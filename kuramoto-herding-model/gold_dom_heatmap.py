"""
Gold DOM Heatmap Dashboard
--------------------------
A daily gold trading dashboard with a DOM-like heatmap, astrophysical styling,
Times New Roman typography, and MP4 export. The visualization blends price,
volume pressure, momentum, and macro pulse into a living heatfield.

Requirements:
    pip install numpy pandas matplotlib requests yfinance scipy
    ffmpeg on PATH for mp4 export

Run:
    python gold_dom_heatmap.py
Output:
    gold_dom_heatmap.mp4
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import requests
import yfinance as yf
from scipy.signal import hilbert

FRED_API_KEY = "f4cc710ee914042b0631cfbd3a4a6c7d"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(series_id, start_date=None, end_date=None):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    if start_date is not None:
        params["observation_start"] = start_date
    if end_date is not None:
        params["observation_end"] = end_date

    try:
        response = requests.get(FRED_BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        observations = data.get("observations", [])
        dates = [obs["date"] for obs in observations]
        values = [float(obs["value"]) if obs["value"] != "." else np.nan for obs in observations]
        return pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    except Exception as exc:
        print(f"[warn] FRED fetch failed for {series_id}: {exc}")
        return pd.Series(dtype=float)


def synthetic_gold_data(days=120, seed=17):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.0095, days)
    rets[25:35] += rng.normal(0.0008, 0.01, 10)
    rets[70:78] -= rng.normal(0.001, 0.014, 8)
    price = 1900 * np.exp(np.cumsum(rets))
    volume = np.abs(rng.normal(1.0, 0.4, days)) * 1e5
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    return pd.DataFrame({"Close": price, "Volume": volume}, index=dates)


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def normalize(series):
    if series.size == 0:
        return series
    arr = np.array(series, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return np.zeros_like(arr)
    low = np.nanmin(arr)
    high = np.nanmax(arr)
    if abs(high - low) < 1e-9:
        return np.zeros_like(arr)
    out = (arr - low) / (high - low)
    return pd.Series(out, index=series.index)


def build_gold_dashboard(days=120):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days * 2)

    try:
        gold_raw = yf.download("GC=F", start=start_date.strftime("%Y-%m-%d"), end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)
        close = gold_raw["Close"]
        volume = gold_raw["Volume"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        gold = pd.DataFrame({"Close": close, "Volume": volume})
        gold = gold.dropna()
        if gold.empty:
            raise ValueError("yfinance returned an empty GC=F dataset")
        gold = gold.tail(days)
    except Exception as exc:
        print(f"[warn] yfinance gold fetch failed, using synthetic data: {exc}")
        gold = synthetic_gold_data(days=days)

    fedfunds = fetch_fred_series("FEDFUNDS", start_date=gold.index.min().strftime("%Y-%m-%d"), end_date=gold.index.max().strftime("%Y-%m-%d"))
    spread = fetch_fred_series("T10Y3M", start_date=gold.index.min().strftime("%Y-%m-%d"), end_date=gold.index.max().strftime("%Y-%m-%d"))
    if fedfunds.empty:
        fedfunds = pd.Series(np.nan, index=gold.index)
        print("[warn] FEDFUNDS unavailable; macro gauge will fallback.")
    if spread.empty:
        spread = pd.Series(np.nan, index=gold.index)
        print("[warn] T10Y3M unavailable; macro gauge will fallback.")

    fedfunds = fedfunds.reindex(gold.index).ffill().bfill()
    spread = spread.reindex(gold.index).ffill().bfill()

    data = gold.copy()
    data["Return"] = data["Close"].pct_change().fillna(0)
    data["LogPrice"] = np.log(data["Close"])
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA34"] = data["Close"].ewm(span=34, adjust=False).mean()
    data["RSI"] = compute_rsi(data["Close"], window=14)
    data["Volatility"] = data["Return"].rolling(12, min_periods=1).std()
    data["Momentum"] = data["Close"].diff(5).fillna(0)
    data["Pulse"] = np.sign(data["Return"]) * data["Volume"]
    data["PulseNorm"] = normalize(np.abs(data["Pulse"]))
    data["TrendStrength"] = normalize(data["Close"].diff(10).fillna(0).abs())
    data["MacroPulse"] = normalize((fedfunds.interpolate().fillna(method="ffill").fillna(method="bfill") + spread.interpolate().fillna(method="ffill").fillna(method="bfill")).fillna(0))
    data["FedFunds"] = fedfunds
    data["Spread"] = spread

    cycle = hilbert(data["Return"].fillna(0).values)
    data["Phase"] = np.angle(cycle)
    data["Amplitude"] = np.abs(cycle)

    levels = 60
    price_pad = 50
    price_min = data["Close"].min() - price_pad
    price_max = data["Close"].max() + price_pad
    price_levels = np.linspace(price_min, price_max, levels)
    heat = np.zeros((levels, len(data)))
    for idx, value in enumerate(data["Close"]):
        width = max(5, 0.008 * value)
        center = value
        gaussian = np.exp(-0.5 * ((price_levels - center) / width) ** 2)
        polarity = np.sign(data["Return"].iloc[idx] if idx > 0 else 1)
        intensity = (0.2 + 0.8 * data["PulseNorm"].iloc[idx]) * gaussian
        heat[:, idx] = intensity * (0.4 + 0.6 * polarity)

    return data, price_levels, heat


def render_dom_mp4(data, price_levels, heat, output_path="gold_dom_heatmap.mp4"):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif", "serif"]
    plt.rcParams["text.color"] = "#e9ecf7"
    plt.rcParams["axes.labelcolor"] = "#d7dbf3"
    plt.rcParams["xtick.color"] = "#b0b7d4"
    plt.rcParams["ytick.color"] = "#b0b7d4"
    plt.rcParams["figure.facecolor"] = "#030312"
    plt.rcParams["axes.facecolor"] = "#030312"

    n = len(data)
    x = np.arange(n)
    star_x = np.random.default_rng(33).uniform(0, 1, 120)
    star_y = np.random.default_rng(33).uniform(0, 1, 120)
    star_sizes = np.random.default_rng(33).uniform(8, 24, 120)
    star_alpha = np.random.default_rng(33).uniform(0.08, 0.33, 120)

    fig = plt.figure(figsize=(18, 10), facecolor="#030312")
    gs = fig.add_gridspec(3, 5, width_ratios=[2.6, 2.6, 1, 1, 1], height_ratios=[1.3, 1, 0.85], wspace=0.1, hspace=0.18)

    ax_heat = fig.add_subplot(gs[0:2, 0:2])
    ax_price = fig.add_subplot(gs[2, 0:2])
    ax_radial = fig.add_subplot(gs[0, 2:], polar=True)
    ax_metrics = fig.add_subplot(gs[1, 2:4])
    ax_macro = fig.add_subplot(gs[1, 4:])

    ax_heat.set_facecolor("#020214")
    ax_price.set_facecolor("#020214")
    ax_radial.set_facecolor("#020214")
    ax_metrics.set_facecolor("#020214")
    ax_macro.set_facecolor("#020214")

    ax_heat.axis("off")
    ax_radial.set_xticks([])
    ax_radial.set_yticks([])
    ax_metrics.axis("off")
    ax_macro.axis("off")

    ax_price.grid(color="#2f3560", linestyle="--", linewidth=0.35, alpha=0.55)
    ax_price.set_xlabel("days", fontsize=12)
    ax_price.set_ylabel("Gold USD/oz", fontsize=12)
    ax_price.set_title("Gold DOM Heatfield — Quantum Positioning", fontsize=22, pad=16, color="#f2f4ff")

    heat_img = ax_heat.imshow(heat, aspect="auto", cmap="magma", origin="lower", extent=[0, n, price_levels[0], price_levels[-1]], vmin=0, vmax=1.2, alpha=0.95)
    ax_heat.set_title("DOM Heatmap — pressure by price level", fontsize=18, color="#f2f4ff", pad=14)
    ax_heat.text(0.02, 0.96, "astro DOM | depth field | daily flow", transform=ax_heat.transAxes, color="#99a4dc", fontsize=10, va="top")

    price_line, = ax_price.plot([], [], color="#ffde73", linewidth=2.4)
    ema_fast_line, = ax_price.plot([], [], color="#8bf4ff", linewidth=1.5)
    ema_slow_line, = ax_price.plot([], [], color="#c493ff", linewidth=1.5)
    current_marker, = ax_price.plot([], [], marker="o", color="#ffffff", markersize=8, markeredgecolor="#ffd97e", markeredgewidth=1.2)

    ax_price.set_xlim(0, n)
    ax_price.set_ylim(data["Close"].min() * 0.993, data["Close"].max() * 1.007)

    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    radial_scatter = ax_radial.scatter([], [], c=[], cmap="plasma", s=120, edgecolors="#ffffff", linewidths=0.8)
    ax_radial.set_title("Quantum Order Flow Field", fontsize=18, color="#f2f4ff", pad=18)

    metrics_text = ax_metrics.text(0.03, 0.96, "", transform=ax_metrics.transAxes, color="#eef0ff", fontsize=13, va="top", family="serif")
    macro_text = ax_macro.text(0.02, 0.96, "", transform=ax_macro.transAxes, color="#eef0ff", fontsize=13, va="top", family="serif")

    frame_label = fig.text(0.01, 0.01, "gold_dom_heatmap.mp4", fontsize=10, color="#7d86b2")

    star_points = ax_heat.scatter(star_x * n, star_y * (price_levels[-1] - price_levels[0]) + price_levels[0], s=star_sizes, color="#9fcfff", alpha=star_alpha, edgecolors="none", zorder=0)

    def init():
        price_line.set_data([], [])
        ema_fast_line.set_data([], [])
        ema_slow_line.set_data([], [])
        current_marker.set_data([], [])
        radial_scatter.set_offsets(np.empty((0, 2)))
        radial_scatter.set_array(np.array([]))
        metrics_text.set_text("")
        macro_text.set_text("")
        return price_line, ema_fast_line, ema_slow_line, current_marker, radial_scatter, metrics_text, macro_text

    def update(frame):
        idx = max(1, frame)
        window = max(30, idx)

        x_data = x[:idx]
        price_line.set_data(x_data, data["Close"].values[:idx])
        ema_fast_line.set_data(x_data, data["EMA12"].values[:idx])
        ema_slow_line.set_data(x_data, data["EMA34"].values[:idx])
        current_marker.set_data([idx - 1], [data["Close"].values[idx - 1]])

        heat_img.set_data(heat[:, :idx])
        heat_img.set_extent([0, idx, price_levels[0], price_levels[-1]])

        phase = data["Phase"].iloc[idx - 1]
        amplitude = data["Amplitude"].iloc[idx - 1]
        radii = 0.65 + 0.18 * np.sin(theta * 4 + phase)
        scatter_x = radii * np.cos(theta)
        scatter_y = radii * np.sin(theta)
        radial_scatter.set_offsets(np.column_stack([theta, radii]))
        radial_scatter.set_array(np.sin(theta + phase))

        rsi = data["RSI"].iloc[idx - 1]
        momentum = data["Momentum"].iloc[idx - 1]
        vol = data["Volatility"].iloc[idx - 1]
        pulse = data["PulseNorm"].iloc[idx - 1]
        fed = data["FedFunds"].iloc[idx - 1]
        spr = data["Spread"].iloc[idx - 1]
        macro = data["MacroPulse"].iloc[idx - 1]

        metrics_text.set_text(
            f"Date: {data.index[idx - 1].strftime('%Y-%m-%d')}\n"
            f"Price: {data['Close'].iloc[idx - 1]:.2f} USD\n"
            f"RSI: {rsi:.1f}   Momentum: {momentum:.1f}\n"
            f"Volatility: {vol:.4f}   Pulse: {pulse:.2f}"
        )

        macro_text.set_text(
            f"Macro Pulse: {macro:.2f}\n"
            f"FedFunds: {fed if not np.isnan(fed) else 0.0:.2f}   Spread: {spr if not np.isnan(spr) else 0.0:.2f}\n"
            f"Trend strength: {data['TrendStrength'].iloc[idx - 1]:.2f}"
        )

        return price_line, ema_fast_line, ema_slow_line, current_marker, radial_scatter, metrics_text, macro_text

    frames = n
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, interval=50, blit=False)
    ani.save(output_path, fps=24, dpi=160, writer="ffmpeg")
    print(f"Saved {output_path} with {frames} frames.")


def main():
    data, levels, heat = build_gold_dashboard(days=120)
    render_dom_mp4(data, levels, heat)


if __name__ == "__main__":
    main()
