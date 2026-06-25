# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy", "pandas", "matplotlib", "requests", "scipy", "imageio"
# ]
# ///

import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import requests
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


def synthetic_gold_series(days=365, seed=11):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, days)
    rets[80:100] += rng.normal(0.0005, 0.01, 20)
    rets[180:200] -= rng.normal(0.0006, 0.012, 20)
    price = 1900 * np.exp(np.cumsum(rets))
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    return pd.Series(price, index=dates)


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
    min_val = np.nanmin(series)
    max_val = np.nanmax(series)
    if abs(max_val - min_val) < 1e-9:
        return np.zeros_like(series)
    return (series - min_val) / (max_val - min_val)


def build_gold_view():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=540)

    try:
        import yfinance as yf
        gold_df = yf.download("GC=F", start=start_date.strftime("%Y-%m-%d"), end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)
        gold = gold_df["Close"]
        if isinstance(gold, pd.DataFrame):
            gold = gold.iloc[:, 0]
        gold = gold.dropna()
        gold.index = pd.to_datetime(gold.index).tz_localize(None)
        if gold.empty:
            raise ValueError("yfinance returned empty GC=F series")
    except Exception as exc:
        print(f"[warn] gold price fetch failed, using synthetic series: {exc}")
        gold = synthetic_gold_series(365)

    macro = fetch_fred_series("FEDFUNDS", start_date=start_date.isoformat(), end_date=end_date.isoformat())
    rate_spread = fetch_fred_series("T10Y3M", start_date=start_date.isoformat(), end_date=end_date.isoformat())

    if macro.empty:
        macro = pd.Series(np.nan, index=gold.index)
        print("[warn] FEDFUNDS series unavailable, leaving blank.")
    if rate_spread.empty:
        rate_spread = pd.Series(np.nan, index=gold.index)
        print("[warn] Rate spread series unavailable, leaving blank.")

    gold = gold.reindex(pd.date_range(gold.index.min(), gold.index.max(), freq="B")).interpolate(method="time")
    macro = macro.reindex(gold.index)
    rate_spread = rate_spread.reindex(gold.index)

    data = pd.DataFrame({"gold": gold, "fedfunds": macro, "spread": rate_spread})
    data["fedfunds"] = data["fedfunds"].ffill().bfill()
    data["spread"] = data["spread"].ffill().bfill()

    data["returns"] = data["gold"].pct_change().fillna(0)
    data["log_gold"] = np.log(data["gold"])
    data["trend"] = data["log_gold"].rolling(50, min_periods=10).mean()
    data["ma_fast"] = data["gold"].rolling(12, min_periods=1).mean()
    data["ma_slow"] = data["gold"].rolling(36, min_periods=1).mean()
    data["momentum"] = data["gold"].diff(8).fillna(0)
    data["volatility"] = data["returns"].rolling(18, min_periods=3).std().fillna(method="bfill")
    data["rsi"] = compute_rsi(data["gold"], window=14)
    data["macd"] = data["gold"].ewm(span=12, adjust=False).mean() - data["gold"].ewm(span=26, adjust=False).mean()
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    data["trend_strength"] = normalize(data["trend"].diff().abs().fillna(0))
    data["pressure"] = normalize(data["volatility"] * np.abs(data["macd_hist"]))
    data["cycle_phase"] = np.angle(hilbert(data["returns"].fillna(0).values))
    data["cycle_amplitude"] = np.abs(hilbert(data["returns"].fillna(0).values))

    return data


def render_gold_mp4(data, output_path="gold_quant_positioning.mp4"):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif", "serif"]
    plt.rcParams["text.color"] = "#f4f4f4"
    plt.rcParams["axes.labelcolor"] = "#d8d8d8"
    plt.rcParams["xtick.color"] = "#b8b8b8"
    plt.rcParams["ytick.color"] = "#b8b8b8"
    plt.rcParams["figure.facecolor"] = "#02020a"
    plt.rcParams["axes.facecolor"] = "#02020a"

    n = len(data)
    x = np.arange(n)
    star_count = 180
    rng = np.random.default_rng(29)
    stars_x = rng.uniform(-1.2, 1.2, star_count)
    stars_y = rng.uniform(-0.7, 0.95, star_count)
    star_sizes = rng.uniform(2, 10, star_count)
    star_alpha = rng.uniform(0.08, 0.35, star_count)

    fig = plt.figure(figsize=(18, 10), facecolor="#02020a")
    gs = fig.add_gridspec(3, 4, width_ratios=[2, 2, 1, 1], height_ratios=[1.2, 1.0, 0.9], wspace=0.12, hspace=0.16)

    ax_price = fig.add_subplot(gs[0:2, 0:2])
    ax_orbit = fig.add_subplot(gs[0:2, 2:], polar=True)
    ax_heat = fig.add_subplot(gs[2, 0:2])
    ax_gauge = fig.add_subplot(gs[2, 2:])

    ax_price.set_facecolor("#030315")
    ax_orbit.set_facecolor("#02020a")
    ax_heat.set_facecolor("#02020a")
    ax_gauge.set_facecolor("#02020a")

    ax_price.grid(color="#2c2f4a", linestyle="--", linewidth=0.35, alpha=0.55)
    ax_heat.grid(False)
    ax_gauge.grid(False)
    ax_orbit.set_xticks([])
    ax_orbit.set_yticks([])

    price_line, = ax_price.plot([], [], color="#f7d86d", linewidth=2.3, solid_capstyle="round")
    ma_fast_line, = ax_price.plot([], [], color="#8bf4ff", linewidth=1.3, linestyle="-")
    ma_slow_line, = ax_price.plot([], [], color="#c48bff", linewidth=1.3, linestyle="-")
    current_dot = ax_price.scatter([], [], s=80, c="#ffdd72", edgecolors="#ffffff", linewidths=0.8, zorder=4)
    ax_price.set_title("Gold Quantum Positioning Engine", fontsize=28, pad=18, color="#f2f2f2", weight="bold")
    subtitle = ax_price.text(0.01, 0.94, "A daily astrophysical heatmap for gold trading and positioning", transform=ax_price.transAxes, color="#b8c3d8", fontsize=12, va="top")

    ax_price.text(0.02, 0.08, "Times New Roman | macro-informed | quantum-inspired", transform=ax_price.transAxes, color="#8da7dc", fontsize=11)
    ax_price.set_xlim(0, n)
    ax_price.set_ylim(data["gold"].min() * 0.995, data["gold"].max() * 1.005)
    ax_price.set_xlabel("Trading Days", fontsize=12, labelpad=8)
    ax_price.set_ylabel("Gold Price USD/oz", fontsize=12, labelpad=10)

    orbit_nodes = ax_orbit.scatter([], [], s=120, c=[], cmap="plasma", vmin=-1, vmax=1, edgecolors="#ffffff", linewidths=0.7)
    orbit_ring = patches.Circle((0, 0), 1.0, transform=ax_orbit.transData._b, fill=False, color="#5b5f8f", linewidth=1.4, alpha=0.7)
    ax_orbit.add_patch(orbit_ring)
    ax_orbit.text(0.5, 1.05, "Quantum Phase Field", transform=ax_orbit.transAxes, color="#f4f4f8", fontsize=18, ha="center")

    heat_data = np.vstack([
        normalize(data["momentum"]),
        normalize(data["volatility"]),
        normalize(data["rsi"]),
        normalize(data["macd_hist"]),
        normalize(data["pressure"]),
    ])
    heat = ax_heat.imshow(heat_data[:, :1], aspect="auto", cmap="magma", vmin=0, vmax=1, origin="lower")
    ax_heat.set_yticks([0, 1, 2, 3, 4])
    ax_heat.set_yticklabels(["Momentum", "Volatility", "RSI", "MACD", "Pulse"], fontsize=11, color="#f1f1f1")
    ax_heat.set_xticks([])
    ax_heat.set_title("Signal Heatmap — live fields of market pressure", fontsize=16, color="#f0f0f0", pad=10)

    gauge_bar = ax_gauge.barh([0], [0], height=0.4, color="#7ee6ff", alpha=0.85)
    gauge_text = ax_gauge.text(0.5, 0.55, "", transform=ax_gauge.transAxes, ha="center", va="center", color="#fdfdfd", fontsize=18)
    ax_gauge.set_xlim(0, 1)
    ax_gauge.set_ylim(-0.5, 1.5)
    ax_gauge.set_yticks([])
    ax_gauge.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_gauge.set_xticklabels(["Low", "", "Medium", "", "High"], fontsize=11, color="#c3c9eb")
    ax_gauge.set_title("Macro Pressure Gauge", fontsize=16, color="#f0f0f0", pad=10)

    clock_text = fig.text(0.78, 0.07, "", fontsize=14, color="#d8d8ff")
    frame_text = fig.text(0.02, 0.02, "Generated by gold_quant_positioning", fontsize=10, color="#8088a0")

    star_scatter = ax_price.scatter(stars_x * n * 0.025 + n * 0.5, stars_y * (data["gold"].max() - data["gold"].min()) * 0.18 + data["gold"].min(),
                                     s=star_sizes, c="#b6d8ff", alpha=star_alpha, edgecolors="none", zorder=1)

    def init():
        price_line.set_data([], [])
        ma_fast_line.set_data([], [])
        ma_slow_line.set_data([], [])
        current_dot.set_offsets(np.empty((0, 2)))
        orbit_nodes.set_offsets(np.empty((0, 2)))
        orbit_nodes.set_array(np.array([]))
        heat.set_data(np.zeros((5, 1)))
        gauge_bar[0].set_width(0)
        gauge_text.set_text("")
        clock_text.set_text("")
        return price_line, ma_fast_line, ma_slow_line, current_dot, orbit_nodes, heat, gauge_bar[0], gauge_text, clock_text

    def update(frame):
        idx = max(1, frame)
        x_data = x[:idx]
        price_line.set_data(x_data, data["gold"].values[:idx])
        ma_fast_line.set_data(x_data, data["ma_fast"].values[:idx])
        ma_slow_line.set_data(x_data, data["ma_slow"].values[:idx])
        current_dot.set_offsets([[idx - 1, data["gold"].values[idx - 1]]])

        phase = data["cycle_phase"].iloc[idx - 1]
        amplitude = data["cycle_amplitude"].iloc[idx - 1]
        node_angles = np.linspace(0, 2 * np.pi, 22, endpoint=False)
        node_radii = 0.85 + 0.1 * np.sin(node_angles * 2 + phase)
        node_x = node_radii * np.cos(node_angles)
        node_y = node_radii * np.sin(node_angles)
        colors = np.sin(node_angles + phase)
        orbit_nodes.set_offsets(np.column_stack([node_angles, node_radii]))
        orbit_nodes.set_array(colors)

        segment_start = max(0, idx - 90)
        heat_segment = heat_data[:, segment_start:idx]
        padded = np.zeros((heat_data.shape[0], max(1, heat_data.shape[1])))
        padded[:, -heat_segment.shape[1]:] = heat_segment
        heat.set_data(padded)

        fed_value = data["fedfunds"].iloc[idx - 1]
        spread_value = data["spread"].iloc[idx - 1]
        if np.isnan(fed_value) and np.isnan(spread_value):
            macro_signal = data["pressure"].iloc[idx - 1]
        else:
            macro_signal = np.nanmean([fed_value, spread_value])
            if np.isnan(macro_signal):
                macro_signal = data["pressure"].iloc[idx - 1]
        macro_norm = normalize(pd.Series([macro_signal, data["pressure"].iloc[idx - 1]])).iloc[1]
        gauge_bar[0].set_width(float(macro_norm))
        gauge_text.set_text(
            f"Macro pulse {macro_norm:0.2f} | Fed funds {fed_value if not np.isnan(fed_value) else 0.0:0.2f} | Spread {spread_value if not np.isnan(spread_value) else 0.0:0.2f}"
        )

        clock_text.set_text(f"{data.index[idx - 1].strftime('%Y-%m-%d')}  |  frame {idx}/{n}")

        return price_line, ma_fast_line, ma_slow_line, current_dot, orbit_nodes, heat, gauge_bar[0], gauge_text, clock_text

    n_frames = n
    ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=45, blit=False)
    ani.save(output_path, fps=25, dpi=160, writer="ffmpeg")
    print(f"Saved {output_path} with {n_frames} frames.")


def main() -> None:
    data = build_gold_view()
    render_gold_mp4(data)


if __name__ == "__main__":
    main()
