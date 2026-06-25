"""
Order Book Action Potential
----------------------------
Behavioral Quant Layer: Microstructure Toxicity (VPIN-adjacent) Visualization

Treats signed order-flow imbalance as an input current into a Leaky
Integrate-and-Fire (LIF) neuron model. When cumulative directional pressure
crosses a threshold, the model "fires" - visualizing toxic, one-sided order
flow as action-potential spikes, rendered in an MRI-style grayscale.

Requirements:
    pip install numpy matplotlib yfinance
    ffmpeg must be installed and on PATH for video export.

Run:
    python action_potential_orderbook.py
Output:
    action_potential_orderbook.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 1. DATA: order-flow imbalance proxy from price/volume
# ----------------------------

def load_intraday_gold(period="5d", interval="5m"):
    """Pull intraday GC=F via yfinance. Falls back to a synthetic tick path
    with two injected toxic order-flow bursts if offline or yfinance fails."""
    try:
        import yfinance as yf
        df = yf.download("GC=F", period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError("empty dataframe")
        close = df["Close"].values.flatten()
        vol = df["Volume"].values.flatten()
        if vol.sum() == 0:
            raise ValueError("no volume data in this feed")
        return close, vol
    except Exception as e:
        print(f"[warn] yfinance unavailable ({e}), using synthetic tick path.")
        n = 600
        rng = np.random.default_rng(11)
        rets = rng.normal(0, 0.0008, n)
        rets[200:230] += rng.normal(0.0009, 0.002, 30)    # buy-side toxic burst
        rets[400:420] -= rng.normal(0.0011, 0.0022, 20)   # sell-side toxic burst
        close = 2050 * np.exp(np.cumsum(rets))
        vol = rng.lognormal(8, 0.5, n)
        return close, vol


def order_flow_imbalance(close, vol):
    """Signed-volume proxy (tick rule): sign(price change) * volume, normalized
    and lightly smoothed -> acts as the 'input current' to the neuron model."""
    sign = np.sign(np.diff(close, prepend=close[0]))
    signed_vol = sign * vol
    imbalance = signed_vol / (np.abs(signed_vol).max() + 1e-9)
    kernel = np.ones(3) / 3
    imbalance = np.convolve(imbalance, kernel, mode="same")
    return imbalance


# ----------------------------
# 2. MODEL: Leaky Integrate-and-Fire neuron
# ----------------------------

TAU = 8.0       # membrane time constant (in bars)
V_RESET = -0.2  # reset potential after a spike
V_THRESH = 1.0  # firing threshold
GAIN = 6.0      # how strongly order-flow imbalance drives membrane potential


def simulate_lif(imbalance, dt=1.0):
    n = len(imbalance)
    V = np.zeros(n)
    spikes = np.zeros(n, dtype=bool)
    v = 0.0
    for t in range(n):
        dv = (-v / TAU + GAIN * imbalance[t]) * dt
        v += dv
        if v >= V_THRESH:
            spikes[t] = True
            v = V_RESET
        V[t] = v
    return V, spikes


close, vol = load_intraday_gold()
imbalance = order_flow_imbalance(close, vol)
V, spikes = simulate_lif(imbalance)
n = len(close)

# ----------------------------
# 3. FIGURE: MRI-grayscale action potential render
# ----------------------------

plt.style.use("dark_background")
fig, (ax_price, ax_v) = plt.subplots(
    2, 1, figsize=(11, 7), facecolor="black",
    gridspec_kw={"height_ratios": [1, 1.3]}, sharex=True
)
fig.subplots_adjust(hspace=0.08, left=0.08, right=0.97, top=0.93, bottom=0.08)

for ax in (ax_price, ax_v):
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("gray")
    ax.tick_params(colors="white", labelsize=8)

ax_price.set_ylabel("Gold Price", color="white", fontsize=9)
ax_v.set_ylabel("Membrane Potential\n(Order Flow Pressure)", color="white", fontsize=9)
ax_v.axhline(V_THRESH, color="firebrick", linestyle="--", linewidth=1, alpha=0.7)
ax_v.text(2, V_THRESH + 0.05, "fire threshold", color="firebrick", fontsize=7)

price_line, = ax_price.plot([], [], color="bisque", linewidth=1.2)
v_line, = ax_v.plot([], [], color="white", linewidth=1.0)
spike_scatter = ax_v.scatter([], [], color="orangered", s=25, zorder=5)

ax_price.set_xlim(0, n)
ax_price.set_ylim(close.min() * 0.999, close.max() * 1.001)
ax_v.set_xlim(0, n)
ax_v.set_ylim(min(V.min(), V_RESET) - 0.2, V_THRESH + 0.3)

fig.suptitle("Order Book Action Potential — Gold Microstructure Toxicity",
             color="white", fontsize=12, fontfamily="serif")

STEP = 2  # advance multiple bars per rendered frame for a smoother, shorter video


def init():
    price_line.set_data([], [])
    v_line.set_data([], [])
    spike_scatter.set_offsets(np.empty((0, 2)))
    return price_line, v_line, spike_scatter


def update(frame):
    idx = min(frame * STEP, n)
    x = np.arange(idx)
    price_line.set_data(x, close[:idx])
    v_line.set_data(x, V[:idx])
    spike_x = x[spikes[:idx]]
    spike_y = V[:idx][spikes[:idx]]
    offsets = np.column_stack([spike_x, spike_y]) if len(spike_x) else np.empty((0, 2))
    spike_scatter.set_offsets(offsets)
    return price_line, v_line, spike_scatter


n_frames = n // STEP + 1
ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init,
                               interval=30, blit=False)

if __name__ == "__main__":
    out_path = "action_potential_orderbook.mp4"
    print(f"Rendering {n_frames} frames to {out_path} ...")
    ani.save(out_path, fps=30, dpi=170, writer="ffmpeg")
    print("Done.")
