"""
Kuramoto Synchronization Cascade
---------------------------------
Behavioral Quant Layer: Bayesian Herding Cascade Visualization

Models gold market participants as coupled oscillators (Kuramoto model).
Coupling strength K(t) is driven by a realized-volatility stress proxy computed
from price data. As stress rises, traders' "sentiment phases" synchronize -
visualizing a herding cascade building, peaking, and dispersing.

Order parameter r(t) = |mean(e^{i*theta})| is the standard Kuramoto sync metric:
  r -> 0   : phases random / dispersed sentiment
  r -> 1   : phases locked / full herding cascade

Requirements:
    pip install numpy matplotlib yfinance
    ffmpeg must be installed and on PATH for video export.

Run:
    python kuramoto_herding_cascade.py
Output:
    kuramoto_herding_cascade.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 1. DATA: pull gold prices, derive a stress proxy
# ----------------------------

def load_gold_data(period="1y", interval="1d"):
    """Pull GC=F (COMEX gold futures) via yfinance. Falls back to a synthetic
    path with two injected volatility clusters if offline or yfinance fails."""
    try:
        import yfinance as yf
        df = yf.download("GC=F", period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError("Empty dataframe returned")
        close = df["Close"].values.flatten()
        return close
    except Exception as e:
        print(f"[warn] yfinance unavailable ({e}), using synthetic GBM path instead.")
        n = 360
        rng = np.random.default_rng(7)
        rets = rng.normal(0.0003, 0.012, n)
        rets[90:120] += rng.normal(0, 0.03, 30)    # stress cluster 1
        rets[240:270] -= rng.normal(0, 0.035, 30)  # stress cluster 2
        close = 2000 * np.exp(np.cumsum(rets))
        return close


def stress_proxy(close, window=10):
    """Rolling realized volatility, normalized to [0,1]. Drives coupling strength K."""
    rets = np.diff(np.log(close))
    vol = np.array([
        np.std(rets[max(0, i - window):i + 1]) for i in range(len(rets))
    ])
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
    return vol_norm  # length = len(close) - 1


# ----------------------------
# 2. MODEL: Kuramoto oscillators
# ----------------------------

N_OSC = 64                  # number of "trader" oscillators
K_MIN, K_MAX = 0.5, 8.0      # coupling strength range, mapped from stress proxy
DT = 0.05
STEPS_PER_FRAME = 1

close = load_gold_data()
stress = stress_proxy(close)
n_frames = len(stress)

rng = np.random.default_rng(42)
omega = rng.normal(0, 1.0, N_OSC)         # natural frequencies = idiosyncratic bias
theta = rng.uniform(0, 2 * np.pi, N_OSC)  # initial phases = initial sentiment


def kuramoto_step(theta, omega, K, dt):
    diff = theta[None, :] - theta[:, None]
    coupling = (K / N_OSC) * np.sin(-diff).sum(axis=1)
    theta_new = theta + dt * (omega + coupling)
    return theta_new % (2 * np.pi)


def order_parameter(theta):
    z = np.mean(np.exp(1j * theta))
    return np.abs(z), np.angle(z)  # r, psi


# ----------------------------
# 3. FIGURE: MRI-style dark render
# ----------------------------

plt.style.use("dark_background")
fig = plt.figure(figsize=(9, 9), facecolor="black")
ax_main = fig.add_axes([0.05, 0.25, 0.9, 0.7])
ax_r = fig.add_axes([0.05, 0.06, 0.9, 0.14])

ax_main.set_xlim(-1.4, 1.4)
ax_main.set_ylim(-1.4, 1.4)
ax_main.set_aspect("equal")
ax_main.axis("off")

ax_r.set_facecolor("black")
ax_r.set_xlim(0, n_frames)
ax_r.set_ylim(0, 1)
ax_r.set_ylabel("Order Parameter r(t)\n(herding intensity)", color="white", fontsize=8)
ax_r.tick_params(colors="white", labelsize=7)
for spine in ax_r.spines.values():
    spine.set_color("gray")

r_history = []

angles_static = np.linspace(0, 2 * np.pi, N_OSC, endpoint=False)
ring_x = np.cos(angles_static)
ring_y = np.sin(angles_static)

scat = ax_main.scatter([], [], s=60, c=[], cmap="bone", vmin=0, vmax=1,
                        edgecolors="white", linewidths=0.3)
lines = LineCollection([], colors="white", linewidths=0.3, alpha=0.15)
ax_main.add_collection(lines)
title_txt = ax_main.text(0, 1.3, "", ha="center", color="white", fontsize=13,
                          fontfamily="serif")
r_line, = ax_r.plot([], [], color="lightgray", linewidth=1.2)


def get_positions(theta):
    # oscillators sit on a fixed ring; only color/connections move with phase
    return ring_x.copy(), ring_y.copy()


def local_sync_color(theta):
    r, psi = order_parameter(theta)
    align = np.cos(theta - psi)  # -1..1, how aligned each node is with mean field
    return (align + 1) / 2       # rescale to 0..1 for colormap


def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat, lines, r_line, title_txt


def update(frame):
    global theta, r_history
    K = K_MIN + (K_MAX - K_MIN) * stress[min(frame, len(stress) - 1)]
    for _ in range(STEPS_PER_FRAME):
        theta = kuramoto_step(theta, omega, K, DT)

    r, psi = order_parameter(theta)
    r_history.append(r)

    x, y = get_positions(theta)
    colors = local_sync_color(theta)
    scat.set_offsets(np.column_stack([x, y]))
    scat.set_array(colors)

    # connect oscillators that are nearly phase-locked
    segs = []
    for i in range(N_OSC):
        for j in range(i + 1, N_OSC):
            if np.cos(theta[i] - theta[j]) > 0.95:
                segs.append([(x[i], y[i]), (x[j], y[j])])
    lines.set_segments(segs)
    lines.set_alpha(min(0.5, 0.08 + r * 0.4))

    r_line.set_data(range(len(r_history)), r_history)

    regime = "CASCADE" if r > 0.75 else ("BUILDING" if r > 0.4 else "DISPERSED")
    title_txt.set_text(f"Bayesian Herding Cascade  |  r = {r:.2f}  [{regime}]")

    return scat, lines, r_line, title_txt


ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init,
                               interval=40, blit=False)

# ----------------------------
# 4. EXPORT
# ----------------------------
if __name__ == "__main__":
    out_path = "kuramoto_herding_cascade.mp4"
    print(f"Rendering {n_frames} frames to {out_path} ...")
    ani.save(out_path, fps=20, dpi=180, writer="ffmpeg")
    print("Done.")
