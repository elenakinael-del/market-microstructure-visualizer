"""
NEURAL DIFFUSION NETWORK  →  MARKET VOLATILITY
================================================
Neuroimaging-quality 3D animation.

Visual concept:
  - Nodes arranged in a realistic brain-shaped 3D point cloud
  - Edges = white-matter tractography style (DTI fiber tracts)
  - Signal propagates like fMRI BOLD response: node lights up, dims
  - Herding cascade: clusters synchronize → gold volatility spike
  - Camera orbits slowly (like a rotating brain scan render)

Color science:
  - Background: scanner black #030308
  - Resting nodes: deep indigo (#1a1060)
  - Active nodes: hot white-cyan glow (#e0f8ff → #00d4ff → #0033aa)
  - Fiber tracts: platinum (#c0c8d8) at low alpha
  - Volatility surface: thermal (blue→purple→red→yellow)

Requirements:
    pip install numpy matplotlib scipy pillow

Run:
    python neural_diffusion_market.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg not available
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

np.random.seed(2025)

# ══════════════════════════════════════════════════════════════════════
#  PALETTE
# ══════════════════════════════════════════════════════════════════════
BG          = "#030308"
NODE_REST   = "#1a1060"
NODE_PEAK   = "#e8f8ff"
TRACT_COL   = "#8090b8"
SPIKE_COL   = "#ffe566"
VOL_LOW     = "#0d1b6e"
VOL_MED     = "#9b2288"
VOL_HIGH    = "#ff4500"
TEXT_COL    = "#b0c4de"

CMAP_NODE  = mcolors.LinearSegmentedColormap.from_list(
    "neural", ["#0a0840", "#1a3a9c", "#00b4d8", "#caf0f8", "#ffffff"])

CMAP_VOL   = mcolors.LinearSegmentedColormap.from_list(
    "thermal", ["#03045e","#0077b6","#48cae4","#7b2d8b","#e63946","#ffba08"])

# ══════════════════════════════════════════════════════════════════════
#  BRAIN POINT CLOUD  (prolate spheroid + cortical folding noise)
# ══════════════════════════════════════════════════════════════════════
N_NODES   = 180
N_FRAMES  = 220
CASCADE_START = 70    # frame when herding begins
LOCK_FRAME    = 155   # frame of full synchrony / vol spike

def brain_cloud(n):
    """Nodes distributed inside a brain-shaped volume."""
    pts = []
    while len(pts) < n:
        x = np.random.uniform(-1.6, 1.6)
        y = np.random.uniform(-1.2, 1.2)
        z = np.random.uniform(-0.9, 0.9)
        # prolate ellipsoid test
        if (x/1.55)**2 + (y/1.15)**2 + (z/0.85)**2 < 1.0:
            # cortical surface folding: slight repulsion toward shell
            r = np.sqrt(x**2 + y**2 + z**2)
            shell_bias = 0.35 * np.random.random()
            pts.append([x*(1+shell_bias), y*(1+shell_bias), z*(1+shell_bias)])
    pts = np.array(pts[:n])
    # scale to cm-like units
    return pts * 45

POS = brain_cloud(N_NODES)

# ══════════════════════════════════════════════════════════════════════
#  FIBER TRACTS  (k-NN edges, short-range only)
# ══════════════════════════════════════════════════════════════════════
K_NEIGHBORS = 4
dmat = cdist(POS, POS)
np.fill_diagonal(dmat, 1e9)

EDGES = []
for i in range(N_NODES):
    nbrs = np.argsort(dmat[i])[:K_NEIGHBORS]
    for j in nbrs:
        if (j, i) not in [(e[0],e[1]) for e in EDGES]:
            EDGES.append((i, j))

# ══════════════════════════════════════════════════════════════════════
#  KURAMOTO DYNAMICS  (herding = coupling ramp)
# ══════════════════════════════════════════════════════════════════════
OMEGA  = np.random.normal(0, 0.6, N_NODES)   # natural frequencies
PHASES = np.random.uniform(0, 2*np.pi, N_NODES)

def K_coupling(frame):
    if frame < CASCADE_START:
        return 0.25
    t = (frame - CASCADE_START) / (LOCK_FRAME - CASCADE_START)
    return 0.25 + 5.5 / (1 + np.exp(-10*(t-0.5)))

print("Computing Kuramoto dynamics …")
ALL_PHASES  = np.zeros((N_FRAMES, N_NODES))
ALL_PHASES[0] = PHASES.copy()
DT = 0.10

for f in range(1, N_FRAMES):
    th  = ALL_PHASES[f-1]
    K   = K_coupling(f)
    Rc  = np.mean(np.exp(1j*th))
    psi = np.angle(Rc)
    R   = np.abs(Rc)
    dth = OMEGA + K * R * np.sin(psi - th)
    ALL_PHASES[f] = (th + DT*dth) % (2*np.pi)

ORDER_R = np.array([np.abs(np.mean(np.exp(1j*ALL_PHASES[f])))
                    for f in range(N_FRAMES)])

# Node "activation" = coherence with mean phase (0→1)
def node_activation(frame):
    th   = ALL_PHASES[frame]
    psi  = np.angle(np.mean(np.exp(1j*th)))
    R    = ORDER_R[frame]
    delta = np.abs(((th - psi + np.pi) % (2*np.pi)) - np.pi) / np.pi
    base  = (1 - delta)                        # 1 = aligned
    # add traveling wave pulse pre-cascade
    if frame < CASCADE_START:
        t_wave  = frame * 0.18
        dists   = np.linalg.norm(POS - POS.mean(0), axis=1)
        wave    = np.exp(-((dists/30 - (t_wave % 3.0))**2) / 0.4) * 0.45
        return np.clip(wave + 0.08, 0, 1)
    return np.clip(base * R + (1-R)*0.05, 0, 1)

# ══════════════════════════════════════════════════════════════════════
#  VOL SURFACE  (emotion-coded Heston-like surface)
# ══════════════════════════════════════════════════════════════════════
STRIKE_GRID = np.linspace(0.7, 1.3, 28)
TENOR_GRID  = np.linspace(0.02, 2.0, 28)
S, T        = np.meshgrid(STRIKE_GRID, TENOR_GRID)

def vol_surface(frame):
    R  = ORDER_R[frame]
    K  = K_coupling(frame)
    # base surface: ATM smile
    atm   = 0.14 + 0.18 * R
    skew  = -0.12 * (1 + 0.8*R)
    smile = 0.08 * (1 + 1.5*R)
    tenor_term = 0.06 / np.sqrt(T + 0.01)
    iv = atm + skew*(S-1) + smile*(S-1)**2 + tenor_term
    # add herding spike
    if frame >= CASCADE_START:
        t_since = frame - CASCADE_START
        spike   = 0.22 * R * np.exp(-((S-1.0)**2)/0.04) * np.exp(-T/0.5)
        iv += spike * min(1.0, t_since/40)
    return np.clip(iv, 0.02, 0.65)

# ══════════════════════════════════════════════════════════════════════
#  GOLD PRICE  (random walk + cascade spike)
# ══════════════════════════════════════════════════════════════════════
PRICE = np.cumsum(np.random.normal(0, 0.3, N_FRAMES))
PRICE -= PRICE.min()
PRICE /= PRICE.max()
for i in range(LOCK_FRAME, N_FRAMES):
    t = i - LOCK_FRAME
    PRICE[i] += 0.85 * (1 - np.exp(-t/8)) * np.exp(-t/55)
PRICE = (PRICE - PRICE.min()) / (PRICE.max() - PRICE.min())

# ══════════════════════════════════════════════════════════════════════
#  FIGURE LAYOUT
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9), facecolor=BG)
fig.patch.set_facecolor(BG)

# Title
fig.text(0.5, 0.965,
         "NEURAL DIFFUSION NETWORK  ·  EMOTION → VOLATILITY CASCADE",
         ha="center", va="top", color=TEXT_COL,
         fontsize=13, fontweight="bold", fontfamily="monospace",
)

fig.text(0.5, 0.942,
         "Kuramoto phase synchrony  ·  Bayesian herding  ·  Gold IV surface",
         ha="center", va="top", color="#5070a0",
         fontsize=8, fontfamily="monospace")

# Grid: [brain 3D left] [vol surface right-top] [price right-bottom]
ax_brain = fig.add_axes([0.01, 0.06, 0.54, 0.86], projection="3d")
ax_vol   = fig.add_axes([0.58, 0.38, 0.40, 0.52], projection="3d")
ax_price = fig.add_axes([0.60, 0.06, 0.37, 0.25])

# ── Brain axis styling ─────────────────────────────────────────────
ax_brain.set_facecolor(BG)
ax_brain.xaxis.pane.fill = False
ax_brain.yaxis.pane.fill = False
ax_brain.zaxis.pane.fill = False
ax_brain.xaxis.pane.set_edgecolor("#0a0a20")
ax_brain.yaxis.pane.set_edgecolor("#0a0a20")
ax_brain.zaxis.pane.set_edgecolor("#0a0a20")
ax_brain.grid(False)
ax_brain.set_axis_off()
ax_brain.set_xlim(-80, 80); ax_brain.set_ylim(-80, 80); ax_brain.set_zlim(-60, 60)

# Label
ax_brain.text2D(0.5, 0.01,
    "TRADER NETWORK  ·  SYNAPTIC SYNCHRONIZATION",
    transform=ax_brain.transAxes, ha="center", va="bottom",
    color="#4060a0", fontsize=7, fontfamily="monospace")

# ── Vol surface axis styling ───────────────────────────────────────
ax_vol.set_facecolor(BG)
ax_vol.xaxis.pane.fill = False
ax_vol.yaxis.pane.fill = False
ax_vol.zaxis.pane.fill = False
for pane in [ax_vol.xaxis.pane, ax_vol.yaxis.pane, ax_vol.zaxis.pane]:
    pane.set_edgecolor("#0d0d22")
ax_vol.grid(False)
ax_vol.tick_params(colors="#303060", labelsize=6)
ax_vol.set_xlabel("Strike / ATM", color="#303060", fontsize=6, labelpad=2)
ax_vol.set_ylabel("Tenor (yr)", color="#303060", fontsize=6, labelpad=2)
ax_vol.set_zlabel("Impl. Vol", color="#303060", fontsize=6, labelpad=2)
ax_vol.set_title("GOLD IV SURFACE", color="#4060a0",
                 fontsize=7, fontfamily="monospace", pad=3)
ax_vol.view_init(elev=28, azim=-55)

# ── Price axis styling ─────────────────────────────────────────────
ax_price.set_facecolor(BG)
ax_price.tick_params(colors=TEXT_COL, labelsize=7)
for spine in ax_price.spines.values():
    spine.set_edgecolor("#0d0d22")
ax_price.set_xlim(0, N_FRAMES)
ax_price.set_ylim(-0.1, 1.25)
ax_price.set_ylabel("XAU/USD  (norm)", color=TEXT_COL, fontsize=7)
ax_price.set_xlabel("Time →", color="#303060", fontsize=7)
ax_price.set_title("GOLD PRICE  ·  CASCADE RESPONSE",
                   color="#4060a0", fontsize=7, fontfamily="monospace", pad=3)

# Pre-draw order-R indicator as colored background bar
ax_price.axhline(1.0, color="#303060", lw=0.4, ls="--", alpha=0.5)

# ══════════════════════════════════════════════════════════════════════
#  INITIAL PLOT OBJECTS
# ══════════════════════════════════════════════════════════════════════
act0    = node_activation(0)
colors0 = CMAP_NODE(act0)
sizes0  = 18 + 80 * act0 ** 2

brain_sc = ax_brain.scatter(
    POS[:,0], POS[:,1], POS[:,2],
    c=colors0, s=sizes0, alpha=0.85,
    depthshade=True, zorder=3,
)

# Draw fiber tracts (static geometry, only alpha changes)
tract_lines = []
for (i, j) in EDGES:
    ln, = ax_brain.plot(
        [POS[i,0], POS[j,0]],
        [POS[i,1], POS[j,1]],
        [POS[i,2], POS[j,2]],
        color=TRACT_COL, alpha=0.08, lw=0.6, zorder=1,
    )
    tract_lines.append((i, j, ln))

# Vol surface placeholder
iv0   = vol_surface(0)
vol_surf = [ax_vol.plot_surface(S, T, iv0, cmap=CMAP_VOL,
                                 alpha=0.88, linewidth=0, antialiased=True)]

# Price line
price_line, = ax_price.plot([], [], color=SPIKE_COL, lw=1.8, zorder=3)
price_fill  = [None]
lock_vline  = ax_price.axvline(-1, color="#ff4040", lw=0.9, ls=":", alpha=0)

# ── Overlay text ──────────────────────────────────────────────────
status_txt = fig.text(
    0.285, 0.895, "RESTING STATE  ·  Desynchronized",
    ha="center", color="#3a4a7a", fontsize=9,
    fontfamily="monospace", fontweight="bold",
)
r_val_txt = fig.text(
    0.285, 0.872, "R = 0.00  |  K = 0.25",
    ha="center", color="#3a4a7a", fontsize=8, fontfamily="monospace",
)

# ── Camera azimuth schedule ────────────────────────────────────────
AZ_SCHEDULE = np.linspace(-40, 40, N_FRAMES)   # slow orbit

# ══════════════════════════════════════════════════════════════════════
#  ANIMATE
# ══════════════════════════════════════════════════════════════════════
lock_shown = [False]

def update(frame):
    act  = node_activation(frame)
    R    = ORDER_R[frame]
    K    = K_coupling(frame)

    # ── Node colors & sizes ──────────────────────────────────────
    c = CMAP_NODE(act)
    s = 18 + 110 * act**2
    brain_sc._offsets3d = (POS[:,0], POS[:,1], POS[:,2])
    brain_sc.set_color(c)
    brain_sc.set_sizes(s)

    # ── Tract alpha (brightens with R) ───────────────────────────
    tract_alpha = 0.04 + 0.30 * R
    for (i, j, ln) in tract_lines:
        edge_act = (act[i] + act[j]) / 2
        a = np.clip(tract_alpha * (0.3 + 1.4*edge_act), 0, 0.6)
        ln.set_alpha(a)
        r_val = 0.3 + 0.7 * edge_act
        ln.set_color((r_val*0.5, r_val*0.6, r_val*1.0))

    # ── Camera orbit ─────────────────────────────────────────────
    ax_brain.view_init(elev=22, azim=AZ_SCHEDULE[frame])

    # ── Vol surface ───────────────────────────────────────────────
    for s_obj in vol_surf:
        s_obj.remove()
    vol_surf[0] = ax_vol.plot_surface(
        S, T, vol_surface(frame),
        cmap=CMAP_VOL, alpha=0.88,
        linewidth=0, antialiased=True,
    )

    # ── Price ─────────────────────────────────────────────────────
    xs = np.arange(frame+1)
    price_line.set_data(xs, PRICE[:frame+1])
    if price_fill[0]:
        price_fill[0].remove()
    price_fill[0] = ax_price.fill_between(
        xs, PRICE[:frame+1], alpha=0.15, color=SPIKE_COL,
    )

    # Lock line
    if frame >= LOCK_FRAME and not lock_shown[0]:
        lock_shown[0] = True
        lock_vline.set_xdata([LOCK_FRAME])
        lock_vline.set_alpha(0.7)

    # ── Status text ───────────────────────────────────────────────
    if R > 0.82:
        status_txt.set_text("⚡  PHASE LOCK  ·  HERDING CASCADE  ·  IV SPIKE")
        status_txt.set_color("#ff6030")
    elif R > 0.50:
        status_txt.set_text("⚠   SYNCHRONIZING  ·  Clusters aligning")
        status_txt.set_color("#ff9900")
    else:
        status_txt.set_text("RESTING STATE  ·  Independent oscillators")
        status_txt.set_color("#3a5a9a")
    r_val_txt.set_text(f"Coherence R = {R:.3f}  |  Herding K = {K:.2f}")

    return [brain_sc, price_line, lock_vline]

print("Building animation …")
ani = FuncAnimation(fig, update, frames=N_FRAMES,
                    interval=40, blit=False)

try:
    writer = FFMpegWriter(fps=25, bitrate=3200,
                          metadata={"title": "Neural Diffusion Market"})
    ani.save("neural_diffusion_market.mp4", writer=writer, dpi=160,
             savefig_kwargs={"facecolor": BG})
    print("✓  Saved: neural_diffusion_market.mp4")
except Exception as e:
    print(f"MP4 skipped ({e}) — install ffmpeg to export video.")

plt.show()
