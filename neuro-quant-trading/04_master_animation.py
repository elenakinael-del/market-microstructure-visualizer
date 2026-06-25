"""
04_master_animation.py
-----------------------
The centrepiece animation. Combines:
  - Left panel:  3D transparent brain with activating regions
  - Centre-top:  Scenario label + neuroscience annotation
  - Right panel: Live PnL curves building day by day
  - Bottom bar:  Cost attribution strip

Narrative arc: calm → vol spike → panic exit → recovery → reward chase → recovery

Output: outputs/master_neuro_quant.mp4

Runtime: ~3–5 min on a modern laptop.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import animation
from scipy.ndimage import gaussian_filter
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from brain_config import (
    BRAIN_REGIONS, REGION_COLOURS, TRADING_SCENARIOS,
    ALGO_ACTIVATIONS, SCENARIO_SEQUENCE, TRADING_DAYS,
    BASE_EDGE_BPS, TRADES_PER_DAY,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colours / style constants
# ---------------------------------------------------------------------------
BG     = "#060610"
PANEL  = "#0A0A1A"
SPINE  = "#1C1C2C"
GOLD   = "#D4AF37"
ALGO_C = "#4A90D9"
HUM_C  = "#E84040"
LABEL_C= "#7A8AA0"

# ---------------------------------------------------------------------------
# PnL simulation (same seed as script 02 for consistency)
# ---------------------------------------------------------------------------
np.random.seed(42)

BIAS_EVENTS = [
    (38,  "vol_spike",    -47),
    (51,  "panic_sell",   -112),
    (89,  "reward_chase", -63),
    (117, "vol_spike",    -47),
    (143, "panic_sell",   -112),
    (178, "reward_chase", -63),
    (201, "vol_spike",    -47),
    (220, "panic_sell",   -112),
]
bias_day_map = {ev[0]: ev for ev in BIAS_EVENTS}

base_daily = np.random.normal(BASE_EDGE_BPS * TRADES_PER_DAY, 35, TRADING_DAYS)
algo_daily = base_daily.copy()
human_daily = base_daily.copy()
for d, sc_key, cost in BIAS_EVENTS:
    human_daily[d] += cost

BPS_TO_USD = 10.0
algo_cum  = np.cumsum(algo_daily) * BPS_TO_USD
human_cum = np.cumsum(human_daily) * BPS_TO_USD
days = np.arange(1, TRADING_DAYS + 1)

# ---------------------------------------------------------------------------
# Brain geometry helpers
# ---------------------------------------------------------------------------
def make_brain_surface(n=50, seed=7):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)
    X = 70 * np.sin(U) * np.cos(V)
    Y = 90 * np.sin(U) * np.sin(V)
    Z = 65 * np.cos(U)
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(0, 5, X.shape), sigma=3)
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-9
    X += noise * X / R
    Y += noise * Y / R
    Z += noise * Z / R
    return X, Y, Z


_BX, _BY, _BZ = make_brain_surface()


def sphere(cx, cy, cz, r, n=18):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)
    X = cx + r * np.sin(U) * np.cos(V)
    Y = cy + r * np.sin(U) * np.sin(V)
    Z = cz + r * np.cos(U)
    return X, Y, Z


def lerp_act(a, b, t):
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def scenario_act(sc_key):
    sc = TRADING_SCENARIOS[sc_key]
    base = {k: 0.0 for k in BRAIN_REGIONS}
    base.update(sc["activations"])
    return base


# ---------------------------------------------------------------------------
# Scenario timeline — each scenario maps to a PnL day range
# ---------------------------------------------------------------------------
# We'll divide 252 days evenly across the scenario sequence
SC_SEQ = SCENARIO_SEQUENCE
SC_N   = len(SC_SEQ)
DAYS_PER_SC = TRADING_DAYS // SC_N  # ~25 days per scenario

# For each PnL day, which scenario are we in?
def day_to_scenario(day):
    idx = min(day // DAYS_PER_SC, SC_N - 1)
    return SC_SEQ[idx]


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
BRAIN_FRAMES_PER_DAY = 1       # 1 brain render per trading day
SKIP_DAYS = 3                  # render every N trading days
frame_days = list(range(0, TRADING_DAYS, SKIP_DAYS))
TOTAL_FRAMES = len(frame_days)
FPS = 20

fig = plt.figure(figsize=(18, 10), facecolor=BG)
gs = gridspec.GridSpec(
    3, 3,
    height_ratios=[0.07, 0.78, 0.15],
    width_ratios=[0.42, 0.16, 0.42],
    hspace=0.04, wspace=0.01,
    figure=fig,
    left=0.02, right=0.98, top=0.98, bottom=0.03,
)

ax_hdr   = fig.add_subplot(gs[0, :])
ax_brain = fig.add_subplot(gs[1, 0], projection="3d")
ax_info  = fig.add_subplot(gs[1, 1])
ax_pnl   = fig.add_subplot(gs[1, 2])
ax_bot   = fig.add_subplot(gs[2, :])

for ax in [ax_hdr, ax_info, ax_bot]:
    ax.set_axis_off()
    ax.set_facecolor(BG)

ax_brain.set_facecolor(PANEL)
ax_pnl.set_facecolor(PANEL)

for sp in ax_pnl.spines.values():
    sp.set_color(SPINE)

# Header
ax_hdr.set_facecolor(BG)
ax_hdr.text(
    0.5, 0.5,
    "Neural Signature of Trading Decisions  ·  Gold Futures  ·  Human vs Systematic Algo",
    ha="center", va="center", fontsize=12, fontweight="bold",
    color=GOLD, fontfamily="monospace", transform=ax_hdr.transAxes,
)

# PnL axis
ax_pnl.set_xlim(0, TRADING_DAYS + 5)
pnl_lo = min(human_cum.min(), algo_cum.min()) * 1.18
pnl_hi = algo_cum.max() * 1.18
ax_pnl.set_ylim(pnl_lo, pnl_hi)
ax_pnl.tick_params(colors=LABEL_C, labelsize=8)
ax_pnl.grid(color="#111122", linewidth=0.7, linestyle="--")
ax_pnl.axhline(0, color=SPINE, linewidth=0.8)
ax_pnl.set_ylabel("Cumulative PnL  ($)", color=LABEL_C,
                   fontsize=8.5, fontfamily="monospace")
ax_pnl.set_xlabel("Trading Day", color=LABEL_C,
                   fontsize=8.5, fontfamily="monospace")
ax_pnl.set_title("Year-to-Date PnL  (1 GC contract)", color=GOLD,
                  fontsize=9.5, fontfamily="monospace", pad=6)

legend_patches = [
    mpatches.Patch(color=ALGO_C, label="Systematic Algo"),
    mpatches.Patch(color=HUM_C,  label="Human Trader"),
]
ax_pnl.legend(handles=legend_patches, loc="upper left",
              fontsize=8, framealpha=0, labelcolor=[ALGO_C, HUM_C])

# Brain axis style
ax_brain.set_facecolor(PANEL)
ax_brain.set_axis_off()
ax_brain.set_title("Active Neural Signature", color=GOLD,
                    fontsize=9.5, fontfamily="monospace", pad=4)

# Artists
line_algo,  = ax_pnl.plot([], [], color=ALGO_C, lw=2.0, zorder=5)
line_human, = ax_pnl.plot([], [], color=HUM_C,  lw=2.0, zorder=4,
                           linestyle="--", alpha=0.9)
day_vline = ax_pnl.axvline(0, color="#FFFFFF", alpha=0.15, linewidth=0.6)

# Info panel texts
info_sc   = ax_info.text(0.5, 0.82, "", ha="center", va="top",
                          transform=ax_info.transAxes, fontsize=9,
                          fontweight="bold", color=GOLD, fontfamily="monospace",
                          wrap=True)
info_sub  = ax_info.text(0.5, 0.68, "", ha="center", va="top",
                          transform=ax_info.transAxes, fontsize=7.5,
                          color=LABEL_C, fontfamily="monospace", wrap=True)
info_drag = ax_info.text(0.5, 0.50, "", ha="center", va="top",
                          transform=ax_info.transAxes, fontsize=8,
                          color=HUM_C, fontfamily="monospace")
info_day  = ax_info.text(0.5, 0.36, "", ha="center", va="top",
                          transform=ax_info.transAxes, fontsize=7.5,
                          color="#506070", fontfamily="monospace")
info_pnl_a= ax_info.text(0.5, 0.22, "", ha="center", va="top",
                          transform=ax_info.transAxes, fontsize=8,
                          color=ALGO_C, fontfamily="monospace")
info_pnl_h= ax_info.text(0.5, 0.10, "", ha="center", va="top",
                          transform=ax_info.transAxes, fontsize=8,
                          color=HUM_C, fontfamily="monospace")

# Bottom attribution bar
ax_bot.text(
    0.02, 0.5,
    "Neuroscience refs: Lo & Repin (2002) · Kuhnen & Knutson (2005) · "
    "De Martino et al. (2006) · Frydman et al. (2014)",
    ha="left", va="center", fontsize=7, color="#354050",
    fontfamily="monospace", transform=ax_bot.transAxes,
)
ax_bot.text(
    0.98, 0.5,
    "\"Amygdala activation at 3σ vol event has a quantifiable PnL cost.\"",
    ha="right", va="center", fontsize=7, color="#4A5A6A",
    fontfamily="monospace", transform=ax_bot.transAxes, style="italic",
)

# ---------------------------------------------------------------------------
# Animation logic
# ---------------------------------------------------------------------------
_brain_surfs = []
_prev_sc = [None]
_prev_act = [None]


def init():
    line_algo.set_data([], [])
    line_human.set_data([], [])
    return [line_algo, line_human]


def draw_brain_state(ax, act, azim_offset=0):
    """Clears and redraws brain for current activation state."""
    ax.cla()
    ax.set_facecolor(PANEL)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-110, 110)
    ax.set_zlim(-85, 85)
    ax.set_axis_off()
    ax.set_title("Active Neural Signature", color=GOLD,
                  fontsize=9.5, fontfamily="monospace", pad=4)

    # Brain shell
    ax.plot_surface(_BX, _BY, _BZ, color="#707080", alpha=0.09,
                    linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(_BX, _BY, _BZ, color="#FFFFFF", alpha=0.025,
                      linewidth=0.3, rstride=7, cstride=7)

    # Region spheres
    for key, (cx, cy, cz, r, _) in BRAIN_REGIONS.items():
        a = act.get(key, 0.0)
        if a < 0.04:
            continue
        col = REGION_COLOURS[key]
        sx, sy, sz = sphere(cx, cy, cz, r * (0.55 + 0.45 * a))
        ax.plot_surface(sx, sy, sz, color=col,
                        alpha=min(0.12 + 0.80 * a, 0.95),
                        linewidth=0, antialiased=True, shade=False)
        if a > 0.55:
            sx2, sy2, sz2 = sphere(cx, cy, cz, r * (1.15 + 0.45 * a))
            ax.plot_surface(sx2, sy2, sz2, color=col,
                            alpha=0.06 * a,
                            linewidth=0, antialiased=True, shade=False)

    ax.view_init(elev=18, azim=-62 + azim_offset)


_frame_counter = [0]


def animate(frame_idx):
    day = frame_days[frame_idx]
    d   = day + 1

    # Current scenario
    sc_key = day_to_scenario(day)
    sc     = TRADING_SCENARIOS[sc_key]
    cur_act = scenario_act(sc_key)

    # Smooth transition
    if _prev_sc[0] != sc_key and _prev_act[0] is not None:
        t = min(_frame_counter[0] / 10, 1.0)
        disp_act = lerp_act(_prev_act[0], cur_act, t)
        _frame_counter[0] += 1
    else:
        disp_act = cur_act
        _frame_counter[0] = 0

    if _prev_sc[0] != sc_key:
        _prev_sc[0] = sc_key
        _prev_act[0] = cur_act

    # Slow azimuth rotation
    azim = (frame_idx * 0.3) % 360

    draw_brain_state(ax_brain, disp_act, azim_offset=azim * 0.15)

    # PnL lines
    line_algo.set_data(days[:d], algo_cum[:d])
    line_human.set_data(days[:d], human_cum[:d])
    day_vline.set_xdata([d, d])

    # Info panel
    drag = sc["pnl_drag_bps"]
    drag_str = f"Bias cost: {drag:+d} bps / event" if drag else "Bias cost: negligible"
    info_sc.set_text(sc["label"])
    info_sub.set_text(sc["subtitle"])
    info_drag.set_text(drag_str)
    info_day.set_text(f"Day {d} / {TRADING_DAYS}")
    info_pnl_a.set_text(f"Algo   ${algo_cum[d-1]:,.0f}")
    info_pnl_h.set_text(f"Human  ${human_cum[d-1]:,.0f}")

    return [line_algo, line_human, day_vline,
            info_sc, info_sub, info_drag, info_day,
            info_pnl_a, info_pnl_h]


print(f"Rendering master animation: {TOTAL_FRAMES} frames @ {FPS} fps …")
ani = animation.FuncAnimation(
    fig, animate, frames=TOTAL_FRAMES,
    init_func=init, interval=1000 // FPS, blit=False,
)

writer = animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                extra_args=["-pix_fmt", "yuv420p"])
out_path = os.path.join(OUTPUT_DIR, "master_neuro_quant.mp4")
ani.save(out_path, writer=writer, dpi=110)
plt.close(fig)
print(f"Saved → {out_path}")
