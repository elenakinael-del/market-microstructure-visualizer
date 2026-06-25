"""
02_pnl_human_vs_algo.py
------------------------
Animated PnL build across a 252-day trading year.
Annotates each behavioural bias event with its neuroscience label
and measured PnL cost.

Output: outputs/pnl_human_vs_algo.mp4

Framing: "Here is your brain sabotaging your returns.
          Here is what removing it does."
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation, gridspec
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from brain_config import (
    TRADING_SCENARIOS, TRADING_DAYS, BASE_EDGE_BPS,
    TRADES_PER_DAY, CONTRACT_VALUE,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Simulate daily PnL
# ---------------------------------------------------------------------------
BIAS_EVENTS = [
    # (day, scenario_key, extra_cost_bps)
    (38,  "vol_spike",    -47),
    (51,  "panic_sell",   -112),
    (89,  "reward_chase", -63),
    (117, "vol_spike",    -47),
    (143, "panic_sell",   -112),
    (178, "reward_chase", -63),
    (201, "vol_spike",    -47),
    (220, "panic_sell",   -112),
]

bias_days = {ev[0]: ev for ev in BIAS_EVENTS}

# Daily base returns (bps)
base_daily = np.random.normal(BASE_EDGE_BPS * TRADES_PER_DAY, 35,
                              TRADING_DAYS)

algo_daily = base_daily.copy()

human_daily = base_daily.copy()
for day, (d, sc_key, cost) in bias_days.items():
    human_daily[d] += cost  # add bias drag

# Cumulative (convert bps to $ assuming 1 contract, ~$2000 per 100bps move)
# 1 bps on GC ~= $10 (10 ticks at $1/tick for mini, or $100 for full)
BPS_TO_USD = 10.0

algo_cum = np.cumsum(algo_daily) * BPS_TO_USD
human_cum = np.cumsum(human_daily) * BPS_TO_USD
days = np.arange(1, TRADING_DAYS + 1)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9), facecolor="#0A0A0F")
gs = gridspec.GridSpec(
    3, 2,
    height_ratios=[0.06, 0.62, 0.32],
    width_ratios=[0.72, 0.28],
    hspace=0.06, wspace=0.04,
    figure=fig,
    left=0.06, right=0.97, top=0.97, bottom=0.06,
)

ax_title = fig.add_subplot(gs[0, :])
ax_pnl   = fig.add_subplot(gs[1, 0])
ax_panel = fig.add_subplot(gs[1, 1])
ax_cost  = fig.add_subplot(gs[2, 0])
ax_blank = fig.add_subplot(gs[2, 1])

for ax in [ax_title, ax_panel, ax_blank]:
    ax.set_axis_off()

for ax in [ax_pnl, ax_cost]:
    ax.set_facecolor("#0D0D18")
    for spine in ax.spines.values():
        spine.set_color("#2A2A3A")

# Title
ax_title.text(
    0.0, 0.5,
    "Gold Futures · Human Trader vs Systematic Algo · One Year PnL",
    ha="left", va="center", fontsize=13, fontweight="bold",
    color="#D4AF37", fontfamily="monospace", transform=ax_title.transAxes,
)
ax_title.text(
    1.0, 0.5,
    "Framing: amygdala activation at 3σ vol event has a quantifiable PnL cost",
    ha="right", va="center", fontsize=8, color="#607080",
    fontfamily="monospace", transform=ax_title.transAxes,
)
ax_title.set_facecolor("#0A0A0F")

# Static axis limits
ax_pnl.set_xlim(0, TRADING_DAYS + 5)
ax_pnl.set_ylim(min(human_cum.min(), algo_cum.min()) * 1.15,
                 algo_cum.max() * 1.15)
ax_pnl.set_ylabel("Cumulative PnL  ($)", color="#8899AA",
                   fontsize=9, fontfamily="monospace")
ax_pnl.tick_params(colors="#606878", labelsize=8)
ax_pnl.grid(color="#1A1A2E", linewidth=0.6, linestyle="--")
ax_pnl.axhline(0, color="#2A2A3A", linewidth=0.8)

ax_cost.set_xlim(0, TRADING_DAYS + 5)
ax_cost.set_ylim(-130, 10)
ax_cost.set_ylabel("Daily Bias Cost  (bps)", color="#8899AA",
                    fontsize=9, fontfamily="monospace")
ax_cost.set_xlabel("Trading Day", color="#8899AA",
                    fontsize=9, fontfamily="monospace")
ax_cost.tick_params(colors="#606878", labelsize=8)
ax_cost.grid(color="#1A1A2E", linewidth=0.6, linestyle="--")
ax_cost.axhline(0, color="#2A2A3A", linewidth=0.8)

# Panel: running stats
PANEL_STATS = [
    ("Algo YTD PnL",    "#4A90D9"),
    ("Human YTD PnL",   "#E84040"),
    ("Bias cost YTD",   "#F5A623"),
    ("# Bias events",   "#9B59B6"),
]
panel_texts = []
for i, (label, col) in enumerate(PANEL_STATS):
    ax_panel.text(0.05, 0.92 - i * 0.18, label, transform=ax_panel.transAxes,
                  color="#607080", fontsize=8, fontfamily="monospace")
    t = ax_panel.text(0.05, 0.82 - i * 0.18, "—", transform=ax_panel.transAxes,
                      color=col, fontsize=13, fontweight="bold",
                      fontfamily="monospace")
    panel_texts.append(t)

ax_panel.set_facecolor("#0D0D18")
for sp in ax_panel.spines.values():
    sp.set_color("#2A2A3A")
ax_panel.set_xticks([])
ax_panel.set_yticks([])

# ---------------------------------------------------------------------------
# Animation artists
# ---------------------------------------------------------------------------
line_algo,  = ax_pnl.plot([], [], color="#4A90D9", lw=1.8, label="Algo")
line_human, = ax_pnl.plot([], [], color="#E84040", lw=1.8,
                           linestyle="--", label="Human", alpha=0.9)

ax_pnl.legend(
    loc="upper left", fontsize=8, framealpha=0,
    labelcolor=["#4A90D9", "#E84040"],
)

cost_bars_artists = []
annotation_artists = []


def init():
    line_algo.set_data([], [])
    line_human.set_data([], [])
    for t in panel_texts:
        t.set_text("—")
    return [line_algo, line_human] + panel_texts


def animate(frame):
    d = frame + 1  # current day (1-indexed)

    line_algo.set_data(days[:d], algo_cum[:d])
    line_human.set_data(days[:d], human_cum[:d])

    # Cost bars
    for art in cost_bars_artists:
        art.remove()
    cost_bars_artists.clear()
    for art in annotation_artists:
        art.remove()
    annotation_artists.clear()

    shown_days = [b[0] for b in BIAS_EVENTS if b[0] <= d]
    if shown_days:
        costs = [bias_days[dd][2] for dd in shown_days]
        bars = ax_cost.bar(
            shown_days, costs, width=2.5,
            color="#E84040", alpha=0.7, zorder=3,
        )
        cost_bars_artists.extend(bars)

        # Vertical event lines on PnL chart
        for dd in shown_days:
            vl = ax_pnl.axvline(dd, color="#E84040", alpha=0.25,
                                 linewidth=0.8, linestyle=":")
            annotation_artists.append(vl)

    # Panel stats
    bias_total = sum(bias_days[dd][2] for dd in shown_days) * BPS_TO_USD
    algo_val   = algo_cum[d - 1]
    human_val  = human_cum[d - 1]
    n_events   = len(shown_days)

    panel_texts[0].set_text(f"${algo_val:,.0f}")
    panel_texts[1].set_text(f"${human_val:,.0f}")
    panel_texts[2].set_text(f"${bias_total:,.0f}")
    panel_texts[3].set_text(str(n_events))

    return [line_algo, line_human] + panel_texts + cost_bars_artists + annotation_artists


SKIP = 2  # render every N days for speed
frame_list = list(range(0, TRADING_DAYS, SKIP))

print(f"Rendering {len(frame_list)} frames …")
ani = animation.FuncAnimation(
    fig, animate, frames=frame_list,
    init_func=init, interval=40, blit=False,
)

writer = animation.FFMpegWriter(fps=25, bitrate=2500,
                                extra_args=["-pix_fmt", "yuv420p"])
out_path = os.path.join(OUTPUT_DIR, "pnl_human_vs_algo.mp4")
ani.save(out_path, writer=writer, dpi=120)
plt.close(fig)
print(f"Saved → {out_path}")
