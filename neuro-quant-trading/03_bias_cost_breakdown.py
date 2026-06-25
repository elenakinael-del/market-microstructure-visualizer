"""
03_bias_cost_breakdown.py
--------------------------
Static publication-quality figure:
  Left  – Radar/spider of brain region activation per scenario
  Right – Waterfall of PnL drag by bias type
  Bottom – Brain region activity heatmap (scenarios × regions)

Output: outputs/bias_cost_breakdown.png  +  .pdf

Designed for inclusion in a fund presentation or SSRN appendix.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from brain_config import BRAIN_REGIONS, TRADING_SCENARIOS, REGION_COLOURS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
SCENARIOS = list(TRADING_SCENARIOS.keys())
REGION_KEYS = list(BRAIN_REGIONS.keys())

# Unique region labels (merge L/R)
REGION_LABELS = {
    "amygdala":       ["amygdala_L",       "amygdala_R"],
    "vent_striatum":  ["ventral_striatum_L","ventral_striatum_R"],
    "dlPFC":          ["dlPFC_L",           "dlPFC_R"],
    "ant_insula":     ["anterior_insula_L", "anterior_insula_R"],
    "ACC":            ["ACC"],
    "vmPFC":          ["vmPFC"],
    "OFC":            ["OFC_L",             "OFC_R"],
    "hippocampus":    ["hippocampus_L",     "hippocampus_R"],
}
REG_KEYS   = list(REGION_LABELS.keys())
REG_COLOURS = {
    "amygdala":      "#E84040",
    "vent_striatum": "#F5A623",
    "dlPFC":         "#4A90D9",
    "ant_insula":    "#9B59B6",
    "ACC":           "#2ECC71",
    "vmPFC":         "#1ABC9C",
    "OFC":           "#F39C12",
    "hippocampus":   "#BDC3C7",
}

def avg_act(sc_key, reg_key):
    sc = TRADING_SCENARIOS[sc_key]
    keys = REGION_LABELS[reg_key]
    vals = [sc["activations"].get(k, 0.0) for k in keys]
    return np.mean(vals)

# Matrix: scenarios × regions
act_matrix = np.array([[avg_act(s, r) for r in REG_KEYS] for s in SCENARIOS])

# PnL drag
drag_bps = np.array([TRADING_SCENARIOS[s]["pnl_drag_bps"] for s in SCENARIOS])
sc_labels = [TRADING_SCENARIOS[s]["label"] for s in SCENARIOS]
sc_colours = [TRADING_SCENARIOS[s]["colour"] for s in SCENARIOS]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
BG = "#0A0A0F"
PANEL = "#0D0D18"
SPINE = "#252535"
TICK_C = "#607080"
LABEL_C = "#8899AA"
GOLD = "#D4AF37"

fig = plt.figure(figsize=(18, 12), facecolor=BG)
gs = gridspec.GridSpec(
    2, 2,
    height_ratios=[1.15, 0.85],
    width_ratios=[1, 1],
    hspace=0.38, wspace=0.30,
    figure=fig,
    left=0.06, right=0.97, top=0.94, bottom=0.07,
)

ax_radar  = fig.add_subplot(gs[0, 0], polar=True)
ax_wf     = fig.add_subplot(gs[0, 1])
ax_heat   = fig.add_subplot(gs[1, :])

fig.text(
    0.5, 0.974,
    "Neuroeconomic Cost Atlas  ·  Gold Futures Trading  ·  Human vs Algo",
    ha="center", va="top", fontsize=14, fontweight="bold",
    color=GOLD, fontfamily="monospace",
)
fig.text(
    0.5, 0.958,
    "Lo & Repin (2002)  ·  Kuhnen & Knutson (2005)  ·  De Martino et al. (2006)",
    ha="center", va="top", fontsize=8, color="#505060", fontfamily="monospace",
)

# ── Radar ────────────────────────────────────────────────────────────────
N = len(REG_KEYS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax_radar.set_facecolor(PANEL)
ax_radar.spines["polar"].set_color(SPINE)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(
    REG_KEYS, color=LABEL_C, fontsize=7.5, fontfamily="monospace"
)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                          color="#404050", fontsize=6)
ax_radar.set_ylim(0, 1)
ax_radar.grid(color=SPINE, linewidth=0.6)
ax_radar.set_title("Region Activation by Scenario", color=GOLD,
                    fontsize=10, fontfamily="monospace", pad=14)

for i, (sc_key, col, lbl) in enumerate(zip(SCENARIOS, sc_colours, sc_labels)):
    vals = [avg_act(sc_key, r) for r in REG_KEYS]
    vals += vals[:1]
    ax_radar.plot(angles, vals, color=col, lw=1.4, alpha=0.9)
    ax_radar.fill(angles, vals, color=col, alpha=0.08)

# Legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], color=sc_colours[i], lw=2, label=sc_labels[i])
    for i in range(len(SCENARIOS))
]
ax_radar.legend(
    handles=legend_elems,
    loc="upper right",
    bbox_to_anchor=(1.38, 1.18),
    fontsize=7,
    framealpha=0,
    labelcolor=LABEL_C,
)

# ── Waterfall / Bar: PnL drag ─────────────────────────────────────────────
ax_wf.set_facecolor(PANEL)
for sp in ax_wf.spines.values():
    sp.set_color(SPINE)
ax_wf.tick_params(colors=TICK_C, labelsize=8)

bar_colours = [c if d < 0 else "#2ECC71" for c, d in zip(sc_colours, drag_bps)]
bars = ax_wf.barh(
    range(len(SCENARIOS)), drag_bps,
    color=bar_colours, alpha=0.82, height=0.55, zorder=3,
)
ax_wf.set_yticks(range(len(SCENARIOS)))
ax_wf.set_yticklabels(sc_labels, color=LABEL_C,
                       fontsize=8.5, fontfamily="monospace")
ax_wf.set_xlabel("PnL drag per event  (bps)", color=LABEL_C,
                  fontsize=8.5, fontfamily="monospace")
ax_wf.axvline(0, color=SPINE, linewidth=0.8)
ax_wf.grid(axis="x", color="#1A1A2E", linewidth=0.6, linestyle="--")
ax_wf.set_title("PnL Cost per Behavioural Scenario", color=GOLD,
                 fontsize=10, fontfamily="monospace")

for bar, val in zip(bars, drag_bps):
    ax_wf.text(
        val - 3 if val < 0 else val + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+d} bps",
        va="center",
        ha="right" if val < 0 else "left",
        fontsize=8, color="#CCDDEE", fontfamily="monospace",
    )

# ── Heatmap ───────────────────────────────────────────────────────────────
ax_heat.set_facecolor(PANEL)
for sp in ax_heat.spines.values():
    sp.set_color(SPINE)

im = ax_heat.imshow(
    act_matrix,
    aspect="auto",
    cmap="hot",
    vmin=0, vmax=1,
    interpolation="nearest",
)

ax_heat.set_xticks(range(len(REG_KEYS)))
ax_heat.set_xticklabels(REG_KEYS, color=LABEL_C,
                         fontsize=8.5, fontfamily="monospace", rotation=28, ha="right")
ax_heat.set_yticks(range(len(SCENARIOS)))
ax_heat.set_yticklabels(sc_labels, color=LABEL_C,
                         fontsize=8.5, fontfamily="monospace")
ax_heat.set_title(
    "Activation Heatmap:  Scenario × Region  "
    "(0 = inactive, 1 = maximal activation)",
    color=GOLD, fontsize=10, fontfamily="monospace",
)

# Annotate cells
for i in range(len(SCENARIOS)):
    for j in range(len(REG_KEYS)):
        v = act_matrix[i, j]
        ax_heat.text(
            j, i, f"{v:.2f}",
            ha="center", va="center",
            fontsize=6.5, fontfamily="monospace",
            color="#FFFFFF" if v > 0.45 else "#606070",
        )

cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                    fraction=0.012, pad=0.01)
cbar.ax.tick_params(colors=TICK_C, labelsize=7)
cbar.set_label("Activation level", color=LABEL_C, fontsize=7,
               fontfamily="monospace")

# ── Save ──────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUTPUT_DIR, f"bias_cost_breakdown.{ext}")
    fig.savefig(out, dpi=150 if ext == "png" else 90,
                bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}")

plt.close(fig)
