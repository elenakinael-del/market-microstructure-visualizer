"""
05_interactive_brain_plotly.py
-------------------------------
Fully interactive Plotly 3D brain. Hover over each region to see:
  - Region name + function
  - Activation level per scenario
  - PnL drag attribution

Generates one HTML file you can open in any browser or embed in a fund deck.
Use scenario dropdown to switch between trading states.

Output: outputs/interactive_brain.html
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from brain_config import (
    BRAIN_REGIONS, REGION_COLOURS, TRADING_SCENARIOS, ALGO_ACTIVATIONS,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Build brain ellipsoid mesh
# ---------------------------------------------------------------------------
def make_brain_mesh(n=60, seed=7):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)
    X = 70 * np.sin(U) * np.cos(V)
    Y = 90 * np.sin(U) * np.sin(V)
    Z = 65 * np.cos(U)
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(rng.normal(0, 5, X.shape), sigma=3)
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-9
    X += noise * X / R
    Y += noise * Y / R
    Z += noise * Z / R
    return X.flatten(), Y.flatten(), Z.flatten()


def make_sphere_mesh(cx, cy, cz, r, n=24):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)
    X = (cx + r * np.sin(U) * np.cos(V)).flatten()
    Y = (cy + r * np.sin(U) * np.sin(V)).flatten()
    Z = (cz + r * np.cos(U)).flatten()
    return X, Y, Z


BX, BY, BZ = make_brain_mesh()

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig = go.Figure()

# Brain shell
fig.add_trace(go.Mesh3d(
    x=BX, y=BY, z=BZ,
    alphahull=4,
    color="lightgrey",
    opacity=0.08,
    name="Brain surface",
    showlegend=False,
    hoverinfo="skip",
))

# One scatter trace per region per scenario — use updatemenus to switch
SCENARIOS = list(TRADING_SCENARIOS.keys())

# We'll build one set of traces per scenario; toggle visibility
all_traces = []

for sc_idx, sc_key in enumerate(SCENARIOS):
    sc = TRADING_SCENARIOS[sc_key]
    for reg_key, (cx, cy, cz, r, label) in BRAIN_REGIONS.items():
        act = sc["activations"].get(reg_key, 0.0)
        if act < 0.01:
            continue

        col = REGION_COLOURS[reg_key]
        sx, sy, sz = make_sphere_mesh(cx, cy, cz, r * (0.6 + 0.4 * act))

        hover = (
            f"<b>{label.replace(chr(10), ' ')}</b><br>"
            f"Scenario: {sc['label']}<br>"
            f"Activation: {act:.2f}<br>"
            f"PnL drag: {sc['pnl_drag_bps']:+d} bps<br>"
            f"<i>{sc['subtitle']}</i>"
        )

        trace = go.Mesh3d(
            x=sx, y=sy, z=sz,
            alphahull=5,
            color=col,
            opacity=0.15 + 0.70 * act,
            name=f"{reg_key} | {sc_key}",
            visible=(sc_idx == 0),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        )
        all_traces.append((sc_idx, trace))
        fig.add_trace(trace)

# Algo traces (always silent — zero activation shown as ghost)
for reg_key, (cx, cy, cz, r, label) in BRAIN_REGIONS.items():
    sx, sy, sz = make_sphere_mesh(cx, cy, cz, r * 0.45)
    fig.add_trace(go.Mesh3d(
        x=sx, y=sy, z=sz,
        alphahull=5,
        color="#2A2A3A",
        opacity=0.04,
        name=f"algo_{reg_key}",
        visible=False,
        showlegend=False,
        hoverinfo="skip",
    ))

# ---------------------------------------------------------------------------
# Dropdown buttons — one per scenario
# ---------------------------------------------------------------------------
n_traces = len(fig.data)
trace_sc_map = [sc_idx for sc_idx, _ in all_traces]
# Brain shell = trace 0, region traces = 1..len(all_traces),
# algo traces after

def make_visibility(active_sc):
    vis = [True]  # brain shell always visible
    for sc_i, _ in all_traces:
        vis.append(sc_i == active_sc)
    # algo traces hidden
    vis += [False] * len(BRAIN_REGIONS)
    return vis


buttons = []
for sc_idx, sc_key in enumerate(SCENARIOS):
    sc = TRADING_SCENARIOS[sc_key]
    drag = sc["pnl_drag_bps"]
    buttons.append(dict(
        label=sc["label"],
        method="update",
        args=[
            {"visible": make_visibility(sc_idx)},
            {"title": {
                "text": (
                    f"<b>Neural Activation Atlas  ·  Gold Futures</b><br>"
                    f"<span style='font-size:13px;color:#8899AA'>"
                    f"Scenario: {sc['label']}  |  {sc['subtitle']}"
                    f"  |  PnL drag: {drag:+d} bps</span>"
                ),
                "font": {"color": "#D4AF37"},
            }},
        ],
    ))

fig.update_layout(
    updatemenus=[dict(
        type="dropdown",
        direction="down",
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        bgcolor="#111128",
        bordercolor="#D4AF37",
        font=dict(color="#D4AF37", size=12, family="monospace"),
        buttons=buttons,
        showactive=True,
    )],
    title=dict(
        text=(
            "<b>Neural Activation Atlas  ·  Gold Futures</b><br>"
            "<span style='font-size:13px;color:#8899AA'>"
            "Hover over a region to see its function and PnL cost attribution</span>"
        ),
        font=dict(color="#D4AF37", family="monospace", size=16),
        x=0.5, xanchor="center",
    ),
    scene=dict(
        bgcolor="#060610",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        camera=dict(eye=dict(x=1.6, y=1.0, z=0.7)),
        aspectmode="cube",
    ),
    paper_bgcolor="#060610",
    plot_bgcolor="#060610",
    margin=dict(l=0, r=0, t=110, b=0),
    height=750,
)

# Colour legend annotation
annotations = []
legend_items = [
    ("#E84040", "Amygdala — Loss Aversion"),
    ("#F5A623", "Ventral Striatum — Reward"),
    ("#4A90D9", "dlPFC — Executive Control"),
    ("#9B59B6", "Anterior Insula — Risk"),
    ("#2ECC71", "ACC — Conflict Monitor"),
    ("#1ABC9C", "vmPFC — Value Signal"),
    ("#F39C12", "OFC — Expected Value"),
    ("#BDC3C7", "Hippocampus — Memory"),
]
for i, (col, lbl) in enumerate(legend_items):
    annotations.append(dict(
        x=1.01, y=0.97 - i * 0.085,
        xref="paper", yref="paper",
        text=f"<span style='color:{col}'>■</span>  {lbl}",
        showarrow=False,
        font=dict(size=10, color="#8899AA", family="monospace"),
        align="left",
    ))

annotations.append(dict(
    x=0.5, y=-0.02,
    xref="paper", yref="paper",
    text=(
        "Refs: Lo & Repin (2002) · Kuhnen & Knutson (2005) · "
        "De Martino et al. (2006)"
    ),
    showarrow=False,
    font=dict(size=9, color="#354050", family="monospace"),
))
fig.update_layout(annotations=annotations)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
out_path = os.path.join(OUTPUT_DIR, "interactive_brain.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Saved → {out_path}")
