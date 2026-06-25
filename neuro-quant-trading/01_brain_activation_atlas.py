"""
01_brain_activation_atlas.py
----------------------------
3D transparent brain with animated region activations across trading scenarios.
Renders human trader vs algo side-by-side.

Output: outputs/brain_activation_atlas.mp4

References:
  Lo & Repin (2002) Psychophysiology 39(6)
  Kuhnen & Knutson (2005) Neuron 47(5)
  De Martino et al. (2006) Science 313
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from brain_config import (
    BRAIN_REGIONS, REGION_COLOURS, TRADING_SCENARIOS,
    ALGO_ACTIVATIONS, SCENARIO_SEQUENCE,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Build a low-poly ellipsoid brain surface
# ---------------------------------------------------------------------------
def make_brain_surface(n=60):
    """Prolate ellipsoid with cortical folding noise."""
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)

    # base ellipsoid (x=70, y=90, z=65 mm half-axes)
    X = 70 * np.sin(U) * np.cos(V)
    Y = 90 * np.sin(U) * np.sin(V)
    Z = 65 * np.cos(U)

    # gyral noise
    rng = np.random.default_rng(7)
    noise = gaussian_filter(rng.normal(0, 5, X.shape), sigma=3)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    X += noise * X / (R + 1e-9)
    Y += noise * Y / (R + 1e-9)
    Z += noise * Z / (R + 1e-9)

    return X, Y, Z


def draw_brain(ax, alpha=0.13, colour="#CCCCCC"):
    X, Y, Z = make_brain_surface()
    ax.plot_surface(X, Y, Z, color=colour, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True)


def sphere(cx, cy, cz, r, n=20):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)
    X = cx + r * np.sin(U) * np.cos(V)
    Y = cy + r * np.sin(U) * np.sin(V)
    Z = cz + r * np.cos(U)
    return X, Y, Z


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def lerp_activations(a_prev, a_next, t):
    return {k: a_prev[k] + (a_next[k] - a_prev[k]) * t for k in a_prev}


def get_scenario_activations(scenario_key):
    sc = TRADING_SCENARIOS[scenario_key]
    base = {k: 0.0 for k in BRAIN_REGIONS}
    base.update(sc["activations"])
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FRAMES_PER_SCENE = 40
TRANSITION_FRAMES = 20
FPS = 20

scenes = SCENARIO_SEQUENCE
n_scenes = len(scenes)

fig = plt.figure(figsize=(16, 7), facecolor="#0A0A0F")
fig.patch.set_facecolor("#0A0A0F")

ax_h = fig.add_subplot(121, projection="3d", facecolor="#0A0A0F")
ax_a = fig.add_subplot(122, projection="3d", facecolor="#0A0A0F")

title_ax = fig.add_axes([0, 0.88, 1, 0.12], facecolor="#0A0A0F")
title_ax.axis("off")
title_txt = title_ax.text(
    0.5, 0.55,
    "Neural Atlas of Trading Decisions  ·  Gold Futures",
    ha="center", va="center",
    fontsize=14, fontweight="bold", color="#D4AF37",
    fontfamily="monospace",
)
subtitle_txt = title_ax.text(
    0.5, 0.10, "", ha="center", va="center",
    fontsize=10, color="#8899AA", fontfamily="monospace",
)

# Legend patches
legend_data = [
    ("#E84040", "Amygdala – Loss Aversion"),
    ("#F5A623", "Ventral Striatum – Reward Anticipation"),
    ("#4A90D9", "dlPFC – Rule-Following / Executive Control"),
    ("#9B59B6", "Anterior Insula – Risk / Disgust"),
    ("#2ECC71", "ACC – Conflict Monitor"),
    ("#1ABC9C", "vmPFC – Value Signal"),
    ("#F39C12", "OFC – Expected Value"),
    ("#BDC3C7", "Hippocampus – Memory Bias"),
]
patches = [mpatches.Patch(color=c, label=l) for c, l in legend_data]
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=4,
    fontsize=7,
    framealpha=0,
    labelcolor="#AABBCC",
    handlelength=1.2,
)

plt.tight_layout(rect=[0, 0.08, 1, 0.88])


def style_ax(ax, title):
    ax.set_facecolor("#0A0A0F")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-110, 110)
    ax.set_zlim(-85, 85)
    ax.set_axis_off()
    ax.set_title(title, color="#D4AF37", fontsize=11,
                 fontfamily="monospace", pad=6)
    ax.view_init(elev=18, azim=-65)


# Pre-build brain surface (static)
_BX, _BY, _BZ = make_brain_surface()


def render_state(ax, activations, title, angle_offset=0):
    ax.cla()
    style_ax(ax, title)

    # Transparent brain
    ax.plot_surface(_BX, _BY, _BZ, color="#808090", alpha=0.10,
                    linewidth=0, antialiased=True, shade=True)
    # Subtle wireframe gyri
    ax.plot_wireframe(_BX, _BY, _BZ, color="#FFFFFF", alpha=0.03,
                      linewidth=0.3, rstride=6, cstride=6)

    # Region spheres
    for key, (cx, cy, cz, r, _label) in BRAIN_REGIONS.items():
        act = activations.get(key, 0.0)
        if act < 0.05:
            continue
        col = REGION_COLOURS[key]
        sx, sy, sz = sphere(cx, cy, cz, r * (0.6 + 0.4 * act))
        ax.plot_surface(sx, sy, sz, color=col,
                        alpha=min(0.15 + 0.75 * act, 0.92),
                        linewidth=0, antialiased=True, shade=False)
        # Halo glow (slightly larger, lower alpha)
        if act > 0.5:
            sx2, sy2, sz2 = sphere(cx, cy, cz, r * (1.0 + 0.5 * act))
            ax.plot_surface(sx2, sy2, sz2, color=col,
                            alpha=0.08 * act,
                            linewidth=0, antialiased=True, shade=False)

    ax.view_init(elev=18, azim=-65 + angle_offset)


# ---------------------------------------------------------------------------
# Build frame list
# ---------------------------------------------------------------------------
frame_data = []   # list of (human_activations, algo_activations, scene_idx, label)

for i, scene in enumerate(scenes):
    h_act = get_scenario_activations(scene)
    # Transition frames from previous scene
    if i > 0:
        prev_h = get_scenario_activations(scenes[i - 1])
        for tf in range(TRANSITION_FRAMES):
            t = tf / TRANSITION_FRAMES
            frame_data.append((lerp_activations(prev_h, h_act, t),
                               ALGO_ACTIVATIONS, i, scene))
    for _ in range(FRAMES_PER_SCENE):
        frame_data.append((h_act, ALGO_ACTIVATIONS, i, scene))

total_frames = len(frame_data)
_azim_counter = [0]


def animate(frame_idx):
    h_act, a_act, scene_i, scene_key = frame_data[frame_idx]
    sc = TRADING_SCENARIOS[scene_key]
    drag = sc["pnl_drag_bps"]
    drag_str = f"PnL drag: {drag:+d} bps" if drag != 0 else "PnL drag: —"

    render_state(ax_h, h_act, "Human Trader", angle_offset=0)
    render_state(ax_a, a_act, "Systematic Algo", angle_offset=0)

    subtitle_txt.set_text(
        f"Scenario {scene_i + 1}/{len(scenes)}:  {sc['label']}  |  "
        f"{sc['subtitle']}  |  {drag_str}"
    )
    return []


print(f"Rendering {total_frames} frames …")
ani = animation.FuncAnimation(
    fig, animate, frames=total_frames, interval=1000 // FPS, blit=False
)

writer = animation.FFMpegWriter(fps=FPS, bitrate=3000,
                                extra_args=["-pix_fmt", "yuv420p"])
out_path = os.path.join(OUTPUT_DIR, "brain_activation_atlas.mp4")
ani.save(out_path, writer=writer, dpi=110)
plt.close(fig)
print(f"Saved → {out_path}")
