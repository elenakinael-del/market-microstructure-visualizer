"""
ACTION POTENTIAL ORDER BOOK  ·  fMRI BOLD RESPONSE STYLE
==========================================================
Visual concept:
  - A 3D membrane tube (like an axon cross-section in neuroimaging)
  - VPIN toxicity = depolarization mapped as a heat field on the tube surface
  - When threshold fires: a neon BOLD pulse (like fMRI activation blobs)
    travels along the tube's time axis in 3D
  - Below: the real-time IV skew surface deforms as the AP hits
  - Right panel: price chart with BOLD-style event markers

Aesthetic:
  - #010108 background (MRI dark room)
  - Membrane: deep blue-violet axon shell
  - Depolarization: cold→hot spectral (like brain activation maps)
  - AP wavefront: white-core cyan halo (like a spreading cortical wave)
  - Fiber ambient glow via alpha-blended scatter rings

Requirements:
    pip install numpy matplotlib scipy pillow

Run:
    python ap_bold_orderbook.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
#  PALETTE  (fMRI/neuroimaging)
# ══════════════════════════════════════════════════════════════════════
BG          = "#010108"
TEXT_COL    = "#a0b8d8"
DIM_COL     = "#202840"
CMAP_BOLD   = mcolors.LinearSegmentedColormap.from_list(
    "bold", ["#000080","#0050d0","#00a8e0","#00f0c0",
             "#80ff40","#ffdd00","#ff6600","#ff0020","#ffffff"])

CMAP_VOL    = mcolors.LinearSegmentedColormap.from_list(
    "vol", ["#03045e","#023e8a","#0077b6","#7209b7","#e63946","#ffba08"])

CMAP_AXON   = mcolors.LinearSegmentedColormap.from_list(
    "axon", ["#090020","#1a0050","#2a00a0","#5020e0","#9060ff"])

# ══════════════════════════════════════════════════════════════════════
#  3-D AXON TUBE GEOMETRY
# ══════════════════════════════════════════════════════════════════════
N_ANG   = 64     # circumferential resolution
N_Z     = 90     # axial (time) resolution — price levels
N_FRAMES= 200
FIRE_F  = 65     # frame the AP fires
THRESH  = 0.68   # VPIN threshold

theta   = np.linspace(0, 2*np.pi, N_ANG)
z_ax    = np.linspace(-1, 1, N_Z)         # normalized axial position
THETA_G, Z_G = np.meshgrid(theta, z_ax)

RADIUS  = 1.0    # base axon radius
X_TUBE  = RADIUS * np.cos(THETA_G)
Y_TUBE  = RADIUS * np.sin(THETA_G)
Z_TUBE  = Z_G * 3.5                       # stretch axially

# ══════════════════════════════════════════════════════════════════════
#  VPIN TOXICITY TRACE
# ══════════════════════════════════════════════════════════════════════
def vpin_trace(n_frames, fire_f, thresh):
    v = np.zeros(n_frames)
    for i in range(fire_f):
        v[i] = thresh * (i/fire_f)**1.6 + np.random.normal(0, 0.018)
    v[:fire_f] = np.clip(v[:fire_f], 0, 1)
    for i in range(fire_f, n_frames):
        t = i - fire_f
        v[i] = max(0, thresh * np.exp(-t/22) + np.random.normal(0, 0.008))
    return v

VPIN = vpin_trace(N_FRAMES, FIRE_F, THRESH)

# ══════════════════════════════════════════════════════════════════════
#  MEMBRANE VOLTAGE FIELD ON TUBE SURFACE
# ══════════════════════════════════════════════════════════════════════
def membrane_field(frame, vpin_val):
    """
    Returns color values (0–1) for the tube surface.
    Pre-fire: gradient depolarization spreading from center.
    Post-fire: travelling AP wavefront (Gaussian blob moving along Z).
    """
    # axial coordinate 0→1
    z_norm = (Z_G + 1) / 2

    if frame < FIRE_F:
        # depolarization builds from center outward
        ramp = vpin_val
        field = ramp * np.exp(-((z_norm - 0.5)**2) / (0.15 + 0.6*ramp))
        noise = np.random.normal(0, 0.025, field.shape)
        return np.clip(field + noise * ramp, 0, 1)
    else:
        t  = (frame - FIRE_F) / (N_FRAMES - FIRE_F)
        # AP wavefront travels from center toward both ends
        forward  = np.exp(-((z_norm - (0.5 + t*0.9))**2) / 0.015)
        backward = np.exp(-((z_norm - (0.5 - t*0.5))**2) / 0.025) * 0.4
        refractory = np.clip(z_norm - (0.5 + t*0.9 - 0.12), 0, 0.18) / 0.18
        wave = forward + backward - refractory * 0.5
        noise = np.random.normal(0, 0.015, wave.shape)
        return np.clip(wave + noise, 0, 1)

# Radius modulation: axon swells slightly at AP wavefront
def radius_mod(frame):
    if frame < FIRE_F:
        return np.ones((N_Z, N_ANG)) * RADIUS
    t      = (frame - FIRE_F) / (N_FRAMES - FIRE_F)
    z_norm = (Z_G + 1) / 2
    swell  = 0.18 * np.exp(-((z_norm - (0.5 + t*0.9))**2) / 0.018)
    return RADIUS + swell

# ══════════════════════════════════════════════════════════════════════
#  VOL SKEW SURFACE
# ══════════════════════════════════════════════════════════════════════
SK  = np.linspace(0.75, 1.25, 30)
TN  = np.linspace(0.03, 1.5, 30)
SG, TG = np.meshgrid(SK, TN)

def vol_surface(frame):
    v   = VPIN[frame]
    atm = 0.15 + 0.25*v
    iv  = atm - 0.10*(SG-1) + 0.12*(SG-1)**2 + 0.05/np.sqrt(TG+0.01)
    if frame >= FIRE_F:
        t   = frame - FIRE_F
        sp  = 0.28 * np.exp(-((SG-0.95)**2)/0.03) * np.exp(-TG/0.4)
        iv += sp * min(1.0, t/30)
    return np.clip(iv, 0.01, 0.70)

# ══════════════════════════════════════════════════════════════════════
#  GOLD PRICE
# ══════════════════════════════════════════════════════════════════════
PRICE = np.cumsum(np.random.normal(0, 0.25, N_FRAMES))
for i in range(FIRE_F, N_FRAMES):
    t = i - FIRE_F
    PRICE[i] += 3.2 * np.exp(-t/8) * (1 - np.exp(-t/3))
PRICE = (PRICE - PRICE.min()) / ((PRICE.max() - PRICE.min()) + 1e-9)

# ══════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9), facecolor=BG)

fig.text(0.5, 0.969,
    "ACTION POTENTIAL ORDER BOOK  ·  VPIN DEPOLARIZATION → IV SPIKE",
    ha="center", color=TEXT_COL, fontsize=12,
    fontweight="bold", fontfamily="monospace")
fig.text(0.5, 0.950,
    "VPIN toxic flow accumulation  ·  threshold crossing  ·  spreading cortical wave",
    ha="center", color=DIM_COL, fontsize=7.5, fontfamily="monospace")

ax_axon  = fig.add_axes([0.01, 0.07, 0.50, 0.86], projection="3d")
ax_vol   = fig.add_axes([0.54, 0.40, 0.42, 0.52], projection="3d")
ax_price = fig.add_axes([0.56, 0.07, 0.40, 0.27])

# ── Axon axis ─────────────────────────────────────────────────────
ax_axon.set_facecolor(BG)
for pane in [ax_axon.xaxis.pane, ax_axon.yaxis.pane, ax_axon.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor("#05050f")
ax_axon.grid(False); ax_axon.set_axis_off()
lim = 1.4
ax_axon.set_xlim(-lim, lim); ax_axon.set_ylim(-lim, lim); ax_axon.set_zlim(-4, 4)
ax_axon.view_init(elev=18, azim=-55)

ax_axon.text2D(0.5, 0.005,
    "AXON MEMBRANE  ·  Price Levels = Membrane Segments  ·  VPIN = Ion Buildup",
    transform=ax_axon.transAxes, ha="center", va="bottom",
    color=DIM_COL, fontsize=6.5, fontfamily="monospace")

# ── Vol axis ───────────────────────────────────────────────────────
ax_vol.set_facecolor(BG)
for pane in [ax_vol.xaxis.pane, ax_vol.yaxis.pane, ax_vol.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor("#08081a")
ax_vol.grid(False)
ax_vol.tick_params(colors=DIM_COL, labelsize=5.5)
ax_vol.set_xlabel("Strike", color=DIM_COL, fontsize=6, labelpad=1)
ax_vol.set_ylabel("Tenor", color=DIM_COL, fontsize=6, labelpad=1)
ax_vol.set_zlabel("IV",    color=DIM_COL, fontsize=6, labelpad=1)
ax_vol.set_title("GOLD IMPLIED VOLATILITY SURFACE",
    color=DIM_COL, fontsize=7, fontfamily="monospace", pad=2)
ax_vol.view_init(elev=30, azim=-50)

# ── Price axis ──────────────────────────────────────────────────────
ax_price.set_facecolor(BG)
ax_price.tick_params(colors=TEXT_COL, labelsize=7)
for sp in ax_price.spines.values(): sp.set_edgecolor("#05050f")
ax_price.set_xlim(0, N_FRAMES); ax_price.set_ylim(-0.08, 1.3)
ax_price.set_xlabel("Time →", color=DIM_COL, fontsize=7)
ax_price.set_ylabel("XAU / USD  (norm.)", color=TEXT_COL, fontsize=7)
ax_price.set_title("GOLD PRICE  ·  POST-AP DISCHARGE",
    color=DIM_COL, fontsize=7, fontfamily="monospace", pad=3)

# ══════════════════════════════════════════════════════════════════════
#  INITIAL PLOT OBJECTS
# ══════════════════════════════════════════════════════════════════════
field0   = membrane_field(0, VPIN[0])
R_mod    = radius_mod(0)
Xm = R_mod * np.cos(THETA_G)
Ym = R_mod * np.sin(THETA_G)

tube_surf = [ax_axon.plot_surface(
    Xm, Ym, Z_TUBE, facecolors=CMAP_BOLD(field0),
    alpha=0.92, linewidth=0, antialiased=True, shade=True,
)]

# Interior glow rings (thin cross-sections at multiple z)
glow_rings = []
for z_pos in np.linspace(-3.0, 3.0, 12):
    ring_th = np.linspace(0, 2*np.pi, 80)
    rg      = 0.95
    gx      = rg * np.cos(ring_th)
    gy      = rg * np.sin(ring_th)
    gz      = np.full_like(ring_th, z_pos)
    ring,   = ax_axon.plot(gx, gy, gz, color="#1a0060", alpha=0.15, lw=0.6)
    glow_rings.append((z_pos, ring))

vol_surf_obj = [ax_vol.plot_surface(
    SG, TG, vol_surface(0), cmap=CMAP_VOL,
    alpha=0.90, linewidth=0, antialiased=True)]

price_line, = ax_price.plot([], [], color="#ffe566", lw=2.0, zorder=3)
price_fill  = [None]
fire_vline  = ax_price.axvline(-1, color="#ff3030", lw=0.9, ls=":", alpha=0)
thresh_line = ax_price.axhline(-1, color="#5050ff", lw=0.5, ls="--", alpha=0)

# ── Status overlays ──────────────────────────────────────────────
vpin_txt = fig.text(
    0.255, 0.908, "VPIN  0.000",
    ha="center", color="#2040a0", fontsize=10,
    fontfamily="monospace", fontweight="bold",
)
state_txt = fig.text(
    0.255, 0.887, "RESTING  ·  sub-threshold  ·  ion channels closed",
    ha="center", color=DIM_COL, fontsize=8, fontfamily="monospace",
)

# VPIN bar (left side indicator)
vpin_bar_bg = fig.add_axes([0.005, 0.12, 0.012, 0.72])
vpin_bar_bg.set_facecolor("#04040c")
vpin_bar_bg.set_xlim(0,1); vpin_bar_bg.set_ylim(0,1)
vpin_bar_bg.axis("off")
vbar_fill = vpin_bar_bg.fill_betweenx([0, 0], 0, 1, color=CMAP_BOLD(0.3), alpha=0.6)
vpin_bar_bg.axhline(THRESH, color="#5050ff", lw=1.0, ls="--", alpha=0.7)
vpin_bar_bg.text(0.5, THRESH+0.03, "THRESHOLD", ha="center", va="bottom",
                 color="#5050ff", fontsize=4.5, rotation=0,
                 transform=vpin_bar_bg.transData)
vpin_bar_bg.text(0.5, -0.05, "VPIN", ha="center", color=DIM_COL,
                 fontsize=5, transform=vpin_bar_bg.transData)
vbar_rect = [None]

# ══════════════════════════════════════════════════════════════════════
#  ANIMATION
# ══════════════════════════════════════════════════════════════════════
fired = [False]

def update(frame):
    vpin_val = VPIN[frame]

    # ── Tube surface ──────────────────────────────────────────────
    field   = membrane_field(frame, vpin_val)
    R_mod   = radius_mod(frame)
    Xm = R_mod * np.cos(THETA_G)
    Ym = R_mod * np.sin(THETA_G)
    for s in tube_surf: s.remove()
    tube_surf[0] = ax_axon.plot_surface(
        Xm, Ym, Z_TUBE, facecolors=CMAP_BOLD(field),
        alpha=0.92, linewidth=0, antialiased=True, shade=True,
    )

    # ── Glow rings ────────────────────────────────────────────────
    if frame >= FIRE_F:
        t_since = frame - FIRE_F
        wave_z  = (t_since / (N_FRAMES - FIRE_F)) * 7.0 - 3.5
        for (z_pos, ring) in glow_rings:
            dist = abs(z_pos - wave_z)
            a    = max(0.05, 0.7 * np.exp(-dist**2 / 0.8))
            brightness = min(1.0, a * 2)
            ring.set_color((brightness*0.1, brightness*0.5, brightness*1.0))
            ring.set_alpha(a)
            ring.set_linewidth(0.6 + 2.5*a)
    else:
        for (z_pos, ring) in glow_rings:
            ring.set_color("#1a0060")
            ring.set_alpha(0.10 + 0.08*vpin_val)

    # Camera slow orbit
    az = -55 + frame * 0.18
    ax_axon.view_init(elev=18, azim=az)

    # ── Vol surface ───────────────────────────────────────────────
    for s in vol_surf_obj: s.remove()
    vol_surf_obj[0] = ax_vol.plot_surface(
        SG, TG, vol_surface(frame),
        cmap=CMAP_VOL, alpha=0.90, linewidth=0, antialiased=True,
    )

    # ── Price ─────────────────────────────────────────────────────
    xs = np.arange(frame+1)
    price_line.set_data(xs, PRICE[:frame+1])
    if price_fill[0]: price_fill[0].remove()
    price_fill[0] = ax_price.fill_between(
        xs, PRICE[:frame+1], alpha=0.12, color="#ffe566")
    if frame >= FIRE_F and not fired[0]:
        fired[0] = True
        fire_vline.set_xdata([FIRE_F])
        fire_vline.set_alpha(0.8)

    # ── VPIN bar ─────────────────────────────────────────────────
    for col in list(vpin_bar_bg.collections):
        col.remove()
    vpin_bar_bg.fill_betweenx(
        [0, min(vpin_val, 1)], 0, 1,
        color=CMAP_BOLD(min(vpin_val, 1)), alpha=0.75)
    vpin_bar_bg.axhline(THRESH, color="#5050ff", lw=1.0, ls="--", alpha=0.7)

    # ── Text ──────────────────────────────────────────────────────
    vpin_txt.set_text(f"VPIN  {vpin_val:.3f}")
    pct = vpin_val / THRESH
    if frame >= FIRE_F:
        vpin_txt.set_color("#ff4040")
        state_txt.set_text("⚡  ACTION POTENTIAL  ·  depolarization wave propagating")
        state_txt.set_color("#ff6020")
    elif pct > 0.88:
        vpin_txt.set_color("#ff8000")
        state_txt.set_text(f"⚠  NEAR THRESHOLD  ·  {pct*100:.0f}% depolarized")
        state_txt.set_color("#ff8020")
    else:
        col_pct = pct
        r = int(20 + 180*col_pct); g = int(40 + 80*col_pct); b = int(160)
        vpin_txt.set_color(f"#{r:02x}{g:02x}{b:02x}")
        state_txt.set_text(f"RESTING  ·  {pct*100:.0f}% depolarized  ·  ion channels building")
        state_txt.set_color(DIM_COL)

    return [price_line, fire_vline]

print("Building animation …")
ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=40, blit=False)

try:
    writer = FFMpegWriter(fps=25, bitrate=3200,
                          metadata={"title": "AP Order Book BOLD"})
    ani.save("ap_bold_orderbook.mp4", writer=writer, dpi=160,
             savefig_kwargs={"facecolor": BG})
    print("✓  Saved: ap_bold_orderbook.mp4")
except Exception as e:
    print(f"MP4 skipped ({e}) — install ffmpeg to export.")

plt.show()
