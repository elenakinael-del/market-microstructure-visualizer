# Neuro-Market Visualizations  ·  v2  (Neuroimaging Edition)
## Two production-grade 3D animations

---

## SCRIPTS

| File | Concept | Visual |
|------|---------|--------|
| `neural_diffusion_market.py` | Kuramoto herding → vol spike | 3D brain point cloud + fiber tracts + rotating camera |
| `ap_bold_orderbook.py` | VPIN toxicity → action potential | 3D axon tube + BOLD wave + IV surface deformation |

---

## SETUP IN VS CODE

### Step 1 — Place the scripts
Save both `.py` files into a folder, e.g. `~/Desktop/NeuroMarket/`

### Step 2 — Open in VS Code
**File → Open Folder** → select `NeuroMarket`

### Step 3 — Open Terminal
**View → Terminal** (or press  `` Ctrl+` `` )

### Step 4 — Install dependencies
```bash
pip install numpy matplotlib scipy pillow
```

### Step 5 — Run
```bash
python neural_diffusion_market.py
```
```bash
python ap_bold_orderbook.py
```

---

## MP4 EXPORT  (for Instagram)

### Install ffmpeg first:
- **macOS:**    `brew install ffmpeg`
- **Windows:**  Download from https://ffmpeg.org → add `bin/` folder to PATH
- **Linux:**    `sudo apt install ffmpeg`

Then re-run either script — it auto-saves the MP4.

### Trim to 15 seconds for IG Reels:
```bash
ffmpeg -i neural_diffusion_market.mp4 -t 15 -c copy neural_ig.mp4
ffmpeg -i ap_bold_orderbook.mp4       -t 15 -c copy ap_ig.mp4
```

---

## TROUBLESHOOTING

**"TkAgg" error on Mac:**
Change line 4 in either script:
```python
matplotlib.use("TkAgg")  →  matplotlib.use("MacOSX")
```

**Animation is slow / choppy:**
Reduce resolution at the top of the script:
```python
N_ANG = 40   # (was 64)
N_Z   = 60   # (was 90)
```

**Black window / no display:**
```bash
pip install pyobjc-framework-Cocoa   # macOS only
```

---

## KEY PARAMETERS TO CUSTOMISE

### `neural_diffusion_market.py`
| Variable | Default | Effect |
|---|---|---|
| `N_NODES` | 180 | Brain network density |
| `CASCADE_START` | 70 | Frame herding begins |
| `LOCK_FRAME` | 155 | Full phase-lock frame |
| `K_END` | 5.5 | Max herding coupling |

### `ap_bold_orderbook.py`
| Variable | Default | Effect |
|---|---|---|
| `N_ANG` | 64 | Tube smoothness |
| `N_Z` | 90 | Axial resolution |
| `FIRE_F` | 65 | AP firing frame |
| `THRESH` | 0.68 | VPIN threshold |
