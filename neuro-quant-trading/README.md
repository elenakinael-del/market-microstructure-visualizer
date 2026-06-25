# Neuro-Quant: Neural Architecture of Trading Decisions
### Gold Futures — Human Trader vs Systematic Algo

---

## Concept

> *"Here is your brain sabotaging your returns. Here is what removing it does.
> The algo brain isn't 'better' — it's the absence of the wrong signals at the wrong time."*

This project renders the neuroeconomic cost of human bias in live gold futures trading.
It combines 3D neuroimaging-style visualisation with quantitative PnL attribution,
framed for a quant / fund audience.

---

## Files

| Script | Output | Time |
|---|---|---|
| `brain_config.py` | shared config — no output | — |
| `01_brain_activation_atlas.py` | `outputs/brain_activation_atlas.mp4` | ~2 min |
| `02_pnl_human_vs_algo.py` | `outputs/pnl_human_vs_algo.mp4` | ~1 min |
| `03_bias_cost_breakdown.py` | `outputs/bias_cost_breakdown.png/.pdf` | ~5 s |
| `04_master_animation.py` | `outputs/master_neuro_quant.mp4` | ~4 min |
| `05_interactive_brain_plotly.py` | `outputs/interactive_brain.html` | ~10 s |
| `run_all.py` | runs all above in order | ~8 min |

---

## Setup

```bash
# 1. Clone / copy project folder into VSCode
# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run everything
python run_all.py

# Or run fast outputs first
python 03_bias_cost_breakdown.py
python 05_interactive_brain_plotly.py
```

### FFmpeg (required for MP4 output)
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org and add to PATH
```

---

## Outputs

### `master_neuro_quant.mp4`
The centrepiece: 3D transparent brain (left) with activating regions + live PnL build (right).
Slow azimuth rotation. Scenario transitions are interpolated.

### `brain_activation_atlas.mp4`
Human vs Algo side-by-side across 10 trading scenarios. Regions light up proportionally to
activation strength. Halo glow on high-activation regions.

### `pnl_human_vs_algo.mp4`
Year-to-date PnL curves building day by day. Bias events annotated with cost in bps.
Bottom strip shows per-event cost bars.

### `bias_cost_breakdown.png/.pdf`
Publication-quality static figure. Three panels:
- Radar chart: region activation profile per scenario
- Horizontal waterfall: PnL drag per bias event
- Heatmap: scenario × region activation matrix

### `interactive_brain.html`
Plotly interactive 3D brain. Hover over any region to see:
- Region name + neuroscience function
- Activation level for the active scenario
- PnL cost attribution

Use the dropdown to switch scenarios. Works in any browser.
Embed directly in a fund presentation or SSRN appendix.

---

## Brain Regions & Trading Psychology Map

| Region | Function | Trading Manifestation |
|---|---|---|
| Amygdala | Fear / threat detection | Panic exit, stop-hunt sensitivity |
| Ventral Striatum | Reward anticipation | FOMO entries, position oversizing |
| dlPFC | Executive control | Rule-following, plan adherence |
| Anterior Insula | Risk / disgust aversion | Risk-off bias, early profit-taking |
| ACC | Conflict monitoring | Hesitation, missed entries |
| vmPFC | Value signal integration | Position sizing quality |
| OFC | Expected value computation | Entry timing |
| Hippocampus | Memory / pattern recall | Recency bias, anchoring |

---

## Neuroscience References

- Lo, A. W., & Repin, D. V. (2002). The psychophysiology of real-time financial risk processing. *Journal of Cognitive Neuroscience, 14*(3).
- Kuhnen, C. M., & Knutson, B. (2005). The neural basis of financial risk taking. *Neuron, 47*(5), 763–770.
- De Martino, B., Kumaran, D., Seymour, B., & Dolan, R. J. (2006). Frames, biases, and rational decision-making in the human brain. *Science, 313*(5787).
- Frydman, C., Barberis, N., Camerer, C., Bossaerts, P., & Rangel, A. (2014). Using neural data to test a theory of investor behavior. *Journal of Finance, 69*(2).

---

## Design Notes

All colours, backgrounds, and typography are set by VSCode / matplotlib defaults
where unspecified. The dark `#060610` / `#0A0A1A` palette was chosen because
quant research figures are routinely rendered dark for screen-first presentations.
The gold `#D4AF37` accent is the only editorial colour choice — it references the
underlying instrument.
