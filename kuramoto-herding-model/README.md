# Kuramoto Herding Cascade — Gold Market Behavioral Synchronization

Visualizes market herding dynamics in gold using the **Kuramoto coupled oscillator model**. Models traders as phase oscillators whose synchronization — driven by realized volatility stress — produces the classic "herding cascade" pattern seen in momentum crashes.

## Concept

> As volatility stress rises, individual traders' "sentiment phases" synchronize.  
> The Kuramoto order parameter r(t) → 1 signals full herding.  
> r(t) → 0 signals dispersed, independent sentiment.

This is a behavioral quant model grounded in physics: the same mathematics that describes synchronized fireflies or power grid stability governs crowd behavior in financial markets.

## Outputs

| File | Description |
|---|---|
| `kuramoto_herding_cascade.mp4` | Main animation: oscillators + sync metric over time |
| `gold_dom_heatmap.mp4` | Gold depth-of-market heatmap animation |
| `action_potential_orderbook.mp4` | Order book modeled as neural action potential |
| `gold_quant_positioning.mp4` | COT positioning visualization |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/kuramoto-herding-model.git
cd kuramoto-herding-model
pip install numpy matplotlib yfinance
# FFmpeg required for MP4 export:
brew install ffmpeg  # macOS
```

## Run

```bash
python "kuramoto herding cascade.py"
python gold_dom_heatmap.py
python "action potential orderbook.py"
```

## The Math

The Kuramoto model:
```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```
Where coupling K(t) is driven by realized volatility stress from live gold futures data.

Order parameter:
```
r(t) = |1/N Σⱼ e^{iθⱼ}|
```

## References

- Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*
- Cont, R., & Bouchaud, J.P. (2000). Herd behavior and aggregate fluctuations in financial markets. *Macroeconomic Dynamics*

---
*Built by Elena Hysa | Behavioral Finance × Physics*
