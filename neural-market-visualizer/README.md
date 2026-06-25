# Neural Market Visualizer — Order Book as Action Potential

Renders live gold futures market microstructure as **neural action potentials**. The order book depth, bid-ask spread dynamics, and trade flow are mapped to the voltage spike patterns of biological neurons — a novel visualization bridging neuroscience and market microstructure.

## Visualizations

| File | Description |
|---|---|
| `ap_bold_orderbook.mp4` | Order book rendered as bold action potential spikes |
| `neural_diffusion_market.mp4` | Neural diffusion model of market state propagation |

## Scripts

| Script | Output |
|---|---|
| `ap_bold_orderbook.py` | Action potential order book animation |
| `neural_diffusion_market.py` | Neural diffusion market animation |

## Concept

The action potential metaphor:
- **Resting state** → thin, illiquid order book
- **Depolarization** → aggressive buying/selling building up
- **Spike** → trade execution, price jump
- **Refractory period** → post-trade consolidation

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/neural-market-visualizer.git
cd neural-market-visualizer
pip install numpy matplotlib yfinance
brew install ffmpeg  # macOS — required for MP4
```

## Run

```bash
python ap_bold_orderbook.py
python neural_diffusion_market.py
```

---
*Built by Elena Hysa | Neuroscience × Market Microstructure*
