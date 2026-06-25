# Gold Psychophysical Volatility — Weber-Fechner Market Indicators

Applies the **Weber-Fechner Law** from psychophysics to gold futures volatility modeling. Human perception of stimulus intensity is logarithmic — this project encodes that into quantitative market indicators and compares algo vs human trader performance.

## Weber-Fechner in Markets

> Perception of change = ln(new_level / reference_level)

Applied to trading: a $10 gold move *feels* different at $1,800 vs $2,400. This model captures that perceptual scaling and uses it as a feature for volatility forecasting.

## Contents

| File | Description |
|---|---|
| `gold_psychophysics.py` | Main model: WF indicators + RF forecaster |
| `PsychophysicalAnimation.html` | Interactive visualization |
| `PsychophysicalVolatility.html` | Volatility surface explorer |
| `3D_Surface.html` | 3D psychophysical surface chart |

## Features

- Weber-Fechner log-ratio features from gold + macro data
- Gamma Exposure (GEX) indicator integration
- Random Forest regressor for volatility forecasting
- FRED API integration for macro features (VIX, TED spread, US10Y)
- Interactive Plotly visualizations

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/gold-psychophysical-volatility.git
cd gold-psychophysical-volatility
pip install pandas numpy yfinance scikit-learn plotly fredapi scipy
```

## Run

```bash
python gold_psychophysics.py
```

Open the `.html` files directly in any browser for interactive charts.

## Research Context

Connected to published research:
> *Forecasting Volatility in Gold Futures Contracts: HAR Models, Options-Implied Volatility and the Limits of Directional Inference*  
> SSRN, June 2026 — [ssrn.com/abstract=6978741](https://ssrn.com/abstract=6978741)

---
*Built by Elena Hysa | Psychology × Quantitative Finance*
