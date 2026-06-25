# XAUUSD Multi-Strategy Backtest — Athens Session Model

A multi-strategy backtesting framework for COMEX Gold Futures (XAUUSD), using 1-hour bars segmented by trading session (Asia, London, NY) calibrated to Athens/EET timezone.

## Strategies

| Strategy | Logic |
|---|---|
| **A** | Liquidity sweep + Break of Structure (BOS) entry |
| **B** | Volume spike (2× average) reversal |
| **C** | Session open range breakout |
| **E** | Sweep + BOS with ATR trailing stop |
| **F** | Multi-confirmation entry filter |

## Session Definitions (Athens EET/EEST)

| Session | Hours (Athens) |
|---|---|
| Asia | 02:00 – 09:00 |
| London | 09:00 – 16:00 |
| New York | 16:00 – 23:00 |

## Features

- Fetches live XAUUSD data via yfinance (GC=F)
- ATR-based trailing stops (14-period, 1.5× multiplier)
- Per-trade risk sizing ($100 risk per trade)
- Session-filtered entries
- Full results CSV export
- Win rate, expectancy, and drawdown metrics

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/xauusd-backtest.git
cd xauusd-backtest
pip install pandas numpy yfinance pytz
```

## Run

```bash
python xauusd_backtest.py
```

Results are printed to console and saved as `xauusd_backtest_results.csv`.

## Sample Output

```
Strategy A | Trades: 47 | Win Rate: 58.5% | Avg R: 1.3 | Max DD: -$340
Strategy B | Trades: 31 | Win Rate: 52.3% | Avg R: 0.9 | Max DD: -$210
...
```

## Key Parameters (edit in script)

```python
SWEEP_THRESHOLD = 0.30   # oz beyond level to qualify as sweep
VOL_MULT        = 2.0    # volume spike multiplier
TRAIL_ATR_MULT  = 1.5    # trailing stop ATR multiplier
RISK_PER_TRADE  = 100.0  # USD risk per trade
```

---
*Built by Elena Hysa | Athens-based trading session analysis*
