"""
XAUUSD Multi-Strategy Backtest
Strategies: A, B, C, E, F
Timeframe: 3 months of 1H bars (Athens/EET timezone)
Data: yfinance GC=F or XAUUSD via Yahoo Finance

Athens timezone session definitions (EET = UTC+2 / EEST = UTC+3):
  Asia    : 02:00 – 09:00 Athens
  London  : 09:00 – 16:00 Athens
  NY      : 16:00 – 23:00 Athens

Run: python xauusd_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TICKER          = "GC=F"          # Gold futures; swap to "XAUUSD=X" if preferred
INTERVAL        = "1h"
LOOKBACK_DAYS   = 100             # ~3 months of trading days + buffer
SWEEP_THRESHOLD = 0.30            # 0.30 oz sweep beyond level (Strategies A & E)
VOL_MULT        = 2.0             # 2× average volume trigger (Strategy B)
VOL_LOOKBACK    = 60              # bars for average volume
MAX_BARS_BOS    = 8               # bars to wait for BOS after sweep (A & E)
MAX_BARS_RETEST = 10              # bars to wait for retest (B)
TRAIL_ATR_MULT  = 1.5             # trailing stop = ATR × multiplier
ATR_PERIOD      = 14
RISK_PER_TRADE  = 100.0           # USD risk per trade (for position sizing display)

ATHENS_TZ = pytz.timezone("Europe/Athens")

# ── SESSION TIMES (Athens hour, 24h) ──────────────────────────────────────────
ASIA_START   = 2;   ASIA_END   = 9
LONDON_START = 9;   LONDON_END = 16
NY_START     = 16;  NY_END     = 23


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    end   = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)
    print(f"Fetching {TICKER} {INTERVAL} from {start.date()} to {end.date()} …")
    df = yf.download(TICKER, start=start, end=end, interval=INTERVAL,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {TICKER}. Try 'XAUUSD=X'.")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # BUG FIX 1: Drop NaNs from OHLCV first.
    # Later, drop NaNs introduced by indicator calculations.
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    # Localise index to Athens time
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ATHENS_TZ)
    df.index.name = "datetime"

    # Helper columns
    df["date"]    = df.index.date
    df["hour"]    = df.index.hour
    df["weekday"] = df.index.weekday   # 0=Mon … 4=Fri

    # ATR
    prev_close    = df["Close"].shift(1)
    tr            = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"]     = tr.rolling(ATR_PERIOD).mean()

    # Rolling average volume
    df["AvgVol"]  = df["Volume"].rolling(VOL_LOOKBACK).mean()

    # BUG FIX 1 (cont.): Drop any remaining NaNs introduced by rolling calculations (ATR, AvgVol)
    df.dropna(inplace=True)

    print(f"  Loaded {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SESSION / DAILY / WEEKLY LEVELS
# ══════════════════════════════════════════════════════════════════════════════

def build_session_levels(df):
    """
    For every bar, attach:
      asia_high / asia_low  — that day's Asia session H/L  (known after Asia closes)
      prev_asia_high / prev_asia_low — previous day's Asia (for E: London range)
      london_high / london_low — that day's London session H/L
      pdh / pdl              — previous calendar day's high/low
      pdw_high / pdw_low     — previous ISO week's high/low
    """
    levels = df.copy()

    # ── Daily Asia H/L ─────────────────────────────────────────────────────────
    asia_mask = (df["hour"] >= ASIA_START) & (df["hour"] < ASIA_END)
    asia_bars = df[asia_mask].copy()
    daily_asia = asia_bars.groupby("date").agg(
        asia_high=("High",  "max"),
        asia_low =("Low",   "min")
    )

    # ── London H/L ─────────────────────────────────────────────────────────────
    lon_mask  = (df["hour"] >= LONDON_START) & (df["hour"] < LONDON_END)
    lon_bars  = df[lon_mask].copy()
    daily_lon = lon_bars.groupby("date").agg(
        london_high=("High", "max"),
        london_low =("Low",  "min")
    )

    # ── Previous day H/L (PDH/PDL) ─────────────────────────────────────────────
    daily_hl  = df.groupby("date").agg(
        day_high=("High", "max"),
        day_low =("Low",  "min")
    )
    daily_hl["pdh"] = daily_hl["day_high"].shift(1)
    daily_hl["pdl"] = daily_hl["day_low"].shift(1)

    # ── Previous week H/L (PDW) ────────────────────────────────────────────────
    # isocalendar() returns a DataFrame in pandas >= 1.1; extract columns explicitly
    iso = df.index.isocalendar()
    levels["iso_week"] = iso["week"].astype(int).values # Use levels to keep these
    levels["iso_year"] = iso["year"].astype(int).values # columns for merging
    weekly_hl = levels.groupby(["iso_year", "iso_week"]).agg(
        wk_high=("High", "max"),
        wk_low =("Low",  "min")
    )
    weekly_hl["pdw_high"] = weekly_hl["wk_high"].shift(1)
    weekly_hl["pdw_low"]  = weekly_hl["wk_low"].shift(1)

    # ── Merge back ─────────────────────────────────────────────────────────────
    # Added how="left" for clarity, though it's the default.
    levels = levels.join(daily_asia,  on="date", how="left")
    levels = levels.join(daily_lon,   on="date", how="left")
    levels = levels.join(daily_hl[["pdh", "pdl"]], on="date", how="left")
    # For PDW: merge on iso_year + iso_week using a temporary key (existing code is robust)
    levels = levels.reset_index().merge(
        weekly_hl[["pdw_high", "pdw_low"]].reset_index(),
        on=["iso_year", "iso_week"],
        how="left"
    ).set_index("datetime")

    # Drop the temporary iso_year and iso_week columns if they are not needed further.
    # They are used for the weekly_hl merge, but not necessarily needed afterwards.
    # Keeping them for now as they don't cause harm and might be useful for debug/extension.
    # If not needed, uncomment: levels = levels.drop(columns=["iso_year", "iso_week"])

    return levels


# ══════════════════════════════════════════════════════════════════════════════
# TRADE MANAGEMENT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def manage_trade(df, entry_idx, direction, atr_at_entry, target_price=None):
    """
    Walk forward bar-by-bar after entry.
    Trailing stop: starts at ATR_MULT × ATR below/above entry, trails highest/lowest close.
    Returns: (exit_price, exit_idx, exit_reason, pnl)
    direction: 'long' or 'short'
    """
    stop   = atr_at_entry * TRAIL_ATR_MULT
    entry  = df.at[entry_idx, "Close"]
    best   = entry          # best price seen (used to trail stop)

    rows   = df.index.tolist()
    pos    = rows.index(entry_idx)

    if direction == "long":
        trail_stop = entry - stop
        for i in range(pos + 1, len(rows)):
            idx = rows[i]
            lo, hi, cl = df.at[idx, "Low"], df.at[idx, "High"], df.at[idx, "Close"]
            # target hit?
            if target_price and hi >= target_price:
                return target_price, idx, "TARGET", target_price - entry
            # stop hit?
            if lo <= trail_stop:
                return trail_stop, idx, "TRAIL_STOP", trail_stop - entry
            # trail
            if cl > best:
                best       = cl
                trail_stop = best - stop
        # reached end of data
        # If pos is the last bar, min(pos+1, len(rows)-1) ensures last_idx is pos.
        # Otherwise, it's the next bar after pos.
        last_idx = rows[min(pos + 1, len(rows) - 1)]
        last_cl  = df.at[last_idx, "Close"]
        return last_cl, last_idx, "EOD", last_cl - entry
    else:  # short
        trail_stop = entry + stop
        for i in range(pos + 1, len(rows)):
            idx = rows[i]
            lo, hi, cl = df.at[idx, "Low"], df.at[idx, "High"], df.at[idx, "Close"]
            if target_price and lo <= target_price:
                return target_price, idx, "TARGET", entry - target_price
            if hi >= trail_stop:
                return trail_stop, idx, "TRAIL_STOP", entry - trail_stop
            if cl < best:
                best       = cl
                trail_stop = best + stop
        last_idx = rows[min(pos + 1, len(rows) - 1)]
        last_cl  = df.at[last_idx, "Close"]
        return last_cl, last_idx, "EOD", entry - last_cl


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY A — Asia Sweep + BOS (London session)
# ══════════════════════════════════════════════════════════════════════════════

def strategy_A(df):
    """
    London bars only.
    1. Bar wicks < Asia Low - 0.3  (LONG) or > Asia High + 0.3  (SHORT)
    2. Within 8 bars: close back through Asia Low/High → entry at that close
    3. Trail out
    """
    trades = []
    rows   = df.index.tolist() # Full dataframe indices for manage_trade

    london_bars = df[(df["hour"] >= LONDON_START) & (df["hour"] < LONDON_END)].index.tolist()

    i = 0
    while i < len(london_bars):
        idx        = london_bars[i]
        row        = df.loc[idx]
        asia_hi    = row["asia_high"]
        asia_lo    = row["asia_low"]
        if pd.isna(asia_hi) or pd.isna(asia_lo):
            i += 1; continue

        direction = None
        if row["Low"] < asia_lo - SWEEP_THRESHOLD:
            direction = "long"
            level     = asia_lo
        elif row["High"] > asia_hi + SWEEP_THRESHOLD:
            direction = "short"
            level     = asia_hi

        if direction:
            # look for BOS within next MAX_BARS_BOS bars (still London session)
            bos_found = False
            for j in range(i + 1, min(i + MAX_BARS_BOS + 1, len(london_bars))):
                bos_idx = london_bars[j]
                bos_row = df.loc[bos_idx]
                
                # Ensure ATR is not NaN at bos_idx
                atr_at_entry = df.at[bos_idx, "ATR"]
                if pd.isna(atr_at_entry):
                    continue # Skip this potential entry if ATR is unavailable

                if direction == "long"  and bos_row["Close"] > level:
                    ep, ex_idx, reason, pnl = manage_trade(df, bos_idx, "long", atr_at_entry)
                    trades.append(dict(strategy="A", direction="LONG",  entry_time=bos_idx,
                                       entry=bos_row["Close"], exit_time=ex_idx,
                                       exit=ep, reason=reason, pnl=pnl, sweep_idx=idx))
                    # Move 'i' to the position of the exit bar in london_bars, or end if exit is outside
                    i = london_bars.index(ex_idx) if ex_idx in london_bars else len(london_bars) - 1
                    bos_found = True
                    break # Exit inner for-loop after trade
                if direction == "short" and bos_row["Close"] < level:
                    ep, ex_idx, reason, pnl = manage_trade(df, bos_idx, "short", atr_at_entry)
                    trades.append(dict(strategy="A", direction="SHORT", entry_time=bos_idx,
                                       entry=bos_row["Close"], exit_time=ex_idx,
                                       exit=ep, reason=reason, pnl=pnl, sweep_idx=idx))
                    # Move 'i' to the position of the exit bar in london_bars, or end if exit is outside
                    i = london_bars.index(ex_idx) if ex_idx in london_bars else len(london_bars) - 1
                    bos_found = True
                    break # Exit inner for-loop after trade
            if bos_found:
                continue # Skip incrementing i again if a trade was found and i was updated
        i += 1 # Increment i if no trade was found for the current sweep or no sweep happened

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY B — High Volume Breakout + Retest (London + NY)
# ══════════════════════════════════════════════════════════════════════════════

def strategy_B(df):
    """
    London + NY bars.
    1. Bar closes above Asia High (LONG) or below Asia Low (SHORT) on Vol ≥ 2× AvgVol
    2. Within 10 bars: price wicks back to that level AND closes back through it → entry
    3. Trail out
    """
    trades = []
    active = (df["hour"] >= LONDON_START) & (df["hour"] < NY_END)
    active_bars = df[active].index.tolist()

    i = 0
    while i < len(active_bars):
        idx     = active_bars[i]
        row     = df.loc[idx]
        asia_hi = row["asia_high"]
        asia_lo = row["asia_low"]
        avg_vol = row["AvgVol"]
        vol     = row["Volume"]

        if pd.isna(asia_hi) or pd.isna(asia_lo) or pd.isna(avg_vol) or avg_vol == 0:
            i += 1; continue

        direction = None
        if row["Close"] > asia_hi and vol >= VOL_MULT * avg_vol:
            direction = "long";  level = asia_hi
        elif row["Close"] < asia_lo and vol >= VOL_MULT * avg_vol:
            direction = "short"; level = asia_lo

        if direction:
            retest_found = False
            for j in range(i + 1, min(i + MAX_BARS_RETEST + 1, len(active_bars))):
                rt_idx = active_bars[j]
                rt_row = df.loc[rt_idx]

                # Ensure ATR is not NaN at rt_idx
                atr_at_entry = df.at[rt_idx, "ATR"]
                if pd.isna(atr_at_entry):
                    continue # Skip this potential entry if ATR is unavailable

                if direction == "long":
                    # wick touches level and closes back above
                    if rt_row["Low"] <= level and rt_row["Close"] > level:
                        ep, ex_idx, reason, pnl = manage_trade(df, rt_idx, "long", atr_at_entry)
                        trades.append(dict(strategy="B", direction="LONG",  entry_time=rt_idx,
                                           entry=rt_row["Close"], exit_time=ex_idx,
                                           exit=ep, reason=reason, pnl=pnl, breakout_idx=idx))
                        i = active_bars.index(ex_idx) if ex_idx in active_bars else len(active_bars) - 1
                        retest_found = True
                        break # Exit inner for-loop after trade
                else: # short
                    if rt_row["High"] >= level and rt_row["Close"] < level:
                        ep, ex_idx, reason, pnl = manage_trade(df, rt_idx, "short", atr_at_entry)
                        trades.append(dict(strategy="B", direction="SHORT", entry_time=rt_idx,
                                           entry=rt_row["Close"], exit_time=ex_idx,
                                           exit=ep, reason=reason, pnl=pnl, breakout_idx=idx))
                        i = active_bars.index(ex_idx) if ex_idx in active_bars else len(active_bars) - 1
                        retest_found = True
                        break # Exit inner for-loop after trade
            if retest_found:
                continue # Skip incrementing i again if a trade was found and i was updated
        i += 1 # Increment i if no trade was found for the current breakout or no breakout happened

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY C — PDH/PDL Sweep + Reclaim (London + NY)
# ══════════════════════════════════════════════════════════════════════════════

def strategy_C(df):
    """
    London + NY bars.
    1. Bar wicks below PDL (LONG) or above PDH (SHORT) AND closes back through on same candle
    2. Entry = next bar open
    3. Target = today's Asia High (LONG) or Asia Low (SHORT)
    4. Trail out
    """
    trades = []
    active = (df["hour"] >= LONDON_START) & (df["hour"] < NY_END)
    active_bars = df[active].index.tolist()
    rows_all    = df.index.tolist()

    i = 0
    while i < len(active_bars):
        idx = active_bars[i]
        row = df.loc[idx]
        pdh = row["pdh"]; pdl = row["pdl"]
        if pd.isna(pdh) or pd.isna(pdl):
            i += 1
            continue

        direction = None
        if row["Low"] < pdl and row["Close"] > pdl:          # sweep + reclaim PDL
            direction = "long";  target = row["asia_high"]
        elif row["High"] > pdh and row["Close"] < pdh:        # sweep + reclaim PDH
            direction = "short"; target = row["asia_low"]

        if direction:
            # entry = next bar open
            pos_all = rows_all.index(idx)
            if pos_all + 1 >= len(rows_all): # Check if next bar exists
                i += 1
                continue
            entry_idx = rows_all[pos_all + 1]
            entry_p   = df.at[entry_idx, "Open"]

            # BUG FIX / Robustness: Ensure ATR is not NaN at entry_idx
            atr_at_entry = df.at[entry_idx, "ATR"]
            if pd.isna(atr_at_entry):
                i += 1 # Cannot manage trade without ATR
                continue

            tgt       = target if not pd.isna(target) else None
            ep, ex_idx, reason, pnl = manage_trade(df, entry_idx, direction,
                                                    atr_at_entry, tgt)
            trades.append(dict(strategy="C", direction=direction.upper(),
                               entry_time=entry_idx, entry=entry_p,
                               exit_time=ex_idx, exit=ep, reason=reason, pnl=pnl,
                               sweep_idx=idx))
            # Move 'i' to the position of the exit bar in active_bars, or beyond
            i = active_bars.index(ex_idx) if ex_idx in active_bars else len(active_bars) - 1
            continue # Skip normal increment if a trade was found
        i += 1 # Increment i if no trade was found for the current bar

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY E — NY Open Sweep of London Range + BOS
# ══════════════════════════════════════════════════════════════════════════════

def strategy_E(df):
    """
    NY bars only.
    1. Bar wicks below London Low - 0.3 (LONG) or above London High + 0.3 (SHORT)
    2. Within 8 NY bars: close back through London level → entry
    3. Trail out
    """
    trades = []
    ny_bars = df[(df["hour"] >= NY_START) & (df["hour"] < NY_END)].index.tolist()

    i = 0
    while i < len(ny_bars):
        idx     = ny_bars[i]
        row     = df.loc[idx]
        lon_hi  = row["london_high"]
        lon_lo  = row["london_low"]
        if pd.isna(lon_hi) or pd.isna(lon_lo):
            i += 1; continue

        direction = None
        if row["Low"] < lon_lo - SWEEP_THRESHOLD:
            direction = "long";  level = lon_lo
        elif row["High"] > lon_hi + SWEEP_THRESHOLD:
            direction = "short"; level = lon_hi

        if direction:
            bos_found = False
            for j in range(i + 1, min(i + MAX_BARS_BOS + 1, len(ny_bars))):
                bos_idx = ny_bars[j]
                bos_row = df.loc[bos_idx]

                # Ensure ATR is not NaN at bos_idx
                atr_at_entry = df.at[bos_idx, "ATR"]
                if pd.isna(atr_at_entry):
                    continue # Skip this potential entry if ATR is unavailable

                if direction == "long"  and bos_row["Close"] > level:
                    ep, ex_idx, reason, pnl = manage_trade(df, bos_idx, "long",  atr_at_entry)
                    trades.append(dict(strategy="E", direction="LONG",  entry_time=bos_idx,
                                       entry=bos_row["Close"], exit_time=ex_idx,
                                       exit=ep, reason=reason, pnl=pnl, sweep_idx=idx))
                    i = ny_bars.index(ex_idx) if ex_idx in ny_bars else len(ny_bars) - 1
                    bos_found = True
                    break
                if direction == "short" and bos_row["Close"] < level:
                    ep, ex_idx, reason, pnl = manage_trade(df, bos_idx, "short", atr_at_entry)
                    trades.append(dict(strategy="E", direction="SHORT", entry_time=bos_idx,
                                       entry=bos_row["Close"], exit_time=ex_idx,
                                       exit=ep, reason=reason, pnl=pnl, sweep_idx=idx))
                    i = ny_bars.index(ex_idx) if ex_idx in ny_bars else len(ny_bars) - 1
                    bos_found = True
                    break
            if bos_found:
                continue
        i += 1

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY F — PDW Sweep + Reclaim (London + NY)
# ══════════════════════════════════════════════════════════════════════════════

def strategy_F(df):
    """
    London + NY bars.
    Same logic as C but using previous week's high/low.
    1. Wick beyond PDW level + close back through on same candle
    2. Entry = next bar open
    3. Target = today's Asia High (LONG) or Asia Low (SHORT)
    4. Trail out
    """
    trades = []
    active = (df["hour"] >= LONDON_START) & (df["hour"] < NY_END)
    active_bars = df[active].index.tolist()
    rows_all    = df.index.tolist()

    i = 0
    while i < len(active_bars):
        idx    = active_bars[i]
        row    = df.loc[idx]
        pdw_hi = row["pdw_high"]; pdw_lo = row["pdw_low"]
        if pd.isna(pdw_hi) or pd.isna(pdw_lo):
            i += 1
            continue

        direction = None
        if row["Low"] < pdw_lo and row["Close"] > pdw_lo:
            direction = "long";  target = row["asia_high"]
        elif row["High"] > pdw_hi and row["Close"] < pdw_hi:
            direction = "short"; target = row["asia_low"]

        if direction:
            pos_all = rows_all.index(idx)
            if pos_all + 1 >= len(rows_all):
                i += 1
                continue
            entry_idx = rows_all[pos_all + 1]
            entry_p   = df.at[entry_idx, "Open"]

            # BUG FIX / Robustness: Ensure ATR is not NaN at entry_idx
            atr_at_entry = df.at[entry_idx, "ATR"]
            if pd.isna(atr_at_entry):
                i += 1 # Cannot manage trade without ATR
                continue

            tgt       = target if not pd.isna(target) else None
            ep, ex_idx, reason, pnl = manage_trade(df, entry_idx, direction,
                                                    atr_at_entry, tgt)
            trades.append(dict(strategy="F", direction=direction.upper(),
                               entry_time=entry_idx, entry=entry_p,
                               exit_time=ex_idx, exit=ep, reason=reason, pnl=pnl,
                               sweep_idx=idx))
            i = active_bars.index(ex_idx) if ex_idx in active_bars else len(active_bars) - 1
            continue
        i += 1

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS & REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(trades_df, name=""):
    if trades_df.empty:
        # BUG FIX / Cosmetic: Ensure consistent string formatting for empty results
        return {"strategy": name, "trades": 0, "win_rate": "0.0%",
                "total_pnl": "0.00", "avg_pnl": "0.00", "max_win": "0.00", "max_loss": "0.00",
                "profit_factor": "0.00", "avg_winner": "n/a", "avg_loser": "n/a"}
    t = trades_df
    wins  = t[t["pnl"] > 0]
    losses= t[t["pnl"] <= 0] # Includes zero-pnl as loss for conservative PF calculation
    gross_profit = wins["pnl"].sum()   if not wins.empty   else 0
    gross_loss   = losses["pnl"].sum() if not losses.empty else 0
    pf = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    # Ensure all numerical outputs are formatted consistently
    total_pnl_str = f"{t['pnl'].sum():.2f}"
    avg_pnl_str = f"{t['pnl'].mean():.2f}"
    max_win_str = f"{t['pnl'].max():.2f}"
    max_loss_str = f"{t['pnl'].min():.2f}"
    profit_factor_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    avg_winner_str = f"{wins['pnl'].mean():.2f}"  if not wins.empty   else "n/a"
    avg_loser_str = f"{losses['pnl'].mean():.2f}" if not losses.empty else "n/a"

    return {
        "strategy"      : name,
        "trades"        : len(t),
        "win_rate"      : f"{len(wins)/len(t)*100:.1f}%",
        "total_pnl"     : total_pnl_str,
        "avg_pnl"       : avg_pnl_str,
        "max_win"       : max_win_str,
        "max_loss"      : max_loss_str,
        "profit_factor" : profit_factor_str,
        "avg_winner"    : avg_winner_str,
        "avg_loser"     : avg_loser_str,
    }

def print_banner(text):
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)

def print_results(all_trades):
    print_banner("INDIVIDUAL TRADE LOG")
    cols = ["strategy", "direction", "entry_time", "entry", "exit_time", "exit", "reason", "pnl"]
    display = all_trades[cols].copy()
    display["entry"]      = display["entry"].round(2)
    display["exit"]       = display["exit"].round(2)
    display["pnl"]        = display["pnl"].round(2)
    display["entry_time"] = display["entry_time"].astype(str).str[:16]
    display["exit_time"]  = display["exit_time"].astype(str).str[:16]
    print(display.to_string(index=False))

    print_banner("SUMMARY BY STRATEGY")
    stats = []
    # Ensure consistent order of strategies
    strategy_names = sorted(all_trades["strategy"].unique().tolist()) if not all_trades.empty else []
    for strat in strategy_names:
        sub = all_trades[all_trades["strategy"] == strat]
        stats.append(compute_stats(sub, strat))

    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    print_banner("OVERALL COMBINED PERFORMANCE")
    overall = compute_stats(all_trades, "ALL")
    for k, v in overall.items():
        print(f"  {k:<20} {v}")

    print_banner("EXIT REASON BREAKDOWN")
    er = all_trades.groupby(["strategy", "reason"]).size().unstack(fill_value=0)
    print(er.to_string())

    # Direction breakdown
    print_banner("DIRECTION BREAKDOWN")
    dd = all_trades.groupby(["strategy", "direction"])["pnl"].agg(["count", "mean", "sum"])
    dd.columns = ["trades", "avg_pnl", "total_pnl"]
    dd = dd.round(2)
    print(dd.to_string())

    print("\n")
    print("NOTE: PnL is in USD per oz (price points). Scale by your contract size.")
    print("      GC futures = 100 oz/contract → multiply PnL × 100 for $ P&L.")
    print("      Trailing stop = ATR(14) × 1.5 from best price seen.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_banner("XAUUSD MULTI-STRATEGY BACKTEST  (A / B / C / E / F)")
    df = load_data()
    df = build_session_levels(df)

    print("\nRunning Strategy A — Asia Sweep + BOS (London) …")
    trades_A = strategy_A(df)
    print(f"  → {len(trades_A)} trades")

    print("Running Strategy B — HV Breakout + Retest (London+NY) …")
    trades_B = strategy_B(df)
    print(f"  → {len(trades_B)} trades")

    print("Running Strategy C — PDH/PDL Sweep + Reclaim (London+NY) …")
    trades_C = strategy_C(df)
    print(f"  → {len(trades_C)} trades")

    print("Running Strategy E — NY Sweep of London Range + BOS (NY) …")
    trades_E = strategy_E(df)
    print(f"  → {len(trades_E)} trades")

    print("Running Strategy F — PDW Sweep + Reclaim (London+NY) …")
    trades_F = strategy_F(df)
    print(f"  → {len(trades_F)} trades")

    all_dfs = [t for t in [trades_A, trades_B, trades_C, trades_E, trades_F] if not t.empty]
    if not all_dfs:
        print("\n⚠️  No trades generated. Check data availability and session times.")
        return

    all_trades = pd.concat(all_dfs, ignore_index=True)
    all_trades['entry_time'] = pd.to_datetime(all_trades['entry_time'])
    # BUG FIX 2: Removed unused and potentially problematic iso_year calculation.
    # all_trades['iso_year'] = all_trades['entry_time'].dt.isocalendar().year.astype(int)
    all_trades.sort_values("entry_time", inplace=True)

    print_results(all_trades)

    # Save to CSV
    out = "xauusd_backtest_results.csv"
    all_trades.to_csv(out, index=False)
    print(f"Full trade log saved → {out}")


if __name__ == "__main__":
    main()
