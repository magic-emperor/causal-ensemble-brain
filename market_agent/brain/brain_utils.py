"""
brain_utils.py — Shared Math Foundation for All Brains
=======================================================
Single source of truth for all indicator calculations.
Every brain imports from here. No local duplicate implementations.

Why this matters:
  - ATR was duplicated 6+ times with minor variations
  - RSI had 3 different implementations (simple MA vs Wilder's EMA diverge by 3-8 points)
  - Swing high/low detection existed only in liquidity_sweep.py; Fibonacci used all-time max/min

Rule: If a calculation appears in more than one brain file, it lives here.
Rule: Wilder's EMA is the ONLY RSI implementation. Simple rolling mean RSI is wrong.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict


# ── ATR (Average True Range) ──────────────────────────────────────────────────

def calc_atr(hist: pd.DataFrame, period: int = 14) -> float:
    """
    ATR using Wilder's rolling mean on True Range.
    Returns float. Safe fallback to 0.0 on insufficient data.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    """
    if hist is None or len(hist) < period + 1:
        return 0.0
    high       = hist['High']
    low        = hist['Low']
    prev_close = hist['Close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    result = tr.rolling(period).mean().iloc[-1]
    return float(result) if not pd.isna(result) else 0.0


# ── RSI (Relative Strength Index) — Wilder's EMA ONLY ────────────────────────

def calc_rsi_series(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    RSI as a full Series using Wilder's EMA smoothing.

    IMPORTANT: Do NOT use rolling().mean() for RSI — that is a simple moving
    average approximation that diverges from Wilder's at extreme values (RSI < 30
    or > 70) by 3-8 points. This is exactly when signals fire. Use this function.

    Returns pd.Series of RSI values (0-100).
    """
    close    = hist['Close']
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float('nan'))
    return 100 - (100 / (1 + rs))


def calc_rsi_float(hist: pd.DataFrame, period: int = 14) -> float:
    """
    RSI as a single float (last value). Convenience wrapper around calc_rsi_series.
    Returns 50.0 on insufficient data.
    """
    if hist is None or len(hist) < period + 1:
        return 50.0
    s = calc_rsi_series(hist, period)
    val = s.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


# ── MACD ──────────────────────────────────────────────────────────────────────

def calc_macd(
    hist: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD: returns (macd_line, signal_line, histogram) as Series.
    macd_line  = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal)
    histogram   = macd_line - signal_line
    """
    close       = hist['Close']
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def calc_bollinger(
    hist: pd.DataFrame,
    period: int = 20,
    k: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: returns (upper, mid, lower) as Series.
    mid   = SMA(period)
    upper = mid + k * std
    lower = mid - k * std
    """
    close = hist['Close']
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def calc_bb_pct_b(hist: pd.DataFrame, period: int = 20, k: float = 2.0) -> float:
    """
    %B: position of price within Bollinger Bands.
    0.0 = at lower band, 1.0 = at upper band, 0.5 = at mid.
    Returns 0.5 on insufficient data.
    """
    if hist is None or len(hist) < period:
        return 0.5
    upper, mid, lower = calc_bollinger(hist, period, k)
    price  = float(hist['Close'].iloc[-1])
    up_val = float(upper.iloc[-1])
    lo_val = float(lower.iloc[-1])
    band   = up_val - lo_val
    return (price - lo_val) / band if band > 0 else 0.5


# ── VWAP ─────────────────────────────────────────────────────────────────────

def calc_vwap(hist: pd.DataFrame, window: int = 20) -> float:
    """
    Rolling VWAP over the last `window` bars.
    Returns float price. Falls back to last close if Volume missing.
    """
    if hist is None or len(hist) < window:
        return float(hist['Close'].iloc[-1]) if hist is not None and len(hist) > 0 else 0.0
    need = ['High', 'Low', 'Close', 'Volume']
    if not all(c in hist.columns for c in need):
        return float(hist['Close'].iloc[-1])
    df       = hist.tail(window)
    typical  = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol_sum  = df['Volume'].sum()
    if vol_sum <= 0:
        return float(df['Close'].iloc[-1])
    return float((typical * df['Volume']).sum() / vol_sum)


# ── ADX (Average Directional Index) ──────────────────────────────────────────

def calc_adx(hist: pd.DataFrame, period: int = 14) -> float:
    """
    ADX: trend strength indicator (0-100).
    ADX > 25: strong trend. ADX < 20: ranging/no trend.
    Returns float. Falls back to 15.0 on insufficient data.
    """
    if hist is None or len(hist) < period * 2 + 1:
        return 15.0
    high  = hist['High']
    low   = hist['Low']
    close = hist['Close']

    plus_dm  = (high.diff()).where(high.diff() > low.diff().abs(), 0.0).where(high.diff() > 0, 0.0)
    minus_dm = (low.diff().abs()).where(low.diff().abs() > high.diff(), 0.0).where(low.diff() < 0, 0.0)
    tr_vals  = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr14    = tr_vals.rolling(period).mean()
    plus_di  = 100 * plus_dm.rolling(period).mean() / atr14.replace(0, float('nan'))
    minus_di = 100 * minus_dm.rolling(period).mean() / atr14.replace(0, float('nan'))
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float('nan'))
    adx_val  = float(dx.rolling(period).mean().iloc[-1])
    return adx_val if not pd.isna(adx_val) else 15.0


# ── Swing High / Low Detection ────────────────────────────────────────────────

def find_swing_levels(hist: pd.DataFrame, lookback: int = 30) -> Tuple[float, float]:
    """
    Find the most recent significant swing high and swing low within `lookback` bars.

    Uses a fractal/pivot approach: a swing high is a bar where High is greater than
    the surrounding 2 bars on each side. This is more meaningful than absolute max/min
    over the entire history window (which could be weeks-old irrelevant extremes).

    Falls back to max/min of the lookback window if no fractal pivots are found.

    Returns: (swing_high, swing_low) as floats.
    """
    if hist is None or len(hist) < 5:
        return (float(hist['High'].max()), float(hist['Low'].min())) if hist is not None and len(hist) > 0 else (0.0, 0.0)

    window = hist.tail(lookback) if len(hist) > lookback else hist
    highs  = window['High'].values
    lows   = window['Low'].values
    n      = len(highs)

    pivot_highs = []
    pivot_lows  = []

    # Simple 2-bar fractal pivot: bar[i] is a pivot high if it's > neighbours on both sides
    for i in range(2, n - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            pivot_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            pivot_lows.append(lows[i])

    swing_high = max(pivot_highs) if pivot_highs else float(window['High'].max())
    swing_low  = min(pivot_lows)  if pivot_lows  else float(window['Low'].min())

    return swing_high, swing_low


# ── Delta Flow (Crypto only) ──────────────────────────────────────────────────

def compute_delta_flow(hist: pd.DataFrame) -> Optional[float]:
    """
    Cumulative buying vs selling pressure over last 5 candles.
    Only available for crypto (Binance data has TakerBase column).

    +0.30 to +1.0 = buyers in control
    -0.30 to -1.0 = sellers in control
    -0.30 to +0.30 = balanced

    Returns None for equity/forex (no TakerBase column).
    """
    if hist is None or 'TakerBase' not in hist.columns:
        return None
    if len(hist) < 5:
        return None

    recent     = hist.iloc[-5:]
    total_vol  = recent['Volume'].replace(0, float('nan'))
    taker_buy  = recent['TakerBase']
    taker_sell = total_vol - taker_buy

    delta_per_candle = (taker_buy - taker_sell) / total_vol
    cumulative       = float(delta_per_candle.mean())

    return round(cumulative, 3) if not pd.isna(cumulative) else None
