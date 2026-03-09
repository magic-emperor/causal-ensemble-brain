"""
Brain 7: Causal-Ensemble — SQUEEZE Specialist (Mean Reversion)
==============================================================
Works best in: SQUEEZE regime only (confirmed by 90-day grid run 2026-03-08).

CONFIRMED FINDINGS (grid run 2026-03-08, 90 days, 27 combos, 16 symbols):
  - RANGING  regime: WR=21.8%, EV=−0.299 — NO EDGE. Blocked.
  - VOLATILE regime: WR=0.0%,  EV=−1.000 — NO EDGE. Blocked.
  - SQUEEZE  regime breakdown:
      * SQUEEZE 1h equity:    WR=42.1%, EV=+0.389  ✅
      * SQUEEZE 1h crypto:    WR=42.9%, EV=+0.414  ✅
      * SQUEEZE 4h fx_comm:   WR=62.5%, EV=+1.062  ✅
      * SQUEEZE 1h fx_comm:   WR=20.5%, EV=−0.325  ❌ BLOCKED
      * SQUEEZE 4h crypto:    WR=0.0%,  EV=−1.000  ❌ (n=1, data thin — blocked for now)
  - RSI sweep (25/30/35) had ZERO IMPACT on signal counts or EV.
    Brain fires almost entirely on SQUEEZE direction, not RSI extremes.
  - Price outside BB bands (pct_b < 0 or > 1): 54 signals, WR=20.4%, EV=−0.328.
    These are late entries after breakout already happened. Gate added.

DESIGN CHANGES vs v1 (pre-grid):
  1. RANGING and VOLATILE blocked (gate 5 — data confirmed no edge)
  2. FX/Commodities on 1h blocked (gate 3 — structural 1h SQUEEZE no-edge)
  3. Outside-band entry gate added (gate 6 — pct_b outside [0,1] → HOLD)
  4. Symbol exclusion list: HDFCBANK.NS, RELIANCE.NS, USDJPY=X (confirmed losses)
  5. symbol + timeframe params added to function signature for routing

Returns: BrainSignal (brain_contract.py)
"""
from __future__ import annotations

import pandas as pd
from market_agent.brain.brain_contract import BrainSignal
from market_agent.brain.brain_utils import calc_rsi_series, calc_atr


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTED CONSTANTS — imported by grid/confirm runners, never hardcode there
# ══════════════════════════════════════════════════════════════════════════════

_MIN_DATA_BARS         = 30
_BB_PERIOD             = 20
_BB_STD_K              = 2.0
_SQUEEZE_RATIO         = 0.70
_STRONG_TREND_RATIO    = 1.60
_ATR_EXTREME_MULT      = 2.0
_VOLUME_HEALTH_RATIO   = 0.60

_RSI_OVERSOLD_STRONG   = 30
_RSI_OVERSOLD_MEDIUM   = 40
_RSI_OVERBOUGHT_STRONG = 70
_RSI_OVERBOUGHT_MEDIUM = 60

_PCT_B_EXTREME_LOW     = 0.05
_PCT_B_NEAR_LOW        = 0.15
_PCT_B_WEAK_LOW        = 0.20
_PCT_B_EXTREME_HIGH    = 0.95
_PCT_B_NEAR_HIGH       = 0.85
_PCT_B_WEAK_HIGH       = 0.80

# Outside-band gate: price outside BB = breakout already happened = no entry
# Grid: 54 outside-band signals, WR=20.4%, EV=−0.328
_PCT_B_MIN_VALID       = 0.0
_PCT_B_MAX_VALID       = 1.0

_SQUEEZE_CANDLES       = 5

_RR_T1_STRONG = 3.0;  _RR_T2_STRONG = 5.0;  _RR_SL_STRONG = 1.0
_RR_T1_MEDIUM = 2.5;  _RR_T2_MEDIUM = 4.0;  _RR_SL_MEDIUM = 1.0
_RR_T1_WEAK   = 2.0;  _RR_T2_WEAK   = 3.5;  _RR_SL_WEAK   = 1.0

# ── Confirmed exclusions ──────────────────────────────────────────────────────
# Evidence required to add: p-value < 5% OR consistent across both 90d + 180d.
# Evidence required to REMOVE: 365d confirm with EV > 0.
CAUSAL_EXCLUDED_SYMBOLS = {
    'HDFCBANK.NS',   # 90d+180d: SQUEEZE WR=0% EV=−1.000. All SELL signals below lower band.
    'RELIANCE.NS',   # 90d: SQUEEZE WR=25% EV=−0.175. 180d contradicts — re-evaluate at 365d.
    'USDJPY=X',      # 90d+180d: both negative across 1h and 4h consistently.
    'TATASTEEL.NS',  # 180d SQUEEZE WR=16.7% EV=−0.452. p=1.95% — statistically confirmed bad.
    'CL=F',          # 365d: SQUEEZE WR=0% EV=−1.000 (7 signals). p=0.78% — lowest p-value seen.
                     # Crude oil does not mean-revert on squeezes — supply/demand shocks break them.
    'BTC-USD',       # 365d: SQUEEZE WR=25% EV=−0.237 (20 signals). RSI gate not enough.
                     # BTC trends through squeezes — revisit after 180 more live signals.
}

# ── BTC RSI confirmation gate ─────────────────────────────────────────────────
# BTC squeeze signals without RSI alignment have no directional conviction.
# 180d data: RSI-gated BTC EV=+0.158 vs ungated EV=−0.147 (18 bad signals filtered).
# Applied only to BTC-USD — other symbols do not show same mid-RSI problem.
_BTC_RSI_BUY_MIN  = 55.0   # BTC BUY squeeze: RSI must be above this (momentum up)
_BTC_RSI_SELL_MAX = 45.0   # BTC SELL squeeze: RSI must be below this (momentum down)

# ── Asset-class routing ───────────────────────────────────────────────────────
FX_COMM_SYMBOLS = {'GBPJPY=X', 'USDJPY=X', 'GC=F', 'CL=F'}


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _volume_is_healthy(hist: pd.DataFrame) -> bool:
    if 'Volume' not in hist.columns or len(hist) < 20:
        return True
    avg_vol  = float(hist['Volume'].rolling(20).mean().iloc[-1])
    last_vol = float(hist['Volume'].iloc[-1])
    if avg_vol <= 0:
        return True
    return (last_vol / avg_vol) >= _VOLUME_HEALTH_RATIO


def _trend_is_not_extreme(hist: pd.DataFrame) -> bool:
    if len(hist) < 50:
        return True
    atr   = calc_atr(hist, period=14)
    sma50 = hist['Close'].rolling(50).mean().iloc[-1]
    price = float(hist['Close'].iloc[-1])
    if pd.isna(sma50) or atr <= 0 or price <= 0:
        return True
    return abs(price - float(sma50)) / atr < _ATR_EXTREME_MULT


def _squeeze_breakout_direction(hist: pd.DataFrame) -> str:
    close = hist['Close']
    if len(close) < _SQUEEZE_CANDLES:
        return 'NEUTRAL'
    sma20 = float(close.rolling(20).mean().iloc[-1])
    if pd.isna(sma20):
        return 'NEUTRAL'
    candles = [float(close.iloc[-(j+1)]) for j in range(_SQUEEZE_CANDLES)]
    all_rising  = all(candles[i] > candles[i+1] for i in range(_SQUEEZE_CANDLES - 1))
    all_falling = all(candles[i] < candles[i+1] for i in range(_SQUEEZE_CANDLES - 1))
    if all_rising  and candles[0] > sma20:
        return 'BUY'
    if all_falling and candles[0] < sma20:
        return 'SELL'
    return 'NEUTRAL'


def _get_rr_multipliers(sig_strength: str) -> tuple:
    if sig_strength == 'STRONG':
        return (_RR_T1_STRONG, _RR_T2_STRONG, _RR_SL_STRONG)
    elif sig_strength == 'MEDIUM':
        return (_RR_T1_MEDIUM, _RR_T2_MEDIUM, _RR_SL_MEDIUM)
    else:
        return (_RR_T1_WEAK, _RR_T2_WEAK, _RR_SL_WEAK)


def _hold(reason: str, decision_factor: str, base_meas: dict,
          confidence: float = 0.30, contra: str = '') -> BrainSignal:
    """Convenience: return a HOLD BrainSignal."""
    return BrainSignal(
        brain_name='Causal-Ensemble', specialization='SQUEEZE Specialist',
        method='BB Squeeze Direction | 2:1 R:R minimum',
        direction='HOLD', confidence=confidence, signal_strength=0.0,
        signal_age_candles=0, primary_evidence=reason,
        supporting_factors=[], contra_factors=[contra] if contra else [],
        method_confidence=0.50, regime_suitability='LOW',
        measurements={**base_meas, 'decision_factor': decision_factor},
        rr_t1_mult=_RR_T1_WEAK, rr_t2_mult=_RR_T2_WEAK, rr_sl_mult=_RR_SL_WEAK,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BRAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def causal_ensemble_signal(
    hist:      pd.DataFrame,
    regime:    str = None,
    symbol:    str = None,     # for exclusion list + FX 1h routing
    timeframe: str = None,     # for FX 1h routing
) -> BrainSignal:
    """
    Brain 7: Causal-Ensemble — SQUEEZE Specialist.

    Gates (in order):
      1. Data sufficiency (>= _MIN_DATA_BARS bars)
      2. Symbol exclusion (HDFCBANK, RELIANCE, USDJPY — grid confirmed no-edge)
      3. FX/Comm on 1h blocked (grid: WR=16.9%, EV=−0.402 structural)
      4. Extreme trend gate (price > 2×ATR14 from SMA50)
      5. Regime gate: only SQUEEZE (and TRENDING+squeeze if direction matches).
         RANGING and VOLATILE blocked — confirmed no edge.
      6. Outside-band gate: pct_b outside [0,1] → already broken out → HOLD
      7. Signal tier: SQUEEZE direction → STRONG RSI → MEDIUM → WEAK
    """

    # ── 1. Data sufficiency ───────────────────────────────────────────────────
    if len(hist) < _MIN_DATA_BARS:
        return _hold(
            f'Insufficient data (need >= {_MIN_DATA_BARS}, have {len(hist)})',
            'GATE_INSUFFICIENT_DATA',
            {'bars_used': len(hist)},
        )

    # ── Compute indicators (needed by all subsequent gates) ───────────────────
    close = hist['Close']
    price = float(close.iloc[-1])

    rsi_s = calc_rsi_series(hist, period=14)
    rsi   = float(rsi_s.iloc[-1]) if (not rsi_s.empty and not pd.isna(rsi_s.iloc[-1])) else 50.0

    sma20_s = close.rolling(_BB_PERIOD).mean()
    std20_s = close.rolling(_BB_PERIOD).std()
    upper_s = sma20_s + _BB_STD_K * std20_s
    lower_s = sma20_s - _BB_STD_K * std20_s
    bb_w_s  = (upper_s - lower_s) / sma20_s.replace(0, float('nan'))

    up_val = float(upper_s.iloc[-1])
    lo_val = float(lower_s.iloc[-1])
    width  = float(bb_w_s.iloc[-1]) if (not bb_w_s.empty and not pd.isna(bb_w_s.iloc[-1])) else 0.04

    valid_bw = bb_w_s.dropna()
    if len(valid_bw) >= 20:
        avg_w = float(valid_bw.rolling(20).mean().iloc[-1])
    elif len(valid_bw) >= 10:
        avg_w = float(valid_bw.iloc[-len(valid_bw):].mean())
    else:
        avg_w = width

    atr = calc_atr(hist, period=14)

    band_range   = up_val - lo_val
    pct_b        = (price - lo_val) / band_range if band_range > 0 else 0.5
    is_squeeze   = (avg_w > 0) and (width < avg_w * _SQUEEZE_RATIO)
    strong_trend = (avg_w > 0) and (width > avg_w * _STRONG_TREND_RATIO)

    # Base measurements — shared by all returns
    base_meas = {
        'pct_b':             round(pct_b, 3),
        'rsi':               round(rsi, 1),
        'bb_width':          round(width, 4),
        'avg_bb_width':      round(avg_w, 4),
        'is_squeeze':        float(is_squeeze),
        'price_at_signal':   round(price, 6),
        'atr_at_signal':     round(float(atr), 6),
        'atr_pct_at_signal': round(float(atr) / price * 100, 3) if price > 0 else 0.0,
        'bars_used':         len(hist),
    }

    # ── 2. Symbol exclusion ───────────────────────────────────────────────────
    if symbol and symbol in CAUSAL_EXCLUDED_SYMBOLS:
        return _hold(
            f'{symbol} excluded — confirmed no-edge (grid 2026-03-08)',
            'GATE_EXCLUDED_SYMBOL', base_meas, confidence=0.30, contra='symbol_excluded',
        )

    # ── 3. FX/Comm on 1h blocked ──────────────────────────────────────────────
    if symbol and timeframe and symbol in FX_COMM_SYMBOLS and timeframe == '1h':
        return _hold(
            f'FX/Comm {symbol} on 1h blocked (grid: WR=16.9%, EV=−0.402). Only 4h valid.',
            'GATE_FX_1H_NO_EDGE', base_meas, confidence=0.30, contra='fx_1h_no_edge',
        )

    # ── 4. Extreme trend gate ─────────────────────────────────────────────────
    if not _trend_is_not_extreme(hist):
        return _hold(
            f'Extreme trend — mean reversion blocked (pct_b={pct_b:.2f})',
            'GATE_EXTREME_TREND', base_meas, confidence=0.35,
            contra=f'price > {_ATR_EXTREME_MULT}×ATR14 from SMA50',
        )

    # ── 5. Regime gate ────────────────────────────────────────────────────────
    if regime in ('RANGING', 'VOLATILE', 'CHAOS'):
        return _hold(
            f'Regime {regime} blocked — no edge confirmed (grid 2026-03-08)',
            f'GATE_REGIME_{regime}', base_meas, confidence=0.30,
            contra=f'{regime}: EV negative in 90-day grid',
        )

    if regime in ('TRENDING_UP', 'TRENDING_DOWN'):
        if not is_squeeze:
            return _hold(
                f'Trending ({regime}) without squeeze — blocked',
                'GATE_TRENDING_NO_SQUEEZE', base_meas, confidence=0.40,
                contra=f'no squeeze in {regime}',
            )
        sq_dir_check = _squeeze_breakout_direction(hist)
        trend_dir    = 'BUY' if regime == 'TRENDING_UP' else 'SELL'
        if sq_dir_check != 'NEUTRAL' and sq_dir_check != trend_dir:
            return _hold(
                f'Squeeze ({sq_dir_check}) opposes trend ({regime}) — fade risk',
                'GATE_SQUEEZE_VS_TREND', base_meas, confidence=0.38,
                contra='squeeze direction opposes trend',
            )

    # ── 6. Outside-band gate ──────────────────────────────────────────────────
    if pct_b < _PCT_B_MIN_VALID or pct_b > _PCT_B_MAX_VALID:
        return _hold(
            f'Price outside BB bands (pct_b={pct_b:.3f}) — breakout already done, no chase',
            'GATE_OUTSIDE_BAND', base_meas, confidence=0.32,
            contra='outside_band: WR=20.4% EV=−0.328 in grid',
        )

    # ── 7. Signal tiers ───────────────────────────────────────────────────────
    if is_squeeze:
        sq_dir = _squeeze_breakout_direction(hist)
        if sq_dir == 'NEUTRAL':
            direction, confidence, sig_strength = 'HOLD', 0.50, 'NONE'
            reason = f'BB Squeeze — direction unclear (width={width:.4f})'
        else:
            direction    = sq_dir
            confidence   = 0.65
            sig_strength = 'MEDIUM'
            reason       = f'BB Squeeze: {sq_dir} (width={width:.4f} < {_SQUEEZE_RATIO}×avg {avg_w:.4f})'

        # BTC RSI confirmation gate
        # BTC squeeze with neutral RSI (avg 50.8) loses consistently.
        # 180d: ungated EV=-0.147, RSI-gated EV=+0.158 (18 bad signals blocked).
        if direction != 'HOLD' and symbol == 'BTC-USD':
            btc_rsi_ok = (
                (direction == 'BUY'  and rsi >= _BTC_RSI_BUY_MIN) or
                (direction == 'SELL' and rsi <= _BTC_RSI_SELL_MAX)
            )
            if not btc_rsi_ok:
                return _hold(
                    f'BTC SQUEEZE: RSI={rsi:.1f} no conviction (BUY>{_BTC_RSI_BUY_MIN}, SELL<{_BTC_RSI_SELL_MAX})',
                    'GATE_BTC_RSI_NO_CONVICTION', base_meas, confidence=0.40,
                    contra='BTC mid-RSI squeeze EV=-0.282 in 180d',
                )

    elif pct_b < _PCT_B_EXTREME_LOW and rsi < _RSI_OVERSOLD_STRONG:
        direction, confidence, sig_strength = 'BUY', 0.85, 'STRONG'
        reason = f'STRONG oversold: pct_b={pct_b:.2f} RSI={rsi:.1f}'

    elif pct_b > _PCT_B_EXTREME_HIGH and rsi > _RSI_OVERBOUGHT_STRONG:
        direction, confidence, sig_strength = 'SELL', 0.85, 'STRONG'
        reason = f'STRONG overbought: pct_b={pct_b:.2f} RSI={rsi:.1f}'

    elif pct_b < _PCT_B_NEAR_LOW and rsi < _RSI_OVERSOLD_MEDIUM:
        direction, confidence, sig_strength = 'BUY', 0.70, 'MEDIUM'
        reason = f'MEDIUM oversold: pct_b={pct_b:.2f} RSI={rsi:.1f}'

    elif pct_b > _PCT_B_NEAR_HIGH and rsi > _RSI_OVERBOUGHT_MEDIUM:
        direction, confidence, sig_strength = 'SELL', 0.70, 'MEDIUM'
        reason = f'MEDIUM overbought: pct_b={pct_b:.2f} RSI={rsi:.1f}'

    elif pct_b < _PCT_B_WEAK_LOW:
        rsi_rising = (
            float(rsi_s.iloc[-1]) > float(rsi_s.iloc[-3])
            if (len(rsi_s) >= 3 and not pd.isna(rsi_s.iloc[-3])) else True
        )
        if rsi_rising:
            direction, confidence, sig_strength = 'BUY', 0.58, 'WEAK'
            reason = f'Weak lower BB: pct_b={pct_b:.2f} RSI={rsi:.1f} rising'
        else:
            direction, confidence, sig_strength = 'HOLD', 0.44, 'NONE'
            reason = f'Weak BB: pct_b={pct_b:.2f} RSI={rsi:.1f} still falling'

    elif pct_b > _PCT_B_WEAK_HIGH:
        rsi_falling = (
            float(rsi_s.iloc[-1]) < float(rsi_s.iloc[-3])
            if (len(rsi_s) >= 3 and not pd.isna(rsi_s.iloc[-3])) else True
        )
        if rsi_falling:
            direction, confidence, sig_strength = 'SELL', 0.58, 'WEAK'
            reason = f'Weak upper BB: pct_b={pct_b:.2f} RSI={rsi:.1f} falling'
        else:
            direction, confidence, sig_strength = 'HOLD', 0.44, 'NONE'
            reason = f'Weak BB: pct_b={pct_b:.2f} RSI={rsi:.1f} still rising'

    else:
        direction, confidence, sig_strength = 'HOLD', 0.45, 'NONE'
        reason = f'Mid-band (pct_b={pct_b:.2f}) — no edge'

    # Volume penalty
    volume_ok = _volume_is_healthy(hist)
    if not volume_ok and direction != 'HOLD':
        confidence = round(confidence * 0.80, 3)
        reason    += ' [vol −20% conf]'

    t1_mult, t2_mult, sl_mult = _get_rr_multipliers(sig_strength)

    if direction == 'BUY' and atr > 0:
        entry_price = price
        target_1    = round(price + t1_mult * atr, 6)
        stop_loss   = round(price - sl_mult * atr, 6)
    elif direction == 'SELL' and atr > 0:
        entry_price = price
        target_1    = round(price - t1_mult * atr, 6)
        stop_loss   = round(price + sl_mult * atr, 6)
    else:
        entry_price = price; target_1 = 0.0; stop_loss = 0.0

    sig_strength_val = (
        round(abs(rsi - 50) / 50.0, 3)
        if sig_strength in ('STRONG', 'MEDIUM') and direction != 'HOLD'
        else (round(max(0.0, 1.0 - width / avg_w), 3) if avg_w > 0 else 0.0)
    )

    return BrainSignal(
        brain_name='Causal-Ensemble',
        specialization='SQUEEZE Specialist',
        method='BB Squeeze Direction | 2:1 R:R minimum',
        direction=direction,
        confidence=round(confidence, 3),
        signal_strength=sig_strength_val,
        signal_age_candles=1,
        primary_evidence=reason,
        supporting_factors=[f'Tier: {sig_strength}', f'pct_b={pct_b:.2f}', f'RSI={rsi:.1f}'],
        contra_factors=['low_volume'] if not volume_ok else [],
        method_confidence=0.85,
        regime_suitability='LOW' if strong_trend else 'HIGH',
        reliability_flags={
            'squeeze_mode': is_squeeze, 'strong_trend': strong_trend,
            'low_volume':   not volume_ok,
        },
        measurements={
            'entry_price':       round(entry_price, 6),
            'target_1':          round(target_1, 6),
            'stop_loss':         round(stop_loss, 6),
            **base_meas,
            'sig_strength_tier': sig_strength,
            'decision_factor':   f'BB_{sig_strength}',
            'indicator_1_name':  'pct_b',    'indicator_1_value': round(pct_b, 3),
            'indicator_2_name':  'rsi',      'indicator_2_value': round(rsi, 1),
            'indicator_3_name':  'bb_width', 'indicator_3_value': round(width, 4),
        },
        rr_t1_mult=t1_mult, rr_t2_mult=t2_mult, rr_sl_mult=sl_mult,
        recent_accuracy=None, regime_accuracy=None,
    )