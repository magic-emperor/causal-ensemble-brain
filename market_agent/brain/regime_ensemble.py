"""
Brain 2: Regime-Ensemble — Market Condition Classifier (Meta Brain)
===================================================================
Classifies the current market regime. Does NOT generate a directional signal.
Output is used by signal_generators.py to gate which other brains are allowed.

Taxonomy (Unified):
  TRENDING_UP    — ADX > 25 (strong), 22-25 (developing), price > SMA50
  TRENDING_DOWN  — ADX > 25 (strong), 22-25 (developing), price < SMA50
  RANGING        — ADX < 22 (no trend separation)
  VOLATILE       -- ATR% > 2.5x its own 60-bar median (adaptive per asset class)
  SQUEEZE        -- BB width < 70% of its 20-bar average (coiling)
  CHAOS          -- ATR% > 5x its own 60-bar median (extreme -- NO trades allowed)

KEY CHANGES:
  - ADX threshold raised from 18 → 22 for weak trend, 25+ for strong trend.
    ADX 18-22 is "developing/borderline" — calling it TRENDING_UP/DOWN at ADX=18
    was activating trend brains in ranging markets and causing losses.
  - Added 'computed_regime' to measurements dict.
    Previously the regime was buried inside primary_evidence as a string like
    "Regime: TRENDING_UP | ...". signal_generators.py had to string-parse to
    extract it. Now: bs.measurements['computed_regime'] gives the clean string.

Returns: BrainSignal with direction='HOLD' (meta brain — no direction)
"""
from __future__ import annotations

import pandas as pd
from market_agent.brain.brain_contract import BrainSignal
from market_agent.brain.brain_utils import calc_atr, calc_adx

# Must match UNIFIED_REGIMES in signal_generators.py
UNIFIED_REGIMES = {
    'TRENDING_UP':    {'trust_trend_brains': 0.90, 'trust_mean_rev': 0.15, 'volatility_brains': 0.50},
    'TRENDING_DOWN':  {'trust_trend_brains': 0.90, 'trust_mean_rev': 0.15, 'volatility_brains': 0.50},
    'RANGING':        {'trust_trend_brains': 0.25, 'trust_mean_rev': 0.90, 'volatility_brains': 0.60},
    'VOLATILE':       {'trust_trend_brains': 0.40, 'trust_mean_rev': 0.10, 'volatility_brains': 0.85},
    'SQUEEZE':        {'trust_trend_brains': 0.20, 'trust_mean_rev': 0.70, 'volatility_brains': 0.95},
    'CHAOS':          {'trust_trend_brains': 0.00, 'trust_mean_rev': 0.00, 'volatility_brains': 0.00},
}


def regime_ensemble_signal(hist: pd.DataFrame) -> BrainSignal:
    """
    Brain 2: Market Condition Classifier (Meta Brain).
    Returns HOLD direction — no directional signal.
    signal_generators.py reads regime from measurements['computed_regime']
    and uses it to gate all other brains.
    """
    if len(hist) < 50:
        return BrainSignal(
            brain_name='Regime-Ensemble',
            specialization='Market Condition Classifier -- Meta Brain',
            method='Unified Taxonomy (ATR%, BB width, ADX)',
            direction='HOLD',
            confidence=0.40,
            signal_strength=0.0,
            signal_age_candles=0,
            primary_evidence='Insufficient data for regime classification',
            supporting_factors=[],
            contra_factors=[],
            method_confidence=0.40,
            regime_suitability='HIGH',
            reliability_flags={'insufficient_data': True},
            measurements={'computed_regime': 'RANGING'},   # safe default
            recent_accuracy=None,
            regime_accuracy=None,
        )

    close = hist['Close']
    high  = hist['High']
    low   = hist['Low']

    atr     = calc_atr(hist, 14)                          # ← brain_utils canonical
    adx     = calc_adx(hist, 14)                          # ← brain_utils canonical
    price   = float(close.iloc[-1])
    atr_pct = atr / price if price > 0 else 0.01

    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    std20 = float(close.rolling(20).std().iloc[-1])
    bb_w  = (sma20 + 2 * std20 - (sma20 - 2 * std20)) / sma20 if sma20 > 0 else 0.04

    upper_s       = close.rolling(20).mean() + 2 * close.rolling(20).std()
    lower_s       = close.rolling(20).mean() - 2 * close.rolling(20).std()
    bb_width_series = (upper_s - lower_s) / close.rolling(20).mean()
    avg_bb_w      = float(bb_width_series.rolling(20).mean().iloc[-1]) if len(bb_width_series) > 20 else bb_w

    # ── Regime decision tree ──────────────────────────────────────────────────
    # Priority order: CHAOS > VOLATILE > SQUEEZE > TRENDING (strong) > TRENDING (weak) > RANGING
    evidence_parts    = []
    regime_suitability = 'HIGH'   # A4: overridden to MEDIUM for borderline ADX 22-25

    # ── A3: Adaptive ATR thresholds ─────────────────────────────────────────
    # Compute 60-bar median ATR% as the asset's own "normal" volatility baseline.
    # This makes thresholds self-calibrating:
    #   BTC  (median ~3.5%) → volatile=8.75%, chaos=17.5%  (4%/8% was too low)
    #   NIFTY (median ~0.8%) → volatile=2.0%,  chaos=5.0%   (floors prevent zero)
    # Fallback: if < 60 bars, use absolute thresholds (same as before).
    close_s  = hist['Close']
    high_s   = hist['High']
    low_s    = hist['Low']
    prev_c   = close_s.shift(1)
    tr_s     = pd.concat([high_s - low_s,
                           (high_s - prev_c).abs(),
                           (low_s  - prev_c).abs()], axis=1).max(axis=1)
    atr_pct_series = tr_s / close_s

    if len(hist) >= 60:
        median_atr_pct     = float(atr_pct_series.iloc[-60:].median())
        volatile_threshold = max(0.02, median_atr_pct * 2.5)
        chaos_threshold    = max(0.05, median_atr_pct * 5.0)
    else:
        # Insufficient history — fall back to fixed absolute thresholds
        median_atr_pct     = None
        volatile_threshold = 0.04
        chaos_threshold    = 0.08
    # ─────────────────────────────────────────────────────────────────────────

    if atr_pct > chaos_threshold:
        regime     = 'CHAOS'
        confidence = 0.90
        evidence_parts.append(
            f'ATR%={atr_pct:.1%} > chaos_threshold={chaos_threshold:.1%} -- extreme, no trades'
        )

    elif atr_pct > volatile_threshold:
        regime     = 'VOLATILE'
        confidence = 0.80
        evidence_parts.append(
            f'ATR%={atr_pct:.1%} > volatile_threshold={volatile_threshold:.1%} -- high volatility'
        )

    elif bb_w < avg_bb_w * 0.70:
        regime     = 'SQUEEZE'
        confidence = 0.80
        evidence_parts.append(f'BB squeeze: width={bb_w:.3f} < avg {avg_bb_w:.3f}')

    elif adx > 25:
        # Strong confirmed trend
        regime     = 'TRENDING_UP' if price > sma50 else 'TRENDING_DOWN'
        confidence = min(0.90, 0.65 + (adx - 25) / 100)
        evidence_parts.append(f'Strong trend: ADX={adx:.1f} > 25 | price {">" if price > sma50 else "<"} SMA50')

    elif adx > 22:
        # Developing/borderline trend — lower confidence to reflect uncertainty.
        # ADX 22-25 is NOT a confirmed trend. We still call it TRENDING for gate purposes,
        # but confidence is only 0.55 and suitability is MEDIUM to signal this to callers.
        regime             = 'TRENDING_UP' if price > sma50 else 'TRENDING_DOWN'
        confidence         = 0.55
        regime_suitability = 'MEDIUM'    # A4: borderline — less certain classification
        evidence_parts.append(f'Developing trend: ADX={adx:.1f} (22-25 borderline)')

    else:
        # ADX < 22 = NO trend. RANGING.
        regime     = 'RANGING'
        confidence = 0.72
        evidence_parts.append(f'Ranging: ADX={adx:.1f} < 22 | no trend separation')

    weights = UNIFIED_REGIMES.get(regime, UNIFIED_REGIMES['RANGING'])

    return BrainSignal(
        brain_name='Regime-Ensemble',
        specialization='Market Condition Classifier -- Meta Brain',
        method='Unified Taxonomy (ATR%, BB width, ADX)',
        direction='HOLD',   # meta brain does NOT vote direction
        confidence=confidence,
        signal_strength=min(1.0, adx / 50.0),
        signal_age_candles=0,
        primary_evidence=f'Regime: {regime} | ' + ' | '.join(evidence_parts),
        supporting_factors=[
            f'Trend brain trust weight: {weights["trust_trend_brains"]:.0%}',
            f'Mean-rev brain trust weight: {weights["trust_mean_rev"]:.0%}',
            f'Volatility brain trust weight: {weights["volatility_brains"]:.0%}',
        ],
        contra_factors=[],
        method_confidence=0.85,
        regime_suitability=regime_suitability,
        reliability_flags={},
        measurements={
            'computed_regime':    regime,
            'atr_pct':            round(atr_pct, 4),
            'adx':                round(adx, 1),
            'bb_width':           round(bb_w, 4),
            'avg_bb_w':           round(avg_bb_w, 4),
            'price_vs_sma50_pct': round((price - sma50) / sma50 * 100, 2),
            'volatile_threshold': round(volatile_threshold, 4),
            'chaos_threshold':    round(chaos_threshold, 4),
            'median_atr_pct':     round(median_atr_pct, 4) if median_atr_pct is not None else None,
            # A12: decision logging
            'decision_factor':    'REGIME_CLASSIFY',
            'price_at_signal':    round(price, 6),
            'atr_at_signal':      round(float(atr), 6),
            'atr_pct_at_signal':  round(float(atr) / price * 100, 3) if price > 0 else 0.0,
            'bars_used':          len(hist),
        },
        recent_accuracy=None,
        regime_accuracy=None,
    )
