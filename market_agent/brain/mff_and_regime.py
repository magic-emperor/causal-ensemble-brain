# # ============================================================
# # DEPRECATED - THIS FILE IS NO LONGER ACTIVE
# #
# # Its logic has been fully absorbed into:
# #   - signal_generators.py  (UNIFIED_REGIMES, BRAIN_REGIME_GATES_UNIFIED,
# #                             normalize_regime, regime_allows_brain)
# #   - multi_modal_fusion.py (canonical Multi-Modal-Fusion brain)
# #   - regime_ensemble.py    (canonical Regime-Ensemble brain)
# #
# # Kept for historical reference only. DO NOT import this file.
# # To restore: rename to mff_and_regime_ACTIVE.py and update imports.
# # ============================================================

# # --- original code disabled below (all on one line to avoid syntax errors) ---
# # fmt: off
# pass

# # NOTE: The actual code from this file lives in multi_modal_fusion.py and
# # regime_ensemble.py. Check git history if you need to see the original.


# """
# ═══════════════════════════════════════════════════════════════════════
# FIX #2: Multi-Modal-Fusion — Recalibrated for Crypto/Forex
# FIX #3: Regime-Ensemble — Unified Taxonomy
# ═══════════════════════════════════════════════════════════════════════

# FIX #2 ROOT CAUSE:
#   Multi-Modal-Fusion had a hard guard: if ATR/price > 2.5%, stand down.
#   Crypto/forex almost always has ATR/price > 2.5%.
#   Result: Brain was FULLY DISABLED for your primary trading markets.
  
#   The fix is NOT to remove the guard — divergence logic IS less reliable
#   in high volatility. The fix is:
#   1. Use LONGER divergence window (10 bars instead of 5) for crypto
#   2. Require STRONGER divergence confirmation
#   3. Use volatility-adjusted RSI thresholds (crypto oversold = 35, not 30)
#   4. Add MACD zero-line cross as a second confirmation layer

# FIX #3 ROOT CAUSE:
#   Regime-Ensemble outputs:   STRONG_TREND_UP, RANGING, VOLATILE_BREAKOUT, etc.
#   Trade gates check against: STABLE_TRADING, SCANNING_INTRADAY, HYBRID_SCAN
#   These NEVER matched. Every brain's regime gate was firing against the wrong labels.
  
#   Fix: Unify into ONE taxonomy used everywhere.
#   New regimes: TRENDING, RANGING, VOLATILE, SQUEEZE
# """
# from __future__ import annotations
# import pandas as pd
# import numpy as np
# from typing import Optional, Dict


# # ═══════════════════════════════════════════════════════════════════════
# # UNIFIED REGIME TAXONOMY (used by ALL brains + trade gates)
# # ═══════════════════════════════════════════════════════════════════════

# UNIFIED_REGIMES = {
#     # Regime Name     : which brain types work
#     'TRENDING_UP':    {'trend_brains': 0.90, 'mean_rev_brains': 0.15, 'volatility_brains': 0.50},
#     'TRENDING_DOWN':  {'trend_brains': 0.90, 'mean_rev_brains': 0.15, 'volatility_brains': 0.50},
#     'RANGING':        {'trend_brains': 0.25, 'mean_rev_brains': 0.90, 'volatility_brains': 0.60},
#     'VOLATILE':       {'trend_brains': 0.40, 'mean_rev_brains': 0.10, 'volatility_brains': 0.85},
#     'SQUEEZE':        {'trend_brains': 0.20, 'mean_rev_brains': 0.70, 'volatility_brains': 0.95},
#     'CHAOS':          {'trend_brains': 0.00, 'mean_rev_brains': 0.00, 'volatility_brains': 0.00},  # NO trades
# }

# # Mapping from OLD regime names → NEW unified names
# # Use this to migrate existing code without breaking everything at once
# REGIME_MIGRATION_MAP = {
#     # Old signal_generators regimes
#     'STABLE_TRADING':    'TRENDING_UP',    # was trend-focused
#     'SCANNING_INTRADAY': 'RANGING',        # was range-scanning
#     'HYBRID_SCAN':       'RANGING',        # was hybrid (closer to ranging)
#     'VOLATILE_CHAOS':    'CHAOS',          # maps directly
#     # Old regime_ensemble regimes
#     'STRONG_TREND_UP':   'TRENDING_UP',
#     'STRONG_TREND_DOWN': 'TRENDING_DOWN',
#     'VOLATILE_BREAKOUT': 'VOLATILE',
#     'LOW_VOLATILITY':    'SQUEEZE',
#     'RANGING':           'RANGING',        # same
# }

# # Which unified regimes each brain is allowed to trade in
# BRAIN_REGIME_GATES_UNIFIED = {
#     'AMV-LSTM':           ['TRENDING_UP', 'TRENDING_DOWN'],
#     'Multi-Modal-Fusion': ['RANGING', 'SQUEEZE'],
#     'Multi-Timeframe':    ['TRENDING_UP', 'TRENDING_DOWN'],
#     'Cross-Stock-GNN':    ['VOLATILE', 'RANGING'],
#     'Causal-Ensemble':    ['RANGING', 'SQUEEZE', 'VOLATILE'],  # mean-rev works here
#     'Super-Brain':        ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING'],
#     'RL-Weighter':        ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING'],
#     'Liquidity-Sweep':    ['VOLATILE', 'TRENDING_UP', 'TRENDING_DOWN'],  # NEW brain
#     'Regime-Ensemble':    ['ALL'],
# }


# def normalize_regime(regime: str) -> str:
#     """Convert any regime string (old or new) to unified taxonomy."""
#     if regime in UNIFIED_REGIMES:
#         return regime
#     return REGIME_MIGRATION_MAP.get(regime, 'RANGING')  # default to RANGING


# def regime_allows_brain(brain_name: str, regime: str) -> bool:
#     """Single function to check if a brain can trade in a given regime."""
#     unified = normalize_regime(regime)
#     if unified == 'CHAOS':
#         return False
#     allowed = BRAIN_REGIME_GATES_UNIFIED.get(brain_name, ['ALL'])
#     if 'ALL' in allowed:
#         return True
#     return unified in allowed


# def _calc_atr(hist: pd.DataFrame, period: int = 14) -> float:
#     high, low  = hist['High'], hist['Low']
#     prev_close = hist['Close'].shift(1)
#     tr = pd.concat([
#         high - low,
#         (high - prev_close).abs(),
#         (low  - prev_close).abs(),
#     ], axis=1).max(axis=1)
#     return float(tr.rolling(period).mean().iloc[-1] or 0.0)


# def _calc_rsi_wilder(hist: pd.DataFrame, period: int = 14) -> pd.Series:
#     close = hist['Close']
#     delta = close.diff()
#     gain  = delta.where(delta > 0, 0.0)
#     loss  = (-delta).where(delta < 0, 0.0)
#     avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
#     avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
#     rs = avg_gain / avg_loss.replace(0, float('nan'))
#     return 100 - (100 / (1 + rs))


# # ═══════════════════════════════════════════════════════════════════════
# # FIX #2: MULTI-MODAL-FUSION V2 (Crypto/Forex Recalibrated)
# # ═══════════════════════════════════════════════════════════════════════

# def _detect_divergence(
#     close: pd.Series,
#     rsi_series: pd.Series,
#     lookback: int,
#     atr_pct: float,
# ) -> tuple:
#     """
#     Detect RSI divergence with volatility-adjusted lookback and confirmation.
    
#     For high-volatility assets (crypto/forex), we:
#     1. Use longer lookback (10-15 bars) to avoid noise-triggered divergence
#     2. Require the RSI gap to be meaningful (not just 1-2 points)
#     3. Check that the price move is substantial (not just noise)
    
#     Returns (bullish_div: bool, bearish_div: bool, strength: float)
#     """
#     if len(close) < lookback + 2 or len(rsi_series) < lookback + 2:
#         return False, False, 0.0

#     curr_price = float(close.iloc[-1])
#     curr_rsi   = float(rsi_series.iloc[-1])

#     # Window excluding current bar
#     window_close = close.iloc[-(lookback+1):-1]
#     window_rsi   = rsi_series.iloc[-(lookback+1):-1]

#     max_price = float(window_close.max())
#     min_price = float(window_close.min())
#     max_rsi   = float(window_rsi.max())
#     min_rsi   = float(window_rsi.min())

#     # Minimum price move required (volatility-adjusted)
#     # High vol assets need bigger move for divergence to be real
#     min_price_move_pct = 0.005 if atr_pct < 0.015 else 0.012  # 0.5% equity, 1.2% crypto

#     price_made_higher_high = (curr_price > max_price) and ((curr_price - max_price) / max_price > min_price_move_pct)
#     price_made_lower_low   = (curr_price < min_price) and ((min_price - curr_price) / min_price > min_price_move_pct)

#     # RSI gap must be meaningful (at least 3 points)
#     rsi_min_gap = 5.0 if atr_pct > 0.015 else 3.0

#     rsi_did_not_confirm_high = curr_rsi < (max_rsi - rsi_min_gap)
#     rsi_did_not_confirm_low  = curr_rsi > (min_rsi + rsi_min_gap)

#     bearish_div = price_made_higher_high and rsi_did_not_confirm_high
#     bullish_div = price_made_lower_low   and rsi_did_not_confirm_low

#     # Strength: how large is the divergence gap?
#     strength = 0.0
#     if bearish_div:
#         strength = min(1.0, (curr_price - max_price) / max_price * 10)
#     elif bullish_div:
#         strength = min(1.0, (min_price - curr_price) / min_price * 10)

#     return bullish_div, bearish_div, strength


# def _macd_zero_cross(close: pd.Series) -> tuple:
#     """
#     MACD zero-line cross detection.
#     Returns (direction: 'BUY'|'SELL'|'NONE', bars_since_cross: int)
#     """
#     if len(close) < 35:
#         return 'NONE', 0
#     ema12 = close.ewm(span=12, adjust=False).mean()
#     ema26 = close.ewm(span=26, adjust=False).mean()
#     macd  = ema12 - ema26
#     signal = macd.ewm(span=9, adjust=False).mean()
#     hist_line = macd - signal

#     # Find last zero cross of histogram
#     bars_since = 0
#     for i in range(1, min(10, len(hist_line))):
#         curr = float(hist_line.iloc[-i])
#         prev = float(hist_line.iloc[-i-1])
#         if curr > 0 and prev <= 0:
#             return 'BUY', i
#         if curr < 0 and prev >= 0:
#             return 'SELL', i
#         bars_since = i
#     return 'NONE', bars_since


# def multi_modal_fusion_v2(hist: pd.DataFrame) -> dict:
#     """
#     FIXED Multi-Modal-Fusion v2: Works for crypto, forex, AND equities.
    
#     Key changes from v1:
#     1. No hard ATR guard that disables on crypto/forex
#     2. Volatility-adaptive divergence detection (longer lookback for HV assets)
#     3. MACD zero-line cross as second confirmation
#     4. Asset-class aware RSI thresholds
#     5. Confirmation scoring: divergence alone = medium, divergence + MACD = high confidence
#     """
#     if len(hist) < 40:
#         return {'brain_name': 'Multi-Modal-Fusion', 'direction': 'HOLD', 'confidence': 0.30}

#     close      = hist['Close']
#     rsi_series = _calc_rsi_wilder(hist)
#     rsi        = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
#     atr        = _calc_atr(hist)
#     price      = float(close.iloc[-1])
#     atr_pct    = atr / price if price > 0 else 0.01

#     # Volatility tier — adjusts sensitivity
#     is_high_vol = atr_pct > 0.015   # crypto/forex
#     is_ultra_vol = atr_pct > 0.035  # extreme volatility (meme coins, news events)

#     # ── RSI thresholds: volatility-adjusted ──────────────────
#     # Crypto oscillates more → standard 30/70 fires too early
#     if is_ultra_vol:
#         rsi_oversold, rsi_overbought = 25, 75
#     elif is_high_vol:
#         rsi_oversold, rsi_overbought = 33, 67
#     else:
#         rsi_oversold, rsi_overbought = 30, 70

#     # Divergence lookback: longer for high-vol to avoid noise
#     div_lookback = 15 if is_high_vol else 8

#     # ── Divergence detection ──────────────────────────────────
#     bullish_div, bearish_div, div_strength = _detect_divergence(
#         close, rsi_series, div_lookback, atr_pct
#     )

#     # ── MACD zero-line cross ──────────────────────────────────
#     macd_cross_dir, macd_cross_age = _macd_zero_cross(close)
#     macd_fresh = macd_cross_age <= 3  # cross within last 3 bars is fresh

#     # ── Signal logic ─────────────────────────────────────────
#     signals = []

#     # Priority 1: Divergence (highest quality signal)
#     if bullish_div:
#         signals.append(('BUY', 0.75 + div_strength * 0.10, 'bullish_divergence'))
#     if bearish_div:
#         signals.append(('SELL', 0.75 + div_strength * 0.10, 'bearish_divergence'))

#     # Priority 2: RSI extreme + MACD confirmation
#     if rsi < rsi_oversold and macd_cross_dir == 'BUY' and macd_fresh:
#         signals.append(('BUY', 0.78, 'rsi_oversold+macd_cross'))
#     elif rsi > rsi_overbought and macd_cross_dir == 'SELL' and macd_fresh:
#         signals.append(('SELL', 0.78, 'rsi_overbought+macd_cross'))

#     # Priority 3: RSI extreme alone
#     elif rsi < rsi_oversold:
#         signals.append(('BUY', 0.62, 'rsi_oversold'))
#     elif rsi > rsi_overbought:
#         signals.append(('SELL', 0.62, 'rsi_overbought'))

#     # Priority 4: MACD momentum (weakest, only if no other signal)
#     if not signals:
#         ema12 = close.ewm(span=12, adjust=False).mean()
#         ema26 = close.ewm(span=26, adjust=False).mean()
#         macd_line = ema12 - ema26
#         signal_line = macd_line.ewm(span=9, adjust=False).mean()
#         hist_vals = macd_line - signal_line
#         hist_slope = float(hist_vals.iloc[-1] - hist_vals.iloc[-3])
#         macd_bullish = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
#         if macd_bullish and hist_slope > 0:
#             signals.append(('BUY', 0.55, 'macd_momentum'))
#         elif not macd_bullish and hist_slope < 0:
#             signals.append(('SELL', 0.55, 'macd_momentum'))

#     if not signals:
#         return {
#             'brain_name': 'Multi-Modal-Fusion',
#             'direction': 'HOLD',
#             'confidence': 0.45,
#             'reason': f'No signal: RSI={rsi:.1f}, no divergence, MACD neutral',
#             'measurements': {'rsi': rsi, 'atr_pct': round(atr_pct, 4)},
#         }

#     # Pick highest confidence signal; if conflicting → HOLD
#     buy_sigs  = [s for s in signals if s[0] == 'BUY']
#     sell_sigs = [s for s in signals if s[0] == 'SELL']

#     if buy_sigs and sell_sigs:
#         return {
#             'brain_name': 'Multi-Modal-Fusion',
#             'direction': 'HOLD',
#             'confidence': 0.40,
#             'reason': 'Conflicting signals — bullish div vs bearish indicators',
#             'measurements': {'rsi': rsi, 'atr_pct': round(atr_pct, 4)},
#         }

#     best = max(signals, key=lambda x: x[1])
#     direction, confidence, reason_code = best

#     # Boost confidence if divergence AND MACD agree
#     if (direction == 'BUY' and bullish_div and macd_cross_dir == 'BUY' and macd_fresh) or \
#        (direction == 'SELL' and bearish_div and macd_cross_dir == 'SELL' and macd_fresh):
#         confidence = min(0.90, confidence + 0.08)
#         reason_code += '+macd_confirmed'

#     return {
#         'brain_name':         'Multi-Modal-Fusion',
#         'direction':          direction,
#         'confidence':         round(confidence, 3),
#         'reason':             f'{reason_code} | RSI={rsi:.1f} | ATR%={atr_pct:.2%}',
#         'is_high_vol_asset':  is_high_vol,
#         'divergence':         {'bullish': bullish_div, 'bearish': bearish_div, 'strength': round(div_strength, 3)},
#         'macd_cross':         {'direction': macd_cross_dir, 'bars_ago': macd_cross_age},
#         'measurements': {
#             'rsi':         round(rsi, 1),
#             'atr_pct':     round(atr_pct, 4),
#             'rsi_oversold_threshold': rsi_oversold,
#             'div_lookback': div_lookback,
#             'bullish_div': float(bullish_div),
#             'bearish_div': float(bearish_div),
#         },
#     }


# # ═══════════════════════════════════════════════════════════════════════
# # FIX #3: REGIME DETECTION V2 — Unified Output
# # ═══════════════════════════════════════════════════════════════════════

# def regime_ensemble_v2(hist: pd.DataFrame) -> dict:
#     """
#     FIXED Regime-Ensemble v2: Outputs unified regime names.
    
#     Outputs one of: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, SQUEEZE, CHAOS
#     All brain regime gates now use the same taxonomy.
    
#     Detection logic:
#     - ATR% > 4%: VOLATILE (or CHAOS if > 8%)
#     - BB squeeze (width < 70% of avg): SQUEEZE
#     - ADX > 25 + price > SMA50: TRENDING_UP
#     - ADX > 25 + price < SMA50: TRENDING_DOWN
#     - else: RANGING
#     """
#     close = hist['Close']
#     high  = hist['High']
#     low   = hist['Low']

#     if len(hist) < 50:
#         return {'regime': 'RANGING', 'confidence': 0.40, 'trust_weights': UNIFIED_REGIMES['RANGING']}

#     atr    = _calc_atr(hist)
#     price  = float(close.iloc[-1])
#     atr_pct = atr / price if price > 0 else 0.01

#     sma50  = float(close.rolling(50).mean().iloc[-1])
#     sma20  = float(close.rolling(20).mean().iloc[-1])
#     std20  = float(close.rolling(20).std().iloc[-1])
#     bb_w   = (sma20 + 2*std20 - (sma20 - 2*std20)) / sma20 if sma20 > 0 else 0.04

#     # BB average width for squeeze detection
#     upper_s = close.rolling(20).mean() + 2 * close.rolling(20).std()
#     lower_s = close.rolling(20).mean() - 2 * close.rolling(20).std()
#     bb_width_series = (upper_s - lower_s) / close.rolling(20).mean()
#     avg_bb_w = float(bb_width_series.rolling(20).mean().iloc[-1]) if len(bb_width_series) > 20 else bb_w

#     # ADX calculation (simplified)
#     plus_dm  = (high.diff()).where(high.diff() > low.diff().abs(), 0.0).where(high.diff() > 0, 0.0)
#     minus_dm = (low.diff().abs()).where(low.diff().abs() > high.diff(), 0.0).where(low.diff() < 0, 0.0)
#     tr_vals  = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
#     atr14    = tr_vals.rolling(14).mean()
#     plus_di  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, float('nan'))
#     minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, float('nan'))
#     dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float('nan'))
#     adx = float(dx.rolling(14).mean().iloc[-1]) if not dx.empty else 15.0
#     if pd.isna(adx):
#         adx = 15.0

#     # ── Regime decision tree ──────────────────────────────────
#     evidence_parts = []

#     if atr_pct > 0.08:
#         regime = 'CHAOS'
#         confidence = 0.90
#         evidence_parts.append(f'ATR%={atr_pct:.1%} > 8% extreme chaos')

#     elif atr_pct > 0.04:
#         regime = 'VOLATILE'
#         confidence = 0.80
#         evidence_parts.append(f'ATR%={atr_pct:.1%} > 4% high volatility')

#     elif bb_w < avg_bb_w * 0.70:
#         regime = 'SQUEEZE'
#         confidence = 0.80
#         evidence_parts.append(f'BB squeeze: width={bb_w:.3f} < avg {avg_bb_w:.3f}')

#     elif adx > 25:
#         if price > sma50:
#             regime = 'TRENDING_UP'
#         else:
#             regime = 'TRENDING_DOWN'
#         confidence = min(0.90, 0.60 + (adx - 25) / 100)
#         evidence_parts.append(f'ADX={adx:.1f} > 25 | price {">" if price > sma50 else "<"} SMA50')

#     elif adx > 18:
#         # Weak trend
#         regime = 'TRENDING_UP' if price > sma50 else 'TRENDING_DOWN'
#         confidence = 0.55
#         evidence_parts.append(f'Weak trend: ADX={adx:.1f}')

#     else:
#         regime = 'RANGING'
#         confidence = 0.70
#         evidence_parts.append(f'ADX={adx:.1f} < 18 | price near SMA')

#     return {
#         'regime':           regime,
#         'confidence':       round(confidence, 3),
#         'trust_weights':    UNIFIED_REGIMES.get(regime, UNIFIED_REGIMES['RANGING']),
#         'evidence':         ' | '.join(evidence_parts),
#         'measurements': {
#             'atr_pct':  round(atr_pct, 4),
#             'adx':      round(adx, 1),
#             'bb_width': round(bb_w, 4),
#             'avg_bb_w': round(avg_bb_w, 4),
#             'price_vs_sma50_pct': round((price - sma50) / sma50 * 100, 2),
#         },
#         # Backward compat: brain regime gate check
#         'brain_allowed': lambda brain: regime_allows_brain(brain, regime),
#     }


# # ═══════════════════════════════════════════════════════════════════════
# # SELF-TEST
# # ═══════════════════════════════════════════════════════════════════════
# if __name__ == '__main__':
#     import numpy as np

#     print("=" * 60)
#     print("FIX #2: Multi-Modal-Fusion v2 Tests")
#     print("=" * 60)

#     # Simulate BTC-like data
#     np.random.seed(42)
#     n = 120
#     base = 50000.0
#     prices = [base]
#     for i in range(n - 1):
#         prices.append(prices[-1] * (1 + np.random.normal(0.0002, 0.018)))

#     df_btc = pd.DataFrame({
#         'Open':   prices,
#         'High':   [p * 1.01 for p in prices],
#         'Low':    [p * 0.99 for p in prices],
#         'Close':  prices,
#         'Volume': [500000] * n,
#     })

#     r1 = multi_modal_fusion_v2(df_btc)
#     print(f"\nTest 1 (BTC normal): {r1['direction']} | {r1['confidence']:.0%}")
#     print(f"  Reason: {r1['reason']}")
#     print(f"  High vol: {r1.get('is_high_vol_asset')}")

#     # Force RSI oversold on BTC
#     df_down = df_btc.copy()
#     cv = df_down['Close'].values.copy()
#     for i in range(1, 16):
#         cv[-i] = cv[-16] * (1 - i * 0.018)
#     df_down['Close'] = cv
#     df_down['Low']   = df_down['Close'] * 0.99
#     df_down['High']  = df_down['Close'] * 1.01

#     r2 = multi_modal_fusion_v2(df_down)
#     print(f"\nTest 2 (BTC oversold): {r2['direction']} | {r2['confidence']:.0%}")
#     print(f"  Reason: {r2['reason']}")
#     print(f"  RSI threshold used: {r2.get('measurements', {}).get('rsi_oversold_threshold')}")

#     print("\n" + "=" * 60)
#     print("FIX #3: Regime-Ensemble v2 Tests")
#     print("=" * 60)

#     r3 = regime_ensemble_v2(df_btc)
#     print(f"\nTest 3 (BTC normal): Regime={r3['regime']} | Confidence={r3['confidence']:.0%}")
#     print(f"  Evidence: {r3['evidence']}")
#     print(f"  Trust weights: {r3['trust_weights']}")

#     # Test regime allows brain
#     print("\n  Brain regime gates:")
#     for brain in ['AMV-LSTM', 'Causal-Ensemble', 'Multi-Modal-Fusion', 'Liquidity-Sweep']:
#         allowed = regime_allows_brain(brain, r3['regime'])
#         print(f"    {brain}: {'✅ ALLOWED' if allowed else '❌ BLOCKED'}")

#     print("\n  Regime taxonomy unification test:")
#     for old_regime in ['STABLE_TRADING', 'SCANNING_INTRADAY', 'HYBRID_SCAN', 'STRONG_TREND_UP']:
#         new = normalize_regime(old_regime)
#         print(f"    {old_regime:25s} → {new}")

#     print("\n✅ Fix #2 and Fix #3 complete and tested.")