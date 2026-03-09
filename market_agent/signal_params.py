"""
Single source of truth for ALL trading-critical parameters.

RULE: Nothing in backtest_engine, run_live_tournament, metrics, or position
should hardcode a trading threshold — import from here.

R:R = BASE_ATR_T1_MULT / BASE_ATR_SL_MULT = 0.75 / 0.50 = 1.5:1
EV at 44% WR: (0.44 × 0.75) - (0.56 × 0.50) = +0.05 → profitable
EV at 40% WR: (0.40 × 0.75) - (0.60 × 0.50) = 0.00 → breakeven
Evidence: Round 5/6 — old 1:1 R:R required >55% WR, never achieved by any brain.
"""

# ── R:R Multipliers ───────────────────────────────────────────────────────────
BASE_ATR_T1_MULT = 0.75    # Target 1 distance = ATR × this
BASE_ATR_T2_MULT = 1.50    # Target 2 distance = ATR × this
BASE_ATR_SL_MULT = 0.50    # Stop Loss distance = ATR × this → R:R = 1.5:1
                            # CHANGED from 0.75: Round 5/6 proved 1:1 is structurally losing

# Asset class adjustment factors are applied ON TOP of these values
# via market_agent/asset_class.py
# Final T1 dist = ATR * BASE_ATR_T1_MULT * asset_class.atr_t1_factor

# ── Signal Entry Gate ─────────────────────────────────────────────────────────
MIN_SIGNAL_CONFIDENCE      = 0.50   # Below this → treat as HOLD
                                     # MUST be same in backtest_engine AND run_live_tournament
SIGNAL_FLIP_MIN_CONFIDENCE = 0.60   # Brain must reach this confidence to force early exit

# ── Position Sizing ───────────────────────────────────────────────────────────
KELLY_MAX_POSITION_PCT  = 0.15    # Max fraction of capital deployed per trade (15%)
KELLY_MIN_POSITION_PCT  = 0.08    # Min fraction of capital deployed per trade (8%)
CONFIDENCE_SIZING_CAP   = 0.64    # Cap confidence used for Kelly sizing
                                   # Evidence: Round 4-5 — 0.80-conf signals != higher WR
                                   # Oversizing on overconfident signals increased losses

# ── Regime Detection Thresholds ───────────────────────────────────────────────
# Used by _detect_regime() in backtest_engine.py and run_live_tournament.py
REGIME_VOL_VOLATILE_THRESHOLD = 0.025   # rolling std > this → VOLATILE_CHAOS
REGIME_TREND_STABLE_THRESHOLD = 0.03    # |20-bar price change / price| > this → STABLE_TRADING
REGIME_VOL_HYBRID_THRESHOLD   = 0.005   # rolling std < this → HYBRID_SCAN
                                         # else → SCANNING_INTRADAY

# ── ATR Contraction Filter ────────────────────────────────────────────────────
ATR_CONTRACTION_THRESHOLD = 0.80   # Skip entry if current ATR-14 < this × 20-bar ATR average
                                    # Exception: mean-reversion brains in HYBRID_SCAN are exempt

# ── Expiry Extension ──────────────────────────────────────────────────────────
EXPIRY_EXTENSION_PROGRESS_MIN = 0.50   # Extend hold if trade moved >= this fraction toward T1
                                        # Only in non-BAD regimes (not VOLATILE_CHAOS/HYBRID_SCAN)

# ── System Limits ─────────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS  = 3         # Per-brain max concurrent open positions
CAPITAL_FLOOR_INR   = 5_000     # Brain enters CAPITAL_PRES mode below this INR
MIN_HISTORY_CANDLES = 25        # Minimum candles needed before generating any signal

# ── Per-Brain Post-T1 Mode ────────────────────────────────────────────────────
# Option A: No partial exit at T1 — full position runs to T2 or post-T1 SL
# Option C: 50% partial exit at T1, trail SL to T1 price (lock in breakeven)
BRAIN_POST_T1_MODE = {
    "Causal-Ensemble":    "A",   # A: -87 INR vs C: -647 INR
    "AMV-LSTM":           "A",   # A: -726 INR vs C: -1008 INR
    "Multi-Modal-Fusion": "A",   # A: -1153 INR vs C: -1633 INR
    "Cross-Stock-GNN":    "A",   # A: -1594 INR vs C: -1942 INR
    "Multi-Timeframe":    "C",   # C: -347 INR vs A: -1964 INR
    "Super-Brain":        "C",   # Round 6: STABLE_TRADING 71% WR — protect profits at T1
    "Consensus":          "C",   # C: -762 INR vs A: -894 INR
}


def get_post_t1_mode(brain_name: str) -> str:
    """Return 'A' or 'C' for the given brain. Defaults to C (safer)."""
    return BRAIN_POST_T1_MODE.get(brain_name, "C")