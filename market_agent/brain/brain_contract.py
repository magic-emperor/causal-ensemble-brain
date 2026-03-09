"""
Phase 4, Step 4.1 — BrainSignal Contract

This dataclass is the single contract that every brain function must return.
All 7 brains (Steps 4.2–4.8) must return a BrainSignal instance.
The council (cortex.py) only consumes BrainSignal objects — no plain dicts.

Usage:
    from market_agent.brain.brain_contract import BrainSignal
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BrainSignal:
    """
    Standardised output for every brain in the council.

    Required fields     — every brain must set these.
    Optional fields     — default to sensible values if a brain can't compute them.
    Computed fields     — filled by the health_monitor / cortex layer, not the brain.
    """

    # ── Identity ─────────────────────────────────────────────
    brain_name:       str          # e.g. 'AMV-LSTM', 'Regime Ensemble'
    specialization:   str          # one-line description of what this brain watches
    method:           str          # algorithm/model used e.g. 'LSTM + SMA crossover'

    # ── Core signal ──────────────────────────────────────────
    direction:        str          # 'BUY', 'SELL', or 'HOLD'
    confidence:       float        # 0.0–1.0  (how certain is this signal)

    # ── Evidence quality ─────────────────────────────────────
    signal_strength:      float    # 0.0–1.0  (magnitude, not direction)
    signal_age_candles:   int      # how many candles ago the pattern formed (0 = current)
    primary_evidence:     str      # the single most important reason for this signal
    supporting_factors:   List[str]    # additional evidence that supports the direction
    contra_factors:       List[str]    # evidence that argues AGAINST this direction
    method_confidence:    float        # 0.0–1.0  (how reliable is the method in general)
    regime_suitability:   str          # 'HIGH', 'MEDIUM', 'LOW' — is method right for current regime?

    # ── Optional / defaulted fields ───────────────────────────
    symbol:           str = ""     # symbol this signal was generated for (optional, added by caller)

    # ── Reliability flags ─────────────────────────────────────
    # Key: flag name  Value: True = flag raised (potential problem)
    # Each brain defines its own flags (e.g. 'low_volume', 'missing_timeframe')
    reliability_flags:    Dict[str, bool] = field(default_factory=dict)

    # ── Raw measurements ──────────────────────────────────────
    # Brain-specific numeric values for the Boss Brain to inspect
    # e.g. {'rsi': 34.2, 'macd_hist': 0.15, 'lstm_p_up': 0.72}
    measurements:         Dict[str, float] = field(default_factory=dict)

    # ── Accuracy (filled by health_monitor, not by brain) ────
    recent_accuracy:  Optional[float] = None   # last 30-trade accuracy for this brain
    regime_accuracy:  Optional[float] = None   # accuracy in *current* regime specifically

    # ── Dynamic Risk/Reward Multipliers (Optional) ───────────
    rr_t1_mult:       Optional[float] = None
    rr_t2_mult:       Optional[float] = None
    rr_sl_mult:       Optional[float] = None

    def to_debate_context(self) -> str:
        """
        Returns a compact, human-readable summary of this brain's signal.
        Used directly in the Boss Brain prompt and in council debate logging.
        """
        flags_str = ', '.join(
            k for k, v in self.reliability_flags.items() if v
        ) or 'none'

        supporting_str = ' | '.join(self.supporting_factors[:3]) or 'none'
        contra_str     = ' | '.join(self.contra_factors[:2])     or 'none'

        accuracy_str = ''
        if self.recent_accuracy is not None:
            accuracy_str = f"  Accuracy: {self.recent_accuracy:.0%} recent"
            if self.regime_accuracy is not None:
                accuracy_str += f" / {self.regime_accuracy:.0%} in current regime"

        meas_str = ', '.join(
            f"{k}={v:.3f}" for k, v in list(self.measurements.items())[:4]
        ) or 'N/A'

        return (
            f"[{self.brain_name}]\n"
            f"  Method: {self.method} ({self.specialization})\n"
            f"  Signal: {self.direction} | Confidence: {self.confidence:.0%} "
            f"| Strength: {self.signal_strength:.2f} | Age: {self.signal_age_candles} candles\n"
            f"  Evidence: {self.primary_evidence}\n"
            f"  Supporting: {supporting_str}\n"
            f"  Contra: {contra_str}\n"
            f"  Method confidence: {self.method_confidence:.0%} | "
            f"Regime suitability: {self.regime_suitability}\n"
            f"  Measurements: {meas_str}\n"
            f"  Flags: {flags_str}"
            f"{accuracy_str}"
        )

    def is_abstaining(self) -> bool:
        """Brain is not casting a vote — direction is HOLD and confidence is low."""
        return self.direction == 'HOLD' and self.confidence < 0.45

    def effective_confidence(self) -> float:
        """
        Confidence adjusted down by reliability flags.
        Each raised flag reduces confidence by 5% (max 30% penalty total).
        """
        penalties = sum(1 for v in self.reliability_flags.values() if v)
        penalty   = min(penalties * 0.05, 0.30)
        return max(0.0, self.confidence - penalty)
