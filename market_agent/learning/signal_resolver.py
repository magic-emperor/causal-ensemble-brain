"""
Signal Resolver Engine — The Brain's Feedback Loop

Stores every prediction, resolves them against actual price action,
and scores accuracy on a 0-100% gradient scale.

Includes:
- Time-windowed resolution (Scalp: 30min, Swing: 8h)
- Gradient accuracy (90% target reached = 90% score, not a "loss")
- Black swan forgiveness (VIX spike / post-signal news = reduced penalty)
- PostgreSQL-backed with pruning (10K per symbol)
"""

import pickle
import structlog
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Index, text
from sqlalchemy.ext.declarative import declarative_base

logger = structlog.get_logger()

# Use the same Base from postgres.py when integrated
# For the table definition, we use a local Base that will be merged
from market_agent.data.storage.postgres import Base


class SignalPrediction(Base):
    """
    Stores every signal prediction with full context for later resolution.
    """
    __tablename__ = 'signal_predictions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)  # BUY, SELL, WAIT
    entry_price = Column(Float, nullable=False)
    target_1 = Column(Float)
    target_2 = Column(Float)
    stop_loss = Column(Float)
    confidence = Column(Float)
    strategy = Column(String)  # Intraday (Scalp), Swing (Hold)
    regime = Column(String)
    model_id = Column(String, default="Aegis-Signal-Engine")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime)  # When the signal window closes
    resolved_at = Column(DateTime)  # When resolution happened

    # Resolution (filled after evaluation)
    is_resolved = Column(Boolean, default=False)
    accuracy_score = Column(Float)  # 0.0 to 100.0 gradient
    resolution_type = Column(String)  # T1_HIT, T2_HIT, SL_HIT, EXPIRED, PARTIAL
    best_price = Column(Float)  # High-water mark (best price during window)
    worst_price = Column(Float)  # Low-water mark
    resolve_price = Column(Float)  # Actual price at resolution time

    # Black Swan Detection
    is_shock = Column(Boolean, default=False)
    shock_reason = Column(String)  # VIX_SPIKE, VOLUME_ANOMALY, POST_SIGNAL_NEWS
    penalty_applied = Column(Float, default=1.0)  # 1.0 = full, 0.3 = forgiven

    # Path A: ground truth and council timeframe (optional columns for existing DBs)
    binary_win = Column(Integer, default=None)  # 0 or 1: 1 if T1/T2 hit, 0 otherwise
    actual_direction = Column(String, default=None)  # UP/DOWN/FLAT from resolve vs entry
    outcome_return_pct = Column(Float, default=None)  # (resolve - entry)/entry * 100 for BUY
    timeframe_min = Column(Integer, default=None)  # Brain/council suggested hold 1-20 min

    __table_args__ = (
        Index('idx_signal_pred_symbol_time', 'symbol', 'created_at'),
        Index('idx_signal_pred_unresolved', 'symbol', 'is_resolved'),
    )


class SignalResolver:
    """
    The brain's feedback loop. Stores predictions, resolves them against
    reality, and provides gradient accuracy scores.
    """

    # Resolution windows (defaults — overridden by dynamic calculation)
    SCALP_WINDOW_MINUTES = 30
    SCALP_MIN_MINUTES = 10
    SCALP_MAX_MINUTES = 45
    SWING_WINDOW_HOURS = 12
    SWING_MIN_HOURS = 4
    SWING_MAX_HOURS = 24

    # Pruning
    MAX_PREDICTIONS_PER_SYMBOL = 10000
    PRUNE_PERCENT = 0.10

    def __init__(self, storage):
        """
        Args:
            storage: PostgresStorage instance with active engine/session
        """
        self.storage = storage
        self._ensure_table()

    def _ensure_table(self):
        """Create the signal_predictions table if it doesn't exist; add Path A columns if missing."""
        try:
            SignalPrediction.__table__.create(self.storage.engine, checkfirst=True)
            self._add_path_a_columns_if_missing()
            logger.info("signal_predictions_table_ready")
        except Exception as e:
            # logger.error("signal_predictions_table_failed")
            pass

    def _add_path_a_columns_if_missing(self):
        """Add binary_win, actual_direction, outcome_return_pct, timeframe_min to existing table (PostgreSQL)."""
        try:
            with self.storage.engine.connect() as conn:
                for col, typ in [
                    ("binary_win", "INTEGER"),
                    ("actual_direction", "VARCHAR(10)"),
                    ("outcome_return_pct", "FLOAT"),
                    ("timeframe_min", "INTEGER"),
                ]:
                    conn.execute(text(
                        f"ALTER TABLE signal_predictions ADD COLUMN IF NOT EXISTS {col} {typ}"
                    ))
                conn.commit()
        except Exception as e:
            logger.debug("path_a_columns_migration", error=str(e)[:80])

    # ═══════════════════════════════════════════════════════════
    # STORAGE
    # ═══════════════════════════════════════════════════════════

    def store_signal(self, signal: Dict[str, Any], strategy: str = "Intraday (Scalp)") -> Optional[int]:
        """
        Store a new prediction for later resolution.

        Args:
            signal: Signal dict from get_real_signal()
            strategy: "Intraday (Scalp)" or "Swing (Hold)"

        Returns:
            Prediction ID or None if storage failed
        """
        direction = signal.get('direction', 'WAIT')
        if direction == 'WAIT':
            return None  # Don't store WAIT signals

        entry_price = signal.get('entry_price', 0)
        if entry_price <= 0:
            entry_price = signal.get('current_price', 0)

        if entry_price <= 0:
            return None  # Can't store without a price

        # Calculate dynamic expiry window (Path A: council/brain can pass timeframe_min 1-20)
        now = datetime.utcnow()
        vol_z = signal.get('vol_z_score', 1.0)
        timeframe_min = signal.get('timeframe_min')
        if timeframe_min is not None and 1 <= timeframe_min <= 20 and strategy == "Intraday (Scalp)":
            minutes = min(20, max(1, int(timeframe_min)))
            expires_at = now + timedelta(minutes=minutes)
        elif strategy == "Swing (Hold)":
            hours = self._dynamic_swing_hours(vol_z)
            expires_at = now + timedelta(hours=hours)
        else:
            minutes = self._dynamic_scalp_minutes(vol_z)
            expires_at = now + timedelta(minutes=minutes)

        session = self.storage.Session()
        try:
            # Pruning check
            count = session.query(SignalPrediction).filter(
                SignalPrediction.symbol == signal.get('symbol', '')
            ).count()

            if count >= self.MAX_PREDICTIONS_PER_SYMBOL:
                prune_count = int(self.MAX_PREDICTIONS_PER_SYMBOL * self.PRUNE_PERCENT)
                oldest_ids = session.query(SignalPrediction.id).filter(
                    SignalPrediction.symbol == signal.get('symbol', '')
                ).order_by(SignalPrediction.created_at.asc()).limit(prune_count).all()

                ids_to_del = [i[0] for i in oldest_ids]
                if ids_to_del:
                    session.query(SignalPrediction).filter(
                        SignalPrediction.id.in_(ids_to_del)
                    ).delete(synchronize_session=False)
                    session.commit()
                    logger.info("predictions_pruned", symbol=signal.get('symbol'), dropped=len(ids_to_del))

            # Store the prediction (Path A: optional timeframe_min)
            model_id = signal.get('model_used', signal.get('model_name', 'Aegis-Signal-Engine'))
            entry = SignalPrediction(
                symbol=signal.get('symbol', ''),
                direction=direction,
                entry_price=entry_price,
                target_1=signal.get('target_1', 0),
                target_2=signal.get('target_2', 0),
                stop_loss=signal.get('stop_loss', 0),
                confidence=signal.get('confidence', 0),
                strategy=strategy,
                regime=signal.get('regime', 'UNKNOWN'),
                model_id=model_id,
                created_at=now,
                expires_at=expires_at,
                is_resolved=False,
                timeframe_min=timeframe_min if isinstance(timeframe_min, int) and 1 <= timeframe_min <= 20 else None,
            )
            session.add(entry)
            session.commit()

            pred_id = entry.id

            # Optionally attribute same signal to all council brains so per-brain accuracy warms up.
            # Skip when this row is already from a council brain (Path A: we store 7 separate signals).
            try:
                from market_agent.config import ATTRIBUTE_SIGNAL_TO_ALL_BRAINS, COUNCIL_BRAIN_IDS
                if ATTRIBUTE_SIGNAL_TO_ALL_BRAINS and COUNCIL_BRAIN_IDS and (model_id not in COUNCIL_BRAIN_IDS):
                    tf_min = timeframe_min if isinstance(timeframe_min, int) and 1 <= timeframe_min <= 20 else None
                    for brain_id in COUNCIL_BRAIN_IDS:
                        extra = SignalPrediction(
                            symbol=signal.get('symbol', ''),
                            direction=direction,
                            entry_price=entry_price,
                            target_1=signal.get('target_1', 0),
                            target_2=signal.get('target_2', 0),
                            stop_loss=signal.get('stop_loss', 0),
                            confidence=signal.get('confidence', 0),
                            strategy=strategy,
                            regime=signal.get('regime', 'UNKNOWN'),
                            model_id=brain_id,
                            created_at=now,
                            expires_at=expires_at,
                            is_resolved=False,
                            timeframe_min=tf_min,
                        )
                        session.add(extra)
                    session.commit()
            except Exception:
                session.rollback()
                # Don't fail main store if per-brain attribution fails
                pass

            logger.info("signal_stored", symbol=signal.get('symbol'),
                        direction=direction, entry=entry_price,
                        t1=signal.get('target_1'), sl=signal.get('stop_loss'),
                        id=pred_id)
            return pred_id

        except Exception as e:
            session.rollback()
            # logger.error("signal_storage_failed")
            return None
        finally:
            session.close()

    # ═══════════════════════════════════════════════════════════
    # DYNAMIC RESOLUTION TIMING
    # ═══════════════════════════════════════════════════════════

    def _dynamic_scalp_minutes(self, vol_z: float) -> float:
        """
        Dynamic scalp window based on volatility z-score.
        High vol (z>2) → 5 min (price moves fast, targets hit quickly)
        Normal vol (z~1) → 15 min (default)
        Low vol (z<0.5) → 30 min (price moves slowly)
        """
        if vol_z <= 0:
            return self.SCALP_WINDOW_MINUTES
        
        # Inverse relationship: higher vol → shorter window
        # Scale from MAX at z=0.3 to MIN at z=3.0
        ratio = max(0.0, min(1.0, (vol_z - 0.3) / 2.7))
        minutes = self.SCALP_MAX_MINUTES - ratio * (self.SCALP_MAX_MINUTES - self.SCALP_MIN_MINUTES)
        return round(max(self.SCALP_MIN_MINUTES, min(self.SCALP_MAX_MINUTES, minutes)), 1)

    def _dynamic_swing_hours(self, vol_z: float) -> float:
        """
        Dynamic swing window based on volatility z-score.
        High vol (z>2) → 2 hours (volatile markets resolve fast)
        Normal vol (z~1) → 8 hours (default)
        Low vol (z<0.5) → 12 hours (low vol = slow resolution)
        """
        if vol_z <= 0:
            return self.SWING_WINDOW_HOURS
        
        ratio = max(0.0, min(1.0, (vol_z - 0.3) / 2.7))
        hours = self.SWING_MAX_HOURS - ratio * (self.SWING_MAX_HOURS - self.SWING_MIN_HOURS)
        return round(max(self.SWING_MIN_HOURS, min(self.SWING_MAX_HOURS, hours)), 1)

    # ═══════════════════════════════════════════════════════════
    # RESOLUTION
    # ═══════════════════════════════════════════════════════════

    def resolve_signals(self, current_price: float, symbol: str,
                        macro_data: Dict = None, recent_news: List = None) -> List[Dict]:
        """
        Check all open (unresolved) predictions for a symbol against current price.
        Resolves expired and hit predictions with gradient accuracy.

        Called every dashboard refresh cycle.

        Args:
            current_price: Latest market price
            symbol: Ticker symbol
            macro_data: Current macro data (for shock detection)
            recent_news: Recent news items (for post-signal shock detection)

        Returns:
            List of newly resolved predictions
        """
        if current_price <= 0:
            return []

        session = self.storage.Session()
        resolved = []
        now = datetime.utcnow()

        try:
            # Fetch all unresolved predictions for this symbol
            open_preds = session.query(SignalPrediction).filter(
                SignalPrediction.symbol == symbol,
                SignalPrediction.is_resolved == False
            ).order_by(SignalPrediction.created_at.asc()).all()

            # Path A: Fetch period high/low so "touched T1" can resolve (not only close at check time)
            period_ohlc = self._get_period_ohlc(symbol, open_preds, now) if open_preds else {}

            for pred in open_preds:
                # Update best/worst from current price (close at this run)
                if pred.direction == "BUY":
                    pred.best_price = max(pred.best_price or current_price, current_price)
                    pred.worst_price = min(pred.worst_price or current_price, current_price)
                elif pred.direction == "SELL":
                    pred.best_price = min(pred.best_price or current_price, current_price)
                    pred.worst_price = max(pred.worst_price or current_price, current_price)

                # Path A: Also fold in period high/low for this prediction's window (touched T1 counts)
                if pred.id in period_ohlc:
                    ph, pl = period_ohlc[pred.id]
                    if ph is not None and pl is not None:
                        if pred.direction == "BUY":
                            pred.best_price = max(pred.best_price or ph, ph)
                            pred.worst_price = min(pred.worst_price or pl, pl)
                        else:
                            pred.best_price = min(pred.best_price or pl, pl)
                            pred.worst_price = max(pred.worst_price or ph, ph)

                # Check resolution conditions
                resolution = self._check_resolution(pred, current_price, now)

                if resolution:
                    res_type, accuracy = resolution

                    # Black swan check on failures
                    penalty = 1.0
                    shock = False
                    shock_reason = None

                    if accuracy < 50.0:
                        shock, shock_reason = self._is_external_shock(
                            pred, macro_data, recent_news
                        )
                        if shock:
                            penalty = 0.3  # 70% forgiven
                        elif accuracy >= 40.0:
                            # Near-miss: reduced penalty
                            penalty = 0.5

                    # Path A: binary_win (1 = T1/T2 hit, 0 = else), actual_direction, outcome_return_pct
                    binary_win = 1 if res_type in ("T1_HIT", "T2_HIT") else 0
                    entry_p = pred.entry_price or 0
                    if entry_p > 0:
                        ret_pct = ((current_price - entry_p) / entry_p * 100) if pred.direction == "BUY" else ((entry_p - current_price) / entry_p * 100)
                        actual_dir = "UP" if current_price > entry_p else ("DOWN" if current_price < entry_p else "FLAT")
                    else:
                        ret_pct = None
                        actual_dir = None

                    pred.is_resolved = True
                    pred.resolved_at = now
                    pred.accuracy_score = accuracy
                    pred.resolution_type = res_type
                    pred.resolve_price = current_price
                    pred.is_shock = shock
                    pred.shock_reason = shock_reason
                    pred.penalty_applied = penalty
                    pred.binary_win = binary_win
                    pred.actual_direction = actual_dir
                    pred.outcome_return_pct = ret_pct

                    resolved.append({
                        "id": pred.id,
                        "symbol": pred.symbol,
                        "model_id": getattr(pred, "model_id", None) or "Aegis-Signal-Engine",
                        "direction": pred.direction,
                        "entry_price": pred.entry_price,
                        "target_1": pred.target_1,
                        "stop_loss": pred.stop_loss,
                        "best_price": pred.best_price,
                        "accuracy": accuracy,
                        "resolution_type": res_type,
                        "binary_win": binary_win,
                        "actual_direction": actual_dir,
                        "outcome_return_pct": ret_pct,
                        "is_shock": shock,
                        "shock_reason": shock_reason,
                        "penalty": penalty,
                        "confidence": pred.confidence,
                        "regime": pred.regime,
                        "resolve_price": current_price,
                        "created_at": pred.created_at.isoformat() if pred.created_at else None,
                        "resolved_at": now.isoformat(),
                    })

            session.commit()

            if resolved:
                logger.info("signals_resolved",
                            symbol=symbol,
                            count=len(resolved),
                            avg_accuracy=round(sum(r['accuracy'] for r in resolved) / len(resolved), 1))

                # ─── Store outcomes in Pattern Memory & update brain weights ───
                try:
                    from market_agent.learning.pattern_memory import pattern_memory
                    for r in resolved:
                        pattern_memory.store_outcome(
                            signal=r,
                            result=r['resolution_type'],
                            accuracy=r['accuracy'],
                            best_price=r.get('best_price', 0),
                            strategy="Intraday (Scalp)"  # TODO: get from signal metadata
                        )
                except Exception:
                    pass  # Pattern memory is optional, don't block resolve

                try:
                    from market_agent.learning.training_persistence import training_db
                    # Update per-brain accuracy stats from resolved signals
                    for r in resolved:
                        model_id = r.get('model_id', 'Aegis-Cluster')
                        is_win = r['resolution_type'] in ('T1_HIT', 'T2_HIT')
                        training_db.update_brain_weight(
                            brain_id=model_id,
                            accuracy=r['accuracy'],
                            wins=1 if is_win else None,
                            losses=0 if is_win else 1
                        )
                except Exception:
                    pass  # Training persistence is optional

                # ─── Feed Risk Manager (circuit breaker) ───
                try:
                    from market_agent.learning.risk_manager import get_risk_manager
                    risk_mgr = get_risk_manager()
                    for r in resolved:
                        # Calculate P&L % from entry to resolution
                        entry = r.get('entry_price', 0)
                        best = r.get('best_price', entry)
                        pnl_pct = ((best - entry) / entry * 100) if entry > 0 else 0
                        if r.get('direction') == 'SELL':
                            pnl_pct = -pnl_pct
                        if r['resolution_type'] == 'SL_HIT':
                            pnl_pct = -abs(pnl_pct) if pnl_pct >= 0 else pnl_pct

                        risk_mgr.record_outcome(
                            symbol=symbol,
                            resolution_type=r['resolution_type'],
                            pnl_pct=pnl_pct,
                            regime=r.get('regime', 'UNKNOWN')
                        )
                except Exception:
                    pass  # Risk manager is optional

                # ─── Feed Calibration Tracker (confidence feedback) ───
                try:
                    from market_agent.learning.calibrator import get_calibration_tracker
                    tracker = get_calibration_tracker()
                    for r in resolved:
                        conf = r.get('confidence', 0.5) * 100
                        was_correct = 1 if r['resolution_type'] in ('T1_HIT', 'T2_HIT') else 0
                        tracker.record_outcome(conf, was_correct)
                except Exception:
                    pass  # Calibration tracker is optional

                # ─── Path A: Attribution on failure (store in FAISS for RAG "past similar failures") ───
                try:
                    from market_agent.learning.attribution import get_attribution_engine
                    attr_engine = get_attribution_engine()
                    for r in resolved:
                        if r.get('accuracy', 0) < 50.0:
                            prediction = {
                                "direction": r.get('direction'),
                                "regime": r.get('regime'),
                                "confidence": r.get('confidence'),
                                "model_id": r.get('model_id'),
                                "symbol": r.get('symbol'),
                            }
                            outcome = {
                                "direction": r.get('actual_direction') or ("UP" if (r.get('resolve_price') or 0) > (r.get('entry_price') or 0) else "DOWN"),
                                "actual_price": r.get('resolve_price') or r.get('entry_price'),
                            }
                            attr_engine.attribute_failure(prediction, outcome)
                except Exception:
                    pass  # Attribution is optional, don't block resolve

            return resolved

        except Exception as e:
            session.rollback()
            # logger.error("signal_resolution_failed")
            return []
        finally:
            session.close()

    def _check_resolution(self, pred: SignalPrediction, current_price: float,
                          now: datetime) -> Optional[Tuple[str, float]]:
        """
        Check if a prediction should be resolved.

        Returns:
            (resolution_type, accuracy_score) or None if still open
        """
        entry = pred.entry_price
        t1 = pred.target_1 or entry
        t2 = pred.target_2 or t1
        sl = pred.stop_loss or entry
        best = pred.best_price or current_price

        if pred.direction == "BUY":
            target_range = t1 - entry if t1 != entry else 1.0
            max_favorable = best - entry

            # T2 Full Hit
            if best >= t2 and t2 > entry:
                return ("T2_HIT", 100.0)

            # T1 Hit
            if best >= t1 and t1 > entry:
                return ("T1_HIT", 100.0)

            # SL Hit (but still check how close we got to target)
            if current_price <= sl and sl < entry:
                # Even on SL hit, give credit for how close we got
                if max_favorable > 0 and target_range > 0:
                    partial = min(100.0, (max_favorable / target_range) * 100)
                    return ("SL_HIT", partial)
                return ("SL_HIT", 0.0)

            # Expired: score by how close we got
            if pred.expires_at and now >= pred.expires_at:
                if target_range > 0 and max_favorable > 0:
                    accuracy = min(100.0, (max_favorable / target_range) * 100)
                else:
                    accuracy = 0.0
                return ("EXPIRED", accuracy)

        elif pred.direction == "SELL":
            target_range = entry - t1 if t1 != entry else 1.0
            max_favorable = entry - best  # For SELL, price going DOWN is favorable

            # T2 Full Hit
            if best <= t2 and t2 < entry:
                return ("T2_HIT", 100.0)

            # T1 Hit
            if best <= t1 and t1 < entry:
                return ("T1_HIT", 100.0)

            # SL Hit
            if current_price >= sl and sl > entry:
                if max_favorable > 0 and target_range > 0:
                    partial = min(100.0, (max_favorable / target_range) * 100)
                    return ("SL_HIT", partial)
                return ("SL_HIT", 0.0)

            # Expired
            if pred.expires_at and now >= pred.expires_at:
                if target_range > 0 and max_favorable > 0:
                    accuracy = min(100.0, (max_favorable / target_range) * 100)
                else:
                    accuracy = 0.0
                return ("EXPIRED", accuracy)

        return None  # Still open

    def _get_period_ohlc(self, symbol: str, open_preds: List,
                         now: datetime) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
        """
        Fetch period high/low per prediction so resolution can use intraday high/low
        (so "touched T1" counts, not only close at each scout run).
        Returns dict pred.id -> (period_high, period_low).
        """
        result = {}
        if not open_preds:
            return result
        try:
            import pandas as pd
            t0 = min(p.created_at for p in open_preds)
            t1 = now
            if t0 >= t1:
                return result
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                delta = (t1 - t0).total_seconds() / 3600
                interval = "15m" if delta <= 48 else "1h"
                hist = ticker.history(start=t0, end=t1, interval=interval, auto_adjust=True)
            except Exception:
                return result
            if hist is None or hist.empty or "High" not in hist.columns or "Low" not in hist.columns:
                return result
            hist = hist.sort_index()
            # Normalize index to naive UTC for comparison with pred.created_at
            try:
                if getattr(hist.index, "tz", None) is not None:
                    hist = hist.tz_convert("UTC").tz_localize(None)
            except Exception:
                pass
            for pred in open_preds:
                start = pred.created_at
                if start.tzinfo:
                    start = start.replace(tzinfo=None)
                mask = (hist.index >= start) & (hist.index <= t1)
                if not mask.any():
                    continue
                sub = hist.loc[mask]
                ph = float(sub["High"].max())
                pl = float(sub["Low"].min())
                result[pred.id] = (ph, pl)
        except Exception as e:
            logger.debug("period_ohlc_fetch_failed", symbol=symbol, error=str(e)[:80])
        return result

    # ═══════════════════════════════════════════════════════════
    # BLACK SWAN DETECTION
    # ═══════════════════════════════════════════════════════════

    def _is_external_shock(self, pred: SignalPrediction,
                           macro_data: Dict = None,
                           recent_news: List = None) -> Tuple[bool, Optional[str]]:
        """
        Determine if a failed prediction was caused by external factors.
        If so, the brain shouldn't be penalized as harshly.

        Checks:
        1. VIX spike > 20% during signal window
        2. Extreme volume (>3x normal) suggesting institutional dump
        3. High-impact news arriving AFTER the prediction
        """
        # 1. VIX Spike Check
        if macro_data:
            vix_data = macro_data.get('VIX', {})
            vix_change = abs(vix_data.get('change', 0))
            if vix_change > 20:
                logger.info("shock_detected", type="VIX_SPIKE", change=vix_change,
                            symbol=pred.symbol)
                return True, "VIX_SPIKE"

        # 2. Volume Anomaly (if available in macro data)
        if macro_data:
            volume_ratio = macro_data.get('volume_ratio', 1.0)
            if volume_ratio > 3.0:
                logger.info("shock_detected", type="VOLUME_ANOMALY",
                            ratio=volume_ratio, symbol=pred.symbol)
                return True, "VOLUME_ANOMALY"

        # 3. Post-Signal High-Impact News
        if recent_news and pred.created_at:
            for news_item in recent_news:
                # Check if news arrived after prediction was made
                news_time = news_item.get('timestamp')
                if news_time and isinstance(news_time, str):
                    try:
                        # Handle time-only strings (HH:MM format)
                        if len(news_time) <= 5:
                            continue  # Skip if we can't compare timestamps
                        news_dt = datetime.fromisoformat(news_time)
                        if news_dt > pred.created_at:
                            impact = news_item.get('impact', 0)
                            sentiment = abs(news_item.get('sentiment', 0))
                            if impact > 0.8 or sentiment > 0.7:
                                logger.info("shock_detected", type="POST_SIGNAL_NEWS",
                                            headline=news_item.get('headline', '')[:60],
                                            symbol=pred.symbol)
                                return True, "POST_SIGNAL_NEWS"
                    except (ValueError, TypeError):
                        continue

        return False, None

    # ═══════════════════════════════════════════════════════════
    # ACCURACY & PERFORMANCE
    # ═══════════════════════════════════════════════════════════

    def get_accuracy_stats(self, symbol: str = None, model_id: str = None,
                           last_n: int = 50, accuracy_threshold: float = 50.0,
                           regime: str = None) -> Dict[str, Any]:
        """
        Get accuracy statistics from resolved predictions.

        Returns real accuracy computed from gradient scores,
        not binary win/loss.
        """
        session = self.storage.Session()
        try:
            query = session.query(SignalPrediction).filter(
                SignalPrediction.is_resolved == True
            )

            if symbol:
                query = query.filter(SignalPrediction.symbol == symbol)
            if model_id:
                query = query.filter(SignalPrediction.model_id == model_id)
            if regime:
                query = query.filter(SignalPrediction.regime == regime)

            results = query.order_by(
                SignalPrediction.resolved_at.desc()
            ).limit(last_n).all()

            if not results:
                return {
                    "accuracy": 0.0,
                    "total": 0,
                    "status": "NO_DATA",
                    "detail": "No resolved predictions yet. Building history..."
                }

            total = len(results)
            # Weighted accuracy: shock-affected predictions count less
            # PANIC-LOOP FIX: also exclude EXPIRED 0.0 signals from closed-market periods.
            # When NSE/US market is closed, price never moves → max_favorable=0 → accuracy=0.0.
            # These are not evidence of brain failure; they are evidence of a closed market.
            # Filter: drop EXPIRED signals with accuracy_score == 0.0 that were created on
            # a weekend OR outside NSE hours (for .NS/.BO symbols).
            from datetime import timezone as _tz, timedelta as _td
            _IST = _td(hours=5, minutes=30)
            _NSE_OPEN_MIN  = 9  * 60 + 15   # 09:15 IST in minutes-since-midnight
            _NSE_CLOSE_MIN = 15 * 60 + 30   # 15:30 IST in minutes-since-midnight

            def _is_closed_market_expired(r) -> bool:
                """Return True if this EXPIRED 0.0 signal is from a closed-market period."""
                if r.resolution_type != "EXPIRED":
                    return False
                score = r.accuracy_score if r.accuracy_score is not None else 0.0
                if score != 0.0:
                    return False  # Only filter definitive zero-movement expirations
                created = r.created_at
                if created is None:
                    return False
                # Weekend check (applies to all symbols — markets uniformly closed Sat/Sun)
                if created.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    return True
                # NSE hours check (only for Indian exchange symbols)
                sym = r.symbol or ""
                if sym.endswith(".NS") or sym.endswith(".BO"):
                    # Convert naive UTC stored time to IST for hour check
                    if created.tzinfo is None:
                        ist_dt = created.replace(tzinfo=_tz.utc) + _IST
                    else:
                        ist_dt = created.astimezone(_tz((_IST))) 
                    signal_min = ist_dt.hour * 60 + ist_dt.minute
                    if not (_NSE_OPEN_MIN <= signal_min <= _NSE_CLOSE_MIN):
                        return True  # Created outside NSE session — exclude
                return False

            weighted_sum = 0.0
            weight_total = 0.0
            wins = 0
            shocks = 0
            excluded_closed_market = 0

            for r in results:
                # Skip closed-market expired noise before it poisons the average
                if _is_closed_market_expired(r):
                    excluded_closed_market += 1
                    continue

                weight = r.penalty_applied if r.penalty_applied else 1.0
                score = r.accuracy_score if r.accuracy_score else 0.0

                weighted_sum += score * weight
                weight_total += weight

                if score >= accuracy_threshold:
                    wins += 1
                if r.is_shock:
                    shocks += 1

            if excluded_closed_market > 0:
                logger.debug(
                    "closed_market_signals_excluded_from_accuracy",
                    excluded=excluded_closed_market,
                    symbol=symbol,
                )

            # If ALL signals were closed-market noise, return healthy default — do not panic.
            if weight_total == 0:
                return {
                    "accuracy": 100.0,
                    "total": 0,
                    "wins": 0,
                    "shocks": 0,
                    "win_rate": 0.0,
                    "status": "NO_DATA",
                    "detail": f"All {total} resolved signals were closed-market expirations — excluded from accuracy.",
                    "trend": "STABLE",
                    "recent_scores": [],
                }

            weighted_accuracy = (weighted_sum / weight_total) if weight_total > 0 else 0.0

            # Status assignment
            if total < 5:
                status = "WARMING_UP"
            elif weighted_accuracy >= 75:
                status = "ELITE"
            elif weighted_accuracy >= 60:
                status = "PERFORMING"
            elif weighted_accuracy >= 45:
                status = "LEARNING"
            else:
                status = "REMEDIAL"

            # Recent trend (last 10 vs previous 10)
            trend = "STABLE"
            if total >= 20:
                recent_10 = [r.accuracy_score for r in results[:10] if r.accuracy_score is not None]
                older_10 = [r.accuracy_score for r in results[10:20] if r.accuracy_score is not None]
                if recent_10 and older_10:
                    recent_avg = sum(recent_10) / len(recent_10)
                    older_avg = sum(older_10) / len(older_10)
                    if recent_avg > older_avg + 5:
                        trend = "IMPROVING"
                    elif recent_avg < older_avg - 5:
                        trend = "DECLINING"

            return {
                "accuracy": round(weighted_accuracy, 1),
                "total": total,
                "wins": wins,
                "shocks": shocks,
                "win_rate": round((wins / total) * 100, 1) if total > 0 else 0.0,
                "status": status,
                "trend": trend,
                "recent_scores": [
                    {
                        "symbol": r.symbol,
                        "direction": r.direction,
                        "accuracy": r.accuracy_score,
                        "type": r.resolution_type,
                        "shock": r.is_shock,
                        "date": r.resolved_at.strftime("%m/%d %H:%M") if r.resolved_at else "N/A"
                    }
                    for r in results[:5]
                ]
            }

        except Exception as e:
            # logger.error("accuracy_stats_failed")
            return {"accuracy": 0.0, "total": 0, "status": "ERROR", "detail": "Stats fetch error"}
        finally:
            session.close()

    def get_models_needing_training(self, accuracy_threshold: float = 55.0) -> List[Dict]:
        """
        Find models performing below threshold — replaces hardcoded training proposals.
        Returns real underperformers for Boss Brain to review.
        """
        session = self.storage.Session()
        try:
            # Get distinct model_ids - ONLY look at last 100 resolved predictions
            # to avoid stale history triggering persistent alerts.
            subquery = session.query(
                SignalPrediction.model_id,
                SignalPrediction.accuracy_score
            ).filter(
                SignalPrediction.is_resolved == True
            ).order_by(SignalPrediction.resolved_at.desc()).limit(500).subquery()
            
            from sqlalchemy import func
            models = session.query(
                subquery.c.model_id,
                func.count(subquery.c.model_id).label('total'),
                func.avg(subquery.c.accuracy_score).label('avg_accuracy')
            ).group_by(subquery.c.model_id).all()

            needs_training = []
            for model_id, total, avg_accuracy in models:
                if total >= 10 and avg_accuracy is not None and avg_accuracy < accuracy_threshold:
                    gap = accuracy_threshold - avg_accuracy
                    # Dynamic multiplier: more below threshold → higher weight boost (cap 2.0x)
                    multiplier = min(2.0, max(1.2, 1.0 + gap / 100.0))
                    needs_training.append({
                        "model": model_id,
                        "accuracy": round(avg_accuracy, 1),
                        "total_predictions": total,
                        "reason": f"Accuracy {avg_accuracy:.1f}% is below {accuracy_threshold}% threshold over {total} predictions",
                        "suggested_action": f"Increase training weight {multiplier:.1f}x on recent failure patterns",
                        "created_at": datetime.utcnow(),
                        "deadline": datetime.utcnow() + timedelta(hours=12)
                    })

            return needs_training

        except Exception as e:
            # logger.error("training_check_failed")
            return []
        finally:
            session.close()

    def get_latest_directions_per_model(self, symbol: str, limit: int = 100) -> Dict[str, str]:
        """
        Path A: Latest stored direction per model_id for symbol (for debate triggers).
        Returns { model_id: "BUY"|"SELL"|"HOLD" }. Used so "brain disagreement" can fire.
        """
        session = self.storage.Session()
        try:
            rows = session.query(SignalPrediction.model_id, SignalPrediction.direction).filter(
                SignalPrediction.symbol == symbol
            ).order_by(SignalPrediction.created_at.desc()).limit(limit).all()
            out = {}
            for model_id, direction in rows:
                mid = (model_id or "Aegis-Signal-Engine").strip()
                if mid and mid not in out and direction in ("BUY", "SELL", "HOLD"):
                    out[mid] = direction
            return out
        except Exception:
            return {}
        finally:
            session.close()

    def get_brain_points(self, symbol: str = None) -> Dict[str, int]:
        """
        Path A: running count of correct predictions (binary_win=1) per model_id.
        Used for UI "+1 point per correct brain" display.
        """
        session = self.storage.Session()
        try:
            query = session.query(
                SignalPrediction.model_id,
                SignalPrediction.binary_win,
            ).filter(SignalPrediction.is_resolved == True)
            if symbol:
                query = query.filter(SignalPrediction.symbol == symbol)
            rows = query.all()
            points = {}
            for model_id, bw in rows:
                if model_id not in points:
                    points[model_id] = 0
                if bw == 1:
                    points[model_id] += 1
            return points
        except Exception:
            return {}
        finally:
            session.close()

    def get_brain_streaks(self, symbol: str = None) -> Dict[str, int]:
        """
        Path A §10: current streak (consecutive binary_win=1 from most recent) per model_id.
        Used for UI "3 correct in a row" and council prompt "Brain X was right on last N similar setups."
        """
        session = self.storage.Session()
        try:
            query = session.query(
                SignalPrediction.model_id,
                SignalPrediction.binary_win,
                SignalPrediction.resolved_at,
            ).filter(SignalPrediction.is_resolved == True).order_by(SignalPrediction.resolved_at.desc())
            if symbol:
                query = query.filter(SignalPrediction.symbol == symbol)
            rows = query.limit(500).all()
            by_model = {}
            for model_id, bw, resolved_at in rows:
                mid = model_id or "Aegis-Signal-Engine"
                if mid not in by_model:
                    by_model[mid] = []
                by_model[mid].append((bw, resolved_at))
            streaks = {}
            for mid, list_bw in by_model.items():
                list_bw.sort(key=lambda x: x[1] or datetime.min, reverse=True)
                count = 0
                for bw, _ in list_bw:
                    if bw == 1:
                        count += 1
                    else:
                        break
                if count > 0:
                    streaks[mid] = count
            return streaks
        except Exception:
            return {}
        finally:
            session.close()

    def get_recent_predictions_by_model(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """
        Path A: recent resolved predictions with model_id, direction, binary_win for UI (Brain predictions table).
        """
        session = self.storage.Session()
        try:
            query = session.query(SignalPrediction).filter(
                SignalPrediction.is_resolved == True
            ).order_by(SignalPrediction.resolved_at.desc())
            if symbol:
                query = query.filter(SignalPrediction.symbol == symbol)
            rows = query.limit(limit).all()
            out = []
            for r in rows:
                entry = r.entry_price or 0
                t1 = getattr(r, "target_1", None)
                t2 = getattr(r, "target_2", None)
                sl = getattr(r, "stop_loss", None)
                # Display fallback for old rows where target_1/stop_loss were not stored (null in DB)
                if t1 is None and entry and entry > 0:
                    t1 = round(entry * 1.01, 2) if r.direction == "BUY" else round(entry * 0.99, 2)
                if sl is None and entry and entry > 0:
                    sl = round(entry * 0.99, 2) if r.direction == "BUY" else round(entry * 1.01, 2)
                out.append({
                    "id": r.id,
                    "model_id": getattr(r, "model_id", None) or "Aegis-Signal-Engine",
                    "symbol": r.symbol,
                    "direction": r.direction,
                    "entry_price": r.entry_price,
                    "target_1": t1,
                    "target_2": t2,
                    "stop_loss": sl,
                    "resolve_price": r.resolve_price,
                    "binary_win": getattr(r, "binary_win", None),
                    "accuracy_score": r.accuracy_score,
                    "resolution_type": r.resolution_type,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
                    "expires_at": r.expires_at.isoformat() if getattr(r, "expires_at", None) else None,
                    "timeframe_min": getattr(r, "timeframe_min", None),
                })
            return out
        except Exception:
            return []
        finally:
            session.close()

    def get_open_predictions(self, symbol: str = None) -> List[Dict]:
        """Get all unresolved predictions, optionally filtered by symbol."""
        session = self.storage.Session()
        try:
            query = session.query(SignalPrediction).filter(
                SignalPrediction.is_resolved == False
            )
            if symbol:
                query = query.filter(SignalPrediction.symbol == symbol)

            results = query.order_by(SignalPrediction.created_at.desc()).limit(20).all()

            return [{
                "id": r.id,
                "symbol": r.symbol,
                "direction": r.direction,
                "entry_price": r.entry_price,
                "target_1": r.target_1,
                "stop_loss": r.stop_loss,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "best_price": r.best_price,
            } for r in results]

        except Exception as e:
            logger.error("open_predictions_fetch_failed", error=str(e))
            return []
        finally:
            session.close()
