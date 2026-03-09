import pandas as pd
import pickle
import structlog
from datetime import datetime, timedelta
from typing import List, Dict, Any
from market_agent.data.storage.postgres import PostgresStorage, Predictions

logger = structlog.get_logger()

class RegretEngine:
    """
    Layer 4: Self-Awareness
    Compares past predictions against actual price movements.
    Tracks success (praise) and failure (improvement triggers).
    Now integrated with SignalResolver for real gradient accuracy.
    """
    def __init__(self, storage: PostgresStorage):
        self.storage = storage
        self.evaluation_history = []  # In-memory cache
        self.model_performance = {}   # Track per-model accuracy
        self.praise_log = []          # Log of successful predictions
        self.improvement_queue = []   # Queue of models needing training

        # Signal Resolver Integration (real accuracy from stored predictions)
        try:
            from market_agent.learning.signal_resolver import SignalResolver
            self.signal_resolver = SignalResolver(storage)
        except Exception as e:
            # logger.error("signal_resolver_init_failed") # Silenced to survive Windows console encoding issues
            self.signal_resolver = None
    
    def evaluate_prediction(self, prediction: Dict, actual_outcome: str, actual_price: float) -> Dict:
        """
        Compare a single prediction with actual market outcome.
        Returns evaluation with success/failure and praise/improvement trigger.
        """
        predicted_direction = prediction.get('direction', 'UP')
        predicted_confidence = prediction.get('confidence', 0.5)
        model_name = prediction.get('model', 'Unknown')
        symbol = prediction.get('symbol', 'Unknown')
        
        # Calculate actual direction from price change
        basis_price = prediction.get('basis_price', actual_price)
        price_change = (actual_price - basis_price) / basis_price * 100
        
        if price_change > 0.1:
            actual_direction = "UP"
        elif price_change < -0.1:
            actual_direction = "DOWN"
        else:
            actual_direction = "FLAT"
        
        # Determine if prediction was correct
        is_correct = (predicted_direction == actual_direction) or \
                     (predicted_direction == "FLAT" and abs(price_change) < 0.2)
        
        # Calculate error magnitude
        error_magnitude = abs(price_change) if not is_correct else 0.0
        
        # Build evaluation result
        eval_result = {
            "prediction_id": prediction.get('id'),
            "symbol": symbol,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "confidence": predicted_confidence,
            "is_correct": is_correct,
            "price_change_pct": round(price_change, 3),
            "error_magnitude": round(error_magnitude, 3),
            "praise": None,
            "improvement_action": None
        }
        
        # PRAISE for success
        if is_correct:
            if predicted_confidence > 0.8:
                eval_result["praise"] = f"🏆 EXCELLENT! {model_name} predicted {predicted_direction} with {predicted_confidence:.0%} confidence and was CORRECT! High-confidence win."
            elif predicted_confidence > 0.6:
                eval_result["praise"] = f"✅ Good job {model_name}! Correctly identified {actual_direction} move."
            else:
                eval_result["praise"] = f"👍 {model_name} got it right despite low confidence. Learning opportunity."
            self.praise_log.append(eval_result)
        
        # IMPROVEMENT TRIGGER for failure
        else:
            if predicted_confidence > 0.7:
                eval_result["improvement_action"] = f"⚠️ HIGH-REGRET: {model_name} was {predicted_confidence:.0%} confident but WRONG. Needs immediate retraining on {symbol} pattern."
                self.improvement_queue.append({
                    "model": model_name,
                    "reason": f"High-confidence failure on {symbol}",
                    "error_magnitude": error_magnitude,
                    "priority": "HIGH",
                    "timestamp": datetime.now()
                })
            else:
                eval_result["improvement_action"] = f"📊 {model_name} missed this one. Low-confidence failure - add to training batch."
                self.improvement_queue.append({
                    "model": model_name,
                    "reason": f"Low-confidence failure on {symbol}",
                    "error_magnitude": error_magnitude,
                    "priority": "NORMAL",
                    "timestamp": datetime.now()
                })
        
        # Update model performance tracking
        self._update_model_accuracy(model_name, is_correct)
        
        # Store evaluation
        self.evaluation_history.append(eval_result)
        
        return eval_result
    
    def _update_model_accuracy(self, model_name: str, is_correct: bool):
        """Track rolling accuracy per model."""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {"correct": 0, "total": 0, "accuracy": 0.0}
        
        self.model_performance[model_name]["total"] += 1
        if is_correct:
            self.model_performance[model_name]["correct"] += 1
        
        total = self.model_performance[model_name]["total"]
        correct = self.model_performance[model_name]["correct"]
        self.model_performance[model_name]["accuracy"] = round(correct / total * 100, 1)
    
    def get_models_needing_training(self, accuracy_threshold: float = 55.0) -> List[Dict]:
        """Get list of models below accuracy threshold."""
        needs_training = []
        for model, stats in self.model_performance.items():
            if stats["accuracy"] < accuracy_threshold and stats["total"] >= 10:
                needs_training.append({
                    "model": model,
                    "accuracy": stats["accuracy"],
                    "total_predictions": stats["total"],
                    "reason": f"Accuracy {stats['accuracy']}% is below {accuracy_threshold}% threshold",
                    "suggested_action": "Increase training weight 1.5x on recent failures"
                })
        return needs_training
    
    def get_praise_summary(self) -> List[Dict]:
        """Get recent successful predictions for Boss Brain praise."""
        return self.praise_log[-10:]  # Last 10 praises
    
    def get_improvement_queue(self) -> List[Dict]:
        """Get pending improvement actions."""
        return self.improvement_queue
    
    def clear_improvement_queue(self, model_name: str = None):
        """Clear improvement queue after training is dispatched."""
        if model_name:
            self.improvement_queue = [q for q in self.improvement_queue if q['model'] != model_name]
        else:
            self.improvement_queue = []

    def get_real_accuracy(self, model_name: str = None, symbol: str = None,
                          regime: str = None) -> Dict[str, Any]:
        """
        Unified Performance API: Fetches real gradient accuracy from SignalResolver.
        Supports per-regime filtering.
        Falls back to in-memory tracking if resolver unavailable.
        No random numbers — returns honest data.
        """
        # Priority 1: Use SignalResolver (real gradient accuracy from DB)
        if self.signal_resolver:
            stats = self.signal_resolver.get_accuracy_stats(
                symbol=symbol, model_id=model_name, regime=regime
            )
            if stats.get('total', 0) > 0:
                return stats

        # Priority 2: In-memory tracking (from evaluate_prediction calls)
        if model_name and model_name in self.model_performance:
            perf = self.model_performance[model_name]
            return {
                "accuracy": perf["accuracy"],
                "total": perf["total"],
                "wins": perf["correct"],
                "status": "PERFORMING" if perf["accuracy"] > 60 else "LEARNING",
                "trend": "STABLE",
            }

        # Priority 3: Honest "no data" response (NO random numbers)
        return {
            "accuracy": 0.0,
            "total": 0,
            "status": "NO_DATA",
            "detail": "Building prediction history. Accuracy will update after signals resolve."
        }

    def consult_boss_brain(self, model_name: str, failure_timestamp: datetime) -> Dict[str, Any]:
        """
        Phase 3: AI-Powered Boss Consultation.
        Uses FAISS recall + AI analysis + attribution engine.
        No more static 'Greatest Hits' strings.
        """
        # 1. FAISS recall of similar past failures
        similar_context = ""
        try:
            from market_agent.brain.council_memory import get_council_memory
            memory = get_council_memory()
            similar = memory.recall_similar(
                query=f"{model_name} failure analysis",
                memory_type="TRADE_LOSS", k=3
            )
            if similar:
                similar_context = "Past similar failures: " + "; ".join(
                    s.get("text", "")[:100] for s in similar
                )
        except Exception:
            pass
        
        # 2. Get brain's real accuracy
        accuracy = self.get_real_accuracy(model_name)
        acc_pct = accuracy.get("accuracy", 0)
        total = accuracy.get("total", 0)
        trend = accuracy.get("trend", "STABLE")
        
        # 3. Try AI analysis
        audit_reasoning = ""
        try:
            from market_agent.brain.gemini_client import gemini_client
            if gemini_client and gemini_client.is_available:
                prompt = (
                    f"Boss Brain audit for {model_name}:\n"
                    f"- Current accuracy: {acc_pct:.1f}% over {total} predictions\n"
                    f"- Trend: {trend}\n"
                    f"- Failed at: {failure_timestamp}\n"
                    f"{similar_context}\n"
                    f"In 2-3 sentences, explain likely causes and specific corrections."
                )
                audit_reasoning = gemini_client._call_ai(
                    prompt, f"boss_audit_{model_name}"
                )
        except Exception:
            pass
        
        # 4. Data-only fallback
        if not audit_reasoning:
            audit_reasoning = (
                f"Boss Audit: {model_name} at {acc_pct:.1f}% accuracy "
                f"({total} predictions, {trend} trend). "
                f"{similar_context or 'No similar past failures found.'}"
            )
        
        verdict = "RETRAIN_REQUIRED" if acc_pct < 50 else "CONTINUE"
        
        return {
            "verdict": verdict,
            "audit": audit_reasoning,
            "accuracy": acc_pct,
            "trend": trend,
            "timestamp": datetime.now().isoformat()
        }

    def evaluate_performance(self, limit=100, symbol=None) -> List[Dict[str, Any]]:
        """
        Matches predictions with reality and calculates errors.
        Enhanced: praise/improvement triggers + council trainer RL updates.
        """
        pending = self.storage.get_pending_evaluations(limit=limit, symbol=symbol)
        evaluations = []
        
        for pred in pending:
            # Reality check: Get the next candle for this symbol/timestamp
            reality_data = self.storage.get_latest_data(pred.symbol, "1m", limit=1)
            if not reality_data:
                continue
            
            actual_close = reality_data[0]["data"]["close"]
            
            # Create prediction dict for evaluation
            pred_dict = {
                "id": pred.id,
                "symbol": pred.symbol,
                "model": pred.model_id,
                "direction": pred.regime, 
                "confidence": pickle.loads(pred.conf_score) if pred.conf_score else 0.5,
                "basis_price": actual_close * 0.99 # Mocking basis price
            }
            
            eval_result = self.evaluate_prediction(pred_dict, pred.regime, actual_close)
            eval_result["predicted_at"] = pred.timestamp
            evaluations.append(eval_result)
            
            # TRIGGER FEEDBACK LOOP: If accuracy drops, consult the Boss
            accuracy_info = self.get_real_accuracy(pred.model_id)
            if accuracy_info["accuracy"] < 55:
                audit = self.consult_boss_brain(pred.model_id, pred.timestamp)
                eval_result["boss_audit"] = audit
        
        # RL: Auto-resolve debates against actual price outcomes
        try:
            from market_agent.learning.council_trainer import get_council_trainer
            trainer = get_council_trainer(self.storage)
            rl_result = trainer.auto_resolve_from_prices()
            if rl_result.get("resolved", 0) > 0:
                logger.info("rl_debates_resolved",
                             resolved=rl_result["resolved"],
                             checked=rl_result["checked"])
        except Exception:
            pass
        
        return evaluations


if __name__ == "__main__":
    storage = PostgresStorage()
    engine = RegretEngine(storage)
    
    # Test with mock prediction
    mock_pred = {
        "id": 1,
        "symbol": "ITC.NS",
        "model": "AMV-LSTM",
        "direction": "UP",
        "confidence": 0.85,
        "basis_price": 450.0
    }
    
    result = engine.evaluate_prediction(mock_pred, "UP", 455.0)
    print("Evaluation Result:", result)
    
    # Test Boss Consultation
    audit = engine.consult_boss_brain("AMV-LSTM", datetime.now())
    print("Boss Audit:", audit)
