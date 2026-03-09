"""
Market Agent Configuration
Centralizes all hardcoded values, paths, and thresholds.
"""
import os
import json

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
MODELS_DIR = os.path.join(BASE_DIR, 'market_agent', 'models', 'checkpoints')
STRATEGY_PARAMS_PATH = os.path.join(CONFIG_DIR, 'strategy_params.json')
# FAISS index for council memory (override with env FAISS_INDEX_PATH)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(BASE_DIR, "faiss_index"))

# Accuracy stats: how many resolved predictions to consider.
# Default 50 is intentional (balance recency vs stability). Override with env ACCURACY_STATS_LAST_N.
ACCURACY_STATS_LAST_N = int(os.getenv("ACCURACY_STATS_LAST_N", "50"))

# Council brain IDs — used for per-brain signal attribution
# NAMING: must use HYPHENS to match BRAIN_ADAPTERS in brain_adapters.py
# and BRAIN_OPTIMAL_REGIMES in signal_generators.py
COUNCIL_BRAIN_IDS = [
    "AMV-LSTM", "Regime-Ensemble", "Multi-Modal-Fusion", "Multi-Timeframe",
    "Cross-Stock-GNN", "RL-Weighter", "Causal-Ensemble",
]
# When True, store_signal also stores one row per COUNCIL_BRAIN_IDS (same signal) so per-brain accuracy warms up
ATTRIBUTE_SIGNAL_TO_ALL_BRAINS = os.getenv("ATTRIBUTE_SIGNAL_TO_ALL_BRAINS", "false").lower() in ("1", "true", "yes")

# Optional mapping: If a brain is present here, it will ONLY run for the specified symbols.
# If a brain is not in this map, it runs on all symbols in the watchlist.
BRAIN_SYMBOL_MAP = {
    "Causal-Ensemble": [
        "AAPL", "AMD", "NVDA", "GOOGL", "MSFT", "META", "AMZN", "TSLA",
        "ADANIENT.NS", "ADANIPORTS.NS", "LT.NS",
        "GBPJPY=X", "GC=F"
    ]
}

# Path A: when True, scanner runs 7 brain generators and stores 7 real signals (no single-signal duplication)
USE_PATH_A_SEVEN_BRAINS = os.getenv("USE_PATH_A_SEVEN_BRAINS", "false").lower() in ("1", "true", "yes")

# ═══════════════════════════════════════════════════════════
# MACRO RISK (Brain Activation Phase 2)
# ═══════════════════════════════════════════════════════════
MACRO_RISK = {
    "HIGH_IMPACT_FACTOR": 0.5,   # Reduce risk by 50%
    "MEDIUM_IMPACT_FACTOR": 0.8, # Reduce risk by 20%
    "EVENTS": [
        "CPI", "FOMC", "NFP", "GDP", "Interest Rate", 
        "Jerome Powell", "Budget", "Elections"
    ],
    "LOOKAHEAD_MINUTES": 60,     # Check events coming in next 60 mins
    "LOOKBACK_MINUTES": 15,      # Check events that happened 15 mins ago
}

# ═══════════════════════════════════════════════════════════
# NEURAL MODELS (Brain Activation Phase 3)
# ═══════════════════════════════════════════════════════════
NEURAL_MODELS = {
    "MICRO_PRICE": os.path.join(MODELS_DIR, "micro_price_v1.pth"),
    "AMV_LSTM": os.path.join(MODELS_DIR, "best_amv_lstm.pth"),
    "CAUSAL_WEIGHTS": os.path.join(MODELS_DIR, "causal_weights.json"),
}

# ═══════════════════════════════════════════════════════════
# DYNAMIC CONFIDENCE
# ═══════════════════════════════════════════════════════════
def load_dynamic_confidence():
    """Load learnt confidence levels from strategy_params.json"""
    try:
        if os.path.exists(STRATEGY_PARAMS_PATH):
            with open(STRATEGY_PARAMS_PATH, 'r') as f:
                data = json.load(f)
                return data.get("brain_confidence", {})
    except Exception:
        pass
    # Fallback defects
    return {
        "SMA-Crossover": 0.55,
        "RSI-Momentum": 0.60,
        "MACD-Signal": 0.55,
        "Bollinger-Bounce": 0.65,
        "Volume-Breakout": 0.60,
    }

BRAIN_CONFIDENCE = load_dynamic_confidence()

# ═══════════════════════════════════════════════════════════
# TRADING PARAMS
# ═══════════════════════════════════════════════════════════
TRADING = {
    "DEFAULT_RISK": 1.5,
    "MAX_RISK": 2.5,
    "MIN_CONFIDENCE": 0.60,
    "LEVERAGE": 1,
}

# Backtest / paper trading: slippage and commission (Path A)
BACKTEST_SLIPPAGE_BPS = int(os.getenv("BACKTEST_SLIPPAGE_BPS", "5"))   # 5 bps = 0.05%
BACKTEST_COMMISSION_BPS = float(os.getenv("BACKTEST_COMMISSION_BPS", "10"))  # 10 bps = 0.1% per trade round-trip
BACKTEST_COMMISSION_PER_TRADE = float(os.getenv("BACKTEST_COMMISSION_PER_TRADE", "0"))  # Fixed $ per trade (0 = use bps)