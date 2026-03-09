"""
Asset class detection and per-class trading parameters.
Applied ON TOP of base ATR multipliers from signal_params.py.

Usage:
    from market_agent.asset_class import get_asset_class, get_params
    params = get_params(symbol)
    final_t1 = atr * BASE_ATR_T1_MULT * params['atr_t1_factor']
"""

# ── Asset class detection ──────────────────────────────────────────

_CRYPTO  = {'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD'}
_COMMODITY = {'GC=F', 'CL=F', 'SI=F', 'NG=F', 'ZC=F', 'ZW=F'}
_US_EQUITY = {'AAPL', 'NVDA', 'GOOGL', 'AMD', 'MSFT', 'AMZN', 'META', 'TSLA'}


def get_asset_class(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol.endswith('.NS') or symbol.endswith('.BO') or symbol.startswith('^NSE'):
        return 'NSE'
    if symbol in _CRYPTO or symbol.endswith('-USD'):
        return 'CRYPTO'
    if symbol in _COMMODITY or symbol.endswith('=F'):
        return 'COMMODITY'
    if symbol.endswith('=X'):
        return 'FOREX'
    if symbol in _US_EQUITY:
        return 'US_EQUITY'
    return 'US_EQUITY'  # default


# ── Per-class parameters ───────────────────────────────────────────
# ATR factors multiply the BASE values from signal_params.py
# BASE_ATR_T1_MULT = 0.75, BASE_ATR_T2_MULT = 1.50, BASE_ATR_SL_MULT = 0.50 (1.5:1 R:R)
# Final effective values per class shown in comments below (T1 dist / SL dist)

ASSET_CLASS_PARAMS = {
    'NSE': {
        # Less volatile — tighter targets
        'atr_t1_factor':     0.80,     # T1 = 0.75 * 0.80 = 0.60 ATR
        'atr_t2_factor':     0.80,     # T2 = 1.50 * 0.80 = 1.20 ATR
        'atr_sl_factor':     0.80,     # SL = 0.50 * 0.80 = 0.40 ATR → R:R = 0.60/0.40 = 1.5:1 ✓
        'scalp_expiry_min':  90,       # NSE session is only 375 min
        'swing_expiry_min':  375,      # one full session
        'min_confidence':    0.60,
        'trades_24_7':       False,
    },
    'FOREX': {
        'atr_t1_factor':     1.07,     # T1 = 0.75 * 1.07 = 0.80 ATR
        'atr_t2_factor':     1.07,     # T2 = 1.50 * 1.07 = 1.60 ATR
        'atr_sl_factor':     1.00,     # SL = 0.50 * 1.00 = 0.50 ATR → R:R = 0.80/0.50 = 1.6:1 ✓
        'scalp_expiry_min':  240,
        'swing_expiry_min':  720,
        'min_confidence':    0.55,
        'trades_24_7':       True,
    },
    'CRYPTO': {
        # Higher vol — wider targets, more time
        'atr_t1_factor':     1.33,     # T1 = 0.75 * 1.33 = 1.00 ATR
        'atr_t2_factor':     1.33,     # T2 = 1.50 * 1.33 = 2.00 ATR
        'atr_sl_factor':     1.20,     # SL = 0.50 * 1.20 = 0.60 ATR → R:R = 1.00/0.60 = 1.67:1 ✓
        'scalp_expiry_min':  240,
        'swing_expiry_min':  480,
        'min_confidence':    0.58,
        'trades_24_7':       True,
    },
    'COMMODITY': {
        'atr_t1_factor':     1.00,     # T1 = 0.75 * 1.00 = 0.75 ATR
        'atr_t2_factor':     1.00,     # T2 = 1.50 * 1.00 = 1.50 ATR
        'atr_sl_factor':     1.00,     # SL = 0.50 * 1.00 = 0.50 ATR → R:R = 0.75/0.50 = 1.5:1 ✓
        'scalp_expiry_min':  240,
        'swing_expiry_min':  480,
        'min_confidence':    0.57,
        'trades_24_7':       True,
    },
    'US_EQUITY': {
        'atr_t1_factor':     0.93,     # T1 = 0.75 * 0.93 = 0.70 ATR
        'atr_t2_factor':     0.93,     # T2 = 1.50 * 0.93 = 1.40 ATR
        'atr_sl_factor':     0.90,     # SL = 0.50 * 0.90 = 0.45 ATR → R:R = 0.70/0.45 = 1.56:1 ✓
        'scalp_expiry_min':  120,      # US session only 6.5h
        'swing_expiry_min':  390,
        'min_confidence':    0.58,
        'trades_24_7':       False,
    },
}


def get_params(symbol: str) -> dict:
    """Return trading parameters for the asset class of *symbol*."""
    return ASSET_CLASS_PARAMS[get_asset_class(symbol)]