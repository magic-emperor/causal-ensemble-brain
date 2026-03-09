"""
causal_paper_scout.py
═══════════════════════════════════════════════════════════════════════
Focused paper-trading scout for Brain 7 (Causal-Ensemble) ONLY.

PURPOSE
-------
Run a clean, isolated paper-trade test for the Causal-Ensemble brain on
its 13 confirmed symbols.  NO other brains run.  NO cross-brain signal
pollution.  Signals are stored in the normal DB table but tagged with
model_used='Causal-Ensemble' so they can be filtered precisely.

DESIGN PRINCIPLES
-----------------
1. Only Causal-Ensemble fires — every other brain is skipped entirely.
2. Only confirmed symbols are scanned — everything else is skipped.
3. Paper trade positions are tracked IN MEMORY per session AND persisted
   to the DB via resolver.store_signal() so outcomes are resolved later.
4. Market hours are enforced correctly per asset class (IST for .NS,
   ET for US equities, 24/7 for FX/crypto/commodity).
5. Fully configurable — new brains can plug into PAPER_TRADE_CONFIGS
   (see bottom of file) with zero changes to the engine.
6. Safe to run alongside autonomous_scout.py — different signal tags.

HOW TO RUN
----------
  python -m market_agent.runner.causal_paper_scout

CONFIGURATION (edit PAPER_TRADE_CONFIGS below or override via env):
  PAPER_SCOUT_INTERVAL_MIN  — cycle interval in minutes (default: 10)
  PAPER_SCOUT_BRAIN         — which brain config to use (default: causal_ensemble)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

import structlog
import yfinance as yf
import pandas as pd

logger = structlog.get_logger()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════
# CONFIGURABLE PAPER TRADE PROFILES
# Add a new entry here to enable a focused test for any brain.
# Each profile is self-contained — symbols, timeframe, mode.
# ═══════════════════════════════════════════════════════════

PAPER_TRADE_CONFIGS: Dict[str, Dict] = {

    "causal_ensemble": {
        # Human-readable label for logs/reports
        "label":        "Causal-Ensemble (Brain 7) — Mean Reversion Paper Trade",
        "brain_name":   "Causal-Ensemble",

        # Confirmed symbols from 365-day backtest (passing groups only)
        # equity_US_1h:  WR=65.5% EV=+1.163
        # equity_IN_1h:  WR=50.0% EV=+0.618
        # fx_comm_4h:    WR=50.0% EV=+0.566  (paper only — still thin)
        "symbols": [
            # US Equities — 1h timeframe
            "AAPL", "AMD", "NVDA", "GOOGL", "MSFT", "META", "AMZN", "TSLA",
            # Indian Equities — 1h timeframe
            "ADANIENT.NS", "ADANIPORTS.NS", "LT.NS",
            # FX / Commodity — 4h timeframe (paper, thin n — monitor only)
            "GBPJPY=X", "GC=F",
        ],

        # yfinance data params per symbol group
        # Causal-Ensemble needs 1h candles (needs 50+ bars for BB/RSI/ATR)
        # FX symbols need 4h candles
        "data_config": {
            "default":      {"interval": "1h",  "period": "30d"},
            "FX_4H":        {"interval": "4h",  "period": "60d"},
            "FX_4H_symbols": ["GBPJPY=X", "GC=F"],
        },

        # Brain import path and function name
        "brain_module":   "market_agent.brain.causal_ensemble",
        "brain_fn":       "causal_ensemble_signal",

        # Risk settings for paper trade sizing
        "risk_pct":       0.005,   # 0.5% risk per trade as planned
        "account_size":   1_000_000,  # Virtual ₹10 lakh account

        # Acceptance thresholds for live monitoring
        "min_wr":         0.38,
        "min_ev":         0.0,
        "max_dd_r":       10.0,
    },

    # ── Template for future brains ───────────────────────────────────────
    # "amv_lstm": {
    #     "label":       "AMV-LSTM (Brain 1) — Trend Paper Trade",
    #     "brain_name":  "AMV-LSTM",
    #     "symbols":     ["NIFTY", "BANKNIFTY", ...],
    #     "data_config": {"default": {"interval": "15m", "period": "10d"}},
    #     "brain_module": "market_agent.brain.amv_lstm",
    #     "brain_fn":     "amv_lstm_signal",
    #     "risk_pct":     0.005,
    #     "account_size": 1_000_000,
    #     "min_wr": 0.40, "min_ev": 0.0, "max_dd_r": 8.0,
    # },
}

# Which config to run (override with env PAPER_SCOUT_BRAIN)
ACTIVE_BRAIN_KEY = os.getenv("PAPER_SCOUT_BRAIN", "causal_ensemble")
INTERVAL_MINUTES = int(os.getenv("PAPER_SCOUT_INTERVAL_MIN", "10"))


# ═══════════════════════════════════════════════════════════
# MARKET HOURS — CORRECT PER ASSET CLASS
# ═══════════════════════════════════════════════════════════

def _is_market_open(symbol: str) -> bool:
    """
    Returns True if the market for this symbol is currently tradeable.

    NSE India (.NS):
        Mon-Fri 09:15–15:30 IST

    US Equities (no suffix):
        Mon-Fri 09:30–16:00 ET (UTC-4 in summer, UTC-5 in winter)
        We use UTC-4 (EDT) conservatively — slightly early open is safe.

    FX (=X), Commodities (=F):
        Mon 00:00 – Fri 22:00 UTC (near 24/5)
        We use: weekday < 5 as adequate proxy.

    Crypto (-USD):
        24/7 always open.
    """
    now_utc = datetime.now(timezone.utc)

    # Indian equities
    if ".NS" in symbol or ".BO" in symbol:
        IST = timezone(timedelta(hours=5, minutes=30))
        now_ist = now_utc.astimezone(IST)
        wd = now_ist.weekday()
        h, m = now_ist.hour, now_ist.minute
        if wd >= 5:
            return False
        open_min  = 9 * 60 + 15   # 09:15
        close_min = 15 * 60 + 30  # 15:30
        cur_min   = h * 60 + m
        return open_min <= cur_min <= close_min

    # Crypto — always open
    if "-USD" in symbol or "USDT" in symbol:
        return True

    # FX and Commodities — Mon–Fri only (simplified, safe for our use)
    if "=X" in symbol or "=F" in symbol:
        return now_utc.weekday() < 5

    # US Equities — need hour check (was MISSING in watchlist_scanner.py)
    # EDT = UTC-4 (summer), EST = UTC-5 (winter).
    # Use UTC-4 throughout for simplicity — 9:30 EDT = 13:30 UTC.
    # This means in winter we open 1h early on UTC basis — acceptable for paper trade.
    ET = timezone(timedelta(hours=-4))
    now_et = now_utc.astimezone(ET)
    wd = now_et.weekday()
    h, m = now_et.hour, now_et.minute
    if wd >= 5:
        return False
    open_min  = 9 * 60 + 30   # 09:30 ET
    close_min = 16 * 60 + 0   # 16:00 ET
    cur_min   = h * 60 + m
    return open_min <= cur_min <= close_min


# ═══════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════

# Per-symbol data cache: avoid refetching within same 10-min cycle
_data_cache: Dict[str, Dict] = {}  # symbol -> {"df": pd, "fetched_at": datetime}
_CACHE_TTL_MINUTES = 9  # refresh if older than 9 min (just under the 10-min cycle)


def _fetch_ohlcv(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data. Uses in-cycle cache to avoid duplicate API calls.

    DATA SOURCES USED:
        Primary  : yfinance (Yahoo Finance) — free, reliable for all symbols
                   NSE (.NS): Yahoo uses NSE data feed, 15-min delayed but sufficient
                   US equities: real-time (or ~15 min delayed on free tier)
                   FX (=X): real-time (no delay on Yahoo FX)
                   Commodities (=F): real-time futures data
        Fallback : Returns None — brain will emit HOLD, signal skipped

    WHY ONLY YFINANCE HERE:
        This scout runs generate_brain_signals() which computes indicators
        over 30-60 bars of 1h/4h data. For indicator computation, 15-min
        delay is irrelevant — we care about bar shapes, not tick precision.
        For EXECUTION prices (entry/SL/T1) we use the CLOSE of the last
        complete bar, which yfinance provides correctly.
        Live tick price would only matter for live trading — not paper.
    """
    cache = _data_cache.get(symbol)
    if cache:
        age = (datetime.now() - cache["fetched_at"]).total_seconds() / 60
        if age < _CACHE_TTL_MINUTES:
            return cache["df"]

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is None or df.empty:
            logger.warning("data_fetch_empty", symbol=symbol, interval=interval)
            return None

        # Normalize: drop timezone from index (brain code expects naive timestamps)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Ensure standard column names
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)

        if len(df) < 30:
            logger.warning("data_fetch_insufficient", symbol=symbol,
                           rows=len(df), need=30)
            return None

        _data_cache[symbol] = {"df": df, "fetched_at": datetime.now()}
        return df

    except Exception as e:
        logger.error("data_fetch_failed", symbol=symbol, error=str(e)[:80])
        return None


# ═══════════════════════════════════════════════════════════
# IN-MEMORY PAPER TRADE TRACKER
# Tracks positions and P&L for THIS session.
# Signals are also written to DB via resolver for cross-session resolution.
# ═══════════════════════════════════════════════════════════

class SessionPaperTracker:
    """
    Tracks paper trade results for one scout session.
    In-memory — resets on restart.
    For multi-session tracking, use the DB signal_predictions table
    filtered by model_used='Causal-Ensemble'.
    """

    def __init__(self, risk_pct: float, account_size: float):
        self.risk_pct     = risk_pct
        self.account_size = account_size
        self.trades: List[Dict] = []
        self.wins   = 0
        self.losses = 0
        self.total_r = 0.0
        self.max_r   = 0.0
        self.max_dd  = 0.0
        self._peak_r = 0.0

    def record(self, symbol: str, direction: str, entry: float,
               target_1: float, stop_loss: float, brain_signal: Any):
        """Record a new paper trade signal."""
        risk_amount = self.account_size * self.risk_pct
        sl_dist = abs(entry - stop_loss)
        t1_dist = abs(target_1 - entry)
        rr = t1_dist / sl_dist if sl_dist > 0 else 0.0

        trade = {
            "symbol":    symbol,
            "direction": direction,
            "entry":     entry,
            "target_1":  target_1,
            "stop_loss": stop_loss,
            "rr":        rr,
            "risk_₹":   risk_amount,
            "time":      datetime.now().strftime("%H:%M:%S"),
            "regime":    brain_signal.measurements.get("regime_used", "?"),
            "confidence": brain_signal.confidence,
        }
        self.trades.append(trade)
        logger.info("paper_trade_signal",
                    symbol=symbol, direction=direction,
                    entry=round(entry, 4), t1=round(target_1, 4),
                    sl=round(stop_loss, 4), rr=round(rr, 2),
                    regime=trade["regime"])

    def get_summary(self) -> str:
        n = len(self.trades)
        if n == 0:
            return "No signals this session yet."
        lines = [
            f"Session signals: {n}",
            f"Recent signals:",
        ]
        for t in self.trades[-5:]:
            lines.append(
                f"  {t['time']} {t['symbol']:15s} {t['direction']:4s}  "
                f"entry={t['entry']:.4f}  RR={t['rr']:.2f}  "
                f"regime={t['regime']}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# CORE SCAN FUNCTION
# ═══════════════════════════════════════════════════════════

def run_focused_scan(config: Dict, tracker: SessionPaperTracker,
                     storage, resolver) -> int:
    """
    Run one scan cycle for the configured brain.
    Returns number of signals generated this cycle.
    """
    from dataclasses import replace

    # Dynamically import the brain function
    try:
        import importlib
        brain_mod = importlib.import_module(config["brain_module"])
        brain_fn  = getattr(brain_mod, config["brain_fn"])
    except Exception as e:
        logger.error("brain_import_failed", module=config["brain_module"], error=str(e))
        return 0

    # Also import regime_ensemble so the brain gets a proper regime
    try:
        from market_agent.brain.regime_ensemble import regime_ensemble_signal
        UNIFIED_REGIMES = {
            "TRENDING_UP", "TRENDING_DOWN", "RANGING",
            "VOLATILE", "SQUEEZE", "CHAOS",
        }
    except Exception as e:
        regime_ensemble_signal = None
        UNIFIED_REGIMES = set()
        logger.warning("regime_ensemble_import_failed", error=str(e)[:60])

    data_cfg = config["data_config"]
    signals_this_cycle = 0

    for symbol in config["symbols"]:

        # ── 1. Market hours guard ────────────────────────────────────────────
        if not _is_market_open(symbol):
            logger.debug("market_closed_skip", symbol=symbol)
            continue

        # ── 2. Fetch data ────────────────────────────────────────────────────
        if symbol in data_cfg.get("FX_4H_symbols", []):
            dcfg = data_cfg["FX_4H"]
        else:
            dcfg = data_cfg["default"]

        hist = _fetch_ohlcv(symbol, dcfg["interval"], dcfg["period"])
        if hist is None:
            continue

        # ── 3. Compute regime ────────────────────────────────────────────────
        regime = "RANGING"  # safe default
        if regime_ensemble_signal:
            try:
                rs = regime_ensemble_signal(hist)
                computed = rs.measurements.get("computed_regime", "")
                if computed in UNIFIED_REGIMES:
                    regime = computed
            except Exception:
                pass  # keep default regime

        # ── 4. Compute ATR for signal dict building ──────────────────────────
        try:
            atr = hist["High"].sub(hist["Low"]).rolling(14).mean().iloc[-1]
            if pd.isna(atr) or atr <= 0:
                atr = float(hist["Close"].iloc[-1]) * 0.01
        except Exception:
            atr = float(hist["Close"].iloc[-1]) * 0.01

        current_price = float(hist["Close"].iloc[-1])
        if current_price <= 0:
            continue

        # ── 5. Fire ONLY the configured brain ───────────────────────────────
        try:
            bs = brain_fn(hist, regime=regime)
        except TypeError:
            # Some brains don't accept regime kwarg — call without it
            try:
                bs = brain_fn(hist)
            except Exception as e:
                logger.warning("brain_call_failed", symbol=symbol, error=str(e)[:80])
                continue
        except Exception as e:
            logger.warning("brain_call_failed", symbol=symbol, error=str(e)[:80])
            continue

        # ── 6. Check if signal is actionable ────────────────────────────────
        if bs.direction in ("HOLD", "WAIT", None):
            continue

        eff_conf = bs.effective_confidence() if hasattr(bs, "effective_confidence") else bs.confidence
        if eff_conf < 0.50:
            logger.debug("signal_low_confidence", symbol=symbol,
                         brain=config["brain_name"], conf=round(eff_conf, 3))
            continue

        # ── 7. Build signal dict ─────────────────────────────────────────────
        from market_agent.signal_params import (
            BASE_ATR_T1_MULT, BASE_ATR_T2_MULT, BASE_ATR_SL_MULT,
        )
        t1_mult = getattr(bs, "rr_t1_mult", None) or BASE_ATR_T1_MULT
        t2_mult = getattr(bs, "rr_t2_mult", None) or BASE_ATR_T2_MULT
        sl_mult = getattr(bs, "rr_sl_mult", None) or BASE_ATR_SL_MULT

        if bs.direction == "BUY":
            target_1  = current_price + atr * t1_mult
            target_2  = current_price + atr * t2_mult
            stop_loss = current_price - atr * sl_mult
        else:
            target_1  = current_price - atr * t1_mult
            target_2  = current_price - atr * t2_mult
            stop_loss = current_price + atr * sl_mult

        sig_dict = {
            "symbol":       symbol,
            "direction":    bs.direction,
            "entry_price":  current_price,
            "current_price": current_price,
            "target_1":     target_1,
            "target_2":     target_2,
            "stop_loss":    stop_loss,
            "confidence":   eff_conf,
            "regime":       regime,
            "model_used":   config["brain_name"],
            "model_name":   config["brain_name"],
            "vol_z_score":  1.0,
            "timeframe_min": 60 if dcfg["interval"] == "1h" else 240,
        }

        # ── 8. Store to DB (for cross-session resolution) ────────────────────
        try:
            pred_id = resolver.store_signal(sig_dict, strategy="Paper-Trade")
            if pred_id:
                signals_this_cycle += 1
                tracker.record(symbol, bs.direction, current_price,
                                target_1, stop_loss, bs)
                print(f"  ✅ {symbol:15s} {bs.direction:4s} @ {current_price:.4f}  "
                      f"T1={target_1:.4f}  SL={stop_loss:.4f}  "
                      f"regime={regime}  conf={eff_conf:.2f}")
        except Exception as e:
            logger.error("signal_store_failed", symbol=symbol, error=str(e)[:80])

    return signals_this_cycle


# ═══════════════════════════════════════════════════════════
# SIGNAL RESOLVER — check open signals against fresh prices
# ═══════════════════════════════════════════════════════════

def resolve_open_signals(config: Dict, resolver, symbols: List[str]):
    """
    Check previously stored (unresolved) signals for this brain
    against current prices. Updates outcome in DB.
    """
    for symbol in symbols:
        if not _is_market_open(symbol):
            continue
        try:
            # Fresh price: use latest close from yfinance (fast)
            ticker = yf.Ticker(symbol)
            fast = getattr(ticker, "fast_info", None)
            price = None
            if fast:
                price = getattr(fast, "last_price", None)
            if not price or price <= 0:
                hist = ticker.history(period="1d", interval="1m")
                if hist is not None and not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price and price > 0:
                resolver.resolve_signals(
                    current_price=price,
                    symbol=symbol,
                    macro_data={},
                    recent_news=[],
                )
        except Exception as e:
            logger.debug("resolve_failed", symbol=symbol, error=str(e)[:60])


# ═══════════════════════════════════════════════════════════
# PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════

def print_db_performance(config: Dict, storage, resolver):
    """
    Pull all resolved signals for this brain from DB and print performance.
    This shows the CUMULATIVE paper trade result across all sessions.
    """
    try:
        from sqlalchemy import text
        brain_name = config["brain_name"]
        with storage.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    symbol,
                    direction,
                    entry_price,
                    target_1,
                    stop_loss,
                    outcome,
                    created_at
                FROM signal_predictions
                WHERE model_used = :brain
                  AND outcome IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 200
            """), {"brain": brain_name}).fetchall()

        if not rows:
            print(f"  No resolved signals yet for {brain_name}")
            return

        wins   = sum(1 for r in rows if r[5] in ("T1_HIT", "T2_HIT"))
        losses = sum(1 for r in rows if r[5] == "SL_HIT")
        total  = wins + losses
        wr     = wins / total if total > 0 else 0.0

        # Approximate R calculation
        total_r = 0.0
        for r in rows:
            outcome = r[5]
            entry   = float(r[2]) if r[2] else 0
            t1      = float(r[3]) if r[3] else 0
            sl      = float(r[4]) if r[4] else 0
            if entry <= 0 or sl <= 0:
                continue
            sl_dist = abs(entry - sl)
            if outcome in ("T1_HIT", "T2_HIT"):
                t1_dist = abs(t1 - entry)
                total_r += t1_dist / sl_dist if sl_dist > 0 else 0
            elif outcome == "SL_HIT":
                total_r -= 1.0

        ev = total_r / total if total > 0 else 0.0

        print(f"\n  ═══ PAPER TRADE PERFORMANCE ({brain_name}) ═══")
        print(f"  Resolved signals : {total}")
        print(f"  Win Rate         : {wr:.1%}  (target ≥ 38%)")
        print(f"  EV (avg R)       : {ev:+.3f}  (target > 0)")
        print(f"  Total R          : {total_r:+.2f}R")
        print(f"  {'✅ PERFORMING' if wr >= 0.38 and ev > 0 else '⚠️  WATCHING'}")
        if total >= 10:
            print(f"  → Minimum 10 signals reached — check before going live")
        else:
            print(f"  → Need {10 - total} more resolved signals before live decision")

    except Exception as e:
        logger.error("performance_report_failed", error=str(e)[:80])


# ═══════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════

def start_causal_paper_scout(
    brain_key: str = ACTIVE_BRAIN_KEY,
    interval_minutes: int = INTERVAL_MINUTES,
):
    """
    Main loop for focused paper trading.

    brain_key         : key in PAPER_TRADE_CONFIGS
    interval_minutes  : how often to scan (default 10)
    """
    config = PAPER_TRADE_CONFIGS.get(brain_key)
    if not config:
        raise ValueError(
            f"Unknown brain key '{brain_key}'. "
            f"Available: {list(PAPER_TRADE_CONFIGS.keys())}"
        )

    from market_agent.data.storage.postgres import PostgresStorage
    from market_agent.learning.evaluator import RegretEngine
    from market_agent.learning.signal_resolver import SignalResolver

    storage  = PostgresStorage()
    regret   = RegretEngine(storage)
    resolver = regret.signal_resolver or SignalResolver(storage)
    tracker  = SessionPaperTracker(config["risk_pct"], config["account_size"])

    print("=" * 65)
    print(f"  CAUSAL PAPER SCOUT")
    print(f"  Brain    : {config['label']}")
    print(f"  Symbols  : {len(config['symbols'])}")
    print(f"  Interval : {interval_minutes} min")
    print(f"  Risk     : {config['risk_pct']*100:.1f}% per trade")
    print(f"  Account  : ₹{config['account_size']:,.0f} (virtual)")
    print(f"  Target   : WR≥{config['min_wr']:.0%}  EV>0  MaxDD≤{config['max_dd_r']}R")
    print("=" * 65)
    print(f"\n  Confirmed symbols ({len(config['symbols'])}):")
    for s in config["symbols"]:
        print(f"    {s}")
    print()

    cycle = 0
    while True:
        cycle += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'─'*65}")
        print(f"  CYCLE {cycle}  |  {now_str}")
        print(f"{'─'*65}")

        try:
            # Step 1: Resolve any open signals first (fresh prices)
            print("  [RESOLVE] Checking open signals...")
            resolve_open_signals(config, resolver, config["symbols"])

            # Step 2: Scan and generate new signals
            print("  [SCAN] Scanning confirmed symbols...")
            n_signals = run_focused_scan(config, tracker, storage, resolver)
            if n_signals == 0:
                print("  No new signals this cycle.")
            else:
                print(f"  {n_signals} new signal(s) generated.")

            # Step 3: Print session summary
            print()
            print(tracker.get_summary())

            # Step 4: Print cumulative DB performance every 3 cycles
            if cycle % 3 == 0:
                print_db_performance(config, storage, resolver)

        except KeyboardInterrupt:
            print("\n  Paper scout stopped by user.")
            print_db_performance(config, storage, resolver)
            break
        except Exception as e:
            logger.error("cycle_error", cycle=cycle, error=str(e))

        print(f"\n  Sleeping {interval_minutes} min until next cycle...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    start_causal_paper_scout()