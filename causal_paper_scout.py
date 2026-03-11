"""
causal_paper_scout.py
═══════════════════════════════════════════════════════════════════
Paper-trading scout for Brain 7 (Causal-Ensemble) ONLY.

FULLY SELF-CONTAINED:
  - No dependency on market_agent.learning, SignalResolver, RegretEngine
  - Signal storage and resolution done via PostgresStorage directly
    using the paper_trade_signals table (added to postgres.py by our patch)
  - Data source: yfinance only (free, no API key, covers all 13 symbols)

WHAT HAPPENS EACH CYCLE (every 55 minutes by default):
  1. RESOLVE  — check every open signal vs current price → T1/SL/EXPIRED
  2. SCAN     — fetch fresh 1h / 4h data, compute regime, run brain
  3. GATE     — skip HOLD, skip confidence < 0.50, skip CHAOS regime
  4. STORE    — write new signals to paper_trade_signals table
  5. REPORT   — print full cumulative performance every 3 cycles

HOW TO ADD ANOTHER BRAIN FOR PAPER TESTING:
  Add one entry to PAPER_TRADE_CONFIGS below. Zero changes to the
  core loop needed — it is fully data-driven.

HOW TO RUN LOCALLY:
  python causal_paper_scout.py

HOW TO DEPLOY ON RENDER:
  Deployed as a FREE Web Service via render.yaml.
  A health server runs on PORT (set by Render) so UptimeRobot can
  ping /health every 5 min to prevent the free tier from spinning down.
  Set DATABASE_URL in Render Environment tab.
"""

from __future__ import annotations

import os
import sys
import time
import importlib
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import structlog
import pandas as pd
import yfinance as yf

# Load .env file when running locally (no-op if dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════
# PAPER TRADE BRAIN CONFIGS
#
# Each entry is a complete brain configuration.
# The main loop uses the config selected by PAPER_SCOUT_BRAIN env var.
# Add new brains here without touching the engine.
# ═══════════════════════════════════════════════════════════

PAPER_TRADE_CONFIGS: Dict[str, Dict] = {

    "causal_ensemble": {
        # Label printed in banner
        "label": "Causal-Ensemble (Brain 7) — Mean Reversion",

        # Name stored in DB rows and used in performance queries
        "brain_name": "Causal-Ensemble",

        # Confirmed symbols — passed 365-day backtest
        # equity_US_1h : WR=65.5%  EV=+1.163R  ✅ deploy
        # equity_IN_1h : WR=50.0%  EV=+0.618R  ✅ deploy
        # fx_comm_4h   : WR=50.0%  EV=+0.566R  ⚠ paper-only (thin n)
        # Excluded: BTC-USD (EV negative), CL=F (excluded from 365d test)
        "symbols": [
            # US equities — 1h timeframe
            "AAPL", "AMD", "NVDA", "GOOGL", "MSFT", "META", "AMZN", "TSLA",
            # Indian equities — 1h timeframe
            "ADANIENT.NS", "ADANIPORTS.NS", "LT.NS",
            # FX / Commodity — 4h timeframe
            "GBPJPY=X", "GC=F",
        ],

        # Data fetch settings per symbol group
        "data_config": {
            # Default: 1h bars, 30 days back (gives ~500 bars — enough for all indicators)
            "default":        {"interval": "1h", "period": "30d"},
            # FX and Commodity symbols that need 4h bars instead
            "FX_4H_symbols":  ["GBPJPY=X", "GC=F"],
            "FX_4H":          {"interval": "4h", "period": "60d"},
        },

        # Brain to run — must match module path in the repo
        "brain_module": "market_agent.brain.causal_ensemble",
        "brain_fn":     "causal_ensemble_signal",

        # Risk settings (all virtual for paper trading)
        "risk_pct":     0.005,       # 0.5% of account per trade
        "account_size": 1_000_000,   # ₹10,00,000 virtual

        # Go-live decision thresholds (from 365d backtest baseline)
        "min_wr":      0.38,         # minimum win rate to go live
        "min_ev":      0.0,          # minimum EV in R to go live
        "max_dd_r":    10.0,         # maximum drawdown in R before stopping

        # Minimum resolved signals before making a live/no-live decision
        "min_resolved": 10,
    },

    # ── Template for next brain ─────────────────────────────
    # "amv_lstm": {
    #     "label":        "AMV-LSTM (Brain 1) — Trend Following",
    #     "brain_name":   "AMV-LSTM",
    #     "symbols":      ["RELIANCE.NS", "INFY.NS", "TCS.NS"],
    #     "data_config":  {"default": {"interval": "15m", "period": "10d"}},
    #     "brain_module": "market_agent.brain.amv_lstm",
    #     "brain_fn":     "amv_lstm_signal",
    #     "risk_pct":     0.005, "account_size": 1_000_000,
    #     "min_wr": 0.40, "min_ev": 0.0, "max_dd_r": 8.0, "min_resolved": 10,
    # },
}

# ── Active config — set via env var or falls back to causal_ensemble ──────────
ACTIVE_BRAIN_KEY   = os.getenv("PAPER_SCOUT_BRAIN",        "causal_ensemble")
INTERVAL_MINUTES   = int(os.getenv("PAPER_SCOUT_INTERVAL_MIN", "55"))

# Regimes where the Causal-Ensemble brain should NOT trade
NO_TRADE_REGIMES   = {"CHAOS"}


# ═══════════════════════════════════════════════════════════
# MARKET HOURS — CORRECT PER ASSET CLASS
#
# CRITICAL RULE: Only scan when the relevant exchange is open.
# Previous bug in the main system: US equities had no hour check —
# brains were firing at 2 AM ET on stale bars. Fixed here correctly.
# ═══════════════════════════════════════════════════════════

def _is_market_open(symbol: str) -> bool:
    """
    Returns True only when the relevant exchange is open for this symbol.

    Indian equities (.NS / .BO) : Mon–Fri  09:15–15:30 IST  (UTC+5:30)
    US equities (no suffix)     : Mon–Fri  09:30–16:00 ET   (UTC-4 used year-round)
    FX (=X) / Commodities (=F)  : Mon–Fri only (near 24/5 but closed weekends)
    Crypto (-USD / USDT)        : 24/7 — always True
    """
    now_utc = datetime.now(timezone.utc)

    # ── Indian equities ───────────────────────────────────────────
    if ".NS" in symbol or ".BO" in symbol:
        IST     = timezone(timedelta(hours=5, minutes=30))
        now_ist = now_utc.astimezone(IST)
        if now_ist.weekday() >= 5:                  # Saturday or Sunday
            return False
        t = now_ist.hour * 60 + now_ist.minute
        return (9 * 60 + 15) <= t <= (15 * 60 + 30)

    # ── Crypto — always open ──────────────────────────────────────
    if symbol.endswith("-USD") or "USDT" in symbol:
        return True

    # ── FX and Commodities — weekdays only ───────────────────────
    if "=X" in symbol or "=F" in symbol:
        return now_utc.weekday() < 5               # Mon=0 … Fri=4

    # ── US Equities — 09:30–16:00 ET, weekdays only ──────────────
    # Using UTC-4 (EDT) throughout the year. In winter (EST, UTC-5) this
    # means the window shifts 1h early in UTC terms — acceptable for paper.
    ET     = timezone(timedelta(hours=-4))
    now_et = now_utc.astimezone(ET)
    if now_et.weekday() >= 5:
        return False
    t = now_et.hour * 60 + now_et.minute
    return (9 * 60 + 30) <= t <= (16 * 60 + 0)


# ═══════════════════════════════════════════════════════════
# DATA FETCHING — yfinance with in-cycle caching
# ═══════════════════════════════════════════════════════════

_cache: Dict[str, Dict] = {}
_CACHE_TTL_MINUTES = 50   # re-use fetched data within same 55-min cycle


def _fetch_ohlcv(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars from yfinance with a 50-minute in-process cache.

    WHY YFINANCE ONLY (not Breeze/AngelOne for paper trading):
      - Paper trade entry/exit uses bar CLOSE — 15-min delay on yfinance
        is irrelevant for 1h/4h bars where indicator math is what matters.
      - All 13 confirmed symbols are covered by yfinance:
          NSE (.NS)  : Yahoo uses NSE feed, ~15-min delay
          US equities: ~15-min delay on free tier
          FX (=X)    : real-time from Yahoo
          GC=F       : real-time futures from Yahoo
      - No API key. No session token. No rate limits at 55-min intervals.

    For live trading (not paper), we would switch to Breeze for .NS and
    direct exchange feeds for US — but that is a future concern.
    """
    cache_key = f"{symbol}_{interval}"
    hit = _cache.get(cache_key)
    if hit:
        age_min = (datetime.now() - hit["fetched_at"]).total_seconds() / 60
        if age_min < _CACHE_TTL_MINUTES:
            return hit["df"]

    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)

        if df is None or df.empty:
            logger.warning("yf_data_empty", symbol=symbol, interval=interval)
            return None

        # Drop timezone info — brain code expects tz-naive DatetimeIndex
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)

        if len(df) < 30:
            logger.warning("yf_data_insufficient", symbol=symbol, rows=len(df))
            return None

        _cache[cache_key] = {"df": df, "fetched_at": datetime.now()}
        return df

    except Exception as e:
        logger.error("yf_fetch_failed", symbol=symbol, error=str(e)[:120])
        return None


def _get_current_price(symbol: str, fallback_df: pd.DataFrame) -> float:
    """
    Attempt near-real-time price via yfinance fast_info.
    Falls back to the last bar's Close if fast_info is unavailable.
    fast_info does not require an API key and is near-real-time.
    """
    try:
        price = yf.Ticker(symbol).fast_info.last_price
        if price and float(price) > 0:
            return float(price)
    except Exception:
        pass
    return float(fallback_df["Close"].iloc[-1])


# ═══════════════════════════════════════════════════════════
# IN-SESSION PERFORMANCE TRACKER (in-memory)
# Tracks signals fired this process run. DB has full history.
# ═══════════════════════════════════════════════════════════

class SessionTracker:
    def __init__(self):
        self.signals: List[Dict] = []

    def record(self, symbol, direction, entry, t1, sl, regime, conf):
        sl_dist = abs(entry - sl)
        rr      = round(abs(t1 - entry) / sl_dist, 2) if sl_dist > 0 else 0.0
        self.signals.append({
            "time": datetime.now().strftime("%H:%M"), "symbol": symbol,
            "direction": direction, "entry": entry, "t1": t1,
            "sl": sl, "rr": rr, "regime": regime, "conf": round(conf, 2),
        })

    def print_summary(self):
        n = len(self.signals)
        if n == 0:
            print("  Session signals: 0  (no signals fired yet)")
            return
        print(f"  Session signals: {n}  (last 5 shown)")
        for s in self.signals[-5:]:
            print(f"    {s['time']}  {s['symbol']:<15}  {s['direction']:<4}  "
                  f"entry={s['entry']:.4f}  T1={s['t1']:.4f}  SL={s['sl']:.4f}  "
                  f"RR={s['rr']:.1f}  regime={s['regime']}  conf={s['conf']}")


# ═══════════════════════════════════════════════════════════
# PERFORMANCE REPORT — reads from DB, printed every 3 cycles
# ═══════════════════════════════════════════════════════════

def _print_performance(storage, config: Dict):
    brain_name  = config["brain_name"]
    min_wr      = config["min_wr"]
    min_ev      = config["min_ev"]
    max_dd_r    = config["max_dd_r"]
    min_resolved = config["min_resolved"]

    p = storage.get_paper_trade_performance(brain_name)
    if not p:
        print("  [Performance] No resolved signals yet.")
        return

    total    = p["total"]
    decided  = p["decided"]
    wins     = p["wins"]
    losses   = p["losses"]
    wr       = p["wr"]
    total_r  = p["total_r"]
    ev       = p["ev"]
    max_dd   = p["max_dd"]
    open_n   = p["open"]

    wr_ok = wr     >= min_wr
    ev_ok = ev     >= min_ev
    dd_ok = max_dd <= max_dd_r

    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print(f"  ║  PAPER TRADE PERFORMANCE — {brain_name:<25}  ║")
    print("  ╠═══════════════════════════════════════════════════════╣")
    print(f"  ║  Total signals  : {total:>4}  ({open_n} still open)               ║")
    print(f"  ║  Decided (W+L)  : {decided:>4}  ({wins}W / {losses}L)                     ║")
    print(f"  ║  Win Rate       : {wr:>6.1%}   target ≥{min_wr:.0%}     {'✅' if wr_ok else '❌'}          ║")
    print(f"  ║  Avg EV (R)     : {ev:>+7.3f}  target >{min_ev:.1f}      {'✅' if ev_ok else '❌'}          ║")
    print(f"  ║  Total R        : {total_r:>+7.2f}R                              ║")
    print(f"  ║  Max Drawdown   : {max_dd:>6.2f}R   limit ≤{max_dd_r:.0f}R      {'✅' if dd_ok else '❌'}          ║")
    print("  ╠═══════════════════════════════════════════════════════╣")

    if decided < min_resolved:
        remaining = min_resolved - decided
        print(f"  ║  ⏳  Need {remaining:>2} more resolved signals before deciding.    ║")
        print(f"  ║     Continue paper trading.                           ║")
    elif wr_ok and ev_ok and dd_ok:
        print(f"  ║  🟢  READY FOR LIVE TRADING — all thresholds passed   ║")
        print(f"  ║     Go live at 0.5% risk/trade. Monitor daily.        ║")
    else:
        failing = []
        if not wr_ok: failing.append(f"WR={wr:.1%}")
        if not ev_ok: failing.append(f"EV={ev:+.3f}")
        if not dd_ok: failing.append(f"DD={max_dd:.2f}R")
        print(f"  ║  🔴  BELOW THRESHOLD: {', '.join(failing):<34}  ║")
        print(f"  ║     Continue paper trading — do NOT go live yet.      ║")

    print("  ╚═══════════════════════════════════════════════════════╝")

    recent = p.get("recent", [])
    if recent:
        print(f"\n  Last {len(recent)} resolved signals:")
        for r in recent:
            icon = "✅" if r["outcome"] == "T1_HIT" else ("❌" if r["outcome"] == "SL_HIT" else "⏸ ")
            pnl  = f"{r['pnl_r']:+.2f}R" if r["pnl_r"] is not None else "  N/A"
            ts   = r["created_at"].strftime("%m-%d %H:%M") if r["created_at"] else "?"
            print(f"    {icon}  {r['symbol']:<15}  {r['direction']:<4}  "
                  f"{r['outcome']:<8}  {pnl}  regime={r['regime'] or '?':<12}  {ts}")
    print()


# ═══════════════════════════════════════════════════════════
# CORE SCAN — one complete cycle
# ═══════════════════════════════════════════════════════════

UNIFIED_REGIMES = {
    "TRENDING_UP", "TRENDING_DOWN", "RANGING",
    "VOLATILE", "SQUEEZE", "CHAOS",
}


def _run_scan_cycle(config: Dict, storage, tracker: SessionTracker,
                    brain_fn, regime_fn) -> int:
    """
    Run one complete scan over all symbols in the config.
    Returns number of new signals stored this cycle.
    """
    data_cfg    = config["data_config"]
    brain_name  = config["brain_name"]
    new_signals = 0

    for symbol in config["symbols"]:

        # ── 1. Market hours gate ──────────────────────────────────
        if not _is_market_open(symbol):
            logger.debug("market_closed_skip", symbol=symbol)
            continue

        # ── 2. Determine data fetch settings ─────────────────────
        if symbol in data_cfg.get("FX_4H_symbols", []):
            dcfg = data_cfg["FX_4H"]
        else:
            dcfg = data_cfg["default"]

        # ── 3. Fetch OHLCV ────────────────────────────────────────
        hist = _fetch_ohlcv(symbol, dcfg["interval"], dcfg["period"])
        if hist is None or hist.empty:
            continue

        # ── 4. Current price ──────────────────────────────────────
        current_price = _get_current_price(symbol, hist)
        if current_price <= 0:
            logger.warning("bad_price", symbol=symbol, price=current_price)
            continue

        # ── 5. Resolve open signals BEFORE scanning ───────────────
        # This ensures we always resolve with the freshest price
        # before deciding whether to fire another signal.
        storage.resolve_paper_signals(symbol, current_price)

        # ── 6. Compute regime (optional but improves signal quality)
        regime = "RANGING"   # safe default
        if regime_fn is not None:
            try:
                rs = regime_fn(hist)
                computed = rs.measurements.get("computed_regime", "")
                if computed in UNIFIED_REGIMES:
                    regime = computed
            except Exception as re:
                logger.debug("regime_compute_failed", symbol=symbol, error=str(re)[:60])

        # ── 7. Skip no-trade regimes ──────────────────────────────
        if regime in NO_TRADE_REGIMES:
            logger.debug("regime_blocked", symbol=symbol, regime=regime)
            continue

        # ── 8. Compute ATR for target/SL placement ────────────────
        try:
            atr = hist["High"].sub(hist["Low"]).rolling(14).mean().iloc[-1]
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.01   # 1% fallback
        except Exception:
            atr = current_price * 0.01

        # ── 9. Run brain ──────────────────────────────────────────
        try:
            brain_signal = brain_fn(hist, regime=regime)
        except TypeError:
            # Some brain versions do not accept regime kwarg yet
            try:
                brain_signal = brain_fn(hist)
            except Exception as be:
                logger.warning("brain_call_failed", symbol=symbol, error=str(be)[:80])
                continue
        except Exception as be:
            logger.warning("brain_call_failed", symbol=symbol, error=str(be)[:80])
            continue

        # ── 10. Direction gate ────────────────────────────────────
        if brain_signal.direction in ("HOLD", "WAIT", None, ""):
            continue

        # ── 11. Confidence gate ───────────────────────────────────
        eff_conf = (
            brain_signal.effective_confidence()
            if hasattr(brain_signal, "effective_confidence")
            else brain_signal.confidence
        )
        if eff_conf < 0.50:
            logger.debug("low_conf_skip", symbol=symbol, conf=round(eff_conf, 3))
            continue

        # ── 12. Build targets using brain's RR multipliers ────────
        # Brain can override defaults via rr_t1_mult / rr_sl_mult.
        # Causal-Ensemble sets: STRONG=3:1, MEDIUM=2.5:1, WEAK=2:1
        t1_mult = getattr(brain_signal, "rr_t1_mult", None) or 2.5
        t2_mult = getattr(brain_signal, "rr_t2_mult", None) or 4.0
        sl_mult = getattr(brain_signal, "rr_sl_mult", None) or 1.0

        if brain_signal.direction == "BUY":
            target_1  = current_price + atr * t1_mult
            target_2  = current_price + atr * t2_mult
            stop_loss = current_price - atr * sl_mult
        else:  # SELL
            target_1  = current_price - atr * t1_mult
            target_2  = current_price - atr * t2_mult
            stop_loss = current_price + atr * sl_mult

        # ── 13. Sanity check ──────────────────────────────────────
        sl_dist = abs(current_price - stop_loss)
        if sl_dist <= 0 or sl_dist / current_price > 0.15:
            # SL more than 15% away — something wrong with ATR calc
            logger.warning("bad_sl_distance", symbol=symbol,
                           sl_dist_pct=round(sl_dist / current_price * 100, 2))
            continue

        # ── 14. Store signal ──────────────────────────────────────
        sig_id = storage.store_paper_signal(
            brain_name  = brain_name,
            symbol      = symbol,
            direction   = brain_signal.direction,
            entry_price = current_price,
            target_1    = round(target_1, 6),
            target_2    = round(target_2, 6),
            stop_loss   = round(stop_loss, 6),
            confidence  = round(eff_conf, 4),
            regime      = regime,
            timeframe   = dcfg["interval"],
            strategy    = "Paper-Trade",
        )

        if sig_id:
            new_signals += 1
            tracker.record(symbol, brain_signal.direction, current_price,
                           target_1, stop_loss, regime, eff_conf)
            print(f"  ✅ [{sig_id:>5}] {symbol:<15}  {brain_signal.direction:<4}  "
                  f"@ {current_price:<12.4f}  "
                  f"T1={target_1:.4f}  SL={stop_loss:.4f}  "
                  f"RR={t1_mult:.1f}  regime={regime:<14}  conf={eff_conf:.2f}")

    return new_signals


# ═══════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# HEALTH SERVER — keeps Render Free Web Service alive
#
# Render free Web Services spin down after 15 min of inactivity.
# This tiny HTTP server answers GET / and GET /health so that
# UptimeRobot can ping it every 5 minutes and prevent spin-down.
#
# Uses Python's built-in http.server — ZERO new dependencies.
# Runs in a daemon thread so it never blocks the scanning loop.
#
# Port: read from PORT env var (Render sets this automatically).
#       Falls back to 8080 for local testing.
# ═══════════════════════════════════════════════════════════

def _start_health_server():
    """
    Start a minimal HTTP health server in a background daemon thread.
    Responds 200 OK to any GET request with a plain-text status line.
    Called once at startup — before the main scanning loop begins.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"causal-paper-scout: alive\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass   # silence per-request access logs — keep Render logs clean

    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"  ✅ Health server running on port {port}  (UptimeRobot target: /health)")


def start_scout(brain_key: str = ACTIVE_BRAIN_KEY,
                interval_minutes: int = INTERVAL_MINUTES):
    """
    Main entry point. Runs forever until stopped.
    Render Background Worker keeps this running 24/7 without spinning down.
    """
    # ── Start health server (keeps Render free tier alive) ───────
    _start_health_server()

    # ── Validate brain key ────────────────────────────────────────
    config = PAPER_TRADE_CONFIGS.get(brain_key)
    if not config:
        print(f"❌ ERROR: Unknown brain key '{brain_key}'")
        print(f"   Available keys: {list(PAPER_TRADE_CONFIGS.keys())}")
        print(f"   Set PAPER_SCOUT_BRAIN env var to one of the above.")
        sys.exit(1)

    # ── Connect to database ───────────────────────────────────────
    try:
        from market_agent.data.storage.postgres import PostgresStorage
        storage = PostgresStorage()
        print("  ✅ Database connected. All tables created/verified.")
    except Exception as e:
        print(f"  ❌ FATAL — Cannot connect to database: {e}")
        print("     Ensure DATABASE_URL environment variable is set correctly.")
        sys.exit(1)

    # ── Load brain ────────────────────────────────────────────────
    try:
        brain_mod = importlib.import_module(config["brain_module"])
        brain_fn  = getattr(brain_mod, config["brain_fn"])
        print(f"  ✅ Brain loaded: {config['brain_name']}")
    except Exception as e:
        print(f"  ❌ FATAL — Cannot load brain module '{config['brain_module']}': {e}")
        sys.exit(1)

    # ── Load regime brain (optional — degrades gracefully) ────────
    regime_fn = None
    try:
        from market_agent.brain.regime_ensemble import regime_ensemble_signal
        regime_fn = regime_ensemble_signal
        print("  ✅ Regime-Ensemble loaded.")
    except Exception as e:
        print(f"  ⚠️  Regime-Ensemble unavailable: {e}")
        print("     Defaulting to RANGING for all symbols. Brain will still run.")

    # ── Session tracker ───────────────────────────────────────────
    tracker = SessionTracker()

    # ── Startup banner ────────────────────────────────────────────
    print()
    print("═" * 65)
    print("  CAUSAL PAPER SCOUT  —  LIVE")
    print(f"  Brain    : {config['label']}")
    print(f"  Symbols  : {len(config['symbols'])}")
    print(f"  Interval : {interval_minutes} min per cycle")
    print(f"  Risk     : {config['risk_pct']*100:.1f}% per trade  (VIRTUAL)")
    print(f"  Account  : ₹{config['account_size']:>12,.0f}  (VIRTUAL)")
    print(f"  Targets  : WR≥{config['min_wr']:.0%}  EV>0  MaxDD≤{config['max_dd_r']:.0f}R")
    print(f"  Go-live  : After {config['min_resolved']} resolved signals passing all thresholds")
    print("═" * 65)
    print()
    print("  Symbols and market status at startup:")
    for s in config["symbols"]:
        status = "🟢 OPEN" if _is_market_open(s) else "🔴 closed"
        print(f"    {s:<22} {status}")
    print()

    # ── Main loop ─────────────────────────────────────────────────
    cycle = 0
    while True:
        cycle += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'─' * 65}")
        print(f"  CYCLE {cycle}  |  {ts}  |  brain={config['brain_name']}")
        print(f"{'─' * 65}")

        try:
            n = _run_scan_cycle(config, storage, tracker, brain_fn, regime_fn)

            if n == 0:
                print("  No new signals this cycle.")
            else:
                print(f"\n  {n} new signal(s) stored.")

            print()
            tracker.print_summary()

            # Print full DB performance every 3 cycles
            if cycle % 3 == 0:
                _print_performance(storage, config)

        except KeyboardInterrupt:
            print("\n\n  Stopped by user (Ctrl+C).")
            print("  Final performance report:")
            _print_performance(storage, config)
            sys.exit(0)

        except Exception as e:
            # Non-fatal — log and keep running
            logger.error("cycle_unhandled_error", cycle=cycle, error=str(e))
            print(f"  ⚠️  Unhandled error in cycle {cycle}: {e}")
            print("     Scout will continue on next cycle.")

        print(f"\n  ⏱  Sleeping {interval_minutes} min until next cycle...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    start_scout()