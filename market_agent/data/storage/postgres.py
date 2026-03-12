import os
import pickle
import structlog
from typing import Optional
from sqlalchemy import (
    create_engine, Column, String, Integer,
    Float as SAFloat, Float, LargeBinary, DateTime, Index, text,
    Numeric, BigInteger, Boolean
)
import json as _json
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime



logger = structlog.get_logger()

# ── Sentence-transformer model for semantic embeddings ───────────────────────
# Loaded lazily on first call — does not slow down DB init or import time.
# Used by store_brain_thought(). Same model (all-MiniLM-L6-v2, 384-dim) that
# MarketSituations already uses for cosine search in pgvector.
_ST_MODEL = None

def _get_st_model():
    """Return the cached sentence-transformer model, loading it on first call."""
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info('sentence_transformer_loaded', model='all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning('sentence_transformer_unavailable', error=str(e))
            _ST_MODEL = None
    return _ST_MODEL
# ─────────────────────────────────────────────────────────────────────────────

Base = declarative_base()

class MarketData(Base):
    """
    Layer 0: Pure Data Storage
    """
    __tablename__ = 'market_data'
    
    id         = Column(Integer, primary_key=True)
    symbol     = Column(String, nullable=False, index=True)
    timestamp  = Column(DateTime, nullable=False, index=True)
    timeframe  = Column(String, nullable=False)
    data_binary = Column(LargeBinary, nullable=True)   # LEGACY — kept for backfill period
    created_at = Column(DateTime, default=datetime.utcnow)

    # ── Phase 2 columnar fields (Step 2.3) ──────────────────
    # Added by migrate_schema.py. store_ohlc writes to BOTH data_binary
    # and these fields. get_latest_data reads from these first (columnar
    # path), falls back to Pickle only if close_price is NULL.
    open_price  = Column(Numeric(18, 6), nullable=True)
    high_price  = Column(Numeric(18, 6), nullable=True)
    low_price   = Column(Numeric(18, 6), nullable=True)
    close_price = Column(Numeric(18, 6), nullable=True)
    volume_val  = Column(BigInteger, nullable=True)
    data_source = Column(String(20), default='legacy')

class Predictions(Base):
    """
    Layer 4: Self-Awareness Tracking
    Stores model forecasts to compare against future reality.
    """
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_id = Column(String, nullable=False)

    # Probabilities for Down, Flat, Up (Index 0, 1, 2)
    prob_down = Column(LargeBinary)   # Stored as binary floats for precision
    prob_flat = Column(LargeBinary)
    prob_up = Column(LargeBinary)

    # Expected price range/volatility
    predicted_range = Column(LargeBinary)

    # Link to regime detected at time of prediction
    regime = Column(String)

    # Metadata
    conf_score = Column(LargeBinary)  # Calibrated confidence
    created_at = Column(DateTime, default=datetime.utcnow)

    # PATCH 2: Outcome tracking columns.
    # Previously only added by migrate_schema.py ALTER TABLE but never declared
    # in the ORM model — causing silent write failures in update_prediction_outcome()
    # on any DB where migration had not been run. Declaring them here ensures
    # SQLAlchemy always knows about them and Base.metadata.create_all() creates them.
    actual_direction = Column(String(5),      nullable=True)
    actual_price_6h  = Column(Numeric(18, 6), nullable=True)
    was_correct      = Column(Boolean,         nullable=True)
    checked_at       = Column(DateTime,        nullable=True)

class MarketSituations(Base):
    """
    Layer 8: Long-Term Memory (Vector)
    Stores vectorized market states and corresponding findings.
    """
    __tablename__ = 'market_situations'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # The Vector Embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding = Column(Vector(384)) 
    
    # Human-readable context and outcome
    context = Column(String)
    research_finding = Column(String)
    regime = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class NeuralCouncilArchive(Base):
    """
    Phase 41 & 42: Long-Term Reasoning Memory
    Stores debate outcomes, participants, and causal insights.
    """
    __tablename__ = 'neural_council_archive'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    topic = Column(String, nullable=False)
    verdict = Column(String)
    participants = Column(String) # List of brains
    critical_insight = Column(String)
    causal_explanation = Column(String)


class CompanyFundamentals(Base):
    """
    Phase 46: Holding Strategy - Company Financials
    Stores fundamental data for long-term holding analysis.
    """
    __tablename__ = 'company_fundamentals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String)  # 'screener.in', 'nse', 'yfinance'
    
    # Core Financials
    market_cap = Column(LargeBinary)  # Stored as pickle for precision
    revenue = Column(LargeBinary)
    net_profit = Column(LargeBinary)
    eps = Column(LargeBinary)
    pe_ratio = Column(LargeBinary)
    book_value = Column(LargeBinary)
    
    # Debt & Cash Position
    total_debt = Column(LargeBinary)
    total_cash = Column(LargeBinary)
    debt_to_equity = Column(LargeBinary)
    current_ratio = Column(LargeBinary)
    free_cash_flow = Column(LargeBinary)
    
    # Shareholding Pattern
    promoter_holding = Column(LargeBinary)
    fii_holding = Column(LargeBinary)
    dii_holding = Column(LargeBinary)
    public_holding = Column(LargeBinary)
    
    # Health Scores (calculated)
    altman_z_score = Column(LargeBinary)
    piotroski_f_score = Column(LargeBinary)
    
    # Raw JSON for additional data
    raw_data = Column(LargeBinary)
    
    __table_args__ = (
        Index('idx_fundamentals_symbol_time', 'symbol', 'timestamp'),
    )


class CorporateActions(Base):
    """
    Phase 47: Corporate Actions & Deals
    Tracks dividends, splits, M&A, investments, etc.
    """
    __tablename__ = 'corporate_actions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    action_date = Column(DateTime, nullable=False, index=True)
    announcement_date = Column(DateTime)
    
    # Action Type: DIVIDEND, SPLIT, BONUS, M&A, JV, INVESTMENT, BUYBACK
    action_type = Column(String, nullable=False)
    
    # Details
    description = Column(String)
    value = Column(LargeBinary)  # Dividend amount, split ratio, deal value
    
    # AI Analysis
    impact_score = Column(LargeBinary)  # -1.0 (negative) to +1.0 (positive)
    sentiment = Column(String)  # 'bullish', 'bearish', 'neutral'
    ai_analysis = Column(String)  # Natural language explanation
    
    # Source
    source = Column(String)  # 'bse', 'nse', 'moneycontrol', 'rss'
    source_url = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
class BrainAnalystLogs(Base):
    """
    Phase 45: Autonomous Analyst Persistence
    Stores the 'thoughts' and conclusions of the Autonomous Analyst.
    """
    __tablename__ = 'brain_analyst_logs'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    sentiment = Column(LargeBinary) # Float
    conclusion = Column(String)
    missing_data = Column(String) # JSON list
    active_news_json = Column(String) # Top headlines at time of thought
    vector_embedding = Column(Vector(384)) # Phase 48: Semantic Neural Memory
    
    __table_args__ = (
        Index('idx_analyst_symbol_time', 'symbol', 'timestamp'),
    )


class BrainTrainingRun(Base):
    """
    Phase 49: Brain Training Ledger
    Tracks when each brain/model was trained and on roughly how much data.
    """
    __tablename__ = 'brain_training_runs'

    id = Column(Integer, primary_key=True)
    model_id = Column(String, nullable=False, index=True)
    mode = Column(String)  # e.g. "Refining AMV-LSTM", "Auto-Refining RL Weighter"
    bars_learned = Column(Integer)  # approximate number of bars/samples
    epochs = Column(Integer)  # approximate epochs in that run
    notes = Column(String)  # free-form notes ("dashboard-triggered", etc.)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class CouncilDebate(Base):
    """
    Phase 3: Council Debate Records
    Stores every council debate with positions, verdicts, and outcomes.
    """
    __tablename__ = 'council_debates'

    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    symbol = Column(String, index=True)
    regime = Column(String)
    trigger_type = Column(String)  # accuracy_drop, brain_disagreement, etc.
    participants_json = Column(String)  # JSON: [{brain, position, confidence, reasoning}]
    verdict = Column(String)
    verdict_confidence = Column(SAFloat)
    outcome = Column(String, default='PENDING')  # CORRECT, WRONG, PENDING, EXCELLENT
    debate_hash = Column(String(64), index=True)  # SHA-256 for deduplication
    debate_duration_sec = Column(SAFloat)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime)
    embedding = Column(Vector(384))  # pgvector for semantic search

    __table_args__ = (
        Index('idx_debate_symbol_time', 'symbol', 'created_at'),
    )

class AICodeProposal(Base):
    """
    Phase 3.5: AI Code Enhancement Proposals
    Tracks brain-proposed improvements through peer review → testing → approval.
    """
    __tablename__ = 'ai_code_proposals'

    id = Column(Integer, primary_key=True)
    pr_id = Column(String(30), unique=True, index=True)  # e.g. "AEP-001"
    proposing_brain = Column(String(50), nullable=False)
    file_path = Column(String)
    diagnosis = Column(String)          # What the brain found
    proposed_diff = Column(String)      # Suggested code change
    expected_impact = Column(String)    # "Improve accuracy by ~5% in VOLATILE regime"
    test_code = Column(String)          # Unit test the brain wrote
    test_result = Column(String(20), default='PENDING')  # PASSED, FAILED, PENDING
    peer_reviews_json = Column(String)  # JSON: [{brain, vote, reasoning}]
    consensus = Column(String(30), default='PENDING')  # AI_APPROVED, NEEDS_DISCUSSION, REJECTED
    user_status = Column(String(30), default='PENDING')  # PENDING, APPROVED, REJECTED
    accuracy_before = Column(SAFloat)
    accuracy_after = Column(SAFloat)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    applied_at = Column(DateTime)

    __table_args__ = (
        Index('idx_proposal_brain', 'proposing_brain'),
    )


class CouncilVerdict(Base):
    """
    Path A: Council's agreed decision per symbol (entry, T1, T2, SL, direction, timeframe_min).
    One row per council decision; used to display "Council decided" and resolve council accuracy.
    """
    __tablename__ = 'council_verdicts'

    id = Column(Integer, primary_key=True)
    council_session_id = Column(String(50), nullable=True, index=True)

    symbol = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    direction = Column(String, nullable=False)  # BUY, SELL, HOLD
    entry_price = Column(SAFloat, nullable=False)
    target_1 = Column(SAFloat)
    target_2 = Column(SAFloat)
    stop_loss = Column(SAFloat)
    timeframe_min = Column(Integer)  # 1-20 min suggested hold
    participants_json = Column(String)  # JSON: [{brain, position, ...}]
    source_debate_id = Column(Integer)  # optional FK to council_debates.id
    
    # Execution Tracking (Gap 6 / Phase 2)
    actual_exit_price = Column(SAFloat)
    outcome = Column(String(10))
    pnl_pct = Column(SAFloat)
    exit_timestamp = Column(DateTime)
    holding_candles = Column(Integer)

    __table_args__ = (
        Index('idx_council_verdict_symbol_time', 'symbol', 'created_at'),
    )


class BrainPrediction(Base):
    """
    Tier 1 Fix (Gap 5): Individual brain prediction BEFORE council vote.
    Required because council verdict is identical for all brains.
    """
    __tablename__ = 'brain_predictions'

    id = Column(Integer, primary_key=True)
    council_session_id = Column(String(50), nullable=False, index=True)
    brain_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(5), nullable=False)
    confidence = Column(SAFloat)
    signal_strength = Column(SAFloat)
    method_confidence = Column(SAFloat)
    regime_suitability = Column(String(10))
    regime = Column(String(30))
    predicted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Filled by resolver:
    outcome = Column(String(10))
    was_correct = Column(Boolean)
    actual_direction = Column(String(5))
    resolved_at = Column(DateTime)

    __table_args__ = (
        Index('idx_bp_brain_regime', 'brain_name', 'regime', 'was_correct'),
        Index('idx_bp_unresolved', 'symbol', 'predicted_at', postgresql_where=(Column('was_correct') == None)),
    )


# ── Paper Trade Signals ───────────────────────────────────────────────────────
# Stores every virtual signal fired during paper trading.
# outcome: NULL = still open, T1_HIT = winner, SL_HIT = loser, EXPIRED = timed out

class PaperTradeSignal(Base):
    __tablename__ = 'paper_trade_signals'

    id          = Column(Integer, primary_key=True, autoincrement=True)
    brain_name  = Column(String,  nullable=False, index=True)
    symbol      = Column(String,  nullable=False, index=True)
    direction   = Column(String,  nullable=False)          # BUY | SELL
    entry_price = Column(Float,   nullable=False)
    target_1    = Column(Float,   nullable=False)
    target_2    = Column(Float,   nullable=True)
    stop_loss   = Column(Float,   nullable=False)
    confidence  = Column(Float,   nullable=True)
    regime      = Column(String,  nullable=True)
    timeframe   = Column(String,  nullable=True)
    strategy    = Column(String,  nullable=True)
    outcome     = Column(String,  nullable=True)           # NULL | T1_HIT | SL_HIT | EXPIRED
    exit_price  = Column(Float,   nullable=True)
    pnl_r       = Column(Float,   nullable=True)           # profit/loss in R units
    created_at  = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('idx_pts_brain_open', 'brain_name', 'outcome'),
        Index('idx_pts_symbol_open', 'symbol', 'outcome'),
    )


class PostgresStorage:
    def __init__(self, connection_string=None):
        if not connection_string:
            # Priority 1: Render injects a single DATABASE_URL — use it directly
            db_url = os.getenv("DATABASE_URL", "")
            if db_url:
                # Render uses postgres:// scheme; SQLAlchemy requires postgresql://
                connection_string = db_url.replace("postgres://", "postgresql://", 1)
            else:
                # Priority 2: individual env vars (local dev)
                user     = os.getenv("DB_USER",     "agent_user")
                password = os.getenv("DB_PASSWORD", "agent_password")
                host     = os.getenv("DB_HOST",     "localhost")
                port     = os.getenv("DB_PORT",     "5433")
                db_name  = os.getenv("DB_NAME",     "market_data")
                connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)

        # Enable pgvector extension
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        Base.metadata.create_all(self.engine)
        # logger.info("database_initialized") # Silenced to prevent Windows OSError [Errno 22]

    def store_ohlc(self, symbol, timestamp, timeframe, data_dict,
                   source: str = None):
        """
        Step 2.3 DUAL-WRITE shim.
        Writes to both the legacy data_binary (Pickle) AND the new
        columnar numeric fields simultaneously.  This keeps all existing
        readers working while the new columnar path fills up.
        After Phase 6 validation, data_binary will be dropped.

        PATCH 1: source default changed from 'unknown' to None.
        If source is not passed, a warning is logged with the caller location
        and the row is tagged 'unset' — clearly flagged for later review.
        This was the root cause of the 39,002 mystery 1m rows (data_source='unknown').
        All callers must now pass source= explicitly.
        """
        # GUARD: Every ingested row must have a named source.
        if not source:
            import traceback
            caller = ''.join(traceback.format_stack(limit=3)[-2:-1]).strip()
            logger.warning(
                'store_ohlc_missing_source',
                symbol=symbol,
                timeframe=timeframe,
                hint='Pass source= explicitly. Rows tagged unset are excluded from brain training.',
                caller=caller[:200],
            )
            source = 'unset'
        session = self.Session()
        try:
            # Check for existing row (upsert semantics)
            existing = session.query(MarketData).filter(
                MarketData.symbol    == symbol,
                MarketData.timestamp == timestamp,
                MarketData.timeframe == timeframe,
            ).first()

            fields = dict(
                data_binary  = pickle.dumps(data_dict),       # LEGACY — keep during transition
                open_price   = float(data_dict.get('Open',  0) or 0),
                high_price   = float(data_dict.get('High',  0) or 0),
                low_price    = float(data_dict.get('Low',   0) or 0),
                close_price  = float(data_dict.get('Close', 0) or 0),
                volume_val   = int(data_dict.get('Volume',  0) or 0),
                data_source  = source,
            )

            if existing:
                for k, v in fields.items():
                    setattr(existing, k, v)
            else:
                entry = MarketData(
                    symbol=symbol, timestamp=timestamp, timeframe=timeframe,
                    **fields
                )
                session.add(entry)

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error('store_ohlc_failed', symbol=symbol, error=str(e))
            raise
        finally:
            session.close()

    def get_latest_data(self, symbol, timeframe, limit=100):
        """
        Step 2.3 COLUMNAR-READ-FIRST shim.
        Reads from new numeric columns when available (close_price is not NULL).
        Falls back to Pickle for rows that haven't been backfilled yet.
        Return signature is IDENTICAL to old code: list of {timestamp, data} dicts.
        """
        session = self.Session()
        try:
            results = session.query(MarketData).filter(
                MarketData.symbol    == symbol,
                MarketData.timeframe == timeframe,
            ).order_by(MarketData.timestamp.desc()).limit(limit).all()

            output = []
            for r in reversed(results):
                # Columnar path (new, preferred)
                if r.close_price is not None and float(r.close_price) > 0:
                    data = {
                        'Open':   float(r.open_price  or 0),
                        'High':   float(r.high_price  or 0),
                        'Low':    float(r.low_price   or 0),
                        'Close':  float(r.close_price),
                        'Volume': int(r.volume_val or 0),
                    }
                # Legacy Pickle path (pre-migration rows)
                elif r.data_binary:
                    try:
                        data = pickle.loads(r.data_binary)
                    except Exception:
                        continue  # corrupt row — skip silently
                else:
                    continue  # empty row — skip
                output.append({'timestamp': r.timestamp, 'data': data})

            return output
        finally:
            session.close()

    def get_last_council_debate(self, symbol: str) -> dict:
        """
        Returns the most recent council_debates row for a given symbol as a dict.
        Used by Command Center to read the real regime instead of hardcoding.
        Returns {} if no debate exists yet.
        """
        session = self.Session()
        try:
            row = session.execute(
                text("""
                    SELECT regime, verdict, verdict_confidence, symbol, created_at
                    FROM council_debates
                    WHERE symbol = :sym
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"sym": symbol}
            ).fetchone()
            if row:
                return {
                    "regime":     row[0] or "UNKNOWN",
                    "verdict":    row[1],
                    "confidence": row[2],
                    "symbol":     row[3],
                    "created_at": row[4],
                }
            return {}
        except Exception as e:
            logger.debug("get_last_council_debate_failed", symbol=symbol, error=str(e))
            return {}
        finally:
            session.close()

    def store_prediction(self, symbol, timestamp, model_id, probs, predicted_range, regime, confidence):
        """Stores a model prediction for later evaluation."""
        session = self.Session()
        try:
            entry = Predictions(
                symbol=symbol,
                timestamp=timestamp,
                model_id=model_id,
                prob_down=pickle.dumps(probs[0]),
                prob_flat=pickle.dumps(probs[1]),
                prob_up=pickle.dumps(probs[2]),
                predicted_range=pickle.dumps(predicted_range),
                regime=regime,
                conf_score=pickle.dumps(confidence)
            )
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            # logger.error("prediction_storage_failed")
            raise
        finally:
            session.close()

    def get_pending_evaluations(self, limit=100, symbol=None):
        """Retrieves predictions that haven't been matched with reality yet."""
        session = self.Session()
        try:
            query = session.query(Predictions)
            if symbol:
                query = query.filter(Predictions.symbol == symbol)
            return query.order_by(Predictions.timestamp.desc()).limit(limit).all()
        finally:
            session.close()

    def store_situation(self, symbol, timestamp, embedding, context, finding, regime):
        """Stores a vectorized market situation for long-term memory."""
        session = self.Session()
        try:
            entry = MarketSituations(
                symbol=symbol,
                timestamp=timestamp,
                embedding=embedding,
                context=context,
                research_finding=finding,
                regime=regime
            )
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            # logger.error("situation_storage_failed")
            raise
        finally:
            session.close()

    def search_similar_situations(self, embedding, limit=5):
        """Finds historical market situations similar to the current one."""
        session = self.Session()
        try:
            # Note: cosine_distance is 1 - cosine_similarity
            return session.query(MarketSituations).order_by(
                MarketSituations.embedding.cosine_distance(embedding)
            ).limit(limit).all()
        finally:
            session.close()

    def archive_council_session(self, topic, verdict, participants, insight, causal_exp="N/A", limit=10000):
        """
        Stores a Neural Council session for long-term reasoning.
        Includes Pruning Logic: Drops oldest 10% if memory exceeds 'limit'.
        """
        session = self.Session()
        try:
            # 1. Check current count
            count = session.query(NeuralCouncilArchive).count()
            
            if count >= limit:
                # Prune oldest 10%
                prune_count = int(limit * 0.1)
                logger.info("memory_threshold_reached", action="pruning", count=count, dropping=prune_count)
                
                # SQLAlchemy subquery to find IDs of oldest N records
                oldest_ids = session.query(NeuralCouncilArchive.id).order_by(
                    NeuralCouncilArchive.timestamp.asc()
                ).limit(prune_count).all()
                
                ids_to_del = [i[0] for i in oldest_ids]
                session.query(NeuralCouncilArchive).filter(
                    NeuralCouncilArchive.id.in_(ids_to_del)
                ).delete(synchronize_session=False)
                session.commit()

            # 2. Add new entry
            entry = NeuralCouncilArchive(
                topic=topic,
                verdict=verdict,
                participants=",".join(participants) if isinstance(participants, list) else participants,
                critical_insight=insight,
                causal_explanation=causal_exp
            )
            session.add(entry)
            session.commit()
            logger.info("council_session_archived", topic=topic)
        except Exception as e:
            session.rollback()
            # logger.error("council_archival_failed")
        finally:
            session.close()

    def store_fundamentals(self, symbol, data_dict, source="screener.in"):
        """Phase 46: Stores company fundamental data."""
        session = self.Session()
        try:
            entry = CompanyFundamentals(
                symbol=symbol,
                source=source,
                market_cap=pickle.dumps(data_dict.get('market_cap')),
                revenue=pickle.dumps(data_dict.get('revenue')),
                net_profit=pickle.dumps(data_dict.get('net_profit')),
                eps=pickle.dumps(data_dict.get('eps')),
                pe_ratio=pickle.dumps(data_dict.get('pe_ratio')),
                book_value=pickle.dumps(data_dict.get('book_value')),
                total_debt=pickle.dumps(data_dict.get('total_debt')),
                total_cash=pickle.dumps(data_dict.get('total_cash')),
                debt_to_equity=pickle.dumps(data_dict.get('debt_to_equity')),
                current_ratio=pickle.dumps(data_dict.get('current_ratio')),
                free_cash_flow=pickle.dumps(data_dict.get('free_cash_flow')),
                promoter_holding=pickle.dumps(data_dict.get('promoter_holding')),
                fii_holding=pickle.dumps(data_dict.get('fii_holding')),
                dii_holding=pickle.dumps(data_dict.get('dii_holding')),
                public_holding=pickle.dumps(data_dict.get('public_holding')),
                altman_z_score=pickle.dumps(data_dict.get('altman_z')),
                piotroski_f_score=pickle.dumps(data_dict.get('piotroski_f')),
                raw_data=pickle.dumps(data_dict),
            )
            session.add(entry)
            session.commit()
            logger.info("fundamentals_stored", symbol=symbol, source=source)
        except Exception as e:
            session.rollback()
            # logger.error("fundamentals_storage_failed")
        finally:
            session.close()

    def get_latest_fundamentals(self, symbol):
        """Retrieves the most recent fundamental data for a symbol."""
        session = self.Session()
        try:
            result = session.query(CompanyFundamentals).filter(
                CompanyFundamentals.symbol == symbol
            ).order_by(CompanyFundamentals.timestamp.desc()).first()
            
            if result:
                return {
                    'symbol': symbol,
                    'timestamp': result.timestamp,
                    'source': result.source,
                    'market_cap': pickle.loads(result.market_cap) if result.market_cap else None,
                    'revenue': pickle.loads(result.revenue) if result.revenue else None,
                    'net_profit': pickle.loads(result.net_profit) if result.net_profit else None,
                    'eps': pickle.loads(result.eps) if result.eps else None,
                    'promoter_holding': pickle.loads(result.promoter_holding) if result.promoter_holding else None,
                    'fii_holding': pickle.loads(result.fii_holding) if result.fii_holding else None,
                    'altman_z': pickle.loads(result.altman_z_score) if result.altman_z_score else None,
                    'piotroski_f': pickle.loads(result.piotroski_f_score) if result.piotroski_f_score else None,
                }
            return None
        finally:
            session.close()

    def store_corporate_action(self, symbol, action_type, action_date, description, 
                                value=None, impact_score=None, source="moneycontrol"):
        """Phase 47: Stores a corporate action (dividend, M&A, etc.)."""
        session = self.Session()
        try:
            entry = CorporateActions(
                symbol=symbol,
                action_type=action_type,
                action_date=action_date,
                description=description,
                value=pickle.dumps(value) if value else None,
                impact_score=pickle.dumps(impact_score) if impact_score else None,
                source=source,
            )
            session.add(entry)
            session.commit()
            logger.info("corporate_action_stored", symbol=symbol, action_type=action_type)
        except Exception as e:
            session.rollback()
            # logger.error("corporate_action_storage_failed")
        finally:
            session.close()

    def get_corporate_actions(self, symbol, limit=20):
        """Retrieves recent corporate actions for a symbol."""
        session = self.Session()
        try:
            results = session.query(CorporateActions).filter(
                CorporateActions.symbol == symbol
            ).order_by(CorporateActions.action_date.desc()).limit(limit).all()
            
            return [{
                'action_type': r.action_type,
                'action_date': r.action_date,
                'description': r.description,
                'value': pickle.loads(r.value) if r.value else None,
                'impact_score': pickle.loads(r.impact_score) if r.impact_score else None,
                'source': r.source,
            } for r in results]
        finally:
            session.close()

    def store_brain_thought(self, symbol, sentiment, conclusion, missing_data, news_list, limit=10000):
        """Stores the analyst's conclusion in DB with pruning."""
        session = self.Session()
        try:
            import json
            # 1. Pruning logic
            count = session.query(BrainAnalystLogs).count()
            if count >= limit:
                prune_count = int(limit * 0.1)
                oldest_ids = session.query(BrainAnalystLogs.id).order_by(
                    BrainAnalystLogs.timestamp.asc()
                ).limit(prune_count).all()
                ids_to_del = [i[0] for i in oldest_ids]
                session.query(BrainAnalystLogs).filter(
                    BrainAnalystLogs.id.in_(ids_to_del)
                ).delete(synchronize_session=False)
                session.commit()

            # 2. Add new entry
            # PATCH 3: Real semantic embedding using sentence-transformers all-MiniLM-L6-v2.
            # Previously used [sentiment] * 384 — a vector of identical floats with zero
            # semantic meaning. pgvector cosine search on it returned garbage results.
            # Now embeds the conclusion text so get_brain_history() semantic search works.
            # Falls back to sentiment-seeded vector only if the model is unavailable.
            st_model = _get_st_model()
            if st_model is not None and conclusion:
                try:
                    embedding = st_model.encode(conclusion, normalize_embeddings=True).tolist()
                except Exception as emb_err:
                    logger.warning('embedding_failed', symbol=symbol, error=str(emb_err))
                    embedding = [float(sentiment)] * 384
            else:
                # Fallback: sentiment-seeded vector (no semantic meaning, but doesn't crash)
                embedding = [float(sentiment)] * 384

            entry = BrainAnalystLogs(
                symbol=symbol,
                sentiment=pickle.dumps(sentiment),
                conclusion=conclusion,
                missing_data=json.dumps(missing_data),
                active_news_json=json.dumps(news_list[:5]),
                vector_embedding=embedding
            )
            session.add(entry)
            session.commit()
            logger.info("brain_thought_stored", symbol=symbol, semantic=True)
        except Exception as e:
            session.rollback()
            # logger.error("thought_storage_failed")
        finally:
            session.close()

    def get_brain_history(self, symbol, limit=10, semantic_query_vec=None):
        """Retrieves historical thoughts for a symbol, with optional semantic search."""
        session = self.Session()
        try:
            import json
            query = session.query(BrainAnalystLogs).filter(BrainAnalystLogs.symbol == symbol)
            
            if semantic_query_vec:
                # Semantic retrieval using pgvector
                query = query.order_by(BrainAnalystLogs.vector_embedding.cosine_distance(semantic_query_vec))
            else:
                query = query.order_by(BrainAnalystLogs.timestamp.desc())

            results = query.limit(limit).all()
            
            return [{
                'timestamp': r.timestamp,
                'sentiment': pickle.loads(r.sentiment),
                'conclusion': r.conclusion,
                'missing_data': json.loads(r.missing_data),
                'news': json.loads(r.active_news_json)
            } for r in results]
        finally:
            session.close()

    # ═══════════════════════════════════════════════════════════
    # Brain Training Ledger Helpers
    # ═══════════════════════════════════════════════════════════

    def log_brain_training_run(self, model_id: str, mode: str,
                               bars_learned: int = None, epochs: int = None,
                               notes: str = ""):
        """Persist a single training run for a given brain/model."""
        session = self.Session()
        try:
            run = BrainTrainingRun(
                model_id=model_id,
                mode=mode,
                bars_learned=bars_learned,
                epochs=epochs,
                notes=notes,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )
            session.add(run)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()

    def get_brain_training_stats(self, model_id: str):
        """
        Get simple training stats for a brain:
        - total_runs
        - last_run_at
        - last_mode
        """
        session = self.Session()
        try:
            from sqlalchemy import func

            total = session.query(func.count(BrainTrainingRun.id)).filter(
                BrainTrainingRun.model_id == model_id
            ).scalar()

            last = session.query(BrainTrainingRun).filter(
                BrainTrainingRun.model_id == model_id
            ).order_by(BrainTrainingRun.completed_at.desc()).first()

            if not total:
                return {
                    "total_runs": 0,
                    "last_run_at": None,
                    "last_mode": None,
                }

            return {
                "total_runs": int(total),
                "last_run_at": last.completed_at if last else None,
                "last_mode": last.mode if last else None,
            }
        finally:
            session.close()

    # ═══════════════════════════════════════
    # COUNCIL DEBATE METHODS (Phase 3)
    # ═══════════════════════════════════════

    def store_council_debate(self, topic: str, symbol: str, regime: str,
                             trigger_type: str, participants: list,
                             verdict: str, verdict_confidence: float,
                             debate_hash: str, debate_duration_sec: float = 0.0,
                             embedding: list = None):
        """Store a council debate record."""
        session = self.Session()
        try:
            debate = CouncilDebate(
                topic=topic[:1000],
                symbol=symbol,
                regime=regime,
                trigger_type=trigger_type,
                participants_json=_json.dumps(participants) if participants else '[]',
                verdict=verdict[:1000],
                verdict_confidence=verdict_confidence,
                debate_hash=debate_hash,
                debate_duration_sec=debate_duration_sec,
                embedding=embedding,
            )
            session.add(debate)
            session.commit()
            return debate.id
        except Exception as e:
            session.rollback()
            logger.error("store_debate_failed", error=str(e))
            return None
        finally:
            session.close()

    def get_council_debates(self, symbol: str = None, limit: int = 20) -> list:
        """Get recent council debates, optionally filtered by symbol."""
        session = self.Session()
        try:
            query = session.query(CouncilDebate).order_by(
                CouncilDebate.created_at.desc()
            )
            if symbol:
                query = query.filter(CouncilDebate.symbol == symbol)
            
            results = query.limit(limit).all()
            return [
                {
                    "id": r.id,
                    "topic": r.topic,
                    "symbol": r.symbol,
                    "regime": r.regime,
                    "trigger_type": r.trigger_type,
                    "participants": _json.loads(r.participants_json) if r.participants_json else [],
                    "verdict": r.verdict,
                    "verdict_confidence": r.verdict_confidence,
                    "outcome": r.outcome,
                    "debate_hash": r.debate_hash,
                    "debate_duration_sec": r.debate_duration_sec,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
                }
                for r in results
            ]
        except Exception as e:
            logger.error("get_debates_failed", error=str(e))
            return []
        finally:
            session.close()

    def store_council_verdict(self, symbol: str, direction: str, entry_price: float,
                              target_1: float = None, target_2: float = None, stop_loss: float = None,
                              timeframe_min: int = None, participants_json: str = None,
                              source_debate_id: int = None, council_session_id: str = None) -> Optional[int]:
        """Path A: Store council's agreed decision (entry, T1, T2, SL). Returns verdict id or None."""
        session = self.Session()
        try:
            v = CouncilVerdict(
                council_session_id=council_session_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                target_1=target_1,
                target_2=target_2,
                stop_loss=stop_loss,
                timeframe_min=timeframe_min,
                participants_json=participants_json,
                source_debate_id=source_debate_id,
            )
            session.add(v)
            session.commit()
            return v.id
        except Exception as e:
            session.rollback()
            logger.error("store_council_verdict_failed", error=str(e))
            return None
        finally:
            session.close()

    def get_latest_council_verdict(self, symbol: str) -> Optional[dict]:
        """Path A: Get most recent council verdict for symbol for UI (Council decided row)."""
        session = self.Session()
        try:
            r = session.query(CouncilVerdict).filter(
                CouncilVerdict.symbol == symbol
            ).order_by(CouncilVerdict.created_at.desc()).first()
            if r:
                return {
                    "id": r.id,
                    "council_session_id": r.council_session_id,
                    "symbol": r.symbol,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "direction": r.direction,
                    "entry_price": r.entry_price,
                    "target_1": r.target_1,
                    "target_2": r.target_2,
                    "stop_loss": r.stop_loss,
                    "timeframe_min": r.timeframe_min,
                    "participants_json": r.participants_json,
                    "source_debate_id": r.source_debate_id,
                }
            return None
        except Exception as e:
            logger.error("get_latest_council_verdict_failed", error=str(e))
            return None
        finally:
            session.close()

    def store_brain_prediction(self,
                               council_session_id: str,
                               brain_name: str,
                               symbol: str,
                               direction: str,
                               confidence: float,
                               regime: str,
                               signal_strength: float = None,
                               method_confidence: float = None,
                               regime_suitability: str = None) -> Optional[int]:
        """Store individual brain prediction BEFORE council vote."""
        session = self.Session()
        try:
            pred = BrainPrediction(
                council_session_id=council_session_id,
                brain_name=brain_name,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                regime=regime,
                signal_strength=signal_strength,
                method_confidence=method_confidence,
                regime_suitability=regime_suitability,
                predicted_at=datetime.utcnow()
            )
            session.add(pred)
            session.commit()
            return pred.id
        except Exception as e:
            session.rollback()
            logger.error('store_brain_prediction_failed',
                         brain=brain_name, symbol=symbol, error=str(e))
            return None
        finally:
            session.close()

    def get_brain_predictions_for_session(self, session_id: str):
        """Fetch all individual brain predictions linked to a council session."""
        session = self.Session()
        try:
            return session.query(BrainPrediction).filter(
                BrainPrediction.council_session_id == session_id
            ).all()
        finally:
            session.close()

    def update_brain_prediction_outcome(self, pred_id: int, outcome: str,
                                        actual_direction: str, was_correct: bool):
        """Update outcome for an individual brain pred after signal resolution."""
        session = self.Session()
        try:
            pred = session.query(BrainPrediction).get(pred_id)
            if pred:
                pred.outcome = outcome
                pred.actual_direction = actual_direction
                pred.was_correct = was_correct
                pred.resolved_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error('update_brain_pred_outcome_failed',
                         pred_id=pred_id, error=str(e))
        finally:
            session.close()

    def get_debate_by_hash(self, debate_hash: str, days: int = 7) -> dict:
        """Get the most recent debate with this hash from the last N days."""
        session = self.Session()
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=days)
            result = session.query(CouncilDebate).filter(
                CouncilDebate.debate_hash == debate_hash,
                CouncilDebate.created_at >= cutoff
            ).order_by(CouncilDebate.created_at.desc()).first()

            if result:
                return {
                    "id": result.id,
                    "outcome": result.outcome,
                    "verdict": result.verdict,
                    "created_at": result.created_at.isoformat() if result.created_at else None,
                }
            return None
        except Exception as e:
            logger.error("get_debate_hash_failed", error=str(e))
            return None
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────────
    # PHASE 2 OUTCOME TRACKING — Step 2.2 / 2.3
    # ─────────────────────────────────────────────────────────────────

    def update_prediction_outcome(self, prediction_id: int,
                                   actual_price: float,
                                   actual_direction: str) -> None:
        """Fill was_correct on a past prediction once outcome is known."""
        session = self.Session()
        try:
            pred = session.query(Predictions).get(prediction_id)
            if not pred:
                return
            # Determine predicted direction from stored probs
            p_up   = pickle.loads(pred.prob_up)   if pred.prob_up   else 0.0
            p_down = pickle.loads(pred.prob_down) if pred.prob_down else 0.0
            pred_dir = 'UP' if p_up > p_down else 'DOWN'
            pred.actual_price_6h  = actual_price
            pred.actual_direction = actual_direction
            pred.was_correct      = (pred_dir == actual_direction)
            pred.checked_at       = datetime.utcnow()
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error('update_prediction_outcome_failed', id=prediction_id, error=str(e))
        finally:
            session.close()

    def record_verdict_outcome(self, verdict_id: int, exit_price: float,
                                outcome: str, pnl_pct: float,
                                exit_ts, holding_candles: int) -> None:
        """Record T1/SL/EXPIRED results on a closed council verdict."""
        session = self.Session()
        try:
            v = session.query(CouncilVerdict).get(verdict_id)
            if not v:
                return
            v.actual_exit_price = exit_price
            v.outcome           = outcome
            v.pnl_pct           = pnl_pct
            v.exit_timestamp    = exit_ts
            v.holding_candles   = holding_candles
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error('record_verdict_outcome_failed', id=verdict_id, error=str(e))
        finally:
            session.close()

    def get_open_verdicts(self, symbol: str = None) -> list:
        """Return all council verdicts without a closed outcome yet."""
        session = self.Session()
        try:
            q = session.query(CouncilVerdict).filter(
                CouncilVerdict.outcome == None  # noqa: E711
            )
            if symbol:
                q = q.filter(CouncilVerdict.symbol == symbol)
            return q.order_by(CouncilVerdict.created_at.desc()).all()
        finally:
            session.close()

    def clear_all_market_data(self) -> None:
        """
        DANGER — wipes ALL rows from market_data.
        Only called by rebuild_db.py after explicit 'YES' confirmation.
        """
        session = self.Session()
        try:
            session.execute(text('TRUNCATE TABLE market_data'))
            session.commit()
            logger.warning('market_data_cleared_all')
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Paper Trade Methods ───────────────────────────────────────────────────

    def store_paper_signal(self, brain_name: str, symbol: str, direction: str,
                           entry_price: float, target_1: float, target_2: float,
                           stop_loss: float, confidence: float = None,
                           regime: str = None, timeframe: str = None,
                           strategy: str = None) -> Optional[int]:
        """
        Store a new paper trade signal. Returns the row ID or None on failure.
        outcome is NULL (open) until resolve_paper_signals() closes it.
        """
        session = self.Session()
        try:
            row = PaperTradeSignal(
                brain_name  = brain_name,
                symbol      = symbol,
                direction   = direction,
                entry_price = entry_price,
                target_1    = target_1,
                target_2    = target_2,
                stop_loss   = stop_loss,
                confidence  = confidence,
                regime      = regime,
                timeframe   = timeframe,
                strategy    = strategy,
                outcome     = None,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id
        except Exception as e:
            session.rollback()
            logger.error("store_paper_signal_failed", error=str(e)[:120])
            return None
        finally:
            session.close()

    def resolve_paper_signals(self, symbol: str, current_price: float,
                              expiry_hours: int = 48) -> int:
        """
        Check all open signals for *symbol* against current_price.
        Closes signals that hit T1, SL, or are older than expiry_hours.
        Returns count of signals resolved this call.
        """
        session = self.Session()
        resolved = 0
        try:
            open_signals = session.query(PaperTradeSignal).filter(
                PaperTradeSignal.symbol  == symbol,
                PaperTradeSignal.outcome == None,
            ).all()

            now = datetime.utcnow()
            for sig in open_signals:
                outcome    = None
                exit_price = current_price
                pnl_r      = None

                sl_dist = abs(sig.entry_price - sig.stop_loss)
                if sl_dist <= 0:
                    continue

                if sig.direction == "BUY":
                    if current_price >= sig.target_1:
                        outcome = "T1_HIT"
                        pnl_r   = round((sig.target_1 - sig.entry_price) / sl_dist, 3)
                    elif current_price <= sig.stop_loss:
                        outcome = "SL_HIT"
                        pnl_r   = -1.0
                else:  # SELL
                    if current_price <= sig.target_1:
                        outcome = "T1_HIT"
                        pnl_r   = round((sig.entry_price - sig.target_1) / sl_dist, 3)
                    elif current_price >= sig.stop_loss:
                        outcome = "SL_HIT"
                        pnl_r   = -1.0

                # Expiry check
                if outcome is None and sig.created_at:
                    age_hours = (now - sig.created_at).total_seconds() / 3600
                    if age_hours >= expiry_hours:
                        outcome    = "EXPIRED"
                        exit_price = current_price
                        pnl_r      = round((current_price - sig.entry_price) /
                                           sl_dist * (1 if sig.direction == "BUY" else -1), 3)

                if outcome:
                    sig.outcome     = outcome
                    sig.exit_price  = exit_price
                    sig.pnl_r       = pnl_r
                    sig.resolved_at = now
                    resolved += 1

            if resolved:
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error("resolve_paper_signals_failed", symbol=symbol, error=str(e)[:120])
        finally:
            session.close()
        return resolved

    def get_paper_trade_performance(self, brain_name: str) -> Optional[dict]:
        """
        Compute cumulative paper trade performance for a brain.
        Returns dict with WR, EV, total_R, max_drawdown, open count, recent signals.
        Returns None if no signals exist yet.
        """
        session = self.Session()
        try:
            all_sigs = session.query(PaperTradeSignal).filter(
                PaperTradeSignal.brain_name == brain_name
            ).order_by(PaperTradeSignal.created_at).all()

            if not all_sigs:
                return None

            total    = len(all_sigs)
            open_n   = sum(1 for s in all_sigs if s.outcome is None)
            decided  = [s for s in all_sigs if s.outcome in ("T1_HIT", "SL_HIT")]
            wins     = [s for s in decided if s.outcome == "T1_HIT"]
            losses   = [s for s in decided if s.outcome == "SL_HIT"]

            n_decided = len(decided)
            wr        = len(wins) / n_decided if n_decided > 0 else 0.0
            pnls      = [s.pnl_r for s in decided if s.pnl_r is not None]
            total_r   = round(sum(pnls), 3) if pnls else 0.0
            ev        = round(total_r / n_decided, 3) if n_decided > 0 else 0.0

            # Max drawdown in R (peak-to-trough on running cumulative R)
            max_dd   = 0.0
            peak     = 0.0
            running  = 0.0
            for p in pnls:
                running += p
                if running > peak:
                    peak = running
                dd = peak - running
                if dd > max_dd:
                    max_dd = dd

            # Last 10 resolved for display
            resolved_all = [s for s in all_sigs if s.outcome is not None]
            recent = [
                {
                    "symbol":    s.symbol,
                    "direction": s.direction,
                    "outcome":   s.outcome,
                    "pnl_r":     s.pnl_r,
                    "regime":    s.regime,
                    "created_at": s.created_at,
                }
                for s in resolved_all[-10:]
            ]

            return {
                "total":    total,
                "open":     open_n,
                "decided":  n_decided,
                "wins":     len(wins),
                "losses":   len(losses),
                "wr":       round(wr, 4),
                "total_r":  total_r,
                "ev":       ev,
                "max_dd":   round(max_dd, 3),
                "recent":   recent,
            }
        except Exception as e:
            logger.error("get_paper_perf_failed", brain=brain_name, error=str(e)[:120])
            return None
        finally:
            session.close()


if __name__ == "__main__":
    storage = PostgresStorage()
    print("PostgreSQL Storage ready with Fundamentals, Corporate Actions, and Council Debates support.")