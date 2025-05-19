from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import sqlite3

app = FastAPI()

# Database utility functions

def get_db():
    conn = sqlite3.connect("ai_trading.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    db = get_db()
    # Trades table
    db.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            asset TEXT,
            asset_type TEXT,
            strategy TEXT,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            confidence REAL,
            result TEXT,
            pnl REAL,
            notes TEXT,
            indicators TEXT
        )
    ''')
    # Strategy parameters
    db.execute('''
        CREATE TABLE IF NOT EXISTS strategy_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT UNIQUE,
            stop_loss_pct REAL,
            confidence_threshold REAL,
            macro_weight REAL,
            updated_at TEXT
        )
    ''')
    # Seed default commodity and stock strategies
    for name, sl, conf, macro in [
        ("commodity_default", 0.03, 0.75, 0.2),
        ("stock_default", 0.02, 0.80, 0.3)
    ]:
        db.execute('''
            INSERT INTO strategy_parameters (strategy_name, stop_loss_pct, confidence_threshold, macro_weight, updated_at)
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM strategy_parameters WHERE strategy_name = ?
            )
        ''', (name, sl, conf, macro, datetime.utcnow().isoformat(), name))
    db.commit()
    db.close()

# Initialize DB on startup
init_db()

# Request models
class TradeRequest(BaseModel):
    asset: str
    asset_type: str
    price_data: List[float]
    volume_data: Optional[List[float]] = None
    indicators: Optional[Dict] = {}
    account_balance: float
    risk_tolerance: Optional[float] = 0.02
    volatility_index: Optional[float] = 1.0
    correlation: Optional[float] = 0.0
    # New fields for news data
    news_sentiment: Optional[float] = None
    news_events: Optional[List[Dict]] = None

class UpdateStrategySettings(BaseModel):
    strategy_name: str
    confidence_threshold: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    macro_weight: Optional[float] = None

class LogRequest(BaseModel):
    timestamp: Optional[str] = None
    asset: str
    asset_type: str
    strategy: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: Optional[float] = None
    result: Optional[str] = None
    pnl: Optional[float] = None
    notes: Optional[str] = None
    indicators: Optional[Dict] = {}
    # Add news fields to logging
    news_sentiment: Optional[float] = None
    news_events: Optional[List[Dict]] = None

# Helper: fetch all trades

def fetch_trades():
    db = get_db()
    rows = db.execute("SELECT * FROM trades").fetchall()
    db.close()
    return rows

# Strategy endpoints
@app.post("/strategy/commodities")
def commodity_strategy(req: TradeRequest):
    return _apply_strategy(req, "commodity_default")

@app.post("/strategy/stocks")
def stock_strategy(req: TradeRequest):
    return _apply_strategy(req, "stock_default")


def _apply_strategy(req: TradeRequest, strategy_name: str):
    db = get_db()
    params = db.execute(
        "SELECT * FROM strategy_parameters WHERE strategy_name = ?", (strategy_name,)
    ).fetchone()
    if not params:
        db.close()
        raise HTTPException(status_code=404, detail=f"Missing strategy parameters: {strategy_name}")

    price = req.price_data[-1]
    stop_loss_pct = params["stop_loss_pct"]
    confidence_threshold = params["confidence_threshold"]
    macro_weight = params["macro_weight"]
    
    # Basic TP at +5%
    stop_loss = round(price * (1 - stop_loss_pct), 2)
    take_profit = round(price * 1.05, 2)
    confidence = round(confidence_threshold + 0.03, 2)
    
    # Process news sentiment if available
    news_impact = 0
    if req.news_sentiment is not None:
        # Apply sentiment impact scaled by macro_weight
        news_impact = req.news_sentiment * macro_weight
        confidence += news_impact
    
    # Check for significant news events
    significant_event = False
    event_note = ""
    if req.news_events:
        for event in req.news_events:
            # Look for high importance events
            if event.get('importance', 0) > 0.7:
                significant_event = True
                event_note = f"Significant event: {event.get('title', 'Unnamed event')}"
                
                # Adjust take profit for positive events
                if event.get('sentiment', 0) > 0:
                    take_profit = round(price * 1.07, 2)  # Higher target on positive news
                break
    
    # Calculate position size
    position_size = round(req.account_balance * req.risk_tolerance / stop_loss_pct, 2)
    
    # Determine action with news override for very negative news
    action = "BUY" if confidence > confidence_threshold else "WAIT"
    if req.news_sentiment is not None and req.news_sentiment < -0.7:
        action = "WAIT"  # Override on very negative news
        event_note += " | Trading paused due to negative news"
    
    # Log trade with news info
    notes = "Auto-logged"
    if event_note:
        notes += f" | {event_note}"
    
    news_info = {
        "sentiment": req.news_sentiment,
        "events": req.news_events[:2] if req.news_events else []  # Limit to first 2 events
    }
    
    # Log the trade
    db.execute(
        '''INSERT INTO trades (
            timestamp, asset, asset_type, strategy,
            entry_price, stop_loss, take_profit, confidence,
            result, pnl, notes, indicators
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        datetime.utcnow().isoformat(), 
        req.asset, 
        req.asset_type, 
        strategy_name,
        price, 
        stop_loss, 
        take_profit, 
        confidence,
        None, 
        None, 
        notes,
        str({**req.indicators, "news": news_info})  # Include news in indicators
    ))
    db.commit()
    db.close()

    # Return response with news impact info
    return {
        "asset": req.asset,
        "action": action,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "strategy": strategy_name,
        "position_size": position_size,
        "news_impact": news_impact,
        "significant_event": significant_event
    }

# Manual logging endpoint
@app.post("/log")
def manual_log(log: LogRequest):
    db = get_db()
    ts = log.timestamp or datetime.utcnow().isoformat()
    
    # Prepare indicators with news data if available
    indicators_data = log.indicators or {}
    if log.news_sentiment is not None or log.news_events:
        news_info = {
            "sentiment": log.news_sentiment,
            "events": log.news_events[:2] if log.news_events else []
        }
        indicators_data["news"] = news_info
    
    db.execute(
        '''INSERT INTO trades (
            timestamp, asset, asset_type, strategy,
            entry_price, stop_loss, take_profit, confidence,
            result, pnl, notes, indicators
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        ts, log.asset, log.asset_type, log.strategy,
        log.entry_price, log.stop_loss, log.take_profit, log.confidence,
        log.result, log.pnl, log.notes or "", str(indicators_data)
    ))
    db.commit()
    db.close()
    return {"message": "Trade logged successfully"}

# Analytics endpoint
@app.get("/learn")
def learn():
    rows = fetch_trades()
    total = len(rows)
    wins = sum(1 for r in rows if r["pnl"] and r["pnl"] > 0)
    losses = sum(1 for r in rows if r["pnl"] and r["pnl"] <= 0)
    win_rate = wins / total if total else None
    avg_pnl = sum(r["pnl"] for r in rows if r["pnl"] is not None) / total if total else None
    
    # Add news analysis section
    news_influenced_trades = []
    for r in rows:
        indicators_str = r["indicators"]
        if '"news":' in indicators_str:
            news_influenced_trades.append(r)
    
    news_trades_count = len(news_influenced_trades)
    news_wins = sum(1 for r in news_influenced_trades if r["pnl"] and r["pnl"] > 0)
    news_win_rate = news_wins / news_trades_count if news_trades_count else None
    
    # Max drawdown calculation
    drawdowns = [r["pnl"] for r in rows if r["pnl"] is not None]
    max_dd = min(drawdowns) if drawdowns else None
    
    # Confidence accuracy
    correct_conf = sum(1 for r in rows if r["confidence"] and r["pnl"] is not None and 
                    ((r["confidence"]>=0.5 and r["pnl"]>0) or (r["confidence"]<0.5 and r["pnl"]<=0)))
    accuracy = correct_conf / total if total else None

    # Return with news analysis
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_dd,
        "confidence_accuracy": accuracy,
        "news_analysis": {
            "news_influenced_trades": news_trades_count,
            "news_win_rate": news_win_rate,
            "difference_from_overall": (news_win_rate - win_rate) if news_win_rate and win_rate else None
        }
    }

# Update strategy parameters
@app.post("/update-strategy-settings")
def update_strategy_settings(settings: UpdateStrategySettings):
    db = get_db()
    row = db.execute(
        "SELECT * FROM strategy_parameters WHERE strategy_name = ?", (settings.strategy_name,)
    ).fetchone()
    if not row:
        db.close()
        raise HTTPException(status_code=404, detail=f"Strategy '{settings.strategy_name}' not found")

    new_stop = settings.stop_loss_pct if settings.stop_loss_pct is not None else row["stop_loss_pct"]
    new_conf = settings.confidence_threshold if settings.confidence_threshold is not None else row["confidence_threshold"]
    new_macro = settings.macro_weight if settings.macro_weight is not None else row["macro_weight"]
    updated_at = datetime.utcnow().isoformat()

    db.execute(
        '''UPDATE strategy_parameters
           SET stop_loss_pct = ?, confidence_threshold = ?, macro_weight = ?, updated_at = ?
           WHERE strategy_name = ?''',
        (new_stop, new_conf, new_macro, updated_at, settings.strategy_name)
    )
    db.commit()
    db.close()
    return {
        "message": f"Strategy '{settings.strategy_name}' updated successfully",
        "strategy_parameters": {
            "stop_loss_pct": new_stop,
            "confidence_threshold": new_conf,
            "macro_weight": new_macro,
            "updated_at": updated_at
        }
    }

# Health check
@app.get("/")
def root():
    return {"message": "AI strategy backend v7 with news integration is running."}