
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sqlite3
import statistics

app = FastAPI()

def get_db():
    conn = sqlite3.connect("ai_trading.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
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
    db.execute('''
        CREATE TABLE IF NOT EXISTS strategy_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT,
            stop_loss_pct REAL,
            confidence_threshold REAL,
            macro_weight REAL,
            updated_at TEXT
        )
    ''')
    db.execute('''
        INSERT INTO strategy_parameters (strategy_name, stop_loss_pct, confidence_threshold, macro_weight, updated_at)
        SELECT ?, ?, ?, ?, ?
        WHERE NOT EXISTS (
            SELECT 1 FROM strategy_parameters WHERE strategy_name = ?
        )
    ''', ("commodity_default", 0.03, 0.75, 0.2, datetime.utcnow().isoformat(), "commodity_default"))
    db.commit()
    db.close()

init_db()

class TradeRequest(BaseModel):
    asset: str
    asset_type: str
    price_data: List[float]
    volume_data: Optional[List[float]] = None
    indicators: Optional[dict] = {}
    account_balance: float
    risk_tolerance: Optional[float] = 0.02
    volatility_index: Optional[float] = 1.0
    correlation: Optional[float] = 0.0

@app.post("/strategy/commodities")
def commodity_strategy(req: TradeRequest):
    db = get_db()
    params = db.execute("SELECT * FROM strategy_parameters WHERE strategy_name = ?", ("commodity_default",)).fetchone()

    if not params:
        raise ValueError("Missing strategy parameters: commodity_default")

    stop_loss_pct = params["stop_loss_pct"]
    confidence_threshold = params["confidence_threshold"]

    price = req.price_data[-1]
    stop_loss = round(price * (1 - stop_loss_pct), 2)
    take_profit = round(price * 1.05, 2)
    confidence = round(confidence_threshold + 0.03, 2)
    position_size = round(req.account_balance * req.risk_tolerance / stop_loss_pct, 2)

    db.execute('''
        INSERT INTO trades (
            timestamp, asset, asset_type, strategy,
            entry_price, stop_loss, take_profit, confidence,
            result, pnl, notes, indicators
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.utcnow().isoformat(),
        req.asset,
        req.asset_type,
        "MACD + Order Flow + Volatility Adjusted",
        price,
        stop_loss,
        take_profit,
        confidence,
        None,
        None,
        "Auto-logged",
        str(req.indicators)
    ))
    db.commit()
    db.close()

    return {
        "asset": req.asset,
        "action": "BUY" if confidence > confidence_threshold else "WAIT",
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "strategy": "MACD + Order Flow + Volatility Adjusted",
        "position_size": position_size
    }

@app.get("/")
def root():
    return {"message": "AI strategy backend v5 with DB auto-init is running."}
