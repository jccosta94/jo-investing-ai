
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sqlite3
import statistics

app = FastAPI()

# ------------------------------
# Database Setup
# ------------------------------
def get_db():
    conn = sqlite3.connect("trades.db")
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
            exit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            result TEXT,
            confidence REAL,
            pnl REAL,
            notes TEXT
        )
    ''')
    db.commit()
    db.close()

init_db()

# ------------------------------
# Models
# ------------------------------
class TradeRequest(BaseModel):
    asset: str
    asset_type: str
    price_data: List[float]
    volume_data: Optional[List[float]] = None
    indicators: Optional[dict] = {}
    account_balance: float
    risk_tolerance: Optional[float] = 0.02
    volatility_index: Optional[float] = 1.0  # default neutral
    correlation: Optional[float] = 0.0  # with broader market

class TradeLog(BaseModel):
    asset: str
    asset_type: str
    strategy: str
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    result: Optional[str]
    confidence: float
    pnl: Optional[float]
    notes: Optional[str] = None

# ------------------------------
# Strategy Logic
# ------------------------------
@app.post("/strategy/stocks")
def stock_strategy(req: TradeRequest):
    price = req.price_data[-1]
    signal = "BUY" if req.indicators.get("vwma_trend") == "up" else "HOLD"
    confidence = 0.78
    if req.indicators.get("volume_profile_zone") == "LVN":
        confidence += 0.05

    # Adjust risk based on volatility
    adjusted_risk = req.risk_tolerance / req.volatility_index
    stop_loss_pct = 0.03
    position_size = req.account_balance * adjusted_risk / stop_loss_pct

    response = {
        "asset": req.asset,
        "action": signal,
        "entry_price": price,
        "stop_loss": round(price * (1 - stop_loss_pct), 2),
        "take_profit": round(price * 1.05, 2),
        "confidence": round(confidence, 2),
        "strategy": "VWMA + Volume Profile + SMC + Market Regime",
        "position_size": round(position_size, 2)
    }
    return response

@app.post("/strategy/commodities")
def commodity_strategy(req: TradeRequest):
    price = req.price_data[-1]
    signal = "BUY" if req.indicators.get("orderflow") == "bullish" else "WAIT"
    confidence = 0.73
    if req.indicators.get("macro_bias") == "positive":
        confidence += 0.05

    # Adjust risk for volatility and correlation
    adjusted_risk = req.risk_tolerance / (req.volatility_index * (1 + abs(req.correlation)))
    stop_loss_pct = 0.025
    position_size = req.account_balance * adjusted_risk / stop_loss_pct

    response = {
        "asset": req.asset,
        "action": signal,
        "entry_price": price,
        "stop_loss": round(price * (1 - stop_loss_pct), 2),
        "take_profit": round(price * 1.05, 2),
        "confidence": round(confidence, 2),
        "strategy": "MACD + Order Flow + Volatility Adjusted",
        "position_size": round(position_size, 2)
    }
    return response

# ------------------------------
# Logging & Learning
# ------------------------------
@app.post("/log")
def log_trade(trade: TradeLog):
    db = get_db()
    db.execute('''
        INSERT INTO trades (
            timestamp, asset, asset_type, strategy, entry_price,
            exit_price, stop_loss, take_profit, result, confidence,
            pnl, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.utcnow().isoformat(), trade.asset, trade.asset_type,
        trade.strategy, trade.entry_price, trade.exit_price, trade.stop_loss,
        trade.take_profit, trade.result, trade.confidence, trade.pnl, trade.notes
    ))
    db.commit()
    db.close()
    return {"status": "logged"}

@app.get("/learn")
def learn_from_trades():
    db = get_db()
    cur = db.execute("SELECT * FROM trades ORDER BY timestamp ASC")
    rows = cur.fetchall()
    db.close()

    if not rows:
        return {"message": "No trades logged yet."}

    stats = {}
    equity_curve = []
    balance = 10000  # starting balance
    max_balance = balance
    max_drawdown = 0

    for row in rows:
        strat = row["strategy"]
        pnl = row["pnl"] or 0.0
        result = row["result"]
        if strat not in stats:
            stats[strat] = {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "pnl_list": []}
        if result == "win":
            stats[strat]["wins"] += 1
        elif result == "loss":
            stats[strat]["losses"] += 1
        stats[strat]["total"] += 1
        stats[strat]["total_pnl"] += pnl
        stats[strat]["pnl_list"].append(pnl)

        # equity tracking
        balance += pnl
        equity_curve.append(balance)
        max_balance = max(max_balance, balance)
        drawdown = (max_balance - balance) / max_balance
        max_drawdown = max(max_drawdown, drawdown)

    for s in stats:
        wins = stats[s]["wins"]
        total = stats[s]["total"]
        pnl_list = stats[s]["pnl_list"]
        avg_pnl = statistics.mean(pnl_list) if pnl_list else 0
        rr_ratio = avg_pnl / 0.02 if 0.02 else 0
        stats[s].update({
            "win_rate": round(wins / total, 2),
            "avg_pnl": round(avg_pnl, 2),
            "rr_ratio": round(rr_ratio, 2)
        })

    return {
        "strategy_stats": stats,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "total_trades": len(rows),
        "final_balance": round(balance, 2)
    }

@app.get("/")
def root():
    return {"message": "Jo Investing AI API v4 — full market-aware, learning-based system running."}
