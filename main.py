
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
    indicators: Optional[dict]
    notes: Optional[str] = None

@app.post("/strategy/stocks")
def stock_strategy(req: TradeRequest):
    price = req.price_data[-1]
    stop_loss_pct = 0.03
    stop_loss = round(price * (1 - stop_loss_pct), 2)
    take_profit = round(price * 1.05, 2)
    confidence = 0.78
    position_size = round(req.account_balance * req.risk_tolerance / stop_loss_pct, 2)

    return {
        "asset": req.asset,
        "action": "BUY",
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "strategy": "VWMA + Volume Profile + SMC",
        "position_size": position_size
    }

@app.post("/strategy/commodities")
def commodity_strategy(req: TradeRequest):
    db = get_db()
    params = db.execute("SELECT * FROM strategy_parameters WHERE strategy_name = ?", ("commodity_default",)).fetchone()
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

@app.post("/log")
def log_trade(trade: TradeLog):
    db = get_db()
    db.execute('''
        INSERT INTO trades (
            timestamp, asset, asset_type, strategy,
            entry_price, stop_loss, take_profit,
            confidence, result, pnl, notes, indicators
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.utcnow().isoformat(),
        trade.asset,
        trade.asset_type,
        trade.strategy,
        trade.entry_price,
        trade.stop_loss,
        trade.take_profit,
        trade.confidence,
        trade.result,
        trade.pnl,
        trade.notes,
        str(trade.indicators)
    ))
    db.commit()
    db.close()
    return {"status": "logged"}

@app.get("/learn")
def learn_from_trades():
    db = get_db()
    rows = db.execute("SELECT * FROM trades ORDER BY timestamp ASC").fetchall()
    db.close()

    if not rows:
        return {"message": "No trades logged yet."}

    stats = {}
    equity_curve = []
    balance = 10000
    max_balance = balance
    max_drawdown = 0
    degradation_window = 5

    for row in rows:
        strat = row["strategy"]
        pnl = row["pnl"] or 0.0
        confidence = row["confidence"] or 0.0
        result = row["result"]

        if strat not in stats:
            stats[strat] = {
                "wins": 0, "losses": 0, "total": 0, "pnl_list": [],
                "confidences": [], "accuracy_high_conf": 0, "high_conf_total": 0
            }

        if result == "win":
            stats[strat]["wins"] += 1
        elif result == "loss":
            stats[strat]["losses"] += 1

        stats[strat]["total"] += 1
        stats[strat]["pnl_list"].append(pnl)
        stats[strat]["confidences"].append(confidence)

        if confidence >= 0.8:
            stats[strat]["high_conf_total"] += 1
            if result == "win":
                stats[strat]["accuracy_high_conf"] += 1

        balance += pnl
        equity_curve.append(balance)
        if balance > max_balance:
            max_balance = balance
        drawdown = (max_balance - balance) / max_balance
        max_drawdown = max(max_drawdown, drawdown)

    for s in stats:
        total = stats[s]["total"]
        stats[s]["win_rate"] = round(stats[s]["wins"] / total, 2) if total else 0
        stats[s]["avg_pnl"] = round(statistics.mean(stats[s]["pnl_list"]), 2) if stats[s]["pnl_list"] else 0
        stats[s]["avg_confidence"] = round(statistics.mean(stats[s]["confidences"]), 2) if stats[s]["confidences"] else 0
        if stats[s]["high_conf_total"] > 0:
            stats[s]["confidence_accuracy"] = round(stats[s]["accuracy_high_conf"] / stats[s]["high_conf_total"], 2)

        if len(stats[s]["pnl_list"]) >= degradation_window * 2:
            early = stats[s]["pnl_list"][:degradation_window]
            late = stats[s]["pnl_list"][-degradation_window:]
            stats[s]["strategy_degradation"] = round(statistics.mean(late) - statistics.mean(early), 2)

    return {
        "strategy_stats": stats,
        "final_balance": round(balance, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "total_trades": len(rows)
    }

@app.get("/")
def root():
    return {"message": "AI strategy backend v5 with full analytics is running."}
