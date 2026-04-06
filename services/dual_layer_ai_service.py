from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict

DB_PATH = os.getenv("DB_PATH", "/data/trading_data.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exploration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT,
            setup_key TEXT,
            market_regime TEXT,
            result TEXT,
            confidence REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            pnl REAL DEFAULT 0,
            leverage REAL DEFAULT 0,
            margin REAL DEFAULT 0,
            fee REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


@dataclass
class EntryClassification:
    learning_tier: str
    setup_key: str
    effective_margin_floor: float
    desired_margin: float
    leverage: int
    leverage_cap: int


class DualLayerAIService:
    """
    Dual-layer AI:
    - exploration: small/noisy results only affect confidence
    - effective learning: only meaningful live results affect learning
    """

    def __init__(self) -> None:
        pass

    def _setup_key(self, symbol: str, side: str, market_regime: str) -> str:
        return f"{symbol}:{side}:{market_regime}"

    def get_confidence(self, symbol: str) -> float:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(confidence), 0.0) FROM exploration WHERE symbol=?",
                (symbol,),
            ).fetchone()
            return float(row[0] or 0.0)
        finally:
            conn.close()

    def classify_entry(
        self,
        *,
        symbol: str,
        side: str,
        market_regime: str,
        confidence: float,
        leverage: int,
        leverage_cap: int,
        desired_margin: float,
        equity: float,
    ) -> Dict[str, Any]:
        effective_margin_floor = max(0.1, float(equity or 0.0) * 0.01)
        learning_tier = "effective" if (
            desired_margin >= effective_margin_floor and leverage >= leverage_cap
        ) else "exploration"

        setup_key = self._setup_key(symbol, side, market_regime or "unknown")
        return EntryClassification(
            learning_tier=learning_tier,
            setup_key=setup_key,
            effective_margin_floor=round(effective_margin_floor, 8),
            desired_margin=round(float(desired_margin or 0.0), 8),
            leverage=int(leverage or 1),
            leverage_cap=int(leverage_cap or 1),
        ).__dict__

    def record_exploration(
        self,
        symbol: str,
        side: str,
        market_regime: str,
        result: str,
        confidence_delta: float,
    ) -> None:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO exploration (symbol, side, setup_key, market_regime, result, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    side,
                    self._setup_key(symbol, side, market_regime or "unknown"),
                    market_regime or "unknown",
                    result,
                    float(confidence_delta or 0.0),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_learning_trade(
        self,
        symbol: str,
        pnl: float,
        leverage: float,
        margin: float,
        fee: float,
    ) -> None:
        if abs(float(pnl or 0.0)) < 0.5:
            return
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO learning (symbol, pnl, leverage, margin, fee)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    float(pnl or 0.0),
                    float(leverage or 0.0),
                    float(margin or 0.0),
                    float(fee or 0.0),
                ),
            )
            conn.commit()
        finally:
            conn.close()
