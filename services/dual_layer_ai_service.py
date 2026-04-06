# DUAL LAYER AI ENGINE (Exploration + Learning)

import os
import sqlite3
import datetime

DB_PATH = os.getenv("DB_PATH", "/data/trading_data.db")


class DualLayerAI:

    def __init__(self):
        pass

    # -----------------------------
    # 🔵 Exploration Layer (small trades)
    # -----------------------------
    def record_exploration(self, symbol, setup, result, confidence_delta):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS exploration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            setup TEXT,
            result TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        INSERT INTO exploration (symbol, setup, result, confidence)
        VALUES (?, ?, ?, ?)
        """, (symbol, setup, result, confidence_delta))

        conn.commit()
        conn.close()

    def get_confidence(self, symbol):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT SUM(confidence) FROM exploration WHERE symbol=?
        """, (symbol,))

        val = cursor.fetchone()[0]
        conn.close()

        return val if val else 0

    # -----------------------------
    # 🟢 Learning Layer (real trades)
    # -----------------------------
    def record_learning_trade(self, symbol, pnl, leverage, margin, fee):
        if abs(pnl) < 0.5:
            return  # ❗過濾小單

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            pnl REAL,
            leverage REAL,
            margin REAL,
            fee REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        INSERT INTO learning (symbol, pnl, leverage, margin, fee)
        VALUES (?, ?, ?, ?, ?)
        """, (symbol, pnl, leverage, margin, fee))

        conn.commit()
        conn.close()

    def get_learning_stats(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT COUNT(*), SUM(pnl) FROM learning
        """)

        count, pnl = cursor.fetchone()
        conn.close()

        return {
            "count": count or 0,
            "pnl": pnl or 0
        }

    # -----------------------------
    # 🧠 Decision Logic
    # -----------------------------
    def should_enter(self, symbol):
        confidence = self.get_confidence(symbol)

        # ❗信心太低不做
        if confidence < -0.5:
            return False

        return True

