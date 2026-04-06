from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config.settings import settings


class SQLiteStore:
    def __init__(self) -> None:
        data_dir = Path(settings.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "trading_data.db"
        self._init_db()

    @contextmanager
    def conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    learning_tier TEXT,
                    count_in_learning INTEGER DEFAULT 0,
                    is_full_close INTEGER DEFAULT 1,
                    pnl_net REAL DEFAULT 0,
                    pnl_gross REAL DEFAULT 0,
                    fee_usdt REAL DEFAULT 0,
                    leverage REAL DEFAULT 0,
                    requested_leverage REAL DEFAULT 0,
                    leverage_cap REAL DEFAULT 0,
                    margin_used REAL DEFAULT 0,
                    margin_pct REAL DEFAULT 0,
                    pnl_on_margin_pct REAL DEFAULT 0,
                    hold_seconds REAL DEFAULT 0,
                    entry_time TEXT,
                    close_time TEXT,
                    reason TEXT,
                    review_area TEXT,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time);
                CREATE INDEX IF NOT EXISTS idx_trades_learning ON trades(learning_tier, count_in_learning);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

                CREATE TABLE IF NOT EXISTS exploration_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    setup_key TEXT,
                    side TEXT,
                    market_regime TEXT,
                    confidence_before REAL DEFAULT 0,
                    confidence_delta REAL DEFAULT 0,
                    confidence_after REAL DEFAULT 0,
                    result_label TEXT,
                    leverage REAL DEFAULT 0,
                    margin_used REAL DEFAULT 0,
                    pnl_net REAL DEFAULT 0,
                    fee_usdt REAL DEFAULT 0,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_exploration_symbol ON exploration_samples(symbol, setup_key);
                CREATE INDEX IF NOT EXISTS idx_exploration_time ON exploration_samples(timestamp);

                CREATE TABLE IF NOT EXISTS gpt_daily_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    day TEXT,
                    created_at TEXT,
                    summary TEXT,
                    effective_trade_count INTEGER DEFAULT 0,
                    exploratory_trade_count INTEGER DEFAULT 0,
                    raw_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_gpt_daily_day ON gpt_daily_reviews(day);
                """
            )

    def append_trade(self, record: Dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False)
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    timestamp, symbol, side, learning_tier, count_in_learning, is_full_close,
                    pnl_net, pnl_gross, fee_usdt, leverage, requested_leverage, leverage_cap,
                    margin_used, margin_pct, pnl_on_margin_pct, hold_seconds,
                    entry_time, close_time, reason, review_area, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.get("timestamp", "")),
                    str(record.get("symbol", "")),
                    str(record.get("side", "")),
                    str(record.get("learning_tier", "")),
                    1 if bool(record.get("count_in_learning")) else 0,
                    1 if bool(record.get("is_full_close", True)) else 0,
                    float(record.get("pnl_net", record.get("pnl", 0.0)) or 0.0),
                    float(record.get("pnl_gross", record.get("realized_pnl_gross", 0.0)) or 0.0),
                    float(record.get("fee_usdt", record.get("fill_fee", 0.0)) or 0.0),
                    float(record.get("leverage", 0.0) or 0.0),
                    float(record.get("requested_leverage", 0.0) or 0.0),
                    float(record.get("leverage_cap", 0.0) or 0.0),
                    float(record.get("margin_used", record.get("effective_margin_used", 0.0)) or 0.0),
                    float(record.get("margin_pct", 0.0) or 0.0),
                    float(record.get("pnl_on_margin_pct", 0.0) or 0.0),
                    float(record.get("hold_seconds", 0.0) or 0.0),
                    str(record.get("entry_time", "")),
                    str(record.get("close_time", record.get("timestamp", ""))),
                    str(record.get("reason", "")),
                    str(record.get("review_area", "")),
                    payload,
                ),
            )

    def append_exploration_sample(self, sample: Dict[str, Any]) -> None:
        payload = json.dumps(sample, ensure_ascii=False)
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO exploration_samples (
                    timestamp, symbol, setup_key, side, market_regime,
                    confidence_before, confidence_delta, confidence_after,
                    result_label, leverage, margin_used, pnl_net, fee_usdt, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(sample.get("timestamp", "")),
                    str(sample.get("symbol", "")),
                    str(sample.get("setup_key", "")),
                    str(sample.get("side", "")),
                    str(sample.get("market_regime", "")),
                    float(sample.get("confidence_before", 0.0) or 0.0),
                    float(sample.get("confidence_delta", 0.0) or 0.0),
                    float(sample.get("confidence_after", 0.0) or 0.0),
                    str(sample.get("result_label", "unknown")),
                    float(sample.get("leverage", 0.0) or 0.0),
                    float(sample.get("margin_used", 0.0) or 0.0),
                    float(sample.get("pnl_net", 0.0) or 0.0),
                    float(sample.get("fee_usdt", 0.0) or 0.0),
                    payload,
                ),
            )

    def upsert_daily_review(self, day: str, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO gpt_daily_reviews (day, created_at, summary, effective_trade_count, exploratory_trade_count, raw_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    created_at=excluded.created_at,
                    summary=excluded.summary,
                    effective_trade_count=excluded.effective_trade_count,
                    exploratory_trade_count=excluded.exploratory_trade_count,
                    raw_json=excluded.raw_json
                """,
                (
                    day,
                    str(payload.get("created_at", "")),
                    str(payload.get("summary", "")),
                    int(payload.get("effective_trade_count", 0) or 0),
                    int(payload.get("exploratory_trade_count", 0) or 0),
                    raw,
                ),
            )

    def _decode_rows(self, rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                out.append(json.loads(row["raw_json"]))
            except Exception:
                out.append(dict(row))
        return out

    def recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.conn() as conn:
            rows = conn.execute("SELECT raw_json FROM trades ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return list(reversed(self._decode_rows(rows)))

    def recent_exploration(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.conn() as conn:
            rows = conn.execute("SELECT raw_json FROM exploration_samples ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return list(reversed(self._decode_rows(rows)))

    def latest_daily_reviews(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.conn() as conn:
            rows = conn.execute("SELECT raw_json FROM gpt_daily_reviews ORDER BY day DESC LIMIT ?", (int(limit),)).fetchall()
        return self._decode_rows(rows)

    def stats(self) -> Dict[str, Any]:
        with self.conn() as conn:
            trade_count = int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])
            effective_count = int(conn.execute("SELECT COUNT(*) FROM trades WHERE count_in_learning=1").fetchone()[0])
            exploration_count = int(conn.execute("SELECT COUNT(*) FROM exploration_samples").fetchone()[0])
            review_count = int(conn.execute("SELECT COUNT(*) FROM gpt_daily_reviews").fetchone()[0])
            fee_total = float(conn.execute("SELECT COALESCE(SUM(fee_usdt),0) FROM trades").fetchone()[0])
            pnl_total = float(conn.execute("SELECT COALESCE(SUM(pnl_net),0) FROM trades WHERE count_in_learning=1").fetchone()[0])
        return {
            "db_path": str(self.db_path),
            "trade_count": trade_count,
            "effective_trade_count": effective_count,
            "exploration_sample_count": exploration_count,
            "daily_review_count": review_count,
            "effective_pnl_total": round(pnl_total, 6),
            "fee_total": round(fee_total, 6),
        }
