from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict

from openai import OpenAI

DB_PATH = os.getenv("DB_PATH", "/data/trading_data.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TIMEZONE_OFFSET = 8


class GPTAdvisorService:
    """
    Safe daily-only GPT path.
    - no live control
    - no per-symbol calls
    - no huge payloads
    - only fixed compact summary at 00:00~00:05 Taiwan time
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.last_run_date = None

    def available(self) -> bool:
        return bool(self.client)

    def _now_local(self) -> datetime:
        return datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)

    def should_run_daily(self) -> bool:
        now = self._now_local()
        today = now.date()
        if now.hour == 0 and now.minute < 5 and self.last_run_date != today:
            self.last_run_date = today
            return True
        return False

    def _fetch_today_summary(self) -> Dict[str, Any]:
        conn = sqlite3.connect(DB_PATH)
        try:
            today = self._now_local().date().isoformat()
            rows = conn.execute(
                """
                SELECT
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN COALESCE(pnl_net, pnl, 0) > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN COALESCE(pnl_net, pnl, 0) <= 0 THEN 1 ELSE 0 END) as losses,
                    COALESCE(SUM(COALESCE(pnl_net, pnl, 0)), 0) as total_pnl,
                    COALESCE(SUM(ABS(COALESCE(fee_usdt, fill_fee, 0))), 0) as total_fee,
                    COALESCE(AVG(COALESCE(drawdown, 0)), 0) as avg_drawdown
                FROM trades
                WHERE DATE(COALESCE(close_time, entry_time, timestamp)) = ?
                  AND COALESCE(count_in_learning, 0) = 1
                """,
                (today,),
            ).fetchone()

            trade_count = int(rows[0] or 0)
            wins = int(rows[1] or 0)
            losses = int(rows[2] or 0)
            total_pnl = float(rows[3] or 0.0)
            total_fee = float(rows[4] or 0.0)
            avg_drawdown = float(rows[5] or 0.0)
            return {
                "day": today,
                "trade_count": trade_count,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / max(trade_count, 1), 6),
                "avg_pnl": round(total_pnl / max(trade_count, 1), 6) if trade_count else 0.0,
                "total_pnl": round(total_pnl, 6),
                "fee": round(total_fee, 6),
                "avg_drawdown": round(avg_drawdown, 6),
            }
        finally:
            conn.close()

    def _call_gpt(self, summary: Dict[str, Any]) -> str | None:
        if not self.client:
            return None
        try:
            response = self.client.responses.create(
                model=os.getenv("GPT_MODEL", "gpt-4.1-mini"),
                input=[
                    {"role": "system", "content": "你是交易AI優化顧問。只根據摘要提供簡短優化建議。"},
                    {"role": "user", "content": f"今日交易摘要：{summary}。請給 TP/SL、槓桿、進場邏輯的精簡建議。"},
                ],
            )
            text = getattr(response, "output_text", None)
            if text:
                return text
            try:
                return response.output[0].content[0].text
            except Exception:
                return None
        except Exception as exc:
            print("GPT ERROR:", exc)
            return None

    def _save_advice(self, advice: str) -> None:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gpt_advice (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("INSERT INTO gpt_advice (content) VALUES (?)", (advice,))
            conn.commit()
        finally:
            conn.close()

    def run(self) -> None:
        if not self.should_run_daily():
            return
        summary = self._fetch_today_summary()
        advice = self._call_gpt(summary)
        if advice:
            self._save_advice(advice)
