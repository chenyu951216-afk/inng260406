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

    Important:
    - NO live GPT control
    - NO per-symbol calls
    - NO large payloads
    - Only runs once daily around 00:00~00:05 Taiwan time
    - Keeps compatibility methods so old runtime code will not crash
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.last_run_date = None

    # -----------------------------
    # Compatibility for old runtime
    # -----------------------------
    def available(self) -> bool:
        return bool(self.client)

    def live_available(self) -> bool:
        # Explicitly disable live GPT overlay to avoid cost explosion
        return False

    def advise_live(self, candidate: Dict[str, Any], account_summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # No live calls. Return candidate unchanged with a marker.
        result = dict(candidate or {})
        result["gpt_live_used"] = False
        result["gpt_live_mode"] = "disabled"
        return result

    def overlay_live(self, candidate: Dict[str, Any], account_summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # Alias for compatibility if older code calls another method name
        return self.advise_live(candidate, account_summary)

    # -----------------------------
    # Daily summary only
    # -----------------------------
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

            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS trade_count,
                    COALESCE(SUM(CASE WHEN COALESCE(pnl_net, pnl, 0) > 0 THEN 1 ELSE 0 END), 0) AS wins,
                    COALESCE(SUM(CASE WHEN COALESCE(pnl_net, pnl, 0) <= 0 THEN 1 ELSE 0 END), 0) AS losses,
                    COALESCE(SUM(COALESCE(pnl_net, pnl, 0)), 0) AS total_pnl,
                    COALESCE(SUM(ABS(COALESCE(fee_usdt, fill_fee, 0))), 0) AS total_fee,
                    COALESCE(AVG(COALESCE(drawdown, 0)), 0) AS avg_drawdown,
                    COALESCE(AVG(COALESCE(leverage, 0)), 0) AS avg_leverage,
                    COALESCE(AVG(COALESCE(margin_used, margin, 0)), 0) AS avg_margin
                FROM trades
                WHERE DATE(COALESCE(close_time, entry_time, timestamp, created_at)) = ?
                  AND COALESCE(count_in_learning, 0) = 1
                """,
                (today,),
            ).fetchone()

            trade_count = int(row[0] or 0)
            wins = int(row[1] or 0)
            losses = int(row[2] or 0)
            total_pnl = float(row[3] or 0.0)
            total_fee = float(row[4] or 0.0)
            avg_drawdown = float(row[5] or 0.0)
            avg_leverage = float(row[6] or 0.0)
            avg_margin = float(row[7] or 0.0)

            return {
                "day": today,
                "trade_count": trade_count,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / max(trade_count, 1), 6) if trade_count else 0.0,
                "avg_pnl": round(total_pnl / max(trade_count, 1), 6) if trade_count else 0.0,
                "total_pnl": round(total_pnl, 6),
                "fee": round(total_fee, 6),
                "avg_drawdown": round(avg_drawdown, 6),
                "avg_leverage": round(avg_leverage, 6),
                "avg_margin": round(avg_margin, 6),
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
                    {
                        "role": "system",
                        "content": "你是交易AI優化顧問。只根據固定摘要提供精簡優化建議，禁止要求逐筆原始交易明細。"
                    },
                    {
                        "role": "user",
                        "content": (
                            f"今日交易摘要：{summary}。"
                            "請用精簡條列回覆：1.TP/SL優化 2.槓桿建議 3.進場過濾建議 4.風控提醒。"
                        ),
                    },
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
