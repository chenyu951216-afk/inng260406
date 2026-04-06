from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any, Dict
from zoneinfo import ZoneInfo

from config.settings import settings
from storage.trade_store import TradeStore


class DailyTradeDigestService:
    """
    Safe compact digest:
    - no huge per-trade payload to GPT
    - only fixed-size summary fields
    """

    def __init__(self) -> None:
        self.store = TradeStore()

    def today_key(self) -> str:
        return datetime.now(ZoneInfo(settings.gpt_review_timezone)).date().isoformat()

    def build_digest(self, day: str | None = None) -> Dict[str, Any]:
        target_day = day or self.today_key()
        rows = self.store.records_for_day(target_day, settings.gpt_review_timezone, effective_only=True)

        if not rows:
            return {
                "day": target_day,
                "trade_count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "total_fees": 0.0,
                "avg_drawdown": 0.0,
            }

        pnls = [float(r.get("pnl_net", r.get("pnl", 0.0)) or 0.0) for r in rows]
        fees = [abs(float(r.get("fee_usdt", r.get("fill_fee", 0.0)) or 0.0)) for r in rows]
        dds = [float(r.get("drawdown", 0.0) or 0.0) for r in rows]
        wins = len([p for p in pnls if p > 0])
        losses = len(pnls) - wins

        return {
            "day": target_day,
            "trade_count": len(rows),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(len(rows), 1), 6),
            "avg_pnl": round(mean(pnls), 6),
            "total_pnl": round(sum(pnls), 6),
            "total_fees": round(sum(fees), 6),
            "avg_drawdown": round(mean(dds), 6),
        }
