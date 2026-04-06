from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from config.settings import settings
from storage.sqlite_store import SQLiteStore


class TradeStore:
    def __init__(self) -> None:
        p = Path(settings.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.file_path = p / "trade_records.jsonl"
        self.sqlite = SQLiteStore()

    def append(self, record: Dict[str, Any]) -> None:
        payload = dict(record)
        now_ts = datetime.now(timezone.utc).isoformat()
        payload.setdefault("timestamp", now_ts)
        payload.setdefault("close_time", payload.get("timestamp", now_ts))
        payload.setdefault("count_in_learning", payload.get("learning_tier") == "effective")
        payload.setdefault("pnl_net", payload.get("pnl", 0.0))
        payload.setdefault("pnl_amount", payload.get("pnl_net", payload.get("pnl", 0.0)))
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.sqlite.append_trade(payload)

    def append_exploration_sample(self, sample: Dict[str, Any]) -> None:
        self.sqlite.append_exploration_sample(sample)

    def save_daily_review(self, day: str, payload: Dict[str, Any]) -> None:
        self.sqlite.upsert_daily_review(day, payload)

    def _all(self) -> List[Dict[str, Any]]:
        rows = self.sqlite.recent_trades(limit=5000)
        if rows:
            return rows
        if not self.file_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.sqlite.recent_trades(limit=limit)
        return rows if rows else self._all()[-limit:]

    def recent_exploration(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.sqlite.recent_exploration(limit=limit)

    def storage_stats(self) -> Dict[str, Any]:
        return self.sqlite.stats()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, "", "None"):
                return default
            return float(value)
        except Exception:
            return default

    def _is_full_close(self, row: Dict[str, Any]) -> bool:
        return bool(row.get("is_full_close", True))

    def _is_effective_learning(self, row: Dict[str, Any]) -> bool:
        if not self._is_full_close(row):
            return False
        if not bool(row.get("count_in_learning", row.get("learning_tier") == "effective")):
            return False
        pnl_net = self._safe_float(row.get("pnl_net", row.get("pnl", 0.0)), 0.0)
        return abs(pnl_net) >= settings.effective_learning_min_abs_pnl_usdt

    def reflection_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.recent(limit=limit * 6)
        filtered = [r for r in rows if self._is_effective_learning(r)]
        return filtered[-limit:] if filtered else []

    def records_for_day(self, day: str, tz_name: str | None = None, *, effective_only: bool = False) -> List[Dict[str, Any]]:
        tz = ZoneInfo(tz_name or settings.gpt_review_timezone)
        out: List[Dict[str, Any]] = []
        for row in self._all():
            ts = str(row.get("close_time", row.get("timestamp", "")))
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt.astimezone(tz).date().isoformat() != day:
                continue
            if effective_only and not self._is_effective_learning(row):
                continue
            out.append(row)
        return out

    def latest_trading_day(self, tz_name: str | None = None) -> str:
        tz = ZoneInfo(tz_name or settings.gpt_review_timezone)
        rows = self._all()
        if not rows:
            return ""
        for row in reversed(rows):
            ts = str(row.get("close_time", row.get("timestamp", "")))
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(tz).date().isoformat()
            except Exception:
                continue
        return ""

    def consecutive_losses(self, limit: int = 50) -> int:
        streak = 0
        for row in reversed(self.reflection_recent(limit=limit)):
            if self._safe_float(row.get("pnl_net", row.get("pnl", 0.0)), 0.0) > 0:
                break
            streak += 1
        return streak

    def summary(self, limit: int = 50) -> Dict[str, Any]:
        rows = self.recent(limit=limit * 8)
        effective_rows = [row for row in rows if self._is_effective_learning(row)]
        effective_rows = effective_rows[-limit:]
        exploratory_rows = [row for row in rows if self._is_full_close(row) and not self._is_effective_learning(row)]
        exploration_samples = self.recent_exploration(limit=limit * 8)
        total_realized = sum(self._safe_float(r.get("pnl_net", r.get("pnl", 0.0)), 0.0) for r in effective_rows)
        total_fees = sum(abs(self._safe_float(r.get("fee_usdt", r.get("fill_fee", 0.0)), 0.0)) for r in effective_rows)
        wins = sum(1 for row in effective_rows if self._safe_float(row.get("pnl_net", row.get("pnl", 0.0)), 0.0) > 0)
        losses = sum(1 for row in effective_rows if self._safe_float(row.get("pnl_net", row.get("pnl", 0.0)), 0.0) <= 0)
        avg_pnl = total_realized / len(effective_rows) if effective_rows else 0.0
        avg_margin_return_pct = (
            sum(self._safe_float(r.get("pnl_on_margin_pct", 0.0), 0.0) for r in effective_rows) / len(effective_rows)
            if effective_rows else 0.0
        )
        exp_success = sum(1 for s in exploration_samples if str(s.get("result_label", "")).lower() in {"win","positive","success","effective"})
        exp_total = len(exploration_samples)
        return {
            "count": len(effective_rows),
            "wins": wins,
            "losses": losses,
            "win_count": wins,
            "loss_count": losses,
            "win_rate": round((wins / len(effective_rows) * 100.0), 2) if effective_rows else 0.0,
            "avg_pnl": round(avg_pnl, 6),
            "avg_margin_return_pct": round(avg_margin_return_pct, 4),
            "realized_pnl": round(total_realized, 6),
            "total_fees": round(total_fees, 6),
            "consecutive_losses": self.consecutive_losses(limit=limit),
            "total_count": len(rows),
            "closed_count": len(effective_rows),
            "exploratory_closed_count": len(exploratory_rows),
            "exploration_sample_count": exp_total,
            "exploration_success_rate": round((exp_success / exp_total * 100.0), 2) if exp_total else 0.0,
            "storage": self.storage_stats(),
        }
