from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from config.settings import settings
from storage.trade_store import TradeStore


class DualLayerAIService:
    def __init__(self) -> None:
        self.trades = TradeStore()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, "", "None"):
                return default
            return float(value)
        except Exception:
            return default

    def effective_margin_floor(self, equity: float) -> float:
        return max(settings.min_margin_floor_usdt, max(float(equity or 0.0), 0.0) * settings.effective_learning_min_margin_ratio)

    def classify_entry(self, *, symbol: str, side: str, market_regime: str, confidence: float, leverage: int, leverage_cap: int, desired_margin: float, equity: float) -> Dict[str, Any]:
        floor = self.effective_margin_floor(equity)
        effective = desired_margin >= floor and (not settings.effective_learning_force_max_leverage or leverage >= leverage_cap)
        tier = "effective" if effective else "exploration"
        setup_key = f"{symbol}:{side}:{market_regime or 'unknown'}"
        return {
            "learning_tier": tier,
            "count_in_learning": effective,
            "setup_key": setup_key,
            "effective_margin_floor": floor,
            "requires_max_leverage": settings.effective_learning_force_max_leverage,
            "entry_confidence": confidence,
        }

    def classify_close(self, record: Dict[str, Any]) -> Dict[str, Any]:
        pnl_net = self._safe_float(record.get("pnl_net", record.get("pnl", 0.0)), 0.0)
        margin_used = self._safe_float(record.get("margin_used", 0.0), 0.0)
        floor = self.effective_margin_floor(self._safe_float(record.get("equity_snapshot", 0.0), 0.0))
        leverage = int(self._safe_float(record.get("leverage", 0.0), 0.0))
        leverage_cap = int(self._safe_float(record.get("leverage_cap", leverage), leverage))
        is_effective = abs(pnl_net) >= settings.effective_learning_min_abs_pnl_usdt and margin_used >= floor and (not settings.effective_learning_force_max_leverage or leverage >= leverage_cap)
        return {
            "learning_tier": "effective" if is_effective else "exploration",
            "count_in_learning": is_effective,
            "effective_after_close": is_effective,
            "effective_reason": "meets_realized_pnl_margin_and_leverage" if is_effective else "exploration_or_below_effective_threshold",
        }

    def record_exploration_outcome(self, record: Dict[str, Any]) -> None:
        sample = {
            "timestamp": record.get("close_time") or record.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "symbol": record.get("symbol"),
            "setup_key": record.get("setup_key") or f"{record.get('symbol')}:{record.get('side')}:{record.get('market_regime', 'unknown')}",
            "side": record.get("side"),
            "market_regime": record.get("market_regime", "unknown"),
            "confidence_before": self._safe_float(record.get("entry_confidence", 0.0), 0.0),
            "confidence_delta": self._confidence_delta(record),
            "confidence_after": self._safe_float(record.get("entry_confidence", 0.0), 0.0) + self._confidence_delta(record),
            "result_label": self._result_label(record),
            "leverage": self._safe_float(record.get("leverage", 0.0), 0.0),
            "margin_used": self._safe_float(record.get("margin_used", 0.0), 0.0),
            "pnl_net": self._safe_float(record.get("pnl_net", 0.0), 0.0),
            "fee_usdt": self._safe_float(record.get("fee_usdt", 0.0), 0.0),
        }
        self.trades.append_exploration_sample(sample)

    def _result_label(self, record: Dict[str, Any]) -> str:
        pnl = self._safe_float(record.get("pnl_net", 0.0), 0.0)
        if pnl > 0:
            return "win"
        if pnl < 0:
            return "loss"
        return "flat"

    def _confidence_delta(self, record: Dict[str, Any]) -> float:
        pnl = self._safe_float(record.get("pnl_net", 0.0), 0.0)
        fee = abs(self._safe_float(record.get("fee_usdt", 0.0), 0.0))
        effective_pnl = pnl - fee
        if effective_pnl > 0.5:
            return 0.03
        if effective_pnl > 0:
            return 0.01
        if effective_pnl < -0.5:
            return -0.03
        if effective_pnl < 0:
            return -0.01
        return 0.0
