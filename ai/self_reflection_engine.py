from typing import Any, Dict, List

from ai.adaptive_policy_store import AdaptivePolicyStore
from config.settings import settings
from services.gpt_advisor_service import GPTAdvisorService


class SelfReflectionEngine:
    def __init__(self) -> None:
        self.policy_store = AdaptivePolicyStore()
        self.gpt = GPTAdvisorService()

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def summarize(self, recent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        policy = self.policy_store.load()
        rows = [x for x in recent_results if bool(x.get("count_in_learning", x.get("learning_tier") == "effective"))]
        if not rows:
            policy["last_reflection"] = "目前尚無足夠有效學習單，AI 先沿用現有自主策略；探索單與同步補記不納入正式學習。"
            saved = self.policy_store.save(policy)
            return {"plain_text": policy["last_reflection"], "adaptations": [], "policy": saved, "gpt_status": self.gpt.available()}

        wins = sum(1 for x in rows if float(x.get("pnl_net", x.get("pnl", 0.0)) or 0.0) > 0)
        losses = sum(1 for x in rows if float(x.get("pnl_net", x.get("pnl", 0.0)) or 0.0) <= 0)
        avg_pnl = sum(float(x.get("pnl_net", x.get("pnl", 0.0)) or 0.0) for x in rows) / len(rows)
        avg_margin_return_pct = sum(float(x.get("pnl_on_margin_pct", 0.0) or 0.0) for x in rows) / len(rows)

        adaptations: List[Dict[str, Any]] = []
        if losses > wins:
            policy["entry_confidence_shift"] = self._clamp(float(policy.get("entry_confidence_shift", 0.0)) + 0.012, -0.08, 0.08)
            policy["size_multiplier_bias"] = self._clamp(float(policy.get("size_multiplier_bias", 1.0)) - 0.05, settings.adaptive_size_floor, settings.adaptive_size_ceiling)
            adaptations.append({"target": "entry_selectivity", "action": "slightly_raise"})
        else:
            policy["entry_confidence_shift"] = self._clamp(float(policy.get("entry_confidence_shift", 0.0)) - 0.008, -0.08, 0.08)
            policy["size_multiplier_bias"] = self._clamp(float(policy.get("size_multiplier_bias", 1.0)) + 0.03, settings.adaptive_size_floor, settings.adaptive_size_ceiling)
            adaptations.append({"target": "sample_growth", "action": "slightly_expand"})

        summary = (
            f"AI 近期自我摘要：有效學習單樣本數 {len(rows)}，勝 {wins}、負 {losses}，平均單筆淨損益 {avg_pnl:.4f}，"
            f"平均保證金報酬 {avg_margin_return_pct:.2f}% 。探索單與同步補記不納入正式學習。"
        )
        policy["last_reflection"] = summary
        saved = self.policy_store.save(policy)
        return {"plain_text": summary, "adaptations": adaptations, "policy": saved, "gpt_status": self.gpt.available()}

    def reflect(self, recent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.summarize(recent_results)
