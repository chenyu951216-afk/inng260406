import json
import logging
from typing import Any, Dict

from config.settings import settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class GPTAdvisorService:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = OpenAI(api_key=settings.openai_api_key) if (OpenAI and settings.openai_api_key) else None

    def available(self) -> bool:
        return bool(self.client and settings.enable_gpt_reflection and settings.enable_daily_gpt_review)

    def live_available(self) -> bool:
        return bool(self.client and settings.enable_gpt_live_control)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}

    def _build_request_kwargs(self, instructions: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": settings.gpt_model,
            "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False),
            "timeout": settings.gpt_timeout_sec,
        }
        # Some models / endpoints reject reasoning.effort. Only send it for known reasoning-capable models.
        model_name = str(settings.gpt_model or "").lower()
        supports_reasoning_effort = any(
            token in model_name for token in ("o1", "o3", "o4", "gpt-5")
        )
        if supports_reasoning_effort and getattr(settings, "gpt_reasoning_effort", None):
            kwargs["reasoning"] = {"effort": settings.gpt_reasoning_effort}
        return kwargs

    def _call_json(self, instructions: str, payload: Dict[str, Any], *, live: bool = False) -> Dict[str, Any]:
        if live:
            if not self.live_available():
                return {"enabled": False, "reason": "gpt_live_not_configured"}
        else:
            if not self.available():
                return {"enabled": False, "reason": "gpt_not_configured"}
        try:
            response = self.client.responses.create(**self._build_request_kwargs(instructions, payload))
            text = getattr(response, "output_text", "")
            parsed = self._extract_json(text)
            if not parsed:
                return {"enabled": True, "reason": "gpt_empty_or_unparseable", "raw": text}
            parsed["enabled"] = True
            return parsed
        except Exception as exc:  # pragma: no cover
            self.logger.warning("GPT advisor call failed: %s", exc)
            return {"enabled": True, "reason": f"gpt_call_failed:{exc}"}

    def advise_live_trade_candidate(self, candidate: Dict[str, Any], policy: Dict[str, Any], account_summary: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "goal": "You are a live crypto futures trade co-pilot. Templates are assistive only. Decide whether to enter now, adjust leverage, and adjust TP/SL. Keep risk controlled. Output strict JSON only.",
            "candidate": {
                "symbol": candidate.get("symbol"),
                "side": candidate.get("side"),
                "entry_decision": candidate.get("entry_decision", {}),
                "leverage_decision": candidate.get("leverage_decision", {}),
                "sizing_decision": candidate.get("sizing_decision", {}),
                "features": candidate.get("features", {}),
                "market_snapshot": candidate.get("market_snapshot", {}),
            },
            "policy": policy,
            "account_summary": {
                "equity": account_summary.get("equity", 0),
                "available_equity": account_summary.get("available_equity", account_summary.get("available", 0)),
                "used_margin": account_summary.get("used_margin", 0),
            },
            "required_json_schema": {
                "action": "enter|wait|hold",
                "reason": "string",
                "leverage": "integer",
                "margin_pct": "float",
                "tp_bias": "float between -0.5 and 0.5",
                "sl_bias": "float between -0.5 and 0.5",
                "market_structure": "breakout|trend|range|reversal|unknown",
                "confidence_boost": "float between -0.15 and 0.15"
            }
        }
        return self._call_json(
            "You are directly participating in live trade entry, leverage, and TP/SL adjustment. Output strict JSON only.",
            payload,
            live=True,
        )

    def review_daily_trades(self, digest: Dict[str, Any], current_policy: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "goal": "Review one day of REAL crypto futures trades. Templates are assistive only. Focus on entry logic, position management, protection orders, exit style, and trade lifecycle handling. Do not propose hard symbol bans or same-direction caps.",
            "digest": digest,
            "current_policy": current_policy,
            "required_json_schema": {
                "top_findings": ["string"],
                "recommendations": [
                    {
                        "area": "entry|position_management|protection|exit|trade_lifecycle",
                        "change": "string",
                        "direction": "increase|decrease|tighten|loosen|hold|rebalance",
                        "confidence": "0..1",
                        "reason": "string"
                    }
                ],
                "bot_questions": ["string"],
                "summary": "string"
            },
        }
        return self._call_json(
            "You are reviewing a live trading bot's real daily trading data. Output strict JSON only.",
            payload,
        )

    def discuss_disagreement(self, digest: Dict[str, Any], review: Dict[str, Any], objection: Dict[str, Any], current_policy: Dict[str, Any], round_index: int) -> Dict[str, Any]:
        payload = {
            "goal": "Resolve disagreement between the local trading bot and GPT review.",
            "round": round_index,
            "digest": digest,
            "prior_review": review,
            "local_objection": objection,
            "current_policy": current_policy,
            "required_json_schema": {
                "updated_recommendations": [
                    {
                        "area": "entry|position_management|protection|exit|trade_lifecycle",
                        "change": "string",
                        "direction": "increase|decrease|tighten|loosen|hold|rebalance",
                        "confidence": "0..1",
                        "reason": "string"
                    }
                ],
                "consensus_summary": "string",
                "needs_more_discussion": "boolean"
            },
        }
        return self._call_json(
            "You are in a structured deliberation with an autonomous trading bot. Respect the bot's evidence. Output strict JSON only.",
            payload,
        )

    def recommend_adjustments(self, digest: Dict[str, Any], consensus: Dict[str, Any], current_policy: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "goal": "Translate the agreed direction into controlled trading adjustments.",
            "digest": digest,
            "consensus": consensus,
            "current_policy": current_policy,
            "required_json_schema": {
                "entry_confidence_shift": "float between -0.08 and 0.08",
                "size_multiplier_bias": "float between 0.35 and 2.4",
                "leverage_bias": "float between 0.7 and 1.4",
                "breakout_bias": "float between -0.2 and 0.2",
                "trend_follow_bias": "float between -0.2 and 0.2",
                "entry_aggression": "float between -0.2 and 0.2",
                "breakout_tolerance": "float between -0.2 and 0.2",
                "pullback_preference": "float between -0.2 and 0.2",
                "scale_in_aggression": "float between -0.2 and 0.35",
                "scale_out_aggression": "float between -0.2 and 0.35",
                "reentry_after_partial": "float between -0.2 and 0.25",
                "protection_refresh_bias": "float between -0.2 and 0.35",
                "protection_profile": "tight|balanced|wide",
                "exit_style": "fast|balanced|runner",
                "position_management_profile": "defensive|balanced|press_winners",
                "initial_stop_loss_atr": "float between 1.0 and 3.5",
                "initial_take_profit_atr": "float between 1.2 and 5.0",
                "break_even_trigger_rr": "float between 0.6 and 1.8",
                "trailing_activation_rr": "float between 0.8 and 2.5",
                "trailing_buffer_atr": "float between 0.4 and 1.6",
                "partial_take_profit_rr": "float between 1.0 and 3.0",
                "break_even_lock_ratio": "float between 0.05 and 0.5",
                "trailing_step_rr": "float between 0.2 and 0.8",
                "tp1_fraction": "float between 0.1 and 0.5",
                "tp2_fraction": "float between 0.15 and 0.7",
                "summary": "string"
            },
        }
        return self._call_json(
            "You are converting a live-trading discussion consensus into careful bot parameter updates. Output strict JSON only.",
            payload,
        )

    def status(self) -> Dict[str, Any]:
        return {
            "configured": bool(settings.openai_api_key),
            "available": self.available(),
            "live_available": self.live_available(),
            "model": settings.gpt_model,
        }
