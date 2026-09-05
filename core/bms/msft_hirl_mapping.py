"""BMS adapter for the MSFT-HIRL multicast SFT mapping policy.

This is additive: the existing HRL planner, checkpoint, migration pipeline,
graph encoder, action masks, imitation pretraining and Double-DQN logic stay
unchanged.  The adapter only exposes them through BMS contracts.
"""
from __future__ import annotations

from typing import Any, Mapping

from .base import BMSContext, BaseModule, BaseSkill, _check_layer
from .registry import register_module, register_skill


@register_skill("msft_hirl_mapping")
class MSFTHIRLMappingSkill(BaseSkill):
    """Wrap a planner exposing ``plan_next`` and/or ``plan_batch``."""

    name = "msft_hirl_mapping"

    def __init__(self, planner: Any | None = None) -> None:
        self.planner = planner

    @_check_layer("skill")
    def execute(self, context: BMSContext, params: dict) -> dict:
        planner = params.get("planner") or self.planner
        if planner is None:
            return self._result("rejected", None, {"reason": "planner_not_configured"})
        requests = params.get("requests")
        if requests is None and params.get("request") is not None:
            requests = [params["request"]]
        if requests is None:
            requests = list(context.active_requests)
        requests = [dict(row) for row in requests]
        if not requests:
            return self._result("rejected", [], {"reason": "no_requests"})
        try:
            if len(requests) > 1 and callable(getattr(planner, "plan_batch", None)):
                plans = planner.plan_batch(requests)
            elif callable(getattr(planner, "plan_next", None)):
                plans = [planner.plan_next(row) for row in requests]
            else:
                return self._result("rejected", None, {"reason": "planner_missing_plan_api"})
        except Exception as exc:
            return self._result("error", None, {"reason": str(exc), "exception": type(exc).__name__})
        plans = [dict(plan) for plan in plans]
        accepted = sum(bool(plan.get("accepted", False)) for plan in plans)
        return self._result(
            "success" if accepted else "rejected",
            plans if len(plans) != 1 else plans[0],
            {"planner": type(planner).__name__, "request_count": len(requests),
             "accepted_count": accepted, "algorithm": "MSFT-HIRL",
             "training": "behavior_cloning_plus_double_dqn"},
        )

    @staticmethod
    def _result(status: str, data: Any, metadata: Mapping[str, Any]) -> dict:
        return {"status": status, "data": data, "metadata": dict(metadata)}


@register_module("msft_hirl_deployment")
class MSFTHIRLDeploymentModule(BaseModule):
    """Deployment module that delegates mapping to the MSFT-HIRL skill."""

    def __init__(self, planner: Any | None = None, skills=None) -> None:
        self.mapping_skill = MSFTHIRLMappingSkill(planner)
        super().__init__(skills=list(skills or [self.mapping_skill]))

    @_check_layer("module")
    def run(self, context: BMSContext, instruction: dict) -> dict:
        result = self.mapping_skill.execute(context, dict(instruction or {}))
        return {"proposals": result.get("data"),
                "confidence": 1.0 if result.get("status") == "success" else 0.0,
                "evidence": {"skill": self.mapping_skill.name,
                             "status": result.get("status"),
                             "metadata": result.get("metadata", {})}}


__all__ = ["MSFTHIRLMappingSkill", "MSFTHIRLDeploymentModule"]
