"""Rule-backed role agents for the first runnable role-MARL prototype.

These agents intentionally expose the same shape that trainable agents will use
later: observe -> select_action -> return a small discrete action and metadata.
For now the policy is deterministic and delegates concrete migration/reroute
target generation to ReconfigurationManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RoleDecision:
    role: str
    action: int
    label: str
    req_id: Optional[int] = None
    score: float = 0.0
    proposal: Optional[Dict[str, Any]] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "action": self.action,
            "label": self.label,
            "req_id": self.req_id,
            "score": self.score,
            "proposal": self.proposal,
            "reason": self.reason,
        }


class SFTSelectionAgent:
    """Select one target SFT from the Top-K risky SFT list."""

    # 0: no-op, 1..K: choose rank action-1
    def __init__(self, top_k: int = 3, min_risk_score: float = 0.0):
        self.top_k = int(top_k)
        self.min_risk_score = float(min_risk_score)

    def observe(self, manager) -> Dict[str, Any]:
        risks = manager.select_topk_risky_sfts(self.top_k)
        return {
            "risks": risks,
            "metrics": manager.metrics_snapshot(),
        }

    def select_action(self, obs: Dict[str, Any]) -> RoleDecision:
        risks = obs.get("risks", [])
        if not risks or risks[0].score <= self.min_risk_score:
            return RoleDecision(
                role="sft_selector",
                action=0,
                label="no_target",
                reason="no SFT exceeds the risk threshold",
            )
        target = risks[0]
        return RoleDecision(
            role="sft_selector",
            action=1,
            label="select_top1",
            req_id=target.req_id,
            score=float(target.score),
            reason="select the highest-risk active SFT",
        )


class VNFMigrationAgent:
    """Decide whether to propose VNF migration for the selected SFT."""

    # 0: no-op, 1: migrate hottest-node VNF
    def __init__(self, min_gain: float = 0.0):
        self.min_gain = float(min_gain)

    def observe(self, manager, req_id: Optional[int]) -> Dict[str, Any]:
        action = manager.plan_greedy_migrate(req_id=req_id) if req_id is not None else None
        return {"planned_action": action}

    def select_action(self, obs: Dict[str, Any]) -> RoleDecision:
        action = obs.get("planned_action")
        if action is None or action.action_type == "noop" or action.estimated_gain <= self.min_gain:
            return RoleDecision(
                role="vnf_migration",
                action=0,
                label="no_migration",
                req_id=getattr(action, "req_id", None),
                reason=getattr(action, "reason", "no feasible migration"),
            )
        return RoleDecision(
            role="vnf_migration",
            action=1,
            label="migrate_hottest_vnf",
            req_id=action.req_id,
            score=float(action.estimated_gain),
            proposal=action.to_dict(),
            reason=action.reason,
        )


class TreeRerouteAgent:
    """Decide whether to propose local tree-edge rerouting."""

    # 0: no-op, 1: reroute hottest tree edge
    def __init__(self, min_gain: float = 0.0):
        self.min_gain = float(min_gain)

    def observe(self, manager, req_id: Optional[int]) -> Dict[str, Any]:
        action = manager.plan_greedy_reroute(req_id=req_id) if req_id is not None else None
        return {"planned_action": action}

    def select_action(self, obs: Dict[str, Any]) -> RoleDecision:
        action = obs.get("planned_action")
        if action is None or action.action_type == "noop" or action.estimated_gain <= self.min_gain:
            return RoleDecision(
                role="tree_reroute",
                action=0,
                label="no_reroute",
                req_id=getattr(action, "req_id", None),
                reason=getattr(action, "reason", "no feasible reroute"),
            )
        return RoleDecision(
            role="tree_reroute",
            action=1,
            label="reroute_hottest_edge",
            req_id=action.req_id,
            score=float(action.estimated_gain),
            proposal=action.to_dict(),
            reason=action.reason,
        )


def decisions_to_dict(decisions: List[RoleDecision]) -> List[Dict[str, Any]]:
    return [decision.to_dict() for decision in decisions]

