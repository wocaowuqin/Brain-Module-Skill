"""Rule coordinator for role-based SFT reconfiguration agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.marl.role_agents import RoleDecision


@dataclass
class CoordinationResult:
    selected: str
    req_id: Optional[int]
    executed: List[Dict[str, Any]]
    skipped: List[Dict[str, Any]]
    reward_proxy: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected,
            "req_id": self.req_id,
            "executed": self.executed,
            "skipped": self.skipped,
            "reward_proxy": self.reward_proxy,
            "reason": self.reason,
        }


class RoleCoordinator:
    """Resolve migration/reroute proposals from fixed role agents."""

    def __init__(
        self,
        migration_cost: float = 0.15,
        reroute_cost_per_extra_edge: float = 0.08,
        delay_penalty_weight: float = 0.05,
        execute_both_margin: float = 0.10,
    ):
        self.migration_cost = float(migration_cost)
        self.reroute_cost_per_extra_edge = float(reroute_cost_per_extra_edge)
        self.delay_penalty_weight = float(delay_penalty_weight)
        self.execute_both_margin = float(execute_both_margin)

    def coordinate(self, manager, req_id: Optional[int],
                   migration_decision: RoleDecision,
                   reroute_decision: RoleDecision,
                   apply: bool = True) -> CoordinationResult:
        if req_id is None:
            return CoordinationResult(
                selected="noop",
                req_id=None,
                executed=[],
                skipped=[migration_decision.to_dict(), reroute_decision.to_dict()],
                reward_proxy=0.0,
                reason="no selected SFT",
            )

        rec = manager.rm.request_table.get(req_id)
        delay_before = manager.estimate_sft_delay(rec) if rec is not None else 0.0

        migration_net = self._migration_net_score(migration_decision)
        reroute_net = self._reroute_net_score(reroute_decision)

        executed: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        selected = "noop"
        reason = "no positive proposal"

        can_migrate = migration_decision.action != 0 and migration_net > 0
        can_reroute = reroute_decision.action != 0 and reroute_net > 0

        if can_migrate and can_reroute:
            if abs(migration_net - reroute_net) <= self.execute_both_margin:
                # The proposals share one resource fingerprint. Applying the
                # first invalidates the second, so execute only the stronger
                # action until a joint migration+reroute transaction exists.
                if migration_net >= reroute_net:
                    selected = "migrate"
                    reason = "migration wins the atomic single-action tie break"
                    if apply:
                        executed.append(manager.apply_migration_proposal(migration_decision.proposal).to_dict())
                    skipped.append(reroute_decision.to_dict())
                else:
                    selected = "reroute"
                    reason = "reroute wins the atomic single-action tie break"
                    if apply:
                        executed.append(manager.apply_reroute_proposal(reroute_decision.proposal).to_dict())
                    skipped.append(migration_decision.to_dict())
            elif migration_net > reroute_net:
                selected = "migrate"
                reason = "migration has larger net value"
                if apply:
                    executed.append(manager.apply_migration_proposal(migration_decision.proposal).to_dict())
                skipped.append(reroute_decision.to_dict())
            else:
                selected = "reroute"
                reason = "reroute has larger net value"
                if apply:
                    executed.append(manager.apply_reroute_proposal(reroute_decision.proposal).to_dict())
                skipped.append(migration_decision.to_dict())
        elif can_migrate:
            selected = "migrate"
            reason = "only migration has positive net value"
            if apply:
                executed.append(manager.apply_migration_proposal(migration_decision.proposal).to_dict())
            skipped.append(reroute_decision.to_dict())
        elif can_reroute:
            selected = "reroute"
            reason = "only reroute has positive net value"
            if apply:
                executed.append(manager.apply_reroute_proposal(reroute_decision.proposal).to_dict())
            skipped.append(migration_decision.to_dict())
        else:
            skipped.extend([migration_decision.to_dict(), reroute_decision.to_dict()])

        rec_after = manager.rm.request_table.get(req_id)
        delay_after = manager.estimate_sft_delay(rec_after) if rec_after is not None else delay_before
        delay_delta = delay_after - delay_before
        reward_proxy = max(migration_net, 0.0) + max(reroute_net, 0.0)
        reward_proxy -= self.delay_penalty_weight * max(0.0, delay_delta)

        if apply and executed:
            successful = [item for item in executed if item.get("success")]
            if not successful:
                selected = "failed"
                reason = "selected action executor failed"

        return CoordinationResult(
            selected=selected,
            req_id=req_id,
            executed=executed,
            skipped=skipped,
            reward_proxy=float(reward_proxy),
            reason=reason,
        )

    def _migration_net_score(self, decision: RoleDecision) -> float:
        if decision.action == 0:
            return 0.0
        gain = float((decision.proposal or {}).get("estimated_gain", 0.0))
        return gain - self.migration_cost

    def _reroute_net_score(self, decision: RoleDecision) -> float:
        if decision.action == 0:
            return 0.0
        extra_edges = 0
        try:
            target = (decision.proposal or {}).get("target", {})
            new_path = target.get("new_path", [])
            extra_edges = int((target.get("safety") or {}).get("extra_edges", max(0, len(new_path) - 2)))
        except Exception:
            extra_edges = 0
        gain = float((decision.proposal or {}).get("estimated_gain", 0.0))
        return gain - self.reroute_cost_per_extra_edge * extra_edges
