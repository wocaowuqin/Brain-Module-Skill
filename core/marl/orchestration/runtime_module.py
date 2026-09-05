"""Brain-managed dispatch for the real Mininet/Ryu reconfiguration pipeline."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional

from core.marl.orchestration.agents import RuleBasedBrainAgent, SkillAgent
from core.marl.orchestration.protocol import (
    AgentRole,
    CommandType,
    EventType,
    ExecutionResult,
    OrchestrationEvent,
    SystemObservation,
)
from core.marl.orchestration.skills import (
    SkillContext,
    SkillRegistry,
    runtime_reconfiguration_skill_registry,
)


class RuntimeReconfigurationModule:
    """Route live events while leaving mutation to the existing safe executor."""

    def __init__(
        self,
        *,
        brain: Optional[RuleBasedBrainAgent] = None,
        registry: Optional[SkillRegistry] = None,
    ) -> None:
        self.brain = brain or RuleBasedBrainAgent()
        self.registry = registry or runtime_reconfiguration_skill_registry()
        self.specialists = {
            role: SkillAgent(role, self.registry)
            for role in (AgentRole.MIGRATION, AgentRole.REROUTE)
        }
        self.dispatch_count = 0
        self.command_counts: Counter[str] = Counter()
        self.skill_counts: Counter[str] = Counter()
        self.outcome_counts: Counter[str] = Counter()
        self._pending: set[str] = set()

    def route_migration(
        self,
        *,
        timestamp: float,
        request_id: int,
        payload: Mapping[str, Any],
        metrics: Mapping[str, Any],
        snapshot_version: Optional[str] = None,
    ) -> ExecutionResult:
        return self._route(
            EventType.NODE_OVERLOAD,
            timestamp=timestamp,
            request_id=request_id,
            payload=payload,
            metrics=metrics,
            snapshot_version=snapshot_version,
        )

    def route_reroute(
        self,
        *,
        timestamp: float,
        request_id: int,
        payload: Mapping[str, Any],
        metrics: Mapping[str, Any],
        snapshot_version: Optional[str] = None,
        sla_alert: bool = False,
    ) -> ExecutionResult:
        return self._route(
            EventType.SLA_ALERT if sla_alert else EventType.LINK_OVERLOAD,
            timestamp=timestamp,
            request_id=request_id,
            payload=payload,
            metrics=metrics,
            snapshot_version=snapshot_version,
        )

    def _route(
        self,
        event_type: EventType,
        *,
        timestamp: float,
        request_id: int,
        payload: Mapping[str, Any],
        metrics: Mapping[str, Any],
        snapshot_version: Optional[str],
    ) -> ExecutionResult:
        cycle = self.dispatch_count
        observation = SystemObservation(
            timestamp=float(timestamp),
            snapshot_version=(
                str(snapshot_version)
                if snapshot_version is not None
                else f"runtime-reconfiguration-{cycle}"
            ),
            metrics=dict(metrics),
        )
        event = OrchestrationEvent(
            event_id=f"runtime-{event_type.value}-{request_id}-{cycle}",
            event_type=event_type,
            timestamp=float(timestamp),
            request_id=int(request_id),
            metadata={
                "priority": float(payload.get("priority", 1.0)),
                "policy": payload.get("policy", "external"),
            },
        )
        command = self.brain.decide(event, observation)
        proposals = tuple(
            proposal
            for role in command.assigned_roles
            for proposal in self.specialists[role].propose(
                command,
                SkillContext(runtime_payload=dict(payload)),
            )
        )
        selected = next((item for item in proposals if item.feasible), None)
        self.dispatch_count += 1
        self.command_counts[command.command_type.value] += 1
        for proposal in proposals:
            self.skill_counts[proposal.skill_name] += 1
        if selected is None:
            self.outcome_counts["dispatch_rejected"] += 1
            return ExecutionResult(
                event_id=event.event_id,
                command=command,
                proposals=proposals,
                selected="skip",
                success=False,
                applied=False,
                status_code="DISPATCH_REJECTED",
                reason=proposals[0].reason if proposals else command.reason,
            )
        self._pending.add(command.command_id)
        return ExecutionResult(
            event_id=event.event_id,
            command=command,
            proposals=proposals,
            selected=selected.action_type,
            success=True,
            applied=False,
            status_code="DISPATCHED",
            reason=selected.reason,
            payload=selected.payload,
        )

    def record_outcome(
        self,
        dispatch: ExecutionResult,
        *,
        success: bool,
        applied: bool,
        attempted: bool,
        reason: str,
    ) -> dict[str, Any]:
        command_id = dispatch.command.command_id
        if command_id not in self._pending:
            raise ValueError(f"unknown or completed runtime command: {command_id}")
        self._pending.remove(command_id)
        status = (
            "applied"
            if success and applied
            else "failed"
            if attempted
            else "skipped"
        )
        self.outcome_counts[status] += 1
        return {
            "command_id": command_id,
            "success": bool(success),
            "applied": bool(applied),
            "attempted": bool(attempted),
            "status": status,
            "reason": str(reason),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "architecture": "brain_module_skill",
            "brain": type(self.brain).__name__,
            "module": type(self).__name__,
            "specialists": {
                role.value: {
                    "agent": type(agent).__name__,
                    "skills": list(agent.skill_names),
                }
                for role, agent in self.specialists.items()
            },
            "dispatch_count": self.dispatch_count,
            "command_counts": dict(self.command_counts),
            "skill_counts": dict(self.skill_counts),
            "outcome_counts": dict(self.outcome_counts),
            "pending_commands": len(self._pending),
            "mutation_boundary": "existing_runtime_atomic_ledger_and_ryu_executor",
        }


__all__ = ["RuntimeReconfigurationModule"]
