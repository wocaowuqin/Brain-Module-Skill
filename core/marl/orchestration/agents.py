"""Central brain and role-specialist agents for SFT orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from core.marl.orchestration.protocol import (
    AgentRole,
    BrainCommand,
    CommandType,
    EventType,
    OrchestrationEvent,
    SkillProposal,
    SystemObservation,
)
from core.marl.orchestration.skills import SkillContext, SkillRegistry


@dataclass(frozen=True)
class BrainPolicyConfig:
    max_reconfiguration_delay_ms: float = 100.0
    default_priority: float = 1.0


class RuleBasedBrainAgent:
    """Event-driven first policy with the same command surface as a learned brain."""

    role = AgentRole.BRAIN

    def __init__(self, config: Optional[BrainPolicyConfig] = None) -> None:
        self.config = config or BrainPolicyConfig()
        self.decision_count = 0

    def decide(
        self, event: OrchestrationEvent, observation: SystemObservation
    ) -> BrainCommand:
        self.decision_count += 1
        command_type, roles, reason = self._route_event(event, observation)
        request_id = event.request_id
        if request_id is None and event.request is not None:
            value = event.request.get("id", event.request.get("request_id"))
            request_id = int(value) if value is not None else None
        if request_id is None and observation.risky_sfts:
            request_id = int(observation.risky_sfts[0]["req_id"])
        priority = float(event.metadata.get("priority", self.config.default_priority))
        return BrainCommand(
            command_id=f"brain-{self.decision_count}:{event.event_id}",
            event_id=event.event_id,
            command_type=command_type,
            target_request_id=request_id,
            assigned_roles=roles,
            snapshot_version=observation.snapshot_version,
            priority=priority,
            deadline_ms=event.deadline_ms,
            reason=reason,
        )

    def _route_event(
        self, event: OrchestrationEvent, observation: SystemObservation
    ) -> tuple[CommandType, tuple[AgentRole, ...], str]:
        if event.event_type == EventType.REQUEST_ARRIVAL:
            if event.request is None:
                return CommandType.REJECT, (), "arrival event has no request payload"
            return (
                CommandType.DEPLOY,
                (AgentRole.DEPLOYMENT,),
                "send the new request to the two-level HRL mapper",
            )
        if event.event_type in {EventType.LINK_OVERLOAD, EventType.EXECUTION_FAILURE}:
            return (
                CommandType.REROUTE,
                (AgentRole.REROUTE,),
                "repair the affected multicast branch",
            )
        if event.event_type == EventType.NODE_OVERLOAD:
            return (
                CommandType.MIGRATE,
                (AgentRole.MIGRATION,),
                "move a VNF away from the overloaded node",
            )

        metrics = observation.metrics
        node_hot = int(metrics.get("node_hotspots", 0) or 0) > 0
        link_hot = int(metrics.get("link_hotspots", 0) or 0) > 0
        if node_hot and link_hot:
            return (
                CommandType.JOINT_RECONFIG,
                (AgentRole.MIGRATION, AgentRole.REROUTE),
                "node and link risks coexist; ask both specialists for proposals",
            )
        if node_hot:
            return CommandType.MIGRATE, (AgentRole.MIGRATION,), "node hotspot detected"
        if link_hot or event.event_type == EventType.SLA_ALERT:
            return CommandType.REROUTE, (AgentRole.REROUTE,), "link or SLA risk detected"
        return CommandType.KEEP, (), "no actionable deployment or reconfiguration risk"


class SkillAgent:
    """A role specialist that may invoke only its explicitly assigned skills."""

    def __init__(
        self,
        role: AgentRole,
        registry: SkillRegistry,
        skill_names: Optional[Iterable[str]] = None,
    ) -> None:
        if role == AgentRole.BRAIN:
            raise ValueError("brain is not a skill agent")
        self.role = role
        self.registry = registry
        self.skill_names = tuple(skill_names or registry.names_for_role(role))
        if not self.skill_names:
            raise ValueError(f"role {role.value} has no registered skills")
        for name in self.skill_names:
            registry.get(name, role=role)

    def propose(
        self, command: BrainCommand, context: SkillContext
    ) -> tuple[SkillProposal, ...]:
        if self.role not in command.assigned_roles:
            raise PermissionError(
                f"command {command.command_id} is not assigned to {self.role.value}"
            )
        proposals = [
            self.registry.get(name, role=self.role).propose(command, context)
            for name in self.skill_names
        ]
        proposals.sort(key=lambda item: (item.feasible, item.score), reverse=True)
        return tuple(proposals)
