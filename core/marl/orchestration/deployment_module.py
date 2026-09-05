"""Brain-managed deployment module for live batched HRL admission."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from core.marl.orchestration.agents import RuleBasedBrainAgent, SkillAgent
from core.marl.orchestration.protocol import (
    AgentRole,
    EventType,
    OrchestrationEvent,
    SystemObservation,
)
from core.marl.orchestration.skills import SkillContext, SkillRegistry, default_skill_registry


class DeploymentModule:
    """Route arrived requests through Brain, Deployment Agent and HRL Skill."""

    def __init__(
        self,
        planner: Any,
        *,
        brain: Optional[RuleBasedBrainAgent] = None,
        registry: Optional[SkillRegistry] = None,
    ) -> None:
        if not callable(getattr(planner, "plan_batch", None)):
            raise TypeError("deployment planner must expose plan_batch(requests)")
        self.planner = planner
        self.brain = brain or RuleBasedBrainAgent()
        self.registry = registry or default_skill_registry()
        self.agent = SkillAgent(AgentRole.DEPLOYMENT, self.registry)
        self.batch_count = 0
        self.request_count = 0
        self.command_counts: Counter[str] = Counter()
        self.skill_counts: Counter[str] = Counter()

    def plan_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        timestamp = max(float(row.get("arrival_time", 0.0)) for row in requests)
        planner_metadata = self.planner.metadata() if hasattr(self.planner, "metadata") else {}
        observation = SystemObservation(
            timestamp=timestamp,
            snapshot_version=f"deployment-batch-{self.batch_count}",
            metrics={
                "batch_size": len(requests),
                "planner_rejected_requests": int(
                    planner_metadata.get("rejected_requests", 0) or 0
                ),
            },
            queue_depth=len(requests),
        )
        commands = []
        for request in requests:
            request_id = int(request["id"])
            event = OrchestrationEvent(
                event_id=f"arrival-{request_id}",
                event_type=EventType.REQUEST_ARRIVAL,
                timestamp=timestamp,
                request=request,
                request_id=request_id,
                deadline_ms=request.get("delay_bound_ms"),
                metadata={"priority": request.get("priority", 1.0)},
            )
            command = self.brain.decide(event, observation)
            if AgentRole.DEPLOYMENT not in command.assigned_roles:
                raise RuntimeError("brain did not assign an arrival to deployment")
            commands.append(command)
            self.command_counts[command.command_type.value] += 1

        skill = self.registry.get("hrl_sft_mapping", role=AgentRole.DEPLOYMENT)
        proposals = skill.propose_batch(
            commands,
            SkillContext(hrl_planner=self.planner, requests=tuple(requests)),
        )
        plans: list[dict[str, Any]] = []
        for command, proposal in zip(commands, proposals):
            self.skill_counts[proposal.skill_name] += 1
            plan = dict(proposal.payload)
            plan.setdefault("request_id", command.target_request_id)
            plan.setdefault("accepted", proposal.feasible)
            plan.setdefault("reason", proposal.reason)
            plan.setdefault("orchestration", {}).update({
                "architecture": "brain_module_skill",
                "brain": type(self.brain).__name__,
                "module": type(self).__name__,
                "agent_role": AgentRole.DEPLOYMENT.value,
                "skill": proposal.skill_name,
                "command_id": command.command_id,
                "command_type": command.command_type.value,
                "snapshot_version": command.snapshot_version,
            })
            plans.append(plan)
        self.batch_count += 1
        self.request_count += len(requests)
        return plans

    def metadata(self) -> dict[str, Any]:
        return {
            "architecture": "brain_module_skill",
            "brain": type(self.brain).__name__,
            "module": type(self).__name__,
            "agent": type(self.agent).__name__,
            "role": AgentRole.DEPLOYMENT.value,
            "skills": list(self.agent.skill_names),
            "batch_count": self.batch_count,
            "request_count": self.request_count,
            "command_counts": dict(self.command_counts),
            "skill_counts": dict(self.skill_counts),
        }


class BrainManagedDeploymentPlanner:
    """Compatibility wrapper preserving the runtime planner API."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self.module = DeploymentModule(planner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.planner, name)

    def plan_batch(self, requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return self.module.plan_batch(requests)

    def plan_next(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.plan_batch([request])[0]

    def release(self, request_id: int) -> bool:
        return bool(self.planner.release(int(request_id)))

    def metadata(self) -> dict[str, Any]:
        base = dict(self.planner.metadata())
        base.update({
            "architecture": "brain_module_skill",
            "orchestration": self.module.metadata(),
        })
        return base

    def close(self) -> None:
        self.planner.close()


__all__ = ["BrainManagedDeploymentPlanner", "DeploymentModule"]
