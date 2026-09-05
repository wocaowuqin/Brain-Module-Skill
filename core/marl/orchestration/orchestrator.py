"""Runnable central-brain orchestration over HRL, migration and rerouting."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Mapping, Optional

from core.marl.orchestration.agents import RuleBasedBrainAgent, SkillAgent
from core.marl.orchestration.protocol import (
    AgentRole,
    CommandType,
    ExecutionResult,
    OrchestrationEvent,
    SkillProposal,
    SystemObservation,
)
from core.marl.orchestration.skills import (
    SkillContext,
    SkillRegistry,
    default_skill_registry,
)
from core.marl.role_agents import RoleDecision
from core.marl.role_coordinator import RoleCoordinator
from envs.modules.reconfiguration_manager import ReconfigurationManager


DeploymentExecutor = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class MultiAgentSFTOrchestrator:
    """Coordinate specialists while keeping all resource mutation centralized."""

    def __init__(
        self,
        env_or_resource_mgr,
        *,
        hrl_planner: Any = None,
        deployment_executor: Optional[DeploymentExecutor] = None,
        brain: Optional[RuleBasedBrainAgent] = None,
        registry: Optional[SkillRegistry] = None,
        top_k_risks: int = 3,
        node_util_threshold: float = 0.80,
        link_util_threshold: float = 0.80,
    ) -> None:
        self.manager = (
            env_or_resource_mgr
            if isinstance(env_or_resource_mgr, ReconfigurationManager)
            else ReconfigurationManager(
                env_or_resource_mgr,
                node_util_threshold=node_util_threshold,
                link_util_threshold=link_util_threshold,
            )
        )
        self.hrl_planner = hrl_planner
        self.deployment_executor = deployment_executor
        self.brain = brain or RuleBasedBrainAgent()
        self.registry = registry or default_skill_registry()
        self.top_k_risks = max(1, int(top_k_risks))
        self.specialists: Dict[AgentRole, SkillAgent] = {
            role: SkillAgent(role, self.registry)
            for role in (AgentRole.DEPLOYMENT, AgentRole.MIGRATION, AgentRole.REROUTE)
        }
        # This component arbitrates already-generated reconfiguration proposals.
        # It is not the brain policy and remains the exact safe execution boundary.
        self.reconfiguration_arbiter = RoleCoordinator()
        self.cycle = 0

    def observe(self, timestamp: float) -> SystemObservation:
        metrics = self.manager.metrics_snapshot()
        risks = tuple(
            asdict(item) for item in self.manager.select_topk_risky_sfts(self.top_k_risks)
        )
        return SystemObservation(
            timestamp=float(timestamp),
            snapshot_version=f"orchestration-cycle-{self.cycle}",
            metrics=metrics,
            risky_sfts=risks,
        )

    def handle(self, event: OrchestrationEvent, *, apply: bool = True) -> ExecutionResult:
        observation = self.observe(event.timestamp)
        command = self.brain.decide(event, observation)
        if command.command_type in {CommandType.KEEP, CommandType.REJECT}:
            return ExecutionResult(
                event_id=event.event_id,
                command=command,
                proposals=(),
                selected=command.command_type.value,
                success=command.command_type == CommandType.KEEP,
                applied=False,
                status_code=command.command_type.value.upper(),
                reason=command.reason,
            )

        context = SkillContext(
            manager=self.manager,
            hrl_planner=self.hrl_planner,
            request=event.request,
        )
        proposals = tuple(
            proposal
            for role in command.assigned_roles
            for proposal in self.specialists[role].propose(command, context)
        )
        if command.command_type == CommandType.DEPLOY:
            result = self._handle_deployment(event, command, proposals, apply)
        else:
            result = self._handle_reconfiguration(event, command, proposals, apply)
        if result.applied:
            self.cycle += 1
        return result

    def release(self, request_id: int) -> bool:
        if self.hrl_planner is None or not hasattr(self.hrl_planner, "release"):
            return False
        return bool(self.hrl_planner.release(int(request_id)))

    def metadata(self) -> Dict[str, Any]:
        return {
            "brain": type(self.brain).__name__,
            "hrl_planner": type(self.hrl_planner).__name__ if self.hrl_planner else None,
            "deployment_executor": bool(self.deployment_executor),
            "skills": self.registry.metadata(),
            "specialists": {
                role.value: list(agent.skill_names)
                for role, agent in self.specialists.items()
            },
            "joint_reconfiguration": "single-safe-action-arbitration",
        }

    def _handle_deployment(
        self, event, command, proposals: tuple[SkillProposal, ...], apply: bool
    ) -> ExecutionResult:
        proposal = next((item for item in proposals if item.feasible), None)
        if proposal is None:
            reason = proposals[0].reason if proposals else "deployment agent returned no proposal"
            return ExecutionResult(
                event_id=event.event_id,
                command=command,
                proposals=proposals,
                selected="reject",
                success=False,
                applied=False,
                status_code="HRL_REJECTED",
                reason=reason,
            )
        if not apply or self.deployment_executor is None:
            return ExecutionResult(
                event_id=event.event_id,
                command=command,
                proposals=proposals,
                selected=proposal.action_type,
                success=True,
                applied=False,
                status_code="PLANNED_ONLY",
                reason=(
                    "HRL plan is ready; no deployment executor was configured"
                    if self.deployment_executor is None
                    else "plan-only mode"
                ),
                payload=proposal.payload,
            )
        raw = dict(self.deployment_executor(proposal.payload))
        success = bool(raw.get("success", raw.get("ok", False)))
        return ExecutionResult(
            event_id=event.event_id,
            command=command,
            proposals=proposals,
            selected=proposal.action_type,
            success=success,
            applied=success,
            status_code=str(raw.get("status_code", "APPLIED" if success else "FAILED")),
            reason=str(raw.get("reason", "deployment executor completed")),
            payload=raw,
        )

    def _handle_reconfiguration(
        self, event, command, proposals: tuple[SkillProposal, ...], apply: bool
    ) -> ExecutionResult:
        migration = self._role_decision(proposals, AgentRole.MIGRATION)
        reroute = self._role_decision(proposals, AgentRole.REROUTE)
        coordination = self.reconfiguration_arbiter.coordinate(
            self.manager,
            req_id=command.target_request_id,
            migration_decision=migration,
            reroute_decision=reroute,
            apply=apply,
        )
        payload = coordination.to_dict()
        selected = coordination.selected
        applied = bool(apply and coordination.executed)
        success = selected not in {"failed", "noop"}
        status = "APPLIED" if applied and success else (
            "PLANNED_ONLY" if success else selected.upper()
        )
        return ExecutionResult(
            event_id=event.event_id,
            command=command,
            proposals=proposals,
            selected=selected,
            success=success,
            applied=applied,
            status_code=status,
            reason=coordination.reason,
            payload=payload,
        )

    @staticmethod
    def _role_decision(
        proposals: tuple[SkillProposal, ...], role: AgentRole
    ) -> RoleDecision:
        proposal = next(
            (item for item in proposals if item.role == role and item.feasible), None
        )
        if proposal is None:
            return RoleDecision(
                role=role.value,
                action=0,
                label=f"no_{role.value}",
                reason="no feasible specialist proposal",
            )
        return RoleDecision(
            role=role.value,
            action=1,
            label=proposal.action_type,
            req_id=proposal.request_id,
            score=proposal.score,
            proposal=dict(proposal.payload),
            reason=proposal.reason,
        )
