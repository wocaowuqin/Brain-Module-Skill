"""Skill adapters around the existing HRL and safe reconfiguration planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence

from core.marl.orchestration.protocol import (
    AgentRole,
    BrainCommand,
    SkillProposal,
)


@dataclass(frozen=True)
class SkillContext:
    manager: Any = None
    hrl_planner: Any = None
    request: Optional[Mapping[str, Any]] = None
    requests: Sequence[Mapping[str, Any]] = ()
    runtime_payload: Mapping[str, Any] = None


class AgentSkill(Protocol):
    name: str
    role: AgentRole

    def propose(self, command: BrainCommand, context: SkillContext) -> SkillProposal:
        ...


class SkillRegistry:
    """Role-scoped registry; agents cannot invoke another role's skills."""

    def __init__(self) -> None:
        self._skills: Dict[str, AgentSkill] = {}

    def register(self, skill: AgentSkill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str, *, role: Optional[AgentRole] = None) -> AgentSkill:
        try:
            skill = self._skills[name]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {name}") from exc
        if role is not None and skill.role != role:
            raise PermissionError(
                f"role {role.value} cannot invoke {skill.role.value} skill {name}"
            )
        return skill

    def names_for_role(self, role: AgentRole) -> tuple[str, ...]:
        return tuple(
            name for name, skill in self._skills.items() if skill.role == role
        )

    def metadata(self) -> Dict[str, str]:
        return {name: skill.role.value for name, skill in self._skills.items()}


class HRLDeploymentSkill:
    """Use the project's two-level HRL planner for initial SFT mapping."""

    name = "hrl_sft_mapping"
    role = AgentRole.DEPLOYMENT

    def propose(self, command: BrainCommand, context: SkillContext) -> SkillProposal:
        if context.hrl_planner is None:
            return self._reject(command, "HRL planner is not configured")
        if context.request is None:
            return self._reject(command, "deployment command has no request")
        plan = context.hrl_planner.plan_next(dict(context.request))
        accepted = bool(plan.get("accepted", False))
        return SkillProposal(
            proposal_id=f"{command.command_id}:{self.name}",
            command_id=command.command_id,
            role=self.role,
            skill_name=self.name,
            request_id=_request_id(context.request),
            action_type="deploy_sft" if accepted else "reject",
            score=1.0 if accepted else 0.0,
            feasible=accepted,
            snapshot_version=command.snapshot_version,
            payload=plan,
            reason="HRL produced a complete SFT plan" if accepted else str(
                plan.get("reason", "HRL rejected the request")
            ),
            metadata={"planner": type(context.hrl_planner).__name__},
        )

    def propose_batch(
        self,
        commands: Sequence[BrainCommand],
        context: SkillContext,
    ) -> tuple[SkillProposal, ...]:
        """Invoke one true micro-batch planner call for deployment requests."""

        if context.hrl_planner is None:
            return tuple(
                self._reject(command, "HRL planner is not configured")
                for command in commands
            )
        requests = tuple(context.requests)
        if len(commands) != len(requests):
            raise ValueError("deployment commands and requests must have equal length")
        if not callable(getattr(context.hrl_planner, "plan_batch", None)):
            return tuple(
                self.propose(
                    command,
                    SkillContext(
                        manager=context.manager,
                        hrl_planner=context.hrl_planner,
                        request=request,
                    ),
                )
                for command, request in zip(commands, requests)
            )
        plans = context.hrl_planner.plan_batch([dict(row) for row in requests])
        if len(plans) != len(requests):
            raise RuntimeError("HRL batch planner returned the wrong result count")
        proposals = []
        for command, request, raw_plan in zip(commands, requests, plans):
            plan = dict(raw_plan)
            request_id = _request_id(request)
            plan_id = plan.get("request_id", request_id)
            if plan_id is not None and request_id is not None and int(plan_id) != request_id:
                raise RuntimeError("HRL batch planner changed request ordering")
            accepted = bool(plan.get("accepted", False))
            proposals.append(SkillProposal(
                proposal_id=f"{command.command_id}:{self.name}",
                command_id=command.command_id,
                role=self.role,
                skill_name=self.name,
                request_id=request_id,
                action_type="deploy_sft" if accepted else "reject",
                score=1.0 if accepted else 0.0,
                feasible=accepted,
                snapshot_version=command.snapshot_version,
                payload=plan,
                reason=(
                    "HRL batch produced a complete SFT plan"
                    if accepted
                    else str(plan.get("reason", "HRL batch rejected the request"))
                ),
                metadata={
                    "planner": type(context.hrl_planner).__name__,
                    "batch_size": len(requests),
                },
            ))
        return tuple(proposals)

    def _reject(self, command: BrainCommand, reason: str) -> SkillProposal:
        return SkillProposal(
            proposal_id=f"{command.command_id}:{self.name}",
            command_id=command.command_id,
            role=self.role,
            skill_name=self.name,
            request_id=command.target_request_id,
            action_type="reject",
            score=0.0,
            feasible=False,
            snapshot_version=command.snapshot_version,
            reason=reason,
        )


class VNFMigrationPlanningSkill:
    name = "plan_vnf_migration"
    role = AgentRole.MIGRATION

    def propose(self, command: BrainCommand, context: SkillContext) -> SkillProposal:
        action = context.manager.plan_greedy_migrate(command.target_request_id)
        payload = action.to_dict()
        feasible = action.action_type == "migrate_vnf" and bool(action.target)
        return SkillProposal(
            proposal_id=f"{command.command_id}:{self.name}",
            command_id=command.command_id,
            role=self.role,
            skill_name=self.name,
            request_id=action.req_id,
            action_type=action.action_type,
            score=float(action.estimated_gain),
            feasible=feasible,
            snapshot_version=command.snapshot_version,
            payload=payload,
            reason=action.reason,
            metadata={"executor": "apply_migration_proposal"},
        )


class TreeReroutePlanningSkill:
    name = "plan_tree_reroute"
    role = AgentRole.REROUTE

    def propose(self, command: BrainCommand, context: SkillContext) -> SkillProposal:
        action, diagnostics = context.manager.plan_greedy_reroute_with_diagnostics(
            command.target_request_id
        )
        payload = action.to_dict()
        feasible = action.action_type == "reroute_edge" and bool(action.target)
        return SkillProposal(
            proposal_id=f"{command.command_id}:{self.name}",
            command_id=command.command_id,
            role=self.role,
            skill_name=self.name,
            request_id=action.req_id,
            action_type=action.action_type,
            score=float(action.estimated_gain),
            feasible=feasible,
            snapshot_version=command.snapshot_version,
            payload=payload,
            reason=action.reason,
            metadata={
                "executor": "apply_reroute_proposal",
                "diagnostics": diagnostics,
            },
        )


class RuntimeMigrationExecutionSkill:
    """Authorize the existing live make-before-break migration pipeline."""

    name = "runtime_vnf_migration"
    role = AgentRole.MIGRATION

    def propose(self, command: BrainCommand, context: SkillContext) -> SkillProposal:
        payload = dict(context.runtime_payload or {})
        active = bool(payload.get("request_active", True))
        feasible = command.target_request_id is not None and active
        return SkillProposal(
            proposal_id=f"{command.command_id}:{self.name}",
            command_id=command.command_id,
            role=self.role,
            skill_name=self.name,
            request_id=command.target_request_id,
            action_type="execute_vnf_migration" if feasible else "skip",
            score=float(payload.get("estimated_gain", command.priority)),
            feasible=feasible,
            snapshot_version=command.snapshot_version,
            payload=payload,
            reason=(
                "dispatch to migration WQMIX and make-before-break executor"
                if feasible
                else "migration request is not active"
            ),
            metadata={
                "planner": str(payload.get("policy", "external")),
                "mutation_boundary": "shared_runtime_ledger_and_ryu",
            },
        )


class RuntimeTreeRerouteExecutionSkill:
    """Authorize the existing live gated tree-reroute pipeline."""

    name = "runtime_tree_reroute"
    role = AgentRole.REROUTE

    def propose(self, command: BrainCommand, context: SkillContext) -> SkillProposal:
        payload = dict(context.runtime_payload or {})
        active = bool(payload.get("request_active", True))
        feasible = command.target_request_id is not None and active
        return SkillProposal(
            proposal_id=f"{command.command_id}:{self.name}",
            command_id=command.command_id,
            role=self.role,
            skill_name=self.name,
            request_id=command.target_request_id,
            action_type="execute_tree_reroute" if feasible else "skip",
            score=float(payload.get("estimated_gain", command.priority)),
            feasible=feasible,
            snapshot_version=command.snapshot_version,
            payload=payload,
            reason=(
                "dispatch to strict-gate tree-reroute executor"
                if feasible
                else "reroute request is not active"
            ),
            metadata={
                "planner": str(payload.get("policy", "external")),
                "mutation_boundary": "shared_runtime_ledger_and_ryu",
            },
        )


def default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in (
        HRLDeploymentSkill(),
        VNFMigrationPlanningSkill(),
        TreeReroutePlanningSkill(),
    ):
        registry.register(skill)
    return registry


def runtime_reconfiguration_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(RuntimeMigrationExecutionSkill())
    registry.register(RuntimeTreeRerouteExecutionSkill())
    return registry


def _request_id(request: Mapping[str, Any]) -> Optional[int]:
    value = request.get("id", request.get("request_id"))
    return int(value) if value is not None else None
