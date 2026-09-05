"""Typed messages exchanged by the hierarchical multi-agent orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


class AgentRole(str, Enum):
    BRAIN = "brain"
    DEPLOYMENT = "deployment"
    MIGRATION = "migration"
    REROUTE = "reroute"


class EventType(str, Enum):
    REQUEST_ARRIVAL = "request_arrival"
    PERIODIC_REVIEW = "periodic_review"
    SLA_ALERT = "sla_alert"
    NODE_OVERLOAD = "node_overload"
    LINK_OVERLOAD = "link_overload"
    EXECUTION_FAILURE = "execution_failure"


class CommandType(str, Enum):
    KEEP = "keep"
    DEPLOY = "deploy"
    MIGRATE = "migrate"
    REROUTE = "reroute"
    JOINT_RECONFIG = "joint_reconfig"
    REJECT = "reject"


@dataclass(frozen=True)
class OrchestrationEvent:
    event_id: str
    event_type: EventType
    timestamp: float
    request: Optional[Mapping[str, Any]] = None
    request_id: Optional[int] = None
    deadline_ms: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemObservation:
    timestamp: float
    snapshot_version: str
    metrics: Mapping[str, Any]
    risky_sfts: Tuple[Mapping[str, Any], ...] = ()
    queue_depth: int = 0
    busy_roles: Tuple[AgentRole, ...] = ()


@dataclass(frozen=True)
class BrainCommand:
    command_id: str
    event_id: str
    command_type: CommandType
    target_request_id: Optional[int]
    assigned_roles: Tuple[AgentRole, ...]
    snapshot_version: str
    priority: float
    deadline_ms: Optional[float]
    reason: str


@dataclass(frozen=True)
class SkillProposal:
    proposal_id: str
    command_id: str
    role: AgentRole
    skill_name: str
    request_id: Optional[int]
    action_type: str
    score: float
    feasible: bool
    snapshot_version: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    event_id: str
    command: BrainCommand
    proposals: Tuple[SkillProposal, ...]
    selected: str
    success: bool
    applied: bool
    status_code: str
    reason: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _enum_values(asdict(self))


def _enum_values(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_values(item) for item in value]
    return value
