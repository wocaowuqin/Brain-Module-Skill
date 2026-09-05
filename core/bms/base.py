from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Mapping
from .exceptions import LayerViolationError

@dataclass(frozen=True)
class BMSContext:
    """Small immutable hand-off object shared by Brain, Module and Skill."""
    timestamp: float
    ledger_snapshot: Any = None
    active_requests: tuple[Mapping[str, Any], ...] = ()
    traffic_history: Mapping[str, Any] = field(default_factory=dict)
    migration_history: Mapping[str, Any] = field(default_factory=dict)
    sla_violations: tuple[Mapping[str, Any], ...] = ()
    queue_backlog: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    external_events: tuple[Mapping[str, Any], ...] = ()

def _check_layer(layer: str):
    """Validate BMS return contracts and reject cross-layer objects."""
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            result = fn(*args, **kwargs)
            if not isinstance(result, dict):
                raise LayerViolationError(f"{layer} method {fn.__name__} must return dict")
            forbidden = result.get("_layer_object")
            if layer == "skill" and forbidden in {"BrainCommand", "BaseBrain"}:
                raise LayerViolationError("Skill cannot return a BrainCommand")
            if layer == "module" and any(k in result for k in ("ledger", "ledger_handle")):
                raise LayerViolationError("Module cannot expose a ledger handle")
            return result
        return wrapped
    return decorator

def _require(result, keys, layer):
    missing = [key for key in keys if key not in result]
    if missing: raise LayerViolationError(f"{layer} result missing keys: {missing}")

class BaseSkill(ABC):
    layer = "skill"
    @abstractmethod
    @_check_layer("skill")
    def execute(self, context: BMSContext, params: dict) -> dict:
        raise NotImplementedError

class BaseModule(ABC):
    layer = "module"
    def __init__(self, skills=None): self.skills = list(skills or [])
    @abstractmethod
    @_check_layer("module")
    def run(self, context: BMSContext, instruction: dict) -> dict:
        raise NotImplementedError

class BaseBrain(ABC):
    layer = "brain"
    @abstractmethod
    def decide(self, context: BMSContext) -> dict:
        raise NotImplementedError
    @abstractmethod
    def arbitrate(self, module_result: dict, context: BMSContext) -> dict:
        raise NotImplementedError
