"""Bounded, prediction-aware scheduling for online VNF migration.

The scheduler deliberately separates cheap fleet-wide screening from learned
joint target selection.  A VNF is a logical agent only after it has passed the
overload, lifetime, ownership, and cooldown gates.  This keeps the WQMIX input
bounded even when hundreds of VNFs are active.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Mapping, Sequence


@dataclass(frozen=True)
class MigrationTask:
    task_id: str
    request_id: int
    stage: int
    vnf_type: int
    old_node: int
    cpu: float
    memory: float
    bandwidth_mbps: float
    state_size_mb: float
    current_utilization: float
    predicted_utilization: float
    utilization_relief: float
    sla_risk: float
    remaining_lifetime_s: float
    estimated_migration_ms: float
    priority: float
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MigrationTask":
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})


class EWMATrendForecaster:
    """Small online forecaster with an EWMA level and bounded linear trend."""

    def __init__(
        self,
        alpha: float = 0.35,
        trend_alpha: float = 0.20,
        history_size: int = 32,
    ) -> None:
        if not 0.0 < alpha <= 1.0 or not 0.0 < trend_alpha <= 1.0:
            raise ValueError("EWMA coefficients must be in (0, 1]")
        if history_size < 2:
            raise ValueError("history_size must be at least two")
        self.alpha = float(alpha)
        self.trend_alpha = float(trend_alpha)
        self.history: Dict[int, Deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=int(history_size))
        )
        self.level: Dict[int, float] = {}
        self.trend: Dict[int, float] = {}

    def update(self, timestamp: float, values: Mapping[int, float]) -> None:
        now = float(timestamp)
        for raw_node, raw_value in values.items():
            node = int(raw_node)
            value = min(1.5, max(0.0, float(raw_value)))
            previous_level = self.level.get(node, value)
            samples = self.history[node]
            previous_time = samples[-1][0] if samples else now
            dt = max(1e-6, now - previous_time)
            instantaneous_trend = (value - previous_level) / dt
            previous_trend = self.trend.get(node, 0.0)
            level = self.alpha * value + (1.0 - self.alpha) * previous_level
            trend = (
                self.trend_alpha * instantaneous_trend
                + (1.0 - self.trend_alpha) * previous_trend
            )
            self.level[node] = level
            self.trend[node] = min(0.25, max(-0.25, trend))
            samples.append((now, value))

    def predict(self, node: int, horizon_s: float = 1.0) -> float:
        node = int(node)
        level = self.level.get(node, 0.0)
        trend = self.trend.get(node, 0.0)
        return min(1.5, max(0.0, level + max(0.0, float(horizon_s)) * trend))


def task_conflict(first: MigrationTask, second: MigrationTask) -> bool:
    """Return conflicts known before target candidates have been generated."""

    return (
        first.task_id == second.task_id
        or first.request_id == second.request_id
        or first.old_node == second.old_node
    )


def migration_execution_waves(
    tasks: Sequence[MigrationTask],
    max_inflight: int = 4,
) -> list[list[MigrationTask]]:
    """Pack tasks into bounded waves without same-request/source conflicts."""

    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    pending = list(tasks)
    waves: list[list[MigrationTask]] = []
    while pending:
        wave: list[MigrationTask] = []
        deferred: list[MigrationTask] = []
        for task in pending:
            if len(wave) < max_inflight and not any(
                task_conflict(task, selected) for selected in wave
            ):
                wave.append(task)
            else:
                deferred.append(task)
        if not wave:
            wave, deferred = [pending[0]], pending[1:]
        waves.append(wave)
        pending = deferred
    return waves
