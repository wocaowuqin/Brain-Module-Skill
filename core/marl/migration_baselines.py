"""Common heuristic and MILP baselines for batched VNF migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from core.marl.batch_deployment_wqmix import ResourceFootprint, ResourceSnapshot
from core.marl.deployment_topk import deserialize_footprint
from core.marl.migration_reward import migration_action_reward


HEURISTIC_POLICIES = (
    "no_migration",
    "random_feasible",
    "reactive_greedy",
    "predictive_greedy",
    "delay_aware_greedy",
)


def snapshot_from_migration_record(record: Mapping[str, Any]) -> ResourceSnapshot:
    raw = record["snapshot"]

    def edges(values: Mapping[str, Any]) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for key, value in values.items():
            if isinstance(key, str):
                u, v = key.split(",", 1)
                result[(int(u), int(v))] = float(value)
            else:
                result[(int(key[0]), int(key[1]))] = float(value)
        return result

    return ResourceSnapshot(
        version=int(raw.get("version", 0)),
        cpu_remaining={int(key): float(value) for key, value in raw["cpu_remaining"].items()},
        memory_remaining={
            int(key): float(value) for key, value in raw["memory_remaining"].items()
        },
        bandwidth_remaining=edges(raw["bandwidth_remaining"]),
    )


def candidate_footprints(
    record: Mapping[str, Any],
) -> list[list[ResourceFootprint | None]]:
    return [
        [
            None
            if candidate.get("resource_footprint") is None
            else deserialize_footprint(candidate["resource_footprint"])
            for candidate in agent["candidates"]
        ]
        for agent in record["agents"]
    ]


def _candidate_score(agent: Mapping[str, Any], action: int, policy: str) -> float:
    reject = int(agent["reject_action"])
    if action == reject:
        return migration_action_reward(agent, action)
    candidate = agent["candidates"][action]
    metrics = candidate.get("metrics") or {}
    if policy in {"reactive_greedy", "predictive_greedy"}:
        return float(candidate.get("objective", -math.inf))
    if policy == "delay_aware_greedy":
        return (
            -8.0 * max(0.0, float(metrics.get("delay_ratio", 0.0)) - 0.8)
            - 2.0 * float(metrics.get("projected_target_utilization", 1.0))
            - 0.002 * float(metrics.get("migration_ms", 0.0))
            + 2.0 * float(metrics.get("utilization_relief", 0.0))
        )
    return migration_action_reward(agent, action)


def heuristic_rankings(
    record: Mapping[str, Any],
    policy: str,
    *,
    rng: random.Random | None = None,
    overload_threshold: float = 0.85,
) -> tuple[list[list[int]], list[list[float]]]:
    """Build masked candidate rankings for one deterministic baseline."""

    if policy not in HEURISTIC_POLICIES:
        raise ValueError(f"unknown migration heuristic: {policy}")
    rng = rng or random.Random(0)
    rankings: list[list[int]] = []
    scores: list[list[float]] = []
    for agent in record["agents"]:
        reject = int(agent["reject_action"])
        valid = [
            index
            for index, allowed in enumerate(agent["action_mask"])
            if bool(allowed) and index != reject
        ]
        row_scores = [
            _candidate_score(agent, index, policy)
            for index in range(len(agent["candidates"]))
        ]
        if policy == "no_migration":
            # The joint decoder treats reject/noop as a fallback rather than a
            # searchable candidate.  Exposing non-noop actions here would turn
            # this lower-bound policy into an accidental migration policy.
            ranking = [reject]
        elif policy == "random_feasible":
            rng.shuffle(valid)
            ranking = valid + [reject]
        else:
            task = agent["task"]
            current = float(task.get("current_utilization", 0.0))
            predicted = float(task.get("predicted_utilization", current))
            trigger = current >= overload_threshold
            if policy in {"predictive_greedy", "delay_aware_greedy"}:
                trigger = max(current, predicted) >= overload_threshold
            valid.sort(key=lambda action: (-row_scores[action], action))
            ranking = valid + [reject] if trigger else [reject]
        rankings.append(ranking)
        scores.append(row_scores)
    return rankings, scores


@dataclass(frozen=True)
class MigrationOracleResult:
    status: str
    optimal: bool
    objective: float
    actions: list[int]
    accepted: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MigrationBatchMILPOracle:
    """Exact small-batch upper bound under the shared migration reward."""

    STATUS_NAMES = {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }

    def __init__(self, time_limit_seconds: float = 2.0) -> None:
        if time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive")
        self.time_limit_seconds = float(time_limit_seconds)

    def solve(self, record: Mapping[str, Any]) -> MigrationOracleResult:
        agents = list(record["agents"])
        snapshot = snapshot_from_migration_record(record)
        footprints = candidate_footprints(record)
        action_variables: list[list[int]] = []
        variable_count = 0
        for agent in agents:
            row = list(range(variable_count, variable_count + len(agent["candidates"])))
            action_variables.append(row)
            variable_count += len(row)

        objective = np.zeros(variable_count, dtype=np.float64)
        upper = np.ones(variable_count, dtype=np.float64)
        for agent_index, agent in enumerate(agents):
            for action, variable in enumerate(action_variables[agent_index]):
                if not bool(agent["action_mask"][action]):
                    upper[variable] = 0.0
                objective[variable] = -migration_action_reward(agent, action)
                objective[variable] += 1e-9 * (agent_index * 1000 + action)

        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []

        def constraint(coefficients: Mapping[int, float], lower: float, upper_value: float) -> None:
            row = len(lower_bounds)
            for column, value in coefficients.items():
                if abs(float(value)) <= 1e-12:
                    continue
                row_indices.append(row)
                column_indices.append(int(column))
                values.append(float(value))
            lower_bounds.append(float(lower))
            upper_bounds.append(float(upper_value))

        for variables in action_variables:
            constraint({variable: 1.0 for variable in variables}, 1.0, 1.0)

        cpu_rows: dict[int, dict[int, float]] = {}
        memory_rows: dict[int, dict[int, float]] = {}
        bandwidth_rows: dict[tuple[int, int], dict[int, float]] = {}
        for agent_index, agent_footprints in enumerate(footprints):
            for action, footprint in enumerate(agent_footprints):
                if footprint is None:
                    continue
                variable = action_variables[agent_index][action]
                for node, amount in footprint.cpu.items():
                    cpu_rows.setdefault(node, {})[variable] = float(amount)
                for node, amount in footprint.memory.items():
                    memory_rows.setdefault(node, {})[variable] = float(amount)
                for edge, amount in footprint.bandwidth.items():
                    bandwidth_rows.setdefault(edge, {})[variable] = float(amount)
        for node, coefficients in cpu_rows.items():
            constraint(coefficients, -math.inf, snapshot.cpu_remaining.get(node, 0.0))
        for node, coefficients in memory_rows.items():
            constraint(coefficients, -math.inf, snapshot.memory_remaining.get(node, 0.0))
        for edge, coefficients in bandwidth_rows.items():
            constraint(coefficients, -math.inf, snapshot.bandwidth_remaining.get(edge, 0.0))

        matrix = coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(lower_bounds), variable_count),
        ).tocsr()
        result = milp(
            c=objective,
            integrality=np.ones(variable_count, dtype=np.int8),
            bounds=Bounds(np.zeros(variable_count), upper),
            constraints=LinearConstraint(matrix, lower_bounds, upper_bounds),
            options={
                "time_limit": self.time_limit_seconds,
                "mip_rel_gap": 0.0,
                "presolve": True,
            },
        )
        status = self.STATUS_NAMES.get(int(result.status), f"status_{result.status}")
        if result.x is None:
            actions = [int(agent["reject_action"]) for agent in agents]
            return MigrationOracleResult(
                status, False, -math.inf, actions, 0, str(result.message)
            )
        actions = [
            int(np.argmax(result.x[variables]))
            for variables in action_variables
        ]
        reward = sum(
            migration_action_reward(agent, action)
            for agent, action in zip(agents, actions)
        )
        accepted = sum(
            int(action != int(agent["reject_action"]))
            for agent, action in zip(agents, actions)
        )
        return MigrationOracleResult(
            status=status,
            optimal=int(result.status) == 0,
            objective=float(reward),
            actions=actions,
            accepted=accepted,
            message=str(result.message),
        )
