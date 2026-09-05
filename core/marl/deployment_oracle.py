"""MILP joint-action oracle for deployment_topk_v3 request batches."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from core.marl.batch_deployment_wqmix import ResourceFootprint, ResourceSnapshot
from core.marl.deployment_topk import deserialize_footprint


InstanceKey = Tuple[int, int]
ResourceKey = Tuple[str, Any]


@dataclass
class OracleSolution:
    status: str
    optimal: bool
    objective: float
    mip_gap: Optional[float]
    actions: List[int]
    accepted: int
    rejected: int
    selected_sources: List[str]
    resource_usage: Dict[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "optimal": self.optimal,
            "objective": self.objective,
            "mip_gap": self.mip_gap,
            "actions": self.actions,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "selected_sources": self.selected_sources,
            "resource_usage": self.resource_usage,
            "message": self.message,
        }


def snapshot_from_batch(batch: Mapping[str, Any]) -> ResourceSnapshot:
    raw = batch["resource_snapshot"]
    return ResourceSnapshot(
        version=int(raw["version"]),
        cpu_remaining={
            int(node): float(value) for node, value in raw["cpu_remaining"].items()
        },
        memory_remaining={
            int(node): float(value) for node, value in raw["memory_remaining"].items()
        },
        bandwidth_remaining={
            (int(row["u"]), int(row["v"])): float(row["mbps"])
            for row in raw["bandwidth_remaining"]
        },
        vnf_instances={
            (int(row["node"]), int(row["vnf_type"])): (
                float(row["cpu"]),
                float(row["memory"]),
                int(row["ref_count"]),
            )
            for row in raw.get("vnf_instances", [])
        },
    )


def candidate_footprints(batch: Mapping[str, Any]) -> List[List[Optional[ResourceFootprint]]]:
    rows = []
    for agent in batch["agents"]:
        rows.append([
            None
            if candidate.get("resource_footprint") is None
            else deserialize_footprint(candidate["resource_footprint"])
            for candidate in agent["candidates"]
        ])
    return rows


class DeploymentBatchOracle:
    """Solve one request batch as a binary resource-allocation MILP."""

    STATUS_NAMES = {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }

    def __init__(
        self,
        reject_penalty: float = 1_000_000.0,
        priority_penalty: float = 1_000.0,
        candidate_cost_scale: float = 0.01,
        sla_risk_scale: float = 0.0,
        sla_violation_penalty: float = 0.0,
        queue_safety_factor: float = 1.0,
        bandwidth_capacity_mbps: Optional[float] = None,
        q0_risk_weight: float = 4.0,
        q1_risk_weight: float = 1.5,
        q2_risk_weight: float = 1.0,
        time_limit_seconds: float = 10.0,
    ) -> None:
        self.reject_penalty = float(reject_penalty)
        self.priority_penalty = float(priority_penalty)
        self.candidate_cost_scale = float(candidate_cost_scale)
        self.sla_risk_scale = float(sla_risk_scale)
        self.sla_violation_penalty = float(sla_violation_penalty)
        self.queue_safety_factor = float(queue_safety_factor)
        self.bandwidth_capacity_mbps = (
            None
            if bandwidth_capacity_mbps is None
            else float(bandwidth_capacity_mbps)
        )
        self.qos_risk_weights = {
            3: float(q0_risk_weight),
            2: float(q1_risk_weight),
            1: float(q2_risk_weight),
        }
        self.time_limit_seconds = float(time_limit_seconds)
        if (
            self.sla_risk_scale < 0.0
            or self.sla_violation_penalty < 0.0
            or self.queue_safety_factor < 0.0
            or (
                self.bandwidth_capacity_mbps is not None
                and self.bandwidth_capacity_mbps <= 0.0
            )
            or any(weight < 0.0 for weight in self.qos_risk_weights.values())
        ):
            raise ValueError("SLA-aware Oracle parameters must be nonnegative")

    def solve(self, batch: Mapping[str, Any]) -> OracleSolution:
        agents = batch["agents"]
        snapshot = snapshot_from_batch(batch)
        footprints = candidate_footprints(batch)
        action_variables: List[List[int]] = []
        variable_count = 0
        for agent in agents:
            indices = list(range(variable_count, variable_count + len(agent["candidates"])))
            action_variables.append(indices)
            variable_count += len(indices)

        new_instance_requirements = self._new_instance_requirements(footprints, snapshot)
        instance_variables = {
            key: variable_count + index
            for index, key in enumerate(sorted(new_instance_requirements))
        }
        variable_count += len(instance_variables)

        objective = np.zeros(variable_count, dtype=np.float64)
        upper = np.ones(variable_count, dtype=np.float64)
        for agent_index, agent in enumerate(agents):
            priority = float(agent.get("request_features", [0.0] * 11)[10])
            for action_index, candidate in enumerate(agent["candidates"]):
                variable = action_variables[agent_index][action_index]
                if not bool(candidate.get("action_valid", False)):
                    upper[variable] = 0.0
                if candidate.get("source") == "reject" or candidate.get("plan") is None:
                    objective[variable] = self.reject_penalty + self.priority_penalty * priority
                else:
                    raw_cost = max(0.0, float(candidate.get("objective", 0.0)))
                    objective[variable] = self.candidate_cost_scale * raw_cost
                    if self.sla_risk_scale > 0.0:
                        risk = self._candidate_sla_risk(candidate, footprints[agent_index][action_index], snapshot)
                        qos_weight = self.qos_risk_weights.get(int(round(priority)), 1.0)
                        objective[variable] += (
                            self.sla_risk_scale * qos_weight * min(2.0, risk) ** 2
                        )
                        if risk > 1.0:
                            objective[variable] += self.sla_violation_penalty * qos_weight
                    objective[variable] += 1e-8 * (agent_index * 1_000 + action_index)
        for key, variable in instance_variables.items():
            requirement = new_instance_requirements[key]
            objective[variable] = 1e-6 * (requirement[0] + requirement[1])

        row_indices: List[int] = []
        column_indices: List[int] = []
        values: List[float] = []
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []

        def add_constraint(coefficients: Mapping[int, float], lower: float, upper_value: float) -> None:
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
            add_constraint({variable: 1.0 for variable in variables}, 1.0, 1.0)

        cpu_rows: Dict[int, Dict[int, float]] = {
            node: {} for node in snapshot.cpu_remaining
        }
        memory_rows: Dict[int, Dict[int, float]] = {
            node: {} for node in snapshot.memory_remaining
        }
        bandwidth_rows: Dict[Tuple[int, int], Dict[int, float]] = {
            edge: {} for edge in snapshot.bandwidth_remaining
        }

        for key, variable in instance_variables.items():
            cpu, memory = new_instance_requirements[key]
            cpu_rows.setdefault(key[0], {})[variable] = cpu
            memory_rows.setdefault(key[0], {})[variable] = memory

        for agent_index, request_footprints in enumerate(footprints):
            for action_index, footprint in enumerate(request_footprints):
                if footprint is None:
                    continue
                variable = action_variables[agent_index][action_index]
                if footprint.vnf_instances:
                    needed_keys = {
                        (requirement.node, requirement.vnf_type)
                        for requirement in footprint.vnf_instances
                        if (requirement.node, requirement.vnf_type) not in snapshot.vnf_instances
                    }
                    for key in needed_keys:
                        add_constraint(
                            {variable: 1.0, instance_variables[key]: -1.0},
                            -math.inf,
                            0.0,
                        )
                else:
                    for node, amount in footprint.cpu.items():
                        cpu_rows.setdefault(node, {})[variable] = amount
                    for node, amount in footprint.memory.items():
                        memory_rows.setdefault(node, {})[variable] = amount
                for edge, amount in footprint.bandwidth.items():
                    bandwidth_rows.setdefault(edge, {})[variable] = amount

        for node, coefficients in cpu_rows.items():
            add_constraint(coefficients, -math.inf, snapshot.cpu_remaining.get(node, 0.0))
        for node, coefficients in memory_rows.items():
            add_constraint(coefficients, -math.inf, snapshot.memory_remaining.get(node, 0.0))
        for edge, coefficients in bandwidth_rows.items():
            add_constraint(
                coefficients, -math.inf, snapshot.bandwidth_remaining.get(edge, 0.0)
            )

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
                "time_limit": max(0.01, self.time_limit_seconds),
                "mip_rel_gap": 0.0,
                "presolve": True,
            },
        )
        status = self.STATUS_NAMES.get(int(result.status), f"status_{result.status}")
        if result.x is None:
            return OracleSolution(
                status=status,
                optimal=False,
                objective=math.inf,
                mip_gap=None,
                actions=[int(agent["reject_action"]) for agent in agents],
                accepted=0,
                rejected=len(agents),
                selected_sources=["reject"] * len(agents),
                resource_usage={},
                message=str(result.message),
            )

        actions = []
        selected_sources = []
        for agent_index, variables in enumerate(action_variables):
            values_for_agent = result.x[variables]
            action = int(np.argmax(values_for_agent))
            actions.append(action)
            selected_sources.append(str(agents[agent_index]["candidates"][action]["source"]))
        usage = self.resource_usage(batch, actions)
        accepted = sum(source != "reject" for source in selected_sources)
        mip_gap_raw = getattr(result, "mip_gap", None)
        mip_gap = None if mip_gap_raw is None else float(mip_gap_raw)
        return OracleSolution(
            status=status,
            optimal=int(result.status) == 0,
            objective=float(result.fun),
            mip_gap=mip_gap,
            actions=actions,
            accepted=accepted,
            rejected=len(agents) - accepted,
            selected_sources=selected_sources,
            resource_usage=usage,
            message=str(result.message),
        )

    def _candidate_sla_risk(
        self,
        candidate: Mapping[str, Any],
        footprint: Optional[ResourceFootprint],
        snapshot: ResourceSnapshot,
    ) -> float:
        metrics = candidate.get("metrics") or {}
        delay_bound = max(float(metrics.get("delay_bound_ms", 0.0)), 1e-6)
        static_delay = max(float(metrics.get("estimated_delay_ms", 0.0)), 0.0)
        plan = candidate.get("plan")
        if (
            footprint is None
            or not isinstance(plan, Mapping)
            or self.bandwidth_capacity_mbps is None
        ):
            return static_delay / delay_bound

        capacity = self.bandwidth_capacity_mbps

        def edge_multiplier(u: int, v: int) -> float:
            edge = (int(u), int(v))
            remaining = float(snapshot.bandwidth_remaining.get(edge, 0.0))
            current_used = max(0.0, capacity - remaining)
            selected = float(footprint.bandwidth.get(edge, 0.0))
            utilization = min(0.95, max(0.0, (current_used + selected) / capacity))
            return 1.0 + self.queue_safety_factor * utilization / max(1e-6, 1.0 - utilization)

        def path_weight(path: Sequence[Any]) -> Tuple[float, int]:
            values = [int(node) for node in path]
            multipliers = [
                edge_multiplier(u, v) for u, v in zip(values, values[1:])
            ]
            return sum(multipliers), len(multipliers)

        weighted_hops = 0.0
        hop_count = 0
        for segment in plan.get("segments", []):
            weight, count = path_weight(segment.get("path", []))
            weighted_hops += weight
            hop_count += count
        branch_results = [
            path_weight(path)
            for path in ((plan.get("multicast") or {}).get("paths") or {}).values()
        ]
        if branch_results:
            branch_weight, branch_hops = max(
                branch_results,
                key=lambda row: row[0] / max(1, row[1]),
            )
            weighted_hops += branch_weight
            hop_count += branch_hops
        if hop_count <= 0:
            return static_delay / delay_bound
        modeled_delay = static_delay * weighted_hops / hop_count
        return modeled_delay / delay_bound

    @staticmethod
    def _new_instance_requirements(
        footprints: Sequence[Sequence[Optional[ResourceFootprint]]],
        snapshot: ResourceSnapshot,
    ) -> Dict[InstanceKey, Tuple[float, float]]:
        requirements: Dict[InstanceKey, Tuple[float, float]] = {}
        for request_candidates in footprints:
            for footprint in request_candidates:
                if footprint is None:
                    continue
                for requirement in footprint.vnf_instances:
                    key = (requirement.node, requirement.vnf_type)
                    if key in snapshot.vnf_instances:
                        continue
                    old_cpu, old_memory = requirements.get(key, (0.0, 0.0))
                    requirements[key] = (
                        max(old_cpu, float(requirement.cpu)),
                        max(old_memory, float(requirement.memory)),
                    )
        return requirements

    def resource_usage(
        self,
        batch: Mapping[str, Any],
        actions: Sequence[int],
    ) -> Dict[str, Any]:
        snapshot = snapshot_from_batch(batch)
        footprints = candidate_footprints(batch)
        selected = [footprints[index][int(action)] for index, action in enumerate(actions)]
        new_instances = self._new_instance_requirements(
            [[footprint] for footprint in selected if footprint is not None], snapshot
        )
        cpu: Dict[int, float] = {}
        memory: Dict[int, float] = {}
        bandwidth: Dict[Tuple[int, int], float] = {}
        for (node, _), (cpu_amount, memory_amount) in new_instances.items():
            cpu[node] = cpu.get(node, 0.0) + cpu_amount
            memory[node] = memory.get(node, 0.0) + memory_amount
        for footprint in selected:
            if footprint is None:
                continue
            if not footprint.vnf_instances:
                for node, amount in footprint.cpu.items():
                    cpu[node] = cpu.get(node, 0.0) + amount
                for node, amount in footprint.memory.items():
                    memory[node] = memory.get(node, 0.0) + amount
            for edge, amount in footprint.bandwidth.items():
                bandwidth[edge] = bandwidth.get(edge, 0.0) + amount
        return {
            "cpu": {str(node): amount for node, amount in sorted(cpu.items())},
            "memory": {str(node): amount for node, amount in sorted(memory.items())},
            "bandwidth": [
                {"u": edge[0], "v": edge[1], "mbps": amount}
                for edge, amount in sorted(bandwidth.items())
            ],
            "new_vnf_instances": len(new_instances),
        }


def validate_solution(batch: Mapping[str, Any], solution: OracleSolution) -> List[str]:
    errors = []
    snapshot = snapshot_from_batch(batch)
    if len(solution.actions) != len(batch["agents"]):
        return ["action_count_mismatch"]
    for agent, action in zip(batch["agents"], solution.actions):
        if not 0 <= int(action) < len(agent["candidates"]):
            errors.append("action_out_of_range")
            continue
        if not bool(agent["candidates"][int(action)].get("action_valid", False)):
            errors.append("invalid_action_selected")
    usage = solution.resource_usage
    for node, amount in usage.get("cpu", {}).items():
        if float(amount) > snapshot.cpu_remaining.get(int(node), 0.0) + 1e-7:
            errors.append(f"cpu_exceeded:{node}")
    for node, amount in usage.get("memory", {}).items():
        if float(amount) > snapshot.memory_remaining.get(int(node), 0.0) + 1e-7:
            errors.append(f"memory_exceeded:{node}")
    for row in usage.get("bandwidth", []):
        edge = (int(row["u"]), int(row["v"]))
        if float(row["mbps"]) > snapshot.bandwidth_remaining.get(edge, 0.0) + 1e-7:
            errors.append(f"bandwidth_exceeded:{edge[0]}->{edge[1]}")
    return errors
