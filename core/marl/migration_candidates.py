"""Complete Top-K target candidates for stateful VNF migration."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass
import heapq
import math
from typing import Any, Mapping, Optional, Sequence

from core.marl.batch_deployment_wqmix import (
    ResourceFootprint,
    ResourceSnapshot,
    candidate_conflict_features,
    footprint_feasible,
)
from core.marl.deployment_topk import serialize_footprint
from core.marl.migration_scheduler import MigrationTask


MIGRATION_REQUEST_DIM = 14
MIGRATION_CANDIDATE_DIM = 24
MIGRATION_STATE_DIM = 12


@dataclass(frozen=True)
class MigrationCandidate:
    candidate_id: str
    target_node: int
    plan: dict[str, Any]
    footprint: ResourceFootprint
    metrics: dict[str, float]
    objective: float
    feasible: bool
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_node": self.target_node,
            "plan": self.plan,
            "resource_footprint": serialize_footprint(self.footprint),
            "metrics": dict(self.metrics),
            "objective": self.objective,
            "feasible": self.feasible,
            "rejection_reason": self.rejection_reason,
        }


def _placement_rows(plan: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = plan.get("placement_by_vnf") or {}
    if isinstance(raw, Mapping):
        return {int(key): value for key, value in raw.items()}
    return {index: value for index, value in enumerate(raw)}


def plan_edge_multiplicity(plan: Mapping[str, Any]) -> Counter[tuple[int, int]]:
    edges: Counter[tuple[int, int]] = Counter()
    for segment in plan.get("segments") or []:
        path = [int(node) for node in segment.get("path") or []]
        edges.update(zip(path, path[1:]))
    multicast = plan.get("multicast") or {}
    tree_edges = multicast.get("tree_edges")
    if tree_edges:
        edges.update((int(edge[0]), int(edge[1])) for edge in tree_edges)
    elif multicast.get("paths"):
        seen: set[tuple[int, int]] = set()
        for path in multicast["paths"].values():
            nodes = [int(node) for node in path]
            seen.update(zip(nodes, nodes[1:]))
        edges.update(seen)
    return edges


class MigrationCandidateGenerator:
    """Rebuild adjacent SFC segments for each feasible target DC."""

    def __init__(
        self,
        profile: Mapping[str, Any],
        max_candidates: int = 8,
        cpu_capacity: float | Mapping[int, float] = 55.0,
        memory_capacity: float | Mapping[int, float] = 45.0,
        max_projected_utilization: float = 0.90,
        control_plane_ms: float = 344.0,
        state_copy_mbps: float = 100.0,
        delay_safety_ratio: float = 1.0,
    ) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if not 0.0 < max_projected_utilization <= 1.0:
            raise ValueError("max_projected_utilization must be in (0, 1]")
        self.profile = dict(profile)
        self.max_candidates = int(max_candidates)
        self.max_projected_utilization = float(max_projected_utilization)
        self.control_plane_ms = float(control_plane_ms)
        self.state_copy_mbps = max(1e-6, float(state_copy_mbps))
        self.delay_safety_ratio = float(delay_safety_ratio)
        self.nodes = {
            int(row.get("dpid", row.get("id"))): dict(row)
            for row in self.profile.get("nodes", [])
        }
        self.dc_nodes = [
            int(node)
            for node in self.profile.get("dc_nodes_1based", [])
        ] or [node for node, row in self.nodes.items() if bool(row.get("is_dc"))]
        self.default_bandwidth = float(self.profile.get("default_bandwidth_mbps", 90.0))
        self.default_delay = float(self.profile.get("default_delay_ms", 1.0))
        self.edge_capacity: dict[tuple[int, int], float] = {}
        self.edge_delay: dict[tuple[int, int], float] = {}
        self.edge_ports: dict[tuple[int, int], int] = {}
        self.adjacency: dict[int, list[int]] = {node: [] for node in self.nodes}
        for row in self.profile.get("edges", []):
            u, v = int(row["u"]), int(row["v"])
            capacity = float(row.get("bandwidth_mbps", self.default_bandwidth))
            delay = float(row.get("delay_ms", self.default_delay))
            if "u_port" not in row or "v_port" not in row:
                raise ValueError(f"topology edge {u}<->{v} has no switch port mapping")
            for edge in ((u, v), (v, u)):
                self.edge_capacity[edge] = capacity
                self.edge_delay[edge] = delay
            self.edge_ports[(u, v)] = int(row["u_port"])
            self.edge_ports[(v, u)] = int(row["v_port"])
            self.adjacency.setdefault(u, []).append(v)
            self.adjacency.setdefault(v, []).append(u)
        self.cpu_capacity = self._capacity_map(cpu_capacity, 55.0)
        self.memory_capacity = self._capacity_map(memory_capacity, 45.0)

    def _capacity_map(
        self, value: float | Mapping[int, float], fallback: float
    ) -> dict[int, float]:
        if isinstance(value, Mapping):
            return {node: float(value.get(node, fallback)) for node in self.dc_nodes}
        return {node: float(value) for node in self.dc_nodes}

    def _shortest_path(
        self,
        source: int,
        target: int,
        snapshot: ResourceSnapshot,
        demand_mbps: float,
    ) -> Optional[list[int]]:
        return self._shortest_paths_from(
            source, snapshot, demand_mbps
        ).get(int(target))

    def _shortest_paths_from(
        self,
        source: int,
        snapshot: ResourceSnapshot,
        demand_mbps: float,
        *,
        reverse: bool = False,
    ) -> dict[int, list[int]]:
        """One Dijkstra pass for every target under one immutable snapshot."""

        source = int(source)
        heap: list[tuple[float, tuple[int, ...]]] = [(0.0, (source,))]
        best = {source: 0.0}
        paths = {source: [source]}
        while heap:
            cost, path = heapq.heappop(heap)
            node = path[-1]
            if cost > best.get(node, math.inf) + 1e-12:
                continue
            for neighbor in self.adjacency.get(node, []):
                edge = (neighbor, node) if reverse else (node, neighbor)
                if neighbor in path:
                    continue
                remaining = float(snapshot.bandwidth_remaining.get(edge, 0.0))
                if remaining + 1e-9 < demand_mbps:
                    continue
                capacity = max(1e-9, self.edge_capacity.get(edge, self.default_bandwidth))
                utilization = max(0.0, 1.0 - remaining / capacity)
                edge_cost = self.edge_delay.get(edge, self.default_delay) * (
                    1.0 + 2.0 * utilization / max(0.05, 1.0 - utilization)
                )
                new_cost = cost + edge_cost
                if new_cost + 1e-12 < best.get(neighbor, math.inf):
                    best[neighbor] = new_cost
                    if reverse:
                        resolved_path = (neighbor, *paths[node])
                        heap_path = path + (neighbor,)
                    else:
                        resolved_path = path + (neighbor,)
                        heap_path = resolved_path
                    paths[neighbor] = list(resolved_path)
                    heapq.heappush(heap, (new_cost, heap_path))
        return paths

    def _path_delay(self, path: Sequence[int]) -> float:
        return sum(
            self.edge_delay.get((int(u), int(v)), self.default_delay)
            for u, v in zip(path, path[1:])
        )

    def _path_outputs(self, path: Sequence[int]) -> dict[str, list[int]]:
        nodes = [int(node) for node in path]
        if not nodes:
            raise ValueError("migration segment path cannot be empty")
        outputs: dict[str, list[int]] = {}
        for u, v in zip(nodes, nodes[1:]):
            if (u, v) not in self.edge_ports:
                raise ValueError(f"topology has no directed switch port for {u}->{v}")
            outputs[str(u)] = [self.edge_ports[(u, v)]]
        target = self.nodes.get(nodes[-1])
        if target is None or "host_port" not in target:
            raise ValueError(f"topology node {nodes[-1]} has no host port")
        outputs[str(nodes[-1])] = [int(target["host_port"])]
        return outputs

    def _plan_delay(self, plan: Mapping[str, Any]) -> float:
        segment = sum(
            self._path_delay([int(node) for node in row.get("path", [])])
            for row in plan.get("segments") or []
        )
        multicast = plan.get("multicast") or {}
        branch = max(
            (
                self._path_delay([int(node) for node in path])
                for path in (multicast.get("paths") or {}).values()
            ),
            default=0.0,
        )
        return float(segment + branch)

    def _migrated_plan(
        self,
        plan: Mapping[str, Any],
        stage: int,
        target: int,
        snapshot: ResourceSnapshot,
        demand_mbps: float,
        incoming_path: Optional[Sequence[int]] = None,
        outgoing_path: Optional[Sequence[int]] = None,
    ) -> Optional[dict[str, Any]]:
        migrated = copy.deepcopy(dict(plan))
        placements = _placement_rows(migrated)
        if stage <= 0 or stage >= max(placements):
            return None
        if stage - 1 not in placements or stage + 1 not in placements:
            return None
        old_dc = int(placements[stage]["dc_node"])
        previous = int(placements[stage - 1]["dc_node"])
        following = int(placements[stage + 1]["dc_node"])
        incoming = list(incoming_path) if incoming_path is not None else self._shortest_path(
            previous, target, snapshot, demand_mbps
        )
        outgoing = list(outgoing_path) if outgoing_path is not None else self._shortest_path(
            target, following, snapshot, demand_mbps
        )
        if incoming is None or outgoing is None:
            return None
        raw_placements = migrated["placement_by_vnf"]
        key: Any = str(stage) if isinstance(raw_placements, Mapping) and str(stage) in raw_placements else stage
        placement = raw_placements[key]
        placement["dc_node"] = int(target)
        node = self.nodes.get(int(target), {})
        target_ip = str(node.get("host_ip", node.get("ip", f"10.0.0.{target}"))).split("/", 1)[0]
        placement["listen_ip"] = target_ip
        if "chain_nodes" in migrated and stage < len(migrated["chain_nodes"]):
            migrated["chain_nodes"][stage] = int(target)
        segments = sorted(migrated.get("segments") or [], key=lambda row: int(row["stage"]))
        by_stage = {int(row["stage"]): row for row in segments}
        if stage not in by_stage or stage + 1 not in by_stage:
            return None
        by_stage[stage].update({
            "from_dpid": previous,
            "to_dpid": int(target),
            "target_ip": target_ip,
            "path": incoming,
            "switch_outputs": self._path_outputs(incoming),
        })
        next_key: Any = str(stage + 1) if isinstance(raw_placements, Mapping) and str(stage + 1) in raw_placements else stage + 1
        next_placement = raw_placements[next_key]
        by_stage[stage + 1].update({
            "from_dpid": int(target),
            "to_dpid": following,
            "target_ip": str(next_placement.get("listen_ip", f"10.0.0.{following}")),
            "path": outgoing,
            "switch_outputs": self._path_outputs(outgoing),
        })
        migrated["segments"] = segments
        migrated.setdefault("migration", {}).update({
            "stage": int(stage),
            "old_dc": old_dc,
            "target_dc": int(target),
            "snapshot_version": int(snapshot.version),
        })
        return migrated

    def generate(
        self,
        task: MigrationTask,
        current_plan: Mapping[str, Any],
        snapshot: ResourceSnapshot,
        *,
        delay_bound_ms: float = math.inf,
    ) -> list[MigrationCandidate]:
        old_edges = plan_edge_multiplicity(current_plan)
        old_delay = self._plan_delay(current_plan)
        placements = _placement_rows(current_plan)
        if task.stage - 1 not in placements or task.stage + 1 not in placements:
            return []
        previous = int(placements[task.stage - 1]["dc_node"])
        following = int(placements[task.stage + 1]["dc_node"])
        incoming_paths = self._shortest_paths_from(
            previous, snapshot, task.bandwidth_mbps
        )
        outgoing_paths = self._shortest_paths_from(
            following, snapshot, task.bandwidth_mbps, reverse=True
        )
        rows: list[MigrationCandidate] = []
        for target in self.dc_nodes:
            if int(target) == int(task.old_node):
                continue
            cpu_remaining = float(snapshot.cpu_remaining.get(target, 0.0))
            memory_remaining = float(snapshot.memory_remaining.get(target, 0.0))
            if cpu_remaining + 1e-9 < task.cpu or memory_remaining + 1e-9 < task.memory:
                continue
            migrated = self._migrated_plan(
                current_plan,
                task.stage,
                target,
                snapshot,
                task.bandwidth_mbps,
                incoming_paths.get(int(target)),
                outgoing_paths.get(int(target)),
            )
            if migrated is None:
                continue
            new_edges = plan_edge_multiplicity(migrated)
            temporary_bw = {
                edge: float(max(0, count - old_edges.get(edge, 0))) * task.bandwidth_mbps
                for edge, count in new_edges.items()
                if count > old_edges.get(edge, 0)
            }
            footprint = ResourceFootprint(
                cpu={target: task.cpu},
                memory={target: task.memory},
                bandwidth=temporary_bw,
            )
            cpu_cap = max(1e-9, self.cpu_capacity[target])
            mem_cap = max(1e-9, self.memory_capacity[target])
            target_util = max(
                0.0,
                1.0 - cpu_remaining / cpu_cap,
                1.0 - memory_remaining / mem_cap,
            )
            projected = max(
                1.0 - (cpu_remaining - task.cpu) / cpu_cap,
                1.0 - (memory_remaining - task.memory) / mem_cap,
            )
            delay = self._plan_delay(migrated)
            copy_ms = task.state_size_mb * 8.0 / self.state_copy_mbps * 1000.0
            migration_ms = self.control_plane_ms + copy_ms
            lifetime_ratio = task.remaining_lifetime_s / max(1e-6, migration_ms / 1000.0)
            delay_ratio = delay / max(1e-6, float(delay_bound_ms)) if math.isfinite(delay_bound_ms) else 0.0
            peak_bw = max(temporary_bw.values(), default=0.0)
            objective = (
                10.0 * task.utilization_relief
                + 3.0 * task.sla_risk
                - 2.0 * projected
                - max(0.0, delay - old_delay) / max(1.0, float(delay_bound_ms) if math.isfinite(delay_bound_ms) else 100.0)
                - migration_ms / 1000.0
            )
            reason = ""
            if projected > self.max_projected_utilization + 1e-9:
                reason = "projected_target_utilization"
            elif math.isfinite(delay_bound_ms) and delay > delay_bound_ms * self.delay_safety_ratio + 1e-9:
                reason = "delay_bound"
            elif task.remaining_lifetime_s <= migration_ms / 1000.0 + 1.0:
                reason = "insufficient_remaining_lifetime"
            elif not footprint_feasible(footprint, snapshot):
                reason = "resource_snapshot"
            metrics = {
                "cpu": task.cpu,
                "memory": task.memory,
                "temporary_bandwidth_sum": sum(temporary_bw.values()),
                "temporary_bandwidth_peak": peak_bw,
                "state_size_mb": task.state_size_mb,
                "path_hops": float(sum(new_edges.values())),
                "changed_edges": float(len(set(new_edges) ^ set(old_edges))),
                "target_utilization": target_util,
                "projected_target_utilization": projected,
                "current_delay_ms": old_delay,
                "predicted_delay_ms": delay,
                "delay_delta_ms": delay - old_delay,
                "delay_ratio": delay_ratio,
                # Counterfactual one-step SLA loss proxy consumed by the
                # lifecycle-aware reward.  Online predictors may overwrite
                # these fields with a multi-step estimate.
                "future_sla_loss_before": float(task.sla_risk),
                "future_sla_loss_after": max(0.0, delay_ratio - 1.0),
                "state_copy_ms": copy_ms,
                "migration_ms": migration_ms,
                "utilization_relief": task.utilization_relief,
                "sla_risk": task.sla_risk,
                "lifetime_migration_ratio": min(100.0, lifetime_ratio),
                "snapshot_version": float(snapshot.version),
            }
            rows.append(MigrationCandidate(
                candidate_id=f"{task.task_id}-n{target}",
                target_node=int(target),
                plan=migrated,
                footprint=footprint,
                metrics=metrics,
                objective=float(objective),
                feasible=not reason,
                rejection_reason=reason,
            ))
        rows.sort(key=lambda row: (
            not row.feasible,
            -row.objective,
            row.metrics["migration_ms"],
            row.target_node,
        ))
        return rows[: self.max_candidates]


def migration_request_features(task: MigrationTask) -> list[float]:
    values = [
        task.cpu,
        task.memory,
        task.bandwidth_mbps,
        task.state_size_mb,
        task.current_utilization,
        task.predicted_utilization,
        task.utilization_relief,
        task.sla_risk,
        min(1000.0, task.remaining_lifetime_s),
        task.estimated_migration_ms,
        task.priority,
        float(task.stage),
        float(task.vnf_type),
        1.0,
    ]
    assert len(values) == MIGRATION_REQUEST_DIM
    return values


def migration_candidate_base_features(
    candidate: Optional[MigrationCandidate],
) -> list[float]:
    if candidate is None:
        return [0.0] * 19 + [1.0]
    metric = candidate.metrics
    values = [
        metric["cpu"], metric["memory"],
        metric["temporary_bandwidth_sum"], metric["temporary_bandwidth_peak"],
        metric["state_size_mb"], metric["path_hops"], metric["changed_edges"],
        metric["target_utilization"], metric["projected_target_utilization"],
        metric["current_delay_ms"], metric["predicted_delay_ms"],
        metric["delay_delta_ms"], metric["delay_ratio"], metric["state_copy_ms"],
        metric["migration_ms"], metric["utilization_relief"], metric["sla_risk"],
        metric["lifetime_migration_ratio"], candidate.objective, 0.0,
    ]
    return values


def migration_batch_candidate_features(
    candidates: Sequence[Sequence[Optional[MigrationCandidate]]],
    snapshot: ResourceSnapshot,
) -> list[list[list[float]]]:
    footprints = [
        [candidate.footprint if candidate is not None else None for candidate in row]
        for row in candidates
    ]
    conflicts = candidate_conflict_features(footprints, snapshot)
    return [
        [
            migration_candidate_base_features(candidate) + conflicts[agent][index]
            for index, candidate in enumerate(row)
        ]
        for agent, row in enumerate(candidates)
    ]


def migration_global_state_features(
    snapshot: ResourceSnapshot,
    tasks: Sequence[MigrationTask],
) -> list[float]:
    cpu = list(map(float, snapshot.cpu_remaining.values()))
    memory = list(map(float, snapshot.memory_remaining.values()))
    bandwidth = list(map(float, snapshot.bandwidth_remaining.values()))

    def stats(values: Sequence[float]) -> tuple[float, float, float]:
        return (
            min(values, default=0.0),
            sum(values) / max(1, len(values)),
            max(values, default=0.0),
        )

    values = [
        *stats(cpu), *stats(memory), *stats(bandwidth),
        float(len(tasks)),
        max((task.predicted_utilization for task in tasks), default=0.0),
        sum(task.sla_risk for task in tasks) / max(1, len(tasks)),
    ]
    assert len(values) == MIGRATION_STATE_DIM
    return values


def candidate_mask(
    candidates: Sequence[Optional[MigrationCandidate]],
    snapshot: ResourceSnapshot,
) -> list[bool]:
    return [
        True if candidate is None else bool(
            candidate.feasible and footprint_feasible(candidate.footprint, snapshot)
        )
        for candidate in candidates
    ]
