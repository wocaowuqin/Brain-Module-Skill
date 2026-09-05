"""Complete deployment-plan candidate generation for request-batch MARL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

from core.marl.batch_deployment_wqmix import (
    ResourceFootprint,
    ResourceSnapshot,
    VNFInstanceRequirement,
    footprint_feasible,
)
from scripts.export_hrl_sfc_plans import (
    multicast_outputs,
    path_outputs,
    profile_maps,
)


Edge = Tuple[int, int]


@dataclass
class DeploymentCandidate:
    candidate_id: str
    source: str
    plan: Dict[str, Any]
    footprint: ResourceFootprint
    metrics: Dict[str, float]
    objective: float

    def to_dict(self, features: Sequence[float], action_valid: bool) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "action_valid": bool(action_valid),
            "objective": float(self.objective),
            "metrics": self.metrics,
            "resource_footprint": serialize_footprint(self.footprint),
            "candidate_features": [float(value) for value in features],
            "plan": self.plan,
        }


def serialize_footprint(footprint: ResourceFootprint) -> Dict[str, Any]:
    return {
        "cpu": {str(node): amount for node, amount in sorted(footprint.cpu.items())},
        "memory": {
            str(node): amount for node, amount in sorted(footprint.memory.items())
        },
        "bandwidth": [
            {"u": edge[0], "v": edge[1], "mbps": amount}
            for edge, amount in sorted(footprint.bandwidth.items())
        ],
        "vnf_instances": [
            {
                "node": requirement.node,
                "vnf_type": requirement.vnf_type,
                "cpu": requirement.cpu,
                "memory": requirement.memory,
            }
            for requirement in footprint.vnf_instances
        ],
    }


def deserialize_footprint(raw: Mapping[str, Any]) -> ResourceFootprint:
    return ResourceFootprint(
        cpu={int(node): float(value) for node, value in raw.get("cpu", {}).items()},
        memory={
            int(node): float(value) for node, value in raw.get("memory", {}).items()
        },
        bandwidth={
            (int(row["u"]), int(row["v"])): float(row["mbps"])
            for row in raw.get("bandwidth", [])
        },
        vnf_instances=tuple(
            VNFInstanceRequirement(
                node=int(row["node"]),
                vnf_type=int(row["vnf_type"]),
                cpu=float(row["cpu"]),
                memory=float(row["memory"]),
            )
            for row in raw.get("vnf_instances", [])
        ),
    )


def plan_signature(plan: Mapping[str, Any]) -> str:
    essential = {
        "chain_nodes": [int(value) for value in plan.get("chain_nodes", [])],
        "segments": [
            [int(value) for value in segment.get("path", [])]
            for segment in sorted(plan.get("segments", []), key=lambda row: int(row["stage"]))
        ],
        "tree_edges": sorted(
            [list(map(int, edge)) for edge in (plan.get("multicast") or {}).get("tree_edges", [])]
        ),
    }
    return hashlib.sha256(
        json.dumps(essential, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class CompletePlanCandidateGenerator:
    """Generate diverse full SFC plans against one immutable snapshot."""

    PLACEMENT_MODES = ("balanced", "compact", "spread", "residual")
    ROUTING_MODES = (
        "delay", "hop", "residual", "spread",
        "tree_delay", "tree_residual",
    )

    def __init__(
        self,
        profile: Mapping[str, Any],
        max_candidates: int = 8,
        placement_beam: int = 16,
        placement_chains: int = 12,
        pool_limit: int = 64,
        stage_port_base: int = 20000,
        objective_delay_weight: float = 1.0,
        objective_cpu_weight: float = 0.0,
        objective_memory_weight: float = 0.0,
        objective_bandwidth_weight: float = 0.03,
        objective_pressure_weight: float = 8.0,
    ) -> None:
        self.profile = dict(profile)
        self.max_candidates = int(max_candidates)
        self.placement_beam = int(placement_beam)
        self.placement_chains = int(placement_chains)
        self.pool_limit = int(pool_limit)
        self.stage_port_base = int(stage_port_base)
        self.objective_weights = {
            "delay": float(objective_delay_weight),
            "cpu": float(objective_cpu_weight),
            "memory": float(objective_memory_weight),
            "bandwidth": float(objective_bandwidth_weight),
            "pressure": float(objective_pressure_weight),
        }
        if any(value < 0.0 for value in self.objective_weights.values()):
            raise ValueError("candidate objective weights must be nonnegative")
        self.ports, self.nodes = profile_maps(self.profile)
        self.dc_nodes = sorted(int(value) for value in self.profile["dc_nodes_1based"])
        self.graph = nx.Graph()
        self.edge_delay: Dict[Edge, float] = {}
        for raw in self.profile["edges"]:
            u, v = int(raw["u"]), int(raw["v"])
            delay = float(raw.get("delay_ms", self.profile.get("default_delay_ms", 1.0)))
            self.graph.add_edge(u, v, delay_ms=delay)
            self.edge_delay[(u, v)] = delay
            self.edge_delay[(v, u)] = delay
        self.hops = dict(nx.all_pairs_shortest_path_length(self.graph))
        self.cached_paths: Dict[Tuple[int, int], List[List[int]]] = {}

    def prewarm_paths(self, max_paths: int = 8) -> None:
        """Cache delay-ranked topology paths before online request timing starts."""

        nodes = sorted(map(int, self.graph.nodes))
        for index, source in enumerate(nodes):
            for target in nodes[index + 1:]:
                try:
                    paths = [
                        list(map(int, path))
                        for path in islice(
                            nx.shortest_simple_paths(
                                self.graph, source, target, weight="delay_ms"
                            ),
                            max_paths,
                        )
                    ]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    paths = []
                self.cached_paths[(source, target)] = paths
                self.cached_paths[(target, source)] = [
                    list(reversed(path)) for path in paths
                ]

    def _cached_feasible_path(
        self,
        source: int,
        target: int,
        snapshot: ResourceSnapshot,
        bandwidth: float,
    ) -> Optional[List[int]]:
        if source == target:
            return [int(source)]
        best: Optional[Tuple[Tuple[float, float, int, Tuple[int, ...]], List[int]]] = None
        for path in self.cached_paths.get((int(source), int(target)), []):
            remaining = [
                float(snapshot.bandwidth_remaining.get((u, v), 0.0))
                for u, v in zip(path, path[1:])
            ]
            if any(value + 1e-9 < bandwidth for value in remaining):
                continue
            peak = max(
                (bandwidth / max(value, 1e-9) for value in remaining),
                default=0.0,
            )
            delay = self._path_delay(path)
            key = (peak, delay, len(path), tuple(path))
            if best is None or key < best[0]:
                best = (key, path)
        return list(best[1]) if best is not None else None

    def generate(
        self,
        request: Mapping[str, Any],
        snapshot: ResourceSnapshot,
        baseline_plan: Optional[Mapping[str, Any]] = None,
    ) -> List[DeploymentCandidate]:
        request_id = int(request["id"])
        pool: Dict[str, DeploymentCandidate] = {}

        if baseline_plan and baseline_plan.get("accepted", True):
            try:
                plan = json.loads(json.dumps(baseline_plan))
                validate_complete_plan(plan, request, self.profile)
                footprint = ResourceFootprint.from_sfc_plan(
                    plan, float(request["bw_origin"]), snapshot
                )
                if footprint_feasible(footprint, snapshot):
                    candidate = self._candidate(request, plan, footprint, "legacy_hrl", snapshot)
                    pool[plan_signature(plan)] = candidate
            except (KeyError, TypeError, ValueError):
                pass

        placement_rows = self._placement_chains(request, snapshot)
        for chain_index, (chain, placement_mode, _) in enumerate(placement_rows):
            for route_index, route_mode in enumerate(self.ROUTING_MODES):
                if len(pool) >= self.pool_limit:
                    break
                plan = self._build_plan(
                    request,
                    chain,
                    snapshot,
                    route_mode,
                    variant=chain_index * len(self.ROUTING_MODES) + route_index,
                )
                if plan is None:
                    continue
                try:
                    validate_complete_plan(plan, request, self.profile)
                except ValueError:
                    continue
                footprint = ResourceFootprint.from_sfc_plan(
                    plan, float(request["bw_origin"]), snapshot
                )
                if not footprint_feasible(footprint, snapshot):
                    continue
                signature = plan_signature(plan)
                candidate = self._candidate(
                    request,
                    plan,
                    footprint,
                    f"beam_{placement_mode}_{route_mode}",
                    snapshot,
                )
                previous = pool.get(signature)
                if previous is None or candidate.objective < previous.objective:
                    pool[signature] = candidate

        ranked = sorted(pool.values(), key=lambda row: (row.objective, row.candidate_id))
        selected = self._select_diverse(ranked)
        return selected[: self.max_candidates]

    def generate_first_feasible(
        self,
        request: Mapping[str, Any],
        snapshot: ResourceSnapshot,
    ) -> Optional[DeploymentCandidate]:
        """Return the first complete live plan without building a Top-K pool.

        This path is intended for online repair of reject-only offline rows.  It
        preserves the same placement, routing, validation, footprint, and
        objective semantics as :meth:`generate`, but stops as soon as one hard-
        feasible candidate is available.
        """

        bandwidth = float(request["bw_origin"])
        path_finder = lambda source, target: self._cached_feasible_path(
            source, target, snapshot, bandwidth
        )
        for chain_index, (chain, placement_mode) in enumerate(
            self._greedy_placement_chains(request, snapshot)
        ):
            plan = self._build_plan(
                request,
                chain,
                snapshot,
                "residual",
                variant=chain_index,
                path_finder=path_finder,
            )
            if plan is None:
                continue
            try:
                validate_complete_plan(plan, request, self.profile)
                footprint = ResourceFootprint.from_sfc_plan(
                    plan, bandwidth, snapshot
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not footprint_feasible(footprint, snapshot):
                continue
            return self._candidate(
                request,
                plan,
                footprint,
                f"cached_path_repair_{placement_mode}",
                snapshot,
            )
        return None

    def _greedy_placement_chains(
        self,
        request: Mapping[str, Any],
        snapshot: ResourceSnapshot,
    ) -> List[Tuple[Tuple[int, ...], str]]:
        """Build a few O(stages * DCs) live placement alternatives."""

        source = int(request["source_dpid"])
        cpu = list(map(float, request["cpu_origin"]))
        memory = list(map(float, request["memory_origin"]))
        vnf_types = list(map(int, request["vnf"]))
        results: List[Tuple[Tuple[int, ...], str]] = []
        seen: set[Tuple[int, ...]] = set()
        for mode in ("compact", "residual", "balanced", "spread"):
            chain: List[int] = []
            cpu_used: Dict[int, float] = {}
            memory_used: Dict[int, float] = {}
            feasible = True
            for stage, (cpu_need, memory_need, vnf_type) in enumerate(
                zip(cpu, memory, vnf_types)
            ):
                previous = source if not chain else chain[-1]
                choices = []
                for dc in self.dc_nodes:
                    reused = (dc, vnf_type) in snapshot.vnf_instances or any(
                        chain[index] == dc and vnf_types[index] == vnf_type
                        for index in range(len(chain))
                    )
                    cpu_increment = 0.0 if reused else cpu_need
                    memory_increment = 0.0 if reused else memory_need
                    next_cpu = cpu_used.get(dc, 0.0) + cpu_increment
                    next_memory = memory_used.get(dc, 0.0) + memory_increment
                    cpu_remaining = float(snapshot.cpu_remaining.get(dc, 0.0))
                    memory_remaining = float(snapshot.memory_remaining.get(dc, 0.0))
                    if (
                        next_cpu > cpu_remaining + 1e-9
                        or next_memory > memory_remaining + 1e-9
                    ):
                        continue
                    distance = float(self.hops.get(previous, {}).get(dc, 10_000))
                    if distance >= 10_000:
                        continue
                    pressure = max(
                        next_cpu / max(cpu_remaining, 1e-9),
                        next_memory / max(memory_remaining, 1e-9),
                    )
                    repeated = float(dc in chain)
                    if mode == "compact":
                        score = distance + 1.5 * pressure - 0.75 * repeated
                    elif mode == "spread":
                        score = distance + 7.0 * pressure + 2.0 * repeated
                    elif mode == "residual":
                        score = 0.5 * distance + 14.0 * pressure
                    else:
                        score = distance + 4.0 * pressure + 0.5 * repeated
                    choices.append((score, dc, next_cpu, next_memory))
                if not choices:
                    feasible = False
                    break
                _, dc, next_cpu, next_memory = min(choices)
                chain.append(dc)
                cpu_used[dc] = next_cpu
                memory_used[dc] = next_memory
            chain_tuple = tuple(chain)
            if feasible and chain_tuple not in seen:
                seen.add(chain_tuple)
                results.append((chain_tuple, mode))
        return results

    def _placement_chains(
        self,
        request: Mapping[str, Any],
        snapshot: ResourceSnapshot,
    ) -> List[Tuple[Tuple[int, ...], str, float]]:
        cpu = [float(value) for value in request["cpu_origin"]]
        memory = [float(value) for value in request["memory_origin"]]
        vnf_types = [int(value) for value in request["vnf"]]
        source = int(request["source_dpid"])
        all_results: Dict[Tuple[int, ...], Tuple[str, float]] = {}
        for mode_index, mode in enumerate(self.PLACEMENT_MODES):
            beams: List[Tuple[float, Tuple[int, ...], Dict[int, float], Dict[int, float]]] = [
                (0.0, tuple(), {}, {})
            ]
            for stage, (cpu_need, memory_need, vnf_type) in enumerate(
                zip(cpu, memory, vnf_types)
            ):
                expanded = []
                for score, chain, cpu_used, memory_used in beams:
                    previous = source if not chain else chain[-1]
                    for dc in self.dc_nodes:
                        instance_key = (dc, vnf_type)
                        reused = instance_key in snapshot.vnf_instances or any(
                            chain[index] == dc and vnf_types[index] == vnf_type
                            for index in range(len(chain))
                        )
                        cpu_increment = 0.0 if reused else cpu_need
                        memory_increment = 0.0 if reused else memory_need
                        next_cpu = cpu_used.get(dc, 0.0) + cpu_increment
                        next_memory = memory_used.get(dc, 0.0) + memory_increment
                        cpu_remaining = snapshot.cpu_remaining.get(dc, 0.0)
                        memory_remaining = snapshot.memory_remaining.get(dc, 0.0)
                        if next_cpu > cpu_remaining + 1e-9:
                            continue
                        if next_memory > memory_remaining + 1e-9:
                            continue
                        distance = float(self.hops.get(previous, {}).get(dc, 10_000))
                        if distance >= 10_000:
                            continue
                        pressure = max(
                            next_cpu / max(cpu_remaining, 1e-9),
                            next_memory / max(memory_remaining, 1e-9),
                        )
                        repeated = float(dc in chain)
                        jitter = self._jitter(int(request["id"]), mode_index, stage, dc)
                        if mode == "compact":
                            increment = distance + 1.5 * pressure - 0.75 * repeated + jitter
                        elif mode == "spread":
                            increment = distance + 7.0 * pressure + 2.0 * repeated + jitter
                        elif mode == "residual":
                            increment = 0.5 * distance + 14.0 * pressure + jitter
                        else:
                            increment = distance + 4.0 * pressure + 0.5 * repeated + jitter
                        new_cpu = dict(cpu_used)
                        new_memory = dict(memory_used)
                        new_cpu[dc] = next_cpu
                        new_memory[dc] = next_memory
                        expanded.append(
                            (score + increment, chain + (dc,), new_cpu, new_memory)
                        )
                expanded.sort(key=lambda row: (row[0], row[1]))
                beams = expanded[: self.placement_beam]
                if not beams:
                    break
            for score, chain, _, _ in beams[: self.placement_chains]:
                previous = all_results.get(chain)
                if previous is None or score < previous[1]:
                    all_results[chain] = (mode, score)
        rows = [
            (chain, mode, score)
            for chain, (mode, score) in all_results.items()
        ]
        rows.sort(key=lambda row: (row[2], row[0]))
        return rows[: self.placement_chains]

    @staticmethod
    def _jitter(request_id: int, mode: int, stage: int, node: int) -> float:
        value = (
            request_id * 1_000_003
            + mode * 97_409
            + stage * 7_919
            + node * 101
        ) % 10_000
        return value / 100_000.0

    def _edge_weight(
        self,
        snapshot: ResourceSnapshot,
        bandwidth: float,
        mode: str,
        request_id: int,
        variant: int,
    ):
        def weight(u: int, v: int, data: Mapping[str, Any]) -> Optional[float]:
            remaining = float(snapshot.bandwidth_remaining.get((int(u), int(v)), 0.0))
            if remaining + 1e-9 < bandwidth:
                return None
            delay = float(data.get("delay_ms", 1.0))
            pressure = bandwidth / max(remaining, 1e-9)
            if mode == "hop":
                base = 1.0
            elif mode == "residual":
                base = 0.25 * delay + 12.0 * pressure
            elif mode == "spread":
                base = delay + 25.0 * pressure * pressure
            else:
                base = delay + 2.0 * pressure
            jitter = (
                request_id * 65_537 + variant * 4_099 + int(u) * 131 + int(v) * 17
            ) % 1_000
            return base + jitter / 100_000.0

        return weight

    def _shortest_path(
        self,
        source: int,
        target: int,
        weight,
    ) -> Optional[List[int]]:
        if source == target:
            return [source]
        try:
            return [int(node) for node in nx.shortest_path(self.graph, source, target, weight=weight)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _build_plan(
        self,
        request: Mapping[str, Any],
        chain: Sequence[int],
        snapshot: ResourceSnapshot,
        routing_mode: str,
        variant: int,
        path_finder=None,
    ) -> Optional[Dict[str, Any]]:
        request_id = int(request["id"])
        bandwidth = float(request["bw_origin"])
        tree_growth = routing_mode.startswith("tree_")
        edge_mode = routing_mode[5:] if tree_growth else routing_mode
        weight = self._edge_weight(
            snapshot, bandwidth, edge_mode, request_id, variant
        )
        terminals = [int(request["source_dpid"]), *map(int, chain)]
        spine_paths = []
        for source, target in zip(terminals, terminals[1:]):
            path = (
                path_finder(source, target)
                if path_finder is not None
                else self._shortest_path(source, target, weight)
            )
            if path is None:
                return None
            spine_paths.append(path)

        root = int(chain[-1])
        destinations = [int(value) for value in request["destination_dpids"]]
        if tree_growth and path_finder is None:
            branch_paths = self._grow_multicast_paths(root, destinations, weight)
            if branch_paths is None:
                return None
        else:
            branch_paths = {}
            for destination in destinations:
                path = (
                    path_finder(root, destination)
                    if path_finder is not None
                    else self._shortest_path(root, destination, weight)
                )
                if path is None:
                    return None
                branch_paths[str(destination)] = path
        outputs, tree_edges = multicast_outputs(branch_paths, self.ports, self.nodes)

        chain_length = len(chain)
        base_port = self.stage_port_base + (request_id - 1) * chain_length
        if base_port + chain_length - 1 > 65_535:
            raise ValueError("stage UDP port range exceeds 65535")
        segments = []
        placements = {}
        for stage, path in enumerate(spine_paths):
            target = int(chain[stage])
            target_ip = str(self.nodes[target]["host_ip"]).split("/")[0]
            stage_port = base_port + stage
            segments.append({
                "stage": stage,
                "from_dpid": int(path[0]),
                "to_dpid": target,
                "target_ip": target_ip,
                "udp_port": stage_port,
                "path": path,
                "switch_outputs": path_outputs(path, self.ports, self.nodes),
            })
            placements[str(stage)] = {
                "dc_node": target,
                "vnf_type": int(request["vnf"][stage]),
                "cpu_units": float(request["cpu_origin"][stage]),
                "memory_units": float(request["memory_origin"][stage]),
                "listen_ip": target_ip,
                "listen_port": stage_port,
            }
        return {
            "version": "deployment_topk_candidate_v3",
            "request_id": request_id,
            "accepted": True,
            "source_dpid": int(request["source_dpid"]),
            "destination_dpids": [int(value) for value in request["destination_dpids"]],
            "chain_nodes": [int(value) for value in chain],
            "placement_by_vnf": placements,
            "segments": segments,
            "multicast": {
                "root_dpid": root,
                "dst_ip": str(request["multicast_ip"]),
                "udp_port": int(request["udp_port"]),
                "paths": branch_paths,
                "tree_edges": tree_edges,
                "switch_outputs": outputs,
            },
            "candidate_generation": {
                "routing_mode": routing_mode,
                "snapshot_version": int(snapshot.version),
            },
        }

    def _grow_multicast_paths(
        self,
        root: int,
        destinations: Sequence[int],
        weight,
    ) -> Optional[Dict[str, List[int]]]:
        """Grow one directed shared tree by cheapest incremental attachment."""

        tree = nx.DiGraph()
        tree.add_node(int(root))
        reachable = {int(root)}
        pending = set(map(int, destinations)) - reachable

        while pending:
            best = None
            for start in sorted(reachable):
                for destination in sorted(pending):
                    path = self._shortest_path(start, destination, weight)
                    if path is None:
                        continue
                    new_edges = [
                        (u, v) for u, v in zip(path, path[1:])
                        if not tree.has_edge(u, v)
                    ]
                    incremental_cost = 0.0
                    feasible = True
                    for u, v in new_edges:
                        value = weight(u, v, self.graph[u][v])
                        if value is None:
                            feasible = False
                            break
                        incremental_cost += float(value)
                    if not feasible:
                        continue
                    key = (
                        incremental_cost,
                        len(new_edges),
                        len(path),
                        destination,
                        start,
                        tuple(path),
                    )
                    if best is None or key < best[0]:
                        best = (key, path)
            if best is None:
                return None
            path = best[1]
            tree.add_edges_from(zip(path, path[1:]))
            reachable.update(path)
            pending.difference_update(reachable)

        paths: Dict[str, List[int]] = {}
        for destination in map(int, destinations):
            try:
                paths[str(destination)] = [
                    int(node)
                    for node in nx.shortest_path(tree, int(root), destination)
                ]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
        return paths

    def _candidate(
        self,
        request: Mapping[str, Any],
        plan: Dict[str, Any],
        footprint: ResourceFootprint,
        source: str,
        snapshot: ResourceSnapshot,
    ) -> DeploymentCandidate:
        metrics = self._metrics(request, plan, footprint, snapshot)
        objective = (
            self.objective_weights["delay"] * metrics["estimated_delay_ms"]
            + self.objective_weights["cpu"] * metrics["cpu_units"]
            + self.objective_weights["memory"] * metrics["memory_units"]
            + self.objective_weights["bandwidth"] * metrics["bandwidth_mbps"]
            + self.objective_weights["pressure"] * metrics["peak_resource_pressure"]
            + 0.15 * metrics["flowmod_estimate"]
            + 0.10 * metrics["tree_edges"]
        )
        signature = plan_signature(plan)
        return DeploymentCandidate(
            candidate_id=f"r{int(request['id'])}-{signature[:16]}",
            source=source,
            plan=plan,
            footprint=footprint,
            metrics=metrics,
            objective=float(objective),
        )

    def _metrics(
        self,
        request: Mapping[str, Any],
        plan: Mapping[str, Any],
        footprint: ResourceFootprint,
        snapshot: ResourceSnapshot,
    ) -> Dict[str, float]:
        segment_delay = 0.0
        segment_hops = 0
        flow_switches = set()
        for segment in plan["segments"]:
            path = list(map(int, segment["path"]))
            segment_hops += max(0, len(path) - 1)
            segment_delay += self._path_delay(path)
            flow_switches.update(path)
        multicast_paths = (plan.get("multicast") or {}).get("paths", {})
        branch_delay = max(
            (self._path_delay(list(map(int, path))) for path in multicast_paths.values()),
            default=0.0,
        )
        tree_edges = len((plan.get("multicast") or {}).get("tree_edges", []))
        flow_switches.update(
            int(node) for node in (plan.get("multicast") or {}).get("switch_outputs", {})
        )

        def pressure(demand: Mapping[Any, float], remaining: Mapping[Any, float]) -> float:
            return max(
                (
                    amount / max(float(remaining.get(resource, 0.0)), 1e-9)
                    for resource, amount in demand.items()
                ),
                default=0.0,
            )

        return {
            "estimated_delay_ms": float(segment_delay + branch_delay),
            "delay_bound_ms": float(request.get("delay_bound_ms", math.inf)),
            "cpu_units": float(sum(footprint.cpu.values())),
            "memory_units": float(sum(footprint.memory.values())),
            "bandwidth_mbps": float(sum(footprint.bandwidth.values())),
            "segment_hops": float(segment_hops),
            "tree_edges": float(tree_edges),
            "flowmod_estimate": float(len(flow_switches)),
            "peak_resource_pressure": max(
                pressure(footprint.cpu, snapshot.cpu_remaining),
                pressure(footprint.memory, snapshot.memory_remaining),
                pressure(footprint.bandwidth, snapshot.bandwidth_remaining),
            ),
        }

    def _path_delay(self, path: Sequence[int]) -> float:
        return sum(self.edge_delay[(int(u), int(v))] for u, v in zip(path, path[1:]))

    def _select_diverse(
        self,
        ranked: Sequence[DeploymentCandidate],
    ) -> List[DeploymentCandidate]:
        if len(ranked) <= self.max_candidates:
            return list(ranked)
        selected = [ranked[0]]
        remaining = list(ranked[1:])
        while remaining and len(selected) < self.max_candidates:
            best_index = 0
            best_key = None
            for index, candidate in enumerate(remaining):
                diversity = min(
                    candidate_distance(candidate.plan, chosen.plan)
                    for chosen in selected
                )
                normalized_cost = candidate.objective / max(ranked[0].objective, 1e-9)
                key = (diversity - 0.05 * normalized_cost, -candidate.objective)
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
            selected.append(remaining.pop(best_index))
        selected.sort(key=lambda row: (row.objective, row.candidate_id))
        return selected


def candidate_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    first_chain = tuple(map(int, first.get("chain_nodes", [])))
    second_chain = tuple(map(int, second.get("chain_nodes", [])))
    chain_difference = sum(a != b for a, b in zip(first_chain, second_chain))
    chain_denominator = max(1, len(first_chain), len(second_chain))
    first_edges = {
        tuple(map(int, edge)) for edge in (first.get("multicast") or {}).get("tree_edges", [])
    }
    second_edges = {
        tuple(map(int, edge)) for edge in (second.get("multicast") or {}).get("tree_edges", [])
    }
    union = first_edges | second_edges
    tree_distance = 1.0 - len(first_edges & second_edges) / max(1, len(union))
    return 0.6 * chain_difference / chain_denominator + 0.4 * tree_distance


def validate_complete_plan(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    request_id = int(request["id"])
    if int(plan.get("request_id", -1)) != request_id:
        raise ValueError("request id mismatch")
    if not plan.get("accepted", True):
        raise ValueError("candidate plan is rejected")
    chain = [int(value) for value in plan.get("chain_nodes", [])]
    if len(chain) != len(request["vnf"]):
        raise ValueError("VNF chain length mismatch")
    dc_nodes = set(map(int, profile["dc_nodes_1based"]))
    if any(node not in dc_nodes for node in chain):
        raise ValueError("VNF placed on a non-DC node")
    edge_set = set()
    for raw in profile["edges"]:
        u, v = int(raw["u"]), int(raw["v"])
        edge_set.add((u, v))
        edge_set.add((v, u))

    segments = sorted(plan.get("segments", []), key=lambda row: int(row["stage"]))
    if len(segments) != len(chain):
        raise ValueError("segment count mismatch")
    expected_from = int(request["source_dpid"])
    for stage, segment in enumerate(segments):
        if int(segment["stage"]) != stage:
            raise ValueError("non-contiguous segment stages")
        path = [int(value) for value in segment.get("path", [])]
        if not path or path[0] != expected_from or path[-1] != chain[stage]:
            raise ValueError("segment endpoint mismatch")
        if any((u, v) not in edge_set for u, v in zip(path, path[1:])):
            raise ValueError("segment uses a nonexistent edge")
        placement = plan.get("placement_by_vnf", {}).get(str(stage))
        if not placement or int(placement["dc_node"]) != chain[stage]:
            raise ValueError("placement/chain mismatch")
        if int(placement["vnf_type"]) != int(request["vnf"][stage]):
            raise ValueError("VNF type mismatch")
        expected_from = chain[stage]

    multicast = plan.get("multicast") or {}
    if int(multicast.get("root_dpid", -1)) != chain[-1]:
        raise ValueError("multicast root is not the last VNF")
    paths = multicast.get("paths") or {}
    expected_destinations = set(map(int, request["destination_dpids"]))
    if set(map(int, paths)) != expected_destinations:
        raise ValueError("multicast destinations mismatch")
    tree_edges = {tuple(map(int, edge)) for edge in multicast.get("tree_edges", [])}
    tree = nx.DiGraph()
    tree.add_edges_from(tree_edges)
    if not nx.is_directed_acyclic_graph(tree):
        raise ValueError("multicast tree contains a directed cycle")
    for destination in expected_destinations:
        path = [int(value) for value in paths[str(destination)]]
        if not path or path[0] != chain[-1] or path[-1] != destination:
            raise ValueError("multicast path endpoint mismatch")
        if any((u, v) not in edge_set for u, v in zip(path, path[1:])):
            raise ValueError("multicast path uses a nonexistent edge")
        if any((u, v) not in tree_edges for u, v in zip(path, path[1:])):
            raise ValueError("multicast path is absent from tree_edges")
    reachable = nx.descendants(tree, chain[-1]) | {chain[-1]} if tree_edges else {chain[-1]}
    if not expected_destinations <= reachable:
        raise ValueError("multicast tree disconnects a destination")
