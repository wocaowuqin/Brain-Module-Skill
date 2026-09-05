#!/usr/bin/env python3
"""Generate residual-bandwidth trees and optional runtime reroute events."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.trainable_role_agents import (
    TrainableSFTSelectionAgent,
    TrainableTreeRerouteAgent,
    _metrics_vector,
    _proposal_vector,
)


Edge = tuple[int, int]


@dataclass
class ActiveTree:
    request: dict[str, Any]
    edges: set[Edge]
    paths: dict[str, list[int]]
    reconfig_count: int = 0


@dataclass
class Candidate:
    request_id: int
    old_edge: Edge
    new_path: list[int]
    estimated_gain: float
    old_utilization: float
    max_new_utilization: float
    extra_edges: int
    old_delay_ms: float
    new_path_delay_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "old_edge": list(self.old_edge),
            "new_path": self.new_path,
            "estimated_gain": self.estimated_gain,
            "old_utilization": self.old_utilization,
            "max_new_utilization": self.max_new_utilization,
            "utilization_drop": self.old_utilization - self.max_new_utilization,
            "extra_edges": self.extra_edges,
            "old_delay_ms": self.old_delay_ms,
            "new_path_delay_ms": self.new_path_delay_ms,
        }

    def action(self) -> SimpleNamespace:
        return SimpleNamespace(
            req_id=self.request_id,
            action_type="reroute_edge",
            estimated_gain=self.estimated_gain,
            target={
                "old_edge": list(self.old_edge),
                "new_path": self.new_path,
                "safety": {
                    "old_utilization": self.old_utilization,
                    "max_new_utilization": self.max_new_utilization,
                    "extra_edges": self.extra_edges,
                    "old_edge_delay_ms": self.old_delay_ms,
                    "new_path_delay_ms": self.new_path_delay_ms,
                },
            },
        )


class DynamicTreePlanner:
    def __init__(
        self,
        profile: dict[str, Any],
        hotspot_threshold: float,
        max_projected_utilization: float,
        utilization_weight: float,
        penalty_threshold: float,
        penalty_exponent: float,
        max_reroute_hops: int,
        max_tree_edge_growth: int,
        max_delay_ratio: float,
        max_delay_increase_ms: float,
    ) -> None:
        self.profile = profile
        self.nodes = {int(row["dpid"]): row for row in profile["nodes"]}
        self.adjacency: dict[int, list[int]] = {node: [] for node in self.nodes}
        self.capacity: dict[Edge, float] = {}
        self.delay: dict[Edge, float] = {}
        self.port: dict[Edge, int] = {}
        for row in profile["edges"]:
            u, v = int(row["u"]), int(row["v"])
            capacity = float(row.get("bandwidth_mbps", profile["default_bandwidth_mbps"]))
            delay = float(row.get("delay_ms", profile["default_delay_ms"]))
            self.adjacency[u].append(v)
            self.adjacency[v].append(u)
            self.capacity[(u, v)] = self.capacity[(v, u)] = capacity
            self.delay[(u, v)] = self.delay[(v, u)] = delay
            self.port[(u, v)] = int(row["u_port"])
            self.port[(v, u)] = int(row["v_port"])
        for node in self.adjacency:
            self.adjacency[node].sort()
        self.load = {edge: 0.0 for edge in self.capacity}
        self.active: dict[int, ActiveTree] = {}
        self.hotspot_threshold = float(hotspot_threshold)
        self.max_projected_utilization = float(max_projected_utilization)
        self.utilization_weight = float(utilization_weight)
        self.penalty_threshold = float(penalty_threshold)
        self.penalty_exponent = float(penalty_exponent)
        self.max_reroute_hops = int(max_reroute_hops)
        self.max_tree_edge_growth = int(max_tree_edge_growth)
        self.max_delay_ratio = float(max_delay_ratio)
        self.max_delay_increase_ms = float(max_delay_increase_ms)
        self.capacity_fallbacks = 0
        self.max_observed_utilization = 0.0

    def utilization(self, edge: Edge, extra: float = 0.0) -> float:
        return (self.load[edge] + float(extra)) / max(self.capacity[edge], 1e-9)

    def _shortest_parent_tree(
        self, source: int, bw: float, utilization_limit: float | None
    ) -> tuple[dict[int, int], dict[int, float]]:
        distances = {int(source): 0.0}
        parent: dict[int, int] = {}
        queue = [(0.0, int(source))]
        while queue:
            distance, current = heapq.heappop(queue)
            if distance > distances.get(current, math.inf) + 1e-12:
                continue
            for neighbor in self.adjacency[current]:
                edge = (current, neighbor)
                projected = self.utilization(edge, bw)
                if utilization_limit is not None and projected > utilization_limit + 1e-12:
                    continue
                cost = (
                    self.delay[edge]
                    + self.utilization_weight
                    * max(0.0, projected - self.penalty_threshold)
                    ** self.penalty_exponent
                    + 1e-6 * neighbor
                )
                candidate = distance + cost
                old = distances.get(neighbor, math.inf)
                if candidate + 1e-12 < old:
                    distances[neighbor] = candidate
                    parent[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        return parent, distances

    def build_initial_tree(self, request: dict[str, Any]) -> ActiveTree:
        source = int(request["source_dpid"])
        destinations = sorted({int(value) for value in request["destination_dpids"]})
        bw = float(request["bw_origin"])
        parent: dict[int, int] | None = None
        distances: dict[int, float] = {}
        for limit in (self.max_projected_utilization, 1.0, None):
            candidate_parent, candidate_distances = self._shortest_parent_tree(
                source, bw, limit
            )
            if all(destination in candidate_distances for destination in destinations):
                parent, distances = candidate_parent, candidate_distances
                if limit != self.max_projected_utilization:
                    self.capacity_fallbacks += 1
                break
        if parent is None:
            raise ValueError(f"request {request['id']} has unreachable destinations")

        edges: set[Edge] = set()
        paths: dict[str, list[int]] = {}
        for destination in destinations:
            reverse = [destination]
            current = destination
            while current != source:
                previous = parent[current]
                edges.add((previous, current))
                reverse.append(previous)
                current = previous
            paths[str(destination)] = list(reversed(reverse))
        tree = ActiveTree(request=request, edges=edges, paths=paths)
        self._validate_tree(tree)
        return tree

    def build_shortest_tree(self, request: dict[str, Any]) -> ActiveTree:
        source = int(request["source_dpid"])
        destinations = sorted({int(value) for value in request["destination_dpids"]})
        parent: dict[int, int] = {}
        visited = {source}
        queue = [source]
        for current in queue:
            for neighbor in self.adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
        if any(destination not in visited for destination in destinations):
            raise ValueError(f"request {request['id']} has unreachable destinations")
        edges: set[Edge] = set()
        paths: dict[str, list[int]] = {}
        for destination in destinations:
            reverse = [destination]
            current = destination
            while current != source:
                previous = parent[current]
                edges.add((previous, current))
                reverse.append(previous)
                current = previous
            paths[str(destination)] = list(reversed(reverse))
        tree = ActiveTree(request=request, edges=edges, paths=paths)
        self._validate_tree(tree)
        return tree

    def install(self, tree: ActiveTree) -> None:
        request_id = int(tree.request["id"])
        if request_id in self.active:
            raise ValueError(f"request {request_id} is already active")
        bw = float(tree.request["bw_origin"])
        for edge in tree.edges:
            self.load[edge] += bw
            self.max_observed_utilization = max(
                self.max_observed_utilization, self.utilization(edge)
            )
        self.active[request_id] = tree

    def remove(self, request_id: int) -> None:
        tree = self.active.pop(int(request_id), None)
        if tree is None:
            return
        bw = float(tree.request["bw_origin"])
        for edge in tree.edges:
            self.load[edge] -= bw
            if self.load[edge] < -1e-8:
                raise ValueError(f"negative bandwidth ledger on {edge}")
            self.load[edge] = max(0.0, self.load[edge])

    @staticmethod
    def _descendants(edges: Iterable[Edge], root: int) -> set[int]:
        children: dict[int, list[int]] = {}
        for u, v in edges:
            children.setdefault(u, []).append(v)
        result: set[int] = set()
        stack = [int(root)]
        while stack:
            node = stack.pop()
            if node in result:
                continue
            result.add(node)
            stack.extend(children.get(node, []))
        return result

    def _alternate_path(
        self,
        source: int,
        target: int,
        bw: float,
        forbidden_internal: set[int],
        avoid_edge: Edge,
    ) -> list[int] | None:
        queue = [(0.0, 0, int(source), [int(source)])]
        best: dict[tuple[int, int], float] = {(int(source), 0): 0.0}
        while queue:
            cost, hops, current, path = heapq.heappop(queue)
            if current == int(target):
                return path
            if hops >= self.max_reroute_hops:
                continue
            for neighbor in self.adjacency[current]:
                edge = (current, neighbor)
                if edge == avoid_edge or neighbor in path:
                    continue
                if neighbor != int(target) and neighbor in forbidden_internal:
                    continue
                projected = self.utilization(edge, bw)
                if projected > self.max_projected_utilization + 1e-12:
                    continue
                next_hops = hops + 1
                next_cost = (
                    cost
                    + self.delay[edge]
                    + self.utilization_weight
                    * max(0.0, projected - self.penalty_threshold)
                    ** self.penalty_exponent
                )
                key = (neighbor, next_hops)
                if next_cost + 1e-12 >= best.get(key, math.inf):
                    continue
                best[key] = next_cost
                heapq.heappush(queue, (next_cost, next_hops, neighbor, path + [neighbor]))
        return None

    def candidates(self, request_id: int, limit: int = 4) -> list[Candidate]:
        tree = self.active.get(int(request_id))
        if tree is None:
            return []
        bw = float(tree.request["bw_origin"])
        all_nodes = {node for edge in tree.edges for node in edge}
        all_nodes.add(int(tree.request["source_dpid"]))
        candidates: dict[tuple[Edge, tuple[int, ...]], Candidate] = {}
        ranked_edges = sorted(
            tree.edges,
            key=lambda edge: (-self.utilization(edge), edge),
        )
        for old_edge in ranked_edges:
            old_util = self.utilization(old_edge)
            if old_util + 1e-12 < self.hotspot_threshold:
                break
            child = old_edge[1]
            subtree = self._descendants(tree.edges, child)
            remaining = tree.edges - {old_edge}
            main_nodes = (all_nodes - subtree) | {old_edge[0]}
            anchors = sorted(main_nodes, key=lambda node: (node != old_edge[0], node))
            for anchor in anchors:
                forbidden = all_nodes - {anchor, child}
                path = self._alternate_path(
                    anchor, child, bw, forbidden, old_edge
                )
                if path is None:
                    continue
                path_edges = set(zip(path, path[1:]))
                extra_edges = len(path_edges) - 1
                if extra_edges > self.max_tree_edge_growth:
                    continue
                old_delay = self.delay[old_edge]
                new_delay = sum(self.delay[edge] for edge in path_edges)
                if new_delay > old_delay * self.max_delay_ratio + self.max_delay_increase_ms + 1e-12:
                    continue
                new_edges = remaining | path_edges
                if not self._is_directed_tree(
                    new_edges, int(tree.request["source_dpid"])
                ):
                    continue
                max_new_util = max(self.utilization(edge, bw) for edge in path_edges)
                gain = (
                    old_util
                    - max_new_util
                    - 0.02 * max(0, extra_edges)
                    - 0.01 * max(0.0, new_delay - old_delay)
                )
                if gain <= 1e-12:
                    continue
                candidate = Candidate(
                    request_id=int(request_id),
                    old_edge=old_edge,
                    new_path=path,
                    estimated_gain=float(gain),
                    old_utilization=float(old_util),
                    max_new_utilization=float(max_new_util),
                    extra_edges=int(extra_edges),
                    old_delay_ms=float(old_delay),
                    new_path_delay_ms=float(new_delay),
                )
                key = (candidate.old_edge, tuple(candidate.new_path))
                previous = candidates.get(key)
                if previous is None or candidate.estimated_gain > previous.estimated_gain:
                    candidates[key] = candidate
        ranked = sorted(
            candidates.values(),
            key=lambda value: (
                -value.estimated_gain,
                value.extra_edges,
                value.new_path_delay_ms,
                value.request_id,
                value.old_edge,
                value.new_path,
            ),
        )
        return ranked[: max(1, int(limit))]

    def best_candidate(self, request_id: int) -> Candidate | None:
        candidates = self.candidates(request_id, limit=1)
        return candidates[0] if candidates else None

    def apply_candidate(self, candidate: Candidate) -> ActiveTree:
        tree = self.active[candidate.request_id]
        bw = float(tree.request["bw_origin"])
        new_path_edges = set(zip(candidate.new_path, candidate.new_path[1:]))
        new_edges = (tree.edges - {candidate.old_edge}) | new_path_edges
        for edge in new_path_edges:
            self.load[edge] += bw
            self.max_observed_utilization = max(
                self.max_observed_utilization, self.utilization(edge)
            )
        self.load[candidate.old_edge] -= bw
        self.load[candidate.old_edge] = max(0.0, self.load[candidate.old_edge])
        tree.edges = new_edges
        tree.paths = self._paths_from_edges(tree)
        tree.reconfig_count += 1
        self._validate_tree(tree)
        return tree

    def outputs(self, tree: ActiveTree) -> dict[str, list[int]]:
        outputs: dict[int, set[int]] = {}
        for u, v in tree.edges:
            outputs.setdefault(u, set()).add(self.port[(u, v)])
        for destination in tree.request["destination_dpids"]:
            dpid = int(destination)
            outputs.setdefault(dpid, set()).add(int(self.nodes[dpid]["host_port"]))
        return {
            str(dpid): sorted(ports) for dpid, ports in sorted(outputs.items())
        }

    def risks(self) -> list[SimpleNamespace]:
        values = []
        for request_id, tree in self.active.items():
            hot_score = sum(
                self.utilization(edge)
                for edge in tree.edges
                if self.utilization(edge) >= self.hotspot_threshold
            )
            delay_estimate = max(
                (
                    sum(self.delay[edge] for edge in zip(path, path[1:]))
                    for path in tree.paths.values()
                ),
                default=0.0,
            )
            values.append(
                SimpleNamespace(
                    req_id=request_id,
                    score=float(hot_score + 0.03 * tree.reconfig_count),
                    link_hot_score=float(hot_score),
                    node_hot_score=0.0,
                    delay_estimate=float(delay_estimate),
                    migration_count=0,
                    reconfig_count=tree.reconfig_count,
                )
            )
        values.sort(key=lambda item: (-item.score, item.req_id))
        return values

    def metrics(self) -> dict[str, Any]:
        delays = []
        for tree in self.active.values():
            delays.append(
                max(
                    (
                        sum(self.delay[edge] for edge in zip(path, path[1:]))
                        for path in tree.paths.values()
                    ),
                    default=0.0,
                )
            )
        return {
            "active_sfts": len(self.active),
            "node_hotspots": 0,
            "link_hotspots": sum(
                self.utilization(edge) >= self.hotspot_threshold
                for edge in self.load
            ),
            "total_tree_edges": sum(len(tree.edges) for tree in self.active.values()),
            "avg_delay_total_ms": sum(delays) / len(delays) if delays else 0.0,
            "avg_queueing_delay_ms": 0.0,
            "max_delay_total_ms": max(delays) if delays else 0.0,
        }

    def _paths_from_edges(self, tree: ActiveTree) -> dict[str, list[int]]:
        children: dict[int, list[int]] = {}
        for u, v in tree.edges:
            children.setdefault(u, []).append(v)
        parent = {v: u for u, v in tree.edges}
        source = int(tree.request["source_dpid"])
        paths = {}
        for destination in tree.request["destination_dpids"]:
            current = int(destination)
            reverse = [current]
            while current != source:
                if current not in parent:
                    raise ValueError(
                        f"request {tree.request['id']} destination {destination} disconnected"
                    )
                current = parent[current]
                reverse.append(current)
            paths[str(int(destination))] = list(reversed(reverse))
        return paths

    @staticmethod
    def _is_directed_tree(edges: Iterable[Edge], root: int) -> bool:
        edges = set(edges)
        parents: dict[int, int] = {}
        children: dict[int, list[int]] = {}
        for u, v in edges:
            if v == root or v in parents:
                return False
            parents[v] = u
            children.setdefault(u, []).append(v)
        visited: set[int] = set()
        stack = [int(root)]
        while stack:
            node = stack.pop()
            if node in visited:
                return False
            visited.add(node)
            stack.extend(children.get(node, []))
        return len(visited) == len({root} | {node for edge in edges for node in edge})

    def _validate_tree(self, tree: ActiveTree) -> None:
        source = int(tree.request["source_dpid"])
        if not self._is_directed_tree(tree.edges, source):
            raise ValueError(f"request {tree.request['id']} is not a rooted directed tree")
        paths = self._paths_from_edges(tree)
        if paths != tree.paths:
            raise ValueError(f"request {tree.request['id']} path/tree mismatch")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_requests(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_qmix(checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selector = TrainableSFTSelectionAgent(top_k=3, epsilon=0.0, device="cpu")
    rerouter = TrainableTreeRerouteAgent(epsilon=0.0, device="cpu")
    selector.load_state_dict(state["selector"])
    rerouter.load_state_dict(state["reroute"])
    selector.config.epsilon = 0.0
    rerouter.config.epsilon = 0.0
    return selector, rerouter


def plan_policy(
    policy: str,
    profile: dict[str, Any],
    requests: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if policy == "greedy":
        utilization_weight = args.utilization_weight
        penalty_threshold = args.greedy_penalty_threshold
        penalty_exponent = args.greedy_penalty_exponent
    else:
        utilization_weight = args.reroute_utilization_weight
        penalty_threshold = args.reroute_penalty_threshold
        penalty_exponent = args.reroute_penalty_exponent
    planner = DynamicTreePlanner(
        profile,
        args.hotspot_threshold,
        args.max_projected_utilization,
        utilization_weight,
        penalty_threshold,
        penalty_exponent,
        args.max_reroute_hops,
        args.max_tree_edge_growth,
        args.max_delay_ratio,
        args.max_delay_increase_ms,
    )
    selector = rerouter = None
    if policy == "qmix":
        selector, rerouter = load_qmix(Path(args.qmix_checkpoint))

    plans: list[dict[str, Any]] = []
    reroutes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    request_by_id = {int(row["id"]): row for row in requests}
    events = []
    for request in requests:
        events.append((float(request["arrival_time"]), 1, "arrive", int(request["id"])))
        events.append((float(request["leave_time"]), 0, "leave", int(request["id"])))
    if policy in {"local", "qmix"}:
        start = min(float(row["arrival_time"]) for row in requests)
        end = max(float(row["leave_time"]) for row in requests)
        tick = math.ceil(start / args.reconfig_interval) * args.reconfig_interval
        while tick < end:
            events.append((tick, 2, "reconfigure", -1))
            tick += args.reconfig_interval
    events.sort()

    for event_time, _, event_type, request_id in events:
        if event_type == "arrive":
            tree = (
                planner.build_initial_tree(request_by_id[request_id])
                if policy == "greedy"
                else planner.build_shortest_tree(request_by_id[request_id])
            )
            planner.install(tree)
            plans.append(
                {
                    "request_id": request_id,
                    "source": (
                        "residual_bandwidth_greedy"
                        if policy == "greedy"
                        else f"shortest_path_with_{policy}_hotspot_reroute"
                    ),
                    "switch_outputs": planner.outputs(tree),
                    "paths": tree.paths,
                    "tree_edges": [list(edge) for edge in sorted(tree.edges)],
                    "bw_mbps": float(tree.request["bw_origin"]),
                }
            )
            continue
        if event_type == "leave":
            planner.remove(request_id)
            continue
        risks = planner.risks()[:3]
        if not risks or risks[0].score <= 0.0:
            continue

        selected_request = None
        selector_action = None
        selector_q = None
        reroute_action = None
        reroute_q = None
        candidate = None
        candidate_options: list[Candidate] = []
        if policy == "local":
            candidates_by_request = {
                risk.req_id: planner.candidates(risk.req_id, args.max_candidates)
                for risk in risks
            }
            candidates = [
                values[0] for values in candidates_by_request.values() if values
            ]
            if candidates:
                candidate = max(
                    candidates,
                    key=lambda value: (value.estimated_gain, -value.request_id),
                )
                selected_request = candidate.request_id
                candidate_options = candidates_by_request[selected_request]
        else:
            metrics = planner.metrics()
            selector_obs = selector._vectorize(metrics, risks)
            valid_selector = [0] + list(range(1, len(risks) + 1))
            selector_action, selector_q = selector.select_discrete_action(
                selector_obs, valid_selector
            )
            if selector_action > 0:
                selected_request = risks[selector_action - 1].req_id
                candidate_options = planner.candidates(
                    selected_request, args.max_candidates
                )
                candidate = candidate_options[0] if candidate_options else None
            action_obj = candidate.action() if candidate is not None else None
            proposal_obs = _proposal_vector(metrics, action_obj)
            valid_reroute = [0, 1] if candidate is not None else [0]
            reroute_action, reroute_q = rerouter.select_discrete_action(
                proposal_obs, valid_reroute
            )
            if reroute_action == 0:
                candidate = None
            decisions.append(
                {
                    "time": event_time,
                    "top3_request_ids": [risk.req_id for risk in risks],
                    "selector_obs": selector_obs,
                    "selector_action": selector_action,
                    "selector_q": selector_q,
                    "selected_request_id": selected_request,
                    "reroute_obs": proposal_obs,
                    "reroute_action": reroute_action,
                    "reroute_q": reroute_q,
                    "candidate_available": action_obj is not None,
                    "candidate_count": len(candidate_options),
                    "candidate_options": [
                        value.to_dict() for value in candidate_options
                    ],
                }
            )
        if candidate is None:
            continue
        tree = planner.apply_candidate(candidate)
        reroutes.append(
            {
                "time": event_time,
                "request_id": candidate.request_id,
                "policy": policy,
                "switch_outputs": planner.outputs(tree),
                "paths": tree.paths,
                "old_edge": list(candidate.old_edge),
                "new_path": candidate.new_path,
                "estimated_gain": candidate.estimated_gain,
                "old_utilization": candidate.old_utilization,
                "max_new_utilization": candidate.max_new_utilization,
                "extra_edges": candidate.extra_edges,
                "tree_edges": [list(edge) for edge in sorted(tree.edges)],
                "selector_action": selector_action,
                "selector_q": selector_q,
                "reroute_action": reroute_action,
                "reroute_q": reroute_q,
                "candidate_count": len(candidate_options),
                "candidate_options": [
                    value.to_dict() for value in candidate_options
                ],
            }
        )

    if planner.active:
        raise ValueError(f"active ledger was not drained: {sorted(planner.active)}")
    if any(value < -1e-8 for value in planner.load.values()):
        raise ValueError("negative final bandwidth ledger")
    plan_path = output_dir / f"{policy}_plans.jsonl"
    reroute_path = output_dir / f"{policy}_reroutes.jsonl"
    decision_path = output_dir / f"{policy}_decisions.jsonl"
    write_jsonl(plan_path, plans)
    write_jsonl(reroute_path, reroutes)
    write_jsonl(decision_path, decisions)
    return {
        "policy": policy,
        "requests": len(requests),
        "plans": len(plans),
        "reroutes": len(reroutes),
        "qmix_decisions": len(decisions),
        "qmix_actions": sum(row.get("reroute_action") == 1 for row in decisions),
        "capacity_fallbacks": planner.capacity_fallbacks,
        "max_observed_utilization": planner.max_observed_utilization,
        "utilization_cost": {
            "weight": utilization_weight,
            "threshold": penalty_threshold,
            "exponent": penalty_exponent,
        },
        "plan_file": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "reroute_file": str(reroute_path),
        "reroute_sha256": sha256(reroute_path),
        "decision_file": str(decision_path),
        "decision_sha256": sha256(decision_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--profile", default="sdn/topologies/us_backbone_28.json")
    parser.add_argument(
        "--output", default="artifacts/runs/sdn/policy_plans/seed7301"
    )
    parser.add_argument("--policy", choices=("greedy", "local", "qmix", "all"), default="all")
    parser.add_argument("--skip-requests", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--hotspot-threshold", type=float, default=0.75)
    parser.add_argument("--max-projected-utilization", type=float, default=0.90)
    parser.add_argument("--utilization-weight", type=float, default=200.0)
    parser.add_argument("--greedy-penalty-threshold", type=float, default=0.65)
    parser.add_argument("--greedy-penalty-exponent", type=float, default=2.0)
    parser.add_argument("--reroute-utilization-weight", type=float, default=20.0)
    parser.add_argument("--reroute-penalty-threshold", type=float, default=0.0)
    parser.add_argument("--reroute-penalty-exponent", type=float, default=4.0)
    parser.add_argument("--reconfig-interval", type=float, default=1.0)
    parser.add_argument("--max-reroute-hops", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--max-tree-edge-growth", type=int, default=2)
    parser.add_argument("--max-delay-ratio", type=float, default=1.25)
    parser.add_argument("--max-delay-increase-ms", type=float, default=2.0)
    parser.add_argument(
        "--qmix-checkpoint",
        default="artifacts/runs/reconfiguration/role_marl_offline_v2/qmix/best_model.pth",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.skip_requests < 0 or args.max_requests < 0:
        raise ValueError("request limits must be non-negative")
    if not 0.0 < args.hotspot_threshold <= 1.0:
        raise ValueError("hotspot threshold must be in (0, 1]")
    if not 0.0 < args.max_projected_utilization <= 1.0:
        raise ValueError("projected utilization limit must be in (0, 1]")
    if args.reconfig_interval <= 0.0:
        raise ValueError("reconfiguration interval must be positive")
    if args.max_candidates <= 0:
        raise ValueError("max candidates must be positive")
    if not 0.0 <= args.greedy_penalty_threshold < args.max_projected_utilization:
        raise ValueError("greedy penalty threshold must be below the projected limit")
    if not 0.0 <= args.reroute_penalty_threshold < args.max_projected_utilization:
        raise ValueError("reroute penalty threshold must be below the projected limit")
    if args.greedy_penalty_exponent <= 0.0 or args.reroute_penalty_exponent <= 0.0:
        raise ValueError("utilization penalty exponents must be positive")
    request_path = Path(args.requests)
    if not request_path.is_absolute():
        request_path = ROOT / request_path
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    requests = read_requests(request_path)
    if args.skip_requests:
        requests = requests[args.skip_requests :]
    if args.max_requests:
        requests = requests[: args.max_requests]
    if not requests:
        raise ValueError("request selection is empty")
    checkpoint = Path(args.qmix_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    args.qmix_checkpoint = str(checkpoint)
    policies = ("greedy", "local", "qmix") if args.policy == "all" else (args.policy,)
    summaries = [
        plan_policy(policy, read_json(profile_path), requests, output_dir, args)
        for policy in policies
    ]
    summary = {
        "valid": True,
        "requests_file": str(request_path),
        "requests_sha256": sha256(request_path),
        "profile_file": str(profile_path),
        "profile_sha256": sha256(profile_path),
        "skip_requests": args.skip_requests,
        "max_requests": args.max_requests,
        "hotspot_threshold": args.hotspot_threshold,
        "max_projected_utilization": args.max_projected_utilization,
        "utilization_weight": args.utilization_weight,
        "greedy_penalty_threshold": args.greedy_penalty_threshold,
        "greedy_penalty_exponent": args.greedy_penalty_exponent,
        "reroute_utilization_weight": args.reroute_utilization_weight,
        "reroute_penalty_threshold": args.reroute_penalty_threshold,
        "reroute_penalty_exponent": args.reroute_penalty_exponent,
        "reconfig_interval": args.reconfig_interval,
        "max_candidates": args.max_candidates,
        "qmix_checkpoint": str(checkpoint),
        "qmix_checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        "policies": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
