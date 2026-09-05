#!/usr/bin/env python3
"""Replay generated SFT requests against Mininet/OVS/Ryu in real time."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import queue
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any


_SYSTEM_MONOTONIC_RESOLUTION = time.get_clock_info("monotonic").resolution
_PERF_COUNTER_RESOLUTION = time.get_clock_info("perf_counter").resolution
_RUNTIME_CLOCK = (
    "perf_counter"
    if _PERF_COUNTER_RESOLUTION < _SYSTEM_MONOTONIC_RESOLUTION
    else "monotonic"
)
if _RUNTIME_CLOCK == "perf_counter":
    # This Conda Windows build exposes a 15.625 ms monotonic clock, which is
    # too coarse for 2-5 ms online batching.  perf_counter is also monotonic.
    time.monotonic = time.perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdn.ryu_client import RyuSFTClient
from sdn.migration_safety import make_before_break_compatible
from core.marl.batch_deployment_wqmix import ResourceFootprint
from core.marl.deployment_topk import validate_complete_plan
from core.marl.migration_scheduler import MigrationTask, migration_execution_waves


DEFAULT_PACKET_OVERHEAD_BYTES = 42
_MININET_RUNTIME_CONNECTIONS = threading.local()
_MININET_RUNTIME_HOST = "127.0.0.1"


class _MigrationPolicyNoop(RuntimeError):
    """Internal control flow for an intentional WQMIX no-migration action."""


def bandwidth_mbps_to_pps(
    bandwidth_mbps: float,
    payload_bytes: int,
    packet_overhead_bytes: int = DEFAULT_PACKET_OVERHEAD_BYTES,
) -> float:
    """Convert a link-rate demand to UDP packets/s without exceeding it."""
    packet_bytes = int(payload_bytes) + int(packet_overhead_bytes)
    if bandwidth_mbps < 0.0 or payload_bytes <= 0 or packet_bytes <= 0:
        raise ValueError("invalid bandwidth or packet size")
    return float(bandwidth_mbps) * 1_000_000.0 / (packet_bytes * 8.0)


def load_json(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return json.loads(value.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    rows = []
    with value.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{value}:{line_number}: {exc}") from exc
    return rows


def runtime_host_dpids(
    profile: dict[str, Any],
    requests: list[dict[str, Any]],
    *,
    online_wqmix_data: str | None = None,
    sfc_plans: dict[int, dict[str, Any]] | None = None,
    sfc_candidate_plans: dict[int, list[dict[str, Any]]] | None = None,
) -> set[int]:
    """Return every host that can be used by the selected request workload."""
    selected_ids = {int(request["id"]) for request in requests}
    selected = {
        int(value)
        for request in requests
        for value in [request["source_dpid"], *request["destination_dpids"]]
    }
    if sfc_plans:
        for request_id in selected_ids:
            plan = sfc_plans.get(request_id)
            if plan and plan.get("accepted", True):
                selected.update(map(int, plan.get("chain_nodes", [])))
    if sfc_candidate_plans:
        for request_id in selected_ids:
            for plan in sfc_candidate_plans.get(request_id, []):
                if plan.get("accepted", True):
                    selected.update(map(int, plan.get("chain_nodes", [])))
    if online_wqmix_data:
        folder = Path(online_wqmix_data)
        if not folder.is_absolute():
            folder = ROOT / folder
        batches = folder / "batches.jsonl"
        if not batches.is_file():
            raise FileNotFoundError(
                f"active-host selection requires candidate batches: {batches}"
            )
        for batch in read_jsonl(batches):
            for agent in batch.get("agents", []):
                if int(agent.get("request_id", -1)) not in selected_ids:
                    continue
                for candidate in agent.get("candidates", []):
                    plan = candidate.get("plan") if isinstance(candidate, dict) else None
                    if plan and plan.get("accepted", True):
                        selected.update(map(int, plan.get("chain_nodes", [])))
    available = {int(node["dpid"]) for node in profile["nodes"]}
    missing = sorted(selected - available)
    if missing:
        raise ValueError(f"runtime hosts are outside the topology: {missing}")
    return selected


def topology_index(profile: dict[str, Any]):
    nodes = {int(node["dpid"]): node for node in profile["nodes"]}
    adjacency = defaultdict(list)
    for edge in profile["edges"]:
        u = int(edge["u"])
        v = int(edge["v"])
        adjacency[u].append((v, int(edge["u_port"])))
        adjacency[v].append((u, int(edge["v_port"])))
    for dpid in adjacency:
        adjacency[dpid].sort(key=lambda item: (item[0], item[1]))
    return nodes, adjacency


def shortest_segment(
    profile: dict[str, Any], source_dpid: int, target_dpid: int
) -> tuple[list[int], dict[str, list[int]]]:
    """Build one shortest-hop unicast segment using profile port mappings."""
    nodes, adjacency = topology_index(profile)
    source, target = int(source_dpid), int(target_dpid)
    if source not in nodes or target not in nodes:
        raise ValueError(f"segment endpoint is outside topology: {source}->{target}")
    parent: dict[int, tuple[int, int]] = {}
    queue_nodes = deque([source])
    visited = {source}
    while queue_nodes and target not in visited:
        current = queue_nodes.popleft()
        for neighbor, output_port in adjacency[current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = (current, output_port)
            queue_nodes.append(neighbor)
    if target not in visited:
        raise ValueError(f"topology has no path from {source} to {target}")
    path = [target]
    while path[-1] != source:
        path.append(parent[path[-1]][0])
    path.reverse()
    outputs: dict[str, list[int]] = {}
    for u, v in zip(path, path[1:]):
        output_port = next(port for neighbor, port in adjacency[u] if neighbor == v)
        outputs[str(u)] = [int(output_port)]
    outputs[str(target)] = [int(nodes[target]["host_port"])]
    return path, outputs


def segment_outputs_for_path(
    profile: dict[str, Any], path: list[int] | tuple[int, ...]
) -> dict[str, list[int]]:
    """Materialize switch outputs for an already selected path."""

    nodes, adjacency = topology_index(profile)
    resolved = [int(node) for node in path]
    if not resolved:
        raise ValueError("migration candidate segment has an empty path")
    if any(node not in nodes for node in resolved):
        raise ValueError(f"migration candidate path is outside topology: {resolved}")
    outputs: dict[str, list[int]] = {}
    for source, target in zip(resolved, resolved[1:]):
        ports = [
            int(port)
            for neighbor, port in adjacency.get(source, [])
            if int(neighbor) == target
        ]
        if not ports:
            raise ValueError(
                f"migration candidate uses a non-topology edge: {source}->{target}"
            )
        outputs[str(source)] = [ports[0]]
    outputs[str(resolved[-1])] = [int(nodes[resolved[-1]]["host_port"])]
    return outputs


def directed_edge_key(u: int, v: int) -> tuple[int, int]:
    """Return the forwarding direction used by Mininet's full-duplex link."""
    return (int(u), int(v))


def topology_bandwidth_caps(profile: dict[str, Any]) -> dict[tuple[int, int], float]:
    """Return per-direction link capacities for Mininet's full-duplex links."""
    capacities: dict[tuple[int, int], float] = {}
    default_capacity = float(profile.get("default_bandwidth_mbps", 0.0))
    for edge in profile.get("edges", []):
        u, v = int(edge["u"]), int(edge["v"])
        capacity = float(edge.get("bandwidth_mbps", default_capacity))
        if capacity <= 0.0:
            continue
        for key in ((u, v), (v, u)):
            previous = capacities.get(key)
            capacities[key] = capacity if previous is None else min(previous, capacity)
    return capacities


def plan_physical_edges(plan: dict[str, Any]) -> dict[tuple[int, int], int]:
    """Extract directed links and traversal multiplicity from an execution plan."""
    edges: Counter[tuple[int, int]] = Counter()

    def add_path(path: Any) -> None:
        if not isinstance(path, (list, tuple)):
            return
        values = [int(value) for value in path]
        for index in range(len(values) - 1):
            edges[directed_edge_key(values[index], values[index + 1])] += 1

    for segment in plan.get("segments", []) or []:
        add_path(segment.get("path", []))
    multicast = plan.get("multicast", {}) or {}
    tree_edges = multicast.get("tree_edges", []) or []
    if tree_edges:
        for edge in tree_edges:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                edges[directed_edge_key(int(edge[0]), int(edge[1]))] += 1
    else:
        multicast_edges: set[tuple[int, int]] = set()
        for path in (multicast.get("paths", {}) or {}).values():
            if isinstance(path, (list, tuple)):
                values = [int(value) for value in path]
                multicast_edges.update(
                    directed_edge_key(values[index], values[index + 1])
                    for index in range(len(values) - 1)
                )
        for edge in multicast_edges:
            edges[edge] += 1
    # A plain multicast plan exposes one path per receiver. Those paths share
    # a single physical tree, so reserve the stream once per unique edge.
    if not plan.get("segments") and not tree_edges:
        multicast_edges = set()
        for path in (plan.get("paths", {}) or {}).values():
            if isinstance(path, (list, tuple)):
                values = [int(value) for value in path]
                multicast_edges.update(
                    directed_edge_key(values[index], values[index + 1])
                    for index in range(len(values) - 1)
                )
        for edge in multicast_edges:
            edges[edge] = 1
    return dict(edges)


def migration_overlap_footprint(
    current_plan: dict[str, Any],
    replacement_plan: dict[str, Any],
    stage: int,
    bandwidth_mbps: float,
) -> ResourceFootprint:
    """Return resources simultaneously needed while old and new VNFs overlap."""

    stage = int(stage)
    placement = replacement_plan["placement_by_vnf"][str(stage)]
    old_edges = plan_physical_edges(current_plan)
    new_edges = plan_physical_edges(replacement_plan)
    temporary_bandwidth = {
        edge: float(max(0, count - old_edges.get(edge, 0)))
        * float(bandwidth_mbps)
        for edge, count in new_edges.items()
        if count > old_edges.get(edge, 0)
    }
    target = int(placement["dc_node"])
    return ResourceFootprint(
        cpu={target: float(placement.get("cpu_units", 0.0))},
        memory={target: float(placement.get("memory_units", 0.0))},
        bandwidth=temporary_bandwidth,
    )


def ryu_status_matches_sfc_plan(
    status: dict[str, Any], request_id: int, plan: dict[str, Any]
) -> bool:
    """Confirm a migration commit after an ambiguous REST response."""

    active = (status.get("sfcs") or {}).get(str(int(request_id)))
    if not isinstance(active, dict):
        return False

    def segment_signature(raw: dict[str, Any]) -> tuple[Any, ...]:
        outputs = tuple(
            sorted(
                (
                    int(dpid),
                    tuple(sorted(int(port) for port in ports)),
                )
                for dpid, ports in (raw.get("switch_outputs") or {}).items()
            )
        )
        return (
            int(raw["stage"]),
            str(raw["target_ip"]),
            int(raw["udp_port"]),
            tuple(int(node) for node in raw.get("path", [])),
            outputs,
        )

    actual_segments = tuple(
        segment_signature(segment)
        for segment in sorted(active.get("segments") or [], key=lambda row: int(row["stage"]))
    )
    expected_segments = tuple(
        segment_signature(segment)
        for segment in sorted(plan.get("segments") or [], key=lambda row: int(row["stage"]))
    )
    return bool(expected_segments) and actual_segments == expected_segments


def build_tail_reroute_plan(
    current_plan: dict[str, Any],
    reroute: dict[str, Any],
    request: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Replace only the multicast tree after the last VNF and validate it."""

    paths = reroute.get("paths")
    outputs = reroute.get("switch_outputs")
    if not isinstance(paths, dict) or not paths:
        raise ValueError("SFC tail reroute requires non-empty destination paths")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("SFC tail reroute requires non-empty switch outputs")

    segments = sorted(
        current_plan.get("segments", []), key=lambda row: int(row["stage"])
    )
    if not segments:
        raise ValueError("SFC tail reroute requires at least one VNF segment")
    root = int(segments[-1]["to_dpid"])
    expected_destinations = {int(value) for value in request["destination_dpids"]}
    normalized_paths = {
        str(int(destination)): [int(node) for node in path]
        for destination, path in paths.items()
    }
    if set(map(int, normalized_paths)) != expected_destinations:
        raise ValueError("reroute destinations do not match the active request")

    tree_edges: set[tuple[int, int]] = set()
    for destination in sorted(expected_destinations):
        path = normalized_paths[str(destination)]
        if not path or path[0] != root or path[-1] != destination:
            raise ValueError(
                f"reroute path {root}->{destination} has invalid endpoints"
            )
        tree_edges.update(zip(path, path[1:]))

    new_plan = copy.deepcopy(current_plan)
    multicast = dict(new_plan.get("multicast") or {})
    multicast.update(
        {
            "root_dpid": root,
            "paths": normalized_paths,
            "tree_edges": [list(edge) for edge in sorted(tree_edges)],
            "switch_outputs": {
                str(int(dpid)): sorted({int(port) for port in ports})
                for dpid, ports in outputs.items()
            },
        }
    )
    new_plan["multicast"] = multicast
    new_plan.setdefault("runtime_reconfiguration", []).append(
        {
            "type": "tail_multicast_reroute",
            "time": float(reroute.get("time", 0.0)),
            "policy": str(reroute.get("policy", "external")),
            "old_tree_edges": copy.deepcopy(
                (current_plan.get("multicast") or {}).get("tree_edges", [])
            ),
            "new_tree_edges": multicast["tree_edges"],
        }
    )
    validate_complete_plan(new_plan, request, profile)
    return new_plan


def shortest_tree_outputs(
    profile: dict[str, Any], source_dpid: int, destination_dpids: list[int]
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    nodes, adjacency = topology_index(profile)
    source = int(source_dpid)
    destinations = sorted({int(value) for value in destination_dpids})
    if source not in nodes:
        raise ValueError(f"source DPID {source} is absent from the topology")
    if not destinations or any(value not in nodes for value in destinations):
        raise ValueError("destination DPIDs are empty or outside the topology")
    if source in destinations:
        raise ValueError("source cannot also be a destination")

    parent: dict[int, tuple[int, int]] = {}
    queue = deque([source])
    visited = {source}
    while queue:
        current = queue.popleft()
        for neighbor, output_port in adjacency[current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = (current, output_port)
            queue.append(neighbor)

    missing = sorted(set(destinations) - visited)
    if missing:
        raise ValueError(f"destinations are unreachable from {source}: {missing}")

    outputs = defaultdict(set)
    paths = {}
    for destination in destinations:
        reverse_path = [destination]
        current = destination
        while current != source:
            previous, output_port = parent[current]
            outputs[previous].add(output_port)
            current = previous
            reverse_path.append(current)
        path = list(reversed(reverse_path))
        paths[str(destination)] = path
        outputs[destination].add(int(nodes[destination]["host_port"]))

    return (
        {str(dpid): sorted(ports) for dpid, ports in sorted(outputs.items())},
        paths,
    )


def request_events(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for request in requests:
        events.append(
            {
                "time": float(request["arrival_time"]),
                "type": "arrive",
                "request": request,
            }
        )
        events.append(
            {
                "time": float(request["leave_time"]),
                "type": "leave",
                "request": request,
            }
        )
    events.sort(key=lambda item: (item["time"], 0 if item["type"] == "leave" else 1))
    return events


def probe_agent_worker_plan(
    requests: list[dict[str, Any]],
    profile: dict[str, Any],
    sender_override: int = 0,
    receiver_override: int = 0,
) -> dict[str, Any]:
    nodes = {int(node["dpid"]): node for node in profile["nodes"]}
    sender_events: dict[str, list[tuple[float, int]]] = defaultdict(list)
    receiver_events: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for request in requests:
        arrival = float(request["arrival_time"])
        leave = float(request["leave_time"])
        source_host = str(nodes[int(request["source_dpid"])]["host"])
        sender_events[source_host].extend(((arrival, 1), (leave, -1)))
        for destination in request["destination_dpids"]:
            receiver_host = str(nodes[int(destination)]["host"])
            receiver_events[receiver_host].extend(((arrival, 1), (leave, -1)))

    def peaks_by_host(events_by_host: dict[str, list[tuple[float, int]]]):
        peaks = {}
        for host, events in events_by_host.items():
            active = 0
            peak = 0
            for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
                active += delta
                peak = max(peak, active)
            peaks[host] = peak
        return peaks

    sender_peaks = peaks_by_host(sender_events)
    receiver_peaks = peaks_by_host(receiver_events)
    max_sender_peak = max(sender_peaks.values(), default=1)
    max_receiver_peak = max(receiver_peaks.values(), default=1)
    auto_sender = min(128, max(16, math.ceil(1.5 * max_sender_peak) + 4))
    auto_receiver = min(128, max(32, math.ceil(1.5 * max_receiver_peak) + 8))
    return {
        "sender_workers": int(sender_override or auto_sender),
        "receiver_workers": int(receiver_override or auto_receiver),
        "auto_sender_workers": int(auto_sender),
        "auto_receiver_workers": int(auto_receiver),
        "max_sender_concurrency": int(max_sender_peak),
        "max_receiver_concurrency": int(max_receiver_peak),
        "sender_peaks_by_host": sender_peaks,
        "receiver_peaks_by_host": receiver_peaks,
        "sender_override": int(sender_override),
        "receiver_override": int(receiver_override),
        "capacity_capped": bool(
            auto_sender == 128 or auto_receiver == 128
        ),
    }


def wait_until(target: float) -> None:
    while True:
        remaining = target - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.05))


def summarize_milliseconds(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    def nearest_rank(probability: float) -> float:
        index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
        return ordered[max(0, index)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
    }


class RollingSetupEstimator:
    """Thread-safe rolling setup-time estimate used for deadline admission."""

    def __init__(
        self,
        initial_ms: float,
        window_size: int,
        min_samples: int,
        safety_factor: float,
    ) -> None:
        self.initial_ms = float(initial_ms)
        self.window_size = int(window_size)
        self.min_samples = int(min_samples)
        self.safety_factor = float(safety_factor)
        self._samples: deque[float] = deque(maxlen=self.window_size)
        self._lock = threading.Lock()

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )

    def estimate_ms(self) -> float:
        with self._lock:
            if self.initial_ms <= 0.0:
                return 0.0
            if len(self._samples) < self.min_samples:
                return self.initial_ms
            return self._percentile(list(self._samples), 0.95) * self.safety_factor

    def observe(self, setup_ms: float) -> None:
        if not math.isfinite(setup_ms) or setup_ms < 0.0:
            return
        with self._lock:
            self._samples.append(float(setup_ms))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
        estimate = (
            0.0
            if self.initial_ms <= 0.0
            else self.initial_ms
            if len(samples) < self.min_samples
            else self._percentile(samples, 0.95) * self.safety_factor
        )
        return {
            "estimate_ms": estimate,
            "initial_ms": self.initial_ms,
            "sample_count": len(samples),
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "safety_factor": self.safety_factor,
            "observed_ms": summarize_milliseconds(samples),
        }


class StageCapacityGate:
    """Non-blocking capacity gate with auditable occupancy statistics."""

    def __init__(self, name: str, limit: int) -> None:
        self.name = str(name)
        self.limit = int(limit)
        self._current = 0
        self._peak = 0
        self._rejections = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self.limit > 0 and self._current >= self.limit:
                self._rejections += 1
                return False
            self._current += 1
            self._peak = max(self._peak, self._current)
            return True

    def release(self) -> None:
        with self._lock:
            if self._current <= 0:
                raise RuntimeError(f"{self.name} capacity released without acquisition")
            self._current -= 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "limit": self.limit,
                "current": self._current,
                "peak": self._peak,
                "rejections": self._rejections,
            }


class VnfEndpointPool:
    """Allocate unique reusable VNF receive ports independently per DC."""

    def __init__(self, port_base: int, ports_per_dc: int):
        self.port_base = int(port_base)
        self.ports_per_dc = int(ports_per_dc)
        self._lock = threading.Lock()
        self._free_by_dc: dict[int, list[int]] = {}
        self._allocations: dict[int, list[tuple[int, int]]] = {}
        self._active_by_dc: Counter[int] = Counter()
        self._peak_active_by_dc: Counter[int] = Counter()
        self._peak_active_endpoints = 0
        self._exhaustions = 0
        self._exhaustions_by_dc: Counter[int] = Counter()
        self._pending_migrations: dict[
            str, tuple[int, int, int, int, int, int]
        ] = {}
        self._retired_migration_sources: dict[
            str, tuple[int, int, int, int]
        ] = {}

    def _free_ports(self, dc_node: int) -> list[int]:
        return self._free_by_dc.setdefault(
            int(dc_node),
            list(
                reversed(
                    range(self.port_base, self.port_base + self.ports_per_dc)
                )
            ),
        )

    def assign(
        self, request_id: int, plan: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        request_id = int(request_id)
        with self._lock:
            if request_id in self._allocations:
                raise ValueError(f"request {request_id} already owns VNF endpoints")
            staged: list[tuple[int, int, dict[str, Any]]] = []
            pooled_plan = copy.deepcopy(plan)
            segments = sorted(
                pooled_plan["segments"], key=lambda value: int(value["stage"])
            )
            for segment in segments:
                stage = int(segment["stage"])
                placement = pooled_plan["placement_by_vnf"][str(stage)]
                dc_node = int(placement["dc_node"])
                free_ports = self._free_ports(dc_node)
                if not free_ports:
                    for rollback_dc, rollback_port, _ in staged:
                        self._free_ports(rollback_dc).append(rollback_port)
                    self._exhaustions += 1
                    self._exhaustions_by_dc[dc_node] += 1
                    return None, {
                        "enabled": True,
                        "reason": "vnf_endpoint_pool_exhausted",
                        "dc_node": dc_node,
                        "ports_per_dc": self.ports_per_dc,
                        "active_endpoints_on_dc": self._active_by_dc[dc_node],
                    }
                port = free_ports.pop()
                staged.append((dc_node, port, segment))
            allocations = []
            for dc_node, port, segment in staged:
                segment["udp_port"] = int(port)
                stage = int(segment["stage"])
                placement = pooled_plan["placement_by_vnf"][str(stage)]
                if "listen_port" in placement:
                    placement["listen_port"] = int(port)
                allocations.append((dc_node, port))
            self._allocations[request_id] = allocations
            for dc_node, _ in allocations:
                self._active_by_dc[dc_node] += 1
                self._peak_active_by_dc[dc_node] = max(
                    self._peak_active_by_dc[dc_node],
                    self._active_by_dc[dc_node],
                )
            active_endpoints = (
                sum(len(value) for value in self._allocations.values())
                + len(self._pending_migrations)
                + len(self._retired_migration_sources)
            )
            self._peak_active_endpoints = max(
                self._peak_active_endpoints, active_endpoints
            )
            return pooled_plan, {
                "enabled": True,
                "ports": [port for _, port in allocations],
                "dc_nodes": [dc_node for dc_node, _ in allocations],
            }

    def release(self, request_id: int) -> bool:
        with self._lock:
            request_id = int(request_id)
            allocations = self._allocations.pop(request_id, None)
            if allocations is None:
                return False
            for dc_node, port in allocations:
                if self._active_by_dc[dc_node] <= 0:
                    raise RuntimeError(
                        f"DC {dc_node} endpoint released without allocation"
                    )
                self._active_by_dc[dc_node] -= 1
                self._free_ports(dc_node).append(port)
            for token, pending in list(self._pending_migrations.items()):
                pending_request, _, _, _, target_dc, new_port = pending
                if pending_request != request_id:
                    continue
                self._pending_migrations.pop(token, None)
                self._active_by_dc[target_dc] -= 1
                self._free_ports(target_dc).append(new_port)
            for token, retired in list(self._retired_migration_sources.items()):
                retired_request, _, old_dc, old_port = retired
                if retired_request != request_id:
                    continue
                self._retired_migration_sources.pop(token, None)
                self._active_by_dc[old_dc] -= 1
                self._free_ports(old_dc).append(old_port)
            return True

    def reserve_migration(
        self, request_id: int, stage: int, target_dc: int
    ) -> tuple[str | None, int | None, dict[str, Any]]:
        """Reserve one target endpoint while retaining the live source endpoint."""
        request_id, stage, target_dc = int(request_id), int(stage), int(target_dc)
        with self._lock:
            allocations = self._allocations.get(request_id)
            if allocations is None or not 0 <= stage < len(allocations):
                return None, None, {"reason": "request_or_stage_not_allocated"}
            old_dc, old_port = allocations[stage]
            free_ports = self._free_ports(target_dc)
            if not free_ports:
                self._exhaustions += 1
                self._exhaustions_by_dc[target_dc] += 1
                return None, None, {"reason": "vnf_endpoint_pool_exhausted"}
            new_port = int(free_ports.pop())
            token = f"{request_id}-{stage}-{time.time_ns()}"
            self._pending_migrations[token] = (
                request_id, stage, int(old_dc), int(old_port), target_dc, new_port
            )
            self._active_by_dc[target_dc] += 1
            self._peak_active_by_dc[target_dc] = max(
                self._peak_active_by_dc[target_dc], self._active_by_dc[target_dc]
            )
            self._peak_active_endpoints = max(
                self._peak_active_endpoints,
                sum(len(value) for value in self._allocations.values())
                + len(self._pending_migrations)
                + len(self._retired_migration_sources),
            )
            return token, new_port, {
                "old_dc": int(old_dc), "old_port": int(old_port),
                "target_dc": target_dc, "target_port": new_port,
            }

    def finish_migration(self, token: str, commit: bool) -> bool:
        with self._lock:
            token = str(token)
            if commit and token in self._retired_migration_sources:
                return True
            pending = self._pending_migrations.pop(token, None)
            if pending is None:
                return False
            request_id, stage, old_dc, old_port, target_dc, new_port = pending
            if commit:
                self._allocations[request_id][stage] = (target_dc, new_port)
                self._retired_migration_sources[token] = (
                    request_id, stage, old_dc, old_port
                )
            else:
                self._active_by_dc[target_dc] -= 1
                self._free_ports(target_dc).append(new_port)
            return True

    def release_migration_source(self, token: str) -> bool:
        """Release the old endpoint only after its VNF unregister is confirmed."""

        with self._lock:
            retired = self._retired_migration_sources.pop(str(token), None)
            if retired is None:
                return False
            _, _, old_dc, old_port = retired
            if self._active_by_dc[old_dc] <= 0:
                raise RuntimeError(
                    f"DC {old_dc} migration source released without allocation"
                )
            self._active_by_dc[old_dc] -= 1
            self._free_ports(old_dc).append(old_port)
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "port_base": self.port_base,
                "ports_per_dc": self.ports_per_dc,
                "active_requests": len(self._allocations),
                "active_endpoints": sum(
                    len(value) for value in self._allocations.values()
                ) + len(self._pending_migrations) + len(self._retired_migration_sources),
                "peak_active_endpoints": self._peak_active_endpoints,
                "pending_migrations": len(self._pending_migrations),
                "retired_migration_sources": len(self._retired_migration_sources),
                "exhaustions": self._exhaustions,
                "active_by_dc": {
                    str(dc): count
                    for dc, count in sorted(self._active_by_dc.items())
                    if count > 0
                },
                "peak_active_by_dc": {
                    str(dc): count
                    for dc, count in sorted(self._peak_active_by_dc.items())
                },
                "exhaustions_by_dc": {
                    str(dc): count
                    for dc, count in sorted(self._exhaustions_by_dc.items())
                },
            }


class RuntimeFifoAckTimeout(RuntimeError):
    pass


class ReceiverStartupTimeout(RuntimeError):
    pass


class RyuBatchCommitter:
    """Collect ready SFCs briefly and commit them with one controller barrier."""

    def __init__(
        self,
        client: RyuSFTClient,
        window_ms: float,
        max_batch_size: int,
    ) -> None:
        self.client = client
        self.window_seconds = float(window_ms) / 1000.0
        self.max_batch_size = int(max_batch_size)
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._batch_sizes: list[int] = []
        self._commit_times_ms: list[float] = []
        self._thread = threading.Thread(
            target=self._run, name="ryu-batch-committer", daemon=True
        )
        self._thread.start()

    def submit(self, plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Ryu batch committer is closed")
        item = {
            "plan": plan,
            "submitted": time.monotonic(),
            "done": threading.Event(),
            "response": None,
            "error": None,
            "timing": None,
        }
        self._queue.put(item)
        item["done"].wait()
        if item["error"] is not None:
            raise item["error"]
        return item["response"], item["timing"]

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            items = [first]
            stop_after_batch = False
            close_at_ns = time.perf_counter_ns() + int(self.window_seconds * 1e9)
            while len(items) < self.max_batch_size:
                if time.perf_counter_ns() >= close_at_ns:
                    break
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    # A positive sub-millisecond sleep is commonly rounded to
                    # 15.6 ms on Windows.  Yield without a timed sleep so the
                    # configured 1-2 ms batching deadline remains meaningful.
                    time.sleep(0)
                    continue
                if item is None:
                    self._queue.task_done()
                    stop_after_batch = True
                    break
                items.append(item)
            commit_started = time.monotonic()
            try:
                responses = self.client.install_sfc_batch(
                    [item["plan"] for item in items]
                )
                if len(responses) != len(items):
                    raise RuntimeError(
                        "Ryu batch response count does not match request count"
                    )
                commit_ms = (time.monotonic() - commit_started) * 1000.0
                completed = time.monotonic()
                with self._stats_lock:
                    self._batch_sizes.append(len(items))
                    self._commit_times_ms.append(commit_ms)
                for batch_index, (item, response) in enumerate(zip(items, responses)):
                    item["response"] = response
                    item["timing"] = {
                        "batch_size": len(items),
                        "batch_index": batch_index,
                        "queue_wait_ms": (
                            commit_started - item["submitted"]
                        )
                        * 1000.0,
                        "commit_ms": commit_ms,
                        "total_ms": (completed - item["submitted"]) * 1000.0,
                    }
            except BaseException as exc:
                for item in items:
                    item["error"] = exc
            finally:
                for item in items:
                    item["done"].set()
                    self._queue.task_done()
            if stop_after_batch:
                return

    def snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "enabled": True,
                "window_ms": self.window_seconds * 1000.0,
                "max_batch_size": self.max_batch_size,
                "batches": len(self._batch_sizes),
                "requests": sum(self._batch_sizes),
                "request_batch_size": summarize_milliseconds(
                    [float(value) for value in self._batch_sizes]
                ),
                "commit_ms": summarize_milliseconds(self._commit_times_ms),
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()


class VnfControlBatcher:
    """Coalesce cross-request VNF FIFO transactions into one Mininet RTT."""

    def __init__(
        self,
        port: int,
        timeout: float,
        window_ms: float,
        max_batch_size: int,
        name: str,
        transport: Any | None = None,
    ) -> None:
        self.port = int(port)
        self.timeout = float(timeout)
        self.window_seconds = float(window_ms) / 1000.0
        self.max_batch_size = int(max_batch_size)
        self.name = str(name)
        self.transport = transport or mininet_runtime_fifo_messages_wait_ack
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._batch_sizes: list[int] = []
        self._message_counts: list[int] = []
        self._thread = threading.Thread(
            target=self._run,
            name=f"vnf-{self.name}-batcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"VNF {self.name} batcher is closed")
        if not messages:
            raise ValueError(f"VNF {self.name} messages cannot be empty")
        item = {
            "messages": list(messages),
            "submitted": time.monotonic(),
            "done": threading.Event(),
            "response": None,
            "error": None,
            "timing": None,
        }
        self._queue.put(item)
        item["done"].wait()
        if item["error"] is not None:
            raise item["error"]
        return {**item["response"], "batch_timing": item["timing"]}

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            items = [first]
            stop_after_batch = False
            close_at_ns = time.perf_counter_ns() + int(self.window_seconds * 1e9)
            while len(items) < self.max_batch_size:
                if time.perf_counter_ns() >= close_at_ns:
                    break
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    remaining_seconds = max(
                        0.0, (close_at_ns - time.perf_counter_ns()) / 1e9
                    )
                    time.sleep(min(0.0005, remaining_seconds))
                    continue
                if item is None:
                    self._queue.task_done()
                    stop_after_batch = True
                    break
                items.append(item)

            messages = [
                message for item in items for message in item["messages"]
            ]
            transaction_started = time.monotonic()
            try:
                response = self.transport(
                    self.port,
                    messages,
                    self.timeout,
                )
                completed = time.monotonic()
                with self._stats_lock:
                    self._batch_sizes.append(len(items))
                    self._message_counts.append(len(messages))
                for item in items:
                    item["response"] = response
                    item["timing"] = {
                        "request_batch_size": len(items),
                        "message_count": len(messages),
                        "queue_wait_ms": (
                            transaction_started - item["submitted"]
                        )
                        * 1000.0,
                        "transaction_ms": (
                            completed - transaction_started
                        )
                        * 1000.0,
                        "total_ms": (completed - item["submitted"]) * 1000.0,
                    }
            except BaseException as exc:
                for item in items:
                    item["error"] = exc
            finally:
                for item in items:
                    item["done"].set()
                    self._queue.task_done()
            if stop_after_batch:
                return

    def snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "enabled": True,
                "name": self.name,
                "window_ms": self.window_seconds * 1000.0,
                "max_batch_size": self.max_batch_size,
                "batches": len(self._batch_sizes),
                "requests": sum(self._batch_sizes),
                "messages": sum(self._message_counts),
                "request_batch_size": summarize_milliseconds(
                    [float(value) for value in self._batch_sizes]
                ),
                "messages_per_batch": summarize_milliseconds(
                    [float(value) for value in self._message_counts]
                ),
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()


def to_wsl_path(path: Path) -> str:
    value = path.resolve()
    drive = value.drive.rstrip(":").lower()
    if not drive:
        return str(value).replace("\\", "/")
    tail = str(value)[len(value.drive) :].replace("\\", "/")
    return f"/mnt/{drive}{tail}"


def normalize_cpu_set(value: str | None) -> str:
    if not value or not value.strip():
        return ""
    cpus: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("CPU sets cannot contain empty entries")
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range {item!r}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU number {cpu}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set must contain at least one CPU")
    return ",".join(str(cpu) for cpu in sorted(cpus))


def wait_for_controller(client: RyuSFTClient, switch_count: int, timeout: float):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            status = client.status()
            if len(status.get("datapaths", [])) >= switch_count:
                return status
        except OSError as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"Ryu did not observe {switch_count} switches: {last_error}")


def set_service_realtime_scheduler(
    wsl: list[str], service: str, priority: int, timeout: float = 5.0
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "service": service,
        "requested_priority": int(priority),
        "applied": False,
        "main_pid": None,
        "scheduler": None,
    }
    if priority <= 0:
        return result

    deadline = time.monotonic() + timeout
    pid = 0
    while time.monotonic() < deadline:
        query = subprocess.run(
            wsl
            + [
                "systemctl",
                "show",
                "--property",
                "MainPID",
                "--value",
                service,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            pid = int(query.stdout.strip())
        except ValueError:
            pid = 0
        if pid > 0:
            break
        time.sleep(0.05)
    if pid <= 0:
        raise RuntimeError(f"systemd service {service} has no live MainPID")

    changed = subprocess.run(
        wsl + ["sudo", "-n", "chrt", "-r", "-p", str(int(priority)), str(pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if changed.returncode != 0:
        raise RuntimeError(
            f"failed to set SCHED_RR on {service} PID {pid}: {changed.stderr.strip()}"
        )
    status = subprocess.run(
        wsl + ["chrt", "-p", str(pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    result.update(
        {
            "applied": True,
            "main_pid": pid,
            "scheduler": status.stdout.strip(),
        }
    )
    return result


def restore_service_scheduler(wsl: list[str], scheduler: dict[str, Any]) -> None:
    if not scheduler.get("applied") or not scheduler.get("main_pid"):
        return
    subprocess.run(
        wsl
        + [
            "sudo",
            "-n",
            "chrt",
            "-o",
            "-p",
            "0",
            str(int(scheduler["main_pid"])),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def set_service_cpu_affinity(
    wsl: list[str], service: str, cpu_set: str, timeout: float = 5.0
) -> dict[str, Any]:
    requested = normalize_cpu_set(cpu_set)
    result: dict[str, Any] = {
        "service": service,
        "requested_cpu_set": requested,
        "applied": False,
        "main_pid": None,
        "original_cpu_set": None,
        "effective_cpu_set": None,
    }
    if not requested:
        return result

    deadline = time.monotonic() + timeout
    pid = 0
    while time.monotonic() < deadline:
        query = subprocess.run(
            wsl
            + [
                "systemctl",
                "show",
                "--property",
                "MainPID",
                "--value",
                service,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            pid = int(query.stdout.strip())
        except ValueError:
            pid = 0
        if pid > 0:
            break
        time.sleep(0.05)
    if pid <= 0:
        raise RuntimeError(f"systemd service {service} has no live MainPID")

    before = subprocess.run(
        wsl + ["taskset", "-pc", str(pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if before.returncode != 0 or ":" not in before.stdout:
        raise RuntimeError(
            f"failed to read CPU affinity for {service} PID {pid}: {before.stderr.strip()}"
        )
    original = before.stdout.rsplit(":", 1)[-1].strip()
    changed = subprocess.run(
        wsl + ["sudo", "-n", "taskset", "-apc", requested, str(pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if changed.returncode != 0:
        raise RuntimeError(
            f"failed to set CPU affinity on {service} PID {pid}: {changed.stderr.strip()}"
        )
    result.update(
        {
            "applied": True,
            "main_pid": pid,
            "original_cpu_set": original,
            "effective_cpu_set": requested,
        }
    )
    return result


def restore_service_cpu_affinity(wsl: list[str], affinity: dict[str, Any]) -> None:
    if (
        not affinity.get("applied")
        or not affinity.get("main_pid")
        or not affinity.get("original_cpu_set")
    ):
        return
    subprocess.run(
        wsl
        + [
            "sudo",
            "-n",
            "taskset",
            "-apc",
            str(affinity["original_cpu_set"]),
            str(int(affinity["main_pid"])),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def mininet_runtime_request(
    port: int, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    connections = getattr(_MININET_RUNTIME_CONNECTIONS, "by_port", None)
    if connections is None:
        connections = {}
        _MININET_RUNTIME_CONNECTIONS.by_port = connections
    connection_key = (_MININET_RUNTIME_HOST, int(port))
    state = connections.get(connection_key)
    if state is None:
        connection = socket.create_connection(
            (_MININET_RUNTIME_HOST, int(port)), timeout=timeout
        )
        connection.settimeout(timeout)
        state = (connection, connection.makefile("rb"))
        connections[connection_key] = state
    connection, reader = state
    try:
        connection.settimeout(timeout)
        connection.sendall(request)
        response = reader.readline(16 * 1024 * 1024 + 1)
    except (OSError, ValueError):
        connections.pop(connection_key, None)
        try:
            reader.close()
        finally:
            connection.close()
        raise
    if len(response) > 16 * 1024 * 1024:
        raise RuntimeError("Mininet runtime response exceeds 16 MiB")
    if not response:
        raise RuntimeError("Mininet runtime returned an empty response")
    result = json.loads(response.decode("utf-8"))
    if not result.get("ok"):
        error = str(result.get("error"))
        if "FIFO ACK timed out" in error or "VNF registration ACK timed out" in error:
            raise RuntimeFifoAckTimeout(f"Mininet runtime command failed: {error}")
        raise RuntimeError(f"Mininet runtime command failed: {error}")
    return result


def wait_for_mininet_runtime(port: int, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return mininet_runtime_request(port, {"operation": "health"}, 2.0)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"Mininet runtime did not become ready: {last_error}")


def mininet_runtime_commands(
    port: int, commands: list[tuple[str, str]], timeout: float
) -> dict[str, Any]:
    return mininet_runtime_request(
        port,
        {
            "operation": "commands",
            "commands": [
                {"host": host, "command": command} for host, command in commands
            ],
        },
        timeout,
    )


def mininet_runtime_fifo_messages(
    port: int, messages: list[dict[str, Any]], timeout: float
) -> dict[str, Any]:
    return mininet_runtime_request(
        port,
        {"operation": "fifo_messages", "messages": messages},
        timeout,
    )


def mininet_runtime_fifo_messages_wait_ack(
    port: int,
    messages: list[dict[str, Any]],
    timeout: float,
    ack_timeout: float | None = None,
) -> dict[str, Any]:
    return mininet_runtime_request(
        port,
        {
            "operation": "fifo_messages_wait_ack",
            "messages": messages,
            "ack_timeout": float(timeout if ack_timeout is None else ack_timeout),
        },
        timeout,
    )


def mininet_runtime_diagnostics(port: int, timeout: float) -> dict[str, Any]:
    return mininet_runtime_request(port, {"operation": "diagnostics"}, timeout)


def mininet_clock_offset(
    port: int, timeout: float, samples: int = 7
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("clock samples must be positive")
    observations = []
    for _ in range(samples):
        before_ns = time.time_ns()
        response = mininet_runtime_request(port, {"operation": "clock"}, timeout)
        after_ns = time.time_ns()
        remote_ns = int(response["time_ns"])
        observations.append(
            {
                "rtt_ns": after_ns - before_ns,
                "offset_ns": remote_ns - ((before_ns + after_ns) // 2),
            }
        )
    best = min(observations, key=lambda row: row["rtt_ns"])
    return {
        "offset_ns": int(best["offset_ns"]),
        "best_rtt_ms": float(best["rtt_ns"]) / 1_000_000.0,
        "samples": len(observations),
        "offset_range_ms": (
            max(row["offset_ns"] for row in observations)
            - min(row["offset_ns"] for row in observations)
        ) / 1_000_000.0,
    }


def remote_deadline_ns(
    local_time_ns: int,
    remote_offset_ns: int,
    budget_seconds: float,
) -> int:
    """Convert a fresh local/remote wall-clock sample into a remote deadline."""

    if budget_seconds < 0.0 or not math.isfinite(budget_seconds):
        raise ValueError("deadline budget must be finite and non-negative")
    return (
        int(local_time_ns)
        + int(remote_offset_ns)
        + int(float(budget_seconds) * 1_000_000_000)
    )


def probe_stop_deadlines_ns(
    local_time_ns: int,
    remote_offset_ns: int,
    remaining_lifetime_seconds: float,
    sender_stop_margin_seconds: float,
) -> tuple[int, int]:
    """Return distinct sender and receiver deadlines on the remote clock.

    The sender stops before request expiry. Receivers remain active until the
    request leaves so packets already inside the SFC and multicast tree can
    drain instead of being counted as artificial tail loss.
    """

    if remaining_lifetime_seconds < 0.0:
        raise ValueError("remaining lifetime must be non-negative")
    if not 0.0 <= sender_stop_margin_seconds <= remaining_lifetime_seconds:
        raise ValueError("sender stop margin must fit the remaining lifetime")
    sender_budget = remaining_lifetime_seconds - sender_stop_margin_seconds
    sender_deadline = remote_deadline_ns(
        local_time_ns, remote_offset_ns, sender_budget
    )
    receiver_deadline = remote_deadline_ns(
        local_time_ns, remote_offset_ns, remaining_lifetime_seconds
    )
    return sender_deadline, receiver_deadline


def resolve_wsl_ipv4(wsl: list[str]) -> str:
    completed = subprocess.run(
        wsl + ["hostname", "-I"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for candidate in completed.stdout.split():
        try:
            socket.inet_aton(candidate)
        except OSError:
            continue
        return candidate
    raise RuntimeError(
        f"could not discover a WSL IPv4 address: {completed.stdout.strip()}"
    )


def resolve_controller_rest_url(configured: str, wsl_ipv4: str) -> str:
    value = configured.strip()
    if value.lower() == "auto":
        if not wsl_ipv4:
            raise ValueError("WSL IPv4 address cannot be empty for automatic REST discovery")
        return f"http://{wsl_ipv4}:8080"
    if not value:
        raise ValueError("Ryu REST URL cannot be empty")
    return value.rstrip("/")


def network_diagnostic_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    udp_fields = (
        "InDatagrams", "NoPorts", "InErrors", "OutDatagrams",
        "RcvbufErrors", "SndbufErrors", "InCsumErrors", "IgnoredMulti",
    )
    host_udp = {}
    for host in sorted(set(before.get("host_udp", {})) | set(after.get("host_udp", {}))):
        old = before.get("host_udp", {}).get(host, {})
        new = after.get("host_udp", {}).get(host, {})
        delta = {field: int(new.get(field, 0)) - int(old.get(field, 0)) for field in udp_fields}
        if any(delta.values()):
            host_udp[host] = delta

    qdisc_fields = ("dropped", "overlimits", "requeues")
    qdiscs = {}
    for interface in sorted(set(before.get("qdiscs", {})) | set(after.get("qdiscs", {}))):
        old = before.get("qdiscs", {}).get(interface, {})
        new = after.get("qdiscs", {}).get(interface, {})
        delta = {field: int(new.get(field, 0)) - int(old.get(field, 0)) for field in qdisc_fields}
        if any(delta.values()):
            qdiscs[interface] = delta
    return {"host_udp": host_udp, "qdiscs": qdiscs}


def wait_for_rest(client: RyuSFTClient, timeout: float):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return client.status()
        except OSError as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"Ryu REST API did not become ready: {last_error}")


def load_external_plans(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    plans = {}
    for row in read_jsonl(path):
        request_id = int(row["request_id"])
        if request_id in plans:
            raise ValueError(f"duplicate external tree plan for request {request_id}")
        outputs = row.get("switch_outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"external tree plan {request_id} has no switch_outputs")
        plans[request_id] = {
            "switch_outputs": {
                str(int(dpid)): sorted({int(port) for port in ports})
                for dpid, ports in outputs.items()
            },
            "paths": row.get("paths", {}),
            "source": str(row.get("source", "external")),
        }
    return plans


def load_sfc_plans(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    plans = {}
    for row in read_jsonl(path):
        request_id = int(row["request_id"])
        if request_id in plans:
            raise ValueError(f"duplicate SFC plan for request {request_id}")
        if not row.get("accepted", True):
            plans[request_id] = row
            continue
        segments = row.get("segments")
        multicast = row.get("multicast")
        placements = row.get("placement_by_vnf")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"SFC plan {request_id} has no segments")
        if not isinstance(multicast, dict) or not multicast.get("switch_outputs"):
            raise ValueError(f"SFC plan {request_id} has no multicast tree")
        if not isinstance(placements, dict) or len(placements) != len(segments):
            raise ValueError(f"SFC plan {request_id} has invalid VNF placements")
        stages = sorted(int(segment["stage"]) for segment in segments)
        if stages != list(range(len(segments))):
            raise ValueError(f"SFC plan {request_id} has non-contiguous stages")
        plans[request_id] = row
    return plans


def _validate_sfc_plan_row(row: dict[str, Any], label: str) -> dict[str, Any]:
    """Validate one complete SFC plan and return a detached copy."""
    plan = copy.deepcopy(row)
    request_id = int(plan["request_id"])
    if not plan.get("accepted", True):
        return plan
    segments = plan.get("segments")
    multicast = plan.get("multicast")
    placements = plan.get("placement_by_vnf")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{label} {request_id} has no segments")
    if not isinstance(multicast, dict) or not multicast.get("switch_outputs"):
        raise ValueError(f"{label} {request_id} has no multicast tree")
    if not isinstance(placements, dict) or len(placements) != len(segments):
        raise ValueError(f"{label} {request_id} has invalid VNF placements")
    stages = sorted(int(segment["stage"]) for segment in segments)
    if stages != list(range(len(segments))):
        raise ValueError(f"{label} {request_id} has non-contiguous stages")
    return plan


def load_sfc_candidate_plans(
    path: str | None, top_k: int = 0
) -> dict[int, list[dict[str, Any]]]:
    """Load multiple complete SFC candidates from batches or candidate JSONL.

    Supported records are either deployment-topk ``batches.jsonl`` rows with
    ``agents[].candidates[].plan`` or flat rows containing ``candidates``.
    The reject action and candidates without a complete plan are ignored.
    """
    if not path:
        return {}
    rows = read_jsonl(path)
    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        agents = row.get("agents")
        if isinstance(agents, list):
            records = agents
        elif isinstance(row.get("candidates"), list):
            records = [row]
        else:
            continue
        for agent in records:
            request_id = int(agent.get("request_id", -1))
            raw_candidates = agent.get("candidates")
            if not isinstance(raw_candidates, list) or request_id < 0:
                continue
            candidates: list[dict[str, Any]] = []
            for index, candidate in enumerate(raw_candidates):
                if not isinstance(candidate, dict):
                    continue
                plan = candidate.get("plan")
                if not isinstance(plan, dict) or not plan.get("accepted", True):
                    continue
                plan = _validate_sfc_plan_row(
                    {**plan, "request_id": request_id},
                    "SFC candidate",
                )
                selection = {
                    "candidate_index": int(index),
                    "candidate_id": str(candidate.get("candidate_id", f"r{request_id}-c{index}")),
                    "source": str(candidate.get("source", "candidate_file")),
                    "objective": candidate.get("objective"),
                    "candidate_metrics": copy.deepcopy(candidate.get("metrics", {})),
                }
                plan.setdefault("topk_selection", {}).update(selection)
                candidates.append(plan)
            if top_k > 0:
                candidates = candidates[: int(top_k)]
            if request_id in result:
                raise ValueError(f"duplicate SFC candidate request {request_id}")
            result[request_id] = candidates
    return result


def sfc_candidate_sla_score(
    plan: dict[str, Any], projected_utilizations: list[float], rank: int
) -> tuple[float, float, float, float, float, int]:
    """Rank a feasible candidate by estimated delay under its live link load."""
    metrics = plan.get("topk_selection", {}).get("candidate_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    estimated_delay_ms = max(0.0, float(metrics.get("estimated_delay_ms", 0.0)))
    delay_bound_ms = max(1e-9, float(metrics.get("delay_bound_ms", 1.0)))
    max_utilization = max(projected_utilizations, default=0.0)
    # M/M/1-inspired risk term. It is used only to rank already feasible plans;
    # the hard bandwidth ledger remains the admission authority.
    delay_risk = (estimated_delay_ms / delay_bound_ms) / max(
        1.0 - min(max_utilization, 0.999999), 1e-6
    )
    route_size = float(metrics.get("segment_hops", 0.0)) + float(
        metrics.get("tree_edges", 0.0)
    )
    flowmod_estimate = float(metrics.get("flowmod_estimate", route_size))
    return (
        delay_risk,
        max_utilization,
        route_size,
        flowmod_estimate,
        sum(projected_utilizations),
        int(rank),
    )


def index_sfc_execution_plan(
    request: dict[str, Any], sfc_plan: dict[str, Any], source: str
) -> dict[str, Any]:
    request_id = int(request["id"])
    if not sfc_plan.get("accepted", True):
        return {
            "switch_outputs": {},
            "paths": {},
            "source": f"{source}_rejected",
            "reason": str(sfc_plan.get("reason", "hrl_rejected")),
        }
    first_segment = min(
        sfc_plan["segments"], key=lambda value: int(value["stage"])
    )
    if int(first_segment["path"][0]) != int(request["source_dpid"]):
        raise ValueError(f"SFC plan {request_id} starts at the wrong source")
    if str(sfc_plan["multicast"]["dst_ip"]) != str(request["multicast_ip"]):
        raise ValueError(f"SFC plan {request_id} uses the wrong multicast address")
    return {
        "switch_outputs": sfc_plan["multicast"]["switch_outputs"],
        "paths": sfc_plan["multicast"].get("paths", {}),
        "source": source,
    }


def deterministic_host_mac(dpid: int) -> str:
    value = int(dpid)
    if value <= 0 or value > 0xFFFFFF:
        raise ValueError(f"DPID {value} exceeds deterministic host MAC range")
    return (
        f"02:00:00:{(value >> 16) & 0xff:02x}:"
        f"{(value >> 8) & 0xff:02x}:{value & 0xff:02x}"
    )


def ensure_sfc_plan_neighbors(
    command_port: int,
    plan: dict[str, Any],
    nodes: dict[int, dict[str, Any]],
    timeout: float,
) -> int:
    """Install and verify only the host-neighbor entries used by an SFC plan."""
    commands = []
    expected = []
    seen = set()
    for segment in plan["segments"]:
        source = int(segment["from_dpid"])
        target = int(segment["to_dpid"])
        if source == target or (source, target) in seen:
            continue
        seen.add((source, target))
        source_node = nodes[source]
        target_node = nodes[target]
        host = str(source_node["host"])
        interface = f"{host}-eth0"
        target_ip = str(target_node["host_ip"]).split("/", 1)[0]
        target_mac = deterministic_host_mac(target)
        commands.append(
            (
                host,
                f"ip neigh replace {shlex.quote(target_ip)} lladdr "
                f"{shlex.quote(target_mac)} nud permanent dev "
                f"{shlex.quote(interface)} && ip neigh show "
                f"{shlex.quote(target_ip)} dev {shlex.quote(interface)}",
            )
        )
        expected.append((host, target_ip, target_mac))
    if not commands:
        return 0
    response = mininet_runtime_commands(command_port, commands, timeout)
    outputs = response.get("outputs", [])
    if len(outputs) != len(expected):
        raise RuntimeError("SFC neighbor setup returned the wrong result count")
    for output, (host, target_ip, target_mac) in zip(outputs, expected):
        text = str(output.get("output", ""))
        normalized = text.lower()
        if target_ip not in text or target_mac.lower() not in normalized or "permanent" not in normalized:
            raise RuntimeError(
                f"SFC neighbor verification failed on {host} for {target_ip}: {text!r}"
            )
    return len(commands)


def sfc_sender_request(
    request: dict[str, Any], plan: dict[str, Any] | None
) -> dict[str, Any]:
    if plan is None:
        return request
    first = min(plan["segments"], key=lambda value: int(value["stage"]))
    return {
        **request,
        "multicast_ip": str(first["target_ip"]),
        "udp_port": int(first["udp_port"]),
    }


def sfc_runtime_files(
    probe_dir: Path, request_id: int, stage: int
) -> dict[str, Path]:
    prefix = probe_dir / f"sfc-r{int(request_id)}-stage{int(stage)}"
    return {
        "ready": Path(f"{prefix}.ready"),
        "stats": Path(f"{prefix}.stats.json"),
        "drained": Path(f"{prefix}.drained.json"),
        "pid": Path(f"{prefix}.pid"),
        "log": Path(f"{prefix}.log"),
    }


def sfc_forwarder_command(
    forwarder_path: str,
    plan: dict[str, Any],
    stage: int,
    files: dict[str, str],
) -> str:
    segments = sorted(plan["segments"], key=lambda value: int(value["stage"]))
    segment = segments[int(stage)]
    placement = plan["placement_by_vnf"][str(int(stage))]
    if int(stage) + 1 < len(segments):
        next_host = str(segments[int(stage) + 1]["target_ip"])
        next_port = int(segments[int(stage) + 1]["udp_port"])
    else:
        next_host = str(plan["multicast"]["dst_ip"])
        next_port = int(plan["multicast"]["udp_port"])
    values = [
        "python3",
        forwarder_path,
        "--listen-port",
        str(segment["udp_port"]),
        "--next-host",
        next_host,
        "--next-port",
        str(next_port),
        "--vnf-type",
        str(placement["vnf_type"]),
        "--ready-file",
        files["ready"],
        "--stats-output",
        files["stats"],
    ]
    return (
        f"rm -f {shlex.quote(files['ready'])} {shlex.quote(files['stats'])} "
        f"{shlex.quote(files['pid'])}; "
        + " ".join(shlex.quote(value) for value in values)
        + f" > {shlex.quote(files['log'])} 2>&1 & "
        + f"echo $! > {shlex.quote(files['pid'])}"
    )


def sfc_stop_command(pid_file: str) -> str:
    return (
        f"if test -s {shlex.quote(pid_file)}; then "
        f"kill $(cat {shlex.quote(pid_file)}) 2>/dev/null || true; fi"
    )


def fifo_json_command(fifo: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return f"printf '%s\\n' {shlex.quote(encoded)} > {shlex.quote(fifo)}"


def vnf_agent_start_command(
    agent_path: str,
    fifo: str,
    ready_file: str,
    log_file: str,
    pid_file: str,
    workers: int,
    initial_workers: int,
    bindings_per_worker: int,
    prewarm_workers: bool,
    drain_timeout_ms: float,
    drain_idle_ms: float,
    realtime_priority: int,
    cpu_set: str = "",
    max_packets_per_socket_event: int = 64,
    q0_max_packets_per_socket_event: int = 0,
    backend: str = "python",
    prebound_port_base: int = 30000,
    prebound_port_count: int = 0,
    dscp_scheduling: bool = False,
) -> str:
    if backend not in {"python", "native"}:
        raise ValueError(f"unsupported VNF agent backend: {backend}")
    prefix = ["python3", agent_path] if backend == "python" else [agent_path]
    values = (["taskset", "-c", normalize_cpu_set(cpu_set)] if cpu_set else []) + prefix + [
        "--command-fifo", fifo,
        "--ready-file", ready_file,
        "--drain-timeout-ms", str(float(drain_timeout_ms)),
        "--drain-idle-ms", str(float(drain_idle_ms)),
        "--realtime-priority", str(int(realtime_priority)),
        "--max-packets-per-socket-event", str(int(max_packets_per_socket_event)),
        "--q0-max-packets-per-socket-event", str(int(q0_max_packets_per_socket_event)),
    ]
    if backend == "python":
        values.extend(
            [
                "--workers", str(int(workers)),
                "--initial-workers", str(int(initial_workers)),
                "--bindings-per-worker", str(int(bindings_per_worker)),
                "--prebound-port-base", str(int(prebound_port_base)),
                "--prebound-port-count", str(int(prebound_port_count)),
            ]
        )
    elif dscp_scheduling:
        values.append("--dscp-scheduling")
    if prewarm_workers:
        values.append("--prewarm-workers")
    return (
        f"rm -f {shlex.quote(fifo)} {shlex.quote(ready_file)} {shlex.quote(pid_file)}; "
        + " ".join(shlex.quote(value) for value in values)
        + f" > {shlex.quote(log_file)} 2>&1 & echo $! > {shlex.quote(pid_file)}"
    )


def vnf_agent_register_command(
    fifo: str,
    plan: dict[str, Any],
    stage: int,
    files: dict[str, str],
) -> str:
    segments = sorted(plan["segments"], key=lambda value: int(value["stage"]))
    segment = segments[int(stage)]
    placement = plan["placement_by_vnf"][str(int(stage))]
    if int(stage) + 1 < len(segments):
        next_host = str(segments[int(stage) + 1]["target_ip"])
        next_port = int(segments[int(stage) + 1]["udp_port"])
    else:
        next_host = str(plan["multicast"]["dst_ip"])
        next_port = int(plan["multicast"]["udp_port"])
    payload = {
        "operation": "register",
        "request_id": int(plan["request_id"]),
        "stage": int(stage),
        "listen_port": int(segment["udp_port"]),
        "next_host": next_host,
        "next_port": next_port,
        "vnf_type": int(placement["vnf_type"]),
        "ready_file": files["ready"],
        "stats_output": files["stats"],
    }
    return (
        f"rm -f {shlex.quote(files['ready'])} {shlex.quote(files['stats'])}; "
        + fifo_json_command(fifo, payload)
    )


def vnf_agent_register_message(
    fifo: str,
    plan: dict[str, Any],
    stage: int,
    files: dict[str, str],
    dscp: int = 0,
    include_ready_file: bool = True,
    include_stats_file: bool = True,
) -> dict[str, Any]:
    segments = sorted(plan["segments"], key=lambda value: int(value["stage"]))
    segment = segments[int(stage)]
    placement = plan["placement_by_vnf"][str(int(stage))]
    if int(stage) + 1 < len(segments):
        next_host = str(segments[int(stage) + 1]["target_ip"])
        next_port = int(segments[int(stage) + 1]["udp_port"])
    else:
        next_host = str(plan["multicast"]["dst_ip"])
        next_port = int(plan["multicast"]["udp_port"])
    payload = {
        "operation": "register",
        "request_id": int(plan["request_id"]),
        "stage": int(stage),
        "listen_port": int(segment["udp_port"]),
        "next_host": next_host,
        "next_port": next_port,
        "vnf_type": int(placement["vnf_type"]),
        "dscp": int(dscp),
        "stats_output": files["stats"] if include_stats_file else "",
    }
    remove_paths = [files["stats"]] if include_stats_file else []
    if include_ready_file:
        payload["ready_file"] = files["ready"]
        remove_paths.append(files["ready"])
    return {
        "fifo": fifo,
        "remove_paths": remove_paths,
        "payload": payload,
    }


def vnf_agent_unregister_command(fifo: str, request_id: int, stage: int) -> str:
    return fifo_json_command(
        fifo,
        {"operation": "unregister", "request_id": int(request_id), "stage": int(stage)},
    )


def vnf_agent_unregister_message(
    fifo: str, request_id: int, stage: int
) -> dict[str, Any]:
    return {
        "fifo": fifo,
        "payload": {
            "operation": "unregister",
            "request_id": int(request_id),
            "stage": int(stage),
        },
    }


def vnf_agent_drain_message(
    fifo: str,
    request_id: int,
    stage: int,
    drain_ack: str | None,
    drain_timeout_ms: float,
    drain_idle_ms: float,
) -> dict[str, Any]:
    message = {
        "fifo": fifo,
        "payload": {
            "operation": "drain",
            "request_id": int(request_id),
            "stage": int(stage),
            "drain_timeout_ms": float(drain_timeout_ms),
            "drain_idle_ms": float(drain_idle_ms),
        },
    }
    if drain_ack:
        message["remove_paths"] = [drain_ack]
        message["payload"]["drain_ack"] = drain_ack
    return message


def vnf_agent_fifo_for_binding(
    fifos_by_host: dict[str, list[str]],
    host: str,
    request_id: int,
    stage: int,
) -> str:
    """Keep one request stage on the same native-agent shard for its lifetime."""
    fifos = fifos_by_host.get(str(host), [])
    if not fifos:
        raise KeyError(f"no VNF agent FIFO is registered for host {host}")
    shard = (int(request_id) * 31 + int(stage)) % len(fifos)
    return fifos[shard]


def load_reroute_events(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    rows = []
    for row in read_jsonl(path):
        request_id = int(row["request_id"])
        event_time = float(row["time"])
        outputs = row.get("switch_outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"reroute event for request {request_id} has no switch_outputs")
        rows.append(
            {
                **row,
                "request_id": request_id,
                "time": event_time,
                "switch_outputs": {
                    str(int(dpid)): sorted({int(port) for port in ports})
                    for dpid, ports in outputs.items()
                },
            }
        )
    rows.sort(key=lambda row: (row["time"], row["request_id"]))
    return rows


def load_vnf_control_events(
    path: str | None, event_kind: str
) -> list[dict[str, Any]]:
    if not path:
        return []
    rows = []
    for raw in read_jsonl(path):
        row = dict(raw)
        row["request_id"] = int(row["request_id"])
        row["time"] = float(row["time"])
        row["stage"] = int(row["stage"])
        if row["stage"] < 0:
            raise ValueError(f"{event_kind} stage must be non-negative")
        if event_kind == "migration":
            row["target_dc"] = int(row["target_dc"])
        elif event_kind == "impairment":
            row["processing_delay_us"] = max(
                0, int(row.get("processing_delay_us", 0))
            )
            row["drop_every"] = max(0, int(row.get("drop_every", 0)))
        rows.append(row)
    rows.sort(key=lambda row: (row["time"], row["request_id"], row["stage"]))
    return rows


def migrated_sft_plan(
    profile: dict[str, Any],
    plan: dict[str, Any],
    stage: int,
    target_dc: int,
    target_port: int,
    *,
    candidate_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one internal-VNF migration without changing an SFT candidate path.

    WQMIX candidates already contain the two bandwidth-aware paths used for
    scoring and joint decoding.  Runtime materialization may fill endpoint and
    switch-port fields, but it must not replace those paths.  Manual migration
    events without a candidate retain the explicit shortest-path fallback.
    """

    migrated = copy.deepcopy(candidate_plan if candidate_plan is not None else plan)
    segments = sorted(migrated["segments"], key=lambda value: int(value["stage"]))
    stage, target_dc = int(stage), int(target_dc)
    if not 0 < stage < len(segments):
        raise ValueError("online migration currently requires an internal SFT VNF")
    nodes, _ = topology_index(profile)
    if target_dc not in nodes or not bool(nodes[target_dc].get("is_dc")):
        raise ValueError(f"migration target {target_dc} is not a DC node")
    placement = migrated["placement_by_vnf"][str(stage)]
    current_placements = plan["placement_by_vnf"]
    old_dc = int(placement["dc_node"])
    current_dc = int(current_placements[str(stage)]["dc_node"])
    if current_dc == target_dc:
        raise ValueError("migration target equals the current VNF node")
    if candidate_plan is not None and old_dc != target_dc:
        raise ValueError(
            f"migration candidate target mismatch: plan={old_dc}, event={target_dc}"
        )
    for raw_stage, current_placement in current_placements.items():
        index = int(raw_stage)
        if index == stage:
            continue
        candidate_placement = migrated["placement_by_vnf"].get(str(index))
        if candidate_placement is None or int(candidate_placement["dc_node"]) != int(
            current_placement["dc_node"]
        ):
            raise ValueError(f"migration candidate changed unrelated VNF stage {index}")
    if candidate_plan is not None and migrated.get("multicast") != plan.get("multicast"):
        raise ValueError("migration candidate changed the multicast delivery tree")
    previous_dc = int(
        migrated["placement_by_vnf"][str(stage - 1)]["dc_node"]
    )
    next_dc = int(migrated["placement_by_vnf"][str(stage + 1)]["dc_node"])
    by_stage = {int(segment["stage"]): segment for segment in segments}
    if stage not in by_stage or stage + 1 not in by_stage:
        raise ValueError("migration candidate is missing an adjacent SFT segment")
    if candidate_plan is None:
        incoming_path, incoming_outputs = shortest_segment(
            profile, previous_dc, target_dc
        )
        outgoing_path, outgoing_outputs = shortest_segment(
            profile, target_dc, next_dc
        )
    else:
        incoming_path = [int(node) for node in by_stage[stage].get("path", [])]
        outgoing_path = [int(node) for node in by_stage[stage + 1].get("path", [])]
        if not incoming_path or incoming_path[0] != previous_dc or incoming_path[-1] != target_dc:
            raise ValueError("migration candidate incoming path has invalid endpoints")
        if not outgoing_path or outgoing_path[0] != target_dc or outgoing_path[-1] != next_dc:
            raise ValueError("migration candidate outgoing path has invalid endpoints")
        incoming_outputs = segment_outputs_for_path(profile, incoming_path)
        outgoing_outputs = segment_outputs_for_path(profile, outgoing_path)
    target_ip = str(nodes[target_dc]["host_ip"]).split("/", 1)[0]
    placement["dc_node"] = target_dc
    placement["listen_ip"] = target_ip
    placement["listen_port"] = int(target_port)
    migrated["chain_nodes"][stage] = target_dc
    by_stage[stage].update(
        {
            "from_dpid": previous_dc,
            "to_dpid": target_dc,
            "target_ip": target_ip,
            "udp_port": int(target_port),
            "path": incoming_path,
            "switch_outputs": incoming_outputs,
        }
    )
    next_placement = migrated["placement_by_vnf"][str(stage + 1)]
    by_stage[stage + 1].update(
        {
            "from_dpid": target_dc,
            "to_dpid": next_dc,
            "target_ip": str(next_placement["listen_ip"]),
            "udp_port": int(by_stage[stage + 1]["udp_port"]),
            "path": outgoing_path,
            "switch_outputs": outgoing_outputs,
        }
    )
    migrated["segments"] = segments
    migrated.setdefault("migration", {}).update(
        {"stage": stage, "old_dc": current_dc, "target_dc": target_dc}
    )
    return migrated


def migrated_sfc_plan(
    profile: dict[str, Any],
    plan: dict[str, Any],
    stage: int,
    target_dc: int,
    target_port: int,
) -> dict[str, Any]:
    """Backward-compatible manual-migration wrapper."""

    return migrated_sft_plan(profile, plan, stage, target_dc, target_port)


def live_edge_utilization(
    status: dict[str, Any], profile: dict[str, Any], edge: list[int] | tuple[int, int]
) -> tuple[float | None, dict[str, Any] | None]:
    if len(edge) != 2:
        return None, None
    u, v = int(edge[0]), int(edge[1])
    output_port = None
    for row in profile["edges"]:
        if int(row["u"]) == u and int(row["v"]) == v:
            output_port = int(row["u_port"])
            break
        if int(row["v"]) == u and int(row["u"]) == v:
            output_port = int(row["v_port"])
            break
    if output_port is None:
        return None, None
    sample = status.get("port_stats", {}).get(str(u), {}).get(str(output_port))
    if not isinstance(sample, dict) or sample.get("utilization") is None:
        return None, sample if isinstance(sample, dict) else None
    return float(sample["utilization"]), sample


def mininet_command(process: subprocess.Popen, host: str, command: str) -> None:
    mininet_commands(process, [(host, command)])


def mininet_commands(
    process: subprocess.Popen, commands: list[tuple[str, str]]
) -> None:
    if process.stdin is None:
        raise RuntimeError("Mininet stdin is unavailable")
    process.stdin.write(
        "".join(f"{host} {command}\n" for host, command in commands)
    )
    process.stdin.flush()


def mininet_barrier(
    process: subprocess.Popen,
    host: str,
    marker: str,
    log_path: Path,
    timeout: float,
) -> None:
    mininet_command(process, host, f"echo {shlex.quote(marker)}")
    wait_for_mininet_output(process, log_path, marker.encode("ascii"), timeout)


def wait_for_mininet_output(
    process: subprocess.Popen,
    log_path: Path,
    needle: bytes,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            output = log_path.read_bytes()
        except OSError:
            output = b""
        if needle in output:
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"Mininet exited before emitting {needle.decode('ascii', errors='replace')!r}"
            )
        time.sleep(0.05)
    raise RuntimeError(
        f"Mininet did not emit {needle.decode('ascii', errors='replace')!r} "
        f"within {timeout:.1f}s; inspect {log_path}"
    )


def probe_receiver_command(
    probe_path: str,
    request: dict[str, Any],
    interface_ip: str,
    duration: float,
    expected_packets: int,
    ready_file: str | None,
    stop_time_ns: int | None,
    expected_file: str,
    receive_buffer_bytes: int,
    output: str,
    agent_fifo: str | None = None,
    schedule_origin_file: str | None = None,
    start_offset_ns: int | None = None,
    stop_offset_ns: int | None = None,
    return_agent_message: bool = False,
    destination_id: int | None = None,
    receiver_backend: str = "python",
    native_receiver_path: str | None = None,
) -> str | dict[str, Any]:
    values = [
        "python3",
        probe_path,
        "receiver",
        "--group",
        str(request["multicast_ip"]),
        "--port",
        str(request["udp_port"]),
        "--duration",
        f"{duration:.6f}",
        "--interface-ip",
        interface_ip,
        "--delay-bound-ms",
        str(request["delay_bound_ms"]),
        "--delay-compliance-ratio",
        str(request.get("delay_compliance_ratio", 0.99)),
        "--packet-loss-bound",
        str(request["packet_loss_bound"]),
        "--grace-seconds",
        "0.5",
        "--expected-packets",
        str(expected_packets),
        "--expected-file",
        expected_file,
        "--receive-buffer-bytes",
        str(receive_buffer_bytes),
        "--output",
        output,
    ]
    if receiver_backend == "native":
        if not native_receiver_path:
            raise ValueError("native receiver backend requires its executable path")
        values.extend(
            [
                "--receiver-backend",
                "native",
                "--native-receiver-path",
                native_receiver_path,
            ]
        )
    elif receiver_backend != "python":
        raise ValueError(f"unsupported receiver backend {receiver_backend!r}")
    if ready_file:
        values.extend(["--ready-file", ready_file])
    if stop_time_ns is not None:
        values.extend(["--stop-time-ns", str(stop_time_ns)])
    elif schedule_origin_file is not None and stop_offset_ns is not None:
        values.extend(
            [
                "--schedule-origin-file",
                schedule_origin_file,
                "--stop-offset-ns",
                str(stop_offset_ns),
            ]
        )
    else:
        raise ValueError("receiver requires an absolute or scheduled stop time")
    if request.get("jitter_bound_ms") is not None:
        values.extend(["--jitter-bound-ms", str(request["jitter_bound_ms"])])
    if agent_fifo:
        message = probe_agent_task_message(
            agent_fifo,
            values[2:],
            schedule_origin_file,
            start_offset_ns,
            metadata={
                "request_id": int(request["id"]),
                "destination_id": (
                    int(destination_id) if destination_id is not None else None
                ),
            },
        )
        if return_agent_message:
            return message
        payload = json.dumps(message["payload"], separators=(",", ":"))
        return f"printf '%s\\n' {shlex.quote(payload)} > {shlex.quote(agent_fifo)}"
    return " ".join(shlex.quote(value) for value in values) + f" > {shlex.quote(output + '.stdout')} 2>&1 &"


def probe_sender_command(
    probe_path: str,
    request: dict[str, Any],
    duration: float,
    payload_bytes: int,
    packets_per_second: float,
    ready_files: list[str],
    ready_timeout: float,
    stop_time_ns: int | None,
    minimum_duration: float,
    expected_file: str,
    output: str,
    agent_fifo: str | None = None,
    schedule_origin_file: str | None = None,
    start_offset_ns: int | None = None,
    stop_offset_ns: int | None = None,
    activation_file: str | None = None,
    activation_timeout: float = 0.0,
    post_activation_delay: float = 0.0,
    max_catch_up_packets: int = 0,
    sender_backend: str = "python",
    native_sender_path: str | None = None,
    native_realtime_priority: int = 0,
    return_agent_message: bool = False,
    cpu_set: str = "",
) -> str | dict[str, Any]:
    pps = packets_per_second
    if pps <= 0.0:
        pps = bandwidth_mbps_to_pps(
            float(request["bw_origin"]), payload_bytes
        )
    values = [
        "python3",
        probe_path,
        "sender",
        "--destination",
        str(request["multicast_ip"]),
        "--port",
        str(request["udp_port"]),
        "--duration",
        f"{duration:.6f}",
        "--packets-per-second",
        f"{pps:.6f}",
        "--payload-bytes",
        str(payload_bytes),
        "--dscp",
        str(request["dscp"]),
        "--max-catch-up-packets",
        str(max_catch_up_packets),
        "--sender-backend",
        sender_backend,
        "--ready-timeout",
        f"{ready_timeout:.6f}",
        "--minimum-duration",
        f"{minimum_duration:.6f}",
        "--expected-file",
        expected_file,
        "--output",
        output,
    ]
    if sender_backend == "native":
        if not native_sender_path:
            raise ValueError("native sender backend requires an executable path")
        values.extend(["--native-sender-path", native_sender_path])
        values.extend(
            ["--native-realtime-priority", str(int(native_realtime_priority))]
        )
    if stop_time_ns is not None:
        values.extend(["--stop-time-ns", str(stop_time_ns)])
    elif schedule_origin_file is not None and stop_offset_ns is not None:
        values.extend(
            [
                "--schedule-origin-file",
                schedule_origin_file,
                "--stop-offset-ns",
                str(stop_offset_ns),
            ]
        )
    else:
        raise ValueError("sender requires an absolute or scheduled stop time")
    if activation_file:
        values.extend(
            [
                "--activation-file",
                activation_file,
                "--activation-timeout",
                f"{activation_timeout:.6f}",
                "--post-activation-delay",
                f"{post_activation_delay:.6f}",
            ]
        )
    for ready_file in ready_files:
        values.extend(["--ready-file", ready_file])
    if agent_fifo:
        message = probe_agent_task_message(
            agent_fifo,
            values[2:],
            schedule_origin_file,
            start_offset_ns,
            metadata={"request_id": int(request["id"])},
        )
        if return_agent_message:
            return message
        payload = json.dumps(message["payload"], separators=(",", ":"))
        return f"printf '%s\\n' {shlex.quote(payload)} > {shlex.quote(agent_fifo)}"
    command = " ".join(shlex.quote(value) for value in values)
    if cpu_set:
        command = f"taskset -c {shlex.quote(cpu_set)} {command}"
    return command + f" > {shlex.quote(output + '.stdout')} 2>&1 &"


def probe_agent_task_message(
    agent_fifo: str,
    argv: list[str],
    schedule_origin_file: str | None = None,
    start_offset_ns: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload_data: dict[str, Any] = {"argv": argv}
    if metadata:
        payload_data.update(
            {key: value for key, value in metadata.items() if value is not None}
        )
    if schedule_origin_file is not None or start_offset_ns is not None:
        if schedule_origin_file is None or start_offset_ns is None:
            raise ValueError("scheduled task requires both origin file and offset")
        payload_data["schedule_origin_file"] = schedule_origin_file
        payload_data["start_offset_ns"] = int(start_offset_ns)
    return {"fifo": agent_fifo, "payload": payload_data}


def probe_agent_task_command(
    agent_fifo: str,
    argv: list[str],
    schedule_origin_file: str | None = None,
    start_offset_ns: int | None = None,
) -> str:
    message = probe_agent_task_message(
        agent_fifo, argv, schedule_origin_file, start_offset_ns
    )
    payload = json.dumps(message["payload"], separators=(",", ":"))
    return (
        f"printf '%s\\n' {shlex.quote(payload)} > {shlex.quote(agent_fifo)}"
    )


def probe_agent_start_command(
    probe_path: str,
    agent_fifo: str,
    ready_file: str,
    log_file: str,
    sender_workers: int = 16,
    receiver_workers: int = 32,
    cpu_set: str = "",
    realtime_priority: int = 0,
) -> str:
    values = (
        ["chrt", "-r", str(int(realtime_priority))]
        if int(realtime_priority) > 0
        else []
    ) + (["taskset", "-c", normalize_cpu_set(cpu_set)] if cpu_set else []) + [
        "python3",
        probe_path,
        "agent",
        "--command-fifo",
        agent_fifo,
        "--ready-file",
        ready_file,
        "--sender-workers",
        str(sender_workers),
        "--receiver-workers",
        str(receiver_workers),
    ]
    return (
        f"rm -f {shlex.quote(agent_fifo)}; "
        + " ".join(shlex.quote(value) for value in values)
        + f" > {shlex.quote(log_file)} 2>&1 &"
    )


def read_probe_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def startup_failure_sender(request: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "mode": "sender",
        "destination": str(request["multicast_ip"]),
        "port": int(request["udp_port"]),
        "planned_packets": 0,
        "sent_packets": 0,
        "ready_files": len(request["destination_dpids"]),
        "ready_wait_seconds": 0.0,
        "status": "startup_deadline_missed",
        "error": reason,
    }


def startup_failure_receiver(
    request: dict[str, Any], destination: int, reason: str
) -> dict[str, Any]:
    return {
        "mode": "receiver",
        "group": str(request["multicast_ip"]),
        "port": int(request["udp_port"]),
        "destination_dpid": int(destination),
        "expected_packets": 0,
        "received_packets": 0,
        "lost_packets": 0,
        "packet_loss_rate": 1.0,
        "mean_delay_ms": None,
        "p50_delay_ms": None,
        "p95_delay_ms": None,
        "p99_delay_ms": None,
        "max_delay_ms": None,
        "jitter_ms": None,
        "delay_bound_ms": float(request["delay_bound_ms"]),
        "delay_compliance_ratio": float(
            request.get("delay_compliance_ratio", 0.99)
        ),
        "jitter_bound_ms": request.get("jitter_bound_ms"),
        "packet_loss_bound": float(request["packet_loss_bound"]),
        "expected_packets_source": "none",
        "measurement_status": "startup_deadline_missed",
        "error": reason,
        "delay_sla_met": False,
        "jitter_sla_met": False,
        "loss_sla_met": False,
        "sla_met": False,
    }


def wait_for_probe_files(
    paths: list[Path], timeout: float, poll_seconds: float = 0.1
) -> int:
    if timeout <= 0.0 or poll_seconds <= 0.0:
        raise ValueError("probe file timeout and poll interval must be positive")
    deadline = time.monotonic() + timeout
    observed = 0
    while time.monotonic() < deadline:
        observed = sum(path.is_file() for path in paths)
        if observed >= len(paths):
            return observed
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    return observed


def wait_for_json_file(
    path: Path, timeout: float, poll_seconds: float = 0.002
) -> dict[str, Any] | None:
    """Wait through WSL/Windows creation and permission propagation races."""
    if timeout <= 0.0 or poll_seconds <= 0.0:
        raise ValueError("JSON file timeout and poll interval must be positive")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = path.read_text(encoding="utf-8")
            value = json.loads(payload)
            if isinstance(value, dict):
                return value
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay dynamic SFT requests on Mininet.")
    parser.add_argument("--requests", required=True, help="requests.jsonl from runtime_request_generator.py")
    parser.add_argument(
        "--profile",
        default="sdn/topologies/us_backbone_28_bw90.json",
        help="Mininet profile; default matches the current 90 Mbps training datasets",
    )
    parser.add_argument(
        "--tree-plans",
        default=None,
        help="optional JSONL request_id/switch_outputs plans from Greedy, IDQN, or QMIX",
    )
    parser.add_argument(
        "--sfc-plans",
        default=None,
        help="HRL JSONL plans with VNF placement, chain segments, and multicast tree",
    )
    parser.add_argument(
        "--sfc-candidate-plans",
        default=None,
        help=(
            "deployment-topk batches.jsonl containing multiple complete SFC "
            "candidates per request"
        ),
    )
    parser.add_argument(
        "--sfc-candidate-selection",
        choices=("first", "greedy-feasible", "least-loaded", "sla-aware"),
        default="greedy-feasible",
        help="offline candidate selection policy at arrival time",
    )
    parser.add_argument(
        "--sfc-candidate-top-k",
        type=int,
        default=0,
        help="limit offline candidates per request; zero keeps all candidates",
    )
    parser.add_argument("--online-hrl-checkpoint", default=None)
    parser.add_argument("--online-hrl-data", default=None)
    parser.add_argument("--online-hrl-legacy-root", default=None)
    parser.add_argument("--online-hrl-seed", type=int, default=0)
    parser.add_argument("--online-hrl-max-steps", type=int, default=600)
    parser.add_argument(
        "--online-hrl-failure-step-budget",
        type=int,
        default=0,
        help="optional bounded step budget for no-progress HRL requests; zero keeps the legacy ceiling",
    )
    parser.add_argument(
        "--online-hrl-async-prefetch-workers",
        type=int,
        default=0,
        help="background ordered HRL prefetch workers; enable only when the next batch can overlap data-plane execution",
    )
    parser.add_argument("--online-hrl-bw-cap", type=float, default=90.0)
    parser.add_argument("--online-hrl-cpu-cap", type=float, default=55.0)
    parser.add_argument("--online-hrl-mem-cap", type=float, default=45.0)
    parser.add_argument("--online-hrl-torch-threads", type=int, default=1)
    parser.add_argument(
        "--online-hrl-batched-policy",
        action="store_true",
        help=(
            "score current-snapshot complete plans with frozen batched HRL "
            "policy heads instead of running the stateful legacy rollout"
        ),
    )
    parser.add_argument("--online-hrl-batched-microbatch-ms", type=float, default=5.0)
    parser.add_argument("--online-hrl-batched-max-agents", type=int, default=32)
    parser.add_argument("--online-hrl-batched-top-k", type=int, default=4)
    parser.add_argument("--online-hrl-batched-workers", type=int, default=4)
    parser.add_argument(
        "--online-hrl-no-static-topology-cache",
        action="store_true",
        help="disable immutable topology/hop/path caching for the online HRL ablation",
    )
    parser.add_argument("--online-hrl-safe-dest-recovery", action="store_true")
    parser.add_argument("--online-hrl-planner-destinations", action="store_true")
    parser.add_argument(
        "--online-hrl-k-path-candidate-filter",
        action="store_true",
        help="prewarm K topology paths and let the frozen low-level RL policy select among their feasible first hops",
    )
    parser.add_argument(
        "--online-hrl-k-path-candidate-k",
        type=int,
        default=2,
        help="number of prewarmed paths used to constrain low-level RL candidates (use at least 2 to preserve a meaningful choice)",
    )
    parser.add_argument(
        "--online-hrl-macro-path-rollout",
        action="store_true",
        help="reuse the low-level RL selected complete path for its suffix hops",
    )
    parser.add_argument(
        "--online-hrl-fast-mode",
        action="store_true",
        help=(
            "enable the validated online HRL speed profile: Top-2 path filtering, "
            "macro path rollout, four completion-certified high-level choices, "
            "beam-8 multicast decoding, and bounded failure-step handling"
        ),
    )
    parser.add_argument(
        "--online-hrl-process-isolation",
        action="store_true",
        help="run the single stateful HRL planner in a dedicated Python process; fast mode enables this automatically",
    )
    parser.add_argument(
        "--online-hrl-completion-candidate-budget",
        type=int,
        default=0,
        help="number of fully completion-certified DC choices; zero uses 4 in fast mode and unlimited otherwise",
    )
    parser.add_argument(
        "--online-hrl-destination-beam-width",
        type=int,
        default=0,
        help="multicast completion beam width; zero uses 8 in fast mode and 64 otherwise",
    )
    parser.add_argument(
        "--online-hrl-fast-k-path-candidates",
        action="store_true",
        help="derive low-level candidates directly from the prewarmed K-path table; experimental",
    )
    parser.add_argument("--online-hrl-reseed-per-request", action="store_true")
    parser.add_argument("--online-hrl-verbose", action="store_true")
    parser.add_argument("--online-wqmix-checkpoint", default=None)
    parser.add_argument(
        "--online-wqmix-data",
        default=None,
        help="deployment_topk_v3 folder containing batches.jsonl and dataset_spec.json",
    )
    parser.add_argument("--online-wqmix-microbatch-ms", type=float, default=5.0)
    parser.add_argument(
        "--online-wqmix-bandwidth-utilization-limit",
        type=float,
        default=1.0,
    )
    parser.add_argument("--online-wqmix-device", default="cpu")
    parser.add_argument("--online-wqmix-torch-threads", type=int, default=1)
    parser.add_argument("--online-wqmix-decoder-top-r", type=int, default=4)
    parser.add_argument(
        "--online-wqmix-decoder-time-budget-ms", type=float, default=2.0
    )
    parser.add_argument(
        "--online-wqmix-repair-missing-plans",
        action="store_true",
        help=(
            "generate current-snapshot complete plans when an offline Top-K row "
            "is missing or contains only reject"
        ),
    )
    parser.add_argument(
        "--online-wqmix-hybrid-least-loaded",
        action="store_true",
        help=(
            "use least-loaded for singleton batches and fuse least-loaded with "
            "WQMIX rankings for multi-request batches"
        ),
    )
    parser.add_argument(
        "--online-wqmix-hybrid-dataset-selected",
        action="store_true",
        help=(
            "keep each request's validated dataset action first while feasible, "
            "and use WQMIX only for conflicting multi-request batches"
        ),
    )
    parser.add_argument("--online-wqmix-hybrid-rl-weight", type=float, default=0.25)
    parser.add_argument(
        "--online-wqmix-q0-sla-safety-margin",
        type=float,
        default=0.0,
        help="prefer only Q0 candidates below this predicted delay/bound ratio; zero disables",
    )
    parser.add_argument("--online-wqmix-q0-hard-sla-gate", action="store_true")
    parser.add_argument(
        "--online-wqmix-sla-queue-safety-factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--online-wqmix-sla-calibration",
        "--online-wqmix-sla-predictor",
        dest="online_wqmix_sla_calibration",
        default=None,
        help=(
            "supervised SLA predictor checkpoint or legacy empirical calibration; "
            "both must be built from real probe results"
        ),
    )
    parser.add_argument(
        "--online-wqmix-sla-calibration-rank-weight",
        type=float,
        default=0.25,
        help="weight of calibrated failure probability during candidate reranking",
    )
    parser.add_argument(
        "--online-wqmix-sla-min-rerank-delta",
        type=float,
        default=0.05,
        help="minimum candidate failure-probability difference required to alter ranking",
    )
    parser.add_argument(
        "--online-wqmix-sla-max-ood-score",
        type=float,
        default=4.0,
        help="ignore learned SLA predictions beyond this standardized OOD score",
    )
    parser.add_argument(
        "--online-wqmix-max-sla-failure-probability",
        type=float,
        default=None,
        help="optional hard rejection threshold in [0,1]; disabled by default",
    )
    parser.add_argument(
        "--reroute-events",
        default=None,
        help="optional JSONL runtime reroutes with time/request_id/switch_outputs",
    )
    parser.add_argument("--reroute-drain-seconds", type=float, default=0.05)
    parser.add_argument(
        "--vnf-migration-events",
        default=None,
        help="optional JSONL events with time/request_id/stage/target_dc",
    )
    parser.add_argument(
        "--vnf-impairment-events",
        default=None,
        help=(
            "optional JSONL hotspot events with time/request_id/stage and "
            "processing_delay_us/drop_every"
        ),
    )
    parser.add_argument("--vnf-migration-drain-ms", type=float, default=50.0)
    parser.add_argument("--vnf-migration-idle-ms", type=float, default=8.0)
    parser.add_argument(
        "--online-wqmix-auto-migration",
        action="store_true",
        help=(
            "turn impairment events into WQMIX migration decisions; each active "
            "request scores all feasible DC targets online"
        ),
    )
    parser.add_argument(
        "--online-migration-wqmix-checkpoint",
        default=None,
        help=(
            "migration_wqmix_v1 checkpoint used for VNF target selection; "
            "deployment WQMIX checkpoints are rejected"
        ),
    )
    parser.add_argument("--online-migration-max-agents", type=int, default=8)
    parser.add_argument("--online-migration-max-inflight", type=int, default=4)
    parser.add_argument("--online-migration-cpu-capacity", type=float, default=55.0)
    parser.add_argument("--online-migration-memory-capacity", type=float, default=45.0)
    parser.add_argument(
        "--predictive-migration-scan-seconds",
        type=float,
        default=0.0,
        help="periodic online ledger scan; zero disables prediction-driven triggers",
    )
    parser.add_argument("--predictive-migration-overload-threshold", type=float, default=0.85)
    parser.add_argument("--predictive-migration-safe-utilization", type=float, default=0.75)
    parser.add_argument("--predictive-migration-horizon-seconds", type=float, default=1.0)
    parser.add_argument("--predictive-migration-cooldown-seconds", type=float, default=5.0)
    parser.add_argument(
        "--predictive-migration-min-remaining-lifetime-seconds",
        type=float,
        default=1.344,
        help=(
            "do not migrate a VNF when its request has less trace lifetime left; "
            "the default preserves the existing 1.344 second safety gate"
        ),
    )
    parser.add_argument("--online-wqmix-migration-delay-seconds", type=float, default=0.4)
    parser.add_argument(
        "--strict-reroute-gates",
        action="store_true",
        help="require lifetime, gain, modeled drop, cooldown, and live hotspot gates",
    )
    parser.add_argument("--reroute-min-remaining-lifetime", type=float, default=1.0)
    parser.add_argument("--reroute-min-estimated-gain", type=float, default=0.30)
    parser.add_argument("--reroute-min-old-utilization", type=float, default=0.85)
    parser.add_argument("--reroute-min-utilization-drop", type=float, default=0.20)
    parser.add_argument("--reroute-min-cooldown-seconds", type=float, default=1.0)
    parser.add_argument("--reroute-live-utilization-threshold", type=float, default=0.75)
    parser.add_argument(
        "--output", default="artifacts/runs/mininet/runtime_replay/result.json"
    )
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument(
        "--rest-url",
        default="auto",
        help=(
            "Ryu REST base URL; auto uses the current WSL IPv4 address and "
            "avoids relying on Windows localhost forwarding"
        ),
    )
    parser.add_argument("--controller-service", default="ryu-sft-static-controller.service")
    parser.add_argument(
        "--controller-realtime-priority",
        type=int,
        default=0,
        help=(
            "temporarily run the Ryu systemd MainPID with SCHED_RR; "
            "zero disables it and SCHED_OTHER is restored on exit"
        ),
    )
    parser.add_argument("--controller-cpu-set", default="")
    parser.add_argument("--ovs-service", default="ovs-vswitchd.service")
    parser.add_argument("--ovs-cpu-set", default="")
    parser.add_argument("--mininet-cpu-set", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--skip-requests", type=int, default=0)
    parser.add_argument("--payload-bytes", type=int, default=1200)
    parser.add_argument(
        "--packet-overhead-bytes",
        type=int,
        default=DEFAULT_PACKET_OVERHEAD_BYTES,
        help=(
            "per-packet Ethernet+IPv4+UDP bytes included when converting "
            "bw_origin link Mbps to packet rate"
        ),
    )
    parser.add_argument(
        "--probe-receive-buffer-bytes",
        type=int,
        default=4 * 1024 * 1024,
        help="requested SO_RCVBUF for each UDP receiver",
    )
    parser.add_argument("--packets-per-second", type=float, default=0.0)
    parser.add_argument(
        "--offered-load-compliance-ratio",
        type=float,
        default=0.99,
        help=(
            "minimum sent/planned packet ratio required for request-level SLA; "
            "sender pacing misses count as failures"
        ),
    )
    parser.add_argument(
        "--sender-max-catch-up-packets",
        type=int,
        default=0,
        help=(
            "maximum overdue pacing slots sent as a bounded burst; zero drops "
            "all overdue slots"
        ),
    )
    parser.add_argument(
        "--receiver-ready-seconds",
        type=float,
        default=0.2,
        help="maximum receiver handshake budget reserved before traffic starts",
    )
    parser.add_argument(
        "--receiver-ready-protocol",
        choices=("ack", "file"),
        default="ack",
        help=(
            "wait for receiver bind ACKs over local Unix datagrams or use the "
            "legacy sender-side ready-file polling protocol"
        ),
    )
    parser.add_argument(
        "--probe-tree-warmup-seconds",
        type=float,
        default=0.2,
        help="sender start offset reserved for initial tree installation",
    )
    parser.add_argument(
        "--sender-stop-margin-seconds",
        type=float,
        default=0.1,
        help="drain window between the final send and SFC teardown",
    )
    parser.add_argument("--minimum-traffic-seconds", type=float, default=0.05)
    parser.add_argument("--probe-result-timeout", type=float, default=5.0)
    parser.add_argument(
        "--probe-launch-mode",
        choices=("agent", "process"),
        default="agent",
        help="persistent per-host agents avoid Python process startup on every request",
    )
    parser.add_argument(
        "--probe-sender-launch-mode",
        choices=("agent", "process"),
        default="agent",
        help=(
            "run senders in the probe agent thread pool or isolated processes; "
            "process mode avoids sender GIL contention while receivers stay resident"
        ),
    )
    parser.add_argument(
        "--probe-sender-prelaunch",
        action="store_true",
        help=(
            "experimental: launch the sender before receiver ACKs and gate it "
            "on receiver ready files; disabled by default because it can increase "
            "control/data-plane contention"
        ),
    )
    parser.add_argument(
        "--probe-sender-backend",
        choices=("python", "native"),
        default="python",
        help="Python pacing loop or compiled CLOCK_MONOTONIC C sender",
    )
    parser.add_argument(
        "--probe-receiver-backend",
        choices=("python", "native"),
        default="python",
        help=(
            "Python receive loop or compiled C receiver using kernel packet "
            "timestamps"
        ),
    )
    parser.add_argument(
        "--probe-native-realtime-priority",
        type=int,
        default=0,
        help=(
            "SCHED_RR priority for native senders; zero is the scalable default "
            "because many realtime sender processes can starve Ryu and receivers"
        ),
    )
    parser.add_argument(
        "--vnf-launch-mode",
        choices=("agent", "process"),
        default="agent",
        help="persistent per-DC VNF agents or one process per request stage",
    )
    parser.add_argument(
        "--vnf-ready-protocol",
        choices=("ack", "file"),
        default="ack",
        help=(
            "local Unix datagram ACKs avoid WSL/Windows ready-file polling; "
            "file preserves the legacy compatibility path"
        ),
    )
    parser.add_argument(
        "--vnf-agent-backend",
        choices=("python", "native"),
        default="python",
        help="persistent VNF data path; native uses the compiled C FIFO-compatible agent",
    )
    parser.add_argument(
        "--vnf-agent-workers",
        type=int,
        default=4,
        help="internal worker-process count for each Python VNF agent",
    )
    parser.add_argument(
        "--vnf-agent-native-shards",
        type=int,
        default=1,
        help=(
            "resident C agent processes per DC; one is the measured default, "
            "larger values are an experimental data-plane sharding control"
        ),
    )
    parser.add_argument(
        "--vnf-agent-initial-workers",
        type=int,
        default=0,
        help="initial workers per DC; zero starts all --vnf-agent-workers",
    )
    parser.add_argument(
        "--vnf-agent-bindings-per-worker",
        type=int,
        default=0,
        help="dynamic scale-out threshold; zero disables VNF worker scaling",
    )
    parser.add_argument(
        "--vnf-agent-prewarm-workers",
        action="store_true",
        help="pre-fork and freeze idle dynamic VNF workers",
    )
    parser.add_argument("--vnf-agent-prebound-port-base", type=int, default=30000)
    parser.add_argument(
        "--vnf-agent-prebound-port-count",
        type=int,
        default=0,
        help=(
            "reserve reusable UDP endpoints per DC and rewrite SFC stage ports; "
            "Python agents prebind them and native agents bind on registration"
        ),
    )
    parser.add_argument(
        "--vnf-drain-timeout-ms",
        type=float,
        default=100.0,
        help="maximum explicit queue-drain wait for each VNF stage",
    )
    parser.add_argument(
        "--vnf-drain-idle-ms",
        type=float,
        default=5.0,
        help="socket idle interval that confirms one VNF stage is drained",
    )
    parser.add_argument(
        "--vnf-agent-realtime-priority",
        type=int,
        default=0,
        help="SCHED_RR priority for persistent VNF workers; zero disables it",
    )
    parser.add_argument(
        "--vnf-agent-cpu-set",
        default="",
        help="optional Linux CPU list inherited by persistent VNF workers",
    )
    parser.add_argument(
        "--vnf-agent-packet-batch",
        type=int,
        default=64,
        help="packets handled per ready VNF binding before yielding",
    )
    parser.add_argument(
        "--vnf-agent-q0-packet-batch",
        type=int,
        default=0,
        help="Q0/DSCP-EF packet burst override; zero uses --vnf-agent-packet-batch",
    )
    parser.add_argument(
        "--vnf-agent-dscp-scheduling",
        action="store_true",
        help=(
            "serve ready native VNF bindings by Q0/Q1/Q2 class each cycle; "
            "all classes still receive service once per cycle"
        ),
    )
    parser.add_argument(
        "--probe-preschedule",
        action="store_true",
        help="experimental: preload absolute-time probe tasks into role-separated agents",
    )
    parser.add_argument(
        "--probe-agent-sender-workers",
        type=int,
        default=0,
        help="sender pool size; zero derives it from trace peak concurrency",
    )
    parser.add_argument(
        "--probe-agent-receiver-workers",
        type=int,
        default=0,
        help="receiver pool size; zero derives it from trace peak concurrency",
    )
    parser.add_argument(
        "--probe-agent-cpu-set",
        default="",
        help="optional Linux CPU list inherited by probe agents and native senders",
    )
    parser.add_argument(
        "--probe-agent-realtime-priority",
        type=int,
        default=0,
        help="SCHED_RR priority inherited by persistent probe receiver threads; zero disables it",
    )
    parser.add_argument("--switch-start-delay", type=float, default=0.02)
    parser.add_argument(
        "--mininet-qdisc",
        choices=(
            "htb",
            "htb_fq",
            "htb_fq_codel",
            "htb_prio",
            "htb_prio_fq_codel",
            "tbf",
            "hfsc",
        ),
        default="htb",
    )
    parser.add_argument("--mininet-max-queue-size", type=int, default=1000)
    parser.add_argument(
        "--mininet-active-hosts-only",
        action="store_true",
        help=(
            "create hosts only for selected sources, destinations, and all Top-K "
            "VNF placements; all topology switches and links remain present"
        ),
    )
    parser.add_argument("--mininet-command-port", type=int, default=8765)
    parser.add_argument(
        "--mininet-command-host",
        default="auto",
        help=(
            "Windows-reachable Mininet runtime host; auto discovers the current "
            "WSL IPv4 address instead of relying on localhost forwarding"
        ),
    )
    parser.add_argument(
        "--sfc-barrier-mode",
        choices=("staged", "single"),
        default="staged",
        help="staged safety barriers or one all-switch barrier per new SFC",
    )
    parser.add_argument(
        "--ryu-commit-batch-ms",
        type=float,
        default=0.0,
        help=(
            "collect VNF-ready SFCs for this many milliseconds and commit them "
            "with one Ryu batch barrier; zero keeps per-request REST commits"
        ),
    )
    parser.add_argument(
        "--ryu-commit-batch-size",
        type=int,
        default=16,
        help="maximum SFC requests in one Ryu commit batch",
    )
    parser.add_argument(
        "--ryu-batch-sender-stagger-ms",
        type=float,
        default=0.0,
        help="sender start spacing by position within one Ryu commit batch",
    )
    parser.add_argument(
        "--deployment-workers",
        type=int,
        default=1,
        help=(
            "request-event worker count; values above one deploy independent "
            "requests concurrently while preserving per-request event order"
        ),
    )
    parser.add_argument(
        "--parallel-deployment-pipeline",
        action="store_true",
        help=(
            "enable the bounded deployment pipeline: overlap neighbor setup with "
            "VNF registration, micro-batch VNF control and Ryu commits, and use "
            "the online WQMIX ledger as the only admission authority"
        ),
    )
    parser.add_argument(
        "--vnf-registration-concurrency",
        type=int,
        default=0,
        help=(
            "maximum concurrent Mininet VNF register/ready transactions; "
            "zero matches --deployment-workers"
        ),
    )
    parser.add_argument(
        "--vnf-register-batch-ms",
        type=float,
        default=0.0,
        help=(
            "collect concurrent request registrations for one Mininet FIFO/ACK "
            "transaction; zero keeps one transaction per request"
        ),
    )
    parser.add_argument(
        "--vnf-register-batch-size",
        type=int,
        default=16,
        help="maximum requests in one cross-request VNF registration batch",
    )
    parser.add_argument(
        "--vnf-unregister-batch-ms",
        type=float,
        default=0.0,
        help=(
            "collect drained request unregistrations for one Mininet FIFO/ACK "
            "transaction; zero keeps one transaction per request"
        ),
    )
    parser.add_argument(
        "--vnf-unregister-batch-size",
        type=int,
        default=32,
        help="maximum requests in one cross-request VNF unregister batch",
    )
    parser.add_argument(
        "--vnf-fast-unregister",
        action="store_true",
        help=(
            "let unregister perform drain+close with one Unix ACK while Ryu paths "
            "remain installed; removes the separate drain-file transaction"
        ),
    )
    parser.add_argument(
        "--deployment-max-outstanding",
        type=int,
        default=0,
        help=(
            "maximum admitted arrival requests that may be planning, deploying, "
            "or waiting; zero keeps the legacy unlimited queue"
        ),
    )
    parser.add_argument(
        "--planning-max-outstanding",
        type=int,
        default=0,
        help=(
            "maximum online requests queued or active in the planning phase; "
            "zero disables the planning-phase limit"
        ),
    )
    parser.add_argument(
        "--execution-max-outstanding",
        type=int,
        default=0,
        help=(
            "maximum planned requests queued or active in deployment; zero "
            "disables the execution-phase limit"
        ),
    )
    parser.add_argument(
        "--deployment-max-queue-wait-ms",
        type=float,
        default=0.0,
        help=(
            "reject an admitted arrival if deployment starts this long after "
            "its trace arrival; zero disables the queue-wait deadline"
        ),
    )
    parser.add_argument(
        "--cleanup-workers",
        type=int,
        default=4,
        help=(
            "background worker count for leave events in parallel mode; "
            "four sustains the default 8 requests/s trace without delaying "
            "VNF endpoint reuse"
        ),
    )
    parser.add_argument(
        "--deployment-admission-estimate-ms",
        type=float,
        default=150.0,
        help=(
            "estimated setup time reserved by pre-deployment lifetime admission; "
            "zero disables the additional setup reserve"
        ),
    )
    parser.add_argument(
        "--deployment-admission-window",
        type=int,
        default=128,
        help="number of recent successful setup measurements used by rolling P95",
    )
    parser.add_argument(
        "--deployment-admission-min-samples",
        type=int,
        default=8,
        help="successful setups required before replacing the cold-start estimate",
    )
    parser.add_argument(
        "--deployment-admission-safety-factor",
        type=float,
        default=1.20,
        help="multiplier applied to the rolling setup-time P95",
    )
    parser.add_argument(
        "--deployment-bandwidth-utilization-limit",
        type=float,
        default=0.0,
        help=(
            "maximum reserved bandwidth fraction per physical link before an "
            "arrival is rejected; 0 disables this admission check"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_sla_predictor_runtime_contract(
    contract: dict[str, Any] | None, args: argparse.Namespace
) -> dict[str, Any] | None:
    if not contract:
        return None
    expected = {
        "probe_sender_backend": args.probe_sender_backend,
        "probe_receiver_backend": args.probe_receiver_backend,
        "vnf_agent_backend": args.vnf_agent_backend,
        "mininet_qdisc": args.mininet_qdisc,
        "vnf_agent_packet_batch": int(args.vnf_agent_packet_batch),
        "vnf_agent_q0_packet_batch": int(args.vnf_agent_q0_packet_batch),
        "vnf_agent_dscp_scheduling": bool(args.vnf_agent_dscp_scheduling),
    }
    mismatches = {
        key: {"trained": contract.get(key), "runtime": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    runtime_drain_ms = float(args.sender_stop_margin_seconds) * 1000.0
    required_drain_ms = float(contract.get("required_receiver_drain_ms", 0.0))
    if runtime_drain_ms + 1e-6 < required_drain_ms:
        mismatches["receiver_drain_ms"] = {
            "trained_minimum": required_drain_ms,
            "runtime": runtime_drain_ms,
        }
    if mismatches:
        raise ValueError(
            "SLA predictor runtime contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return {
        "validated": True,
        "label_semantics": contract.get("label_semantics"),
        "runtime_receiver_drain_ms": runtime_drain_ms,
        "matched": expected,
    }


def main() -> int:
    global _MININET_RUNTIME_HOST
    args = parse_args()
    if args.parallel_deployment_pipeline:
        if args.vnf_launch_mode != "agent" or args.vnf_ready_protocol != "ack":
            raise ValueError(
                "--parallel-deployment-pipeline requires --vnf-launch-mode "
                "agent and --vnf-ready-protocol ack"
            )
        args.deployment_workers = max(4, int(args.deployment_workers))
        args.vnf_registration_concurrency = max(
            2, int(args.vnf_registration_concurrency)
        )
        args.vnf_register_batch_ms = max(1.0, float(args.vnf_register_batch_ms))
        args.ryu_commit_batch_ms = max(1.0, float(args.ryu_commit_batch_ms))
    for cpu_set_argument in (
        "controller_cpu_set",
        "ovs_cpu_set",
        "mininet_cpu_set",
        "vnf_agent_cpu_set",
        "probe_agent_cpu_set",
    ):
        setattr(
            args,
            cpu_set_argument,
            normalize_cpu_set(getattr(args, cpu_set_argument)),
        )
    online_hrl_enabled = bool(args.online_hrl_checkpoint)
    online_wqmix_enabled = bool(args.online_wqmix_checkpoint)
    online_planner_enabled = online_hrl_enabled or online_wqmix_enabled
    plan_modes = sum(
        bool(value)
        for value in (args.sfc_plans, args.sfc_candidate_plans, args.tree_plans)
    )
    if plan_modes + int(online_hrl_enabled) + int(online_wqmix_enabled) > 1:
        raise ValueError(
            "precomputed plans, online HRL, and online WQMIX are mutually exclusive"
        )
    if online_hrl_enabled and not (
        args.online_hrl_data and args.online_hrl_legacy_root and args.online_hrl_seed
    ):
        raise ValueError(
            "online HRL requires --online-hrl-data, --online-hrl-legacy-root, "
            "and a non-zero --online-hrl-seed"
        )
    if online_hrl_enabled and args.skip_requests:
        raise ValueError("online HRL currently requires --skip-requests 0")
    if online_wqmix_enabled and not args.online_wqmix_data:
        raise ValueError("online WQMIX requires --online-wqmix-data")
    if online_wqmix_enabled and args.skip_requests:
        raise ValueError("online WQMIX currently requires --skip-requests 0")
    if args.online_wqmix_sla_calibration and not online_wqmix_enabled:
        raise ValueError("SLA calibration requires --online-wqmix-checkpoint")
    if args.online_wqmix_sla_calibration_rank_weight < 0.0:
        raise ValueError("SLA calibration rank weight must be non-negative")
    if not 0.0 <= args.online_wqmix_sla_min_rerank_delta <= 1.0:
        raise ValueError("SLA rerank delta must be in [0, 1]")
    if args.online_wqmix_sla_max_ood_score <= 0.0:
        raise ValueError("SLA maximum OOD score must be positive")
    if args.online_wqmix_max_sla_failure_probability is not None and not (
        0.0 <= args.online_wqmix_max_sla_failure_probability <= 1.0
    ):
        raise ValueError("maximum SLA failure probability must be in [0, 1]")
    if (
        args.online_wqmix_max_sla_failure_probability is not None
        and not args.online_wqmix_sla_calibration
    ):
        raise ValueError("SLA probability gate requires --online-wqmix-sla-calibration")
    if (args.vnf_migration_events or args.vnf_impairment_events) and (
        args.vnf_launch_mode != "agent" or args.vnf_ready_protocol != "ack"
    ):
        raise ValueError("VNF migration/hotspot events require agent ACK mode")
    if args.vnf_migration_events and not args.vnf_agent_prebound_port_count:
        raise ValueError("VNF migration requires a non-empty prebound endpoint pool")
    if args.online_wqmix_auto_migration and not online_wqmix_enabled:
        raise ValueError("online WQMIX migration requires --online-wqmix-checkpoint")
    if args.online_wqmix_auto_migration and not args.online_migration_wqmix_checkpoint:
        raise ValueError(
            "online automatic migration requires "
            "--online-migration-wqmix-checkpoint; deployment weights are not migration weights"
        )
    if args.online_migration_max_agents <= 0 or args.online_migration_max_inflight <= 0:
        raise ValueError("online migration batch and inflight limits must be positive")
    if args.predictive_migration_scan_seconds < 0.0:
        raise ValueError("predictive migration scan interval must be non-negative")
    if args.predictive_migration_min_remaining_lifetime_seconds < 0.0:
        raise ValueError("predictive migration minimum remaining lifetime must be non-negative")
    if args.predictive_migration_scan_seconds > 0.0 and not (
        online_wqmix_enabled and args.online_migration_wqmix_checkpoint
    ):
        raise ValueError(
            "prediction-driven migration requires online deployment WQMIX and "
            "a migration-specific WQMIX checkpoint"
        )
    if args.online_wqmix_migration_delay_seconds < 0.0:
        raise ValueError("online WQMIX migration delay must be non-negative")
    if (args.sfc_plans or args.sfc_candidate_plans or online_planner_enabled) and args.probe_preschedule:
        raise ValueError("--probe-preschedule is not supported with dynamic VNF startup")
    if args.probe_preschedule and args.probe_sender_launch_mode != "agent":
        raise ValueError("--probe-preschedule requires --probe-sender-launch-mode agent")
    if args.vnf_agent_dscp_scheduling and args.vnf_agent_backend != "native":
        raise ValueError("--vnf-agent-dscp-scheduling requires the native backend")
    if args.time_scale <= 0.0 or args.max_requests < 0 or args.skip_requests < 0:
        raise ValueError("time-scale must be positive and request limits non-negative")
    if (
        args.receiver_ready_seconds <= 0.0
        or args.payload_bytes <= 0
        or args.packet_overhead_bytes < 0
        or args.sender_stop_margin_seconds < 0.0
        or not 0.0 < args.offered_load_compliance_ratio <= 1.0
        or args.sender_max_catch_up_packets < 0
        or not 0 <= args.probe_native_realtime_priority <= 50
        or args.minimum_traffic_seconds <= 0.0
        or args.probe_result_timeout <= 0.0
        or args.probe_receive_buffer_bytes <= 0
        or args.mininet_max_queue_size <= 0
        or not 0 <= args.probe_agent_sender_workers <= 128
        or not 0 <= args.probe_agent_receiver_workers <= 128
        or not 0 <= args.probe_agent_realtime_priority <= 50
        or not 1 <= args.vnf_agent_workers <= 32
        or not 1 <= args.vnf_agent_native_shards <= 32
        or not 0 <= args.vnf_agent_initial_workers <= args.vnf_agent_workers
        or args.vnf_agent_bindings_per_worker < 0
        or args.vnf_agent_prebound_port_count < 0
        or args.vnf_drain_timeout_ms <= 0.0
        or args.vnf_drain_idle_ms <= 0.0
        or args.vnf_drain_idle_ms > args.vnf_drain_timeout_ms
        or not 0 <= args.vnf_agent_realtime_priority <= 50
        or args.vnf_agent_packet_batch <= 0
        or args.vnf_agent_q0_packet_batch < 0
        or not 0 <= args.controller_realtime_priority <= 50
        or not 0.0 <= args.probe_tree_warmup_seconds <= 1.0
        or not 0.0 <= args.reroute_drain_seconds <= 10.0
        or args.reroute_min_remaining_lifetime < 0.0
        or args.reroute_min_estimated_gain < 0.0
        or not 0.0 <= args.reroute_min_old_utilization <= 1.0
        or not 0.0 <= args.reroute_min_utilization_drop <= 1.0
        or args.reroute_min_cooldown_seconds < 0.0
        or not 0.0 < args.reroute_live_utilization_threshold <= 1.0
        or not 1 <= args.mininet_command_port <= 65535
        or args.ryu_commit_batch_ms < 0.0
        or args.ryu_batch_sender_stagger_ms < 0.0
        or not 1 <= args.ryu_commit_batch_size <= 64
        or not 1 <= args.deployment_workers <= 32
        or not 0 <= args.vnf_registration_concurrency <= 32
        or args.vnf_register_batch_ms < 0.0
        or not 1 <= args.vnf_register_batch_size <= 64
        or args.vnf_unregister_batch_ms < 0.0
        or not 1 <= args.vnf_unregister_batch_size <= 64
        or args.deployment_max_outstanding < 0
        or args.planning_max_outstanding < 0
        or args.execution_max_outstanding < 0
        or args.deployment_max_queue_wait_ms < 0.0
        or not 1 <= args.cleanup_workers <= 8
        or args.deployment_admission_estimate_ms < 0.0
        or args.deployment_admission_window <= 0
        or not 1 <= args.deployment_admission_min_samples <= args.deployment_admission_window
        or args.deployment_admission_safety_factor <= 0.0
        or not 0.0 <= args.deployment_bandwidth_utilization_limit <= 1.0
        or args.sfc_candidate_top_k < 0
        or args.online_hrl_torch_threads < 0
        or args.online_hrl_failure_step_budget < 0
        or args.online_hrl_async_prefetch_workers < 0
        or args.online_hrl_k_path_candidate_k < 2
        or args.online_hrl_completion_candidate_budget < 0
        or args.online_hrl_destination_beam_width < 0
        or args.online_wqmix_microbatch_ms < 0.0
        or not 0.0 < args.online_wqmix_bandwidth_utilization_limit <= 1.0
        or args.online_wqmix_q0_sla_safety_margin < 0.0
        or args.online_wqmix_sla_queue_safety_factor < 0.0
        or not 0.0 <= args.online_wqmix_hybrid_rl_weight <= 1.0
        or args.online_wqmix_torch_threads < 0
        or args.online_wqmix_decoder_top_r <= 0
        or args.online_wqmix_decoder_time_budget_ms < 0.0
    ):
        raise ValueError("invalid probe timing value or Mininet command port")
    if args.vnf_agent_prebound_port_count:
        if not (
            1024 <= args.vnf_agent_prebound_port_base <= 65535
            and args.vnf_agent_prebound_port_base
            + args.vnf_agent_prebound_port_count
            <= 65536
        ):
            raise ValueError("prebound VNF port range must stay within 1024..65535")
        if (
            args.vnf_agent_backend == "python"
            and int(args.vnf_agent_initial_workers or args.vnf_agent_workers)
            != int(args.vnf_agent_workers)
        ):
            raise ValueError("prebound VNF endpoints require all workers resident")
    if (
        args.online_wqmix_hybrid_least_loaded
        and args.online_wqmix_hybrid_dataset_selected
    ):
        raise ValueError(
            "online WQMIX hybrid least-loaded and dataset-selected modes are "
            "mutually exclusive"
        )
    if (args.vnf_register_batch_ms > 0.0 or args.vnf_unregister_batch_ms > 0.0) and (
        args.vnf_launch_mode != "agent" or args.vnf_ready_protocol != "ack"
    ):
        raise ValueError("VNF control batching requires agent launch mode and ACK protocol")
    if args.vnf_fast_unregister and (
        args.vnf_launch_mode != "agent"
        or args.vnf_ready_protocol != "ack"
        or args.sender_stop_margin_seconds
        < args.vnf_drain_timeout_ms / 1000.0
    ):
        raise ValueError(
            "fast VNF unregister requires agent ACK mode and a sender stop margin "
            "at least as long as the VNF drain timeout"
        )
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    profile = load_json(profile_path)
    requests = read_jsonl(args.requests)
    if args.skip_requests:
        requests = requests[args.skip_requests :]
    if args.max_requests:
        requests = requests[: args.max_requests]
    if not requests:
        raise ValueError("request trace is empty")

    external_plans = load_external_plans(args.tree_plans)
    sfc_plans = load_sfc_plans(args.sfc_plans)
    sfc_candidate_plans = load_sfc_candidate_plans(
        args.sfc_candidate_plans, args.sfc_candidate_top_k
    )
    if args.sfc_candidate_plans:
        missing_candidates = sorted(
            int(request["id"])
            for request in requests
            if int(request["id"]) not in sfc_candidate_plans
        )
        if missing_candidates:
            raise ValueError(
                "no SFC candidates for selected requests: "
                f"{missing_candidates[:10]}"
            )
    reroute_rows = load_reroute_events(args.reroute_events)
    migration_rows = load_vnf_control_events(
        args.vnf_migration_events, "migration"
    )
    impairment_rows = load_vnf_control_events(
        args.vnf_impairment_events, "impairment"
    )
    if args.online_wqmix_auto_migration and not migration_rows:
        migration_rows = [
            {
                "time": float(row["time"])
                + args.online_wqmix_migration_delay_seconds,
                "request_id": int(row["request_id"]),
                "stage": int(row["stage"]),
                "target_dc": 0,
                "policy": "online_wqmix_multi_agent",
                "trigger_time": float(row["time"]),
            }
            for row in impairment_rows
        ]
    online_planner = None
    migration_online_planner = None
    predictive_migration_monitor = None
    online_plan_records = []
    sla_predictor_contract_validation = None
    from core.marl.orchestration import RuntimeReconfigurationModule

    runtime_reconfiguration = RuntimeReconfigurationModule()
    if online_hrl_enabled:
        planner_kwargs = dict(
            legacy_root=args.online_hrl_legacy_root,
            checkpoint=args.online_hrl_checkpoint,
            data_path=args.online_hrl_data,
            runtime_requests=args.requests,
            profile=profile_path,
            seed=args.online_hrl_seed,
            max_steps=args.online_hrl_max_steps,
            failure_step_budget=(
                args.online_hrl_failure_step_budget
                if args.online_hrl_failure_step_budget > 0
                else (240 if args.online_hrl_fast_mode else None)
            ),
            async_prefetch_workers=args.online_hrl_async_prefetch_workers,
            fast_k_path_candidates=args.online_hrl_fast_k_path_candidates,
            completion_candidate_budget=(
                args.online_hrl_completion_candidate_budget
                if args.online_hrl_completion_candidate_budget > 0
                else (4 if args.online_hrl_fast_mode else 0)
            ),
            destination_beam_width=(
                args.online_hrl_destination_beam_width
                if args.online_hrl_destination_beam_width > 0
                else (8 if args.online_hrl_fast_mode else 64)
            ),
            optimize_static_topology=not args.online_hrl_no_static_topology_cache,
            bw_cap=args.online_hrl_bw_cap,
            cpu_cap=args.online_hrl_cpu_cap,
            mem_cap=args.online_hrl_mem_cap,
            safe_dest_recovery=args.online_hrl_safe_dest_recovery,
            planner_destinations=args.online_hrl_planner_destinations,
            k_path_candidate_filter=(
                args.online_hrl_k_path_candidate_filter
                or args.online_hrl_fast_mode
            ),
            k_path_candidate_k=args.online_hrl_k_path_candidate_k,
            macro_path_rollout=(
                args.online_hrl_macro_path_rollout
                or args.online_hrl_fast_mode
            ),
            reseed_per_request=args.online_hrl_reseed_per_request,
            torch_threads=args.online_hrl_torch_threads,
            quiet=not args.online_hrl_verbose,
            collect_timing=args.online_hrl_verbose,
        )
        if args.online_hrl_batched_policy:
            from core.marl.orchestration import BrainManagedDeploymentPlanner
            from sdn.online_hrl_planner import OnlineLegacyHRLPlanner
            from sdn.online_batched_hrl_planner import OnlineBatchedHRLPlanner

            loaded_hrl = OnlineLegacyHRLPlanner(**planner_kwargs)
            batched_planner = OnlineBatchedHRLPlanner(
                loaded_hrl,
                profile,
                microbatch_ms=args.online_hrl_batched_microbatch_ms,
                max_agents=args.online_hrl_batched_max_agents,
                top_k=args.online_hrl_batched_top_k,
                worker_count=args.online_hrl_batched_workers,
                cpu_capacity=args.online_hrl_cpu_cap,
                memory_capacity=args.online_hrl_mem_cap,
            )
            online_planner = BrainManagedDeploymentPlanner(batched_planner)
        elif args.online_hrl_process_isolation or args.online_hrl_fast_mode:
            from sdn.online_hrl_process import IsolatedOnlineLegacyHRLPlanner

            online_planner = IsolatedOnlineLegacyHRLPlanner(
                reserve_cpu_affinity=args.online_hrl_fast_mode,
                **planner_kwargs,
            )
        else:
            from sdn.online_hrl_planner import OnlineLegacyHRLPlanner

            online_planner = OnlineLegacyHRLPlanner(**planner_kwargs)
    elif online_wqmix_enabled:
        from sdn.online_wqmix_planner import OnlineWQMIXPlanner

        online_planner = OnlineWQMIXPlanner(
            checkpoint=args.online_wqmix_checkpoint,
            data_folder=args.online_wqmix_data,
            profile=profile_path,
            microbatch_ms=args.online_wqmix_microbatch_ms,
            bandwidth_utilization_limit=(
                args.online_wqmix_bandwidth_utilization_limit
            ),
            q0_sla_safety_margin=args.online_wqmix_q0_sla_safety_margin,
            q0_hard_sla_gate=args.online_wqmix_q0_hard_sla_gate,
            sla_queue_safety_factor=args.online_wqmix_sla_queue_safety_factor,
            sla_calibration=args.online_wqmix_sla_calibration,
            sla_calibration_rank_weight=(
                args.online_wqmix_sla_calibration_rank_weight
            ),
            sla_min_rerank_probability_delta=(
                args.online_wqmix_sla_min_rerank_delta
            ),
            sla_max_ood_score=args.online_wqmix_sla_max_ood_score,
            max_sla_failure_probability=(
                args.online_wqmix_max_sla_failure_probability
            ),
            hybrid_least_loaded=args.online_wqmix_hybrid_least_loaded,
            hybrid_dataset_selected=(
                args.online_wqmix_hybrid_dataset_selected
            ),
            hybrid_rl_weight=args.online_wqmix_hybrid_rl_weight,
            decoder_top_r=args.online_wqmix_decoder_top_r,
            decoder_time_budget_ms=(
                args.online_wqmix_decoder_time_budget_ms
            ),
            repair_missing_plans=args.online_wqmix_repair_missing_plans,
            device=args.online_wqmix_device,
            torch_threads=args.online_wqmix_torch_threads,
        )
        sla_metadata = online_planner.metadata().get("sla_calibration") or {}
        sla_predictor_contract_validation = validate_sla_predictor_runtime_contract(
            sla_metadata.get("measurement_contract"), args
        )
    if args.online_migration_wqmix_checkpoint:
        from sdn.online_migration_wqmix_planner import OnlineMigrationWQMIXPlanner

        migration_online_planner = OnlineMigrationWQMIXPlanner(
            checkpoint=args.online_migration_wqmix_checkpoint,
            profile=profile_path,
            cpu_capacity=args.online_migration_cpu_capacity,
            memory_capacity=args.online_migration_memory_capacity,
            device=args.online_wqmix_device,
            torch_threads=args.online_wqmix_torch_threads,
            decoder_top_r=args.online_wqmix_decoder_top_r,
            decoder_time_budget_ms=args.online_wqmix_decoder_time_budget_ms,
        )
        if args.predictive_migration_scan_seconds > 0.0:
            from sdn.migration_monitor import OnlineMigrationMonitor

            predictive_migration_monitor = OnlineMigrationMonitor(
                args.online_migration_cpu_capacity,
                args.online_migration_memory_capacity,
                max_agents=min(
                    args.online_migration_max_agents,
                    migration_online_planner.max_agents,
                ),
                overload_threshold=args.predictive_migration_overload_threshold,
                safe_utilization=args.predictive_migration_safe_utilization,
                prediction_horizon_s=args.predictive_migration_horizon_seconds,
                cooldown_s=args.predictive_migration_cooldown_seconds,
                minimum_remaining_lifetime_s=(
                    args.predictive_migration_min_remaining_lifetime_seconds
                ),
            )
    plans = {}
    for request in requests:
        request_id = int(request["id"])
        if online_planner is not None:
            plans[request_id] = {
                "switch_outputs": {},
                "paths": {},
                "source": "online_planner_pending",
            }
        elif args.sfc_plans:
            if request_id not in sfc_plans:
                raise ValueError(f"no SFC plan for selected request {request_id}")
            sfc_plan = sfc_plans[request_id]
            plans[request_id] = index_sfc_execution_plan(
                request, sfc_plan, "legacy_hrl_sfc"
            )
        elif sfc_candidate_plans:
            candidates = sfc_candidate_plans[request_id]
            if candidates:
                sfc_plan = candidates[0]
                sfc_plans[request_id] = copy.deepcopy(sfc_plan)
                plans[request_id] = index_sfc_execution_plan(
                    request, sfc_plans[request_id], "offline_sfc_candidate"
                )
            else:
                plans[request_id] = {
                    "switch_outputs": {},
                    "paths": {},
                    "source": "offline_sfc_candidate_pending_rejection",
                }
        elif request_id in external_plans:
            plans[request_id] = external_plans[request_id]
        else:
            outputs, paths = shortest_tree_outputs(
                profile,
                int(request["source_dpid"]),
                [int(value) for value in request["destination_dpids"]],
            )
            plans[request_id] = {
                "switch_outputs": outputs,
                "paths": paths,
                "source": "shortest_path_baseline",
            }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        selected_ids = {int(request["id"]) for request in requests}
        selected_reroutes = [
            row for row in reroute_rows if int(row["request_id"]) in selected_ids
        ]
        result = {
            "valid": True,
            "dry_run": True,
            "requests": len(requests),
            "events": len(requests) * 2 + len(selected_reroutes),
            "reroute_events": len(selected_reroutes),
            "first_request": requests[0],
            "first_plan": plans[int(requests[0]["id"])],
            "first_sfc_plan": sfc_plans.get(int(requests[0]["id"])),
            "probe_agent_worker_plan": probe_agent_worker_plan(
                requests,
                profile,
                args.probe_agent_sender_workers,
                args.probe_agent_receiver_workers,
            ),
        }
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    wsl = ["wsl", "-d", args.distro, "--"]
    _MININET_RUNTIME_HOST = (
        resolve_wsl_ipv4(wsl)
        if args.mininet_command_host.strip().lower() == "auto"
        else args.mininet_command_host.strip()
    )
    if not _MININET_RUNTIME_HOST:
        raise ValueError("Mininet command host cannot be empty")
    controller_rest_url = resolve_controller_rest_url(
        args.rest_url, _MININET_RUNTIME_HOST
    )
    probe_dir = output_path.parent / f"{output_path.stem}_probes"
    shutil.rmtree(probe_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_dir_wsl = to_wsl_path(probe_dir)
    repo_wsl = to_wsl_path(ROOT)
    profile_wsl = to_wsl_path(profile_path)
    probe_wsl = f"{repo_wsl}/sdn/udp_sla_probe.py"
    native_sender_wsl = None
    if args.probe_sender_backend == "native":
        native_sender_wsl = (
            f"/tmp/sft-udp-sla-sender-{int(time.time() * 1000)}"
        )
        subprocess.run(
            wsl
            + [
                "gcc",
                "-O2",
                "-std=c11",
                f"{repo_wsl}/sdn/udp_sla_sender.c",
                "-lm",
                "-o",
                native_sender_wsl,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    native_receiver_wsl = None
    if args.probe_receiver_backend == "native":
        native_receiver_wsl = (
            f"/tmp/sft-udp-sla-receiver-{int(time.time() * 1000)}"
        )
        subprocess.run(
            wsl
            + [
                "gcc",
                "-O2",
                "-std=c11",
                f"{repo_wsl}/sdn/udp_sla_receiver.c",
                "-o",
                native_receiver_wsl,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    forwarder_wsl = f"{repo_wsl}/sdn/vnf_forwarder.py"
    vnf_agent_wsl = f"{repo_wsl}/sdn/vnf_agent.py"
    native_vnf_agent_wsl = None
    if args.vnf_agent_backend == "native":
        native_vnf_agent_wsl = (
            f"{repo_wsl}/sdn/.vnf-agent-native-{os.getpid()}"
        )
        subprocess.run(
            wsl
            + [
                "gcc",
                "-O3",
                "-std=c11",
                f"{repo_wsl}/sdn/vnf_agent_native.c",
                "-o",
                native_vnf_agent_wsl,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    for service in ("ryu-controller.service", "ryu-sft-controller.service", "ryu-sft-static-controller.service"):
        subprocess.run(
            wsl + ["sudo", "-n", "systemctl", "stop", service],
            check=False,
            stderr=subprocess.PIPE,
        )
    # Mininet cleanup kills controller processes, so it must run before the
    # controller used by this replay is started.
    subprocess.run(
        wsl + ["sudo", "-n", "mn", "-c"],
        check=False,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        wsl + ["sudo", "-n", "systemctl", "start", args.controller_service],
        check=True,
        stderr=subprocess.PIPE,
    )

    client = RyuSFTClient(controller_rest_url, timeout=max(5.0, args.timeout))
    wait_for_rest(client, args.timeout)
    controller_config = {
        "topology_file": profile_wsl,
        "stats_interval": 1.0,
        "sfc_barrier_mode": args.sfc_barrier_mode,
    }
    if args.strict_reroute_gates:
        controller_config["threshold"] = args.reroute_live_utilization_threshold
    client.configure(**controller_config)
    mininet_log_path = output_path.with_suffix(".mininet.log")
    mininet_log = mininet_log_path.open("wb")
    selected_host_dpids = (
        runtime_host_dpids(
            profile,
            requests,
            online_wqmix_data=(
                args.online_wqmix_data if online_wqmix_enabled else None
            ),
            sfc_plans=sfc_plans,
            sfc_candidate_plans=sfc_candidate_plans,
        )
        if args.mininet_active_hosts_only
        else {int(node["dpid"]) for node in profile["nodes"]}
    )
    # Live HRL plans are produced only after an arrival and therefore cannot
    # be inspected when the Mininet host set is created.  Keep every DC host
    # available in this mode so a valid placement selected at runtime always
    # has a VNF agent FIFO.  The source/destination hosts are still added by
    # ``runtime_host_dpids`` above.
    if args.mininet_active_hosts_only and online_hrl_enabled:
        selected_host_dpids.update(
            int(dpid) for dpid in profile.get("dc_nodes_1based", [])
        )
    selected_host_dpids.update(
        int(row["target_dc"])
        for row in migration_rows
        if int(row["request_id"]) in {int(request["id"]) for request in requests}
    )
    mininet_command = ["sudo", "-n"]
    if args.mininet_cpu_set:
        mininet_command.extend(["taskset", "-c", args.mininet_cpu_set])
    mininet_command.extend(
        [
            "python3",
            f"{repo_wsl}/sdn/run_profile_mininet.py",
            "--profile",
            profile_wsl,
            "--controller-ip",
            "127.0.0.1",
            "--controller-port",
            "6653",
            "--switch-start-delay",
            str(args.switch_start_delay),
            "--link-mode",
            "tc",
            "--qdisc",
            args.mininet_qdisc,
            "--max-queue-size",
            str(args.mininet_max_queue_size),
            "--command-port",
            str(args.mininet_command_port),
        ]
    )
    if args.mininet_active_hosts_only:
        mininet_command.extend(
            ["--host-dpids", ",".join(map(str, sorted(selected_host_dpids)))]
        )
    mininet = subprocess.Popen(
        wsl + mininet_command,
        stdin=subprocess.DEVNULL,
        stdout=mininet_log,
        stderr=subprocess.STDOUT,
    )

    installed = set()
    execution_rejected = set()
    event_rows = []
    setup_estimator = RollingSetupEstimator(
        args.deployment_admission_estimate_ms,
        args.deployment_admission_window,
        args.deployment_admission_min_samples,
        args.deployment_admission_safety_factor,
    )
    total_capacity = StageCapacityGate(
        "total deployment", args.deployment_max_outstanding
    )
    planning_capacity = StageCapacityGate(
        "planning", args.planning_max_outstanding
    )
    execution_capacity = StageCapacityGate(
        "execution", args.execution_max_outstanding
    )
    vnf_endpoint_pool = (
        VnfEndpointPool(
            args.vnf_agent_prebound_port_base,
            args.vnf_agent_prebound_port_count,
        )
        if args.vnf_agent_prebound_port_count > 0
        else None
    )
    probe_paths = []
    sender_paths = []
    precomputed_receiver_results = []
    precomputed_sender_results = []
    vnf_stats_paths = []
    active_vnfs = set()
    run_token = f"{int(time.time())}-{requests[0]['id']}-{len(requests)}"
    handshake_dir_wsl = f"/tmp/sft-runtime-{run_token}"
    agent_fifos: dict[str, str] = {}
    sender_agent_fifos: dict[str, str] = {}
    vnf_agent_fifos: dict[str, list[str]] = {}
    vnf_agent_pid_files: dict[str, list[str]] = {}
    nodes = {int(node["dpid"]): node for node in profile["nodes"]}
    runtime_nodes = {
        dpid: node for dpid, node in nodes.items() if dpid in selected_host_dpids
    }
    probe_host_dpids = {
        int(value)
        for request in requests
        for value in [request["source_dpid"], *request["destination_dpids"]]
    }
    preschedule_enabled = (
        args.probe_launch_mode == "agent" and args.probe_preschedule
    )
    source_hosts = {
        str(nodes[int(request["source_dpid"])]["host"])
        for request in requests
    }
    agent_worker_plan = probe_agent_worker_plan(
        requests,
        profile,
        args.probe_agent_sender_workers,
        args.probe_agent_receiver_workers,
    )
    agent_sender_workers = int(agent_worker_plan["sender_workers"])
    agent_receiver_workers = int(agent_worker_plan["receiver_workers"])
    diagnostics_before: dict[str, Any] = {}
    diagnostics_after: dict[str, Any] = {}
    clock_sync: dict[str, Any] = {}
    dynamic_clock_sync_observations: list[dict[str, Any]] = []
    dynamic_clock_sync_lock = threading.Lock()
    controller_scheduler: dict[str, Any] = {
        "service": args.controller_service,
        "requested_priority": int(args.controller_realtime_priority),
        "applied": False,
        "main_pid": None,
        "scheduler": None,
    }
    controller_affinity: dict[str, Any] = {
        "service": args.controller_service,
        "requested_cpu_set": args.controller_cpu_set,
        "applied": False,
    }
    ovs_affinity: dict[str, Any] = {
        "service": args.ovs_service,
        "requested_cpu_set": args.ovs_cpu_set,
        "applied": False,
    }
    ryu_batch_committer: RyuBatchCommitter | None = None
    vnf_register_batcher: VnfControlBatcher | None = None
    vnf_unregister_batcher: VnfControlBatcher | None = None
    neighbor_executor = (
        ThreadPoolExecutor(
            max_workers=min(4, int(args.deployment_workers)),
            thread_name_prefix="neighbor-stage",
        )
        if args.parallel_deployment_pipeline
        else None
    )
    try:
        controller_affinity = set_service_cpu_affinity(
            wsl, args.controller_service, args.controller_cpu_set
        )
        ovs_affinity = set_service_cpu_affinity(
            wsl, args.ovs_service, args.ovs_cpu_set
        )
        controller_scheduler = set_service_realtime_scheduler(
            wsl,
            args.controller_service,
            args.controller_realtime_priority,
        )
        wait_for_controller(client, len(nodes), args.timeout)
        runtime_status = wait_for_mininet_runtime(args.mininet_command_port, args.timeout)
        if len(runtime_status.get("hosts", [])) != len(runtime_nodes):
            raise RuntimeError("Mininet runtime host count does not match the topology")
        subprocess.run(
            wsl + ["mkdir", "-p", handshake_dir_wsl],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        clock_sync = mininet_clock_offset(
            args.mininet_command_port, args.timeout
        )
        diagnostics_before = mininet_runtime_diagnostics(
            args.mininet_command_port, args.timeout
        )
        if args.ryu_commit_batch_ms > 0.0:
            ryu_batch_committer = RyuBatchCommitter(
                client,
                args.ryu_commit_batch_ms,
                args.ryu_commit_batch_size,
            )
        route_commands = []
        for node in runtime_nodes.values():
            host = str(node["host"])
            interface = f"{host}-eth0"
            neighbor_commands = []
            for peer in runtime_nodes.values():
                if int(peer["dpid"]) == int(node["dpid"]):
                    continue
                peer_ip = str(peer["host_ip"]).split("/", 1)[0]
                peer_mac = deterministic_host_mac(int(peer["dpid"]))
                neighbor_commands.append(
                    f"ip neigh replace {shlex.quote(peer_ip)} lladdr "
                    f"{shlex.quote(peer_mac)} nud permanent dev {shlex.quote(interface)}"
                )
            route_commands.append(
                (
                    host,
                    f"ip route replace 239.0.0.0/8 dev {shlex.quote(interface)}; "
                    + "; ".join(neighbor_commands),
                )
            )
        mininet_runtime_commands(
            args.mininet_command_port, route_commands, args.timeout
        )
        if args.probe_launch_mode == "agent":
            agent_commands = []
            agent_ready_paths = []
            for node in runtime_nodes.values():
                if int(node["dpid"]) not in probe_host_dpids:
                    continue
                host = str(node["host"])
                receiver_role = "-receiver" if preschedule_enabled else ""
                agent_fifo = (
                    f"{handshake_dir_wsl}/agent-{host}{receiver_role}.fifo"
                )
                agent_ready_name = f"agent-{host}{receiver_role}.ready"
                agent_ready_wsl = f"{probe_dir_wsl}/{agent_ready_name}"
                agent_log_wsl = f"{probe_dir_wsl}/agent-{host}{receiver_role}.log"
                agent_fifos[host] = agent_fifo
                agent_ready_paths.append(probe_dir / agent_ready_name)
                agent_commands.append(
                    (
                        host,
                        probe_agent_start_command(
                            probe_wsl,
                            agent_fifo,
                            agent_ready_wsl,
                            agent_log_wsl,
                            agent_sender_workers,
                            agent_receiver_workers,
                            args.probe_agent_cpu_set,
                            args.probe_agent_realtime_priority,
                        ),
                    )
                )
                if preschedule_enabled and host in source_hosts:
                    sender_fifo = (
                        f"{handshake_dir_wsl}/agent-{host}-sender.fifo"
                    )
                    sender_ready_name = f"agent-{host}-sender.ready"
                    sender_ready_wsl = f"{probe_dir_wsl}/{sender_ready_name}"
                    sender_log_wsl = f"{probe_dir_wsl}/agent-{host}-sender.log"
                    sender_agent_fifos[host] = sender_fifo
                    agent_ready_paths.append(probe_dir / sender_ready_name)
                    agent_commands.append(
                        (
                            host,
                            probe_agent_start_command(
                                probe_wsl,
                                sender_fifo,
                                sender_ready_wsl,
                                sender_log_wsl,
                                agent_sender_workers,
                                agent_receiver_workers,
                                args.probe_agent_cpu_set,
                                args.probe_agent_realtime_priority,
                            ),
                        )
                    )
            mininet_runtime_commands(
                args.mininet_command_port, agent_commands, args.timeout
            )
            ready_agents = wait_for_probe_files(agent_ready_paths, args.timeout)
            if ready_agents != len(agent_ready_paths):
                raise RuntimeError(
                    f"only {ready_agents}/{len(agent_ready_paths)} probe agents became ready"
                )

        if (sfc_plans or online_planner is not None) and args.vnf_launch_mode == "agent":
            vnf_agent_commands = []
            vnf_agent_ready_paths = []
            native_shards = (
                args.vnf_agent_native_shards
                if args.vnf_agent_backend == "native"
                else 1
            )
            for raw_dpid in profile.get("dc_nodes_1based", []):
                dpid = int(raw_dpid)
                if dpid not in runtime_nodes:
                    continue
                host = str(nodes[dpid]["host"])
                host_fifos = []
                host_pid_files = []
                for shard in range(native_shards):
                    suffix = f"-w{shard + 1}" if native_shards > 1 else ""
                    fifo = f"{handshake_dir_wsl}/vnf-agent-{host}{suffix}.fifo"
                    ready_name = f"vnf-agent-{host}{suffix}.ready"
                    ready_wsl = f"{probe_dir_wsl}/{ready_name}"
                    log_wsl = f"{probe_dir_wsl}/vnf-agent-{host}{suffix}.log"
                    pid_wsl = f"{handshake_dir_wsl}/vnf-agent-{host}{suffix}.pid"
                    host_fifos.append(fifo)
                    host_pid_files.append(pid_wsl)
                    vnf_agent_ready_paths.append(probe_dir / ready_name)
                    vnf_agent_commands.append(
                        (
                            host,
                            vnf_agent_start_command(
                                vnf_agent_wsl
                                if args.vnf_agent_backend == "python"
                                else str(native_vnf_agent_wsl),
                                fifo,
                                ready_wsl,
                                log_wsl,
                                pid_wsl,
                                args.vnf_agent_workers,
                                args.vnf_agent_initial_workers,
                                args.vnf_agent_bindings_per_worker,
                                args.vnf_agent_prewarm_workers,
                                args.vnf_drain_timeout_ms,
                                args.vnf_drain_idle_ms,
                                args.vnf_agent_realtime_priority,
                                args.vnf_agent_cpu_set,
                                args.vnf_agent_packet_batch,
                                args.vnf_agent_q0_packet_batch,
                                args.vnf_agent_backend,
                                args.vnf_agent_prebound_port_base,
                                args.vnf_agent_prebound_port_count,
                                dscp_scheduling=args.vnf_agent_dscp_scheduling,
                            ),
                        )
                    )
                vnf_agent_fifos[host] = host_fifos
                vnf_agent_pid_files[host] = host_pid_files
            mininet_runtime_commands(
                args.mininet_command_port, vnf_agent_commands, args.timeout
            )
            ready_agents = wait_for_probe_files(vnf_agent_ready_paths, args.timeout)
            if ready_agents != len(vnf_agent_ready_paths):
                raise RuntimeError(
                    f"only {ready_agents}/{len(vnf_agent_ready_paths)} VNF agents became ready"
                )
            if args.vnf_register_batch_ms > 0.0:
                vnf_register_batcher = VnfControlBatcher(
                    args.mininet_command_port,
                    args.timeout,
                    args.vnf_register_batch_ms,
                    args.vnf_register_batch_size,
                    "register",
                )
            if args.vnf_unregister_batch_ms > 0.0:
                vnf_unregister_batcher = VnfControlBatcher(
                    args.mininet_command_port,
                    args.timeout,
                    args.vnf_unregister_batch_ms,
                    args.vnf_unregister_batch_size,
                    "unregister",
                )

        events = request_events(requests)
        online_arrival_events = [
            event for event in events if event.get("type") == "arrive"
        ]
        online_arrival_index = {
            int(event["request"]["id"]): index
            for index, event in enumerate(online_arrival_events)
        }
        request_by_id = {int(request["id"]): request for request in requests}
        selected_reroutes = []
        for row in reroute_rows:
            request = request_by_id.get(int(row["request_id"]))
            if request is None:
                continue
            if not (
                float(request["arrival_time"])
                <= float(row["time"])
                < float(request["leave_time"])
            ):
                raise ValueError(
                    f"reroute for request {row['request_id']} at {row['time']} "
                    "is outside its active lifetime"
                )
            selected_reroutes.append(row)
            events.append(
                {
                    "time": float(row["time"]),
                    "type": "reroute",
                    "request": request,
                    "reroute": row,
                }
            )
        selected_migrations = []
        selected_impairments = []
        for rows, event_type, payload_key, selected in (
            (impairment_rows, "impairment", "impairment", selected_impairments),
            (migration_rows, "migration", "migration", selected_migrations),
        ):
            for row in rows:
                request = request_by_id.get(int(row["request_id"]))
                if request is None:
                    continue
                if not (
                    float(request["arrival_time"])
                    <= float(row["time"])
                    < float(request["leave_time"])
                ):
                    raise ValueError(
                        f"{event_type} for request {row['request_id']} at "
                        f"{row['time']} is outside its active lifetime"
                    )
                selected.append(row)
                events.append(
                    {
                        "time": float(row["time"]),
                        "type": event_type,
                        "request": request,
                        payload_key: row,
                    }
                )
        origin = float(events[0]["time"])
        selected_predictive_scans = 0
        if predictive_migration_monitor is not None and requests:
            scan_time = float(origin) + args.predictive_migration_scan_seconds
            final_time = max(float(request["leave_time"]) for request in requests)
            while scan_time < final_time - 1e-12:
                events.append({
                    "time": scan_time,
                    "type": "migration_scan",
                    "request": requests[0],
                })
                selected_predictive_scans += 1
                scan_time += args.predictive_migration_scan_seconds
        event_priority = {
            "leave": 0, "arrive": 1, "impairment": 2,
            "migration_scan": 3, "migration": 4, "reroute": 5,
        }
        events.sort(key=lambda item: (item["time"], event_priority[item["type"]]))
        applied_reroutes = 0
        failed_reroutes = 0
        skipped_reroutes = 0
        reroute_gate_rejections: Counter[str] = Counter()
        reroute_chain_valid: dict[int, bool] = defaultdict(lambda: True)
        last_reroute_time: dict[int, float] = {}
        applied_migrations = 0
        failed_migrations = 0
        skipped_migrations = 0
        migration_decision_cache: dict[tuple[int, int, float], dict[str, Any]] = {}
        applied_impairments = 0
        skipped_impairments = 0
        migration_records: list[dict[str, Any]] = []
        prescheduled_probes = preschedule_enabled
        schedule_origin_wsl = f"{handshake_dir_wsl}/schedule.origin"
        if prescheduled_probes:
            preload_commands = []
            for request in sorted(
                requests, key=lambda row: (float(row["arrival_time"]), int(row["id"]))
            ):
                request_id = int(request["id"])
                scaled_lifetime = (
                    float(request["leave_time"])
                    - float(request["arrival_time"])
                ) * args.time_scale
                traffic_duration = (
                    scaled_lifetime - args.sender_stop_margin_seconds
                )
                if traffic_duration < args.minimum_traffic_seconds:
                    reason = (
                        f"request {request_id} has only {scaled_lifetime:.3f}s "
                        "of scaled lifetime; no traffic window remains"
                    )
                    precomputed_sender_results.append(
                        {
                            "request_id": request_id,
                            "result": startup_failure_sender(request, reason),
                        }
                    )
                    for destination in request["destination_dpids"]:
                        precomputed_receiver_results.append(
                            {
                                "request_id": request_id,
                                "destination_dpid": int(destination),
                                "result": startup_failure_receiver(
                                    request, int(destination), reason
                                ),
                            }
                        )
                    continue
                start_offset_ns = int(
                    round(
                        (float(request["arrival_time"]) - origin)
                        * args.time_scale
                        * 1e9
                    )
                )
                sender_stop_offset_ns = int(
                    round(
                        (
                            (float(request["leave_time"]) - origin)
                            * args.time_scale
                            - args.sender_stop_margin_seconds
                        )
                        * 1e9
                    )
                )
                receiver_stop_offset_ns = int(
                    round(
                        (float(request["leave_time"]) - origin)
                        * args.time_scale
                        * 1e9
                    )
                )
                pps = args.packets_per_second
                if pps <= 0.0:
                    pps = bandwidth_mbps_to_pps(
                        float(request["bw_origin"]),
                        args.payload_bytes,
                        args.packet_overhead_bytes,
                    )
                expected_file = f"{handshake_dir_wsl}/r{request_id}.expected"
                ready_files = []
                for destination in request["destination_dpids"]:
                    destination = int(destination)
                    node = nodes[destination]
                    probe_name = (
                        f"sft-runtime-{run_token}-r{request_id}-h{destination}.json"
                    )
                    probe_output = f"{probe_dir_wsl}/{probe_name}"
                    ready_output = (
                        f"{handshake_dir_wsl}/r{request_id}-h{destination}.ready"
                    )
                    ready_files.append(ready_output)
                    probe_paths.append(
                        (request_id, destination, probe_dir / probe_name)
                    )
                    preload_commands.append(
                        (
                            str(node["host"]),
                            probe_receiver_command(
                                probe_wsl,
                                request,
                                str(node["host_ip"]).split("/", 1)[0],
                                traffic_duration,
                                0,
                                ready_output,
                                None,
                                expected_file,
                                args.probe_receive_buffer_bytes,
                                probe_output,
                                agent_fifos[str(node["host"])],
                                schedule_origin_wsl,
                                start_offset_ns,
                                receiver_stop_offset_ns,
                                receiver_backend=args.probe_receiver_backend,
                                native_receiver_path=native_receiver_wsl,
                            ),
                        )
                    )
                sender_name = (
                    f"sft-runtime-{run_token}-r{request_id}-sender.json"
                )
                sender_output = f"{probe_dir_wsl}/{sender_name}"
                sender_paths.append((request_id, probe_dir / sender_name))
                source_host = str(nodes[int(request["source_dpid"])]["host"])
                preload_commands.append(
                    (
                        source_host,
                        probe_sender_command(
                            probe_wsl,
                            request,
                            traffic_duration,
                            args.payload_bytes,
                            pps,
                            ready_files,
                            min(
                                args.receiver_ready_seconds,
                                max(
                                    0.0,
                                    traffic_duration
                                    - args.minimum_traffic_seconds,
                                ),
                            ),
                            None,
                            args.minimum_traffic_seconds,
                            expected_file,
                            sender_output,
                            sender_agent_fifos[source_host],
                            schedule_origin_wsl,
                            start_offset_ns
                            + int(args.probe_tree_warmup_seconds * 1e9),
                            sender_stop_offset_ns,
                            sender_backend=args.probe_sender_backend,
                            native_sender_path=native_sender_wsl,
                            native_realtime_priority=(
                                args.probe_native_realtime_priority
                            ),
                        ),
                    )
                )
            for start in range(0, len(preload_commands), 256):
                mininet_runtime_commands(
                    args.mininet_command_port,
                    preload_commands[start : start + 256],
                    args.timeout,
                )
            lead_seconds = 1.0
            epoch_now_ns = time.time_ns()
            monotonic_now = time.monotonic()
            schedule_origin_ns = epoch_now_ns + int(lead_seconds * 1e9)
            mininet_runtime_commands(
                args.mininet_command_port,
                [
                    (
                        sorted(source_hosts)[0],
                        f"printf '%s' {schedule_origin_ns} > "
                        f"{shlex.quote(schedule_origin_wsl)}",
                    )
                ],
                args.timeout,
            )
            wall_start = monotonic_now + lead_seconds
        else:
            wall_start = time.monotonic()
        request_locks = {
            int(request["id"]): threading.Lock() for request in requests
        }
        link_capacities = topology_bandwidth_caps(profile)
        link_reserved_mbps: dict[tuple[int, int], float] = defaultdict(float)
        request_link_reservations: dict[int, dict[tuple[int, int], float]] = {}
        link_admission_rejected = 0
        bandwidth_lock = threading.Lock()
        vnf_registration_slots = (
            threading.BoundedSemaphore(args.vnf_registration_concurrency)
            if args.vnf_registration_concurrency > 0
            else None
        )
        candidate_selection_fallbacks = 0
        candidate_selection_no_feasible = 0
        candidate_selection_attempts = 0
        candidate_selection_timings_ms: list[float] = []
        scheduled_bandwidth_releases = 0

        def select_sfc_candidate(
            request: dict[str, Any],
        ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            """Select a complete offline candidate against the live link ledger."""
            nonlocal candidate_selection_fallbacks
            nonlocal candidate_selection_no_feasible, candidate_selection_attempts
            request_id = int(request["id"])
            candidates = sfc_candidate_plans.get(request_id, [])
            if not candidates:
                candidate_selection_no_feasible += 1
                return None, {"reason": "missing_sfc_candidates"}
            demand = max(0.0, float(request.get("bw_origin", 0.0)))
            limit = float(args.deployment_bandwidth_utilization_limit)
            reasons: list[dict[str, Any]] = []
            best_plan: dict[str, Any] | None = None
            best_detail: dict[str, Any] | None = None
            best_score: tuple[float, ...] | None = None
            for rank, candidate in enumerate(candidates):
                candidate_selection_attempts += 1
                plan = candidate
                if args.sfc_candidate_selection == "first" or (
                    args.sfc_candidate_selection == "greedy-feasible" and limit <= 0.0
                ):
                    selected_rank = rank
                    if selected_rank > 0:
                        candidate_selection_fallbacks += 1
                    selected_plan = copy.deepcopy(plan)
                    selected_plan.setdefault("topk_selection", {}).update(
                        {
                            "selection_policy": args.sfc_candidate_selection,
                            "selected_rank": selected_rank,
                            "candidate_count": len(candidates),
                            "fallback": bool(selected_rank > 0),
                        }
                    )
                    return selected_plan, {
                        "selected_rank": selected_rank,
                        "candidate_count": len(candidates),
                        "fallback": bool(selected_rank > 0),
                    }
                edges = plan_physical_edges(plan)
                overloaded = []
                missing = []
                projected_utilizations = []
                for edge, multiplicity in sorted(edges.items()):
                    capacity = link_capacities.get(edge)
                    if capacity is None:
                        missing.append(list(edge))
                        continue
                    requested = demand * int(multiplicity)
                    reserved = float(link_reserved_mbps[edge])
                    projected_utilizations.append(
                        (reserved + requested) / max(float(capacity), 1e-9)
                    )
                    if limit > 0.0 and reserved + requested > capacity * limit + 1e-9:
                        overloaded.append(
                            {
                                "edge": list(edge),
                                "reserved_mbps": reserved,
                                "requested_mbps": requested,
                                "limit_mbps": capacity * limit,
                            }
                        )
                if missing or overloaded:
                    reasons.append(
                        {
                            "rank": rank,
                            "candidate_index": plan.get("topk_selection", {}).get(
                                "candidate_index", rank
                            ),
                            "missing_edges": missing[:8],
                            "overloaded_links": overloaded[:8],
                        }
                    )
                    continue
                detail = {
                    "selected_rank": rank,
                    "candidate_count": len(candidates),
                    "fallback": bool(rank > 0),
                    "rejected_candidates": reasons,
                }
                if args.sfc_candidate_selection not in ("least-loaded", "sla-aware"):
                    if rank > 0:
                        candidate_selection_fallbacks += 1
                    selected_plan = copy.deepcopy(plan)
                    selected_plan.setdefault("topk_selection", {}).update(
                        {
                            "selection_policy": args.sfc_candidate_selection,
                            "selected_rank": rank,
                            "candidate_count": len(candidates),
                            "fallback": bool(rank > 0),
                        }
                    )
                    return selected_plan, detail
                if args.sfc_candidate_selection == "sla-aware":
                    score = sfc_candidate_sla_score(
                        plan, projected_utilizations, rank
                    )
                else:
                    score = (
                        max(projected_utilizations, default=0.0),
                        sum(projected_utilizations),
                        rank,
                    )
                if best_score is None or score < best_score:
                    best_score = score
                    best_plan = plan
                    best_detail = detail
            if best_plan is not None and best_detail is not None:
                selected_rank = int(best_detail["selected_rank"])
                if selected_rank > 0:
                    candidate_selection_fallbacks += 1
                selected_plan = copy.deepcopy(best_plan)
                selected_plan.setdefault("topk_selection", {}).update(
                    {
                        "selection_policy": args.sfc_candidate_selection,
                        "selected_rank": selected_rank,
                        "candidate_count": len(candidates),
                        "fallback": bool(selected_rank > 0),
                        "selection_score": list(best_score)
                        if best_score is not None
                        else None,
                    }
                )
                return selected_plan, best_detail
            candidate_selection_no_feasible += 1
            return None, {
                "reason": "all_sfc_candidates_infeasible",
                "candidate_count": len(candidates),
                "rejected_candidates": reasons,
            }

        def reserve_request_bandwidth(
            request: dict[str, Any], plan: dict[str, Any]
        ) -> tuple[bool, dict[str, Any]]:
            """Atomically reserve one multicast stream on every link in the plan."""
            nonlocal link_admission_rejected
            request_id = int(request["id"])
            demand = max(0.0, float(request.get("bw_origin", 0.0)))
            if args.deployment_bandwidth_utilization_limit <= 0.0:
                return True, {
                    "enabled": False,
                    "requested_mbps": demand,
                }
            edge_multiplicity = plan_physical_edges(plan)
            missing_edges = sorted(
                edge for edge in edge_multiplicity if edge not in link_capacities
            )
            if missing_edges:
                link_admission_rejected += 1
                return False, {
                    "reason": "plan_edge_missing_from_profile",
                    "missing_edges": [list(edge) for edge in missing_edges],
                }
            limit = float(args.deployment_bandwidth_utilization_limit)
            overloaded = []
            for edge in sorted(edge_multiplicity):
                capacity = link_capacities[edge]
                reserved = link_reserved_mbps[edge]
                requested = demand * edge_multiplicity[edge]
                if reserved + requested > capacity * limit + 1e-9:
                    overloaded.append(
                        {
                            "edge": list(edge),
                            "capacity_mbps": capacity,
                            "reserved_mbps": reserved,
                            "requested_mbps": requested,
                            "traversal_multiplicity": edge_multiplicity[edge],
                            "limit_mbps": capacity * limit,
                        }
                    )
            if overloaded:
                link_admission_rejected += 1
                return False, {
                    "reason": "physical_link_bandwidth_admission",
                    "bandwidth_utilization_limit": limit,
                    "overloaded_links": overloaded[:8],
                    "overloaded_link_count": len(overloaded),
                }
            reservation = {
                edge: demand * multiplicity
                for edge, multiplicity in edge_multiplicity.items()
                if demand > 0.0
            }
            for edge, amount in reservation.items():
                link_reserved_mbps[edge] += amount
            request_link_reservations[request_id] = reservation
            return True, {
                "enabled": True,
                "reserved_mbps": demand,
                "reserved_edge_count": len(reservation),
                "reserved_edges": [list(edge) for edge in sorted(reservation)],
                "bandwidth_utilization_limit": limit,
            }

        def mirror_committed_request_bandwidth(
            request: dict[str, Any], plan: dict[str, Any]
        ) -> tuple[bool, dict[str, Any]]:
            """Mirror an already committed online-ledger allocation.

            Online WQMIX has atomically committed the complete CPU, memory, and
            bandwidth footprint before this function is called.  This mirror is
            retained for runtime diagnostics and migration accounting only; a
            mismatch is an invariant violation, never a second admission result.
            """

            request_id = int(request["id"])
            if request_id in request_link_reservations:
                raise RuntimeError(
                    f"request {request_id} already has a runtime bandwidth mirror"
                )
            demand = max(0.0, float(request.get("bw_origin", 0.0)))
            edge_multiplicity = plan_physical_edges(plan)
            missing_edges = sorted(
                edge for edge in edge_multiplicity if edge not in link_capacities
            )
            if missing_edges:
                raise RuntimeError(
                    "online WQMIX committed edges absent from the runtime profile: "
                    f"request={request_id}, edges={missing_edges[:8]}"
                )
            reservation = {
                edge: demand * multiplicity
                for edge, multiplicity in edge_multiplicity.items()
                if demand > 0.0
            }
            authoritative_limit = float(
                args.online_wqmix_bandwidth_utilization_limit
            )
            inconsistent = []
            for edge, amount in reservation.items():
                projected = link_reserved_mbps[edge] + amount
                limit_mbps = link_capacities[edge] * authoritative_limit
                if projected > limit_mbps + 1e-9:
                    inconsistent.append(
                        {
                            "edge": list(edge),
                            "projected_mbps": projected,
                            "limit_mbps": limit_mbps,
                        }
                    )
            if inconsistent:
                raise RuntimeError(
                    "runtime bandwidth mirror diverged from the authoritative "
                    f"online WQMIX ledger for request {request_id}: "
                    f"{inconsistent[:8]}"
                )
            for edge, amount in reservation.items():
                link_reserved_mbps[edge] += amount
            request_link_reservations[request_id] = reservation
            return True, {
                "enabled": True,
                "mode": "authoritative_ledger_mirror",
                "authoritative_ledger": "online_wqmix_atomic_ledger",
                "reserved_mbps": demand,
                "reserved_edge_count": len(reservation),
                "reserved_edges": [list(edge) for edge in sorted(reservation)],
                "bandwidth_utilization_limit": authoritative_limit,
            }

        def release_request_bandwidth(request_id: int) -> bool:
            with bandwidth_lock:
                reservation = request_link_reservations.pop(int(request_id), None)
                if not reservation:
                    return False
                for edge, amount in reservation.items():
                    link_reserved_mbps[edge] = max(
                        0.0, link_reserved_mbps[edge] - amount
                    )
                return True

        def check_replacement_bandwidth(
            request: dict[str, Any], replacement: dict[str, Any]
        ) -> tuple[bool, dict[tuple[int, int], float], dict[str, Any]]:
            request_id = int(request["id"])
            demand = max(0.0, float(request.get("bw_origin", 0.0)))
            multiplicity = plan_physical_edges(replacement)
            missing = sorted(edge for edge in multiplicity if edge not in link_capacities)
            if missing:
                return False, {}, {"reason": "plan_edge_missing", "edges": missing}
            proposed = {
                edge: demand * count for edge, count in multiplicity.items()
                if demand > 0.0
            }
            old = request_link_reservations.get(request_id, {})
            limit = float(args.deployment_bandwidth_utilization_limit)
            overloaded = []
            if limit > 0.0:
                for edge, amount in proposed.items():
                    used_without_old = link_reserved_mbps[edge] - old.get(edge, 0.0)
                    if used_without_old + amount > link_capacities[edge] * limit + 1e-9:
                        overloaded.append(list(edge))
            return not overloaded, proposed, {
                "overloaded_edges": overloaded,
                "old_edge_count": len(old),
                "new_edge_count": len(proposed),
            }

        def commit_replacement_bandwidth(
            request_id: int, proposed: dict[tuple[int, int], float]
        ) -> None:
            old = request_link_reservations.get(int(request_id), {})
            for edge, amount in old.items():
                link_reserved_mbps[edge] = max(0.0, link_reserved_mbps[edge] - amount)
            for edge, amount in proposed.items():
                link_reserved_mbps[edge] += amount
            request_link_reservations[int(request_id)] = proposed

        def reject_arrival_before_execution(
            event: dict[str, Any],
            *,
            code: str,
            source: str,
            detail: str,
            extra: dict[str, Any] | None = None,
        ) -> None:
            request = event["request"]
            request_id = int(request["id"])
            target = wall_start + (
                float(event["time"]) - origin
            ) * args.time_scale
            execution_rejected.add(request_id)
            if online_planner is not None and hasattr(online_planner, "release"):
                online_planner.release(request_id)
            precomputed_sender_results.append(
                {
                    "request_id": request_id,
                    "result": startup_failure_sender(request, detail),
                }
            )
            for destination in request["destination_dpids"]:
                precomputed_receiver_results.append(
                    {
                        "request_id": request_id,
                        "destination_dpid": int(destination),
                        "result": startup_failure_receiver(
                            request, int(destination), detail
                        ),
                    }
                )
            row = {
                "request_id": request_id,
                "type": "arrive",
                "trace_time": float(event["time"]),
                "wall_elapsed": time.monotonic() - wall_start,
                "scheduler_lag_ms": max(
                    0.0, (time.monotonic() - target) * 1000.0
                ),
                "controller": {
                    "accepted": False,
                    "reason": code,
                    "source": source,
                    "detail": detail,
                },
                "paths": {},
                "probe_status": code,
                "online_planning": (
                    sfc_plans.get(request_id, {}).get("online_planning")
                ),
            }
            if extra:
                row.update(extra)
            event_rows.append(row)

        def deadline_admission(
            request: dict[str, Any], *, phase: str
        ) -> dict[str, Any]:
            now = time.monotonic()
            leave_target = wall_start + (
                float(request["leave_time"]) - origin
            ) * args.time_scale
            remaining_lifetime_ms = (leave_target - now) * 1000.0
            setup_estimate_ms = setup_estimator.estimate_ms()
            probe_reserve_ms = (
                args.sender_stop_margin_seconds + args.minimum_traffic_seconds
            ) * 1000.0
            required_lifetime_ms = setup_estimate_ms + probe_reserve_ms
            return {
                "phase": phase,
                "remaining_lifetime_ms": remaining_lifetime_ms,
                "setup_estimate_p95_ms": setup_estimate_ms,
                "probe_reserve_ms": probe_reserve_ms,
                "required_lifetime_ms": required_lifetime_ms,
                "slack_ms": remaining_lifetime_ms - required_lifetime_ms,
            }

        def prepare_online_plans(batch_events: list[dict[str, Any]]) -> None:
            if online_planner is None or not batch_events:
                return
            requests_for_batch = [event["request"] for event in batch_events]
            planning_started = time.monotonic()
            if hasattr(online_planner, "plan_batch"):
                selected_plans = online_planner.plan_batch(requests_for_batch)
            else:
                selected_plans = [
                    online_planner.plan_next(request) for request in requests_for_batch
                ]
            planning_ready = time.monotonic()
            if len(selected_plans) != len(batch_events):
                raise RuntimeError("online planner returned the wrong batch size")
            source = "online_wqmix_sfc" if online_wqmix_enabled else "online_hrl_sfc"
            for event, sfc_plan in zip(batch_events, selected_plans):
                request = event["request"]
                request_id = int(request["id"])
                target = wall_start + (
                    float(event["time"]) - origin
                ) * args.time_scale
                sfc_plan.setdefault("online_planning", {}).update(
                    {
                        "arrival_to_planning_start_ms": max(
                            0.0, (planning_started - target) * 1000.0
                        ),
                        "arrival_to_plan_ready_ms": max(
                            0.0, (planning_ready - target) * 1000.0
                        ),
                    }
                )
                sfc_plans[request_id] = sfc_plan
                plans[request_id] = index_sfc_execution_plan(
                    request, sfc_plan, source
                )
                online_plan_records.append(sfc_plan)
            if (
                online_hrl_enabled
                and args.online_hrl_async_prefetch_workers > 0
                and hasattr(online_planner, "prefetch")
            ):
                last_index = max(
                    (
                        online_arrival_index.get(int(event["request"]["id"]), -1)
                        for event in batch_events
                    ),
                    default=-1,
                )
                prefetch_requests = [
                    event["request"]
                    for event in online_arrival_events[
                        last_index + 1 : last_index + 1 + args.online_hrl_async_prefetch_workers
                    ]
                ]
                online_planner.prefetch(prefetch_requests)

        def prepare_online_plan(event: dict[str, Any]) -> None:
            if online_planner is None or event["type"] != "arrive":
                return
            prepare_online_plans([event])

        def process_event_unlocked(event: dict[str, Any]) -> None:
            nonlocal applied_reroutes, failed_reroutes, skipped_reroutes
            nonlocal applied_migrations, failed_migrations, skipped_migrations
            nonlocal applied_impairments, skipped_impairments
            target = wall_start + (float(event["time"]) - origin) * args.time_scale
            wait_until(target)
            request = event["request"]
            request_id = int(request["id"])
            reconfiguration_dispatch = None
            lag_ms = max(0.0, (time.monotonic() - target) * 1000.0)

            def finish_event(detail: dict[str, Any]) -> None:
                if reconfiguration_dispatch is not None:
                    controller_detail = detail.get("controller", {})
                    skipped = bool(controller_detail.get("skipped", False))
                    applied = (
                        bool(controller_detail.get("accepted", False)) and not skipped
                    )
                    attempted = not skipped
                    reason = str(
                        detail.get("reroute_reason")
                        or (detail.get("migration") or {}).get("reason")
                        or controller_detail.get("reason")
                        or (
                            "runtime reconfiguration applied"
                            if applied
                            else "runtime reconfiguration rejected"
                        )
                    )
                    if reconfiguration_dispatch.status_code == "DISPATCHED":
                        outcome = runtime_reconfiguration.record_outcome(
                            reconfiguration_dispatch,
                            success=applied,
                            applied=applied,
                            attempted=attempted,
                            reason=reason,
                        )
                    else:
                        outcome = {
                            "command_id": (
                                reconfiguration_dispatch.command.command_id
                            ),
                            "success": False,
                            "applied": False,
                            "attempted": False,
                            "status": "dispatch_rejected",
                            "reason": reconfiguration_dispatch.reason,
                        }
                    detail["orchestration"] = {
                        "architecture": "brain_module_skill",
                        "dispatch": reconfiguration_dispatch.to_dict(),
                        "outcome": outcome,
                    }
                    migration_record = detail.get("migration")
                    if event["type"] == "migration" and isinstance(
                        migration_record, dict
                    ):
                        migration_record["orchestration"] = detail["orchestration"]
                event_rows.append(
                    {
                        "request_id": request_id,
                        "type": event["type"],
                        "trace_time": float(event["time"]),
                        "wall_elapsed": time.monotonic() - wall_start,
                        "scheduler_lag_ms": lag_ms,
                        **detail,
                    }
                )

            sfc_plan_for_event = sfc_plans.get(request_id)
            sfc_rejected = (
                sfc_plan_for_event is not None
                and not sfc_plan_for_event.get("accepted", True)
            )
            if event["type"] == "arrive" and sfc_rejected:
                reason = str(sfc_plan_for_event.get("reason", "hrl_rejected"))
                precomputed_sender_results.append(
                    {
                        "request_id": request_id,
                        "result": startup_failure_sender(request, reason),
                    }
                )
                for destination in request["destination_dpids"]:
                    precomputed_receiver_results.append(
                        {
                            "request_id": request_id,
                            "destination_dpid": int(destination),
                            "result": startup_failure_receiver(
                                request, int(destination), reason
                            ),
                        }
                    )
                event_rows.append(
                    {
                        "request_id": request_id,
                        "type": "arrive",
                        "trace_time": float(event["time"]),
                        "wall_elapsed": time.monotonic() - wall_start,
                        "scheduler_lag_ms": lag_ms,
                        "controller": {
                            "accepted": False,
                            "reason": reason,
                            "source": (
                                "online_wqmix" if online_wqmix_enabled else "legacy_hrl"
                            ),
                        },
                        "paths": {},
                        "probe_status": "hrl_rejected",
                        "online_planning": sfc_plan_for_event.get("online_planning"),
                    }
                )
                return
            if event["type"] == "arrive":
                if (
                    args.deployment_max_queue_wait_ms > 0.0
                    and lag_ms > args.deployment_max_queue_wait_ms
                ):
                    reject_arrival_before_execution(
                        event,
                        code="deployment_queue_wait_rejected",
                        source="bounded_admission",
                        detail=(
                            f"request {request_id} rejected after waiting "
                            f"{lag_ms:.3f}ms for deployment; limit is "
                            f"{args.deployment_max_queue_wait_ms:.3f}ms"
                        ),
                        extra={
                            "deployment_queue_wait_ms": lag_ms,
                            "deployment_max_queue_wait_ms": (
                                args.deployment_max_queue_wait_ms
                            ),
                        },
                    )
                    return
                deployment_admission = deadline_admission(
                    request, phase="execution_start"
                )
                if isinstance(event.get("_enqueue_admission"), dict):
                    deployment_admission["enqueue"] = event["_enqueue_admission"]
                if deployment_admission["slack_ms"] <= 0.0:
                    reason = (
                        f"request {request_id} expired before deployment: "
                        f"{deployment_admission['remaining_lifetime_ms']:.3f}ms "
                        "remaining, "
                        f"{deployment_admission['required_lifetime_ms']:.3f}ms "
                        "required"
                    )
                    reject_arrival_before_execution(
                        event,
                        code="expired_before_deployment",
                        source="deadline_admission",
                        detail=reason,
                        extra={"deployment_admission": deployment_admission},
                    )
                    return
                plan = plans[request_id]
                sfc_plan = sfc_plans.get(request_id)
                candidate_selection_detail: dict[str, Any] | None = None
                candidate_selection_started = time.monotonic()
                with bandwidth_lock:
                    if sfc_candidate_plans:
                        sfc_plan, candidate_selection_detail = select_sfc_candidate(
                            request
                        )
                        if sfc_plan is None:
                            bandwidth_admitted = False
                            bandwidth_detail = {
                                "reason": "all_sfc_candidates_infeasible",
                                "candidate_selection": candidate_selection_detail,
                            }
                        else:
                            sfc_plans[request_id] = sfc_plan
                            plans[request_id] = index_sfc_execution_plan(
                                request, sfc_plan, "offline_sfc_candidate"
                            )
                            plan = plans[request_id]
                            bandwidth_admitted, bandwidth_detail = reserve_request_bandwidth(
                                request, sfc_plan
                            )
                    elif online_wqmix_enabled and sfc_plan is not None:
                        bandwidth_admitted, bandwidth_detail = (
                            mirror_committed_request_bandwidth(request, sfc_plan)
                        )
                    else:
                        admission_plan = sfc_plan if sfc_plan is not None else plan
                        bandwidth_admitted, bandwidth_detail = reserve_request_bandwidth(
                            request, admission_plan
                        )
                candidate_selection_elapsed_ms = (
                    time.monotonic() - candidate_selection_started
                ) * 1000.0
                if sfc_candidate_plans:
                    candidate_selection_timings_ms.append(
                        candidate_selection_elapsed_ms
                    )
                    if candidate_selection_detail is not None:
                        candidate_selection_detail["selection_elapsed_ms"] = (
                            candidate_selection_elapsed_ms
                        )
                if not bandwidth_admitted:
                    rejection_code = (
                        "sfc_candidate_infeasible"
                        if sfc_candidate_plans
                        and candidate_selection_detail
                        and candidate_selection_detail.get("reason")
                        else "physical_link_bandwidth_admission"
                    )
                    reject_arrival_before_execution(
                        event,
                        code=rejection_code,
                        source=(
                            "sfc_candidate_selection"
                            if rejection_code == "sfc_candidate_infeasible"
                            else "bandwidth_admission"
                        ),
                        detail=(
                            f"request {request_id} rejected before deployment because "
                            "no selected SFC candidate fits the current admission "
                            "ledger"
                            if rejection_code == "sfc_candidate_infeasible"
                            else f"request {request_id} rejected before deployment because "
                            "its planned link footprint exceeds the reserved "
                            "bandwidth budget"
                        ),
                        extra={
                            "bandwidth_admission": bandwidth_detail,
                            "candidate_selection": candidate_selection_detail,
                        },
                    )
                    return
                endpoint_pool_detail = None
                if vnf_endpoint_pool is not None and sfc_plan is not None:
                    pooled_plan, endpoint_pool_detail = vnf_endpoint_pool.assign(
                        request_id, sfc_plan
                    )
                    if pooled_plan is None:
                        release_request_bandwidth(request_id)
                        reject_arrival_before_execution(
                            event,
                            code="vnf_endpoint_pool_exhausted",
                            source="vnf_endpoint_pool",
                            detail=(
                                f"request {request_id} rejected because a selected "
                                "DC has no free reserved VNF endpoint"
                            ),
                            extra={"vnf_endpoint_pool": endpoint_pool_detail},
                        )
                        return
                    sfc_plan = pooled_plan
                    sfc_plans[request_id] = sfc_plan
                    plans[request_id] = index_sfc_execution_plan(
                        request, sfc_plan, "reserved_vnf_endpoint_pool"
                    )
                    plan = plans[request_id]
                sfc_plan_for_event = sfc_plan
                deployment_started = time.monotonic()
                neighbor_setup_ms = 0.0
                neighbor_entries = 0
                vnf_slot_wait_ms = 0.0
                vnf_command_ms = 0.0
                vnf_ready_wait_ms = 0.0
                vnf_register_batch_timing = None
                ryu_install_ms = 0.0
                ryu_batch_timing = None
                neighbor_vnf_overlap = False
                if sfc_plan is not None:
                    neighbor_started = time.monotonic()
                    neighbor_future = None
                    if neighbor_executor is not None:
                        neighbor_future = neighbor_executor.submit(
                            ensure_sfc_plan_neighbors,
                            args.mininet_command_port,
                            sfc_plan,
                            nodes,
                            args.timeout,
                        )
                        neighbor_vnf_overlap = True
                    else:
                        neighbor_entries = ensure_sfc_plan_neighbors(
                            args.mininet_command_port,
                            sfc_plan,
                            nodes,
                            args.timeout,
                        )
                        neighbor_setup_ms = (
                            time.monotonic() - neighbor_started
                        ) * 1000.0
                    active_vnfs.add(request_id)
                    segments = sorted(
                        sfc_plan["segments"],
                        key=lambda value: int(value["stage"]),
                        reverse=True,
                    )
                    registration_commands = []
                    registration_messages = []
                    ready_paths = []
                    use_registration_ack = (
                        args.vnf_launch_mode == "agent"
                        and args.vnf_ready_protocol == "ack"
                    )
                    for segment in segments:
                        stage = int(segment["stage"])
                        placement = sfc_plan["placement_by_vnf"][str(stage)]
                        dc_node = int(placement["dc_node"])
                        local_files = sfc_runtime_files(probe_dir, request_id, stage)
                        wsl_files = {
                            key: to_wsl_path(value) for key, value in local_files.items()
                        }
                        vnf_stats_paths.append(
                            (request_id, stage, dc_node, local_files["stats"])
                        )
                        if not use_registration_ack:
                            ready_paths.append(local_files["ready"])
                        host = str(nodes[dc_node]["host"])
                        if args.vnf_launch_mode == "agent":
                            registration_messages.append(
                                vnf_agent_register_message(
                                    vnf_agent_fifo_for_binding(
                                        vnf_agent_fifos,
                                        host,
                                        request_id,
                                        stage,
                                    ),
                                    sfc_plan,
                                    stage,
                                    wsl_files,
                                    dscp=int(request.get("dscp", 0)),
                                    include_ready_file=not use_registration_ack,
                                    include_stats_file=(
                                        args.vnf_agent_backend != "native"
                                    ),
                                )
                            )
                        else:
                            command = sfc_forwarder_command(
                                forwarder_wsl, sfc_plan, stage, wsl_files
                            )
                            registration_commands.append((host, command))

                    registration_context = (
                        vnf_registration_slots
                        if vnf_registration_slots is not None
                        else nullcontext()
                    )
                    vnf_slot_wait_started = time.monotonic()
                    with registration_context:
                        vnf_slot_wait_ms = (
                            time.monotonic() - vnf_slot_wait_started
                        ) * 1000.0
                        vnf_command_started = time.monotonic()
                        if args.vnf_launch_mode == "agent":
                            if use_registration_ack:
                                if vnf_register_batcher is not None:
                                    registration_result = vnf_register_batcher.submit(
                                        registration_messages
                                    )
                                else:
                                    registration_result = (
                                        mininet_runtime_fifo_messages_wait_ack(
                                            args.mininet_command_port,
                                            registration_messages,
                                            args.timeout,
                                        )
                                    )
                                vnf_register_batch_timing = registration_result.get(
                                    "batch_timing"
                                )
                            else:
                                registration_result = mininet_runtime_fifo_messages(
                                    args.mininet_command_port,
                                    registration_messages,
                                    args.timeout,
                                )
                        else:
                            registration_result = mininet_runtime_commands(
                                args.mininet_command_port,
                                registration_commands,
                                args.timeout,
                            )
                        registration_round_trip_ms = (
                            time.monotonic() - vnf_command_started
                        ) * 1000.0
                        if use_registration_ack:
                            vnf_command_ms = float(
                                registration_result.get(
                                    "dispatch_ms", registration_round_trip_ms
                                )
                            )
                            vnf_ready_wait_ms = float(
                                registration_result.get(
                                    "ack_wait_ms",
                                    max(
                                        0.0,
                                        registration_round_trip_ms - vnf_command_ms,
                                    ),
                                )
                            )
                        else:
                            vnf_command_ms = registration_round_trip_ms
                            vnf_ready_started = time.monotonic()
                            ready_count = wait_for_probe_files(
                                ready_paths, args.timeout, poll_seconds=0.005
                            )
                            vnf_ready_wait_ms = (
                                time.monotonic() - vnf_ready_started
                            ) * 1000.0
                            if ready_count != len(ready_paths):
                                missing = [
                                    str(path)
                                    for path in ready_paths
                                    if not path.is_file()
                                ]
                                raise RuntimeError(
                                    f"VNF request {request_id} has {len(missing)} "
                                    "stages not ready: " + ", ".join(missing)
                                )
                    if neighbor_future is not None:
                        neighbor_entries = int(neighbor_future.result())
                        neighbor_setup_ms = (
                            time.monotonic() - neighbor_started
                        ) * 1000.0
                    ryu_started = time.monotonic()
                    if ryu_batch_committer is not None:
                        response, ryu_batch_timing = ryu_batch_committer.submit(
                            sfc_plan
                        )
                    else:
                        response = client.install_sfc(sfc_plan)
                    ryu_install_ms = (time.monotonic() - ryu_started) * 1000.0
                else:
                    ryu_started = time.monotonic()
                    response = client.install_tree(
                        request["group_id"],
                        request["multicast_ip"],
                        plan["switch_outputs"],
                        source_dpid=request["source_dpid"],
                    )
                    ryu_install_ms = (time.monotonic() - ryu_started) * 1000.0
                installed.add(request_id)
                deployment_elapsed_ms = (
                    time.monotonic() - deployment_started
                ) * 1000.0
                deployment_timing = {
                    "neighbor_setup_ms": neighbor_setup_ms,
                    "neighbor_entries": neighbor_entries,
                    "vnf_slot_wait_ms": vnf_slot_wait_ms,
                    "vnf_command_ms": vnf_command_ms,
                    "vnf_ready_wait_ms": vnf_ready_wait_ms,
                    "vnf_register_batch": vnf_register_batch_timing,
                    "ryu_install_ms": ryu_install_ms,
                    "ryu_batch": ryu_batch_timing,
                    "total_ms": deployment_elapsed_ms,
                    "vnf_stages": len(sfc_plan["segments"]) if sfc_plan else 0,
                    "parallel_vnf_registration": bool(sfc_plan),
                    "parallel_pipeline": bool(args.parallel_deployment_pipeline),
                    "neighbor_vnf_overlap": neighbor_vnf_overlap,
                    "vnf_ready_protocol": (
                        args.vnf_ready_protocol if sfc_plan else None
                    ),
                }
                if prescheduled_probes:
                    setup_estimator.observe(deployment_elapsed_ms)
                    deployment_admission["observed_setup_ms"] = deployment_elapsed_ms
                    detail = {
                        "controller": response,
                        "paths": plan["paths"],
                        "probe_status": "prescheduled",
                        "deployment_elapsed_ms": deployment_elapsed_ms,
                        "deployment_timing": deployment_timing,
                        "deployment_admission": deployment_admission,
                        "online_planning": (
                            sfc_plan.get("online_planning") if sfc_plan else None
                        ),
                    }
                else:
                    leave_target = wall_start + (
                        float(request["leave_time"]) - origin
                    ) * args.time_scale
                    remaining_lifetime = leave_target - time.monotonic()
                    available_for_ready = (
                        remaining_lifetime
                        - args.sender_stop_margin_seconds
                        - args.minimum_traffic_seconds
                    )
                    if available_for_ready <= 0.0:
                        setup_estimator.observe(deployment_elapsed_ms)
                        deployment_admission["observed_setup_ms"] = (
                            deployment_elapsed_ms
                        )
                        reason = (
                            f"request {request_id} has only "
                            f"{remaining_lifetime:.3f}s left after tree installation; "
                            "no receiver readiness window remains"
                        )
                        precomputed_sender_results.append(
                            {
                                "request_id": request_id,
                                "result": startup_failure_sender(request, reason),
                            }
                        )
                        for destination in request["destination_dpids"]:
                            precomputed_receiver_results.append(
                                {
                                    "request_id": request_id,
                                    "destination_dpid": int(destination),
                                    "result": startup_failure_receiver(
                                        request, int(destination), reason
                                    ),
                                }
                            )
                        event_rows.append(
                            {
                                "request_id": request_id,
                                "type": event["type"],
                                "trace_time": float(event["time"]),
                                "wall_elapsed": time.monotonic() - wall_start,
                                "scheduler_lag_ms": lag_ms,
                                "controller": response,
                                "paths": plan["paths"],
                            "probe_status": "startup_deadline_missed",
                            "deployment_elapsed_ms": deployment_elapsed_ms,
                            "deployment_timing": deployment_timing,
                            "deployment_admission": deployment_admission,
                            }
                        )
                        return
                    ready_budget = min(
                        args.receiver_ready_seconds, available_for_ready
                    )
                    traffic_duration = (
                        remaining_lifetime - args.sender_stop_margin_seconds
                    )
                    if traffic_duration < args.minimum_traffic_seconds:
                        raise RuntimeError(
                            f"request {request_id} has only "
                            f"{remaining_lifetime:.3f}s left after tree installation; "
                            "increase --time-scale or reduce probe startup/margin values"
                        )
                    pps = args.packets_per_second
                    if pps <= 0.0:
                        pps = bandwidth_mbps_to_pps(
                            float(request["bw_origin"]),
                            args.payload_bytes,
                            args.packet_overhead_bytes,
                        )
                    # Windows and WSL wall clocks can be stepped independently
                    # during a run. Refresh the target-clock offset immediately
                    # before constructing this request's absolute probe deadline.
                    with dynamic_clock_sync_lock:
                        request_clock_sync = mininet_clock_offset(
                            args.mininet_command_port,
                            args.timeout,
                            samples=1,
                        )
                        request_clock_sync["request_id"] = request_id
                        request_clock_sync["trace_time"] = float(event["time"])
                        dynamic_clock_sync_observations.append(request_clock_sync)
                    remaining_stop_budget_seconds = max(
                        0.0, leave_target - time.monotonic()
                    )
                    sender_stop_budget_seconds = max(
                        0.0,
                        remaining_stop_budget_seconds
                        - args.sender_stop_margin_seconds,
                    )
                    sender_stop_time_ns, receiver_stop_time_ns = (
                        probe_stop_deadlines_ns(
                            time.time_ns(),
                            int(request_clock_sync["offset_ns"]),
                            remaining_stop_budget_seconds,
                            min(
                                args.sender_stop_margin_seconds,
                                remaining_stop_budget_seconds,
                            ),
                        )
                    )
                    expected_file = f"{handshake_dir_wsl}/r{request_id}.expected"
                    host_commands = []
                    receiver_messages = []
                    ready_files = []
                    receiver_ready_timing = None
                    sender_dispatch_ms = 0.0
                    sender_stagger_ms = (
                        float(ryu_batch_timing.get("batch_index", 0))
                        * args.ryu_batch_sender_stagger_ms
                        if ryu_batch_timing is not None
                        else 0.0
                    )
                    use_receiver_ack = (
                        args.receiver_ready_protocol == "ack"
                        and args.probe_launch_mode == "agent"
                    )
                    prelaunch_sender = bool(
                        args.probe_sender_prelaunch and use_receiver_ack
                    )
                    for destination in request["destination_dpids"]:
                        node = nodes[int(destination)]
                        probe_name = (
                            f"sft-runtime-{run_token}-r{request_id}-h{destination}.json"
                        )
                        probe_output = f"{probe_dir_wsl}/{probe_name}"
                        ready_output = (
                            f"{handshake_dir_wsl}/r{request_id}-h{destination}.ready"
                        )
                        if not use_receiver_ack or prelaunch_sender:
                            ready_files.append(ready_output)
                        probe_paths.append(
                            (request_id, int(destination), probe_dir / probe_name)
                        )
                        command = probe_receiver_command(
                            probe_wsl,
                            request,
                            str(node["host_ip"]).split("/", 1)[0],
                            remaining_lifetime,
                            0,
                            (
                                ready_output
                                if not use_receiver_ack or prelaunch_sender
                                else None
                            ),
                            receiver_stop_time_ns,
                            expected_file,
                            args.probe_receive_buffer_bytes,
                            probe_output,
                            agent_fifos.get(str(node["host"])),
                            return_agent_message=use_receiver_ack,
                            destination_id=int(destination),
                            receiver_backend=args.probe_receiver_backend,
                            native_receiver_path=native_receiver_wsl,
                        )
                        if use_receiver_ack:
                            if not isinstance(command, dict):
                                raise RuntimeError(
                                    "receiver ACK protocol requires an agent message"
                                )
                            receiver_messages.append(command)
                        else:
                            if not isinstance(command, str):
                                raise RuntimeError(
                                    "receiver file protocol requires a host command"
                                )
                            host_commands.append((str(node["host"]), command))

                    sender_name = (
                        f"sft-runtime-{run_token}-r{request_id}-sender.json"
                    )
                    sender_output = f"{probe_dir_wsl}/{sender_name}"
                    sender_paths.append((request_id, probe_dir / sender_name))
                    source_host = str(nodes[int(request["source_dpid"])]["host"])
                    sender_uses_agent = args.probe_sender_launch_mode == "agent"
                    sender_command = probe_sender_command(
                        probe_wsl,
                        sfc_sender_request(request, sfc_plan),
                        traffic_duration,
                        args.payload_bytes,
                        pps,
                        ready_files,
                        ready_budget,
                        sender_stop_time_ns,
                        args.minimum_traffic_seconds,
                        expected_file,
                        sender_output,
                        agent_fifos.get(source_host) if sender_uses_agent else None,
                        max_catch_up_packets=args.sender_max_catch_up_packets,
                        sender_backend=args.probe_sender_backend,
                        native_sender_path=native_sender_wsl,
                        native_realtime_priority=(
                            args.probe_native_realtime_priority
                        ),
                        return_agent_message=sender_uses_agent and use_receiver_ack,
                        cpu_set=(
                            args.probe_agent_cpu_set
                            if not sender_uses_agent
                            else ""
                        ),
                    )

                    # With receiver ACKs, prelaunch the sender before receiver setup.
                    # It blocks on the ready files above, overlapping process/agent
                    # scheduling with receiver bind and multicast membership setup.
                    if prelaunch_sender:
                        sender_dispatch_started = time.monotonic()
                        if isinstance(sender_command, dict):
                            mininet_runtime_fifo_messages(
                                args.mininet_command_port,
                                [sender_command],
                                args.timeout,
                            )
                        else:
                            mininet_runtime_commands(
                                args.mininet_command_port,
                                [(source_host, sender_command)],
                                args.timeout,
                            )
                        sender_dispatch_ms = (
                            time.monotonic() - sender_dispatch_started
                        ) * 1000.0
                    if use_receiver_ack:
                        try:
                            receiver_ack_result = (
                                mininet_runtime_fifo_messages_wait_ack(
                                    args.mininet_command_port,
                                    receiver_messages,
                                    args.timeout,
                                    ack_timeout=max(0.001, ready_budget),
                                )
                            )
                        except RuntimeFifoAckTimeout as exc:
                            raise ReceiverStartupTimeout(
                                f"request {request_id} receiver startup failed: {exc}"
                            ) from exc
                        receiver_ready_timing = {
                            "protocol": "unix_ack",
                            "receivers": len(receiver_ack_result.get("acks", [])),
                            "dispatch_ms": float(
                                receiver_ack_result.get("dispatch_ms", 0.0)
                            ),
                            "ack_wait_ms": float(
                                receiver_ack_result.get("ack_wait_ms", 0.0)
                            ),
                            "total_ms": float(
                                receiver_ack_result.get("total_ms", 0.0)
                            ),
                        }
                        if not prelaunch_sender:
                            if sender_stagger_ms > 0.0:
                                wait_until(
                                    time.monotonic() + sender_stagger_ms / 1000.0
                                )
                            sender_dispatch_started = time.monotonic()
                            if isinstance(sender_command, dict):
                                mininet_runtime_fifo_messages(
                                    args.mininet_command_port,
                                    [sender_command],
                                    args.timeout,
                                )
                            else:
                                mininet_runtime_commands(
                                    args.mininet_command_port,
                                    [(source_host, sender_command)],
                                    args.timeout,
                                )
                            sender_dispatch_ms = (
                                time.monotonic() - sender_dispatch_started
                            ) * 1000.0
                    else:
                        if isinstance(sender_command, dict):
                            raise AssertionError(
                                "non-ACK probe startup produced an agent message"
                            )
                        host_commands.append((source_host, sender_command))
                        if sender_stagger_ms > 0.0:
                            wait_until(
                                time.monotonic() + sender_stagger_ms / 1000.0
                            )
                        sender_dispatch_started = time.monotonic()
                        mininet_runtime_commands(
                            args.mininet_command_port, host_commands, args.timeout
                        )
                        sender_dispatch_ms = (
                            time.monotonic() - sender_dispatch_started
                        ) * 1000.0

                    setup_observed_ms = (
                        time.monotonic() - deployment_started
                    ) * 1000.0
                    setup_estimator.observe(setup_observed_ms)
                    deployment_admission["observed_setup_ms"] = setup_observed_ms
                    deployment_timing["sender_dispatch_ms"] = sender_dispatch_ms
                    deployment_timing["sender_stagger_ms"] = sender_stagger_ms
                    deployment_timing["probe_clock_offset_ns"] = int(
                        request_clock_sync["offset_ns"]
                    )
                    deployment_timing["sender_stop_budget_ms"] = (
                        sender_stop_budget_seconds * 1000.0
                    )
                    deployment_timing["receiver_stop_budget_ms"] = (
                        remaining_stop_budget_seconds * 1000.0
                    )
                    deployment_timing["probe_drain_window_ms"] = (
                        max(
                            0,
                            receiver_stop_time_ns - sender_stop_time_ns,
                        )
                        / 1_000_000.0
                    )
                    deployment_timing["setup_observed_ms"] = setup_observed_ms
                    detail = {
                        "controller": response,
                        "paths": plan["paths"],
                        "deployment_elapsed_ms": deployment_elapsed_ms,
                        "deployment_timing": deployment_timing,
                        "deployment_admission": deployment_admission,
                        "receiver_ready_timing": receiver_ready_timing,
                        "bandwidth_admission": bandwidth_detail,
                        "vnf_endpoint_pool": endpoint_pool_detail,
                        "candidate_selection": candidate_selection_detail,
                        "online_planning": (
                            sfc_plan.get("online_planning") if sfc_plan else None
                        ),
                        "sfc": (
                            {
                                "chain_nodes": sfc_plan["chain_nodes"],
                                "placement_by_vnf": sfc_plan["placement_by_vnf"],
                                "segments": sfc_plan["segments"],
                            }
                            if sfc_plan is not None
                            else None
                        ),
                    }
            elif event["type"] == "impairment":
                impairment = event["impairment"]
                if request_id not in installed or request_id not in active_vnfs:
                    skipped_impairments += 1
                    detail = {
                        "controller": {"accepted": False, "skipped": True},
                        "reason": "request_not_active",
                        "impairment": impairment,
                    }
                else:
                    stage = int(impairment["stage"])
                    placement = sfc_plans[request_id]["placement_by_vnf"].get(
                        str(stage)
                    )
                    if placement is None:
                        skipped_impairments += 1
                        detail = {
                            "controller": {"accepted": False, "skipped": True},
                            "reason": "stage_not_found",
                            "impairment": impairment,
                        }
                    else:
                        dc_node = int(placement["dc_node"])
                        host = str(nodes[dc_node]["host"])
                        message = {
                            "fifo": vnf_agent_fifo_for_binding(
                                vnf_agent_fifos, host, request_id, stage
                            ),
                            "payload": {
                                "operation": "update_impairment",
                                "request_id": request_id,
                                "stage": stage,
                                "processing_delay_us": int(
                                    impairment["processing_delay_us"]
                                ),
                                "drop_every": int(impairment["drop_every"]),
                            },
                        }
                        response = mininet_runtime_fifo_messages_wait_ack(
                            args.mininet_command_port, [message], args.timeout
                        )
                        applied_impairments += 1
                        detail = {
                            "controller": {"accepted": True},
                            "agent": response.get("acks", [{}])[0],
                            "dc_node": dc_node,
                            "impairment": impairment,
                        }
            elif event["type"] == "migration":
                migration = event["migration"]
                live_ledger = getattr(online_planner, "ledger", None)
                live_ledger_snapshot = (
                    live_ledger.snapshot() if live_ledger is not None else None
                )
                reconfiguration_dispatch = runtime_reconfiguration.route_migration(
                    timestamp=float(event["time"]),
                    request_id=request_id,
                    payload={
                        **migration,
                        "request_active": (
                            request_id in installed and request_id in active_vnfs
                        ),
                    },
                    metrics={
                        "active_requests": len(installed),
                        "node_hotspots": 1,
                        "link_hotspots": 0,
                    },
                    snapshot_version=(
                        f"online-ledger-{live_ledger_snapshot.version}"
                        if live_ledger_snapshot is not None
                        else None
                    ),
                )
                migration_started = time.monotonic()
                record: dict[str, Any] = {
                    "request_id": request_id,
                    "stage": int(migration["stage"]),
                    "target_dc": int(migration["target_dc"]),
                    "trace_time": float(event["time"]),
                    "policy": migration.get("policy", "external"),
                }
                for key in (
                    "trigger",
                    "target_rankings",
                    "migration_decision_ms",
                ):
                    if key in migration:
                        record[key] = migration[key]
                migration_records.append(record)
                if request_id not in installed or request_id not in active_vnfs:
                    skipped_migrations += 1
                    record.update(success=False, skipped=True, reason="request_not_active")
                    detail = {"controller": {"accepted": False, "skipped": True},
                              "migration": record}
                else:
                    stage = int(migration["stage"])
                    target_dc = int(migration["target_dc"])
                    current_plan = sfc_plans[request_id]
                    placement = current_plan["placement_by_vnf"].get(str(stage))
                    endpoint_token = None
                    ryu_token = None
                    target_registered = False
                    ledger_replaced = False
                    ledger_replacement = None
                    old_ledger_footprint = None
                    new_ledger_footprint = None
                    ledger_preparation = None
                    ledger_preparation_token = None
                    bandwidth_committed = False
                    endpoint_committed = False
                    runtime_plan_updated = False
                    migration_counted = False
                    upstream_switched = False
                    upstream_switch_attempted = False
                    upstream_fifo = None
                    old_next_host = None
                    old_next_port = None
                    migration_epoch = None
                    controller_committed = False
                    controller_commit_attempted = False
                    controller_commit_uncertain = False
                    try:
                        if placement is None:
                            raise ValueError(f"SFC stage {stage} does not exist")
                        old_dc = int(placement["dc_node"])
                        migration_policy = str(migration.get("policy", "external"))
                        migration_rankings = None
                        private_candidate_plans = event.get(
                            "_candidate_plans_by_target", {}
                        )
                        selected_candidate_plan = event.get("_target_plan")
                        if selected_candidate_plan is None:
                            selected_candidate_plan = private_candidate_plans.get(
                                target_dc,
                                private_candidate_plans.get(str(target_dc)),
                            )
                        if target_dc <= 0:
                            if not (
                                args.online_wqmix_auto_migration
                                and online_wqmix_enabled
                                and migration_online_planner is not None
                            ):
                                raise ValueError(
                                    "automatic migration target requires migration-specific WQMIX"
                                )
                            decision_key = (request_id, stage, float(event["time"]))
                            if decision_key not in migration_decision_cache:
                                same_time = [migration] + [
                                    row for row in selected_migrations
                                    if row is not migration
                                    and int(row.get("target_dc", 0)) <= 0
                                    and abs(float(row["time"]) - float(event["time"])) <= 1e-12
                                ]
                                batch_limit = min(
                                    int(args.online_migration_max_agents),
                                    int(migration_online_planner.max_agents),
                                )
                                batch_entries = []
                                batch_keys = []
                                seen_tasks = set()
                                for row in same_time:
                                    row_request_id = int(row["request_id"])
                                    row_stage = int(row["stage"])
                                    row_key = (
                                        row_request_id, row_stage, float(row["time"])
                                    )
                                    if row_key in seen_tasks or row_request_id not in sfc_plans:
                                        continue
                                    row_request = request_by_id.get(row_request_id)
                                    if row_request is None:
                                        continue
                                    seen_tasks.add(row_key)
                                    enriched_request = dict(row_request)
                                    online_prediction = (
                                        plans.get(row_request_id, {}).get(
                                            "online_planning"
                                        )
                                        or {}
                                    )
                                    predicted_sla_risk = online_prediction.get(
                                        "predicted_sla_risk",
                                        enriched_request.get("predicted_sla_risk"),
                                    )
                                    if predicted_sla_risk is not None:
                                        enriched_request["predicted_sla_risk"] = float(
                                            predicted_sla_risk
                                        )
                                    batch_entries.append({
                                        "request": enriched_request,
                                        "current_plan": sfc_plans[row_request_id],
                                        "stage": row_stage,
                                    })
                                    batch_keys.append(row_key)
                                    if len(batch_entries) >= batch_limit:
                                        break
                                batch_decisions = migration_online_planner.plan_batch(
                                    batch_entries,
                                    online_planner.ledger.snapshot(),
                                    float(event["time"]),
                                )
                                migration_decision_cache.update(
                                    zip(batch_keys, batch_decisions)
                                )
                            migration_decision = migration_decision_cache[decision_key]
                            if not migration_decision["accepted"]:
                                raise _MigrationPolicyNoop(
                                    "migration_wqmix_selected_no_migration"
                                )
                            target_dc = int(migration_decision["target_dc"])
                            selected_candidate_plan = migration_decision.get("target_plan")
                            if selected_candidate_plan is None:
                                raise RuntimeError("migration_wqmix_selected_plan_missing")
                            trial_plan = migrated_sft_plan(
                                profile,
                                current_plan,
                                stage,
                                target_dc,
                                args.vnf_agent_prebound_port_base,
                                candidate_plan=selected_candidate_plan,
                            )
                            safety_rejections = []
                            flow_safe, flow_reason = make_before_break_compatible(
                                current_plan, trial_plan
                            )
                            if not flow_safe:
                                safety_rejections.append({
                                    "target_dc": target_dc,
                                    "reason": flow_reason,
                                })
                                raise RuntimeError(
                                    "migration_wqmix_selected_plan_not_make_before_break_safe"
                                )
                            with bandwidth_lock:
                                bandwidth_safe, _, bandwidth_gate = (
                                    check_replacement_bandwidth(request, trial_plan)
                                )
                            if not bandwidth_safe:
                                safety_rejections.append({
                                    "target_dc": target_dc,
                                    "reason": "replacement_bandwidth",
                                    "detail": bandwidth_gate,
                                })
                                raise RuntimeError(
                                    "migration_wqmix_selected_plan_bandwidth_stale"
                                )
                            migration_policy = "migration_wqmix_joint"
                            record["target_rankings"] = migration_decision["rankings"]
                            record["target_safety_rejections"] = safety_rejections
                            record["target_safety_fallback"] = False
                            selected_candidate = migration_decision.get(
                                "selected_candidate"
                            ) or {}
                            if selected_candidate.get("candidate_id") is not None:
                                record["candidate_id"] = selected_candidate["candidate_id"]
                            record["migration_decision_ms"] = float(
                                migration_decision["decision_ms"]
                            )
                            record["migration_decoder"] = migration_decision["decoder"]
                            record["policy"] = migration_policy
                        elif migration_policy == "predictive_migration_wqmix":
                            if selected_candidate_plan is None:
                                raise RuntimeError(
                                    "predictive_migration_wqmix_selected_plan_missing"
                                )
                            record["target_safety_fallback"] = False
                        safety_plan = migrated_sft_plan(
                            profile,
                            current_plan,
                            stage,
                            target_dc,
                            args.vnf_agent_prebound_port_base,
                            candidate_plan=selected_candidate_plan,
                        )
                        flow_safe, flow_reason = make_before_break_compatible(
                            current_plan, safety_plan
                        )
                        if not flow_safe:
                            raise RuntimeError(
                                f"migration_make_before_break_gate:{flow_reason}"
                            )
                        if live_ledger is not None:
                            old_ledger_footprint = ResourceFootprint.from_sfc_plan(
                                current_plan, float(request.get("bw_origin", 0.0))
                            )
                            new_ledger_footprint = ResourceFootprint.from_sfc_plan(
                                safety_plan, float(request.get("bw_origin", 0.0))
                            )
                            temporary_footprint = migration_overlap_footprint(
                                current_plan,
                                safety_plan,
                                stage,
                                float(request.get("bw_origin", 0.0)),
                            )
                            prepare_replacement = getattr(
                                online_planner, "prepare_replacement", None
                            )
                            if callable(prepare_replacement):
                                ledger_preparation = prepare_replacement(
                                    request_id, temporary_footprint
                                )
                            else:
                                ledger_preparation = live_ledger.prepare_replacement(
                                    request_id, temporary_footprint
                                )
                            if not ledger_preparation.get("prepared"):
                                raise RuntimeError(
                                    "migration_ledger_prepare_"
                                    + str(ledger_preparation.get("reason", "failed"))
                                )
                            ledger_preparation_token = str(
                                ledger_preparation["token"]
                            )
                            record["ledger_preparation"] = ledger_preparation
                        endpoint_token, target_port, endpoint_detail = (
                            vnf_endpoint_pool.reserve_migration(
                                request_id, stage, target_dc
                            )
                        )
                        if endpoint_token is None or target_port is None:
                            raise RuntimeError(
                                str(endpoint_detail.get("reason", "endpoint_reservation_failed"))
                            )
                        new_plan = migrated_sft_plan(
                            profile,
                            current_plan,
                            stage,
                            target_dc,
                            target_port,
                            candidate_plan=selected_candidate_plan,
                        )
                        with bandwidth_lock:
                            bandwidth_ok, proposed_bandwidth, bandwidth_detail = (
                                check_replacement_bandwidth(request, new_plan)
                            )
                        if not bandwidth_ok:
                            raise RuntimeError("migration_bandwidth_infeasible")
                        old_host = str(nodes[old_dc]["host"])
                        target_host = str(nodes[target_dc]["host"])
                        old_fifo = vnf_agent_fifo_for_binding(
                            vnf_agent_fifos, old_host, request_id, stage
                        )
                        target_fifo = vnf_agent_fifo_for_binding(
                            vnf_agent_fifos, target_host, request_id, stage
                        )
                        old_snapshot = mininet_runtime_fifo_messages_wait_ack(
                            args.mininet_command_port,
                            [{"fifo": old_fifo, "payload": {
                                "operation": "snapshot", "request_id": request_id,
                                "stage": stage,
                            }}], args.timeout,
                        )["acks"][0]
                        state = old_snapshot["state"]
                        files = sfc_runtime_files(probe_dir, request_id, stage)
                        register_message = vnf_agent_register_message(
                            target_fifo, new_plan, stage,
                            {key: to_wsl_path(value) for key, value in files.items()},
                            dscp=int(request.get("dscp", 0)),
                            include_ready_file=False,
                            include_stats_file=(args.vnf_agent_backend != "native"),
                        )
                        register_message["payload"].update(
                            {
                                "restore_received": int(state["received"]),
                                "restore_forwarded": int(state["forwarded"]),
                                "restore_dropped": int(state["dropped"]),
                                "migration_epoch": int(state.get("migration_epoch", 0)) + 1,
                            }
                        )
                        target_ack = mininet_runtime_fifo_messages_wait_ack(
                            args.mininet_command_port, [register_message], args.timeout
                        )["acks"][0]
                        target_registered = True
                        prepared = client.prepare_sfc_migration(new_plan)
                        ryu_token = str(prepared["migration_token"])
                        if live_ledger is not None:
                            new_ledger_footprint = ResourceFootprint.from_sfc_plan(
                                new_plan, float(request.get("bw_origin", 0.0))
                            )
                        previous_stage = stage - 1
                        previous_dc = int(
                            current_plan["placement_by_vnf"][str(previous_stage)]["dc_node"]
                        )
                        previous_host = str(nodes[previous_dc]["host"])
                        upstream_fifo = vnf_agent_fifo_for_binding(
                            vnf_agent_fifos,
                            previous_host,
                            request_id,
                            previous_stage,
                        )
                        current_incoming_segment = next(
                            segment
                            for segment in current_plan["segments"]
                            if int(segment["stage"]) == stage
                        )
                        new_incoming_segment = next(
                            segment
                            for segment in new_plan["segments"]
                            if int(segment["stage"]) == stage
                        )
                        old_next_host = str(current_incoming_segment["target_ip"])
                        old_next_port = int(current_incoming_segment["udp_port"])
                        migration_epoch = int(state.get("migration_epoch", 0)) + 1
                        switch_started = time.monotonic()
                        upstream_switch_attempted = True
                        # Treat a missing ACK as possibly applied.  The failure path
                        # must explicitly restore the old endpoint before cleanup.
                        upstream_switched = True
                        switched = mininet_runtime_fifo_messages_wait_ack(
                            args.mininet_command_port,
                            [{"fifo": upstream_fifo, "payload": {
                                "operation": "update_next", "request_id": request_id,
                                "stage": previous_stage,
                                "next_host": str(new_incoming_segment["target_ip"]),
                                "next_port": int(new_incoming_segment["udp_port"]),
                                "migration_epoch": migration_epoch,
                            }}], args.timeout,
                        )["acks"][0]
                        switch_rpc_ms = (time.monotonic() - switch_started) * 1000.0
                        switch_ms = float(
                            switched.get("control_latency_ms", switch_rpc_ms)
                        )
                        drained = mininet_runtime_fifo_messages_wait_ack(
                            args.mininet_command_port,
                            [{"fifo": old_fifo, "payload": {
                                "operation": "drain", "request_id": request_id,
                                "stage": stage,
                                "drain_timeout_ms": args.vnf_migration_drain_ms,
                                "drain_idle_ms": args.vnf_migration_idle_ms,
                            }}], args.timeout,
                            ack_timeout=min(args.timeout, args.vnf_migration_drain_ms / 1000.0 + 1.0),
                        )["acks"][0]
                        final_state = mininet_runtime_fifo_messages_wait_ack(
                            args.mininet_command_port,
                            [{"fifo": old_fifo, "payload": {
                                "operation": "snapshot", "request_id": request_id,
                                "stage": stage,
                            }}], args.timeout,
                        )["acks"][0]["state"]
                        delta = {
                            key: max(0, int(final_state[key]) - int(state[key]))
                            for key in ("received", "forwarded", "dropped")
                        }
                        epoch = int(migration_epoch)
                        merged = mininet_runtime_fifo_messages_wait_ack(
                            args.mininet_command_port,
                            [{"fifo": target_fifo, "payload": {
                                "operation": "restore_delta", "request_id": request_id,
                                "stage": stage, "migration_epoch": epoch,
                                "delta_received": delta["received"],
                                "delta_forwarded": delta["forwarded"],
                                "delta_dropped": delta["dropped"],
                            }}], args.timeout,
                        )["acks"][0]
                        controller_commit_attempted = True
                        try:
                            committed = client.commit_sfc_migration(
                                request_id, ryu_token, drain_seconds=0.0
                            )
                            controller_committed = True
                        except Exception as commit_exc:
                            try:
                                controller_status = client.status()
                                controller_committed = ryu_status_matches_sfc_plan(
                                    controller_status, request_id, new_plan
                                )
                                record["controller_commit_probe"] = {
                                    "matched_new_plan": controller_committed,
                                    "error": str(commit_exc),
                                }
                            except Exception as probe_exc:
                                record["controller_commit_probe"] = {
                                    "matched_new_plan": False,
                                    "error": str(commit_exc),
                                    "probe_error": str(probe_exc),
                                }
                            if not controller_committed:
                                controller_commit_uncertain = True
                                raise
                            committed = {
                                "accepted": True,
                                "committed": True,
                                "request_id": request_id,
                                "recovered_from_status": True,
                                "commit_response_error": str(commit_exc),
                            }

                        post_commit_errors = []
                        if (
                            live_ledger is not None
                            and ledger_preparation_token is not None
                            and new_ledger_footprint is not None
                        ):
                            try:
                                commit_prepared_replacement = getattr(
                                    online_planner,
                                    "commit_prepared_replacement",
                                    None,
                                )
                                if callable(commit_prepared_replacement):
                                    ledger_replacement = commit_prepared_replacement(
                                        request_id,
                                        ledger_preparation_token,
                                        new_ledger_footprint,
                                    )
                                else:
                                    ledger_replacement = (
                                        live_ledger.commit_prepared_replacement(
                                            request_id,
                                            ledger_preparation_token,
                                            new_ledger_footprint,
                                        )
                                    )
                                if ledger_replacement.get("replaced"):
                                    ledger_replaced = True
                                    ledger_preparation_token = None
                                else:
                                    reason = str(
                                        ledger_replacement.get("reason", "failed")
                                    )
                                    post_commit_errors.append({
                                        "component": "ledger",
                                        "error": reason,
                                    })
                                    record["ledger_reconciliation_required"] = True
                                    record["ledger_reconciliation_reason"] = reason
                            except Exception as ledger_exc:
                                post_commit_errors.append({
                                    "component": "ledger",
                                    "error": str(ledger_exc),
                                })
                                record["ledger_reconciliation_required"] = True
                                record["ledger_reconciliation_reason"] = str(ledger_exc)
                        try:
                            with bandwidth_lock:
                                commit_replacement_bandwidth(
                                    request_id, proposed_bandwidth
                                )
                            bandwidth_committed = True
                        except Exception as bandwidth_exc:
                            post_commit_errors.append({
                                "component": "bandwidth_mirror",
                                "error": str(bandwidth_exc),
                            })
                        try:
                            endpoint_committed = bool(
                                vnf_endpoint_pool.finish_migration(
                                    endpoint_token, commit=True
                                )
                            )
                            if not endpoint_committed:
                                raise RuntimeError("endpoint_migration_token_not_found")
                        except Exception as endpoint_exc:
                            post_commit_errors.append({
                                "component": "endpoint_pool",
                                "error": str(endpoint_exc),
                            })
                        try:
                            sfc_plans[request_id] = new_plan
                            plans[request_id] = index_sfc_execution_plan(
                                request, new_plan, "runtime_vnf_migration"
                            )
                            runtime_plan_updated = True
                        except Exception as plan_exc:
                            post_commit_errors.append({
                                "component": "runtime_plan",
                                "error": str(plan_exc),
                            })
                        old_unregistered = {
                            "accepted": False,
                            "cleanup_pending": True,
                            "reason": "endpoint_pool_not_committed",
                        }
                        if endpoint_committed:
                            try:
                                old_unregistered = (
                                    mininet_runtime_fifo_messages_wait_ack(
                                        args.mininet_command_port,
                                        [vnf_agent_unregister_message(
                                            old_fifo, request_id, stage
                                        )],
                                        args.timeout,
                                    )["acks"][0]
                                )
                                source_endpoint_released = bool(
                                    vnf_endpoint_pool.release_migration_source(
                                        endpoint_token
                                    )
                                )
                                old_unregistered[
                                    "endpoint_released"
                                ] = source_endpoint_released
                                if not source_endpoint_released:
                                    raise RuntimeError(
                                        "retired_source_endpoint_not_found"
                                    )
                            except Exception as cleanup_exc:
                                old_unregistered = {
                                    "accepted": False,
                                    "cleanup_pending": True,
                                    "error": str(cleanup_exc),
                                }
                                post_commit_errors.append({
                                    "component": "source_unregister",
                                    "error": str(cleanup_exc),
                                })
                        if predictive_migration_monitor is not None:
                            try:
                                predictive_migration_monitor.mark_migrated(
                                    request_id, stage, float(event["time"])
                                )
                            except Exception as monitor_exc:
                                post_commit_errors.append({
                                    "component": "migration_monitor",
                                    "error": str(monitor_exc),
                                })
                        applied_migrations += 1
                        migration_counted = True
                        record.update(
                            success=True, old_dc=old_dc, target_port=target_port,
                            target_dc=target_dc,
                            prepare_ms=float(prepared.get("prepare_ms", 0.0)),
                            switch_ms=switch_ms,
                            switch_rpc_ms=switch_rpc_ms,
                            drain_ms=float(drained.get("drain_wait_ms", 0.0)),
                            total_ms=(time.monotonic() - migration_started) * 1000.0,
                            state_delta=delta,
                            ledger_replacement=ledger_replacement,
                        )
                        if post_commit_errors:
                            record["post_commit_reconciliation_required"] = True
                            record["post_commit_errors"] = post_commit_errors
                        detail = {
                            "controller": committed, "migration": record,
                            "target_register": target_ack, "switch": switched,
                            "state_merge": merged, "source_unregister": old_unregistered,
                            "bandwidth": bandwidth_detail,
                        }
                    except _MigrationPolicyNoop as exc:
                        skipped_migrations += 1
                        record.update(
                            success=False,
                            skipped=True,
                            reason="policy_noop",
                            detail=str(exc),
                        )
                        detail = {
                            "controller": {
                                "accepted": False,
                                "skipped": True,
                                "reason": "policy_noop",
                            },
                            "migration": record,
                        }
                    except Exception as exc:
                        if controller_committed:
                            if not migration_counted:
                                applied_migrations += 1
                                migration_counted = True
                            if (
                                live_ledger is not None
                                and ledger_preparation_token is not None
                                and new_ledger_footprint is not None
                                and not ledger_replaced
                            ):
                                try:
                                    commit_prepared_replacement = getattr(
                                        online_planner,
                                        "commit_prepared_replacement",
                                        None,
                                    )
                                    ledger_replacement = (
                                        commit_prepared_replacement(
                                            request_id,
                                            ledger_preparation_token,
                                            new_ledger_footprint,
                                        )
                                        if callable(commit_prepared_replacement)
                                        else live_ledger.commit_prepared_replacement(
                                            request_id,
                                            ledger_preparation_token,
                                            new_ledger_footprint,
                                        )
                                    )
                                    if ledger_replacement.get("replaced"):
                                        ledger_replaced = True
                                        ledger_preparation_token = None
                                    else:
                                        record["ledger_reconciliation_reason"] = str(
                                            ledger_replacement.get("reason", "failed")
                                        )
                                except Exception as ledger_exc:
                                    record["ledger_reconciliation_reason"] = str(
                                        ledger_exc
                                    )
                                if not ledger_replaced:
                                    record["ledger_reconciliation_required"] = True
                            if not bandwidth_committed:
                                try:
                                    with bandwidth_lock:
                                        commit_replacement_bandwidth(
                                            request_id, proposed_bandwidth
                                        )
                                    bandwidth_committed = True
                                except Exception as bandwidth_exc:
                                    record["bandwidth_reconciliation_error"] = str(
                                        bandwidth_exc
                                    )
                            if endpoint_token is not None and not endpoint_committed:
                                try:
                                    endpoint_committed = bool(
                                        vnf_endpoint_pool.finish_migration(
                                            endpoint_token, commit=True
                                        )
                                    )
                                except Exception as endpoint_exc:
                                    record["endpoint_reconciliation_error"] = str(
                                        endpoint_exc
                                    )
                            if not runtime_plan_updated:
                                try:
                                    sfc_plans[request_id] = new_plan
                                    plans[request_id] = index_sfc_execution_plan(
                                        request, new_plan, "runtime_vnf_migration"
                                    )
                                    runtime_plan_updated = True
                                except Exception as plan_exc:
                                    record["runtime_plan_reconciliation_error"] = str(
                                        plan_exc
                                    )
                            record.update(
                                success=True,
                                applied_with_reconciliation=True,
                                post_commit_error=str(exc),
                                post_commit_reconciliation_required=True,
                                total_ms=(time.monotonic() - migration_started) * 1000.0,
                            )
                            detail = {
                                "controller": {
                                    "accepted": True,
                                    "committed": True,
                                    "reconciliation_required": True,
                                },
                                "migration": record,
                            }
                            finish_event(detail)
                            return

                        failed_migrations += 1
                        if (
                            upstream_switched
                            and not controller_commit_uncertain
                            and upstream_fifo is not None
                            and old_next_host is not None
                            and old_next_port is not None
                            and migration_epoch is not None
                        ):
                            try:
                                record["upstream_rollback"] = (
                                    mininet_runtime_fifo_messages_wait_ack(
                                        args.mininet_command_port,
                                        [{"fifo": upstream_fifo, "payload": {
                                            "operation": "update_next",
                                            "request_id": request_id,
                                            "stage": stage - 1,
                                            "next_host": old_next_host,
                                            "next_port": old_next_port,
                                            "migration_epoch": migration_epoch,
                                        }}],
                                        args.timeout,
                                    )["acks"][0]
                                )
                                upstream_switched = False
                            except Exception as rollback_exc:
                                record["upstream_rollback"] = {
                                    "accepted": False,
                                    "error": str(rollback_exc),
                                }
                        record.update(
                            success=False, error=str(exc),
                            upstream_switch_attempted=upstream_switch_attempted,
                            controller_commit_attempted=controller_commit_attempted,
                            controller_commit_uncertain=controller_commit_uncertain,
                            total_ms=(time.monotonic() - migration_started) * 1000.0,
                        )
                        cleanup_confirmed = (
                            not controller_commit_uncertain and not upstream_switched
                        )
                        if cleanup_confirmed and ryu_token is not None:
                            try:
                                record["controller_abort"] = client.abort_sfc_migration(
                                    request_id, ryu_token
                                )
                            except Exception as abort_exc:
                                cleanup_confirmed = False
                                record["controller_abort"] = {
                                    "accepted": False,
                                    "error": str(abort_exc),
                                }
                        if cleanup_confirmed and target_registered and endpoint_token is not None:
                            try:
                                target_host = str(nodes[target_dc]["host"])
                                target_fifo = vnf_agent_fifo_for_binding(
                                    vnf_agent_fifos, target_host, request_id, stage
                                )
                                record["target_unregister"] = mininet_runtime_fifo_messages_wait_ack(
                                    args.mininet_command_port,
                                    [vnf_agent_unregister_message(
                                        target_fifo, request_id, stage
                                    )], args.timeout,
                                )["acks"][0]
                            except Exception as unregister_exc:
                                cleanup_confirmed = False
                                record["target_unregister"] = {
                                    "accepted": False,
                                    "error": str(unregister_exc),
                                }
                        if cleanup_confirmed and endpoint_token is not None:
                            cleanup_confirmed = bool(
                                vnf_endpoint_pool.finish_migration(
                                    endpoint_token, commit=False
                                )
                            )
                            record["endpoint_reservation_aborted"] = cleanup_confirmed
                        if cleanup_confirmed and ledger_preparation_token is not None:
                            abort_prepared_replacement = getattr(
                                online_planner,
                                "abort_prepared_replacement",
                                None,
                            )
                            try:
                                record["ledger_preparation_abort"] = (
                                    abort_prepared_replacement(
                                        ledger_preparation_token
                                    )
                                    if callable(abort_prepared_replacement)
                                    else live_ledger.abort_prepared_replacement(
                                        ledger_preparation_token
                                    )
                                )
                                cleanup_confirmed = bool(
                                    record["ledger_preparation_abort"].get("aborted")
                                )
                                if cleanup_confirmed:
                                    ledger_preparation_token = None
                            except Exception as ledger_abort_exc:
                                cleanup_confirmed = False
                                record["ledger_preparation_abort"] = {
                                    "aborted": False,
                                    "error": str(ledger_abort_exc),
                                }
                        if not cleanup_confirmed:
                            record["cleanup_reconciliation_required"] = True
                            record["resources_preserved_for_safety"] = True
                        detail = {"controller": {"accepted": False}, "migration": record}
            elif event["type"] == "reroute":
                reroute = event["reroute"]
                live_ledger = getattr(online_planner, "ledger", None)
                live_ledger_snapshot = (
                    live_ledger.snapshot() if live_ledger is not None else None
                )
                reconfiguration_dispatch = runtime_reconfiguration.route_reroute(
                    timestamp=float(event["time"]),
                    request_id=request_id,
                    payload={
                        **reroute,
                        "request_active": request_id in installed,
                    },
                    metrics={
                        "active_requests": len(installed),
                        "node_hotspots": 0,
                        "link_hotspots": 1,
                    },
                    sla_alert=(
                        str(reroute.get("trigger", "")).lower() == "sla_alert"
                    ),
                    snapshot_version=(
                        f"online-ledger-{live_ledger_snapshot.version}"
                        if live_ledger_snapshot is not None
                        else None
                    ),
                )
                if request_id not in installed:
                    skipped_reroutes += 1
                    reroute_chain_valid[request_id] = False
                    reroute_gate_rejections["request_not_installed"] += 1
                    detail = {
                        "controller": {"accepted": False, "skipped": True},
                        "paths": reroute.get("paths", {}),
                        "reroute_policy": reroute.get("policy", "external"),
                        "reroute_reason": "request is no longer installed",
                    }
                else:
                    gate_reasons = []
                    gate_evidence: dict[str, Any] = {}
                    if args.strict_reroute_gates:
                        trace_remaining = (
                            float(request["leave_time"]) - float(event["time"])
                        ) * args.time_scale
                        wall_leave = wall_start + (
                            float(request["leave_time"]) - origin
                        ) * args.time_scale
                        wall_remaining = wall_leave - time.monotonic()
                        estimated_gain = float(reroute.get("estimated_gain", 0.0))
                        old_utilization = float(reroute.get("old_utilization", 0.0))
                        new_utilization = float(
                            reroute.get("max_new_utilization", old_utilization)
                        )
                        utilization_drop = old_utilization - new_utilization
                        gate_evidence.update(
                            {
                                "trace_remaining_seconds": trace_remaining,
                                "wall_remaining_seconds": wall_remaining,
                                "estimated_gain": estimated_gain,
                                "modeled_old_utilization": old_utilization,
                                "modeled_new_utilization": new_utilization,
                                "modeled_utilization_drop": utilization_drop,
                            }
                        )
                        if not reroute_chain_valid[request_id]:
                            gate_reasons.append("prior_action_rejected")
                        if min(trace_remaining, wall_remaining) < args.reroute_min_remaining_lifetime:
                            gate_reasons.append("insufficient_remaining_lifetime")
                        if estimated_gain < args.reroute_min_estimated_gain:
                            gate_reasons.append("insufficient_estimated_gain")
                        if old_utilization < args.reroute_min_old_utilization:
                            gate_reasons.append("modeled_edge_not_hot")
                        if utilization_drop < args.reroute_min_utilization_drop:
                            gate_reasons.append("insufficient_modeled_drop")
                        previous_time = last_reroute_time.get(request_id)
                        if (
                            previous_time is not None
                            and float(event["time"]) - previous_time
                            < args.reroute_min_cooldown_seconds
                        ):
                            gate_reasons.append("request_cooldown")
                        live_status = client.status()
                        live_utilization, live_sample = live_edge_utilization(
                            live_status, profile, reroute.get("old_edge", [])
                        )
                        gate_evidence["live_old_edge_utilization"] = live_utilization
                        gate_evidence["live_port_sample"] = live_sample
                        if live_utilization is None:
                            gate_reasons.append("live_utilization_unavailable")
                        elif live_utilization < args.reroute_live_utilization_threshold:
                            gate_reasons.append("live_edge_not_hot")
                    if gate_reasons:
                        skipped_reroutes += 1
                        reroute_chain_valid[request_id] = False
                        reroute_gate_rejections.update(gate_reasons)
                        detail = {
                            "controller": {
                                "accepted": False,
                                "skipped": True,
                                "strict_gate_rejected": True,
                            },
                            "paths": reroute.get("paths", {}),
                            "reroute_policy": reroute.get("policy", "external"),
                            "reroute_gate": {
                                "accepted": False,
                                "reasons": gate_reasons,
                                "evidence": gate_evidence,
                            },
                        }
                    else:
                        old_plan = copy.deepcopy(sfc_plans[request_id])
                        old_footprint = None
                        ledger_replaced = False
                        ledger_replacement = None
                        ledger_rollback = None
                        controller_rollback = None
                        controller_attempted = False
                        controller_ms = None
                        reroute_started = time.monotonic()
                        try:
                            new_plan = build_tail_reroute_plan(
                                old_plan, reroute, request, profile
                            )
                            with bandwidth_lock:
                                bandwidth_ok, proposed_bandwidth, bandwidth_detail = (
                                    check_replacement_bandwidth(request, new_plan)
                                )
                            if not bandwidth_ok:
                                raise RuntimeError("reroute_bandwidth_infeasible")

                            if live_ledger is not None:
                                old_footprint = ResourceFootprint.from_sfc_plan(
                                    old_plan, float(request.get("bw_origin", 0.0))
                                )
                                new_footprint = ResourceFootprint.from_sfc_plan(
                                    new_plan, float(request.get("bw_origin", 0.0))
                                )
                                replace_allocation = getattr(
                                    online_planner, "replace_allocation", None
                                )
                                ledger_replacement = (
                                    replace_allocation(request_id, new_footprint)
                                    if callable(replace_allocation)
                                    else live_ledger.replace(request_id, new_footprint)
                                )
                                if not ledger_replacement.get("replaced"):
                                    raise RuntimeError(
                                        "reroute_ledger_replace_"
                                        + str(ledger_replacement.get("reason", "failed"))
                                    )
                                ledger_replaced = True

                            root_dpid = int(new_plan["multicast"]["root_dpid"])
                            controller_attempted = True
                            controller_started = time.monotonic()
                            response = client.reroute(
                                request["group_id"],
                                request["multicast_ip"],
                                new_plan["multicast"]["switch_outputs"],
                                source_dpid=root_dpid,
                                drain_seconds=args.reroute_drain_seconds,
                            )
                            controller_ms = (
                                time.monotonic() - controller_started
                            ) * 1000.0
                            with bandwidth_lock:
                                commit_replacement_bandwidth(
                                    request_id, proposed_bandwidth
                                )
                            sfc_plans[request_id] = new_plan
                            plans[request_id] = index_sfc_execution_plan(
                                request, new_plan, "runtime_tail_reroute"
                            )
                            applied_reroutes += 1
                            last_reroute_time[request_id] = float(event["time"])
                            detail = {
                                "controller": response,
                                "paths": new_plan["multicast"]["paths"],
                                "reroute_policy": reroute.get("policy", "external"),
                                "reroute_gate": {
                                    "accepted": True,
                                    "reasons": [],
                                    "evidence": gate_evidence,
                                },
                                "reroute_plan": {
                                    key: value
                                    for key, value in reroute.items()
                                    if key not in {"switch_outputs", "paths"}
                                },
                                "tail_root_dpid": root_dpid,
                                "old_tree_edges": old_plan["multicast"].get(
                                    "tree_edges", []
                                ),
                                "new_tree_edges": new_plan["multicast"].get(
                                    "tree_edges", []
                                ),
                                "bandwidth": bandwidth_detail,
                                "ledger_replacement": ledger_replacement,
                                "reroute_timing_ms": {
                                    "controller_ms": controller_ms,
                                    "total_ms": (
                                        time.monotonic() - reroute_started
                                    ) * 1000.0,
                                },
                            }
                        except Exception as exc:
                            failed_reroutes += 1
                            reroute_chain_valid[request_id] = False
                            if ledger_replaced and old_footprint is not None:
                                replace_allocation = getattr(
                                    online_planner, "replace_allocation", None
                                )
                                ledger_rollback = (
                                    replace_allocation(request_id, old_footprint)
                                    if callable(replace_allocation)
                                    else live_ledger.replace(request_id, old_footprint)
                                )
                            if controller_attempted:
                                try:
                                    controller_rollback = client.reroute(
                                        request["group_id"],
                                        request["multicast_ip"],
                                        old_plan["multicast"]["switch_outputs"],
                                        source_dpid=int(
                                            old_plan["multicast"]["root_dpid"]
                                        ),
                                        drain_seconds=0.0,
                                    )
                                except Exception as rollback_exc:
                                    controller_rollback = {
                                        "accepted": False,
                                        "reason": str(rollback_exc),
                                    }
                            detail = {
                                "controller": {
                                    "accepted": False,
                                    "reason": str(exc),
                                },
                                "paths": reroute.get("paths", {}),
                                "reroute_policy": reroute.get("policy", "external"),
                                "reroute_reason": str(exc),
                                "reroute_gate": {
                                    "accepted": True,
                                    "reasons": [],
                                    "evidence": gate_evidence,
                                },
                                "ledger_replacement": ledger_replacement,
                                "ledger_rollback": ledger_rollback,
                                "controller_rollback": controller_rollback,
                                "reroute_timing_ms": {
                                    "controller_ms": controller_ms,
                                    "total_ms": (
                                        time.monotonic() - reroute_started
                                    ) * 1000.0,
                                },
                            }
            else:
                if sfc_rejected:
                    detail = {
                        "controller": {
                            "accepted": False,
                            "skipped": True,
                            "reason": "HRL rejected the request at arrival",
                        }
                    }
                elif request_id in execution_rejected:
                    detail = {
                        "controller": {
                            "accepted": False,
                            "skipped": True,
                            "reason": "expired before deployment",
                        }
                    }
                elif request_id in sfc_plans:
                    drain_results = []
                    drain_dispatch_ms = 0.0
                    drain_wait_ms = 0.0
                    controller_delete_ms = 0.0
                    unregister_ms = 0.0
                    unregister_batch_timing = None
                    if args.vnf_launch_mode == "agent" and not args.vnf_fast_unregister:
                        # The sender has already stopped. Arm every stage in one
                        # transaction; downstream idle timers are refreshed by any
                        # remaining tail packets while the Ryu path stays installed.
                        drain_messages = []
                        drain_files = []
                        for segment in sorted(
                            sfc_plans[request_id]["segments"],
                            key=lambda value: int(value["stage"]),
                        ):
                            stage = int(segment["stage"])
                            placement = sfc_plans[request_id]["placement_by_vnf"][
                                str(stage)
                            ]
                            dc_node = int(placement["dc_node"])
                            host = str(nodes[dc_node]["host"])
                            local_files = sfc_runtime_files(
                                probe_dir, request_id, stage
                            )
                            native_drain_ack = args.vnf_agent_backend == "native"
                            drain_message = vnf_agent_drain_message(
                                vnf_agent_fifo_for_binding(
                                    vnf_agent_fifos,
                                    host,
                                    request_id,
                                    stage,
                                ),
                                request_id,
                                stage,
                                (
                                    None
                                    if native_drain_ack
                                    else to_wsl_path(local_files["drained"])
                                ),
                                args.vnf_drain_timeout_ms,
                                args.vnf_drain_idle_ms,
                            )
                            drain_messages.append(drain_message)
                            if not native_drain_ack:
                                drain_files.append(
                                    (stage, local_files["drained"])
                                )
                        drain_dispatch_started = time.monotonic()
                        drain_wait_timeout = min(
                            args.timeout,
                            max(
                                1.0,
                                args.vnf_drain_timeout_ms / 1000.0 + 0.5,
                            ),
                        )
                        if native_drain_ack:
                            drain_response = mininet_runtime_fifo_messages_wait_ack(
                                args.mininet_command_port,
                                drain_messages,
                                args.timeout,
                                ack_timeout=drain_wait_timeout,
                            )
                            drain_dispatch_ms = float(
                                drain_response.get("dispatch_ms", 0.0)
                            )
                            drain_wait_ms = float(
                                drain_response.get("ack_wait_ms", 0.0)
                            )
                            drain_results = list(drain_response.get("acks", []))
                        else:
                            mininet_runtime_fifo_messages(
                                args.mininet_command_port,
                                drain_messages,
                                args.timeout,
                            )
                            drain_dispatch_ms = (
                                time.monotonic() - drain_dispatch_started
                            ) * 1000.0
                            drain_wait_started = time.monotonic()
                            wait_for_probe_files(
                                [path for _, path in drain_files],
                                drain_wait_timeout,
                                poll_seconds=0.002,
                            )
                            drain_wait_ms = (
                                time.monotonic() - drain_wait_started
                            ) * 1000.0
                            for stage, drain_path in drain_files:
                                drain_result = read_probe_json(drain_path)
                                if drain_result is not None:
                                    drain_results.append(drain_result)
                                else:
                                    drain_results.append(
                                        {
                                            "request_id": request_id,
                                            "stage": stage,
                                            "missing_ack": True,
                                        }
                                    )
                    stop_commands = []
                    unregister_messages = []
                    for segment in sfc_plans[request_id]["segments"]:
                        stage = int(segment["stage"])
                        placement = sfc_plans[request_id]["placement_by_vnf"][str(stage)]
                        dc_node = int(placement["dc_node"])
                        host = str(nodes[dc_node]["host"])
                        local_files = sfc_runtime_files(probe_dir, request_id, stage)
                        if args.vnf_launch_mode == "agent":
                            unregister_messages.append(
                                vnf_agent_unregister_message(
                                    vnf_agent_fifo_for_binding(
                                        vnf_agent_fifos,
                                        host,
                                        request_id,
                                        stage,
                                    ),
                                    request_id,
                                    stage,
                                )
                            )
                        else:
                            command = sfc_stop_command(to_wsl_path(local_files["pid"]))
                            stop_commands.append((host, command))
                    def execute_unregister() -> tuple[float, dict[str, Any] | None, list[dict[str, Any]]]:
                        started = time.monotonic()
                        batch_timing = None
                        acks: list[dict[str, Any]] = []
                        if args.vnf_launch_mode == "agent":
                            if vnf_unregister_batcher is not None:
                                result = vnf_unregister_batcher.submit(
                                    unregister_messages
                                )
                                batch_timing = result.get("batch_timing")
                                acks = list(result.get("acks", []))
                            elif vnf_endpoint_pool is not None or args.vnf_fast_unregister:
                                result = mininet_runtime_fifo_messages_wait_ack(
                                    args.mininet_command_port,
                                    unregister_messages,
                                    args.timeout,
                                )
                                acks = list(result.get("acks", []))
                            else:
                                mininet_runtime_fifo_messages(
                                    args.mininet_command_port,
                                    unregister_messages,
                                    args.timeout,
                                )
                        else:
                            mininet_runtime_commands(
                                args.mininet_command_port, stop_commands, args.timeout
                            )
                        return (
                            (time.monotonic() - started) * 1000.0,
                            batch_timing,
                            acks,
                        )

                    unregister_acks: list[dict[str, Any]] = []
                    if args.vnf_fast_unregister:
                        (
                            unregister_ms,
                            unregister_batch_timing,
                            unregister_acks,
                        ) = execute_unregister()
                        drain_results = [
                            {
                                "request_id": ack.get("request_id"),
                                "stage": ack.get("stage"),
                                "worker_id": ack.get("worker_id"),
                                "drain_wait_ms": ack.get("drain_wait_ms"),
                                "timed_out": bool(ack.get("drain_timed_out", False)),
                                "received": ack.get("received"),
                                "forwarded": ack.get("forwarded"),
                                "protocol": "unregister_ack",
                            }
                            for ack in unregister_acks
                        ]

                    controller_delete_started = time.monotonic()
                    controller_result = client.delete_sfc(request_id)
                    controller_delete_ms = (
                        time.monotonic() - controller_delete_started
                    ) * 1000.0

                    if not args.vnf_fast_unregister:
                        (
                            unregister_ms,
                            unregister_batch_timing,
                            unregister_acks,
                        ) = execute_unregister()
                    active_vnfs.discard(request_id)
                    cleanup_phase = {
                        "drain_dispatch_ms": drain_dispatch_ms,
                        "drain_wait_ms": drain_wait_ms,
                        "controller_delete_ms": controller_delete_ms,
                        "unregister_ms": unregister_ms,
                        "unregister_batch": unregister_batch_timing,
                    }
                    with cleanup_timing_lock:
                        cleanup_drain_dispatch_ms.append(drain_dispatch_ms)
                        cleanup_drain_wait_ms.append(drain_wait_ms)
                        cleanup_controller_delete_ms.append(controller_delete_ms)
                        cleanup_unregister_ms.append(unregister_ms)
                    detail = {
                        "controller": controller_result,
                        "vnfs_stopped": True,
                        "vnf_drain": drain_results,
                        "cleanup_timing": cleanup_phase,
                    }
                else:
                    detail = {"controller": client.delete_tree(request["group_id"])}
                installed.discard(request_id)
                release_request_bandwidth(request_id)
                if vnf_endpoint_pool is not None and request_id not in active_vnfs:
                    vnf_endpoint_pool.release(request_id)
                if online_planner is not None and hasattr(online_planner, "release"):
                    online_planner.release(request_id)
            finish_event(detail)

        cleanup_timing_lock = threading.Lock()
        cleanup_queue_wait_ms: list[float] = []
        cleanup_service_ms: list[float] = []
        cleanup_drain_dispatch_ms: list[float] = []
        cleanup_drain_wait_ms: list[float] = []
        cleanup_controller_delete_ms: list[float] = []
        cleanup_unregister_ms: list[float] = []
        cleanup_peak_queue_depth = 0

        arrival_done = {
            int(request["id"]): threading.Event() for request in requests
        }

        def recover_receiver_startup_timeout(
            event: dict[str, Any], exc: ReceiverStartupTimeout
        ) -> None:
            request = event["request"]
            request_id = int(request["id"])
            cleanup_detail: dict[str, Any] = {}
            if request_id in installed:
                try:
                    if request_id in sfc_plans:
                        cleanup_detail["controller"] = client.delete_sfc(request_id)
                    else:
                        cleanup_detail["controller"] = client.delete_tree(
                            request["group_id"]
                        )
                    installed.discard(request_id)
                except Exception as cleanup_exc:
                    cleanup_detail["controller_error"] = str(cleanup_exc)

            endpoint_released = vnf_endpoint_pool is None
            if request_id in active_vnfs and request_id in sfc_plans:
                unregister_messages = []
                for segment in sfc_plans[request_id]["segments"]:
                    stage = int(segment["stage"])
                    placement = sfc_plans[request_id]["placement_by_vnf"][str(stage)]
                    host = str(nodes[int(placement["dc_node"])]["host"])
                    unregister_messages.append(
                        vnf_agent_unregister_message(
                            vnf_agent_fifo_for_binding(
                                vnf_agent_fifos,
                                host,
                                request_id,
                                stage,
                            ),
                            request_id,
                            stage,
                        )
                    )
                try:
                    if vnf_unregister_batcher is not None:
                        unregister_result = vnf_unregister_batcher.submit(
                            unregister_messages
                        )
                    else:
                        unregister_result = mininet_runtime_fifo_messages_wait_ack(
                            args.mininet_command_port,
                            unregister_messages,
                            args.timeout,
                        )
                    cleanup_detail["vnf_unregister"] = {
                        "acknowledged": True,
                        "acks": len(unregister_result.get("acks", [])),
                        "batch_timing": unregister_result.get("batch_timing"),
                    }
                    active_vnfs.discard(request_id)
                    if vnf_endpoint_pool is not None:
                        endpoint_released = vnf_endpoint_pool.release(request_id)
                except Exception as cleanup_exc:
                    cleanup_detail["vnf_unregister_error"] = str(cleanup_exc)

            release_request_bandwidth(request_id)
            reject_arrival_before_execution(
                event,
                code="receiver_startup_timeout",
                source="probe_receiver_ack",
                detail=str(exc),
                extra={
                    "runtime_failure_cleanup": cleanup_detail,
                    "vnf_endpoint_released": endpoint_released,
                },
            )

        def process_event(event: dict[str, Any]) -> None:
            if event["type"] == "migration_scan":
                scan_time = float(event["time"])
                target = wall_start + (scan_time - origin) * args.time_scale
                wait_until(target)
                migratable_plans = {
                    request_id: plan
                    for request_id, plan in sfc_plans.items()
                    if request_id in installed and request_id in active_vnfs
                }
                entries = predictive_migration_monitor.scan(
                    online_planner.ledger.snapshot(),
                    migratable_plans,
                    request_by_id,
                    scan_time,
                )
                if not entries:
                    return
                decisions = migration_online_planner.plan_batch(
                    entries,
                    online_planner.ledger.snapshot(),
                    scan_time,
                )
                migration_events = []
                for entry, decision in zip(entries, decisions):
                    if not decision["accepted"]:
                        continue
                    request = entry["request"]
                    request_id = int(request["id"])
                    stage = int(entry["stage"])
                    row = {
                        "time": scan_time,
                        "request_id": request_id,
                        "stage": stage,
                        "target_dc": int(decision["target_dc"]),
                        "policy": "predictive_migration_wqmix",
                        "trigger": "ewma_predicted_overload",
                        "target_rankings": decision["rankings"],
                        "migration_decision_ms": decision["decision_ms"],
                    }
                    selected_migrations.append(row)
                    candidate_plans_by_target = {
                        int(candidate["target_dc"]): candidate["target_plan"]
                        for candidate in decision.get("ranked_candidates", [])
                        if candidate.get("target_dc") is not None
                        and candidate.get("target_plan") is not None
                    }
                    if decision.get("target_plan") is not None:
                        candidate_plans_by_target.setdefault(
                            int(decision["target_dc"]), decision["target_plan"]
                        )
                    migration_events.append({
                        "time": scan_time,
                        "type": "migration",
                        "request": request,
                        "migration": row,
                        "_target_plan": decision.get("target_plan"),
                        "_candidate_plans_by_target": candidate_plans_by_target,
                        "migration_task": MigrationTask.from_dict(decision["task"]),
                    })
                if len(migration_events) == 1 or args.online_migration_max_inflight == 1:
                    for migration_event in migration_events:
                        process_event(migration_event)
                elif migration_events:
                    event_by_task = {
                        event["migration_task"].task_id: event
                        for event in migration_events
                    }
                    waves = migration_execution_waves(
                        [event["migration_task"] for event in migration_events],
                        max_inflight=args.online_migration_max_inflight,
                    )
                    for wave in waves:
                        wave_events = [event_by_task[task.task_id] for task in wave]
                        if len(wave_events) == 1:
                            process_event(wave_events[0])
                            continue
                        with ThreadPoolExecutor(
                            max_workers=len(wave_events),
                            thread_name_prefix="vnf-migration",
                        ) as executor:
                            futures = [
                                executor.submit(process_event, migration_event)
                                for migration_event in wave_events
                            ]
                            for future in futures:
                                future.result()
                return
            request_id = int(event["request"]["id"])
            if event["type"] == "arrive":
                with request_locks[request_id]:
                    try:
                        try:
                            process_event_unlocked(event)
                        except ReceiverStartupTimeout as exc:
                            recover_receiver_startup_timeout(event, exc)
                    finally:
                        arrival_done[request_id].set()
                return
            arrival_done[request_id].wait()
            with request_locks[request_id]:
                process_event_unlocked(event)

        if (
            args.deployment_workers == 1
            and args.deployment_max_outstanding == 0
            and args.planning_max_outstanding == 0
            and args.execution_max_outstanding == 0
        ):
            for event in events:
                target = wall_start + (
                    float(event["time"]) - origin
                ) * args.time_scale
                wait_until(target)
                prepare_online_plan(event)
                process_event(event)
        else:
            deployment_queue: queue.PriorityQueue = queue.PriorityQueue()
            cleanup_queue: queue.PriorityQueue = queue.PriorityQueue()
            migration_queue: queue.PriorityQueue = queue.PriorityQueue()
            planning_queue: queue.Queue | None = (
                queue.Queue() if online_planner is not None else None
            )
            worker_errors = []
            worker_errors_lock = threading.Lock()

            def release_gate_slot(
                queued_event: dict[str, Any],
                key: str,
                gate: StageCapacityGate,
            ) -> None:
                if queued_event.pop(key, False):
                    gate.release()

            def event_worker(work_queue: queue.PriorityQueue) -> None:
                while True:
                    _, _, queued_event = work_queue.get()
                    cleanup_started = None
                    try:
                        if queued_event is None:
                            return
                        if queued_event["type"] == "leave":
                            cleanup_started = time.monotonic()
                            enqueued = float(
                                queued_event.pop(
                                    "_cleanup_enqueued_monotonic",
                                    cleanup_started,
                                )
                            )
                            with cleanup_timing_lock:
                                cleanup_queue_wait_ms.append(
                                    max(0.0, cleanup_started - enqueued) * 1000.0
                                )
                        process_event(queued_event)
                    except BaseException as exc:
                        if queued_event is not None:
                            failed_request_id = int(queued_event["request"]["id"])
                            release_request_bandwidth(failed_request_id)
                            if vnf_endpoint_pool is not None:
                                vnf_endpoint_pool.release(failed_request_id)
                        with worker_errors_lock:
                            worker_errors.append(exc)
                    finally:
                        if cleanup_started is not None:
                            with cleanup_timing_lock:
                                cleanup_service_ms.append(
                                    max(0.0, time.monotonic() - cleanup_started)
                                    * 1000.0
                                )
                        if queued_event is not None:
                            release_gate_slot(
                                queued_event,
                                "_execution_capacity_slot",
                                execution_capacity,
                            )
                            release_gate_slot(
                                queued_event,
                                "_total_capacity_slot",
                                total_capacity,
                            )
                        work_queue.task_done()

            def planning_worker() -> None:
                assert planning_queue is not None
                pending_item = None
                while True:
                    if pending_item is None:
                        sequence, queued_event = planning_queue.get()
                    else:
                        sequence, queued_event = pending_item
                        pending_item = None
                    batch_items = [(sequence, queued_event)]
                    try:
                        if queued_event is None:
                            return
                        microbatch_seconds = float(
                            getattr(online_planner, "microbatch_seconds", 0.0)
                        )
                        if microbatch_seconds > 0.0:
                            first_target = wall_start + (
                                float(queued_event["time"]) - origin
                            ) * args.time_scale
                            batch_close = first_target + microbatch_seconds
                            wait_until(batch_close)
                            while len(batch_items) < int(
                                getattr(online_planner, "max_agents", 1)
                            ):
                                try:
                                    next_item = planning_queue.get_nowait()
                                except queue.Empty:
                                    break
                                next_sequence, next_event = next_item
                                if next_event is None:
                                    pending_item = next_item
                                    break
                                next_target = wall_start + (
                                    float(next_event["time"]) - origin
                                ) * args.time_scale
                                if next_target > batch_close + 1e-9:
                                    pending_item = next_item
                                    break
                                batch_items.append(next_item)
                        batch_events = [item[1] for item in batch_items]
                        prepare_online_plans(batch_events)
                        for batch_sequence, batch_event in batch_items:
                            release_gate_slot(
                                batch_event,
                                "_planning_capacity_slot",
                                planning_capacity,
                            )
                            if not execution_capacity.try_acquire():
                                request_id = int(batch_event["request"]["id"])
                                reject_arrival_before_execution(
                                    batch_event,
                                    code="execution_capacity_rejected",
                                    source="bounded_execution",
                                    detail=(
                                        f"request {request_id} rejected after planning: "
                                        f"execution phase limit "
                                        f"{args.execution_max_outstanding} is occupied"
                                    ),
                                    extra={
                                        "execution_capacity": (
                                            execution_capacity.snapshot()
                                        )
                                    },
                                )
                                arrival_done[request_id].set()
                                release_gate_slot(
                                    batch_event,
                                    "_total_capacity_slot",
                                    total_capacity,
                                )
                                continue
                            batch_event["_execution_capacity_slot"] = True
                            deadline = float(batch_event["request"]["leave_time"])
                            deployment_queue.put(
                                (deadline, batch_sequence, batch_event)
                            )
                    except BaseException as exc:
                        for _, failed_event in batch_items:
                            request_id = int(failed_event["request"]["id"])
                            execution_rejected.add(request_id)
                            if online_planner is not None and hasattr(
                                online_planner, "release"
                            ):
                                online_planner.release(request_id)
                            arrival_done[request_id].set()
                            release_gate_slot(
                                failed_event,
                                "_planning_capacity_slot",
                                planning_capacity,
                            )
                            release_gate_slot(
                                failed_event,
                                "_total_capacity_slot",
                                total_capacity,
                            )
                        with worker_errors_lock:
                            worker_errors.append(exc)
                    finally:
                        for _ in batch_items:
                            planning_queue.task_done()

            deployment_threads = [
                threading.Thread(
                    target=event_worker,
                    args=(deployment_queue,),
                    name=f"deployment-{index + 1}",
                    daemon=True,
                )
                for index in range(args.deployment_workers)
            ]
            cleanup_threads = [
                threading.Thread(
                    target=event_worker,
                    args=(cleanup_queue,),
                    name=f"cleanup-{index + 1}",
                    daemon=True,
                )
                for index in range(args.cleanup_workers)
            ]
            migration_thread = threading.Thread(
                target=event_worker,
                args=(migration_queue,),
                name="migration-control-1",
                daemon=True,
            )
            for thread in deployment_threads + cleanup_threads + [migration_thread]:
                thread.start()
            planner_thread = None
            if planning_queue is not None:
                planner_thread = threading.Thread(
                    target=planning_worker,
                    name="online-hrl-planner",
                    daemon=True,
                )
                planner_thread.start()

            for sequence, event in enumerate(events):
                target = wall_start + (
                    float(event["time"]) - origin
                ) * args.time_scale
                wait_until(target)
                if event["type"] == "leave":
                    # Native senders stop against the same absolute trace clock.
                    # Release admission capacity now; flow/VNF cleanup may lag in
                    # its own worker without creating false resource occupancy.
                    if release_request_bandwidth(int(event["request"]["id"])):
                        scheduled_bandwidth_releases += 1
                    if online_planner is not None and hasattr(online_planner, "release"):
                        online_planner.release(int(event["request"]["id"]))
                    event["_cleanup_enqueued_monotonic"] = time.monotonic()
                    cleanup_queue.put((float(event["time"]), sequence, event))
                    with cleanup_timing_lock:
                        cleanup_peak_queue_depth = max(
                            cleanup_peak_queue_depth,
                            cleanup_queue.qsize(),
                        )
                elif event["type"] in {"migration", "migration_scan"}:
                    # Migration has its own control worker so a slow
                    # make-before-break transaction cannot stall SFC setup.
                    migration_queue.put((float(event["time"]), sequence, event))
                elif event["type"] != "arrive":
                    # Runtime control operations must observe the installed SFC
                    # and serialize with that request's arrival/leave operations.
                    deployment_queue.put((float(event["time"]), sequence, event))
                else:
                    request_id = int(event["request"]["id"])
                    precomputed_sfc_plan = sfc_plans.get(request_id)
                    if (
                        precomputed_sfc_plan is not None
                        and not precomputed_sfc_plan.get("accepted", True)
                    ):
                        process_event(event)
                        continue
                    enqueue_admission = deadline_admission(
                        event["request"], phase="enqueue"
                    )
                    if enqueue_admission["slack_ms"] <= 0.0:
                        reject_arrival_before_execution(
                            event,
                            code="expired_before_deployment",
                            source="deadline_admission",
                            detail=(
                                f"request {request_id} rejected at enqueue with "
                                f"{enqueue_admission['slack_ms']:.3f}ms slack"
                            ),
                            extra={"deployment_admission": enqueue_admission},
                        )
                        arrival_done[request_id].set()
                        continue
                    if not total_capacity.try_acquire():
                        reject_arrival_before_execution(
                            event,
                            code="deployment_capacity_rejected",
                            source="bounded_admission",
                            detail=(
                                f"request {request_id} rejected immediately: "
                                f"{args.deployment_max_outstanding} deployment slots "
                                "are already occupied"
                            ),
                            extra={
                                "deployment_max_outstanding": args.deployment_max_outstanding,
                                "total_capacity": total_capacity.snapshot(),
                            },
                        )
                        arrival_done[request_id].set()
                        continue
                    event["_total_capacity_slot"] = True
                    event["_enqueue_admission"] = enqueue_admission
                    if planning_queue is not None:
                        if not planning_capacity.try_acquire():
                            reject_arrival_before_execution(
                                event,
                                code="planning_capacity_rejected",
                                source="bounded_planning",
                                detail=(
                                    f"request {request_id} rejected immediately: "
                                    f"planning phase limit "
                                    f"{args.planning_max_outstanding} is occupied"
                                ),
                                extra={
                                    "planning_capacity": planning_capacity.snapshot()
                                },
                            )
                            arrival_done[request_id].set()
                            release_gate_slot(
                                event,
                                "_total_capacity_slot",
                                total_capacity,
                            )
                            continue
                        event["_planning_capacity_slot"] = True
                        planning_queue.put((sequence, event))
                        continue
                    if not execution_capacity.try_acquire():
                        reject_arrival_before_execution(
                            event,
                            code="execution_capacity_rejected",
                            source="bounded_execution",
                            detail=(
                                f"request {request_id} rejected immediately: "
                                f"execution phase limit "
                                f"{args.execution_max_outstanding} is occupied"
                            ),
                            extra={
                                "execution_capacity": execution_capacity.snapshot()
                            },
                        )
                        arrival_done[request_id].set()
                        release_gate_slot(
                            event, "_total_capacity_slot", total_capacity
                        )
                        continue
                    event["_execution_capacity_slot"] = True
                    # Earliest deadline first among requests that have arrived.
                    deadline = float(event["request"]["leave_time"])
                    deployment_queue.put((deadline, sequence, event))

            if planning_queue is not None:
                planning_queue.join()
                planning_queue.put((len(events), None))
                assert planner_thread is not None
                planner_thread.join()
            deployment_queue.join()
            migration_queue.join()
            cleanup_queue.join()
            for sequence, _ in enumerate(deployment_threads):
                deployment_queue.put((math.inf, sequence, None))
            for sequence, _ in enumerate(cleanup_threads):
                cleanup_queue.put((math.inf, sequence, None))
            migration_queue.put((math.inf, 0, None))
            for thread in deployment_threads + cleanup_threads + [migration_thread]:
                thread.join()
            if worker_errors:
                raise worker_errors[0]
            event_rows.sort(
                key=lambda row: (
                    float(row["trace_time"]),
                    event_priority[str(row["type"])],
                    int(row["request_id"]),
                )
            )

        wait_for_probe_files(
            [
                path
                for request_id, _, path in probe_paths
                if request_id not in execution_rejected
            ]
            + [
                path
                for request_id, path in sender_paths
                if request_id not in execution_rejected
            ],
            args.probe_result_timeout,
        )
        if args.vnf_agent_backend != "native":
            wait_for_probe_files(
                [path for _, _, _, path in vnf_stats_paths],
                args.probe_result_timeout,
            )
        receiver_results = list(precomputed_receiver_results)
        for request_id, destination, path in probe_paths:
            if request_id in execution_rejected:
                continue
            receiver_results.append(
                {
                    "request_id": request_id,
                    "destination_dpid": destination,
                    "result": read_probe_json(path),
                }
            )
        sender_results = list(precomputed_sender_results) + [
            {"request_id": request_id, "result": read_probe_json(path)}
            for request_id, path in sender_paths
            if request_id not in execution_rejected
        ]
        if args.vnf_agent_backend == "native":
            native_drain_results = {}
            for event_row in event_rows:
                for drain_result in event_row.get("vnf_drain", []):
                    key = (
                        int(drain_result.get("request_id", -1)),
                        int(drain_result.get("stage", -1)),
                    )
                    native_drain_results[key] = {
                        **drain_result,
                        "runtime": "native_vnf_agent",
                    }
            vnf_results = [
                {
                    "request_id": request_id,
                    "stage": stage,
                    "dc_node": dc_node,
                    "result": native_drain_results.get(
                        (request_id, stage),
                        {
                            "request_id": request_id,
                            "stage": stage,
                            "received": 0,
                            "forwarded": 0,
                            "runtime": "native_vnf_agent",
                            "measurement_status": "no_drain_result",
                        },
                    ),
                }
                for request_id, stage, dc_node, _ in vnf_stats_paths
            ]
        else:
            vnf_results = [
                {
                    "request_id": request_id,
                    "stage": stage,
                    "dc_node": dc_node,
                    "result": read_probe_json(path),
                }
                for request_id, stage, dc_node, path in vnf_stats_paths
            ]
        diagnostics_after = mininet_runtime_diagnostics(
            args.mininet_command_port, args.timeout
        )
        measured = [row["result"] for row in receiver_results if row["result"]]
        measured_senders = [row["result"] for row in sender_results if row["result"]]
        sender_by_request = {}
        for row in sender_results:
            sender_result = row.get("result")
            if not sender_result:
                continue
            planned_packets = int(sender_result.get("planned_packets", 0))
            sent_packets = int(sender_result.get("sent_packets", 0))
            completion_ratio = (
                sent_packets / planned_packets if planned_packets > 0 else 0.0
            )
            compliance_met = bool(
                sender_result.get("status") == "completed"
                and planned_packets > 0
                and completion_ratio + 1e-12
                >= args.offered_load_compliance_ratio
            )
            sender_result["offered_load_completion_ratio"] = completion_ratio
            sender_result["offered_load_compliance_ratio"] = (
                args.offered_load_compliance_ratio
            )
            sender_result["offered_load_compliance_met"] = compliance_met
            sender_by_request[int(row["request_id"])] = sender_result
        ready_waits = sorted(
            float(row.get("ready_wait_seconds", 0.0)) for row in measured_senders
        )
        deployment_timings = [
            row["deployment_timing"]
            for row in event_rows
            if row.get("type") == "arrive"
            and isinstance(row.get("deployment_timing"), dict)
        ]
        receiver_sla_by_request = defaultdict(int)
        for row in receiver_results:
            if row["result"] and bool(row["result"].get("sla_met")):
                receiver_sla_by_request[int(row["request_id"])] += 1
        sla_met_requests = sum(
            bool(
                sender_by_request.get(int(request["id"]), {}).get(
                    "offered_load_compliance_met", False
                )
            )
            and receiver_sla_by_request[int(request["id"])]
            == len(request["destination_dpids"])
            for request in requests
        )
        total_planned_packets = sum(
            int(row.get("planned_packets", 0)) for row in measured_senders
        )
        total_sent_packets = sum(
            int(row.get("sent_packets", 0)) for row in measured_senders
        )
        delays = [
            float(row["mean_delay_ms"])
            for row in measured
            if row.get("mean_delay_ms") is not None
        ]
        planned_vnf_stages = sum(
            len(sfc_plans[int(request["id"])]["segments"])
            for request in requests
            if int(request["id"]) in sfc_plans
            and sfc_plans[int(request["id"])].get("accepted", True)
        )
        expected_vnf_stages = len(vnf_stats_paths)
        measured_vnf_results = [
            row
            for row in vnf_results
            if row["result"] is not None
        ]
        valid_vnf_results = [
            row
            for row in measured_vnf_results
            if int(row["result"].get("received") or 0) > 0
            and int(row["result"].get("forwarded") or 0) > 0
        ]
        vnf_results_by_request = defaultdict(list)
        for row in vnf_results:
            vnf_results_by_request[int(row["request_id"])].append(row)
        sfc_traversed_requests = sum(
            len(rows) == len(sfc_plans[request_id]["segments"])
            and all(
                row["result"] is not None
                and int(row["result"].get("received") or 0) > 0
                for row in rows
            )
            for request_id, rows in vnf_results_by_request.items()
        )
        arrival_scheduler_lags = [
            float(row["scheduler_lag_ms"])
            for row in event_rows
            if row.get("type") == "arrive"
        ]
        online_planning_timings = [
            row["online_planning"]
            for row in online_plan_records
            if isinstance(row.get("online_planning"), dict)
        ]
        final_live_ledger = getattr(online_planner, "ledger", None)
        ledger_integrity = (
            final_live_ledger.integrity_report()
            if final_live_ledger is not None
            else None
        )
        bandwidth_mirror_integrity = {
            "fully_released": (
                not request_link_reservations
                and all(abs(float(value)) <= 1e-9 for value in link_reserved_mbps.values())
            ),
            "active_request_reservations": len(request_link_reservations),
            "residual_edges": {
                f"{edge[0]}->{edge[1]}": float(value)
                for edge, value in link_reserved_mbps.items()
                if abs(float(value)) > 1e-9
            },
        }
        valid = (
            len(measured) == len(receiver_results)
            and len(measured_senders) == len(sender_results)
            and (
                not sfc_plans
                or len(measured_vnf_results) == expected_vnf_stages
            )
            and (ledger_integrity is None or ledger_integrity["fully_released"])
            and bandwidth_mirror_integrity["fully_released"]
        )
        dynamic_clock_offsets = [
            int(row["offset_ns"]) for row in dynamic_clock_sync_observations
        ]
        dynamic_clock_sync_summary = {
            "strategy": "per_dynamic_request",
            "samples": len(dynamic_clock_offsets),
            "initial_offset_ns": int(clock_sync.get("offset_ns", 0)),
            "first_offset_ns": (
                dynamic_clock_offsets[0] if dynamic_clock_offsets else None
            ),
            "last_offset_ns": (
                dynamic_clock_offsets[-1] if dynamic_clock_offsets else None
            ),
            "minimum_offset_ns": (
                min(dynamic_clock_offsets) if dynamic_clock_offsets else None
            ),
            "maximum_offset_ns": (
                max(dynamic_clock_offsets) if dynamic_clock_offsets else None
            ),
            "offset_range_ms": (
                (max(dynamic_clock_offsets) - min(dynamic_clock_offsets))
                / 1_000_000.0
                if dynamic_clock_offsets
                else None
            ),
            "maximum_shift_from_initial_ms": (
                max(
                    abs(value - int(clock_sync.get("offset_ns", 0)))
                    for value in dynamic_clock_offsets
                )
                / 1_000_000.0
                if dynamic_clock_offsets
                else None
            ),
        }
        result = {
            "valid": valid,
            "dry_run": False,
            "topology": profile.get("name", profile_path.stem),
            "requests": len(requests),
            "online_hrl": (
                online_planner.metadata() if online_hrl_enabled else None
            ),
            "online_wqmix": (
                online_planner.metadata() if online_wqmix_enabled else None
            ),
            "online_migration_wqmix": (
                {
                    **migration_online_planner.metadata(),
                    "monitor": (
                        predictive_migration_monitor.metadata()
                        if predictive_migration_monitor is not None
                        else None
                    ),
                }
                if migration_online_planner is not None
                else None
            ),
            "multiagent_orchestration": {
                "architecture": "brain_module_skill",
                "deployment": (
                    online_planner.metadata().get("orchestration")
                    if online_hrl_enabled
                    and isinstance(online_planner.metadata(), dict)
                    else None
                ),
                "reconfiguration": runtime_reconfiguration.metadata(),
            },
            "resource_integrity": {
                "atomic_ledger": ledger_integrity,
                "bandwidth_mirror": bandwidth_mirror_integrity,
            },
            "online_plans": online_plan_records,
            "events": event_rows,
            "receiver_results": receiver_results,
            "sender_results": sender_results,
            "vnf_results": vnf_results,
            "network_diagnostics": {
                "before": diagnostics_before,
                "after": diagnostics_after,
                "delta": network_diagnostic_delta(
                    diagnostics_before, diagnostics_after
                ),
            },
            "probe_directory": str(probe_dir),
            "probe_timing": {
                "runtime_clock": {
                    "name": _RUNTIME_CLOCK,
                    "resolution_seconds": (
                        _PERF_COUNTER_RESOLUTION
                        if _RUNTIME_CLOCK == "perf_counter"
                        else _SYSTEM_MONOTONIC_RESOLUTION
                    ),
                    "system_monotonic_resolution_seconds": (
                        _SYSTEM_MONOTONIC_RESOLUTION
                    ),
                },
                "receiver_ready_budget_seconds": args.receiver_ready_seconds,
                "tree_warmup_seconds": args.probe_tree_warmup_seconds,
                "sender_stop_margin_seconds": args.sender_stop_margin_seconds,
                "minimum_traffic_seconds": args.minimum_traffic_seconds,
                "handshake_directory": handshake_dir_wsl,
                "mininet_command_host": _MININET_RUNTIME_HOST,
                "mininet_command_port": args.mininet_command_port,
                "controller_rest_url": controller_rest_url,
                "probe_launch_mode": args.probe_launch_mode,
                "probe_sender_launch_mode": args.probe_sender_launch_mode,
                "probe_sender_prelaunch": bool(args.probe_sender_prelaunch),
                "probe_sender_backend": args.probe_sender_backend,
                "probe_receiver_backend": args.probe_receiver_backend,
                "native_sender_path": native_sender_wsl,
                "native_receiver_path": native_receiver_wsl,
                "native_sender_realtime_priority": (
                    args.probe_native_realtime_priority
                ),
                "probes_prescheduled": prescheduled_probes,
                "separate_sender_receiver_agents": prescheduled_probes,
                "agent_sender_workers": agent_sender_workers,
                "agent_receiver_workers": agent_receiver_workers,
                "agent_worker_plan": agent_worker_plan,
                "schedule_origin_file": (
                    schedule_origin_wsl if prescheduled_probes else None
                ),
                "probe_receive_buffer_bytes": args.probe_receive_buffer_bytes,
                "payload_bytes": args.payload_bytes,
                "packet_overhead_bytes": args.packet_overhead_bytes,
                "bandwidth_rate_model": (
                    "bw_origin_is_link_rate_including_packet_overhead"
                    if args.packets_per_second <= 0.0
                    else "fixed_packets_per_second_override"
                ),
                "packets_per_second_override": args.packets_per_second,
                "offered_load_compliance_ratio": (
                    args.offered_load_compliance_ratio
                ),
                "sender_max_catch_up_packets": args.sender_max_catch_up_packets,
                "sla_predictor_contract_validation": (
                    sla_predictor_contract_validation
                ),
                "windows_to_wsl_clock_sync": clock_sync,
                "dynamic_windows_to_wsl_clock_sync": (
                    dynamic_clock_sync_summary
                ),
                "vnf_launch_mode": args.vnf_launch_mode,
                "vnf_ready_protocol": args.vnf_ready_protocol,
                "vnf_agent_workers_per_dc": (
                    args.vnf_agent_workers
                    if args.vnf_agent_backend == "python"
                    else args.vnf_agent_native_shards
                ),
                "vnf_agent_python_worker_setting": args.vnf_agent_workers,
                "vnf_agent_processes_per_dc": (
                    args.vnf_agent_native_shards
                    if args.vnf_agent_backend == "native"
                    else 1
                ),
                "vnf_agent_backend": args.vnf_agent_backend,
                "vnf_agent_initial_workers_per_dc": args.vnf_agent_initial_workers,
                "vnf_agent_bindings_per_worker": args.vnf_agent_bindings_per_worker,
                "vnf_agent_prewarm_workers": args.vnf_agent_prewarm_workers,
                "vnf_endpoint_pool": (
                    vnf_endpoint_pool.snapshot()
                    if vnf_endpoint_pool is not None
                    else {
                        "enabled": False,
                        "port_base": args.vnf_agent_prebound_port_base,
                        "ports_per_dc": 0,
                    }
                ),
                "vnf_drain_timeout_ms": args.vnf_drain_timeout_ms,
                "vnf_drain_idle_ms": args.vnf_drain_idle_ms,
                "vnf_fast_unregister": bool(args.vnf_fast_unregister),
                "vnf_agent_realtime_priority": (
                    args.vnf_agent_realtime_priority
                ),
                "vnf_agent_packet_batch": args.vnf_agent_packet_batch,
                "vnf_agent_q0_packet_batch": args.vnf_agent_q0_packet_batch,
                "vnf_agent_dscp_scheduling": bool(
                    args.vnf_agent_dscp_scheduling
                ),
                "controller_scheduler": controller_scheduler,
                "cpu_affinity": {
                    "controller": controller_affinity,
                    "ovs": ovs_affinity,
                    "mininet_cpu_set": args.mininet_cpu_set,
                    "vnf_agent_cpu_set": args.vnf_agent_cpu_set,
                    "probe_agent_cpu_set": args.probe_agent_cpu_set,
                    "probe_agent_realtime_priority": args.probe_agent_realtime_priority,
                },
                "vnf_control_transport": (
                    (
                        "direct_fifo_unix_ack"
                        if args.vnf_ready_protocol == "ack"
                        else "direct_fifo_ready_file"
                    )
                    if args.vnf_launch_mode == "agent"
                    else "host_command_ready_file"
                ),
                "deployment_workers": args.deployment_workers,
                "deployment_pipeline": {
                    "enabled": bool(args.parallel_deployment_pipeline),
                    "stages": [
                        "planning_atomic_commit",
                        "neighbor_and_vnf_prepare",
                        "ryu_batch_commit",
                        "receiver_ready",
                        "sender_dispatch",
                    ],
                    "neighbor_workers": (
                        min(4, int(args.deployment_workers))
                        if args.parallel_deployment_pipeline
                        else 0
                    ),
                    "neighbor_vnf_overlap": bool(
                        args.parallel_deployment_pipeline
                    ),
                    "authoritative_resource_ledger": (
                        "online_wqmix_atomic_ledger"
                        if online_wqmix_enabled
                        else "runtime_bandwidth_ledger"
                    ),
                    "vnf_register_batch_ms": args.vnf_register_batch_ms,
                    "ryu_commit_batch_ms": args.ryu_commit_batch_ms,
                },
                "vnf_registration_concurrency": (
                    args.vnf_registration_concurrency
                    if args.vnf_registration_concurrency > 0
                    else args.deployment_workers
                ),
                "vnf_register_batch": (
                    vnf_register_batcher.snapshot()
                    if vnf_register_batcher is not None
                    else {
                        "enabled": False,
                        "window_ms": args.vnf_register_batch_ms,
                        "max_batch_size": args.vnf_register_batch_size,
                    }
                ),
                "vnf_unregister_batch": (
                    vnf_unregister_batcher.snapshot()
                    if vnf_unregister_batcher is not None
                    else {
                        "enabled": False,
                        "window_ms": args.vnf_unregister_batch_ms,
                        "max_batch_size": args.vnf_unregister_batch_size,
                    }
                ),
                "deployment_max_outstanding": args.deployment_max_outstanding,
                "planning_max_outstanding": args.planning_max_outstanding,
                "execution_max_outstanding": args.execution_max_outstanding,
                "stage_capacity": {
                    "total": total_capacity.snapshot(),
                    "planning": planning_capacity.snapshot(),
                    "execution": execution_capacity.snapshot(),
                },
                "deployment_max_queue_wait_ms": args.deployment_max_queue_wait_ms,
                "cleanup_workers": args.cleanup_workers,
                "cleanup_timing": {
                    "peak_queue_depth": cleanup_peak_queue_depth,
                    "queue_wait_ms": summarize_milliseconds(
                        cleanup_queue_wait_ms
                    ),
                    "service_ms": summarize_milliseconds(cleanup_service_ms),
                    "drain_dispatch_ms": summarize_milliseconds(
                        cleanup_drain_dispatch_ms
                    ),
                    "drain_wait_ms": summarize_milliseconds(
                        cleanup_drain_wait_ms
                    ),
                    "controller_delete_ms": summarize_milliseconds(
                        cleanup_controller_delete_ms
                    ),
                    "unregister_ms": summarize_milliseconds(
                        cleanup_unregister_ms
                    ),
                },
                "deployment_scheduler": (
                    "edf"
                    if (
                        args.deployment_workers > 1
                        or args.deployment_max_outstanding > 0
                        or args.planning_max_outstanding > 0
                        or args.execution_max_outstanding > 0
                    )
                    else "trace_order"
                ),
                "planning_mode": (
                    "online_wqmix"
                    if online_wqmix_enabled
                    else "online_hrl"
                    if online_hrl_enabled
                    else "offline_sfc_candidates"
                    if sfc_candidate_plans
                    else "precomputed"
                ),
                "sfc_candidate_selection": {
                    "enabled": bool(sfc_candidate_plans),
                    "source": args.sfc_candidate_plans,
                    "policy": args.sfc_candidate_selection,
                    "top_k": args.sfc_candidate_top_k,
                    "requests": len(sfc_candidate_plans),
                    "fallbacks": candidate_selection_fallbacks,
                    "attempts": candidate_selection_attempts,
                    "no_feasible_requests": candidate_selection_no_feasible,
                    "scheduled_bandwidth_releases": scheduled_bandwidth_releases,
                },
                "deployment_admission": setup_estimator.snapshot(),
                "deployment_bandwidth_utilization_limit": (
                    args.deployment_bandwidth_utilization_limit
                ),
                "deployment_bandwidth_admission": {
                    "physical_links": len(profile.get("edges", [])),
                    "directed_link_capacities": len(link_capacities),
                    "directional_capacity_mbps": sum(link_capacities.values()),
                    "rejected_requests": link_admission_rejected,
                },
                "sfc_barrier_mode": args.sfc_barrier_mode,
                "ryu_commit_batch_ms": args.ryu_commit_batch_ms,
                "ryu_commit_batch_size": args.ryu_commit_batch_size,
                "ryu_commit_batch": (
                    ryu_batch_committer.snapshot()
                    if ryu_batch_committer is not None
                    else {
                        "enabled": False,
                        "window_ms": args.ryu_commit_batch_ms,
                        "max_batch_size": args.ryu_commit_batch_size,
                    }
                ),
                "ryu_batch_sender_stagger_ms": (
                    args.ryu_batch_sender_stagger_ms
                ),
                "mininet_qdisc": args.mininet_qdisc,
                "mininet_max_queue_size": args.mininet_max_queue_size,
                "reroute_drain_seconds": args.reroute_drain_seconds,
                "vnf_migration": {
                    "events_file": args.vnf_migration_events,
                    "hotspot_events_file": args.vnf_impairment_events,
                    "migration_wqmix_checkpoint": (
                        args.online_migration_wqmix_checkpoint
                    ),
                    "max_agents": args.online_migration_max_agents,
                    "max_inflight": args.online_migration_max_inflight,
                    "predictive_scan_seconds": (
                        args.predictive_migration_scan_seconds
                    ),
                    "predictive_scans_scheduled": selected_predictive_scans,
                    "drain_ms": args.vnf_migration_drain_ms,
                    "idle_ms": args.vnf_migration_idle_ms,
                    "records": migration_records,
                },
                "strict_reroute_gates": args.strict_reroute_gates,
                "reroute_gate_config": {
                    "min_remaining_lifetime": args.reroute_min_remaining_lifetime,
                    "min_estimated_gain": args.reroute_min_estimated_gain,
                    "min_old_utilization": args.reroute_min_old_utilization,
                    "min_utilization_drop": args.reroute_min_utilization_drop,
                    "min_cooldown_seconds": args.reroute_min_cooldown_seconds,
                    "live_utilization_threshold": args.reroute_live_utilization_threshold,
                },
            },
            "measurement_summary": {
                "expected_receivers": len(receiver_results),
                "measured_receivers": len(measured),
                "measured_senders": len(measured_senders),
                "sender_start_failures": sum(
                    row.get("status") != "completed" for row in measured_senders
                ),
                "traffic_started_requests": sum(
                    row.get("status") == "completed" for row in measured_senders
                ),
                "offered_load_compliant_requests": sum(
                    bool(row.get("offered_load_compliance_met"))
                    for row in measured_senders
                ),
                "offered_load_noncompliant_started_requests": sum(
                    row.get("status") == "completed"
                    and not bool(row.get("offered_load_compliance_met"))
                    for row in measured_senders
                ),
                "planned_sender_packets": total_planned_packets,
                "sent_sender_packets": total_sent_packets,
                "offered_load_completion_ratio": (
                    total_sent_packets / total_planned_packets
                    if total_planned_packets > 0
                    else 0.0
                ),
                "hrl_rejected_requests": sum(
                    row.get("probe_status") == "hrl_rejected"
                    for row in event_rows
                ),
                "planner_rejected_requests": sum(
                    row.get("probe_status") == "hrl_rejected"
                    for row in event_rows
                ),
                "expired_before_deployment_requests": sum(
                    row.get("probe_status") == "expired_before_deployment"
                    for row in event_rows
                ),
                "deployment_capacity_rejected_requests": sum(
                    row.get("probe_status") == "deployment_capacity_rejected"
                    for row in event_rows
                ),
                "planning_capacity_rejected_requests": sum(
                    row.get("probe_status") == "planning_capacity_rejected"
                    for row in event_rows
                ),
                "execution_capacity_rejected_requests": sum(
                    row.get("probe_status") == "execution_capacity_rejected"
                    for row in event_rows
                ),
                "deployment_queue_wait_rejected_requests": sum(
                    row.get("probe_status") == "deployment_queue_wait_rejected"
                    for row in event_rows
                ),
                "bandwidth_admission_rejected_requests": sum(
                    row.get("probe_status") == "physical_link_bandwidth_admission"
                    for row in event_rows
                ),
                "sfc_candidate_infeasible_requests": sum(
                    row.get("probe_status") == "sfc_candidate_infeasible"
                    for row in event_rows
                ),
                "vnf_endpoint_pool_rejected_requests": sum(
                    row.get("probe_status") == "vnf_endpoint_pool_exhausted"
                    for row in event_rows
                ),
                "deployment_attempted_requests": len(deployment_timings),
                "online_planning_requests": len(online_planning_timings),
                "reroute_events_requested": len(selected_reroutes),
                "reroute_events_applied": applied_reroutes,
                "reroute_events_failed": failed_reroutes,
                "reroute_events_skipped": skipped_reroutes,
                "reroute_gate_rejections": dict(reroute_gate_rejections),
                "reroute_timing_ms": {
                    field: summarize_milliseconds([
                        float(row["reroute_timing_ms"][field])
                        for row in event_rows
                        if row.get("type") == "reroute"
                        and isinstance(row.get("reroute_timing_ms"), dict)
                        and row["reroute_timing_ms"].get(field) is not None
                    ])
                    for field in ("controller_ms", "total_ms")
                },
                "vnf_migration_events_requested": len(selected_migrations),
                "vnf_migration_events_applied": applied_migrations,
                "vnf_migration_events_failed": failed_migrations,
                "vnf_migration_events_skipped": skipped_migrations,
                "vnf_impairment_events_requested": len(selected_impairments),
                "vnf_impairment_events_applied": applied_impairments,
                "vnf_impairment_events_skipped": skipped_impairments,
                "vnf_migration_timing_ms": {
                    field: summarize_milliseconds(
                        [float(row.get(field, 0.0)) for row in migration_records
                         if row.get("success")]
                    )
                    for field in (
                        "prepare_ms", "switch_ms", "switch_rpc_ms", "drain_ms", "total_ms"
                    )
                },
                "sla_met_requests": sla_met_requests,
                "request_acceptance_rate": sla_met_requests / len(requests),
                "traffic_observed_receivers": sum(
                    int(row.get("received_packets", 0)) > 0 for row in measured
                ),
                "sla_met_receivers": sum(bool(row.get("sla_met")) for row in measured),
                "delay_sla_met_receivers": sum(
                    bool(row.get("delay_sla_met")) for row in measured
                ),
                "jitter_sla_met_receivers": sum(
                    bool(row.get("jitter_sla_met")) for row in measured
                ),
                "loss_sla_met_receivers": sum(
                    bool(row.get("loss_sla_met")) for row in measured
                ),
                "sender_failure_receivers": sum(
                    row.get("measurement_status") == "sender_failed"
                    for row in measured
                ),
                "vnf_stages_planned": planned_vnf_stages,
                "vnf_stages_expected": expected_vnf_stages,
                "vnf_stages_measured": sum(
                    row["result"] is not None for row in vnf_results
                ),
                "vnf_stages_with_traffic": sum(
                        row["result"] is not None
                        and int(row["result"].get("received") or 0) > 0
                    for row in vnf_results
                ),
                "sfc_traversed_requests": sfc_traversed_requests,
                "mean_delay_ms": sum(delays) / len(delays) if delays else None,
                "mean_packet_loss_rate": (
                    sum(float(row.get("packet_loss_rate", 1.0)) for row in measured)
                    / max(1, len(measured))
                ),
                "mean_ready_wait_ms": (
                    1000.0 * sum(ready_waits) / len(ready_waits)
                    if ready_waits
                    else None
                ),
                "p95_ready_wait_ms": (
                    1000.0
                    * ready_waits[
                        min(len(ready_waits) - 1, math.ceil(0.95 * len(ready_waits)) - 1)
                    ]
                    if ready_waits
                    else None
                ),
                "max_ready_wait_ms": 1000.0 * max(ready_waits) if ready_waits else None,
                "deployment_timing_ms": {
                    field: summarize_milliseconds(
                        [float(row.get(field, 0.0)) for row in deployment_timings]
                    )
                    for field in (
                        "neighbor_setup_ms",
                        "vnf_slot_wait_ms",
                        "vnf_command_ms",
                        "vnf_ready_wait_ms",
                        "ryu_install_ms",
                        "sender_dispatch_ms",
                        "setup_observed_ms",
                        "total_ms",
                    )
                },
                "candidate_selection_timing_ms": summarize_milliseconds(
                    candidate_selection_timings_ms
                ),
                "arrival_scheduler_lag_ms": summarize_milliseconds(
                    arrival_scheduler_lags
                ),
                "online_planning_timing_ms": {
                    field: summarize_milliseconds(
                        [
                            float(row[field])
                            for row in online_planning_timings
                            if row.get(field) is not None
                        ]
                    )
                    for field in (
                        "inference_ms",
                        "conversion_ms",
                        "total_ms",
                        "arrival_to_planning_start_ms",
                        "arrival_to_plan_ready_ms",
                    )
                },
            },
        }
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["measurement_summary"], ensure_ascii=False, indent=2))
        if not valid:
            raise RuntimeError(
                "probe output is incomplete; inspect receiver_results and sender_results "
                f"in {output_path}"
            )
    finally:
        if online_planner is not None and hasattr(online_planner, "close"):
            try:
                online_planner.close()
            except (BrokenPipeError, EOFError, OSError):
                # A worker may already have exited after a runtime failure;
                # cleanup must not mask the original deployment exception.
                pass
        if neighbor_executor is not None:
            neighbor_executor.shutdown(wait=True, cancel_futures=True)
        if vnf_register_batcher is not None:
            vnf_register_batcher.close()
        if vnf_unregister_batcher is not None:
            vnf_unregister_batcher.close()
        if ryu_batch_committer is not None:
            ryu_batch_committer.close()
        restore_service_scheduler(wsl, controller_scheduler)
        restore_service_cpu_affinity(wsl, controller_affinity)
        restore_service_cpu_affinity(wsl, ovs_affinity)
        for request_id in sorted(installed):
            try:
                if request_id in sfc_plans and sfc_plans[request_id].get("accepted", True):
                    client.delete_sfc(request_id)
                else:
                    client.delete_tree(request_id)
            except Exception:
                pass
        if mininet.poll() is None:
            for request_id in sorted(active_vnfs):
                try:
                    stop_commands = []
                    for segment in sfc_plans[request_id]["segments"]:
                        stage = int(segment["stage"])
                        placement = sfc_plans[request_id]["placement_by_vnf"][str(stage)]
                        dc_node = int(placement["dc_node"])
                        host = str(nodes[dc_node]["host"])
                        files = sfc_runtime_files(probe_dir, request_id, stage)
                        if args.vnf_launch_mode == "agent":
                            command = vnf_agent_unregister_command(
                                vnf_agent_fifo_for_binding(
                                    vnf_agent_fifos,
                                    host,
                                    request_id,
                                    stage,
                                ),
                                request_id,
                                stage,
                            )
                        else:
                            command = sfc_stop_command(to_wsl_path(files["pid"]))
                        stop_commands.append((host, command))
                    mininet_runtime_commands(
                        args.mininet_command_port, stop_commands, 5.0
                    )
                except Exception:
                    pass
        if mininet.poll() is None and vnf_agent_fifos:
            try:
                shutdown_commands = []
                for host, fifos in vnf_agent_fifos.items():
                    for fifo, pid_file in zip(
                        fifos, vnf_agent_pid_files[host], strict=True
                    ):
                        shutdown = fifo_json_command(
                            fifo, {"operation": "shutdown"}
                        )
                        shutdown_commands.append(
                            (
                                host,
                                f"{shutdown}; "
                                f"if test -s {shlex.quote(pid_file)}; then "
                                f"for i in $(seq 1 50); do "
                                f"kill -0 $(cat {shlex.quote(pid_file)}) 2>/dev/null || break; "
                                f"sleep 0.02; done; fi",
                            )
                        )
                mininet_runtime_commands(
                    args.mininet_command_port, shutdown_commands, 10.0
                )
            except Exception:
                pass
        if mininet.poll() is None:
            try:
                mininet_runtime_request(
                    args.mininet_command_port,
                    {"operation": "shutdown"},
                    5.0,
                )
            except Exception:
                pass
        try:
            mininet.wait(timeout=30)
        except subprocess.TimeoutExpired:
            mininet.kill()
            mininet.wait()
        mininet_log.close()
        subprocess.run(
            wsl + ["rm", "-rf", handshake_dir_wsl],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if native_vnf_agent_wsl:
            subprocess.run(
                wsl + ["rm", "-f", native_vnf_agent_wsl],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if native_receiver_wsl:
            subprocess.run(
                wsl + ["rm", "-f", native_receiver_wsl],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
