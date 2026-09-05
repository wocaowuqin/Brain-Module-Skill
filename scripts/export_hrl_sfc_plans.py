#!/usr/bin/env python3
"""Run a legacy HRL checkpoint and export executable SFC deployment plans."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="legacy-compatible requests.pkl")
    parser.add_argument("--runtime-requests", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="print export progress every N requests; use 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=7303)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--bw-cap", type=float, default=90.0)
    parser.add_argument("--cpu-cap", type=float, default=55.0)
    parser.add_argument("--mem-cap", type=float, default=45.0)
    parser.add_argument("--stage-port-base", type=int, default=20000)
    parser.add_argument(
        "--resource-csv",
        type=Path,
        default=None,
        help="per-request authoritative AllResourceManager ledger CSV",
    )
    parser.add_argument(
        "--safe-dest-recovery",
        action="store_true",
        help="retry destination soft failures with a cycle-safe deterministic path",
    )
    parser.add_argument(
        "--reseed-per-request",
        action="store_true",
        help="derive deterministic RNG state from the trace seed and request id",
    )
    parser.add_argument(
        "--planner-destinations",
        action="store_true",
        help="use the cycle-safe bandwidth planner for destination routing",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Python logging level for HRL export; WARNING reduces diagnostic I/O",
    )
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def request_signature(request: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that must describe the same request in both inputs."""
    return {
        "id": int(request["id"]),
        "source": int(request["source"]),
        "dest": [int(value) for value in request.get("dest", [])],
        "vnf": [int(value) for value in request.get("vnf", [])],
        "bw_origin": float(request.get("bw_origin", 0.0)),
        "cpu_origin": [float(value) for value in request.get("cpu_origin", [])],
        "memory_origin": [
            float(value) for value in request.get("memory_origin", [])
        ],
        "arrival_time": float(request.get("arrival_time", 0.0)),
        "lifetime": float(request.get("lifetime", 0.0)),
    }


def _signature_sha256(signature: dict[str, Any]) -> str:
    payload = json.dumps(
        signature, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_request_alignment(
    dataset_requests: Iterable[dict[str, Any]],
    runtime_requests: Iterable[dict[str, Any]],
) -> None:
    """Fail before inference when the HRL dataset and runtime trace differ."""
    dataset_rows = list(dataset_requests)
    runtime_rows = list(runtime_requests)
    for position, (dataset_row, runtime_row) in enumerate(
        zip(dataset_rows, runtime_rows)
    ):
        dataset_signature = request_signature(dataset_row)
        runtime_signature = request_signature(runtime_row)
        if dataset_signature == runtime_signature:
            continue
        differences = {
            field: {
                "dataset": dataset_signature[field],
                "runtime": runtime_signature[field],
            }
            for field in dataset_signature
            if dataset_signature[field] != runtime_signature[field]
        }
        raise ValueError(
            "HRL dataset/runtime request mismatch before inference: "
            f"position={position}, "
            f"dataset_id={dataset_signature['id']}, "
            f"runtime_id={runtime_signature['id']}, "
            f"differences={json.dumps(differences, ensure_ascii=False)}, "
            f"dataset_signature_sha256={_signature_sha256(dataset_signature)}, "
            f"runtime_signature_sha256={_signature_sha256(runtime_signature)}"
        )
    if len(dataset_rows) != len(runtime_rows):
        raise ValueError(
            "HRL dataset/runtime request count mismatch before inference: "
            f"dataset_count={len(dataset_rows)}, runtime_count={len(runtime_rows)}"
        )


def load_legacy_evaluator(legacy_root: Path):
    evaluator_path = legacy_root / "scripts" / "evaluate_checkpoint.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(evaluator_path)
    sys.path.insert(0, str(legacy_root))
    spec = importlib.util.spec_from_file_location("legacy_evaluate_checkpoint", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluator_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=None,
        topo="us_backbone",
        goal_strategy="adaptive",
        epsilon=0.0,
        bw_cap=args.bw_cap,
        cap_cpu=args.cpu_cap,
        cap_mem=args.mem_cap,
        ablation_variant="full",
        ablation_hop=False,
        ablation_reach=False,
        zero_candidate_feats=False,
        minimal_mlp_state=False,
        checkpoint=resolved(args.checkpoint),
        model_mode="eval",
        start_episode=0,
    )


def profile_maps(profile: dict[str, Any]) -> tuple[dict[tuple[int, int], int], dict[int, dict[str, Any]]]:
    ports: dict[tuple[int, int], int] = {}
    for edge in profile["edges"]:
        u, v = int(edge["u"]), int(edge["v"])
        ports[(u, v)] = int(edge["u_port"])
        ports[(v, u)] = int(edge["v_port"])
    nodes = {int(row["dpid"]): row for row in profile["nodes"]}
    return ports, nodes


def positive_edges(tree_snapshot: dict[str, Any] | None) -> set[tuple[int, int]]:
    edges = set()
    for raw_edge, flow in (tree_snapshot or {}).get("tree", {}).items():
        if float(flow) <= 0.0:
            continue
        if isinstance(raw_edge, str):
            text = raw_edge.strip("()[] ")
            values = [int(value.strip()) for value in text.split(",")]
            raw_edge = values
        u, v = raw_edge
        edges.add((int(u), int(v)))
    return edges


def tree_path(edges: Iterable[tuple[int, int]], source: int, destination: int) -> list[int]:
    """Find a path using only the committed directed SFT edges."""
    adjacency: dict[int, set[int]] = {}
    for u, v in edges:
        adjacency.setdefault(int(u), set()).add(int(v))
    queue = deque([int(source)])
    parent = {int(source): None}
    while queue:
        node = queue.popleft()
        if node == int(destination):
            break
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if int(destination) not in parent:
        raise ValueError(f"tree has no path from {source} to {destination}")
    result = [int(destination)]
    while result[-1] != int(source):
        result.append(parent[result[-1]])
    return list(reversed(result))


def path_outputs(
    path: list[int],
    ports: dict[tuple[int, int], int],
    nodes: dict[int, dict[str, Any]],
) -> dict[str, list[int]]:
    if not path:
        raise ValueError("path cannot be empty")
    outputs: dict[str, list[int]] = {}
    for u, v in zip(path, path[1:]):
        if (u, v) not in ports:
            raise ValueError(f"profile has no directed edge {u}->{v}")
        outputs[str(u)] = [ports[(u, v)]]
    outputs[str(path[-1])] = [int(nodes[path[-1]]["host_port"])]
    return outputs


def multicast_outputs(
    paths: dict[str, list[int]],
    ports: dict[tuple[int, int], int],
    nodes: dict[int, dict[str, Any]],
) -> tuple[dict[str, list[int]], list[list[int]]]:
    outputs: dict[str, set[int]] = {}
    edges = set()
    for raw_destination, path in paths.items():
        destination = int(raw_destination)
        for u, v in zip(path, path[1:]):
            if (u, v) not in ports:
                raise ValueError(f"profile has no directed edge {u}->{v}")
            outputs.setdefault(str(u), set()).add(ports[(u, v)])
            edges.add((u, v))
        outputs.setdefault(str(destination), set()).add(
            int(nodes[destination]["host_port"])
        )
    return (
        {key: sorted(values) for key, values in sorted(outputs.items(), key=lambda row: int(row[0]))},
        [list(edge) for edge in sorted(edges)],
    )


def to_dpid_path(path: Iterable[Any]) -> list[int]:
    return [int(value) + 1 for value in path]


def convert_plan(
    info: dict[str, Any],
    original: dict[str, Any],
    profile: dict[str, Any],
    stage_port_base: int,
) -> dict[str, Any]:
    if not info.get("success"):
        return {
            "request_id": int(original["id"]),
            "accepted": False,
            "reason": str(info.get("reason", "hrl_rejected")),
        }
    if not (info.get("tree_snapshot") or {}).get("tree"):
        raise ValueError(
            f"request {original['id']} succeeded without an authoritative tree snapshot"
        )
    chain_zero = [int(value) for value in info.get("chain_nodes", [])]
    chain = to_dpid_path(chain_zero)
    vnf_types = [int(value) for value in original["vnf"]]
    cpu = [float(value) for value in original["cpu_origin"]]
    memory = [float(value) for value in original["memory_origin"]]
    if len(chain) != len(vnf_types):
        raise ValueError(
            f"request {original['id']} has {len(vnf_types)} VNFs but HRL exported {len(chain)} placements"
        )

    ports, nodes = profile_maps(profile)
    edges_zero = positive_edges(info.get("tree_snapshot"))
    terminals_zero = [int(original["source"]), *chain_zero]
    spine_zero = [
        tree_path(edges_zero, source, destination)
        for source, destination in zip(terminals_zero, terminals_zero[1:])
    ]

    root_zero = chain_zero[-1]
    destination_zero = [int(value) for value in original["dest"]]
    branch_paths_zero = {
        str(destination): tree_path(edges_zero, root_zero, destination)
        for destination in destination_zero
    }
    branch_paths = {
        str(destination + 1): to_dpid_path(path)
        for destination, path in ((int(key), value) for key, value in branch_paths_zero.items())
    }
    outputs, tree_edges = multicast_outputs(branch_paths, ports, nodes)

    base_port = int(stage_port_base) + (int(original["id"]) - 1) * len(chain)
    if base_port + len(chain) - 1 > 65535:
        raise ValueError("stage UDP port range exceeds 65535")
    segments = []
    spine = [to_dpid_path(path) for path in spine_zero]
    for stage, path in enumerate(spine):
        target = chain[stage]
        target_ip = str(nodes[target]["host_ip"]).split("/")[0]
        stage_port = base_port + stage
        segments.append(
            {
                "stage": stage,
                "from_dpid": path[0],
                "to_dpid": target,
                "target_ip": target_ip,
                "udp_port": stage_port,
                "path": path,
                "switch_outputs": path_outputs(path, ports, nodes),
            }
        )

    placements = {
        str(stage): {
            "dc_node": chain[stage],
            "vnf_type": vnf_types[stage],
            "cpu_units": cpu[stage],
            "memory_units": memory[stage],
            "listen_ip": str(nodes[chain[stage]]["host_ip"]).split("/")[0],
            "listen_port": base_port + stage,
        }
        for stage in range(len(chain))
    }
    return {
        "version": "hrl_sfc_plan_v1",
        "request_id": int(original["id"]),
        "accepted": True,
        "source_dpid": int(original["source_dpid"]),
        "destination_dpids": [int(value) for value in original["destination_dpids"]],
        "chain_nodes": chain,
        "placement_by_vnf": placements,
        "segments": segments,
        "multicast": {
            "root_dpid": chain[-1],
            "dst_ip": str(original["multicast_ip"]),
            "udp_port": int(original["udp_port"]),
            "paths": branch_paths,
            "tree_edges": tree_edges,
            "switch_outputs": outputs,
        },
        "hrl": {
            "steps": int(info.get("steps", 0)),
            "reward": float(info.get("reward", 0.0)),
            "completion_status": str(info.get("completion_status", "")),
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _node_peak(
    capacity: Any,
    available: Any,
    eligible_nodes: set[int],
) -> dict[str, float | int]:
    values = {
        index: max(0.0, float(cap) - float(avail))
        for index, (cap, avail) in enumerate(zip(capacity, available))
        if index in eligible_nodes
    }
    if not values:
        return {"dpid": 0, "used_units": 0.0, "capacity_units": 0.0, "utilization": 0.0}
    index = max(values, key=values.__getitem__)
    cap = float(capacity[index])
    return {
        "dpid": index + 1,
        "used_units": values[index],
        "capacity_units": cap,
        "utilization": values[index] / cap if cap > 0 else 0.0,
    }


def resource_ledger_snapshot(env: Any, request_id: int) -> dict[str, Any]:
    """Serialize the live FusedResourceManager state after one decision.

    The pool is the authority for reuse and lifecycle release.  Exporting this
    snapshot avoids re-estimating live CPU/MEM/BW from placement JSON later.
    """
    resource_manager = getattr(env, "resource_mgr", None)
    pool = getattr(resource_manager, "pool", None)
    if resource_manager is None or pool is None:
        return {"available": False}

    cpu_capacity = list(getattr(pool, "cpu_cap", []))
    cpu_available = list(getattr(pool, "cpu_avail", []))
    memory_capacity = list(getattr(pool, "mem_cap", []))
    memory_available = list(getattr(pool, "mem_avail", []))
    dc_nodes = {int(node) for node in getattr(env, "dc_nodes", [])}
    if not dc_nodes:
        dc_nodes = set(range(len(cpu_capacity)))
    cpu_used = sum(
        max(0.0, float(cap) - float(avail))
        for index, (cap, avail) in enumerate(zip(cpu_capacity, cpu_available))
        if index in dc_nodes
    )
    memory_used = sum(
        max(0.0, float(cap) - float(avail))
        for index, (cap, avail) in enumerate(zip(memory_capacity, memory_available))
        if index in dc_nodes
    )
    cpu_total_capacity = sum(float(cpu_capacity[index]) for index in dc_nodes)
    memory_total_capacity = sum(float(memory_capacity[index]) for index in dc_nodes)

    bandwidth_capacity = getattr(pool, "bw_cap", {})
    bandwidth_available = getattr(pool, "bw_avail", {})
    bandwidth_rows = []
    for edge, cap in bandwidth_capacity.items():
        available = float(bandwidth_available.get(edge, cap))
        used = max(0.0, float(cap) - available)
        bandwidth_rows.append((used, float(cap), int(edge[0]), int(edge[1])))
    bandwidth_used = sum(row[0] for row in bandwidth_rows)
    bandwidth_cap = sum(row[1] for row in bandwidth_rows)
    hottest = max(bandwidth_rows, default=(0.0, 0.0, -1, -1))

    request_table = getattr(resource_manager, "request_table", {})
    active_records = [
        record for record in request_table.values()
        if getattr(record, "state", "") in {"PENDING", "ACTIVE"}
    ]
    instances = getattr(resource_manager, "instance_table", {})
    active_instances = [
        instance for instance in instances.values()
        if getattr(instance, "state", "") == "ACTIVE"
    ]
    current = request_table.get(request_id)
    bindings = list(getattr(current, "vnf_bindings", [])) if current is not None else []
    allocations = list(getattr(current, "edge_allocations", [])) if current is not None else []
    reused_bindings = sum(bool(getattr(binding, "reused", False)) for binding in bindings)

    return {
        "available": True,
        "time_s": float(getattr(env, "time_step", 0.0)),
        "active_request_count": len(active_records),
        "active_instance_count": len(active_instances),
        "cpu": {
            "used_units": cpu_used,
            "capacity_units": cpu_total_capacity,
            "utilization": cpu_used / cpu_total_capacity if cpu_total_capacity > 0 else 0.0,
            "peak_node": _node_peak(cpu_capacity, cpu_available, dc_nodes),
        },
        "memory": {
            "used_units": memory_used,
            "capacity_units": memory_total_capacity,
            "utilization": memory_used / memory_total_capacity if memory_total_capacity > 0 else 0.0,
            "peak_node": _node_peak(memory_capacity, memory_available, dc_nodes),
        },
        "bandwidth": {
            "used_mbps": bandwidth_used,
            "capacity_mbps": bandwidth_cap,
            "utilization": bandwidth_used / bandwidth_cap if bandwidth_cap > 0 else 0.0,
            "peak_directed_edge": {
                "from_dpid": hottest[2] + 1 if hottest[2] >= 0 else 0,
                "to_dpid": hottest[3] + 1 if hottest[3] >= 0 else 0,
                "used_mbps": hottest[0],
                "capacity_mbps": hottest[1],
                "utilization": hottest[0] / hottest[1] if hottest[1] > 0 else 0.0,
            },
        },
        "current_request": {
            "recorded": current is not None,
            "edge_allocation_count": len(allocations),
            "edge_bandwidth_mbps": sum(float(getattr(item, "bw", 0.0)) for item in allocations),
            "vnf_binding_count": len(bindings),
            "reused_vnf_binding_count": reused_bindings,
            "new_vnf_binding_count": len(bindings) - reused_bindings,
        },
    }


def write_resource_csv(
    path: Path,
    plans: Iterable[dict[str, Any]],
    requests: dict[int, dict[str, Any]],
) -> None:
    fields = [
        "request_id", "arrival_time", "leave_time", "accepted", "reason",
        "ledger_time_s", "active_request_count", "active_instance_count",
        "cpu_used_units", "cpu_capacity_units", "cpu_utilization",
        "cpu_peak_dpid", "cpu_peak_used_units", "cpu_peak_utilization",
        "memory_used_units", "memory_capacity_units", "memory_utilization",
        "memory_peak_dpid", "memory_peak_used_units", "memory_peak_utilization",
        "bandwidth_used_mbps", "bandwidth_capacity_mbps", "bandwidth_utilization",
        "bandwidth_peak_from_dpid", "bandwidth_peak_to_dpid", "bandwidth_peak_used_mbps",
        "bandwidth_peak_utilization", "current_edge_allocation_count",
        "current_edge_bandwidth_mbps", "current_vnf_binding_count",
        "current_reused_vnf_binding_count", "current_new_vnf_binding_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for plan in plans:
            request = requests[int(plan["request_id"])]
            ledger = plan.get("resource_ledger", {})
            cpu = ledger.get("cpu", {})
            memory = ledger.get("memory", {})
            bandwidth = ledger.get("bandwidth", {})
            current = ledger.get("current_request", {})
            writer.writerow({
                "request_id": plan["request_id"],
                "arrival_time": request.get("arrival_time"),
                "leave_time": request.get("leave_time"),
                "accepted": bool(plan.get("accepted")),
                "reason": plan.get("reason", ""),
                "ledger_time_s": ledger.get("time_s", ""),
                "active_request_count": ledger.get("active_request_count", ""),
                "active_instance_count": ledger.get("active_instance_count", ""),
                "cpu_used_units": cpu.get("used_units", ""),
                "cpu_capacity_units": cpu.get("capacity_units", ""),
                "cpu_utilization": cpu.get("utilization", ""),
                "cpu_peak_dpid": cpu.get("peak_node", {}).get("dpid", ""),
                "cpu_peak_used_units": cpu.get("peak_node", {}).get("used_units", ""),
                "cpu_peak_utilization": cpu.get("peak_node", {}).get("utilization", ""),
                "memory_used_units": memory.get("used_units", ""),
                "memory_capacity_units": memory.get("capacity_units", ""),
                "memory_utilization": memory.get("utilization", ""),
                "memory_peak_dpid": memory.get("peak_node", {}).get("dpid", ""),
                "memory_peak_used_units": memory.get("peak_node", {}).get("used_units", ""),
                "memory_peak_utilization": memory.get("peak_node", {}).get("utilization", ""),
                "bandwidth_used_mbps": bandwidth.get("used_mbps", ""),
                "bandwidth_capacity_mbps": bandwidth.get("capacity_mbps", ""),
                "bandwidth_utilization": bandwidth.get("utilization", ""),
                "bandwidth_peak_from_dpid": bandwidth.get("peak_directed_edge", {}).get("from_dpid", ""),
                "bandwidth_peak_to_dpid": bandwidth.get("peak_directed_edge", {}).get("to_dpid", ""),
                "bandwidth_peak_used_mbps": bandwidth.get("peak_directed_edge", {}).get("used_mbps", ""),
                "bandwidth_peak_utilization": bandwidth.get("peak_directed_edge", {}).get("utilization", ""),
                "current_edge_allocation_count": current.get("edge_allocation_count", ""),
                "current_edge_bandwidth_mbps": current.get("edge_bandwidth_mbps", ""),
                "current_vnf_binding_count": current.get("vnf_binding_count", ""),
                "current_reused_vnf_binding_count": current.get("reused_vnf_binding_count", ""),
                "current_new_vnf_binding_count": current.get("new_vnf_binding_count", ""),
            })


def main() -> None:
    args = parse_args()
    # Configure logging before loading the legacy evaluator.  The evaluator
    # imports the HRL stack lazily, so this controls its diagnostic output too.
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        force=True,
    )
    selected_level = getattr(logging, str(args.log_level).upper())
    # Some legacy modules set their own logger level to INFO.  ``disable``
    # applies globally and makes the quiet benchmark independent of those
    # module-local settings while preserving WARNING/ERROR diagnostics.
    logging.disable(max(0, int(selected_level) - 1))
    legacy_root = resolved(args.legacy_root)
    checkpoint = resolved(args.checkpoint)
    data_path = resolved(args.data)
    runtime_path = resolved(args.runtime_requests)
    profile_path = resolved(args.profile)
    output_path = resolved(args.output)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every must be non-negative")

    runtime_requests = load_jsonl(runtime_path)
    original_by_id = {int(row["id"]): row for row in runtime_requests}
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    evaluator = load_legacy_evaluator(legacy_root)
    eval_args = evaluator_args(args)
    env, _, coordinator = evaluator.build_eval_stack(eval_args, args.seed, data_path)
    validate_request_alignment(
        getattr(env, "all_requests", []), runtime_requests
    )
    env._safe_dest_recovery_enabled = bool(args.safe_dest_recovery)
    env._planner_destinations_enabled = bool(args.planner_destinations)

    plans = []
    for request_index in range(min(args.episodes, len(runtime_requests))):
        if args.reseed_per_request:
            request_id = int(runtime_requests[request_index]["id"])
            episode_seed = (int(args.seed) * 1_000_003 + request_id) & 0x7FFFFFFF
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
        with torch.no_grad():
            _, info = coordinator.run_episode(training=False, max_steps=args.max_steps)
        snapshot = info.get("req_snapshot") or {}
        request_id = int(snapshot.get("id", -1))
        if request_id not in original_by_id:
            raise ValueError(f"HRL returned unknown request id {request_id}")
        plan = convert_plan(
            info,
            original_by_id[request_id],
            profile,
            args.stage_port_base,
        )
        plan["resource_ledger"] = resource_ledger_snapshot(env, request_id)
        plans.append(plan)
        completed = request_index + 1
        if args.progress_every and (
            completed % args.progress_every == 0
            or completed == min(args.episodes, len(runtime_requests))
        ):
            accepted_count = sum(1 for row in plans if row.get("accepted"))
            print(
                json.dumps(
                    {
                        "progress": completed,
                        "total": min(args.episodes, len(runtime_requests)),
                        "accepted": accepted_count,
                        "rejected": completed - accepted_count,
                        "latest_request_id": request_id,
                        "latest_reason": plan.get("reason", ""),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    write_jsonl(output_path, plans)

    resource_csv_path = (
        resolved(args.resource_csv)
        if args.resource_csv is not None
        else output_path.with_name(f"{output_path.stem}_resource_ledger.csv")
    )
    write_resource_csv(resource_csv_path, plans, original_by_id)

    summary_path = resolved(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    accepted = [row for row in plans if row.get("accepted")]
    summary = {
        "valid": True,
        "plans": len(plans),
        "accepted": len(accepted),
        "rejected": len(plans) - len(accepted),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "runtime_requests": str(runtime_path),
        "runtime_requests_sha256": sha256(runtime_path),
        "profile": str(profile_path),
        "profile_sha256": sha256(profile_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "resource_ledger_csv": str(resource_csv_path),
        "resource_ledger_csv_sha256": sha256(resource_csv_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
