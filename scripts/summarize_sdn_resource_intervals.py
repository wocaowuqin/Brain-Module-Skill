#!/usr/bin/env python3
"""Summarize declared SDN resource demand over fixed trace-time intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.sfc_plan import declared_compute, directed_physical_edges, plan_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, help="request JSONL")
    parser.add_argument(
        "--tree-plans",
        required=True,
        help="legacy SDN tree-plan or hrl_sfc_plan_v1 JSONL",
    )
    parser.add_argument("--result", help="runtime result JSON used to filter accepted requests")
    parser.add_argument("--output", required=True, help="output CSV")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="summarize only the first N trace requests; useful for plan smoke tests",
    )
    parser.add_argument("--duration", type=float, default=400.0)
    parser.add_argument("--interval", type=float, default=50.0)
    parser.add_argument("--dc-nodes", type=int, default=20)
    parser.add_argument("--cpu-capacity-per-node", type=float, default=55.0)
    parser.add_argument("--memory-capacity-per-node", type=float, default=45.0)
    parser.add_argument("--link-capacity-mbps", type=float, default=90.0)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def total(values: Any) -> float:
    if isinstance(values, list):
        return sum(float(value) for value in values)
    return float(values)


def ledger_value(job: dict[str, Any], *keys: str) -> float | None:
    value: Any = job.get("ledger")
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if value is not None else None


def accepted_ids(result_path: Path | None) -> set[int] | None:
    if result_path is None:
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        int(event["request_id"])
        for event in result.get("events", [])
        if event.get("type") == "arrive"
        and bool(event.get("controller", {}).get("accepted"))
    }


def add_job(state: dict[str, Any], job: dict[str, Any], sign: int) -> None:
    state["requests"] += sign
    state["cpu"] += sign * job["cpu"]
    state["memory"] += sign * job["memory"]
    state["ingress_bw"] += sign * job["bw"]
    state["tree_edge_bw"] += sign * job["bw"] * len(job["edges"])
    for edge in job["edges"]:
        state["links"][edge] += sign * job["bw"]
        if abs(state["links"][edge]) < 1e-9:
            state["links"][edge] = 0.0


def update_peaks(peaks: dict[str, float], state: dict[str, Any]) -> None:
    peaks["requests"] = max(peaks["requests"], state["requests"])
    peaks["cpu"] = max(peaks["cpu"], state["cpu"])
    peaks["memory"] = max(peaks["memory"], state["memory"])
    peaks["ingress_bw"] = max(peaks["ingress_bw"], state["ingress_bw"])
    peaks["tree_edge_bw"] = max(peaks["tree_edge_bw"], state["tree_edge_bw"])
    peaks["directed_link_bw"] = max(
        peaks["directed_link_bw"],
        max(state["links"].values(), default=0.0),
    )


def interval_row(
    start: float,
    end: float,
    all_requests: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    duration = end - start
    requested_arrivals = [
        row for row in all_requests if start <= float(row["arrival_time"]) < end
    ]
    arrivals = [job for job in jobs if start <= job["arrival"] < end]
    active_at_start = [
        job for job in jobs if job["arrival"] <= start < job["leave"]
    ]
    active_at_end = [
        job for job in jobs if job["arrival"] <= end < job["leave"]
    ]
    ledger_rows = [job["ledger"] for job in arrivals if job.get("ledger", {}).get("available")]

    def ledger_peak(*keys: str) -> float | str:
        values = [ledger_value({"ledger": ledger}, *keys) for ledger in ledger_rows]
        values = [value for value in values if value is not None]
        return max(values) if values else ""

    def ledger_mean(*keys: str) -> float | str:
        values = [ledger_value({"ledger": ledger}, *keys) for ledger in ledger_rows]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else ""

    state: dict[str, Any] = {
        "requests": 0,
        "cpu": 0.0,
        "memory": 0.0,
        "ingress_bw": 0.0,
        "tree_edge_bw": 0.0,
        "links": defaultdict(float),
    }
    for job in active_at_start:
        add_job(state, job, 1)
    peaks = {
        "requests": 0.0,
        "cpu": 0.0,
        "memory": 0.0,
        "ingress_bw": 0.0,
        "tree_edge_bw": 0.0,
        "directed_link_bw": 0.0,
    }
    update_peaks(peaks, state)

    events: list[tuple[float, int, dict[str, Any]]] = []
    for job in jobs:
        if start < job["arrival"] < end:
            events.append((job["arrival"], 1, job))
        if start < job["leave"] < end:
            events.append((job["leave"], -1, job))
    for _, sign, job in sorted(events, key=lambda value: (value[0], value[1])):
        add_job(state, job, sign)
        update_peaks(peaks, state)

    cpu_unit_seconds = 0.0
    memory_unit_seconds = 0.0
    ingress_mbit = 0.0
    tree_edge_mbit = 0.0
    request_seconds = 0.0
    for job in jobs:
        overlap = max(0.0, min(job["leave"], end) - max(job["arrival"], start))
        if overlap == 0.0:
            continue
        request_seconds += overlap
        cpu_unit_seconds += job["cpu"] * overlap
        memory_unit_seconds += job["memory"] * overlap
        ingress_mbit += job["bw"] * overlap
        tree_edge_mbit += job["bw"] * len(job["edges"]) * overlap

    cpu_capacity = args.dc_nodes * args.cpu_capacity_per_node
    memory_capacity = args.dc_nodes * args.memory_capacity_per_node
    end_cpu = sum(job["cpu"] for job in active_at_end)
    end_memory = sum(job["memory"] for job in active_at_end)
    end_ingress_bw = sum(job["bw"] for job in active_at_end)
    return {
        "interval_start_s": start,
        "interval_end_s": end,
        "requested_arrivals": len(requested_arrivals),
        "deployed_arrivals": len(arrivals),
        "cumulative_requested_arrivals": sum(
            float(row["arrival_time"]) < end for row in all_requests
        ),
        "cumulative_deployed_arrivals": sum(job["arrival"] < end for job in jobs),
        "request_seconds": request_seconds,
        "cpu_unit_seconds": cpu_unit_seconds,
        "memory_unit_seconds": memory_unit_seconds,
        "ingress_mbit": ingress_mbit,
        "tree_edge_mbit": tree_edge_mbit,
        "avg_active_requests": request_seconds / duration,
        "avg_cpu_demand_units": cpu_unit_seconds / duration,
        "avg_cpu_utilization": cpu_unit_seconds / duration / cpu_capacity,
        "avg_memory_demand_units": memory_unit_seconds / duration,
        "avg_memory_utilization": memory_unit_seconds / duration / memory_capacity,
        "avg_ingress_bw_mbps": ingress_mbit / duration,
        "avg_tree_edge_bw_mbps": tree_edge_mbit / duration,
        "peak_active_requests": int(peaks["requests"]),
        "peak_cpu_demand_units": peaks["cpu"],
        "peak_cpu_utilization": peaks["cpu"] / cpu_capacity,
        "peak_memory_demand_units": peaks["memory"],
        "peak_memory_utilization": peaks["memory"] / memory_capacity,
        "peak_ingress_bw_mbps": peaks["ingress_bw"],
        "peak_total_tree_edge_bw_mbps": peaks["tree_edge_bw"],
        "peak_directed_link_bw_mbps": peaks["directed_link_bw"],
        "peak_directed_link_utilization": peaks["directed_link_bw"] / args.link_capacity_mbps,
        "active_requests_at_end": len(active_at_end),
        "cpu_demand_at_end_units": end_cpu,
        "memory_demand_at_end_units": end_memory,
        "ingress_bw_at_end_mbps": end_ingress_bw,
        # These values come directly from AllResourceManager after each
        # accepted/rejected decision. They include instance reuse and expiry.
        "ledger_observations": len(ledger_rows),
        "ledger_mean_active_requests": ledger_mean("active_request_count"),
        "ledger_peak_active_requests": ledger_peak("active_request_count"),
        "ledger_mean_cpu_used_units": ledger_mean("cpu", "used_units"),
        "ledger_peak_cpu_used_units": ledger_peak("cpu", "used_units"),
        "ledger_peak_cpu_utilization": ledger_peak("cpu", "utilization"),
        "ledger_mean_memory_used_units": ledger_mean("memory", "used_units"),
        "ledger_peak_memory_used_units": ledger_peak("memory", "used_units"),
        "ledger_peak_memory_utilization": ledger_peak("memory", "utilization"),
        "ledger_mean_bandwidth_used_mbps": ledger_mean("bandwidth", "used_mbps"),
        "ledger_peak_bandwidth_used_mbps": ledger_peak("bandwidth", "used_mbps"),
        "ledger_peak_bandwidth_utilization": ledger_peak("bandwidth", "utilization"),
        "ledger_peak_directed_link_used_mbps": ledger_peak(
            "bandwidth", "peak_directed_edge", "used_mbps"
        ),
        "ledger_peak_directed_link_utilization": ledger_peak(
            "bandwidth", "peak_directed_edge", "utilization"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.interval <= 0:
        raise ValueError("duration and interval must be positive")
    requests = read_jsonl(resolve(args.requests))
    if args.max_requests is not None:
        if args.max_requests <= 0:
            raise ValueError("max-requests must be positive")
        requests = requests[:args.max_requests]
    plans = {
        int(row["request_id"]): row for row in read_jsonl(resolve(args.tree_plans))
    }
    accepted = accepted_ids(resolve(args.result) if args.result else None)

    jobs: list[dict[str, Any]] = []
    for request in requests:
        request_id = int(request["id"])
        if accepted is not None and request_id not in accepted:
            continue
        if request_id not in plans:
            raise ValueError(f"accepted request {request_id} has no tree plan")
        plan = plans[request_id]
        if not plan.get("accepted", True):
            continue
        cpu, memory = declared_compute(plan, request)
        jobs.append(
            {
                "request_id": request_id,
                "arrival": float(request["arrival_time"]),
                "leave": float(request["leave_time"]),
                "cpu": cpu,
                "memory": memory,
                "bw": float(request["bw_origin"]),
                "edges": directed_physical_edges(plan),
                "plan_format": plan_format(plan),
                "ledger": plan.get("resource_ledger", {}),
            }
        )

    rows = []
    cumulative_cpu = 0.0
    cumulative_memory = 0.0
    cumulative_ingress = 0.0
    cumulative_tree_edge = 0.0
    cpu_capacity = args.dc_nodes * args.cpu_capacity_per_node
    memory_capacity = args.dc_nodes * args.memory_capacity_per_node
    for index in range(math.ceil(args.duration / args.interval)):
        start = index * args.interval
        end = min(args.duration, start + args.interval)
        row = interval_row(start, end, requests, jobs, args)
        cumulative_cpu += row["cpu_unit_seconds"]
        cumulative_memory += row["memory_unit_seconds"]
        cumulative_ingress += row["ingress_mbit"]
        cumulative_tree_edge += row["tree_edge_mbit"]
        row.update(
            {
                "cumulative_cpu_unit_seconds": cumulative_cpu,
                "cumulative_memory_unit_seconds": cumulative_memory,
                "cumulative_ingress_mbit": cumulative_ingress,
                "cumulative_tree_edge_mbit": cumulative_tree_edge,
                "cumulative_avg_cpu_utilization": cumulative_cpu / end / cpu_capacity,
                "cumulative_avg_memory_utilization": cumulative_memory
                / end
                / memory_capacity,
            }
        )
        rows.append(row)

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    totals = {
        "output": str(output_path),
        "interval_rows": len(rows),
        "requests": len(requests),
        "deployed_requests": len(jobs),
        "cpu_unit_seconds_0_to_duration": sum(row["cpu_unit_seconds"] for row in rows),
        "memory_unit_seconds_0_to_duration": sum(row["memory_unit_seconds"] for row in rows),
        "ingress_mbit_0_to_duration": sum(row["ingress_mbit"] for row in rows),
        "tree_edge_mbit_0_to_duration": sum(row["tree_edge_mbit"] for row in rows),
    }
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
