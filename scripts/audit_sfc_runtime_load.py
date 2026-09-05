#!/usr/bin/env python3
"""Audit SFC plan bandwidth against the Mininet profile and runtime probes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


Edge = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--profile", default="sdn/topologies/us_backbone_28.json")
    parser.add_argument("--runtime", help="optional Mininet/Ryu result JSON")
    parser.add_argument("--output", required=True, help="summary JSON output")
    parser.add_argument("--edge-csv", help="optional directed-edge audit CSV")
    parser.add_argument("--payload-bytes", type=int, default=1200)
    parser.add_argument(
        "--packet-overhead-bytes",
        type=int,
        default=42,
        help="Ethernet + IPv4 + UDP bytes seen by a Linux link qdisc",
    )
    parser.add_argument("--utilization-limit", type=float, default=0.80)
    parser.add_argument(
        "--bandwidth-demand-layer",
        choices=("wire", "payload"),
        default="wire",
        help="whether bw_origin already includes per-packet link overhead",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def path_edges(path: Iterable[Any]) -> list[Edge]:
    nodes = [int(value) for value in path]
    return list(zip(nodes, nodes[1:]))


def plan_edge_traversals(plan: dict[str, Any]) -> list[Edge]:
    edges: list[Edge] = []
    for segment in plan.get("segments", []):
        edges.extend(path_edges(segment.get("path", [])))
    edges.extend(
        (int(edge[0]), int(edge[1]))
        for edge in plan.get("multicast", {}).get("tree_edges", [])
    )
    return edges


def profile_capacities(
    profile: dict[str, Any],
) -> tuple[dict[Edge, float], dict[str, Edge]]:
    default = float(profile.get("default_bandwidth_mbps", 0.0))
    capacities: dict[Edge, float] = {}
    interfaces: dict[str, Edge] = {}
    for row in profile["edges"]:
        u, v = int(row["u"]), int(row["v"])
        capacity = float(row.get("bandwidth_mbps", default))
        capacities[(u, v)] = capacity
        capacities[(v, u)] = capacity
        if row.get("u_port") is not None:
            interfaces[f"s{u}-eth{int(row['u_port'])}"] = (u, v)
        if row.get("v_port") is not None:
            interfaces[f"s{v}-eth{int(row['v_port'])}"] = (v, u)
    return capacities, interfaces


def sweep_peak_load(
    requests: dict[int, dict[str, Any]],
    plans: dict[int, dict[str, Any]],
    selected_ids: set[int],
    wire_factor: float,
) -> tuple[dict[Edge, dict[str, float]], int]:
    events: list[tuple[float, int, int, float, Counter[Edge]]] = []
    for request_id in selected_ids:
        request = requests.get(request_id)
        plan = plans.get(request_id)
        if request is None or plan is None or not plan.get("accepted"):
            continue
        footprint = Counter(plan_edge_traversals(plan))
        bandwidth = float(request["bw_origin"])
        events.append(
            (float(request["arrival_time"]), 1, request_id, bandwidth, footprint)
        )
        events.append(
            (float(request["leave_time"]), -1, request_id, bandwidth, footprint)
        )

    # Release before allocate when timestamps are equal.
    events.sort(key=lambda row: (row[0], row[1]))
    active_requests: set[int] = set()
    active_load: Counter[Edge] = Counter()
    peaks: dict[Edge, dict[str, float]] = {}
    max_concurrency = 0
    for timestamp, direction, request_id, bandwidth, footprint in events:
        if direction < 0:
            active_requests.discard(request_id)
        else:
            active_requests.add(request_id)
        max_concurrency = max(max_concurrency, len(active_requests))
        for edge, multiplicity in footprint.items():
            active_load[edge] += direction * bandwidth * multiplicity
            if abs(active_load[edge]) < 1e-9:
                active_load.pop(edge, None)
            payload = max(0.0, active_load.get(edge, 0.0))
            previous = peaks.setdefault(
                edge,
                {"payload_mbps": 0.0, "wire_mbps": 0.0, "time_s": timestamp},
            )
            if payload > previous["payload_mbps"]:
                previous.update(
                    {
                        "payload_mbps": payload,
                        "wire_mbps": payload * wire_factor,
                        "time_s": timestamp,
                    }
                )
    return peaks, max_concurrency


def runtime_request_ids(runtime: dict[str, Any]) -> set[int]:
    return {
        int(row["request_id"])
        for row in runtime.get("sender_results", [])
        if row.get("result", {}).get("status") == "completed"
    }


def runtime_packet_summary(runtime: dict[str, Any]) -> dict[str, Any]:
    senders = [
        row.get("result", {})
        for row in runtime.get("sender_results", [])
        if row.get("result", {}).get("status") == "completed"
    ]
    receivers = [
        row.get("result", {})
        for row in runtime.get("receiver_results", [])
        if row.get("result", {}).get("measurement_status") == "completed"
    ]
    planned = sum(int(row.get("planned_packets", 0)) for row in senders)
    sent = sum(int(row.get("sent_packets", 0)) for row in senders)
    expected = sum(int(row.get("expected_packets", 0)) for row in receivers)
    received = sum(int(row.get("received_packets", 0)) for row in receivers)
    return {
        "completed_senders": len(senders),
        "completed_receivers": len(receivers),
        "planned_sender_packets": planned,
        "sent_sender_packets": sent,
        "sender_delivery_ratio": sent / planned if planned else None,
        "sender_schedule_shortfall_packets": max(0, planned - sent),
        "expected_receiver_packets": expected,
        "received_receiver_packets": received,
        "network_delivery_ratio": received / expected if expected else None,
        "receiver_loss_packets": max(0, expected - received),
    }


def qdisc_by_edge(
    runtime: dict[str, Any], interfaces: dict[str, Edge]
) -> dict[Edge, dict[str, int]]:
    rows: dict[Edge, dict[str, int]] = {}
    qdiscs = runtime.get("network_diagnostics", {}).get("delta", {}).get("qdiscs", {})
    for interface, values in qdiscs.items():
        edge = interfaces.get(interface)
        if edge is None:
            continue
        row = rows.setdefault(edge, {"dropped": 0, "overlimits": 0, "requeues": 0})
        for field in row:
            row[field] += int(values.get(field, 0))
    return rows


def main() -> int:
    args = parse_args()
    if (
        args.payload_bytes <= 0
        or args.packet_overhead_bytes < 0
        or not 0.0 < args.utilization_limit <= 1.0
    ):
        raise ValueError("invalid payload, overhead, or utilization limit")

    requests = {
        int(row["id"]): row for row in load_jsonl(resolve(args.requests))
    }
    plans = {
        int(row["request_id"]): row for row in load_jsonl(resolve(args.plans))
    }
    profile = load_json(resolve(args.profile))
    capacities, interfaces = profile_capacities(profile)
    wire_factor = (
        (args.payload_bytes + args.packet_overhead_bytes) / args.payload_bytes
        if args.bandwidth_demand_layer == "payload"
        else 1.0
    )

    accepted_ids = {
        request_id for request_id, plan in plans.items() if plan.get("accepted")
    }
    planned_peaks, planned_concurrency = sweep_peak_load(
        requests, plans, accepted_ids, wire_factor
    )

    runtime: dict[str, Any] | None = None
    runtime_ids: set[int] = set()
    runtime_peaks: dict[Edge, dict[str, float]] = {}
    runtime_concurrency = 0
    runtime_packets = None
    runtime_qdiscs: dict[Edge, dict[str, int]] = {}
    if args.runtime:
        runtime = load_json(resolve(args.runtime))
        runtime_ids = runtime_request_ids(runtime)
        runtime_peaks, runtime_concurrency = sweep_peak_load(
            requests, plans, runtime_ids, wire_factor
        )
        runtime_packets = runtime_packet_summary(runtime)
        runtime_qdiscs = qdisc_by_edge(runtime, interfaces)

    ledger_capacity_samples = []
    ledger_errors = []
    duplicate_traversal_plans = []
    for request_id in sorted(accepted_ids):
        plan = plans[request_id]
        request = requests.get(request_id)
        if request is None:
            ledger_errors.append({"request_id": request_id, "reason": "missing_request"})
            continue
        traversals = plan_edge_traversals(plan)
        if len(traversals) != len(set(traversals)):
            duplicate_traversal_plans.append(request_id)
        ledger = plan.get("resource_ledger")
        if isinstance(ledger, dict) and ledger.get("available"):
            current = ledger.get("current_request", {})
            declared_count = int(current.get("edge_allocation_count", -1))
            declared_bandwidth = float(
                current.get("edge_bandwidth_mbps", math.nan)
            )
            expected_bandwidth = len(traversals) * float(request["bw_origin"])
            if declared_count != len(traversals) or not math.isclose(
                declared_bandwidth, expected_bandwidth, abs_tol=1e-6
            ):
                ledger_errors.append(
                    {
                        "request_id": request_id,
                        "declared_edge_count": declared_count,
                        "traversal_count": len(traversals),
                        "declared_bandwidth_mbps": declared_bandwidth,
                        "expected_bandwidth_mbps": expected_bandwidth,
                    }
                )
        total_capacity = (ledger or {}).get("bandwidth", {}).get("capacity_mbps")
        if total_capacity is not None and capacities:
            ledger_capacity_samples.append(float(total_capacity) / len(capacities))

    assumed_capacity = (
        statistics.median(ledger_capacity_samples) if ledger_capacity_samples else None
    )
    profile_capacity_values = sorted(set(capacities.values()))

    edge_rows = []
    for edge, capacity in sorted(capacities.items()):
        planned = planned_peaks.get(edge, {})
        observed = runtime_peaks.get(edge, {})
        qdisc = runtime_qdiscs.get(edge, {})
        planned_payload = float(planned.get("payload_mbps", 0.0))
        planned_wire = float(planned.get("wire_mbps", 0.0))
        observed_payload = float(observed.get("payload_mbps", 0.0))
        observed_wire = float(observed.get("wire_mbps", 0.0))
        edge_rows.append(
            {
                "from_dpid": edge[0],
                "to_dpid": edge[1],
                "capacity_mbps": capacity,
                "admission_limit_mbps": capacity * args.utilization_limit,
                "planned_peak_payload_mbps": planned_payload,
                "planned_peak_wire_mbps": planned_wire,
                "planned_peak_wire_utilization": planned_wire / capacity,
                "planned_peak_time_s": planned.get("time_s"),
                "runtime_set_peak_payload_mbps": observed_payload,
                "runtime_set_peak_wire_mbps": observed_wire,
                "runtime_set_peak_wire_utilization": observed_wire / capacity,
                "runtime_set_peak_time_s": observed.get("time_s"),
                "qdisc_dropped": int(qdisc.get("dropped", 0)),
                "qdisc_overlimits": int(qdisc.get("overlimits", 0)),
                "qdisc_requeues": int(qdisc.get("requeues", 0)),
            }
        )

    hottest_planned = sorted(
        edge_rows,
        key=lambda row: row["planned_peak_wire_utilization"],
        reverse=True,
    )
    hottest_runtime = sorted(
        edge_rows,
        key=lambda row: row["runtime_set_peak_wire_utilization"],
        reverse=True,
    )
    result = {
        "valid": not ledger_errors,
        "requests": len(requests),
        "plans": len(plans),
        "accepted_plans": len(accepted_ids),
        "runtime_started_requests": len(runtime_ids) if runtime is not None else None,
        "packet_model": {
            "payload_bytes": args.payload_bytes,
            "packet_overhead_bytes": args.packet_overhead_bytes,
            "bandwidth_demand_layer": args.bandwidth_demand_layer,
            "wire_to_payload_factor": wire_factor,
        },
        "capacity_model": {
            "profile_directed_edges": len(capacities),
            "profile_capacity_values_mbps": profile_capacity_values,
            "plan_ledger_assumed_capacity_mbps": assumed_capacity,
            "plan_profile_capacity_match": (
                assumed_capacity is None
                or all(
                    math.isclose(value, assumed_capacity, abs_tol=1e-6)
                    for value in profile_capacity_values
                )
            ),
            "utilization_limit": args.utilization_limit,
        },
        "ledger": {
            "duplicate_traversal_plan_count": len(duplicate_traversal_plans),
            "duplicate_traversal_request_ids": duplicate_traversal_plans,
            "error_count": len(ledger_errors),
            "errors": ledger_errors,
        },
        "planned_load": {
            "max_concurrent_requests": planned_concurrency,
            "edges_over_physical_capacity": sum(
                row["planned_peak_wire_utilization"] > 1.0 + 1e-9
                for row in edge_rows
            ),
            "edges_over_admission_limit": sum(
                row["planned_peak_wire_mbps"]
                > row["admission_limit_mbps"] + 1e-9
                for row in edge_rows
            ),
            "hottest_edges": hottest_planned[:10],
        },
        "runtime_started_set_load": (
            {
                "max_concurrent_requests": runtime_concurrency,
                "edges_over_physical_capacity": sum(
                    row["runtime_set_peak_wire_utilization"] > 1.0 + 1e-9
                    for row in edge_rows
                ),
                "edges_over_admission_limit": sum(
                    row["runtime_set_peak_wire_mbps"]
                    > row["admission_limit_mbps"] + 1e-9
                    for row in edge_rows
                ),
                "hottest_edges": hottest_runtime[:10],
            }
            if runtime is not None
            else None
        ),
        "runtime_packets": runtime_packets,
    }

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.edge_csv:
        edge_csv = resolve(args.edge_csv)
        edge_csv.parent.mkdir(parents=True, exist_ok=True)
        with edge_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0]))
            writer.writeheader()
            writer.writerows(edge_rows)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
