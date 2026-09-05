"""Run a fast, deterministic SFC placement simulation without Mininet/Ryu.

The simulator reuses the project topology, request trace, complete-plan
candidate generator, and authoritative resource ledger.  It models request
arrival and lifetime release events, resource reservations, a lightweight
queue/loss SLA model, and per-slot resource/operating-cost records.  It is
intended for algorithm comparisons; it does not measure real control-plane or
packet timing.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    ResourceFootprint,
    candidate_action_mask,
)
from core.marl.deployment_topk import (  # noqa: E402
    CompletePlanCandidateGenerator,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return float(values[index])


def path_edges(plan: Mapping[str, Any]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for segment in plan.get("segments") or []:
        path = [int(node) for node in segment.get("path") or []]
        edges.extend(zip(path, path[1:]))
    for edge in (plan.get("multicast") or {}).get("tree_edges") or []:
        if len(edge) >= 2:
            edges.append((int(edge[0]), int(edge[1])))
    return edges


def mm1k_loss_probability(utilization: float, queue_packets: int) -> float:
    """Return the stationary blocking probability of an M/M/1/K queue."""

    rho = max(0.0, float(utilization))
    if rho >= 1.0:
        return 1.0
    if rho <= 1e-12:
        return 0.0
    if abs(rho - 1.0) <= 1e-9:
        return 1.0 / (queue_packets + 1.0)
    numerator = (1.0 - rho) * rho**queue_packets
    denominator = 1.0 - rho ** (queue_packets + 1)
    return max(0.0, min(1.0, numerator / max(denominator, 1e-15)))


def evaluate_candidate_qos(
    request: Mapping[str, Any],
    candidate: Any,
    footprint: ResourceFootprint,
    snapshot: Any,
    capacities: Mapping[tuple[int, int], float],
    propagation_delays: Mapping[tuple[int, int], float],
    *,
    packet_size_bytes: int,
    queue_packets: int,
    vnf_processing_delay_ms: float,
) -> dict[str, Any]:
    """Evaluate every receiver against the candidate's post-commit state."""

    packet_bits = float(packet_size_bytes * 8)
    traversed = set(path_edges(candidate.plan))
    edge_metrics: dict[tuple[int, int], dict[str, float]] = {}
    for edge in traversed:
        capacity = max(1e-9, float(capacities[edge]))
        currently_used = capacity - float(snapshot.bandwidth_remaining[edge])
        used_after = currently_used + float(footprint.bandwidth.get(edge, 0.0))
        utilization = max(0.0, used_after / capacity)
        if utilization >= 1.0 - 1e-12:
            system_delay_ms = float("inf")
            queue_delay_ms = float("inf")
        else:
            service_ms = packet_bits / (capacity * 1_000.0)
            queue_delay_ms = service_ms * utilization / max(1e-12, 1.0 - utilization)
            system_delay_ms = (
                float(propagation_delays[edge]) + service_ms + queue_delay_ms
            )
        edge_metrics[edge] = {
            "utilization": utilization,
            "delay_ms": system_delay_ms,
            "queue_delay_ms": queue_delay_ms,
            "loss_rate": mm1k_loss_probability(utilization, queue_packets),
        }

    prefix_edges: list[tuple[int, int]] = []
    for segment in sorted(
        candidate.plan.get("segments") or [], key=lambda row: int(row.get("stage", 0))
    ):
        path = [int(node) for node in segment.get("path") or []]
        prefix_edges.extend(zip(path, path[1:]))
    multicast_paths = list(
        ((candidate.plan.get("multicast") or {}).get("paths") or {}).values()
    )
    receiver_edges = []
    for raw_path in multicast_paths:
        path = [int(node) for node in raw_path]
        receiver_edges.append(prefix_edges + list(zip(path, path[1:])))
    if not receiver_edges:
        receiver_edges = [prefix_edges]

    receiver_rows = []
    processing_delay = (
        len(request.get("vnf") or []) * float(vnf_processing_delay_ms)
    )
    for edges in receiver_edges:
        metrics = [edge_metrics[edge] for edge in edges]
        delay_ms = processing_delay + sum(row["delay_ms"] for row in metrics)
        jitter_ms = math.sqrt(
            sum(row["queue_delay_ms"] ** 2 for row in metrics)
        )
        survival = math.prod(1.0 - row["loss_rate"] for row in metrics)
        receiver_rows.append({
            "delay_ms": delay_ms,
            "jitter_ms": jitter_ms,
            "loss_rate": 1.0 - survival,
        })

    max_delay = max((row["delay_ms"] for row in receiver_rows), default=0.0)
    max_jitter = max((row["jitter_ms"] for row in receiver_rows), default=0.0)
    max_loss = max((row["loss_rate"] for row in receiver_rows), default=0.0)
    peak_utilization = max(
        (row["utilization"] for row in edge_metrics.values()), default=0.0
    )
    delay_bound = float(request.get("delay_bound_ms", math.inf))
    jitter_bound_raw = request.get("jitter_bound_ms")
    jitter_bound = (
        float(jitter_bound_raw) if jitter_bound_raw is not None else math.inf
    )
    loss_bound = float(request.get("packet_loss_bound", 1.0))
    delay_ok = max_delay <= delay_bound + 1e-9
    jitter_ok = max_jitter <= jitter_bound + 1e-9
    loss_ok = max_loss <= loss_bound + 1e-12
    return {
        "max_delay_ms": max_delay,
        "max_jitter_ms": max_jitter,
        "max_loss_rate": max_loss,
        "peak_link_utilization": peak_utilization,
        "delay_sla_met": delay_ok,
        "jitter_sla_met": jitter_ok,
        "loss_sla_met": loss_ok,
        "strict_sla": delay_ok and jitter_ok and loss_ok,
        "receiver_count": len(receiver_rows),
    }


def main() -> None:
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cpu-capacity", type=float, default=55.0)
    parser.add_argument("--memory-capacity", type=float, default=45.0)
    parser.add_argument("--bandwidth-utilization-limit", type=float, default=0.9)
    parser.add_argument("--slot-seconds", type=float, default=1.0)
    parser.add_argument(
        "--sla-admission-mode", choices=("posthoc", "hard"), default="posthoc"
    )
    parser.add_argument("--packet-size-bytes", type=int, default=1500)
    parser.add_argument("--queue-packets", type=int, default=100)
    parser.add_argument("--vnf-processing-delay-ms", type=float, default=0.2)
    args = parser.parse_args()

    if (
        args.top_k <= 0
        or args.slot_seconds <= 0
        or args.packet_size_bytes <= 0
        or args.queue_packets <= 0
        or args.vnf_processing_delay_ms < 0.0
    ):
        raise ValueError("simulation sizes must be positive and delays nonnegative")
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    requests = sorted(
        read_jsonl(Path(args.requests)), key=lambda row: (float(row["arrival_time"]), int(row["id"]))
    )
    if args.max_requests > 0:
        requests = requests[: args.max_requests]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    default_bw = float(profile.get("default_bandwidth_mbps", 90.0))
    capacities: dict[tuple[int, int], float] = {}
    delays: dict[tuple[int, int], float] = {}
    for raw in profile["edges"]:
        u, v = int(raw["u"]), int(raw["v"])
        capacity = float(raw.get("bandwidth_mbps", default_bw)) * args.bandwidth_utilization_limit
        delay = float(raw.get("delay_ms", profile.get("default_delay_ms", 1.0)))
        capacities[(u, v)] = capacities[(v, u)] = capacity
        delays[(u, v)] = delays[(v, u)] = delay
    dc_nodes = [int(node) for node in profile["dc_nodes_1based"]]
    ledger = AtomicResourceLedger(
        {node: args.cpu_capacity for node in dc_nodes},
        {node: args.memory_capacity for node in dc_nodes},
        capacities,
    )
    generator = CompletePlanCandidateGenerator(
        profile,
        max_candidates=args.top_k,
        placement_beam=8,
        placement_chains=6,
        pool_limit=max(12, args.top_k * 2),
        objective_pressure_weight=8.0,
    )
    generator.prewarm_paths()

    leave_heap: list[tuple[float, int]] = []
    active: set[int] = set()
    request_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    slot_stats: dict[int, dict[str, float]] = {}
    candidate_sources: Counter[str] = Counter()
    delays_ms: list[float] = []
    jitters_ms: list[float] = []
    losses: list[float] = []

    def release_until(timestamp: float) -> int:
        released = 0
        while leave_heap and leave_heap[0][0] <= timestamp + 1e-12:
            _, request_id = heapq.heappop(leave_heap)
            if ledger.release(request_id):
                active.discard(request_id)
                released += 1
        return released

    for request in requests:
        arrival = float(request["arrival_time"])
        released = release_until(arrival)
        snapshot = ledger.snapshot()
        candidates = generator.generate(request, snapshot)
        footprints = [candidate.footprint for candidate in candidates]
        mask = candidate_action_mask([footprints], snapshot)[0]
        feasible = [candidate for candidate, valid in zip(candidates, mask) if valid]
        evaluated = [
            (
                candidate,
                evaluate_candidate_qos(
                    request,
                    candidate,
                    candidate.footprint,
                    snapshot,
                    capacities,
                    delays,
                    packet_size_bytes=args.packet_size_bytes,
                    queue_packets=args.queue_packets,
                    vnf_processing_delay_ms=args.vnf_processing_delay_ms,
                ),
            )
            for candidate in feasible
        ]
        selectable = (
            [row for row in evaluated if row[1]["strict_sla"]]
            if args.sla_admission_mode == "hard"
            else evaluated
        )
        selected_row = min(
            selectable,
            key=lambda row: (
                row[1]["peak_link_utilization"],
                row[1]["max_delay_ms"] / max(1e-9, float(request.get("delay_bound_ms", math.inf))),
                row[1]["max_loss_rate"] / max(1e-12, float(request.get("packet_loss_bound", 1.0))),
                row[0].objective,
                row[0].candidate_id,
            ),
            default=None,
        )
        selected = selected_row[0] if selected_row is not None else None
        selected_qos = selected_row[1] if selected_row is not None else None
        slot = int(math.floor(arrival / args.slot_seconds))
        stats = slot_stats.setdefault(slot, {"arrivals": 0, "accepted": 0, "rejected": 0, "released": 0, "samples": 0, "cpu_util_sum": 0.0, "memory_util_sum": 0.0, "bandwidth_util_sum": 0.0, "active_sum": 0.0, "max_active_requests": 0.0, "max_cpu_util": 0.0, "max_memory_util": 0.0, "max_bandwidth_util": 0.0})
        stats["arrivals"] += 1
        stats["released"] += released
        base = {
            "request_id": int(request["id"]),
            "arrival_time": arrival,
            "leave_time": float(request.get("leave_time", arrival)),
            "lifetime": float(request.get("lifetime", 0.0)),
            "candidate_count": len(candidates),
            "candidate_generation": "online_complete_plan",
            "candidate_generation_ok": bool(candidates),
            "accepted": False,
            "reject_reason": (
                "no_resource_feasible_candidate"
                if not feasible
                else "no_sla_safe_candidate"
                if not selected
                else ""
            ),
            "selected_source": "",
            "estimated_delay_ms": "",
            "modeled_delay_ms": "",
            "modeled_jitter_ms": "",
            "modeled_loss_rate": "",
            "peak_link_utilization": "",
            "delay_sla_met": False,
            "jitter_sla_met": False,
            "loss_sla_met": False,
            "strict_sla": False,
            "cpu_cost": 0.0,
            "memory_cost": 0.0,
            "bandwidth_cost": 0.0,
            "active_after": len(active),
        }
        if selected is not None:
            footprint = selected.footprint
            commit = ledger.commit_exact([int(request["id"])], [[footprint]], [0], expected_version=snapshot.version)
            if commit.get("committed") and commit.get("accepted", 0) == 1:
                active.add(int(request["id"]))
                heapq.heappush(leave_heap, (float(request.get("leave_time", arrival)), int(request["id"])))
                stats["accepted"] += 1
                base.update({
                    "accepted": True,
                    "reject_reason": "",
                    "selected_source": selected.source,
                    "estimated_delay_ms": float(selected.metrics["estimated_delay_ms"]),
                    "cpu_cost": float(sum(footprint.cpu.values())),
                    "memory_cost": float(sum(footprint.memory.values())),
                    "bandwidth_cost": float(sum(footprint.bandwidth.values())),
                })
                candidate_sources[selected.source] += 1
                assert selected_qos is not None
                base.update({
                    "modeled_delay_ms": selected_qos["max_delay_ms"],
                    "modeled_jitter_ms": selected_qos["max_jitter_ms"],
                    "modeled_loss_rate": selected_qos["max_loss_rate"],
                    "peak_link_utilization": selected_qos["peak_link_utilization"],
                    "delay_sla_met": selected_qos["delay_sla_met"],
                    "jitter_sla_met": selected_qos["jitter_sla_met"],
                    "loss_sla_met": selected_qos["loss_sla_met"],
                    "strict_sla": selected_qos["strict_sla"],
                    "active_after": len(active),
                })
                delays_ms.append(float(selected_qos["max_delay_ms"]))
                jitters_ms.append(float(selected_qos["max_jitter_ms"]))
                losses.append(float(selected_qos["max_loss_rate"]))
            else:
                stats["rejected"] += 1
                base["reject_reason"] = str(commit.get("reason", "commit_failed"))
        else:
            stats["rejected"] += 1
        live_snapshot = ledger.snapshot()
        cpu_util = 1.0 - sum(live_snapshot.cpu_remaining.values()) / max(1e-9, sum(ledger.cpu_capacity.values()))
        mem_util = 1.0 - sum(live_snapshot.memory_remaining.values()) / max(1e-9, sum(ledger.memory_capacity.values()))
        bw_util = 1.0 - sum(live_snapshot.bandwidth_remaining.values()) / max(1e-9, sum(ledger.bandwidth_capacity.values()))
        stats["samples"] += 1
        stats["cpu_util_sum"] += cpu_util
        stats["memory_util_sum"] += mem_util
        stats["bandwidth_util_sum"] += bw_util
        stats["active_sum"] += len(active)
        stats["max_active_requests"] = max(stats["max_active_requests"], len(active))
        stats["max_cpu_util"] = max(stats["max_cpu_util"], cpu_util)
        stats["max_memory_util"] = max(stats["max_memory_util"], mem_util)
        stats["max_bandwidth_util"] = max(stats["max_bandwidth_util"], bw_util)
        request_rows.append(base)

    if requests:
        end_time = max(float(row.get("leave_time", row["arrival_time"])) for row in requests)
        release_until(end_time + 1e-9)
    for slot in sorted(slot_stats):
        stats = slot_stats[slot]
        samples = max(1, int(stats.pop("samples", 0)))
        cpu_util = float(stats.pop("cpu_util_sum", 0.0)) / samples
        mem_util = float(stats.pop("memory_util_sum", 0.0)) / samples
        bw_util = float(stats.pop("bandwidth_util_sum", 0.0)) / samples
        mean_active = float(stats.pop("active_sum", 0.0)) / samples
        slot_rows.append({"slot": slot, **stats, "cpu_utilization": cpu_util, "memory_utilization": mem_util, "bandwidth_utilization": bw_util, "mean_active_requests": mean_active, "energy_proxy": float(sum(ledger.cpu_capacity.values()) * (0.7 + 0.3 * cpu_util))})

    strict = sum(int(row["strict_sla"]) for row in request_rows)
    accepted = sum(int(row["accepted"]) for row in request_rows)
    summary = {
        "mode": "pure_discrete_event_sfc_simulation",
        "requests": len(request_rows),
        "accepted": accepted,
        "rejected": len(request_rows) - accepted,
        "acceptance_rate": accepted / max(1, len(request_rows)),
        "strict_sla": strict,
        "strict_sla_rate_all": strict / max(1, len(request_rows)),
        "strict_sla_rate_accepted": strict / max(1, accepted),
        "mean_delay_ms": statistics.fmean(delays_ms) if delays_ms else 0.0,
        "p95_delay_ms": percentile(delays_ms, 0.95),
        "mean_jitter_ms": statistics.fmean(jitters_ms) if jitters_ms else 0.0,
        "p95_jitter_ms": percentile(jitters_ms, 0.95),
        "mean_packet_loss_rate": statistics.fmean(losses) if losses else 0.0,
        "mean_cpu_cost_per_accepted": sum(float(row["cpu_cost"]) for row in request_rows) / max(1, accepted),
        "mean_memory_cost_per_accepted": sum(float(row["memory_cost"]) for row in request_rows) / max(1, accepted),
        "mean_bandwidth_cost_per_accepted": sum(float(row["bandwidth_cost"]) for row in request_rows) / max(1, accepted),
        "candidate_sources": dict(candidate_sources),
        "migration_count": 0,
        "sla_admission_mode": args.sla_admission_mode,
        "modeled_sla_dimensions": ["delay", "jitter", "loss"],
        "jitter_modeled": True,
        "queue_model": {
            "type": "M/M/1/K",
            "packet_size_bytes": args.packet_size_bytes,
            "queue_packets": args.queue_packets,
            "vnf_processing_delay_ms": args.vnf_processing_delay_ms,
        },
        "wall_runtime_seconds": time.perf_counter() - wall_started,
        "note": "Pure model-based simulation; no Mininet/Ryu or real packet measurements.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "request_metrics.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in request_rows) + "\n", encoding="utf-8")
    with (out / "request_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(request_rows[0]) if request_rows else ["request_id"])
        writer.writeheader(); writer.writerows(request_rows)
    with (out / "slot_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slot_rows[0]) if slot_rows else ["slot"])
        writer.writeheader(); writer.writerows(slot_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
