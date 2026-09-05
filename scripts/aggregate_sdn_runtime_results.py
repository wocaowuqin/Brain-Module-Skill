#!/usr/bin/env python3
"""Aggregate comparable metrics from multiple SDN runtime result files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="runtime result JSON paths or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for per_seed.csv, per_qos.csv, and hotspots.csv",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in patterns:
        path = resolve(value)
        if any(character in value for character in "*?["):
            anchor = ROOT if not Path(value).is_absolute() else Path(path.anchor)
            relative_pattern = str(path.relative_to(anchor))
            paths.update(candidate for candidate in anchor.glob(relative_pattern))
        elif path.exists():
            paths.add(path)
        else:
            raise FileNotFoundError(path)
    return sorted(paths)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def average(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def qos_name(result: dict[str, Any]) -> str:
    delay = float(result.get("delay_bound_ms") or math.inf)
    jitter = result.get("jitter_bound_ms")
    if delay <= 100.0 and jitter is not None:
        return "Q0"
    if delay <= 400.0 and jitter is not None:
        return "Q1"
    return "Q2"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def accepted_plan(plan: dict[str, Any]) -> bool:
    return bool(plan.get("accepted"))


def started_sender(result: dict[str, Any]) -> bool:
    return int(result.get("sent_packets") or 0) > 0


def event_timings(events: list[dict[str, Any]]) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.get("type") != "arrive":
            continue
        timing = event.get("deployment_timing") or {}
        for key in ("vnf_command_ms", "vnf_ready_wait_ms", "ryu_install_ms", "total_ms"):
            value = timing.get(key)
            if isinstance(value, (int, float)):
                metrics[key].append(float(value))
        planning = event.get("online_planning") or {}
        value = planning.get("total_ms")
        if isinstance(value, (int, float)):
            metrics["planning_ms"].append(float(value))
        lag = event.get("scheduler_lag_ms")
        if isinstance(lag, (int, float)):
            metrics["scheduler_lag_ms"].append(float(lag))
    return metrics


def summarize(
    path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("measurement_summary") or {}
    # Runtime replays conventionally store every payload as ``result.json``.
    # Use the run directory in that case; otherwise the filename remains the
    # natural experiment identifier.  This keeps multi-seed aggregation
    # traceable when several runs share the same JSON basename.
    experiment_name = path.parent.name if path.stem == "result" else path.stem
    seed = next(
        (part[4:] for part in experiment_name.split("_") if part.startswith("seed")),
        experiment_name,
    )

    plans = [plan for plan in payload.get("online_plans", []) if accepted_plan(plan)]
    sender_by_request = {
        int(row["request_id"]): row.get("result") or {}
        for row in payload.get("sender_results", [])
    }
    receiver_by_request: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("receiver_results", []):
        receiver_by_request[int(row["request_id"])].append(row.get("result") or {})
    arrival_by_request = {
        int(event["request_id"]): event
        for event in payload.get("events", [])
        if event.get("type") == "arrive"
    }
    plan_by_request = {int(plan["request_id"]): plan for plan in plans}

    started_ids = {
        request_id for request_id, result in sender_by_request.items()
        if started_sender(result)
    }
    strict_ids = {
        request_id for request_id, results in receiver_by_request.items()
        if results and all(bool(result.get("sla_met")) for result in results)
    }
    traversed_ids = {
        request_id for request_id in started_ids
        if receiver_by_request.get(request_id)
        and all(int(result.get("received_packets") or 0) > 0 for result in receiver_by_request[request_id])
    }

    actual_expected = 0
    actual_lost = 0
    for request_id in started_ids:
        for result in receiver_by_request.get(request_id, []):
            actual_expected += int(result.get("expected_packets") or 0)
            actual_lost += int(result.get("lost_packets") or 0)

    tree_edges = [len((plan.get("multicast") or {}).get("tree_edges") or []) for plan in plans]
    segment_hops = [
        sum(max(0, len(segment.get("path") or []) - 1) for segment in plan.get("segments") or [])
        for plan in plans
    ]
    destination_hops = [
        max(0, len(path_nodes) - 1)
        for plan in plans
        for path_nodes in ((plan.get("multicast") or {}).get("paths") or {}).values()
    ]
    timings = event_timings(payload.get("events", []))

    # A dry-run result contains the same top-level schema as an executed
    # replay, but it has no receiver measurements.  Keep the numeric counters
    # for diagnostics while exposing an explicit qualification flag so a
    # caller cannot mistake an unmeasured run for a measured strict-SLA rate.
    dry_run = bool(payload.get("dry_run", False))
    if dry_run:
        strict_rate_status = "dry_run_unqualified"
    elif receiver_by_request:
        strict_rate_status = "measured"
    else:
        strict_rate_status = "no_receiver_measurements"
    measurement_qualified = strict_rate_status == "measured"

    seed_row = {
        "experiment": experiment_name,
        "seed": seed,
        "dry_run": dry_run,
        "measurement_qualified": measurement_qualified,
        "strict_rate_status": strict_rate_status,
        "vnf_agent_packet_batch": (payload.get("probe_timing") or {}).get(
            "vnf_agent_packet_batch", 64
        ),
        "vnf_agent_q0_packet_batch": (payload.get("probe_timing") or {}).get(
            "vnf_agent_q0_packet_batch", 0
        ),
        "requests": int(payload.get("requests") or len(sender_by_request)),
        "planned_accepted": len(plans),
        "started": len(started_ids),
        "planned_to_started_rate": len(started_ids) / max(1, len(plans)),
        "sfc_traversed": len(traversed_ids),
        "strict_sla": len(strict_ids),
        "strict_rate": len(strict_ids) / max(1, int(payload.get("requests") or len(sender_by_request))),
        "started_sla_rate": len(strict_ids & started_ids) / max(1, len(started_ids)),
        "actual_lost_packets": actual_lost,
        "actual_expected_packets": actual_expected,
        "actual_traffic_loss_rate": actual_lost / max(1, actual_expected),
        "ledger_violations": int(summary.get("ledger_violations") or 0),
        "planner_rejected": int(summary.get("planner_rejected_requests") or 0),
        "deployment_attempted": int(summary.get("deployment_attempted_requests") or 0),
        "deployment_queue_wait_rejected": int(
            summary.get("deployment_queue_wait_rejected_requests") or 0
        ),
        "deployment_capacity_rejected": int(
            summary.get("deployment_capacity_rejected_requests") or 0
        ),
        "expired_before_deployment": int(
            summary.get("expired_before_deployment_requests") or 0
        ),
        "mean_delay_ms": float(summary.get("mean_delay_ms") or math.nan),
        "planning_mean_ms": average(timings["planning_ms"]),
        "deployment_mean_ms": average(timings["total_ms"]),
        "deployment_p95_ms": percentile(timings["total_ms"], 0.95),
        "vnf_command_mean_ms": average(timings["vnf_command_ms"]),
        "vnf_ready_mean_ms": average(timings["vnf_ready_wait_ms"]),
        "ryu_install_mean_ms": average(timings["ryu_install_ms"]),
        "scheduler_lag_mean_ms": average(timings["scheduler_lag_ms"]),
        "scheduler_lag_p95_ms": percentile(timings["scheduler_lag_ms"], 0.95),
        "mean_tree_edges": average([float(value) for value in tree_edges]),
        "mean_segment_hops": average([float(value) for value in segment_hops]),
        "mean_destination_hops": average([float(value) for value in destination_hops]),
        "max_destination_hops": max(destination_hops, default=0),
    }

    request_rows: list[dict[str, Any]] = []
    request_ids = sorted(set(sender_by_request) | set(receiver_by_request))
    for request_id in request_ids:
        results = receiver_by_request.get(request_id, [])
        sender = sender_by_request.get(request_id, {})
        event = arrival_by_request.get(request_id, {})
        timing = event.get("deployment_timing") or {}
        planning = event.get("online_planning") or {}
        plan = plan_by_request.get(request_id, {})
        paths = ((plan.get("multicast") or {}).get("paths") or {}).values()
        receiver_delays = [
            float(result["mean_delay_ms"])
            for result in results if result.get("mean_delay_ms") is not None
        ]
        dc_nodes = [
            int(placement["dc_node"])
            for placement in (plan.get("placement_by_vnf") or {}).values()
        ]
        request_rows.append({
            "experiment": experiment_name,
            "seed": seed,
            "request_id": request_id,
            "qos_class": qos_name(results[0]) if results else "unknown",
            "planned_accepted": bool(plan),
            "deployment_accepted": bool((event.get("controller") or {}).get("accepted")),
            "started": request_id in started_ids,
            "sfc_traversed": request_id in traversed_ids,
            "strict_sla": request_id in strict_ids,
            "delay_failed": bool(results) and not all(bool(result.get("delay_sla_met")) for result in results),
            "jitter_failed": bool(results) and not all(bool(result.get("jitter_sla_met")) for result in results),
            "loss_failed": bool(results) and not all(bool(result.get("loss_sla_met")) for result in results),
            "sender_status": sender.get("status", ""),
            "scheduler_lag_ms": event.get("scheduler_lag_ms", ""),
            "planning_ms": planning.get("total_ms", ""),
            "deployment_ms": timing.get("total_ms", ""),
            "vnf_command_ms": timing.get("vnf_command_ms", ""),
            "vnf_ready_ms": timing.get("vnf_ready_wait_ms", ""),
            "ryu_install_ms": timing.get("ryu_install_ms", ""),
            "receiver_mean_delay_ms": average(receiver_delays),
            "receiver_max_mean_delay_ms": max(receiver_delays, default=math.nan),
            "tree_edges": len((plan.get("multicast") or {}).get("tree_edges") or []),
            "segment_hops": sum(
                max(0, len(segment.get("path") or []) - 1)
                for segment in plan.get("segments") or []
            ),
            "mean_destination_hops": average([
                float(max(0, len(path_nodes) - 1)) for path_nodes in paths
            ]),
            "vnf_dc_nodes": ";".join(str(node) for node in dc_nodes),
            "unique_vnf_dcs": len(set(dc_nodes)),
        })

    qos_rows: list[dict[str, Any]] = []
    qos_request_ids: dict[str, set[int]] = defaultdict(set)
    for request_id, results in receiver_by_request.items():
        if results:
            qos_request_ids[qos_name(results[0])].add(request_id)
    for qos in sorted(qos_request_ids):
        ids = qos_request_ids[qos]
        started = ids & started_ids
        strict = ids & strict_ids
        delay_failed = jitter_failed = loss_failed = 0
        delays: list[float] = []
        for request_id in started:
            results = receiver_by_request[request_id]
            delay_failed += not all(bool(result.get("delay_sla_met")) for result in results)
            jitter_failed += not all(bool(result.get("jitter_sla_met")) for result in results)
            loss_failed += not all(bool(result.get("loss_sla_met")) for result in results)
            delays.extend(
                float(result["mean_delay_ms"])
                for result in results if result.get("mean_delay_ms") is not None
            )
        qos_rows.append({
            "experiment": experiment_name,
            "seed": seed,
            "qos_class": qos,
            "requests": len(ids),
            "started": len(started),
            "strict_sla": len(strict),
            "strict_rate_all": len(strict) / max(1, len(ids)),
            "strict_rate_started": len(strict) / max(1, len(started)),
            "started_delay_failed": delay_failed,
            "started_jitter_failed": jitter_failed,
            "started_loss_failed": loss_failed,
            "receiver_mean_delay_ms": average(delays),
            "receiver_p95_mean_delay_ms": percentile(delays, 0.95),
        })

    dc_counts: Counter[int] = Counter()
    for plan in plans:
        for placement in (plan.get("placement_by_vnf") or {}).values():
            dc_counts[int(placement["dc_node"])] += 1
    total_stages = sum(dc_counts.values())
    ordered_dc_counts = [count for _, count in dc_counts.most_common()]
    seed_row.update({
        "unique_vnf_dcs": len(dc_counts),
        "top1_dc_stage_share": (
            ordered_dc_counts[0] / total_stages if ordered_dc_counts else math.nan
        ),
        "top3_dc_stage_share": sum(ordered_dc_counts[:3]) / max(1, total_stages),
        "dc_stage_hhi": sum((count / max(1, total_stages)) ** 2 for count in ordered_dc_counts),
    })
    hotspot_rows = [
        {
            "experiment": experiment_name,
            "seed": seed,
            "dc_node": dc_node,
            "placed_vnf_stages": count,
            "stage_share": count / max(1, total_stages),
        }
        for dc_node, count in dc_counts.most_common()
    ]
    return seed_row, qos_rows, hotspot_rows, request_rows


def aggregate_row(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [float(row["strict_rate"]) for row in seed_rows]
    mean_rate = average(rates)
    std_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0
    # Student's t is appropriate for the small number of independent seeds.
    t_critical_95 = {
        1: math.nan,
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(len(rates), 1.96)
    margin = t_critical_95 * std_rate / math.sqrt(len(rates)) if len(rates) > 1 else math.nan
    total_requests = sum(int(row["requests"]) for row in seed_rows)
    total_strict = sum(int(row["strict_sla"]) for row in seed_rows)
    total_expected = sum(int(row["actual_expected_packets"]) for row in seed_rows)
    total_lost = sum(int(row["actual_lost_packets"]) for row in seed_rows)
    qualified = [row for row in seed_rows if bool(row.get("measurement_qualified", False))]
    statuses = Counter(str(row.get("strict_rate_status", "unknown")) for row in seed_rows)
    return {
        "seeds": len(seed_rows),
        "requests": total_requests,
        "measurement_qualified_seeds": len(qualified),
        "strict_rate_statuses": ";".join(
            f"{key}:{statuses[key]}" for key in sorted(statuses)
        ),
        "strict_sla": total_strict,
        "micro_strict_rate": total_strict / max(1, total_requests),
        "macro_strict_mean": mean_rate,
        "macro_strict_std": std_rate,
        "macro_95ci_low": mean_rate - margin,
        "macro_95ci_high": mean_rate + margin,
        "planned_accepted": sum(int(row["planned_accepted"]) for row in seed_rows),
        "started": sum(int(row["started"]) for row in seed_rows),
        "planned_to_started_rate": (
            sum(int(row["started"]) for row in seed_rows)
            / max(1, sum(int(row["planned_accepted"]) for row in seed_rows))
        ),
        "sfc_traversed": sum(int(row["sfc_traversed"]) for row in seed_rows),
        "deployment_queue_wait_rejected": sum(
            int(row["deployment_queue_wait_rejected"]) for row in seed_rows
        ),
        "actual_lost_packets": total_lost,
        "actual_expected_packets": total_expected,
        "actual_traffic_loss_rate": total_lost / max(1, total_expected),
    }


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.results)
    if not paths:
        raise SystemExit("no result files matched")
    seed_rows: list[dict[str, Any]] = []
    qos_rows: list[dict[str, Any]] = []
    hotspot_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    for path in paths:
        seed_row, seed_qos, seed_hotspots, seed_requests = summarize(path)
        seed_rows.append(seed_row)
        qos_rows.extend(seed_qos)
        hotspot_rows.extend(seed_hotspots)
        request_rows.extend(seed_requests)

    output_dir = resolve(args.output_dir)
    write_csv(output_dir / "per_seed.csv", seed_rows)
    write_csv(output_dir / "per_qos.csv", qos_rows)
    write_csv(output_dir / "hotspots.csv", hotspot_rows)
    write_csv(output_dir / "per_request.csv", request_rows)
    aggregate = aggregate_row(seed_rows)
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        **aggregate,
        "per_seed_csv": str(output_dir / "per_seed.csv"),
        "per_qos_csv": str(output_dir / "per_qos.csv"),
        "hotspots_csv": str(output_dir / "hotspots.csv"),
        "per_request_csv": str(output_dir / "per_request.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
