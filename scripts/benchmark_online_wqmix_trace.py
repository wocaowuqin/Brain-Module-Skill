#!/usr/bin/env python3
"""Replay one request trace through online WQMIX planning without Mininet."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdn.online_wqmix_planner import OnlineWQMIXPlanner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--profile", default="sdn/topologies/us_backbone_28_bw90.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--diagnostics-csv",
        default=None,
        help="rejected-request diagnostics; defaults beside --output",
    )
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--microbatch-ms", type=float, default=20.0)
    parser.add_argument("--max-agents", type=int, default=32)
    parser.add_argument("--bandwidth-utilization-limit", type=float, default=0.9)
    parser.add_argument("--decoder-top-r", type=int, default=4)
    parser.add_argument("--decoder-time-budget-ms", type=float, default=2.0)
    parser.add_argument("--repair-missing-plans", action="store_true")
    parser.add_argument("--hybrid-least-loaded", action="store_true")
    parser.add_argument("--hybrid-rl-weight", type=float, default=0.25)
    parser.add_argument("--q0-sla-safety-margin", type=float, default=0.0)
    parser.add_argument("--q0-hard-sla-gate", action="store_true")
    parser.add_argument(
        "--sla-predictor",
        default=None,
        help="supervised runtime SLA predictor checkpoint (.pt)",
    )
    parser.add_argument("--sla-rank-weight", type=float, default=0.25)
    parser.add_argument("--sla-min-rerank-delta", type=float, default=0.05)
    parser.add_argument("--sla-max-ood-score", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile + 0.999999) - 1))
    return float(ordered[index])


def read_requests(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: (float(row["arrival_time"]), int(row["id"])))
    return rows[:maximum] if maximum > 0 else rows


def build_batches(
    requests: list[dict[str, Any]], microbatch_ms: float, max_agents: int
) -> list[list[dict[str, Any]]]:
    window = float(microbatch_ms) / 1000.0
    result = []
    index = 0
    while index < len(requests):
        cutoff = float(requests[index]["arrival_time"]) + window
        batch = []
        while index < len(requests) and len(batch) < max_agents:
            request = requests[index]
            if batch and float(request["arrival_time"]) > cutoff + 1e-12:
                break
            batch.append(request)
            index += 1
        result.append(batch)
    return result


def main() -> int:
    args = parse_args()
    if (
        args.max_requests < 0
        or args.microbatch_ms < 0.0
        or args.max_agents <= 0
        or not 0.0 < args.bandwidth_utilization_limit <= 1.0
        or args.decoder_top_r <= 0
        or args.decoder_time_budget_ms < 0.0
        or not 0.0 <= args.hybrid_rl_weight <= 1.0
        or args.q0_sla_safety_margin < 0.0
        or args.sla_rank_weight < 0.0
        or not 0.0 <= args.sla_min_rerank_delta <= 1.0
        or args.sla_max_ood_score <= 0.0
    ):
        raise ValueError("invalid benchmark configuration")
    requests_path = resolve(args.requests)
    requests = read_requests(requests_path, args.max_requests)
    batches = build_batches(requests, args.microbatch_ms, args.max_agents)
    planner = OnlineWQMIXPlanner(
        checkpoint=resolve(args.checkpoint),
        data_folder=resolve(args.data),
        profile=resolve(args.profile),
        microbatch_ms=args.microbatch_ms,
        bandwidth_utilization_limit=args.bandwidth_utilization_limit,
        decoder_top_r=args.decoder_top_r,
        decoder_time_budget_ms=args.decoder_time_budget_ms,
        repair_missing_plans=args.repair_missing_plans,
        hybrid_least_loaded=args.hybrid_least_loaded,
        hybrid_rl_weight=args.hybrid_rl_weight,
        q0_sla_safety_margin=args.q0_sla_safety_margin,
        q0_hard_sla_gate=args.q0_hard_sla_gate,
        sla_calibration=(resolve(args.sla_predictor) if args.sla_predictor else None),
        sla_calibration_rank_weight=args.sla_rank_weight,
        sla_min_rerank_probability_delta=args.sla_min_rerank_delta,
        sla_max_ood_score=args.sla_max_ood_score,
        device=args.device,
    )

    started = time.perf_counter()
    accepted_ids = []
    rejection_reasons: Counter[str] = Counter()
    rejection_diagnostic_categories: Counter[str] = Counter()
    rejection_diagnostic_rows: list[dict[str, Any]] = []
    selected_sources: Counter[str] = Counter()
    plan_records: list[dict[str, Any]] = []
    batch_sizes = []
    accepted_by_qos: Counter[str] = Counter()
    total_by_qos: Counter[str] = Counter()
    for batch in batches:
        batch_sizes.append(len(batch))
        plans = planner.plan_batch(batch)
        for request, plan in zip(batch, plans):
            planning = plan.get("online_planning") or {}
            plan_records.append(
                {
                    "request_id": int(request["id"]),
                    "accepted": bool(plan.get("accepted")),
                    "reason": plan.get("reason"),
                    "selected_candidate_index": planning.get(
                        "selected_candidate_index"
                    ),
                    "proposed_candidate_index": planning.get(
                        "proposed_candidate_index"
                    ),
                    "selection_mode": planning.get("selection_mode"),
                    "sla_candidate_diagnostics": planning.get(
                        "sla_candidate_diagnostics", []
                    ),
                }
            )
            qos = str(request.get("qos_class", "unknown"))
            total_by_qos[qos] += 1
            if plan.get("accepted"):
                request_id = int(request["id"])
                accepted_ids.append(request_id)
                accepted_by_qos[qos] += 1
                selection = plan.get("online_wqmix_selection") or {}
                selected_sources[str(selection.get("source", "unknown"))] += 1
            else:
                rejection_reasons[str(plan.get("reason", "unknown"))] += 1
                diagnostic = dict(
                    (plan.get("online_planning") or {}).get(
                        "rejection_diagnostics"
                    )
                    or {}
                )
                category = str(diagnostic.get("category", "unclassified"))
                rejection_diagnostic_categories[category] += 1
                rejection_diagnostic_rows.append(
                    {
                        "request_id": int(request["id"]),
                        "arrival_time": float(request["arrival_time"]),
                        "leave_time": float(request["leave_time"]),
                        "qos_class": qos,
                        "reason": str(plan.get("reason", "unknown")),
                        "category": category,
                        "batch_size": diagnostic.get("batch_size"),
                        "plan_candidates": diagnostic.get("plan_candidates"),
                        "locally_feasible_candidates": diagnostic.get(
                            "locally_feasible_candidates"
                        ),
                        "candidate_failure_counts": json.dumps(
                            diagnostic.get("candidate_failure_counts") or {},
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        "joint_conflict_counts": json.dumps(
                            diagnostic.get("joint_conflict_counts") or {},
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        "peak_cpu_demand_to_remaining": diagnostic.get(
                            "peak_cpu_demand_to_remaining"
                        ),
                        "peak_cpu_node": diagnostic.get("peak_cpu_node"),
                        "peak_memory_demand_to_remaining": diagnostic.get(
                            "peak_memory_demand_to_remaining"
                        ),
                        "peak_memory_node": diagnostic.get("peak_memory_node"),
                        "peak_bandwidth_demand_to_remaining": diagnostic.get(
                            "peak_bandwidth_demand_to_remaining"
                        ),
                        "peak_bandwidth_edge": diagnostic.get(
                            "peak_bandwidth_edge"
                        ),
                    }
                )
    elapsed = time.perf_counter() - started
    active_metadata = planner.metadata()
    for request_id in accepted_ids:
        planner.release(request_id)
    cleaned_metadata = planner.metadata()

    first_arrival = float(requests[0]["arrival_time"]) if requests else 0.0
    last_arrival = float(requests[-1]["arrival_time"]) if requests else 0.0
    trace_span = max(0.0, last_arrival - first_arrival)
    accepted = len(accepted_ids)
    output = resolve(args.output)
    diagnostics_output = (
        resolve(args.diagnostics_csv)
        if args.diagnostics_csv
        else output.with_name(f"{output.stem}_rejections.csv")
    )
    result = {
        "valid": True,
        "scope": "planning and resource-ledger replay only; Ryu/Mininet and strict SLA are not measured",
        "requests_file": str(requests_path.resolve()),
        "checkpoint": str(resolve(args.checkpoint).resolve()),
        "candidate_data": str(resolve(args.data).resolve()),
        "requests": len(requests),
        "accepted": accepted,
        "rejected": len(requests) - accepted,
        "planning_acceptance_rate": accepted / max(1, len(requests)),
        "trace_arrival_span_seconds": trace_span,
        "observed_trace_arrival_rate": len(requests) / max(trace_span, 1e-9),
        "microbatch_ms": args.microbatch_ms,
        "batches": len(batches),
        "mean_batch_size": sum(batch_sizes) / max(1, len(batch_sizes)),
        "max_batch_size": max(batch_sizes, default=0),
        "p95_batch_size": percentile([float(value) for value in batch_sizes], 0.95),
        "wall_planning_seconds": elapsed,
        "planning_throughput_requests_per_second": len(requests) / max(elapsed, 1e-9),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rejection_diagnostic_categories": dict(
            sorted(rejection_diagnostic_categories.items())
        ),
        "rejection_diagnostics_csv": str(diagnostics_output.resolve()),
        "selected_candidate_sources": dict(sorted(selected_sources.items())),
        "plan_records": plan_records,
        "qos": {
            qos: {
                "requests": total,
                "accepted": accepted_by_qos[qos],
                "acceptance_rate": accepted_by_qos[qos] / max(1, total),
            }
            for qos, total in sorted(total_by_qos.items())
        },
        "planner_before_cleanup": active_metadata,
        "planner_after_cleanup": cleaned_metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rejection_diagnostic_rows[0]) if rejection_diagnostic_rows else [
        "request_id",
        "arrival_time",
        "leave_time",
        "qos_class",
        "reason",
        "category",
    ]
    with diagnostics_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejection_diagnostic_rows)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
