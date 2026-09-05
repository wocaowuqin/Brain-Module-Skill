#!/usr/bin/env python3
"""Replay frozen HRL on one shared online ledger.

Unlike ``benchmark_online_hrl_parallel.py``, this benchmark never creates a
planner replica per worker and never clears the resource ledger between
requests. Requests are consumed in arrival order; the environment's
TimeSlotManager releases expired lifecycle records before the next request is
planned. The result measures online planning/admission under resource
competition. It does not claim measured Mininet/Ryu delay or packet loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--runtime-requests", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--failure-step-budget", type=int, default=240)
    parser.add_argument("--k-path-candidate-filter", action="store_true")
    parser.add_argument("--k-path-candidate-k", type=int, default=2)
    parser.add_argument("--macro-path-rollout", action="store_true")
    parser.add_argument("--fast-k-path-candidates", action="store_true")
    parser.add_argument("--completion-candidate-budget", type=int, default=0)
    parser.add_argument("--destination-beam-width", type=int, default=64)
    parser.add_argument(
        "--safe-dest-recovery",
        action="store_true",
        help="enable the planner's bounded destination-tree recovery",
    )
    parser.add_argument(
        "--planner-destinations",
        action="store_true",
        help="enable planner-side destination connection assistance",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _util(snapshot: dict[str, Any], key: str) -> float:
    value = snapshot.get(key) or {}
    return float(value.get("utilization", 0.0) or 0.0)


def main() -> int:
    args = parse_args()
    from sdn.online_hrl_planner import OnlineLegacyHRLPlanner
    requests_path = Path(args.runtime_requests).resolve()
    rows = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: (float(row.get("arrival_time", 0.0)), int(row["id"])))
    if args.max_requests > 0:
        rows = rows[: args.max_requests]
    if not rows:
        raise ValueError("runtime request stream is empty")

    planner = OnlineLegacyHRLPlanner(
        legacy_root=args.legacy_root,
        checkpoint=args.checkpoint,
        data_path=args.data,
        runtime_requests=requests_path,
        profile=args.profile,
        seed=args.seed,
        max_steps=args.max_steps,
        failure_step_budget=args.failure_step_budget if args.failure_step_budget > 0 else None,
        k_path_candidate_filter=args.k_path_candidate_filter,
        k_path_candidate_k=args.k_path_candidate_k,
        macro_path_rollout=args.macro_path_rollout,
        fast_k_path_candidates=args.fast_k_path_candidates,
        completion_candidate_budget=args.completion_candidate_budget,
        destination_beam_width=args.destination_beam_width,
        safe_dest_recovery=args.safe_dest_recovery,
        planner_destinations=args.planner_destinations,
        torch_threads=args.torch_threads,
        collect_timing=False,
        quiet=True,
    )
    expected = planner.request_order[: len(rows)]
    actual = [int(row["id"]) for row in rows]
    if actual != expected:
        raise RuntimeError(
            "runtime request order differs from planner order; use the same trace "
            f"(expected prefix {expected[:3]}, got {actual[:3]})"
        )

    started = time.perf_counter()
    decisions: list[dict[str, Any]] = []
    active_peak = 0
    accepted = 0
    planning_ms: list[float] = []
    strict_sla_planning_feasible = 0
    max_cpu = max_mem = max_bw = 0.0
    releases = 0
    for row in rows:
        plan = planner.plan_next(row)
        info = plan.get("online_planning") or {}
        snapshot = plan.get("resource_ledger") or {}
        rm = planner.env.resource_mgr
        active = len(getattr(rm.request_manager, "active_requests", {}))
        expired = int(getattr(rm.request_manager, "stats", {}).get("total_expired", 0))
        releases = max(releases, expired)
        accepted_now = bool(plan.get("accepted", False))
        accepted += int(accepted_now)
        planning_ms.append(float(info.get("total_ms", info.get("inference_ms", 0.0))))
        active_peak = max(active_peak, active)
        cpu_u, mem_u, bw_u = (_util(snapshot, key) for key in ("cpu", "memory", "bandwidth"))
        max_cpu, max_mem, max_bw = max(max_cpu, cpu_u), max(max_mem, mem_u), max(max_bw, bw_u)
        # This is a planning-side strict feasibility flag only. Runtime probe
        # delay/loss must be used for the actual strict SLA rate.
        strict_sla_planning_feasible += int(
            accepted_now and bool(plan.get("segments")) and bool(plan.get("multicast"))
        )
        decisions.append({
            "request_id": int(row["id"]),
            "arrival_time": float(row.get("arrival_time", 0.0)),
            "lifetime": float(row.get("lifetime", 0.0)),
            "accepted": accepted_now,
            "reason": plan.get("reason"),
            "planning_ms": float(info.get("total_ms", info.get("inference_ms", 0.0))),
            "active_requests": active,
            "expired_total": expired,
            "resource_ledger": snapshot,
        })

    elapsed = time.perf_counter() - started
    result = {
        "valid": True,
        "benchmark_scope": "shared_ledger_online_planning_only",
        "safe_for_online_deployment": False,
        "safe_for_online_planning_replay": True,
        "data_plane_sla_measured": False,
        "config": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data": str(Path(args.data).resolve()),
            "runtime_requests": str(requests_path),
            "profile": str(Path(args.profile).resolve()),
            "seed": args.seed,
            "max_requests": len(rows),
        },
        "summary": {
            "arrival_count": len(rows),
            "accepted_count": accepted,
            "online_acceptance_rate": accepted / len(rows),
            "planning_feasible_strict_sla_count": strict_sla_planning_feasible,
            "planning_feasible_strict_sla_rate": strict_sla_planning_feasible / len(rows),
            "elapsed_seconds": elapsed,
            "throughput_requests_per_second": len(rows) / max(elapsed, 1e-9),
            "mean_planning_ms": statistics.mean(planning_ms) if planning_ms else 0.0,
            "p95_planning_ms": sorted(planning_ms)[min(len(planning_ms) - 1, max(0, int(0.95 * len(planning_ms)) - 1))],
            "max_planning_ms": max(planning_ms) if planning_ms else 0.0,
            "active_request_peak": active_peak,
            "expired_request_count": releases,
            "max_cpu_utilization": max_cpu,
            "max_memory_utilization": max_mem,
            "max_bandwidth_utilization": max_bw,
        },
        "planner_metadata": planner.metadata(),
        "decisions": decisions,
    }
    planner.close()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
