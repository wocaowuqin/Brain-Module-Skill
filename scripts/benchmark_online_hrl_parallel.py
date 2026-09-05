#!/usr/bin/env python3
"""Measure isolated legacy-HRL replica throughput without shared commits.

This is a hardware scaling benchmark, not a deployment mode. Every process
owns an independent resource ledger and replays the same request prefix so its
decisions can be checked for determinism.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
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
    parser.add_argument("--episodes-per-worker", type=int, default=5)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", default=None)
    parser.add_argument("--k-path-candidate-filter", action="store_true")
    parser.add_argument("--k-path-candidate-k", type=int, default=4)
    parser.add_argument("--macro-path-rollout", action="store_true")
    parser.add_argument("--profile-functions", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--skip-high-topk", action="store_true")
    parser.add_argument("--failure-step-budget", type=int, default=0)
    parser.add_argument("--async-prefetch-workers", type=int, default=0)
    parser.add_argument("--fast-k-path-candidates", action="store_true")
    parser.add_argument("--completion-candidate-budget", type=int, default=0)
    parser.add_argument("--destination-beam-width", type=int, default=64)
    parser.add_argument(
        "--disable-timing",
        action="store_true",
        help="disable detailed per-call timing probes for a raw throughput measurement",
    )
    return parser.parse_args()


def worker_run(config: dict[str, Any], episodes: int, barrier) -> dict[str, Any]:
    from sdn.online_hrl_planner import OnlineLegacyHRLPlanner

    runtime_path = Path(config["runtime_requests"])
    requests = [
        json.loads(line)
        for line in runtime_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:episodes]
    planner = OnlineLegacyHRLPlanner(
        legacy_root=config["legacy_root"],
        checkpoint=config["checkpoint"],
        data_path=config["data"],
        runtime_requests=runtime_path,
        profile=config["profile"],
        seed=config["seed"],
        planner_destinations=True,
        torch_threads=int(config.get("torch_threads", 1)),
        max_steps=int(config.get("max_steps", 600)),
        skip_high_topk=bool(config.get("skip_high_topk", False)),
        quiet=True,
        k_path_candidate_filter=bool(config.get("k_path_candidate_filter", False)),
        k_path_candidate_k=int(config.get("k_path_candidate_k", 4)),
        macro_path_rollout=bool(config.get("macro_path_rollout", False)),
        failure_step_budget=(
            int(config["failure_step_budget"])
            if int(config.get("failure_step_budget", 0)) > 0
            else None
        ),
        async_prefetch_workers=int(config.get("async_prefetch_workers", 0)),
        fast_k_path_candidates=bool(config.get("fast_k_path_candidates", False)),
        completion_candidate_budget=int(
            config.get("completion_candidate_budget", 0)
        ),
        destination_beam_width=int(config.get("destination_beam_width", 64)),
        collect_timing=not bool(config.get("disable_timing", False)),
    )
    barrier.wait()
    started = time.time()
    timings = []
    algorithm_timings = []
    accepted = 0
    rejected_requests: list[dict[str, Any]] = []
    steps = []
    k_path_stats: dict[str, int] = {}
    macro_path_stats: dict[str, int] = {}
    runtime_timing: dict[str, dict[str, float]] = {}
    digest = hashlib.sha256()
    profiler = None
    if int(config.get("profile_functions", 0)) > 0:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    if int(config.get("async_prefetch_workers", 0)) > 0:
        planner.prefetch(requests[: int(config["async_prefetch_workers"])])
    for request_index, request in enumerate(requests):
        plan = planner.plan_next(request)
        next_prefetch_index = request_index + int(config.get("async_prefetch_workers", 0))
        if next_prefetch_index < len(requests):
            planner.prefetch([requests[next_prefetch_index]])
        timings.append(float(plan["online_planning"]["inference_ms"]))
        accepted += int(bool(plan.get("accepted", False)))
        if not bool(plan.get("accepted", False)):
            rejected_requests.append({
                "request_id": int(request["id"]),
                "reason": plan.get("reason"),
                "hrl_reason": (plan.get("hrl") or {}).get("reason")
                if isinstance(plan.get("hrl"), dict) else None,
                "steps": int(plan["online_planning"].get("steps", 0)),
            })
        steps.append(int(plan["online_planning"].get("steps", 0)))
        for key, value in (plan["online_planning"].get("k_path_candidate_stats") or {}).items():
            k_path_stats[key] = k_path_stats.get(key, 0) + int(value)
        for key, value in (plan["online_planning"].get("macro_path_rollout_stats") or {}).items():
            macro_path_stats[key] = macro_path_stats.get(key, 0) + int(value)
        for key, value in (plan["online_planning"].get("runtime_timing_ms") or {}).items():
            row = runtime_timing.setdefault(key, {"count": 0.0, "total_ms": 0.0})
            row["count"] += float(value.get("count", 0))
            row["total_ms"] += float(value.get("total_ms", 0.0))
        algorithm_timings.append({
            "request_id": int(request["id"]),
            **dict(plan["online_planning"].get("algorithm_timing") or {}),
        })
        comparable = {
            key: plan.get(key)
            for key in (
                "accepted",
                "reason",
                "chain_nodes",
                "placement_by_vnf",
                "segments",
                "multicast",
                "hrl",
            )
        }
        digest.update(
            json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    planner.close()
    function_profile = ""
    if profiler is not None:
        import io
        import pstats
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(int(config["profile_functions"]))
        function_profile = stream.getvalue()
    finished = time.time()
    return {
        "pid": __import__("os").getpid(),
        "episodes": len(requests),
        "initialization_ms": planner.initialization_ms,
        "planning_started_epoch": started,
        "planning_finished_epoch": finished,
        "planning_wall_seconds": finished - started,
        "mean_inference_ms": statistics.mean(timings),
        "max_inference_ms": max(timings),
        "accepted": accepted,
        "acceptance_rate": accepted / max(1, len(requests)),
        "rejected_requests": rejected_requests,
        "mean_steps": statistics.mean(steps) if steps else 0.0,
        "k_path_candidate_stats": k_path_stats,
        "macro_path_rollout_stats": macro_path_stats,
        "runtime_timing_ms": runtime_timing,
        "decision_sha256": digest.hexdigest(),
        "algorithm_timings": algorithm_timings,
        "function_profile": function_profile,
    }


def main() -> int:
    args = parse_args()
    if args.episodes_per_worker <= 0 or any(value <= 0 for value in args.workers):
        raise ValueError("episode and worker counts must be positive")
    config = {
        "legacy_root": str(Path(args.legacy_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "runtime_requests": str(Path(args.runtime_requests).resolve()),
        "profile": str(Path(args.profile).resolve()),
        "seed": int(args.seed),
        "k_path_candidate_filter": bool(args.k_path_candidate_filter),
        "k_path_candidate_k": int(args.k_path_candidate_k),
        "macro_path_rollout": bool(args.macro_path_rollout),
        "profile_functions": int(args.profile_functions),
        "torch_threads": int(args.torch_threads),
        "max_steps": int(args.max_steps),
        "skip_high_topk": bool(args.skip_high_topk),
        "failure_step_budget": int(args.failure_step_budget),
        "async_prefetch_workers": int(args.async_prefetch_workers),
        "fast_k_path_candidates": bool(args.fast_k_path_candidates),
        "completion_candidate_budget": max(
            0, int(args.completion_candidate_budget)
        ),
        "destination_beam_width": max(1, int(args.destination_beam_width)),
        "disable_timing": bool(args.disable_timing),
    }
    experiments = []
    context = multiprocessing.get_context("spawn")
    for worker_count in args.workers:
        with context.Manager() as manager:
            barrier = manager.Barrier(worker_count)
            wall_started = time.time()
            with ProcessPoolExecutor(
                max_workers=worker_count, mp_context=context
            ) as executor:
                futures = [
                    executor.submit(
                        worker_run, config, args.episodes_per_worker, barrier
                    )
                    for _ in range(worker_count)
                ]
                rows = [future.result() for future in futures]
            wall_finished = time.time()
        planning_start = min(row["planning_started_epoch"] for row in rows)
        planning_finish = max(row["planning_finished_epoch"] for row in rows)
        total_decisions = sum(row["episodes"] for row in rows)
        decision_hashes = {row["decision_sha256"] for row in rows}
        experiments.append(
            {
                "workers": worker_count,
                "episodes_per_worker": args.episodes_per_worker,
                "total_decisions": total_decisions,
                "end_to_end_wall_seconds": wall_finished - wall_started,
                "synchronized_planning_seconds": planning_finish - planning_start,
                "synchronized_throughput_rps": (
                    total_decisions / max(planning_finish - planning_start, 1e-9)
                ),
                "mean_worker_inference_ms": statistics.mean(
                    row["mean_inference_ms"] for row in rows
                ),
                "max_worker_inference_ms": max(
                    row["max_inference_ms"] for row in rows
                ),
                "rejected_requests": [
                    item for row in rows for item in row.get("rejected_requests", [])
                ],
                "replica_decisions_identical": len(decision_hashes) == 1,
                "workers_detail": rows,
            }
        )
    result = {
        "valid": True,
        "benchmark_scope": "isolated_planner_replicas_without_shared_commit",
        "safe_for_online_deployment": False,
        "config": config,
        "experiments": experiments,
    }
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
