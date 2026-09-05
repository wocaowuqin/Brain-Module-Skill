#!/usr/bin/env python3
"""Benchmark parallel complete-plan candidates with one shared ledger.

This is the safe bridge between full HRL plans and the future online WQMIX
ranker.  HRL plans in ``--baseline-plans`` are treated as optional candidates;
all footprints are rebuilt against the current snapshot before the central
decoder commits them.  Worker threads never own or mutate a resource ledger.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import AtomicResourceLedger
from core.marl.deployment_topk import CompletePlanCandidateGenerator
from core.marl.online_parallel_pipeline import (
    CentralSharedLedgerPipeline,
    GuardedWQMIXRanker,
    ObjectiveCandidateRanker,
    WQMIXCandidateRanker,
    complete_plan_candidate_factory,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _batches(
    rows: Sequence[Mapping[str, Any]], microbatch_ms: float, max_agents: int
) -> list[list[Mapping[str, Any]]]:
    if microbatch_ms < 0.0 or max_agents <= 0:
        raise ValueError("microbatch_ms must be non-negative and max_agents positive")
    window = float(microbatch_ms) / 1000.0
    output: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    cutoff = 0.0
    for row in rows:
        arrival = float(row.get("arrival_time", 0.0))
        if not current:
            current = [row]
            cutoff = arrival + window
            continue
        if len(current) >= max_agents or arrival > cutoff + 1e-12:
            output.append(current)
            current = [row]
            cutoff = arrival + window
        else:
            current.append(row)
    if current:
        output.append(current)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--baseline-plans", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--microbatch-ms", type=float, default=5.0)
    parser.add_argument("--max-agents", type=int, default=32)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--cpu-capacity", type=float, default=55.0)
    parser.add_argument("--memory-capacity", type=float, default=45.0)
    parser.add_argument("--bandwidth-utilization-limit", type=float, default=1.0)
    parser.add_argument("--decoder-time-budget-ms", type=float, default=2.0)
    parser.add_argument(
        "--decoder-top-r",
        type=int,
        default=0,
        help="number of deploy candidates considered by the decoder; 0 uses top-k",
    )
    parser.add_argument(
        "--wqmix-checkpoint",
        default=None,
        help="optional deployment WQMIX checkpoint used only for central ranking",
    )
    parser.add_argument(
        "--wqmix-safety-guard",
        action="store_true",
        help="choose WQMIX only when its bounded joint decode is no worse than objective order",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_path = _path(args.profile).resolve()
    request_path = _path(args.requests).resolve()
    output_path = _path(args.output).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    rows = sorted(_read_jsonl(request_path), key=lambda row: (float(row.get("arrival_time", 0.0)), int(row["id"])))
    if args.max_requests > 0:
        rows = rows[: args.max_requests]
    if not rows:
        raise ValueError("request stream is empty")
    if not 0.0 < args.bandwidth_utilization_limit <= 1.0:
        raise ValueError("bandwidth utilization limit must be in (0, 1]")

    baseline: dict[int, Mapping[str, Any]] = {}
    if args.baseline_plans:
        for row in _read_jsonl(_path(args.baseline_plans).resolve()):
            request_id = row.get("request_id", row.get("id"))
            if request_id is not None and bool(row.get("accepted", True)):
                baseline[int(request_id)] = row

    capacities: dict[tuple[int, int], float] = {}
    default_bandwidth = float(profile.get("default_bandwidth_mbps", 90.0))
    for raw in profile["edges"]:
        u, v = int(raw["u"]), int(raw["v"])
        cap = float(raw.get("bandwidth_mbps", default_bandwidth)) * args.bandwidth_utilization_limit
        capacities[(u, v)] = cap
        capacities[(v, u)] = cap
    dc_nodes = [int(value) for value in profile["dc_nodes_1based"]]
    ledger = AtomicResourceLedger(
        {node: float(args.cpu_capacity) for node in dc_nodes},
        {node: float(args.memory_capacity) for node in dc_nodes},
        capacities,
    )
    generator = CompletePlanCandidateGenerator(profile, max_candidates=args.top_k)
    generator.prewarm_paths(max_paths=8)
    baseline_provider = lambda request: baseline.get(int(request["id"]))
    factory = complete_plan_candidate_factory(
        generator, baseline_plan_provider=baseline_provider if baseline else None
    )
    ranker = None
    if args.wqmix_checkpoint:
        learned_ranker = WQMIXCandidateRanker(args.wqmix_checkpoint, torch_threads=1)
        ranker = (
            GuardedWQMIXRanker(
                learned_ranker,
                ObjectiveCandidateRanker(),
                top_r=(args.decoder_top_r if args.decoder_top_r > 0 else args.top_k),
                time_budget_ms=min(args.decoder_time_budget_ms, 1.0),
            )
            if args.wqmix_safety_guard
            else learned_ranker
        )
    pipeline = CentralSharedLedgerPipeline(
        ledger,
        factory,
        ranker=ranker,
        worker_count=args.worker_count,
        top_r=(args.decoder_top_r if args.decoder_top_r > 0 else args.top_k),
        decoder_time_budget_ms=args.decoder_time_budget_ms,
    )

    started = time.perf_counter()
    batch_rows = []
    accepted = 0
    worker_errors = Counter()
    for batch_id, batch in enumerate(_batches(rows, args.microbatch_ms, args.max_agents), start=1):
        result = pipeline.submit_batch(batch, timestamp=max(float(row.get("arrival_time", 0.0)) for row in batch))
        accepted += result.accepted
        for error in result.worker_errors.values():
            worker_errors[type(error).__name__] += 1
        batch_rows.append({
            "batch_id": batch_id,
            "request_ids": list(result.request_ids),
            "accepted": result.accepted,
            "rejected": result.rejected,
            "retry_count": result.retry_count,
            "timing_ms": dict(result.timing_ms),
            "decoder": dict(result.decoder),
            "worker_errors": dict(result.worker_errors),
        })
    elapsed = time.perf_counter() - started
    metadata = pipeline.metadata()
    pipeline.close()
    timing = [float(row["timing_ms"]["total_ms"]) for row in batch_rows]
    result = {
        "valid": True,
        "scope": "parallel complete-plan generation + central shared ledger; no Ryu/Mininet data-plane measurement",
        "config": {
            "profile": str(profile_path),
            "requests": str(request_path),
            "baseline_plans": str(_path(args.baseline_plans).resolve()) if args.baseline_plans else None,
            "max_requests": len(rows),
            "microbatch_ms": args.microbatch_ms,
            "max_agents": args.max_agents,
            "worker_count": args.worker_count,
            "top_k": args.top_k,
            "bandwidth_utilization_limit": args.bandwidth_utilization_limit,
            "wqmix_checkpoint": str(_path(args.wqmix_checkpoint).resolve()) if args.wqmix_checkpoint else None,
            "wqmix_safety_guard": bool(args.wqmix_safety_guard),
        },
        "summary": {
            "arrival_count": len(rows),
            "accepted_count": accepted,
            "rejected_count": len(rows) - accepted,
            "acceptance_rate": accepted / len(rows),
            "batch_count": len(batch_rows),
            "elapsed_seconds": elapsed,
            "throughput_requests_per_second": len(rows) / max(elapsed, 1e-9),
            "mean_batch_total_ms": statistics.mean(timing) if timing else 0.0,
            "p95_batch_total_ms": sorted(timing)[min(len(timing) - 1, max(0, int(0.95 * len(timing)) - 1))] if timing else 0.0,
            "worker_error_count": sum(worker_errors.values()),
            "worker_errors_by_type": dict(worker_errors),
            "ledger_version": int(ledger.snapshot().version),
        },
        "pipeline": metadata,
        "batches": batch_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
