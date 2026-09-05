#!/usr/bin/env python3
"""Run the real HRL + central-brain + WQMIX online planning loop.

The script measures planning/admission only.  Ryu/Mininet packet probes must
be attached separately before claiming data-plane SLA results.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.orchestration import (
    AgentRole,
    EventType,
    OrchestrationEvent,
    RuleBasedBrainAgent,
    SystemObservation,
)
from core.marl.orchestration.hrl_adapter import HRLPlannerAdapter, build_ledger_from_profile
from core.marl.online_parallel_pipeline import (
    CentralSharedLedgerPipeline,
    GuardedWQMIXRanker,
    ObjectiveCandidateRanker,
    WQMIXCandidateRanker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--hrl-checkpoint", required=True)
    parser.add_argument("--hrl-data", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", default="artifacts/runs/multiagent_online/result.json")
    parser.add_argument("--seed", type=int, default=7071)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--microbatch-ms", type=float, default=5.0)
    parser.add_argument("--max-agents", type=int, default=32)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--hrl-mode",
        choices=("legacy", "batched-policy"),
        default="legacy",
        help=(
            "legacy runs the stateful two-level rollout; batched-policy uses "
            "the loaded HRL encoder/policy heads to rank pure complete-plan "
            "candidates without advancing the legacy environment"
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("default", "fast", "ultra"),
        default="fast",
        help="candidate search budget; fast targets sub-33ms warm-path planning",
    )
    parser.add_argument("--prewarm-paths", type=int, default=8)
    parser.add_argument("--decoder-time-budget-ms", type=float, default=2.0)
    parser.add_argument("--decoder-top-r", type=int, default=0)
    parser.add_argument("--cpu-capacity", type=float, default=55.0)
    parser.add_argument("--memory-capacity", type=float, default=45.0)
    parser.add_argument("--bandwidth-utilization-limit", type=float, default=1.0)
    parser.add_argument("--wqmix-checkpoint", default=None)
    parser.add_argument("--wqmix-safety-guard", action="store_true")
    parser.add_argument("--hrl-max-steps", type=int, default=600)
    parser.add_argument("--hrl-failure-step-budget", type=int, default=0)
    parser.add_argument("--hrl-planner-destinations", action="store_true")
    parser.add_argument("--hrl-safe-dest-recovery", action="store_true")
    parser.add_argument("--hrl-k-path-candidate-filter", action="store_true")
    parser.add_argument("--hrl-k-path-candidate-k", type=int, default=2)
    parser.add_argument("--hrl-macro-path-rollout", action="store_true")
    parser.add_argument("--hrl-fast-k-path-candidates", action="store_true")
    parser.add_argument("--hrl-skip-high-topk", action="store_true")
    parser.add_argument("--hrl-completion-candidate-budget", type=int, default=0)
    parser.add_argument("--hrl-destination-beam-width", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--hrl-prefetch-workers", type=int, default=1)
    parser.add_argument(
        "--trace-lookahead-prefetch",
        action="store_true",
        help="replay-only optimization: prefetch the next trace batch before it arrives",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_batches(
    rows: Sequence[Mapping[str, Any]], microbatch_ms: float, max_agents: int
) -> list[list[Mapping[str, Any]]]:
    if microbatch_ms < 0.0 or max_agents <= 0:
        raise ValueError("microbatch-ms must be non-negative and max-agents positive")
    window = float(microbatch_ms) / 1000.0
    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    cutoff = 0.0
    for row in rows:
        arrival = float(row.get("arrival_time", 0.0))
        if not current:
            current = [row]
            cutoff = arrival + window
        elif len(current) >= max_agents or arrival > cutoff + 1e-12:
            batches.append(current)
            current = [row]
            cutoff = arrival + window
        else:
            current.append(row)
    if current:
        batches.append(current)
    return batches


def percentile95(values: Sequence[float]) -> float:
    """Return the nearest-rank P95 without under-reporting tiny samples."""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    args = parse_args()
    profile_path = resolve(args.profile).resolve()
    request_path = resolve(args.requests).resolve()
    hrl_data = resolve(args.hrl_data).resolve()
    legacy_root = resolve(args.legacy_root).resolve()
    checkpoint = resolve(args.hrl_checkpoint).resolve()
    for path in (profile_path, request_path, hrl_data, legacy_root, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    rows = sorted(
        read_jsonl(request_path),
        key=lambda row: (float(row.get("arrival_time", 0.0)), int(row["id"])),
    )
    if args.max_requests > 0:
        rows = rows[: args.max_requests]
    if not rows:
        raise ValueError("request stream is empty")

    from sdn.online_hrl_planner import OnlineLegacyHRLPlanner

    planner = OnlineLegacyHRLPlanner(
        legacy_root=legacy_root,
        checkpoint=checkpoint,
        data_path=hrl_data,
        runtime_requests=request_path,
        profile=profile_path,
        seed=args.seed,
        max_steps=args.hrl_max_steps,
        failure_step_budget=(args.hrl_failure_step_budget or None),
        planner_destinations=args.hrl_planner_destinations,
        safe_dest_recovery=args.hrl_safe_dest_recovery,
        k_path_candidate_filter=args.hrl_k_path_candidate_filter,
        k_path_candidate_k=max(1, int(args.hrl_k_path_candidate_k)),
        macro_path_rollout=args.hrl_macro_path_rollout,
        fast_k_path_candidates=args.hrl_fast_k_path_candidates,
        skip_high_topk=args.hrl_skip_high_topk,
        completion_candidate_budget=max(0, int(args.hrl_completion_candidate_budget)),
        destination_beam_width=max(1, int(args.hrl_destination_beam_width)),
        async_prefetch_workers=max(0, int(args.hrl_prefetch_workers)),
        torch_threads=args.torch_threads,
        quiet=True,
        collect_timing=False,
    )
    candidate_budgets = {
        "default": {"placement_beam": 16, "placement_chains": 12, "pool_limit": 64},
        "fast": {"placement_beam": 4, "placement_chains": 4, "pool_limit": 16},
        "ultra": {"placement_beam": 2, "placement_chains": 2, "pool_limit": 8},
    }
    if args.hrl_mode == "batched-policy":
        from core.marl.batched_hrl_adapter import BatchedHRLCandidateAdapter

        adapter = BatchedHRLCandidateAdapter(
            planner,
            profile,
            max_candidates=args.top_k,
            placement_beam=candidate_budgets[args.candidate_mode]["placement_beam"],
            placement_chains=candidate_budgets[args.candidate_mode]["placement_chains"],
            pool_limit=candidate_budgets[args.candidate_mode]["pool_limit"],
            cpu_capacity=args.cpu_capacity,
            memory_capacity=args.memory_capacity,
        )
    else:
        adapter = HRLPlannerAdapter(
            planner,
            profile,
            max_candidates=args.top_k,
            **candidate_budgets[args.candidate_mode],
        )
    adapter.prewarm_paths(max_paths=args.prewarm_paths)
    ledger = build_ledger_from_profile(
        profile,
        cpu_capacity=args.cpu_capacity,
        memory_capacity=args.memory_capacity,
        bandwidth_utilization_limit=args.bandwidth_utilization_limit,
    )
    ranker = getattr(adapter, "ranker", None) if args.hrl_mode == "batched-policy" else None
    if args.wqmix_checkpoint:
        learned = WQMIXCandidateRanker(resolve(args.wqmix_checkpoint), torch_threads=1)
        ranker = (
            GuardedWQMIXRanker(
                learned,
                ObjectiveCandidateRanker(),
                top_r=args.decoder_top_r or args.top_k,
                time_budget_ms=min(args.decoder_time_budget_ms, 1.0),
            )
            if args.wqmix_safety_guard
            else learned
        )
    pipeline = CentralSharedLedgerPipeline(
        ledger,
        adapter.candidate_factory,
        ranker=ranker,
        worker_count=args.worker_count,
        top_r=args.decoder_top_r or args.top_k,
        decoder_time_budget_ms=args.decoder_time_budget_ms,
    )
    brain = RuleBasedBrainAgent()
    batches = make_batches(rows, args.microbatch_ms, args.max_agents)
    started = time.perf_counter()
    accepted = 0
    request_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    try:
        for batch_id, batch in enumerate(batches, start=1):
            timestamp = max(float(row.get("arrival_time", 0.0)) for row in batch)
            # In causal online mode, a request is submitted to the HRL worker
            # only after it has entered this arrived batch.  Trace lookahead is
            # opt-in and must not be used for online SLA claims.
            if args.hrl_mode == "legacy" and args.hrl_prefetch_workers > 0:
                for request in batch:
                    adapter.prefetch_batch([request])
            observation = SystemObservation(
                timestamp=timestamp,
                snapshot_version=f"ledger-{ledger.snapshot().version}",
                metrics={
                    "active_allocations": len(ledger.allocations),
                    "ledger_version": ledger.snapshot().version,
                },
                queue_depth=len(batch),
            )
            commands = []
            for request in batch:
                event = OrchestrationEvent(
                    event_id=f"arrival-{int(request['id'])}",
                    event_type=EventType.REQUEST_ARRIVAL,
                    timestamp=timestamp,
                    request=request,
                    request_id=int(request["id"]),
                    deadline_ms=request.get("delay_bound_ms"),
                )
                command = brain.decide(event, observation)
                if AgentRole.DEPLOYMENT not in command.assigned_roles:
                    raise RuntimeError("central brain failed to assign deployment role")
                role_counts[AgentRole.DEPLOYMENT.value] += 1
                commands.append(command)
            preparation = (
                adapter.prepare_batch(batch)
                if args.hrl_mode == "legacy"
                else None
            )
            next_batch = batches[batch_id] if batch_id < len(batches) else None
            if (
                args.trace_lookahead_prefetch
                and args.hrl_mode == "legacy"
                and next_batch is not None
                and args.hrl_prefetch_workers > 0
            ):
                adapter.prefetch_batch(next_batch)
            result = pipeline.submit_batch(batch, timestamp=timestamp)
            accepted += result.accepted
            commit_by_id = {
                int(row["request_id"]): bool(row.get("accepted", False))
                for row in result.commit.get("results", [])
            }
            selected_by_id = {
                int(request_id): (int(action), plan)
                for request_id, action, plan in zip(
                    result.request_ids, result.actions, result.plans
                )
            }
            for request in batch:
                request_id = int(request["id"])
                action, plan = selected_by_id[request_id]
                row = {
                    "request_id": request_id,
                    "batch_id": batch_id,
                    "accepted": bool(commit_by_id.get(request_id, False)),
                    "candidate_action": action,
                    "plan": plan,
                    "arrival_time": float(request.get("arrival_time", 0.0)),
                }
                request_rows.append(row)
            batch_rows.append({
                "batch_id": batch_id,
                "request_ids": [int(row["id"]) for row in batch],
                "hrl_preparation": (
                    preparation.to_dict()
                    if preparation is not None
                    else {
                        "mode": "batched_policy_scoring",
                        "rollout_vectorized": False,
                        "request_ids": [int(row["id"]) for row in batch],
                    }
                ),
                "central_commands": [command.command_type.value for command in commands],
                "pipeline": result.to_dict(),
            })
            if args.hrl_mode == "legacy":
                adapter.forget_batch(batch)
    finally:
        pipeline.close()
        adapter.close()
    elapsed = time.perf_counter() - started
    final_time = max(
        float(row.get("leave_time", row.get("arrival_time", 0.0))) for row in rows
    )
    released_at_end = pipeline.release_expired(final_time + 1e-9)
    output = resolve(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "valid": True,
        "scope": (
            "stateful legacy HRL + parallel complete-plan candidates + central brain + exact shared ledger"
            if args.hrl_mode == "legacy"
            else "batched HRL policy scoring + parallel complete-plan candidates + central brain + exact shared ledger"
        ),
        "data_plane_sla_measured": False,
        "config": {
            "legacy_root": str(legacy_root),
            "hrl_checkpoint": str(checkpoint),
            "hrl_data": str(hrl_data),
            "requests": str(request_path),
            "profile": str(profile_path),
            "max_requests": len(rows),
            "microbatch_ms": args.microbatch_ms,
            "max_agents": args.max_agents,
            "worker_count": args.worker_count,
            "top_k": args.top_k,
            "candidate_mode": args.candidate_mode,
            "hrl_mode": args.hrl_mode,
            "hrl_prefetch_workers": args.hrl_prefetch_workers,
            "trace_lookahead_prefetch": bool(args.trace_lookahead_prefetch),
            "wqmix_checkpoint": str(resolve(args.wqmix_checkpoint).resolve()) if args.wqmix_checkpoint else None,
        },
        "summary": {
            "arrival_count": len(rows),
            "accepted_count": accepted,
            "rejected_count": len(rows) - accepted,
            "acceptance_rate": accepted / max(len(rows), 1),
            "batch_count": len(batches),
            "elapsed_seconds": elapsed,
            "throughput_requests_per_second": len(rows) / max(elapsed, 1e-9),
            "mean_batch_total_ms": statistics.mean(
                float(row["pipeline"]["timing_ms"]["total_ms"]) for row in batch_rows
            ) if batch_rows else 0.0,
            "p95_batch_total_ms": percentile95([
                float(row["pipeline"]["timing_ms"]["total_ms"])
                for row in batch_rows
            ]),
            "released_at_end": int(released_at_end),
            "remaining_ledger_allocations": len(ledger.allocations),
        },
        "multi_agent": {
            "brain": type(brain).__name__,
            "roles": dict(role_counts),
            "deployment_policy": "HRL candidates ranked by optional WQMIX then bounded joint decoder",
            "hard_constraints": "AtomicResourceLedger exact commit with lifecycle release",
        },
        "adapter": adapter.metadata(),
        "pipeline": pipeline.metadata(),
        "batches": batch_rows,
        "requests": request_rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
