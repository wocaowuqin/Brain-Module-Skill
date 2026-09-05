#!/usr/bin/env python3
"""Solve exact joint labels for deployment_topk_v3 micro-batches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.deployment_oracle import (  # noqa: E402
    DeploymentBatchOracle,
    validate_solution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset folder or batches.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--reject-penalty", type=float, default=1_000_000.0)
    parser.add_argument("--priority-penalty", type=float, default=1_000.0)
    parser.add_argument("--candidate-cost-scale", type=float, default=0.01)
    parser.add_argument("--sla-risk-scale", type=float, default=0.0)
    parser.add_argument("--sla-violation-penalty", type=float, default=0.0)
    parser.add_argument("--queue-safety-factor", type=float, default=1.0)
    parser.add_argument("--bandwidth-capacity-mbps", type=float, default=None)
    parser.add_argument("--q0-risk-weight", type=float, default=4.0)
    parser.add_argument("--q1-risk-weight", type=float, default=1.5)
    parser.add_argument("--q2-risk-weight", type=float, default=1.0)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def training_reward(batch, actions) -> float:
    accepted = 0
    delay_risk = 0.0
    bandwidth = 0.0
    new_instances = set()
    for agent, action in zip(batch["agents"], actions):
        candidate = agent["candidates"][int(action)]
        if candidate.get("source") == "reject":
            continue
        accepted += 1
        metrics = candidate.get("metrics", {})
        delay_risk += float(metrics.get("estimated_delay_ms", 0.0)) / max(
            float(metrics.get("delay_bound_ms", 1.0)), 1e-9
        )
        footprint = candidate.get("resource_footprint") or {}
        bandwidth += sum(float(row["mbps"]) for row in footprint.get("bandwidth", []))
        new_instances.update(
            (int(row["node"]), int(row["vnf_type"]))
            for row in footprint.get("vnf_instances", [])
        )
    rejected = len(batch["agents"]) - accepted
    return (
        10.0 * accepted
        - 10.0 * rejected
        - delay_risk
        - 0.001 * bandwidth
        - 0.05 * len(new_instances)
    )


def main() -> int:
    args = parse_args()
    data_path = args.data / "batches.jsonl" if args.data.is_dir() else args.data
    output_dir = args.output or data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = read_jsonl(data_path)
    oracle = DeploymentBatchOracle(
        reject_penalty=args.reject_penalty,
        priority_penalty=args.priority_penalty,
        candidate_cost_scale=args.candidate_cost_scale,
        sla_risk_scale=args.sla_risk_scale,
        sla_violation_penalty=args.sla_violation_penalty,
        queue_safety_factor=args.queue_safety_factor,
        bandwidth_capacity_mbps=args.bandwidth_capacity_mbps,
        q0_risk_weight=args.q0_risk_weight,
        q1_risk_weight=args.q1_risk_weight,
        q2_risk_weight=args.q2_risk_weight,
        time_limit_seconds=args.time_limit,
    )
    labels = []
    solve_times = []
    optimal_batches = 0
    invalid_solutions = 0
    original_accepted = 0
    oracle_accepted = 0
    changed_batches = 0
    improved_batches = 0
    changed_actions = 0

    for index, batch in enumerate(batches, 1):
        started = time.perf_counter()
        solution = oracle.solve(batch)
        solve_ms = (time.perf_counter() - started) * 1000.0
        solve_times.append(solve_ms)
        errors = validate_solution(batch, solution)
        invalid_solutions += int(bool(errors))
        optimal_batches += int(solution.optimal)
        original_actions = [int(agent["selected_action"]) for agent in batch["agents"]]
        original_sources = [
            str(agent["candidates"][action]["source"])
            for agent, action in zip(batch["agents"], original_actions)
        ]
        old_accepted = sum(source != "reject" for source in original_sources)
        original_accepted += old_accepted
        oracle_accepted += solution.accepted
        action_changes = sum(
            int(first != second) for first, second in zip(original_actions, solution.actions)
        )
        changed_actions += action_changes
        changed_batches += int(action_changes > 0)
        improved_batches += int(solution.accepted > old_accepted)
        labels.append({
            "schema_version": "3.0.0",
            "dataset_version": "deployment_topk_oracle_v3",
            "batch_id": int(batch["batch_id"]),
            "snapshot_version": int(batch["snapshot_version"]),
            "agent_count": int(batch["agent_count"]),
            "request_ids": [int(agent["request_id"]) for agent in batch["agents"]],
            "oracle_actions": solution.actions,
            "oracle_selected_sources": solution.selected_sources,
            "oracle_accepted": solution.accepted,
            "original_actions": original_actions,
            "original_accepted": old_accepted,
            "action_changes": action_changes,
            "acceptance_delta": solution.accepted - old_accepted,
            "training_reward": training_reward(batch, solution.actions),
            "solve_ms": solve_ms,
            "solver": {
                "name": "scipy.optimize.milp_highs",
                "status": solution.status,
                "optimal": solution.optimal,
                "objective": solution.objective,
                "mip_gap": solution.mip_gap,
                "message": solution.message,
            },
            "resource_usage": solution.resource_usage,
            "validation_errors": errors,
        })
        if index % 20 == 0:
            print(
                f"solved batches={index} optimal={optimal_batches}/{index} "
                f"mean_ms={mean(solve_times):.3f}"
            )

    sorted_times = sorted(solve_times)
    p95_index = max(0, math.ceil(0.95 * len(sorted_times)) - 1) if sorted_times else 0
    summary = {
        "valid": invalid_solutions == 0 and optimal_batches == len(batches),
        "dataset_version": "deployment_topk_oracle_v3",
        "batches": len(batches),
        "requests": sum(int(batch["agent_count"]) for batch in batches),
        "optimal_batches": optimal_batches,
        "invalid_solutions": invalid_solutions,
        "mean_solve_ms": mean(solve_times) if solve_times else 0.0,
        "p95_solve_ms": sorted_times[p95_index] if sorted_times else 0.0,
        "max_solve_ms": max(solve_times, default=0.0),
        "original_accepted": original_accepted,
        "oracle_accepted": oracle_accepted,
        "acceptance_gain": oracle_accepted - original_accepted,
        "changed_batches": changed_batches,
        "improved_batches": improved_batches,
        "changed_actions": changed_actions,
    }
    spec = {
        "schema_version": "3.0.0",
        "dataset_version": "deployment_topk_oracle_v3",
        "source_batches": str(data_path.resolve()),
        "solver": "scipy.optimize.milp (HiGHS)",
        "objective_order": [
            "maximize accepted request count",
            "prefer higher-priority requests when acceptance counts tie",
            "minimize QoS-weighted M/M/1-style delay risk",
            "minimize candidate deployment objective",
            "minimize new VNF instance resources",
        ],
        "shared_vnf_model": (
            "one activation variable per missing (node, vnf_type); maximum declared "
            "CPU and memory are reserved when multiple requests create it in one batch"
        ),
        "config": {
            "time_limit_seconds": args.time_limit,
            "reject_penalty": args.reject_penalty,
            "priority_penalty": args.priority_penalty,
            "candidate_cost_scale": args.candidate_cost_scale,
            "sla_risk_scale": args.sla_risk_scale,
            "sla_violation_penalty": args.sla_violation_penalty,
            "queue_safety_factor": args.queue_safety_factor,
            "bandwidth_capacity_mbps": args.bandwidth_capacity_mbps,
            "q0_risk_weight": args.q0_risk_weight,
            "q1_risk_weight": args.q1_risk_weight,
            "q2_risk_weight": args.q2_risk_weight,
        },
    }
    write_jsonl(output_dir / "oracle_labels.jsonl", labels)
    (output_dir / "oracle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "oracle_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir.resolve()), **summary}, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
