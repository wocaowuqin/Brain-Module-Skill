#!/usr/bin/env python3
"""Build migration-WQMIX batches from real runtime plans plus hotspot injection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import ResourceSnapshot  # noqa: E402
from core.marl.joint_candidate_decoder import decode_joint_candidates  # noqa: E402
from core.marl.migration_candidates import (  # noqa: E402
    MigrationCandidateGenerator,
    candidate_mask,
    migration_batch_candidate_features,
    migration_global_state_features,
    migration_request_features,
)
from core.marl.migration_scheduler import MigrationTask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-report", action="append", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-report", type=int, default=64)
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cpu-capacity", type=float, default=55.0)
    parser.add_argument("--memory-capacity", type=float, default=45.0)
    parser.add_argument("--bandwidth-utilization-limit", type=float, default=0.90)
    parser.add_argument("--delay-bound-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def snapshot_payload(snapshot: ResourceSnapshot) -> dict[str, Any]:
    return {
        "version": snapshot.version,
        "cpu_remaining": {str(key): value for key, value in snapshot.cpu_remaining.items()},
        "memory_remaining": {str(key): value for key, value in snapshot.memory_remaining.items()},
        "bandwidth_remaining": {
            f"{edge[0]},{edge[1]}": value
            for edge, value in snapshot.bandwidth_remaining.items()
        },
    }


def runtime_examples(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    plans = {
        int(plan["request_id"]): plan
        for plan in report.get("online_plans", [])
        if bool(plan.get("accepted", True)) and plan.get("placement_by_vnf")
    }
    arrivals = {
        int(event["request_id"]): event
        for event in report.get("events", [])
        if event.get("type") == "arrive"
    }
    rows = []
    for request_id, plan in plans.items():
        placements = plan["placement_by_vnf"]
        indices = sorted(map(int, placements))
        internal = [stage for stage in indices if stage > 0 and stage + 1 in indices]
        event = arrivals.get(request_id, {})
        for stage in internal:
            placement = placements[str(stage)]
            admission = event.get("deployment_admission") or {}
            remaining_s = max(
                2.0,
                float(admission.get("remaining_lifetime_ms", 5000.0)) / 1000.0,
            )
            bandwidth = float(
                (event.get("bandwidth_admission") or {}).get("reserved_mbps", 5.0)
            )
            sla_risk = float(
                (event.get("online_planning") or {}).get("predicted_sla_risk", 0.5)
                or 0.5
            )
            rows.append({
                "request_id": request_id,
                "stage": stage,
                "plan": plan,
                "placement": placement,
                "bandwidth": bandwidth,
                "remaining_s": remaining_s,
                "sla_risk": min(2.0, max(0.0, sla_risk)),
            })
    return rows


def make_snapshot(
    profile: Mapping[str, Any],
    examples: list[dict[str, Any]],
    rng: random.Random,
    args: argparse.Namespace,
    version: int,
) -> ResourceSnapshot:
    dc_nodes = list(map(int, profile["dc_nodes_1based"]))
    cpu = {node: args.cpu_capacity * rng.uniform(0.45, 0.90) for node in dc_nodes}
    memory = {node: args.memory_capacity * rng.uniform(0.45, 0.90) for node in dc_nodes}
    for example in examples:
        old = int(example["placement"]["dc_node"])
        cpu[old] = min(cpu[old], args.cpu_capacity * rng.uniform(0.03, 0.14))
        memory[old] = min(memory[old], args.memory_capacity * rng.uniform(0.03, 0.14))
    default_bw = float(profile.get("default_bandwidth_mbps", 90.0))
    bandwidth: dict[tuple[int, int], float] = {}
    for edge in profile["edges"]:
        capacity = float(edge.get("bandwidth_mbps", default_bw))
        capacity *= args.bandwidth_utilization_limit
        remaining = capacity * rng.uniform(0.45, 0.95)
        u, v = int(edge["u"]), int(edge["v"])
        bandwidth[(u, v)] = remaining
        bandwidth[(v, u)] = remaining
    return ResourceSnapshot(version, cpu, memory, bandwidth)


def action_reward(agent: Mapping[str, Any], action: int) -> float:
    if int(action) == int(agent["reject_action"]):
        return -1.0 - 2.0 * float(agent["task"]["sla_risk"])
    metrics = agent["candidates"][int(action)]["metrics"]
    delay_ratio = float(metrics.get("delay_ratio", 0.0))
    sla = 5.0 if delay_ratio <= 1.0 + 1e-9 else -10.0 * delay_ratio
    return (
        sla
        + 2.0 * float(metrics["utilization_relief"])
        - 3.0 * max(0.0, float(metrics["projected_target_utilization"]) - 0.75)
        - 0.002 * float(metrics["migration_ms"])
    )


def build_record(
    trace_id: str,
    batch_id: int,
    examples: list[dict[str, Any]],
    profile: Mapping[str, Any],
    generator: MigrationCandidateGenerator,
    rng: random.Random,
    args: argparse.Namespace,
) -> dict[str, Any]:
    snapshot = make_snapshot(profile, examples, rng, args, batch_id)
    tasks: list[MigrationTask] = []
    generated = []
    for example in examples:
        placement = example["placement"]
        old_node = int(placement["dc_node"])
        current = rng.uniform(0.86, 0.96)
        predicted = min(1.1, current + rng.uniform(0.01, 0.08))
        cpu = float(placement.get("cpu_units", 1.0))
        memory = float(placement.get("memory_units", 1.0))
        relief = max(cpu / args.cpu_capacity, memory / args.memory_capacity)
        state_size = 0.25 * (cpu + memory)
        task = MigrationTask(
            task_id=f"r{example['request_id']}-s{example['stage']}",
            request_id=int(example["request_id"]),
            stage=int(example["stage"]),
            vnf_type=int(placement.get("vnf_type", example["stage"])),
            old_node=old_node,
            cpu=cpu,
            memory=memory,
            bandwidth_mbps=float(example["bandwidth"]),
            state_size_mb=state_size,
            current_utilization=current,
            predicted_utilization=predicted,
            utilization_relief=relief,
            sla_risk=float(example["sla_risk"]),
            remaining_lifetime_s=float(example["remaining_s"]),
            estimated_migration_ms=344.0 + state_size * 80.0,
            priority=(1.0 + 2.0 * float(example["sla_risk"])) * relief,
            created_at=float(batch_id),
        )
        candidates = generator.generate(
            task, example["plan"], snapshot, delay_bound_ms=args.delay_bound_ms
        )
        tasks.append(task)
        generated.append(candidates)

    nullable = [list(row) + [None] for row in generated]
    feature_rows = migration_batch_candidate_features(nullable, snapshot)
    agents = []
    footprints = []
    rankings = []
    masks = []
    for task, candidates, features in zip(tasks, generated, feature_rows):
        serialized = [candidate.to_dict() for candidate in candidates]
        reject_action = len(candidates)
        serialized.append({
            "candidate_id": f"{task.task_id}-reject",
            "target_node": None,
            "plan": None,
            "resource_footprint": None,
            "metrics": {},
            "objective": -1.0,
            "feasible": True,
            "rejection_reason": "",
        })
        mask = candidate_mask(list(candidates) + [None], snapshot)
        ranking = sorted(
            range(len(candidates)),
            key=lambda index: candidates[index].objective,
            reverse=True,
        ) + [reject_action]
        agents.append({
            "task": task.to_dict(),
            "request_features": migration_request_features(task),
            "candidates": serialized,
            "candidate_features": features,
            "action_mask": mask,
            "reject_action": reject_action,
        })
        footprints.append([candidate.footprint for candidate in candidates] + [None])
        rankings.append(ranking)
        masks.append(mask)
    decode = decode_joint_candidates(
        footprints,
        snapshot,
        rankings,
        reject_actions=[agent["reject_action"] for agent in agents],
        action_mask=masks,
        priorities=[(-task.priority, task.task_id) for task in tasks],
        top_r=min(4, args.top_k),
        time_budget_ms=10.0,
        max_greedy_evaluations=max(48, 8 * len(tasks)),
        max_repair_evaluations=max(16, 4 * len(tasks)),
    )
    for agent, action in zip(agents, decode.actions):
        agent["oracle_action"] = int(action)
    individual_top = [ranking[0] for ranking in rankings]
    joint_adjusted = sum(
        int(action != proposed)
        for action, proposed in zip(decode.actions, individual_top)
    )
    reward = sum(action_reward(agent, action) for agent, action in zip(agents, decode.actions))
    return {
        "schema": "migration_wqmix_batch_v1",
        "trace_id": trace_id,
        "batch_id": batch_id,
        "snapshot": snapshot_payload(snapshot),
        "state_features": migration_global_state_features(snapshot, tasks),
        "agents": agents,
        "oracle": decode.to_dict(),
        "individual_top_actions": individual_top,
        "joint_adjusted_tasks": joint_adjusted,
        "reward": reward,
    }


def main() -> int:
    args = parse_args()
    if args.max_agents <= 0 or args.top_k <= 0 or args.samples_per_report <= 0:
        raise ValueError("batch and sample sizes must be positive")
    profile_path = resolve(args.profile)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    generator = MigrationCandidateGenerator(
        profile,
        max_candidates=args.top_k,
        cpu_capacity=args.cpu_capacity,
        memory_capacity=args.memory_capacity,
    )
    records = []
    source_reports = []
    batch_id = 0
    for raw_path in args.runtime_report:
        report_path = resolve(raw_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        examples = runtime_examples(report)
        if not examples:
            raise ValueError(f"no internal-stage migration examples in {report_path}")
        source_reports.append(str(report_path.resolve()))
        trace_id = report_path.stem
        for _ in range(args.samples_per_report):
            batch_id += 1
            size = rng.randint(1, min(args.max_agents, len(examples)))
            selected = rng.sample(examples, size)
            records.append(build_record(
                trace_id, batch_id, selected, profile, generator, rng, args
            ))
    batches_path = output / "batches.jsonl"
    with batches_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    agents = sum(len(record["agents"]) for record in records)
    accepted = sum(int(record["oracle"]["accepted"]) for record in records)
    adjusted = sum(int(record["joint_adjusted_tasks"]) for record in records)
    conflict_batches = sum(int(record["joint_adjusted_tasks"] > 0) for record in records)
    spec = {
        "schema": "migration_wqmix_dataset_v1",
        "profile": str(profile_path.resolve()),
        "source_runtime_reports": source_reports,
        "seed": args.seed,
        "batches": len(records),
        "agents": agents,
        "oracle_accepted": accepted,
        "oracle_acceptance_rate": accepted / max(1, agents),
        "joint_adjusted_tasks": adjusted,
        "joint_adjustment_rate": adjusted / max(1, agents),
        "conflict_batches": conflict_batches,
        "conflict_batch_rate": conflict_batches / max(1, len(records)),
        "max_agents": args.max_agents,
        "max_candidates": args.top_k,
        "hotspots": "synthetic overload snapshots applied to real deployed SFC plans",
    }
    (output / "dataset_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
