#!/usr/bin/env python3
"""Regression-check migration semantics on the original multicast SFT workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    ResourceSnapshot,
    footprint_feasible,
)
from core.marl.deployment_topk import validate_complete_plan  # noqa: E402
from core.marl.migration_candidates import (  # noqa: E402
    MigrationCandidateGenerator,
    plan_edge_multiplicity,
)
from core.marl.migration_scheduler import MigrationTask  # noqa: E402
from scripts.run_sdn_runtime_requests import (  # noqa: E402
    migrated_sfc_plan,
    migrated_sft_plan,
)
from sdn.migration_safety import make_before_break_compatible  # noqa: E402


DEFAULT_REQUESTS = (
    "data/sdn_runtime_requests/seed_7071_lifetime50node_rate8/requests.jsonl"
)
DEFAULT_PLANS = (
    "artifacts/runs/hrl/export_current_seed7071_rate8_100_mask_fix_v2/plans.jsonl"
)
DEFAULT_PROFILE = "sdn/topologies/us_backbone_28_bw90.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", default=DEFAULT_REQUESTS)
    parser.add_argument("--plans", default=DEFAULT_PLANS)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def full_snapshot(profile: dict[str, Any]) -> ResourceSnapshot:
    cpu = {int(node): 55.0 for node in profile["dc_nodes_1based"]}
    memory = {int(node): 45.0 for node in profile["dc_nodes_1based"]}
    bandwidth = {}
    default = float(profile.get("default_bandwidth_mbps", 90.0))
    for edge in profile["edges"]:
        capacity = float(edge.get("bandwidth_mbps", default))
        u, v = int(edge["u"]), int(edge["v"])
        bandwidth[(u, v)] = capacity
        bandwidth[(v, u)] = capacity
    return ResourceSnapshot(7071, cpu, memory, bandwidth)


def task_for(request: dict[str, Any], plan: dict[str, Any]) -> MigrationTask:
    stage = 1
    placement = plan["placement_by_vnf"][str(stage)]
    cpu = float(placement.get("cpu_units", request["cpu_origin"][stage]))
    memory = float(placement.get("memory_units", request["memory_origin"][stage]))
    state_size = 0.25 * (cpu + memory)
    relief = max(cpu / 55.0, memory / 45.0)
    return MigrationTask(
        task_id=f"r{int(request['id'])}-s{stage}",
        request_id=int(request["id"]),
        stage=stage,
        vnf_type=int(placement.get("vnf_type", request["vnf"][stage])),
        old_node=int(placement["dc_node"]),
        cpu=cpu,
        memory=memory,
        bandwidth_mbps=float(request["bw_origin"]),
        state_size_mb=state_size,
        current_utilization=0.86,
        predicted_utilization=0.90,
        utilization_relief=relief,
        sla_risk=0.0,
        remaining_lifetime_s=10.0,
        estimated_migration_ms=344.0 + 80.0 * state_size,
        priority=relief,
        created_at=float(request["arrival_time"]),
    )


def main() -> int:
    args = parse_args()
    if args.limit <= 0 or args.top_k <= 0:
        raise ValueError("limit and top-k must be positive")
    requests = read_jsonl(resolve(args.requests), args.limit)
    plans = read_jsonl(resolve(args.plans), args.limit)
    profile = json.loads(resolve(args.profile).read_text(encoding="utf-8"))
    requests_by_id = {int(request["id"]): request for request in requests}
    plans_by_id = {int(plan["request_id"]): plan for plan in plans}
    common = sorted(requests_by_id.keys() & plans_by_id.keys())[: args.limit]
    if len(common) != args.limit:
        raise ValueError(
            f"expected {args.limit} matched requests/plans, found {len(common)}"
        )

    snapshot = full_snapshot(profile)
    generator = MigrationCandidateGenerator(
        profile, max_candidates=args.top_k, cpu_capacity=55.0, memory_capacity=45.0
    )
    candidates_checked = 0
    requests_with_candidates = 0
    path_mismatches = 0
    output_errors = 0
    multicast_errors = 0
    footprint_errors = 0
    materialization_errors = 0
    make_before_break_safe = 0
    make_before_break_unsafe = 0
    candidate_differs_from_manual_shortest = 0

    for request_id in common:
        request = requests_by_id[request_id]
        current = plans_by_id[request_id]
        validate_complete_plan(current, request, profile)
        if len((current.get("multicast") or {}).get("paths", {})) != 5:
            multicast_errors += 1
        task = task_for(request, current)
        candidates = generator.generate(
            task,
            current,
            snapshot,
            delay_bound_ms=float(request["delay_bound_ms"]),
        )
        requests_with_candidates += int(bool(candidates))
        old_edges = plan_edge_multiplicity(current)
        for candidate in candidates:
            candidates_checked += 1
            candidate_plan = candidate.plan
            materialized = migrated_sft_plan(
                profile,
                current,
                task.stage,
                candidate.target_node,
                29999,
                candidate_plan=candidate_plan,
            )
            validate_complete_plan(materialized, request, profile)
            candidate_paths = {
                int(row["stage"]): [int(node) for node in row["path"]]
                for row in candidate_plan["segments"]
            }
            execution_paths = {
                int(row["stage"]): [int(node) for node in row["path"]]
                for row in materialized["segments"]
            }
            path_mismatches += int(candidate_paths != execution_paths)
            multicast_errors += int(
                materialized.get("multicast") != current.get("multicast")
            )
            for segment in materialized["segments"]:
                outputs = segment.get("switch_outputs") or {}
                for node in segment.get("path") or []:
                    output_errors += int(not outputs.get(str(int(node))))
            expected_bandwidth = {
                edge: float(max(0, count - old_edges.get(edge, 0)))
                * task.bandwidth_mbps
                for edge, count in plan_edge_multiplicity(candidate_plan).items()
                if count > old_edges.get(edge, 0)
            }
            footprint_errors += int(
                candidate.footprint.bandwidth != expected_bandwidth
                or not footprint_feasible(candidate.footprint, snapshot)
            )
            flow_safe, _ = make_before_break_compatible(current, materialized)
            make_before_break_safe += int(flow_safe)
            make_before_break_unsafe += int(not flow_safe)
            manual = migrated_sfc_plan(
                profile, current, task.stage, candidate.target_node, 29999
            )
            manual_paths = {
                int(row["stage"]): [int(node) for node in row["path"]]
                for row in manual["segments"]
            }
            candidate_differs_from_manual_shortest += int(
                candidate_paths != manual_paths
            )

    materialization_errors = (
        path_mismatches + output_errors + multicast_errors + footprint_errors
    )
    output = {
        "valid": (
            len(common) == args.limit
            and requests_with_candidates == args.limit
            and candidates_checked > 0
            and materialization_errors == 0
        ),
        "dataset": str(resolve(args.requests)),
        "plans": str(resolve(args.plans)),
        "profile": str(resolve(args.profile)),
        "requests_checked": len(common),
        "requests_with_candidates": requests_with_candidates,
        "candidates_checked": candidates_checked,
        "path_mismatches": path_mismatches,
        "switch_output_errors": output_errors,
        "multicast_errors": multicast_errors,
        "footprint_errors": footprint_errors,
        "make_before_break_safe": make_before_break_safe,
        "make_before_break_unsafe": make_before_break_unsafe,
        "candidate_differs_from_manual_shortest": (
            candidate_differs_from_manual_shortest
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
