#!/usr/bin/env python3
"""Generate complete Top-K SFC deployment candidates on immutable snapshots."""

from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    ResourceFootprint,
    candidate_action_mask,
    deployment_candidate_features,
    resource_conflict,
)
from core.marl.deployment_topk import (  # noqa: E402
    CompletePlanCandidateGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--baseline-plans", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-agents", type=int, default=32)
    parser.add_argument("--microbatch-ms", type=float, default=5.0)
    parser.add_argument("--cpu-capacity", type=float, default=55.0)
    parser.add_argument("--memory-capacity", type=float, default=45.0)
    parser.add_argument(
        "--bandwidth-utilization-limit",
        type=float,
        default=1.0,
        help="hard admission fraction of each profile link capacity",
    )
    parser.add_argument("--stage-port-base", type=int, default=20000)
    parser.add_argument("--objective-delay-weight", type=float, default=1.0)
    parser.add_argument("--objective-cpu-weight", type=float, default=0.0)
    parser.add_argument("--objective-memory-weight", type=float, default=0.0)
    parser.add_argument("--objective-bandwidth-weight", type=float, default=0.03)
    parser.add_argument("--objective-pressure-weight", type=float, default=8.0)
    parser.add_argument(
        "--commit-ranking",
        choices=("objective", "pressure", "bandwidth"),
        default="objective",
        help="deterministic candidate order used only to advance the dataset ledger",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def request_batches(
    requests: Sequence[Mapping[str, Any]],
    microbatch_seconds: float,
    max_agents: int,
) -> Iterable[tuple[float, List[Mapping[str, Any]]]]:
    index = 0
    while index < len(requests):
        first_arrival = float(requests[index]["arrival_time"])
        cutoff = first_arrival + microbatch_seconds
        batch = []
        while index < len(requests) and len(batch) < max_agents:
            request = requests[index]
            if batch and float(request["arrival_time"]) > cutoff + 1e-12:
                break
            batch.append(request)
            index += 1
        yield cutoff, batch


def serialize_snapshot(snapshot) -> Dict[str, Any]:
    return {
        "version": int(snapshot.version),
        "cpu_remaining": {
            str(node): float(value) for node, value in sorted(snapshot.cpu_remaining.items())
        },
        "memory_remaining": {
            str(node): float(value)
            for node, value in sorted(snapshot.memory_remaining.items())
        },
        "bandwidth_remaining": [
            {"u": edge[0], "v": edge[1], "mbps": float(value)}
            for edge, value in sorted(snapshot.bandwidth_remaining.items())
        ],
        "vnf_instances": [
            {
                "node": key[0],
                "vnf_type": key[1],
                "cpu": float(value[0]),
                "memory": float(value[1]),
                "ref_count": int(value[2]),
            }
            for key, value in sorted(snapshot.vnf_instances.items())
        ],
    }


def request_features(request: Mapping[str, Any], decision_time: float) -> List[float]:
    return [
        float(request["bw_origin"]),
        float(sum(map(float, request["cpu_origin"]))),
        float(sum(map(float, request["memory_origin"]))),
        float(len(request["destination_dpids"])),
        float(len(request["vnf"])),
        float(request["lifetime"]),
        max(0.0, float(request["leave_time"]) - decision_time),
        float(request.get("delay_bound_ms") or 0.0),
        float(request.get("jitter_bound_ms") or 0.0),
        float(request.get("packet_loss_bound") or 0.0),
        float(request.get("priority") or 0.0),
        float(request.get("dscp") or 0.0),
    ]


def global_state_features(snapshot, batch: Sequence[Mapping[str, Any]]) -> List[float]:
    cpu = list(map(float, snapshot.cpu_remaining.values()))
    memory = list(map(float, snapshot.memory_remaining.values()))
    bandwidth = list(map(float, snapshot.bandwidth_remaining.values()))
    return [
        float(len(batch)),
        float(len(snapshot.vnf_instances)),
        min(cpu, default=0.0),
        mean(cpu) if cpu else 0.0,
        min(memory, default=0.0),
        mean(memory) if memory else 0.0,
        min(bandwidth, default=0.0),
        mean(bandwidth) if bandwidth else 0.0,
        mean(float(request["bw_origin"]) for request in batch),
        mean(float(request["lifetime"]) for request in batch),
        mean(len(request["destination_dpids"]) for request in batch),
        mean(len(request["vnf"]) for request in batch),
    ]


def sparse_conflicts(candidates, snapshot) -> List[Dict[str, Any]]:
    rows = []
    for first_agent in range(len(candidates)):
        for second_agent in range(first_agent + 1, len(candidates)):
            for first_index, first in enumerate(candidates[first_agent]):
                if first is None:
                    continue
                for second_index, second in enumerate(candidates[second_agent]):
                    if second is None:
                        continue
                    conflict = resource_conflict(first, second, snapshot)
                    if not any(conflict):
                        continue
                    rows.append({
                        "first_agent": first_agent,
                        "first_candidate": first_index,
                        "second_agent": second_agent,
                        "second_candidate": second_index,
                        "cpu": bool(conflict[0]),
                        "memory": bool(conflict[1]),
                        "bandwidth": bool(conflict[2]),
                    })
    return rows


def main() -> int:
    args = parse_args()
    if (
        args.top_k <= 0
        or args.max_agents <= 0
        or args.microbatch_ms < 0
        or not 0.0 < args.bandwidth_utilization_limit <= 1.0
    ):
        raise ValueError("top-k/max-agents must be positive and microbatch-ms nonnegative")
    requests = sorted(read_jsonl(args.requests), key=lambda row: float(row["arrival_time"]))
    if args.max_requests > 0:
        requests = requests[: args.max_requests]
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    baseline = {}
    if args.baseline_plans:
        baseline = {int(row["request_id"]): row for row in read_jsonl(args.baseline_plans)}

    dc_nodes = [int(value) for value in profile["dc_nodes_1based"]]
    default_bandwidth = float(profile.get("default_bandwidth_mbps", 90.0))
    bandwidth_capacity = {}
    for raw in profile["edges"]:
        u, v = int(raw["u"]), int(raw["v"])
        capacity = (
            float(raw.get("bandwidth_mbps", default_bandwidth))
            * args.bandwidth_utilization_limit
        )
        bandwidth_capacity[(u, v)] = capacity
        bandwidth_capacity[(v, u)] = capacity
    ledger = AtomicResourceLedger(
        cpu_capacity={node: float(args.cpu_capacity) for node in dc_nodes},
        memory_capacity={node: float(args.memory_capacity) for node in dc_nodes},
        bandwidth_capacity=bandwidth_capacity,
    )
    generator = CompletePlanCandidateGenerator(
        profile,
        max_candidates=args.top_k,
        stage_port_base=args.stage_port_base,
        objective_delay_weight=args.objective_delay_weight,
        objective_cpu_weight=args.objective_cpu_weight,
        objective_memory_weight=args.objective_memory_weight,
        objective_bandwidth_weight=args.objective_bandwidth_weight,
        objective_pressure_weight=args.objective_pressure_weight,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    leave_heap: List[tuple[float, int]] = []
    rows = []
    candidate_counts = []
    requests_with_four = 0
    requests_without_plan = 0
    legacy_candidates = 0
    conflict_pairs = 0
    fallback_commits = 0
    accepted_commits = 0
    rejected_commits = 0
    release_count = 0
    candidate_sources: Counter[str] = Counter()
    requests_without_plan_ids = []
    selected_plan_rows = []

    for batch_id, (decision_time, batch) in enumerate(
        request_batches(requests, args.microbatch_ms / 1000.0, args.max_agents), 1
    ):
        while leave_heap and leave_heap[0][0] <= decision_time + 1e-12:
            _, request_id = heapq.heappop(leave_heap)
            release_count += int(ledger.release(request_id))
        snapshot = ledger.snapshot()
        generated = [
            generator.generate(request, snapshot, baseline.get(int(request["id"])))
            for request in batch
        ]
        footprints = [
            [candidate.footprint for candidate in request_candidates] + [None]
            for request_candidates in generated
        ]
        features = deployment_candidate_features(footprints, snapshot)
        hard_masks = candidate_action_mask(footprints, snapshot)
        conflict_rows = sparse_conflicts(footprints, snapshot)
        conflict_pairs += len(conflict_rows)

        agent_rows = []
        rankings = []
        for agent_index, (request, request_candidates) in enumerate(zip(batch, generated)):
            plan_count = len(request_candidates)
            candidate_counts.append(plan_count)
            requests_with_four += int(plan_count >= 4)
            requests_without_plan += int(plan_count == 0)
            if plan_count == 0:
                requests_without_plan_ids.append(int(request["id"]))
            legacy_candidates += sum(
                int(candidate.source == "legacy_hrl") for candidate in request_candidates
            )
            candidate_sources.update(candidate.source for candidate in request_candidates)
            candidate_rows = []
            valid_actions = []
            for candidate_index, candidate in enumerate(request_candidates):
                candidate_features = features[agent_index][candidate_index] + [
                    candidate.metrics["estimated_delay_ms"],
                    candidate.metrics["delay_bound_ms"],
                    candidate.metrics["segment_hops"],
                    candidate.metrics["tree_edges"],
                    candidate.metrics["flowmod_estimate"],
                    candidate.metrics["peak_resource_pressure"],
                    candidate.objective,
                    0.0,
                ]
                delay_ok = (
                    candidate.metrics["estimated_delay_ms"]
                    <= candidate.metrics["delay_bound_ms"] + 1e-9
                )
                action_valid = bool(hard_masks[agent_index][candidate_index] and delay_ok)
                if action_valid:
                    valid_actions.append(candidate_index)
                candidate_rows.append(candidate.to_dict(candidate_features, action_valid))
            reject_index = plan_count
            reject_features = features[agent_index][reject_index] + [0.0] * 8
            candidate_rows.append({
                "candidate_id": f"r{int(request['id'])}-reject",
                "source": "reject",
                "action_valid": True,
                "objective": 1_000_000.0,
                "metrics": {},
                "resource_footprint": None,
                "candidate_features": reject_features,
                "plan": None,
            })
            valid_actions.append(reject_index)
            plan_actions = [index for index in valid_actions if index != reject_index]
            if args.commit_ranking == "pressure":
                plan_actions.sort(key=lambda index: (
                    request_candidates[index].metrics["peak_resource_pressure"],
                    request_candidates[index].metrics["bandwidth_mbps"],
                    request_candidates[index].metrics["cpu_units"]
                    + request_candidates[index].metrics["memory_units"],
                    request_candidates[index].metrics["estimated_delay_ms"],
                    request_candidates[index].objective,
                ))
            elif args.commit_ranking == "bandwidth":
                plan_actions.sort(key=lambda index: (
                    request_candidates[index].metrics["bandwidth_mbps"],
                    request_candidates[index].metrics["peak_resource_pressure"],
                    request_candidates[index].metrics["cpu_units"]
                    + request_candidates[index].metrics["memory_units"],
                    request_candidates[index].metrics["estimated_delay_ms"],
                    request_candidates[index].objective,
                ))
            commit_ranking = plan_actions + [reject_index]
            rankings.append(commit_ranking)
            agent_rows.append({
                "agent_index": agent_index,
                "request_id": int(request["id"]),
                "arrival_time": float(request["arrival_time"]),
                "leave_time": float(request["leave_time"]),
                "request_features": request_features(request, decision_time),
                "candidate_count": plan_count,
                "reject_action": reject_index,
                "valid_actions": valid_actions,
                "commit_ranking": commit_ranking,
                "candidates": candidate_rows,
            })

        edf_order = sorted(
            range(len(batch)),
            key=lambda index: (float(batch[index]["leave_time"]), int(batch[index]["id"])),
        )
        commit = ledger.commit_ranked(
            request_ids=[int(batch[index]["id"]) for index in edf_order],
            candidates=[footprints[index] for index in edf_order],
            rankings=[rankings[index] for index in edf_order],
            expected_version=snapshot.version,
        )
        commit_by_request = {
            int(item["request_id"]): item for item in commit["results"]
        }
        for agent in agent_rows:
            result = commit_by_request[agent["request_id"]]
            if result["accepted"]:
                accepted_commits += 1
                fallback_commits += int(result["attempts"] > 1)
                request = next(row for row in batch if int(row["id"]) == agent["request_id"])
                heapq.heappush(
                    leave_heap, (float(request["leave_time"]), int(request["id"]))
                )
                agent["selected_action"] = int(result["candidate_index"])
                selected_plan = json.loads(json.dumps(
                    agent["candidates"][agent["selected_action"]]["plan"]
                ))
                selected_plan["topk_selection"] = {
                    "dataset_version": "deployment_topk_v3",
                    "batch_id": batch_id,
                    "snapshot_version": int(snapshot.version),
                    "candidate_index": agent["selected_action"],
                    "candidate_id": agent["candidates"][agent["selected_action"]]["candidate_id"],
                    "source": agent["candidates"][agent["selected_action"]]["source"],
                    "bandwidth_utilization_limit": args.bandwidth_utilization_limit,
                }
                selected_plan_rows.append(selected_plan)
            else:
                rejected_commits += 1
                agent["selected_action"] = int(agent["reject_action"])
                selected_plan_rows.append({
                    "version": "deployment_topk_candidate_v3",
                    "request_id": int(agent["request_id"]),
                    "accepted": False,
                    "reason": str(result["reason"]),
                    "topk_selection": {
                        "dataset_version": "deployment_topk_v3",
                        "batch_id": batch_id,
                        "snapshot_version": int(snapshot.version),
                        "candidate_index": int(agent["reject_action"]),
                        "source": "reject",
                        "bandwidth_utilization_limit": args.bandwidth_utilization_limit,
                    },
                })
            agent["commit_result"] = result

        rows.append({
            "schema_version": "3.0.0",
            "dataset_version": "deployment_topk_v3",
            "batch_id": batch_id,
            "decision_time": float(decision_time),
            "snapshot_version": int(snapshot.version),
            "resource_snapshot": serialize_snapshot(snapshot),
            "global_state_features": global_state_features(snapshot, batch),
            "agent_count": len(batch),
            "max_agents": int(args.max_agents),
            "top_k": int(args.top_k),
            "agent_mask": [True] * len(batch) + [False] * (args.max_agents - len(batch)),
            "agents": agent_rows,
            "conflict_pairs": conflict_rows,
            "commit": commit,
        })
        if batch_id % 10 == 0:
            print(
                f"generated batches={batch_id} requests={sum(len(row['agents']) for row in rows)} "
                f"mean_candidates={mean(candidate_counts):.2f}"
            )

    summary = {
        "valid": True,
        "dataset_version": "deployment_topk_v3",
        "requests": len(requests),
        "batches": len(rows),
        "mean_agents_per_batch": mean(row["agent_count"] for row in rows) if rows else 0.0,
        "max_agents_in_batch": max((row["agent_count"] for row in rows), default=0),
        "mean_plan_candidates": mean(candidate_counts) if candidate_counts else 0.0,
        "requests_with_at_least_four_candidates": requests_with_four,
        "four_candidate_coverage": requests_with_four / max(1, len(requests)),
        "requests_without_plan_candidate": requests_without_plan,
        "requests_without_plan_ids": requests_without_plan_ids,
        "legacy_hrl_candidates": legacy_candidates,
        "candidate_sources": dict(sorted(candidate_sources.items())),
        "sparse_conflict_pairs": conflict_pairs,
        "accepted_commits": accepted_commits,
        "rejected_commits": rejected_commits,
        "fallback_commits": fallback_commits,
        "bandwidth_utilization_limit": args.bandwidth_utilization_limit,
        "candidate_objective_weights": generator.objective_weights,
        "commit_ranking": args.commit_ranking,
        "released_requests": release_count,
        "final_ledger_version": ledger.snapshot().version,
        "request_feature_dim": 12,
        "candidate_feature_dim": 24,
        "global_state_feature_dim": 12,
    }
    spec = {
        "schema_version": "3.1.0",
        "dataset_version": "deployment_topk_v3",
        "unit": "request micro-batch with complete Top-K deployment plans",
        "requests": str(args.requests.resolve()),
        "profile": str(args.profile.resolve()),
        "baseline_plans": str(args.baseline_plans.resolve()) if args.baseline_plans else None,
        "top_k": int(args.top_k),
        "max_agents": int(args.max_agents),
        "microbatch_ms": float(args.microbatch_ms),
        "resource_model": {
            "cpu_per_dc": float(args.cpu_capacity),
            "memory_per_dc": float(args.memory_capacity),
            "bandwidth_from_profile": True,
            "bandwidth_utilization_limit": args.bandwidth_utilization_limit,
            "candidate_objective_weights": generator.objective_weights,
            "commit_ranking": args.commit_ranking,
            "vnf_instance_sharing": (
                "same (node, vnf_type) shares one active instance; CPU and memory "
                "are released after the final request reference"
            ),
        },
        "candidate_sources": [
            "legacy_hrl",
            "beam_balanced",
            "beam_compact",
            "beam_spread",
            "beam_residual",
        ],
        "notes": [
            "All candidates for one micro-batch use the same immutable resource snapshot.",
            "The reject action is appended after the complete plan candidates.",
            "Candidate feature 23 is reserved and fixed to zero; candidate source is audit metadata only.",
            "Commit labels use deterministic EDF plus ranked feasible-candidate fallback; they are not global oracle labels.",
        ],
    }
    write_jsonl(args.output / "batches.jsonl", rows)
    write_jsonl(args.output / "selected_plans.jsonl", selected_plan_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "dataset_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve()), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
