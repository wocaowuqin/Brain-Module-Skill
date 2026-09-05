#!/usr/bin/env python3
"""Audit deployment_topk_v3 structure, diversity, and ledger consistency."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Dict, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    ResourceFootprint,
    footprint_feasible,
    resource_conflict,
)
from core.marl.deployment_topk import (  # noqa: E402
    deserialize_footprint,
    plan_signature,
    validate_complete_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset folder or batches.jsonl")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--min-four-coverage", type=float, default=0.90)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def snapshot_maps(raw: Mapping[str, Any]):
    return (
        {int(node): float(value) for node, value in raw["cpu_remaining"].items()},
        {int(node): float(value) for node, value in raw["memory_remaining"].items()},
        {
            (int(row["u"]), int(row["v"])): float(row["mbps"])
            for row in raw["bandwidth_remaining"]
        },
    )


def maps_close(first: Mapping[Any, float], second: Mapping[Any, float]) -> bool:
    return first.keys() == second.keys() and all(
        abs(float(first[key]) - float(second[key])) <= 1e-7 for key in first
    )


def footprint_equal(first: ResourceFootprint, second: ResourceFootprint) -> bool:
    return (
        maps_close(first.cpu, second.cpu)
        and maps_close(first.memory, second.memory)
        and maps_close(first.bandwidth, second.bandwidth)
        and first.vnf_instances == second.vnf_instances
    )


def main() -> int:
    args = parse_args()
    data_path = args.data / "batches.jsonl" if args.data.is_dir() else args.data
    rows = read_jsonl(data_path)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    dc_nodes = [int(value) for value in profile["dc_nodes_1based"]]
    spec_path = data_path.parent / "dataset_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    resource_model = spec.get("resource_model", {})
    cpu_capacity = float(resource_model.get("cpu_per_dc", 55.0))
    memory_capacity = float(resource_model.get("memory_per_dc", 45.0))
    default_bandwidth = float(profile.get("default_bandwidth_mbps", 90.0))
    bandwidth_utilization_limit = float(
        resource_model.get("bandwidth_utilization_limit", 1.0)
    )
    if not 0.0 < bandwidth_utilization_limit <= 1.0:
        raise ValueError("bandwidth_utilization_limit must be in (0, 1]")
    bandwidth_capacity = {}
    for raw in profile["edges"]:
        u, v = int(raw["u"]), int(raw["v"])
        capacity = (
            float(raw.get("bandwidth_mbps", default_bandwidth))
            * bandwidth_utilization_limit
        )
        bandwidth_capacity[(u, v)] = capacity
        bandwidth_capacity[(v, u)] = capacity
    ledger = AtomicResourceLedger(
        {node: cpu_capacity for node in dc_nodes},
        {node: memory_capacity for node in dc_nodes},
        bandwidth_capacity,
    )

    leave_heap = []
    request_count = 0
    plan_counts = []
    four_count = 0
    structural_errors = 0
    footprint_errors = 0
    duplicate_errors = 0
    mask_errors = 0
    snapshot_errors = 0
    commit_errors = 0
    feature_errors = 0
    conflict_errors = 0
    conflict_pairs = 0
    accepted = 0
    fallback = 0

    for expected_batch_id, row in enumerate(rows, 1):
        if row.get("dataset_version") != "deployment_topk_v3":
            raise ValueError(f"batch {expected_batch_id}: wrong dataset version")
        if int(row["batch_id"]) != expected_batch_id:
            raise ValueError("batch ids must be contiguous")
        decision_time = float(row["decision_time"])
        while leave_heap and leave_heap[0][0] <= decision_time + 1e-12:
            _, request_id = heapq.heappop(leave_heap)
            ledger.release(request_id)
        snapshot = ledger.snapshot()
        raw_snapshot = row["resource_snapshot"]
        raw_cpu, raw_memory, raw_bandwidth = snapshot_maps(raw_snapshot)
        raw_instances = {
            (int(item["node"]), int(item["vnf_type"])): (
                float(item["cpu"]),
                float(item["memory"]),
                int(item["ref_count"]),
            )
            for item in raw_snapshot.get("vnf_instances", [])
        }
        if (
            int(raw_snapshot["version"]) != snapshot.version
            or int(row["snapshot_version"]) != snapshot.version
            or not maps_close(raw_cpu, snapshot.cpu_remaining)
            or not maps_close(raw_memory, snapshot.memory_remaining)
            or not maps_close(raw_bandwidth, snapshot.bandwidth_remaining)
            or raw_instances != snapshot.vnf_instances
        ):
            snapshot_errors += 1

        agents = row["agents"]
        max_agents = int(row["max_agents"])
        if len(row["agent_mask"]) != max_agents:
            mask_errors += 1
        if row["agent_mask"] != [True] * len(agents) + [False] * (max_agents - len(agents)):
            mask_errors += 1
        request_count += len(agents)
        footprints_by_agent = []
        rankings = []
        requests_by_id: Dict[int, Mapping[str, Any]] = {}

        for agent_index, agent in enumerate(agents):
            request_id = int(agent["request_id"])
            request = {
                "id": request_id,
                "source_dpid": int(agent["candidates"][0]["plan"]["source_dpid"])
                if agent["candidate_count"] else -1,
                "destination_dpids": (
                    agent["candidates"][0]["plan"]["destination_dpids"]
                    if agent["candidate_count"] else []
                ),
                "vnf": [],
            }
            candidates = agent["candidates"]
            reject_index = int(agent["reject_action"])
            plan_rows = candidates[:reject_index]
            plan_counts.append(len(plan_rows))
            four_count += int(len(plan_rows) >= 4)
            if len(candidates) != reject_index + 1:
                structural_errors += 1
            if candidates[-1].get("source") != "reject" or candidates[-1].get("plan") is not None:
                structural_errors += 1
            signatures = set()
            footprints = []
            for candidate_index, candidate in enumerate(plan_rows):
                plan = candidate["plan"]
                placement_rows = sorted(
                    plan["placement_by_vnf"].values(),
                    key=lambda value: int(value["listen_port"]),
                )
                request["vnf"] = [int(value["vnf_type"]) for value in placement_rows]
                try:
                    validate_complete_plan(plan, request, profile)
                except (KeyError, TypeError, ValueError):
                    structural_errors += 1
                signature = plan_signature(plan)
                if signature in signatures:
                    duplicate_errors += 1
                signatures.add(signature)
                footprint = ResourceFootprint.from_sfc_plan(
                    plan,
                    float(sum(row["mbps"] for row in candidate["resource_footprint"]["bandwidth"]))
                    / max(1, sum(
                        max(0, len(segment["path"]) - 1) for segment in plan["segments"]
                    ) + len(plan["multicast"]["tree_edges"])),
                    snapshot,
                )
                serialized = deserialize_footprint(candidate["resource_footprint"])
                if not footprint_equal(footprint, serialized):
                    footprint_errors += 1
                footprints.append(serialized)
                expected_valid = footprint_feasible(serialized, snapshot)
                delay_ok = float(candidate["metrics"]["estimated_delay_ms"]) <= float(
                    candidate["metrics"]["delay_bound_ms"]
                ) + 1e-9
                if bool(candidate["action_valid"]) != bool(expected_valid and delay_ok):
                    mask_errors += 1
                if len(candidate.get("candidate_features", [])) != 24:
                    feature_errors += 1
            footprints.append(None)
            footprints_by_agent.append(footprints)
            if len(candidates[-1].get("candidate_features", [])) != 24:
                feature_errors += 1
            expected_valid_actions = [
                index for index, candidate in enumerate(candidates) if candidate["action_valid"]
            ]
            if list(map(int, agent["valid_actions"])) != expected_valid_actions:
                mask_errors += 1
            ranking = list(map(
                int, agent.get("commit_ranking", expected_valid_actions)
            ))
            if (
                sorted(ranking) != sorted(expected_valid_actions)
                or len(ranking) != len(set(ranking))
            ):
                mask_errors += 1
            rankings.append(ranking)
            requests_by_id[request_id] = agent

        for conflict in row.get("conflict_pairs", []):
            first = footprints_by_agent[int(conflict["first_agent"])][int(conflict["first_candidate"])]
            second = footprints_by_agent[int(conflict["second_agent"])][int(conflict["second_candidate"])]
            actual = resource_conflict(first, second, snapshot)
            expected = (
                float(bool(conflict["cpu"])),
                float(bool(conflict["memory"])),
                float(bool(conflict["bandwidth"])),
            )
            if actual != expected or not any(actual):
                conflict_errors += 1
            conflict_pairs += 1

        edf_order = sorted(
            range(len(agents)),
            key=lambda index: (
                float(agents[index]["leave_time"]), int(agents[index]["request_id"])
            ),
        )
        reproduced = ledger.commit_ranked(
            [int(agents[index]["request_id"]) for index in edf_order],
            [footprints_by_agent[index] for index in edf_order],
            [rankings[index] for index in edf_order],
            expected_version=snapshot.version,
        )
        original_results = {
            int(item["request_id"]): item for item in row["commit"]["results"]
        }
        for result in reproduced["results"]:
            request_id = int(result["request_id"])
            original = original_results.get(request_id)
            if original is None or (
                bool(result["accepted"]) != bool(original["accepted"])
                or int(result["candidate_index"]) != int(original["candidate_index"])
                or str(result["reason"]) != str(original["reason"])
            ):
                commit_errors += 1
            agent = requests_by_id[request_id]
            if result["accepted"]:
                accepted += 1
                fallback += int(result["attempts"] > 1)
                heapq.heappush(
                    leave_heap, (float(agent["leave_time"]), request_id)
                )

    coverage = four_count / max(1, request_count)
    report = {
        "valid": not any((
            structural_errors,
            footprint_errors,
            duplicate_errors,
            mask_errors,
            snapshot_errors,
            commit_errors,
            feature_errors,
            conflict_errors,
        )) and coverage >= float(args.min_four_coverage),
        "batches": len(rows),
        "requests": request_count,
        "mean_plan_candidates": mean(plan_counts) if plan_counts else 0.0,
        "four_candidate_coverage": coverage,
        "sparse_conflict_pairs": conflict_pairs,
        "accepted_commits": accepted,
        "fallback_commits": fallback,
        "errors": {
            "structure": structural_errors,
            "footprint": footprint_errors,
            "duplicates": duplicate_errors,
            "mask": mask_errors,
            "snapshot": snapshot_errors,
            "commit": commit_errors,
            "feature_dim": feature_errors,
            "conflict": conflict_errors,
        },
    }
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
