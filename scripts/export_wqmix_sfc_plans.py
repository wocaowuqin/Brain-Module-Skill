#!/usr/bin/env python3
"""Export WQMIX-selected complete SFC plans for the existing Ryu executor."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import heapq
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    BatchCandidateQNetwork,
    ResourceFootprint,
)
from core.marl.deployment_dataset import (  # noqa: E402
    DeploymentOracleDataset,
    FeatureNormalizer,
    load_labeled_batches,
)
from core.marl.deployment_oracle import candidate_footprints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="one deployment_v3 dataset folder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--bandwidth-utilization-limit",
        type=float,
        default=1.0,
        help="fraction of each profile link that the authoritative ledger may reserve",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_authoritative_ledger(
    data_folder: Path,
    bandwidth_utilization_limit: float,
) -> tuple[AtomicResourceLedger, dict[str, Any], Path]:
    if not 0.0 < bandwidth_utilization_limit <= 1.0:
        raise ValueError("bandwidth utilization limit must be in (0, 1]")
    spec_path = data_folder / "dataset_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    profile_path = Path(spec["profile"])
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    resource_model = spec["resource_model"]
    cpu_per_dc = float(resource_model["cpu_per_dc"])
    memory_per_dc = float(resource_model["memory_per_dc"])
    dc_nodes = [int(node) for node in profile["dc_nodes_1based"]]
    default_bandwidth = float(profile.get("default_bandwidth_mbps", 0.0))
    bandwidth_capacity: dict[tuple[int, int], float] = {}
    for row in profile["edges"]:
        edge_capacity = float(row.get("bandwidth_mbps", default_bandwidth))
        edge_capacity *= bandwidth_utilization_limit
        u, v = int(row["u"]), int(row["v"])
        bandwidth_capacity[(u, v)] = edge_capacity
        bandwidth_capacity[(v, u)] = edge_capacity
    return (
        AtomicResourceLedger(
            {node: cpu_per_dc for node in dc_nodes},
            {node: memory_per_dc for node in dc_nodes},
            bandwidth_capacity,
        ),
        spec,
        profile_path.resolve(),
    )


def live_plan_rankings(
    q_values: torch.Tensor,
    agents: list[dict[str, Any]],
) -> list[list[int]]:
    rankings: list[list[int]] = []
    for agent_index, agent in enumerate(agents):
        plan_actions = [
            candidate_index
            for candidate_index, candidate in enumerate(agent["candidates"])
            if bool(candidate.get("action_valid")) and candidate.get("plan") is not None
        ]
        plan_actions.sort(
            key=lambda candidate_index: (
                -float(q_values[agent_index, candidate_index].item()),
                candidate_index,
            )
        )
        rankings.append(plan_actions)
    return rankings


def audit_exported_bandwidth(
    rows: list[dict[str, Any]],
    requests_path: Path,
    capacity: dict[tuple[int, int], float],
) -> dict[str, Any]:
    requests = {
        int(row["id"]): row
        for row in (
            json.loads(line)
            for line in requests_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    events: list[tuple[float, int, tuple[int, int], float]] = []
    for plan in rows:
        if not bool(plan.get("accepted")):
            continue
        request = requests[int(plan["request_id"])]
        footprint = ResourceFootprint.from_sfc_plan(
            plan,
            float(request["bw_origin"]),
        )
        for edge, amount in footprint.bandwidth.items():
            events.append((float(request["arrival_time"]), 1, edge, float(amount)))
            events.append((float(request["leave_time"]), 0, edge, -float(amount)))
    current: Counter[tuple[int, int]] = Counter()
    peak: Counter[tuple[int, int]] = Counter()
    for _, _, edge, delta in sorted(events):
        current[edge] += delta
        peak[edge] = max(peak[edge], current[edge])
    peak_rows = sorted(
        (
            {
                "u": int(edge[0]),
                "v": int(edge[1]),
                "reserved_mbps": float(amount),
                "capacity_mbps": float(capacity.get(edge, 0.0)),
            }
            for edge, amount in peak.items()
        ),
        key=lambda row: (-row["reserved_mbps"], row["u"], row["v"]),
    )
    violations = [
        row
        for row in peak_rows
        if row["reserved_mbps"] > row["capacity_mbps"] + 1e-9
    ]
    return {
        "valid": not violations,
        "max_reserved_mbps": max(
            (row["reserved_mbps"] for row in peak_rows), default=0.0
        ),
        "violations": len(violations),
        "violation_edges": violations[:10],
        "peak_edges": peak_rows[:10],
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = resolve(args.checkpoint)
    data_folder = resolve(args.data)
    output_path = resolve(args.output)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ledger, dataset_spec, profile_path = build_authoritative_ledger(
        data_folder,
        float(args.bandwidth_utilization_limit),
    )
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    model = BatchCandidateQNetwork(
        int(checkpoint["request_dim"]),
        int(checkpoint["candidate_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    records = load_labeled_batches([data_folder])
    dataset = DeploymentOracleDataset(
        records,
        normalizer,
        int(checkpoint["max_agents"]),
        int(checkpoint["max_candidates"]),
    )
    rows: list[dict[str, Any]] = []
    selected = 0
    rejected = 0
    fallback = 0
    releases = 0
    rejection_reasons: Counter[str] = Counter()
    leave_heap: list[tuple[float, int]] = []
    peak_bandwidth: dict[tuple[int, int], float] = {}
    processed_requests = 0
    with torch.inference_mode():
        for index, record in enumerate(records):
            if args.max_batches and index >= args.max_batches:
                break
            item = dataset[index]
            q_values = model(
                item["request_observations"].unsqueeze(0).to(device),
                item["candidate_features"].unsqueeze(0).to(device),
                item["agent_mask"].unsqueeze(0).to(device),
            )[0]
            batch = record["batch"]
            # Precomputed plans are replayed from each request's trace arrival,
            # not from the later micro-batch cutoff.  Retain allocations that
            # overlap any arrival in this batch so the exported trace cannot
            # transiently overbook a link inside the collection window.
            admission_time = min(float(agent["arrival_time"]) for agent in batch["agents"])
            while leave_heap and leave_heap[0][0] <= admission_time + 1e-12:
                _, request_id = heapq.heappop(leave_heap)
                releases += int(ledger.release(request_id))

            agents = list(batch["agents"])
            if args.max_requests:
                agents = agents[: max(0, args.max_requests - processed_requests)]
            if not agents:
                break
            rankings = live_plan_rankings(q_values, agents)
            footprints = candidate_footprints(batch)[: len(agents)]
            commit = ledger.commit_ranked(
                [int(agent["request_id"]) for agent in agents],
                footprints,
                rankings,
            )
            for agent, ranking, result in zip(agents, rankings, commit["results"]):
                proposed_action = ranking[0] if ranking else int(agent["reject_action"])
                if result["accepted"]:
                    action = int(result["candidate_index"])
                    candidate = agent["candidates"][action]
                    exported = copy.deepcopy(candidate["plan"])
                    used_fallback = action != proposed_action
                    fallback += int(used_fallback)
                    exported["accepted"] = True
                    exported["wqmix_selection"] = {
                        "algorithm": "deployment_candidate_wqmix",
                        "checkpoint": str(checkpoint_path.resolve()),
                        "batch_id": int(batch["batch_id"]),
                        "dataset_snapshot_version": int(batch["snapshot_version"]),
                        "ledger_version": int(commit["end_version"]),
                        "proposed_candidate_index": int(proposed_action),
                        "candidate_index": action,
                        "candidate_id": str(candidate["candidate_id"]),
                        "source": str(candidate["source"]),
                        "fallback": used_fallback,
                        "attempts": int(result["attempts"]),
                    }
                    rows.append(exported)
                    heapq.heappush(
                        leave_heap,
                        (float(agent["leave_time"]), int(agent["request_id"])),
                    )
                    selected += 1
                else:
                    reason = str(result["reason"])
                    rejection_reasons[reason] += 1
                    rows.append({
                        "request_id": int(agent["request_id"]),
                        "accepted": False,
                        "reason": reason,
                        "wqmix_selection": {
                            "algorithm": "deployment_candidate_wqmix",
                            "batch_id": int(batch["batch_id"]),
                            "dataset_snapshot_version": int(batch["snapshot_version"]),
                            "ledger_version": int(commit["end_version"]),
                            "proposed_candidate_index": int(proposed_action),
                            "candidate_index": -1,
                            "attempts": int(result["attempts"]),
                        },
                    })
                    rejected += 1
                processed_requests += 1
            for edge, amount in ledger.bandwidth_used.items():
                peak_bandwidth[edge] = max(
                    peak_bandwidth.get(edge, 0.0),
                    max(0.0, float(amount)),
                )
            if args.max_requests and processed_requests >= args.max_requests:
                break
    final_snapshot = ledger.snapshot()
    ledger_violations = sum(
        value < -1e-7
        for resources in (
            final_snapshot.cpu_remaining,
            final_snapshot.memory_remaining,
            final_snapshot.bandwidth_remaining,
        )
        for value in resources.values()
    )
    peak_rows = sorted(
        (
            {
                "u": int(edge[0]),
                "v": int(edge[1]),
                "reserved_mbps": float(amount),
                "capacity_mbps": float(ledger.bandwidth_capacity[edge]),
                "utilization": float(amount / ledger.bandwidth_capacity[edge]),
            }
            for edge, amount in peak_bandwidth.items()
        ),
        key=lambda row: (-row["utilization"], row["u"], row["v"]),
    )
    requests_path = Path(dataset_spec["requests"])
    if not requests_path.is_absolute():
        requests_path = ROOT / requests_path
    trace_bandwidth_audit = audit_exported_bandwidth(
        rows,
        requests_path.resolve(),
        ledger.bandwidth_capacity,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "valid": ledger_violations == 0 and trace_bandwidth_audit["valid"],
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(data_folder.resolve()),
        "request_trace": str(requests_path.resolve()),
        "profile": str(profile_path),
        "plans": len(rows),
        "selected": selected,
        "rejected": rejected,
        "fallback": fallback,
        "released": releases,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "bandwidth_utilization_limit": float(args.bandwidth_utilization_limit),
        "peak_bandwidth_utilization": max(
            (row["utilization"] for row in peak_rows), default=0.0
        ),
        "peak_bandwidth_edges": peak_rows[:10],
        "trace_bandwidth_audit": trace_bandwidth_audit,
        "ledger_violations": ledger_violations,
        "final_ledger_version": int(final_snapshot.version),
        "output": str(output_path.resolve()),
        "admission": (
            "cross-batch authoritative atomic ledger with lifetime release and "
            "ranked feasible-candidate fallback"
        ),
    }
    (output_path.with_suffix(".summary.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
