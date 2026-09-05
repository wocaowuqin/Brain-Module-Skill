#!/usr/bin/env python3
"""Validate deployment_topk_oracle_v3 labels and optionally re-solve them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.deployment_oracle import (  # noqa: E402
    DeploymentBatchOracle,
    OracleSolution,
    validate_solution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset folder or batches.jsonl")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--time-limit", type=float, default=10.0)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    data_path = args.data / "batches.jsonl" if args.data.is_dir() else args.data
    labels_path = args.labels or data_path.parent / "oracle_labels.jsonl"
    batches = read_jsonl(data_path)
    labels = read_jsonl(labels_path)
    errors = {
        "count": 0,
        "identity": 0,
        "status": 0,
        "action": 0,
        "resource": 0,
        "resolve": 0,
    }
    if len(batches) != len(labels):
        errors["count"] += 1
    oracle = DeploymentBatchOracle(time_limit_seconds=args.time_limit)
    optimal = 0
    accepted = 0
    changed_actions = 0

    for batch, label in zip(batches, labels):
        if (
            int(batch["batch_id"]) != int(label["batch_id"])
            or int(batch["snapshot_version"]) != int(label["snapshot_version"])
            or [int(agent["request_id"]) for agent in batch["agents"]]
            != list(map(int, label["request_ids"]))
        ):
            errors["identity"] += 1
        solver = label["solver"]
        optimal += int(bool(solver["optimal"]))
        if solver["status"] != "optimal" or not solver["optimal"]:
            errors["status"] += 1
        actions = list(map(int, label["oracle_actions"]))
        selected_sources = []
        for agent, action in zip(batch["agents"], actions):
            if not 0 <= action < len(agent["candidates"]):
                errors["action"] += 1
                selected_sources.append("invalid")
                continue
            candidate = agent["candidates"][action]
            if not bool(candidate["action_valid"]):
                errors["action"] += 1
            selected_sources.append(str(candidate["source"]))
        solution = OracleSolution(
            status=str(solver["status"]),
            optimal=bool(solver["optimal"]),
            objective=float(solver["objective"]),
            mip_gap=None if solver.get("mip_gap") is None else float(solver["mip_gap"]),
            actions=actions,
            accepted=int(label["oracle_accepted"]),
            rejected=len(batch["agents"]) - int(label["oracle_accepted"]),
            selected_sources=selected_sources,
            resource_usage=label["resource_usage"],
            message=str(solver.get("message", "")),
        )
        resource_errors = validate_solution(batch, solution)
        errors["resource"] += len(resource_errors)
        accepted += solution.accepted
        changed_actions += int(label["action_changes"])

        if args.resolve:
            repeated = oracle.solve(batch)
            if (
                not repeated.optimal
                or repeated.actions != actions
                or abs(repeated.objective - solution.objective) > 1e-6
            ):
                errors["resolve"] += 1

    valid = not any(errors.values())
    report = {
        "valid": valid,
        "batches": len(batches),
        "labels": len(labels),
        "optimal_batches": optimal,
        "oracle_accepted": accepted,
        "changed_actions": changed_actions,
        "resolved": bool(args.resolve),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
