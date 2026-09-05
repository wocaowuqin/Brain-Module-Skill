#!/usr/bin/env python3
"""Evaluate deployment BC rankings through the atomic resource committer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    BatchCandidateQNetwork,
    ranked_actions,
)
from core.marl.deployment_dataset import (  # noqa: E402
    DeploymentOracleDataset,
    FeatureNormalizer,
    load_labeled_batches,
    source_request_files,
)
from core.marl.deployment_oracle import (  # noqa: E402
    candidate_footprints,
    snapshot_from_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument(
        "--output", default="artifacts/runs/deployment/bc_eval.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-seen-source", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def trace_seed(folder: Path) -> int | None:
    spec_path = folder / "dataset_spec.json"
    if not spec_path.exists():
        return None
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    requests = Path(spec["requests"])
    scenario_path = requests.parent / "scenario.json"
    if not scenario_path.exists():
        return None
    return int(json.loads(scenario_path.read_text(encoding="utf-8"))["seed"])


def preload_ledger(batch: dict[str, Any]) -> AtomicResourceLedger:
    snapshot = snapshot_from_batch(batch)
    ledger = AtomicResourceLedger(
        snapshot.cpu_remaining,
        snapshot.memory_remaining,
        snapshot.bandwidth_remaining,
    )
    ledger.vnf_instances = {
        key: {"cpu": value[0], "memory": value[1], "ref_count": float(value[2])}
        for key, value in snapshot.vnf_instances.items()
    }
    return ledger


def evaluate_folder(
    model: BatchCandidateQNetwork,
    normalizer: FeatureNormalizer,
    folder: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    records = load_labeled_batches([folder])
    dataset = DeploymentOracleDataset(
        records,
        normalizer,
        int(checkpoint["max_agents"]),
        int(checkpoint["max_candidates"]),
    )
    counters = {
        "requests": 0,
        "oracle_accepted": 0,
        "heuristic_accepted": 0,
        "proposed_accepted": 0,
        "executed_accepted": 0,
        "proposed_correct": 0,
        "executed_correct": 0,
        "exact_proposed_batches": 0,
        "exact_executed_batches": 0,
        "proposed_joint_infeasible_batches": 0,
        "fallback_requests": 0,
        "policy_rejects": 0,
        "ledger_violations": 0,
    }
    end_to_end_ms: list[float] = []
    model.eval()
    with torch.inference_mode():
        for index, record in enumerate(records):
            started = time.perf_counter_ns()
            item = dataset[index]
            request_obs = item["request_observations"].unsqueeze(0).to(device)
            candidate_features = item["candidate_features"].unsqueeze(0).to(device)
            agent_mask = item["agent_mask"].unsqueeze(0).to(device)
            action_mask = item["action_mask"].unsqueeze(0).to(device)
            q_values = model(request_obs, candidate_features, agent_mask)
            rankings = ranked_actions(q_values[0], action_mask[0])
            active_count = int(item["agent_mask"].sum())
            rankings = rankings[:active_count]
            proposed = [ranking[0] for ranking in rankings]

            batch = record["batch"]
            label = record["label"]
            agents = batch["agents"]
            footprints = candidate_footprints(batch)
            ledger = preload_ledger(batch)
            commit = ledger.commit_ranked(
                [int(agent["request_id"]) for agent in agents],
                footprints,
                rankings,
            )
            executed = []
            for agent, result in zip(agents, commit["results"]):
                if result["accepted"]:
                    executed.append(int(result["candidate_index"]))
                else:
                    executed.append(int(agent["reject_action"]))
                counters["fallback_requests"] += int(result["attempts"] > 1)
                counters["policy_rejects"] += int(result["reason"] == "policy_reject")
            end_to_end_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

            oracle_actions = list(map(int, label["oracle_actions"]))
            proposed_sources = [
                str(agent["candidates"][action]["source"])
                for agent, action in zip(agents, proposed)
            ]
            counters["requests"] += active_count
            counters["oracle_accepted"] += int(label["oracle_accepted"])
            counters["heuristic_accepted"] += int(label["original_accepted"])
            counters["proposed_accepted"] += sum(source != "reject" for source in proposed_sources)
            counters["executed_accepted"] += int(commit["accepted"])
            counters["proposed_correct"] += sum(
                int(actual == target) for actual, target in zip(proposed, oracle_actions)
            )
            counters["executed_correct"] += sum(
                int(actual == target) for actual, target in zip(executed, oracle_actions)
            )
            counters["exact_proposed_batches"] += int(proposed == oracle_actions)
            counters["exact_executed_batches"] += int(executed == oracle_actions)

            selected_rankings = [[action] for action in proposed]
            proposed_ledger = preload_ledger(batch)
            proposed_commit = proposed_ledger.commit_ranked(
                [int(agent["request_id"]) for agent in agents],
                footprints,
                selected_rankings,
            )
            proposed_executed = []
            for agent, result in zip(agents, proposed_commit["results"]):
                proposed_executed.append(
                    int(result["candidate_index"])
                    if result["accepted"] else int(agent["reject_action"])
                )
            counters["proposed_joint_infeasible_batches"] += int(
                proposed_executed != proposed
            )
            counters["ledger_violations"] += int(
                any(
                    value < -1e-7
                    for resources in (
                        ledger.snapshot().cpu_remaining,
                        ledger.snapshot().memory_remaining,
                        ledger.snapshot().bandwidth_remaining,
                    )
                    for value in resources.values()
                )
            )

    requests = max(1, counters["requests"])
    batches = max(1, len(records))
    oracle_accepted = max(1, counters["oracle_accepted"])
    return {
        "folder": str(folder.resolve()),
        "trace_seed": trace_seed(folder),
        "batches": len(records),
        **counters,
        "proposed_action_accuracy": counters["proposed_correct"] / requests,
        "executed_action_accuracy": counters["executed_correct"] / requests,
        "exact_proposed_joint_accuracy": counters["exact_proposed_batches"] / batches,
        "exact_executed_joint_accuracy": counters["exact_executed_batches"] / batches,
        "executed_acceptance_rate": counters["executed_accepted"] / requests,
        "oracle_acceptance_rate": counters["oracle_accepted"] / requests,
        "heuristic_acceptance_rate": counters["heuristic_accepted"] / requests,
        "oracle_acceptance_ratio": counters["executed_accepted"] / oracle_accepted,
        "fallback_rate": counters["fallback_requests"] / requests,
        "offline_feature_model_commit_mean_ms": statistics.fmean(end_to_end_ms),
        "offline_feature_model_commit_p95_ms": percentile(end_to_end_ms, 0.95),
    }


def confidence_summary(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    average = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if count > 1 else 0.0
    half_width = (
        float(student_t.ppf(0.975, count - 1)) * standard_deviation / math.sqrt(count)
        if count > 1 else 0.0
    )
    return {
        "count": count,
        "mean": average,
        "std": standard_deviation,
        "ci95_low": max(0.0, average - half_width),
        "ci95_high": min(1.0, average + half_width),
    }


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    folders = [resolve(value) for value in args.data]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    evaluation_sources = source_request_files(folders)
    seen_sources = set(checkpoint.get("train_sources", [])) | set(
        checkpoint.get("validation_sources", [])
    )
    overlap = sorted(evaluation_sources & seen_sources)
    if overlap and not args.allow_seen_source:
        raise ValueError("evaluation source was used in training/validation: " + ", ".join(overlap))

    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    model = BatchCandidateQNetwork(
        int(checkpoint["request_dim"]),
        int(checkpoint["candidate_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    results = [
        evaluate_folder(model, normalizer, folder, checkpoint, device)
        for folder in folders
    ]
    total_requests = sum(result["requests"] for result in results)
    total_batches = sum(result["batches"] for result in results)
    total_oracle = sum(result["oracle_accepted"] for result in results)
    report = {
        "valid": all(result["ledger_violations"] == 0 for result in results),
        "checkpoint": str(checkpoint_path.resolve()),
        "source_overlap": overlap,
        "datasets": results,
        "micro": {
            "requests": total_requests,
            "batches": total_batches,
            "executed_accepted": sum(result["executed_accepted"] for result in results),
            "heuristic_accepted": sum(result["heuristic_accepted"] for result in results),
            "oracle_accepted": total_oracle,
            "executed_acceptance_rate": sum(
                result["executed_accepted"] for result in results
            ) / max(1, total_requests),
            "heuristic_acceptance_rate": sum(
                result["heuristic_accepted"] for result in results
            ) / max(1, total_requests),
            "oracle_acceptance_rate": total_oracle / max(1, total_requests),
            "oracle_acceptance_ratio": sum(
                result["executed_accepted"] for result in results
            ) / max(1, total_oracle),
            "executed_action_accuracy": sum(
                result["executed_correct"] for result in results
            ) / max(1, total_requests),
            "exact_executed_joint_accuracy": sum(
                result["exact_executed_batches"] for result in results
            ) / max(1, total_batches),
            "fallback_rate": sum(
                result["fallback_requests"] for result in results
            ) / max(1, total_requests),
        },
        "macro": {
            "seeds": len({result["trace_seed"] for result in results}),
            "executed_acceptance_rate": confidence_summary([
                result["executed_acceptance_rate"] for result in results
            ]),
            "oracle_acceptance_ratio": confidence_summary([
                result["oracle_acceptance_ratio"] for result in results
            ]),
            "proposed_action_accuracy": confidence_summary([
                result["proposed_action_accuracy"] for result in results
            ]),
            "fallback_rate": confidence_summary([
                result["fallback_rate"] for result in results
            ]),
        },
        "latency_scope": (
            "offline tensor construction, normalization, model forward, ranking, and atomic commit; "
            "candidate generation and Mininet/Ryu deployment are excluded"
        ),
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
