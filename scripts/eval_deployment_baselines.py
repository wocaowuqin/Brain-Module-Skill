#!/usr/bin/env python3
"""Evaluate initial-SFC deployment baselines with one decoder and ledger."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import torch


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
from core.marl.joint_candidate_decoder import decode_joint_candidates  # noqa: E402
from core.marl.joint_candidate_transaction import (  # noqa: E402
    commit_decoded_joint_actions,
)


HEURISTICS = (
    "reject_all",
    "original_policy",
    "independent_top1",
    "random_feasible",
    "objective_greedy",
    "legacy_hrl_only",
    "joint_greedy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--output", default="artifacts/runs/deployment/baseline_eval"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decoder-top-r", type=int, default=4)
    parser.add_argument("--decoder-time-budget-ms", type=float, default=2.0)
    parser.add_argument("--allow-seen-source", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_checkpoints(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" in value:
            name, raw_path = value.split("=", 1)
            path = resolve(raw_path.strip())
        else:
            path = resolve(value)
            name = path.stem
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"invalid or duplicate checkpoint name: {name!r}")
        result[name] = path
    return result


def preload_ledger(batch: Mapping[str, Any]) -> AtomicResourceLedger:
    snapshot = snapshot_from_batch(batch)
    ledger = AtomicResourceLedger(
        snapshot.cpu_remaining,
        snapshot.memory_remaining,
        snapshot.bandwidth_remaining,
    )
    ledger.vnf_instances = {
        key: {
            "cpu": float(value[0]),
            "memory": float(value[1]),
            "ref_count": float(value[2]),
        }
        for key, value in snapshot.vnf_instances.items()
    }
    return ledger


def objective_scores(batch: Mapping[str, Any]) -> list[list[float]]:
    return [
        [
            -float(candidate.get("objective", 1_000_000.0))
            for candidate in agent["candidates"]
        ]
        for agent in batch["agents"]
    ]


def heuristic_rankings(
    record: Mapping[str, Any],
    policy: str,
    rng: random.Random,
) -> tuple[list[list[int]], list[list[float]], list[int] | None]:
    batch, label = record["batch"], record["label"]
    scores = objective_scores(batch)
    if policy == "joint_greedy":
        # Serialized candidate features 8:12 are the four cross-request
        # resource-conflict ratios.  Penalizing them distinguishes this
        # conflict-aware joint heuristic from the pure objective greedy row.
        for agent_index, agent in enumerate(batch["agents"]):
            for action, candidate in enumerate(agent["candidates"]):
                features = list(candidate.get("candidate_features") or [])
                conflict = sum(map(float, features[8:12])) if len(features) >= 12 else 0.0
                scores[agent_index][action] -= 5.0 * conflict
    rankings: list[list[int]] = []
    direct: list[int] | None = None
    if policy == "original_policy":
        direct = list(map(int, label["original_actions"]))
    for agent_index, agent in enumerate(batch["agents"]):
        reject = int(agent["reject_action"])
        valid = [
            index
            for index, candidate in enumerate(agent["candidates"])
            if index != reject and bool(candidate.get("action_valid", False))
        ]
        ordered = sorted(valid, key=lambda index: (-scores[agent_index][index], index))
        if policy == "reject_all":
            ranking = [reject]
        elif policy == "independent_top1":
            ranking = ordered[:1] + [reject]
        elif policy == "random_feasible":
            rng.shuffle(valid)
            ranking = valid + [reject]
        elif policy == "legacy_hrl_only":
            hrl = [
                index for index in ordered
                if str(agent["candidates"][index].get("source")) == "legacy_hrl"
            ]
            ranking = hrl + [reject]
        else:
            ranking = ordered + [reject]
        rankings.append(ranking)
    return rankings, scores, direct


def decode(
    batch: Mapping[str, Any],
    rankings: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    args: argparse.Namespace,
) -> Any:
    return decode_joint_candidates(
        candidate_footprints(batch),
        snapshot_from_batch(batch),
        rankings,
        reject_actions=[int(agent["reject_action"]) for agent in batch["agents"]],
        action_mask=[
            [bool(candidate.get("action_valid", False)) for candidate in agent["candidates"]]
            for agent in batch["agents"]
        ],
        scores=scores,
        priorities=[
            (float(agent.get("leave_time", math.inf)), int(agent["request_id"]))
            for agent in batch["agents"]
        ],
        top_r=args.decoder_top_r,
        time_budget_ms=args.decoder_time_budget_ms,
    )


class Metrics:
    def __init__(self, name: str) -> None:
        self.name = name
        self.batches = self.requests = self.accepted = 0
        self.sla_safe = self.correct = self.exact = 0
        self.decoder_timeouts = self.decoder_adjustments = 0
        self.commit_failures = self.ledger_violations = 0
        self.decision_ms: list[float] = []

    def add(
        self,
        record: Mapping[str, Any],
        actions: Sequence[int],
        elapsed_ms: float,
        *,
        timed_out: bool = False,
        proposed: Sequence[int] | None = None,
    ) -> None:
        batch, label = record["batch"], record["label"]
        agents = batch["agents"]
        oracle = list(map(int, label["oracle_actions"]))
        ledger = preload_ledger(batch)
        footprints = candidate_footprints(batch)
        committed = commit_decoded_joint_actions(
            ledger,
            [int(agent["request_id"]) for agent in agents],
            footprints,
            list(map(int, actions)),
            expected_version=ledger.snapshot().version,
        )
        self.commit_failures += int(not committed["committed"])
        executed = []
        for agent, result in zip(agents, committed["results"]):
            action = (
                int(result["candidate_index"])
                if result["accepted"]
                else int(agent["reject_action"])
            )
            executed.append(action)
            if result["accepted"]:
                self.accepted += 1
                metrics = agent["candidates"][action].get("metrics") or {}
                estimated = float(metrics.get("estimated_delay_ms", math.inf))
                bound = float(metrics.get("delay_bound_ms", -math.inf))
                self.sla_safe += int(estimated <= bound + 1e-9)
        snapshot = ledger.snapshot()
        self.ledger_violations += int(any(
            value < -1e-7
            for resources in (
                snapshot.cpu_remaining,
                snapshot.memory_remaining,
                snapshot.bandwidth_remaining,
            )
            for value in resources.values()
        ))
        self.batches += 1
        self.requests += len(agents)
        self.correct += sum(
            int(action == target) for action, target in zip(executed, oracle)
        )
        self.exact += int(executed == oracle)
        self.decoder_timeouts += int(timed_out)
        if proposed is not None:
            self.decoder_adjustments += sum(
                int(int(action) != int(proposal))
                for action, proposal in zip(executed, proposed)
            )
        self.decision_ms.append(float(elapsed_ms))

    def report(self, oracle_accepted: int) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "batches": self.batches,
            "requests": self.requests,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / max(1, self.requests),
            "oracle_acceptance_ratio": self.accepted / max(1, oracle_accepted),
            "modeled_sla_safe_accepted": self.sla_safe,
            "modeled_sla_safe_rate": self.sla_safe / max(1, self.accepted),
            "action_accuracy_vs_milp": self.correct / max(1, self.requests),
            "exact_joint_accuracy_vs_milp": self.exact / max(1, self.batches),
            "decoder_timeouts": self.decoder_timeouts,
            "decoder_adjustment_rate": self.decoder_adjustments / max(1, self.requests),
            "commit_failures": self.commit_failures,
            "ledger_violations": self.ledger_violations,
            "decision_mean_ms": statistics.fmean(self.decision_ms) if self.decision_ms else 0.0,
            "decision_p95_ms": percentile(self.decision_ms, 0.95),
        }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    folders = [resolve(value) for value in args.data]
    records = load_labeled_batches(folders)
    if not records:
        raise ValueError("deployment evaluation dataset is empty")
    evaluation_sources = source_request_files(folders)
    loaded = {}
    overlaps = {}
    for name, path in parse_checkpoints(args.checkpoint).items():
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("model") not in {
            "deployment_candidate_bc",
            "deployment_candidate_qmix",
            "deployment_candidate_wqmix",
        }:
            raise ValueError(f"{path} is not a deployment candidate policy")
        seen_sources = (
            set(payload.get("train_sources", []))
            | set(payload.get("validation_sources", []))
            | set((payload.get("training") or {}).get("train_traces", []))
        )
        overlap = sorted(evaluation_sources & seen_sources)
        overlaps[name] = overlap
        if overlap and not args.allow_seen_source:
            raise ValueError(f"evaluation source was seen by {name}: " + ", ".join(overlap))
        normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
        model = BatchCandidateQNetwork(
            int(payload["request_dim"]),
            int(payload["candidate_dim"]),
            int(payload["hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        loaded[name] = (payload, model, normalizer)

    metrics = {name: Metrics(name) for name in ("milp_oracle_label", *HEURISTICS)}
    for record_index, record in enumerate(records):
        batch, label = record["batch"], record["label"]
        oracle_actions = list(map(int, label["oracle_actions"]))
        started = time.perf_counter_ns()
        metrics["milp_oracle_label"].add(
            record,
            oracle_actions,
            (time.perf_counter_ns() - started) / 1_000_000.0,
        )
        for policy in HEURISTICS:
            rankings, scores, direct = heuristic_rankings(
                record, policy, random.Random(args.seed + record_index)
            )
            started = time.perf_counter_ns()
            if direct is None:
                result = decode(batch, rankings, scores, args)
                actions = result.actions
                timed_out = result.timed_out
            else:
                actions = direct
                timed_out = False
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            metrics[policy].add(
                record,
                actions,
                elapsed,
                timed_out=timed_out,
                proposed=[ranking[0] for ranking in rankings],
            )

    for name, (payload, model, normalizer) in loaded.items():
        policy_metrics = Metrics(name)
        dataset = DeploymentOracleDataset(
            records,
            normalizer,
            int(payload["max_agents"]),
            int(payload["max_candidates"]),
        )
        with torch.inference_mode():
            for index, record in enumerate(records):
                item = dataset[index]
                active = len(record["batch"]["agents"])
                started = time.perf_counter_ns()
                q_values = model(
                    item["request_observations"].unsqueeze(0).to(device),
                    item["candidate_features"].unsqueeze(0).to(device),
                    item["agent_mask"].unsqueeze(0).to(device),
                )[0].cpu()
                rankings = ranked_actions(q_values, item["action_mask"])[:active]
                result = decode(
                    record["batch"], rankings, q_values[:active].tolist(), args
                )
                elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
                policy_metrics.add(
                    record,
                    result.actions,
                    elapsed,
                    timed_out=result.timed_out,
                    proposed=[ranking[0] for ranking in rankings],
                )
        metrics[name] = policy_metrics

    oracle_accepted = metrics["milp_oracle_label"].accepted
    rows = [metric.report(oracle_accepted) for metric in metrics.values()]
    valid = all(
        row["commit_failures"] == 0 and row["ledger_violations"] == 0
        for row in rows
    )
    report = {
        "valid": valid,
        "datasets": [str(folder.resolve()) for folder in folders],
        "request_sources": sorted(evaluation_sources),
        "checkpoint_source_overlap": overlaps,
        "algorithms": rows,
        "metric_notes": {
            "milp_oracle_label": "precomputed SciPy/HiGHS joint oracle label",
            "legacy_hrl_only": "only candidates whose serialized source is exactly legacy_hrl",
            "modeled_sla_safe_rate": "candidate estimated_delay_ms <= delay_bound_ms; not Mininet measurement",
            "decision_latency": "ranking plus joint decoding; excludes candidate generation and Ryu/Mininet",
        },
    }
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "deployment_baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "deployment_baseline_report.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
