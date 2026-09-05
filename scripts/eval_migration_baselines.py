#!/usr/bin/env python3
"""Evaluate heuristic, MILP, and learned VNF-migration baselines uniformly."""

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
    BatchCandidateQNetwork,
    ranked_actions,
)
from core.marl.joint_candidate_decoder import joint_footprints_feasible  # noqa: E402
from core.marl.migration_baselines import (  # noqa: E402
    HEURISTIC_POLICIES,
    MigrationBatchMILPOracle,
    candidate_footprints,
    heuristic_rankings,
    snapshot_from_migration_record,
)
from core.marl.migration_dataset import (  # noqa: E402
    MigrationFeatureNormalizer,
    MigrationTransitionDataset,
    load_migration_records,
)
from core.marl.migration_reward import migration_action_reward  # noqa: E402
from envs.migration_wqmix_env import MigrationReplayEnv  # noqa: E402


SUPPORTED_CHECKPOINT_TYPES = {"migration_policy_v2", "migration_wqmix_v1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="learned policy checkpoint; repeat for multiple algorithms",
    )
    parser.add_argument(
        "--output", default="artifacts/runs/migration/baseline_eval"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--oracle-time-limit", type=float, default=2.0)
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
    position = (len(ordered) - 1) * float(probability)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_checkpoints(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            path = resolve(value)
            name = path.stem
        else:
            name, raw_path = value.split("=", 1)
            name = name.strip()
            path = resolve(raw_path.strip())
        if not name:
            raise ValueError("checkpoint name cannot be empty")
        if name in result:
            raise ValueError(f"duplicate checkpoint name: {name}")
        result[name] = path
    return result


class PolicyMetrics:
    def __init__(self, name: str) -> None:
        self.name = name
        self.batches = 0
        self.agents = 0
        self.migrations = 0
        self.sla_safe_migrations = 0
        self.correct_actions = 0
        self.exact_joint = 0
        self.decoder_timeouts = 0
        self.decoder_adjustments = 0
        self.resource_violations = 0
        self.reward = 0.0
        self.decision_ms: list[float] = []

    def add(
        self,
        record: Mapping[str, Any],
        actions: Sequence[int],
        oracle_actions: Sequence[int],
        elapsed_ms: float,
        *,
        timed_out: bool = False,
        proposed: Sequence[int] | None = None,
    ) -> None:
        agents = list(record["agents"])
        if len(actions) != len(agents):
            raise ValueError("evaluated actions do not align with migration agents")
        selected_footprints = []
        all_footprints = candidate_footprints(record)
        for index, (agent, raw_action) in enumerate(zip(agents, actions)):
            action = int(raw_action)
            reject = int(agent["reject_action"])
            self.reward += migration_action_reward(agent, action)
            selected_footprints.append(all_footprints[index][action])
            if action != reject:
                self.migrations += 1
                ratio = float(agent["candidates"][action]["metrics"].get("delay_ratio", math.inf))
                self.sla_safe_migrations += int(ratio <= 1.0 + 1e-9)
        feasible, _ = joint_footprints_feasible(
            selected_footprints, snapshot_from_migration_record(record)
        )
        self.resource_violations += int(not feasible)
        self.batches += 1
        self.agents += len(agents)
        self.correct_actions += sum(
            int(int(action) == int(target))
            for action, target in zip(actions, oracle_actions)
        )
        self.exact_joint += int(list(map(int, actions)) == list(map(int, oracle_actions)))
        self.decoder_timeouts += int(timed_out)
        if proposed is not None:
            self.decoder_adjustments += sum(
                int(int(action) != int(proposal))
                for action, proposal in zip(actions, proposed)
            )
        self.decision_ms.append(float(elapsed_ms))

    def report(self, noop_reward: float, oracle_reward: float) -> dict[str, Any]:
        gain_denominator = oracle_reward - noop_reward
        gain_ratio = (
            None
            if abs(gain_denominator) <= 1e-12
            else (self.reward - noop_reward) / gain_denominator
        )
        return {
            "algorithm": self.name,
            "batches": self.batches,
            "agents": self.agents,
            "migrations": self.migrations,
            "migration_rate": self.migrations / max(1, self.agents),
            "sla_safe_migrations": self.sla_safe_migrations,
            "sla_safe_rate_among_migrations": self.sla_safe_migrations / max(1, self.migrations),
            "total_reward": self.reward,
            "mean_reward_per_agent": self.reward / max(1, self.agents),
            "oracle_normalized_gain": gain_ratio,
            "action_accuracy_vs_milp": self.correct_actions / max(1, self.agents),
            "exact_joint_accuracy_vs_milp": self.exact_joint / max(1, self.batches),
            "decoder_timeouts": self.decoder_timeouts,
            "decoder_adjustment_rate": self.decoder_adjustments / max(1, self.agents),
            "resource_violations": self.resource_violations,
            "decision_mean_ms": statistics.fmean(self.decision_ms) if self.decision_ms else 0.0,
            "decision_p95_ms": percentile(self.decision_ms, 0.95),
        }


def checkpoint_sources(payload: Mapping[str, Any]) -> set[str]:
    results = set()
    for key in ("train_sources", "validation_sources"):
        for value in payload.get(key, []):
            path = Path(value)
            results.add(
                str((path / "batches.jsonl" if path.is_dir() else path).resolve())
            )
    return results


def source_files(paths: Sequence[Path]) -> set[str]:
    return {
        str((path / "batches.jsonl" if path.is_dir() else path).resolve())
        for path in paths
    }


def runtime_sources(paths: Sequence[Path]) -> set[str]:
    results: set[str] = set()
    for path in paths:
        folder = path if path.is_dir() else path.parent
        spec_path = folder / "dataset_spec.json"
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        results.update(
            str(Path(value).resolve())
            for value in spec.get("source_runtime_reports", [])
        )
    return results


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_paths = [resolve(value) for value in args.data]
    records = load_migration_records(data_paths)
    if not records:
        raise ValueError("evaluation dataset is empty")
    checkpoint_paths = parse_checkpoints(args.checkpoint)
    loaded: dict[str, tuple[dict[str, Any], BatchCandidateQNetwork, MigrationFeatureNormalizer]] = {}
    overlaps: dict[str, list[str]] = {}
    runtime_overlaps: dict[str, list[str]] = {}
    evaluated_sources = source_files(data_paths)
    evaluated_runtime_sources = runtime_sources(data_paths)
    for name, path in checkpoint_paths.items():
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("checkpoint_type") not in SUPPORTED_CHECKPOINT_TYPES:
            raise ValueError(f"{path} is not a supported migration policy checkpoint")
        overlap = sorted(checkpoint_sources(payload) & evaluated_sources)
        runtime_overlap = sorted(
            set(payload.get("source_runtime_reports", []))
            & evaluated_runtime_sources
        )
        overlaps[name] = overlap
        runtime_overlaps[name] = runtime_overlap
        if (overlap or runtime_overlap) and not args.allow_seen_source:
            raise ValueError(
                f"evaluation data was seen by {name}: "
                + ", ".join([*overlap, *runtime_overlap])
            )
        model = BatchCandidateQNetwork(
            int(payload["request_dim"]),
            int(payload["candidate_dim"]),
            int(payload["hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        loaded[name] = (
            payload,
            model,
            MigrationFeatureNormalizer.from_dict(payload["normalizer"]),
        )

    oracle_solver = MigrationBatchMILPOracle(args.oracle_time_limit)
    oracle_metrics = PolicyMetrics("milp_oracle")
    oracle_actions_by_record: list[list[int]] = []
    oracle_status: dict[str, int] = {}
    for record in records:
        started = time.perf_counter_ns()
        solution = oracle_solver.solve(record)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        oracle_status[solution.status] = oracle_status.get(solution.status, 0) + 1
        oracle_actions_by_record.append(solution.actions)
        oracle_metrics.add(
            record, solution.actions, solution.actions, elapsed
        )

    metrics = {name: PolicyMetrics(name) for name in HEURISTIC_POLICIES}
    for record_index, record in enumerate(records):
        oracle_actions = oracle_actions_by_record[record_index]
        for policy in HEURISTIC_POLICIES:
            env = MigrationReplayEnv(
                [record],
                MigrationFeatureNormalizer.fit([record]),
                max_agents=max(1, len(record["agents"])),
                max_candidates=max(
                    1, max(len(agent["candidates"]) - 1 for agent in record["agents"])
                ),
                decoder_top_r=args.decoder_top_r,
                decoder_time_budget_ms=args.decoder_time_budget_ms,
            )
            started = time.perf_counter_ns()
            rankings, scores = heuristic_rankings(
                record,
                policy,
                rng=random.Random(args.seed + record_index),
            )
            decoded = env.decode_rankings(rankings, scores=scores)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            proposed = [ranking[0] for ranking in rankings]
            metrics[policy].add(
                record,
                decoded.actions,
                oracle_actions,
                elapsed,
                timed_out=decoded.timed_out,
                proposed=proposed,
            )

    for name, (payload, model, normalizer) in loaded.items():
        policy_metrics = PolicyMetrics(name)
        dataset = MigrationTransitionDataset(
            records,
            normalizer,
            int(payload["max_agents"]),
            int(payload["max_candidates"]),
        )
        for record_index, record in enumerate(records):
            env = MigrationReplayEnv(
                [record],
                normalizer,
                int(payload["max_agents"]),
                int(payload["max_candidates"]),
                decoder_top_r=args.decoder_top_r,
                decoder_time_budget_ms=args.decoder_time_budget_ms,
            )
            item = dataset._tensorize(record)
            active = len(record["agents"])
            started = time.perf_counter_ns()
            with torch.inference_mode():
                q_values = model(
                    item["request_observations"].unsqueeze(0).to(device),
                    item["candidate_features"].unsqueeze(0).to(device),
                    item["agent_mask"].unsqueeze(0).to(device),
                )[0].cpu()
            rankings = ranked_actions(q_values, item["action_mask"])[:active]
            decoded = env.decode_rankings(
                rankings, scores=q_values[:active].tolist()
            )
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            policy_metrics.add(
                record,
                decoded.actions,
                oracle_actions_by_record[record_index],
                elapsed,
                timed_out=decoded.timed_out,
                proposed=[ranking[0] for ranking in rankings],
            )
        metrics[name] = policy_metrics

    noop_reward = metrics["no_migration"].reward
    oracle_reward = oracle_metrics.reward
    reports = [
        oracle_metrics.report(noop_reward, oracle_reward),
        *(metrics[name].report(noop_reward, oracle_reward) for name in metrics),
    ]
    valid = all(
        row["resource_violations"] == 0 and row["decoder_timeouts"] == 0
        for row in reports
    )
    report = {
        "valid": valid,
        "replay_semantics": "static_trace_replay",
        "records": len(records),
        "data_sources": sorted(evaluated_sources),
        "checkpoint_source_overlap": overlaps,
        "checkpoint_runtime_source_overlap": runtime_overlaps,
        "reportable": not any(overlaps.values()) and not any(runtime_overlaps.values()),
        "oracle_status": oracle_status,
        "algorithms": reports,
        "metric_notes": {
            "migration_rate": "selected migrations / eligible migration-task agents",
            "sla_safe_rate_among_migrations": "selected candidates with modeled delay_ratio <= 1",
            "oracle_normalized_gain": "(policy reward - noop reward) / (MILP reward - noop reward)",
            "decision_latency": "policy ranking plus bounded joint decoding; MILP row is solver time",
        },
        "scientific_boundary": (
            "This is trace-based counterfactual evaluation. Stored future snapshots do not "
            "change when a policy chooses a different current migration."
        ),
    }
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "migration_baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = list(reports[0].keys())
    with (output / "migration_baseline_report.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reports)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
