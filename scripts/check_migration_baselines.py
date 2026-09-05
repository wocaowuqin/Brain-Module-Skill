#!/usr/bin/env python3
"""One-command smoke check for every migration baseline implementation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import WeightedQMIXLearner  # noqa: E402
from core.marl.joint_candidate_decoder import joint_footprints_feasible  # noqa: E402
from core.marl.migration_baseline_learners import (  # noqa: E402
    BehaviorCloningLearner,
    IndependentDQNLearner,
    MAPPOLearner,
)
from core.marl.migration_baselines import (  # noqa: E402
    HEURISTIC_POLICIES,
    MigrationBatchMILPOracle,
    candidate_footprints,
    heuristic_rankings,
    snapshot_from_migration_record,
)
from core.marl.migration_candidates import (  # noqa: E402
    MIGRATION_CANDIDATE_DIM,
    MIGRATION_REQUEST_DIM,
    MIGRATION_STATE_DIM,
)
from core.marl.migration_dataset import (  # noqa: E402
    MigrationFeatureNormalizer,
    MigrationTransitionDataset,
    load_migration_records,
)
from core.marl.migration_reward import migration_action_reward  # noqa: E402
from envs.migration_wqmix_env import MigrationReplayEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/migration_wqmix_smoke_seed7071")
    parser.add_argument(
        "--output", default="artifacts/runs/migration/baseline_check.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-static-replay",
        action="store_true",
        help="explicitly run the legacy static migration replay smoke check",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def finite(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def main() -> int:
    args = parse_args()
    torch.manual_seed(2026)
    random.seed(2026)
    records = load_migration_records([resolve(args.data)])
    if not records:
        raise ValueError("migration smoke dataset is empty")
    max_agents = max(len(record["agents"]) for record in records)
    max_candidates = max(
        len(agent["candidates"]) - 1
        for record in records
        for agent in record["agents"]
    )
    normalizer = MigrationFeatureNormalizer.fit(records)
    dataset = MigrationTransitionDataset(
        records, normalizer, max_agents, max_candidates
    )
    batch = next(iter(DataLoader(dataset, batch_size=min(4, len(dataset)))))
    device = torch.device(args.device)
    results: dict[str, object] = {}

    for policy in HEURISTIC_POLICIES:
        accepted = timeouts = violations = 0
        for record in records:
            env = MigrationReplayEnv(
                [record], normalizer, max_agents, max_candidates
            )
            rankings, scores = heuristic_rankings(record, policy)
            decoded = env.decode_rankings(rankings, scores=scores)
            accepted += decoded.accepted
            timeouts += int(decoded.timed_out)
            selected = [
                candidate_footprints(record)[index][action]
                for index, action in enumerate(decoded.actions)
            ]
            feasible, _ = joint_footprints_feasible(
                selected, snapshot_from_migration_record(record)
            )
            violations += int(not feasible)
        results[policy] = {
            "accepted": accepted,
            "decoder_timeouts": timeouts,
            "resource_violations": violations,
        }

    oracle = MigrationBatchMILPOracle().solve(records[0])
    results["milp_oracle"] = oracle.to_dict()

    bc = BehaviorCloningLearner(
        MIGRATION_REQUEST_DIM, MIGRATION_CANDIDATE_DIM, device=device
    )
    results["bc"] = bc.train_step(batch)
    idqn = IndependentDQNLearner(
        MIGRATION_REQUEST_DIM, MIGRATION_CANDIDATE_DIM, device=device
    )
    results["idqn"] = idqn.train_step(batch)
    for name, alpha in (("qmix", 1.0), ("wqmix", 0.1)):
        learner = WeightedQMIXLearner(
            MIGRATION_REQUEST_DIM,
            MIGRATION_CANDIDATE_DIM,
            MIGRATION_STATE_DIM,
            max_agents=max_agents,
            alpha=alpha,
            device=device,
        )
        results[name] = learner.train_step(batch)

    mappo = MAPPOLearner(
        MIGRATION_REQUEST_DIM,
        MIGRATION_CANDIDATE_DIM,
        MIGRATION_STATE_DIM,
        device=device,
    )
    proposals, old_logs = mappo.sample_actions(batch)
    returns = []
    for row, record_index in enumerate(batch["record_index"].tolist()):
        record = records[int(record_index)]
        rankings = []
        for agent_index, agent in enumerate(record["agents"]):
            action = int(proposals[row, agent_index])
            reject = int(agent["reject_action"])
            rankings.append([reject] if action == reject else [action, reject])
        env = MigrationReplayEnv(
            [record], normalizer, max_agents, max_candidates
        )
        decoded = env.decode_rankings(rankings)
        returns.append(sum(
            migration_action_reward(agent, action)
            for agent, action in zip(record["agents"], decoded.actions)
        ))
    results["mappo"] = mappo.train_step(
        batch,
        proposals,
        old_logs,
        torch.tensor(returns, dtype=torch.float32),
    )

    learned_are_finite = all(
        finite(results[name]) for name in ("bc", "idqn", "qmix", "wqmix", "mappo")
    )
    valid = (
        learned_are_finite
        and oracle.status in {"optimal", "limit_reached"}
        and all(
            row["decoder_timeouts"] == 0 and row["resource_violations"] == 0
            for name, row in results.items()
            if name in HEURISTIC_POLICIES
        )
    )
    report = {
        "valid": valid,
        "replay_semantics": "static_trace_replay",
        "legacy_offline_opt_in": bool(args.allow_static_replay),
        "records": len(records),
        "max_agents": max_agents,
        "max_candidates": max_candidates,
        "results": results,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
