#!/usr/bin/env python3
"""Train BC, independent DQN, QMIX, Weighted QMIX, and MAPPO baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import WeightedQMIXLearner  # noqa: E402
from core.marl.migration_baseline_learners import (  # noqa: E402
    BehaviorCloningLearner,
    IndependentDQNLearner,
    MAPPOLearner,
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


LEARNED_ALGORITHMS = ("bc", "idqn", "qmix", "wqmix", "mappo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", action="append", required=True)
    parser.add_argument("--validation-data", action="append", default=[])
    parser.add_argument(
        "--algorithm", choices=("all", *LEARNED_ALGORITHMS), default="all"
    )
    parser.add_argument(
        "--output", default="artifacts/runs/migration/baselines_train"
    )
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mixer-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--ppo-updates", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--wqmix-alpha", type=float, default=0.1)
    parser.add_argument("--imitation-weight", type=float, default=0.1)
    parser.add_argument("--no-bc-warm-start", action="store_true")
    parser.add_argument(
        "--allow-static-replay",
        action="store_true",
        help=(
            "explicitly allow the legacy offline trace-replay trainer; "
            "without this flag use a DynamicMigrationEnv rollout entrypoint"
        ),
    )
    parser.add_argument(
        "--allow-source-overlap",
        action="store_true",
        help="allow shared underlying runtime reports for smoke tests only",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def data_source(path: Path) -> str:
    return str((path / "batches.jsonl" if path.is_dir() else path).resolve())


def runtime_sources(paths: Iterable[Path]) -> set[str]:
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {}
    return {
        key: float(sum(float(row[key]) for row in materialized) / len(materialized))
        for key in materialized[0]
    }


@torch.inference_mode()
def policy_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = exact = agents = records = 0
    for batch in loader:
        active = batch["agent_mask"].to(device).bool()
        action_mask = batch["action_mask"].to(device).bool()
        logits = model(
            batch["request_observations"].to(device),
            batch["candidate_features"].to(device),
            active,
        ).masked_fill(~action_mask, -1e9)
        predictions = logits.argmax(dim=-1)
        targets = batch["teacher_actions"].to(device)
        correct += int(((predictions == targets) & active).sum().item())
        exact += int((((predictions == targets) | ~active).all(dim=1)).sum().item())
        agents += int(active.sum().item())
        records += int(active.shape[0])
    model.train()
    return {
        "action_accuracy": correct / max(1, agents),
        "exact_joint_accuracy": exact / max(1, records),
        "agents": float(agents),
        "records": float(records),
    }


def checkpoint_payload(
    *,
    algorithm: str,
    state_dict: Mapping[str, Any],
    normalizer: MigrationFeatureNormalizer,
    args: argparse.Namespace,
    train_paths: list[Path],
    validation_paths: list[Path],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_type": "migration_policy_v2",
        "algorithm": algorithm,
        "state_dict": state_dict,
        "normalizer": normalizer.to_dict(),
        "request_dim": MIGRATION_REQUEST_DIM,
        "candidate_dim": MIGRATION_CANDIDATE_DIM,
        "state_dim": MIGRATION_STATE_DIM,
        "max_agents": int(args.max_agents),
        "max_candidates": int(args.max_candidates),
        "hidden_dim": int(args.hidden_dim),
        "train_sources": [data_source(path) for path in train_paths],
        "validation_sources": [data_source(path) for path in validation_paths],
        "source_runtime_reports": sorted(
            runtime_sources([*train_paths, *validation_paths])
        ),
        "seed": int(args.seed),
        "migration_semantics": {
            "agent": "one selected VNF migration task",
            "actions": "Top-K complete target plans plus noop",
            "hard_constraints": "action mask plus bounded joint decoder",
            "noop_semantics": "noop rank is a candidate-search cutoff",
            "replay_semantics": "static_trace_replay",
        },
    }
    if extra:
        payload.update(extra)
    return payload


def load_bc_warm_start(model: torch.nn.Module, bc_state: Mapping[str, Any] | None) -> None:
    if bc_state is not None:
        model.load_state_dict(bc_state)


def decoded_mappo_returns(
    records: list[Mapping[str, Any]],
    indices: torch.Tensor,
    proposals: torch.Tensor,
    normalizer: MigrationFeatureNormalizer,
    max_agents: int,
    max_candidates: int,
) -> torch.Tensor:
    returns: list[float] = []
    for row, raw_index in enumerate(indices.tolist()):
        record = records[int(raw_index)]
        rankings: list[list[int]] = []
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
    return torch.tensor(returns, dtype=torch.float32)


def main() -> int:
    args = parse_args()
    if not args.allow_static_replay:
        raise RuntimeError(
            "train_migration_baselines.py trains only on static serialized "
            "trace replay. Pass --allow-static-replay to run this legacy "
            "baseline, or use a DynamicMigrationEnv rollout entrypoint for "
            "action-dependent training."
        )
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    train_paths = [resolve(value) for value in args.train_data]
    validation_paths = [resolve(value) for value in args.validation_data]
    train_records = load_migration_records(train_paths)
    validation_records = (
        load_migration_records(validation_paths) if validation_paths else train_records
    )
    if not train_records:
        raise ValueError("training dataset is empty")
    provenance_overlap = sorted(
        runtime_sources(train_paths) & runtime_sources(validation_paths)
    )
    if provenance_overlap and not args.allow_source_overlap:
        raise ValueError(
            "training and validation data reuse underlying runtime reports; "
            "use independent traces, or --allow-source-overlap for a non-reportable "
            "smoke test: " + ", ".join(provenance_overlap)
        )
    selected = list(LEARNED_ALGORITHMS) if args.algorithm == "all" else [args.algorithm]
    normalizer = MigrationFeatureNormalizer.fit(train_records)
    train_dataset = MigrationTransitionDataset(
        train_records, normalizer, args.max_agents, args.max_candidates
    )
    validation_dataset = MigrationTransitionDataset(
        validation_records, normalizer, args.max_agents, args.max_candidates
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    histories: dict[str, list[dict[str, float]]] = {}
    checkpoints: dict[str, str] = {}

    need_bc = "bc" in selected or (
        not args.no_bc_warm_start
        and any(name in selected for name in ("qmix", "wqmix", "mappo"))
    )
    bc_state: Mapping[str, Any] | None = None
    if need_bc:
        learner = BehaviorCloningLearner(
            MIGRATION_REQUEST_DIM,
            MIGRATION_CANDIDATE_DIM,
            args.hidden_dim,
            args.lr,
            device,
        )
        history = []
        for epoch in range(max(0, args.bc_epochs)):
            metrics = mean_metrics(learner.train_step(batch) for batch in train_loader)
            metrics.update(policy_accuracy(learner.q_network, validation_loader, device))
            metrics["epoch"] = float(epoch + 1)
            history.append(metrics)
        bc_state = {
            key: value.detach().cpu().clone()
            for key, value in learner.q_network.state_dict().items()
        }
        path = output / "migration_bc.pt"
        torch.save(
            checkpoint_payload(
                algorithm="bc",
                state_dict=bc_state,
                normalizer=normalizer,
                args=args,
                train_paths=train_paths,
                validation_paths=validation_paths,
                extra={"training_scope": "joint-oracle behavior cloning"},
            ),
            path,
        )
        histories["bc"] = history
        checkpoints["bc"] = str(path.resolve())

    if "idqn" in selected:
        learner = IndependentDQNLearner(
            MIGRATION_REQUEST_DIM,
            MIGRATION_CANDIDATE_DIM,
            args.hidden_dim,
            args.lr,
            device,
        )
        history = []
        for epoch in range(max(0, args.epochs)):
            metrics = mean_metrics(learner.train_step(batch) for batch in train_loader)
            metrics.update(policy_accuracy(learner.q_network, validation_loader, device))
            metrics["epoch"] = float(epoch + 1)
            history.append(metrics)
        path = output / "migration_idqn.pt"
        torch.save(
            checkpoint_payload(
                algorithm="idqn",
                state_dict=learner.q_network.state_dict(),
                normalizer=normalizer,
                args=args,
                train_paths=train_paths,
                validation_paths=validation_paths,
                extra={
                    "training_scope": "one-step fitted Q for ephemeral task agents",
                    "temporal_bootstrap": False,
                },
            ),
            path,
        )
        histories["idqn"] = history
        checkpoints["idqn"] = str(path.resolve())

    for algorithm, alpha in (("qmix", 1.0), ("wqmix", args.wqmix_alpha)):
        if algorithm not in selected:
            continue
        learner = WeightedQMIXLearner(
            request_dim=MIGRATION_REQUEST_DIM,
            candidate_dim=MIGRATION_CANDIDATE_DIM,
            state_dim=MIGRATION_STATE_DIM,
            max_agents=args.max_agents,
            hidden_dim=args.hidden_dim,
            mixer_embed_dim=args.mixer_dim,
            alpha=alpha,
            lr=args.lr,
            gamma=args.gamma,
            imitation_weight=args.imitation_weight,
            device=device,
        )
        if not args.no_bc_warm_start:
            load_bc_warm_start(learner.q_network, bc_state)
            learner.target_q_network.load_state_dict(learner.q_network.state_dict())
        history = []
        for epoch in range(max(0, args.epochs)):
            metrics = mean_metrics(learner.train_step(batch) for batch in train_loader)
            metrics.update(policy_accuracy(learner.q_network, validation_loader, device))
            metrics["epoch"] = float(epoch + 1)
            history.append(metrics)
        path = output / f"migration_{algorithm}.pt"
        torch.save(
            checkpoint_payload(
                algorithm=algorithm,
                state_dict=learner.q_network.state_dict(),
                normalizer=normalizer,
                args=args,
                train_paths=train_paths,
                validation_paths=validation_paths,
                extra={
                    "training_scope": "offline consecutive-trace TD surrogate",
                    "alpha": float(alpha),
                    "gamma": float(args.gamma),
                    "mixer_dim": int(args.mixer_dim),
                    "mixer_state_dict": learner.mixer.state_dict(),
                    "target_state_dict": learner.target_q_network.state_dict(),
                    "target_mixer_state_dict": learner.target_mixer.state_dict(),
                },
            ),
            path,
        )
        histories[algorithm] = history
        checkpoints[algorithm] = str(path.resolve())

    if "mappo" in selected:
        learner = MAPPOLearner(
            MIGRATION_REQUEST_DIM,
            MIGRATION_CANDIDATE_DIM,
            MIGRATION_STATE_DIM,
            args.hidden_dim,
            args.lr,
            device=device,
        )
        if not args.no_bc_warm_start:
            load_bc_warm_start(learner.q_network, bc_state)
        history = []
        for epoch in range(max(0, args.epochs)):
            rows = []
            for batch in train_loader:
                proposals, old_logs = learner.sample_actions(batch)
                returns = decoded_mappo_returns(
                    train_records,
                    batch["record_index"],
                    proposals,
                    normalizer,
                    args.max_agents,
                    args.max_candidates,
                )
                for _ in range(max(1, args.ppo_updates)):
                    rows.append(
                        learner.train_step(batch, proposals, old_logs, returns)
                    )
            metrics = mean_metrics(rows)
            metrics.update(policy_accuracy(learner.q_network, validation_loader, device))
            metrics["epoch"] = float(epoch + 1)
            history.append(metrics)
        path = output / "migration_mappo.pt"
        torch.save(
            checkpoint_payload(
                algorithm="mappo",
                state_dict=learner.q_network.state_dict(),
                normalizer=normalizer,
                args=args,
                train_paths=train_paths,
                validation_paths=validation_paths,
                extra={
                    "training_scope": "one-step decoded contextual MAPPO surrogate",
                    "critic_state_dict": learner.critic.state_dict(),
                    "ppo_updates": int(args.ppo_updates),
                },
            ),
            path,
        )
        histories["mappo"] = history
        checkpoints["mappo"] = str(path.resolve())

    report = {
        "valid": True,
        "reportable": not bool(provenance_overlap),
        "algorithms": selected,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "runtime_source_overlap": provenance_overlap,
        "checkpoints": checkpoints,
        "last_metrics": {
            name: rows[-1] if rows else None for name, rows in histories.items()
        },
        "histories": histories,
        "scientific_boundary": (
            "Stored consecutive migration records do not react to selected actions. "
            "IDQN and MAPPO are therefore explicit one-step/contextual baselines; "
            "QMIX variants use the existing offline trace TD surrogate."
        ),
        "replay_semantics": "static_trace_replay",
    }
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "histories"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
