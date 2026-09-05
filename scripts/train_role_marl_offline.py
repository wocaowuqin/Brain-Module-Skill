#!/usr/bin/env python3
"""Offline IDQN/QMIX training on actionable_reroute_v2 decision snapshots."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Dict, Iterable, List, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.qmix import QMIXLearner
from core.marl.trainable_role_agents import (
    RoleReplayBuffer,
    TrainableSFTSelectionAgent,
    TrainableTreeRerouteAgent,
    TrainableVNFMigrationAgent,
)


@dataclass
class OfflineSample:
    split: str
    trace_seed: int
    scenario_id: str
    state_id: str
    selector_obs: List[float]
    selector_valid: List[int]
    reroute_obs: List[float]
    reroute_valid: List[int]
    selector_action: int
    reroute_action: int
    reward: float
    positive: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train role IDQN/QMIX directly from actionable_reroute_v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/actionable_reroute_v2_controlled")
    parser.add_argument("--algorithm", choices=["independent_dqn", "qmix", "all"], default="all")
    parser.add_argument(
        "--output", default="artifacts/runs/reconfiguration/role_marl_offline_v2"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-epoch", type=int, default=0,
                        help="0 uses one replay-sized pass per epoch")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qmix-embed-dim", type=int, default=32)
    parser.add_argument("--noop-counterfactual-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_split(data_root: Path, split: str) -> tuple[List[OfflineSample], Dict[str, Any]]:
    samples: List[OfflineSample] = []
    files: List[Path] = []
    skipped_invalid = 0
    scenarios = 0
    seen_state_ids = set()

    for spec_path in sorted(data_root.rglob("scenario_spec.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("dataset_version") != "actionable_reroute_v2" or spec.get("split") != split:
            continue
        scenarios += 1
        state_path = spec_path.parent / "states.jsonl"
        decision_path = spec_path.parent / "decisions.jsonl"
        files.extend((state_path, decision_path))
        states = {row["state_id"]: row for row in _read_jsonl(state_path)}
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in _read_jsonl(decision_path):
            grouped.setdefault(row.get("state_id", ""), []).append(row)

        for state_id, state in states.items():
            if state_id in seen_state_ids:
                raise ValueError(f"duplicate state_id across {split}: {state_id}")
            seen_state_ids.add(state_id)
            decisions = [
                row for row in grouped.get(state_id, [])
                if row.get("labels", {}).get("rejection_code") != "INVALID_BASE_STRUCTURE"
                and row.get("request", {}).get("structure", {}).get("base_structure_valid", False)
            ]
            if not decisions:
                skipped_invalid += 1
                continue

            positives = [row for row in decisions if row.get("labels", {}).get("action_available", False)]
            if positives:
                chosen = max(
                    positives,
                    key=lambda row: (float(row.get("planner_action", {}).get("estimated_gain", 0.0)),
                                     -int(row.get("rank", 1))),
                )
                positive = True
                selector_action = int(chosen.get("rank", 1))
                reroute_action = 1
                reward = max(0.0, float(chosen.get("planner_action", {}).get("estimated_gain", 0.0)))
            else:
                chosen = min(decisions, key=lambda row: int(row.get("rank", 1)))
                positive = False
                selector_action = 0
                reroute_action = 0
                reward = 0.0

            selector_obs = [float(value) for value in state.get("selector", {}).get("obs_vector", [])]
            reroute_obs = [float(value) for value in chosen.get("reroute_obs_vector", [])]
            if len(selector_obs) < 13 or (len(selector_obs) - 7) % 6 != 0:
                raise ValueError(f"{state_id}: invalid selector observation length {len(selector_obs)}")
            if len(reroute_obs) != 10:
                raise ValueError(f"{state_id}: invalid reroute observation length {len(reroute_obs)}")
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in selector_obs + reroute_obs):
                raise ValueError(f"{state_id}: non-finite or unbounded observation")

            selector_valid = sorted(set(int(value) for value in state.get("selector", {}).get("valid_actions", [0])))
            reroute_valid = sorted(set(int(value) for value in chosen.get("valid_actions", [0])))
            if selector_action not in selector_valid:
                raise ValueError(f"{state_id}: selector label {selector_action} is masked")
            if reroute_action not in reroute_valid:
                raise ValueError(f"{state_id}: reroute label {reroute_action} is masked")
            samples.append(OfflineSample(
                split=split,
                trace_seed=int(state.get("trace_seed", spec.get("trace_seed", -1))),
                scenario_id=str(state.get("scenario_id", spec.get("scenario_id", ""))),
                state_id=state_id,
                selector_obs=selector_obs,
                selector_valid=selector_valid,
                reroute_obs=reroute_obs,
                reroute_valid=reroute_valid,
                selector_action=selector_action,
                reroute_action=reroute_action,
                reward=reward,
                positive=positive,
            ))

    if not samples:
        raise ValueError(f"no usable actionable_reroute_v2 samples for split={split} under {data_root}")
    return samples, {
        "scenarios": scenarios,
        "samples": len(samples),
        "positives": sum(sample.positive for sample in samples),
        "negatives": sum(not sample.positive for sample in samples),
        "skipped_invalid_base": skipped_invalid,
        "sha256": _sha256_files(files),
        "files": [str(path) for path in files],
    }


def build_agents(top_k: int, args: argparse.Namespace, replay_capacity: int):
    common = {
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "epsilon": 0.0,
        "device": args.device,
    }
    agents = [
        TrainableSFTSelectionAgent(top_k=top_k, **common),
        TrainableVNFMigrationAgent(**common),
        TrainableTreeRerouteAgent(**common),
    ]
    for agent in agents:
        agent.config.replay_capacity = int(replay_capacity)
        agent.replay = RoleReplayBuffer(replay_capacity)
    return agents


def _masked_action(agent, obs: Sequence[float], valid_actions: Sequence[int]) -> tuple[int, List[float]]:
    valid = list(valid_actions) or [0]
    with torch.no_grad():
        tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_values = agent.q_net(tensor)[0]
        masked = torch.full_like(q_values, -1e9)
        masked[valid] = q_values[valid]
        return int(torch.argmax(masked).item()), [float(value) for value in q_values.cpu().tolist()]


def evaluate(agents, samples: Sequence[OfflineSample]) -> Dict[str, Any]:
    selector, _, rerouter = agents
    tp = fp = tn = fn = selector_correct = reroute_correct = exact_joint = 0
    finite = True
    q_abs_max = 0.0
    for sample in samples:
        selector_action, selector_q = _masked_action(selector, sample.selector_obs, sample.selector_valid)
        reroute_action, reroute_q = _masked_action(rerouter, sample.reroute_obs, sample.reroute_valid)
        predicted_positive = selector_action > 0 and reroute_action == 1
        if predicted_positive and sample.positive:
            tp += 1
        elif predicted_positive:
            fp += 1
        elif sample.positive:
            fn += 1
        else:
            tn += 1
        selector_correct += int(selector_action == sample.selector_action)
        reroute_correct += int(reroute_action == sample.reroute_action)
        exact_joint += int(
            selector_action == sample.selector_action and reroute_action == sample.reroute_action
        )
        all_q = selector_q + reroute_q
        finite = finite and all(math.isfinite(value) for value in all_q)
        q_abs_max = max(q_abs_max, *(abs(value) for value in all_q))

    total = len(samples)
    return {
        "samples": total,
        "action_accuracy": (tp + tn) / max(1, total),
        "positive_opportunity_recall": tp / max(1, tp + fn),
        "reroute_precision": tp / max(1, tp + fp),
        "noop_accuracy": tn / max(1, tn + fp),
        "selector_action_accuracy": selector_correct / max(1, total),
        "reroute_action_accuracy": reroute_correct / max(1, total),
        "exact_joint_accuracy": exact_joint / max(1, total),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "q_values_finite": finite,
        "max_abs_q": q_abs_max,
    }


def evaluate_by_seed(agents, samples: Sequence[OfflineSample]) -> Dict[str, Any]:
    grouped: Dict[int, List[OfflineSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.trace_seed, []).append(sample)
    per_seed = {str(seed): evaluate(agents, rows) for seed, rows in sorted(grouped.items())}
    metric_names = [
        "action_accuracy",
        "positive_opportunity_recall",
        "reroute_precision",
        "noop_accuracy",
        "selector_action_accuracy",
        "reroute_action_accuracy",
        "exact_joint_accuracy",
    ]
    macro = {}
    for name in metric_names:
        values = [float(metrics[name]) for metrics in per_seed.values()]
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        macro[name] = {
            "mean": mean_value,
            "std": std_value,
            "ci95": 1.96 * std_value / math.sqrt(max(1, len(values))),
        }
    return {"per_seed": per_seed, "macro": macro}


def classification_baselines(samples: Sequence[OfflineSample]) -> Dict[str, Any]:
    positives = sum(sample.positive for sample in samples)
    negatives = len(samples) - positives
    return {
        "always_noop": {
            "action_accuracy": negatives / max(1, len(samples)),
            "positive_opportunity_recall": 0.0,
            "noop_accuracy": 1.0,
        },
        "safe_greedy_label_oracle": {
            "action_accuracy": 1.0,
            "positive_opportunity_recall": 1.0,
            "noop_accuracy": 1.0,
            "note": "upper bound defined by the dataset's planner labels",
        },
    }


def _checkpoint(algorithm: str, epoch: int, agents, learner, metadata: Dict[str, Any]) -> Dict[str, Any]:
    state = {
        "episode": int(epoch),
        "epoch": int(epoch),
        "training_algorithm": algorithm,
        "dataset_version": "actionable_reroute_v2",
        "selector": copy.deepcopy(agents[0].state_dict()),
        "migration": copy.deepcopy(agents[1].state_dict()),
        "reroute": copy.deepcopy(agents[2].state_dict()),
        "offline_metadata": metadata,
    }
    if learner is not None:
        state["qmix"] = copy.deepcopy(learner.state_dict())
    return state


def train_one(algorithm: str, args: argparse.Namespace, splits, split_meta, output_dir: Path) -> Dict[str, Any]:
    train, validation, test = splits
    top_k = (len(train[0].selector_obs) - 7) // 6
    replay_capacity = max(5000, 2 * len(train) + 1)
    agents = build_agents(top_k, args, replay_capacity)
    learner = None
    if algorithm == "qmix":
        learner = QMIXLearner(
            agents,
            mixer_embed_dim=args.qmix_embed_dim,
            lr=args.lr,
            replay_capacity=replay_capacity,
            target_update_interval=25,
        )

    zero_obs = [[0.0] * agent.config.obs_dim for agent in agents]
    penalty = abs(float(args.noop_counterfactual_penalty))
    for sample in train:
        observations = [sample.selector_obs, [0.0] * 10, sample.reroute_obs]
        actions = [sample.selector_action, 0, sample.reroute_action]
        if learner is not None:
            learner.push_transition(observations, actions, sample.reward, zero_obs,
                                    [[0], [0], [0]], done=True)
            if sample.positive:
                learner.push_transition(observations, [0, 0, 0], -penalty, zero_obs,
                                        [[0], [0], [0]], done=True)
        else:
            agents[0].push_transition(sample.selector_obs, sample.selector_action, sample.reward,
                                      zero_obs[0], True, [0])
            agents[1].push_transition(observations[1], 0, 0.0, zero_obs[1], True, [0])
            agents[2].push_transition(sample.reroute_obs, sample.reroute_action, sample.reward,
                                      zero_obs[2], True, [0])
            if sample.positive:
                agents[0].push_transition(sample.selector_obs, 0, -penalty, zero_obs[0], True, [0])
                agents[2].push_transition(sample.reroute_obs, 0, -penalty, zero_obs[2], True, [0])

    replay_size = len(learner.replay) if learner is not None else max(len(agent.replay) for agent in agents)
    updates_per_epoch = args.updates_per_epoch or max(1, math.ceil(replay_size / args.batch_size))
    best_score = (-1.0, -1.0)
    best_state = None
    best_epoch = 0
    history = []
    last_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        losses = []
        for _ in range(updates_per_epoch):
            if learner is not None:
                result = learner.update_from_replay(args.batch_size)
                losses.append(float(result.get("loss", 0.0)))
            else:
                for agent in agents:
                    result = agent.update_from_replay(args.batch_size)
                    losses.append(float(result.get("loss", 0.0)))
        last_loss = sum(losses) / max(1, len(losses))
        validation_metrics = evaluate(agents, validation)
        score = (
            0.5 * (validation_metrics["positive_opportunity_recall"] + validation_metrics["noop_accuracy"]),
            validation_metrics["action_accuracy"],
        )
        history.append({"epoch": epoch, "loss": last_loss, "validation": validation_metrics})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = _checkpoint(algorithm, epoch, agents, learner, {
                "split_hashes": {name: meta["sha256"] for name, meta in split_meta.items()},
                "top_k": top_k,
                "selection_metric": "mean(positive_opportunity_recall, noop_accuracy)",
            })

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    for agent, key in zip(agents, ("selector", "migration", "reroute")):
        agent.load_state_dict(best_state[key])
    if learner is not None:
        learner.load_state_dict(best_state["qmix"])

    validation_metrics = evaluate(agents, validation)
    test_metrics = evaluate(agents, test)
    validation_seed_metrics = evaluate_by_seed(agents, validation)
    test_seed_metrics = evaluate_by_seed(agents, test)
    algorithm_dir = output_dir / algorithm
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = algorithm_dir / "best_model.pth"
    torch.save(best_state, checkpoint_path)
    report = {
        "algorithm": algorithm,
        "dataset_version": "actionable_reroute_v2",
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "updates_per_epoch": updates_per_epoch,
        "replay_size": replay_size,
        "replay_capacity": replay_capacity,
        "last_loss": last_loss,
        "checkpoint": str(checkpoint_path.resolve()),
        "splits": split_meta,
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_by_seed": validation_seed_metrics,
        "test_by_seed": test_seed_metrics,
        "test_baselines": classification_baselines(test),
        "finite_values": bool(validation_metrics["q_values_finite"] and test_metrics["q_values_finite"]),
        "history": history,
    }
    (algorithm_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {split: load_split(data_root, split) for split in ("train", "validation", "test")}
    splits = tuple(loaded[name][0] for name in ("train", "validation", "test"))
    split_meta = {name: loaded[name][1] for name in loaded}
    algorithms = ["independent_dqn", "qmix"] if args.algorithm == "all" else [args.algorithm]
    reports = []
    for algorithm in algorithms:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        reports.append(train_one(algorithm, args, splits, split_meta, output_dir))

    summary = {
        "ok": all(report["finite_values"] for report in reports),
        "data": str(data_root.resolve()),
        "algorithms": reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {
        "ok": summary["ok"],
        "data": summary["data"],
        "algorithms": [
            {
                "algorithm": report["algorithm"],
                "best_epoch": report["best_epoch"],
                "checkpoint": report["checkpoint"],
                "validation": report["validation"],
                "test": report["test"],
            }
            for report in reports
        ],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
