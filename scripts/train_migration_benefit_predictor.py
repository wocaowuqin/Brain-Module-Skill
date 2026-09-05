#!/usr/bin/env python3
"""Train the migration gate from seed-separated counterfactual replays."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.migration_benefit_predictor import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_VERSION,
    PREDICTION_TARGET,
    MigrationBenefitNet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a counterfactual VNF migration benefit predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--minimum-test-auc", type=float, default=0.65)
    parser.add_argument("--minimum-test-precision", type=float, default=0.80)
    return parser.parse_args()


def _load(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.resolve().open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("prediction_target") != PREDICTION_TARGET:
                    raise ValueError(f"{path}:{line_number}: incompatible prediction target")
                if tuple(row.get("feature_names") or ()) != FEATURE_NAMES:
                    raise ValueError(f"{path}:{line_number}: incompatible feature schema")
                features = list(map(float, row.get("features") or ()))
                if len(features) != len(FEATURE_NAMES) or not all(map(math.isfinite, features)):
                    raise ValueError(f"{path}:{line_number}: invalid features")
                label = int(row.get("label", -1))
                if label not in {0, 1}:
                    raise ValueError(f"{path}:{line_number}: invalid label")
                rows.append({**row, "features": features, "label": label})
    return rows


def _split(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) < 5:
        raise ValueError(
            "at least five independent trace seeds are required; one-seed smoke "
            "data must not train an online migration model"
        )
    rng = random.Random(seed)
    rng.shuffle(seeds)
    validation_count = max(1, round(0.2 * len(seeds)))
    test_count = max(1, round(0.2 * len(seeds)))
    validation_seeds = set(seeds[:validation_count])
    test_seeds = set(seeds[validation_count:validation_count + test_count])
    train_seeds = set(seeds) - validation_seeds - test_seeds
    result = {
        "train": [row for row in rows if int(row["seed"]) in train_seeds],
        "validation": [row for row in rows if int(row["seed"]) in validation_seeds],
        "test": [row for row in rows if int(row["seed"]) in test_seeds],
    }
    for name, split_rows in result.items():
        labels = {int(row["label"]) for row in split_rows}
        if labels != {0, 1}:
            raise ValueError(f"{name} split must contain beneficial and harmful migrations")
    return result


def _tensor(rows: Sequence[dict[str, Any]], mean: torch.Tensor, std: torch.Tensor):
    raw = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    return (raw - mean) / std, labels


def _auc(probabilities: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = [p for p, y in zip(probabilities, labels) if y == 1]
    negatives = [p for p, y in zip(probabilities, labels) if y == 0]
    if not positives or not negatives:
        return None
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def _metrics(probabilities: Sequence[float], labels: Sequence[int], threshold: float):
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    return {
        "samples": len(labels),
        "positives": sum(labels),
        "threshold": threshold,
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / max(1, len(labels)),
        "roc_auc": _auc(probabilities, labels),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _probabilities(model: nn.Module, features: torch.Tensor) -> list[float]:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(features)).tolist()


def _threshold(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    # False-positive migrations are more damaging than missed migrations.
    choices = []
    for threshold in [0.50 + 0.02 * index for index in range(24)]:
        metrics = _metrics(probabilities, labels, threshold)
        choices.append((
            int(metrics["precision"] >= 0.80),
            metrics["recall"],
            -metrics["false_positive_rate"],
            threshold,
        ))
    return float(max(choices)[-1])


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = _load(args.data)
    splits = _split(rows, args.seed)
    train_raw = torch.tensor([row["features"] for row in splits["train"]], dtype=torch.float32)
    mean = train_raw.mean(dim=0)
    std = train_raw.std(dim=0, unbiased=False).clamp_min(1e-6)
    tensors = {name: _tensor(split, mean, std) for name, split in splits.items()}
    train_x, train_y = tensors["train"]
    validation_x, validation_y = tensors["validation"]

    model = MigrationBenefitNet(len(FEATURE_NAMES), args.hidden_dims, args.dropout)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(math.sqrt(negatives / max(positives, 1.0)))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_loss = math.inf
    stale = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss = criterion(model(train_x), train_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    model(validation_x), validation_y
                )
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_probabilities = _probabilities(model, validation_x)
    threshold = _threshold(validation_probabilities, validation_y.int().tolist())

    metrics = {}
    for name, (features, labels) in tensors.items():
        metrics[name] = _metrics(
            _probabilities(model, features), labels.int().tolist(), threshold
        )
    test = metrics["test"]
    deployment_allowed = bool(
        test["roc_auc"] is not None
        and float(test["roc_auc"]) >= args.minimum_test_auc
        and float(test["precision"]) >= args.minimum_test_precision
        and int(test["confusion"]["tp"]) > 0
    )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "migration_benefit_predictor.pt"
    checkpoint = {
        "version": MODEL_VERSION,
        "prediction_target": PREDICTION_TARGET,
        "feature_names": list(FEATURE_NAMES),
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "state_dict": best_state,
        "calibration": {"method": "none", "logit_scale": 1.0, "logit_bias": 0.0},
        "decision_threshold": threshold,
        "deployment_allowed": deployment_allowed,
        "training": {
            "samples": len(splits["train"]),
            "best_epoch": best_epoch,
            "seeds": sorted({int(row["seed"]) for row in splits["train"]}),
        },
        "validation": metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "deployment_allowed": deployment_allowed,
        "decision": (
            "offline criteria passed; fixed-seed online A/B is still required"
            if deployment_allowed
            else "offline criteria failed; runtime loader will reject this checkpoint"
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "decision_threshold": threshold,
        "metrics": metrics,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if deployment_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
