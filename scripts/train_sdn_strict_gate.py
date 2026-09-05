#!/usr/bin/env python3
"""Train a calibration-only execute/no-op gate on sdn_strict_gate_v3."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class StrictGateNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def calibration_split(rows: list[dict[str, Any]]):
    local = [row for row in rows if row["source_policy"] == "local"]
    qmix = [row for row in rows if row["source_policy"] == "qmix"]
    if not local or not qmix:
        raise ValueError("calibration split requires local and qmix source policies")
    train, validation = [], []
    for label in (0, 1):
        values = sorted(
            (row for row in local if int(row["execute_label"]) == label),
            key=lambda row: (row["time"], row["request_id"]),
        )
        for index, row in enumerate(values):
            (validation if index % 5 == 0 else train).append(row)
    if not train or not validation or not qmix:
        raise ValueError("empty train/validation/test calibration split")
    return train, validation, qmix


def seed_split(
    rows: list[dict[str, Any]],
    train_seeds: list[int],
    validation_seeds: list[int],
    test_seeds: list[int],
):
    split_sets = {
        "train": set(train_seeds),
        "validation": set(validation_seeds),
        "test": set(test_seeds),
    }
    for left, left_values in split_sets.items():
        for right, right_values in split_sets.items():
            if left >= right:
                continue
            overlap = left_values & right_values
            if overlap:
                raise ValueError(f"trace seed leakage between {left} and {right}: {sorted(overlap)}")
    available = {int(row["trace_seed"]) for row in rows}
    assigned = set().union(*split_sets.values())
    if assigned != available:
        raise ValueError(
            f"explicit seed split must assign every available seed exactly once; "
            f"available={sorted(available)}, assigned={sorted(assigned)}"
        )
    selected = {
        name: [row for row in rows if int(row["trace_seed"]) in seeds]
        for name, seeds in split_sets.items()
    }
    if any(not values for values in selected.values()):
        raise ValueError("train, validation, and test seed splits must all be non-empty")
    for name in ("train", "validation", "test"):
        labels = {int(row["execute_label"]) for row in selected[name]}
        if labels != {0, 1}:
            raise ValueError(f"{name} split must contain execute and no-op labels")
    return selected["train"], selected["validation"], selected["test"]


def tensors(rows: list[dict[str, Any]]):
    return (
        torch.tensor([row["features"] for row in rows], dtype=torch.float32),
        torch.tensor([row["execute_label"] for row in rows], dtype=torch.long),
    )


def metrics(
    model: nn.Module,
    rows: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    features, labels = tensors(rows)
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = (probabilities >= threshold).long()
    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())
    recall = tp / max(1, tp + fn)
    noop_recall = tn / max(1, tn + fp)
    return {
        "samples": len(rows),
        "positives": int((labels == 1).sum().item()),
        "negatives": int((labels == 0).sum().item()),
        "accuracy": float((predictions == labels).float().mean().item()),
        "balanced_accuracy": 0.5 * (recall + noop_recall),
        "execute_recall": recall,
        "execute_precision": tp / max(1, tp + fp),
        "noop_recall": noop_recall,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "mean_execute_probability": float(probabilities.mean().item()),
        "execute_threshold": float(threshold),
    }


def select_threshold(model: nn.Module, rows: list[dict[str, Any]]) -> float:
    features, _ = tensors(rows)
    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=1)[:, 1].tolist()
    candidates = sorted({0.0, 0.5, 1.0, *map(float, probabilities)})
    scored = []
    for threshold in candidates:
        values = metrics(model, rows, threshold)
        scored.append(
            (
                float(values["balanced_accuracy"]),
                float(values["noop_recall"]),
                -abs(threshold - 0.5),
                threshold,
            )
        )
    return float(max(scored)[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help="candidates.jsonl; repeat for multiple independent trace seeds",
    )
    parser.add_argument(
        "--output", default="artifacts/runs/sla/sdn_strict_gate_v3_calibration"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--test-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--min-test-execute-recall", type=float, default=0.5)
    parser.add_argument("--min-test-noop-recall", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.epochs <= 0
        or args.hidden_dim <= 0
        or args.lr <= 0.0
        or not 0.0 <= args.min_test_execute_recall <= 1.0
        or not 0.0 <= args.min_test_noop_recall <= 1.0
    ):
        raise ValueError("invalid training hyperparameter")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw_data_paths = args.data or ["data/sdn_strict_gate_v3/seed7301/candidates.jsonl"]
    data_paths = []
    for value in raw_data_paths:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        data_paths.append(path)
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    rows = [row for path in data_paths for row in read_jsonl(path)]
    if not rows:
        raise ValueError("strict gate dataset is empty")
    feature_names = rows[0]["feature_names"]
    if any(row["feature_names"] != feature_names for row in rows):
        raise ValueError("inconsistent strict gate feature schema")
    explicit_seed_split = any(
        value is not None
        for value in (args.train_seeds, args.validation_seeds, args.test_seeds)
    )
    if explicit_seed_split:
        if any(
            value is None
            for value in (args.train_seeds, args.validation_seeds, args.test_seeds)
        ):
            raise ValueError(
                "train-seeds, validation-seeds, and test-seeds must be provided together"
            )
        train, validation, test = seed_split(
            rows, args.train_seeds, args.validation_seeds, args.test_seeds
        )
        split_mode = "disjoint_trace_seeds"
    else:
        train, validation, test = calibration_split(rows)
        split_mode = "single_seed_cross_policy_calibration"
    model_name = (
        "sdn_strict_gate_v3_multiseed"
        if split_mode == "disjoint_trace_seeds"
        else "sdn_strict_gate_v3_calibration"
    )
    train_x, train_y = tensors(train)
    counts = torch.bincount(train_y, minlength=2).float()
    class_weights = counts.sum() / torch.clamp(2.0 * counts, min=1.0)
    model = StrictGateNet(len(feature_names), args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        validation_metrics = metrics(model, validation)
        score = float(validation_metrics["balanced_accuracy"])
        history.append(
            {"epoch": epoch, "loss": float(loss.item()), "validation": validation_metrics}
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("strict gate training produced no checkpoint")
    model.load_state_dict(best_state)
    execute_threshold = select_threshold(model, validation)
    split_metrics = {
        "train": metrics(model, train, execute_threshold),
        "validation": metrics(model, validation, execute_threshold),
        "test": metrics(model, test, execute_threshold),
    }
    test_metrics = split_metrics["test"]
    offline_validation_passed = bool(
        split_mode == "disjoint_trace_seeds"
        and test_metrics["execute_recall"] >= args.min_test_execute_recall
        and test_metrics["noop_recall"] >= args.min_test_noop_recall
    )
    if offline_validation_passed:
        reason = (
            "offline seed-isolated recall criteria passed; deployment remains blocked "
            "until paired live runs show a strict-SLA acceptance improvement"
        )
    elif split_mode == "disjoint_trace_seeds":
        reason = "independent test-seed execute/no-op recall criteria were not both met"
    else:
        reason = "only one trace seed is available; checkpoint is calibration-only"
    report = {
        "model": model_name,
        "deployment_allowed": False,
        "offline_validation_passed": offline_validation_passed,
        "reason": reason,
        "data": [str(path.resolve()) for path in data_paths],
        "feature_names": feature_names,
        "hidden_dim": args.hidden_dim,
        "best_epoch": best_epoch,
        "execute_threshold": execute_threshold,
        "class_weights": class_weights.tolist(),
        "split_mode": split_mode,
        "split_trace_seeds": {
            "train": sorted({int(row["trace_seed"]) for row in train}),
            "validation": sorted({int(row["trace_seed"]) for row in validation}),
            "test": sorted({int(row["trace_seed"]) for row in test}),
        },
        "splits": {
            name: {
                "source_policies": sorted({row["source_policy"] for row in values}),
                **split_metrics[name],
            }
            for name, values in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model_name,
        "deployment_allowed": False,
        "offline_validation_passed": offline_validation_passed,
        "feature_names": feature_names,
        "hidden_dim": args.hidden_dim,
        "execute_threshold": execute_threshold,
        "state_dict": best_state,
        "training_trace_seeds": sorted({int(row["trace_seed"]) for row in train}),
        "validation_trace_seeds": sorted(
            {int(row["trace_seed"]) for row in validation}
        ),
        "test_trace_seeds": sorted({int(row["trace_seed"]) for row in test}),
    }
    checkpoint_path = output_dir / "strict_gate_calibration.pth"
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint"] = str(checkpoint_path.resolve())
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
