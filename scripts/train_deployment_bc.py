#!/usr/bin/env python3
"""Behavior-clone exact Oracle actions for batched SFC deployment candidates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import BatchCandidateQNetwork  # noqa: E402
from core.marl.deployment_dataset import (  # noqa: E402
    DeploymentOracleDataset,
    FeatureNormalizer,
    load_labeled_batches,
    source_request_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", action="append", required=True,
        help="training dataset folder; repeat to combine independent trace seeds",
    )
    parser.add_argument(
        "--validation", action="append", required=True,
        help="validation dataset folder; repeat to combine independent trace seeds",
    )
    parser.add_argument("--output", default="artifacts/runs/deployment/bc_train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-agents", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=9)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7071)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-source-overlap", action="store_true",
        help="permit train/validation trace overlap for pipeline smoke tests only",
    )
    return parser.parse_args()


def resolve_paths(values: Iterable[str]) -> list[Path]:
    paths = []
    for value in values:
        path = Path(value)
        paths.append(path if path.is_absolute() else ROOT / path)
    return paths


def masked_logits(model: BatchCandidateQNetwork, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(
        batch["request_observations"],
        batch["candidate_features"],
        batch["agent_mask"],
    )
    return logits.masked_fill(~batch["action_mask"].bool(), torch.finfo(logits.dtype).min)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(
    model: BatchCandidateQNetwork,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float | int]:
    totals = {
        "agents": 0,
        "correct": 0,
        "batches": 0,
        "exact_batches": 0,
        "reject_tp": 0,
        "reject_fp": 0,
        "reject_fn": 0,
    }
    loss_sum = 0.0
    inference_ms = []
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            start = time.perf_counter_ns()
            logits = masked_logits(model, batch)
            predictions = logits.argmax(dim=-1)
            inference_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
            active = batch["agent_mask"].bool()
            targets = batch["actions"].long()
            active_logits = logits[active]
            active_targets = targets[active]
            if active_targets.numel():
                loss_sum += float(F.cross_entropy(active_logits, active_targets, reduction="sum"))
            correct = predictions.eq(targets)
            exact = (correct | ~active).all(dim=1)
            target_reject = targets.eq(batch["reject_actions"].long()) & active
            predicted_reject = predictions.eq(batch["reject_actions"].long()) & active
            totals["agents"] += int(active.sum())
            totals["correct"] += int((correct & active).sum())
            totals["batches"] += int(active.shape[0])
            totals["exact_batches"] += int(exact.sum())
            totals["reject_tp"] += int((target_reject & predicted_reject).sum())
            totals["reject_fp"] += int((~target_reject & predicted_reject & active).sum())
            totals["reject_fn"] += int((target_reject & ~predicted_reject).sum())
    agents = max(1, totals["agents"])
    batches = max(1, totals["batches"])
    reject_precision_denominator = totals["reject_tp"] + totals["reject_fp"]
    reject_recall_denominator = totals["reject_tp"] + totals["reject_fn"]
    ordered_latency = sorted(inference_ms)
    p95_index = max(0, min(len(ordered_latency) - 1, int(0.95 * len(ordered_latency))))
    return {
        **totals,
        "loss": loss_sum / agents,
        "agent_accuracy": totals["correct"] / agents,
        "exact_joint_accuracy": totals["exact_batches"] / batches,
        "reject_precision": (
            totals["reject_tp"] / reject_precision_denominator
            if reject_precision_denominator else 0.0
        ),
        "reject_recall": (
            totals["reject_tp"] / reject_recall_denominator
            if reject_recall_denominator else 0.0
        ),
        "model_batch_mean_ms": sum(inference_ms) / max(1, len(inference_ms)),
        "model_batch_p95_ms": ordered_latency[p95_index] if ordered_latency else 0.0,
    }


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0 or args.lr <= 0:
        raise ValueError("training hyperparameters must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    train_paths = resolve_paths(args.train)
    validation_paths = resolve_paths(args.validation)
    train_sources = source_request_files(train_paths)
    validation_sources = source_request_files(validation_paths)
    overlap = sorted(train_sources & validation_sources)
    if overlap and not args.allow_source_overlap:
        raise ValueError(
            "train/validation source request traces overlap; use independent seeds. "
            "--allow-source-overlap is only for a non-reportable smoke test: " + ", ".join(overlap)
        )

    train_records = load_labeled_batches(train_paths)
    validation_records = load_labeled_batches(validation_paths)
    normalizer = FeatureNormalizer.fit([record["batch"] for record in train_records])
    train_dataset = DeploymentOracleDataset(
        train_records, normalizer, args.max_agents, args.max_candidates
    )
    validation_dataset = DeploymentOracleDataset(
        validation_records, normalizer, args.max_agents, args.max_candidates
    )
    dimensions = (
        train_dataset.request_dim,
        train_dataset.candidate_dim,
        train_dataset.state_dim,
    )
    if dimensions != (
        validation_dataset.request_dim,
        validation_dataset.candidate_dim,
        validation_dataset.state_dim,
    ):
        raise ValueError("train/validation feature dimensions differ")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    evaluation_train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    model = BatchCandidateQNetwork(
        train_dataset.request_dim, train_dataset.candidate_dim, args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state: Dict[str, Any] | None = None
    best_score = (-1.0, -1.0)
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        active_count = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            active = batch["agent_mask"].bool()
            logits = masked_logits(model, batch)
            targets = batch["actions"].long()
            if not torch.all(
                batch["action_mask"].gather(-1, targets.unsqueeze(-1)).squeeze(-1) | ~active
            ):
                raise ValueError("Oracle target is outside the action mask")
            loss = F.cross_entropy(logits[active], targets[active])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            count = int(active.sum())
            running_loss += float(loss.item()) * count
            active_count += count
        validation_metrics = evaluate(model, validation_loader, device)
        score = (
            float(validation_metrics["agent_accuracy"]),
            float(validation_metrics["exact_joint_accuracy"]),
        )
        history.append({
            "epoch": epoch,
            "train_loss": running_loss / max(1, active_count),
            "validation": validation_metrics,
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.patience > 0 and stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("BC training produced no checkpoint")
    model.load_state_dict(best_state)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "bc_pretrained.pt"
    checkpoint = {
        "model": "deployment_candidate_bc",
        "dataset_version": "deployment_topk_oracle_v3",
        "candidate_feature_schema": "v3.1_source_neutral",
        "reportable": not bool(overlap),
        "state_dict": best_state,
        "normalizer": normalizer.to_dict(),
        "request_dim": train_dataset.request_dim,
        "candidate_dim": train_dataset.candidate_dim,
        "state_dim": train_dataset.state_dim,
        "hidden_dim": args.hidden_dim,
        "max_agents": args.max_agents,
        "max_candidates": args.max_candidates,
        "best_epoch": best_epoch,
        "train_sources": sorted(train_sources),
        "validation_sources": sorted(validation_sources),
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "valid": True,
        "reportable": not bool(overlap),
        "warning": (
            "train/validation use the same request trace; smoke-test metrics are not paper results"
            if overlap else ""
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "dimensions": {
            "request": train_dataset.request_dim,
            "candidate": train_dataset.candidate_dim,
            "state": train_dataset.state_dim,
            "max_agents": args.max_agents,
            "max_candidates": args.max_candidates,
        },
        "candidate_feature_schema": "v3.1_source_neutral",
        "train_folders": [str(path.resolve()) for path in train_paths],
        "validation_folders": [str(path.resolve()) for path in validation_paths],
        "source_overlap": overlap,
        "train": evaluate(model, evaluation_train_loader, device),
        "validation": evaluate(model, validation_loader, device),
        "history": history,
    }
    report_path = output_dir / "bc_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
