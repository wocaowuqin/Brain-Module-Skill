#!/usr/bin/env python3
"""Train a leakage-free strict-SLA predictor from real Ryu/Mininet runs."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    ResourceFootprint,
)
from core.marl.sla_risk_predictor import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_VERSION,
    PREDICTION_TARGET,
    RuntimeSlaRiskNet,
    build_runtime_sla_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(
            f"--{split}-source",
            action="append",
            required=True,
            metavar="RESULT_JSON::REQUESTS_JSONL",
            help=f"real runtime source assigned to {split}; repeat as needed",
        )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/sla/sla_risk_predictor_v1"),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-test-auc", type=float, default=0.65)
    parser.add_argument("--minimum-brier-skill", type=float, default=0.0)
    parser.add_argument(
        "--minimum-receiver-drain-ms",
        type=float,
        default=50.0,
        help=(
            "reject sources that do not keep receivers alive after sender stop; "
            "the current runtime normally provides a 100 ms drain window"
        ),
    )
    return parser.parse_args()


def _source_pair(value: str) -> tuple[Path, Path]:
    if "::" not in value:
        raise ValueError("source must be RESULT_JSON::REQUESTS_JSONL")
    result, requests = value.split("::", 1)
    return Path(result).resolve(), Path(requests).resolve()


def _read_requests(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["id"])] = row
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_resources(payload: Mapping[str, Any]) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[tuple[int, int], float],
    dict[str, Any],
]:
    runtime = payload.get("online_wqmix") or {}
    data_folder = Path(str(runtime["data_folder"]))
    profile_path = Path(str(runtime["profile"]))
    spec = json.loads((data_folder / "dataset_spec.json").read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    resource = spec["resource_model"]
    cpu = {
        int(node): float(resource["cpu_per_dc"])
        for node in profile["dc_nodes_1based"]
    }
    memory = {
        int(node): float(resource["memory_per_dc"])
        for node in profile["dc_nodes_1based"]
    }
    default_bandwidth = float(profile.get("default_bandwidth_mbps", 0.0))
    limit = float(runtime.get(
        "bandwidth_utilization_limit",
        resource.get("bandwidth_utilization_limit", 1.0),
    ))
    bandwidth: dict[tuple[int, int], float] = {}
    for edge in profile["edges"]:
        capacity = float(edge.get("bandwidth_mbps", default_bandwidth)) * limit
        u, v = int(edge["u"]), int(edge["v"])
        bandwidth[(u, v)] = capacity
        bandwidth[(v, u)] = capacity
    return cpu, memory, bandwidth, {
        "data_folder": str(data_folder.resolve()),
        "profile": str(profile_path.resolve()),
        "bandwidth_utilization_limit": limit,
        "cpu_per_dc": float(resource["cpu_per_dc"]),
        "memory_per_dc": float(resource["memory_per_dc"]),
    }


def _strict_outcomes(
    payload: Mapping[str, Any], requests: Mapping[int, Mapping[str, Any]]
) -> tuple[dict[int, bool], set[int]]:
    receivers: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload.get("receiver_results", []):
        receivers[int(row["request_id"])].append(row)
    senders: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload.get("sender_results", []):
        senders[int(row["request_id"])].append(row)
    sender_started = {
        request_id
        for request_id, rows in senders.items()
        if rows and all(
            isinstance(row.get("result"), Mapping)
            and row["result"].get("status") == "completed"
            for row in rows
        )
    }
    outcomes = {}
    for request_id, request in requests.items():
        rows = receivers.get(request_id, [])
        expected_destinations = {
            int(value) for value in request.get("destination_dpids", [])
        }
        measured_destinations = {
            int(row["destination_dpid"])
            for row in rows
            if row.get("destination_dpid") is not None
        }
        outcomes[request_id] = bool(expected_destinations) and (
            measured_destinations == expected_destinations
            and len(rows) == len(expected_destinations)
            and all(
                isinstance(row.get("result"), Mapping)
                and bool(row["result"].get("sla_met"))
                for row in rows
            )
        )
    return outcomes, sender_started


def _measurement_contract(
    payload: Mapping[str, Any], minimum_receiver_drain_ms: float
) -> dict[str, Any]:
    probe = payload.get("probe_timing") or {}
    deployed_arrivals = [
        event
        for event in payload.get("events", [])
        if event.get("type") == "arrive"
        and bool((event.get("controller") or {}).get("accepted"))
    ]
    drain_windows = [
        float((event.get("deployment_timing") or {}).get("probe_drain_window_ms"))
        for event in deployed_arrivals
        if (event.get("deployment_timing") or {}).get("probe_drain_window_ms")
        is not None
    ]
    if not deployed_arrivals or len(drain_windows) != len(deployed_arrivals):
        raise ValueError(
            "runtime source does not prove receiver-after-sender drain semantics"
        )
    observed_minimum = min(drain_windows)
    if observed_minimum + 1e-6 < float(minimum_receiver_drain_ms):
        raise ValueError(
            "runtime source receiver drain window is too short: "
            f"{observed_minimum:.3f} ms < {minimum_receiver_drain_ms:.3f} ms"
        )
    required = {
        "probe_sender_backend": probe.get("probe_sender_backend"),
        "probe_receiver_backend": probe.get("probe_receiver_backend"),
        "vnf_agent_backend": probe.get("vnf_agent_backend"),
        "mininet_qdisc": probe.get("mininet_qdisc"),
        "vnf_agent_packet_batch": probe.get("vnf_agent_packet_batch"),
        "vnf_agent_q0_packet_batch": probe.get("vnf_agent_q0_packet_batch"),
        "vnf_agent_dscp_scheduling": bool(
            probe.get("vnf_agent_dscp_scheduling", False)
        ),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "runtime source lacks SLA measurement contract fields: "
            + ", ".join(missing)
        )
    return {
        "label_semantics": "request_strict_sla_with_receiver_drain_v1",
        "required_receiver_drain_ms": float(minimum_receiver_drain_ms),
        "observed_minimum_receiver_drain_ms": float(observed_minimum),
        **required,
    }


def extract_source(
    result_path: Path,
    requests_path: Path,
    *,
    split: str,
    source_index: int,
    minimum_receiver_drain_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    measurement_contract = _measurement_contract(
        payload, minimum_receiver_drain_ms
    )
    requests = _read_requests(requests_path)
    plans = {
        int(row["request_id"]): row
        for row in payload.get("online_plans", [])
        if row.get("request_id") is not None and row.get("accepted")
    }
    strict_outcomes, sender_started = _strict_outcomes(payload, requests)
    cpu, memory, bandwidth, resource_metadata = _runtime_resources(payload)
    ledger = AtomicResourceLedger(cpu, memory, bandwidth)
    recent_arrivals: deque[float] = deque()
    rows: list[dict[str, Any]] = []
    skipped = defaultdict(int)

    for event in payload.get("events", []):
        event_type = str(event.get("type"))
        request_id = int(event.get("request_id", -1))
        if event_type == "leave":
            ledger.release(request_id)
            continue
        if event_type != "arrive":
            continue
        request = requests.get(request_id)
        if request is None:
            raise ValueError(f"request {request_id} is absent from {requests_path}")
        arrival = float(request["arrival_time"])
        recent_arrivals.append(arrival)
        while recent_arrivals and recent_arrivals[0] < arrival - 1.0:
            recent_arrivals.popleft()
        controller = event.get("controller") or {}
        plan = plans.get(request_id)
        if not bool(controller.get("accepted")) or plan is None:
            skipped["not_deployed"] += 1
            continue
        planning = event.get("online_planning") or {}
        modeled_delay_ratio = planning.get("predicted_sla_risk")
        if modeled_delay_ratio is None:
            skipped["missing_modeled_delay"] += 1
            continue

        snapshot = ledger.snapshot()
        footprint = ResourceFootprint.from_sfc_plan(
            plan, float(request["bw_origin"]), snapshot
        )
        features = build_runtime_sla_features(
            request,
            plan,
            footprint,
            snapshot,
            bandwidth,
            modeled_delay_ratio=float(modeled_delay_ratio),
            batch_size=int(planning.get("batch_size", 1)),
            active_request_count=len(ledger.allocations),
            recent_arrival_rate_1s=float(len(recent_arrivals)),
        )
        if request_id in sender_started:
            strict_met = bool(strict_outcomes.get(request_id, False))
            rows.append({
                "split": split,
                "source_index": int(source_index),
                "source_result": str(result_path),
                "request_id": request_id,
                "features": features,
                "failure_label": int(not strict_met),
                "strict_sla_met": strict_met,
            })
        else:
            skipped["sender_not_started"] += 1

        commit = ledger.commit_exact(
            [request_id], [[footprint]], [0], expected_version=snapshot.version
        )
        if not commit["results"][0]["accepted"]:
            raise ValueError(
                f"cannot reconstruct accepted request {request_id}: "
                f"{commit['results'][0]['reason']}"
            )

    return rows, {
        "split": split,
        "source_index": int(source_index),
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "requests": str(requests_path),
        "requests_sha256": _sha256(requests_path),
        "samples": len(rows),
        "failures": sum(row["failure_label"] for row in rows),
        "successes": sum(not row["failure_label"] for row in rows),
        "skipped": dict(skipped),
        "resource_model": resource_metadata,
        "measurement_contract": measurement_contract,
    }


def _auc(probabilities: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = [p for p, y in zip(probabilities, labels) if y == 1]
    negatives = [p for p, y in zip(probabilities, labels) if y == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def _metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    reference_failure_rate: float,
) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("empty or misaligned metric inputs")
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
    prevalence = sum(labels) / len(labels)
    oracle_baseline_brier = sum((prevalence - y) ** 2 for y in labels) / len(labels)
    reference_baseline_brier = sum(
        (reference_failure_rate - y) ** 2 for y in labels
    ) / len(labels)
    epsilon = 1e-7
    log_loss = -sum(
        y * math.log(min(1.0 - epsilon, max(epsilon, p)))
        + (1 - y) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - p)))
        for p, y in zip(probabilities, labels)
    ) / len(labels)
    bins = []
    ece = 0.0
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        selected = [
            (p, y)
            for p, y in zip(probabilities, labels)
            if low <= p < high or (index == 9 and p == 1.0)
        ]
        if not selected:
            continue
        mean_p = sum(row[0] for row in selected) / len(selected)
        observed = sum(row[1] for row in selected) / len(selected)
        ece += len(selected) / len(labels) * abs(mean_p - observed)
        bins.append({
            "lower": low,
            "upper": high,
            "samples": len(selected),
            "mean_probability": mean_p,
            "observed_failure_rate": observed,
        })
    return {
        "samples": len(labels),
        "failures": int(sum(labels)),
        "successes": int(len(labels) - sum(labels)),
        "failure_rate": float(prevalence),
        "roc_auc": _auc(probabilities, labels),
        "brier_score": float(brier),
        "oracle_test_prevalence_brier_score": float(oracle_baseline_brier),
        "training_prevalence_reference": float(reference_failure_rate),
        "training_prevalence_brier_score": float(reference_baseline_brier),
        "brier_skill_score": float(
            1.0 - brier / reference_baseline_brier
            if reference_baseline_brier > 0.0
            else 0.0
        ),
        "log_loss": float(log_loss),
        "expected_calibration_error": float(ece),
        "calibration_bins": bins,
    }


def _tensors(
    rows: Sequence[Mapping[str, Any]], mean: torch.Tensor, std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    labels = torch.tensor([row["failure_label"] for row in rows], dtype=torch.float32)
    return (features - mean) / std, labels


def _probabilities(
    model: nn.Module,
    features: torch.Tensor,
    *,
    logit_scale: float = 1.0,
    logit_bias: float = 0.0,
) -> list[float]:
    model.eval()
    with torch.inference_mode():
        logits = model(features) * float(logit_scale) + float(logit_bias)
        return torch.sigmoid(logits).tolist()


def _fit_platt(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    log_scale = nn.Parameter(torch.zeros(()))
    bias = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_scale, bias], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scale = torch.exp(log_scale).clamp(0.05, 20.0)
        loss = criterion(scale * logits + bias, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_scale).clamp(0.05, 20.0).item()), float(bias.item())


def main() -> int:
    args = parse_args()
    if (
        args.epochs <= 0
        or args.patience <= 0
        or not args.hidden_dims
        or any(value <= 0 for value in args.hidden_dims)
        or not 0.0 <= args.dropout < 1.0
        or args.learning_rate <= 0.0
        or args.weight_decay < 0.0
        or args.minimum_receiver_drain_ms <= 0.0
    ):
        raise ValueError("invalid training hyperparameters")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources = []
    source_index = 0
    for split in ("train", "validation", "test"):
        for raw in getattr(args, f"{split}_source"):
            result_path, requests_path = _source_pair(raw)
            rows, metadata = extract_source(
                result_path,
                requests_path,
                split=split,
                source_index=source_index,
                minimum_receiver_drain_ms=args.minimum_receiver_drain_ms,
            )
            rows_by_split[split].extend(rows)
            sources.append(metadata)
            source_index += 1
    contract_keys = (
        "label_semantics",
        "probe_sender_backend",
        "probe_receiver_backend",
        "vnf_agent_backend",
        "mininet_qdisc",
        "vnf_agent_packet_batch",
        "vnf_agent_q0_packet_batch",
        "vnf_agent_dscp_scheduling",
    )
    reference_contract = dict(sources[0]["measurement_contract"])
    for source in sources[1:]:
        contract = source["measurement_contract"]
        mismatches = [
            key
            for key in contract_keys
            if contract.get(key) != reference_contract.get(key)
        ]
        if mismatches:
            raise ValueError(
                "SLA predictor sources use incompatible runtime contracts: "
                + ", ".join(mismatches)
            )
    measurement_contract = {
        **{key: reference_contract[key] for key in contract_keys},
        "required_receiver_drain_ms": float(args.minimum_receiver_drain_ms),
        "minimum_observed_receiver_drain_ms": min(
            float(source["measurement_contract"][
                "observed_minimum_receiver_drain_ms"
            ])
            for source in sources
        ),
    }
    for split, rows in rows_by_split.items():
        labels = {int(row["failure_label"]) for row in rows}
        if labels != {0, 1}:
            raise ValueError(f"{split} split requires both SLA successes and failures")

    raw_train = torch.tensor(
        [row["features"] for row in rows_by_split["train"]], dtype=torch.float32
    )
    mean = raw_train.mean(dim=0)
    std = raw_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x, train_y = _tensors(rows_by_split["train"], mean, std)
    validation_x, validation_y = _tensors(rows_by_split["validation"], mean, std)
    test_x, test_y = _tensors(rows_by_split["test"], mean, std)

    model = RuntimeSlaRiskNet(
        len(FEATURE_NAMES), args.hidden_dims, dropout=args.dropout
    )
    failures = float(train_y.sum().item())
    successes = float(len(train_y) - failures)
    positive_weight = math.sqrt(successes / max(failures, 1.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    model(validation_x), validation_y
                ).item()
            )
        history.append({
            "epoch": epoch,
            "training_loss": float(loss.item()),
            "validation_log_loss": validation_loss,
        })
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
    model.eval()
    with torch.inference_mode():
        validation_logits = model(validation_x).clone()
    validation_logits = validation_logits.detach().clone()
    logit_scale, logit_bias = _fit_platt(validation_logits, validation_y)

    split_metrics = {}
    training_failure_rate = float(train_y.mean().item())
    tensors_by_split = {
        "train": (train_x, train_y),
        "validation": (validation_x, validation_y),
        "test": (test_x, test_y),
    }
    for split, (features, labels) in tensors_by_split.items():
        probabilities = _probabilities(
            model,
            features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )
        split_metrics[split] = _metrics(
            probabilities,
            labels.int().tolist(),
            reference_failure_rate=training_failure_rate,
        )

    test_metrics = split_metrics["test"]
    deployment_allowed = bool(
        test_metrics["roc_auc"] is not None
        and float(test_metrics["roc_auc"]) >= args.minimum_test_auc
        and float(test_metrics["brier_skill_score"]) >= args.minimum_brier_skill
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "runtime_sla_risk_predictor.pt"
    training_metadata = {
        "samples": len(rows_by_split["train"]),
        "failures": int(train_y.sum().item()),
        "successes": int(len(train_y) - train_y.sum().item()),
        "source_indices": sorted({row["source_index"] for row in rows_by_split["train"]}),
        "best_epoch": best_epoch,
        "positive_weight": positive_weight,
    }
    checkpoint = {
        "version": MODEL_VERSION,
        "prediction_target": PREDICTION_TARGET,
        "feature_names": list(FEATURE_NAMES),
        "hidden_dims": list(map(int, args.hidden_dims)),
        "dropout": float(args.dropout),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "state_dict": best_state,
        "calibration": {
            "method": "platt_scaling_on_disjoint_validation_sources",
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
        },
        "training": training_metadata,
        "validation": split_metrics,
        "measurement_contract": measurement_contract,
        "deployment_allowed": deployment_allowed,
    }
    torch.save(checkpoint, checkpoint_path)
    sample_path = output / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8", newline="\n") as handle:
        for split in ("train", "validation", "test"):
            for row in rows_by_split[split]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "version": MODEL_VERSION,
        "prediction_target": PREDICTION_TARGET,
        "deployment_allowed": deployment_allowed,
        "decision": (
            "passed seed-isolated offline criteria; live A/B is still required"
            if deployment_allowed
            else "failed seed-isolated offline criteria; do not enable online"
        ),
        "feature_names": list(FEATURE_NAMES),
        "checkpoint": str(checkpoint_path),
        "samples": str(sample_path),
        "sources": sources,
        "training": training_metadata,
        "calibration": checkpoint["calibration"],
        "measurement_contract": measurement_contract,
        "splits": split_metrics,
        "criteria": {
            "minimum_test_auc": args.minimum_test_auc,
            "minimum_test_brier_skill": args.minimum_brier_skill,
        },
        "history": history,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
