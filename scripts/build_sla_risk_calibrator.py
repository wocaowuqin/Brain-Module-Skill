#!/usr/bin/env python3
"""Build an empirical strict-SLA risk model from real runtime probe results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.sla_risk_calibration import (  # noqa: E402
    EmpiricalSlaRiskCalibrator,
    build_calibration_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        type=Path,
        required=True,
        help="runtime result JSON; repeat for independent runs",
    )
    parser.add_argument(
        "--requests",
        action="append",
        type=Path,
        default=[],
        help=(
            "matching requests.jsonl; repeat once per --result, or pass one file "
            "when building from one result"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--minimum-cell-samples", type=int, default=8)
    return parser.parse_args()


def file_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest,
    }


def load_requests(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = int(row["id"])
        if request_id in rows:
            raise ValueError(f"duplicate request {request_id} at {path}:{line_number}")
        rows[request_id] = row
    return rows


def qos_class(request: Mapping[str, Any], receiver_rows: list[Mapping[str, Any]]) -> str:
    value = request.get("qos_class")
    if value:
        return str(value)
    priority = int(request.get("priority", -1))
    if priority in (1, 2, 3):
        return {3: "Q0_REALTIME", 2: "Q1_INTERACTIVE", 1: "Q2_ELASTIC"}[priority]
    bounds = [
        float(row["result"]["delay_bound_ms"])
        for row in receiver_rows
        if isinstance(row.get("result"), Mapping)
        and row["result"].get("delay_bound_ms") is not None
    ]
    return f"delay_bound_{min(bounds):g}ms" if bounds else "unknown"


def extract_samples(
    result_path: Path,
    request_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    requests = load_requests(request_path)
    plans = {
        int(row["request_id"]): row
        for row in payload.get("online_plans", [])
        if row.get("request_id") is not None
    }
    receivers: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload.get("receiver_results", []):
        receivers[int(row["request_id"])].append(row)
    senders: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload.get("sender_results", []):
        senders[int(row["request_id"])].append(row)

    samples: list[dict[str, Any]] = []
    skipped_no_risk = 0
    skipped_not_deployed = 0
    skipped_sender_start_failure = 0
    for event in payload.get("events", []):
        if event.get("type") != "arrive":
            continue
        request_id = int(event["request_id"])
        controller = event.get("controller") or {}
        if not bool(controller.get("accepted")):
            skipped_not_deployed += 1
            continue
        planning = event.get("online_planning") or {}
        risk = planning.get("predicted_sla_risk")
        if risk is None:
            skipped_no_risk += 1
            continue

        request = requests.get(request_id, {})
        plan = plans.get(request_id, {})
        expected_destinations = {
            int(value) for value in request.get("destination_dpids", [])
        }
        if not expected_destinations:
            expected_destinations = {
                int(value) for value in plan.get("destination_dpids", [])
            }
        if not expected_destinations:
            expected_destinations = {
                int(value) for value in (event.get("paths") or {}).keys()
            }
        request_receivers = receivers.get(request_id, [])
        if not expected_destinations:
            expected_destinations = {
                int(row["destination_dpid"]) for row in request_receivers
            }
        receiver_by_destination = {
            int(row["destination_dpid"]): row for row in request_receivers
        }
        receiver_success = bool(expected_destinations) and all(
            destination in receiver_by_destination
            and isinstance(receiver_by_destination[destination].get("result"), Mapping)
            and bool(receiver_by_destination[destination]["result"].get("sla_met"))
            for destination in expected_destinations
        )
        request_senders = senders.get(request_id, [])
        sender_success = bool(request_senders) and all(
            isinstance(row.get("result"), Mapping)
            and row["result"].get("status") == "completed"
            for row in request_senders
        )
        if not sender_success:
            skipped_sender_start_failure += 1
            continue
        bandwidth = request.get("bw_origin", request.get("bandwidth_mbps"))
        if bandwidth is None:
            bandwidth = (event.get("bandwidth_admission") or {}).get(
                "reserved_mbps", 0.0
            )
        samples.append(
            {
                "request_id": request_id,
                "modeled_delay_ratio": float(risk),
                "strict_sla_met": bool(receiver_success),
                "qos_class": qos_class(request, request_receivers),
                "bandwidth_mbps": float(bandwidth),
                "destination_count": len(expected_destinations),
                "sender_started": bool(sender_success),
                "receiver_strict_sla_met": bool(receiver_success),
            }
        )
    return samples, {
        "result": file_metadata(result_path),
        "requests": file_metadata(request_path) if request_path is not None else None,
        "samples": len(samples),
        "strict_sla_successes": sum(row["strict_sla_met"] for row in samples),
        "excluded_sender_start_failures": skipped_sender_start_failure,
        "receiver_sla_failures_after_sender_start": sum(
            row["sender_started"] and not row["receiver_strict_sla_met"]
            for row in samples
        ),
        "skipped_no_modeled_risk": skipped_no_risk,
        "skipped_not_deployed": skipped_not_deployed,
    }


def binary_metrics(probabilities: list[float], outcomes: list[int]) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be non-empty and aligned")
    brier = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes)
    ) / len(outcomes)
    positives = [
        probability
        for probability, outcome in zip(probabilities, outcomes)
        if outcome == 1
    ]
    negatives = [
        probability
        for probability, outcome in zip(probabilities, outcomes)
        if outcome == 0
    ]
    auc = None
    if positives and negatives:
        wins = sum(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        )
        auc = wins / (len(positives) * len(negatives))
    return {
        "samples": len(outcomes),
        "failures": sum(outcomes),
        "brier_score": float(brier),
        "roc_auc": float(auc) if auc is not None else None,
        "mean_predicted_failure_probability": float(
            sum(probabilities) / len(probabilities)
        ),
    }


def leave_one_source_out_metrics(
    samples: list[dict[str, Any]],
    source_count: int,
    *,
    prior_alpha: float,
    prior_beta: float,
    minimum_cell_samples: int,
) -> dict[str, Any] | None:
    if source_count < 2:
        return None
    all_probabilities: list[float] = []
    all_outcomes: list[int] = []
    per_source = []
    for source_index in range(source_count):
        training = [
            row for row in samples if int(row["_source_index"]) != source_index
        ]
        held_out = [
            row for row in samples if int(row["_source_index"]) == source_index
        ]
        if not training or not held_out:
            continue
        training_payload = build_calibration_payload(
            training,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            minimum_cell_samples=minimum_cell_samples,
        )
        calibrator = EmpiricalSlaRiskCalibrator(training_payload)
        probabilities = []
        outcomes = []
        for row in held_out:
            request = {
                "qos_class": row["qos_class"],
                "bandwidth_mbps": row["bandwidth_mbps"],
                "destinations": [0] * int(row["destination_count"]),
            }
            probabilities.append(
                calibrator.predict(row["modeled_delay_ratio"], request)
                .failure_probability
            )
            outcomes.append(int(not row["strict_sla_met"]))
        source_metrics = binary_metrics(probabilities, outcomes)
        source_metrics["source_index"] = source_index
        per_source.append(source_metrics)
        all_probabilities.extend(probabilities)
        all_outcomes.extend(outcomes)
    if not all_outcomes:
        return None
    overall = binary_metrics(all_probabilities, all_outcomes)
    baseline_probability = sum(all_outcomes) / len(all_outcomes)
    baseline_brier = sum(
        (baseline_probability - outcome) ** 2 for outcome in all_outcomes
    ) / len(all_outcomes)
    overall["constant_baseline_brier_score"] = float(baseline_brier)
    overall["brier_skill_score"] = float(
        1.0 - overall["brier_score"] / baseline_brier
        if baseline_brier > 0.0
        else 0.0
    )
    return {
        "split": "leave_one_runtime_source_out",
        "overall": overall,
        "per_source": per_source,
    }


def main() -> None:
    args = parse_args()
    if args.requests and len(args.requests) != len(args.result):
        raise ValueError("repeat --requests exactly once per --result")
    all_samples: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, result_path in enumerate(args.result):
        request_path = args.requests[index] if args.requests else None
        samples, source = extract_samples(result_path, request_path)
        for sample in samples:
            sample["_source_index"] = index
        all_samples.extend(samples)
        sources.append(source)
    payload = build_calibration_payload(
        all_samples,
        sources=sources,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        minimum_cell_samples=args.minimum_cell_samples,
    )
    payload["extraction_summary"] = {
        "deployed_probe_samples": len(all_samples),
        "strict_sla_successes": sum(row["strict_sla_met"] for row in all_samples),
        "strict_sla_failures": sum(not row["strict_sla_met"] for row in all_samples),
        "excluded_sender_start_failures": sum(
            int(row["excluded_sender_start_failures"]) for row in sources
        ),
        "receiver_sla_failures_after_sender_start": sum(
            row["sender_started"] and not row["receiver_strict_sla_met"]
            for row in all_samples
        ),
    }
    validation = leave_one_source_out_metrics(
        all_samples,
        len(sources),
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        minimum_cell_samples=args.minimum_cell_samples,
    )
    if validation is not None:
        for row in validation["per_source"]:
            source_index = int(row["source_index"])
            row["source_path"] = sources[source_index]["result"]["path"]
        payload["validation_metrics"] = validation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sample_count": payload["sample_count"],
        "strict_sla_rate": payload["strict_sla_rate"],
        "prediction_target": payload["prediction_target"],
        "data_readiness": payload["data_readiness"],
        "in_sample_brier_score": payload["fit_metrics"]["in_sample_brier_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
