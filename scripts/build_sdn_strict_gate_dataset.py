#!/usr/bin/env python3
"""Build paired strict-SLA execute/no-op labels from live SDN policy runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = [
    "remaining_lifetime_norm",
    "bandwidth_norm",
    "qos_realtime",
    "qos_interactive",
    "qos_elastic",
    "old_utilization_norm",
    "new_utilization_norm",
    "utilization_drop_norm",
    "estimated_gain_norm",
    "extra_edges_norm",
    "old_delay_norm",
    "new_delay_norm",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receiver_metrics(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in result["receiver_results"]:
        grouped.setdefault(int(row["request_id"]), []).append(row["result"])
    metrics = {}
    for request_id, rows in grouped.items():
        metrics[request_id] = {
            "strict_sla": all(bool(row.get("sla_met")) for row in rows),
            "receiver_sla_count": sum(bool(row.get("sla_met")) for row in rows),
            "loss_violation_count": sum(not bool(row.get("loss_sla_met")) for row in rows),
            "delay_violation_count": sum(not bool(row.get("delay_sla_met")) for row in rows),
            "jitter_violation_count": sum(not bool(row.get("jitter_sla_met")) for row in rows),
            "mean_loss_rate": sum(float(row.get("packet_loss_rate", 1.0)) for row in rows)
            / max(1, len(rows)),
        }
    return metrics


def reroute_key(row: dict[str, Any], *, result_event: bool = False) -> tuple[int, float]:
    time_key = "trace_time" if result_event else "time"
    return int(row["request_id"]), round(float(row[time_key]), 6)


def validate_all_actions_applied(
    policy: str,
    actions: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, int]:
    planned = Counter(reroute_key(row) for row in actions)
    reroute_events = [
        row
        for row in result.get("events", [])
        if row.get("type") == "reroute"
        and str(row.get("reroute_policy", policy)) == policy
    ]
    observed = Counter(reroute_key(row, result_event=True) for row in reroute_events)
    applied = Counter(
        reroute_key(row, result_event=True)
        for row in reroute_events
        if bool(row.get("controller", {}).get("accepted"))
    )
    missing = planned - observed
    not_applied = planned - applied
    unexpected = observed - planned
    if missing or not_applied or unexpected:
        def preview(values: Counter[tuple[int, float]]) -> list[list[int | float]]:
            return [list(key) for key in list(values.elements())[:10]]

        raise ValueError(
            f"{policy} result is not an all-actions-applied label run: "
            f"planned={sum(planned.values())}, observed={sum(observed.values())}, "
            f"applied={sum(applied.values())}, missing={preview(missing)}, "
            f"not_applied={preview(not_applied)}, unexpected={preview(unexpected)}"
        )
    return {
        "planned": sum(planned.values()),
        "observed": sum(observed.values()),
        "applied": sum(applied.values()),
    }


def feature_vector(request: dict[str, Any], action: dict[str, Any]) -> list[float]:
    qos = str(request["qos_class"])
    remaining = max(0.0, float(request["leave_time"]) - float(action["time"]))
    old_util = max(0.0, float(action.get("old_utilization", 0.0)))
    new_util = max(0.0, float(action.get("max_new_utilization", old_util)))
    return [
        min(remaining / 6.0, 1.0),
        min(float(request["bw_origin"]) / 10.0, 1.0),
        1.0 if qos == "Q0_REALTIME" else 0.0,
        1.0 if qos == "Q1_INTERACTIVE" else 0.0,
        1.0 if qos == "Q2_ELASTIC" else 0.0,
        min(old_util / 1.5, 1.0),
        min(new_util / 1.5, 1.0),
        min(max(0.0, old_util - new_util) / 1.5, 1.0),
        min(max(0.0, float(action.get("estimated_gain", 0.0))), 1.0),
        min(max(0.0, float(action.get("extra_edges", 0.0))) / 2.0, 1.0),
        min(max(0.0, float(action.get("old_delay_ms", 2.0))) / 20.0, 1.0),
        min(max(0.0, float(action.get("new_path_delay_ms", 2.0))) / 20.0, 1.0),
    ]


def parse_source(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError(
            "policy source must be NAME=ACTIONS_JSONL::RESULT_JSON"
        )
    separator = "::" if "::" in parts[1] else ":"
    if separator not in parts[1] or (
        separator == ":" and parts[1].count(":") != 1
    ):
        raise ValueError(
            "policy source must be NAME=ACTIONS_JSONL::RESULT_JSON; "
            "use the double-colon form for absolute Windows paths"
        )
    actions, result = parts[1].split(separator, 1)
    return parts[0], Path(actions), Path(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--baseline-result", required=True)
    parser.add_argument(
        "--policy-source",
        action="append",
        required=True,
        help="NAME=ACTIONS_JSONL::RESULT_JSON; repeat for Local/QMIX",
    )
    parser.add_argument("--trace-seed", type=int, required=True)
    parser.add_argument("--output", default="data/sdn_strict_gate_v3/seed7301")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_path = Path(args.requests).resolve()
    baseline_path = Path(args.baseline_result).resolve()
    output_dir = Path(args.output).resolve()
    requests = {int(row["id"]): row for row in read_jsonl(request_path)}
    baseline = receiver_metrics(read_json(baseline_path))
    rows = []
    source_files = []
    action_execution = []
    for policy, raw_actions, raw_result in map(parse_source, args.policy_source):
        action_path = raw_actions.resolve()
        result_path = raw_result.resolve()
        source_files.extend((action_path, result_path))
        actions = read_jsonl(action_path)
        policy_result = read_json(result_path)
        execution = validate_all_actions_applied(policy, actions, policy_result)
        action_execution.append({"source_policy": policy, **execution})
        policy_metrics = receiver_metrics(policy_result)
        for action in actions:
            request_id = int(action["request_id"])
            if request_id not in baseline or request_id not in policy_metrics:
                raise ValueError(
                    f"request {request_id} lacks complete baseline or {policy} receiver metrics"
                )
            base = baseline[request_id]
            observed = policy_metrics[request_id]
            strict_delta = int(observed["strict_sla"]) - int(base["strict_sla"])
            receiver_delta = (
                int(observed["receiver_sla_count"])
                - int(base["receiver_sla_count"])
            )
            loss_violation_delta = (
                int(observed["loss_violation_count"])
                - int(base["loss_violation_count"])
            )
            delay_violation_delta = (
                int(observed["delay_violation_count"])
                - int(base["delay_violation_count"])
            )
            jitter_violation_delta = (
                int(observed["jitter_violation_count"])
                - int(base["jitter_violation_count"])
            )
            reward = (
                10.0 * strict_delta
                + 2.0 * receiver_delta / 5.0
                - 4.0 * max(0, loss_violation_delta)
                - 2.0 * max(0, delay_violation_delta)
                - 1.0 * max(0, jitter_violation_delta)
                - 0.5
                - 0.2 * max(0, int(action.get("extra_edges", 0)))
            )
            if strict_delta > 0:
                outcome = "beneficial"
            elif strict_delta < 0:
                outcome = "harmful"
            elif base["strict_sla"]:
                outcome = "neutral_pass"
            else:
                outcome = "neutral_fail"
            rows.append(
                {
                    "dataset_version": "sdn_strict_gate_v3",
                    "trace_seed": int(args.trace_seed),
                    "source_policy": policy,
                    "time": float(action["time"]),
                    "request_id": request_id,
                    "candidate": {
                        key: action.get(key)
                        for key in (
                            "old_edge",
                            "new_path",
                            "estimated_gain",
                            "old_utilization",
                            "max_new_utilization",
                            "extra_edges",
                        )
                    },
                    "feature_names": FEATURE_NAMES,
                    "features": feature_vector(requests[request_id], action),
                    "execute_label": 1 if outcome == "beneficial" else 0,
                    "outcome": outcome,
                    "strict_delta": strict_delta,
                    "receiver_sla_delta": receiver_delta,
                    "reward": reward,
                    "baseline_metrics": base,
                    "observed_metrics": observed,
                }
            )
    rows.sort(key=lambda row: (row["time"], row["source_policy"], row["request_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "candidates.jsonl"
    with data_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = Counter(row["outcome"] for row in rows)
    spec = {
        "dataset_version": "sdn_strict_gate_v3",
        "purpose": "calibration_only_until_multiple_disjoint_trace_seeds_exist",
        "trace_seed": int(args.trace_seed),
        "samples": len(rows),
        "execute_labels": sum(row["execute_label"] for row in rows),
        "noop_labels": sum(not row["execute_label"] for row in rows),
        "outcomes": dict(counts),
        "action_execution": action_execution,
        "feature_names": FEATURE_NAMES,
        "requests_file": str(request_path),
        "requests_sha256": sha256(request_path),
        "baseline_result": str(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in source_files
        ],
        "data_file": str(data_path),
        "data_sha256": sha256(data_path),
    }
    (output_dir / "dataset_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
