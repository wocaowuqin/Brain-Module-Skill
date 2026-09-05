#!/usr/bin/env python3
"""Export per-request HRL decision timings from an SDN runtime JSON result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


TIMING_GROUPS = (
    "high_level",
    "high_vnf_placement",
    "high_destination_connection",
    "low_level",
    "low_vnf_routing",
    "low_destination_routing",
)
GROUP_FIELDS = ("count", "total_ms", "mean_ms", "p50_ms", "p95_ms", "max_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _value(mapping: Mapping[str, Any], path: Iterable[str], default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key, default)
    return current


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _aggregate(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values),
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    plans = payload.get("online_plans") or []
    output = args.output or args.input.with_name(f"{args.input.stem}_hrl_timings.csv")
    summary_path = args.summary or output.with_suffix(".summary.json")

    fieldnames = [
        "request_id", "accepted", "reason", "inference_ms", "conversion_ms",
        "online_total_ms", "request_algorithm_total_ms", "decision_total_ms",
        "non_action_algorithm_ms",
    ]
    fieldnames.extend(
        f"{group}_{field}" for group in TIMING_GROUPS for field in GROUP_FIELDS
    )

    rows = []
    for plan in plans:
        online = plan.get("online_planning") or {}
        algorithm = online.get("algorithm_timing") or {}
        row = {
            "request_id": int(plan.get("request_id", online.get("request_id", -1))),
            "accepted": int(bool(plan.get("accepted", False))),
            "reason": str(plan.get("reason", "")),
            "inference_ms": online.get("inference_ms"),
            "conversion_ms": online.get("conversion_ms"),
            "online_total_ms": online.get("total_ms"),
            "request_algorithm_total_ms": algorithm.get("request_total_ms"),
            "decision_total_ms": algorithm.get("decision_total_ms"),
            "non_action_algorithm_ms": algorithm.get("non_action_algorithm_ms"),
        }
        for group in TIMING_GROUPS:
            group_values = algorithm.get(group) or {}
            for field in GROUP_FIELDS:
                row[f"{group}_{field}"] = group_values.get(field)
        rows.append(row)

    if not rows:
        raise ValueError("input contains no online_plans")
    missing = [row["request_id"] for row in rows if row["request_algorithm_total_ms"] is None]
    if missing:
        raise ValueError(
            "input predates granular HRL timing instrumentation; missing request ids: "
            + ", ".join(map(str, missing[:20]))
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_fields = (
        "inference_ms", "online_total_ms", "request_algorithm_total_ms",
        "decision_total_ms", "high_level_mean_ms", "low_level_mean_ms",
    )
    summary = {
        "valid": True,
        "input": str(args.input.resolve()),
        "output": str(output.resolve()),
        "requests": len(rows),
        "accepted": sum(row["accepted"] for row in rows),
        "metrics": {
            field: _aggregate([
                float(row[field]) for row in rows if row.get(field) is not None
            ])
            for field in aggregate_fields
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
