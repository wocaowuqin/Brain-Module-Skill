#!/usr/bin/env python3
"""Compare per-seed CSV outputs from SDN runtime algorithm experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PER_SEED_CSV",
        help="algorithm name and aggregate_sdn_runtime_results.py per_seed.csv",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_inputs(values: list[str]) -> dict[str, list[dict[str, str]]]:
    algorithms: dict[str, list[dict[str, str]]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --input {value!r}; expected NAME=CSV")
        name, raw_path = value.split("=", 1)
        with resolve(raw_path).open(encoding="utf-8", newline="") as handle:
            algorithms[name] = list(csv.DictReader(handle))
    return algorithms


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def total(rows: list[dict[str, str]], key: str) -> float:
    return sum(number(row, key) for row in rows)


def average(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(number(row, key) for row in rows)


def summarize(name: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    requests = total(rows, "requests")
    planned = total(rows, "planned_accepted")
    started = total(rows, "started")
    expected = total(rows, "actual_expected_packets")
    lost = total(rows, "actual_lost_packets")
    return {
        "algorithm": name,
        "seeds": len(rows),
        "requests": int(requests),
        "planned_accepted": int(planned),
        "started": int(started),
        "planned_to_started_rate": started / max(1.0, planned),
        "strict_sla": int(total(rows, "strict_sla")),
        "strict_rate": total(rows, "strict_sla") / max(1.0, requests),
        "deployment_queue_wait_rejected": int(
            total(rows, "deployment_queue_wait_rejected")
        ),
        "actual_traffic_loss_rate": lost / max(1.0, expected),
        "planning_mean_ms_macro": average(rows, "planning_mean_ms"),
        "deployment_mean_ms_macro": average(rows, "deployment_mean_ms"),
        "deployment_p95_ms_macro": average(rows, "deployment_p95_ms"),
        "scheduler_lag_p95_ms_macro": average(rows, "scheduler_lag_p95_ms"),
    }


def paired_strict_difference(
    left: list[dict[str, str]], right: list[dict[str, str]]
) -> dict[str, Any]:
    left_by_seed = {row["seed"]: row for row in left}
    right_by_seed = {row["seed"]: row for row in right}
    seeds = sorted(set(left_by_seed) & set(right_by_seed))
    differences = [
        number(left_by_seed[seed], "strict_rate")
        - number(right_by_seed[seed], "strict_rate")
        for seed in seeds
    ]
    mean = statistics.fmean(differences) if differences else math.nan
    std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(differences), 1.96
    )
    margin = t95 * std / math.sqrt(len(differences)) if len(differences) > 1 else math.nan
    return {
        "left_minus_right": mean,
        "paired_seeds": seeds,
        "per_seed_differences": differences,
        "95ci_low": mean - margin,
        "95ci_high": mean + margin,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    algorithms = read_inputs(args.input)
    summaries = [summarize(name, rows) for name, rows in algorithms.items()]
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "algorithm_summary.csv", summaries)

    comparison: dict[str, Any] = {"algorithms": summaries}
    names = list(algorithms)
    if len(names) == 2:
        comparison["paired_strict_rate"] = paired_strict_difference(
            algorithms[names[0]], algorithms[names[1]]
        )
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
