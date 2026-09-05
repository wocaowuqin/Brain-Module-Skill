#!/usr/bin/env python3
"""Aggregate role-reconfiguration pilot outputs across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METHODS = ["no_reconfig", "rule", "idqn", "qmix"]
METRICS = ["accept_rate", "node_hotspots", "link_hotspots", "avg_delay_ms",
           "max_delay_ms", "migrations", "reroutes", "failed_actions"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="artifacts/runs/reconfiguration/experiments/pilot"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3101, 3102])
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for seed in args.seeds:
        for method in METHODS:
            folder = root / f"test{seed}_{method}"
            summary = json.loads((folder / "role_reconfig_summary.json").read_text(encoding="utf-8"))
            with (folder / "role_reconfig_eval.csv").open(encoding="utf-8-sig", newline="") as handle:
                eval_rows = list(csv.DictReader(handle))
            metrics = summary["final_metrics"]
            rows.append({
                "seed": seed, "method": method,
                "accept_rate": float(summary["accept_rate"]),
                "node_hotspots": int(metrics["node_hotspots"]),
                "link_hotspots": int(metrics["link_hotspots"]),
                "avg_delay_ms": float(metrics["avg_delay_total_ms"]),
                "max_delay_ms": float(metrics["max_delay_total_ms"]),
                "migrations": int(metrics["total_migrations"]),
                "reroutes": int(metrics["total_reconfigs"]),
                "failed_actions": sum(row.get("selected_action") == "failed" for row in eval_rows),
            })
    with (root / "pilot_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        result = {"method": method, "seeds": len(method_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in method_rows]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate.append(result)
    with (root / "pilot_aggregate.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (root / "pilot_results.json").write_text(
        json.dumps({"raw": rows, "aggregate": aggregate}, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
