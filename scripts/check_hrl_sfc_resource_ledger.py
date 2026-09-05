#!/usr/bin/env python3
"""Validate resource-ledger snapshots emitted with executable HRL SFC plans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.sfc_plan import directed_physical_edges, plan_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True, help="HRL SFC JSONL plans")
    parser.add_argument("--resource-csv", help="optional exporter resource CSV")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nonnegative(value: Any, label: str, errors: list[str]) -> float:
    number = float(value)
    if number < -1e-8:
        errors.append(f"{label} is negative: {number}")
    return number


def main() -> None:
    args = parse_args()
    plans = read_jsonl(resolve(args.plans))
    errors: list[str] = []
    accepted = 0

    for plan in plans:
        request_id = int(plan.get("request_id", -1))
        ledger = plan.get("resource_ledger")
        if not isinstance(ledger, dict) or not ledger.get("available"):
            errors.append(f"request {request_id}: missing authoritative resource_ledger")
            continue

        for resource, unit in (("cpu", "units"), ("memory", "units"), ("bandwidth", "mbps")):
            detail = ledger.get(resource, {})
            used = nonnegative(detail.get(f"used_{unit}", 0.0), f"request {request_id} {resource} used", errors)
            capacity = nonnegative(detail.get(f"capacity_{unit}", 0.0), f"request {request_id} {resource} capacity", errors)
            if capacity <= 0.0:
                errors.append(f"request {request_id}: {resource} capacity is not positive")
            elif used > capacity + 1e-6:
                errors.append(f"request {request_id}: {resource} pool over capacity ({used}>{capacity})")

        if plan.get("accepted"):
            accepted += 1
            try:
                if plan_format(plan) != "hrl_sfc_plan_v1":
                    errors.append(f"request {request_id}: accepted plan is not hrl_sfc_plan_v1")
                elif not directed_physical_edges(plan):
                    errors.append(f"request {request_id}: accepted plan has no physical edges")
            except ValueError as exc:
                errors.append(str(exc))

    csv_rows = None
    if args.resource_csv:
        with resolve(args.resource_csv).open(encoding="utf-8", newline="") as handle:
            csv_rows = sum(1 for _ in csv.DictReader(handle))
        if csv_rows != len(plans):
            errors.append(f"resource CSV has {csv_rows} rows but JSONL has {len(plans)} plans")

    result = {
        "valid": not errors,
        "plans": len(plans),
        "accepted": accepted,
        "resource_csv_rows": csv_rows,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
