#!/usr/bin/env python3
"""Run the Telecom Italia driven dynamic SFC migration sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.telecom_italia_migration.simulator import (  # noqa: E402
    ExperimentConfig,
    PAPER_SFC_COUNTS,
    POLICIES,
    run_sweep,
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activity-csv",
        type=Path,
        default=ROOT / "data" / "telecom_italia" / "processed" / "milan_internet_activity.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "runs" / "telecom_italia_migration" / "paper_parameter_sweep",
    )
    parser.add_argument("--sfc-counts", type=parse_ints, default=list(PAPER_SFC_COUNTS))
    parser.add_argument("--policies", type=parse_strings, default=list(POLICIES))
    parser.add_argument("--max-slots", type=int)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = run_sweep(
        args.activity_csv,
        args.output_dir,
        config=ExperimentConfig(seed=args.seed),
        sfc_counts=args.sfc_counts,
        policies=args.policies,
        max_slots=args.max_slots,
    )
    for row in summaries:
        print(
            f"sfc={row['sfc_count']:>2} policy={row['policy']:<22} "
            f"energy={row['average_energy']:.3f} migrations={row['total_migrations']:>4} "
            f"overload={row['average_overloaded_nodes']:.3f} "
            f"sla={row['sla_satisfaction']:.3%}"
        )
    print(f"results={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
