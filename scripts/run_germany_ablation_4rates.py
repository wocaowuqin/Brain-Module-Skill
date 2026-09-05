#!/usr/bin/env python3
"""Run five HRL variants at six arrival rates on Germany50.

This no-argument entry point prepares the six requested per-source rates and
then delegates the actual experiments to the common resumable orchestrator.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()

DATA_ROOT = ROOT / "data" / "rl_arrival_sweep_seed42"
OUTPUT_ROOT = ROOT / "artifacts" / "runs" / "germany_ablation_4rates"
RATES = ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5"]
# Five requested HRL variants. Checkpoint saving is disabled by default.
ALGORITHMS = [
    "msft_hirl",
    "msft_hrl",
    "msft_ilrl",
    "msft_hirl_gat",
    "msft_hirl_mlp",
]


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index, or -1 for CPU")
    parser.add_argument("--rates", nargs="+", choices=RATES, default=RATES)
    parser.add_argument("--limit-requests", type=int, default=0, help="Smoke test only; 0 uses all requests")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit_requests < 0:
        parser.error("--limit-requests must be nonnegative")
    output = args.output or (
        OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_smoke")
        if args.limit_requests else OUTPUT_ROOT
    )
    missing = [
        rate for rate in args.rates
        if not (DATA_ROOT / "germany50" / f"per_node_rate_{rate.replace('.', 'p')}" / "phase3_requests.pkl").exists()
    ]
    if missing:
        run([
            str(PYTHON),
            str(ROOT / "scripts" / "generate_rl_arrival_sweep.py"),
            "--topologies", "germany50",
            "--rates", *missing,
            "--duration", "400",
            "--seed", "42",
            "--output", str(DATA_ROOT),
        ])
    else:
        print("All Germany50 rate datasets already exist.", flush=True)

    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "run_all_variants_arrival_sweep.py"),
        "--algorithms", *ALGORITHMS,
        "--topologies", "germany50",
        "--rates", *args.rates,
        "--data-root", str(DATA_ROOT),
        "--output", str(output),
        "--python", str(PYTHON),
        "--legacy-root", str(ROOT / "legacy_hrl_runtime"),
        "--gpu", str(args.gpu),
        "--limit-requests", str(args.limit_requests),
        "--continue-on-error",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.force:
        command.append("--force")
    run(command)
    print(f"Finished. Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
