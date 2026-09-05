#!/usr/bin/env python3
"""Run the five current HRL ablations on US Backbone at four rates.

This is a no-argument convenience entry point.  It generates the missing
US rate-3.5 trace, then delegates execution and resume handling to
run_all_variants_arrival_sweep.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path.home() / ".conda" / "envs" / "sfc_ppo" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable).resolve()

DATA_ROOT = ROOT / "data" / "rl_arrival_sweep_seed42"
OUTPUT_ROOT = ROOT / "artifacts" / "runs" / "us_ablation_4rates"
RATES = [ "1.5"]
HRL_VARIANTS = [
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
    # Only create the new rate-3.5 trace when it is absent.  Existing US
    # traces are deliberately left untouched for reproducibility.
    rate35 = DATA_ROOT / "us_backbone" / "per_node_rate_3p5" / "phase3_requests.pkl"
    if not rate35.exists():
        run([
            str(PYTHON),
            str(ROOT / "scripts" / "generate_rl_arrival_sweep.py"),
            "--topologies", "us_backbone",
            "--rates", "3.5",
            "--duration", "400",
            "--seed", "42",
            "--output", str(DATA_ROOT),
        ])
    else:
        print(f"Using existing dataset: {rate35}", flush=True)

    run([
        str(PYTHON),
        str(ROOT / "scripts" / "run_all_variants_arrival_sweep.py"),
        "--algorithms", *HRL_VARIANTS,
        "--topologies", "us_backbone",
        "--rates", *RATES,
        "--data-root", str(DATA_ROOT),
        "--output", str(OUTPUT_ROOT),
        "--gpu", "0",
        "--continue-on-error",
    ])
    print(f"Finished. Results: {OUTPUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
