#!/usr/bin/env python3
"""Run the bounded checks shipped in the minimal delivery bundle."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "train_tahrl.py",
    "requirements.txt",
    "ilModel/us/il_model_best.pth",
    "ilModel/50/il_model_best.pth",
    "configs/multiagent_orchestration.yaml",
    "data/rl_arrival_sweep_seed42/germany50/per_node_rate_0p5/phase3_requests.pkl",
    "data/sdn_runtime_requests/seed_7071_lifetime50node_rate8/requests.jsonl",
    "data/migration_wqmix_smoke_seed7071/batches.jsonl",
    "artifacts/runs/hrl/export_current_seed7071_rate8_100_mask_fix_v2/plans.jsonl",
    "artifacts/reference_runs/migration/migration_baselines_smoke/migration_wqmix.pt",
    "artifacts/reference_runs/migration/runtime_rate8_seed7071_first100.json",
    "sdn/topologies/us_backbone_28_bw90.json",
)

CHECKS = (
    (
        "low candidate/final-mask alignment",
        "-m",
        "unittest",
        "tests.test_low_candidate_mask_alignment",
        "-v",
    ),
    (
        "multiagent orchestration",
        "scripts/check_multiagent_orchestration.py",
        "--json",
    ),
    (
        "atomic replacement ledger",
        "scripts/check_atomic_ledger_replacement.py",
    ),
    (
        "runtime migration transaction",
        "scripts/check_migration_runtime_transaction.py",
    ),
    (
        "rate8 SFT migration candidates",
        "scripts/check_rate8_sft_migration_regression.py",
    ),
    (
        "migration WQMIX pipeline",
        "scripts/check_migration_wqmix_pipeline.py",
        "--data",
        "data/migration_wqmix_smoke_seed7071",
        "--checkpoint",
        "artifacts/reference_runs/migration/migration_baselines_smoke/migration_wqmix.pt",
        "--runtime-report",
        "artifacts/reference_runs/migration/runtime_rate8_seed7071_first100.json",
        "--profile",
        "sdn/topologies/us_backbone_28_bw90.json",
    ),
)


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    if missing:
        print("DELIVERY BUNDLE: FAIL")
        print("Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    failures = []
    for check in CHECKS:
        label, *arguments = check
        command = [sys.executable, *arguments]
        print(f"\n[{label}]", flush=True)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
        )
        if completed.returncode != 0:
            failures.append((label, completed.returncode))

    if failures:
        print("\nDELIVERY BUNDLE: FAIL")
        for label, returncode in failures:
            print(f"  - {label}: exit {returncode}")
        return 1
    print("\nDELIVERY BUNDLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
