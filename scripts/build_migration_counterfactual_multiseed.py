#!/usr/bin/env python3
"""Run the migration counterfactual builder over a fixed seed split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seed-disjoint migration counterfactual data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--topo", default="us_backbone")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--delay-weight", type=float, default=0.02)
    parser.add_argument("--hotspot-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trace_root = args.trace_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    manifest = {
        "schema": "migration_counterfactual_multiseed_v1",
        "trace_root": str(args.trace_root),
        "checkpoint": str(args.checkpoint),
        "seeds": list(map(int, args.seeds)),
        "parameters": {
            "episodes": args.episodes,
            "lookahead": args.lookahead,
            "interval": args.interval,
            "max_points": args.max_points,
            "delay_weight": args.delay_weight,
            "hotspot_weight": args.hotspot_weight,
        },
        "files": [],
    }
    started = time.perf_counter()
    python = sys.executable
    for index, seed in enumerate(args.seeds):
        data_path = args.trace_root / f"seed_{seed}" / "requests.pkl"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        output = args.output_dir / f"seed_{seed}.jsonl"
        cmd = [
            python,
            str(ROOT / "scripts" / "build_migration_counterfactual_dataset.py"),
            "--data", str(data_path),
            "--checkpoint", str(args.checkpoint),
            "--topo", args.topo,
            "--seed", str(seed + args.seed_offset),
            "--episodes", str(args.episodes),
            "--lookahead", str(args.lookahead),
            "--interval", str(args.interval),
            "--max-points", str(args.max_points),
            "--max-steps", str(args.max_steps),
            "--delay-weight", str(args.delay_weight),
            "--hotspot-weight", str(args.hotspot_weight),
            "--output", str(output),
        ]
        env = dict(os.environ)
        env.update({
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": str(seed + args.seed_offset),
        })
        print(f"[{index + 1}/{len(args.seeds)}] seed={seed} starting", flush=True)
        completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"counterfactual builder failed for seed {seed}: {completed.returncode}")
        report = output.with_suffix(".report.json")
        manifest["files"].append({
            "seed": seed,
            "data": str(data_path.resolve()),
            "output": str(output.resolve()),
            "report": str(report.resolve()),
        })
        print(f"[{index + 1}/{len(args.seeds)}] seed={seed} complete", flush=True)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
