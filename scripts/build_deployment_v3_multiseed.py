#!/usr/bin/env python3
"""Build seed-isolated deployment Top-K batches and exact Oracle labels."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/deployment_v3_multiseed")
    parser.add_argument("--train-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--validation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--test-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--microbatch-ms", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--per-source-rate", type=float, default=3.0)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-agents", type=int, default=32)
    parser.add_argument("--cpu-capacity", type=float, default=55.0)
    parser.add_argument("--memory-capacity", type=float, default=45.0)
    parser.add_argument("--bandwidth-utilization-limit", type=float, default=0.8)
    parser.add_argument("--objective-delay-weight", type=float, default=1.0)
    parser.add_argument("--objective-cpu-weight", type=float, default=0.0)
    parser.add_argument("--objective-memory-weight", type=float, default=0.0)
    parser.add_argument("--objective-bandwidth-weight", type=float, default=0.03)
    parser.add_argument("--objective-pressure-weight", type=float, default=8.0)
    parser.add_argument(
        "--commit-ranking",
        choices=("objective", "pressure", "bandwidth"),
        default="objective",
    )
    parser.add_argument("--oracle-sla-risk-scale", type=float, default=100.0)
    parser.add_argument("--oracle-sla-violation-penalty", type=float, default=500.0)
    parser.add_argument("--oracle-queue-safety-factor", type=float, default=1.0)
    parser.add_argument("--profile", default="sdn/topologies/us_backbone_28_bw90.json")
    parser.add_argument("--lifetime-source", default="data/50node_rate8/phase3_requests_by_slot.pkl")
    parser.add_argument(
        "--baseline-template", default=None,
        help="optional HRL plan path containing a {seed} placeholder",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def window_name(value: float) -> str:
    return f"mb{value:g}ms"


def build_seed(
    seed: int,
    split: str,
    args: argparse.Namespace,
    output: Path,
    profile: Path,
    lifetime_source: Path,
    profile_data: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_trace = output / "traces" / f"seed_{seed}_raw"
    trace = output / "traces" / f"seed_{seed}"
    if args.overwrite or not (raw_trace / "requests.jsonl").exists():
        run([
            sys.executable, str(ROOT / "sdn" / "runtime_request_generator.py"),
            "--profile", str(profile), "--output", str(raw_trace),
            "--seed", str(seed), "--duration", str(args.duration),
            "--per-source-rate", str(args.per_source_rate),
        ])
    if args.overwrite or not (trace / "requests.jsonl").exists():
        run([
            sys.executable, str(ROOT / "scripts" / "remap_runtime_request_lifetimes.py"),
            "--input", str(raw_trace), "--lifetime-source", str(lifetime_source),
            "--output", str(trace), "--shuffle-seed", str(seed),
        ])

    rows = []
    for microbatch_ms in args.microbatch_ms:
        dataset = output / split / f"seed_{seed}" / window_name(microbatch_ms)
        required = (dataset / "batches.jsonl", dataset / "oracle_labels.jsonl")
        if args.overwrite or not all(path.exists() for path in required):
            command = [
                sys.executable, str(ROOT / "scripts" / "generate_deployment_topk_v3.py"),
                "--requests", str(trace / "requests.jsonl"),
                "--profile", str(profile), "--output", str(dataset),
                "--max-requests", str(args.max_requests), "--top-k", str(args.top_k),
                "--max-agents", str(args.max_agents),
                "--microbatch-ms", str(microbatch_ms),
                "--cpu-capacity", str(args.cpu_capacity),
                "--memory-capacity", str(args.memory_capacity),
                "--bandwidth-utilization-limit", str(args.bandwidth_utilization_limit),
                "--objective-delay-weight", str(args.objective_delay_weight),
                "--objective-cpu-weight", str(args.objective_cpu_weight),
                "--objective-memory-weight", str(args.objective_memory_weight),
                "--objective-bandwidth-weight", str(args.objective_bandwidth_weight),
                "--objective-pressure-weight", str(args.objective_pressure_weight),
                "--commit-ranking", args.commit_ranking,
            ]
            if args.baseline_template:
                baseline = resolve(args.baseline_template.format(seed=seed))
                if not baseline.exists():
                    raise FileNotFoundError(f"baseline plan does not exist: {baseline}")
                command.extend(("--baseline-plans", str(baseline)))
            run(command)
            run([
                sys.executable,
                str(ROOT / "scripts" / "generate_deployment_oracle_labels.py"),
                "--data", str(dataset),
                "--sla-risk-scale", str(args.oracle_sla_risk_scale),
                "--sla-violation-penalty", str(args.oracle_sla_violation_penalty),
                "--queue-safety-factor", str(args.oracle_queue_safety_factor),
                "--bandwidth-capacity-mbps", str(
                    float(profile_data.get("default_bandwidth_mbps", 90.0))
                    * args.bandwidth_utilization_limit
                ),
            ])
        summary = load_json(dataset / "summary.json")
        oracle = load_json(dataset / "oracle_summary.json")
        rows.append({
            "split": split,
            "seed": seed,
            "microbatch_ms": float(microbatch_ms),
            "folder": str(dataset.resolve()),
            "requests": int(summary["requests"]),
            "batches": int(summary["batches"]),
            "mean_agents_per_batch": float(summary["mean_agents_per_batch"]),
            "conflict_pairs": int(summary["sparse_conflict_pairs"]),
            "heuristic_accepted": int(summary["accepted_commits"]),
            "oracle_accepted": int(oracle["oracle_accepted"]),
            "optimal_batches": int(oracle["optimal_batches"]),
            "oracle_mean_solve_ms": float(oracle["mean_solve_ms"]),
        })
    return rows


def main() -> int:
    args = parse_args()
    split_seeds = {
        "train": list(args.train_seeds),
        "validation": list(args.validation_seeds),
        "test": list(args.test_seeds),
    }
    all_seeds = [seed for seeds in split_seeds.values() for seed in seeds]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("train/validation/test seeds must be disjoint")
    if (
        args.workers <= 0 or args.max_requests <= 0 or args.top_k <= 0
        or args.max_agents <= 0 or not args.microbatch_ms
        or any(value < 0 for value in args.microbatch_ms)
    ):
        raise ValueError("invalid pipeline parameter")

    output = resolve(args.output)
    profile = resolve(args.profile)
    profile_data = load_json(profile)
    lifetime_source = resolve(args.lifetime_source)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (seed, split)
        for split, seeds in split_seeds.items()
        for seed in seeds
    ]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                build_seed, seed, split, args, output, profile, lifetime_source,
                profile_data
            ): (seed, split)
            for seed, split in jobs
        }
        for future in as_completed(futures):
            seed, split = futures[future]
            seed_rows = future.result()
            rows.extend(seed_rows)
            print(f"completed split={split} seed={seed} windows={len(seed_rows)}", flush=True)

    rows.sort(key=lambda row: (row["split"], row["seed"], row["microbatch_ms"]))
    manifest = {
        "valid": True,
        "dataset_version": "deployment_topk_oracle_v3_multiseed",
        "split_seeds": split_seeds,
        "seed_disjoint": True,
        "microbatch_ms": list(map(float, args.microbatch_ms)),
        "max_requests_per_seed": args.max_requests,
        "per_source_rate": args.per_source_rate,
        "expected_global_rate": 8.0 * args.per_source_rate,
        "bandwidth_utilization_limit": args.bandwidth_utilization_limit,
        "candidate_objective_weights": {
            "delay": args.objective_delay_weight,
            "cpu": args.objective_cpu_weight,
            "memory": args.objective_memory_weight,
            "bandwidth": args.objective_bandwidth_weight,
            "pressure": args.objective_pressure_weight,
        },
        "commit_ranking": args.commit_ranking,
        "oracle_sla_risk_scale": args.oracle_sla_risk_scale,
        "oracle_sla_violation_penalty": args.oracle_sla_violation_penalty,
        "oracle_queue_safety_factor": args.oracle_queue_safety_factor,
        "duration_s": args.duration,
        "baseline_template": args.baseline_template,
        "elapsed_s": time.perf_counter() - started,
        "datasets": rows,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "datasets"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
