#!/usr/bin/env python3
"""Sequential, resumable batch runner for actionable_reroute_v2 scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_actionable_reroute_dataset.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate every split/seed as a resumable v2 batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="qmix_sft_v1 root with topology/rate/split/seed folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topo", default="us_backbone")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--node-util-threshold", type=float, default=0.5)
    parser.add_argument("--link-util-threshold", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--deployment-executor",
        default="policy",
        choices=["policy", "bw_planner"],
    )
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 runs every incomplete scenario")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose-runtime", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def discover(source: Path, topo: str, rate: float) -> List[Dict[str, Any]]:
    rate_folder = source / topo / f"rate{rate:g}"
    scenarios = []
    split_order = {"train": 0, "validation": 1, "test": 2}
    for split in ("train", "validation", "test"):
        for request_path in (rate_folder / split).glob("seed_*/requests.pkl"):
            seed = int(request_path.parent.name.split("_", 1)[1])
            with request_path.open("rb") as handle:
                request_count = len(pickle.load(handle))
            scenarios.append({
                "split": split,
                "trace_seed": seed,
                "deploy_seed": seed + 3000,
                "requests": request_path,
                "request_count": request_count,
            })
    scenarios.sort(key=lambda item: (split_order[item["split"]], item["trace_seed"]))
    if not scenarios:
        raise FileNotFoundError(f"no source scenarios under {rate_folder}")
    return scenarios


def spec_path(output: Path, topo: str, rate: float, scenario: Dict[str, Any]) -> Path:
    return (
        output / topo / f"rate{rate:g}" / scenario["split"]
        / f"trace_seed_{scenario['trace_seed']}"
        / f"deploy_seed_{scenario['deploy_seed']}" / "scenario_spec.json"
    )


def completed_spec(
    path: Path,
    episodes: int,
    node_threshold: float,
    link_threshold: float,
    deployment_executor: str,
) -> bool:
    if not path.exists():
        return False
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
        planner = spec.get("planner_config", {})
        return (
            spec.get("dataset_version") == "actionable_reroute_v2"
            and int(spec.get("coverage", {}).get("episodes", 0)) >= episodes
            and bool(spec.get("coverage", {}).get("ledger_consistent", False))
            and spec.get("deployment_executor", "policy") == deployment_executor
            and abs(float(planner.get("node_util_threshold", -1.0)) - node_threshold) <= 1e-9
            and abs(float(planner.get("link_util_threshold", -1.0)) - link_threshold) <= 1e-9
        )
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    source = resolve(args.source)
    output = resolve(args.output)
    checkpoint = resolve(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    scenarios = discover(source, args.topo, args.rate)
    results = []
    run_count = 0
    started = time.time()

    for scenario in scenarios:
        target_spec = spec_path(output, args.topo, args.rate, scenario)
        expected_episodes = (
            scenario["request_count"]
            if args.episodes <= 0
            else min(args.episodes, scenario["request_count"])
        )
        if completed_spec(
            target_spec, expected_episodes,
            args.node_util_threshold, args.link_util_threshold,
            args.deployment_executor,
        ) and not args.rerun_complete:
            results.append({**scenario, "requests": str(scenario["requests"]), "status": "skipped_complete"})
            continue
        if args.max_scenarios > 0 and run_count >= args.max_scenarios:
            results.append({**scenario, "requests": str(scenario["requests"]), "status": "deferred"})
            continue
        if args.dry_run:
            results.append({**scenario, "requests": str(scenario["requests"]), "status": "planned"})
            continue

        command = [
            sys.executable,
            str(GENERATOR),
            "--data", str(scenario["requests"]),
            "--topo", args.topo,
            "--rate", str(args.rate),
            "--split", scenario["split"],
            "--trace-seed", str(scenario["trace_seed"]),
            "--deploy-seed", str(scenario["deploy_seed"]),
            "--checkpoint", str(checkpoint),
            "--episodes", str(args.episodes),
            "--max-steps", str(args.max_steps),
            "--deployment-executor", args.deployment_executor,
            "--top-k", str(args.top_k),
            "--node-util-threshold", str(args.node_util_threshold),
            "--link-util-threshold", str(args.link_util_threshold),
            "--output", str(output),
            "--overwrite",
        ]
        if not args.verbose_runtime:
            command.append("--quiet-runtime")
        scenario_started = time.time()
        print(
            f"running {scenario['split']} seed={scenario['trace_seed']} "
            f"({run_count + 1}/{len(scenarios)})",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        result = {
            **scenario,
            "requests": str(scenario["requests"]),
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - scenario_started,
        }
        results.append(result)
        run_count += 1
        if completed.returncode != 0:
            break

    report = {
        "ok": all(item["status"] in {"completed", "skipped_complete", "deferred", "planned"} for item in results),
        "source": str(source),
        "output": str(output),
        "checkpoint": str(checkpoint),
        "episodes": args.episodes,
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
