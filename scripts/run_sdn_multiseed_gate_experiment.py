#!/usr/bin/env python3
"""Run resumable multi-seed SDN baseline, Local, and strict-gate experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7302, 7303, 7304])
    parser.add_argument("--duration", type=float, default=400.0)
    parser.add_argument("--per-source-rate", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument(
        "--output-root", default="artifacts/runs/mininet/sdn_multiseed_gate"
    )
    parser.add_argument("--request-root", default="data/sdn_runtime_requests")
    parser.add_argument("--dataset-root", default="data/sdn_strict_gate_v3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--probe-result-timeout", type=float, default=20.0)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def complete_result(path: Path, expected_requests: int | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    summary = result.get("measurement_summary", {})
    if not result.get("valid") or not summary:
        return False
    return expected_requests is None or int(result.get("requests", -1)) == expected_requests


def run_stage(name: str, command: list[str], log_path: Path, *, skip: bool) -> None:
    if skip:
        print(f"[skip] {name}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run]  {name}", flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"{name} failed with exit code {process.returncode}; "
            f"see {log_path}\n" + "\n".join(tail)
        )


def runtime_command(
    args: argparse.Namespace,
    requests: Path,
    output: Path,
    *,
    plans: Path | None = None,
    reroutes: Path | None = None,
    strict_gate: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_sdn_runtime_requests.py"),
        "--requests",
        str(requests),
        "--output",
        str(output),
        "--timeout",
        str(args.timeout),
        "--probe-result-timeout",
        str(args.probe_result_timeout),
        "--probe-launch-mode",
        "agent",
    ]
    if args.max_requests:
        command.extend(("--max-requests", str(args.max_requests)))
    if plans is not None:
        command.extend(("--tree-plans", str(plans)))
    if reroutes is not None:
        command.extend(("--reroute-events", str(reroutes), "--reroute-drain-seconds", "0.05"))
    if strict_gate:
        command.extend(
            (
                "--strict-reroute-gates",
                "--reroute-min-remaining-lifetime",
                "0.4",
                "--reroute-min-estimated-gain",
                "0.3",
                "--reroute-min-old-utilization",
                "0.75",
                "--reroute-min-utilization-drop",
                "0.2",
                "--reroute-min-cooldown-seconds",
                "1.0",
                "--reroute-live-utilization-threshold",
                "0.6",
            )
        )
    return command


def result_metrics(path: Path) -> dict[str, Any]:
    result = read_json(path)
    summary = result["measurement_summary"]
    return {
        "requests": int(result["requests"]),
        "strict_passes": int(summary["sla_met_requests"]),
        "strict_acceptance": float(summary["request_acceptance_rate"]),
        "receiver_passes": int(summary["sla_met_receivers"]),
        "mean_delay_ms": float(summary["mean_delay_ms"]),
        "mean_packet_loss_rate": float(summary["mean_packet_loss_rate"]),
        "reroutes_requested": int(summary["reroute_events_requested"]),
        "reroutes_applied": int(summary["reroute_events_applied"]),
        "reroutes_skipped": int(summary["reroute_events_skipped"]),
    }


def strict_request_outcomes(result: dict[str, Any]) -> dict[int, bool]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in result["receiver_results"]:
        grouped.setdefault(int(row["request_id"]), []).append(row["result"])
    return {
        request_id: all(bool(value.get("sla_met")) for value in values)
        for request_id, values in grouped.items()
    }


def paired_action_effects(baseline_path: Path, gated_path: Path) -> dict[str, Any]:
    baseline = read_json(baseline_path)
    gated = read_json(gated_path)
    baseline_outcomes = strict_request_outcomes(baseline)
    gated_outcomes = strict_request_outcomes(gated)
    applied_ids = {
        int(event["request_id"])
        for event in gated["events"]
        if event.get("type") == "reroute"
        and bool(event.get("controller", {}).get("accepted"))
    }
    common_ids = set(baseline_outcomes) & set(gated_outcomes)
    if len(common_ids) != len(baseline_outcomes) or len(common_ids) != len(gated_outcomes):
        raise ValueError("baseline and gated runs do not contain identical request results")
    counts = {"beneficial": 0, "harmful": 0, "neutral": 0}
    for request_id in applied_ids:
        before = baseline_outcomes[request_id]
        after = gated_outcomes[request_id]
        if after and not before:
            counts["beneficial"] += 1
        elif before and not after:
            counts["harmful"] += 1
        else:
            counts["neutral"] += 1
    non_target_ids = common_ids - applied_ids
    return {
        "applied_target_requests": len(applied_ids),
        "target_outcomes": counts,
        "target_strict_pass_delta": sum(gated_outcomes[value] for value in applied_ids)
        - sum(baseline_outcomes[value] for value in applied_ids),
        "non_target_strict_pass_delta": sum(
            gated_outcomes[value] for value in non_target_ids
        )
        - sum(baseline_outcomes[value] for value in non_target_ids),
    }


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["gated_minus_baseline"] for row in rows]
    result: dict[str, Any] = {
        "seeds": len(rows),
        "mean_acceptance_delta": statistics.mean(deltas),
        "per_seed_acceptance_delta": deltas,
    }
    if len(deltas) > 1:
        std = statistics.stdev(deltas)
        t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(
            len(deltas) - 1, 1.96
        )
        half_width = t95 * std / len(deltas) ** 0.5
        result.update(
            {
                "sample_std": std,
                "paired_95ci": [
                    result["mean_acceptance_delta"] - half_width,
                    result["mean_acceptance_delta"] + half_width,
                ],
            }
        )
    return result


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.per_source_rate <= 0 or args.max_requests < 0:
        raise ValueError("duration/rate must be positive and max-requests non-negative")
    output_root = resolve(args.output_root)
    request_root = resolve(args.request_root)
    dataset_root = resolve(args.dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for seed in args.seeds:
        seed_output = output_root / f"seed{seed}"
        request_dir = request_root / f"seed_{seed}"
        plan_dir = seed_output / "policy_plans"
        dataset_dir = dataset_root / f"seed{seed}"
        requests = request_dir / "requests.jsonl"
        plans = plan_dir / "local_plans.jsonl"
        reroutes = plan_dir / "local_reroutes.jsonl"
        baseline = seed_output / "baseline.json"
        ungated = seed_output / "local_ungated.json"
        gated = seed_output / "local_gated.json"
        expected_requests = args.max_requests or None

        scenario_ok = False
        scenario_path = request_dir / "scenario.json"
        if scenario_path.is_file():
            scenario = read_json(scenario_path)
            scenario_ok = (
                int(scenario.get("seed", -1)) == seed
                and float(scenario.get("duration_s", -1)) == args.duration
                and float(scenario.get("per_source_rate", -1)) == args.per_source_rate
            )
        run_stage(
            f"seed {seed}: generate requests",
            [
                sys.executable,
                str(ROOT / "sdn" / "runtime_request_generator.py"),
                "--seed",
                str(seed),
                "--duration",
                str(args.duration),
                "--per-source-rate",
                str(args.per_source_rate),
                "--output",
                str(request_dir),
            ],
            seed_output / "generate_requests.log",
            skip=scenario_ok and not args.force,
        )
        plan_ok = plans.is_file() and reroutes.is_file() and (plan_dir / "summary.json").is_file()
        plan_command = [
            sys.executable,
            str(ROOT / "scripts" / "generate_sdn_policy_plans.py"),
            "--requests",
            str(requests),
            "--output",
            str(plan_dir),
            "--policy",
            "local",
        ]
        if args.max_requests:
            plan_command.extend(("--max-requests", str(args.max_requests)))
        run_stage(
            f"seed {seed}: generate Local plans",
            plan_command,
            seed_output / "generate_plans.log",
            skip=plan_ok and not args.force,
        )
        run_stage(
            f"seed {seed}: shortest baseline",
            runtime_command(args, requests, baseline),
            seed_output / "baseline.runner.log",
            skip=complete_result(baseline, expected_requests) and not args.force,
        )
        run_stage(
            f"seed {seed}: all-actions Local label run",
            runtime_command(args, requests, ungated, plans=plans, reroutes=reroutes),
            seed_output / "local_ungated.runner.log",
            skip=complete_result(ungated, expected_requests) and not args.force,
        )
        run_stage(
            f"seed {seed}: strict-gated Local validation",
            runtime_command(
                args,
                requests,
                gated,
                plans=plans,
                reroutes=reroutes,
                strict_gate=True,
            ),
            seed_output / "local_gated.runner.log",
            skip=complete_result(gated, expected_requests) and not args.force,
        )
        dataset_spec = dataset_dir / "dataset_spec.json"
        run_stage(
            f"seed {seed}: build strict-gate labels",
            [
                sys.executable,
                str(ROOT / "scripts" / "build_sdn_strict_gate_dataset.py"),
                "--requests",
                str(requests),
                "--baseline-result",
                str(baseline),
                "--policy-source",
                f"local={reroutes}::{ungated}",
                "--trace-seed",
                str(seed),
                "--output",
                str(dataset_dir),
            ],
            seed_output / "build_dataset.log",
            skip=dataset_spec.is_file() and not args.force,
        )
        baseline_metrics = result_metrics(baseline)
        ungated_metrics = result_metrics(ungated)
        gated_metrics = result_metrics(gated)
        dataset = read_json(dataset_spec)
        seed_row = {
            "seed": seed,
            "baseline": baseline_metrics,
            "local_ungated": ungated_metrics,
            "local_gated": gated_metrics,
            "gated_minus_baseline": gated_metrics["strict_acceptance"]
            - baseline_metrics["strict_acceptance"],
            "paired_action_effects": paired_action_effects(baseline, gated),
            "dataset": {
                "samples": int(dataset["samples"]),
                "execute_labels": int(dataset["execute_labels"]),
                "noop_labels": int(dataset["noop_labels"]),
                "outcomes": dataset["outcomes"],
            },
        }
        (seed_output / "summary.json").write_text(
            json.dumps(seed_row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        seed_rows.append(seed_row)
    summary = {
        "valid": True,
        "protocol": {
            "duration_s": args.duration,
            "per_source_rate": args.per_source_rate,
            "expected_global_rate": 8.0 * args.per_source_rate,
            "max_requests": args.max_requests,
            "probe_launch_mode": "agent",
            "strict_gate": {
                "remaining_lifetime": 0.4,
                "estimated_gain": 0.3,
                "old_utilization": 0.75,
                "utilization_drop": 0.2,
                "live_utilization": 0.6,
                "drain_seconds": 0.05,
            },
        },
        "results": seed_rows,
        "paired_gated_vs_baseline": paired_summary(seed_rows),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
