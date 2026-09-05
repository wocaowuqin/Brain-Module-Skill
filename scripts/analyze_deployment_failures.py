#!/usr/bin/env python3
"""Summarize online deployment outcomes in actionable-reroute datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    return parser.parse_args()


def avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    root = parse_args().dataset
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_seed: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    by_phase: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    load: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in root.rglob("states.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                deployment = row.get("deployment", {})
                success = bool(deployment.get("success"))
                reason = "success" if success else (
                    str(deployment.get("fail_reason") or "unknown")
                )
                split = str(row.get("split", "unknown"))
                seed = int(row.get("trace_seed", -1))
                sim_time = float(row.get("sim_time", 0.0))
                phase = "warmup" if sim_time < 20.0 else (
                    "mixed" if sim_time < 120.0 else "cooldown"
                )
                outcome = "success" if success else "failure"

                by_split[split][reason] += 1
                by_seed[(split, seed)][reason] += 1
                by_phase[(split, phase)][outcome] += 1

                metrics = row.get("metrics", {})
                network = row.get("network", {})
                active_sfts = row.get("active_sfts", [])
                bucket = load[(split, outcome)]
                bucket["active_sfts"].append(float(len(active_sfts)))
                for name in (
                    "node_util_mean", "node_util_max",
                    "link_util_mean", "link_util_max",
                ):
                    value = metrics.get(name, network.get(name))
                    if isinstance(value, (int, float)):
                        bucket[name].append(float(value))

    print("FAILURE REASONS")
    for split in ("train", "validation", "test"):
        counts = by_split[split]
        total = sum(counts.values())
        print(f"\n{split}: total={total}")
        for reason, count in counts.most_common():
            print(f"  {reason:32s} {count:5d}  {count / max(1, total):7.2%}")

    print("\nPER SEED")
    for (split, seed), counts in sorted(by_seed.items()):
        total = sum(counts.values())
        successes = counts["success"]
        main_failure = next(
            ((name, count) for name, count in counts.most_common()
             if name != "success"),
            ("none", 0),
        )
        print(
            f"  {split:10s} {seed}: {successes:4d}/{total:<4d} "
            f"({successes / max(1, total):7.2%}), "
            f"main_failure={main_failure[0]}:{main_failure[1]}"
        )

    print("\nTRAFFIC PHASE")
    for (split, phase), counts in sorted(by_phase.items()):
        total = sum(counts.values())
        print(
            f"  {split:10s} {phase:8s}: {counts['success']:4d}/{total:<4d} "
            f"({counts['success'] / max(1, total):7.2%})"
        )

    print("\nSTATE LOAD (MEAN)")
    for key in sorted(load):
        values = load[key]
        fields = ", ".join(
            f"{name}={avg(samples):.3f}"
            for name, samples in sorted(values.items())
        )
        print(f"  {key[0]:10s} {key[1]:7s}: {fields}")


if __name__ == "__main__":
    main()
