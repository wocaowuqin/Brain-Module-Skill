#!/usr/bin/env python3
"""Create a runtime trace with lifetimes matched to an empirical dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdn.runtime_request_generator import request_events, write_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source runtime trace directory")
    parser.add_argument("--lifetime-source", required=True, help="empirical request PKL")
    parser.add_argument("--output", required=True, help="new runtime trace directory")
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument(
        "--allow-replacement",
        action="store_true",
        help=(
            "bootstrap lifetimes with replacement when the target trace is "
            "larger than the empirical source"
        ),
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_requests(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(value):
            rows.extend(value[key])
        return rows
    raise TypeError(f"unsupported lifetime source type: {type(value).__name__}")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_s": min(values),
        "p05_s": percentile(values, 0.05),
        "p25_s": percentile(values, 0.25),
        "median_s": percentile(values, 0.50),
        "mean_s": statistics.fmean(values),
        "std_s": statistics.pstdev(values),
        "p75_s": percentile(values, 0.75),
        "p95_s": percentile(values, 0.95),
        "max_s": max(values),
    }


def quantile_sample(
    values: list[float],
    count: int,
    *,
    allow_replacement: bool = False,
    seed: int = 0,
) -> list[float]:
    if count > len(values):
        if not allow_replacement:
            raise ValueError(
                f"target has {count} requests but lifetime source has only {len(values)}"
            )
        # Deterministic empirical bootstrap for long traces.  Sampling whole
        # shuffled source cycles preserves the source distribution more
        # closely than independent draws while still avoiding a periodic
        # lifetime order in the target trace.
        rng = random.Random(seed)
        selected: list[float] = []
        while len(selected) < count:
            cycle = list(values)
            rng.shuffle(cycle)
            selected.extend(cycle[: count - len(selected)])
        rng.shuffle(selected)
        return selected
    ordered = sorted(values)
    selected = [
        ordered[min(int((index + 0.5) * len(ordered) / count), len(ordered) - 1)]
        for index in range(count)
    ]
    if len(set(selected)) != len(selected) and len(set(values)) == len(values):
        raise AssertionError("quantile selection unexpectedly reused source samples")
    return selected


def main() -> None:
    args = parse_args()
    input_dir = resolve(args.input)
    source_path = resolve(args.lifetime_source)
    output_dir = resolve(args.output)
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("output must not overwrite the source trace")

    requests = read_jsonl(input_dir / "requests.jsonl")
    scenario = json.loads((input_dir / "scenario.json").read_text(encoding="utf-8"))
    with source_path.open("rb") as handle:
        empirical_requests = flatten_requests(pickle.load(handle))
    empirical_lifetimes = [float(row["lifetime"]) for row in empirical_requests]
    if not empirical_lifetimes or any(
        not math.isfinite(value) or value <= 0.0 for value in empirical_lifetimes
    ):
        raise ValueError("lifetime source contains invalid values")

    original_lifetimes = [float(row["lifetime"]) for row in requests]
    selected_lifetimes = quantile_sample(
        empirical_lifetimes,
        len(requests),
        allow_replacement=args.allow_replacement,
        seed=args.shuffle_seed,
    )
    random.Random(args.shuffle_seed).shuffle(selected_lifetimes)

    time_fields = {"lifetime", "leave_time"}
    original_non_time = [
        {key: value for key, value in row.items() if key not in time_fields}
        for row in requests
    ]
    remapped = []
    for request, lifetime in zip(requests, selected_lifetimes):
        row = dict(request)
        row["lifetime"] = lifetime
        row["leave_time"] = float(row["arrival_time"]) + lifetime
        remapped.append(row)
    remapped_non_time = [
        {key: value for key, value in row.items() if key not in time_fields}
        for row in remapped
    ]
    if remapped_non_time != original_non_time:
        raise AssertionError("a non-time request field changed during remapping")

    source_summary = describe(empirical_lifetimes)
    selected_summary = describe(selected_lifetimes)
    metadata = {
        **{key: value for key, value in scenario.items() if key not in {"files", "requests", "qos_counts", "source_counts"}},
        "version": "sdn_runtime_requests_v1_empirical_lifetime",
        "parent_trace": str(input_dir.relative_to(ROOT)).replace("\\", "/"),
        "parent_requests_sha256": sha256(input_dir / "requests.jsonl"),
        "lifetime_transform": {
            "method": (
                "empirical_cycle_bootstrap_with_replacement_seeded_shuffle"
                if len(requests) > len(empirical_lifetimes)
                else "empirical_quantile_without_replacement_seeded_shuffle"
            ),
            "shuffle_seed": args.shuffle_seed,
            "source": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256(source_path),
            "source_distribution": source_summary,
            "original_distribution": describe(original_lifetimes),
            "selected_distribution": selected_summary,
        },
        "max_leave_time_s": max(float(row["leave_time"]) for row in remapped),
    }
    result = write_trace(output_dir, remapped, request_events(remapped), metadata)

    roundtrip_json = read_jsonl(output_dir / "requests.jsonl")
    with (output_dir / "requests.pkl").open("rb") as handle:
        roundtrip_pickle = pickle.load(handle)
    if roundtrip_json != remapped or roundtrip_pickle != remapped:
        raise AssertionError("serialized trace does not match remapped requests")

    print(
        json.dumps(
            {
                "output": str(output_dir),
                "requests": len(remapped),
                "events": len(request_events(remapped)),
                "arrival_times_unchanged": True,
                "non_time_fields_unchanged": True,
                "original_lifetime": describe(original_lifetimes),
                "source_lifetime": source_summary,
                "remapped_lifetime": selected_summary,
                "max_leave_time_s": result["max_leave_time_s"],
                "files": result["files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
