#!/usr/bin/env python3
"""Build a reproducible per-source arrival-rate sweep for RL baselines.

The historical dataset directory names use US-backbone aggregate-rate labels:
4/8/12/16/20/24.  Those labels correspond to per-source rates
0.5/1/1.5/2/2.5/3 because the US topology has eight source nodes.  For the
50-node topology, the aggregate Poisson rate is scaled by its 15 historical
source nodes while retaining the same labels for plot compatibility.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_ROOT = Path.home() / "Desktop" / "hrl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rl_arrival_sweep_seed42"

RATE_LABELS = {
    0.5: 4,
    1.0: 8,
    1.5: 12,
    2.0: 16,
    2.5: 20,
    3.0: 24,
    3.5: 28,
}

SOURCE_PROFILES = {
    "us_backbone": [2, 5, 6, 10, 15, 16, 21, 24],
    "germany50": [1, 2, 7, 8, 10, 13, 16, 18, 27, 30, 31, 36, 37, 41, 43],
}

# DC profiles are recorded in the generated dataset metadata so that a
# dataset cannot be accidentally paired with a different placement topology.
# The US profile below is the legacy profile requested for fair comparison
# with C:/Users/11353/Desktop/hrl.
DC_PROFILES = {
    "us_backbone": [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 27, 28],
    "germany50": [3, 4, 5, 6, 9, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 32, 33, 34, 35, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50],
}

NODE_COUNTS = {"us_backbone": 28, "germany50": 50}
LEGACY_TOPOLOGY_NAMES = {"us_backbone": "us_backbone", "germany50": "50node"}

VNF_TABLE = [
    (2.17571289793936, 1.06847854592062),
    (0.327471689918480, 1.67212031605700),
    (0.691000236990969, 1.84641423103628),
    (0.401957140875833, 1.30930392240156),
    (0.804261986990572, 1.66085909342891),
    (2.83555094177359, 1.09494296372459),
    (1.64532198696671, 0.296237077563595),
    (0.983307643923246, 0.754735724747883),
]


def rate_key(value: float) -> str:
    return str(value).replace(".", "p")


def sample_lifetime(rng: random.Random) -> float:
    while True:
        value = rng.expovariate(1.0 / 2.0)
        if 1.0 <= value <= 6.0:
            return value


def sample_resources(vnfs: list[int], bandwidth: int) -> tuple[list[int], list[int]]:
    cpu = [max(1, min(24, int(round(bandwidth * VNF_TABLE[vnf - 1][0])))) for vnf in vnfs]
    memory = [max(1, min(16, int(round(bandwidth * VNF_TABLE[vnf - 1][1])))) for vnf in vnfs]
    return cpu, memory


def generate_requests(
    topology: str,
    per_source_rate: float,
    duration: float,
    seed: int,
) -> tuple[list[dict], dict[int, list[dict]], dict]:
    sources = SOURCE_PROFILES[topology]
    num_nodes = NODE_COUNTS[topology]
    aggregate_rate = per_source_rate * len(sources)
    request_count = int(round(aggregate_rate * duration))
    rng = random.Random(seed)

    requests: list[dict] = []
    by_slot: dict[int, list[dict]] = {}
    current_time = 0.0
    slot_seconds = 0.1

    for request_id in range(1, request_count + 1):
        current_time += rng.expovariate(aggregate_rate)
        lifetime = sample_lifetime(rng)
        leave_time = current_time + lifetime
        arrival_slot = int(current_time / slot_seconds)
        leave_slot = int(leave_time / slot_seconds)
        source = rng.choice(sources)
        destinations = rng.sample(
            [node for node in range(1, num_nodes + 1) if node != source],
            5,
        )
        vnfs = [rng.randint(1, 8) for _ in range(3)]
        bandwidth = rng.choice([4, 5, 6, 7, 8])
        cpu, memory = sample_resources(vnfs, bandwidth)
        request = {
            "id": request_id,
            "source": source,
            "dest": destinations,
            "vnf": vnfs,
            "bw_origin": bandwidth,
            "cpu_origin": cpu,
            "memory_origin": memory,
            "arrival_time": current_time,
            "leave_time": leave_time,
            "lifetime": lifetime,
            "time_slot": arrival_slot,
            "leave_time_slot": leave_slot,
            "duration": max(1, leave_slot - arrival_slot),
            "arrive_time_step": arrival_slot,
            "leave_time_step": leave_slot,
        }
        requests.append(request)
        by_slot.setdefault(arrival_slot, []).append(request)

    observed_span = requests[-1]["arrival_time"] - requests[0]["arrival_time"]
    metadata = {
        "topology": topology,
        "seed": seed,
        "dc_nodes": DC_PROFILES[topology],
        "dc_profile": "legacy_us_20dc" if topology == "us_backbone" else "python_germany50_35dc",
        "simulation_duration_s": duration,
        "source_nodes": sources,
        "source_count": len(sources),
        "per_source_arrival_rate_req_s": per_source_rate,
        "aggregate_arrival_rate_req_s": aggregate_rate,
        "historical_rate_label": RATE_LABELS[per_source_rate],
        "request_count": request_count,
        "observed_arrival_span_s": observed_span,
        "observed_aggregate_rate_req_s": request_count / observed_span,
        "observed_mean_per_source_rate_req_s": request_count / observed_span / len(sources),
        "lifetime_distribution": "truncated exponential(mean=2s, range=[1,6]s)",
        "bandwidth_values": [4, 5, 6, 7, 8],
        "vnf_chain_length": 3,
        "multicast_destination_count": 5,
    }
    return requests, by_slot, metadata


def copy_existing(
    legacy_root: Path,
    output_root: Path,
    topology: str,
    per_source_rate: float,
) -> bool:
    label = RATE_LABELS[per_source_rate]
    source_dir = legacy_root / "data" / f"{LEGACY_TOPOLOGY_NAMES[topology]}_rate{label}"
    target_dir = output_root / topology / f"per_node_rate_{rate_key(per_source_rate)}"
    files = ("phase3_requests.pkl", "phase3_requests_by_slot.pkl")
    if not all((source_dir / name).exists() for name in files):
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(source_dir / name, target_dir / name)
    with (target_dir / "phase3_requests.pkl").open("rb") as stream:
        requests = pickle.load(stream)
    span = float(requests[-1]["arrival_time"] - requests[0]["arrival_time"])
    sources = sorted({int(row["source"]) for row in requests})
    metadata = {
        "topology": topology,
        "seed": 42,
        "copied_from": str(source_dir.resolve()),
        "dc_nodes": DC_PROFILES[topology],
        "dc_profile": "legacy_us_20dc" if topology == "us_backbone" else "python_germany50_35dc",
        "source_nodes": sources,
        "source_count": len(sources),
        "per_source_arrival_rate_req_s": per_source_rate,
        "aggregate_arrival_rate_req_s": per_source_rate * len(sources),
        "historical_rate_label": label,
        "request_count": len(requests),
        "observed_arrival_span_s": span,
        "observed_aggregate_rate_req_s": len(requests) / span,
        "observed_mean_per_source_rate_req_s": len(requests) / span / len(sources),
    }
    (target_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True


def validate_dataset(path: Path, expected_topology: str, expected_rate: float) -> dict:
    with path.open("rb") as stream:
        requests = pickle.load(stream)
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"empty or invalid request list: {path}")
    required = {
        "id", "source", "dest", "vnf", "bw_origin", "cpu_origin",
        "memory_origin", "arrival_time", "leave_time", "lifetime",
    }
    missing = required - set(requests[0])
    if missing:
        raise ValueError(f"{path} misses fields: {sorted(missing)}")
    arrivals = [float(row["arrival_time"]) for row in requests]
    if any(right <= left for left, right in zip(arrivals, arrivals[1:])):
        raise ValueError(f"arrival times are not strictly increasing: {path}")
    sources = sorted({int(row["source"]) for row in requests})
    expected_sources = SOURCE_PROFILES[expected_topology]
    if sources != expected_sources:
        raise ValueError(f"source profile mismatch in {path}: {sources} != {expected_sources}")
    span = arrivals[-1] - arrivals[0]
    observed_per_source = len(requests) / span / len(sources)
    relative_error = abs(observed_per_source - expected_rate) / expected_rate
    if relative_error > 0.08:
        raise ValueError(
            f"observed per-source rate {observed_per_source:.4f} differs from "
            f"target {expected_rate:.4f} by {relative_error:.2%}"
        )
    return {
        "topology": expected_topology,
        "per_source_rate": expected_rate,
        "requests": len(requests),
        "sources": len(sources),
        "observed_per_source_rate": observed_per_source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", choices=sorted(SOURCE_PROFILES), default=sorted(SOURCE_PROFILES))
    parser.add_argument("--rates", nargs="+", type=float, default=sorted(RATE_LABELS))
    parser.add_argument("--duration", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = []
    for rate in args.rates:
        if rate not in RATE_LABELS:
            raise ValueError(f"unsupported per-source rate {rate}; use {sorted(RATE_LABELS)}")
    for topology in args.topologies:
        for rate in args.rates:
            target = output_root / topology / f"per_node_rate_{rate_key(rate)}"
            copied = False
            if not args.force_generate and rate in {1.0, 2.0, 3.0}:
                copied = copy_existing(args.legacy_root.resolve(), output_root, topology, rate)
            if not copied:
                requests, by_slot, metadata = generate_requests(topology, rate, args.duration, args.seed)
                target.mkdir(parents=True, exist_ok=True)
                with (target / "phase3_requests.pkl").open("wb") as stream:
                    pickle.dump(requests, stream, protocol=pickle.HIGHEST_PROTOCOL)
                with (target / "phase3_requests_by_slot.pkl").open("wb") as stream:
                    pickle.dump(by_slot, stream, protocol=pickle.HIGHEST_PROTOCOL)
                (target / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            row = validate_dataset(target / "phase3_requests.pkl", topology, rate)
            row["mode"] = "copied_existing" if copied else "generated"
            row["path"] = str((target / "phase3_requests.pkl").resolve())
            report.append(row)
            print(
                f"{topology:11s} per-node={rate:>3.1f} requests={row['requests']:>5d} "
                f"observed={row['observed_per_source_rate']:.4f} ({row['mode']})"
            )
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Keep the placement topology next to the request files.  The request
    # records themselves do not contain DC information, so this sidecar is the
    # authoritative profile that must be used by the runtime experiment.
    (output_root / "topology_profile.json").write_text(
        json.dumps(
            {
                "us_backbone": {
                    "node_count": NODE_COUNTS["us_backbone"],
                    "dc_nodes": DC_PROFILES["us_backbone"],
                    "dc_profile": "legacy_us_20dc",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
