#!/usr/bin/env python3
"""Generate independent Poisson request streams for every non-DC source."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
VNF_CATALOG = json.loads(
    (ROOT / "sdn" / "vnf_catalog.json").read_text(encoding="utf-8")
)
VNF_TYPES = VNF_CATALOG["types"]
VNF_CPU = [
    float(VNF_TYPES[str(index)]["cpu_per_mbps"])
    for index in range(len(VNF_TYPES))
]
VNF_MEMORY = [
    float(VNF_TYPES[str(index)]["memory_per_mbps"])
    for index in range(len(VNF_TYPES))
]
NUM_NODES = 28
DC_NODES_1BASED = {
    1, 3, 4, 7, 8, 9, 11, 12, 13, 14,
    17, 18, 19, 20, 22, 23, 25, 26, 27, 28,
}
SOURCE_NODES_0BASED = [
    node - 1 for node in range(1, NUM_NODES + 1)
    if node not in DC_NODES_1BASED
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/per_source_rate1_test")
    parser.add_argument("--per-source-rate", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=140.0)
    parser.add_argument("--slot-duration", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(7101, 7106)))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def poisson_arrivals(rng: random.Random, rate: float, duration: float) -> list[float]:
    arrivals = []
    current = 0.0
    while rate > 0.0:
        current += rng.expovariate(rate)
        if current >= duration:
            break
        arrivals.append(current)
    return arrivals


def sample_lifetime(rng: random.Random) -> float:
    while True:
        lifetime = rng.expovariate(1.0 / 2.0)
        if 1.0 <= lifetime <= 6.0:
            return lifetime


def resource_demand(vnfs: list[int], bandwidth: int) -> tuple[list[int], list[int]]:
    cpu = [max(1, min(24, round(bandwidth * VNF_CPU[vnf]))) for vnf in vnfs]
    memory = [max(1, min(16, round(bandwidth * VNF_MEMORY[vnf]))) for vnf in vnfs]
    return cpu, memory


def build_indexes(
    requests: list[dict],
) -> tuple[dict[int, list[dict]], list[dict], dict[int, list[dict]]]:
    """Build the legacy slot indexes consumed by the HRL environment."""
    requests_by_slot: dict[int, list[dict]] = {}
    max_slot = 0
    for request in requests:
        requests_by_slot.setdefault(request["time_slot"], []).append(request)
        max_slot = max(max_slot, request["leave_time_slot"])

    events = [
        {"time_slot": slot, "arrive_event": [], "leave_event": []}
        for slot in range(max_slot + 1)
    ]
    events_by_slot: dict[int, list[dict]] = {}
    for request in requests:
        request_id = request["id"]
        arrival_slot = request["time_slot"]
        leave_slot = request["leave_time_slot"]
        events[arrival_slot]["arrive_event"].append(request_id)
        events[leave_slot]["leave_event"].append(request_id)
        events_by_slot.setdefault(arrival_slot, []).append(
            {
                "time": request["arrival_time"],
                "time_slot": arrival_slot,
                "type": "arrive",
                "request": request,
            }
        )
        events_by_slot.setdefault(leave_slot, []).append(
            {
                "time": request["leave_time"],
                "time_slot": leave_slot,
                "type": "leave",
                "request": request,
            }
        )
    return requests_by_slot, events, events_by_slot


def generate(seed: int, rate: float, duration: float, slot_duration: float) -> list[dict]:
    rng = random.Random(seed)
    source_arrivals = []
    for source in SOURCE_NODES_0BASED:
        source_arrivals.extend(
            (arrival, source) for arrival in poisson_arrivals(rng, rate, duration)
        )
    source_arrivals.sort()

    requests = []
    for req_id, (arrival, source) in enumerate(source_arrivals, start=1):
        destination_candidates = [node for node in range(NUM_NODES) if node != source]
        destinations = rng.sample(destination_candidates, 5)
        vnfs = [rng.randrange(8) for _ in range(3)]
        bandwidth = rng.randint(4, 8)
        cpu, memory = resource_demand(vnfs, bandwidth)
        lifetime = sample_lifetime(rng)
        leave_time = arrival + lifetime
        arrive_slot = int(arrival / slot_duration)
        leave_slot = max(arrive_slot + 1, int(math.ceil(leave_time / slot_duration)))
        requests.append({
            "id": req_id,
            "source": source,
            "dest": destinations,
            "vnf": vnfs,
            "bw_origin": bandwidth,
            "cpu_origin": cpu,
            "memory_origin": memory,
            "arrival_time": arrival,
            "leave_time": leave_time,
            "lifetime": lifetime,
            "time_slot": arrive_slot,
            "leave_time_slot": leave_slot,
            "duration": leave_slot - arrive_slot,
            "arrive_time_step": int(math.ceil(arrival)),
            "leave_time_step": int(math.ceil(leave_time)),
            "traffic_profile": "steady_per_source_poisson",
        })
    return requests


def dump(path: Path, value) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    args = parse_args()
    if args.per_source_rate <= 0 or args.duration <= 0 or args.slot_duration <= 0:
        raise ValueError("rates and durations must be positive")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    global_rate = args.per_source_rate * len(SOURCE_NODES_0BASED)
    rate_folder = f"rate{global_rate:g}"
    rows = []

    for seed in args.seeds:
        folder = output / "us_backbone" / rate_folder / "test" / f"seed_{seed}"
        if folder.exists() and not args.overwrite:
            raise FileExistsError(f"dataset exists: {folder}; use --overwrite")
        folder.mkdir(parents=True, exist_ok=True)
        requests = generate(seed, args.per_source_rate, args.duration, args.slot_duration)
        requests_by_slot, events, events_by_slot = build_indexes(requests)
        dump(folder / "requests.pkl", requests)
        dump(folder / "requests_by_slot.pkl", requests_by_slot)
        dump(folder / "events.pkl", events)
        dump(folder / "events_by_slot.pkl", events_by_slot)

        counts = Counter(request["source"] for request in requests)
        row = {
            "seed": seed,
            "requests": len(requests),
            "observed_global_rate": len(requests) / args.duration,
            "mean_destinations": mean(len(request["dest"]) for request in requests),
            "mean_chain_length": mean(len(request["vnf"]) for request in requests),
            "mean_bandwidth": mean(request["bw_origin"] for request in requests),
            "mean_lifetime": mean(request["lifetime"] for request in requests),
        }
        for source in SOURCE_NODES_0BASED:
            row[f"source_{source + 1}_rate"] = counts[source] / args.duration
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    spec = {
        "version": "per_source_poisson_v1",
        "topology": "us_backbone",
        "num_nodes": NUM_NODES,
        "source_nodes_1based": [source + 1 for source in SOURCE_NODES_0BASED],
        "arrival_process": "independent steady Poisson process per source node",
        "per_source_rate": args.per_source_rate,
        "expected_global_rate": global_rate,
        "duration_s": args.duration,
        "slot_duration_s": args.slot_duration,
        "seeds": args.seeds,
        "request_profile": {
            "destinations": 5,
            "vnfs": 3,
            "bandwidth": [4, 8],
            "lifetime_s": [1, 6],
        },
        "reference": "C:/Users/11353/Desktop/hrl/py_split_timeslot/config.py",
    }
    (output / "dataset_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
