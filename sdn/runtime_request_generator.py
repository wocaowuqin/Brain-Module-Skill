#!/usr/bin/env python3
"""Generate reproducible Poisson SFT requests for Mininet runtime replay."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "sdn" / "topologies" / "us_backbone_28.json"
DEFAULT_QOS = ROOT / "sdn" / "qos_profiles.json"
DEFAULT_VNFS = ROOT / "sdn" / "vnf_catalog.json"


def load_json(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return json.loads(value.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def poisson_arrivals(rng: random.Random, rate: float, duration: float) -> Iterable[float]:
    current = 0.0
    while rate > 0.0:
        current += rng.expovariate(rate)
        if current >= duration:
            return
        yield current


def sample_lifetime(
    rng: random.Random,
    minimum: float,
    maximum: float,
    mean_value: float,
) -> float:
    scale = max(0.01, mean_value - minimum)
    while True:
        value = minimum + rng.expovariate(1.0 / scale)
        if value <= maximum:
            return value


def choose_qos(rng: random.Random, qos_profile: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    classes = qos_profile.get("classes", {})
    if not classes:
        raise ValueError("QoS profile has no classes")
    names = list(classes)
    weights = [float(classes[name].get("weight", 0.0)) for name in names]
    if sum(weights) <= 0.0:
        raise ValueError("QoS class weights must have a positive sum")
    name = rng.choices(names, weights=weights, k=1)[0]
    return name, dict(classes[name])


def multicast_ip(request_id: int) -> str:
    index = int(request_id) - 1
    if index < 0 or index >= 254 * 256:
        raise ValueError("request id exceeds the 239.192.0.0/16 multicast pool")
    return f"239.192.{index // 254}.{index % 254 + 1}"


def resource_demand(
    vnf_chain: list[int], bandwidth_mbps: int, vnf_catalog: dict[str, Any]
) -> tuple[list[int], list[int]]:
    types = vnf_catalog.get("types", {})
    cpu = []
    memory = []
    for vnf_type in vnf_chain:
        definition = types.get(str(vnf_type))
        if definition is None:
            raise ValueError(f"VNF catalog has no type {vnf_type}")
        cpu.append(
            max(1, min(24, round(bandwidth_mbps * float(definition["cpu_per_mbps"]))))
        )
        memory.append(
            max(
                1,
                min(
                    16,
                    round(bandwidth_mbps * float(definition["memory_per_mbps"])),
                ),
            )
        )
    return cpu, memory


def validate_inputs(
    profile: dict[str, Any],
    qos_profile: dict[str, Any],
    vnf_catalog: dict[str, Any],
) -> None:
    nodes = profile.get("nodes", [])
    dpids = {int(node["dpid"]) for node in nodes}
    if not dpids or len(dpids) != len(nodes):
        raise ValueError("topology profile has missing or duplicate DPIDs")
    dc_nodes = {int(value) for value in profile.get("dc_nodes_1based", [])}
    if not dc_nodes or not dc_nodes.issubset(dpids):
        raise ValueError("topology profile has invalid DC nodes")
    choose_qos(random.Random(0), qos_profile)
    types = vnf_catalog.get("types", {})
    if len(types) < 1 or any(str(index) not in types for index in range(len(types))):
        raise ValueError("VNF catalog type ids must be contiguous from zero")


def generate_requests(
    profile: dict[str, Any],
    qos_profile: dict[str, Any],
    vnf_catalog: dict[str, Any],
    *,
    seed: int,
    duration: float,
    per_source_rate: float,
    destination_count: int = 5,
    chain_length: int = 3,
    bandwidth_min: int = 4,
    bandwidth_max: int = 8,
    lifetime_min: float = 1.0,
    lifetime_max: float = 6.0,
    lifetime_mean: float = 2.0,
    port_base: int = 5001,
) -> list[dict[str, Any]]:
    validate_inputs(profile, qos_profile, vnf_catalog)
    if duration <= 0.0 or per_source_rate <= 0.0:
        raise ValueError("duration and per-source rate must be positive")
    if bandwidth_min <= 0 or bandwidth_max < bandwidth_min:
        raise ValueError("invalid bandwidth range")
    if not 0.0 < lifetime_min <= lifetime_mean <= lifetime_max:
        raise ValueError("lifetime must satisfy min <= mean <= max")

    dpids = sorted(int(node["dpid"]) for node in profile["nodes"])
    dc_nodes = {int(value) for value in profile["dc_nodes_1based"]}
    sources = [dpid for dpid in dpids if dpid not in dc_nodes]
    if not sources:
        sources = list(dpids)
    if destination_count <= 0 or destination_count >= len(dpids):
        raise ValueError("destination count must be in [1, node_count - 1]")
    vnf_count = len(vnf_catalog["types"])
    if chain_length <= 0 or chain_length > vnf_count:
        raise ValueError("chain length exceeds the distinct VNF type count")

    arrivals = []
    for source_dpid in sources:
        arrival_rng = random.Random((int(seed) << 16) ^ source_dpid)
        arrivals.extend(
            (arrival, source_dpid)
            for arrival in poisson_arrivals(arrival_rng, per_source_rate, duration)
        )
    arrivals.sort()
    if port_base + len(arrivals) - 1 > 65535:
        raise ValueError("unique UDP port range exceeds 65535")

    attribute_rng = random.Random(int(seed) ^ 0x5DEECE66D)
    requests = []
    for request_id, (arrival, source_dpid) in enumerate(arrivals, start=1):
        destinations = attribute_rng.sample(
            [dpid for dpid in dpids if dpid != source_dpid], destination_count
        )
        vnf_chain = attribute_rng.sample(range(vnf_count), chain_length)
        bandwidth = attribute_rng.randint(bandwidth_min, bandwidth_max)
        cpu, memory = resource_demand(vnf_chain, bandwidth, vnf_catalog)
        qos_class, qos = choose_qos(attribute_rng, qos_profile)
        lifetime = sample_lifetime(
            attribute_rng, lifetime_min, lifetime_max, lifetime_mean
        )
        leave_time = arrival + lifetime
        requests.append(
            {
                "id": request_id,
                "source": source_dpid - 1,
                "source_dpid": source_dpid,
                "dest": [dpid - 1 for dpid in destinations],
                "destination_dpids": destinations,
                "vnf": vnf_chain,
                "bw_origin": bandwidth,
                "cpu_origin": cpu,
                "memory_origin": memory,
                "qos_class": qos_class,
                "delay_bound_ms": float(qos["delay_bound_ms"]),
                "delay_compliance_ratio": float(qos["delay_compliance_ratio"]),
                "jitter_bound_ms": (
                    None
                    if qos.get("jitter_bound_ms") is None
                    else float(qos["jitter_bound_ms"])
                ),
                "packet_loss_bound": float(qos["packet_loss_bound"]),
                "dscp": int(qos["dscp"]),
                "priority": int(qos["priority"]),
                "arrival_time": arrival,
                "leave_time": leave_time,
                "lifetime": lifetime,
                "multicast_ip": multicast_ip(request_id),
                "udp_port": port_base + request_id - 1,
                "group_id": request_id,
                "traffic_profile": "steady_per_source_poisson_qos",
            }
        )
    return requests


def request_events(requests: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for request in requests:
        events.append(
            {
                "time": float(request["arrival_time"]),
                "type": "arrive",
                "request_id": int(request["id"]),
            }
        )
        events.append(
            {
                "time": float(request["leave_time"]),
                "type": "leave",
                "request_id": int(request["id"]),
            }
        )
    events.sort(key=lambda item: (item["time"], 0 if item["type"] == "leave" else 1))
    return events


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_trace(
    output: Path,
    requests: list[dict[str, Any]],
    events: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    requests_jsonl = output / "requests.jsonl"
    events_jsonl = output / "events.jsonl"
    requests_pickle = output / "requests.pkl"
    write_jsonl(requests_jsonl, requests)
    write_jsonl(events_jsonl, events)
    with requests_pickle.open("wb") as handle:
        pickle.dump(requests, handle, protocol=pickle.HIGHEST_PROTOCOL)

    qos_counts = Counter(row["qos_class"] for row in requests)
    source_counts = Counter(int(row["source_dpid"]) for row in requests)
    result = {
        **metadata,
        "requests": len(requests),
        "qos_counts": dict(sorted(qos_counts.items())),
        "source_counts": {str(key): source_counts[key] for key in sorted(source_counts)},
        "files": {
            "requests.jsonl": sha256(requests_jsonl),
            "events.jsonl": sha256(events_jsonl),
            "requests.pkl": sha256(requests_pickle),
        },
    }
    (output / "scenario.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic per-source Poisson requests for Mininet replay."
    )
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--qos-profile", default=str(DEFAULT_QOS))
    parser.add_argument("--vnf-catalog", default=str(DEFAULT_VNFS))
    parser.add_argument("--output", default="data/sdn_runtime_requests/seed_7301")
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--duration", type=float, default=400.0)
    parser.add_argument("--per-source-rate", type=float, default=1.0)
    parser.add_argument("--destinations", type=int, default=5)
    parser.add_argument("--chain-length", type=int, default=3)
    parser.add_argument("--bandwidth-min", type=int, default=4)
    parser.add_argument("--bandwidth-max", type=int, default=8)
    parser.add_argument("--lifetime-min", type=float, default=1.0)
    parser.add_argument("--lifetime-max", type=float, default=6.0)
    parser.add_argument("--lifetime-mean", type=float, default=2.0)
    parser.add_argument("--port-base", type=int, default=5001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_json(args.profile)
    qos_profile = load_json(args.qos_profile)
    vnf_catalog = load_json(args.vnf_catalog)
    requests = generate_requests(
        profile,
        qos_profile,
        vnf_catalog,
        seed=args.seed,
        duration=args.duration,
        per_source_rate=args.per_source_rate,
        destination_count=args.destinations,
        chain_length=args.chain_length,
        bandwidth_min=args.bandwidth_min,
        bandwidth_max=args.bandwidth_max,
        lifetime_min=args.lifetime_min,
        lifetime_max=args.lifetime_max,
        lifetime_mean=args.lifetime_mean,
        port_base=args.port_base,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    result = write_trace(
        output,
        requests,
        request_events(requests),
        {
            "version": "sdn_runtime_requests_v1",
            "topology": str(profile.get("name", Path(args.profile).stem)),
            "seed": args.seed,
            "duration_s": args.duration,
            "per_source_rate": args.per_source_rate,
            "expected_global_rate": args.per_source_rate
            * len(
                set(int(node["dpid"]) for node in profile["nodes"])
                - set(int(value) for value in profile["dc_nodes_1based"])
            ),
            "source_nodes_1based": sorted(
                set(int(node["dpid"]) for node in profile["nodes"])
                - set(int(value) for value in profile["dc_nodes_1based"])
            ),
            "qos_profile_version": qos_profile.get("version"),
            "vnf_catalog_version": vnf_catalog.get("version"),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
