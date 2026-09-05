#!/usr/bin/env python3
"""Convert the legacy full-8 trace into the current SDN runtime schema."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdn.runtime_request_generator import (  # noqa: E402
    choose_qos,
    load_json,
    multicast_ip,
    request_events,
    write_trace,
)


DEFAULT_LEGACY_REQUESTS = Path(
    "C:/Users/11353/Desktop/hrl/data/us_backbone_rate8/phase3_requests.pkl"
)
DEFAULT_EPISODE_CSV = Path(
    "C:/Users/11353/Desktop/结果/us1-6数据/full-8-episodes.csv"
)
DEFAULT_QOS_PROFILE = ROOT / "sdn/qos_profiles.json"
DEFAULT_OUTPUT = ROOT / "data/sdn_runtime_requests/full8_same_trace_qos2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-requests", default=str(DEFAULT_LEGACY_REQUESTS))
    parser.add_argument("--episode-csv", default=str(DEFAULT_EPISODE_CSV))
    parser.add_argument("--qos-profile", default=str(DEFAULT_QOS_PROFILE))
    parser.add_argument("--qos-seed", type=int, default=2026)
    parser.add_argument("--port-base", type=int, default=5001)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"unsupported legacy request type: {type(value)!r}")


def load_legacy_requests(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    requests = {}
    for item in rows:
        request = as_dict(item)
        request_id = int(request["id"])
        if request_id in requests:
            raise ValueError(f"duplicate legacy request id: {request_id}")
        requests[request_id] = request
    return requests


def load_episode_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("episode CSV is empty")
    ids = [int(row["request_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("episode CSV contains duplicate request ids")
    arrivals = [float(row["arrival_time"]) for row in rows]
    if arrivals != sorted(arrivals):
        raise ValueError("episode CSV arrival_time is not monotonic")
    return rows


def verify_legacy_mapping(
    episode_rows: list[dict[str, str]],
    legacy_by_id: dict[int, dict[str, Any]],
) -> dict[str, float]:
    max_errors = {
        "actual_bw_per_req": 0.0,
        "cpu_resource_comp": 0.0,
        "mem_resource_comp": 0.0,
        "bw_resource_comp": 0.0,
    }
    for row in episode_rows:
        request_id = int(row["request_id"])
        if request_id not in legacy_by_id:
            raise ValueError(f"request {request_id} is missing from legacy PKL")
        request = legacy_by_id[request_id]
        success = int(row["success"]) == 1
        tree_edges = int(float(row["tree_len"] or 0))
        explored_edges = int(float(row["directed_tree_edges"] or 0))
        bandwidth = float(request["bw_origin"])
        lifetime = float(request["lifetime"])
        expected = {
            "actual_bw_per_req": bandwidth * explored_edges,
            "cpu_resource_comp": (
                sum(float(value) for value in request["cpu_origin"]) * lifetime
                if success else 0.0
            ),
            "mem_resource_comp": (
                sum(float(value) for value in request["memory_origin"]) * lifetime
                if success else 0.0
            ),
            "bw_resource_comp": (
                bandwidth * tree_edges * lifetime if success else 0.0
            ),
        }
        for field, expected_value in expected.items():
            observed = float(row[field] or 0.0)
            max_errors[field] = max(max_errors[field], abs(observed - expected_value))
    if any(error > 5e-4 for error in max_errors.values()):
        raise ValueError(f"legacy PKL/CSV mapping mismatch: {max_errors}")
    return max_errors


def convert(
    episode_rows: list[dict[str, str]],
    legacy_by_id: dict[int, dict[str, Any]],
    qos_profile: dict[str, Any],
    qos_seed: int,
    port_base: int,
) -> list[dict[str, Any]]:
    if port_base + max(legacy_by_id) - 1 > 65535:
        raise ValueError("UDP port range exceeds 65535")
    qos_rng = random.Random(qos_seed)
    converted = []
    for row in episode_rows:
        request_id = int(row["request_id"])
        legacy = legacy_by_id[request_id]
        source = int(legacy["source"])
        destinations = [int(value) for value in legacy["dest"]]
        if source in destinations:
            raise ValueError(f"request {request_id} contains its source as a destination")
        qos_class, qos = choose_qos(qos_rng, qos_profile)
        arrival_time = float(row["arrival_time"])
        lifetime = float(legacy["lifetime"])
        converted.append(
            {
                "id": request_id,
                "source": source,
                "source_dpid": source + 1,
                "dest": destinations,
                "destination_dpids": [value + 1 for value in destinations],
                "vnf": [int(value) for value in legacy["vnf"]],
                "bw_origin": int(legacy["bw_origin"]),
                "cpu_origin": [int(value) for value in legacy["cpu_origin"]],
                "memory_origin": [int(value) for value in legacy["memory_origin"]],
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
                "arrival_time": arrival_time,
                "leave_time": arrival_time + lifetime,
                "lifetime": lifetime,
                "multicast_ip": multicast_ip(request_id),
                "udp_port": port_base + request_id - 1,
                "group_id": request_id,
                "traffic_profile": "legacy_full8_same_trace_qos_overlay",
                "legacy_episode": int(row["episode"]),
            }
        )
    return converted


def main() -> int:
    args = parse_args()
    legacy_path = Path(args.legacy_requests).resolve()
    episode_path = Path(args.episode_csv).resolve()
    qos_path = Path(args.qos_profile).resolve()
    output = Path(args.output).resolve()

    legacy_by_id = load_legacy_requests(legacy_path)
    episode_rows = load_episode_rows(episode_path)
    if len(legacy_by_id) != len(episode_rows):
        raise ValueError(
            f"row count mismatch: PKL={len(legacy_by_id)} CSV={len(episode_rows)}"
        )
    max_mapping_errors = verify_legacy_mapping(episode_rows, legacy_by_id)
    qos_profile = load_json(qos_path)
    requests = convert(
        episode_rows,
        legacy_by_id,
        qos_profile,
        args.qos_seed,
        args.port_base,
    )
    qos_counts = Counter(request["qos_class"] for request in requests)
    result = write_trace(
        output,
        requests,
        request_events(requests),
        {
            "version": "legacy_full8_sdn_runtime_v1",
            "topology": "us_backbone_28",
            "request_order": "full-8-episodes.csv row order",
            "arrival_time_source": "full-8-episodes.csv",
            "request_attributes_source": "legacy phase3_requests.pkl",
            "qos_assignment": "deterministic overlay; QoS was absent from the legacy trace",
            "qos_seed": int(args.qos_seed),
            "legacy_requests_file": str(legacy_path),
            "legacy_requests_sha256": sha256(legacy_path),
            "episode_csv_file": str(episode_path),
            "episode_csv_sha256": sha256(episode_path),
            "qos_profile_file": str(qos_path),
            "qos_profile_sha256": sha256(qos_path),
            "legacy_mapping_max_abs_error": max_mapping_errors,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "requests": len(requests),
                "first_request_id": requests[0]["id"],
                "last_request_id": requests[-1]["id"],
                "first_arrival": requests[0]["arrival_time"],
                "last_arrival": requests[-1]["arrival_time"],
                "qos_counts": dict(sorted(qos_counts.items())),
                "mapping_verified": True,
                "scenario": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
