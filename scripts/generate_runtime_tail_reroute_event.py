#!/usr/bin/env python3
"""Generate one reproducible alternate SFC-tail tree from a prior live result."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sdn_runtime_requests import (
    build_tail_reroute_plan,
    shortest_tree_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--request-id", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    plan = next(
        row for row in result["online_plans"]
        if int(row["request_id"]) == args.request_id
    )
    request_events = [
        row for row in result["events"]
        if int(row["request_id"]) == args.request_id
        and row["type"] in {"arrive", "leave"}
    ]
    arrival = next(float(row["trace_time"]) for row in request_events if row["type"] == "arrive")
    leave = next(float(row["trace_time"]) for row in request_events if row["type"] == "leave")
    event_time = float(args.time) if args.time is not None else arrival + 0.5 * (leave - arrival)
    if not arrival < event_time < leave:
        raise ValueError("reroute time must be inside the active request lifetime")

    multicast = plan["multicast"]
    root = int(multicast["root_dpid"])
    destinations = [int(value) for value in plan["destination_dpids"]]
    old_edges = {tuple(map(int, edge)) for edge in multicast["tree_edges"]}
    chosen = None
    for blocked in sorted(old_edges):
        filtered = copy.deepcopy(profile)
        filtered["edges"] = [
            edge for edge in profile["edges"]
            if {int(edge["u"]), int(edge["v"])} != {blocked[0], blocked[1]}
        ]
        try:
            outputs, paths = shortest_tree_outputs(filtered, root, destinations)
        except ValueError:
            continue
        candidate_edges = {
            (u, v)
            for path in paths.values()
            for u, v in zip(path, path[1:])
        }
        if candidate_edges == old_edges:
            continue
        request = {
            "id": int(plan["request_id"]),
            "source_dpid": int(plan["source_dpid"]),
            "destination_dpids": destinations,
            "vnf": [
                int(plan["placement_by_vnf"][str(stage)]["vnf_type"])
                for stage in range(len(plan["segments"]))
            ],
            "multicast_ip": str(multicast["dst_ip"]),
        }
        reroute = {
            "request_id": args.request_id,
            "time": event_time,
            "policy": "alternate_bfs_tail_tree",
            "trigger": "link_overload",
            "estimated_gain": 1.0,
            "old_edge": list(blocked),
            "old_utilization": 1.0,
            "max_new_utilization": 0.0,
            "paths": paths,
            "switch_outputs": outputs,
        }
        build_tail_reroute_plan(plan, reroute, request, profile)
        chosen = reroute
        break
    if chosen is None:
        raise RuntimeError("no valid alternate tail tree was found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chosen, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "request_id": args.request_id,
        "arrival": arrival,
        "reroute_time": event_time,
        "leave": leave,
        "root_dpid": root,
        "blocked_edge": chosen["old_edge"],
        "old_tree_edges": len(old_edges),
        "new_tree_edges": len({
            (u, v)
            for path in chosen["paths"].values()
            for u, v in zip(path, path[1:])
        }),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
