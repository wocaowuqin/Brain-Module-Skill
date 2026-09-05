#!/usr/bin/env python3
"""Smoke-check persistent SFT snapshots for research content 2.

This script exercises the first-stage data foundation without running a full
training job. It creates a tiny NFV topology, simulates one successful multicast
SFT deployment, persists the logical SFT snapshot, and validates that the
request record contains both resource-ledger and tree-structure information.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.modules.AllResourceManager import DeployResult, FusedResourceManager
from scripts.export_hrl_sfc_plans import convert_plan


def build_resource_manager() -> FusedResourceManager:
    topo = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=float,
    )
    capacities = {
        "cpu": 100.0,
        "memory": 80.0,
        "bandwidth": 50.0,
        "bandwidth_model": "directed",
    }
    return FusedResourceManager(topo=topo, capacities=capacities, dc_nodes=[1, 2])


def run_smoke_check() -> dict:
    rm = build_resource_manager()
    req_id = 1001

    rm.register_request_record(
        req_id=req_id,
        source=0,
        dests=[2, 3],
        vnfs=[0, 1],
        bw=5.0,
    )

    # Simulate two successful VNF placements written by the deployment path.
    rm.bind_vnf_to_request(
        req_id=req_id,
        node=1,
        vnf_type=0,
        deploy_res=DeployResult(ok=True, inst_id="1:0", new_instance=True),
        req_cpu=10.0,
        req_mem=6.0,
    )
    rm.bind_vnf_to_request(
        req_id=req_id,
        node=2,
        vnf_type=1,
        deploy_res=DeployResult(ok=True, inst_id="2:1", new_instance=True),
        req_cpu=12.0,
        req_mem=8.0,
    )

    for dest in (2, 3):
        rm.mark_dest_connected(req_id, dest)

    # Simulate delayed bandwidth commit.
    for u, v in ((0, 1), (1, 2), (2, 3)):
        if not rm.commit_edge_bandwidth(req_id, u, v, 5.0):
            raise RuntimeError(f"failed to commit edge bandwidth: {(u, v)}")

    current_tree = {
        # The post-chain destination connector is deliberately recorded only
        # as reverse flow=0 evidence. Snapshot construction must orient it in
        # service order as (2, 3).
        "tree": {(0, 1): 1.0, (1, 2): 1.0, (3, 2): 0.0},
        "placement": {
            (1, 0): {
                "node": 1,
                "vnf_type": 0,
                "cpu_used": 10.0,
                "mem_used": 6.0,
                "reused": False,
                "inst_id": "1:0",
            },
            (2, 1): {
                "node": 2,
                "vnf_type": 1,
                "cpu_used": 12.0,
                "mem_used": 8.0,
                "reused": False,
                "inst_id": "2:1",
            },
        },
        "node_stage": {0: 0, 1: 1, 2: 2, 3: 2},
        "tree_usage": {(0, 1): 1, (1, 2): 1, (2, 3): 1},
        "connected_dests": {2, 3},
    }
    if not rm.snapshot_request_sft(
        req_id, current_tree=current_tree, snapshot_time=12.5
    ):
        raise AssertionError("valid ordered SFT snapshot was rejected")

    report = rm.validate_request_sft_snapshot(req_id)
    if not report["ok"]:
        raise AssertionError(f"SFT snapshot validation failed: {report}")
    record = rm.request_table[req_id]
    if set(record.tree_edges) != {(0, 1), (1, 2), (2, 3)}:
        raise AssertionError("flow=0 connector was not recovered into the canonical rooted tree")

    # A plain source-rooted BFS would orient destination 2 before VNF stage 1
    # at node 3. Ordered canonicalization must build the service spine first.
    crossing_rm = build_resource_manager()
    crossing_rm.register_request_record(
        req_id=1002,
        source=0,
        dests=[2],
        vnfs=[0, 1],
        bw=5.0,
    )
    crossing_tree = {
        "tree": {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (1, 3): 1.0},
        "placement": {
            (1, 0): {"node": 1, "vnf_type": 0},
            (3, 1): {"node": 3, "vnf_type": 1},
        },
        "connected_dests": {2},
    }
    crossing = crossing_rm.canonicalize_request_sft(
        1002, crossing_tree, allow_topology_repair=False
    )
    if not crossing.get("ok"):
        raise AssertionError(f"ordered crossing canonicalization failed: {crossing}")
    expected_crossing = {(0, 1), (1, 3), (3, 2)}
    if set(crossing["tree_edges"]) != expected_crossing:
        raise AssertionError(
            f"canonicalization ignored service order: {crossing['tree_edges']}"
        )

    profile = {
        "nodes": [
            {"dpid": node, "host_port": 1, "host_ip": f"10.0.0.{node}/24"}
            for node in range(1, 5)
        ],
        "edges": [
            {"u": 1, "v": 2, "u_port": 2, "v_port": 2},
            {"u": 2, "v": 3, "u_port": 3, "v_port": 2},
            {"u": 3, "v": 4, "u_port": 3, "v_port": 2},
        ],
    }
    plan = convert_plan(
        {
            "success": True,
            "chain_nodes": [record.placement_by_vnf[index] for index in range(2)],
            "tree_snapshot": {"tree": dict(record.tree_edges)},
            "steps": 3,
        },
        {
            "id": req_id,
            "source": 0,
            "source_dpid": 1,
            "dest": [2, 3],
            "destination_dpids": [3, 4],
            "vnf": [0, 1],
            "cpu_origin": [10, 12],
            "memory_origin": [6, 8],
            "multicast_ip": "239.192.3.233",
            "udp_port": 6001,
        },
        profile,
        stage_port_base=20000,
    )
    ledger_edges = set(record.tree_edges)
    exported_segment_edges = {
        (u - 1, v - 1)
        for segment in plan["segments"]
        for u, v in zip(segment["path"], segment["path"][1:])
    }
    if not exported_segment_edges.issubset(ledger_edges):
        raise AssertionError(
            "successful export contains a segment edge absent from the ledger"
        )
    report["exported_segment_edges"] = len(exported_segment_edges)
    report["ordered_crossing_edges"] = sorted(
        [list(edge) for edge in crossing["tree_edges"]]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    report = run_smoke_check()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("SFT snapshot smoke check passed")
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
