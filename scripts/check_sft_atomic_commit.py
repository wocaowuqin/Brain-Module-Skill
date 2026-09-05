#!/usr/bin/env python3
"""Regression checks for atomic SFT commit and rollback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.modules.AllResourceManager import FusedResourceManager
from envs.modules.low_level_controller import LowLevelController


def build_case(req_id: int):
    topo = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    rm = FusedResourceManager(
        topo=topo,
        capacities={
            "cpu": 100.0,
            "memory": 80.0,
            "bandwidth": 50.0,
            "bandwidth_model": "directed",
        },
        dc_nodes=[1],
    )
    rm.register_request_record(req_id, source=0, dests=[2], vnfs=[0], bw=5.0)
    deploy = rm.try_deploy_new_kernel(
        node=1, vnf_type=0, req_cpu=10.0, req_mem=6.0,
        allow_reuse=False, req_id=req_id,
    )
    if not deploy.ok:
        raise AssertionError(f"test VNF deployment failed: {deploy.reason}")
    rm.bind_vnf_to_request(req_id, 1, 0, deploy, req_cpu=10.0, req_mem=6.0)
    rm.mark_dest_connected(req_id, 2)
    current_tree = {
        "tree": {(0, 1): 1.0, (1, 2): 1.0},
        "placement": {
            (1, 0): {
                "node": 1,
                "vnf_type": 0,
                "cpu_used": 10.0,
                "mem_used": 6.0,
                "reused": False,
                "inst_id": deploy.inst_id,
            }
        },
        "connected_dests": {2},
        "node_stage": {0: 0, 1: 1, 2: 1},
        "tree_usage": {(0, 1): 1, (1, 2): 1},
    }
    env = SimpleNamespace(
        resource_mgr=rm,
        request_manager=rm.request_manager,
        config={},
        n=3,
        current_request={"id": req_id, "bw_origin": 5.0, "dest": [2], "vnf": [0]},
        current_tree=current_tree,
        time_step=1.0,
        _bw_already_rolled_back=False,
    )
    controller = LowLevelController(env)
    before = {
        "bw01": rm.get_available_bandwidth(0, 1),
        "bw12": rm.get_available_bandwidth(1, 2),
    }
    return rm, env, controller, before


def assert_rolled_back(rm, env, before_after_deploy):
    req = rm.request_table[env.current_request["id"]]
    if req.state != "FAILED":
        raise AssertionError(f"request state is {req.state}, expected FAILED")
    if req.edge_allocations or req.vnf_bindings or req.connected_dests:
        raise AssertionError("request ledger/bindings/destinations were not cleared")
    if req.tree_edges or req.placement_by_vnf:
        raise AssertionError("failed request retained an active SFT snapshot")
    if not env._bw_already_rolled_back:
        raise AssertionError("controller did not mark rollback completion")
    if abs(rm.get_available_bandwidth(0, 1) - before_after_deploy["bw01"]) > 1e-6:
        raise AssertionError("bandwidth (0,1) leaked")
    if abs(rm.get_available_bandwidth(1, 2) - before_after_deploy["bw12"]) > 1e-6:
        raise AssertionError("bandwidth (1,2) leaked")
    if abs(rm.pool.get_available_cpu(1) - 100.0) > 1e-6:
        raise AssertionError("CPU was not restored")
    if abs(rm.pool.get_available_memory(1) - 80.0) > 1e-6:
        raise AssertionError("memory was not restored")


def check_mid_commit_failure() -> dict:
    rm, env, controller, before = build_case(3001)
    original_commit = rm.commit_edge_bandwidth
    calls = 0

    def fail_second(req_id, u, v, bw):
        nonlocal calls
        calls += 1
        if calls == 2:
            return False
        return original_commit(req_id, u, v, bw)

    rm.commit_edge_bandwidth = fail_second
    result = controller._commit_episode_bandwidth()
    if result is not False or calls != 2:
        raise AssertionError(f"expected second-edge failure, result={result}, calls={calls}")
    assert_rolled_back(rm, env, before)
    return {"ok": True, "commit_calls": calls, "state": rm.request_table[3001].state}


def check_snapshot_failure() -> dict:
    rm, env, controller, before = build_case(3002)
    calls = 0

    def reject_snapshot(*args, **kwargs):
        nonlocal calls
        calls += 1
        return False

    rm.snapshot_request_sft = reject_snapshot
    result = controller._commit_episode_bandwidth()
    if result is not False or calls != 1:
        raise AssertionError(f"expected snapshot rejection, result={result}, calls={calls}")
    assert_rolled_back(rm, env, before)
    return {"ok": True, "snapshot_calls": calls, "state": rm.request_table[3002].state}


def check_vnf_order_failure() -> dict:
    topo = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    rm = FusedResourceManager(
        topo=topo,
        capacities={
            "cpu": 100.0,
            "memory": 80.0,
            "bandwidth": 50.0,
            "bandwidth_model": "directed",
        },
        dc_nodes=[1, 2],
    )
    req_id = 3003
    rm.register_request_record(req_id, source=0, dests=[3], vnfs=[0, 1], bw=5.0)
    placements = [(2, 0), (1, 1)]
    placement = {}
    for node, vnf_type in placements:
        deploy = rm.try_deploy_new_kernel(
            node=node,
            vnf_type=vnf_type,
            req_cpu=10.0,
            req_mem=6.0,
            allow_reuse=False,
            req_id=req_id,
        )
        if not deploy.ok:
            raise AssertionError(f"test VNF deployment failed: {deploy.reason}")
        rm.bind_vnf_to_request(
            req_id, node, vnf_type, deploy, req_cpu=10.0, req_mem=6.0
        )
        stage = len(placement)
        placement[(node, stage)] = {
            "node": node,
            "vnf_idx": stage,
            "vnf_type": vnf_type,
            "cpu_used": 10.0,
            "mem_used": 6.0,
            "reused": False,
            "inst_id": deploy.inst_id,
        }
    rm.mark_dest_connected(req_id, 3)
    env = SimpleNamespace(
        resource_mgr=rm,
        request_manager=rm.request_manager,
        config={},
        n=4,
        current_request={
            "id": req_id,
            "bw_origin": 5.0,
            "dest": [3],
            "vnf": [0, 1],
        },
        current_tree={
            "tree": {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0},
            "placement": placement,
            "connected_dests": {3},
            "node_stage": {0: 0, 1: 2, 2: 1, 3: 2},
            "tree_usage": {(0, 1): 1, (1, 2): 1, (2, 3): 1},
        },
        time_step=1.0,
        _bw_already_rolled_back=False,
    )
    controller = LowLevelController(env)
    result = controller._commit_episode_bandwidth()
    if result is not False:
        raise AssertionError("reverse VNF order was accepted")
    record = rm.request_table[req_id]
    if record.state != "FAILED":
        raise AssertionError(f"request state is {record.state}, expected FAILED")
    if (
        record.edge_allocations
        or record.vnf_bindings
        or record.connected_dests
        or record.tree_edges
        or record.placement_by_vnf
    ):
        raise AssertionError("ordered-SFT validation failure was not rolled back")
    for edge in ((0, 1), (1, 2), (2, 3)):
        if abs(rm.get_available_bandwidth(*edge) - 50.0) > 1e-6:
            raise AssertionError(f"bandwidth {edge} leaked after order rejection")
    for node in (1, 2):
        if abs(rm.pool.get_available_cpu(node) - 100.0) > 1e-6:
            raise AssertionError(f"CPU on node {node} was not restored")
        if abs(rm.pool.get_available_memory(node) - 80.0) > 1e-6:
            raise AssertionError(f"memory on node {node} was not restored")
    return {"ok": True, "state": record.state, "rollback_complete": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = {
        "ok": True,
        "mid_commit_failure": check_mid_commit_failure(),
        "snapshot_failure": check_snapshot_failure(),
        "vnf_order_failure": check_vnf_order_failure(),
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("SFT atomic commit rollback checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
