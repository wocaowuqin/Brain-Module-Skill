#!/usr/bin/env python3
"""Check validated SFC-tail rerouting and atomic bandwidth replacement."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import AtomicResourceLedger, ResourceFootprint
from scripts.run_sdn_runtime_requests import build_tail_reroute_plan


def main() -> int:
    profile = {
        "dc_nodes_1based": [2],
        "edges": [
            {"u": 1, "v": 2},
            {"u": 2, "v": 3},
            {"u": 2, "v": 4},
            {"u": 4, "v": 3},
        ],
    }
    request = {
        "id": 7,
        "source_dpid": 1,
        "destination_dpids": [3],
        "vnf": [0],
        "multicast_ip": "239.192.0.7",
        "bw_origin": 4.0,
    }
    current = {
        "version": "test",
        "request_id": 7,
        "accepted": True,
        "source_dpid": 1,
        "destination_dpids": [3],
        "chain_nodes": [2],
        "placement_by_vnf": {
            "0": {
                "dc_node": 2,
                "vnf_type": 0,
                "cpu_units": 2.0,
                "memory_units": 3.0,
            }
        },
        "segments": [{"stage": 0, "from_dpid": 1, "to_dpid": 2, "path": [1, 2]}],
        "multicast": {
            "root_dpid": 2,
            "dst_ip": "239.192.0.7",
            "paths": {"3": [2, 3]},
            "tree_edges": [[2, 3]],
            "switch_outputs": {"2": [1], "3": [1]},
        },
    }
    reroute = {
        "time": 0.5,
        "policy": "test_alternate_tree",
        "paths": {"3": [2, 4, 3]},
        "switch_outputs": {"2": [2], "4": [2], "3": [1]},
    }
    original = copy.deepcopy(current)
    replacement = build_tail_reroute_plan(current, reroute, request, profile)
    assert current == original
    assert replacement["segments"] == original["segments"]
    assert replacement["placement_by_vnf"] == original["placement_by_vnf"]
    assert replacement["multicast"]["root_dpid"] == 2
    assert replacement["multicast"]["tree_edges"] == [[2, 4], [4, 3]]

    ledger = AtomicResourceLedger(
        {2: 10.0},
        {2: 10.0},
        {(1, 2): 10.0, (2, 3): 10.0, (2, 4): 10.0, (4, 3): 10.0},
    )
    old_footprint = ResourceFootprint.from_sfc_plan(current, request["bw_origin"])
    new_footprint = ResourceFootprint.from_sfc_plan(replacement, request["bw_origin"])
    committed = ledger.commit_exact(
        [7], [[old_footprint]], [0], expected_version=ledger.snapshot().version
    )
    assert committed["committed"]
    switched = ledger.replace(7, new_footprint)
    assert switched["replaced"]
    snapshot = ledger.snapshot()
    assert snapshot.bandwidth_remaining[(2, 3)] == 10.0
    assert snapshot.bandwidth_remaining[(2, 4)] == 6.0
    assert snapshot.bandwidth_remaining[(4, 3)] == 6.0
    assert ledger.release(7)
    integrity = ledger.integrity_report()
    assert integrity["fully_released"]

    bad_reroute = {**reroute, "paths": {"3": [1, 2, 4, 3]}}
    try:
        build_tail_reroute_plan(current, bad_reroute, request, profile)
    except ValueError as exc:
        assert "invalid endpoints" in str(exc)
    else:
        raise AssertionError("reroute rooted before the last VNF was accepted")

    print(json.dumps({
        "valid": True,
        "old_tree_edges": current["multicast"]["tree_edges"],
        "new_tree_edges": replacement["multicast"]["tree_edges"],
        "ledger_switch": switched,
        "final_integrity": integrity,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
