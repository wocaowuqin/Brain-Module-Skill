#!/usr/bin/env python3
"""Smoke-check the heuristic reconfiguration manager.

The script builds a tiny active SFT scenario with a hot VNF-hosting node and a
hot tree edge, then verifies hotspot detection, Top-K SFT selection, baseline
planning, and safe local execution for greedy migration/rerouting.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.modules.AllResourceManager import FusedResourceManager
from envs.modules.reconfiguration_manager import ReconfigurationManager


def build_resource_manager() -> FusedResourceManager:
    topo = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=float,
    )
    capacities = {
        "cpu": 100.0,
        "memory": 80.0,
        "bandwidth": 50.0,
        "bandwidth_model": "directed",
    }
    return FusedResourceManager(topo=topo, capacities=capacities, dc_nodes=[1, 2, 4])


def deploy_active_sft(rm: FusedResourceManager, req_id: int = 2001) -> None:
    rm.register_request_record(req_id=req_id, source=0, dests=[3, 5], vnfs=[0, 1], bw=10.0)

    first = rm.try_deploy_new_kernel(1, 0, req_cpu=72.0, req_mem=10.0, allow_reuse=True, req_id=req_id)
    second = rm.try_deploy_new_kernel(2, 1, req_cpu=10.0, req_mem=8.0, allow_reuse=True, req_id=req_id)
    if not first.ok or not second.ok:
        raise RuntimeError("failed to create active VNF instances")

    rm.bind_vnf_to_request(req_id, 1, 0, first, req_cpu=72.0, req_mem=10.0)
    rm.bind_vnf_to_request(req_id, 2, 1, second, req_cpu=10.0, req_mem=8.0)

    # Keep additional load on node 1 so moving the 72-CPU VNF to an empty
    # in-tree node reduces the peak instead of merely moving it unchanged.
    if not rm.allocate_node_resource(1, 7, 10.0, 0.0):
        raise RuntimeError("failed to create background source-node pressure")

    for dest in (3, 5):
        rm.mark_dest_connected(req_id, dest)

    # The complete chain precedes both branches. Node 4 is an in-tree,
    # stage-safe migration target; 2->5->3 is a safe reroute for hot edge 2->3.
    tree_edges = ((0, 1), (1, 4), (4, 2), (2, 3), (2, 5))
    for u, v in tree_edges:
        if not rm.commit_edge_bandwidth(req_id, u, v, 10.0):
            raise RuntimeError(f"failed to commit tree edge {(u, v)}")
    rm.allocate_bandwidth(2, 3, 32.0)

    current_tree = {
        "tree": {edge: 1.0 for edge in tree_edges},
        "placement": {
            (1, 0): {
                "node": 1,
                "vnf_type": 0,
                "cpu_used": 72.0,
                "mem_used": 10.0,
                "reused": False,
                "inst_id": first.inst_id,
            },
            (2, 1): {
                "node": 2,
                "vnf_type": 1,
                "cpu_used": 10.0,
                "mem_used": 8.0,
                "reused": False,
                "inst_id": second.inst_id,
            },
        },
        "node_stage": {0: 0, 1: 1, 4: 1, 2: 2, 3: 2, 5: 2},
        "tree_usage": {edge: 1 for edge in tree_edges},
    }
    if not rm.snapshot_request_sft(
        req_id, current_tree=current_tree, snapshot_time=1.0
    ):
        raise AssertionError("ordered reconfiguration fixture was rejected")


def check_post_validation_rollback() -> dict:
    rm = build_resource_manager()
    deploy_active_sft(rm)
    mgr = ReconfigurationManager(
        rm, node_util_threshold=0.50, link_util_threshold=0.80
    )
    proposal = mgr.plan_greedy_reroute(2001)
    if proposal.action_type != "reroute_edge" or not proposal.target:
        raise AssertionError("rollback fixture has no valid reroute proposal")

    record = rm.request_table[2001]
    rm.current_request = {"id": 2001}
    rm.current_tree = {
        "tree": copy.deepcopy(record.tree_edges),
        "tree_usage": copy.deepcopy(record.tree_usage),
        "node_stage": copy.deepcopy(record.node_stage),
        "connected_dests": set(record.connected_dests),
    }
    rm.nodes_on_tree = {
        node for edge in record.tree_edges for node in edge
    } | {record.source}

    def snapshot() -> dict:
        rec = rm.request_table[2001]
        return {
            "bw_avail": copy.deepcopy(rm.pool.bw_avail),
            "bw_reserved": copy.deepcopy(rm.pool.bw_reserved),
            "edge_allocations": [
                (item.req_id, item.u, item.v, item.bw)
                for item in rec.edge_allocations
            ],
            "tree_edges": copy.deepcopy(rec.tree_edges),
            "tree_usage": copy.deepcopy(rec.tree_usage),
            "node_stage": copy.deepcopy(rec.node_stage),
            "snapshot_time": rec.snapshot_time,
            "last_reconfig_time": rec.last_reconfig_time,
            "migration_count": rec.migration_count,
            "reconfig_count": rec.reconfig_count,
            "state": rec.state,
            "legacy_tree": copy.deepcopy(rm.current_tree),
            "legacy_nodes": copy.deepcopy(rm.nodes_on_tree),
            "legacy_vnfs": copy.deepcopy(rm.shared_vnf_instances),
            "dbg_alloc": getattr(rm.pool, "_dbg_alloc_bw", None),
            "dbg_release": getattr(rm.pool, "_dbg_rel_bw", None),
        }

    before = snapshot()
    original_validator = rm.validate_request_sft_snapshot

    def reject_after_mutation(req_id):
        report = dict(original_validator(req_id))
        if not report.get("ok", False):
            raise AssertionError(
                f"reroute fixture became invalid before injected failure: {report}"
            )
        report["ok"] = False
        report["reason"] = "injected_post_reroute_failure"
        return report

    rm.validate_request_sft_snapshot = reject_after_mutation
    applied, status, reason = mgr._apply_edge_reroute(
        2001, dict(proposal.target)
    )
    after = snapshot()
    if applied:
        raise AssertionError("reroute unexpectedly survived post-validation failure")
    if status != "ROLLED_BACK" or "injected_post_reroute_failure" not in reason:
        raise AssertionError(
            f"reroute rollback returned unexpected status: {status}: {reason}"
        )
    if after != before:
        changed = sorted(
            key for key in before if before.get(key) != after.get(key)
        )
        raise AssertionError(f"reroute rollback changed state fields: {changed}")
    return {
        "ok": True,
        "applied": applied,
        "status_code": status,
        "full_state_restored": True,
    }


def check_string_request_key() -> dict:
    rm = build_resource_manager()
    deploy_active_sft(rm)
    mgr = ReconfigurationManager(
        rm, node_util_threshold=0.50, link_util_threshold=0.80
    )
    proposal = mgr.plan_greedy_reroute(2001)
    if proposal.action_type != "reroute_edge" or not proposal.target:
        raise AssertionError("string-key fixture has no valid reroute proposal")

    rm.request_table["2001"] = rm.request_table.pop(2001)
    result = mgr.apply_reroute_proposal(proposal)
    if not result.success or result.status_code != "APPLIED":
        raise AssertionError(
            "string-key reroute failed: "
            f"{result.status_code}: {result.reason}"
        )
    report = rm.validate_request_sft_snapshot("2001")
    if not report.get("ok", False):
        raise AssertionError(f"string-key reroute snapshot is invalid: {report}")
    return {
        "ok": True,
        "status_code": result.status_code,
        "snapshot_ok": True,
    }


def run_smoke_check(apply_actions: bool = True) -> dict:
    rm = build_resource_manager()
    deploy_active_sft(rm)
    mgr = ReconfigurationManager(rm, node_util_threshold=0.50, link_util_threshold=0.80)

    before = mgr.metrics_snapshot()
    risks = [risk.__dict__ for risk in mgr.select_topk_risky_sfts(k=3)]
    plans = {
        "no_reconfig": mgr.run_baseline("no-reconfig").to_dict(),
        "full_redeploy": mgr.run_baseline("full-redeploy").to_dict(),
        "greedy_migrate_plan": mgr.run_baseline("greedy-migrate").to_dict(),
        "greedy_reroute_plan": mgr.run_baseline("greedy-reroute").to_dict(),
    }

    applied = {}
    if apply_actions:
        malformed = mgr.apply_reroute_proposal({
            "req_id": 2001,
            "action_type": "reroute_edge",
            "target": {"new_path": [1]},
        })
        applied["malformed_reroute"] = malformed.to_dict()
        applied["greedy_migrate"] = mgr.run_baseline("greedy-migrate", apply=True).to_dict()
        applied["stale_reroute"] = mgr.apply_reroute_proposal(
            plans["greedy_reroute_plan"]
        ).to_dict()
        order_guard_fingerprint = mgr.request_state_fingerprint(2001)
        order_unsafe_success, order_unsafe_status, _ = mgr._apply_edge_reroute(
            2001,
            {"old_edge": (2, 3), "new_path": [1, 3], "bw": 10.0},
        )
        applied["order_unsafe_reroute_rejected"] = not order_unsafe_success
        applied["order_unsafe_reroute_status"] = order_unsafe_status
        applied["order_unsafe_state_unchanged"] = (
            mgr.request_state_fingerprint(2001) == order_guard_fingerprint
        )
        structural_unsafe_success, structural_unsafe_status, _ = mgr._apply_edge_reroute(
            2001,
            {"old_edge": (2, 3), "new_path": [2, 4, 1, 3], "bw": 10.0},
        )
        applied["structural_unsafe_reroute_rejected"] = not structural_unsafe_success
        applied["structural_unsafe_reroute_status"] = structural_unsafe_status
        applied["greedy_reroute"] = mgr.run_baseline("greedy-reroute", apply=True).to_dict()

    after = mgr.metrics_snapshot()
    rollback_check = (
        check_post_validation_rollback()
        if apply_actions
        else {"ok": True, "skipped": True}
    )
    string_key_check = (
        check_string_request_key()
        if apply_actions
        else {"ok": True, "skipped": True}
    )
    record = rm.request_table[2001]
    report = rm.validate_request_sft_snapshot(2001)
    safe_tree = mgr._is_directed_tree(record.tree_edges, record.source)
    ledger_edges = [(ea.u, ea.v) for ea in record.edge_allocations]
    ok = (
        len(risks) >= 1
        and before["node_hotspots"] >= 1
        and before["link_hotspots"] >= 1
        and report["ok"]
        and (not apply_actions or applied.get("malformed_reroute", {}).get("status_code") == "INVALID_PROPOSAL")
        and (not apply_actions or record.migration_count >= 1)
        and (not apply_actions or record.reconfig_count >= 1)
        and (not apply_actions or applied.get("order_unsafe_reroute_rejected") is True)
        and (not apply_actions or applied.get("order_unsafe_reroute_status") == "DESTINATION_ORDER")
        and (not apply_actions or applied.get("order_unsafe_state_unchanged") is True)
        and (not apply_actions or applied.get("structural_unsafe_reroute_rejected") is True)
        and (not apply_actions or applied.get("stale_reroute", {}).get("status_code") == "STALE_PLAN")
        and (not apply_actions or safe_tree)
        and (not apply_actions or (2, 3) not in record.tree_usage)
        and (not apply_actions or (2, 3) not in ledger_edges)
        and (not apply_actions or len(ledger_edges) == len(record.tree_edges))
        and rollback_check["ok"]
        and string_key_check["ok"]
    )
    return {
        "ok": ok,
        "before": before,
        "risks": risks,
        "plans": plans,
        "applied": applied,
        "after": after,
        "snapshot_report": report,
        "safe_tree": safe_tree,
        "ledger_edges": ledger_edges,
        "post_validation_rollback": rollback_check,
        "string_request_key": string_key_check,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true", help="do not apply migration/reroute actions")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    result = run_smoke_check(apply_actions=not args.plan_only)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Reconfiguration manager smoke check", "passed" if result["ok"] else "failed")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
