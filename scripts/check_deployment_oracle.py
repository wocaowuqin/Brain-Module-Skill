#!/usr/bin/env python3
"""Deterministic conflict and VNF-sharing checks for the deployment MILP."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.deployment_oracle import DeploymentBatchOracle, validate_solution  # noqa: E402


def footprint(cpu, memory, bandwidth, node, vnf_type):
    return {
        "cpu": {str(node): cpu},
        "memory": {str(node): memory},
        "bandwidth": [
            {"u": edge[0], "v": edge[1], "mbps": value}
            for edge, value in bandwidth.items()
        ],
        "vnf_instances": [
            {"node": node, "vnf_type": vnf_type, "cpu": cpu, "memory": memory}
        ],
    }


def candidate(source, objective, raw_footprint):
    return {
        "source": source,
        "objective": objective,
        "action_valid": True,
        "plan": None if source == "reject" else {"accepted": True},
        "resource_footprint": raw_footprint,
    }


def main() -> int:
    shared = footprint(6.0, 6.0, {(1, 2): 6.0}, 1, 3)
    first_alternate = footprint(4.0, 4.0, {(2, 3): 4.0}, 2, 4)
    second_alternate = footprint(4.0, 4.0, {(3, 4): 4.0}, 3, 5)
    reject = candidate("reject", 1_000_000.0, None)
    batch = {
        "resource_snapshot": {
            "version": 0,
            "cpu_remaining": {"1": 10.0, "2": 10.0, "3": 10.0},
            "memory_remaining": {"1": 10.0, "2": 10.0, "3": 10.0},
            "bandwidth_remaining": [
                {"u": 1, "v": 2, "mbps": 12.0},
                {"u": 2, "v": 3, "mbps": 10.0},
                {"u": 3, "v": 4, "mbps": 10.0},
            ],
            "vnf_instances": [],
        },
        "agents": [
            {
                "request_features": [0.0] * 12,
                "reject_action": 2,
                "candidates": [
                    candidate("shared_best", 1.0, shared),
                    candidate("first_alternate", 5.0, first_alternate),
                    reject,
                ],
            },
            {
                "request_features": [0.0] * 12,
                "reject_action": 2,
                "candidates": [
                    candidate("shared_best", 1.0, shared),
                    candidate("second_alternate", 5.0, second_alternate),
                    reject,
                ],
            },
        ],
    }
    oracle = DeploymentBatchOracle()
    shared_solution = oracle.solve(batch)
    assert shared_solution.optimal
    assert shared_solution.accepted == 2
    assert shared_solution.actions == [0, 0]
    assert shared_solution.resource_usage["new_vnf_instances"] == 1
    assert not validate_solution(batch, shared_solution)

    # Make the first choices use different VNF instances while retaining the
    # same bottleneck edge.  The oracle must move one request to an alternate.
    batch["agents"][1]["candidates"][0] = candidate(
        "conflicting_best",
        1.0,
        footprint(6.0, 6.0, {(1, 2): 6.0}, 1, 6),
    )
    batch["resource_snapshot"]["bandwidth_remaining"][0]["mbps"] = 10.0
    conflict_solution = oracle.solve(batch)
    assert conflict_solution.optimal
    assert conflict_solution.accepted == 2
    assert conflict_solution.actions in ([0, 1], [1, 0])
    assert not validate_solution(batch, conflict_solution)

    # The SLA-aware tie-breaker must prefer a slightly longer route when the
    # shortest route would drive a Q0 request close to link saturation.
    q0_features = [0.0] * 12
    q0_features[10] = 3.0
    congested = candidate(
        "congested_short",
        1.0,
        footprint(1.0, 1.0, {(1, 2): 8.0}, 1, 7),
    )
    congested["plan"] = {
        "segments": [{"path": [1, 2]}],
        "multicast": {"paths": {}},
    }
    congested["metrics"] = {"estimated_delay_ms": 10.0, "delay_bound_ms": 100.0}
    residual = candidate(
        "residual_longer",
        8.0,
        footprint(1.0, 1.0, {(2, 3): 1.0}, 2, 7),
    )
    residual["plan"] = {
        "segments": [{"path": [2, 3]}],
        "multicast": {"paths": {}},
    }
    residual["metrics"] = {"estimated_delay_ms": 15.0, "delay_bound_ms": 100.0}
    sla_batch = {
        "resource_snapshot": {
            "version": 0,
            "cpu_remaining": {"1": 10.0, "2": 10.0},
            "memory_remaining": {"1": 10.0, "2": 10.0},
            "bandwidth_remaining": [
                {"u": 1, "v": 2, "mbps": 9.0},
                {"u": 2, "v": 3, "mbps": 10.0},
            ],
            "vnf_instances": [],
        },
        "agents": [{
            "request_features": q0_features,
            "reject_action": 2,
            "candidates": [congested, residual, reject],
        }],
    }
    sla_solution = DeploymentBatchOracle(
        sla_risk_scale=100.0,
        bandwidth_capacity_mbps=10.0,
    ).solve(sla_batch)
    assert sla_solution.optimal
    assert sla_solution.actions == [1]
    assert not validate_solution(sla_batch, sla_solution)
    print(json.dumps({
        "ok": True,
        "shared_instance_actions": shared_solution.actions,
        "conflict_resolved_actions": conflict_solution.actions,
        "accepted": conflict_solution.accepted,
        "sla_aware_actions": sla_solution.actions,
        "solver_status": conflict_solution.status,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
