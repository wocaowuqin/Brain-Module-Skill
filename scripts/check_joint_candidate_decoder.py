#!/usr/bin/env python3
"""Correctness and bounded-latency checks for joint candidate decoding."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    ResourceFootprint,
    VNFInstanceRequirement,
)
from core.marl.joint_candidate_decoder import (  # noqa: E402
    decode_joint_candidates,
    joint_footprints_feasible,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile + 0.999999) - 1))
    return float(ordered[index])


def conflict_case() -> dict:
    ledger = AtomicResourceLedger(
        cpu_capacity={1: 10.0, 2: 10.0},
        memory_capacity={1: 10.0, 2: 10.0},
        bandwidth_capacity={(1, 2): 10.0, (2, 3): 10.0},
    )
    candidates = [
        [
            ResourceFootprint({1: 6.0}, {1: 6.0}, {(1, 2): 6.0}),
            ResourceFootprint({2: 4.0}, {2: 4.0}, {(2, 3): 4.0}),
            None,
        ],
        [
            ResourceFootprint({1: 6.0}, {1: 6.0}, {(1, 2): 6.0}),
            ResourceFootprint({2: 5.0}, {2: 5.0}, {(2, 3): 4.0}),
            None,
        ],
    ]
    snapshot = ledger.snapshot()
    decoded = decode_joint_candidates(
        candidates,
        snapshot,
        rankings=[[0, 1, 2], [0, 1, 2]],
        reject_actions=[2, 2],
        action_mask=[[True, True, True], [True, True, True]],
        scores=[[3.0, 2.0, 0.0], [3.0, 2.0, 0.0]],
        priorities=[(1.0, 1), (2.0, 2)],
        time_budget_ms=10.0,
    )
    assert decoded.accepted == 2
    assert decoded.actions == (0, 1)
    commit = ledger.commit_exact(
        [101, 102], candidates, decoded.actions,
        expected_version=decoded.snapshot_version,
    )
    assert commit["committed"] and commit["accepted"] == 2
    assert ledger.snapshot().bandwidth_remaining[(1, 2)] == 4.0

    stale_snapshot = ledger.snapshot()
    extra = [[ResourceFootprint({2: 1.0}, {2: 1.0}, {(2, 3): 1.0}), None]]
    first = ledger.commit_exact(
        [103], extra, [0], expected_version=stale_snapshot.version
    )
    assert first["committed"]
    allocation_count = len(ledger.allocations)
    stale = ledger.commit_exact(
        [104], extra, [0], expected_version=stale_snapshot.version
    )
    assert not stale["committed"] and stale["reason"] == "version_mismatch"
    assert len(ledger.allocations) == allocation_count

    infeasible_ledger = AtomicResourceLedger(
        {1: 10.0, 2: 10.0}, {1: 10.0, 2: 10.0},
        {(1, 2): 10.0, (2, 3): 10.0},
    )
    rejected = infeasible_ledger.commit_exact(
        [201, 202], candidates, [0, 0], expected_version=0
    )
    assert not rejected["committed"]
    assert rejected["reason"] == "joint_infeasible_cpu"
    assert not infeasible_ledger.allocations
    return {"decoded": decoded.to_dict(), "commit": commit, "stale": stale}


def shared_instance_case() -> dict:
    ledger = AtomicResourceLedger(
        {1: 6.0}, {1: 8.0}, {(1, 2): 20.0}
    )
    requirement = VNFInstanceRequirement(1, 7, 6.0, 8.0)
    candidates = [
        [ResourceFootprint({}, {}, {(1, 2): 4.0}, (requirement,)), None],
        [ResourceFootprint({}, {}, {(1, 2): 4.0}, (requirement,)), None],
    ]
    snapshot = ledger.snapshot()
    feasible, reason = joint_footprints_feasible(
        [candidates[0][0], candidates[1][0]], snapshot
    )
    assert feasible, reason
    decoded = decode_joint_candidates(
        candidates, snapshot, [[0, 1], [0, 1]],
        reject_actions=[1, 1], time_budget_ms=10.0,
    )
    commit = ledger.commit_exact(
        [301, 302], candidates, decoded.actions,
        expected_version=decoded.snapshot_version,
    )
    assert commit["committed"] and commit["accepted"] == 2
    assert ledger.snapshot().cpu_remaining[1] == 0.0
    assert ledger.vnf_instances[(1, 7)]["ref_count"] == 2.0
    return {"decoded": decoded.to_dict(), "commit": commit}


def repair_case() -> dict:
    snapshot = AtomicResourceLedger(
        {1: 10.0, 2: 10.0}, {1: 10.0, 2: 10.0},
        {(1, 2): 10.0, (2, 3): 10.0},
    ).snapshot()
    candidates = [
        [
            ResourceFootprint({1: 10.0}, {1: 10.0}, {(1, 2): 10.0}),
            ResourceFootprint({2: 6.0}, {2: 6.0}, {(2, 3): 6.0}),
            None,
        ],
        [ResourceFootprint({1: 5.0}, {1: 5.0}, {(1, 2): 5.0}), None],
    ]
    decoded = decode_joint_candidates(
        candidates, snapshot, [[0, 1, 2], [0, 1]],
        reject_actions=[2, 1], time_budget_ms=10.0,
    )
    assert decoded.actions == (1, 0)
    assert decoded.accepted == 2 and decoded.repair_improvements == 1
    return decoded.to_dict()


def worst_batch_profile(iterations: int = 200) -> dict:
    rng = random.Random(7071)
    node_count = 12
    edge_count = 24
    snapshot = AtomicResourceLedger(
        {node: 10.0 for node in range(node_count)},
        {node: 10.0 for node in range(node_count)},
        {(edge, edge + 1): 20.0 for edge in range(edge_count)},
    ).snapshot()
    candidates = []
    scores = []
    for agent in range(32):
        rows = []
        row_scores = []
        for candidate_index in range(4):
            node = (agent * 3 + candidate_index * 5) % node_count
            edge = (agent * 7 + candidate_index * 3) % edge_count
            rows.append(ResourceFootprint(
                {node: float(rng.randint(2, 6))},
                {node: float(rng.randint(2, 6))},
                {(edge, edge + 1): float(rng.randint(4, 12))},
            ))
            row_scores.append(float(4 - candidate_index) + rng.random() * 0.01)
        rows.append(None)
        row_scores.append(0.0)
        candidates.append(rows)
        scores.append(row_scores)
    rankings = [[0, 1, 2, 3, 4] for _ in range(32)]
    action_mask = [[True] * 5 for _ in range(32)]
    elapsed = []
    timeouts = 0
    accepted = []
    examined = []
    repair_budget_exhaustions = 0
    greedy_budget_exhaustions = 0
    for _ in range(iterations):
        decoded = decode_joint_candidates(
            candidates,
            snapshot,
            rankings,
            reject_actions=[4] * 32,
            action_mask=action_mask,
            scores=scores,
            priorities=[(float(index), index) for index in range(32)],
            top_r=4,
            time_budget_ms=2.0,
        )
        feasible, reason = joint_footprints_feasible(
            [candidates[i][decoded.actions[i]] for i in range(32)], snapshot
        )
        assert feasible, reason
        elapsed.append(decoded.elapsed_ms)
        timeouts += int(decoded.timed_out)
        accepted.append(decoded.accepted)
        examined.append(decoded.candidates_examined)
        repair_budget_exhaustions += int(decoded.repair_budget_exhausted)
        greedy_budget_exhaustions += int(decoded.greedy_budget_exhausted)
    p95 = percentile(elapsed, 0.95)
    # The deadline is cooperative, so one sparse feasibility check may finish
    # just after 2 ms.  A generous regression ceiling catches runaway search.
    assert p95 < 10.0, f"decoder P95 regressed to {p95:.3f} ms"
    return {
        "shape": "32 requests x 4 deploy candidates + reject",
        "iterations": iterations,
        "mean_ms": sum(elapsed) / len(elapsed),
        "p95_ms": p95,
        "max_ms": max(elapsed),
        "timeout_rate": timeouts / iterations,
        "repair_budget_exhaustion_rate": (
            repair_budget_exhaustions / iterations
        ),
        "greedy_budget_exhaustion_rate": (
            greedy_budget_exhaustions / iterations
        ),
        "mean_candidates_examined": sum(examined) / len(examined),
        "mean_accepted": sum(accepted) / len(accepted),
    }


def main() -> int:
    result = {
        "ok": True,
        "conflict_and_versioning": conflict_case(),
        "shared_vnf_instance": shared_instance_case(),
        "bounded_repair": repair_case(),
        "worst_batch_profile": worst_batch_profile(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
