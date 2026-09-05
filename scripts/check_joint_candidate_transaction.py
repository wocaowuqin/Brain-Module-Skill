#!/usr/bin/env python3
"""Regression checks for the shared training/online commit transaction."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    ResourceFootprint,
)
from core.marl.joint_candidate_transaction import (  # noqa: E402
    commit_decoded_joint_actions,
)


def new_ledger() -> AtomicResourceLedger:
    return AtomicResourceLedger(
        cpu_capacity={1: 10.0, 2: 10.0},
        memory_capacity={1: 10.0, 2: 10.0},
        bandwidth_capacity={(1, 2): 10.0, (2, 3): 10.0},
    )


def candidate_batch():
    return [
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


def main() -> int:
    candidates = candidate_batch()

    training_ledger = new_ledger()
    online_ledger = new_ledger()
    training_commit = commit_decoded_joint_actions(
        training_ledger, [101, 102], candidates, [0, 1], expected_version=0
    )
    online_commit = commit_decoded_joint_actions(
        online_ledger, [101, 102], candidates, [0, 1], expected_version=0
    )
    assert training_commit == online_commit
    assert training_ledger.snapshot() == online_ledger.snapshot()
    assert training_commit["committed"] and training_commit["accepted"] == 2

    invalid_ledger = new_ledger()
    try:
        commit_decoded_joint_actions(
            invalid_ledger, [201, 202], candidates, [9, 1], expected_version=0
        )
    except ValueError as exc:
        invalid_action = str(exc)
    else:
        raise AssertionError("invalid action was accepted")
    assert not invalid_ledger.allocations

    infeasible_ledger = new_ledger()
    try:
        commit_decoded_joint_actions(
            infeasible_ledger, [301, 302], candidates, [0, 0], expected_version=0
        )
    except RuntimeError as exc:
        infeasible = str(exc)
    else:
        raise AssertionError("jointly infeasible actions were accepted")
    assert not infeasible_ledger.allocations

    stale_ledger = new_ledger()
    first = commit_decoded_joint_actions(
        stale_ledger, [401], [candidates[0]], [1], expected_version=0
    )
    assert first["committed"]
    allocation_count = len(stale_ledger.allocations)
    stale = commit_decoded_joint_actions(
        stale_ledger, [402], [candidates[0]], [1], expected_version=0
    )
    assert not stale["committed"] and stale["reason"] == "version_mismatch"
    assert len(stale_ledger.allocations) == allocation_count

    print(
        json.dumps(
            {
                "ok": True,
                "training_online_identical": True,
                "accepted": training_commit["accepted"],
                "invalid_action": invalid_action,
                "joint_infeasible": infeasible,
                "stale_commit": stale,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
