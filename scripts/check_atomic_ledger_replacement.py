#!/usr/bin/env python3
"""Verify exact active-allocation replacement and rollback semantics."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import AtomicResourceLedger, ResourceFootprint


def resource_state(ledger: AtomicResourceLedger) -> tuple[dict, dict, dict]:
    snapshot = ledger.snapshot()
    return (
        snapshot.cpu_remaining,
        snapshot.memory_remaining,
        snapshot.bandwidth_remaining,
    )


def main() -> int:
    ledger = AtomicResourceLedger(
        {1: 10.0, 2: 10.0},
        {1: 10.0, 2: 10.0},
        {(1, 2): 10.0, (2, 1): 10.0},
    )
    old = ResourceFootprint({1: 6.0}, {1: 5.0}, {(1, 2): 4.0})
    new = ResourceFootprint({2: 6.0}, {2: 5.0}, {(2, 1): 4.0})
    committed = ledger.commit_exact(
        [7], [[old]], [0], expected_version=ledger.snapshot().version
    )
    assert committed["committed"]
    replaced = ledger.replace(7, new, expected_version=ledger.snapshot().version)
    assert replaced["replaced"]
    snapshot = ledger.snapshot()
    assert snapshot.cpu_remaining == {1: 10.0, 2: 4.0}
    assert snapshot.memory_remaining == {1: 10.0, 2: 5.0}
    assert snapshot.bandwidth_remaining == {(1, 2): 10.0, (2, 1): 6.0}

    impossible = ResourceFootprint({1: 11.0}, {1: 1.0}, {(1, 2): 1.0})
    rejected = ledger.replace(7, impossible, expected_version=ledger.snapshot().version)
    assert not rejected["replaced"] and rejected["reason"] == "insufficient_cpu"
    rollback = ledger.snapshot()
    assert rollback == snapshot
    stale = ledger.replace(7, old, expected_version=snapshot.version - 1)
    assert not stale["replaced"] and stale["reason"] == "version_mismatch"
    assert ledger.release(7)
    released = ledger.snapshot()
    assert all(value == 10.0 for value in released.cpu_remaining.values())
    assert all(value == 10.0 for value in released.memory_remaining.values())
    assert all(value == 10.0 for value in released.bandwidth_remaining.values())
    integrity = ledger.integrity_report()
    assert integrity["fully_released"]

    transactional = AtomicResourceLedger(
        {1: 10.0, 2: 10.0},
        {1: 10.0, 2: 10.0},
        {(1, 2): 10.0, (2, 1): 10.0},
    )
    assert transactional.commit_exact(
        [7], [[old]], [0], expected_version=0
    )["committed"]
    live_state = resource_state(transactional)
    overlap = ResourceFootprint({2: 6.0}, {2: 5.0}, {(2, 1): 4.0})
    prepared = transactional.prepare_replacement(
        7, overlap, expected_version=transactional.snapshot().version
    )
    assert prepared["prepared"]
    prepared_state = resource_state(transactional)
    assert prepared_state == (
        {1: 4.0, 2: 4.0},
        {1: 5.0, 2: 5.0},
        {(1, 2): 6.0, (2, 1): 6.0},
    )
    duplicate = transactional.prepare_replacement(7, overlap)
    assert not duplicate["prepared"]
    assert duplicate["reason"] == "replacement_already_prepared"
    bypass = transactional.replace(7, new)
    assert not bypass["replaced"] and bypass["reason"] == "replacement_prepared"
    aborted = transactional.abort_prepared_replacement(prepared["token"])
    assert aborted["aborted"]
    assert resource_state(transactional) == live_state

    prepared = transactional.prepare_replacement(7, overlap)
    committed_prepared = transactional.commit_prepared_replacement(
        7, prepared["token"], new
    )
    assert committed_prepared["replaced"]
    assert resource_state(transactional) == (
        {1: 10.0, 2: 4.0},
        {1: 10.0, 2: 5.0},
        {(1, 2): 10.0, (2, 1): 6.0},
    )
    stale_abort = transactional.abort_prepared_replacement(prepared["token"])
    assert not stale_abort["aborted"]
    assert transactional.release(7)
    assert transactional.integrity_report()["fully_released"]

    failed_commit = AtomicResourceLedger(
        {1: 10.0, 2: 10.0},
        {1: 10.0, 2: 10.0},
        {(1, 2): 10.0, (2, 1): 10.0},
    )
    assert failed_commit.commit_exact(
        [7], [[old]], [0], expected_version=0
    )["committed"]
    small_overlap = ResourceFootprint({2: 1.0}, {2: 1.0}, {})
    prepared_failure = failed_commit.prepare_replacement(7, small_overlap)
    before_failed_commit = resource_state(failed_commit)
    impossible_new = ResourceFootprint({2: 11.0}, {2: 1.0}, {})
    failed = failed_commit.commit_prepared_replacement(
        7, prepared_failure["token"], impossible_new
    )
    assert not failed["replaced"] and failed["reason"] == "insufficient_cpu"
    assert resource_state(failed_commit) == before_failed_commit
    assert failed_commit.abort_prepared_replacement(
        prepared_failure["token"]
    )["aborted"]
    resource_state_after_old = (
        {1: 4.0, 2: 10.0},
        {1: 5.0, 2: 10.0},
        {(1, 2): 6.0, (2, 1): 10.0},
    )
    assert resource_state(failed_commit) == resource_state_after_old
    release_prepared = failed_commit.prepare_replacement(7, small_overlap)
    assert release_prepared["prepared"]
    assert failed_commit.release(7)
    assert failed_commit.integrity_report()["fully_released"]
    print(json.dumps({
        "valid": True,
        "commit": committed,
        "replace": replaced,
        "rollback_rejection": rejected,
        "stale_rejection": stale,
        "final_version": released.version,
        "integrity": integrity,
        "prepared_abort": aborted,
        "prepared_commit": committed_prepared,
        "failed_prepared_commit": failed,
        "release_cleans_preparation": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
