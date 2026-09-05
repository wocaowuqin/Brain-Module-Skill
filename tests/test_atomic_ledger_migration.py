from core.marl.batch_deployment_wqmix import (
    AtomicResourceLedger,
    ResourceFootprint,
    VNFInstanceRequirement,
)


def _ledger():
    return AtomicResourceLedger(
        cpu_capacity={1: 10.0, 2: 10.0, 3: 10.0},
        memory_capacity={1: 10.0, 2: 10.0, 3: 10.0},
        bandwidth_capacity={(1, 2): 10.0, (2, 1): 10.0, (2, 3): 10.0},
    )


def _fp(node, edge=(1, 2), vnf_type=7):
    return ResourceFootprint(
        cpu={node: 2.0},
        memory={node: 1.0},
        bandwidth={edge: 3.0},
        vnf_instances=(VNFInstanceRequirement(node, vnf_type, 2.0, 1.0),),
    )


def _allocate(ledger, request_id, footprint):
    result = ledger.commit_ranked(
        [request_id], [[footprint]], [[0]], expected_version=0
    )
    assert result["accepted"] == 1


def test_apply_migration_updates_source_target_and_directed_bandwidth():
    ledger = _ledger()
    old = _fp(1, (1, 2))
    new = _fp(2, (2, 3))
    _allocate(ledger, 10, old)
    before = ledger.snapshot()

    result = ledger.apply_migration(10, new, expected_version=before.version)

    assert result["replaced"] is True
    after = ledger.snapshot()
    assert after.version == before.version + 1
    assert after.cpu_remaining[1] == 10.0
    assert after.cpu_remaining[2] == 8.0
    assert after.bandwidth_remaining[(1, 2)] == 10.0
    assert after.bandwidth_remaining[(2, 3)] == 7.0


def test_shared_vnf_instance_is_not_freed_until_last_reference():
    ledger = _ledger()
    first = _fp(1, (1, 2), 9)
    second = _fp(1, (2, 1), 9)
    _allocate(ledger, 1, first)
    result = ledger.commit_ranked([2], [[second]], [[0]])
    assert result["accepted"] == 1
    assert ledger.snapshot().cpu_remaining[1] == 8.0

    moved = _fp(2, (2, 3), 9)
    result = ledger.apply_migration(1, moved)
    assert result["replaced"] is True
    snap = ledger.snapshot()
    # Request 2 still references node 1's shared instance.
    assert snap.cpu_remaining[1] == 8.0
    assert snap.cpu_remaining[2] == 8.0
    assert snap.vnf_instances[(1, 9)][2] == 1


def test_failed_migration_has_zero_writes_and_no_leak():
    ledger = _ledger()
    old = _fp(1)
    _allocate(ledger, 3, old)
    before = ledger.snapshot()
    # Exceeds node 2 capacity.
    impossible = ResourceFootprint(
        cpu={2: 100.0}, memory={2: 1.0}, bandwidth={(2, 3): 1.0}
    )

    result = ledger.apply_migration(3, impossible, expected_version=before.version)

    assert result["replaced"] is False
    assert result["reason"] == "insufficient_cpu"
    assert ledger.snapshot() == before
    assert ledger.integrity_report()["prepared_replacements"] == 0


def test_stale_version_nonexistent_and_inflight_are_rejected_without_writes():
    ledger = _ledger()
    old = _fp(1)
    _allocate(ledger, 4, old)
    before = ledger.snapshot()
    assert ledger.apply_migration(4, _fp(2), expected_version=before.version - 1)["reason"] == "version_mismatch"
    assert ledger.snapshot() == before
    assert ledger.apply_migration(999, _fp(2))["reason"] == "request_not_allocated"

    prep = ledger.prepare_replacement(4, _fp(2))
    assert prep["prepared"] is True
    held = ledger.snapshot()
    result = ledger.apply_migration(4, _fp(3))
    assert result["replaced"] is False
    assert result["reason"] == "replacement_prepared"
    assert ledger.snapshot() == held


def test_release_after_migration_returns_all_resources():
    ledger = _ledger()
    _allocate(ledger, 5, _fp(1))
    assert ledger.apply_migration(5, _fp(2))["replaced"] is True
    assert ledger.release(5) is True
    assert ledger.integrity_report()["fully_released"] is True
