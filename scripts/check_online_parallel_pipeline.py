#!/usr/bin/env python3
"""Check central-ledger parallel candidate generation and exact retries."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import AtomicResourceLedger, ResourceFootprint
from core.marl.deployment_topk import CompletePlanCandidateGenerator
from core.marl.online_parallel_pipeline import (
    CandidateGeneration,
    CentralSharedLedgerPipeline,
    complete_plan_candidate_factory,
)


def main() -> int:
    ledger = AtomicResourceLedger(
        cpu_capacity={1: 10.0, 2: 10.0},
        memory_capacity={1: 10.0, 2: 10.0},
        bandwidth_capacity={(1, 2): 10.0, (2, 1): 10.0},
    )
    worker_versions: list[int] = []
    worker_threads: set[str] = set()

    def factory(request, snapshot):
        worker_versions.append(int(snapshot.version))
        worker_threads.add(threading.current_thread().name)
        request_id = int(request["id"])
        first = ResourceFootprint(
            {1: 6.0}, {1: 6.0}, {(1, 2): 6.0}
        )
        second = ResourceFootprint(
            {2: 4.0}, {2: 4.0}, {(2, 1): 4.0}
        )
        # Requests with an odd id prefer the first candidate; the joint
        # decoder must still choose a feasible combination for the batch.
        return CandidateGeneration(
            request_id=request_id,
            footprints=(first, second, None),
            payloads=(
                {"request_id": request_id, "candidate": 0},
                {"request_id": request_id, "candidate": 1},
                None,
            ),
            rankings=(0, 1, 2) if request_id % 2 else (1, 0, 2),
        )

    pipeline = CentralSharedLedgerPipeline(
        ledger,
        factory,
        worker_count=2,
        top_r=2,
        decoder_time_budget_ms=5.0,
    )
    result = pipeline.submit_batch(
        [{"id": 1, "leave_time": 10.0}, {"id": 2, "leave_time": 10.0}],
        timestamp=0.0,
    )
    assert result.committed and result.accepted == 2
    assert len(worker_threads) >= 1
    assert set(worker_versions) == {0}
    assert len(ledger.allocations) == 2
    released = pipeline.release_expired(11.0)
    assert released == 2 and not ledger.allocations

    # A factory that tries to mutate the ledger is intentionally outside the
    # pipeline contract; this check documents that workers only receive a
    # snapshot and cannot observe a post-commit version during one batch.
    second = pipeline.submit_batch(
        [{"id": 3, "leave_time": 20.0}, {"id": 4, "leave_time": 20.0}],
        timestamp=12.0,
    )
    assert second.accepted == 2
    pipeline.close()

    # Force one external commit between snapshot and exact commit.  The first
    # attempt must be rejected as stale and regenerated from version 1; no
    # partial batch reservation is allowed.
    conflict_ledger = AtomicResourceLedger(
        cpu_capacity={1: 10.0, 2: 10.0},
        memory_capacity={1: 10.0, 2: 10.0},
        bandwidth_capacity={(1, 2): 10.0, (2, 1): 10.0},
    )
    conflict_versions: list[int] = []
    conflict_once = {"done": False}

    def conflict_factory(request, snapshot):
        conflict_versions.append(int(snapshot.version))
        return CandidateGeneration(
            request_id=int(request["id"]),
            footprints=(ResourceFootprint({1: 2.0}, {1: 2.0}, {(1, 2): 2.0}), None),
            payloads=({"request_id": int(request["id"])}, None),
            rankings=(0, 1),
        )

    def conflict_ranker(requests, generations, snapshot, masks):
        if not conflict_once["done"]:
            conflict_once["done"] = True
            external = ResourceFootprint({2: 1.0}, {2: 1.0}, {(2, 1): 1.0})
            external_commit = conflict_ledger.commit_ranked(
                [999], [[external, None]], [[0, 1]], expected_version=snapshot.version
            )
            assert external_commit["accepted"] == 1
        return [list(generation.rankings) for generation in generations]

    conflict_pipeline = CentralSharedLedgerPipeline(
        conflict_ledger,
        conflict_factory,
        ranker=conflict_ranker,
        worker_count=2,
        max_version_retries=2,
    )
    conflict_result = conflict_pipeline.submit_batch(
        [{"id": 10, "leave_time": 30.0}, {"id": 11, "leave_time": 30.0}],
        timestamp=0.0,
    )
    assert conflict_result.committed and conflict_result.retry_count == 1
    assert conflict_versions[:2] == [0, 0]
    assert conflict_versions[2:] == [1, 1]
    assert set(conflict_ledger.allocations) == {999, 10, 11}
    conflict_pipeline.close()

    # Exercise the real complete-plan generator adapter against the current
    # US topology/request schema.  This is still a pure candidate call: the
    # adapter receives a snapshot and does not reserve the ledger.
    profile_path = ROOT / "sdn" / "topologies" / "us_backbone_28_bw90.json"
    request_path = (
        ROOT / "data" / "sdn_runtime_requests"
        / "seed_7071_rate24_duration100_lifetime50node" / "requests.jsonl"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8").splitlines()[0])
    capacities = {}
    for edge in profile["edges"]:
        u, v = int(edge["u"]), int(edge["v"])
        cap = float(edge.get("bandwidth_mbps", profile.get("default_bandwidth_mbps", 90.0)))
        capacities[(u, v)] = cap
        capacities[(v, u)] = cap
    real_ledger = AtomicResourceLedger(
        {int(node): 55.0 for node in profile["dc_nodes_1based"]},
        {int(node): 45.0 for node in profile["dc_nodes_1based"]},
        capacities,
    )
    real_generator = CompletePlanCandidateGenerator(profile, max_candidates=4)
    real_factory = complete_plan_candidate_factory(real_generator)
    real_generation = real_factory(request, real_ledger.snapshot())
    assert real_generation.request_id == int(request["id"])
    assert len(real_generation.footprints) == len(real_generation.payloads)
    assert real_generation.footprints[-1] is None
    assert real_generation.payloads[-1] is None
    assert real_ledger.snapshot().version == 0

    metadata = pipeline.metadata()
    print(json.dumps({"ok": True, "result": result.to_dict(), "metadata": metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
