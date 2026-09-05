#!/usr/bin/env python3
"""Smoke-check the Brain -> Module -> Agent -> Skill deployment contract."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.orchestration import BrainManagedDeploymentPlanner


class FakeBatchPlanner:
    microbatch_seconds = 0.005
    max_agents = 32

    def __init__(self) -> None:
        self.batch_calls = 0
        self.releases: list[int] = []

    def plan_batch(self, requests):
        self.batch_calls += 1
        return [
            {
                "request_id": int(row["id"]),
                "accepted": int(row["id"]) % 2 == 1,
                "reason": "test decision",
            }
            for row in requests
        ]

    def release(self, request_id: int) -> bool:
        self.releases.append(int(request_id))
        return True

    def metadata(self):
        return {"mode": "fake_batch", "rejected_requests": 0}

    def close(self):
        return None


def main() -> int:
    base = FakeBatchPlanner()
    planner = BrainManagedDeploymentPlanner(base)
    requests = [
        {"id": 1, "arrival_time": 1.0, "delay_bound_ms": 50.0},
        {"id": 2, "arrival_time": 1.001, "delay_bound_ms": 50.0},
        {"id": 3, "arrival_time": 1.002, "delay_bound_ms": 50.0},
    ]
    plans = planner.plan_batch(requests)
    assert base.batch_calls == 1, "micro-batch degraded into per-request calls"
    assert [row["request_id"] for row in plans] == [1, 2, 3]
    assert [row["accepted"] for row in plans] == [True, False, True]
    assert all(
        row["orchestration"]["architecture"] == "brain_module_skill"
        for row in plans
    )
    assert all(row["orchestration"]["skill"] == "hrl_sft_mapping" for row in plans)
    assert planner.release(1)
    assert base.releases == [1]
    metadata = planner.metadata()
    assert metadata["architecture"] == "brain_module_skill"
    assert metadata["orchestration"]["request_count"] == 3
    assert metadata["orchestration"]["command_counts"] == {"deploy": 3}
    print("brain-module-skill deployment check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
