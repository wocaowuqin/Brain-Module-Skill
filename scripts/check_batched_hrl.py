#!/usr/bin/env python3
"""Smoke-check vectorized HRL heads and bounded worker backpressure."""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hrl.high_policy import HighLevelPolicy
from core.hrl.low_policy import GoalConditionedLowLevelPolicy
from core.hrl.batched_policy import BatchedHRLPolicy
from core.marl.deployment_worker_pool import StatelessHRLWorkerPool


def main() -> int:
    torch.manual_seed(7)
    high = HighLevelPolicy({
        "use_cuda": False,
        "hidden_dim": 16,
        "goal_dim": 4,
        "gnn_output_dim": 8,
        "environment": {"nb_high_level_goals": 5},
        "dropout": 0.0,
    })
    low = GoalConditionedLowLevelPolicy({
        "use_cuda": False,
        "state_dim": 8,
        "goal_dim": 4,
        "hidden_dim": 16,
        "environment": {"nb_low_level_actions": 5},
        "dropout": 0.0,
    })
    policy = BatchedHRLPolicy(high, low, device="cpu")
    graph = torch.randn(3, 8)
    high_candidates = torch.randn(3, 4, 8)
    high_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.bool)
    high_out = policy.forward_high(
        graph,
        candidate_node_embeddings=high_candidates,
        candidate_local_features=torch.randn(3, 4, 7),
        candidate_mask=high_mask,
    )
    state = torch.randn(3, 6, 8)
    goals = torch.randn(3, 4)
    indices = torch.tensor([[1, 2, 4], [0, 3, 5], [2, 3, 4]])
    current = torch.tensor([0, 1, 2])
    low_mask = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.bool)
    low_out = policy.forward_low(
        state,
        goals,
        indices,
        current,
        candidate_local_features=torch.randn(3, 3, 6),
        candidate_mask=low_mask,
    )

    def slow_identity(batch):
        time.sleep(0.05)
        return list(batch)

    pool = StatelessHRLWorkerPool(slow_identity, worker_count=1, max_in_flight=1)
    first = pool.submit([1])
    second = pool.submit([2])
    if first is None or second is not None:
        raise AssertionError("worker pool did not enforce max_in_flight")
    first.result(timeout=2.0)
    stats = pool.stats()
    pool.close()
    checks = {
        "high_batch_shape": tuple(high_out.candidate_scores.shape) == (3, 4),
        "high_actions_valid": all(bool(high_mask[i, int(high_out.actions[i])]) for i in range(3)),
        "low_batch_shape": tuple(low_out.candidate_scores.shape) == (3, 3),
        "low_actions_valid": all(bool(low_mask[i, int(low_out.actions[i])]) for i in range(3)),
        "worker_backpressure": stats.rejected_backpressure == 1,
        "worker_completed": stats.completed == 1 and stats.failed == 0,
    }
    result = {"ok": all(checks.values()), "checks": checks, "worker_stats": stats.__dict__}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

