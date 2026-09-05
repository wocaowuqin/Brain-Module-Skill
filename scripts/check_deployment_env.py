#!/usr/bin/env python3
"""Smoke-test the counterfactual deployment environment and lifecycle release."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.deployment_dataset import FeatureNormalizer, load_labeled_batches  # noqa: E402
from core.marl.deployment_env import BatchDeploymentEnv  # noqa: E402


def main() -> int:
    data = ROOT / "data" / "deployment_v4_tree_bw_rate24_sla80"
    train_folder = data / "train" / "seed_7101" / "mb5ms"
    records = load_labeled_batches([train_folder])
    normalizer = FeatureNormalizer.fit([record["batch"] for record in records])
    trace = data / "traces" / "seed_7101" / "requests.jsonl"
    profile = ROOT / "sdn" / "topologies" / "us_backbone_28_bw90.json"
    env = BatchDeploymentEnv(
        trace, profile, normalizer, microbatch_ms=5.0, max_requests=100
    )
    observation = env.reset()
    total_reward = 0.0
    total_accepted = 0
    steps = 0
    max_active = 0
    while env.active_agents:
        mask = observation["action_mask"][0]
        active = observation["agent_mask"][0]
        rankings = [
            torch.where(row)[0].tolist() for row in mask[active]
        ]
        decoded = env.decode_rankings(rankings, time_budget_ms=10.0)
        observation, reward, done, info = env.step(
            decoded.actions, expected_version=decoded.snapshot_version
        )
        total_reward += reward
        total_accepted += int(info["accepted"])
        max_active = max(max_active, len(decoded.actions))
        steps += 1
        if done:
            break
    remaining = env.ledger.snapshot()
    leaked = sum(1 for value in env.ledger.allocations.values() if value is not None)
    result = {
        "ok": leaked == 0,
        "steps": steps,
        "accepted": total_accepted,
        "total_reward": total_reward,
        "max_active_agents": max_active,
        "ledger_allocations_after_done": leaked,
        "remaining_min_cpu": min(remaining.cpu_remaining.values()),
        "remaining_min_memory": min(remaining.memory_remaining.values()),
        "observation_shapes": {
            key: list(value.shape) for key, value in observation.items()
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
