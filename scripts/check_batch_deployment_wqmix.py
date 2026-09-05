#!/usr/bin/env python3
"""Deterministic smoke test for request-batch Weighted QMIX deployment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import (  # noqa: E402
    AtomicResourceLedger,
    ResourceFootprint,
    WeightedQMIXLearner,
    candidate_action_mask,
    candidate_conflict_features,
    deployment_candidate_features,
    ranked_actions,
)


def training_batch(
    transitions: int = 8,
    max_agents: int = 4,
    candidates: int = 3,
    request_dim: int = 6,
    candidate_dim: int = 12,
    state_dim: int = 10,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7301)
    request_obs = torch.rand(transitions, max_agents, request_dim, generator=generator)
    candidate_features = torch.rand(
        transitions, max_agents, candidates, candidate_dim, generator=generator
    )
    next_request_obs = torch.rand(transitions, max_agents, request_dim, generator=generator)
    next_candidate_features = torch.rand(
        transitions, max_agents, candidates, candidate_dim, generator=generator
    )
    agent_mask = torch.zeros(transitions, max_agents, dtype=torch.bool)
    for transition in range(transitions):
        agent_mask[transition, : 1 + transition % max_agents] = True
    action_mask = agent_mask.unsqueeze(-1).expand(-1, -1, candidates).clone()
    next_agent_mask = agent_mask.clone()
    next_action_mask = action_mask.clone()
    actions = torch.zeros(transitions, max_agents, dtype=torch.long)
    for transition in range(transitions):
        actions[transition, : 1 + transition % max_agents] = transition % (candidates - 1)
    return {
        "request_observations": request_obs,
        "candidate_features": candidate_features,
        "action_mask": action_mask,
        "agent_mask": agent_mask,
        "states": torch.rand(transitions, state_dim, generator=generator),
        "actions": actions,
        "rewards": torch.linspace(-1.0, 1.0, transitions),
        "next_request_observations": next_request_obs,
        "next_candidate_features": next_candidate_features,
        "next_action_mask": next_action_mask,
        "next_agent_mask": next_agent_mask,
        "next_states": torch.rand(transitions, state_dim, generator=generator),
        "dones": torch.tensor([False] * (transitions - 1) + [True]),
    }


def main() -> int:
    torch.manual_seed(7301)
    torch.set_num_threads(1)
    batch = training_batch()
    learner = WeightedQMIXLearner(
        request_dim=6,
        candidate_dim=12,
        state_dim=10,
        max_agents=4,
        hidden_dim=32,
        mixer_embed_dim=32,
        alpha=0.1,
        target_update_interval=2,
    )
    metrics = {}
    for _ in range(6):
        metrics = learner.train_step(batch)
    assert metrics["loss"] >= 0.0
    assert 0.1 <= metrics["mean_weight"] <= 1.0

    actions, q_values = learner.select_actions(
        batch["request_observations"][:1],
        batch["candidate_features"][:1],
        batch["action_mask"][:1],
        batch["agent_mask"][:1],
    )
    assert actions.shape == (1, 4) and q_values.shape == (1, 4, 3)
    assert actions[0, 1:].eq(0).all(), "padded request agents must be no-op"

    ledger = AtomicResourceLedger(
        cpu_capacity={1: 10.0, 2: 10.0},
        memory_capacity={1: 10.0, 2: 10.0},
        bandwidth_capacity={(1, 2): 10.0, (2, 3): 10.0},
    )
    first_best = ResourceFootprint(
        cpu={1: 6.0}, memory={1: 5.0}, bandwidth={(1, 2): 6.0}
    )
    first_fallback = ResourceFootprint(
        cpu={2: 4.0}, memory={2: 4.0}, bandwidth={(2, 3): 4.0}
    )
    second_best = ResourceFootprint(
        cpu={1: 6.0}, memory={1: 6.0}, bandwidth={(1, 2): 6.0}
    )
    second_fallback = ResourceFootprint(
        cpu={2: 5.0}, memory={2: 5.0}, bandwidth={(2, 3): 4.0}
    )
    impossible = ResourceFootprint(
        cpu={1: 20.0}, memory={1: 1.0}, bandwidth={(1, 2): 1.0}
    )
    deployment_candidates = [
        [first_best, first_fallback, None],
        [second_best, second_fallback, None],
        [impossible, None],
    ]
    snapshot = ledger.snapshot()
    conflicts = candidate_conflict_features(deployment_candidates, snapshot)
    assert conflicts[0][0][3] > 0.0, "shared bottleneck conflict was not detected"
    hard_mask = candidate_action_mask(deployment_candidates, snapshot)
    assert hard_mask == [[True, True, True], [True, True, True], [False, True]]
    deployment_features = deployment_candidate_features(deployment_candidates, snapshot)
    assert len(deployment_features[0][0]) == 16
    assert deployment_features[2][1][-1] == 1.0

    commit = ledger.commit_ranked(
        request_ids=[101, 102, 103],
        candidates=deployment_candidates,
        rankings=[[0, 1, 2], [0, 1, 2], [0, 1]],
        expected_version=snapshot.version,
    )
    assert commit["accepted"] == 2
    assert commit["results"][0]["candidate_index"] == 0
    assert commit["results"][1]["candidate_index"] == 1
    assert commit["results"][1]["attempts"] == 2
    assert commit["results"][2]["reason"] == "policy_reject"
    committed_snapshot = ledger.snapshot()
    assert min(committed_snapshot.cpu_remaining.values()) >= 0.0
    assert min(committed_snapshot.memory_remaining.values()) >= 0.0
    assert min(committed_snapshot.bandwidth_remaining.values()) >= 0.0

    concurrent_ledger = AtomicResourceLedger(
        cpu_capacity={1: 10.0},
        memory_capacity={1: 10.0},
        bandwidth_capacity={(1, 2): 10.0},
    )
    concurrent_candidate = ResourceFootprint(
        cpu={1: 3.0}, memory={1: 3.0}, bandwidth={(1, 2): 3.0}
    )

    def concurrent_commit(request_id: int) -> bool:
        result = concurrent_ledger.commit_ranked(
            [request_id], [[concurrent_candidate]], [[0]], expected_version=0
        )
        return bool(result["results"][0]["accepted"])

    with ThreadPoolExecutor(max_workers=16) as executor:
        concurrent_accepts = sum(executor.map(concurrent_commit, range(1000, 1020)))
    concurrent_snapshot = concurrent_ledger.snapshot()
    assert concurrent_accepts == 3
    assert concurrent_snapshot.cpu_remaining[1] == 1.0
    assert concurrent_snapshot.memory_remaining[1] == 1.0
    assert concurrent_snapshot.bandwidth_remaining[(1, 2)] == 1.0

    real_plan_check = {"available": False}
    plan_path = ROOT / "outputs" / "hrl_sfc_seed7071_rate24" / "plans_first100.jsonl"
    request_path = (
        ROOT
        / "data"
        / "sdn_runtime_requests"
        / "seed_7071_rate24_duration100_lifetime50node"
        / "requests.jsonl"
    )
    if plan_path.exists() and request_path.exists():
        first_plan = json.loads(plan_path.read_text(encoding="utf-8").splitlines()[0])
        first_request = json.loads(request_path.read_text(encoding="utf-8").splitlines()[0])
        real_footprint = ResourceFootprint.from_sfc_plan(
            first_plan, float(first_request["bw_origin"])
        )
        real_plan_check = {
            "available": True,
            "request_id": int(first_plan["request_id"]),
            "cpu_units": sum(real_footprint.cpu.values()),
            "memory_units": sum(real_footprint.memory.values()),
            "bandwidth_mbps": sum(real_footprint.bandwidth.values()),
            "directed_edge_allocations": len(real_footprint.bandwidth),
        }
        assert real_plan_check["cpu_units"] == 10.0
        assert real_plan_check["memory_units"] == 21.0
        assert real_plan_check["bandwidth_mbps"] == 70.0

    sample_rankings = ranked_actions(q_values[0], batch["action_mask"][0])
    assert len(sample_rankings) == 4 and len(sample_rankings[0]) == 3

    started = time.perf_counter()
    iterations = 200
    for _ in range(iterations):
        learner.select_actions(
            batch["request_observations"][:1].expand(1, -1, -1),
            batch["candidate_features"][:1].expand(1, -1, -1, -1),
            batch["action_mask"][:1],
            batch["agent_mask"][:1],
        )
    inference_ms = (time.perf_counter() - started) * 1000.0 / iterations

    large_learner = WeightedQMIXLearner(
        request_dim=16,
        candidate_dim=16,
        state_dim=128,
        max_agents=32,
        hidden_dim=64,
        mixer_embed_dim=64,
    )
    large_requests = torch.rand(1, 32, 16)
    large_candidates = torch.rand(1, 32, 8, 16)
    large_action_mask = torch.ones(1, 32, 8, dtype=torch.bool)
    large_agent_mask = torch.ones(1, 32, dtype=torch.bool)
    large_learner.select_actions(
        large_requests, large_candidates, large_action_mask, large_agent_mask
    )
    started = time.perf_counter()
    large_iterations = 200
    for _ in range(large_iterations):
        large_learner.select_actions(
            large_requests, large_candidates, large_action_mask, large_agent_mask
        )
    large_inference_ms = (time.perf_counter() - started) * 1000.0 / large_iterations

    parameter_count = sum(parameter.numel() for parameter in learner.q_network.parameters())
    parameter_count += sum(parameter.numel() for parameter in learner.mixer.parameters())
    print(json.dumps({
        "ok": True,
        "algorithm": "optimistic_weighted_qmix",
        "agent_semantics": "one request per active agent",
        "max_agents": 4,
        "candidate_actions": 3,
        "parameter_count": parameter_count,
        "mean_inference_ms": inference_ms,
        "batch32_k8_inference_ms": large_inference_ms,
        "batch32_k8_raw_scoring_throughput_rps": 32_000.0 / large_inference_ms,
        "training": metrics,
        "conflict_ratio_first_best": conflicts[0][0],
        "atomic_commit": commit,
        "concurrent_commit": {
            "attempted": 20,
            "accepted": concurrent_accepts,
            "remaining_cpu": concurrent_snapshot.cpu_remaining[1],
            "oversubscribed": False,
        },
        "real_plan_footprint": real_plan_check,
        "remaining": {
            "cpu": committed_snapshot.cpu_remaining,
            "memory": committed_snapshot.memory_remaining,
            "bandwidth": {
                f"{edge[0]}->{edge[1]}": value
                for edge, value in committed_snapshot.bandwidth_remaining.items()
            },
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
