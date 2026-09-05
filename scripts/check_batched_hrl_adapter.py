#!/usr/bin/env python3
"""Smoke-check the request-batched HRL scoring bridge."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gnn.tree_transformer_encoder import TreeTransformerEncoder
from core.hrl.high_policy import HighLevelPolicy
from core.hrl.low_policy import GoalConditionedLowLevelPolicy
from core.marl.batch_deployment_wqmix import ResourceFootprint, ResourceSnapshot
from core.marl.online_parallel_pipeline import CandidateGeneration
from core.marl.batched_hrl_adapter import (
    BatchedHRLPolicyRanker,
    BatchedTopologyCollator,
)


def _request(request_id: int) -> dict:
    return {
        "id": request_id,
        "source": 0,
        "source_dpid": 1,
        "dest": [2, 3],
        "destination_dpids": [3, 4],
        "bw_origin": 4.0,
        "cpu_origin": [2.0, 3.0],
        "memory_origin": [4.0, 5.0],
        "vnf": [1, 2],
    }


def main() -> int:
    torch.manual_seed(7)
    profile = {
        "dc_nodes_1based": [1, 3],
        "default_bandwidth_mbps": 90.0,
        "default_delay_ms": 2.0,
        "edges": [
            {"u": 1, "v": 2, "bandwidth_mbps": 90.0, "delay_ms": 2.0},
            {"u": 2, "v": 3, "bandwidth_mbps": 90.0, "delay_ms": 2.0},
            {"u": 3, "v": 4, "bandwidth_mbps": 90.0, "delay_ms": 2.0},
        ],
    }
    encoder = TreeTransformerEncoder(
        node_dim=12, edge_dim=5, hidden_dim=16, num_heads=4, req_dim=3,
        coupling_mode="full", use_aux_head=False,
    )
    high = HighLevelPolicy({
        "use_cuda": False, "hidden_dim": 16, "goal_dim": 8,
        "gnn_output_dim": 16, "environment": {"nb_high_level_goals": 4},
        "dropout": 0.0,
    })
    low = GoalConditionedLowLevelPolicy({
        "use_cuda": False, "state_dim": 16, "goal_dim": 8, "hidden_dim": 16,
        "environment": {"nb_low_level_actions": 4}, "dropout": 0.0,
    })
    planner = SimpleNamespace(
        coordinator=SimpleNamespace(
            high_agent=SimpleNamespace(encoder=encoder, high_policy=high),
            low_agent=SimpleNamespace(low_policy=low),
        )
    )
    snapshot = ResourceSnapshot(
        version=3,
        cpu_remaining={1: 55.0, 2: 55.0, 3: 55.0, 4: 55.0},
        memory_remaining={1: 45.0, 2: 45.0, 3: 45.0, 4: 45.0},
        bandwidth_remaining={(u, v): 90.0 for u in range(1, 5) for v in range(1, 5) if abs(u - v) == 1},
    )
    collator = BatchedTopologyCollator(profile, encoder)
    state = collator.collate([_request(1), _request(2)], snapshot)
    footprint = ResourceFootprint(cpu={1: 2.0}, memory={1: 4.0}, bandwidth={(1, 2): 4.0})
    payload = {
        "chain_nodes": [1, 3],
        "segments": [{"path": [1, 2, 3]}],
        "multicast": {"paths": {"3": [3, 4]}},
    }
    generations = [
        CandidateGeneration(1, (footprint, None), (payload, None), (0, 1), (1.0, 0.0), (True, True)),
        CandidateGeneration(2, (footprint, None), (payload, None), (0, 1), (1.0, 0.0), (True, True)),
    ]
    ranker = BatchedHRLPolicyRanker(planner, profile)
    rankings = ranker([_request(1), _request(2)], generations, snapshot, [[True, True], [True, True]])
    checks = {
        "node_shape": tuple(state.node_embeddings.shape) == (2, 4, 16),
        "graph_shape": tuple(state.graph_embeddings.shape) == (2, 16),
        "ranking_rows": len(rankings) == 2 and all(len(row) == 2 for row in rankings),
        "reject_last": all(row[-1] == 1 for row in rankings),
        "batched_metadata": ranker.metadata()["mode"] == "batched_hrl_policy_scoring",
    }
    result = {"ok": all(checks.values()), "checks": checks, "rankings": rankings}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
