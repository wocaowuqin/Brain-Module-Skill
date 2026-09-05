#!/usr/bin/env python3
"""Regression checks for directed bandwidth and tree-edge reuse semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch_geometric.data  # noqa: F401
except ImportError:
    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric_data = types.ModuleType("torch_geometric.data")
    torch_geometric_data.Data = type("Data", (), {})
    torch_geometric.data = torch_geometric_data
    sys.modules["torch_geometric"] = torch_geometric
    sys.modules["torch_geometric.data"] = torch_geometric_data

from envs.modules.AllResourceManager import SharedResourcePool
from envs.modules.controller_shared_helper import ControllerSharedHelper
from envs.modules.low_level_controller import LowLevelController


class FakePool:
    def __init__(self, available: dict[tuple[int, int], float]):
        self.available = available
        self.bw_cap = {edge: 10.0 for edge in available}

    def get_available_bandwidth(self, u: int, v: int) -> float:
        return float(self.available.get((u, v), 0.0))


@dataclass
class FakeResourceManager:
    adjacency: dict[int, list[int]]
    pool: FakePool

    @property
    def n(self) -> int:
        return len(self.adjacency)

    def get_neighbors(self, node: int) -> list[int]:
        return list(self.adjacency.get(node, []))


class FakeEnv:
    def __init__(self):
        self.n = 4
        self.config = {"low_topk": 4}
        self.current_request = {"id": 1, "bw_origin": 5.0, "dest": [2]}
        self.current_tree = {
            "tree": {(1, 0): 1.0},
            "tree_usage": {(1, 0): 1},
            "connected_dests": set(),
        }
        adjacency = {0: [1, 3], 1: [0, 2], 2: [], 3: [2]}
        available = {
            (0, 1): 1.0,
            (0, 3): 10.0,
            (1, 0): 10.0,
            (1, 2): 10.0,
            (3, 2): 10.0,
        }
        self.resource_mgr = FakeResourceManager(adjacency, FakePool(available))


def check_pool_directionality() -> None:
    topology = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    pool = SharedResourcePool(
        topology,
        {
            "cpu": 100.0,
            "memory": 100.0,
            "bandwidth": 10.0,
            "bandwidth_model": "directed",
        },
    )
    assert pool.allocate_bandwidth(0, 1, 4.0)
    assert pool.get_available_bandwidth(0, 1) == 6.0
    assert pool.get_available_bandwidth(1, 0) == 10.0


def check_reverse_edge_is_not_free() -> None:
    env = FakeEnv()
    helper = ControllerSharedHelper(env)
    assert helper.is_tree_edge(1, 0)
    assert not helper.is_tree_edge(0, 1)
    assert helper.is_reverse_tree_edge(0, 1)

    controller = LowLevelController(env)
    path = controller.compute_bw_aware_path(0, 2)
    assert path == [0, 3, 2], path

    env.resource_mgr.pool.available[(0, 1)] = 10.0
    path = controller.compute_bw_aware_path(0, 2)
    assert path == [0, 1, 2], path


def check_target_selection_uses_requested_direction() -> None:
    env = FakeEnv()
    helper = ControllerSharedHelper(env)
    selected = helper.select_bw_feasible_target(
        target_node=0,
        start_node=1,
        bw_need=5.0,
        valid_indices=[0, 2],
    )
    assert selected == 0, selected


def main() -> int:
    check_pool_directionality()
    check_reverse_edge_is_not_free()
    check_target_selection_uses_requested_direction()
    print("directed edge accounting checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
