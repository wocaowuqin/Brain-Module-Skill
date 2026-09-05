#!/usr/bin/env python3
"""Regression checks for destination failed-anchor retry semantics."""

from pathlib import Path
from types import SimpleNamespace
import sys

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.modules.HRL_Coordinator import HRL_Coordinator
from envs.modules.low_level_controller import LowLevelController


class FakePool:
    def __init__(self, graph):
        self.bw_cap = {(int(u), int(v)): 90.0 for u, v in graph.edges}

    def get_available_bandwidth(self, u, v):
        return self.bw_cap.get((int(u), int(v)), 0.0)


class FakeResourceManager:
    def __init__(self, graph):
        self.graph = graph
        self.pool = FakePool(graph)

    def get_neighbors(self, node):
        return list(self.graph.successors(int(node)))


class FakeShared:
    def __init__(self, env):
        self.env = env

    def get_connected_dests_view(self):
        return set(self.env.current_tree['connected_dests'])

    def get_positive_tree_edge_set(self):
        return {
            edge for edge, flow in self.env.current_tree['tree'].items()
            if float(flow) > 0.0
        }


def make_controller():
    graph = nx.DiGraph([
        (0, 1), (1, 2), (2, 5),
        (1, 3), (3, 4), (4, 5),
    ])
    env = SimpleNamespace(
        n=6,
        current_request={
            'id': 7071,
            'dest': [5],
            'vnf': [0],
            'bw_origin': 10.0,
        },
        current_tree={
            'tree': {(0, 1): 1.0, (1, 2): 1.0},
            'connected_dests': set(),
            'node_stage': {0: 0, 1: 1, 2: 1},
        },
        chain_nodes=[1],
        nodes_on_tree={0, 1, 2},
        resource_mgr=FakeResourceManager(graph),
        _failed_anchors_for_target={},
    )
    controller = LowLevelController.__new__(LowLevelController)
    controller.env = env
    controller.shared = FakeShared(env)
    return env, controller


def run_checks():
    env, controller = make_controller()
    initial_plan = controller.plan_destination_completion()
    assert initial_plan, initial_plan
    first = initial_plan[0]
    assert first['target'] == 5

    env._failed_anchors_for_target = {5: {first['anchor']}}
    valid, reason = controller.validate_destination_plan_step(first)
    assert not valid and reason == 'failed_anchor', (valid, reason)

    retry_plan = controller.plan_destination_completion()
    assert retry_plan, retry_plan
    retry = retry_plan[0]
    assert retry['anchor'] != first['anchor'], (first, retry)
    assert retry['anchor'] not in env._failed_anchors_for_target[5]

    assert 'cycle_blocked' in HRL_Coordinator._SOFT_DEST_FAILURE_REASONS
    assert 'no_progress' in HRL_Coordinator._SOFT_DEST_FAILURE_REASONS
    return {
        'ok': True,
        'initial': first,
        'retry': retry,
        'validation_reason': reason,
    }


if __name__ == '__main__':
    print(run_checks())
