#!/usr/bin/env python3
"""Regression checks for directed HRL stage-order action masks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.modules.high_level_controller import HighLevelController
from envs.modules.low_level_controller import LowLevelController


class FakePool:
    def __init__(self, edges, capacity=100.0):
        self.available = {tuple(edge): float(capacity) for edge in edges}
        self.bw_cap = dict(self.available)
        self.available_cpu = {}
        self.available_memory = {}

    def get_available_bandwidth(self, u, v):
        return self.available.get((int(u), int(v)), 0.0)

    def get_available_cpu(self, node):
        return self.available_cpu.get(int(node), 100.0)

    def get_available_memory(self, node):
        return self.available_memory.get(int(node), 100.0)


class FakeResourceManager:
    def __init__(self, graph):
        directed_edges = list(graph.edges)
        self.graph = graph
        self.pool = FakePool(directed_edges)
        self.request_manager = SimpleNamespace()

    def get_neighbors(self, node):
        return list(self.graph.successors(int(node)))

    def probe_vnf_deploy(self, node, vnf_type, req_cpu, req_mem):
        return {"ok": True, "reuse": False, "reason": "ok"}


class HopController:
    def __init__(self, graph):
        self.graph = graph

    def _get_hop_distance(self, source, target):
        try:
            return nx.shortest_path_length(self.graph, int(source), int(target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 9999


def make_graph():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 4),
            (4, 2),
            (1, 3),
            (3, 1),
            (3, 5),
            (5, 3),
            (5, 2),
            (2, 5),
        ]
    )
    return graph


def make_env(graph):
    topology = np.zeros((6, 6), dtype=float)
    for u, v in graph.edges:
        topology[u, v] = 1.0
    resource_mgr = FakeResourceManager(graph)
    return SimpleNamespace(
        n=6,
        topology=topology,
        config={"low_topk": 5},
        resource_mgr=resource_mgr,
        request_manager=resource_mgr.request_manager,
        current_request={
            "id": 1,
            "source": 0,
            "dest": [2],
            "vnf": [0],
            "cpu_origin": [1.0],
            "memory_origin": [1.0],
            "bw_origin": 10.0,
        },
        current_tree={
            "tree": {(0, 1): 1.0},
            "placement": {},
            "connected_dests": set(),
        },
        current_phase="vnf_deployment",
        next_vnf_idx=0,
        current_node_location=1,
        current_deployment_target=5,
        current_target_node=None,
        chain_nodes=[1],
        nodes_on_tree={0, 1},
        dc_nodes={4, 5},
        hard_tabu_list=set(),
        current_path_trace=[1],
        current_subgoal_full_path=[1],
        recent_edge_trace=[],
        _candidate_ablation="none",
    )


def run_checks():
    graph = make_graph()
    env = make_env(graph)
    env.low_level_controller = HopController(graph)
    high = HighLevelController(env)
    assert not high._has_ordered_stage_path(1, 0, 10.0, blocked_nodes={2})
    high_mask = high.get_high_level_action_mask()

    # DC 4 is reachable only through destination 2 and must be rejected.
    assert high_mask[4] == 0.0, high_mask.tolist()
    # DC 5 has the ordered extension 1->3->5 and remains feasible.
    assert high_mask[5] == 1.0, high_mask.tolist()

    # A destination switch may host a non-final stage only under the explicit
    # remaining-chain co-location contract.
    env.dc_nodes = {2, 4, 5}
    env.current_request["vnf"] = [0, 1]
    destination_dc_mask = high.get_high_level_action_mask()
    assert destination_dc_mask[2] == 1.0, destination_dc_mask.tolist()

    # Both final-stage DCs below are locally reachable, but DC 3 leaves no
    # legal post-chain route to destination 2. Completion lookahead must keep
    # only DC 4 before the policy can choose the dead end.
    lookahead_graph = nx.DiGraph()
    lookahead_graph.add_edges_from([(0, 1), (1, 3), (1, 4), (4, 2)])
    lookahead_env = make_env(lookahead_graph)
    lookahead_env.dc_nodes = {3, 4}
    lookahead_env.current_request['dest'] = [2]
    lookahead_env.current_request['vnf'] = [0]
    lookahead_env.low_level_controller = HopController(lookahead_graph)
    lookahead_high = HighLevelController(lookahead_env)
    lookahead_mask = lookahead_high.get_high_level_action_mask()
    assert lookahead_mask[3] == 0.0, lookahead_mask.tolist()
    assert lookahead_mask[4] == 1.0, lookahead_mask.tolist()

    # After the first local stage, every remaining stage is pinned to the same
    # destination DC; leaving it would invalidate service order.
    env.next_vnf_idx = 1
    env.chain_nodes = [2]
    env.current_node_location = 2
    pinned_mask = high.get_high_level_action_mask()
    assert pinned_mask[2] == 1.0, pinned_mask.tolist()
    assert float(np.sum(pinned_mask)) == 1.0, pinned_mask.tolist()

    env.next_vnf_idx = 0
    env.chain_nodes = [1]
    env.current_node_location = 1
    env.current_request["vnf"] = [0]
    env.dc_nodes = {4, 5}

    low = LowLevelController(env)
    env.low_level_controller = low
    low_mask = low.get_low_level_action_mask(mutate_env=False)
    assert low_mask[2] == 0.0, low_mask.tolist()
    assert low_mask[0] == 0.0, low_mask.tolist()
    assert low_mask[3] == 1.0, low_mask.tolist()

    # A DC used only as a transit hop must not be filtered by the VNF resource
    # probe.  Placement feasibility is checked for current_deployment_target=5,
    # not for intermediate DC 3 on the ordered path 1->3->5.
    env.dc_nodes = {3, 4, 5}
    original_probe = env.resource_mgr.probe_vnf_deploy
    env.resource_mgr.probe_vnf_deploy = lambda node, *args: (
        {"ok": False, "reuse": False, "reason": "cpu"}
        if int(node) == 3 else original_probe(node, *args)
    )
    transit_dc_mask = low.get_low_level_action_mask(mutate_env=False)
    assert transit_dc_mask[3] == 1.0, transit_dc_mask.tolist()

    # Destination branches start at the last VNF and must not re-enter any
    # earlier node on the ordered SFC spine, even through a reverse edge.
    env.current_phase = "destination_connection"
    env.current_request["dest"] = [4]
    env.current_node_location = 5
    env.current_deployment_target = None
    env.current_target_node = 4
    env.chain_nodes = [5]
    env.current_sfc = {
        "chain_nodes": [5],
        "spine_paths": [[0, 1, 3, 5]],
        "branch_paths": {},
    }
    env.current_tree = {
        "tree": {(0, 1): 1.0, (1, 3): 1.0, (3, 5): 1.0},
        "placement": {},
        "connected_dests": set(),
    }
    env.current_path_trace = [5]
    destination_mask = low.get_low_level_action_mask(mutate_env=False)
    assert destination_mask[3] == 0.0, destination_mask.tolist()
    assert destination_mask[2] == 1.0, destination_mask.tolist()

    # A receiver switch that has seen every VNF stage may replicate traffic to
    # its local host and remain an anchor for a later receiver.  This is valid
    # multicast forwarding, not an SFC-order violation.
    fanout_graph = nx.DiGraph()
    fanout_graph.add_edges_from([(0, 1), (1, 2)])
    fanout_env = make_env(fanout_graph)
    fanout_env.n = 3
    fanout_env.current_phase = 'destination_connection'
    fanout_env.current_request.update({
        'source': 0,
        'dest': [1, 2],
        'vnf': [0],
    })
    fanout_env.current_tree = {
        'tree': {(0, 1): 1.0},
        'placement': {(1, 0): 1},
        'connected_dests': {1},
        'node_stage': {0: 0, 1: 1},
    }
    fanout_env.chain_nodes = [1]
    fanout_env.nodes_on_tree = {0, 1}
    fanout_env.current_node_location = 1
    fanout_env.current_target_node = 2
    fanout_env.current_path_trace = [1]
    fanout_env.current_subgoal_full_path = [1]
    fanout_env.low_level_controller = HopController(fanout_graph)
    fanout_low = LowLevelController(fanout_env)
    fanout_env.low_level_controller = fanout_low
    assert fanout_low.compute_progress_path(1, 2) == [1, 2]
    fanout_plan = fanout_low.plan_destination_completion()
    assert fanout_plan and fanout_plan[0] == {
        'target': 2,
        'anchor': 1,
        'path': [1, 2],
        'cost': 1,
    }, fanout_plan
    fanout_valid, fanout_reason = fanout_low.validate_destination_plan_step(
        fanout_plan[0]
    )
    assert fanout_valid, fanout_reason

    # Completion lookahead must reserve cumulative resources for different
    # VNF types co-located by the destination contract.  Two individually
    # feasible 10-CPU stages do not fit in a 15-CPU live balance.
    cumulative_graph = nx.DiGraph([(0, 1)])
    cumulative_env = make_env(cumulative_graph)
    cumulative_env.n = 2
    cumulative_env.dc_nodes = {1}
    cumulative_env.current_request.update({
        'source': 0,
        'dest': [1],
        'vnf': [0, 1],
        'cpu_origin': [10.0, 10.0],
        'memory_origin': [1.0, 1.0],
    })
    cumulative_env.resource_mgr.pool.available_cpu[1] = 15.0
    cumulative_env.resource_mgr.pool.available_memory[1] = 15.0
    cumulative_env.chain_nodes = []
    cumulative_env.current_tree = {
        'tree': {}, 'placement': {}, 'connected_dests': set(),
    }
    cumulative_env.low_level_controller = HopController(cumulative_graph)
    cumulative_high = HighLevelController(cumulative_env)
    assert not cumulative_high._can_complete_after_placement(
        0, 1, 10.0, {1}
    )

    # Repeating the same VNF type may reuse the first planned instance and
    # therefore consumes the fresh-instance resources only once.
    cumulative_env.current_request['vnf'] = [0, 0]
    assert cumulative_high._can_complete_after_placement(
        0, 1, 10.0, {1}
    )

    # Executor-side validation must reject a node that is not a real directed
    # neighbor even if an upstream caller bypasses or misaligns the mask.
    before_tree = dict(env.current_tree["tree"])
    before_path = list(env.current_subgoal_full_path)
    low.get_state = lambda: {}
    _, _, done, truncated, invalid_info = low._handle_movement(5, 4, 4)
    assert not done and not truncated, invalid_info
    assert invalid_info["reason"] == "invalid_directed_edge", invalid_info
    assert env.current_node_location == 5
    assert env.current_tree["tree"] == before_tree
    assert env.current_subgoal_full_path == before_path

    return {
        "ok": True,
        "high_mask": high_mask.tolist(),
        "low_mask": low_mask.tolist(),
        "transit_dc_mask": transit_dc_mask.tolist(),
        "destination_mask": destination_mask.tolist(),
        "invalid_directed_action": invalid_info,
        "blocked_destination_dc": 4,
        "blocked_pre_chain_ancestor": 3,
        "ordered_dc": 5,
        "lookahead_mask": lookahead_mask.tolist(),
        "receiver_fanout_plan": fanout_plan,
        "cumulative_vnf_reservation": True,
    }


if __name__ == "__main__":
    result = run_checks()
    print(json.dumps(result, ensure_ascii=False, indent=2))
