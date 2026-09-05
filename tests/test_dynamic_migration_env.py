from __future__ import annotations

import copy
import unittest

import numpy as np

from core.marl.batch_deployment_wqmix import AtomicResourceLedger, ResourceFootprint
from core.marl.migration_candidates import MigrationCandidate
from envs.dynamic_migration_env import DynamicMigrationEnv


def _profile() -> dict:
    nodes = [
        {"dpid": i, "host_ip": f"10.0.0.{i + 1}", "host_port": 10000 + i,
         "is_dc": i in {1, 2, 3, 4}}
        for i in range(6)
    ]
    edges = []
    for u, v in ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (1, 4)):
        edges.append({"u": u, "v": v, "u_port": 10 + u, "v_port": 20 + v,
                      "bandwidth_mbps": 100.0, "delay_ms": 1.0})
    return {"nodes": nodes, "dc_nodes_1based": [1, 2, 3, 4], "edges": edges,
            "default_bandwidth_mbps": 100.0, "default_delay_ms": 1.0}


def _request(request_id: int = 1, *, leave: float = 8.0) -> dict:
    return {
        "id": request_id, "arrival_time": 0.0, "leave_time": leave,
        "bw_origin": 5.0, "source_dpid": 0, "destination_dpids": [5],
        "dest": [5], "vnf": [0, 1, 2], "cpu_origin": [2.0, 2.0, 2.0],
        "memory_origin": [1.0, 1.0, 1.0], "multicast_ip": "239.1.1.1",
        "udp_port": 5000, "delay_bound_ms": 100.0,
    }


def _plan(request: dict, chain: list[int]) -> dict:
    segments = []
    for stage, (source, target) in enumerate(zip([0, *chain], chain)):
        segments.append({"stage": stage, "from_dpid": source, "to_dpid": target,
                         "target_ip": f"10.0.0.{target + 1}", "udp_port": 5000 + stage,
                         "path": [source, target], "switch_outputs": {}})
    return {
        "version": "test", "request_id": int(request["id"]), "accepted": True,
        "source_dpid": 0, "destination_dpids": [5], "chain_nodes": chain,
        "placement_by_vnf": {
            str(stage): {"dc_node": node, "vnf_type": int(request["vnf"][stage]),
                         "cpu_units": float(request["cpu_origin"][stage]),
                         "memory_units": float(request["memory_origin"][stage]),
                         "listen_ip": f"10.0.0.{node + 1}", "listen_port": 5000 + stage}
            for stage, node in enumerate(chain)
        },
        "segments": segments,
        "multicast": {"root_dpid": chain[-1], "dst_ip": request["multicast_ip"],
                       "udp_port": request["udp_port"], "paths": {"5": [chain[-1], 4, 5]},
                       "tree_edges": [[chain[-1], 4], [4, 5]], "switch_outputs": {}},
    }


def _make_env(*, traffic=None, leave=8.0, extra_arrival=False):
    request = _request(leave=leave)
    initial = _plan(request, [1, 2, 3])
    requests = [request]
    if extra_arrival:
        arriving = _request(2, leave=8.0)
        arriving["arrival_time"] = 1.0
        arriving["cpu_origin"] = [1.0, 1.0, 1.0]
        arriving["memory_origin"] = [0.5, 0.5, 0.5]
        arriving["plan"] = _plan(arriving, [2, 2, 2])
        requests.append(arriving)
    ledger = AtomicResourceLedger(
        {node: (3.0 if node == 2 else 20.0) for node in [1, 2, 3, 4]},
        {node: 20.0 for node in [1, 2, 3, 4]},
        {(edge[0], edge[1]): 100.0
         for row in _profile()["edges"]
         for edge in ((row["u"], row["v"]), (row["v"], row["u"]))},
    )
    fp = ResourceFootprint.from_sfc_plan(initial, request["bw_origin"])
    result = ledger.commit_exact([1], [[fp]], [0], expected_version=ledger.snapshot().version)
    assert result["committed"]
    env = DynamicMigrationEnv(
        ledger, requests, _profile(), initial_plans={1: initial},
        traffic_trace=traffic or [{"timestamp": 0.0, "request_id": 1, "bandwidth_mbps": 5.0}],
        max_agents=1, top_k=1, slot_seconds=1.0,
    )
    env.reset(seed=3)
    target = _plan(request, [1, 4, 3])
    candidate = MigrationCandidate(
        candidate_id="test-target", target_node=4, plan=target,
        footprint=ResourceFootprint.from_sfc_plan(target, request["bw_origin"]),
        metrics={}, objective=0.0, feasible=True,
    )
    # Replace generated candidates with one legal deterministic candidate so
    # this test exercises the atomic ledger boundary, not ranking heuristics.
    env.candidates = [[None, candidate]]
    env._mask = np.asarray([[1, 1]], dtype=np.int8)
    return env, initial, target


class DynamicMigrationEnvTest(unittest.TestCase):
    def test_migration_changes_placement_resources_and_noop_preserves_ledger(self):
        env, initial, target = _make_env()
        before = env.ledger.snapshot()
        env.step(np.asarray([1], dtype=np.int64))
        self.assertEqual(env.active_plans[1]["chain_nodes"], target["chain_nodes"])
        after = env.ledger.snapshot()
        self.assertNotEqual(dict(before.cpu_remaining), dict(after.cpu_remaining))
        self.assertNotEqual(dict(before.bandwidth_remaining), dict(after.bandwidth_remaining))
        cpu_used = dict(env.ledger.cpu_used); bw_used = dict(env.ledger.bandwidth_used)
        env.step(np.asarray([0], dtype=np.int64))
        self.assertEqual(cpu_used, dict(env.ledger.cpu_used))
        self.assertEqual(bw_used, dict(env.ledger.bandwidth_used))

    def test_action_changes_next_state_under_same_external_trace(self):
        no_move, _, _ = _make_env(traffic=[{"timestamp": 0.0, "request_id": 1, "bandwidth_mbps": 40.0}])
        move, _, _ = _make_env(traffic=[{"timestamp": 0.0, "request_id": 1, "bandwidth_mbps": 40.0}])
        state_noop, *_ = no_move.step(np.asarray([0], dtype=np.int64))
        state_move, *_ = move.step(np.asarray([1], dtype=np.int64))
        self.assertFalse(np.array_equal(state_noop["placements"], state_move["placements"]))
        self.assertFalse(np.array_equal(state_noop["link_state"], state_move["link_state"]))

    def test_migration_changes_later_request_admission(self):
        no_move, _, _ = _make_env(extra_arrival=True)
        moved, _, _ = _make_env(extra_arrival=True)
        no_move.step(np.asarray([0], dtype=np.int64))
        _, _, _, _, no_info = no_move.step(np.asarray([0], dtype=np.int64))
        moved.step(np.asarray([1], dtype=np.int64))
        _, _, _, _, moved_info = moved.step(np.asarray([0], dtype=np.int64))
        self.assertIn(2, no_info["events"]["rejected"])
        self.assertIn(2, moved_info["events"]["accepted"])

    def test_departure_releases_ledger_and_reset_restores_baseline(self):
        env, initial, target = _make_env(leave=1.0)
        baseline = copy.deepcopy(env.ledger.allocations)
        env.step(np.asarray([1], dtype=np.int64))
        env.reset(seed=4)
        self.assertEqual(env.active_plans[1]["chain_nodes"], initial["chain_nodes"])
        self.assertEqual(env.ledger.allocations, baseline)
        env.step(np.asarray([0], dtype=np.int64))
        self.assertNotIn(1, env.ledger.allocations)
        self.assertEqual(env.ledger.integrity_report()["fully_released"], True)

    def test_future_traffic_suffix_cannot_change_early_state(self):
        early = [{"timestamp": 0.0, "request_id": 1, "bandwidth_mbps": 5.0}]
        with_suffix = early + [{"timestamp": 10.0, "request_id": 1, "bandwidth_mbps": 99.0}]
        first, *_ = _make_env(traffic=early)
        second, *_ = _make_env(traffic=with_suffix)
        state_a, reward_a, *_ = first.step(np.asarray([0], dtype=np.int64))
        state_b, reward_b, *_ = second.step(np.asarray([0], dtype=np.int64))
        self.assertTrue(np.array_equal(state_a["node_state"], state_b["node_state"]))
        self.assertTrue(np.array_equal(state_a["link_state"], state_b["link_state"]))
        self.assertEqual(reward_a, reward_b)

    def test_masked_action_is_side_effect_free(self):
        env, _, _ = _make_env()
        env._mask[0, 1] = 0
        before = env.ledger.snapshot()
        state, reward, *_ = env.step(np.asarray([1], dtype=np.int64))
        after = env.ledger.snapshot()
        self.assertEqual(dict(before.cpu_remaining), dict(after.cpu_remaining))
        self.assertEqual(env.active_plans[1]["chain_nodes"], [1, 2, 3])
        self.assertEqual(state["placements"][0, :3].tolist(), [1, 2, 3])
        self.assertLess(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
