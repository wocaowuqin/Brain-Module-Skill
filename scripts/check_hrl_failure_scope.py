#!/usr/bin/env python3
"""Regression check for stage-scoped HRL temporary failure bans."""

from types import SimpleNamespace
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.modules.HRL_Coordinator import HRL_Coordinator


def run_checks():
    coordinator = HRL_Coordinator.__new__(HRL_Coordinator)
    coordinator.env = SimpleNamespace(
        current_request={"id": 7071, "vnf": [0, 1, 2]},
        next_vnf_idx=0,
        _episode_deploy_failed={9},
        hard_tabu_list={3},
        current_path_trace=[1, 2],
        recent_edge_trace=[(1, 2)],
        _last_target_goal=2,
        _stuck_steps=4,
        _last_dist_to_target=3,
    )
    coordinator._episode_deploy_failed = {8}
    coordinator._unreachable_targets = {7}
    coordinator._failure_memory_scope = None

    assert coordinator._sync_failure_memory_scope() == (7071, "vnf", 0)
    assert coordinator._unreachable_targets == set()
    assert coordinator._episode_deploy_failed == set()
    assert coordinator.env._episode_deploy_failed is coordinator._episode_deploy_failed
    assert coordinator.env.hard_tabu_list == set()
    assert coordinator.env.current_path_trace == []
    assert coordinator.env.recent_edge_trace == []
    assert coordinator.env._last_target_goal is None

    coordinator._unreachable_targets.add(6)
    coordinator._episode_deploy_failed.add(5)
    coordinator._sync_failure_memory_scope()
    assert coordinator._unreachable_targets == {6}
    assert coordinator._episode_deploy_failed == {5}

    coordinator.env.next_vnf_idx = 1
    assert coordinator._sync_failure_memory_scope() == (7071, "vnf", 1)
    assert coordinator._unreachable_targets == set()
    assert coordinator._episode_deploy_failed == set()

    coordinator._unreachable_targets.add(4)
    coordinator._episode_deploy_failed.add(3)
    coordinator.env.next_vnf_idx = 3
    assert coordinator._sync_failure_memory_scope() == (7071, "destination")
    assert coordinator._unreachable_targets == set()
    assert coordinator._episode_deploy_failed == set()

    return {"ok": True, "scope": coordinator._failure_memory_scope}


if __name__ == "__main__":
    print(run_checks())
