#!/usr/bin/env python3
"""Regression checks for the HRL low-level planner safety guard."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.modules.HRL_Coordinator import HRL_Coordinator


class HopController:
    def _get_hop_distance(self, source, target):
        return abs(int(target) - int(source))


class CompletionHigh:
    def _can_complete_after_placement(self, *args, **kwargs):
        return int(kwargs['previous_override']) == 3


def run_checks():
    coordinator = HRL_Coordinator.__new__(HRL_Coordinator)
    coordinator.env = SimpleNamespace(
        current_node_location=2,
        current_phase="vnf_deployment",
    )
    coordinator.planner_safety_guard = True
    coordinator._pos_history = [1, 2]
    mask = np.ones(8, dtype=np.float32)
    llc = HopController()

    action, guarded = coordinator._apply_planner_safety_guard(
        policy_action=2, planner_action=3, low_mask=mask,
        llc=llc, target_node=6,
    )
    assert (action, guarded) == (3, True)

    action, guarded = coordinator._apply_planner_safety_guard(
        policy_action=4, planner_action=3, low_mask=mask,
        llc=llc, target_node=6,
    )
    assert (action, guarded) == (4, False)

    mask[3] = 0.0
    action, guarded = coordinator._apply_planner_safety_guard(
        policy_action=2, planner_action=3, low_mask=mask,
        llc=llc, target_node=6,
    )
    assert (action, guarded) == (2, False)

    coordinator.env.current_request = {
        'vnf': [0], 'dest': [6], 'bw_origin': 10.0,
    }
    coordinator.env.next_vnf_idx = 0
    coordinator.env.current_deployment_target = 6
    coordinator.env.high_level_controller = CompletionHigh()
    mask[3] = 1.0
    action, guarded = coordinator._apply_vnf_completion_guard(
        policy_action=4, planner_action=3, low_mask=mask,
        llc=llc, target_node=6,
    )
    assert (action, guarded) == (3, True)

    return {
        "ok": True,
        "guarded_action": 3,
        "completion_guard": True,
    }


if __name__ == "__main__":
    print(run_checks())
