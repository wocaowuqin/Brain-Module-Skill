#!/usr/bin/env python3
"""Smoke check for trainable role-DQN agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.sft_role_marl_env import SFTRoleMARLEnv
from scripts.check_reconfiguration_manager import build_resource_manager, deploy_active_sft


def run_check() -> dict:
    rm = build_resource_manager()
    deploy_active_sft(rm)
    env = SFTRoleMARLEnv(
        rm,
        top_k=3,
        node_util_threshold=0.50,
        link_util_threshold=0.80,
        min_risk_score=0.0,
        min_action_gain=0.0,
        agent_mode="trainable",
        max_tree_edge_growth=2,
        trainable_config={
            "hidden_dim": 32,
            "lr": 1e-3,
            "epsilon": 0.0,
            "device": "cpu",
        },
    )

    selector_obs = env.selector.observe(env.manager)
    selector_decision = env.selector.select_action(selector_obs)
    req_id = selector_decision.req_id

    migration_obs = env.migrator.observe(env.manager, req_id)
    migration_decision = env.migrator.select_action(migration_obs)

    reroute_obs = env.rerouter.observe(env.manager, req_id)
    reroute_decision = env.rerouter.select_action(reroute_obs)

    updates = {}
    for name, agent, obs in [
        ("selector", env.selector, selector_obs),
        ("migration", env.migrator, migration_obs),
        ("reroute", env.rerouter, reroute_obs),
    ]:
        vec = obs["obs_vector"]
        action = agent.last_action if agent.last_action is not None else 0
        agent.push_transition(vec, action, reward=1.0, next_obs=vec, done=True)
        updates[name] = agent.update_from_replay(batch_size=1)

    report = env.step(apply=False)
    ok = (
        selector_obs.get("obs_vector") is not None
        and migration_obs.get("obs_vector") is not None
        and reroute_obs.get("obs_vector") is not None
        and all(
            0.0 <= float(value) <= 1.0
            for obs in (selector_obs, migration_obs, reroute_obs)
            for value in obs["obs_vector"]
        )
        and all(item.get("updated", 0.0) == 1.0 for item in updates.values())
        and report["coordination"]["selected"] in {
            "noop", "migrate", "reroute", "migrate_and_reroute", "failed"
        }
    )
    return {
        "ok": bool(ok),
        "selector": selector_decision.to_dict(),
        "migration": migration_decision.to_dict(),
        "reroute": reroute_decision.to_dict(),
        "updates": updates,
        "pipeline_report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_check()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Trainable role-agent smoke check", "passed" if result["ok"] else "failed")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
