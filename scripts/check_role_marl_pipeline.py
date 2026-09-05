#!/usr/bin/env python3
"""End-to-end smoke check for role-based SFT reconfiguration."""

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


def run_pipeline(apply_actions: bool = True) -> dict:
    rm = build_resource_manager()
    deploy_active_sft(rm)
    env = SFTRoleMARLEnv(
        rm,
        top_k=3,
        node_util_threshold=0.50,
        link_util_threshold=0.80,
        min_risk_score=0.0,
        min_action_gain=0.0,
        max_tree_edge_growth=2,
    )
    report = env.step(apply=apply_actions)
    selected = report["coordination"]["selected"]
    snapshot_ok = bool((report.get("snapshot_report") or {}).get("ok", False))
    progressed = (
        selected in {"migrate", "reroute", "migrate_and_reroute"}
        and report["after"]["total_migrations"] + report["after"]["total_reconfigs"] > 0
    )
    report["ok"] = bool(report["ok"] and snapshot_ok and (progressed or not apply_actions))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true", help="do not apply selected actions")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    result = run_pipeline(apply_actions=not args.plan_only)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Role-MARL pipeline smoke check", "passed" if result["ok"] else "failed")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
