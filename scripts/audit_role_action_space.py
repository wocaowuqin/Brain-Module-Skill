#!/usr/bin/env python3
"""Audit feasible role actions without applying reconfiguration."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_role_reconfig_eval import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Top-K migration/reroute feasibility on a request trace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="phase3")
    parser.add_argument("--data", required=True)
    parser.add_argument("--topo", default="us_backbone")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--node-util-threshold", type=float, default=0.50)
    parser.add_argument("--link-util-threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        data=args.data,
        topo=args.topo,
        max_steps=args.max_steps,
        gpu=-1,
        seed=args.seed,
        goal_strategy="adaptive",
        ablation_variant="single_dqn",
        checkpoint=args.checkpoint,
        bw_cap=None,
        cap_cpu=None,
        cap_mem=None,
        top_k=args.top_k,
        node_util_threshold=args.node_util_threshold,
        link_util_threshold=args.link_util_threshold,
        delay_threshold=None,
        min_risk_score=0.0,
        min_action_gain=0.0,
        role_agent_mode="rule",
        role_hidden_dim=64,
        role_lr=1e-3,
        role_epsilon=0.0,
        role_checkpoint=None,
    )


def ledger_errors(manager) -> list[Dict[str, Any]]:
    errors = []
    for rec in manager.active_sft_records():
        tree_edges = set(rec.tree_edges)
        allocations = [(int(item.u), int(item.v), float(item.bw)) for item in rec.edge_allocations]
        allocation_edges = [(u, v) for u, v, _ in allocations]
        duplicate_edges = sorted(edge for edge, count in Counter(allocation_edges).items() if count != 1)
        wrong_bw = sorted((u, v) for u, v, bw in allocations if abs(bw - float(rec.bw)) > 1e-5)
        if tree_edges != set(allocation_edges) or duplicate_edges or wrong_bw:
            errors.append({
                "req_id": int(rec.req_id),
                "missing_allocations": sorted(tree_edges - set(allocation_edges)),
                "stale_allocations": sorted(set(allocation_edges) - tree_edges),
                "duplicate_allocations": duplicate_edges,
                "wrong_bandwidth": wrong_bw,
            })
    return errors


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _, _, _, coordinator, role_env = build_runtime(runtime_args(args))
    manager = role_env.manager
    reroute_reasons: Counter[str] = Counter()
    migration_reasons: Counter[str] = Counter()
    selected_sfts = 0
    feasible_reroutes = 0
    feasible_migrations = 0
    episodes_with_reroute = 0
    episodes_with_migration = 0
    deployment_successes = 0
    ledger_failures = []

    for episode in range(1, int(args.episodes) + 1):
        _, info = coordinator.run_episode(training=False, max_steps=int(args.max_steps))
        deployment_successes += int(bool(info.get("success", False)))
        episode_reroutes = 0
        episode_migrations = 0
        for risk in manager.select_topk_risky_sfts(args.top_k):
            selected_sfts += 1
            migration = manager.plan_greedy_migrate(req_id=risk.req_id)
            reroute = manager.plan_greedy_reroute(req_id=risk.req_id)
            migration_reasons[migration.reason or migration.action_type] += 1
            reroute_reasons[reroute.reason or reroute.action_type] += 1
            if migration.action_type != "noop" and migration.estimated_gain > 0.0:
                feasible_migrations += 1
                episode_migrations += 1
            if reroute.action_type != "noop" and reroute.estimated_gain > 0.0:
                feasible_reroutes += 1
                episode_reroutes += 1
        episodes_with_migration += int(episode_migrations > 0)
        episodes_with_reroute += int(episode_reroutes > 0)
        errors = ledger_errors(manager)
        if errors:
            ledger_failures.append({"episode": episode, "errors": errors})

    return {
        "episodes": int(args.episodes),
        "deployment_successes": deployment_successes,
        "deployment_accept_rate": deployment_successes / max(1, int(args.episodes)),
        "topk_sfts_audited": selected_sfts,
        "feasible_migrations": feasible_migrations,
        "feasible_migration_rate": feasible_migrations / max(1, selected_sfts),
        "episodes_with_migration": episodes_with_migration,
        "feasible_reroutes": feasible_reroutes,
        "feasible_reroute_rate": feasible_reroutes / max(1, selected_sfts),
        "episodes_with_reroute": episodes_with_reroute,
        "migration_reasons": dict(migration_reasons),
        "reroute_reasons": dict(reroute_reasons),
        "ledger_consistent": not ledger_failures,
        "ledger_failures": ledger_failures,
    }


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    result = run(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if result["ledger_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
