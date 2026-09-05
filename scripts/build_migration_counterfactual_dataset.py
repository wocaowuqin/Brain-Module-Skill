#!/usr/bin/env python3
"""Build paired migrate/noop labels from deterministic HRL trace replay.

Each sample rebuilds the runtime twice, replays the same prefix, and then
compares one exact migration with noop over the same lookahead arrivals.  This
is intentionally slower than online inference: it is an offline label builder.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.migration_benefit_predictor import (  # noqa: E402
    FEATURE_NAMES,
    PREDICTION_TARGET,
    build_migration_benefit_features,
)
from scripts.run_role_reconfig_eval import build_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired future-benefit labels for VNF migration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="phase3")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topo", default="us_backbone")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--lookahead", type=int, default=15)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-weight", type=float, default=0.02)
    parser.add_argument("--hotspot-weight", type=float, default=0.25)
    return parser.parse_args()


def _runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        data=args.data,
        topo=args.topo,
        episodes=args.episodes,
        max_steps=args.max_steps,
        reconfig_interval=0,
        warmup_episodes=0,
        plan_only=False,
        output=str(args.output.parent / "unused"),
        gpu=-1,
        seed=args.seed,
        goal_strategy="adaptive",
        ablation_variant="full",
        checkpoint=args.checkpoint,
        deployment_executor="policy",
        bw_cap=None,
        cap_cpu=None,
        cap_mem=None,
        top_k=3,
        node_util_threshold=0.80,
        link_util_threshold=0.80,
        delay_threshold=None,
        min_risk_score=0.0,
        min_action_gain=0.0,
        max_tree_edge_growth=0,
        max_reroute_link_utilization=0.50,
        min_connectivity_ratio=1.0,
        role_agent_mode="rule",
        role_hidden_dim=64,
        role_lr=1e-3,
        role_epsilon=0.0,
        role_checkpoint=None,
        disable_migration=False,
        disable_reroute=True,
        migration_benefit_predictor=None,
        migration_benefit_threshold=None,
    )


def _run_requests(coordinator: Any, count: int, max_steps: int = 600) -> dict[str, Any]:
    successes = 0
    reasons: dict[str, int] = {}
    for _ in range(max(0, int(count))):
        _, info = coordinator.run_episode(training=False, max_steps=max_steps)
        successes += int(bool(info.get("success", False)))
        reason = str(info.get("reason") or "success")
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"requests": int(count), "accepted": successes, "reasons": reasons}


def _find_proposal(manager: Any) -> tuple[Any, Any] | tuple[None, None]:
    for risk in manager.select_topk_risky_sfts(3):
        proposal = manager.plan_greedy_migrate(risk.req_id)
        if proposal.action_type == "migrate_vnf" and proposal.estimated_gain > 0.0:
            return risk, proposal
    return None, None


def _branch(
    args: argparse.Namespace,
    decision_episode: int,
    *,
    migrate: bool,
    expected_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _, _, _, coordinator, role_env = build_runtime(_runtime_args(args))
    prefix = _run_requests(coordinator, decision_episode, args.max_steps)
    manager = role_env.manager
    risk, proposal = _find_proposal(manager)
    if risk is None or proposal is None:
        return {"available": False, "prefix": prefix}

    action = proposal.to_dict()
    target = action.get("target") or {}
    identity = {
        "req_id": int(action["req_id"]),
        "vnf_idx": int(target["vnf_idx"]),
        "old_node": int(target["old_node"]),
        "new_node": int(target["new_node"]),
    }
    if expected_action is not None and identity != expected_action:
        raise RuntimeError(
            "counterfactual replay diverged before the decision point: "
            f"expected={expected_action}, actual={identity}"
        )
    features = build_migration_benefit_features(manager, risk, proposal)
    apply_result = None
    if migrate:
        apply_result = manager.apply_migration_proposal(proposal).to_dict()
        if not apply_result.get("success"):
            raise RuntimeError(f"counterfactual migration failed: {apply_result}")
    future = _run_requests(coordinator, args.lookahead, args.max_steps)
    return {
        "available": True,
        "identity": identity,
        "risk": asdict(risk),
        "proposal": action,
        "features": features,
        "prefix": prefix,
        "future": future,
        "metrics": manager.metrics_snapshot(),
        "apply_result": apply_result,
    }


def _utility(args: argparse.Namespace, outcome: dict[str, Any]) -> float:
    metrics = outcome["metrics"]
    return (
        100.0 * float(outcome["future"]["accepted"])
        - args.delay_weight * float(metrics.get("avg_delay_total_ms", 0.0))
        - args.hotspot_weight
        * float(metrics.get("node_hotspots", 0) + metrics.get("link_hotspots", 0))
    )


def main() -> int:
    args = parse_args()
    if args.lookahead <= 0 or args.interval <= 0:
        raise ValueError("lookahead and interval must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    points = list(range(args.interval, args.episodes - args.lookahead + 1, args.interval))
    if args.max_points > 0:
        points = points[: args.max_points]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.perf_counter()
    for decision_episode in points:
        noop = _branch(args, decision_episode, migrate=False)
        if not noop.get("available"):
            continue
        migrated = _branch(
            args,
            decision_episode,
            migrate=True,
            expected_action=noop["identity"],
        )
        noop_utility = _utility(args, noop)
        migration_utility = _utility(args, migrated)
        accepted_delta = (
            int(migrated["future"]["accepted"])
            - int(noop["future"]["accepted"])
        )
        delay_delta = (
            float(migrated["metrics"].get("avg_delay_total_ms", 0.0))
            - float(noop["metrics"].get("avg_delay_total_ms", 0.0))
        )
        label = int(migration_utility > noop_utility + 1e-9)
        rows.append({
            "schema": "migration_counterfactual_v1",
            "prediction_target": PREDICTION_TARGET,
            "seed": int(args.seed),
            "decision_episode": int(decision_episode),
            "lookahead_requests": int(args.lookahead),
            "feature_names": list(FEATURE_NAMES),
            "features": noop["features"],
            "label": label,
            "accepted_delta": accepted_delta,
            "modeled_delay_delta_ms": delay_delta,
            "utility_delta": migration_utility - noop_utility,
            "action_identity": noop["identity"],
            "risk": noop["risk"],
            "noop": {"future": noop["future"], "metrics": noop["metrics"]},
            "migrate": {
                "future": migrated["future"],
                "metrics": migrated["metrics"],
                "apply_result": migrated["apply_result"],
            },
        })
        print(
            f"episode={decision_episode} label={label} "
            f"accepted_delta={accepted_delta:+d} delay_delta_ms={delay_delta:+.3f}",
            flush=True,
        )

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "rows": len(rows),
        "positive_labels": sum(int(row["label"]) for row in rows),
        "negative_labels": sum(1 - int(row["label"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(args.output.resolve()),
        "scope": "modeled HRL counterfactual replay; not measured Mininet strict SLA",
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
