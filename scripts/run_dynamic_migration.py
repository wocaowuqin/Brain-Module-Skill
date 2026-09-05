#!/usr/bin/env python3
"""Run a short causal DynamicMigrationEnv rollout.

This is a diagnostic entrypoint for phase 1.  It records action-dependent
ledger and fluid-queue transitions; it does not claim packet-level SLA
measurement or the stage-2 economic reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.orchestration.hrl_adapter import build_ledger_from_profile  # noqa: E402
from envs.dynamic_migration_env import DynamicMigrationEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, help="runtime request JSONL")
    parser.add_argument("--profile", required=True, help="topology/resource profile JSON")
    parser.add_argument(
        "--plans",
        help="optional JSONL mapping request id to plan; plans are attached to arrivals",
    )
    parser.add_argument(
        "--traffic", "--traffic-trace", dest="traffic",
        help="optional offered-traffic JSONL",
    )
    parser.add_argument(
        "--steps", "--max-steps", dest="steps", type=int, default=20,
    )
    parser.add_argument("--slot-seconds", type=float, default=1.0)
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--policy", choices=("noop", "first-feasible"), default="noop")
    parser.add_argument("--output", default="artifacts/runs/migration/dynamic_rollout")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def request_id(row: Mapping[str, Any]) -> int:
    for key in ("id", "request_id"):
        if key in row:
            return int(row[key])
    raise ValueError("request/plan row requires id or request_id")


def attach_plans(requests: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    plans: dict[int, Any] = {}
    for row in rows:
        if "plan" not in row:
            raise ValueError("plan JSONL rows require a plan field")
        plans[request_id(row)] = row["plan"]
    for row in requests:
        rid = int(row["id"])
        if rid in plans:
            row["plan"] = plans[rid]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def output_paths(value: str | Path) -> tuple[Path, Path]:
    path = resolve(value)
    if path.suffix.lower() == ".jsonl":
        return path, path.with_name(f"{path.stem}_summary.json")
    path.mkdir(parents=True, exist_ok=True)
    return path / "transitions.jsonl", path / "summary.json"


def choose_actions(env: DynamicMigrationEnv, policy: str) -> list[int]:
    actions = [0] * env.max_agents
    if policy == "noop":
        return actions
    mask = env.get_state()["action_mask"]
    for index in range(min(env.max_agents, len(env.tasks))):
        feasible = [candidate for candidate in range(1, env.top_k + 1) if mask[index, candidate]]
        if feasible:
            actions[index] = int(feasible[0])
    return actions


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    requests = read_jsonl(resolve(args.requests))
    profile = json.loads(resolve(args.profile).read_text(encoding="utf-8"))
    if args.plans:
        attach_plans(requests, read_jsonl(resolve(args.plans)))
    traffic = read_jsonl(resolve(args.traffic)) if args.traffic else None
    ledger = build_ledger_from_profile(profile)
    env = DynamicMigrationEnv(
        ledger,
        requests,
        profile,
        traffic_trace=traffic,
        slot_seconds=args.slot_seconds,
        max_agents=args.max_agents,
        top_k=args.top_k,
        max_steps=args.steps,
    )
    transitions_path, summary_path = output_paths(args.output)
    transitions_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    terminated = truncated = False
    try:
        observation, info = env.reset()
        for step in range(args.steps):
            actions = choose_actions(env, args.policy)
            observation, reward, terminated, truncated, info = env.step(actions)
            rows.append({
                "step": step,
                "actions": actions,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "observation": {
                    "states": jsonable(observation["states"]),
                    "action_mask": jsonable(observation["action_mask"]),
                    "agent_mask": jsonable(observation["agent_mask"]),
                    "placements": jsonable(observation["placements"]),
                    "node_state": jsonable(observation["node_state"]),
                    "link_state": jsonable(observation["link_state"]),
                    "time": jsonable(observation["time"]),
                },
                "info": jsonable(info),
            })
            if terminated or truncated:
                break
    finally:
        env.close()
    with transitions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "valid": True,
        "steps": len(rows),
        "requested_steps": args.steps,
        "policy": args.policy,
        "accepted_count": int(info.get("accepted_count", 0)) if rows else 0,
        "rejected_count": int(info.get("rejected_count", 0)) if rows else 0,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "transition_semantics": "action_dependent_ledger_and_fluid_queues",
        "data_plane_sla_measured": False,
        "reward_semantics": "diagnostic_queue_work_reduction_minus_failed_actions",
        "transitions": str(transitions_path.resolve()),
        "baseline_groups": ["no_migration", "reactive_threshold", "oracle_future", "lifecycle_aware_wqmix"],
        "metrics": {
            "invalid_migration_rate_30s": None,
            "migration_interruption_time_ms": None,
            "cluster_load_std_reduction": None,
        },
        "metric_note": "filled by phase-4 evaluator when data-plane and migration timing traces are available",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
