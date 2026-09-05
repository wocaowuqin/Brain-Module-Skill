#!/usr/bin/env python3
"""Run a causal MIH-style VNF migration baseline on DynamicMigrationEnv.

MIH is implemented as a transparent heuristic: a causal linear forecast of
offered traffic gates overloaded source nodes, then the feasible candidate
with the lowest migration cost and projected target utilization is selected.
This is a reproduction baseline, not the authors' original implementation.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.orchestration.hrl_adapter import build_ledger_from_profile
from envs.dynamic_migration_env import DynamicMigrationEnv


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def causal_forecast(values: list[float], horizon: int) -> float:
    """Linear trend forecast from history strictly before the current slot."""
    if not values:
        return 0.0
    if len(values) == 1:
        return max(0.0, values[-1])
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, np.asarray(values, dtype=float), 1)
    return max(0.0, float(intercept + slope * (len(values) - 1 + horizon)))


def mih_actions(env: DynamicMigrationEnv, history: dict[int, list[float]], threshold: float, horizon: int) -> list[int]:
    actions = [0] * env.max_agents
    state = env.get_state()
    node_index = {node: i for i, node in enumerate(env.nodes)}
    for i, task in enumerate(env.tasks):
        samples = history.get(task.request_id, [])[-max(4, horizon * 4):]
        forecast = causal_forecast(samples, horizon)
        node_row = state["node_state"][node_index[task.old_node]]
        cpu_util = float(node_row[0]) / max(1e-9, env.ledger.cpu_capacity[task.old_node])
        predicted_util = max(cpu_util, cpu_util * forecast / max(1e-9, samples[-1] if samples else forecast or 1.0))
        if predicted_util <= threshold:
            continue
        feasible = []
        for action in range(1, env.top_k + 1):
            if not state["action_mask"][i, action]:
                continue
            candidate = env.candidates[i][action]
            metrics = candidate.metrics
            score = (
                float(metrics.get("migration_ms", 0.0))
                + 100.0 * float(metrics.get("projected_target_utilization", 0.0))
                - 50.0 * float(metrics.get("utilization_relief", 0.0))
            )
            feasible.append((score, action))
        if feasible:
            actions[i] = min(feasible)[1]
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--plans")
    parser.add_argument("--traffic", required=True)
    parser.add_argument("--output", default="artifacts/runs/migration/mih_baseline")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    requests = read_jsonl(Path(args.requests))
    if args.plans:
        plans = {int(row.get("id", row.get("request_id"))): row["plan"] for row in read_jsonl(Path(args.plans))}
        for row in requests:
            if int(row["id"]) in plans:
                row["plan"] = plans[int(row["id"])]
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    traffic = read_jsonl(Path(args.traffic))
    env = DynamicMigrationEnv(build_ledger_from_profile(profile), requests, profile, traffic_trace=traffic, max_steps=args.steps)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    transitions = []
    try:
        obs, info = env.reset()
        for step in range(args.steps):
            history: dict[int, list[float]] = {}
            for row in traffic:
                if float(row["timestamp"]) < env.now:
                    history.setdefault(int(row["request_id"]), []).append(float(row["bandwidth_mbps"]))
            actions = mih_actions(env, history, args.threshold, args.horizon)
            obs, reward, terminated, truncated, info = env.step(actions)
            transitions.append({"step": step, "time": env.now, "actions": actions, "reward": float(reward), "info": info})
            if terminated or truncated:
                break
    finally:
        env.close()
    (output / "transitions.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in transitions) + "\n", encoding="utf-8")
    summary = {"algorithm": "MIH-style-causal-heuristic", "paper_reference": "10.1109/GLOBECOM46510.2021.9685818", "steps": len(transitions), "threshold": args.threshold, "horizon": args.horizon, "note": "reproduction baseline; not authors' official code"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
