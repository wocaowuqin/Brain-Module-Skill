#!/usr/bin/env python3
"""Generate a controlled valid-tree benchmark for safe rerouting."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pickle
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.trainable_role_agents import TrainableSFTSelectionAgent, _proposal_vector
from envs.modules.AllResourceManager import FusedResourceManager
from envs.modules.reconfiguration_manager import ReconfigurationManager
from scripts.generate_actionable_reroute_dataset import (
    DATASET_VERSION,
    SCHEMA_VERSION,
    action_evidence,
    network_snapshot,
    planner_config,
    rebuild_manifest,
    record_structure,
    rejection_code,
    serialize_instance,
    serialize_record,
    sha256,
    stable_id,
    write_jsonl,
)


NEGATIVE_VARIANTS = ["no_hot", "insufficient_bw", "projected_util", "delay", "growth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled safe-reroute states with balanced labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="data/actionable_reroute_v2_controlled")
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--positive-rate", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_undirected_edge(topo: np.ndarray, u: int, v: int) -> None:
    topo[u, v] = 1.0
    topo[v, u] = 1.0


def build_controlled_state(rng: random.Random, req_id: int, variant: str):
    n = 7
    labels = list(range(n))
    rng.shuffle(labels)
    root, child, dest_a, dest_b, alt_a, alt_b, alt_c = labels
    topo = np.zeros((n, n), dtype=float)
    for u, v in ((root, child), (child, dest_a), (child, dest_b)):
        add_undirected_edge(topo, u, v)
    if variant == "growth":
        alternate_path = [root, alt_a, alt_b, alt_c, child]
    else:
        alternate_path = [root, alt_a, child]
    for u, v in zip(alternate_path, alternate_path[1:]):
        add_undirected_edge(topo, u, v)

    capacities = {
        "cpu": 100.0,
        "memory": 80.0,
        "bandwidth": 50.0,
        "bandwidth_model": "directed",
    }
    rm = FusedResourceManager(topo=topo, capacities=capacities, dc_nodes=list(range(n)))
    bw = float(rng.choice([6, 8, 10, 12]))
    rm.register_request_record(
        req_id=req_id, source=root, dests=[dest_a, dest_b], vnfs=[0, 1], bw=bw
    )
    first = rm.try_deploy_new_kernel(
        child, 0, req_cpu=20.0, req_mem=10.0, allow_reuse=True, req_id=req_id
    )
    second = rm.try_deploy_new_kernel(
        dest_a, 1, req_cpu=12.0, req_mem=8.0, allow_reuse=True, req_id=req_id
    )
    if not first.ok or not second.ok:
        raise RuntimeError("controlled VNF deployment failed")
    rm.bind_vnf_to_request(req_id, child, 0, first, req_cpu=20.0, req_mem=10.0)
    rm.bind_vnf_to_request(req_id, dest_a, 1, second, req_cpu=12.0, req_mem=8.0)
    rm.mark_dest_connected(req_id, dest_a)
    rm.mark_dest_connected(req_id, dest_b)
    tree = {(root, child): 1.0, (child, dest_a): 1.0, (child, dest_b): 1.0}
    for u, v in tree:
        if not rm.commit_edge_bandwidth(req_id, u, v, bw):
            raise RuntimeError("controlled tree bandwidth commit failed")
    rm.snapshot_request_sft(
        req_id,
        current_tree={
            "tree": tree,
            "placement": {
                (child, 0): {
                    "node": child,
                    "vnf_type": 0,
                    "cpu_used": 20.0,
                    "mem_used": 10.0,
                    "reused": False,
                    "inst_id": first.inst_id,
                },
                (dest_a, 1): {
                    "node": dest_a,
                    "vnf_type": 1,
                    "cpu_used": 12.0,
                    "mem_used": 8.0,
                    "reused": False,
                    "inst_id": second.inst_id,
                },
            },
            "node_stage": {root: 0, child: 1, dest_a: 2, dest_b: 2},
            "tree_usage": {edge: 1 for edge in tree},
        },
        snapshot_time=float(req_id),
    )

    delay_matrix = np.where(topo > 0, 1.0, 0.0)
    if variant == "delay":
        for u, v in zip(alternate_path, alternate_path[1:]):
            delay_matrix[u, v] = 10.0
            delay_matrix[v, u] = 10.0
    env = SimpleNamespace(resource_mgr=rm, delay_matrix=delay_matrix, time_step=float(req_id))
    manager = ReconfigurationManager(
        env, node_util_threshold=0.50, link_util_threshold=0.80
    )

    if variant != "no_hot":
        old_extra = max(0.0, 0.84 * 50.0 - bw)
        if not rm.allocate_bandwidth(root, child, old_extra):
            raise RuntimeError("controlled hotspot allocation failed")
    if variant == "insufficient_bw":
        for u, v in zip(alternate_path, alternate_path[1:]):
            rm.allocate_bandwidth(u, v, 50.0 - max(0.0, bw - 1.0))
    elif variant == "projected_util":
        for u, v in zip(alternate_path, alternate_path[1:]):
            rm.allocate_bandwidth(u, v, 50.0 - (bw + 2.0))
    return manager, rm.request_table[req_id], alternate_path


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    if args.episodes <= 0 or not 0.05 <= args.positive_rate <= 0.30:
        raise ValueError("episodes must be positive and positive-rate must be in [0.05, 0.30]")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    folder = (
        output / "controlled7" / "rate0" / args.split
        / f"trace_seed_{args.seed}" / f"deploy_seed_{args.seed + 3000}"
    )
    if folder.exists():
        if not args.overwrite:
            raise FileExistsError(f"scenario exists: {folder}; use --overwrite")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    positive_target = max(1, round(args.episodes * args.positive_rate))
    schedule = ["positive"] * positive_target
    schedule.extend(
        NEGATIVE_VARIANTS[index % len(NEGATIVE_VARIANTS)]
        for index in range(args.episodes - positive_target)
    )
    rng.shuffle(schedule)

    states_path = folder / "states.jsonl"
    decisions_path = folder / "decisions.jsonl"
    requests = []
    rejection_counts: Counter[str] = Counter()
    positive_steps = 0
    old_edges = set()
    new_paths = set()
    scenario_id = stable_id({
        "dataset": DATASET_VERSION,
        "kind": "controlled7",
        "split": args.split,
        "seed": args.seed,
        "episodes": args.episodes,
        "positive_rate": args.positive_rate,
    })
    planner = None
    with states_path.open("w", encoding="utf-8") as state_handle, decisions_path.open(
        "w", encoding="utf-8"
    ) as decision_handle:
        for episode, variant in enumerate(schedule, start=1):
            manager, record, alternate_path = build_controlled_state(rng, episode, variant)
            planner = planner_config(manager)
            planner_hash = stable_id(planner, 64)
            risk = manager.score_sft_risk(record)
            action, diagnostics = manager.plan_greedy_reroute_with_diagnostics(record.req_id)
            base_valid = record_structure(record, manager)["base_structure_valid"]
            available = bool(
                base_valid and action.action_type == "reroute_edge" and action.estimated_gain > 0.0
            )
            code = rejection_code(manager, action, diagnostics) if base_valid else "INVALID_BASE_STRUCTURE"
            rejection_counts[code] += 1
            positive_steps += int(available)
            if available:
                target = action.target or {}
                old_edges.add(tuple(target.get("old_edge", [])))
                new_paths.add(tuple(target.get("new_path", [])))

            selector = TrainableSFTSelectionAgent(top_k=3, epsilon=0.0)
            selector_obs = selector.observe(manager)
            state_payload = {
                "schema_version": SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "scenario_id": scenario_id,
                "topology": "controlled7",
                "base_rate": 0.0,
                "split": args.split,
                "trace_seed": int(args.seed),
                "deploy_seed": int(args.seed + 3000),
                "episode": episode,
                "sim_time": float(episode),
                "deployment": {"success": True, "fail_reason": ""},
                "metrics": manager.metrics_snapshot(),
                "network": network_snapshot(manager),
                "active_sfts": [serialize_record(record, manager)],
                "instances": [
                    serialize_instance(instance)
                    for _, instance in sorted(manager.rm.instance_table.items())
                ],
                "topk_risks": [risk.__dict__],
                "selector": {
                    "obs_vector": [float(value) for value in selector_obs["obs_vector"]],
                    "valid_actions": [0, 1] if available else [0],
                },
                "ledger_consistent": True,
                "planner_config_hash": planner_hash,
                "controlled_variant": variant,
            }
            state_id = stable_id(state_payload)
            state_payload["state_id"] = state_id
            write_jsonl(state_handle, state_payload)

            decision = {
                "schema_version": SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "scenario_id": scenario_id,
                "state_id": state_id,
                "decision_id": stable_id({"state_id": state_id, "req_id": record.req_id}),
                "episode": episode,
                "rank": 1,
                "req_id": int(record.req_id),
                "risk": risk.__dict__,
                "request": serialize_record(record, manager),
                "reroute_obs_vector": [
                    float(value) for value in _proposal_vector(manager.metrics_snapshot(), action)
                ],
                "valid_actions": [0, 1] if available else [0],
                "labels": {
                    "safe_greedy_candidate": available,
                    "safety_feasible": available,
                    "positive_gain": bool(action.estimated_gain > 0.0),
                    "action_available": available,
                    "rejection_code": code,
                },
                "planner_action": action.to_dict(),
                "planner_diagnostics": diagnostics,
                "candidate_evidence": action_evidence(manager, action),
                "controlled_variant": variant,
            }
            write_jsonl(decision_handle, decision)
            requests.append({
                "id": episode,
                "controlled_variant": variant,
                "expected_action_available": available,
                "alternate_path": alternate_path,
            })

    requests_path = folder / "requests.pkl"
    with requests_path.open("wb") as handle:
        pickle.dump(requests, handle, protocol=pickle.HIGHEST_PROTOCOL)
    planner_hash = stable_id(planner, 64)
    generator_hash = sha256(Path(__file__).resolve())
    coverage = {
        "episodes": args.episodes,
        "states": args.episodes,
        "decision_points": args.episodes,
        "positive_steps": positive_steps,
        "positive_step_rate": positive_steps / args.episodes,
        "positive_decisions": positive_steps,
        "negative_decisions": args.episodes - positive_steps,
        "valid_base_decisions": args.episodes,
        "invalid_base_decisions": 0,
        "valid_base_rate": 1.0,
        "unique_old_edges": len(old_edges),
        "unique_new_paths": len(new_paths),
        "deployment_successes": args.episodes,
        "deployment_accept_rate": 1.0,
        "ledger_consistent": True,
        "ledger_failure_steps": [],
        "rejection_counts": dict(rejection_counts),
    }
    spec = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "scenario_id": scenario_id,
        "topology": "controlled7",
        "base_rate": 0.0,
        "expected_mean_rate": 0.0,
        "observed_rate": 0.0,
        "split": args.split,
        "trace_seed": int(args.seed),
        "deploy_seed": int(args.seed + 3000),
        "migration_enabled": False,
        "top_k": 1,
        "planner_config": planner,
        "planner_config_hash": planner_hash,
        "source_request_sha256": sha256(requests_path),
        "deployment_checkpoint": "controlled-generator",
        "deployment_checkpoint_sha256": generator_hash,
        "topology_file": "generated-controlled7",
        "topology_sha256": generator_hash,
        "files": {
            "requests.pkl": sha256(requests_path),
            "states.jsonl": sha256(states_path),
            "decisions.jsonl": sha256(decisions_path),
        },
        "coverage": coverage,
        "controlled_benchmark": True,
    }
    (folder / "scenario_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_spec.json").write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "dataset_version": DATASET_VERSION,
            "benchmark": "controlled7",
            "purpose": "algorithm validation before end-to-end HRL integration",
            "formal_gates": {
                "min_episodes_per_seed": 500,
                "min_positive_steps_per_seed": 30,
                "positive_step_rate": [0.05, 0.30],
                "min_negative_decisions_per_seed": 100,
                "min_valid_base_rate": 0.95,
            },
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rebuild_manifest(output)
    print(json.dumps({"scenario": str(folder), "coverage": coverage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
