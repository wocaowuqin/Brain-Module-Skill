#!/usr/bin/env python3
"""Generate deployed-state snapshots and safe-reroute labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.trainable_role_agents import _proposal_vector
from scripts.audit_role_action_space import ledger_errors
from scripts.run_role_reconfig_eval import build_runtime


SCHEMA_VERSION = "2.0.0"
DATASET_VERSION = "actionable_reroute_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate actionable safe-reroute snapshots from a deployed request trace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="phase3")
    parser.add_argument("--data", required=True, help="source requests.pkl")
    parser.add_argument("--topo", default="us_backbone")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--trace-seed", type=int, required=True)
    parser.add_argument("--deploy-seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=0, help="0 uses every request in the trace")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--node-util-threshold", type=float, default=0.50)
    parser.add_argument("--link-util-threshold", type=float, default=0.50)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--deployment-executor",
        default="policy",
        choices=["policy", "bw_planner"],
    )
    parser.add_argument("--output", default="data/actionable_reroute_v2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet-runtime", action="store_true", help="suppress DEBUG/INFO rollout logs")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(payload: Any, length: int = 24) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def write_jsonl(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def runtime_args(args: argparse.Namespace, data_path: Path, checkpoint: Path) -> SimpleNamespace:
    deploy_seed = int(args.deploy_seed if args.deploy_seed is not None else args.trace_seed)
    return SimpleNamespace(
        config=args.config,
        data=str(data_path),
        topo=args.topo,
        max_steps=args.max_steps,
        gpu=-1,
        seed=deploy_seed,
        goal_strategy="adaptive",
        ablation_variant="single_dqn",
        checkpoint=str(checkpoint),
        deployment_executor=args.deployment_executor,
        bw_cap=None,
        cap_cpu=None,
        cap_mem=None,
        top_k=args.top_k,
        node_util_threshold=args.node_util_threshold,
        link_util_threshold=args.link_util_threshold,
        delay_threshold=None,
        min_risk_score=0.0,
        min_action_gain=0.0,
        role_agent_mode="trainable",
        role_hidden_dim=64,
        role_lr=1e-3,
        role_epsilon=0.0,
        role_checkpoint=None,
        disable_migration=True,
        disable_reroute=False,
    )


def planner_config(manager) -> Dict[str, Any]:
    keys = [
        "node_util_threshold",
        "link_util_threshold",
        "delay_threshold",
        "max_path_hops",
        "queueing_delay_weight",
        "max_tree_edge_growth",
        "max_reroute_delay_ratio",
        "max_reroute_delay_increase_ms",
        "path_utilization_weight",
        "path_delay_weight",
        "max_projected_link_utilization",
        "max_reroute_candidates",
    ]
    return {key: getattr(manager, key) for key in keys}


def record_structure(record, manager) -> Dict[str, bool]:
    tree = set(record.tree_edges)
    ledger_rows = [(int(item.u), int(item.v)) for item in record.edge_allocations]
    ledger = set(ledger_rows)
    tree_nodes = {node for edge in tree for node in edge}
    critical = set(int(node) for node in record.connected_dests)
    critical.update(int(node) for node in record.placement_by_vnf.values())
    rooted_tree_valid = manager._is_directed_tree(tree, int(record.source))
    ledger_consistent = tree == ledger and len(ledger_rows) == len(ledger)
    critical_connected = critical.issubset(tree_nodes)
    return {
        "rooted_tree_valid": rooted_tree_valid,
        "ledger_consistent": ledger_consistent,
        "critical_connected": critical_connected,
        "base_structure_valid": rooted_tree_valid and ledger_consistent and critical_connected,
    }


def serialize_record(record, manager) -> Dict[str, Any]:
    return {
        "req_id": int(record.req_id),
        "source": int(record.source),
        "dests": [int(value) for value in record.dests],
        "vnfs": [int(value) for value in record.vnfs],
        "bw": float(record.bw),
        "state": str(record.state),
        "connected_dests": sorted(int(value) for value in record.connected_dests),
        "vnf_bindings": [
            {
                "node": int(item.node),
                "vnf_type": int(item.vnf_type),
                "inst_id": str(item.inst_id),
                "reused": bool(item.reused),
                "cpu": float(item.cpu),
                "mem": float(item.mem),
            }
            for item in record.vnf_bindings
        ],
        "edge_allocations": [
            {"u": int(item.u), "v": int(item.v), "bw": float(item.bw)}
            for item in record.edge_allocations
        ],
        "tree_edges": [
            {"u": int(u), "v": int(v), "flow": float(flow)}
            for (u, v), flow in sorted(record.tree_edges.items())
        ],
        "tree_usage": [
            {"u": int(u), "v": int(v), "count": int(count)}
            for (u, v), count in sorted(record.tree_usage.items())
        ],
        "placement_by_vnf": [
            {"vnf_index": int(index), "node": int(node)}
            for index, node in sorted(record.placement_by_vnf.items())
        ],
        "placement_detail": [
            {"vnf_index": int(index), **json_safe(detail)}
            for index, detail in sorted(record.placement_detail.items())
        ],
        "node_stage": [
            {"node": int(node), "stage": int(stage)}
            for node, stage in sorted(record.node_stage.items())
        ],
        "snapshot_time": nullable_float(record.snapshot_time),
        "last_reconfig_time": nullable_float(record.last_reconfig_time),
        "migration_count": int(record.migration_count),
        "reconfig_count": int(record.reconfig_count),
        "structure": record_structure(record, manager),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def nullable_float(value: Any) -> Any:
    return None if value is None else float(value)


def serialize_instance(instance) -> Dict[str, Any]:
    return {
        "inst_id": str(instance.inst_id),
        "node": int(instance.node),
        "vnf_type": int(instance.vnf_type),
        "cpu": float(instance.cpu),
        "mem": float(instance.mem),
        "ref_count": int(instance.ref_count),
        "active_req_ids": sorted(int(value) for value in instance.active_req_ids),
        "state": str(instance.state),
    }


def network_snapshot(manager) -> Dict[str, Any]:
    pool = manager.rm.pool
    nodes = []
    for node in range(manager.rm.n):
        cpu_cap = float(pool.cpu_cap[node])
        mem_cap = float(pool.mem_cap[node])
        cpu_avail = float(pool.get_available_cpu(node))
        mem_avail = float(pool.get_available_memory(node))
        nodes.append({
            "node": node,
            "cpu_cap": cpu_cap,
            "cpu_avail": cpu_avail,
            "cpu_util": 1.0 - cpu_avail / max(cpu_cap, 1e-9),
            "mem_cap": mem_cap,
            "mem_avail": mem_avail,
            "mem_util": 1.0 - mem_avail / max(mem_cap, 1e-9),
        })
    links = []
    for u, v in sorted(pool.iter_bandwidth_keys()):
        cap = float(pool.bw_cap[(u, v)])
        avail = float(pool.get_available_bandwidth(u, v))
        links.append({
            "u": int(u),
            "v": int(v),
            "cap": cap,
            "avail": avail,
            "util": 1.0 - avail / max(cap, 1e-9),
            "base_delay_ms": float(manager._edge_delay(u, v, getattr(manager.rm, "topo", None))),
        })
    return {
        "bandwidth_model": str(pool.bandwidth_model),
        "nodes": nodes,
        "links": links,
    }


def action_evidence(manager, action) -> Dict[str, Any]:
    target = action.target or {}
    if action.action_type != "reroute_edge" or not target:
        return {}
    rec = manager.rm.request_table.get(action.req_id)
    old_edge = tuple(target["old_edge"])
    path = [int(node) for node in target["new_path"]]
    new_edges = [(path[index], path[index + 1]) for index in range(len(path) - 1)]
    remaining = set(rec.tree_edges) - {old_edge}
    bw = float(target.get("bw", rec.bw))
    edge_evidence = []
    for u, v in new_edges:
        cap = float(manager.rm.pool.bw_cap.get((u, v), 0.0))
        avail = float(manager.rm.pool.get_available_bandwidth(u, v))
        reused = (u, v) in remaining
        extra_bw = 0.0 if reused else bw
        edge_evidence.append({
            "u": u,
            "v": v,
            "reused": reused,
            "cap": cap,
            "avail": avail,
            "bw_margin": avail - extra_bw,
            "pre_util": manager._edge_utilization(u, v),
            "projected_util": manager._edge_utilization(u, v, extra_bw=extra_bw),
            "projected_delay_ms": manager._projected_edge_delay(u, v, extra_bw=extra_bw),
        })
    safety = target.get("safety", {})
    delay_limit = (
        float(safety.get("old_edge_delay_ms", 0.0)) * manager.max_reroute_delay_ratio
        + manager.max_reroute_delay_increase_ms
    )
    return {
        "candidate_id": target.get("candidate_id"),
        "old_edge": list(old_edge),
        "anchor": path[0] if path else None,
        "child": path[-1] if path else None,
        "new_path": path,
        "new_edges": [list(edge) for edge in new_edges],
        "edge_evidence": edge_evidence,
        "util_margin": manager.max_projected_link_utilization - float(safety.get("max_new_utilization", 1.0)),
        "delay_margin_ms": delay_limit - float(safety.get("new_path_delay_ms", delay_limit)),
        "growth_margin": manager.max_tree_edge_growth - int(safety.get("extra_edges", 0)),
    }


def rejection_code(manager, action, diagnostics: Dict[str, Any]) -> str:
    if action.action_type == "reroute_edge" and action.estimated_gain > 0.0:
        return "ACTION_AVAILABLE"
    if diagnostics.get("outcome") == "no_active_sft":
        return "NO_ACTIVE_SFT"
    if int(diagnostics.get("hot_edges", 0)) == 0:
        return "NO_HOT_EDGE"
    if int(diagnostics.get("candidate_paths", 0)) == 0:
        return "NO_PATH"
    if int(diagnostics.get("nonpositive_gain", 0)) > 0:
        return "NON_POSITIVE_GAIN"
    rejections = diagnostics.get("validation_rejections") or {}
    if rejections:
        reason = max(rejections, key=lambda item: rejections[item])
        return manager._validation_status_code(reason)
    return "NO_FEASIBLE_ALTERNATE_PATH"


def scenario_folder(output: Path, args: argparse.Namespace, deploy_seed: int) -> Path:
    return (
        output / args.topo / f"rate{args.rate:g}" / args.split
        / f"trace_seed_{args.trace_seed}" / f"deploy_seed_{deploy_seed}"
    )


def copy_source_files(source_data: Path, folder: Path) -> Dict[str, str]:
    copied = {}
    for name in ("requests.pkl", "requests_by_slot.pkl", "events.pkl", "events_by_slot.pkl"):
        source = source_data.parent / name
        if source.exists():
            target = folder / name
            shutil.copy2(source, target)
            copied[name] = sha256(target)
    return copied


def rebuild_manifest(output: Path) -> None:
    rows = []
    for spec_path in output.rglob("scenario_spec.json"):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        rows.append({
            "topology": spec["topology"],
            "rate": spec["base_rate"],
            "split": spec["split"],
            "trace_seed": spec["trace_seed"],
            "deploy_seed": spec["deploy_seed"],
            "episodes": spec["coverage"]["episodes"],
            "decision_points": spec["coverage"]["decision_points"],
            "positive_steps": spec["coverage"]["positive_steps"],
            "positive_step_rate": spec["coverage"]["positive_step_rate"],
            "ledger_consistent": spec["coverage"]["ledger_consistent"],
            "scenario_path": str(spec_path.parent.relative_to(output)).replace("\\", "/"),
            "states_sha256": spec["files"]["states.jsonl"],
            "decisions_sha256": spec["files"]["decisions.jsonl"],
        })
    rows.sort(key=lambda row: (row["split"], int(row["trace_seed"]), row["topology"], float(row["rate"])))
    fields = list(rows[0]) if rows else []
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    source_data = Path(args.data)
    if not source_data.is_absolute():
        source_data = ROOT / source_data
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not source_data.exists():
        raise FileNotFoundError(f"request data not found: {source_data}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"deployment checkpoint not found: {checkpoint}")
    with source_data.open("rb") as handle:
        requests = pickle.load(handle)
    episodes = len(requests) if int(args.episodes) <= 0 else min(len(requests), int(args.episodes))
    if episodes <= 0:
        raise ValueError("source trace has no requests")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    deploy_seed = int(args.deploy_seed if args.deploy_seed is not None else args.trace_seed)
    folder = scenario_folder(output, args, deploy_seed)
    if folder.exists():
        if not args.overwrite:
            raise FileExistsError(f"scenario exists: {folder}; use --overwrite")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    config, env, _, coordinator, role_env = build_runtime(
        runtime_args(args, source_data, checkpoint)
    )
    if args.quiet_runtime:
        logging.disable(logging.INFO)
    manager = role_env.manager
    planner = planner_config(manager)
    planner_hash = stable_id(planner, 64)
    scenario_id = stable_id({
        "version": DATASET_VERSION,
        "topology": args.topo,
        "rate": args.rate,
        "split": args.split,
        "trace_seed": args.trace_seed,
        "deploy_seed": deploy_seed,
        "deployment_executor": args.deployment_executor,
        "request_sha256": sha256(source_data),
        "checkpoint_sha256": sha256(checkpoint),
        "planner_hash": planner_hash,
    })

    copied_hashes = copy_source_files(source_data, folder)
    state_count = 0
    decision_count = 0
    deployment_successes = 0
    positive_steps = 0
    positive_decisions = 0
    negative_decisions = 0
    valid_base_decisions = 0
    invalid_base_decisions = 0
    rejection_counts: Dict[str, int] = {}
    ledger_failure_steps = []
    old_edges = set()
    new_paths = set()

    states_path = folder / "states.jsonl"
    decisions_path = folder / "decisions.jsonl"
    with states_path.open("w", encoding="utf-8") as state_handle, decisions_path.open(
        "w", encoding="utf-8"
    ) as decision_handle:
        for episode in range(1, episodes + 1):
            _, info = coordinator.run_episode(training=False, max_steps=int(args.max_steps))
            deployment_successes += int(bool(info.get("success", False)))
            metrics = manager.metrics_snapshot()
            risks = manager.select_topk_risky_sfts(args.top_k)
            selector_obs = role_env.selector.observe(manager)
            errors = ledger_errors(manager)
            if errors:
                ledger_failure_steps.append({"episode": episode, "errors": errors})

            decisions = []
            valid_selector_actions = [0]
            for rank, risk in enumerate(risks, start=1):
                action, diagnostics = manager.plan_greedy_reroute_with_diagnostics(risk.req_id)
                available = bool(action.action_type == "reroute_edge" and action.estimated_gain > 0.0)
                record = manager.rm.request_table.get(risk.req_id)
                base_valid = record_structure(record, manager)["base_structure_valid"]
                available = available and base_valid
                if available:
                    valid_selector_actions.append(rank)
                code = (
                    rejection_code(manager, action, diagnostics)
                    if base_valid else "INVALID_BASE_STRUCTURE"
                )
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
                decisions.append((rank, risk, action, diagnostics, available, code, base_valid))

            state_payload = {
                "schema_version": SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "scenario_id": scenario_id,
                "topology": args.topo,
                "base_rate": float(args.rate),
                "split": args.split,
                "trace_seed": int(args.trace_seed),
                "deploy_seed": deploy_seed,
                "episode": episode,
                "sim_time": nullable_float(getattr(env, "time_step", None)),
                "deployment": {
                    "success": bool(info.get("success", False)),
                    "fail_reason": str(info.get("reason", "") or info.get("fail_reason", "")),
                },
                "metrics": json_safe(metrics),
                "network": network_snapshot(manager),
                "active_sfts": [
                    serialize_record(record, manager) for record in manager.active_sft_records()
                ],
                "instances": [
                    serialize_instance(instance)
                    for _, instance in sorted(manager.rm.instance_table.items())
                    if str(instance.state) == "ACTIVE"
                ],
                "topk_risks": [risk.__dict__ for risk in risks],
                "selector": {
                    "obs_vector": [float(value) for value in selector_obs["obs_vector"]],
                    "valid_actions": valid_selector_actions,
                },
                "ledger_consistent": not errors,
                "planner_config_hash": planner_hash,
            }
            state_id = stable_id(state_payload)
            state_payload["state_id"] = state_id
            write_jsonl(state_handle, state_payload)
            state_count += 1

            step_positive = False
            for rank, risk, action, diagnostics, available, code, base_valid in decisions:
                record = manager.rm.request_table.get(risk.req_id)
                decision_id = stable_id({"state_id": state_id, "req_id": risk.req_id, "rank": rank})
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_version": DATASET_VERSION,
                    "scenario_id": scenario_id,
                    "state_id": state_id,
                    "decision_id": decision_id,
                    "episode": episode,
                    "rank": rank,
                    "req_id": int(risk.req_id),
                    "risk": risk.__dict__,
                    "request": serialize_record(record, manager),
                    "reroute_obs_vector": [float(value) for value in _proposal_vector(metrics, action)],
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
                }
                write_jsonl(decision_handle, row)
                decision_count += 1
                valid_base_decisions += int(base_valid)
                invalid_base_decisions += int(not base_valid)
                step_positive = step_positive or available
                if available:
                    positive_decisions += 1
                    target = action.target or {}
                    old_edges.add(tuple(target.get("old_edge", [])))
                    new_paths.add(tuple(target.get("new_path", [])))
                else:
                    negative_decisions += 1
            positive_steps += int(step_positive)

    duration = max(float(request.get("arrival_time", 0.0)) for request in requests) if requests else 0.0
    coverage = {
        "episodes": episodes,
        "states": state_count,
        "decision_points": decision_count,
        "positive_steps": positive_steps,
        "positive_step_rate": positive_steps / max(1, episodes),
        "positive_decisions": positive_decisions,
        "negative_decisions": negative_decisions,
        "valid_base_decisions": valid_base_decisions,
        "invalid_base_decisions": invalid_base_decisions,
        "valid_base_rate": valid_base_decisions / max(1, decision_count),
        "unique_old_edges": len(old_edges),
        "unique_new_paths": len(new_paths),
        "deployment_successes": deployment_successes,
        "deployment_accept_rate": deployment_successes / max(1, episodes),
        "ledger_consistent": not ledger_failure_steps,
        "ledger_failure_steps": ledger_failure_steps,
        "rejection_counts": rejection_counts,
    }
    topology_file = Path(str(config.get("topology", {}).get("file", "")))
    if topology_file and not topology_file.is_absolute():
        topology_file = ROOT / topology_file
    files = {
        **copied_hashes,
        "states.jsonl": sha256(states_path),
        "decisions.jsonl": sha256(decisions_path),
    }
    spec = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "scenario_id": scenario_id,
        "topology": args.topo,
        "base_rate": float(args.rate),
        "expected_mean_rate": float(args.rate) * 1.07,
        "observed_rate": len(requests) / max(duration, 1e-9),
        "split": args.split,
        "trace_seed": int(args.trace_seed),
        "deploy_seed": deploy_seed,
        "deployment_executor": args.deployment_executor,
        "migration_enabled": False,
        "top_k": int(args.top_k),
        "planner_config": planner,
        "planner_config_hash": planner_hash,
        "source_request_sha256": sha256(source_data),
        "deployment_checkpoint": str(checkpoint),
        "deployment_checkpoint_sha256": sha256(checkpoint),
        "topology_file": str(topology_file),
        "topology_sha256": sha256(topology_file) if topology_file.exists() else None,
        "files": files,
        "coverage": coverage,
    }
    spec_path = folder / "scenario_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    output.mkdir(parents=True, exist_ok=True)
    dataset_spec = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "unit": "deployed decision state with Top-K safe-greedy reroute labels",
        "formal_gates": {
            "min_episodes_per_seed": 500,
            "min_positive_steps_per_seed": 30,
            "positive_step_rate": [0.05, 0.30],
            "min_negative_decisions_per_seed": 100,
            "min_valid_base_rate": 0.95,
            "ledger_violations": 0,
        },
        "notes": [
            "Migration is disabled in v2 reroute-only data.",
            "safe_greedy_candidate is a heuristic label, not a globally optimal path label.",
            "Training samplers may rebalance train data; validation and test distributions must remain intact.",
        ],
    }
    (output / "dataset_spec.json").write_text(
        json.dumps(dataset_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rebuild_manifest(output)
    print(json.dumps({"scenario": str(folder), "coverage": coverage}, ensure_ascii=False, indent=2))
    return 0 if coverage["ledger_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
