#!/usr/bin/env python3
"""Validate migration dataset, checkpoint, bounded batching, and online inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.migration_dataset import load_migration_records  # noqa: E402
from core.marl.deployment_topk import deserialize_footprint  # noqa: E402
from core.marl.joint_candidate_decoder import joint_footprints_feasible  # noqa: E402
from core.marl.migration_scheduler import (  # noqa: E402
    MigrationTask,
    migration_execution_waves,
)
from envs.migration_wqmix_env import snapshot_from_record  # noqa: E402
from sdn.online_migration_wqmix_planner import OnlineMigrationWQMIXPlanner  # noqa: E402
from sdn.migration_monitor import OnlineMigrationMonitor  # noqa: E402
from sdn.migration_safety import make_before_break_compatible  # noqa: E402
from scripts.run_sdn_runtime_requests import (  # noqa: E402
    migrated_sfc_plan,
    migrated_sft_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    records = load_migration_records([resolve(args.data)])
    if not records:
        raise ValueError("migration dataset is empty")
    checkpoint = torch.load(resolve(args.checkpoint), map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") not in {
        "migration_wqmix_v1",
        "migration_policy_v2",
    }:
        raise ValueError("wrong checkpoint type")
    max_agents = int(checkpoint["max_agents"])
    max_candidates = int(checkpoint["max_candidates"])
    reject_mask_errors = 0
    dimension_errors = 0
    for record in records:
        if len(record["agents"]) > max_agents:
            dimension_errors += 1
        for agent in record["agents"]:
            reject = int(agent["reject_action"])
            reject_mask_errors += int(
                reject >= len(agent["action_mask"])
                or not bool(agent["action_mask"][reject])
            )
            dimension_errors += int(len(agent["candidates"]) > max_candidates + 1)

    report = json.loads(resolve(args.runtime_report).read_text(encoding="utf-8"))
    plans = {
        int(plan["request_id"]): plan
        for plan in report.get("online_plans", [])
        if plan.get("placement_by_vnf")
    }
    arrivals = {
        int(event["request_id"]): event
        for event in report.get("events", [])
        if event.get("type") == "arrive"
    }
    base_sft_errors = sum(
        int(
            len((plan.get("multicast") or {}).get("paths", {})) != 5
            or int((plan.get("multicast") or {}).get("root_dpid", -1))
            != int(plan.get("chain_nodes", [-1])[-1])
        )
        for plan in plans.values()
    )
    replay_candidate_errors = 0
    replay_candidates_checked = 0
    replay_candidate_differs_from_manual_shortest = 0
    profile = json.loads(resolve(args.profile).read_text(encoding="utf-8"))
    for record in records:
        for agent in record["agents"]:
            request_id = int(agent["task"]["request_id"])
            stage = int(agent["task"]["stage"])
            current_plan = plans.get(request_id)
            if current_plan is None:
                continue
            for candidate in agent["candidates"]:
                candidate_plan = candidate.get("plan")
                target_dc = candidate.get("target_node")
                if not candidate_plan or target_dc is None:
                    continue
                replay_candidates_checked += 1
                materialized = migrated_sft_plan(
                    profile,
                    current_plan,
                    stage,
                    int(target_dc),
                    29999,
                    candidate_plan=candidate_plan,
                )
                candidate_paths = {
                    int(segment["stage"]): [
                        int(node) for node in segment.get("path", [])
                    ]
                    for segment in candidate_plan.get("segments", [])
                }
                execution_paths = {
                    int(segment["stage"]): [
                        int(node) for node in segment.get("path", [])
                    ]
                    for segment in materialized.get("segments", [])
                }
                replay_candidate_errors += int(
                    candidate_paths != execution_paths
                    or materialized.get("multicast")
                    != current_plan.get("multicast")
                )
                for segment in materialized.get("segments", []):
                    outputs = segment.get("switch_outputs") or {}
                    replay_candidate_errors += sum(
                        int(not outputs.get(str(int(node))))
                        for node in segment.get("path") or []
                    )
                manual = migrated_sfc_plan(
                    profile, current_plan, stage, int(target_dc), 29999
                )
                manual_paths = {
                    int(segment["stage"]): [
                        int(node) for node in segment.get("path", [])
                    ]
                    for segment in manual.get("segments", [])
                }
                replay_candidate_differs_from_manual_shortest += int(
                    candidate_paths != manual_paths
                )
    first_record = records[0]
    online_entries = []
    for agent in first_record["agents"]:
        task = agent["task"]
        request_id = int(task["request_id"])
        if request_id not in plans:
            continue
        event = arrivals.get(request_id, {})
        request = {
            "id": request_id,
            "bw_origin": float(task["bandwidth_mbps"]),
            "leave_time": float(task["created_at"]) + float(task["remaining_lifetime_s"]),
            "delay_bound_ms": 50.0,
            "predicted_sla_risk": float(task["sla_risk"]),
        }
        if event:
            request["leave_time"] = max(
                request["leave_time"],
                float(event.get("trace_time", 0.0)) + float(task["remaining_lifetime_s"]),
            )
        online_entries.append({
            "request": request,
            "current_plan": plans[request_id],
            "stage": int(task["stage"]),
        })
    planner = OnlineMigrationWQMIXPlanner(
        resolve(args.checkpoint), resolve(args.profile)
    )
    online_entries = online_entries[:max_agents]
    decisions = planner.plan_batch(
        online_entries,
        snapshot_from_record(first_record),
        decision_time=float(first_record.get("batch_id", 0)),
    )
    candidate_plan_errors = 0
    execution_path_errors = 0
    multicast_errors = 0
    selected_footprints = []
    materialized_plans = 0
    candidate_differs_from_manual_shortest = 0
    for entry, decision in zip(online_entries, decisions):
        for candidate in decision.get("ranked_candidates", []):
            target_plan = candidate.get("target_plan")
            if candidate.get("target_dc") is None:
                continue
            if not isinstance(target_plan, dict):
                candidate_plan_errors += 1
                continue
            for segment in target_plan.get("segments", []):
                outputs = segment.get("switch_outputs") or {}
                for node in segment.get("path") or []:
                    ports = outputs.get(str(int(node)), outputs.get(int(node)))
                    candidate_plan_errors += int(not ports)
        if not decision["accepted"]:
            continue
        candidate_plan = decision.get("target_plan")
        selected_candidate = decision.get("selected_candidate") or {}
        if not isinstance(candidate_plan, dict) or not selected_candidate.get(
            "resource_footprint"
        ):
            candidate_plan_errors += 1
            continue
        materialized = migrated_sft_plan(
            profile,
            entry["current_plan"],
            int(entry["stage"]),
            int(decision["target_dc"]),
            29999,
            candidate_plan=candidate_plan,
        )
        materialized_plans += 1
        candidate_paths = {
            int(segment["stage"]): [int(node) for node in segment.get("path", [])]
            for segment in candidate_plan.get("segments", [])
        }
        execution_paths = {
            int(segment["stage"]): [int(node) for node in segment.get("path", [])]
            for segment in materialized.get("segments", [])
        }
        execution_path_errors += int(candidate_paths != execution_paths)
        manual_plan = migrated_sfc_plan(
            profile,
            entry["current_plan"],
            int(entry["stage"]),
            int(decision["target_dc"]),
            29999,
        )
        manual_paths = {
            int(segment["stage"]): [int(node) for node in segment.get("path", [])]
            for segment in manual_plan.get("segments", [])
        }
        candidate_differs_from_manual_shortest += int(
            candidate_paths != manual_paths
        )
        multicast_errors += int(
            materialized.get("multicast") != entry["current_plan"].get("multicast")
        )
        flow_safe, _ = make_before_break_compatible(
            entry["current_plan"], materialized
        )
        execution_path_errors += int(not flow_safe)
        selected_footprints.append(
            deserialize_footprint(selected_candidate["resource_footprint"])
        )
    joint_feasible, joint_reason = joint_footprints_feasible(
        selected_footprints, snapshot_from_record(first_record)
    )
    tasks = [MigrationTask.from_dict(agent["task"]) for agent in first_record["agents"]]
    waves = migration_execution_waves(tasks, max_inflight=4)
    monitor = OnlineMigrationMonitor(55.0, 45.0, max_agents=max_agents)
    active_plans = {
        int(entry["request"]["id"]): entry["current_plan"]
        for entry in online_entries
    }
    active_requests = {
        int(entry["request"]["id"]): {
            **entry["request"],
            "arrival_time": 0.0,
            "leave_time": max(10.0, float(entry["request"]["leave_time"])),
        }
        for entry in online_entries
    }
    cooldown_entries_before_scan = len(monitor.last_migration)
    monitor.scan(
        snapshot_from_record(first_record), active_plans, active_requests, 0.0
    )
    monitored = monitor.scan(
        snapshot_from_record(first_record), active_plans, active_requests, 1.0
    )
    cooldown_entries_after_scan = len(monitor.last_migration)
    prediction_field_errors = sum(
        int(
            any(
                key not in entry or key not in entry["request"]
                for key in (
                    "current_utilization",
                    "predicted_utilization",
                    "predicted_sla_risk",
                )
            )
        )
        for entry in monitored
    )
    monitored_decisions = planner.plan_batch(
        monitored,
        snapshot_from_record(first_record),
        decision_time=1.0,
    ) if monitored else []
    prediction_value_errors = sum(
        int(
            abs(
                float(decision["task"]["current_utilization"])
                - float(entry["current_utilization"])
            )
            > 1e-9
            or abs(
                float(decision["task"]["predicted_utilization"])
                - float(entry["predicted_utilization"])
            )
            > 1e-9
            or abs(
                float(decision["task"]["sla_risk"])
                - float(entry["predicted_sla_risk"])
            )
            > 1e-9
        )
        for entry, decision in zip(monitored, monitored_decisions)
    )
    output = {
        "valid": (
            reject_mask_errors == 0
            and dimension_errors == 0
            and len(decisions) == len(online_entries)
            and all(len(wave) <= 4 for wave in waves)
            and len(monitored) <= max_agents
            and candidate_plan_errors == 0
            and execution_path_errors == 0
            and multicast_errors == 0
            and joint_feasible
            and cooldown_entries_before_scan == cooldown_entries_after_scan
            and prediction_field_errors == 0
            and len(monitored_decisions) == len(monitored)
            and prediction_value_errors == 0
            and base_sft_errors == 0
            and replay_candidates_checked > 0
            and replay_candidate_errors == 0
            and replay_candidate_differs_from_manual_shortest > 0
        ),
        "records": len(records),
        "max_agents": max_agents,
        "max_candidates": max_candidates,
        "reject_mask_errors": reject_mask_errors,
        "dimension_errors": dimension_errors,
        "base_sft_plans": len(plans),
        "base_sft_errors": base_sft_errors,
        "replay_candidates_checked": replay_candidates_checked,
        "replay_candidate_errors": replay_candidate_errors,
        "replay_candidate_differs_from_manual_shortest": (
            replay_candidate_differs_from_manual_shortest
        ),
        "online_tasks": len(online_entries),
        "online_selected": sum(int(row["accepted"]) for row in decisions),
        "materialized_selected_plans": materialized_plans,
        "candidate_plan_errors": candidate_plan_errors,
        "execution_path_errors": execution_path_errors,
        "candidate_differs_from_manual_shortest": (
            candidate_differs_from_manual_shortest
        ),
        "multicast_errors": multicast_errors,
        "joint_resource_feasible": joint_feasible,
        "joint_resource_reason": joint_reason,
        "online_metadata": planner.metadata(),
        "execution_wave_sizes": [len(wave) for wave in waves],
        "predictive_monitor_tasks": len(monitored),
        "prediction_field_errors": prediction_field_errors,
        "prediction_value_errors": prediction_value_errors,
        "cooldown_unchanged_by_scan": (
            cooldown_entries_before_scan == cooldown_entries_after_scan
        ),
        "predictive_monitor": monitor.metadata(),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
