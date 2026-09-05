"""Online bounded joint planner for migration-specific WQMIX checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from core.marl.batch_deployment_wqmix import (
    BatchCandidateQNetwork,
    ResourceSnapshot,
    ranked_actions,
)
from core.marl.joint_candidate_decoder import decode_joint_candidates
from core.marl.migration_candidates import (
    MigrationCandidateGenerator,
    candidate_mask,
    migration_batch_candidate_features,
    migration_request_features,
)
from core.marl.migration_dataset import MigrationFeatureNormalizer
from core.marl.migration_scheduler import MigrationTask


class OnlineMigrationWQMIXPlanner:
    """Select jointly feasible target DCs for at most ``max_agents`` VNFs."""

    def __init__(
        self,
        checkpoint: str | Path,
        profile: str | Path,
        *,
        cpu_capacity: float = 55.0,
        memory_capacity: float = 45.0,
        device: str = "cpu",
        torch_threads: int = 1,
        decoder_top_r: int = 4,
        decoder_time_budget_ms: float = 2.0,
    ) -> None:
        if torch_threads:
            torch.set_num_threads(int(torch_threads))
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        if payload.get("checkpoint_type") not in {
            "migration_wqmix_v1",
            "migration_policy_v2",
        }:
            raise ValueError(
                "online migration requires a migration policy checkpoint; "
                "deployment checkpoints are intentionally rejected"
            )
        self.checkpoint_path = str(Path(checkpoint).resolve())
        self.algorithm = str(payload.get("algorithm", "unknown"))
        self.profile_path = str(Path(profile).resolve())
        self.profile = json.loads(Path(profile).read_text(encoding="utf-8"))
        self.max_agents = int(payload["max_agents"])
        self.max_candidates = int(payload["max_candidates"])
        self.normalizer = MigrationFeatureNormalizer.from_dict(payload["normalizer"])
        self.model = BatchCandidateQNetwork(
            int(payload["request_dim"]),
            int(payload["candidate_dim"]),
            int(payload["hidden_dim"]),
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.generator = MigrationCandidateGenerator(
            self.profile,
            max_candidates=self.max_candidates,
            cpu_capacity=cpu_capacity,
            memory_capacity=memory_capacity,
        )
        self.cpu_capacity = float(cpu_capacity)
        self.memory_capacity = float(memory_capacity)
        self.decoder_top_r = int(decoder_top_r)
        self.decoder_time_budget_ms = float(decoder_time_budget_ms)
        self.planned_batches = 0
        self.planned_tasks = 0
        self.selected_migrations = 0
        self.rejected_migrations = 0
        self.decision_times_ms: list[float] = []

    def _task(
        self,
        request: Mapping[str, Any],
        plan: Mapping[str, Any],
        stage: int,
        snapshot: ResourceSnapshot,
        decision_time: float,
        *,
        current_utilization: float | None = None,
        predicted_utilization: float | None = None,
        predicted_sla_risk: float | None = None,
    ) -> MigrationTask:
        request_id = int(request["id"])
        placement = plan["placement_by_vnf"][str(int(stage))]
        old_node = int(placement["dc_node"])
        cpu = float(placement.get("cpu_units", 1.0))
        memory = float(placement.get("memory_units", 1.0))
        cpu_remaining = float(snapshot.cpu_remaining.get(old_node, 0.0))
        mem_remaining = float(snapshot.memory_remaining.get(old_node, 0.0))
        snapshot_current = max(
            0.0,
            1.0 - cpu_remaining / max(1e-9, self.cpu_capacity),
            1.0 - mem_remaining / max(1e-9, self.memory_capacity),
        )
        current = min(1.5, max(0.0, float(
            current_utilization
            if current_utilization is not None
            else request.get("current_utilization", snapshot_current)
        )))
        predicted = min(1.5, max(0.0, float(
            predicted_utilization
            if predicted_utilization is not None
            else request.get("predicted_utilization", current)
        )))
        relief = max(cpu / self.cpu_capacity, memory / self.memory_capacity)
        remaining = max(
            0.0,
            float(request.get("leave_time", math.inf)) - float(decision_time),
        )
        online_planning = plan.get("online_planning") or {}
        raw_sla_risk = (
            predicted_sla_risk
            if predicted_sla_risk is not None
            else request.get(
                "predicted_sla_risk",
                online_planning.get("predicted_sla_risk", 0.0),
            )
        )
        sla_risk = min(1.0, max(0.0, float(raw_sla_risk)))
        state_size = 0.25 * (cpu + memory)
        migration_ms = 344.0 + 80.0 * state_size
        return MigrationTask(
            task_id=f"r{request_id}-s{int(stage)}",
            request_id=request_id,
            stage=int(stage),
            vnf_type=int(placement.get("vnf_type", stage)),
            old_node=old_node,
            cpu=cpu,
            memory=memory,
            bandwidth_mbps=float(request.get("bw_origin", 0.0)),
            state_size_mb=state_size,
            current_utilization=current,
            predicted_utilization=predicted,
            utilization_relief=relief,
            sla_risk=sla_risk,
            remaining_lifetime_s=remaining,
            estimated_migration_ms=migration_ms,
            priority=(1.0 + 2.0 * sla_risk) * relief / max(0.1, migration_ms / 1000.0),
            created_at=float(decision_time),
        )

    def plan_batch(
        self,
        entries: Sequence[Mapping[str, Any]],
        snapshot: ResourceSnapshot,
        decision_time: float,
    ) -> list[dict[str, Any]]:
        if not entries:
            return []
        if len(entries) > self.max_agents:
            raise ValueError(
                f"migration batch has {len(entries)} tasks; checkpoint maximum is {self.max_agents}"
            )
        started = time.perf_counter_ns()
        tasks = [
            self._task(
                entry["request"],
                entry["current_plan"],
                int(entry["stage"]),
                snapshot,
                decision_time,
                current_utilization=(
                    float(entry["current_utilization"])
                    if entry.get("current_utilization") is not None
                    else None
                ),
                predicted_utilization=(
                    float(entry["predicted_utilization"])
                    if entry.get("predicted_utilization") is not None
                    else None
                ),
                predicted_sla_risk=(
                    float(entry["predicted_sla_risk"])
                    if entry.get("predicted_sla_risk") is not None
                    else None
                ),
            )
            for entry in entries
        ]
        generated = [
            self.generator.generate(
                task,
                entry["current_plan"],
                snapshot,
                delay_bound_ms=float(entry["request"].get("delay_bound_ms", math.inf) or math.inf),
            )
            for task, entry in zip(tasks, entries)
        ]
        nullable = [list(row) + [None] for row in generated]
        features = migration_batch_candidate_features(nullable, snapshot)
        max_actions = self.max_candidates + 1
        request_tensor = np.zeros(
            (1, self.max_agents, len(self.normalizer.request_mean)), dtype=np.float32
        )
        candidate_tensor = np.zeros(
            (1, self.max_agents, max_actions, len(self.normalizer.candidate_mean)),
            dtype=np.float32,
        )
        action_mask = np.zeros((self.max_agents, max_actions), dtype=np.bool_)
        agent_mask = np.zeros((1, self.max_agents), dtype=np.bool_)
        footprints = []
        reject_actions = []
        for agent, (task, candidates, feature_rows) in enumerate(zip(tasks, generated, features)):
            agent_mask[0, agent] = True
            request_tensor[0, agent] = self.normalizer.normalize_request(
                migration_request_features(task)
            )
            mask = candidate_mask(list(candidates) + [None], snapshot)
            for action, values in enumerate(feature_rows):
                candidate_tensor[0, agent, action] = self.normalizer.normalize_candidate(values)
                action_mask[agent, action] = bool(mask[action])
            footprints.append([candidate.footprint for candidate in candidates] + [None])
            reject_actions.append(len(candidates))
        with torch.inference_mode():
            q_values = self.model(
                torch.from_numpy(request_tensor).to(self.device),
                torch.from_numpy(candidate_tensor).to(self.device),
                torch.from_numpy(agent_mask).to(self.device),
            )[0].cpu()
        rankings = ranked_actions(q_values, torch.from_numpy(action_mask))[: len(entries)]
        # A no-migration action is a real policy decision, not merely a
        # feasibility fallback.  Candidate actions ranked below noop are hidden
        # from the acceptance-first joint decoder.
        bounded_rankings = []
        for ranking, reject in zip(rankings, reject_actions):
            bounded = []
            for raw_action in ranking:
                action = int(raw_action)
                bounded.append(action)
                if action == reject:
                    break
            if reject not in bounded:
                bounded.append(reject)
            bounded_rankings.append(bounded)
        decode = decode_joint_candidates(
            footprints,
            snapshot,
            bounded_rankings,
            reject_actions=reject_actions,
            action_mask=action_mask[: len(entries)].tolist(),
            scores=q_values[: len(entries)].tolist(),
            priorities=[(-task.priority, task.task_id) for task in tasks],
            remaining_lifetimes_s=[task.remaining_lifetime_s for task in tasks],
            migration_prepare_s=[task.estimated_migration_ms / 1000.0 for task in tasks],
            top_r=self.decoder_top_r,
            time_budget_ms=self.decoder_time_budget_ms,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.planned_batches += 1
        self.planned_tasks += len(tasks)
        self.decision_times_ms.append(elapsed_ms)
        results = []
        for agent, (task, candidates, ranking, action) in enumerate(
            zip(tasks, generated, rankings, decode.actions)
        ):
            ranking_details = [
                {
                    "candidate_index": int(index),
                    "target_dc": (
                        int(candidates[index].target_node)
                        if index < len(candidates)
                        else None
                    ),
                    "q_value": float(q_values[agent, index]),
                    "feasible": bool(action_mask[agent, index]),
                }
                for index in ranking
            ]
            ranked_candidates = [
                {
                    **detail,
                    "candidate_id": (
                        candidates[index].candidate_id
                        if index < len(candidates)
                        else None
                    ),
                    "target_plan": (
                        candidates[index].plan
                        if index < len(candidates)
                        else None
                    ),
                }
                for detail, index in zip(ranking_details, ranking)
            ]
            if int(action) == len(candidates):
                self.rejected_migrations += 1
                selected = None
            else:
                self.selected_migrations += 1
                selected = candidates[int(action)]
            results.append({
                "task": task.to_dict(),
                "accepted": selected is not None,
                "selected_candidate_index": int(action),
                "selected_candidate": selected.to_dict() if selected else None,
                "target_dc": int(selected.target_node) if selected else None,
                "target_plan": selected.plan if selected else None,
                "rankings": ranking_details,
                "ranked_candidates": ranked_candidates,
                "decision_ms": elapsed_ms,
                "decoder": decode.to_dict(),
            })
        return results

    def metadata(self) -> dict[str, Any]:
        ordered = sorted(self.decision_times_ms)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else 0.0
        return {
            "checkpoint": self.checkpoint_path,
            "algorithm": self.algorithm,
            "profile": self.profile_path,
            "planned_batches": self.planned_batches,
            "planned_tasks": self.planned_tasks,
            "selected_migrations": self.selected_migrations,
            "rejected_migrations": self.rejected_migrations,
            "mean_decision_ms": sum(ordered) / max(1, len(ordered)),
            "p95_decision_ms": p95,
        }
