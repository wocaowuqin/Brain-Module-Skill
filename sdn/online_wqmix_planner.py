"""Low-latency online WQMIX selection over pre-generated complete SFC plans."""

from __future__ import annotations

from collections import Counter, deque
import copy
import heapq
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from core.marl.batch_deployment_wqmix import (
    AtomicResourceLedger,
    BatchCandidateQNetwork,
    ResourceFootprint,
    ResourceSnapshot,
    candidate_action_mask,
    deployment_candidate_features,
    ranked_actions,
)
from core.marl.deployment_dataset import FeatureNormalizer
from core.marl.deployment_topk import (
    CompletePlanCandidateGenerator,
    deserialize_footprint,
)
from core.marl.joint_candidate_decoder import (
    decode_joint_candidates,
    joint_footprints_feasible,
)
from core.marl.joint_candidate_transaction import commit_decoded_joint_actions
from core.marl.sla_risk_calibration import (
    EmpiricalSlaRiskCalibrator,
    SlaRiskEstimate,
)
from core.marl.sla_risk_predictor import RuntimeSlaRiskPredictor


class OnlineWQMIXPlanner:
    """Generate/select and atomically reserve complete plans after arrival.

    Offline Top-K rows are a warm cache.  When a request is absent from that
    cache, the same complete-plan generator can build candidates from the live
    resource snapshot before WQMIX ranking and exact versioned commit.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        data_folder: str | Path,
        profile: str | Path,
        microbatch_ms: float = 5.0,
        bandwidth_utilization_limit: float = 1.0,
        q0_sla_safety_margin: float = 0.0,
        q0_hard_sla_gate: bool = False,
        sla_queue_safety_factor: float = 1.0,
        sla_calibration: str | Path | None = None,
        sla_calibration_rank_weight: float = 0.25,
        sla_min_rerank_probability_delta: float = 0.05,
        sla_max_ood_score: float = 4.0,
        max_sla_failure_probability: float | None = None,
        hybrid_least_loaded: bool = False,
        hybrid_dataset_selected: bool = False,
        hybrid_rl_weight: float = 0.25,
        hybrid_conflict_only: bool = True,
        decoder_top_r: int = 4,
        decoder_time_budget_ms: float = 2.0,
        repair_missing_plans: bool = False,
        device: str = "cpu",
        torch_threads: int = 1,
    ) -> None:
        if microbatch_ms < 0.0:
            raise ValueError("microbatch_ms must be non-negative")
        if not 0.0 < bandwidth_utilization_limit <= 1.0:
            raise ValueError("bandwidth utilization limit must be in (0, 1]")
        if q0_sla_safety_margin < 0.0 or sla_queue_safety_factor < 0.0:
            raise ValueError("SLA safety parameters must be nonnegative")
        if sla_calibration_rank_weight < 0.0:
            raise ValueError("SLA calibration rank weight must be nonnegative")
        if not 0.0 <= sla_min_rerank_probability_delta <= 1.0:
            raise ValueError("SLA rerank probability delta must be in [0, 1]")
        if sla_max_ood_score <= 0.0:
            raise ValueError("SLA maximum OOD score must be positive")
        if max_sla_failure_probability is not None and not (
            0.0 <= max_sla_failure_probability <= 1.0
        ):
            raise ValueError("maximum SLA failure probability must be in [0, 1]")
        if max_sla_failure_probability is not None and sla_calibration is None:
            raise ValueError(
                "maximum SLA failure probability requires an SLA calibration file"
            )
        if not 0.0 <= hybrid_rl_weight <= 1.0:
            raise ValueError("hybrid RL weight must be in [0, 1]")
        if hybrid_least_loaded and hybrid_dataset_selected:
            raise ValueError(
                "hybrid least-loaded and dataset-selected modes are mutually exclusive"
            )
        if torch_threads < 0:
            raise ValueError("torch_threads must be non-negative")
        if decoder_top_r <= 0 or decoder_time_budget_ms < 0.0:
            raise ValueError("invalid joint decoder configuration")
        if torch_threads:
            torch.set_num_threads(int(torch_threads))

        self.checkpoint_path = Path(checkpoint).resolve()
        self.data_folder = Path(data_folder).resolve()
        self.profile_path = Path(profile).resolve()
        self.microbatch_ms = float(microbatch_ms)
        self.microbatch_seconds = self.microbatch_ms / 1000.0
        self.bandwidth_utilization_limit = float(bandwidth_utilization_limit)
        self.q0_sla_safety_margin = float(q0_sla_safety_margin)
        self.q0_hard_sla_gate = bool(q0_hard_sla_gate)
        self.sla_queue_safety_factor = float(sla_queue_safety_factor)
        self.sla_calibrator = None
        if sla_calibration is not None:
            sla_path = Path(sla_calibration)
            self.sla_calibrator = (
                EmpiricalSlaRiskCalibrator.from_path(sla_path)
                if sla_path.suffix.lower() == ".json"
                else RuntimeSlaRiskPredictor.from_path(sla_path)
            )
        self.sla_calibration_rank_weight = float(sla_calibration_rank_weight)
        self.sla_min_rerank_probability_delta = float(
            sla_min_rerank_probability_delta
        )
        self.sla_max_ood_score = float(sla_max_ood_score)
        self.max_sla_failure_probability = (
            float(max_sla_failure_probability)
            if max_sla_failure_probability is not None
            else None
        )
        self.hybrid_least_loaded = bool(hybrid_least_loaded)
        self.hybrid_dataset_selected = bool(hybrid_dataset_selected)
        self.hybrid_enabled = bool(hybrid_least_loaded or hybrid_dataset_selected)
        self.hybrid_rl_weight = float(hybrid_rl_weight)
        self.hybrid_conflict_only = bool(hybrid_conflict_only)
        self.decoder_top_r = int(decoder_top_r)
        self.decoder_time_budget_ms = float(decoder_time_budget_ms)
        self.repair_missing_plans = bool(repair_missing_plans)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {self.device}")
        self.recent_arrival_times: deque[float] = deque()
        self.current_arrival_rate_1s = 0.0

        initialized = time.perf_counter_ns()
        checkpoint_payload = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.normalizer = FeatureNormalizer.from_dict(checkpoint_payload["normalizer"])
        self.max_agents = int(checkpoint_payload["max_agents"])
        self.trained_max_candidates = int(checkpoint_payload["max_candidates"])
        self.max_candidates = self.trained_max_candidates
        self.request_dim = int(checkpoint_payload["request_dim"])
        self.candidate_dim = int(checkpoint_payload["candidate_dim"])
        self.model = BatchCandidateQNetwork(
            self.request_dim,
            self.candidate_dim,
            int(checkpoint_payload["hidden_dim"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint_payload["state_dict"])
        self.model.eval()

        spec = json.loads(
            (self.data_folder / "dataset_spec.json").read_text(encoding="utf-8")
        )
        profile_payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        resource_model = spec["resource_model"]
        dc_nodes = [int(node) for node in profile_payload["dc_nodes_1based"]]
        default_bandwidth = float(profile_payload.get("default_bandwidth_mbps", 0.0))
        default_delay = float(profile_payload.get("default_delay_ms", 1.0))
        bandwidth_capacity: dict[tuple[int, int], float] = {}
        edge_delay: dict[tuple[int, int], float] = {}
        for row in profile_payload["edges"]:
            capacity = float(row.get("bandwidth_mbps", default_bandwidth))
            capacity *= self.bandwidth_utilization_limit
            u, v = int(row["u"]), int(row["v"])
            bandwidth_capacity[(u, v)] = capacity
            bandwidth_capacity[(v, u)] = capacity
            delay = float(row.get("delay_ms", default_delay))
            edge_delay[(u, v)] = delay
            edge_delay[(v, u)] = delay
        self.bandwidth_capacity = bandwidth_capacity
        self.edge_delay = edge_delay
        self.ledger = AtomicResourceLedger(
            {node: float(resource_model["cpu_per_dc"]) for node in dc_nodes},
            {node: float(resource_model["memory_per_dc"]) for node in dc_nodes},
            bandwidth_capacity,
        )
        objective_weights = resource_model.get("candidate_objective_weights") or {}
        self.repair_generator = (
            CompletePlanCandidateGenerator(
                profile_payload,
                max_candidates=min(max(1, self.trained_max_candidates), 8),
                placement_beam=8,
                placement_chains=6,
                pool_limit=12,
                objective_delay_weight=float(objective_weights.get("delay", 1.0)),
                objective_cpu_weight=float(objective_weights.get("cpu", 0.0)),
                objective_memory_weight=float(objective_weights.get("memory", 0.0)),
                objective_bandwidth_weight=float(objective_weights.get("bandwidth", 0.03)),
                objective_pressure_weight=float(objective_weights.get("pressure", 8.0)),
            )
            if self.repair_missing_plans
            else None
        )

        batches_path = self.data_folder / "batches.jsonl"
        self.agent_by_request: dict[int, dict[str, Any]] = {}
        self.dataset_batch_by_request: dict[int, int] = {}
        dataset_max_candidates = 0
        for line in batches_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            batch = json.loads(line)
            for agent in batch["agents"]:
                request_id = int(agent["request_id"])
                if request_id in self.agent_by_request:
                    raise ValueError(f"duplicate request {request_id} in {batches_path}")
                candidates = list(agent.get("candidates") or [])
                if not candidates:
                    raise ValueError(
                        f"request {request_id} has no candidates in {batches_path}"
                    )
                if int(agent["reject_action"]) >= len(candidates):
                    raise ValueError(
                        f"request {request_id} has an invalid reject action"
                    )
                for candidate in candidates:
                    if len(candidate.get("candidate_features") or []) != self.candidate_dim:
                        raise ValueError(
                            f"request {request_id} candidate feature dimension does not "
                            f"match checkpoint ({self.candidate_dim})"
                        )
                dataset_max_candidates = max(dataset_max_candidates, len(candidates))
                self.agent_by_request[request_id] = agent
                self.dataset_batch_by_request[request_id] = int(batch["batch_id"])
        # CandidateQNetwork scores candidates with shared parameters and has no
        # positional action embedding, so a checkpoint trained with Top-K can
        # safely rank a larger compatible candidate set at inference time.
        self.max_candidates = max(self.max_candidates, dataset_max_candidates)

        self.planned_requests = 0
        self.planned_batches = 0
        self.accepted_requests = 0
        self.rejected_requests = 0
        self.fallback_requests = 0
        self.release_count = 0
        self.missing_candidate_requests = 0
        self.rejection_reasons: Counter[str] = Counter()
        self.rejection_diagnostic_categories: Counter[str] = Counter()
        self.sla_guarded_requests = 0
        self.sla_guard_filtered_candidates = 0
        self.sla_guard_unsafe_candidates = 0
        self.sla_guard_rejected_requests = 0
        self.sla_probability_guarded_requests = 0
        self.sla_probability_rejected_requests = 0
        self.sla_predictor_candidates_evaluated = 0
        self.sla_predictor_ood_skipped_candidates = 0
        self.sla_predictor_probability_skipped_candidates = 0
        self.sla_predictor_resource_safety_skipped_candidates = 0
        self.sla_predictor_promoted_candidates = 0
        self.hybrid_heuristic_batches = 0
        self.hybrid_heuristic_requests = 0
        self.hybrid_rl_batches = 0
        self.hybrid_rl_requests = 0
        self.hybrid_rl_applied_batches = 0
        self.hybrid_rl_applied_requests = 0
        self.batch_timings_ms: list[float] = []
        self.decoder_timings_ms: list[float] = []
        self.decoder_timeouts = 0
        self.decoder_greedy_budget_exhaustions = 0
        self.decoder_repair_budget_exhaustions = 0
        self.decoder_adjusted_requests = 0
        self.commit_version_retries = 0
        self.online_repair_attempts = 0
        self.online_repair_candidates = 0
        self.online_repair_selected = 0
        self.online_repair_generation_ms: list[float] = []
        self.online_repair_attempted_ids: set[int] = set()
        self.leave_heap: list[tuple[float, int]] = []
        if self.repair_generator is not None:
            self.repair_generator.prewarm_paths()
        self.initialization_ms = (time.perf_counter_ns() - initialized) / 1_000_000.0

    def _repair_agent_without_plan(
        self,
        request: Mapping[str, Any],
        agent: Mapping[str, Any],
        snapshot: ResourceSnapshot,
    ) -> dict[str, Any]:
        """Generate one current-snapshot plan for an offline reject-only row."""

        if self.repair_generator is None or any(
            candidate.get("plan") is not None for candidate in agent["candidates"]
        ):
            return dict(agent)
        request_id = int(request["id"])
        if request_id in self.online_repair_attempted_ids:
            return dict(agent)
        self.online_repair_attempted_ids.add(request_id)
        self.online_repair_attempts += 1
        started_ns = time.perf_counter_ns()
        try:
            generated = [
                candidate
                for candidate in [
                    self.repair_generator.generate_first_feasible(request, snapshot)
                ]
                if candidate is not None
            ]
        except (KeyError, TypeError, ValueError):
            generated = []
        self.online_repair_generation_ms.append(
            (time.perf_counter_ns() - started_ns) / 1_000_000.0
        )
        if not generated:
            return dict(agent)

        candidate = generated[0]
        candidate.source = f"online_repair_{candidate.source}"
        candidate.plan.setdefault("candidate_generation", {})["online_repair"] = True
        static_tail = [
            candidate.metrics["estimated_delay_ms"],
            candidate.metrics["delay_bound_ms"],
            candidate.metrics["segment_hops"],
            candidate.metrics["tree_edges"],
            candidate.metrics["flowmod_estimate"],
            candidate.metrics["peak_resource_pressure"],
            candidate.objective,
            0.0,
        ]
        plan_row = candidate.to_dict(
            [0.0] * 16 + static_tail,
            candidate.metrics["estimated_delay_ms"]
            <= candidate.metrics["delay_bound_ms"] + 1e-9,
        )
        reject_row = next(
            (
                copy.deepcopy(row)
                for row in agent["candidates"]
                if row.get("plan") is None
            ),
            None,
        )
        if reject_row is None:
            reject_row = {
                "candidate_id": f"r{int(request['id'])}-reject",
                "source": "reject",
                "action_valid": True,
                "objective": 1_000_000.0,
                "metrics": {},
                "resource_footprint": None,
                "candidate_features": [0.0] * self.candidate_dim,
                "plan": None,
            }
        repaired = copy.deepcopy(dict(agent))
        repaired["candidate_count"] = 1
        repaired["reject_action"] = 1
        repaired["valid_actions"] = [0, 1]
        repaired["commit_ranking"] = [0, 1]
        repaired["candidates"] = [plan_row, reject_row]
        self.online_repair_candidates += 1
        return repaired

    def _generate_online_agent(
        self,
        request: Mapping[str, Any],
        snapshot: ResourceSnapshot,
    ) -> dict[str, Any] | None:
        """Build a WQMIX agent row from the current live snapshot.

        The generated rows deliberately use the same 16 dynamic footprint
        features plus the eight metric features used by deployment_topk_v3;
        the dynamic prefix is refreshed below by ``deployment_candidate_features``.
        """

        if self.repair_generator is None:
            return None
        request_id = int(request["id"])
        if request_id in self.online_repair_attempted_ids:
            return None
        self.online_repair_attempted_ids.add(request_id)
        self.online_repair_attempts += 1
        started_ns = time.perf_counter_ns()
        try:
            # The live miss path is latency-critical.  Generate one hard-
            # feasible complete plan now; richer Top-K expansion belongs in
            # the offline/background cache and must not block deployment.
            first = self.repair_generator.generate_first_feasible(request, snapshot)
            generated = [first] if first is not None else []
        except (KeyError, TypeError, ValueError):
            generated = []
        self.online_repair_generation_ms.append(
            (time.perf_counter_ns() - started_ns) / 1_000_000.0
        )
        if not generated:
            return None

        candidate_rows = []
        for candidate in generated:
            candidate.source = f"online_repair_{candidate.source}"
            candidate.plan.setdefault("candidate_generation", {})[
                "online_repair"
            ] = True
            metrics = candidate.metrics
            static_tail = [
                metrics["estimated_delay_ms"],
                metrics["delay_bound_ms"],
                metrics["segment_hops"],
                metrics["tree_edges"],
                metrics["flowmod_estimate"],
                metrics["peak_resource_pressure"],
                candidate.objective,
                0.0,
            ]
            candidate_rows.append(
                candidate.to_dict(
                    [0.0] * 16 + static_tail,
                    metrics["estimated_delay_ms"]
                    <= metrics["delay_bound_ms"] + 1e-9,
                )
            )

        reject_index = len(candidate_rows)
        candidate_rows.append({
            "candidate_id": f"r{request_id}-reject",
            "source": "reject",
            "action_valid": True,
            "objective": 1_000_000.0,
            "metrics": {},
            "resource_footprint": None,
            "candidate_features": [0.0] * self.candidate_dim,
            "plan": None,
        })
        self.online_repair_candidates += len(generated)
        return {
            "request_id": request_id,
            "candidate_count": len(generated),
            "reject_action": reject_index,
            "valid_actions": list(range(reject_index + 1)),
            "commit_ranking": list(range(reject_index + 1)),
            "candidates": candidate_rows,
        }

    def _least_loaded_rankings(
        self,
        footprints: Sequence[Sequence[Any]],
        snapshot: Any,
        action_mask: np.ndarray,
    ) -> list[list[int]]:
        """Rank feasible actions by projected directed-link utilization."""

        rankings: list[list[int]] = []
        for agent_index, request_footprints in enumerate(footprints):
            scored: list[tuple[tuple[float, float, int], int]] = []
            reject_action: int | None = None
            for action, footprint in enumerate(request_footprints):
                if not bool(action_mask[agent_index, action]):
                    continue
                if footprint is None:
                    reject_action = action
                    continue
                projected = []
                for edge, amount in footprint.bandwidth.items():
                    capacity = float(self.bandwidth_capacity[edge])
                    used = capacity - float(snapshot.bandwidth_remaining[edge])
                    projected.append((used + float(amount)) / max(capacity, 1e-9))
                scored.append(
                    ((max(projected, default=0.0), sum(projected), action), action)
                )
            ranking = [action for _, action in sorted(scored)]
            if reject_action is not None:
                ranking.append(reject_action)
            rankings.append(ranking)
        return rankings

    @staticmethod
    def _dataset_selected_rankings(
        agents: Sequence[Mapping[str, Any]],
        least_loaded_rankings: Sequence[Sequence[int]],
        action_mask: np.ndarray,
    ) -> list[list[int]]:
        """Keep the validated dataset action first while it remains feasible."""

        rankings: list[list[int]] = []
        for agent_index, (agent, least_loaded) in enumerate(
            zip(agents, least_loaded_rankings)
        ):
            ranking = list(map(int, least_loaded))
            selected = int(agent.get("selected_action", -1))
            selected_is_plan = (
                0 <= selected < len(agent["candidates"])
                and agent["candidates"][selected].get("plan") is not None
            )
            selected_is_feasible = (
                selected_is_plan
                and selected < action_mask.shape[1]
                and bool(action_mask[agent_index, selected])
            )
            if selected_is_feasible:
                ranking = [selected] + [
                    action for action in ranking if action != selected
                ]
            rankings.append(ranking)
        return rankings

    def _hybrid_rankings(
        self,
        learned: Sequence[Sequence[int]],
        heuristic: Sequence[Sequence[int]],
        footprints: Sequence[Sequence[Any]],
        agents: Sequence[Mapping[str, Any]],
        snapshot: Any,
    ) -> tuple[list[list[int]], bool]:
        """Fuse rankings only when the joint action improves the safety baseline."""

        weight = self.hybrid_rl_weight
        fused: list[list[int]] = []
        for learned_order, heuristic_order in zip(learned, heuristic):
            learned_position = {
                int(action): index for index, action in enumerate(learned_order)
            }
            heuristic_position = {
                int(action): index for index, action in enumerate(heuristic_order)
            }
            actions = list(dict.fromkeys(map(int, heuristic_order)))
            denominator = float(max(1, len(actions) - 1))
            actions.sort(
                key=lambda action: (
                    (1.0 - weight)
                    * heuristic_position.get(action, len(actions))
                    / denominator
                    + weight
                    * learned_position.get(action, len(actions))
                    / denominator,
                    heuristic_position.get(action, len(actions)),
                    action,
                )
            )
            fused.append(actions)

        heuristic_actions = [order[0] for order in heuristic if order]
        fused_actions = [order[0] for order in fused if order]
        if len(heuristic_actions) != len(heuristic) or len(fused_actions) != len(fused):
            return [list(order) for order in heuristic], False

        def joint_link_score(actions: Sequence[int]) -> tuple[float, float]:
            added: Counter[tuple[int, int]] = Counter()
            for request_footprints, action in zip(footprints, actions):
                footprint = request_footprints[int(action)]
                if footprint is None:
                    return float("inf"), float("inf")
                added.update(footprint.bandwidth)
            projected = []
            for edge, capacity in self.bandwidth_capacity.items():
                used = float(capacity) - float(snapshot.bandwidth_remaining[edge])
                projected.append(
                    (used + float(added[edge])) / max(float(capacity), 1e-9)
                )
            return max(projected, default=0.0), sum(projected)

        def joint_sla_risk(actions: Sequence[int]) -> tuple[float, float]:
            risks = []
            for agent, action in zip(agents, actions):
                metrics = agent["candidates"][int(action)].get("metrics") or {}
                bound = max(1e-9, float(metrics.get("delay_bound_ms", 0.0)))
                risks.append(float(metrics.get("estimated_delay_ms", 0.0)) / bound)
            return max(risks, default=0.0), sum(risks)

        heuristic_link = joint_link_score(heuristic_actions)
        fused_link = joint_link_score(fused_actions)
        heuristic_sla = joint_sla_risk(heuristic_actions)
        fused_sla = joint_sla_risk(fused_actions)
        improves_link_load = (
            fused_link[0] <= heuristic_link[0] + 1e-9
            and fused_link[1] <= heuristic_link[1] + 1e-9
            and (
                fused_link[0] < heuristic_link[0] - 1e-9
                or fused_link[1] < heuristic_link[1] - 1e-9
            )
        )
        preserves_sla = (
            fused_sla[0] <= heuristic_sla[0] + 1e-9
            and fused_sla[1] <= heuristic_sla[1] + 1e-9
        )
        if improves_link_load and preserves_sla:
            return fused, True
        return [list(order) for order in heuristic], False

    @staticmethod
    def _request_features(request: Mapping[str, Any], decision_time: float) -> list[float]:
        return [
            float(request["bw_origin"]),
            float(sum(map(float, request["cpu_origin"]))),
            float(sum(map(float, request["memory_origin"]))),
            float(len(request["destination_dpids"])),
            float(len(request["vnf"])),
            float(request["lifetime"]),
            max(0.0, float(request["leave_time"]) - decision_time),
            float(request.get("delay_bound_ms") or 0.0),
            float(request.get("jitter_bound_ms") or 0.0),
            float(request.get("packet_loss_bound") or 0.0),
            float(request.get("priority") or 0.0),
            float(request.get("dscp") or 0.0),
        ]

    def _reject_plan(
        self,
        request: Mapping[str, Any],
        reason: str,
        timing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.rejection_reasons[str(reason)] += 1
        return {
            "version": "wqmix_sfc_plan_v1",
            "request_id": int(request["id"]),
            "accepted": False,
            "reason": str(reason),
            "online_planning": dict(timing or {}),
        }

    @staticmethod
    def _candidate_feasibility_diagnostics(
        agent: Mapping[str, Any],
        footprints: Sequence[ResourceFootprint | None],
        snapshot: ResourceSnapshot,
    ) -> dict[str, Any]:
        reason_counts: Counter[str] = Counter()
        feasible_indices: list[int] = []
        plan_candidates = 0
        peaks: dict[str, tuple[float, str | int | None]] = {
            "cpu": (0.0, None),
            "memory": (0.0, None),
            "bandwidth": (0.0, None),
        }
        for candidate_index, (candidate, footprint) in enumerate(
            zip(agent["candidates"], footprints)
        ):
            if footprint is None or candidate.get("plan") is None:
                continue
            plan_candidates += 1
            reasons: set[str] = set()
            for name, demand, remaining in (
                ("cpu", footprint.cpu, snapshot.cpu_remaining),
                ("memory", footprint.memory, snapshot.memory_remaining),
                ("bandwidth", footprint.bandwidth, snapshot.bandwidth_remaining),
            ):
                for resource, amount in demand.items():
                    available = float(remaining.get(resource, 0.0))
                    ratio = float(amount) / max(available, 1e-9)
                    if ratio > peaks[name][0]:
                        label = (
                            f"{int(resource[0])}->{int(resource[1])}"
                            if isinstance(resource, tuple)
                            else int(resource)
                        )
                        peaks[name] = (ratio, label)
                    if float(amount) > available + 1e-9:
                        reasons.add(f"{name}_insufficient")
            metrics = candidate.get("metrics") or {}
            if float(metrics.get("estimated_delay_ms", 0.0)) > (
                float(metrics.get("delay_bound_ms", 0.0)) + 1e-9
            ):
                reasons.add("delay_bound_exceeded")
            if reasons:
                reason_counts.update(reasons)
            else:
                feasible_indices.append(candidate_index)
        return {
            "plan_candidates": plan_candidates,
            "locally_feasible_candidates": len(feasible_indices),
            "locally_feasible_indices": feasible_indices,
            "candidate_failure_counts": dict(sorted(reason_counts.items())),
            "peak_cpu_demand_to_remaining": peaks["cpu"][0],
            "peak_cpu_node": peaks["cpu"][1],
            "peak_memory_demand_to_remaining": peaks["memory"][0],
            "peak_memory_node": peaks["memory"][1],
            "peak_bandwidth_demand_to_remaining": peaks["bandwidth"][0],
            "peak_bandwidth_edge": peaks["bandwidth"][1],
        }

    def _modeled_candidate_delay(
        self,
        candidate: Mapping[str, Any],
        footprint: Any,
        snapshot: Any,
    ) -> float:
        plan = candidate.get("plan") or {}

        def edge_delay(u: int, v: int) -> float:
            edge = (int(u), int(v))
            capacity = float(self.bandwidth_capacity[edge])
            current_used = capacity - float(snapshot.bandwidth_remaining[edge])
            used_after = current_used + float(footprint.bandwidth.get(edge, 0.0))
            utilization = min(0.95, max(0.0, used_after / max(capacity, 1e-6)))
            return float(self.edge_delay[edge]) * (
                1.0
                + self.sla_queue_safety_factor
                * utilization
                / max(1e-6, 1.0 - utilization)
            )

        def path_delay(path: Sequence[Any]) -> float:
            nodes = [int(node) for node in path]
            return sum(edge_delay(u, v) for u, v in zip(nodes, nodes[1:]))

        segment_delay = sum(
            path_delay(segment.get("path", []))
            for segment in plan.get("segments", [])
        )
        branch_delay = max(
            (
                path_delay(path)
                for path in ((plan.get("multicast") or {}).get("paths") or {}).values()
            ),
            default=0.0,
        )
        return float(segment_delay + branch_delay)

    def _predict_candidate_sla_risk(
        self,
        request: Mapping[str, Any],
        candidate: Mapping[str, Any],
        footprint: ResourceFootprint,
        snapshot: ResourceSnapshot,
        modeled_delay_ratio: float,
        batch_size: int,
    ) -> SlaRiskEstimate:
        if isinstance(self.sla_calibrator, RuntimeSlaRiskPredictor):
            return self.sla_calibrator.predict(
                request,
                candidate,
                footprint,
                snapshot,
                self.bandwidth_capacity,
                modeled_delay_ratio=modeled_delay_ratio,
                batch_size=batch_size,
                active_request_count=len(self.ledger.allocations),
                recent_arrival_rate_1s=self.current_arrival_rate_1s,
            )
        if isinstance(self.sla_calibrator, EmpiricalSlaRiskCalibrator):
            return self.sla_calibrator.predict(modeled_delay_ratio, request)
        raise RuntimeError("SLA risk predictor is not configured")

    def _apply_q0_sla_guard(
        self,
        requests: Sequence[Mapping[str, Any]],
        agents: Sequence[Mapping[str, Any]],
        footprints: Sequence[Sequence[Any]],
        snapshot: Any,
        rankings: list[list[int]],
    ) -> tuple[
        list[list[int]],
        list[dict[int, float]],
        list[dict[int, SlaRiskEstimate]],
    ]:
        risks_by_agent: list[dict[int, float]] = []
        for request, agent, request_footprints, ranking in zip(
            requests, agents, footprints, rankings
        ):
            delay_bound = max(1e-6, float(request.get("delay_bound_ms") or 0.0))
            risks: dict[int, float] = {}
            for action in ranking:
                candidate = agent["candidates"][int(action)]
                footprint = request_footprints[int(action)]
                if candidate.get("plan") is None or footprint is None:
                    continue
                risks[int(action)] = (
                    self._modeled_candidate_delay(candidate, footprint, snapshot)
                    / delay_bound
                )
            risks_by_agent.append(risks)

        estimates_by_agent: list[dict[int, SlaRiskEstimate]] = [
            {} for _ in requests
        ]
        if isinstance(self.sla_calibrator, RuntimeSlaRiskPredictor):
            flat_rows = [
                (agent_index, action, request, agent["candidates"][action],
                 request_footprints[action], risks[action])
                for agent_index, (request, agent, request_footprints, risks) in enumerate(
                    zip(requests, agents, footprints, risks_by_agent)
                )
                for action in risks
            ]
            predictions = self.sla_calibrator.predict_many(
                [row[2] for row in flat_rows],
                [row[3] for row in flat_rows],
                [row[4] for row in flat_rows],
                snapshot,
                self.bandwidth_capacity,
                modeled_delay_ratios=[row[5] for row in flat_rows],
                batch_size=len(requests),
                active_request_count=len(self.ledger.allocations),
                recent_arrival_rate_1s=self.current_arrival_rate_1s,
            )
            for row, estimate in zip(flat_rows, predictions):
                estimates_by_agent[row[0]][row[1]] = estimate
        elif self.sla_calibrator is not None:
            for agent_index, (request, agent, request_footprints, risks) in enumerate(
                zip(requests, agents, footprints, risks_by_agent)
            ):
                estimates_by_agent[agent_index] = {
                    action: self._predict_candidate_sla_risk(
                        request,
                        agent["candidates"][int(action)],
                        request_footprints[int(action)],
                        snapshot,
                        risk,
                        len(requests),
                    )
                    for action, risk in risks.items()
                }

        for agent_index, (request, ranking, risks, estimates, request_footprints) in enumerate(
            zip(requests, rankings, risks_by_agent, estimates_by_agent, footprints)
        ):
            self.sla_predictor_candidates_evaluated += len(estimates)

            if estimates and self.sla_calibration_rank_weight > 0.0:
                original_positions = {
                    int(action): position for position, action in enumerate(ranking)
                }
                denominator = max(1, len(ranking) - 1)
                reference = estimates.get(int(ranking[0])) if ranking else None
                if reference is not None and reference.ood_score <= self.sla_max_ood_score:
                    reference_probability = reference.failure_probability
                    reference_action = int(ranking[0])
                    reference_risk = risks.get(reference_action, float("inf"))

                    def projected_link_load(action: int) -> tuple[float, float]:
                        footprint = request_footprints[int(action)]
                        if footprint is None:
                            return float("inf"), float("inf")
                        values = []
                        for edge, amount in footprint.bandwidth.items():
                            capacity = float(self.bandwidth_capacity[edge])
                            used = capacity - float(snapshot.bandwidth_remaining[edge])
                            values.append(
                                (used + float(amount)) / max(capacity, 1e-9)
                            )
                        return max(values, default=0.0), sum(values)

                    reference_load = projected_link_load(reference_action)

                    guarded_probabilities: dict[int, float] = {}
                    for action_value in ranking:
                        action = int(action_value)
                        estimate = estimates.get(int(action))
                        candidate_load = projected_link_load(int(action))
                        if action == reference_action:
                            guarded_probabilities[action] = reference_probability
                        elif estimate is None:
                            guarded_probabilities[action] = reference_probability
                        elif estimate.ood_score > self.sla_max_ood_score:
                            self.sla_predictor_ood_skipped_candidates += 1
                            guarded_probabilities[action] = reference_probability
                        elif (
                            reference_probability - estimate.failure_probability
                            < self.sla_min_rerank_probability_delta
                        ):
                            self.sla_predictor_probability_skipped_candidates += 1
                            guarded_probabilities[action] = reference_probability
                        elif (
                            risks.get(action, float("inf")) > reference_risk + 1e-9
                            or candidate_load[0] > reference_load[0] + 1e-9
                            or candidate_load[1] > reference_load[1] + 1e-9
                        ):
                            self.sla_predictor_resource_safety_skipped_candidates += 1
                            guarded_probabilities[action] = reference_probability
                        else:
                            guarded_probabilities[action] = estimate.failure_probability

                    previous_first = int(ranking[0])
                    ranking = sorted(
                        ranking,
                        key=lambda action: (
                            original_positions[int(action)] / denominator
                            + self.sla_calibration_rank_weight
                            * guarded_probabilities[int(action)],
                            original_positions[int(action)],
                        ),
                    )
                    rankings[agent_index] = ranking
                    self.sla_probability_guarded_requests += 1
                    self.sla_predictor_promoted_candidates += int(
                        int(ranking[0]) != previous_first
                    )
            if int(request.get("priority", 0)) != 3 or (
                self.q0_sla_safety_margin <= 0.0 and not self.q0_hard_sla_gate
            ):
                continue
            threshold = self.q0_sla_safety_margin or 1.0
            safe = {
                action
                for action, risk in risks.items()
                if risk <= threshold
            }
            if self.q0_hard_sla_gate:
                # Preserve the policy/heuristic ordering.  The hard gate is
                # applied to the jointly decoded action below, so enabling it
                # cannot redirect many Q0 requests onto different paths and
                # create a new congestion pattern.
                guarded = list(ranking)
                filtered = 0
            elif safe:
                guarded = [action for action in ranking if action in safe]
                guarded.extend(action for action in ranking if action not in safe)
                filtered = sum(action in risks and action not in safe for action in ranking)
            elif risks:
                guarded = sorted(
                    (action for action in ranking if action in risks),
                    key=lambda action: (risks[action], ranking.index(action)),
                )
                guarded.extend(action for action in ranking if action not in risks)
                filtered = max(0, len(risks) - 1)
            else:
                continue
            rankings[agent_index] = guarded
            self.sla_guarded_requests += 1
            self.sla_guard_filtered_candidates += filtered
            self.sla_guard_unsafe_candidates += sum(
                action in risks and action not in safe for action in ranking
            )
        return rankings, risks_by_agent, estimates_by_agent

    def _enforce_q0_hard_sla_gate(
        self,
        requests: Sequence[Mapping[str, Any]],
        agents: Sequence[Mapping[str, Any]],
        actions: Sequence[int],
        risks_by_agent: Sequence[Mapping[int, float]],
        estimates_by_agent: Sequence[Mapping[int, SlaRiskEstimate]],
    ) -> tuple[tuple[int, ...], dict[int, dict[str, Any]]]:
        """Reject an unsafe decoded Q0 action without changing other plans."""

        selected = [int(action) for action in actions]
        rejected: dict[int, dict[str, Any]] = {}
        if (
            not self.q0_hard_sla_gate
            and self.max_sla_failure_probability is None
        ):
            return tuple(selected), rejected
        threshold = self.q0_sla_safety_margin or 1.0
        for index, (request, agent, risks, estimates) in enumerate(
            zip(requests, agents, risks_by_agent, estimates_by_agent)
        ):
            action = selected[index]
            reject_action = int(agent["reject_action"])
            if action == reject_action:
                continue
            risk = risks.get(action)
            estimate = estimates.get(action)
            q0_unsafe = (
                self.q0_hard_sla_gate
                and int(request.get("priority", 0)) == 3
                and (risk is None or float(risk) > threshold)
            )
            probability_unsafe = (
                self.max_sla_failure_probability is not None
                and (
                    estimate is not None
                    and estimate.ood_score <= self.sla_max_ood_score
                    and estimate.failure_probability > self.max_sla_failure_probability
                )
            )
            if not q0_unsafe and not probability_unsafe:
                continue
            reason = (
                "q0_delay_and_sla_probability_gate"
                if q0_unsafe and probability_unsafe
                else "q0_hard_sla_gate"
                if q0_unsafe
                else "sla_failure_probability_gate"
            )
            rejected[index] = {
                "candidate_index": action,
                "modeled_delay_ratio": (
                    float(risk) if risk is not None else float("inf")
                ),
                "failure_probability": (
                    float(estimate.failure_probability)
                    if estimate is not None
                    else 1.0
                ),
                "delay_ratio_bucket": (
                    estimate.delay_ratio_bucket if estimate is not None else None
                ),
                "effective_support": (
                    float(estimate.effective_support)
                    if estimate is not None
                    else 0.0
                ),
                "model_kind": estimate.model_kind if estimate is not None else None,
                "confidence": (
                    float(estimate.confidence) if estimate is not None else 0.0
                ),
                "ood_score": (
                    float(estimate.ood_score) if estimate is not None else float("inf")
                ),
                "reason": reason,
            }
            self.sla_probability_rejected_requests += int(probability_unsafe)
            selected[index] = reject_action
        return tuple(selected), rejected

    def plan_next(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.plan_batch([request])[0]

    def rank_migration_targets(
        self,
        request: Mapping[str, Any],
        current_plan: Mapping[str, Any],
        stage: int,
        candidate_plans: Sequence[Mapping[str, Any]],
        decision_time: float,
    ) -> list[dict[str, Any]]:
        """Score online VNF migration targets with the shared WQMIX agent.

        Migration candidates are complete plans with one stage moved. They are
        scored by the same shared candidate Q network used for deployment; the
        hard migration executor still validates paths, endpoints and bandwidth.
        """
        if not candidate_plans:
            return []
        # The checkpoint has a fixed action tensor width.  Keep a deterministic
        # bounded target set for online migration; target generation is already
        # filtered by DC eligibility and the executor performs final admission.
        candidate_plans = list(candidate_plans)[: self.max_candidates]
        snapshot = self.ledger.snapshot()
        footprints = [
            ResourceFootprint.from_sfc_plan(
                plan, float(request.get("bw_origin", 0.0)), snapshot
            )
            for plan in candidate_plans
        ]
        live = deployment_candidate_features([footprints], snapshot)[0]
        candidates = []
        for plan, footprint, live_row in zip(candidate_plans, footprints, live):
            paths = [segment.get("path", []) for segment in plan.get("segments", [])]
            tree = (plan.get("multicast") or {}).get("tree_edges") or []
            hops = sum(max(0, len(path) - 1) for path in paths)
            tree_edges = len(tree)
            estimated_delay = float(sum(
                self.edge_delay.get((int(u), int(v)), 1.0)
                for path in paths
                for u, v in zip(path, path[1:])
            ))
            bound = float(request.get("delay_bound_ms") or 0.0)
            static_tail = [
                estimated_delay, bound, float(hops), float(tree_edges),
                max(live_row[12:15], default=0.0), 0.0, 0.0, 0.0,
            ]
            candidates.append(list(live_row[:16]) + static_tail)
        candidate_tensor = np.zeros(
            (1, self.max_agents, self.max_candidates, self.candidate_dim),
            dtype=np.float32,
        )
        request_obs = np.zeros(
            (1, self.max_agents, self.request_dim), dtype=np.float32
        )
        agent_mask = np.zeros((1, self.max_agents), dtype=np.bool_)
        action_mask = np.zeros((self.max_agents, self.max_candidates), dtype=np.bool_)
        request_obs[0, 0] = self.normalizer.normalize_request(
            self._request_features(request, decision_time)
        )
        agent_mask[0, 0] = True
        for index, values in enumerate(candidates):
            candidate_tensor[0, 0, index] = self.normalizer.normalize_candidate(values)
            action_mask[0, index] = True
        with torch.inference_mode():
            q_values = self.model(
                torch.from_numpy(request_obs).to(self.device),
                torch.from_numpy(candidate_tensor).to(self.device),
                torch.from_numpy(agent_mask).to(self.device),
            )[0, 0, : len(candidates)].detach().cpu().tolist()
        return [
            {"candidate_index": int(index), "target_plan": candidate_plans[index],
             "q_value": float(q_values[index])}
            for index in sorted(range(len(candidates)), key=lambda i: q_values[i], reverse=True)
        ]

    def plan_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        if len(requests) > self.max_agents:
            raise ValueError(
                f"online micro-batch has {len(requests)} requests; max is {self.max_agents}"
            )
        started_ns = time.perf_counter_ns()
        decision_time = max(float(request["arrival_time"]) for request in requests)
        for request in requests:
            self.recent_arrival_times.append(float(request["arrival_time"]))
        while (
            self.recent_arrival_times
            and self.recent_arrival_times[0] < decision_time - 1.0
        ):
            self.recent_arrival_times.popleft()
        self.current_arrival_rate_1s = float(len(self.recent_arrival_times))
        while self.leave_heap and self.leave_heap[0][0] <= decision_time + 1e-12:
            _, expired_request_id = heapq.heappop(self.leave_heap)
            self.release(expired_request_id)
        request_ids = [int(request["id"]) for request in requests]
        agents: list[dict[str, Any] | None] = [
            self.agent_by_request.get(request_id) for request_id in request_ids
        ]
        missing = [
            request_id for request_id, agent in zip(request_ids, agents) if agent is None
        ]
        if missing:
            self.missing_candidate_requests += len(missing)

        present_indices = [index for index, agent in enumerate(agents) if agent is not None]
        snapshot = self.ledger.snapshot()
        if self.repair_generator is not None:
            for index, agent in enumerate(agents):
                if agent is None:
                    agents[index] = self._generate_online_agent(
                        requests[index], snapshot
                    )
            present_indices = [index for index, agent in enumerate(agents) if agent is not None]
        if not present_indices:
            self.planned_requests += len(requests)
            self.planned_batches += 1
            self.rejected_requests += len(requests)
            return [self._reject_plan(request, "missing_topk_candidates") for request in requests]

        present_agents = [agents[index] for index in present_indices]
        assert all(agent is not None for agent in present_agents)
        present_agents = [agent for agent in present_agents if agent is not None]
        present_agents = [
            self._repair_agent_without_plan(requests[request_index], agent, snapshot)
            for request_index, agent in zip(present_indices, present_agents)
        ]
        active_agent_by_request = {
            request_ids[request_index]: agent
            for request_index, agent in zip(present_indices, present_agents)
        }
        footprints = [
            [
                None
                if candidate.get("resource_footprint") is None
                else deserialize_footprint(candidate["resource_footprint"])
                for candidate in agent["candidates"]
            ]
            for agent in present_agents
        ]
        live_features = deployment_candidate_features(footprints, snapshot)
        live_masks = candidate_action_mask(footprints, snapshot)

        request_obs = np.zeros(
            (1, self.max_agents, self.request_dim), dtype=np.float32
        )
        candidate_tensor = np.zeros(
            (1, self.max_agents, self.max_candidates, self.candidate_dim),
            dtype=np.float32,
        )
        action_mask = np.zeros(
            (self.max_agents, self.max_candidates), dtype=np.bool_
        )
        agent_mask = np.zeros((1, self.max_agents), dtype=np.bool_)
        for output_index, (request_index, agent) in enumerate(
            zip(present_indices, present_agents)
        ):
            request = requests[request_index]
            request_obs[0, output_index] = self.normalizer.normalize_request(
                self._request_features(request, decision_time)
            )
            agent_mask[0, output_index] = True
            for candidate_index, candidate in enumerate(agent["candidates"]):
                if candidate_index >= self.max_candidates:
                    raise ValueError(
                        f"request {request['id']} has more than {self.max_candidates} actions"
                    )
                # Refresh the snapshot-dependent 16-prefix while preserving the
                # candidate's static delay, hop, FlowMod, and source features.
                static_tail = list(
                    map(float, candidate["candidate_features"][16:])
                )
                if static_tail:
                    # Legacy v3 data used the final feature as a candidate
                    # source flag.  Neutralize it so policy decisions depend
                    # on plan quality rather than generator identity.
                    static_tail[-1] = 0.0
                values = live_features[output_index][candidate_index] + static_tail
                candidate_tensor[0, output_index, candidate_index] = (
                    self.normalizer.normalize_candidate(values)
                )
                metrics = candidate.get("metrics") or {}
                delay_valid = (
                    candidate.get("plan") is None
                    or float(metrics.get("estimated_delay_ms", 0.0))
                    <= float(metrics.get("delay_bound_ms", 0.0)) + 1e-9
                )
                action_mask[output_index, candidate_index] = bool(
                    live_masks[output_index][candidate_index] and delay_valid
                )

        features_ready_ns = time.perf_counter_ns()
        least_loaded_rankings = self._least_loaded_rankings(
            footprints, snapshot, action_mask
        )
        heuristic_rankings = (
            self._dataset_selected_rankings(
                present_agents, least_loaded_rankings, action_mask
            )
            if self.hybrid_dataset_selected
            else least_loaded_rankings
        )
        has_resource_conflict = any(
            values[11] > 0.0
            for request_values in live_features
            for values in request_values
        )
        use_hybrid_rl = (
            self.hybrid_enabled
            and len(present_agents) > 1
            and self.hybrid_rl_weight > 0.0
            and (has_resource_conflict or not self.hybrid_conflict_only)
        )
        use_model = not self.hybrid_enabled or use_hybrid_rl
        q_values = None
        if use_model:
            with torch.inference_mode():
                q_values = self.model(
                    torch.from_numpy(request_obs).to(self.device),
                    torch.from_numpy(candidate_tensor).to(self.device),
                    torch.from_numpy(agent_mask).to(self.device),
                )[0].cpu()
            learned_rankings = ranked_actions(
                q_values, torch.from_numpy(action_mask)
            )[: len(present_agents)]
            if use_hybrid_rl:
                rankings, hybrid_rl_applied = self._hybrid_rankings(
                    learned_rankings,
                    heuristic_rankings,
                    footprints,
                    present_agents,
                    snapshot,
                )
            else:
                rankings = learned_rankings
                hybrid_rl_applied = False
        else:
            rankings = heuristic_rankings
            hybrid_rl_applied = False
        inference_ready_ns = time.perf_counter_ns()
        if self.hybrid_enabled:
            if use_hybrid_rl:
                self.hybrid_rl_batches += 1
                self.hybrid_rl_requests += len(present_agents)
                if hybrid_rl_applied:
                    self.hybrid_rl_applied_batches += 1
                    self.hybrid_rl_applied_requests += len(present_agents)
            else:
                self.hybrid_heuristic_batches += 1
                self.hybrid_heuristic_requests += len(present_agents)
        present_requests = [requests[index] for index in present_indices]
        rankings, sla_risks, sla_estimates = self._apply_q0_sla_guard(
            present_requests,
            present_agents,
            footprints,
            snapshot,
            rankings,
        )
        # The learned network ranks deployable plans.  Admission rejection is
        # authoritative only after every currently feasible plan has failed.
        # This prevents a shifted live snapshot from turning the reject logit
        # into a false capacity rejection.
        for index, agent in enumerate(present_agents):
            reject_action = int(agent["reject_action"])
            rankings[index] = [
                action for action in rankings[index] if action != reject_action
            ] + ([reject_action] if reject_action in rankings[index] else [])

        reject_actions = [int(agent["reject_action"]) for agent in present_agents]
        score_rows = (
            q_values[: len(present_agents)].tolist()
            if q_values is not None
            else None
        )
        priorities = [
            (
                float(requests[present_indices[index]]["leave_time"]),
                int(requests[present_indices[index]]["id"]),
            )
            for index in range(len(present_agents))
        ]
        effective_action_mask = action_mask[: len(present_agents)].copy()
        decode = decode_joint_candidates(
            footprints,
            snapshot,
            rankings,
            reject_actions=reject_actions,
            action_mask=effective_action_mask.tolist(),
            scores=score_rows,
            priorities=priorities,
            top_r=self.decoder_top_r,
            time_budget_ms=self.decoder_time_budget_ms,
        )
        decode_ready_ns = time.perf_counter_ns()
        selected_actions, sla_gate_rejections = self._enforce_q0_hard_sla_gate(
            present_requests,
            present_agents,
            decode.actions,
            sla_risks,
            sla_estimates,
        )
        present_request_ids = [
            request_ids[index] for index in present_indices
        ]
        commit = commit_decoded_joint_actions(
            self.ledger,
            present_request_ids,
            footprints,
            selected_actions,
            expected_version=decode.snapshot_version,
        )
        version_retry_count = 0
        while (
            not commit["committed"]
            and commit["reason"] == "version_mismatch"
            and version_retry_count < 16
        ):
            version_retry_count += 1
            self.commit_version_retries += 1
            snapshot = self.ledger.snapshot()
            retry_mask = np.zeros_like(action_mask)
            for agent_index, values in enumerate(
                candidate_action_mask(footprints, snapshot)
            ):
                retry_mask[agent_index, : len(values)] = values
            for agent_index, agent in enumerate(present_agents):
                for candidate_index, candidate in enumerate(agent["candidates"]):
                    metrics = candidate.get("metrics") or {}
                    delay_valid = (
                        candidate.get("plan") is None
                        or float(metrics.get("estimated_delay_ms", 0.0))
                        <= float(metrics.get("delay_bound_ms", 0.0)) + 1e-9
                    )
                    retry_mask[agent_index, candidate_index] &= delay_valid
            effective_action_mask = retry_mask[: len(present_agents)].copy()
            decode = decode_joint_candidates(
                footprints,
                snapshot,
                rankings,
                reject_actions=reject_actions,
                action_mask=retry_mask[: len(present_agents)].tolist(),
                scores=score_rows,
                priorities=priorities,
                top_r=self.decoder_top_r,
                time_budget_ms=self.decoder_time_budget_ms,
            )
            _, retry_sla_risks, retry_sla_estimates = self._apply_q0_sla_guard(
                present_requests,
                present_agents,
                footprints,
                snapshot,
                [list(ranking) for ranking in rankings],
            )
            selected_actions, sla_gate_rejections = (
                self._enforce_q0_hard_sla_gate(
                    present_requests,
                    present_agents,
                    decode.actions,
                    retry_sla_risks,
                    retry_sla_estimates,
                )
            )
            sla_risks = retry_sla_risks
            sla_estimates = retry_sla_estimates
            commit = commit_decoded_joint_actions(
                self.ledger,
                present_request_ids,
                footprints,
                selected_actions,
                expected_version=decode.snapshot_version,
            )
        version_retried = version_retry_count > 0
        if not commit["committed"]:
            raise RuntimeError(
                "jointly decoded online batch could not be committed exactly: "
                f"{commit['reason']}"
            )
        commit_ready_ns = time.perf_counter_ns()
        self.sla_guard_rejected_requests += len(sla_gate_rejections)
        self.decoder_timings_ms.append(float(decode.elapsed_ms))
        self.decoder_timeouts += int(decode.timed_out)
        self.decoder_greedy_budget_exhaustions += int(
            decode.greedy_budget_exhausted
        )
        self.decoder_repair_budget_exhaustions += int(
            decode.repair_budget_exhausted
        )
        commit_by_request = {
            int(result["request_id"]): result for result in commit["results"]
        }
        selected_actions = [
            int(commit_by_request[request_id]["candidate_index"])
            for request_id in present_request_ids
        ]
        rejection_diagnostics_by_request: dict[int, dict[str, Any]] = {}
        for agent_index, (request_id, agent, request_footprints) in enumerate(
            zip(present_request_ids, present_agents, footprints)
        ):
            result = commit_by_request[request_id]
            if result["accepted"]:
                continue
            diagnostic = self._candidate_feasibility_diagnostics(
                agent, request_footprints, snapshot
            )
            feasible_indices = list(diagnostic.pop("locally_feasible_indices"))
            if agent_index in sla_gate_rejections:
                category = str(sla_gate_rejections[agent_index]["reason"])
            elif diagnostic["plan_candidates"] == 0:
                category = "no_plan_candidate"
            elif not feasible_indices:
                counts = diagnostic["candidate_failure_counts"]
                dominant = max(
                    counts,
                    key=lambda key: (
                        int(counts[key]),
                        key == "bandwidth_insufficient",
                        key == "cpu_insufficient",
                        key == "memory_insufficient",
                    ),
                    default="unknown",
                )
                category = f"no_local_feasible_{dominant}"
            else:
                direct_conflicts: Counter[str] = Counter()
                direct_fit = False
                for candidate_index in feasible_indices:
                    proposal = list(selected_actions)
                    proposal[agent_index] = int(candidate_index)
                    feasible, failure = joint_footprints_feasible(
                        [
                            footprints[index][action]
                            for index, action in enumerate(proposal)
                        ],
                        snapshot,
                    )
                    if feasible:
                        direct_fit = True
                        break
                    direct_conflicts[failure or "unknown"] += 1
                if direct_fit:
                    category = (
                        "decoder_budget_exhausted"
                        if decode.timed_out
                        or decode.greedy_budget_exhausted
                        or decode.repair_budget_exhausted
                        else "decoder_ranking_limited"
                    )
                else:
                    dominant = max(
                        direct_conflicts,
                        key=lambda key: int(direct_conflicts[key]),
                        default="unknown",
                    )
                    category = f"batch_conflict_{dominant}"
                diagnostic["joint_conflict_counts"] = dict(
                    sorted(direct_conflicts.items())
                )
            diagnostic["category"] = category
            diagnostic["batch_size"] = len(present_agents)
            rejection_diagnostics_by_request[request_id] = diagnostic
            self.rejection_diagnostic_categories[category] += 1
        ranking_by_request = {
            request_ids[present_indices[index]]: rankings[index]
            for index in range(len(present_indices))
        }
        sla_risk_by_request = {
            request_ids[present_indices[index]]: sla_risks[index]
            for index in range(len(present_indices))
        }
        sla_estimate_by_request = {
            request_ids[present_indices[index]]: sla_estimates[index]
            for index in range(len(present_indices))
        }
        sla_gate_rejection_by_request = {
            request_ids[present_indices[index]]: value
            for index, value in sla_gate_rejections.items()
        }
        agent_by_id = {
            request_ids[present_indices[index]]: present_agents[index]
            for index in range(len(present_indices))
        }
        live_features_by_request = {
            request_ids[present_indices[index]]: live_features[index]
            for index in range(len(present_indices))
        }

        feature_ms = (features_ready_ns - started_ns) / 1_000_000.0
        inference_ms = (inference_ready_ns - features_ready_ns) / 1_000_000.0
        decoder_ms = (decode_ready_ns - inference_ready_ns) / 1_000_000.0
        commit_ms = (commit_ready_ns - decode_ready_ns) / 1_000_000.0
        total_ms = (commit_ready_ns - started_ns) / 1_000_000.0
        base_timing = {
            "batch_size": len(requests),
            "selection_mode": (
                "hybrid_wqmix"
                if use_hybrid_rl
                else "hybrid_dataset_selected"
                if self.hybrid_dataset_selected
                else "hybrid_least_loaded"
                if self.hybrid_least_loaded
                else "wqmix"
            ),
            "batch_resource_conflict": bool(has_resource_conflict),
            "hybrid_rl_applied": bool(hybrid_rl_applied),
            "feature_ms": feature_ms,
            "inference_ms": inference_ms,
            "decoder_ms": decoder_ms,
            "decoder_timed_out": bool(decode.timed_out),
            "decoder_candidates_examined": int(decode.candidates_examined),
            "decoder_greedy_budget_exhausted": bool(
                decode.greedy_budget_exhausted
            ),
            "decoder_repair_improvements": int(decode.repair_improvements),
            "decoder_repair_budget_exhausted": bool(
                decode.repair_budget_exhausted
            ),
            "commit_ms": commit_ms,
            "conversion_ms": feature_ms + decoder_ms + commit_ms,
            "total_ms": total_ms,
            "ledger_start_version": int(commit["start_version"]),
            "ledger_end_version": int(commit["end_version"]),
            "version_mismatch": bool(version_retried),
            "version_retry_count": int(version_retry_count),
        }
        results: list[dict[str, Any]] = []
        for request, request_id, agent in zip(requests, request_ids, agents):
            if agent is None:
                self.rejection_diagnostic_categories["missing_topk_candidates"] += 1
                results.append(
                    self._reject_plan(request, "missing_topk_candidates", base_timing)
                )
                continue
            agent = active_agent_by_request[request_id]
            result = commit_by_request[request_id]
            ranking = ranking_by_request[request_id]
            proposed = int(ranking[0]) if ranking else int(agent["reject_action"])
            sla_gate_rejection = sla_gate_rejection_by_request.get(request_id)
            selected_candidate_index = int(result["candidate_index"])
            selected_sla_estimate = sla_estimate_by_request[request_id].get(
                selected_candidate_index
            )
            candidate_risks = sla_risk_by_request[request_id]
            candidate_estimates = sla_estimate_by_request[request_id]
            candidate_live_features = live_features_by_request[request_id]
            sla_candidate_diagnostics = []
            for rank_position, action in enumerate(ranking):
                candidate_index = int(action)
                candidate = agent["candidates"][candidate_index]
                if candidate.get("plan") is None:
                    continue
                estimate = candidate_estimates.get(candidate_index)
                live_row = candidate_live_features[candidate_index]
                metrics = candidate.get("metrics") or {}
                sla_candidate_diagnostics.append(
                    {
                        "candidate_index": candidate_index,
                        "final_rank": rank_position,
                        "selected": candidate_index == selected_candidate_index,
                        "modeled_delay_ratio": candidate_risks.get(
                            candidate_index
                        ),
                        "failure_probability": (
                            float(estimate.failure_probability)
                            if estimate is not None
                            else None
                        ),
                        "confidence": (
                            float(estimate.confidence)
                            if estimate is not None
                            else None
                        ),
                        "ood_score": (
                            float(estimate.ood_score)
                            if estimate is not None
                            else None
                        ),
                        "predictor_eligible": bool(
                            estimate is not None
                            and estimate.ood_score <= self.sla_max_ood_score
                        ),
                        "peak_cpu_pressure": float(live_row[12]),
                        "peak_memory_pressure": float(live_row[13]),
                        "peak_bandwidth_pressure": float(live_row[14]),
                        "estimated_delay_ms": metrics.get(
                            "estimated_delay_ms"
                        ),
                        "segment_hops": metrics.get("segment_hops"),
                        "tree_edges": metrics.get("tree_edges"),
                    }
                )
            decoder_adjusted = int(result["candidate_index"]) != proposed
            self.decoder_adjusted_requests += int(decoder_adjusted)
            timing = dict(base_timing)
            timing.update(
                {
                    "dataset_batch_id": self.dataset_batch_by_request.get(request_id),
                    "proposed_candidate_index": proposed,
                    "selected_candidate_index": int(result["candidate_index"]),
                    "decoder_adjusted": bool(decoder_adjusted),
                    "fallback": bool(result["accepted"] and result["candidate_index"] != proposed),
                    "attempts": int(result["attempts"]),
                    "predicted_sla_risk": (
                        sla_gate_rejection["modeled_delay_ratio"]
                        if sla_gate_rejection is not None
                        else sla_risk_by_request[request_id].get(
                            selected_candidate_index
                        )
                    ),
                    "predicted_sla_failure_probability": (
                        sla_gate_rejection["failure_probability"]
                        if sla_gate_rejection is not None
                        else selected_sla_estimate.failure_probability
                        if selected_sla_estimate is not None
                        else None
                    ),
                    "sla_calibration_delay_bucket": (
                        sla_gate_rejection["delay_ratio_bucket"]
                        if sla_gate_rejection is not None
                        else selected_sla_estimate.delay_ratio_bucket
                        if selected_sla_estimate is not None
                        else None
                    ),
                    "sla_calibration_effective_support": (
                        sla_gate_rejection["effective_support"]
                        if sla_gate_rejection is not None
                        else selected_sla_estimate.effective_support
                        if selected_sla_estimate is not None
                        else None
                    ),
                    "sla_predictor_model_kind": (
                        sla_gate_rejection.get("model_kind")
                        if sla_gate_rejection is not None
                        else selected_sla_estimate.model_kind
                        if selected_sla_estimate is not None
                        else None
                    ),
                    "sla_predictor_confidence": (
                        sla_gate_rejection.get("confidence")
                        if sla_gate_rejection is not None
                        else selected_sla_estimate.confidence
                        if selected_sla_estimate is not None
                        else None
                    ),
                    "sla_predictor_ood_score": (
                        sla_gate_rejection.get("ood_score")
                        if sla_gate_rejection is not None
                        else selected_sla_estimate.ood_score
                        if selected_sla_estimate is not None
                        else None
                    ),
                    "sla_gate_rejected": sla_gate_rejection is not None,
                    "sla_gate_rejected_candidate_index": (
                        sla_gate_rejection["candidate_index"]
                        if sla_gate_rejection is not None
                        else None
                    ),
                    "sla_gate_rejection_reason": (
                        sla_gate_rejection["reason"]
                        if sla_gate_rejection is not None
                        else None
                    ),
                    "sla_candidate_diagnostics": sla_candidate_diagnostics,
                }
            )
            if not result["accepted"]:
                timing["rejection_diagnostics"] = (
                    rejection_diagnostics_by_request[request_id]
                )
                reason = (
                    str(sla_gate_rejection["reason"])
                    if sla_gate_rejection is not None
                    else str(result["reason"])
                )
                results.append(self._reject_plan(request, reason, timing))
                continue
            selected_index = int(result["candidate_index"])
            candidate = agent["candidates"][selected_index]
            self.online_repair_selected += int(
                str(candidate.get("source", "")).startswith("online_repair_")
            )
            plan = copy.deepcopy(candidate["plan"])
            plan["accepted"] = True
            plan["online_wqmix_selection"] = {
                "algorithm": "deployment_candidate_wqmix",
                "candidate_index": selected_index,
                "candidate_id": str(candidate["candidate_id"]),
                "source": str(candidate["source"]),
                "proposed_candidate_index": proposed,
                "fallback": bool(selected_index != proposed),
                "attempts": int(result["attempts"]),
            }
            plan["online_planning"] = timing
            results.append(plan)
            self.accepted_requests += 1
            self.fallback_requests += int(selected_index != proposed)
            heapq.heappush(
                self.leave_heap,
                (float(request["leave_time"]), request_id),
            )

        self.planned_requests += len(requests)
        self.planned_batches += 1
        self.rejected_requests += sum(not plan.get("accepted", False) for plan in results)
        self.batch_timings_ms.append(total_ms)
        return results

    def release(self, request_id: int) -> bool:
        released = self.ledger.release(int(request_id))
        self.release_count += int(released)
        return released

    def metadata(self) -> dict[str, Any]:
        snapshot = self.ledger.snapshot()
        timings = sorted(self.batch_timings_ms)
        decoder_timings = sorted(self.decoder_timings_ms)
        p95 = (
            timings[min(len(timings) - 1, max(0, int(np.ceil(0.95 * len(timings))) - 1))]
            if timings
            else None
        )
        ledger_violations = sum(
            value < -1e-7
            for resources in (
                snapshot.cpu_remaining,
                snapshot.memory_remaining,
                snapshot.bandwidth_remaining,
            )
            for value in resources.values()
        )
        return {
            "mode": "frozen_wqmix_online_inference",
            "checkpoint": str(self.checkpoint_path),
            "data_folder": str(self.data_folder),
            "profile": str(self.profile_path),
            "weights_updated_online": False,
            "candidate_feature_schema": "v3.1_source_neutral_at_inference",
            "candidate_generation": (
                "offline_topk_plus_online_repair"
                if self.repair_missing_plans
                else "offline_topk"
            ),
            "online_repair_mode": (
                "first_feasible_live"
                if self.repair_missing_plans
                else None
            ),
            "repair_missing_plans": self.repair_missing_plans,
            "online_repair_attempts": self.online_repair_attempts,
            "online_repair_candidates": self.online_repair_candidates,
            "online_repair_selected": self.online_repair_selected,
            "mean_online_repair_generation_ms": (
                sum(self.online_repair_generation_ms)
                / max(1, len(self.online_repair_generation_ms))
            ),
            "trained_max_candidates": self.trained_max_candidates,
            "inference_max_candidates": self.max_candidates,
            "selection_and_commit": "joint_decode_then_exact_versioned_commit",
            "microbatch_ms": self.microbatch_ms,
            "bandwidth_utilization_limit": self.bandwidth_utilization_limit,
            "q0_sla_safety_margin": self.q0_sla_safety_margin,
            "q0_hard_sla_gate": self.q0_hard_sla_gate,
            "sla_queue_safety_factor": self.sla_queue_safety_factor,
            "sla_calibration": (
                self.sla_calibrator.metadata()
                if self.sla_calibrator is not None
                else None
            ),
            "sla_calibration_rank_weight": self.sla_calibration_rank_weight,
            "sla_min_rerank_probability_delta": (
                self.sla_min_rerank_probability_delta
            ),
            "sla_max_ood_score": self.sla_max_ood_score,
            "max_sla_failure_probability": self.max_sla_failure_probability,
            "hybrid_least_loaded": self.hybrid_least_loaded,
            "hybrid_dataset_selected": self.hybrid_dataset_selected,
            "hybrid_rl_weight": self.hybrid_rl_weight,
            "hybrid_conflict_only": self.hybrid_conflict_only,
            "decoder_top_r": self.decoder_top_r,
            "decoder_time_budget_ms": self.decoder_time_budget_ms,
            "hybrid_heuristic_batches": self.hybrid_heuristic_batches,
            "hybrid_heuristic_requests": self.hybrid_heuristic_requests,
            "hybrid_rl_batches": self.hybrid_rl_batches,
            "hybrid_rl_requests": self.hybrid_rl_requests,
            "hybrid_rl_applied_batches": self.hybrid_rl_applied_batches,
            "hybrid_rl_applied_requests": self.hybrid_rl_applied_requests,
            "sla_guarded_requests": self.sla_guarded_requests,
            "sla_guard_filtered_candidates": self.sla_guard_filtered_candidates,
            "sla_guard_unsafe_candidates": self.sla_guard_unsafe_candidates,
            "sla_guard_rejected_requests": self.sla_guard_rejected_requests,
            "sla_probability_guarded_requests": (
                self.sla_probability_guarded_requests
            ),
            "sla_probability_rejected_requests": (
                self.sla_probability_rejected_requests
            ),
            "sla_predictor_candidates_evaluated": (
                self.sla_predictor_candidates_evaluated
            ),
            "sla_predictor_ood_skipped_candidates": (
                self.sla_predictor_ood_skipped_candidates
            ),
            "sla_predictor_probability_skipped_candidates": (
                self.sla_predictor_probability_skipped_candidates
            ),
            "sla_predictor_resource_safety_skipped_candidates": (
                self.sla_predictor_resource_safety_skipped_candidates
            ),
            "sla_predictor_promoted_candidates": (
                self.sla_predictor_promoted_candidates
            ),
            "initialization_ms": self.initialization_ms,
            "planned_batches": self.planned_batches,
            "planned_requests": self.planned_requests,
            "accepted_requests": self.accepted_requests,
            "rejected_requests": self.rejected_requests,
            "fallback_requests": self.fallback_requests,
            "decoder_adjusted_requests": self.decoder_adjusted_requests,
            "decoder_timeouts": self.decoder_timeouts,
            "decoder_greedy_budget_exhaustions": (
                self.decoder_greedy_budget_exhaustions
            ),
            "decoder_repair_budget_exhaustions": (
                self.decoder_repair_budget_exhaustions
            ),
            "commit_version_retries": self.commit_version_retries,
            "released_requests": self.release_count,
            "missing_candidate_requests": self.missing_candidate_requests,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "rejection_diagnostic_categories": dict(
                sorted(self.rejection_diagnostic_categories.items())
            ),
            "mean_batch_planning_ms": (
                sum(timings) / len(timings) if timings else None
            ),
            "p95_batch_planning_ms": p95,
            "mean_decoder_ms": (
                sum(decoder_timings) / len(decoder_timings)
                if decoder_timings
                else None
            ),
            "p95_decoder_ms": (
                decoder_timings[
                    min(
                        len(decoder_timings) - 1,
                        max(0, int(np.ceil(0.95 * len(decoder_timings))) - 1),
                    )
                ]
                if decoder_timings
                else None
            ),
            "ledger_version": int(snapshot.version),
            "ledger_violations": int(ledger_violations),
        }
