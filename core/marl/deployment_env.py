"""Counterfactual request-batch environment for online deployment RL."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from core.marl.batch_deployment_wqmix import (
    AtomicResourceLedger,
    ResourceFootprint,
    candidate_action_mask,
    deployment_candidate_features,
)
from core.marl.joint_candidate_decoder import (
    JointDecodeResult,
    decode_joint_candidates,
)
from core.marl.joint_candidate_transaction import commit_decoded_joint_actions
from core.marl.deployment_dataset import FeatureNormalizer
from core.marl.deployment_topk import CompletePlanCandidateGenerator
from scripts.generate_deployment_topk_v3 import (
    global_state_features,
    request_features,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_profile(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class BatchDeploymentEnv:
    """Regenerate candidates after every policy action.

    A step is one request micro-batch.  The environment owns the authoritative
    lifecycle ledger; action masks are generated from its current snapshot.
    Candidate rankings are converted into one jointly feasible action vector
    by the same bounded decoder used online.  The ledger then performs an
    exact versioned commit without hidden candidate fallback.
    """

    def __init__(
        self,
        requests: str | Path,
        profile: str | Path,
        normalizer: FeatureNormalizer,
        microbatch_ms: float = 5.0,
        max_agents: int = 32,
        top_k: int = 8,
        max_requests: int = 0,
        cpu_capacity: float = 55.0,
        memory_capacity: float = 45.0,
        bandwidth_utilization_limit: float = 0.8,
        queue_safety_factor: float = 2.0,
        sla_risk_weight: float = 8.0,
        sla_violation_penalty: float = 20.0,
        hidden_candidate_dim: int = 24,
    ) -> None:
        self.requests_path = Path(requests)
        self.profile_path = Path(profile)
        self.profile = load_profile(self.profile_path)
        rows = sorted(read_jsonl(self.requests_path), key=lambda row: float(row["arrival_time"]))
        self.requests = rows[:max_requests] if max_requests > 0 else rows
        self.normalizer = normalizer
        self.microbatch_seconds = float(microbatch_ms) / 1000.0
        self.max_agents = int(max_agents)
        self.top_k = int(top_k)
        self.cpu_capacity = float(cpu_capacity)
        self.memory_capacity = float(memory_capacity)
        self.bandwidth_utilization_limit = float(bandwidth_utilization_limit)
        self.queue_safety_factor = float(queue_safety_factor)
        self.sla_risk_weight = float(sla_risk_weight)
        self.sla_violation_penalty = float(sla_violation_penalty)
        if not 0.0 < self.bandwidth_utilization_limit <= 1.0:
            raise ValueError("bandwidth_utilization_limit must be in (0, 1]")
        if (
            self.queue_safety_factor < 0.0
            or self.sla_risk_weight < 0.0
            or self.sla_violation_penalty < 0.0
        ):
            raise ValueError("SLA reward parameters must be non-negative")
        self.edge_delay: Dict[tuple[int, int], float] = {}
        self.generator = CompletePlanCandidateGenerator(
            self.profile, max_candidates=self.top_k
        )
        self.request_dim = len(normalizer.request_mean)
        self.candidate_dim = len(normalizer.candidate_mean)
        self.state_dim = len(normalizer.state_mean)
        if self.candidate_dim != int(hidden_candidate_dim):
            raise ValueError(
                f"normalizer candidate dimension {self.candidate_dim} != {hidden_candidate_dim}"
            )
        self._reset_runtime()

    def _reset_runtime(self) -> None:
        bandwidth_capacity: Dict[tuple[int, int], float] = {}
        default_bandwidth = float(self.profile.get("default_bandwidth_mbps", 90.0))
        for raw in self.profile["edges"]:
            u, v = int(raw["u"]), int(raw["v"])
            capacity = (
                float(raw.get("bandwidth_mbps", default_bandwidth))
                * self.bandwidth_utilization_limit
            )
            bandwidth_capacity[(u, v)] = capacity
            bandwidth_capacity[(v, u)] = capacity
            delay = float(raw.get("delay_ms", self.profile.get("default_delay_ms", 1.0)))
            self.edge_delay[(u, v)] = delay
            self.edge_delay[(v, u)] = delay
        self.bandwidth_capacity = bandwidth_capacity
        dc_nodes = [int(value) for value in self.profile["dc_nodes_1based"]]
        self.ledger = AtomicResourceLedger(
            {node: self.cpu_capacity for node in dc_nodes},
            {node: self.memory_capacity for node in dc_nodes},
            bandwidth_capacity,
        )
        self._index = 0
        self._batch_id = 0
        self._leave_heap: List[tuple[float, int]] = []
        self._current: Optional[Dict[str, torch.Tensor]] = None
        self._current_agents: List[Dict[str, Any]] = []
        self._current_time = 0.0
        self._pending_decode: Optional[JointDecodeResult] = None

    def reset(self) -> Dict[str, torch.Tensor]:
        self._reset_runtime()
        self._prepare_next()
        if self._current is None:
            return self._terminal_observation()
        return self._current

    def _terminal_observation(self) -> Dict[str, torch.Tensor]:
        return {
            "request_observations": torch.zeros(1, self.max_agents, self.request_dim),
            "candidate_features": torch.zeros(1, self.max_agents, self.top_k + 1, self.candidate_dim),
            "action_mask": torch.zeros(1, self.max_agents, self.top_k + 1, dtype=torch.bool),
            "agent_mask": torch.zeros(1, self.max_agents, dtype=torch.bool),
            "states": torch.zeros(1, self.state_dim),
        }

    def _prepare_next(self) -> None:
        if self._index >= len(self.requests):
            while self._leave_heap:
                _, request_id = heapq.heappop(self._leave_heap)
                self.ledger.release(request_id)
            self._current = None
            self._current_agents = []
            return
        first_arrival = float(self.requests[self._index]["arrival_time"])
        cutoff = first_arrival + self.microbatch_seconds
        batch: List[Mapping[str, Any]] = []
        while self._index < len(self.requests) and len(batch) < self.max_agents:
            request = self.requests[self._index]
            if batch and float(request["arrival_time"]) > cutoff + 1e-12:
                break
            batch.append(request)
            self._index += 1
        self._current_time = cutoff
        while self._leave_heap and self._leave_heap[0][0] <= cutoff + 1e-12:
            _, request_id = heapq.heappop(self._leave_heap)
            self.ledger.release(request_id)
        snapshot = self.ledger.snapshot()
        generated = [self.generator.generate(request, snapshot) for request in batch]
        footprints = [
            [candidate.footprint for candidate in candidates] + [None]
            for candidates in generated
        ]
        features = deployment_candidate_features(footprints, snapshot)
        hard_masks = candidate_action_mask(footprints, snapshot)
        request_obs = np.zeros((self.max_agents, self.request_dim), dtype=np.float32)
        candidate_obs = np.zeros(
            (self.max_agents, self.top_k + 1, self.candidate_dim), dtype=np.float32
        )
        action_mask = np.zeros((self.max_agents, self.top_k + 1), dtype=np.bool_)
        agent_mask = np.zeros(self.max_agents, dtype=np.bool_)
        agents: List[Dict[str, Any]] = []
        for agent_index, (request, candidates) in enumerate(zip(batch, generated)):
            agent_mask[agent_index] = True
            request_obs[agent_index] = self.normalizer.normalize_request(
                request_features(request, cutoff)
            )
            for candidate_index, candidate in enumerate(candidates):
                metric_features = [
                    candidate.metrics["estimated_delay_ms"],
                    candidate.metrics["delay_bound_ms"],
                    candidate.metrics["segment_hops"],
                    candidate.metrics["tree_edges"],
                    candidate.metrics["flowmod_estimate"],
                    candidate.metrics["peak_resource_pressure"],
                    candidate.objective,
                    0.0,
                ]
                values = features[agent_index][candidate_index] + metric_features
                candidate_obs[agent_index, candidate_index] = self.normalizer.normalize_candidate(values)
                valid = bool(
                    hard_masks[agent_index][candidate_index]
                    and candidate.metrics["estimated_delay_ms"]
                    <= candidate.metrics["delay_bound_ms"] + 1e-9
                )
                action_mask[agent_index, candidate_index] = valid
            reject_index = len(candidates)
            reject_values = features[agent_index][reject_index] + [0.0] * 8
            candidate_obs[agent_index, reject_index] = self.normalizer.normalize_candidate(reject_values)
            action_mask[agent_index, reject_index] = True
            agents.append({
                "request": request,
                "candidates": candidates,
                "footprints": footprints[agent_index],
                "reject_action": reject_index,
            })
        self._batch_id += 1
        self._current_agents = agents
        self._current = {
            "request_observations": torch.from_numpy(request_obs).unsqueeze(0),
            "candidate_features": torch.from_numpy(candidate_obs).unsqueeze(0),
            "action_mask": torch.from_numpy(action_mask).unsqueeze(0),
            "agent_mask": torch.from_numpy(agent_mask).unsqueeze(0),
            "states": torch.from_numpy(
                self.normalizer.normalize_state(global_state_features(snapshot, batch))
            ).unsqueeze(0),
        }

    def decode_rankings(
        self,
        rankings: Sequence[Sequence[int]],
        *,
        scores: Optional[Sequence[Sequence[float]]] = None,
        top_r: int = 4,
        time_budget_ms: float = 2.0,
    ) -> JointDecodeResult:
        """Decode policy rankings under the current immutable snapshot."""

        if self._current is None:
            raise RuntimeError("cannot decode a terminated environment")
        if len(rankings) != len(self._current_agents):
            raise ValueError("rankings must contain one row per active request")
        snapshot = self.ledger.snapshot()
        result = decode_joint_candidates(
            [agent["footprints"] for agent in self._current_agents],
            snapshot,
            rankings,
            reject_actions=[
                int(agent["reject_action"]) for agent in self._current_agents
            ],
            action_mask=self._current["action_mask"][0][
                : len(self._current_agents)
            ].tolist(),
            scores=scores,
            priorities=[
                (
                    float(agent["request"]["leave_time"]),
                    int(agent["request"]["id"]),
                )
                for agent in self._current_agents
            ],
            top_r=top_r,
            time_budget_ms=time_budget_ms,
        )
        self._pending_decode = result
        return result

    def _modeled_candidate_delay(
        self,
        candidate: Any,
        footprint: ResourceFootprint,
        snapshot: Any,
    ) -> float:
        def edge_delay(u: int, v: int) -> float:
            edge = (int(u), int(v))
            capacity = float(self.bandwidth_capacity[edge])
            current_used = capacity - float(snapshot.bandwidth_remaining[edge])
            used_after = current_used + float(footprint.bandwidth.get(edge, 0.0))
            utilization = min(0.95, max(0.0, used_after / max(capacity, 1e-6)))
            propagation = float(self.edge_delay[edge])
            return propagation * (
                1.0
                + self.queue_safety_factor
                * utilization
                / max(1e-6, 1.0 - utilization)
            )

        def path_delay(path: Sequence[int]) -> float:
            return sum(edge_delay(u, v) for u, v in zip(path, path[1:]))

        plan = candidate.plan
        segment_delay = sum(
            path_delay([int(node) for node in segment.get("path", [])])
            for segment in plan.get("segments", [])
        )
        multicast_paths = (plan.get("multicast") or {}).get("paths") or {}
        branch_delay = max(
            (
                path_delay([int(node) for node in path])
                for path in multicast_paths.values()
            ),
            default=0.0,
        )
        return float(segment_delay + branch_delay)

    def step(
        self,
        actions: Sequence[int],
        *,
        expected_version: Optional[int] = None,
    ) -> tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        if self._current is None:
            raise RuntimeError("cannot step a terminated environment")
        if len(actions) != len(self._current_agents):
            raise ValueError("actions must contain one candidate index per active request")
        mask = self._current["action_mask"][0]
        for agent_index, action in enumerate(actions):
            if not 0 <= int(action) < mask.shape[1] or not bool(mask[agent_index, int(action)]):
                raise ValueError(f"action {action} is masked for active agent {agent_index}")
        request_ids = [int(agent["request"]["id"]) for agent in self._current_agents]
        footprints = [agent["footprints"] for agent in self._current_agents]
        snapshot = self.ledger.snapshot()
        version = (
            int(expected_version)
            if expected_version is not None
            else int(snapshot.version)
        )
        commit = commit_decoded_joint_actions(
            self.ledger,
            request_ids,
            footprints,
            [int(action) for action in actions],
            expected_version=version,
        )
        if not commit["committed"]:
            raise RuntimeError(
                "decoded training action could not be committed exactly: "
                f"{commit['reason']}"
            )
        accepted = int(commit["accepted"])
        rejected = len(actions) - accepted
        for agent, result in zip(self._current_agents, commit["results"]):
            if result["accepted"]:
                request = agent["request"]
                heapq.heappush(self._leave_heap, (float(request["leave_time"]), int(request["id"])))
        sla_risks = []
        modeled_sla_violations = 0
        reward = 10.0 * accepted - 10.0 * rejected
        for agent, action, result in zip(self._current_agents, actions, commit["results"]):
            if not result["accepted"]:
                continue
            candidate = agent["candidates"][int(action)]
            footprint = agent["footprints"][int(action)]
            modeled_delay = self._modeled_candidate_delay(
                candidate, footprint, snapshot
            )
            delay_bound = max(
                1e-6, float(agent["request"].get("delay_bound_ms", 0.0))
            )
            risk = modeled_delay / delay_bound
            sla_risks.append(risk)
            bounded_risk = min(2.0, risk)
            reward -= self.sla_risk_weight * bounded_risk * bounded_risk
            if risk > 1.0:
                reward -= self.sla_violation_penalty
                modeled_sla_violations += 1
        reward -= 0.1 * sum(int(result["reason"] == "insufficient_bandwidth") for result in commit["results"])
        info = {
            "batch_id": self._batch_id,
            "decision_time": self._current_time,
            "proposed_actions": [int(value) for value in actions],
            "executed_actions": [int(value) for value in actions],
            "accepted": accepted,
            "rejected": rejected,
            "commit": commit,
            "ledger_version": self.ledger.version,
            "mean_modeled_sla_risk": (
                sum(sla_risks) / len(sla_risks) if sla_risks else 0.0
            ),
            "modeled_sla_violations": modeled_sla_violations,
            "decoder": (
                self._pending_decode.to_dict()
                if self._pending_decode is not None
                and tuple(map(int, actions)) == self._pending_decode.actions
                else None
            ),
        }
        self._pending_decode = None
        self._prepare_next()
        done = self._current is None
        return self._terminal_observation() if done else self._current, reward, done, info

    @property
    def active_agents(self) -> int:
        return len(self._current_agents)

    @property
    def batch_id(self) -> int:
        return self._batch_id
