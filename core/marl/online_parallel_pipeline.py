"""Central-ledger online pipeline for parallel complete-plan generation.

The pipeline deliberately separates pure candidate work from authoritative
resource mutation:

* workers receive one immutable :class:`ResourceSnapshot` and return complete
  candidates; they never touch the ledger;
* an optional central ranker (for example WQMIX) orders those candidates;
* the bounded joint decoder and a versioned exact commit run in the central
  coordinator; a concurrent commit causes a fresh snapshot and retry.

This module is intentionally policy-agnostic.  An HRL adapter can provide the
candidate factory without changing the hard-constraint or lifecycle semantics.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import heapq
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

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
from core.marl.joint_candidate_decoder import JointDecodeResult, decode_joint_candidates
from core.marl.joint_candidate_transaction import commit_decoded_joint_actions


CandidateFactory = Callable[
    [Mapping[str, Any], ResourceSnapshot], "CandidateGeneration"
]
Ranker = Callable[
    [Sequence[Mapping[str, Any]], Sequence["CandidateGeneration"], ResourceSnapshot,
     Sequence[Sequence[bool]]],
    Sequence[Sequence[int]],
]


def complete_plan_candidate_factory(
    generator: Any,
    *,
    baseline_plan_provider: Optional[Callable[[Mapping[str, Any]], Optional[Mapping[str, Any]]]] = None,
) -> CandidateFactory:
    """Adapt a complete-plan generator (including HRL-derived candidates).

    ``generator`` must expose ``generate(request, snapshot, baseline_plan)``
    and return objects with ``footprint``, ``plan`` and ``objective`` fields,
    such as :class:`CompletePlanCandidateGenerator`.  The optional provider
    supplies a current HRL complete plan as the first candidate; it is read
    before generation and never mutated by this adapter.
    """

    if not callable(getattr(generator, "generate", None)):
        raise TypeError("generator must expose generate(request, snapshot, baseline_plan)")

    def factory(
        request: Mapping[str, Any], snapshot: ResourceSnapshot
    ) -> CandidateGeneration:
        baseline = (
            baseline_plan_provider(request)
            if baseline_plan_provider is not None
            else None
        )
        rows = list(generator.generate(request, snapshot, baseline))
        rows.sort(key=lambda row: (float(getattr(row, "objective", 0.0)), str(getattr(row, "candidate_id", ""))))
        footprints = [getattr(row, "footprint") for row in rows]
        payloads = [getattr(row, "plan") for row in rows]
        scores = [-float(getattr(row, "objective", 0.0)) for row in rows]
        valid_mask = [
            float(getattr(row, "metrics", {}).get("estimated_delay_ms", 0.0))
            <= float(getattr(row, "metrics", {}).get("delay_bound_ms", float("inf"))) + 1e-9
            for row in rows
        ]
        # CompletePlanCandidateGenerator intentionally returns only deploy
        # candidates; make reject explicit for the central decoder contract.
        footprints.append(None)
        payloads.append(None)
        scores.append(0.0)
        valid_mask.append(True)
        return CandidateGeneration(
            request_id=int(request["id"]),
            footprints=tuple(footprints),
            payloads=tuple(payloads),
            rankings=tuple(range(len(rows))),
            scores=tuple(scores),
            valid_mask=tuple(valid_mask),
            metadata={
                "generator": type(generator).__name__,
                "baseline_plan_used": baseline is not None,
                "candidate_metrics": [
                    {
                        **dict(getattr(row, "metrics", {}) or {}),
                        "objective": float(getattr(row, "objective", 0.0)),
                    }
                    for row in rows
                ] + [{}],
            },
        )

    return factory


@dataclass
class CandidateGeneration:
    """Pure result returned by one candidate worker.

    ``None`` is the explicit reject action.  Candidate payloads are optional
    and are kept aligned with footprints so the caller can recover the final
    complete HRL plan after the exact commit.
    """

    request_id: int
    footprints: Sequence[Optional[ResourceFootprint]]
    payloads: Sequence[Optional[Mapping[str, Any]]] = ()
    rankings: Sequence[int] = ()
    scores: Sequence[float] = ()
    valid_mask: Sequence[bool] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WQMIXCandidateRanker:
    """Central WQMIX ranker for :class:`CandidateGeneration` rows.

    The ranker intentionally performs inference only.  Resource and SLA
    validity arrive through ``masks`` and remain enforced by the decoder and
    exact ledger transaction.
    """

    def __init__(
        self,
        checkpoint: str | Any,
        *,
        device: str = "cpu",
        torch_threads: int = 1,
    ) -> None:
        if torch_threads:
            torch.set_num_threads(int(torch_threads))
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
        self.max_agents = int(payload.get("max_agents", 32))
        self.max_candidates = int(payload.get("max_candidates", 9))
        self.request_dim = int(payload["request_dim"])
        self.candidate_dim = int(payload["candidate_dim"])
        self.model = BatchCandidateQNetwork(
            self.request_dim,
            self.candidate_dim,
            int(payload["hidden_dim"]),
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

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

    @staticmethod
    def _metric_tail(
        generation: CandidateGeneration, candidate_index: int
    ) -> list[float]:
        rows = generation.metadata.get("candidate_metrics", [])
        metrics = rows[candidate_index] if candidate_index < len(rows) else {}
        metrics = metrics if isinstance(metrics, Mapping) else {}
        return [
            float(metrics.get("estimated_delay_ms", 0.0)),
            float(metrics.get("delay_bound_ms", 0.0)),
            float(metrics.get("segment_hops", 0.0)),
            float(metrics.get("tree_edges", 0.0)),
            float(metrics.get("flowmod_estimate", 0.0)),
            float(metrics.get("peak_resource_pressure", 0.0)),
            float(metrics.get("objective", 0.0)),
            0.0,
        ]

    def __call__(
        self,
        requests: Sequence[Mapping[str, Any]],
        generations: Sequence[CandidateGeneration],
        snapshot: ResourceSnapshot,
        masks: Sequence[Sequence[bool]],
    ) -> Sequence[Sequence[int]]:
        if len(requests) != len(generations) or len(masks) != len(generations):
            raise ValueError("WQMIX ranker inputs must have equal request counts")
        if len(requests) > self.max_agents:
            raise ValueError("batch exceeds WQMIX max_agents")
        footprints = [list(generation.footprints) for generation in generations]
        dynamic = deployment_candidate_features(footprints, snapshot)
        decision_time = max(float(request.get("arrival_time", 0.0)) for request in requests)
        request_obs = np.zeros((1, self.max_agents, self.request_dim), dtype=np.float32)
        candidate_obs = np.zeros(
            (1, self.max_agents, self.max_candidates, self.candidate_dim),
            dtype=np.float32,
        )
        agent_mask = np.zeros((1, self.max_agents), dtype=np.bool_)
        action_mask = np.zeros((self.max_agents, self.max_candidates), dtype=np.bool_)
        for agent_index, (request, generation, mask) in enumerate(
            zip(requests, generations, masks)
        ):
            request_obs[0, agent_index] = self.normalizer.normalize_request(
                self._request_features(request, decision_time)
            )
            agent_mask[0, agent_index] = True
            if len(generation.footprints) > self.max_candidates:
                raise ValueError("candidate generation exceeds WQMIX action width")
            for candidate_index, _ in enumerate(generation.footprints):
                values = dynamic[agent_index][candidate_index] + self._metric_tail(
                    generation, candidate_index
                )
                candidate_obs[0, agent_index, candidate_index] = (
                    self.normalizer.normalize_candidate(values)
                )
                action_mask[agent_index, candidate_index] = bool(mask[candidate_index])
        with torch.inference_mode():
            q_values = self.model(
                torch.from_numpy(request_obs).to(self.device),
                torch.from_numpy(candidate_obs).to(self.device),
                torch.from_numpy(agent_mask).to(self.device),
            )[0]
        return ranked_actions(
            q_values,
            torch.from_numpy(action_mask),
        )[: len(generations)]


class ObjectiveCandidateRanker:
    """Deterministic objective-order baseline for safety gating."""

    def __call__(
        self,
        requests: Sequence[Mapping[str, Any]],
        generations: Sequence[CandidateGeneration],
        snapshot: ResourceSnapshot,
        masks: Sequence[Sequence[bool]],
    ) -> Sequence[Sequence[int]]:
        rankings = []
        for generation in generations:
            reject = len(generation.footprints) - 1
            if generation.scores and len(generation.scores) == len(generation.footprints):
                order = sorted(
                    range(reject),
                    key=lambda index: (-float(generation.scores[index]), index),
                )
            else:
                order = list(range(reject))
            rankings.append(order + [reject])
        return rankings


class GuardedWQMIXRanker:
    """Use WQMIX only when its decoded acceptance is no worse than baseline.

    This is a safety gate, not a reward change: both rankers see exactly the
    same immutable snapshot and masks, and the selected order is decoded again
    by the authoritative central pipeline.
    """

    def __init__(
        self,
        learned: Ranker,
        fallback: Ranker,
        *,
        top_r: int = 4,
        time_budget_ms: float = 1.0,
    ) -> None:
        self.learned = learned
        self.fallback = fallback
        self.top_r = int(top_r)
        self.time_budget_ms = float(time_budget_ms)
        self.learned_selected = 0
        self.fallback_selected = 0

    @staticmethod
    def _decoded_acceptance(
        generations: Sequence[CandidateGeneration],
        rankings: Sequence[Sequence[int]],
        snapshot: ResourceSnapshot,
        masks: Sequence[Sequence[bool]],
        top_r: int,
        time_budget_ms: float,
    ) -> int:
        result = decode_joint_candidates(
            [list(generation.footprints) for generation in generations],
            snapshot,
            rankings,
            reject_actions=[len(generation.footprints) - 1 for generation in generations],
            action_mask=masks,
            top_r=top_r,
            time_budget_ms=time_budget_ms,
        )
        return int(result.accepted)

    def __call__(
        self,
        requests: Sequence[Mapping[str, Any]],
        generations: Sequence[CandidateGeneration],
        snapshot: ResourceSnapshot,
        masks: Sequence[Sequence[bool]],
    ) -> Sequence[Sequence[int]]:
        learned = [list(row) for row in self.learned(requests, generations, snapshot, masks)]
        fallback = [list(row) for row in self.fallback(requests, generations, snapshot, masks)]
        learned_acceptance = self._decoded_acceptance(
            generations, learned, snapshot, masks, self.top_r, self.time_budget_ms
        )
        fallback_acceptance = self._decoded_acceptance(
            generations, fallback, snapshot, masks, self.top_r, self.time_budget_ms
        )
        if learned_acceptance >= fallback_acceptance:
            self.learned_selected += 1
            return learned
        self.fallback_selected += 1
        return fallback


@dataclass
class ParallelBatchResult:
    """Observable result of one central-ledger batch attempt."""

    request_ids: tuple[int, ...]
    actions: tuple[int, ...]
    accepted: int
    rejected: int
    plans: tuple[Optional[Mapping[str, Any]], ...]
    commit: Mapping[str, Any]
    decoder: Mapping[str, Any]
    timing_ms: Mapping[str, float]
    retry_count: int
    worker_errors: Mapping[int, str] = field(default_factory=dict)

    @property
    def committed(self) -> bool:
        return bool(self.commit.get("committed", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_ids": list(self.request_ids),
            "actions": list(self.actions),
            "accepted": int(self.accepted),
            "rejected": int(self.rejected),
            "plans": list(self.plans),
            "commit": dict(self.commit),
            "decoder": dict(self.decoder),
            "timing_ms": dict(self.timing_ms),
            "retry_count": int(self.retry_count),
            "worker_errors": dict(self.worker_errors),
        }


class CentralSharedLedgerPipeline:
    """Run parallel candidate generation with one authoritative ledger.

    The class is safe for callers that submit batches from multiple threads.
    Batches are serialized only around snapshot/decode/commit; candidate
    generation runs outside the ledger lock and can use ``worker_count``
    threads.  Lifecycle releases happen before every batch snapshot.
    """

    def __init__(
        self,
        ledger: AtomicResourceLedger,
        candidate_factory: CandidateFactory,
        *,
        ranker: Optional[Ranker] = None,
        worker_count: int = 4,
        top_r: int = 4,
        decoder_time_budget_ms: float = 2.0,
        max_greedy_evaluations: int = 48,
        max_repair_evaluations: int = 16,
        max_version_retries: int = 3,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if top_r <= 0 or decoder_time_budget_ms < 0.0:
            raise ValueError("invalid decoder configuration")
        if max_version_retries < 0:
            raise ValueError("max_version_retries must be non-negative")
        self.ledger = ledger
        self.candidate_factory = candidate_factory
        self.ranker = ranker
        self.worker_count = int(worker_count)
        self.top_r = int(top_r)
        self.decoder_time_budget_ms = float(decoder_time_budget_ms)
        self.max_greedy_evaluations = int(max_greedy_evaluations)
        self.max_repair_evaluations = int(max_repair_evaluations)
        self.max_version_retries = int(max_version_retries)
        self._executor = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="sfc-candidate",
        )
        self._submit_lock = threading.Lock()
        self._leave_heap: list[tuple[float, int]] = []
        self.stats = {
            "batches": 0,
            "requests": 0,
            "accepted": 0,
            "version_retries": 0,
            "worker_errors": 0,
            "decoder_timeouts": 0,
            "lifecycle_releases": 0,
            "allocation_replacements": 0,
            "replacement_failures": 0,
            "replacement_preparations": 0,
            "replacement_preparation_failures": 0,
            "prepared_replacement_commits": 0,
            "prepared_replacement_commit_failures": 0,
            "prepared_replacement_aborts": 0,
        }

    def register_lifetime(self, request_id: int, leave_time: float) -> None:
        """Register an accepted request for deterministic lifecycle release."""

        heapq.heappush(self._leave_heap, (float(leave_time), int(request_id)))

    def _release_expired_unlocked(self, timestamp: float) -> int:
        released = 0
        while self._leave_heap and self._leave_heap[0][0] <= float(timestamp) + 1e-12:
            _, request_id = heapq.heappop(self._leave_heap)
            released += int(self.ledger.release(request_id))
        self.stats["lifecycle_releases"] += released
        return released

    def release_expired(self, timestamp: float) -> int:
        """Release expired allocations through the serialized mutation channel."""

        with self._submit_lock:
            return self._release_expired_unlocked(timestamp)

    def release(self, request_id: int) -> bool:
        with self._submit_lock:
            released = bool(self.ledger.release(int(request_id)))
            self.stats["lifecycle_releases"] += int(released)
            return released

    def replace_allocation(
        self,
        request_id: int,
        footprint: ResourceFootprint,
        *,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        with self._submit_lock:
            result = self.ledger.replace(
                int(request_id),
                footprint,
                expected_version=expected_version,
            )
            if result.get("replaced"):
                self.stats["allocation_replacements"] += 1
            else:
                self.stats["replacement_failures"] += 1
            return result

    def prepare_replacement(
        self,
        request_id: int,
        temporary_footprint: ResourceFootprint,
        *,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        with self._submit_lock:
            result = self.ledger.prepare_replacement(
                int(request_id),
                temporary_footprint,
                expected_version=expected_version,
            )
            key = (
                "replacement_preparations"
                if result.get("prepared")
                else "replacement_preparation_failures"
            )
            self.stats[key] += 1
            return result

    def commit_prepared_replacement(
        self,
        request_id: int,
        token: str,
        new_footprint: ResourceFootprint,
    ) -> dict[str, Any]:
        with self._submit_lock:
            result = self.ledger.commit_prepared_replacement(
                int(request_id), token, new_footprint
            )
            if result.get("replaced"):
                self.stats["prepared_replacement_commits"] += 1
                self.stats["allocation_replacements"] += 1
            else:
                self.stats["prepared_replacement_commit_failures"] += 1
                self.stats["replacement_failures"] += 1
            return result

    def abort_prepared_replacement(self, token: str) -> dict[str, Any]:
        with self._submit_lock:
            result = self.ledger.abort_prepared_replacement(token)
            self.stats["prepared_replacement_aborts"] += int(
                bool(result.get("aborted"))
            )
            return result

    @staticmethod
    def _normalise_generation(
        request: Mapping[str, Any], generation: CandidateGeneration
    ) -> CandidateGeneration:
        request_id = int(request["id"])
        if int(generation.request_id) != request_id:
            raise ValueError(
                f"candidate worker returned request {generation.request_id}, "
                f"expected {request_id}"
            )
        footprints = list(generation.footprints)
        payloads = list(generation.payloads)
        scores = list(map(float, generation.scores))
        valid_mask = list(map(bool, generation.valid_mask))
        if payloads and len(payloads) != len(footprints):
            raise ValueError("candidate payloads and footprints must align")
        if not payloads:
            payloads = [None] * len(footprints)
        reject_indices = [index for index, item in enumerate(footprints) if item is None]
        if not reject_indices:
            footprints.append(None)
            payloads.append(None)
            if scores:
                scores.append(0.0)
            if valid_mask:
                valid_mask.append(True)
        elif reject_indices[-1] != len(footprints) - 1:
            raise ValueError("reject action must be the final candidate")
        rankings = list(map(int, generation.rankings))
        if not rankings:
            rankings = list(range(len(footprints)))
        if scores and len(scores) != len(footprints):
            raise ValueError("candidate scores and footprints must align")
        if valid_mask and len(valid_mask) != len(footprints):
            raise ValueError("candidate valid_mask and footprints must align")
        if not valid_mask:
            valid_mask = [True] * len(footprints)
        return CandidateGeneration(
            request_id=request_id,
            footprints=tuple(footprints),
            payloads=tuple(payloads),
            rankings=tuple(rankings),
            scores=tuple(scores),
            valid_mask=tuple(valid_mask),
            metadata=dict(generation.metadata),
        )

    def _generate(
        self,
        requests: Sequence[Mapping[str, Any]],
        snapshot: ResourceSnapshot,
    ) -> tuple[list[CandidateGeneration], dict[int, str]]:
        futures = {
            self._executor.submit(self.candidate_factory, request, snapshot): request
            for request in requests
        }
        generations: list[Optional[CandidateGeneration]] = [None] * len(requests)
        errors: dict[int, str] = {}
        index_by_id = {int(request["id"]): index for index, request in enumerate(requests)}
        for future, request in futures.items():
            request_id = int(request["id"])
            try:
                value = future.result()
                if not isinstance(value, CandidateGeneration):
                    raise TypeError("candidate factory must return CandidateGeneration")
                generations[index_by_id[request_id]] = self._normalise_generation(
                    request, value
                )
            except Exception as exc:  # worker failure becomes an explicit reject
                errors[request_id] = f"{type(exc).__name__}: {exc}"
                generations[index_by_id[request_id]] = CandidateGeneration(
                    request_id=request_id,
                    footprints=(None,),
                    payloads=(None,),
                    rankings=(0,),
                    metadata={"worker_error": errors[request_id]},
                )
        return [value for value in generations if value is not None], errors

    def _rank(
        self,
        requests: Sequence[Mapping[str, Any]],
        generations: Sequence[CandidateGeneration],
        snapshot: ResourceSnapshot,
        masks: Sequence[Sequence[bool]],
    ) -> list[list[int]]:
        if self.ranker is None:
            raw = [list(generation.rankings) for generation in generations]
        else:
            raw = [
                list(map(int, row))
                for row in self.ranker(requests, generations, snapshot, masks)
            ]
        if len(raw) != len(generations):
            raise ValueError("ranker must return one ranking per request")
        result: list[list[int]] = []
        for index, (ranking, generation, mask) in enumerate(zip(raw, generations, masks)):
            reject = len(generation.footprints) - 1
            seen: set[int] = set()
            valid = []
            for action in ranking:
                action = int(action)
                if action in seen or not 0 <= action < len(generation.footprints):
                    continue
                seen.add(action)
                if bool(mask[action]) or action == reject:
                    valid.append(action)
            for action in range(len(generation.footprints)):
                if action not in seen and bool(mask[action]) and action != reject:
                    valid.append(action)
            if reject not in valid:
                valid.append(reject)
            result.append(valid)
        return result

    def submit_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        timestamp: Optional[float] = None,
    ) -> ParallelBatchResult:
        """Generate, rank, decode and exactly commit one request batch.

        A version conflict never partially commits.  The batch is regenerated
        against the fresh snapshot and retried up to ``max_version_retries``.
        """

        if not requests:
            raise ValueError("requests must not be empty")
        request_ids = tuple(int(request["id"]) for request in requests)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request ids in a batch must be unique")
        with self._submit_lock:
            if timestamp is not None:
                self._release_expired_unlocked(float(timestamp))
            started = time.perf_counter_ns()
            retry_count = 0
            last_errors: dict[int, str] = {}
            while True:
                snapshot = self.ledger.snapshot()
                generation_started = time.perf_counter_ns()
                generations, errors = self._generate(requests, snapshot)
                last_errors = errors
                generation_ms = (time.perf_counter_ns() - generation_started) / 1e6
                footprints = [list(generation.footprints) for generation in generations]
                masks = candidate_action_mask(footprints, snapshot)
                masks = [
                    [
                        bool(resource_ok and generation.valid_mask[index])
                        for index, resource_ok in enumerate(row)
                    ]
                    for row, generation in zip(masks, generations)
                ]
                rankings = self._rank(requests, generations, snapshot, masks)
                score_rows = [
                    list(generation.scores) for generation in generations
                ]
                # The decoder accepts one score row per agent.  Use positional
                # scores only when every worker supplied an aligned row;
                # otherwise rankings remain the deterministic fallback.
                scores_arg = (
                    score_rows
                    if all(len(row) == len(generation.footprints)
                           for row, generation in zip(score_rows, generations))
                    else None
                )
                decode_started = time.perf_counter_ns()
                decode: JointDecodeResult = decode_joint_candidates(
                    footprints,
                    snapshot,
                    rankings,
                    reject_actions=[len(row) - 1 for row in footprints],
                    action_mask=masks,
                    scores=scores_arg,
                    priorities=[
                        (float(request.get("leave_time", float("inf"))), int(request["id"]))
                        for request in requests
                    ],
                    top_r=self.top_r,
                    time_budget_ms=self.decoder_time_budget_ms,
                    max_greedy_evaluations=self.max_greedy_evaluations,
                    max_repair_evaluations=self.max_repair_evaluations,
                )
                decode_ms = (time.perf_counter_ns() - decode_started) / 1e6
                commit = commit_decoded_joint_actions(
                    self.ledger,
                    request_ids,
                    footprints,
                    decode.actions,
                    expected_version=decode.snapshot_version,
                )
                if commit.get("committed"):
                    break
                if commit.get("reason") != "version_mismatch":
                    raise RuntimeError(
                        f"central batch commit failed: {commit.get('reason')}"
                    )
                if retry_count >= self.max_version_retries:
                    raise RuntimeError(
                        "central batch commit exceeded version retry budget"
                    )
                retry_count += 1
                self.stats["version_retries"] += 1

            for request, row in zip(requests, commit["results"]):
                if row.get("accepted"):
                    leave_time = request.get("leave_time")
                    if leave_time is not None:
                        self.register_lifetime(int(request["id"]), float(leave_time))
            plans: list[Optional[Mapping[str, Any]]] = []
            for generation, action, row in zip(generations, decode.actions, commit["results"]):
                plans.append(
                    generation.payloads[int(action)]
                    if row.get("accepted") and int(action) < len(generation.payloads)
                    else None
                )
            total_ms = (time.perf_counter_ns() - started) / 1e6
            self.stats["batches"] += 1
            self.stats["requests"] += len(requests)
            self.stats["accepted"] += int(commit["accepted"])
            self.stats["worker_errors"] += len(last_errors)
            self.stats["decoder_timeouts"] += int(decode.timed_out)
            return ParallelBatchResult(
                request_ids=request_ids,
                actions=tuple(map(int, decode.actions)),
                accepted=int(commit["accepted"]),
                rejected=len(requests) - int(commit["accepted"]),
                plans=tuple(plans),
                commit=commit,
                decoder=decode.to_dict(),
                timing_ms={
                    "candidate_generation_ms": float(generation_ms),
                    "decode_ms": float(decode_ms),
                    "total_ms": float(total_ms),
                },
                retry_count=retry_count,
                worker_errors=last_errors,
            )

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "central_shared_ledger_parallel_candidate_pipeline",
            "worker_count": self.worker_count,
            "top_r": self.top_r,
            "decoder_time_budget_ms": self.decoder_time_budget_ms,
            "max_version_retries": self.max_version_retries,
            "ranker": type(self.ranker).__name__ if self.ranker is not None else None,
            "stats": dict(self.stats),
            "ledger_version": int(self.ledger.snapshot().version),
            "ledger_integrity": self.ledger.integrity_report(),
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
