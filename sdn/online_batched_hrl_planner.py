"""Live micro-batch planner using frozen HRL policy heads and an atomic ledger."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from core.marl.batched_hrl_adapter import BatchedHRLCandidateAdapter
from core.marl.online_parallel_pipeline import CentralSharedLedgerPipeline
from core.marl.orchestration.hrl_adapter import build_ledger_from_profile


class OnlineBatchedHRLPlanner:
    """Plan arrived requests without running the legacy environment rollout.

    The frozen HRL encoder/high/low policy heads score complete plans generated
    against one current resource snapshot.  A bounded joint decoder and the
    versioned ledger make the final admission decision.  This is deliberately
    reported as policy scoring, not as a vectorized legacy rollout.
    """

    def __init__(
        self,
        planner: Any,
        profile: Mapping[str, Any],
        *,
        microbatch_ms: float = 5.0,
        max_agents: int = 32,
        top_k: int = 4,
        worker_count: int = 4,
        decoder_time_budget_ms: float = 2.0,
        cpu_capacity: float = 55.0,
        memory_capacity: float = 45.0,
        bandwidth_utilization_limit: float = 1.0,
    ) -> None:
        self.microbatch_ms = float(microbatch_ms)
        self.microbatch_seconds = self.microbatch_ms / 1000.0
        self.max_agents = int(max_agents)
        self.planned_requests = 0
        self.rejected_requests = 0
        self.batch_timings_ms: list[float] = []
        self.adapter = BatchedHRLCandidateAdapter(
            planner,
            profile,
            max_candidates=int(top_k),
            placement_beam=2,
            placement_chains=2,
            pool_limit=8,
            cpu_capacity=float(cpu_capacity),
            memory_capacity=float(memory_capacity),
        )
        self.adapter.prewarm_paths(max_paths=4)
        self.ledger = build_ledger_from_profile(
            profile,
            cpu_capacity=float(cpu_capacity),
            memory_capacity=float(memory_capacity),
            bandwidth_utilization_limit=float(bandwidth_utilization_limit),
        )
        self.pipeline = CentralSharedLedgerPipeline(
            self.ledger,
            self.adapter.candidate_factory,
            ranker=self.adapter.ranker,
            worker_count=int(worker_count),
            top_r=int(top_k),
            decoder_time_budget_ms=float(decoder_time_budget_ms),
        )

    @staticmethod
    def _reject_plan(request: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {
            "version": "batched_hrl_sfc_plan_v1",
            "request_id": int(request["id"]),
            "accepted": False,
            "reason": str(reason),
        }

    def plan_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        if len(requests) > self.max_agents:
            raise ValueError("live HRL micro-batch exceeds max_agents")
        started = time.perf_counter_ns()
        decision_time = max(float(row.get("arrival_time", 0.0)) for row in requests)
        result = self.pipeline.submit_batch(requests, timestamp=decision_time)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.batch_timings_ms.append(elapsed_ms)
        accepted_by_id = {
            int(row["request_id"]): bool(row.get("accepted"))
            for row in result.commit.get("results", [])
        }
        action_by_id = dict(zip(result.request_ids, result.actions))
        plan_by_id = dict(zip(result.request_ids, result.plans))
        plans: list[dict[str, Any]] = []
        for request in requests:
            request_id = int(request["id"])
            payload = plan_by_id.get(request_id)
            if not accepted_by_id.get(request_id, False) or payload is None:
                plans.append(self._reject_plan(request, "joint_decoder_rejected"))
                self.rejected_requests += 1
                continue
            plan = dict(payload)
            plan["accepted"] = True
            plan["request_id"] = request_id
            plan["online_planning"] = {
                "mode": "batched_hrl_policy_scoring",
                "rollout_vectorized": False,
                "candidate_action": int(action_by_id[request_id]),
                "batch_size": len(requests),
                "candidate_generation_ms": result.timing_ms["candidate_generation_ms"],
                "decoder_ms": result.timing_ms["decode_ms"],
                "total_ms": result.timing_ms["total_ms"],
            }
            plans.append(plan)
        self.planned_requests += len(requests)
        return plans

    def plan_next(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.plan_batch([request])[0]

    def release(self, request_id: int) -> bool:
        return self.pipeline.release(int(request_id))

    def replace_allocation(
        self,
        request_id: int,
        footprint,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        return self.pipeline.replace_allocation(
            int(request_id),
            footprint,
            expected_version=expected_version,
        )

    def metadata(self) -> dict[str, Any]:
        ordered = sorted(self.batch_timings_ms)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None
        return {
            "mode": "live_batched_hrl_policy_scoring",
            "rollout_vectorized": False,
            "microbatch_ms": self.microbatch_ms,
            "max_agents": self.max_agents,
            "planned_requests": self.planned_requests,
            "rejected_requests": self.rejected_requests,
            "mean_batch_ms": (
                sum(self.batch_timings_ms) / len(self.batch_timings_ms)
                if self.batch_timings_ms else None
            ),
            "p95_batch_ms": p95,
            "adapter": self.adapter.metadata(),
            "pipeline": self.pipeline.metadata(),
        }

    def close(self) -> None:
        self.pipeline.close()
        self.adapter.close()


__all__ = ["OnlineBatchedHRLPlanner"]
