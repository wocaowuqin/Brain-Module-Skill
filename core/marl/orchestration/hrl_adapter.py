"""Bridge the stateful legacy HRL planner into the pure MARL pipeline.

The legacy planner owns a simulated ledger and therefore cannot safely run as
the worker function of a parallel candidate pipeline.  This adapter calls it
once, in request order, extracts the complete plan, releases the planner's
temporary reservation, and then lets the central shared-ledger pipeline
rebuild and validate the plan against its authoritative snapshot.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

from core.marl.batch_deployment_wqmix import AtomicResourceLedger
from core.marl.deployment_topk import CompletePlanCandidateGenerator
from core.marl.online_parallel_pipeline import (
    CandidateGeneration,
    CandidateFactory,
    complete_plan_candidate_factory,
)


@dataclass(frozen=True)
class HRLPreparationReport:
    """Auditable result of the stateful HRL preparation pass."""

    request_ids: tuple[int, ...]
    accepted: int
    rejected: int
    released: int
    planner_type: str
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_ids": list(self.request_ids),
            "accepted": int(self.accepted),
            "rejected": int(self.rejected),
            "released": int(self.released),
            "planner_type": self.planner_type,
            "elapsed_ms": float(self.elapsed_ms),
        }


class HRLPlannerAdapter:
    """Use a real HRL checkpoint as the first candidate for each request.

    ``planner`` must expose the existing ``plan_next`` and ``release`` API.
    The adapter deliberately keeps HRL plans in memory only; it never writes a
    candidate file and never mutates the central :class:`AtomicResourceLedger`.
    """

    def __init__(
        self,
        planner: Any,
        profile: Mapping[str, Any],
        *,
        max_candidates: int = 8,
        placement_beam: int = 16,
        placement_chains: int = 12,
        pool_limit: int = 64,
        stage_port_base: int = 20000,
        generator_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(getattr(planner, "plan_next", None)):
            raise TypeError("planner must expose plan_next(request)")
        self.planner = planner
        kwargs = dict(generator_kwargs or {})
        self.generator = CompletePlanCandidateGenerator(
            profile,
            max_candidates=int(max_candidates),
            placement_beam=int(placement_beam),
            placement_chains=int(placement_chains),
            pool_limit=int(pool_limit),
            stage_port_base=int(stage_port_base),
            **kwargs,
        )
        self._baseline_plans: dict[int, Mapping[str, Any]] = {}
        self._last_report: HRLPreparationReport | None = None
        self._factory: CandidateFactory = complete_plan_candidate_factory(
            self.generator,
            baseline_plan_provider=lambda request: self._baseline_plans.get(
                int(request["id"])
            ),
        )

    def prewarm_paths(self, max_paths: int = 8) -> None:
        self.generator.prewarm_paths(max_paths=max_paths)

    def prepare_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> HRLPreparationReport:
        """Run HRL once per request, then remove its temporary reservation.

        Calls are intentionally sequential because the legacy HRL environment
        advances an episode cursor and its own lifecycle ledger.  The result is
        later revalidated by the shared ledger, so a stale or invalid HRL plan
        is simply excluded by the candidate mask.
        """

        ids = tuple(int(request["id"]) for request in requests)
        if len(set(ids)) != len(ids):
            raise ValueError("request ids in an HRL preparation batch must be unique")
        started = time.perf_counter_ns()
        accepted = rejected = released = 0
        for request in requests:
            request_id = int(request["id"])
            plan = self.planner.plan_next(dict(request))
            if not isinstance(plan, Mapping):
                raise TypeError("plan_next must return a mapping")
            plan_copy = deepcopy(dict(plan))
            if bool(plan_copy.get("accepted", False)):
                self._baseline_plans[request_id] = plan_copy
                accepted += 1
            else:
                self._baseline_plans.pop(request_id, None)
                rejected += 1
            release = getattr(self.planner, "release", None)
            if bool(plan_copy.get("accepted", False)) and callable(release):
                released += int(bool(release(request_id)))
        self._last_report = HRLPreparationReport(
            request_ids=ids,
            accepted=accepted,
            rejected=rejected,
            released=released,
            planner_type=type(self.planner).__name__,
            elapsed_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
        )
        return self._last_report

    def prefetch_batch(self, requests: Sequence[Mapping[str, Any]]) -> int:
        """Schedule the next ordered HRL pass while the current batch commits."""

        prefetch = getattr(self.planner, "prefetch", None)
        if not callable(prefetch):
            return 0
        return int(prefetch([dict(request) for request in requests]))

    def candidate_factory(
        self, request: Mapping[str, Any], snapshot: Any
    ) -> CandidateGeneration:
        """Generate pure candidates against the supplied immutable snapshot."""

        return self._factory(request, snapshot)

    def forget_batch(self, requests: Sequence[Mapping[str, Any]]) -> None:
        """Drop copied HRL plans after the central transaction has committed."""

        for request in requests:
            self._baseline_plans.pop(int(request["id"]), None)

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "stateful_hrl_baseline_plus_pure_topk_candidates",
            "planner": type(self.planner).__name__,
            "generator": type(self.generator).__name__,
            "max_candidates": int(self.generator.max_candidates),
            "prewarmed_paths": len(self.generator.cached_paths),
            "last_preparation": (
                self._last_report.to_dict() if self._last_report is not None else None
            ),
            "baseline_plan_count": len(self._baseline_plans),
        }

    def close(self) -> None:
        close = getattr(self.planner, "close", None)
        if callable(close):
            close()


def build_ledger_from_profile(
    profile: Mapping[str, Any],
    *,
    cpu_capacity: float = 55.0,
    memory_capacity: float = 45.0,
    bandwidth_utilization_limit: float = 1.0,
) -> AtomicResourceLedger:
    """Build the central ledger using the profile's directed edge capacities."""

    if not 0.0 < float(bandwidth_utilization_limit) <= 1.0:
        raise ValueError("bandwidth_utilization_limit must be in (0, 1]")
    default_bw = float(profile.get("default_bandwidth_mbps", 90.0))
    bandwidth: dict[tuple[int, int], float] = {}
    for raw in profile["edges"]:
        u, v = int(raw["u"]), int(raw["v"])
        cap = float(raw.get("bandwidth_mbps", default_bw)) * float(
            bandwidth_utilization_limit
        )
        bandwidth[(u, v)] = cap
        bandwidth[(v, u)] = cap
    dc_nodes = [int(value) for value in profile["dc_nodes_1based"]]
    return AtomicResourceLedger(
        {node: float(cpu_capacity) for node in dc_nodes},
        {node: float(memory_capacity) for node in dc_nodes},
        bandwidth,
    )
