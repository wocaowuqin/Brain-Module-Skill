"""Shared exact-commit transaction for training and online deployment."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from core.marl.batch_deployment_wqmix import (
    AtomicResourceLedger,
    ResourceFootprint,
)
from core.marl.joint_candidate_decoder import joint_footprints_feasible


def commit_decoded_joint_actions(
    ledger: AtomicResourceLedger,
    request_ids: Sequence[int],
    candidates: Sequence[Sequence[Optional[ResourceFootprint]]],
    actions: Sequence[int],
    *,
    expected_version: int,
) -> dict[str, Any]:
    """Validate and atomically commit one decoder action vector.

    Version mismatch is returned so an online caller can refresh its snapshot
    and decode again. Every other invariant violation is a programming error.
    """

    if not (len(request_ids) == len(candidates) == len(actions)):
        raise ValueError("joint commit request, candidate, and action counts differ")
    selected: list[Optional[ResourceFootprint]] = []
    for index, (request_candidates, raw_action) in enumerate(zip(candidates, actions)):
        action = int(raw_action)
        if not 0 <= action < len(request_candidates):
            raise ValueError(f"invalid action {action} for request index {index}")
        selected.append(request_candidates[action])

    snapshot = ledger.snapshot()
    if int(snapshot.version) == int(expected_version):
        feasible, reason = joint_footprints_feasible(selected, snapshot)
        if not feasible:
            raise RuntimeError(
                "decoder action vector is infeasible before exact commit: "
                f"{reason}"
            )

    commit = ledger.commit_exact(
        [int(value) for value in request_ids],
        candidates,
        [int(value) for value in actions],
        expected_version=int(expected_version),
    )
    if not commit["committed"] and commit["reason"] != "version_mismatch":
        raise RuntimeError(
            "exact joint commit failed after feasibility validation: "
            f"{commit['reason']}"
        )
    return commit
