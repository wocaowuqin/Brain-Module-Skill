"""Shared reward semantics for VNF-migration policies and baselines."""

from __future__ import annotations

from typing import Any, Mapping


def get_remaining_lifetime(
    request_id: int,
    requests: Mapping[int, Mapping[str, Any]] | None = None,
    now: float = 0.0,
    *,
    default: float = 0.0,
) -> float:
    """Return seconds until a request leaves, without inventing a horizon."""
    if requests is None:
        return float(default)
    row = requests.get(int(request_id))
    if row is None:
        return float(default)
    try:
        return max(0.0, float(row["leave_time"]) - float(now))
    except (KeyError, TypeError, ValueError, OverflowError):
        return float(default)


def migration_action_reward(agent: Mapping[str, Any], action: int) -> float:
    """Score one migration action using the runtime candidate metrics.

    The explicit reject action means ``noop`` for migration.  Keeping this
    function outside the environment prevents training and evaluation scripts
    from silently using different reward definitions.
    """

    reject = int(agent.get("reject_action", 0))
    task = agent.get("task") or {}
    if int(action) == reject:
        return 0.0

    candidates = agent.get("candidates") or []
    index = int(action)
    if index < 0 or index >= len(candidates):
        return -1.0
    candidate = candidates[index]
    metrics = candidate.get("metrics", candidate) if isinstance(candidate, Mapping) else {}
    remaining = max(0.0, float(task.get("remaining_lifetime_s", agent.get("remaining_lifetime_s", 0.0))))
    migration_s = float(metrics.get("migration_ms", task.get("estimated_migration_ms", 0.0))) / 1000.0
    if remaining <= migration_s + 1e-9:
        return -abs(float(metrics.get("migration_cost", migration_s)))
    horizon = max(0.0, float(metrics.get("horizon_s", agent.get("horizon_s", 1.0))))
    reduction = metrics.get("future_sla_loss_reduction")
    if reduction is None:
        reduction = float(metrics.get("future_sla_loss_before", metrics.get("sla_loss_before", task.get("sla_risk", 0.0)))) - float(metrics.get("future_sla_loss_after", metrics.get("sla_loss_after", max(0.0, float(metrics.get("delay_ratio", 0.0)) - 1.0))))
    benefit = float(reduction) * min(remaining, horizon)
    cost = float(metrics.get("migration_cost", metrics.get("migration_ms", 0.0) * 0.002))
    reward = benefit - cost
    if int(metrics.get("migration_count_window", task.get("migration_count_window", 0))) > 2:
        reward *= 0.5
    return float(reward)


def migration_action_rewards(agent: Mapping[str, Any]) -> list[float]:
    """Return rewards aligned with every serialized candidate action."""

    return [
        migration_action_reward(agent, action)
        for action in range(len(agent["candidates"]))
    ]
