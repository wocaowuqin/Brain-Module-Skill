"""Prediction-based overload screening for the live deployment ledger."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

from core.marl.batch_deployment_wqmix import ResourceSnapshot
from core.marl.migration_scheduler import EWMATrendForecaster


class OnlineMigrationMonitor:
    """Return the minimum useful set of internal VNF stages to migrate."""

    def __init__(
        self,
        cpu_capacity: float,
        memory_capacity: float,
        *,
        max_agents: int = 8,
        overload_threshold: float = 0.85,
        safe_utilization: float = 0.75,
        recovery_threshold: float = 0.70,
        prediction_horizon_s: float = 1.0,
        cooldown_s: float = 5.0,
        thrashing_window_s: float = 30.0,
        minimum_remaining_lifetime_s: float = 1.344,
    ) -> None:
        self.cpu_capacity = float(cpu_capacity)
        self.memory_capacity = float(memory_capacity)
        self.max_agents = int(max_agents)
        self.overload_threshold = float(overload_threshold)
        self.safe_utilization = float(safe_utilization)
        self.recovery_threshold = float(recovery_threshold)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.cooldown_s = float(cooldown_s)
        self.thrashing_window_s = max(0.0, float(thrashing_window_s))
        self.minimum_remaining_lifetime_s = float(minimum_remaining_lifetime_s)
        self.forecaster = EWMATrendForecaster()
        self.overloaded: set[int] = set()
        self.last_migration: dict[tuple[int, int], float] = {}
        self.migration_history: dict[tuple[int, int], list[float]] = defaultdict(list)
        self.scans = 0
        self.triggers = 0
        self.short_lifetime_skips = 0

    def _utilizations(self, snapshot: ResourceSnapshot) -> dict[int, float]:
        nodes = set(snapshot.cpu_remaining) | set(snapshot.memory_remaining)
        return {
            int(node): max(
                0.0,
                1.0 - float(snapshot.cpu_remaining.get(node, 0.0))
                / max(1e-9, self.cpu_capacity),
                1.0 - float(snapshot.memory_remaining.get(node, 0.0))
                / max(1e-9, self.memory_capacity),
            )
            for node in nodes
        }

    def scan(
        self,
        snapshot: ResourceSnapshot,
        active_plans: Mapping[int, Mapping[str, Any]],
        requests: Mapping[int, Mapping[str, Any]],
        now: float,
    ) -> list[dict[str, Any]]:
        self.scans += 1
        utilization = self._utilizations(snapshot)
        self.forecaster.update(now, utilization)
        predicted_utilization = {
            node: self.forecaster.predict(node, self.prediction_horizon_s)
            for node in utilization
        }
        for node, current in utilization.items():
            predicted = predicted_utilization[node]
            if max(current, predicted) >= self.overload_threshold:
                self.overloaded.add(node)
            elif max(current, predicted) <= self.recovery_threshold:
                self.overloaded.discard(node)

        by_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for request_id, plan in active_plans.items():
            request = requests.get(int(request_id))
            if request is None or not (
                float(request["arrival_time"]) <= now < float(request["leave_time"])
            ):
                continue
            remaining = float(request["leave_time"]) - now
            if remaining < self.minimum_remaining_lifetime_s:
                self.short_lifetime_skips += 1
                continue
            placements = plan.get("placement_by_vnf") or {}
            indices = sorted(map(int, placements))
            for stage in indices:
                if stage <= 0 or stage + 1 not in indices:
                    continue
                key = str(stage) if str(stage) in placements else stage
                placement = placements[key]
                node = int(placement["dc_node"])
                if node not in self.overloaded:
                    continue
                if now - self.last_migration.get((int(request_id), stage), -math.inf) < self.cooldown_s:
                    continue
                cpu = float(placement.get("cpu_units", 1.0))
                memory = float(placement.get("memory_units", 1.0))
                relief = max(cpu / self.cpu_capacity, memory / self.memory_capacity)
                online_planning = plan.get("online_planning") or {}
                raw_sla_risk = request.get(
                    "predicted_sla_risk",
                    online_planning.get("predicted_sla_risk", 0.0),
                )
                sla_risk = min(1.0, max(0.0, float(raw_sla_risk)))
                current = float(utilization[node])
                predicted = float(predicted_utilization[node])
                enriched_request = dict(request)
                enriched_request.update({
                    "current_utilization": current,
                    "predicted_utilization": predicted,
                    "predicted_sla_risk": sla_risk,
                })
                by_node[node].append({
                    "request": enriched_request,
                    "current_plan": plan,
                    "stage": stage,
                    "old_node": node,
                    "current_utilization": current,
                    "predicted_utilization": predicted,
                    "predicted_sla_risk": sla_risk,
                    "relief": relief,
                    "priority": (1.0 + 2.0 * sla_risk) * relief,
                    "migration_count_window": self.migration_count(
                        int(request_id), stage, now
                    ),
                })

        selected: list[dict[str, Any]] = []
        for node in sorted(self.overloaded):
            predicted = predicted_utilization.get(
                node, self.forecaster.predict(node, self.prediction_horizon_s)
            )
            required = max(0.0, predicted - self.safe_utilization)
            accumulated = 0.0
            candidates = sorted(
                by_node.get(node, []),
                key=lambda row: (
                    -float(row["priority"]),
                    int(row["request"]["id"]),
                    int(row["stage"]),
                ),
            )
            for row in candidates:
                selected.append(row)
                accumulated += float(row["relief"])
                if accumulated + 1e-9 >= required or len(selected) >= self.max_agents:
                    break
            if len(selected) >= self.max_agents:
                break
        self.triggers += len(selected)
        return selected

    def mark_migrated(self, request_id: int, stage: int, now: float) -> None:
        key = (int(request_id), int(stage))
        timestamp = float(now)
        self.last_migration[key] = timestamp
        history = self.migration_history[key]
        history.append(timestamp)
        cutoff = timestamp - self.thrashing_window_s
        self.migration_history[key] = [value for value in history if value >= cutoff]

    def migration_count(
        self, request_id: int, stage: int, now: float, window_s: float | None = None
    ) -> int:
        """Count moves for one VNF in the active anti-thrashing window."""
        window = self.thrashing_window_s if window_s is None else max(0.0, float(window_s))
        cutoff = float(now) - window
        key = (int(request_id), int(stage))
        history = self.migration_history.get(key, [])
        self.migration_history[key] = [value for value in history if value >= cutoff]
        return len(self.migration_history[key])

    def metadata(self) -> dict[str, Any]:
        return {
            "scans": self.scans,
            "triggers": self.triggers,
            "overloaded_nodes": sorted(self.overloaded),
            "tracked_migrations": len(self.last_migration),
            "migration_history_events": sum(len(v) for v in self.migration_history.values()),
            "minimum_remaining_lifetime_s": self.minimum_remaining_lifetime_s,
            "thrashing_window_s": self.thrashing_window_s,
            "short_lifetime_skips": self.short_lifetime_skips,
        }
