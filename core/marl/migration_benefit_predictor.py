"""Predict whether a proposed VNF migration beats keeping the current SFT.

The model is deliberately a gate, not an executor.  It consumes only state
available before migration and predicts a label produced by paired
counterfactual rollouts: migrate versus noop over the same future arrivals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


MODEL_VERSION = "migration_benefit_mlp_v1"
PREDICTION_TARGET = "migration_improves_future_horizon_utility_over_noop"

FEATURE_NAMES = (
    "risk_score",
    "link_hot_score",
    "node_hot_score",
    "modeled_delay_ms",
    "migration_count",
    "reroute_count",
    "active_sfts",
    "node_hotspots",
    "link_hotspots",
    "mean_active_delay_ms",
    "max_active_delay_ms",
    "tree_edges",
    "vnf_stages",
    "source_node_utilization",
    "target_node_utilization",
    "projected_target_utilization",
    "estimated_peak_relief",
    "cpu_units",
    "memory_units",
    "remaining_lifetime_s",
)


class MigrationBenefitNet(nn.Module):
    """Small CPU-friendly binary classifier for the online migration gate."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            width = int(width)
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass(frozen=True)
class MigrationBenefitEstimate:
    probability: float
    allowed: bool
    threshold: float
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "allowed": self.allowed,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def _value(source: Any, name: str, default: float = 0.0) -> float:
    if isinstance(source, Mapping):
        raw = source.get(name, default)
    else:
        raw = getattr(source, name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _active_request_info(manager: Any, req_id: int) -> Mapping[str, Any]:
    lifecycle = getattr(manager.rm, "request_manager", None)
    active = getattr(lifecycle, "active_requests", {}) if lifecycle is not None else {}
    return active.get(req_id) or active.get(str(req_id)) or {}


def build_migration_benefit_features(
    manager: Any,
    risk: Any,
    proposal: Any,
) -> list[float]:
    """Build the leakage-free feature vector used by training and inference."""

    payload = proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal or {})
    target = dict(payload.get("target") or {})
    req_id = int(payload.get("req_id", _value(risk, "req_id", -1)))
    record = manager.rm.request_table.get(req_id)
    if record is None:
        raise ValueError(f"migration request {req_id} is not active")

    metrics = manager.metrics_snapshot()
    old_node = int(target.get("old_node", -1))
    new_node = int(target.get("new_node", -1))
    source_util = manager._node_utilization(old_node) if old_node >= 0 else 1.0
    target_util = manager._node_utilization(new_node) if new_node >= 0 else 1.0
    lifecycle = _active_request_info(manager, req_id)
    now = float(getattr(manager.env, "time_step", 0.0) or 0.0)
    expire_time = float(lifecycle.get("expire_time", now))
    remaining = max(0.0, expire_time - now)

    values = {
        "risk_score": _value(risk, "score"),
        "link_hot_score": _value(risk, "link_hot_score"),
        "node_hot_score": _value(risk, "node_hot_score"),
        "modeled_delay_ms": _value(risk, "delay_estimate"),
        "migration_count": _value(record, "migration_count"),
        "reroute_count": _value(record, "reconfig_count"),
        "active_sfts": _value(metrics, "active_sfts"),
        "node_hotspots": _value(metrics, "node_hotspots"),
        "link_hotspots": _value(metrics, "link_hotspots"),
        "mean_active_delay_ms": _value(metrics, "avg_delay_total_ms"),
        "max_active_delay_ms": _value(metrics, "max_delay_total_ms"),
        "tree_edges": float(len(record.tree_edges)),
        "vnf_stages": float(len(record.placement_by_vnf)),
        "source_node_utilization": source_util,
        "target_node_utilization": target_util,
        "projected_target_utilization": _value(
            target, "projected_target_utilization", target_util
        ),
        "estimated_peak_relief": _value(payload, "estimated_gain"),
        "cpu_units": _value(target, "cpu"),
        "memory_units": _value(target, "mem"),
        "remaining_lifetime_s": remaining,
    }
    return [float(values[name]) for name in FEATURE_NAMES]


class MigrationBenefitPredictor:
    """Read-only predictor used to reject migrations without expected benefit."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        path: str | Path | None = None,
        *,
        threshold: float | None = None,
    ) -> None:
        if str(payload.get("version")) != MODEL_VERSION:
            raise ValueError("unsupported migration benefit predictor version")
        if str(payload.get("prediction_target")) != PREDICTION_TARGET:
            raise ValueError("migration predictor has an incompatible target")
        if tuple(payload.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError("migration predictor feature schema does not match runtime")
        if not bool(payload.get("deployment_allowed", False)):
            raise ValueError("migration predictor did not pass offline deployment criteria")

        self.path = Path(path).resolve() if path is not None else None
        self.hidden_dims = tuple(map(int, payload.get("hidden_dims") or (32, 16)))
        self.dropout = float(payload.get("dropout", 0.0))
        normalization = payload.get("normalization") or {}
        self.mean = torch.tensor(normalization.get("mean"), dtype=torch.float32)
        self.std = torch.tensor(normalization.get("std"), dtype=torch.float32)
        if self.mean.numel() != len(FEATURE_NAMES) or self.std.numel() != len(FEATURE_NAMES):
            raise ValueError("migration predictor normalization has invalid length")
        if torch.any(self.std <= 0.0):
            raise ValueError("migration predictor normalization has non-positive scale")

        self.model = MigrationBenefitNet(
            len(FEATURE_NAMES), self.hidden_dims, self.dropout
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        calibration = payload.get("calibration") or {}
        self.logit_scale = float(calibration.get("logit_scale", 1.0))
        self.logit_bias = float(calibration.get("logit_bias", 0.0))
        configured = float(payload.get("decision_threshold", 0.5))
        self.threshold = configured if threshold is None else float(threshold)
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("migration probability threshold must be between zero and one")
        self.training = dict(payload.get("training") or {})
        self.validation = dict(payload.get("validation") or {})

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        threshold: float | None = None,
    ) -> "MigrationBenefitPredictor":
        resolved = Path(path).resolve()
        try:
            payload = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(resolved, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("migration predictor checkpoint is malformed")
        return cls(payload, resolved, threshold=threshold)

    def predict_features(self, features: Sequence[float]) -> MigrationBenefitEstimate:
        raw = torch.tensor(list(features), dtype=torch.float32)
        if raw.numel() != len(FEATURE_NAMES) or not torch.isfinite(raw).all():
            raise ValueError("migration predictor received invalid features")
        standardized = (raw - self.mean) / self.std
        with torch.inference_mode():
            logit = self.model(standardized.unsqueeze(0))[0]
            probability = float(torch.sigmoid(self.logit_scale * logit + self.logit_bias))
        ood_score = float(standardized.abs().max())
        confidence = float(torch.exp(torch.tensor(-max(0.0, ood_score - 2.0))))
        allowed = probability >= self.threshold and confidence >= 0.25
        if confidence < 0.25:
            reason = "out-of-distribution state; keep current placement"
        elif allowed:
            reason = "predicted migration benefit exceeds threshold"
        else:
            reason = "noop has greater predicted future utility"
        return MigrationBenefitEstimate(
            probability=probability,
            allowed=allowed,
            threshold=self.threshold,
            confidence=confidence,
            reason=reason,
        )

    def predict(self, manager: Any, risk: Any, proposal: Any) -> MigrationBenefitEstimate:
        return self.predict_features(
            build_migration_benefit_features(manager, risk, proposal)
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "prediction_target": PREDICTION_TARGET,
            "path": str(self.path) if self.path is not None else None,
            "threshold": self.threshold,
            "training": self.training,
            "validation": self.validation,
        }
