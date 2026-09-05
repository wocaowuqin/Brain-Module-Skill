"""Supervised strict-SLA risk prediction from pre-deployment state.

The prediction target is receiver-level strict SLA failure conditional on a
sender that started successfully.  Every feature in this module is available
before committing a candidate, which prevents runtime-measurement leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from core.marl.batch_deployment_wqmix import ResourceFootprint, ResourceSnapshot
from core.marl.sla_risk_calibration import SlaRiskEstimate


MODEL_VERSION = "runtime_sla_risk_mlp_v2"
PREDICTION_TARGET = "request_strict_sla_failure_given_sender_started"
LEGACY_MODEL_TARGETS = {
    "runtime_sla_risk_mlp_v1": "receiver_strict_sla_failure_given_sender_started",
}

FEATURE_NAMES = (
    "modeled_delay_ratio",
    "bandwidth_mbps",
    "destination_count",
    "vnf_count",
    "cpu_demand_total",
    "memory_demand_total",
    "lifetime_seconds",
    "delay_bound_ms",
    "jitter_bound_ms",
    "packet_loss_bound_log10",
    "qos_realtime",
    "qos_interactive",
    "qos_elastic",
    "segment_hops",
    "tree_edges",
    "max_receiver_hops",
    "mean_receiver_hops",
    "unique_directed_edges",
    "edge_traversals",
    "new_vnf_instances",
    "reused_vnf_stages",
    "active_requests",
    "batch_size",
    "recent_arrival_rate_1s",
    "max_current_link_utilization",
    "mean_current_candidate_link_utilization",
    "max_projected_link_utilization",
    "mean_projected_candidate_link_utilization",
    "p95_projected_candidate_link_utilization",
    "minimum_projected_link_headroom",
    "bandwidth_footprint_total",
    "peak_bandwidth_demand_to_remaining",
)


class RuntimeSlaRiskNet(nn.Module):
    """Small CPU-friendly MLP used for online candidate scoring."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16),
        dropout: float = 0.1,
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


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(map(float, values))
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _qos_class(request: Mapping[str, Any]) -> str:
    value = request.get("qos_class")
    if value:
        return str(value)
    return {3: "Q0_REALTIME", 2: "Q1_INTERACTIVE", 1: "Q2_ELASTIC"}.get(
        int(request.get("priority", -1)), "unknown"
    )


def _plan_geometry(plan: Mapping[str, Any]) -> dict[str, float]:
    segments = plan.get("segments") or []
    multicast = plan.get("multicast") or {}
    receiver_paths = list((multicast.get("paths") or {}).values())
    tree_edges = multicast.get("tree_edges") or []
    segment_hops = sum(max(0, len(row.get("path") or []) - 1) for row in segments)
    receiver_hops = [max(0, len(path or []) - 1) for path in receiver_paths]
    return {
        "segment_hops": float(segment_hops),
        "tree_edges": float(len(tree_edges)),
        "max_receiver_hops": float(max(receiver_hops, default=0)),
        "mean_receiver_hops": float(
            sum(receiver_hops) / max(1, len(receiver_hops))
        ),
    }


def build_runtime_sla_features(
    request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    footprint: ResourceFootprint,
    snapshot: ResourceSnapshot,
    bandwidth_capacity: Mapping[tuple[int, int], float],
    *,
    modeled_delay_ratio: float,
    batch_size: int,
    active_request_count: int,
    recent_arrival_rate_1s: float,
) -> list[float]:
    """Build the leakage-free feature vector used in training and inference."""

    plan = candidate.get("plan") or candidate
    geometry = _plan_geometry(plan)
    qos = _qos_class(request)
    projected: list[float] = []
    current_on_candidate: list[float] = []
    all_current: list[float] = []
    demand_to_remaining: list[float] = []
    for edge, capacity_value in bandwidth_capacity.items():
        capacity = max(float(capacity_value), 1e-9)
        remaining = float(snapshot.bandwidth_remaining.get(edge, capacity))
        current = max(0.0, (capacity - remaining) / capacity)
        all_current.append(current)
        if edge not in footprint.bandwidth:
            continue
        demand = float(footprint.bandwidth[edge])
        current_on_candidate.append(current)
        projected.append(max(0.0, (capacity - remaining + demand) / capacity))
        demand_to_remaining.append(demand / max(remaining, 1e-9))

    requirements = tuple(footprint.vnf_instances)
    new_instances: set[tuple[int, int]] = set()
    reused = 0
    for requirement in requirements:
        key = (int(requirement.node), int(requirement.vnf_type))
        if key in snapshot.vnf_instances or key in new_instances:
            reused += 1
        else:
            new_instances.add(key)

    bandwidth = float(request.get("bw_origin", request.get("bandwidth_mbps", 0.0)))
    destinations = request.get("destination_dpids") or request.get("destinations") or []
    vnf_types = request.get("vnf") or []
    cpu = request.get("cpu_origin") or []
    memory = request.get("memory_origin") or []
    loss_bound = max(float(request.get("packet_loss_bound") or 0.0), 1e-9)
    values = {
        "modeled_delay_ratio": max(0.0, float(modeled_delay_ratio)),
        "bandwidth_mbps": bandwidth,
        "destination_count": float(len(destinations)),
        "vnf_count": float(len(vnf_types)),
        "cpu_demand_total": float(sum(map(float, cpu))),
        "memory_demand_total": float(sum(map(float, memory))),
        "lifetime_seconds": max(0.0, float(request.get("lifetime") or 0.0)),
        "delay_bound_ms": max(0.0, float(request.get("delay_bound_ms") or 0.0)),
        "jitter_bound_ms": max(0.0, float(request.get("jitter_bound_ms") or 0.0)),
        "packet_loss_bound_log10": math.log10(loss_bound),
        "qos_realtime": float(qos == "Q0_REALTIME"),
        "qos_interactive": float(qos == "Q1_INTERACTIVE"),
        "qos_elastic": float(qos == "Q2_ELASTIC"),
        **geometry,
        "unique_directed_edges": float(len(footprint.bandwidth)),
        "edge_traversals": float(
            sum(float(amount) / max(bandwidth, 1e-9) for amount in footprint.bandwidth.values())
        ),
        "new_vnf_instances": float(len(new_instances)),
        "reused_vnf_stages": float(reused),
        "active_requests": float(max(0, active_request_count)),
        "batch_size": float(max(1, batch_size)),
        "recent_arrival_rate_1s": max(0.0, float(recent_arrival_rate_1s)),
        "max_current_link_utilization": max(all_current, default=0.0),
        "mean_current_candidate_link_utilization": float(
            sum(current_on_candidate) / max(1, len(current_on_candidate))
        ),
        "max_projected_link_utilization": max(projected, default=0.0),
        "mean_projected_candidate_link_utilization": float(
            sum(projected) / max(1, len(projected))
        ),
        "p95_projected_candidate_link_utilization": _percentile(projected, 0.95),
        "minimum_projected_link_headroom": max(
            0.0, 1.0 - max(projected, default=0.0)
        ),
        "bandwidth_footprint_total": float(sum(footprint.bandwidth.values())),
        "peak_bandwidth_demand_to_remaining": max(demand_to_remaining, default=0.0),
    }
    return [float(values[name]) for name in FEATURE_NAMES]


class RuntimeSlaRiskPredictor:
    """Read-only supervised probability predictor for online planning."""

    def __init__(self, payload: Mapping[str, Any], path: str | Path | None = None) -> None:
        version = str(payload.get("version") or "")
        prediction_target = str(payload.get("prediction_target") or "")
        supported_target = (
            PREDICTION_TARGET
            if version == MODEL_VERSION
            else LEGACY_MODEL_TARGETS.get(version)
        )
        if supported_target is None:
            raise ValueError(f"unsupported SLA predictor version: {version}")
        if prediction_target != supported_target:
            raise ValueError("SLA predictor has an incompatible prediction target")
        self.version = version
        self.prediction_target = prediction_target
        names = tuple(payload.get("feature_names") or ())
        if names != FEATURE_NAMES:
            raise ValueError("SLA predictor feature schema does not match this runtime")
        self.path = Path(path).resolve() if path is not None else None
        self.hidden_dims = tuple(map(int, payload.get("hidden_dims") or (32, 16)))
        self.dropout = float(payload.get("dropout", 0.0))
        self.mean = torch.tensor(payload["normalization"]["mean"], dtype=torch.float32)
        self.std = torch.tensor(payload["normalization"]["std"], dtype=torch.float32)
        if self.mean.numel() != len(FEATURE_NAMES) or self.std.numel() != len(FEATURE_NAMES):
            raise ValueError("SLA predictor normalization vector has invalid length")
        if torch.any(self.std <= 0.0) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("SLA predictor normalization is invalid")
        self.model = RuntimeSlaRiskNet(
            len(FEATURE_NAMES), self.hidden_dims, self.dropout
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        calibration = payload.get("calibration") or {}
        self.logit_scale = float(calibration.get("logit_scale", 1.0))
        self.logit_bias = float(calibration.get("logit_bias", 0.0))
        self.validation = dict(payload.get("validation") or {})
        self.training = dict(payload.get("training") or {})
        self.measurement_contract = dict(payload.get("measurement_contract") or {})

    @classmethod
    def from_path(cls, path: str | Path) -> "RuntimeSlaRiskPredictor":
        resolved = Path(path).resolve()
        try:
            payload = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(resolved, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("SLA predictor checkpoint is malformed")
        return cls(payload, resolved)

    def predict_features(
        self,
        features: Sequence[float],
        *,
        modeled_delay_ratio: float,
    ) -> SlaRiskEstimate:
        return self.predict_features_batch(
            [features], modeled_delay_ratios=[modeled_delay_ratio]
        )[0]

    def predict_features_batch(
        self,
        features: Sequence[Sequence[float]],
        *,
        modeled_delay_ratios: Sequence[float],
    ) -> list[SlaRiskEstimate]:
        """Score a candidate set with one tensor/model invocation."""

        if len(features) != len(modeled_delay_ratios):
            raise ValueError("SLA predictor feature and delay-ratio counts differ")
        if not features:
            return []
        raw = torch.tensor([list(row) for row in features], dtype=torch.float32)
        if (
            raw.ndim != 2
            or raw.shape[1] != len(FEATURE_NAMES)
            or not torch.isfinite(raw).all()
        ):
            raise ValueError("SLA predictor received invalid features")
        standardized = (raw - self.mean.unsqueeze(0)) / self.std.unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(standardized)
            calibrated = self.logit_scale * logits + self.logit_bias
            probabilities = torch.sigmoid(calibrated).tolist()
        estimates: list[SlaRiskEstimate] = []
        for probability, modeled_delay_ratio, row in zip(
            probabilities, modeled_delay_ratios, standardized.tolist()
        ):
            ood_score = float(max(map(abs, row), default=0.0))
            confidence = math.exp(-max(0.0, ood_score - 2.0))
            top = sorted(
                zip(FEATURE_NAMES, row),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:4]
            estimates.append(
                SlaRiskEstimate(
                    failure_probability=float(probability),
                    modeled_delay_ratio=float(modeled_delay_ratio),
                    effective_support=float(self.training.get("samples", 0)),
                    delay_ratio_bucket="supervised",
                    components=tuple(
                        {"name": name, "standardized_value": float(value)}
                        for name, value in top
                    ),
                    model_kind=self.version,
                    confidence=float(confidence),
                    ood_score=ood_score,
                )
            )
        return estimates

    def predict(
        self,
        request: Mapping[str, Any],
        candidate: Mapping[str, Any],
        footprint: ResourceFootprint,
        snapshot: ResourceSnapshot,
        bandwidth_capacity: Mapping[tuple[int, int], float],
        *,
        modeled_delay_ratio: float,
        batch_size: int,
        active_request_count: int,
        recent_arrival_rate_1s: float,
    ) -> SlaRiskEstimate:
        features = build_runtime_sla_features(
            request,
            candidate,
            footprint,
            snapshot,
            bandwidth_capacity,
            modeled_delay_ratio=modeled_delay_ratio,
            batch_size=batch_size,
            active_request_count=active_request_count,
            recent_arrival_rate_1s=recent_arrival_rate_1s,
        )
        return self.predict_features(features, modeled_delay_ratio=modeled_delay_ratio)

    def predict_many(
        self,
        requests: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        footprints: Sequence[ResourceFootprint],
        snapshot: ResourceSnapshot,
        bandwidth_capacity: Mapping[tuple[int, int], float],
        *,
        modeled_delay_ratios: Sequence[float],
        batch_size: int,
        active_request_count: int,
        recent_arrival_rate_1s: float,
    ) -> list[SlaRiskEstimate]:
        """Build and score all pre-deployment candidate features as one batch."""

        row_count = len(requests)
        if not (
            len(candidates)
            == len(footprints)
            == len(modeled_delay_ratios)
            == row_count
        ):
            raise ValueError("SLA predictor candidate batch dimensions differ")
        features = [
            build_runtime_sla_features(
                request,
                candidate,
                footprint,
                snapshot,
                bandwidth_capacity,
                modeled_delay_ratio=modeled_delay_ratio,
                batch_size=batch_size,
                active_request_count=active_request_count,
                recent_arrival_rate_1s=recent_arrival_rate_1s,
            )
            for request, candidate, footprint, modeled_delay_ratio in zip(
                requests, candidates, footprints, modeled_delay_ratios
            )
        ]
        return self.predict_features_batch(
            features, modeled_delay_ratios=modeled_delay_ratios
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prediction_target": self.prediction_target,
            "path": str(self.path) if self.path is not None else None,
            "feature_count": len(FEATURE_NAMES),
            "training": self.training,
            "validation": self.validation,
            "measurement_contract": self.measurement_contract or None,
            "calibration": {
                "logit_scale": self.logit_scale,
                "logit_bias": self.logit_bias,
            },
        }
