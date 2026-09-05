"""Empirical strict-SLA calibration from real SDN probe outcomes.

The deployment planner already computes a modeled end-to-end delay ratio for
every complete candidate.  This module calibrates that ratio against strict
Mininet/Ryu probe outcomes without changing the resource feasibility model.
The result is intentionally small, deterministic, and auditable so it can be
loaded in the online planning path without adding a training framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_VERSION = "empirical_strict_sla_calibration_v2"
PREDICTION_TARGET = "receiver_strict_sla_given_sender_started"
DELAY_RATIO_UPPER_BOUNDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00)
BANDWIDTH_UPPER_BOUNDS = (5.0, 10.0, 20.0, 40.0, 80.0)
DESTINATION_UPPER_BOUNDS = (1.0, 2.0, 4.0, 8.0)


def _finite_nonnegative(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def numeric_bucket(value: float, upper_bounds: Sequence[float]) -> str:
    value = float(value)
    lower = 0.0
    for upper in upper_bounds:
        if value <= float(upper) + 1e-12:
            return f"({lower:g},{float(upper):g}]"
        lower = float(upper)
    return f"({lower:g},inf)"


def _cell(samples: int, failures: int, alpha: float, beta: float) -> dict[str, Any]:
    probability = (float(failures) + alpha) / (float(samples) + alpha + beta)
    return {
        "samples": int(samples),
        "failures": int(failures),
        "failure_probability": float(probability),
    }


def _monotonic_probabilities(
    cells: Sequence[dict[str, Any]],
    *,
    alpha: float,
    beta: float,
) -> list[float]:
    """Weighted pool-adjacent-violators fit over ordered delay-ratio bins."""

    blocks: list[dict[str, float | list[int]]] = []
    for index, cell in enumerate(cells):
        weight = float(cell["samples"]) + alpha + beta
        probability = float(cell["failure_probability"])
        blocks.append({"indices": [index], "weight": weight, "mean": probability})
        while len(blocks) >= 2 and float(blocks[-2]["mean"]) > float(blocks[-1]["mean"]):
            right = blocks.pop()
            left = blocks.pop()
            total_weight = float(left["weight"]) + float(right["weight"])
            mean = (
                float(left["mean"]) * float(left["weight"])
                + float(right["mean"]) * float(right["weight"])
            ) / max(total_weight, 1e-12)
            blocks.append({
                "indices": list(left["indices"]) + list(right["indices"]),
                "weight": total_weight,
                "mean": mean,
            })
    result = [0.0] * len(cells)
    for block in blocks:
        for index in block["indices"]:
            result[int(index)] = float(block["mean"])
    return result


def _categorical_stats(
    samples: Sequence[Mapping[str, Any]],
    key: str,
    *,
    alpha: float,
    beta: float,
) -> dict[str, dict[str, Any]]:
    counts: dict[str, list[int]] = {}
    for sample in samples:
        label = str(sample[key])
        values = counts.setdefault(label, [0, 0])
        values[0] += 1
        values[1] += int(not bool(sample["strict_sla_met"]))
    return {
        label: _cell(values[0], values[1], alpha, beta)
        for label, values in sorted(counts.items())
    }


def _numeric_stats(
    samples: Sequence[Mapping[str, Any]],
    key: str,
    upper_bounds: Sequence[float],
    *,
    alpha: float,
    beta: float,
    monotonic: bool = False,
) -> dict[str, Any]:
    labels = [numeric_bucket(float(sample[key]), upper_bounds) for sample in samples]
    ordered_labels = []
    for value in (*upper_bounds, float("inf")):
        probe = float(value) if math.isfinite(float(value)) else float(upper_bounds[-1]) + 1.0
        label = numeric_bucket(probe, upper_bounds)
        if label not in ordered_labels:
            ordered_labels.append(label)
    counts = {label: [0, 0] for label in ordered_labels}
    for sample, label in zip(samples, labels):
        counts[label][0] += 1
        counts[label][1] += int(not bool(sample["strict_sla_met"]))
    cells = [_cell(*counts[label], alpha, beta) for label in ordered_labels]
    if monotonic:
        for cell, probability in zip(
            cells,
            _monotonic_probabilities(cells, alpha=alpha, beta=beta),
        ):
            cell["raw_failure_probability"] = cell["failure_probability"]
            cell["failure_probability"] = float(probability)
    return {
        "upper_bounds": [float(value) for value in upper_bounds],
        "cells": {label: cell for label, cell in zip(ordered_labels, cells)},
    }


def build_calibration_payload(
    raw_samples: Iterable[Mapping[str, Any]],
    *,
    sources: Sequence[Mapping[str, Any]] | None = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    minimum_cell_samples: int = 8,
) -> dict[str, Any]:
    """Build a serializable calibration artifact from selected-plan outcomes."""

    alpha = _finite_nonnegative(prior_alpha, name="prior_alpha")
    beta = _finite_nonnegative(prior_beta, name="prior_beta")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("calibration priors must be positive")
    if minimum_cell_samples <= 0:
        raise ValueError("minimum_cell_samples must be positive")
    samples = []
    for row in raw_samples:
        samples.append({
            "modeled_delay_ratio": _finite_nonnegative(
                row["modeled_delay_ratio"], name="modeled_delay_ratio"
            ),
            "strict_sla_met": bool(row["strict_sla_met"]),
            "qos_class": str(row.get("qos_class") or "unknown"),
            "bandwidth_mbps": _finite_nonnegative(
                row.get("bandwidth_mbps", 0.0), name="bandwidth_mbps"
            ),
            "destination_count": _finite_nonnegative(
                row.get("destination_count", 0.0), name="destination_count"
            ),
        })
    if not samples:
        raise ValueError("cannot build SLA calibration without probe samples")
    failures = sum(not sample["strict_sla_met"] for sample in samples)
    payload: dict[str, Any] = {
        "version": MODEL_VERSION,
        "prediction_target": PREDICTION_TARGET,
        "sample_count": len(samples),
        "failure_count": int(failures),
        "strict_sla_rate": float((len(samples) - failures) / len(samples)),
        "prior": {"alpha": alpha, "beta": beta},
        "minimum_cell_samples": int(minimum_cell_samples),
        "global": _cell(len(samples), failures, alpha, beta),
        "dimensions": {
            "modeled_delay_ratio": _numeric_stats(
                samples,
                "modeled_delay_ratio",
                DELAY_RATIO_UPPER_BOUNDS,
                alpha=alpha,
                beta=beta,
                monotonic=True,
            ),
            "qos_class": {"cells": _categorical_stats(
                samples, "qos_class", alpha=alpha, beta=beta
            )},
            "bandwidth_mbps": _numeric_stats(
                samples,
                "bandwidth_mbps",
                BANDWIDTH_UPPER_BOUNDS,
                alpha=alpha,
                beta=beta,
            ),
            "destination_count": _numeric_stats(
                samples,
                "destination_count",
                DESTINATION_UPPER_BOUNDS,
                alpha=alpha,
                beta=beta,
            ),
        },
        "sources": list(sources or []),
    }
    payload["data_readiness"] = {
        "status": (
            "ready"
            if failures >= 10 and len(samples) - failures >= 10
            else "insufficient_failure_examples"
            if failures < 10
            else "insufficient_success_examples"
        ),
        "minimum_recommended_failures": 10,
        "minimum_recommended_successes": 10,
        "observed_failures": int(failures),
        "observed_successes": int(len(samples) - failures),
    }
    calibrator = EmpiricalSlaRiskCalibrator(payload)
    probabilities = [
        calibrator.predict(sample["modeled_delay_ratio"], sample).failure_probability
        for sample in samples
    ]
    payload["fit_metrics"] = {
        "in_sample_brier_score": float(sum(
            (probability - float(not sample["strict_sla_met"])) ** 2
            for probability, sample in zip(probabilities, samples)
        ) / len(samples)),
        "mean_predicted_failure_probability": float(
            sum(probabilities) / len(probabilities)
        ),
    }
    return payload


@dataclass(frozen=True)
class SlaRiskEstimate:
    failure_probability: float
    modeled_delay_ratio: float
    effective_support: float
    delay_ratio_bucket: str
    components: tuple[dict[str, Any], ...]
    model_kind: str = MODEL_VERSION
    confidence: float = 1.0
    ood_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmpiricalSlaRiskCalibrator:
    """Read-only empirical risk lookup with hierarchical shrinkage."""

    _COMPONENT_WEIGHTS = {
        "global": 1.0,
        "modeled_delay_ratio": 4.0,
        "qos_class": 1.5,
        "bandwidth_mbps": 1.0,
        "destination_count": 1.0,
    }

    def __init__(self, payload: Mapping[str, Any], path: str | Path | None = None):
        self.payload = dict(payload)
        if self.payload.get("version") != MODEL_VERSION:
            raise ValueError(
                f"unsupported SLA calibration version: {self.payload.get('version')}"
            )
        if self.payload.get("prediction_target") != PREDICTION_TARGET:
            raise ValueError(
                "SLA calibration does not target receiver strict SLA after "
                "successful sender startup"
            )
        if int(self.payload.get("sample_count", 0)) <= 0:
            raise ValueError("SLA calibration has no samples")
        self.minimum_cell_samples = int(self.payload.get("minimum_cell_samples", 8))
        if self.minimum_cell_samples <= 0:
            raise ValueError("invalid minimum_cell_samples in SLA calibration")
        self.path = Path(path).resolve() if path is not None else None
        self.dimensions = self.payload.get("dimensions") or {}
        self.global_cell = self.payload.get("global") or {}
        self._validate_cell(self.global_cell, "global")
        for dimension in self._COMPONENT_WEIGHTS:
            if dimension == "global":
                continue
            spec = self.dimensions.get(dimension)
            if not isinstance(spec, Mapping):
                raise ValueError(f"SLA calibration is missing dimension {dimension}")
            cells = spec.get("cells")
            if not isinstance(cells, Mapping) or not cells:
                raise ValueError(f"SLA calibration dimension {dimension} has no cells")
            if dimension != "qos_class":
                bounds = spec.get("upper_bounds")
                if not isinstance(bounds, list) or not bounds:
                    raise ValueError(
                        f"SLA calibration dimension {dimension} has no bounds"
                    )
                if any(
                    not math.isfinite(float(value)) or float(value) <= 0.0
                    for value in bounds
                ) or any(
                    float(left) >= float(right)
                    for left, right in zip(bounds, bounds[1:])
                ):
                    raise ValueError(
                        f"SLA calibration dimension {dimension} has invalid bounds"
                    )
            for label, cell in cells.items():
                self._validate_cell(cell, f"{dimension}:{label}")

    @staticmethod
    def _validate_cell(cell: Any, label: str) -> None:
        if not isinstance(cell, Mapping):
            raise ValueError(f"SLA calibration cell {label} is malformed")
        samples = int(cell.get("samples", -1))
        failures = int(cell.get("failures", -1))
        probability = float(cell.get("failure_probability", float("nan")))
        if (
            samples < 0
            or failures < 0
            or failures > samples
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError(f"SLA calibration cell {label} is invalid")

    @classmethod
    def from_path(cls, path: str | Path) -> "EmpiricalSlaRiskCalibrator":
        resolved = Path(path).resolve()
        return cls(json.loads(resolved.read_text(encoding="utf-8")), resolved)

    @staticmethod
    def _request_value(request: Mapping[str, Any], key: str) -> Any:
        if key == "qos_class":
            value = request.get("qos_class")
            if value:
                return str(value)
            priority = int(request.get("priority", -1))
            return {3: "Q0_REALTIME", 2: "Q1_INTERACTIVE", 1: "Q2_ELASTIC"}.get(
                priority, "unknown"
            )
        if key == "bandwidth_mbps":
            return float(request.get("bw_origin", request.get("bandwidth_mbps", 0.0)))
        if key == "destination_count":
            destinations = request.get("destination_dpids") or request.get("destinations")
            return float(len(destinations or []))
        raise KeyError(key)

    def _resolve_cell(self, dimension: str, value: Any) -> tuple[str, Mapping[str, Any]] | None:
        spec = self.dimensions.get(dimension) or {}
        cells = spec.get("cells") or {}
        if dimension == "qos_class":
            label = str(value)
        else:
            label = numeric_bucket(float(value), spec.get("upper_bounds") or [])
        cell = cells.get(label)
        return (label, cell) if cell is not None else None

    def predict(
        self,
        modeled_delay_ratio: float,
        request: Mapping[str, Any],
    ) -> SlaRiskEstimate:
        ratio = _finite_nonnegative(modeled_delay_ratio, name="modeled_delay_ratio")
        components: list[dict[str, Any]] = []

        def add_component(name: str, label: str, cell: Mapping[str, Any]) -> None:
            samples = max(0, int(cell.get("samples", 0)))
            reliability = min(1.0, samples / float(self.minimum_cell_samples))
            base_weight = self._COMPONENT_WEIGHTS[name]
            weight = base_weight * reliability
            if name == "global":
                weight = base_weight
            if weight <= 0.0:
                return
            components.append({
                "name": name,
                "bucket": label,
                "samples": samples,
                "failure_probability": float(cell["failure_probability"]),
                "weight": float(weight),
            })

        add_component("global", "all", self.global_cell)
        delay = self._resolve_cell("modeled_delay_ratio", ratio)
        delay_label = numeric_bucket(
            ratio,
            (self.dimensions.get("modeled_delay_ratio") or {}).get("upper_bounds") or [],
        )
        if delay is not None:
            delay_label, cell = delay
            add_component("modeled_delay_ratio", delay_label, cell)
        for dimension in ("qos_class", "bandwidth_mbps", "destination_count"):
            resolved = self._resolve_cell(
                dimension, self._request_value(request, dimension)
            )
            if resolved is not None:
                add_component(dimension, resolved[0], resolved[1])
        total_weight = sum(float(row["weight"]) for row in components)
        probability = sum(
            float(row["weight"]) * float(row["failure_probability"])
            for row in components
        ) / max(total_weight, 1e-12)
        support = sum(
            float(row["weight"]) * int(row["samples"])
            for row in components
        ) / max(total_weight, 1e-12)
        return SlaRiskEstimate(
            failure_probability=float(min(1.0, max(0.0, probability))),
            modeled_delay_ratio=ratio,
            effective_support=float(support),
            delay_ratio_bucket=delay_label,
            components=tuple(components),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "prediction_target": PREDICTION_TARGET,
            "path": str(self.path) if self.path is not None else None,
            "sample_count": int(self.payload["sample_count"]),
            "failure_count": int(self.payload.get("failure_count", 0)),
            "strict_sla_rate": float(self.payload.get("strict_sla_rate", 0.0)),
            "minimum_cell_samples": self.minimum_cell_samples,
            "fit_metrics": dict(self.payload.get("fit_metrics") or {}),
            "data_readiness": dict(self.payload.get("data_readiness") or {}),
        }
