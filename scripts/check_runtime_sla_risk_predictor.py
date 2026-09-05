#!/usr/bin/env python3
"""Regression checks for the supervised runtime SLA risk predictor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.sla_risk_predictor import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_VERSION,
    RuntimeSlaRiskPredictor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "artifacts/reference_runs/sla/sla_risk_predictor_v2/runtime_sla_risk_predictor.pt"
        ),
    )
    args = parser.parse_args()
    predictor = RuntimeSlaRiskPredictor.from_path(args.checkpoint)
    center = predictor.mean.tolist()
    first = predictor.predict_features(center, modeled_delay_ratio=0.2)
    second = predictor.predict_features(center, modeled_delay_ratio=0.2)
    batched = predictor.predict_features_batch(
        [center, center], modeled_delay_ratios=[0.2, 0.2]
    )
    assert first.model_kind == MODEL_VERSION
    assert first.failure_probability == second.failure_probability
    assert 0.0 <= first.failure_probability <= 1.0
    assert first.ood_score == 0.0
    assert first.confidence == 1.0
    assert len(batched) == 2
    assert math.isclose(
        batched[0].failure_probability,
        first.failure_probability,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        batched[1].failure_probability,
        second.failure_probability,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert batched[0].ood_score == first.ood_score
    assert batched[0].model_kind == first.model_kind

    shifted = list(center)
    shifted[FEATURE_NAMES.index("recent_arrival_rate_1s")] += (
        8.0 * float(predictor.std[FEATURE_NAMES.index("recent_arrival_rate_1s")])
    )
    out_of_distribution = predictor.predict_features(
        shifted, modeled_delay_ratio=0.2
    )
    assert out_of_distribution.ood_score >= 7.99
    assert out_of_distribution.confidence < 0.01

    malformed_rejected = False
    try:
        predictor.predict_features([0.0], modeled_delay_ratio=0.2)
    except ValueError:
        malformed_rejected = True
    assert malformed_rejected
    metadata = predictor.metadata()
    assert metadata["feature_count"] == len(FEATURE_NAMES)
    assert metadata["validation"]["test"]["roc_auc"] >= 0.65
    print(json.dumps({
        "status": "ok",
        "version": MODEL_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "center_probability": first.failure_probability,
        "ood_score": out_of_distribution.ood_score,
        "test_metrics": metadata["validation"]["test"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
