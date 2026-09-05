#!/usr/bin/env python3
"""Regression checks for empirical strict-SLA risk calibration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.sla_risk_calibration import (  # noqa: E402
    EmpiricalSlaRiskCalibrator,
    build_calibration_payload,
)


def synthetic_samples() -> list[dict]:
    rows = []
    for ratio, failures in ((0.1, 1), (0.35, 3), (0.7, 7), (1.2, 10)):
        for index in range(12):
            rows.append({
                "modeled_delay_ratio": ratio,
                "strict_sla_met": index >= failures,
                "qos_class": "Q0_REALTIME",
                "bandwidth_mbps": 20.0,
                "destination_count": 5,
            })
    return rows


def main() -> None:
    payload = build_calibration_payload(
        synthetic_samples(), minimum_cell_samples=8
    )
    calibrator = EmpiricalSlaRiskCalibrator(payload)
    request = {
        "qos_class": "Q0_REALTIME",
        "bw_origin": 20.0,
        "destination_dpids": [1, 2, 3, 4, 5],
    }
    probabilities = [
        calibrator.predict(ratio, request).failure_probability
        for ratio in (0.1, 0.35, 0.7, 1.2)
    ]
    assert probabilities == sorted(probabilities), probabilities

    sparse = calibrator.predict(
        3.0,
        {"qos_class": "unseen", "bw_origin": 500.0, "destination_dpids": []},
    )
    assert any(row["name"] == "global" for row in sparse.components)
    assert 0.0 <= sparse.failure_probability <= 1.0

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = EmpiricalSlaRiskCalibrator.from_path(path)
        assert loaded.predict(0.7, request) == calibrator.predict(0.7, request)

    malformed = copy.deepcopy(payload)
    malformed["global"]["failure_probability"] = 2.0
    try:
        EmpiricalSlaRiskCalibrator(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed calibration did not fail closed")

    print(json.dumps({
        "status": "ok",
        "samples": payload["sample_count"],
        "probabilities": probabilities,
        "sparse_probability": sparse.failure_probability,
        "brier_score": payload["fit_metrics"]["in_sample_brier_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
