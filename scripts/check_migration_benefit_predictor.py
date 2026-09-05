#!/usr/bin/env python3
"""Smoke-check the counterfactual migration benefit gate."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.migration_benefit_predictor import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_VERSION,
    PREDICTION_TARGET,
    MigrationBenefitNet,
    MigrationBenefitPredictor,
    build_migration_benefit_features,
)
from envs.modules.reconfiguration_manager import ReconfigurationManager  # noqa: E402
from envs.sft_role_marl_env import SFTRoleMARLEnv  # noqa: E402
from scripts.check_reconfiguration_manager import (  # noqa: E402
    build_resource_manager,
    deploy_active_sft,
)


def main() -> int:
    rm = build_resource_manager()
    deploy_active_sft(rm)
    manager = ReconfigurationManager(rm, node_util_threshold=0.5)
    risk = manager.select_topk_risky_sfts(1)[0]
    proposal = manager.plan_greedy_migrate(risk.req_id)
    if proposal.action_type != "migrate_vnf":
        raise AssertionError("smoke fixture did not produce a migration proposal")
    features = build_migration_benefit_features(manager, risk, proposal)
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError("migration feature dimension mismatch")

    model = MigrationBenefitNet(len(FEATURE_NAMES), (8,), 0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(-4.0)
    payload = {
        "version": MODEL_VERSION,
        "prediction_target": PREDICTION_TARGET,
        "feature_names": list(FEATURE_NAMES),
        "hidden_dims": [8],
        "dropout": 0.0,
        "normalization": {
            "mean": [0.0] * len(FEATURE_NAMES),
            "std": [1.0] * len(FEATURE_NAMES),
        },
        "state_dict": model.state_dict(),
        "calibration": {"logit_scale": 1.0, "logit_bias": 0.0},
        "decision_threshold": 0.5,
        "deployment_allowed": True,
        "training": {"samples": 10},
        "validation": {"smoke": True},
    }
    predictor = MigrationBenefitPredictor(payload)
    estimate = predictor.predict(manager, risk, proposal)
    if estimate.allowed or estimate.probability >= 0.5:
        raise AssertionError("negative-benefit smoke model failed to block migration")
    gated_env = SFTRoleMARLEnv(
        rm,
        top_k=1,
        node_util_threshold=0.5,
        enable_migration=True,
        enable_reroute=False,
        migration_benefit_predictor=predictor,
    )
    report = gated_env.step(apply=True)
    if report["coordination"]["selected"] != "noop":
        raise AssertionError("role coordinator executed a migration rejected by the gate")
    if report["after"]["total_migrations"] != 0:
        raise AssertionError("migration ledger changed despite a negative gate decision")
    print("Migration benefit predictor smoke check passed")
    print(estimate.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
