"""Runnable role-based MARL wrapper for SFT reconfiguration.

This is not yet a Gym training environment. It is the first executable wrapper
that wires fixed role agents to the heuristic reconfiguration manager:

1. SFT selection role chooses a risky SFT.
2. VNF migration role proposes a migration.
3. Tree reroute role proposes local rerouting.
4. Role coordinator resolves proposals and executes a low-disturbance action.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.marl.role_agents import (
    SFTSelectionAgent,
    TreeRerouteAgent,
    VNFMigrationAgent,
)
from core.marl.role_coordinator import RoleCoordinator
from core.marl.role_agents import RoleDecision
from envs.modules.reconfiguration_manager import ReconfigurationManager


class SFTRoleMARLEnv:
    """One-step role-MARL reconfiguration runner."""

    def __init__(
        self,
        env_or_resource_mgr,
        top_k: int = 3,
        node_util_threshold: float = 0.80,
        link_util_threshold: float = 0.80,
        delay_threshold: Optional[float] = None,
        min_risk_score: float = 0.0,
        min_action_gain: float = 0.0,
        agent_mode: str = "rule",
        trainable_config: Optional[Dict[str, Any]] = None,
        enable_migration: bool = True,
        enable_reroute: bool = True,
        max_tree_edge_growth: int = 2,
        max_projected_link_utilization: float = 0.90,
        min_residual_connectivity_ratio: float = 0.0,
        migration_benefit_predictor=None,
    ):
        self.agent_mode = str(agent_mode).lower()
        self.enable_migration = bool(enable_migration)
        self.enable_reroute = bool(enable_reroute)
        self.migration_benefit_predictor = migration_benefit_predictor
        self.manager = ReconfigurationManager(
            env_or_resource_mgr,
            node_util_threshold=node_util_threshold,
            link_util_threshold=link_util_threshold,
            delay_threshold=delay_threshold,
            max_tree_edge_growth=max_tree_edge_growth,
            max_projected_link_utilization=max_projected_link_utilization,
            min_residual_connectivity_ratio=min_residual_connectivity_ratio,
        )
        if self.agent_mode == "trainable":
            cfg = trainable_config or {}
            from core.marl.trainable_role_agents import (
                TrainableSFTSelectionAgent,
                TrainableTreeRerouteAgent,
                TrainableVNFMigrationAgent,
            )
            self.selector = TrainableSFTSelectionAgent(
                top_k=top_k,
                min_risk_score=min_risk_score,
                hidden_dim=int(cfg.get("hidden_dim", 64)),
                lr=float(cfg.get("lr", 1e-3)),
                epsilon=float(cfg.get("epsilon", 0.10)),
                device=str(cfg.get("device", "cpu")),
            )
            self.migrator = TrainableVNFMigrationAgent(
                min_gain=min_action_gain,
                hidden_dim=int(cfg.get("hidden_dim", 64)),
                lr=float(cfg.get("lr", 1e-3)),
                epsilon=float(cfg.get("epsilon", 0.10)),
                device=str(cfg.get("device", "cpu")),
            )
            self.rerouter = TrainableTreeRerouteAgent(
                min_gain=min_action_gain,
                hidden_dim=int(cfg.get("hidden_dim", 64)),
                lr=float(cfg.get("lr", 1e-3)),
                epsilon=float(cfg.get("epsilon", 0.10)),
                device=str(cfg.get("device", "cpu")),
            )
        else:
            self.selector = SFTSelectionAgent(top_k=top_k, min_risk_score=min_risk_score)
            self.migrator = VNFMigrationAgent(min_gain=min_action_gain)
            self.rerouter = TreeRerouteAgent(min_gain=min_action_gain)
        self.coordinator = RoleCoordinator()
        self.last_report: Optional[Dict[str, Any]] = None

    def observe(self) -> Dict[str, Any]:
        selector_obs = self._selector_observation()
        return {
            "metrics": selector_obs["metrics"],
            "risks": [risk.__dict__ for risk in selector_obs["risks"]],
        }

    def step(self, apply: bool = True) -> Dict[str, Any]:
        before = self.manager.metrics_snapshot()

        selector_obs = self._selector_observation()
        select_decision = self.selector.select_action(selector_obs)
        req_id = select_decision.req_id

        migration_obs = self.migrator.observe(self.manager, req_id if self.enable_migration else None)
        migration_decision = self.migrator.select_action(migration_obs)
        migration_benefit = None
        if (
            migration_decision.action != 0
            and migration_decision.proposal
            and self.migration_benefit_predictor is not None
        ):
            rec = self.manager.rm.request_table.get(req_id)
            risk = self.manager.score_sft_risk(rec) if rec is not None else None
            migration_benefit = self.migration_benefit_predictor.predict(
                self.manager, risk, migration_decision.proposal
            )
            if not migration_benefit.allowed:
                migration_decision = RoleDecision(
                    role="vnf_migration",
                    action=0,
                    label="predictive_no_migration",
                    req_id=req_id,
                    score=float(migration_benefit.probability),
                    proposal=None,
                    reason=migration_benefit.reason,
                )

        reroute_obs = self.rerouter.observe(self.manager, req_id if self.enable_reroute else None)
        reroute_decision = self.rerouter.select_action(reroute_obs)

        coordination = self.coordinator.coordinate(
            self.manager,
            req_id=req_id,
            migration_decision=migration_decision,
            reroute_decision=reroute_decision,
            apply=apply,
        )
        after = self.manager.metrics_snapshot()

        snapshot_report = None
        if req_id is not None:
            snapshot_report = self.manager.rm.validate_request_sft_snapshot(req_id)

        role_training = self._build_role_training_payload(
            req_id=req_id,
            selector_obs=selector_obs,
            select_action=select_decision.action,
            migration_obs=migration_obs,
            migration_action=migration_decision.action,
            reroute_obs=reroute_obs,
            reroute_action=reroute_decision.action,
        )

        report = {
            "ok": coordination.selected not in {"failed"},
            "before": before,
            "selection": select_decision.to_dict(),
            "migration": migration_decision.to_dict(),
            "migration_benefit": (
                migration_benefit.to_dict() if migration_benefit is not None else None
            ),
            "reroute": reroute_decision.to_dict(),
            "coordination": coordination.to_dict(),
            "after": after,
            "snapshot_report": snapshot_report,
        }
        if role_training:
            report["role_training"] = role_training
        self.last_report = report
        return report

    def _build_role_training_payload(
        self,
        req_id: Optional[int],
        selector_obs: Dict[str, Any],
        select_action: int,
        migration_obs: Dict[str, Any],
        migration_action: int,
        reroute_obs: Dict[str, Any],
        reroute_action: int,
    ) -> Dict[str, Dict[str, Any]]:
        if self.agent_mode != "trainable":
            return {}

        payload: Dict[str, Dict[str, Any]] = {}
        next_selector_obs = self._selector_observation()
        payload["selector"] = {
            "obs": selector_obs.get("obs_vector"),
            "action": int(select_action),
            "next_obs": next_selector_obs.get("obs_vector"),
            "valid_actions": selector_obs.get("valid_actions"),
            "next_valid_actions": next_selector_obs.get("valid_actions"),
        }

        next_migration_obs = self.migrator.observe(
            self.manager, req_id if self.enable_migration else None
        )
        payload["migration"] = {
            "obs": migration_obs.get("obs_vector"),
            "action": int(migration_action),
            "next_obs": next_migration_obs.get("obs_vector"),
            "valid_actions": migration_obs.get("valid_actions"),
            "next_valid_actions": next_migration_obs.get("valid_actions"),
        }

        next_reroute_obs = self.rerouter.observe(
            self.manager, req_id if self.enable_reroute else None
        )
        payload["reroute"] = {
            "obs": reroute_obs.get("obs_vector"),
            "action": int(reroute_action),
            "next_obs": next_reroute_obs.get("obs_vector"),
            "valid_actions": reroute_obs.get("valid_actions"),
            "next_valid_actions": next_reroute_obs.get("valid_actions"),
        }
        return payload

    def _selector_observation(self) -> Dict[str, Any]:
        obs = self.selector.observe(self.manager)
        if self.enable_migration and self.enable_reroute:
            return obs

        actionable = []
        valid_actions = [0]
        for index, risk in enumerate(obs.get("risks", []), start=1):
            can_migrate = False
            can_reroute = False
            if self.enable_migration:
                action = self.manager.plan_greedy_migrate(req_id=risk.req_id)
                can_migrate = action.action_type != "noop" and action.estimated_gain > 0.0
            if self.enable_reroute:
                action = self.manager.plan_greedy_reroute(req_id=risk.req_id)
                can_reroute = action.action_type != "noop" and action.estimated_gain > 0.0
            if can_migrate or can_reroute:
                actionable.append(risk)
                valid_actions.append(index)

        if self.agent_mode == "trainable":
            obs["valid_actions"] = valid_actions
        else:
            obs["risks"] = actionable
        return obs
