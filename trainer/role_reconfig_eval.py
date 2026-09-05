#!/usr/bin/env python3
"""Evaluation loop for role-based SFT reconfiguration.

The evaluator reuses the existing HRL deployment coordinator. After each
request is deployed or blocked, it optionally triggers the role-based
reconfiguration wrapper and writes a compact CSV row for thesis experiments.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class RoleReconfigEvaluator:
    """Run deployment episodes and periodically trigger role collaboration."""

    FIELDNAMES = [
        "episode",
        "request_id",
        "arrival_time",
        "success",
        "fail_reason",
        "total_reward",
        "steps",
        "reconfig_triggered",
        "reconfig_ok",
        "selected_req",
        "selected_action",
        "coord_reason",
        "reward_proxy",
        "migration_label",
        "migration_score",
        "migration_benefit_probability",
        "migration_benefit_allowed",
        "migration_benefit_confidence",
        "reroute_label",
        "reroute_score",
        "executed_actions",
        "active_sfts_before",
        "active_sfts_after",
        "node_hotspots_before",
        "node_hotspots_after",
        "link_hotspots_before",
        "link_hotspots_after",
        "total_tree_edges_before",
        "total_tree_edges_after",
        "total_migrations_after",
        "total_reconfigs_after",
        "avg_delay_before",
        "avg_delay_after",
        "avg_propagation_delay_before",
        "avg_propagation_delay_after",
        "avg_processing_delay_before",
        "avg_processing_delay_after",
        "avg_queueing_delay_before",
        "avg_queueing_delay_after",
        "avg_reconfiguration_delay_before",
        "avg_reconfiguration_delay_after",
        "max_delay_before",
        "max_delay_after",
        "success_count",
        "accept_rate",
        "block_rate",
        "reconfig_error",
    ]

    def __init__(
        self,
        env,
        coordinator,
        role_env,
        output_dir: str | Path,
        num_episodes: int,
        max_steps_per_episode: int = 600,
        reconfig_interval: int = 10,
        warmup_episodes: int = 0,
        apply_reconfig: bool = True,
        csv_name: str = "role_reconfig_eval.csv",
    ):
        self.env = env
        self.coordinator = coordinator
        self.role_env = role_env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_episodes = int(num_episodes)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.reconfig_interval = int(reconfig_interval)
        self.warmup_episodes = int(warmup_episodes)
        self.apply_reconfig = bool(apply_reconfig)
        self.csv_path = self.output_dir / csv_name
        self.rows: list[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        success_count = 0

        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

            for episode in range(1, self.num_episodes + 1):
                total_reward, info = self.coordinator.run_episode(
                    training=False,
                    max_steps=self.max_steps_per_episode,
                )
                success = bool(info.get("success", False))
                if success:
                    success_count += 1

                trigger = self._should_reconfigure(episode)
                report: Optional[Dict[str, Any]] = None
                reconfig_error = ""
                before_metrics = self.role_env.manager.metrics_snapshot()
                after_metrics = before_metrics

                if trigger:
                    try:
                        report = self.role_env.step(apply=self.apply_reconfig)
                        before_metrics = report.get("before", before_metrics)
                        after_metrics = report.get("after", after_metrics)
                    except Exception as exc:  # keep long evaluations alive
                        reconfig_error = str(exc)
                        logger.exception("role reconfiguration failed at episode %s", episode)
                        after_metrics = self.role_env.manager.metrics_snapshot()

                row = self._build_row(
                    episode=episode,
                    total_reward=total_reward,
                    info=info,
                    success=success,
                    success_count=success_count,
                    trigger=trigger,
                    report=report,
                    before=before_metrics,
                    after=after_metrics,
                    reconfig_error=reconfig_error,
                )
                writer.writerow(row)
                f.flush()
                self.rows.append(row)

                if episode % 10 == 0 or episode == self.num_episodes:
                    logger.info(
                        "role-reconfig eval %s/%s accept_rate=%.3f active_sfts=%s",
                        episode,
                        self.num_episodes,
                        row["accept_rate"],
                        row["active_sfts_after"],
                    )

        summary = {
            "episodes": self.num_episodes,
            "success_count": success_count,
            "accept_rate": success_count / max(1, self.num_episodes),
            "block_rate": 1.0 - success_count / max(1, self.num_episodes),
            "csv_path": str(self.csv_path),
            "final_metrics": self.role_env.manager.metrics_snapshot(),
            "migration_benefit_predictor": (
                self.role_env.migration_benefit_predictor.metadata()
                if self.role_env.migration_benefit_predictor is not None
                else None
            ),
        }
        summary_path = self.output_dir / "role_reconfig_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _should_reconfigure(self, episode: int) -> bool:
        if self.reconfig_interval <= 0:
            return False
        if episode <= self.warmup_episodes:
            return False
        return episode % self.reconfig_interval == 0

    def _build_row(
        self,
        episode: int,
        total_reward: float,
        info: Dict[str, Any],
        success: bool,
        success_count: int,
        trigger: bool,
        report: Optional[Dict[str, Any]],
        before: Dict[str, Any],
        after: Dict[str, Any],
        reconfig_error: str,
    ) -> Dict[str, Any]:
        req_snapshot = info.get("req_snapshot") or {}
        request_id = req_snapshot.get("id", "")
        arrival_time = req_snapshot.get("arrival_time", "")
        coordination = (report or {}).get("coordination", {})
        migration = (report or {}).get("migration", {})
        migration_benefit = (report or {}).get("migration_benefit") or {}
        reroute = (report or {}).get("reroute", {})
        selection = (report or {}).get("selection", {})
        executed = coordination.get("executed", [])

        accept_rate = success_count / max(1, episode)
        return {
            "episode": episode,
            "request_id": request_id,
            "arrival_time": arrival_time,
            "success": int(success),
            "fail_reason": info.get("reason", ""),
            "total_reward": float(total_reward),
            "steps": info.get("steps", 0),
            "reconfig_triggered": int(trigger),
            "reconfig_ok": int(bool((report or {}).get("ok", False))) if trigger else 0,
            "selected_req": selection.get("req_id", ""),
            "selected_action": coordination.get("selected", ""),
            "coord_reason": coordination.get("reason", ""),
            "reward_proxy": coordination.get("reward_proxy", ""),
            "migration_label": migration.get("label", ""),
            "migration_score": migration.get("score", ""),
            "migration_benefit_probability": migration_benefit.get("probability", ""),
            "migration_benefit_allowed": (
                int(bool(migration_benefit.get("allowed"))) if migration_benefit else ""
            ),
            "migration_benefit_confidence": migration_benefit.get("confidence", ""),
            "reroute_label": reroute.get("label", ""),
            "reroute_score": reroute.get("score", ""),
            "executed_actions": json.dumps(executed, ensure_ascii=False),
            "active_sfts_before": before.get("active_sfts", 0),
            "active_sfts_after": after.get("active_sfts", 0),
            "node_hotspots_before": before.get("node_hotspots", 0),
            "node_hotspots_after": after.get("node_hotspots", 0),
            "link_hotspots_before": before.get("link_hotspots", 0),
            "link_hotspots_after": after.get("link_hotspots", 0),
            "total_tree_edges_before": before.get("total_tree_edges", 0),
            "total_tree_edges_after": after.get("total_tree_edges", 0),
            "total_migrations_after": after.get("total_migrations", 0),
            "total_reconfigs_after": after.get("total_reconfigs", 0),
            "avg_delay_before": before.get("avg_delay_estimate", 0.0),
            "avg_delay_after": after.get("avg_delay_estimate", 0.0),
            "avg_propagation_delay_before": before.get("avg_propagation_delay_ms", 0.0),
            "avg_propagation_delay_after": after.get("avg_propagation_delay_ms", 0.0),
            "avg_processing_delay_before": before.get("avg_processing_delay_ms", 0.0),
            "avg_processing_delay_after": after.get("avg_processing_delay_ms", 0.0),
            "avg_queueing_delay_before": before.get("avg_queueing_delay_ms", 0.0),
            "avg_queueing_delay_after": after.get("avg_queueing_delay_ms", 0.0),
            "avg_reconfiguration_delay_before": before.get("avg_reconfiguration_delay_ms", 0.0),
            "avg_reconfiguration_delay_after": after.get("avg_reconfiguration_delay_ms", 0.0),
            "max_delay_before": before.get("max_delay_total_ms", 0.0),
            "max_delay_after": after.get("max_delay_total_ms", 0.0),
            "success_count": success_count,
            "accept_rate": accept_rate,
            "block_rate": 1.0 - accept_rate,
            "reconfig_error": reconfig_error,
        }
