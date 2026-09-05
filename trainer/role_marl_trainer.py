#!/usr/bin/env python3
"""Training loop for role-based trainable SFT reconfiguration agents."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch


logger = logging.getLogger(__name__)


class RoleMARLTrainer:
    """Train role-DQN agents from online reconfiguration transitions."""

    FIELDNAMES = [
        "episode",
        "request_id",
        "success",
        "fail_reason",
        "reconfig_triggered",
        "selected_req",
        "selected_action",
        "reward",
        "hotspot_reward",
        "delay_reward",
        "migration_penalty",
        "reroute_penalty",
        "tree_penalty",
        "failure_penalty",
        "noop_penalty",
        "coord_reward",
        "selector_loss",
        "migration_loss",
        "reroute_loss",
        "selector_replay",
        "migration_replay",
        "reroute_replay",
        "active_sfts",
        "node_hotspots",
        "link_hotspots",
        "avg_delay_total_ms",
        "avg_queueing_delay_ms",
        "total_migrations",
        "total_reconfigs",
        "accept_rate",
    ]

    def __init__(
        self,
        env,
        coordinator,
        role_env,
        output_dir: str | Path,
        num_episodes: int = 200,
        max_steps_per_episode: int = 600,
        reconfig_interval: int = 10,
        warmup_episodes: int = 0,
        train_every: int = 1,
        updates_per_train: int = 1,
        batch_size: int = 64,
        save_every: int = 50,
        apply_reconfig: bool = True,
        reward_weights: Optional[Dict[str, float]] = None,
        epsilon_final: Optional[float] = None,
        epsilon_decay_episodes: Optional[int] = None,
        training_algorithm: str = "independent_dqn",
        qmix_embed_dim: int = 32,
        qmix_lr: Optional[float] = None,
        role_checkpoint: Optional[str | Path] = None,
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
        self.train_every = int(train_every)
        self.updates_per_train = int(updates_per_train)
        self.batch_size = int(batch_size)
        self.save_every = int(save_every)
        self.apply_reconfig = bool(apply_reconfig)
        self.epsilon_final = epsilon_final
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.initial_epsilons = self._role_epsilons()
        self.training_algorithm = str(training_algorithm).lower()
        if self.training_algorithm not in {"independent_dqn", "qmix"}:
            raise ValueError("training_algorithm must be independent_dqn or qmix")
        self.qmix = None
        if self.training_algorithm == "qmix":
            from core.marl.qmix import QMIXLearner
            agents = [role_env.selector, role_env.migrator, role_env.rerouter]
            self.qmix = QMIXLearner(
                agents, mixer_embed_dim=qmix_embed_dim,
                lr=float(qmix_lr if qmix_lr is not None else agents[0].config.lr),
                gamma=float(agents[0].config.gamma),
                replay_capacity=int(agents[0].config.replay_capacity),
                target_update_interval=int(agents[0].config.target_update_interval),
            )
        self.reward_weights = {
            "hotspot": 2.0,
            "delay": 1.0,
            "delay_scale": 10.0,
            "migration": 0.50,
            "reroute": 0.30,
            "tree": 0.10,
            "failure": 2.0,
            "noop": 0.05,
            "coord": 0.20,
            **(reward_weights or {}),
        }
        self.csv_path = self.output_dir / "role_marl_train.csv"
        self.summary_path = self.output_dir / "role_marl_train_summary.json"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if getattr(role_env, "agent_mode", "") != "trainable":
            raise ValueError("RoleMARLTrainer requires SFTRoleMARLEnv(agent_mode='trainable')")
        if role_checkpoint:
            self.load_checkpoint(role_checkpoint)

    def run(self) -> Dict[str, Any]:
        success_count = 0
        reconfig_count = 0
        last_losses = {"selector": 0.0, "migration": 0.0, "reroute": 0.0}

        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

            for episode in range(1, self.num_episodes + 1):
                self._update_epsilon(episode)
                total_reward, info = self.coordinator.run_episode(
                    training=False,
                    max_steps=self.max_steps_per_episode,
                )
                success = bool(info.get("success", False))
                if success:
                    success_count += 1

                report = None
                reward = 0.0
                reward_detail = self._empty_reward_detail()
                triggered = self._should_reconfigure(episode)

                if triggered:
                    report = self.role_env.step(apply=self.apply_reconfig)
                    reward, reward_detail = self.compute_reward(report)
                    self._store_role_transitions(report, reward)
                    reconfig_count += 1
                    if reconfig_count % max(1, self.train_every) == 0:
                        last_losses = self._train_roles()

                row = self._build_row(
                    episode=episode,
                    info=info,
                    success=success,
                    success_count=success_count,
                    triggered=triggered,
                    report=report,
                    reward=reward,
                    reward_detail=reward_detail,
                    losses=last_losses,
                )
                writer.writerow(row)
                f.flush()

                if self.save_every > 0 and episode % self.save_every == 0:
                    self.save_checkpoint(self.checkpoint_dir / f"role_marl_ep{episode}.pth", episode)

                if episode % 10 == 0 or episode == self.num_episodes:
                    logger.info(
                        "role-MARL train %s/%s accept_rate=%.3f reward=%.3f selected=%s",
                        episode,
                        self.num_episodes,
                        row["accept_rate"],
                        reward,
                        row["selected_action"],
                    )

        final_ckpt = self.checkpoint_dir / "role_marl_final.pth"
        self.save_checkpoint(final_ckpt, self.num_episodes)
        summary = {
            "episodes": self.num_episodes,
            "success_count": success_count,
            "accept_rate": success_count / max(1, self.num_episodes),
            "block_rate": 1.0 - success_count / max(1, self.num_episodes),
            "reconfiguration_events": reconfig_count,
            "csv_path": str(self.csv_path),
            "final_checkpoint": str(final_ckpt),
            "final_metrics": self.role_env.manager.metrics_snapshot(),
            "training_algorithm": self.training_algorithm,
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def compute_reward(self, report: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        before = report.get("before", {})
        after = report.get("after", {})
        coordination = report.get("coordination", {})

        hotspot_before = float(before.get("node_hotspots", 0.0)) + float(before.get("link_hotspots", 0.0))
        hotspot_after = float(after.get("node_hotspots", 0.0)) + float(after.get("link_hotspots", 0.0))
        hotspot_delta = hotspot_before - hotspot_after

        delay_before = float(before.get("avg_delay_total_ms", before.get("avg_delay_estimate", 0.0)))
        delay_after = float(after.get("avg_delay_total_ms", after.get("avg_delay_estimate", 0.0)))
        delay_delta = delay_before - delay_after

        migration_delta = max(
            0.0,
            float(after.get("total_migrations", 0.0)) - float(before.get("total_migrations", 0.0)),
        )
        reroute_delta = max(
            0.0,
            float(after.get("total_reconfigs", 0.0)) - float(before.get("total_reconfigs", 0.0)),
        )
        tree_delta = max(
            0.0,
            float(after.get("total_tree_edges", 0.0)) - float(before.get("total_tree_edges", 0.0)),
        )
        selected = str(coordination.get("selected", ""))
        failed = 1.0 if selected == "failed" or not bool(report.get("ok", False)) else 0.0
        noop = 1.0 if selected == "noop" and hotspot_before > 0 else 0.0

        w = self.reward_weights
        hotspot_reward = w["hotspot"] * hotspot_delta
        delay_reward = w["delay"] * (delay_delta / max(float(w["delay_scale"]), 1e-9))
        migration_penalty = -w["migration"] * migration_delta
        reroute_penalty = -w["reroute"] * reroute_delta
        tree_penalty = -w["tree"] * tree_delta
        failure_penalty = -w["failure"] * failed
        noop_penalty = -w["noop"] * noop
        coord_reward = w["coord"] * float(coordination.get("reward_proxy", 0.0) or 0.0)

        detail = {
            "hotspot_reward": float(hotspot_reward),
            "delay_reward": float(delay_reward),
            "migration_penalty": float(migration_penalty),
            "reroute_penalty": float(reroute_penalty),
            "tree_penalty": float(tree_penalty),
            "failure_penalty": float(failure_penalty),
            "noop_penalty": float(noop_penalty),
            "coord_reward": float(coord_reward),
        }
        reward = sum(detail.values())
        return float(reward), detail

    def _store_role_transitions(self, report: Dict[str, Any], reward: float) -> None:
        payload = report.get("role_training", {})
        role_to_agent = {
            "selector": self.role_env.selector,
            "migration": self.role_env.migrator,
            "reroute": self.role_env.rerouter,
        }
        if self.qmix is not None:
            ordered = [payload.get(role, {}) for role in ("selector", "migration", "reroute")]
            if all(item.get("obs") is not None and item.get("next_obs") is not None and item.get("action") is not None for item in ordered):
                self.qmix.push_transition(
                    [item["obs"] for item in ordered], [item["action"] for item in ordered], reward,
                    [item["next_obs"] for item in ordered],
                    [item.get("next_valid_actions") or [] for item in ordered], done=False,
                )
            return
        for role, transition in payload.items():
            obs = transition.get("obs")
            next_obs = transition.get("next_obs")
            action = transition.get("action")
            if obs is None or next_obs is None or action is None:
                continue
            role_to_agent[role].push_transition(
                obs, int(action), reward, next_obs, done=False,
                next_valid_actions=transition.get("next_valid_actions"),
            )

    def _train_roles(self) -> Dict[str, float]:
        if self.qmix is not None:
            total_loss = 0.0
            updates = 0.0
            for _ in range(max(1, self.updates_per_train)):
                result = self.qmix.update_from_replay(batch_size=self.batch_size)
                total_loss += float(result.get("loss", 0.0))
                updates += float(result.get("updated", 0.0))
            loss = total_loss / max(updates, 1.0)
            return {"selector": loss, "migration": loss, "reroute": loss}
        losses: Dict[str, float] = {}
        for role, agent in [
            ("selector", self.role_env.selector),
            ("migration", self.role_env.migrator),
            ("reroute", self.role_env.rerouter),
        ]:
            total_loss = 0.0
            updates = 0.0
            for _ in range(max(1, self.updates_per_train)):
                result = agent.update_from_replay(batch_size=self.batch_size)
                total_loss += float(result.get("loss", 0.0))
                updates += float(result.get("updated", 0.0))
            losses[role] = total_loss / max(updates, 1.0)
        return losses

    def _build_row(
        self,
        episode: int,
        info: Dict[str, Any],
        success: bool,
        success_count: int,
        triggered: bool,
        report: Optional[Dict[str, Any]],
        reward: float,
        reward_detail: Dict[str, float],
        losses: Dict[str, float],
    ) -> Dict[str, Any]:
        req_snapshot = info.get("req_snapshot") or {}
        metrics = (report or {}).get("after") or self.role_env.manager.metrics_snapshot()
        selection = (report or {}).get("selection", {})
        coordination = (report or {}).get("coordination", {})
        return {
            "episode": episode,
            "request_id": req_snapshot.get("id", ""),
            "success": int(success),
            "fail_reason": info.get("reason", ""),
            "reconfig_triggered": int(triggered),
            "selected_req": selection.get("req_id", ""),
            "selected_action": coordination.get("selected", ""),
            "reward": float(reward),
            **reward_detail,
            "selector_loss": losses.get("selector", 0.0),
            "migration_loss": losses.get("migration", 0.0),
            "reroute_loss": losses.get("reroute", 0.0),
            "selector_replay": len(self.qmix.replay) if self.qmix is not None else len(self.role_env.selector.replay),
            "migration_replay": len(self.qmix.replay) if self.qmix is not None else len(self.role_env.migrator.replay),
            "reroute_replay": len(self.qmix.replay) if self.qmix is not None else len(self.role_env.rerouter.replay),
            "active_sfts": metrics.get("active_sfts", 0),
            "node_hotspots": metrics.get("node_hotspots", 0),
            "link_hotspots": metrics.get("link_hotspots", 0),
            "avg_delay_total_ms": metrics.get("avg_delay_total_ms", metrics.get("avg_delay_estimate", 0.0)),
            "avg_queueing_delay_ms": metrics.get("avg_queueing_delay_ms", 0.0),
            "total_migrations": metrics.get("total_migrations", 0),
            "total_reconfigs": metrics.get("total_reconfigs", 0),
            "accept_rate": success_count / max(1, episode),
        }

    def save_checkpoint(self, path: str | Path, episode: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
                "episode": int(episode),
                "training_algorithm": self.training_algorithm,
                "selector": self.role_env.selector.state_dict(),
                "migration": self.role_env.migrator.state_dict(),
                "reroute": self.role_env.rerouter.state_dict(),
                "reward_weights": self.reward_weights,
            }
        if self.qmix is not None:
            state["qmix"] = self.qmix.state_dict()
        torch.save(state, str(path))

    def load_checkpoint(self, path: str | Path) -> int:
        state = torch.load(str(path), map_location=self.role_env.selector.device, weights_only=False)
        algorithm = str(state.get("training_algorithm", "independent_dqn"))
        if algorithm != self.training_algorithm:
            raise ValueError(f"role checkpoint algorithm mismatch: {algorithm} != {self.training_algorithm}")
        self.role_env.selector.load_state_dict(state["selector"])
        self.role_env.migrator.load_state_dict(state["migration"])
        self.role_env.rerouter.load_state_dict(state["reroute"])
        if self.qmix is not None:
            if "qmix" not in state:
                raise ValueError("QMIX checkpoint has no mixer state")
            self.qmix.load_state_dict(state["qmix"])
        return int(state.get("episode", 0))

    def _should_reconfigure(self, episode: int) -> bool:
        if self.reconfig_interval <= 0:
            return False
        if episode <= self.warmup_episodes:
            return False
        return episode % self.reconfig_interval == 0

    def _role_epsilons(self) -> Dict[str, float]:
        return {
            "selector": float(getattr(self.role_env.selector.config, "epsilon", 0.0)),
            "migration": float(getattr(self.role_env.migrator.config, "epsilon", 0.0)),
            "reroute": float(getattr(self.role_env.rerouter.config, "epsilon", 0.0)),
        }

    def _update_epsilon(self, episode: int) -> None:
        if self.epsilon_final is None or not self.epsilon_decay_episodes:
            return
        frac = min(1.0, max(0.0, episode / max(1, int(self.epsilon_decay_episodes))))
        for name, agent in [
            ("selector", self.role_env.selector),
            ("migration", self.role_env.migrator),
            ("reroute", self.role_env.rerouter),
        ]:
            start = self.initial_epsilons.get(name, float(agent.config.epsilon))
            agent.config.epsilon = float(start + frac * (float(self.epsilon_final) - start))

    @staticmethod
    def _empty_reward_detail() -> Dict[str, float]:
        return {
            "hotspot_reward": 0.0,
            "delay_reward": 0.0,
            "migration_penalty": 0.0,
            "reroute_penalty": 0.0,
            "tree_penalty": 0.0,
            "failure_penalty": 0.0,
            "noop_penalty": 0.0,
            "coord_reward": 0.0,
        }
