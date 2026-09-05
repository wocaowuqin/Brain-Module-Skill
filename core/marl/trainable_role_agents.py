"""Trainable DQN role agents for SFT reconfiguration.

These agents keep the same observe/select_action surface as the rule-backed
roles. They are intentionally small: a fixed-size feature encoder, an MLP
Q-network, a target network, and a replay buffer. Concrete action executors
still live in ReconfigurationManager.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.marl.role_agents import RoleDecision


TensorBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class DQNRoleConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 64
    lr: float = 1e-3
    gamma: float = 0.95
    epsilon: float = 0.10
    target_update_interval: int = 50
    replay_capacity: int = 5000
    batch_size: int = 64
    device: str = "cpu"


class RoleReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buffer: Deque[Tuple[List[float], int, float, List[float], bool, List[bool]]] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, obs: Iterable[float], action: int, reward: float,
             next_obs: Iterable[float], done: bool, next_valid_actions: Iterable[int],
             action_dim: int) -> None:
        valid = set(int(action) for action in next_valid_actions)
        mask = [index in valid for index in range(int(action_dim))]
        self.buffer.append((list(obs), int(action), float(reward), list(next_obs), bool(done), mask))

    def sample(self, batch_size: int, device: torch.device) -> TensorBatch:
        batch = random.sample(self.buffer, min(int(batch_size), len(self.buffer)))
        obs, actions, rewards, next_obs, dones, next_masks = zip(*batch)
        return (
            torch.tensor(obs, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_obs, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
            torch.tensor(next_masks, dtype=torch.bool, device=device),
        )


class RoleQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrainableDQNRoleAgent:
    """Base class for small DQN role agents."""

    def __init__(self, role: str, config: DQNRoleConfig):
        self.role = role
        self.config = config
        self.device = torch.device(config.device)
        self.q_net = RoleQNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_net = RoleQNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(config.lr))
        self.replay = RoleReplayBuffer(config.replay_capacity)
        self.update_steps = 0
        self.last_obs_vector: Optional[List[float]] = None
        self.last_action: Optional[int] = None

    def select_discrete_action(self, obs_vector: Iterable[float],
                               valid_actions: Optional[Iterable[int]] = None) -> Tuple[int, float]:
        obs = list(float(x) for x in obs_vector)
        valid = list(valid_actions) if valid_actions is not None else list(range(self.config.action_dim))
        if not valid:
            valid = [0]
        epsilon = float(self.config.epsilon)
        # Evaluation with epsilon=0 must not advance the process-wide RNG.
        # The deployment HRL shares that RNG, so an unnecessary draw makes
        # reconfiguration and no-reconfiguration A/B traces incomparable.
        if epsilon > 0.0 and random.random() < epsilon:
            action = int(random.choice(valid))
            score = 0.0
        else:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_net(x)[0]
                mask = torch.full_like(q_values, -1e9)
                mask[valid] = 0.0
                masked = q_values + mask
                action = int(torch.argmax(masked).item())
                score = float(q_values[action].item())
        self.last_obs_vector = obs
        self.last_action = action
        return action, score

    def push_transition(self, obs: Iterable[float], action: int, reward: float,
                        next_obs: Iterable[float], done: bool,
                        next_valid_actions: Optional[Iterable[int]] = None) -> None:
        valid = list(next_valid_actions) if next_valid_actions is not None else list(range(self.config.action_dim))
        self.replay.push(obs, action, reward, next_obs, done, valid, self.config.action_dim)

    def update_from_replay(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        if len(self.replay) == 0:
            return {"loss": 0.0, "updated": 0.0}
        bs = int(batch_size or self.config.batch_size)
        obs, actions, rewards, next_obs, dones, next_masks = self.replay.sample(bs, self.device)
        q = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = self.target_net(next_obs).masked_fill(~next_masks, -1e9)
            next_q = next_values.max(dim=1).values
            target = rewards + float(self.config.gamma) * (1.0 - dones) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % int(self.config.target_update_interval) == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return {"loss": float(loss.item()), "updated": 1.0}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "config": self.config.__dict__,
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state.get("target_net", state["q_net"]))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.update_steps = int(state.get("update_steps", 0))


class TrainableSFTSelectionAgent(TrainableDQNRoleAgent):
    """DQN selector: action 0 is noop, action 1..K selects Top-K rank."""

    RISK_DIM = 6
    METRIC_DIM = 7

    def __init__(self, top_k: int = 3, min_risk_score: float = 0.0,
                 hidden_dim: int = 64, lr: float = 1e-3,
                 epsilon: float = 0.10, device: str = "cpu"):
        self.top_k = int(top_k)
        self.min_risk_score = float(min_risk_score)
        obs_dim = self.METRIC_DIM + self.top_k * self.RISK_DIM
        super().__init__(
            "sft_selector",
            DQNRoleConfig(
                obs_dim=obs_dim,
                action_dim=self.top_k + 1,
                hidden_dim=hidden_dim,
                lr=lr,
                epsilon=epsilon,
                device=device,
            ),
        )

    def observe(self, manager) -> Dict[str, Any]:
        risks = manager.select_topk_risky_sfts(self.top_k)
        metrics = manager.metrics_snapshot()
        valid_actions = [0] + [idx for idx, risk in enumerate(risks[: self.top_k], start=1)
                               if risk.score > self.min_risk_score]
        return {
            "risks": risks,
            "metrics": metrics,
            "obs_vector": self._vectorize(metrics, risks),
            "valid_actions": valid_actions,
        }

    def select_action(self, obs: Dict[str, Any]) -> RoleDecision:
        risks = obs.get("risks", [])
        valid = obs.get("valid_actions", [0])
        action, q_score = self.select_discrete_action(obs["obs_vector"], valid)
        if action <= 0 or action - 1 >= len(risks):
            return RoleDecision(
                role=self.role,
                action=0,
                label="no_target",
                score=q_score,
                reason="DQN selected no-op or no valid risky SFT",
            )
        target = risks[action - 1]
        return RoleDecision(
            role=self.role,
            action=action,
            label=f"select_top{action}",
            req_id=target.req_id,
            score=float(q_score),
            proposal={"risk": target.__dict__},
            reason="DQN selected a Top-K risky SFT",
        )

    def _vectorize(self, metrics: Dict[str, Any], risks: List[Any]) -> List[float]:
        vec = _metrics_vector(metrics)
        for idx in range(self.top_k):
            if idx < len(risks):
                risk = risks[idx]
                vec.extend([
                    _squash(risk.score, 20.0),
                    _squash(risk.link_hot_score, 20.0),
                    _squash(risk.node_hot_score, 10.0),
                    _squash(risk.delay_estimate, 1000.0),
                    _squash(risk.migration_count, 10.0),
                    _squash(risk.reconfig_count, 10.0),
                ])
            else:
                vec.extend([0.0] * self.RISK_DIM)
        return vec


class TrainableVNFMigrationAgent(TrainableDQNRoleAgent):
    """DQN migration role: action 0 no-op, action 1 execute planned migration."""

    def __init__(self, min_gain: float = 0.0, hidden_dim: int = 64,
                 lr: float = 1e-3, epsilon: float = 0.10, device: str = "cpu"):
        self.min_gain = float(min_gain)
        super().__init__(
            "vnf_migration",
            DQNRoleConfig(obs_dim=10, action_dim=2, hidden_dim=hidden_dim,
                          lr=lr, epsilon=epsilon, device=device),
        )

    def observe(self, manager, req_id: Optional[int]) -> Dict[str, Any]:
        action = manager.plan_greedy_migrate(req_id=req_id) if req_id is not None else None
        metrics = manager.metrics_snapshot()
        feasible = action is not None and action.action_type != "noop" and action.estimated_gain > self.min_gain
        return {
            "planned_action": action,
            "metrics": metrics,
            "obs_vector": _proposal_vector(metrics, action),
            "valid_actions": [0, 1] if feasible else [0],
        }

    def select_action(self, obs: Dict[str, Any]) -> RoleDecision:
        action_obj = obs.get("planned_action")
        feasible = (
            action_obj is not None
            and action_obj.action_type != "noop"
            and action_obj.estimated_gain > self.min_gain
        )
        valid = obs.get("valid_actions", [0, 1] if feasible else [0])
        action, q_score = self.select_discrete_action(obs["obs_vector"], valid)
        if action == 0 or not feasible:
            return RoleDecision(
                role=self.role,
                action=0,
                label="no_migration",
                req_id=getattr(action_obj, "req_id", None),
                score=q_score,
                reason=getattr(action_obj, "reason", "DQN selected no migration"),
            )
        return RoleDecision(
            role=self.role,
            action=1,
            label="migrate_hottest_vnf",
            req_id=action_obj.req_id,
            score=float(q_score),
            proposal=action_obj.to_dict(),
            reason="DQN accepted the migration proposal",
        )


class TrainableTreeRerouteAgent(TrainableDQNRoleAgent):
    """DQN reroute role: action 0 no-op, action 1 execute planned reroute."""

    def __init__(self, min_gain: float = 0.0, hidden_dim: int = 64,
                 lr: float = 1e-3, epsilon: float = 0.10, device: str = "cpu"):
        self.min_gain = float(min_gain)
        super().__init__(
            "tree_reroute",
            DQNRoleConfig(obs_dim=10, action_dim=2, hidden_dim=hidden_dim,
                          lr=lr, epsilon=epsilon, device=device),
        )

    def observe(self, manager, req_id: Optional[int]) -> Dict[str, Any]:
        action = manager.plan_greedy_reroute(req_id=req_id) if req_id is not None else None
        metrics = manager.metrics_snapshot()
        feasible = action is not None and action.action_type != "noop" and action.estimated_gain > self.min_gain
        return {
            "planned_action": action,
            "metrics": metrics,
            "obs_vector": _proposal_vector(metrics, action),
            "valid_actions": [0, 1] if feasible else [0],
        }

    def select_action(self, obs: Dict[str, Any]) -> RoleDecision:
        action_obj = obs.get("planned_action")
        feasible = (
            action_obj is not None
            and action_obj.action_type != "noop"
            and action_obj.estimated_gain > self.min_gain
        )
        valid = obs.get("valid_actions", [0, 1] if feasible else [0])
        action, q_score = self.select_discrete_action(obs["obs_vector"], valid)
        if action == 0 or not feasible:
            return RoleDecision(
                role=self.role,
                action=0,
                label="no_reroute",
                req_id=getattr(action_obj, "req_id", None),
                score=q_score,
                reason=getattr(action_obj, "reason", "DQN selected no reroute"),
            )
        return RoleDecision(
            role=self.role,
            action=1,
            label="reroute_hottest_edge",
            req_id=action_obj.req_id,
            score=float(q_score),
            proposal=action_obj.to_dict(),
            reason="DQN accepted the reroute proposal",
        )


def _metrics_vector(metrics: Dict[str, Any]) -> List[float]:
    return [
        _squash(metrics.get("active_sfts", 0.0), 50.0),
        _squash(metrics.get("node_hotspots", 0.0), 50.0),
        _squash(metrics.get("link_hotspots", 0.0), 100.0),
        _squash(metrics.get("total_tree_edges", 0.0), 1000.0),
        _squash(metrics.get("avg_delay_total_ms", metrics.get("avg_delay_estimate", 0.0)), 1000.0),
        _squash(metrics.get("avg_queueing_delay_ms", 0.0), 1000.0),
        _squash(metrics.get("max_delay_total_ms", 0.0), 2000.0),
    ]


def _proposal_vector(metrics: Dict[str, Any], action: Optional[Any]) -> List[float]:
    target = getattr(action, "target", None) or {}
    new_path = target.get("new_path", []) if isinstance(target, dict) else []
    safety = target.get("safety", {}) if isinstance(target, dict) else {}
    return [
        _squash(getattr(action, "estimated_gain", 0.0) if action is not None else 0.0, 1.0),
        1.0 if action is not None and getattr(action, "action_type", "noop") != "noop" else 0.0,
        *_metrics_vector(metrics),
        _squash(safety.get("extra_edges", max(0, len(new_path) - 2)), 2.0),
    ]


def _squash(value: Any, scale: float) -> float:
    value = max(0.0, float(value or 0.0))
    return value / (value + max(float(scale), 1e-9))
