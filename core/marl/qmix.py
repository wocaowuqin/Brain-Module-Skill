"""Centralized QMIX learner for the three decentralized role policies."""

from __future__ import annotations

from collections import deque
import random
from typing import Any, Deque, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """Monotonic state-conditioned mixer from per-role Q values to Q_tot."""

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = int(n_agents)
        self.embed_dim = int(embed_dim)
        self.hyper_w1 = nn.Linear(state_dim, self.n_agents * self.embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, self.embed_dim)
        self.value = nn.Sequential(nn.Linear(state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(states)).view(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(batch, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(batch, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(batch, self.embed_dim, 1)
        return (torch.bmm(hidden, w2) + self.value(states).view(batch, 1, 1)).view(batch)


class JointReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buffer: Deque[Tuple[Any, ...]] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, observations: Sequence[Iterable[float]], actions: Sequence[int], reward: float,
             next_observations: Sequence[Iterable[float]], done: bool,
             next_valid_actions: Sequence[Iterable[int]], action_dims: Sequence[int]) -> None:
        obs = [list(map(float, item)) for item in observations]
        next_obs = [list(map(float, item)) for item in next_observations]
        masks = []
        for valid_actions, action_dim in zip(next_valid_actions, action_dims):
            valid = set(map(int, valid_actions))
            masks.append([index in valid for index in range(int(action_dim))])
        self.buffer.append((obs, list(map(int, actions)), float(reward), next_obs, bool(done), masks))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, min(int(batch_size), len(self.buffer)))
        obs, actions, rewards, next_obs, dones, next_masks = zip(*batch)
        role_obs = [torch.tensor([row[i] for row in obs], dtype=torch.float32, device=device) for i in range(len(obs[0]))]
        role_next = [torch.tensor([row[i] for row in next_obs], dtype=torch.float32, device=device) for i in range(len(obs[0]))]
        role_next_masks = [torch.tensor([row[i] for row in next_masks], dtype=torch.bool, device=device)
                           for i in range(len(next_masks[0]))]
        return (
            role_obs,
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            role_next,
            torch.tensor(dones, dtype=torch.float32, device=device),
            role_next_masks,
        )


class QMIXLearner:
    """Joint learner that optimizes role Q-networks through a QMIX mixer."""

    def __init__(self, agents: Sequence[Any], mixer_embed_dim: int = 32, lr: float = 1e-3,
                 gamma: float = 0.95, replay_capacity: int = 5000,
                 target_update_interval: int = 50):
        self.agents = list(agents)
        if not self.agents:
            raise ValueError("QMIX requires at least one role agent")
        self.device = self.agents[0].device
        if any(agent.device != self.device for agent in self.agents):
            raise ValueError("all QMIX role agents must use the same device")
        self.state_dim = sum(int(agent.config.obs_dim) for agent in self.agents)
        self.mixer = QMixer(len(self.agents), self.state_dim, mixer_embed_dim).to(self.device)
        self.target_mixer = QMixer(len(self.agents), self.state_dim, mixer_embed_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        parameters = list(self.mixer.parameters())
        for agent in self.agents:
            parameters.extend(agent.q_net.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=float(lr))
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.replay = JointReplayBuffer(replay_capacity)
        self.update_steps = 0

    def push_transition(self, observations, actions, reward, next_observations,
                        next_valid_actions=None, done=False) -> None:
        if next_valid_actions is None:
            next_valid_actions = [range(agent.config.action_dim) for agent in self.agents]
        self.replay.push(observations, actions, reward, next_observations, done,
                         next_valid_actions, [agent.config.action_dim for agent in self.agents])

    def update_from_replay(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self.replay) == 0:
            return {"loss": 0.0, "updated": 0.0}
        obs, actions, rewards, next_obs, dones, next_masks = self.replay.sample(batch_size, self.device)
        chosen_qs = torch.stack([
            agent.q_net(role_obs).gather(1, actions[:, i:i + 1]).squeeze(1)
            for i, (agent, role_obs) in enumerate(zip(self.agents, obs))
        ], dim=1)
        states = torch.cat(obs, dim=1)
        q_tot = self.mixer(chosen_qs, states)
        with torch.no_grad():
            next_qs = torch.stack([
                agent.target_net(role_obs).masked_fill(~mask, -1e9).max(dim=1).values
                for agent, role_obs, mask in zip(self.agents, next_obs, next_masks)
            ], dim=1)
            target_tot = self.target_mixer(next_qs, torch.cat(next_obs, dim=1))
            targets = rewards + self.gamma * (1.0 - dones) * target_tot
        loss = F.smooth_l1_loss(q_tot, targets)
        self.optimizer.zero_grad()
        loss.backward()
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % max(1, self.target_update_interval) == 0:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            for agent in self.agents:
                agent.target_net.load_state_dict(agent.q_net.state_dict())
        return {"loss": float(loss.item()), "updated": 1.0, "q_tot": float(q_tot.mean().item())}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mixer": self.mixer.state_dict(), "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(), "update_steps": self.update_steps,
            "state_dim": self.state_dim,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("state_dim", self.state_dim)) != self.state_dim:
            raise ValueError(f"QMIX state_dim mismatch: checkpoint={state.get('state_dim')} current={self.state_dim}")
        self.mixer.load_state_dict(state["mixer"])
        self.target_mixer.load_state_dict(state.get("target_mixer", state["mixer"]))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.update_steps = int(state.get("update_steps", 0))
