"""Trainable policy baselines for bounded batched VNF migration.

The migration agents are temporary: one active VNF migration task is one
agent, and it disappears after the micro-batch.  Consequently the independent
DQN baseline uses fitted one-step candidate returns rather than pretending
that padded agent slot ``i`` denotes the same entity in the next batch.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.marl.batch_deployment_wqmix import BatchCandidateQNetwork


def masked_logits(values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Mask invalid discrete actions without introducing NaNs."""

    return values.masked_fill(~action_mask.bool(), torch.finfo(values.dtype).min)


class BehaviorCloningLearner:
    """Shared candidate policy trained from joint-oracle actions."""

    def __init__(
        self,
        request_dim: int,
        candidate_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.q_network = BatchCandidateQNetwork(
            request_dim, candidate_dim, hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=float(lr))

    def train_step(self, raw_batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        batch = {key: value.to(self.device) for key, value in raw_batch.items()}
        active = batch["agent_mask"].bool()
        logits = self.q_network(
            batch["request_observations"],
            batch["candidate_features"],
            active,
        )
        logits = masked_logits(logits, batch["action_mask"])
        if not active.any():
            raise ValueError("BC batch contains no active migration agents")
        targets = batch["teacher_actions"].long()
        valid = batch["action_mask"].gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1).bool()
        if not torch.all(valid | ~active):
            raise ValueError("BC teacher action is masked")
        loss = F.cross_entropy(logits[active], targets[active])
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        accuracy = (logits.argmax(dim=-1)[active] == targets[active]).float().mean()
        return {
            "loss": float(loss.item()),
            "accuracy": float(accuracy.item()),
            "grad_norm": float(grad_norm),
        }


class IndependentDQNLearner:
    """Fitted one-step Q baseline for ephemeral migration-task agents."""

    def __init__(
        self,
        request_dim: int,
        candidate_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.q_network = BatchCandidateQNetwork(
            request_dim, candidate_dim, hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=float(lr))

    def train_step(self, raw_batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        batch = {key: value.to(self.device) for key, value in raw_batch.items()}
        valid = batch["action_mask"].bool() & batch["agent_mask"].bool().unsqueeze(-1)
        if not valid.any():
            raise ValueError("IDQN batch contains no valid migration actions")
        estimates = self.q_network(
            batch["request_observations"],
            batch["candidate_features"],
            batch["agent_mask"],
        )
        targets = batch["action_rewards"].float()
        loss = F.smooth_l1_loss(estimates[valid], targets[valid])
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "mean_q": float(estimates[valid].mean().item()),
            "mean_target": float(targets[valid].mean().item()),
            "grad_norm": float(grad_norm),
        }


class CentralizedValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


class MAPPOLearner:
    """Shared actor and centralized critic with clipped PPO updates.

    Rollouts are supplied by the caller after the sampled proposals have gone
    through the same joint decoder used online.  The current trace simulator is
    action-independent across records, so returns are intentionally one-step.
    """

    def __init__(
        self,
        request_dim: int,
        candidate_dim: int,
        state_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.q_network = BatchCandidateQNetwork(
            request_dim, candidate_dim, hidden_dim
        ).to(self.device)
        self.critic = CentralizedValueNetwork(state_dim, hidden_dim).to(self.device)
        self.clip_ratio = float(clip_ratio)
        self.value_weight = float(value_weight)
        self.entropy_weight = float(entropy_weight)
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.critic.parameters()),
            lr=float(lr),
        )

    @torch.no_grad()
    def sample_actions(
        self, raw_batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request = raw_batch["request_observations"].to(self.device)
        candidates = raw_batch["candidate_features"].to(self.device)
        agent_mask = raw_batch["agent_mask"].to(self.device).bool()
        action_mask = raw_batch["action_mask"].to(self.device).bool()
        logits = masked_logits(
            self.q_network(request, candidates, agent_mask), action_mask
        )
        inactive = ~agent_mask
        if inactive.any():
            logits = logits.clone()
            logits[inactive] = 0.0
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        actions = torch.where(agent_mask, actions, torch.zeros_like(actions))
        log_probs = distribution.log_prob(actions)
        return actions.cpu(), log_probs.cpu()

    def train_step(
        self,
        raw_batch: Mapping[str, torch.Tensor],
        sampled_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        decoded_returns: torch.Tensor,
    ) -> dict[str, float]:
        batch = {key: value.to(self.device) for key, value in raw_batch.items()}
        actions = sampled_actions.to(self.device).long()
        old_logs = old_log_probs.to(self.device).float()
        returns = decoded_returns.to(self.device).float()
        active = batch["agent_mask"].bool()
        logits = masked_logits(
            self.q_network(
                batch["request_observations"],
                batch["candidate_features"],
                active,
            ),
            batch["action_mask"],
        )
        if (~active).any():
            logits = logits.clone()
            logits[~active] = 0.0
        distribution = torch.distributions.Categorical(logits=logits)
        new_logs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic(batch["states"])
        advantages = (returns - values.detach()).unsqueeze(-1).expand_as(new_logs)
        ratios = torch.exp(new_logs - old_logs)
        unclipped = ratios * advantages
        clipped = ratios.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        if not active.any():
            raise ValueError("MAPPO batch contains no active migration agents")
        actor_loss = -torch.minimum(unclipped, clipped)[active].mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_mean = entropy[active].mean()
        loss = (
            actor_loss
            + self.value_weight * critic_loss
            - self.entropy_weight * entropy_mean
        )
        self.optimizer.zero_grad()
        loss.backward()
        parameters = list(self.q_network.parameters()) + list(self.critic.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_mean.item()),
            "mean_return": float(returns.mean().item()),
            "grad_norm": float(grad_norm),
        }
