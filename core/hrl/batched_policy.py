"""Batch inference primitives for the two-level HRL policy.

This module contains only tensor-side policy evaluation.  It deliberately
does not reset an environment, mutate a resource manager, or construct paths.
The caller supplies padded graph/node representations and masks, then hands
the selected actions to a pure plan builder and the central ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class BatchedHighOutput:
    q_values: torch.Tensor
    goal_embeddings: torch.Tensor
    candidate_scores: torch.Tensor
    actions: torch.Tensor


@dataclass(frozen=True)
class BatchedLowOutput:
    candidate_scores: torch.Tensor
    actions: torch.Tensor


def _masked_argmax(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.argmax(dim=-1)
    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.shape != values.shape:
        raise ValueError(
            f"action mask shape {tuple(mask.shape)} does not match values {tuple(values.shape)}"
        )
    safe = values.masked_fill(~mask, torch.finfo(values.dtype).min)
    actions = safe.argmax(dim=-1)
    # An all-false row is a malformed state.  Return a deterministic reject or
    # zero action while making the condition observable to the caller.
    empty = ~mask.any(dim=-1)
    return torch.where(empty, torch.zeros_like(actions), actions)


class BatchedHRLPolicy:
    """Vectorized high/low policy heads for already encoded states.

    ``high_policy`` and ``low_policy`` are the existing
    :class:`HighLevelPolicy` and :class:`GoalConditionedLowLevelPolicy`
    instances.  The graph encoder is intentionally outside this class because
    graph sizes and tree masks vary by request; a separate graph collator can
    produce the padded tensors without coupling policy inference to accounting.
    """

    def __init__(self, high_policy, low_policy, *, device: str | torch.device | None = None):
        self.high_policy = high_policy
        self.low_policy = low_policy
        self.device = torch.device(device) if device is not None else getattr(
            high_policy, "device", torch.device("cpu")
        )
        self.high_policy.eval()
        self.low_policy.eval()

    @torch.inference_mode()
    def forward_high(
        self,
        graph_embeddings: torch.Tensor,
        *,
        goal_mask: Optional[torch.Tensor] = None,
        candidate_node_embeddings: Optional[torch.Tensor] = None,
        candidate_local_features: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> BatchedHighOutput:
        """Evaluate all high-level requests in one forward pass.

        Shapes:
          ``graph_embeddings``: ``[B, gnn_dim]``
          ``candidate_node_embeddings``: ``[B, K, gnn_dim]``
          ``candidate_local_features``: ``[B, K, 9]`` (shorter rows are padded)
        """

        graph_embeddings = graph_embeddings.to(self.device)
        if graph_embeddings.ndim != 2:
            raise ValueError("graph_embeddings must have shape [B, gnn_dim]")
        q_values, goal_embeddings, _ = self.high_policy(
            graph_embeddings, return_subgoal=True
        )
        if goal_mask is not None:
            q_values = q_values.masked_fill(
                ~goal_mask.to(device=q_values.device, dtype=torch.bool),
                torch.finfo(q_values.dtype).min,
            )

        batch_size = graph_embeddings.shape[0]
        if candidate_node_embeddings is None:
            candidate_scores = graph_embeddings.new_zeros((batch_size, 0))
            actions = _masked_argmax(q_values, goal_mask)
            return BatchedHighOutput(q_values, goal_embeddings, candidate_scores, actions)

        candidate_node_embeddings = candidate_node_embeddings.to(self.device)
        if candidate_node_embeddings.ndim != 3 or candidate_node_embeddings.shape[0] != batch_size:
            raise ValueError("candidate_node_embeddings must have shape [B, K, gnn_dim]")
        candidate_z = self.high_policy.state_projection(candidate_node_embeddings)
        request_z = self.high_policy.state_projection(graph_embeddings).unsqueeze(1)
        k = candidate_z.shape[1]
        if candidate_local_features is None:
            local = candidate_z.new_zeros((batch_size, k, 9))
        else:
            local = candidate_local_features.to(self.device)
            if local.ndim != 3 or local.shape[:2] != (batch_size, k):
                raise ValueError("candidate_local_features must have shape [B, K, F]")
            if local.shape[-1] < 9:
                local = torch.nn.functional.pad(local, (0, 9 - local.shape[-1]))
            elif local.shape[-1] > 9:
                local = local[..., :9]
        fused = torch.cat(
            [request_z.expand(-1, k, -1), candidate_z, local], dim=-1
        )
        candidate_scores = self.high_policy.goal_candidate_scorer(fused).squeeze(-1)
        actions = _masked_argmax(
            candidate_scores,
            candidate_mask if candidate_mask is not None else torch.ones_like(candidate_scores),
        )
        return BatchedHighOutput(q_values, goal_embeddings, candidate_scores, actions)

    @torch.inference_mode()
    def forward_low(
        self,
        state_embeddings: torch.Tensor,
        goal_embeddings: Optional[torch.Tensor],
        candidate_indices: torch.Tensor,
        current_indices: torch.Tensor,
        *,
        candidate_local_features: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> BatchedLowOutput:
        """Score padded legal next-hop candidates for a whole request batch."""

        state_embeddings = state_embeddings.to(self.device)
        candidate_indices = candidate_indices.to(self.device, dtype=torch.long)
        current_indices = current_indices.to(self.device, dtype=torch.long).view(-1)
        if state_embeddings.ndim != 3 or candidate_indices.ndim != 2:
            raise ValueError("state_embeddings=[B,N,D], candidate_indices=[B,K] required")
        batch_size, node_count, _ = state_embeddings.shape
        if candidate_indices.shape[0] != batch_size or current_indices.shape[0] != batch_size:
            raise ValueError("low-level batch dimensions do not align")
        if (current_indices < 0).any() or (current_indices >= node_count).any():
            raise ValueError("current_indices contains an out-of-range node")
        if (candidate_indices < 0).any() or (candidate_indices >= node_count).any():
            raise ValueError("candidate_indices contains an out-of-range node")

        projected = self.low_policy.state_projection(state_embeddings)
        if goal_embeddings is None:
            goal = projected.new_zeros((batch_size, self.low_policy.hidden_dim))
        else:
            goal = self.low_policy.goal_projection(goal_embeddings.to(self.device))
        k = candidate_indices.shape[1]
        gather_index = candidate_indices.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
        candidate = projected.gather(1, gather_index)
        current = projected[
            torch.arange(batch_size, device=self.device), current_indices
        ].unsqueeze(1).expand(-1, k, -1)
        goal_expanded = goal.unsqueeze(1).expand(-1, k, -1)
        local = projected.new_zeros((batch_size, k, 6))
        if candidate_local_features is not None:
            local = candidate_local_features.to(self.device)
            if local.ndim != 3 or local.shape[:2] != (batch_size, k):
                raise ValueError("candidate_local_features must have shape [B, K, F]")
            if local.shape[-1] < 6:
                local = torch.nn.functional.pad(local, (0, 6 - local.shape[-1]))
            elif local.shape[-1] > 6:
                local = local[..., :6]
        features = torch.cat([candidate, goal_expanded, current, candidate - current, local], dim=-1)
        scores = self.low_policy.candidate_scorer(features).squeeze(-1)
        actions = _masked_argmax(
            scores,
            candidate_mask if candidate_mask is not None else torch.ones_like(scores),
        )
        return BatchedLowOutput(scores, actions)


__all__ = ["BatchedHRLPolicy", "BatchedHighOutput", "BatchedLowOutput"]

