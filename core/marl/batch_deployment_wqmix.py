"""Weighted QMIX components for batched online SFC deployment.

Each active agent represents one request in a micro-batch.  Its actions are
complete deployment-plan candidates, not physical nodes or individual hops.
The policy only ranks candidates; :class:`AtomicResourceLedger` remains the
authoritative hard-constraint boundary for CPU, memory, and bandwidth.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import copy
import math
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NodeAmount = Dict[int, float]
Edge = Tuple[int, int]
EdgeAmount = Dict[Edge, float]


def _positive_amounts(values: Mapping[Any, float]) -> Dict[Any, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if float(value) > 1e-9
    }


@dataclass
class VNFInstanceRequirement:
    node: int
    vnf_type: int
    cpu: float
    memory: float


@dataclass
class ResourceFootprint:
    """Sparse resources required by one complete deployment candidate."""

    cpu: NodeAmount
    memory: NodeAmount
    bandwidth: EdgeAmount
    vnf_instances: Tuple[VNFInstanceRequirement, ...] = ()
    invalid_reason: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        invalid: Optional[str] = None
        for label, values in (("cpu", self.cpu), ("memory", self.memory), ("bandwidth", self.bandwidth)):
            try:
                iterator = values.items()
            except AttributeError:
                invalid = f"{label}_mapping"
                break
            for _, raw in iterator:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    invalid = f"{label}_not_numeric"
                    break
                if not math.isfinite(value) or value < -1e-9:
                    invalid = f"{label}_nonfinite_or_negative"
                    break
            if invalid:
                break
        self.cpu = _positive_amounts(self.cpu)
        self.memory = _positive_amounts(self.memory)
        self.bandwidth = {
            (int(edge[0]), int(edge[1])): amount
            for edge, amount in _positive_amounts(self.bandwidth).items()
        }
        self.vnf_instances = tuple(
            requirement
            if isinstance(requirement, VNFInstanceRequirement)
            else VNFInstanceRequirement(**requirement)
            for requirement in self.vnf_instances
        )
        if invalid is None:
            for requirement in self.vnf_instances:
                if (
                    not math.isfinite(float(requirement.cpu))
                    or not math.isfinite(float(requirement.memory))
                    or float(requirement.cpu) < -1e-9
                    or float(requirement.memory) < -1e-9
                ):
                    invalid = "vnf_requirement_nonfinite_or_negative"
                    break
        self.invalid_reason = invalid

    @classmethod
    def from_sfc_plan(
        cls,
        plan: Mapping[str, Any],
        bandwidth_mbps: float,
        snapshot: Optional["ResourceSnapshot"] = None,
    ) -> "ResourceFootprint":
        """Extract the conservative reservation represented by an HRL plan."""

        cpu: Counter[int] = Counter()
        memory: Counter[int] = Counter()
        placements = plan.get("placement_by_vnf") or {}
        placement_rows = placements.values() if isinstance(placements, Mapping) else placements
        requirements = []
        seen_instances = set()
        for placement in placement_rows:
            node = int(placement["dc_node"])
            vnf_type = int(placement.get("vnf_type", -1))
            cpu_amount = float(placement.get("cpu_units", 0.0))
            memory_amount = float(placement.get("memory_units", 0.0))
            requirements.append(VNFInstanceRequirement(
                node=node,
                vnf_type=vnf_type,
                cpu=cpu_amount,
                memory=memory_amount,
            ))
            instance_key = (node, vnf_type)
            reusable = snapshot is not None and (
                instance_key in snapshot.vnf_instances or instance_key in seen_instances
            )
            if not reusable:
                cpu[node] += cpu_amount
                memory[node] += memory_amount
            seen_instances.add(instance_key)

        edge_uses: Counter[Edge] = Counter()
        for segment in plan.get("segments") or []:
            path = [int(node) for node in segment.get("path") or []]
            edge_uses.update(zip(path, path[1:]))
        multicast = plan.get("multicast") or {}
        tree_edges = multicast.get("tree_edges") or []
        edge_uses.update((int(edge[0]), int(edge[1])) for edge in tree_edges)
        bandwidth = {
            edge: float(count) * float(bandwidth_mbps)
            for edge, count in edge_uses.items()
        }
        return cls(dict(cpu), dict(memory), bandwidth, tuple(requirements))

    def feature_vector(self) -> List[float]:
        """Topology-size-independent aggregate features for candidate scoring."""

        cpu_values = list(self.cpu.values())
        memory_values = list(self.memory.values())
        bandwidth_values = list(self.bandwidth.values())
        return [
            sum(cpu_values),
            sum(memory_values),
            sum(bandwidth_values),
            max(cpu_values, default=0.0),
            max(memory_values, default=0.0),
            max(bandwidth_values, default=0.0),
            float(len(self.cpu)),
            float(len(self.bandwidth)),
        ]


@dataclass(frozen=True)
class ResourceSnapshot:
    version: int
    cpu_remaining: Mapping[int, float]
    memory_remaining: Mapping[int, float]
    bandwidth_remaining: Mapping[Edge, float]
    vnf_instances: Mapping[Tuple[int, int], Tuple[float, float, int]] = field(
        default_factory=dict
    )


def resource_conflict(
    first: ResourceFootprint,
    second: ResourceFootprint,
    snapshot: ResourceSnapshot,
) -> Tuple[float, float, float]:
    if first.vnf_instances or second.vnf_instances:
        combined_instances: Dict[Tuple[int, int], VNFInstanceRequirement] = {}
        for requirement in (*first.vnf_instances, *second.vnf_instances):
            key = (requirement.node, requirement.vnf_type)
            if key in snapshot.vnf_instances:
                continue
            previous = combined_instances.get(key)
            if previous is None or requirement.cpu + requirement.memory > previous.cpu + previous.memory:
                combined_instances[key] = requirement
        combined_cpu: Counter[int] = Counter()
        combined_memory: Counter[int] = Counter()
        for requirement in combined_instances.values():
            combined_cpu[requirement.node] += requirement.cpu
            combined_memory[requirement.node] += requirement.memory
        cpu_conflict = any(
            amount > snapshot.cpu_remaining.get(node, 0.0) + 1e-9
            for node, amount in combined_cpu.items()
        )
        memory_conflict = any(
            amount > snapshot.memory_remaining.get(node, 0.0) + 1e-9
            for node, amount in combined_memory.items()
        )
    else:
        cpu_conflict = any(
            first.cpu.get(node, 0.0) + second.cpu.get(node, 0.0)
            > snapshot.cpu_remaining.get(node, 0.0) + 1e-9
            for node in first.cpu.keys() & second.cpu.keys()
        )
        memory_conflict = any(
            first.memory.get(node, 0.0) + second.memory.get(node, 0.0)
            > snapshot.memory_remaining.get(node, 0.0) + 1e-9
            for node in first.memory.keys() & second.memory.keys()
        )
    bandwidth_conflict = any(
        first.bandwidth.get(edge, 0.0) + second.bandwidth.get(edge, 0.0)
        > snapshot.bandwidth_remaining.get(edge, 0.0) + 1e-9
        for edge in first.bandwidth.keys() & second.bandwidth.keys()
    )
    return float(cpu_conflict), float(memory_conflict), float(bandwidth_conflict)


def candidate_conflict_features(
    candidates: Sequence[Sequence[Optional[ResourceFootprint]]],
    snapshot: ResourceSnapshot,
) -> List[List[List[float]]]:
    """Return four conflict ratios for every request/candidate pair.

    The first three values are CPU, memory, and bandwidth conflict ratios
    against valid candidates of other requests.  The fourth is the ratio that
    conflicts on at least one resource.  A ``None`` footprint is the explicit
    reject action and therefore has no resource conflict.
    """

    result: List[List[List[float]]] = []
    for agent_index, agent_candidates in enumerate(candidates):
        rows: List[List[float]] = []
        for footprint in agent_candidates:
            if footprint is None:
                rows.append([0.0, 0.0, 0.0, 0.0])
                continue
            totals = [0.0, 0.0, 0.0, 0.0]
            comparisons = 0
            for other_index, other_candidates in enumerate(candidates):
                if other_index == agent_index:
                    continue
                for other in other_candidates:
                    if other is None:
                        continue
                    conflicts = resource_conflict(footprint, other, snapshot)
                    for index, value in enumerate(conflicts):
                        totals[index] += value
                    totals[3] += float(any(conflicts))
                    comparisons += 1
            denominator = float(max(1, comparisons))
            rows.append([value / denominator for value in totals])
        result.append(rows)
    return result


def footprint_feasible(
    footprint: Optional[ResourceFootprint],
    snapshot: ResourceSnapshot,
) -> bool:
    """Check one candidate against an immutable resource snapshot."""

    if footprint is None:
        return True
    return (
        all(
            amount <= snapshot.cpu_remaining.get(node, 0.0) + 1e-9
            for node, amount in footprint.cpu.items()
        )
        and all(
            amount <= snapshot.memory_remaining.get(node, 0.0) + 1e-9
            for node, amount in footprint.memory.items()
        )
        and all(
            amount <= snapshot.bandwidth_remaining.get(edge, 0.0) + 1e-9
            for edge, amount in footprint.bandwidth.items()
        )
    )


def candidate_action_mask(
    candidates: Sequence[Sequence[Optional[ResourceFootprint]]],
    snapshot: ResourceSnapshot,
) -> List[List[bool]]:
    """Build the hard per-request candidate mask for a snapshot."""

    return [
        [footprint_feasible(footprint, snapshot) for footprint in request_candidates]
        for request_candidates in candidates
    ]


def deployment_candidate_features(
    candidates: Sequence[Sequence[Optional[ResourceFootprint]]],
    snapshot: ResourceSnapshot,
) -> List[List[List[float]]]:
    """Build 16 topology-independent features for every candidate.

    Features are eight aggregate footprint values, four batch-conflict ratios,
    three peak demand-to-remaining-resource ratios, and one reject indicator.
    Raw aggregate demands should be normalized with training-set statistics.
    """

    conflicts = candidate_conflict_features(candidates, snapshot)
    result: List[List[List[float]]] = []
    for agent_index, request_candidates in enumerate(candidates):
        rows: List[List[float]] = []
        for candidate_index, footprint in enumerate(request_candidates):
            if footprint is None:
                rows.append([0.0] * 15 + [1.0])
                continue

            def peak_ratio(
                demand: Mapping[Any, float],
                remaining: Mapping[Any, float],
            ) -> float:
                return max(
                    (
                        amount / max(remaining.get(resource, 0.0), 1e-9)
                        for resource, amount in demand.items()
                    ),
                    default=0.0,
                )

            rows.append(
                footprint.feature_vector()
                + conflicts[agent_index][candidate_index]
                + [
                    peak_ratio(footprint.cpu, snapshot.cpu_remaining),
                    peak_ratio(footprint.memory, snapshot.memory_remaining),
                    peak_ratio(footprint.bandwidth, snapshot.bandwidth_remaining),
                    0.0,
                ]
            )
        result.append(rows)
    return result


class BatchCandidateQNetwork(nn.Module):
    """Shared parameterized Q-network for variable request candidates."""

    def __init__(
        self,
        request_dim: int,
        candidate_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.request_dim = int(request_dim)
        self.candidate_dim = int(candidate_dim)
        self.hidden_dim = int(hidden_dim)
        self.request_encoder = nn.Sequential(
            nn.Linear(self.request_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Dedicated value head for the explicit no-migration decision.  This
        # prevents action 0 from being treated as an ordinary destination.
        self.no_migration_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        request_observations: torch.Tensor,
        candidate_features: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        if request_observations.ndim != 3 or candidate_features.ndim != 4:
            raise ValueError("expected request [T,A,O] and candidate [T,A,K,F] tensors")
        request_embedding = self.request_encoder(request_observations)
        active = agent_mask.to(request_embedding.dtype).unsqueeze(-1)
        context = (request_embedding * active).sum(dim=1, keepdim=True)
        context = context / active.sum(dim=1, keepdim=True).clamp_min(1.0)
        candidate_embedding = self.candidate_encoder(candidate_features)
        candidate_count = candidate_features.shape[2]
        request_expanded = request_embedding.unsqueeze(2).expand(-1, -1, candidate_count, -1)
        context_expanded = context.unsqueeze(2).expand_as(request_expanded)
        q_values = self.q_head(
            torch.cat((request_expanded, candidate_embedding, context_expanded), dim=-1)
        ).squeeze(-1)
        no_migration = self.no_migration_head(
            torch.cat((request_embedding, context), dim=-1)
        ).squeeze(-1)
        if candidate_count > 0:
            q_values = q_values.clone()
            q_values[:, :, 0] = no_migration
        return q_values


class MaskedQMixer(nn.Module):
    """Monotonic mixer supporting padded request agents."""

    def __init__(self, max_agents: int, state_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.max_agents = int(max_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)
        self.hyper_w1 = nn.Linear(self.state_dim, self.max_agents * self.embed_dim)
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_w2 = nn.Linear(self.state_dim, self.embed_dim)
        self.value = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(
        self,
        agent_qs: torch.Tensor,
        states: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = agent_qs.shape[0]
        if agent_qs.shape[1] != self.max_agents:
            raise ValueError(f"expected {self.max_agents} agents, got {agent_qs.shape[1]}")
        active = agent_mask.to(agent_qs.dtype)
        w1 = torch.abs(self.hyper_w1(states)).view(batch, self.max_agents, self.embed_dim)
        w1 = w1 * active.unsqueeze(-1)
        b1 = self.hyper_b1(states).view(batch, 1, self.embed_dim)
        hidden = F.elu(torch.bmm((agent_qs * active).view(batch, 1, self.max_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(batch, self.embed_dim, 1)
        return (torch.bmm(hidden, w2) + self.value(states).view(batch, 1, 1)).view(batch)


class WeightedQMIXLearner:
    """OW-QMIX-style learner for padded request micro-batches.

    Samples where the current joint value underestimates the TD target receive
    full weight.  Other samples receive ``alpha``.  This is the optimistic
    weighting variant; hard feasibility is deliberately outside the network.
    """

    REQUIRED_BATCH_KEYS = (
        "request_observations",
        "candidate_features",
        "action_mask",
        "agent_mask",
        "states",
        "actions",
        "rewards",
        "next_request_observations",
        "next_candidate_features",
        "next_action_mask",
        "next_agent_mask",
        "next_states",
        "dones",
    )

    def __init__(
        self,
        request_dim: int,
        candidate_dim: int,
        state_dim: int,
        max_agents: int = 32,
        hidden_dim: int = 64,
        mixer_embed_dim: int = 64,
        alpha: float = 0.1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        target_update_interval: int = 200,
        imitation_weight: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.device = torch.device(device)
        self.max_agents = int(max_agents)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.imitation_weight = float(imitation_weight)
        if self.imitation_weight < 0.0:
            raise ValueError("imitation_weight must be non-negative")
        self.target_update_interval = int(target_update_interval)
        self.q_network = BatchCandidateQNetwork(request_dim, candidate_dim, hidden_dim).to(self.device)
        self.target_q_network = BatchCandidateQNetwork(request_dim, candidate_dim, hidden_dim).to(self.device)
        self.mixer = MaskedQMixer(max_agents, state_dim, mixer_embed_dim).to(self.device)
        self.target_mixer = MaskedQMixer(max_agents, state_dim, mixer_embed_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=float(lr),
        )
        self.update_steps = 0

    def _to_device(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        missing = [key for key in self.REQUIRED_BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError(f"missing batch keys: {missing}")
        return {key: value.to(self.device) for key, value in batch.items()}

    @staticmethod
    def _masked_q(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        return q_values.masked_fill(~action_mask.bool(), torch.finfo(q_values.dtype).min)

    def select_actions(
        self,
        request_observations: torch.Tensor,
        candidate_features: torch.Tensor,
        action_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.q_network.eval()
        with torch.inference_mode():
            q_values = self.q_network(
                request_observations.to(self.device),
                candidate_features.to(self.device),
                agent_mask.to(self.device),
            )
            masked = self._masked_q(q_values, action_mask.to(self.device))
            actions = masked.argmax(dim=-1)
            actions = torch.where(agent_mask.to(self.device).bool(), actions, torch.zeros_like(actions))
        self.q_network.train()
        return actions, q_values

    def train_step(self, raw_batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        batch = self._to_device(raw_batch)
        active_has_action = batch["action_mask"].bool().any(dim=-1)
        if not torch.all(active_has_action | ~batch["agent_mask"].bool()):
            raise ValueError("every active request agent needs at least one valid action")
        next_active_has_action = batch["next_action_mask"].bool().any(dim=-1)
        if not torch.all(next_active_has_action | ~batch["next_agent_mask"].bool()):
            raise ValueError("every active next-state agent needs at least one valid action")
        chosen_is_valid = batch["action_mask"].gather(
            -1, batch["actions"].long().unsqueeze(-1)
        ).squeeze(-1).bool()
        if not torch.all(chosen_is_valid | ~batch["agent_mask"].bool()):
            raise ValueError("the training batch contains an invalid chosen action")

        all_qs = self.q_network(
            batch["request_observations"], batch["candidate_features"], batch["agent_mask"]
        )
        chosen_qs = all_qs.gather(-1, batch["actions"].long().unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_qs, batch["states"], batch["agent_mask"])

        with torch.no_grad():
            next_online = self._masked_q(
                self.q_network(
                    batch["next_request_observations"],
                    batch["next_candidate_features"],
                    batch["next_agent_mask"],
                ),
                batch["next_action_mask"],
            )
            next_actions = next_online.argmax(dim=-1)
            next_target_all = self.target_q_network(
                batch["next_request_observations"],
                batch["next_candidate_features"],
                batch["next_agent_mask"],
            )
            next_qs = next_target_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            next_tot = self.target_mixer(
                next_qs, batch["next_states"], batch["next_agent_mask"]
            )
            targets = batch["rewards"].float() + self.gamma * (
                1.0 - batch["dones"].float()
            ) * next_tot

        weights = torch.where(
            q_tot.detach() < targets,
            torch.ones_like(targets),
            torch.full_like(targets, self.alpha),
        )
        per_sample_loss = F.smooth_l1_loss(q_tot, targets, reduction="none")
        td_loss = (weights * per_sample_loss).mean()
        imitation_loss = torch.zeros((), dtype=td_loss.dtype, device=self.device)
        if self.imitation_weight > 0.0 and "teacher_actions" in batch:
            teacher_actions = batch["teacher_actions"].long()
            teacher_is_valid = batch["action_mask"].gather(
                -1, teacher_actions.unsqueeze(-1)
            ).squeeze(-1).bool()
            if not torch.all(teacher_is_valid | ~batch["agent_mask"].bool()):
                raise ValueError("the training batch contains an invalid teacher action")
            active = batch["agent_mask"].bool()
            masked_logits = self._masked_q(all_qs, batch["action_mask"])
            if active.any():
                imitation_loss = F.cross_entropy(
                    masked_logits[active], teacher_actions[active]
                )
        loss = td_loss + self.imitation_weight * imitation_loss
        self.optimizer.zero_grad()
        loss.backward()
        parameters = list(self.q_network.parameters()) + list(self.mixer.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % max(1, self.target_update_interval) == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "imitation_loss": float(imitation_loss.item()),
            "q_tot": float(q_tot.mean().item()),
            "target": float(targets.mean().item()),
            "mean_weight": float(weights.mean().item()),
            "grad_norm": float(grad_norm),
        }


class AtomicResourceLedger:
    """Versioned hard-resource ledger with ranked-candidate fallback."""

    def __init__(
        self,
        cpu_capacity: Mapping[int, float],
        memory_capacity: Mapping[int, float],
        bandwidth_capacity: Mapping[Edge, float],
    ) -> None:
        self.cpu_capacity = _positive_amounts(cpu_capacity)
        self.memory_capacity = _positive_amounts(memory_capacity)
        self.bandwidth_capacity = {
            (int(edge[0]), int(edge[1])): amount
            for edge, amount in _positive_amounts(bandwidth_capacity).items()
        }
        self.cpu_used: Counter[int] = Counter()
        self.memory_used: Counter[int] = Counter()
        self.bandwidth_used: Counter[Edge] = Counter()
        self.allocations: Dict[int, ResourceFootprint] = {}
        self.allocation_bindings: Dict[int, List[Tuple[int, int]]] = {}
        self.vnf_instances: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.prepared_replacements: Dict[str, Dict[str, Any]] = {}
        self.prepared_replacement_by_request: Dict[int, str] = {}
        self._next_preparation_id = 1
        self.version = 0
        self._lock = threading.Lock()

    def snapshot(self) -> ResourceSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def integrity_report(self) -> Dict[str, Any]:
        """Report whether every active allocation has been fully returned."""

        with self._lock:
            cpu_residual = {
                int(node): float(amount)
                for node, amount in self.cpu_used.items()
                if abs(float(amount)) > 1e-9
            }
            memory_residual = {
                int(node): float(amount)
                for node, amount in self.memory_used.items()
                if abs(float(amount)) > 1e-9
            }
            bandwidth_residual = {
                f"{int(edge[0])}->{int(edge[1])}": float(amount)
                for edge, amount in self.bandwidth_used.items()
                if abs(float(amount)) > 1e-9
            }
            fully_released = not (
                self.allocations
                or self.allocation_bindings
                or self.vnf_instances
                or self.prepared_replacements
                or cpu_residual
                or memory_residual
                or bandwidth_residual
            )
            return {
                "fully_released": fully_released,
                "active_allocations": len(self.allocations),
                "active_bindings": len(self.allocation_bindings),
                "active_vnf_instances": len(self.vnf_instances),
                "prepared_replacements": len(self.prepared_replacements),
                "prepared_replacement_tokens": sorted(self.prepared_replacements),
                "prepared_replacement_requests": sorted(
                    self.prepared_replacement_by_request
                ),
                "cpu_residual": cpu_residual,
                "memory_residual": memory_residual,
                "bandwidth_residual": bandwidth_residual,
                "ledger_version": int(self.version),
            }

    def _snapshot_unlocked(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            version=self.version,
            cpu_remaining={
                node: capacity - self.cpu_used[node]
                for node, capacity in self.cpu_capacity.items()
            },
            memory_remaining={
                node: capacity - self.memory_used[node]
                for node, capacity in self.memory_capacity.items()
            },
            bandwidth_remaining={
                edge: capacity - self.bandwidth_used[edge]
                for edge, capacity in self.bandwidth_capacity.items()
            },
            vnf_instances={
                key: (
                    float(value["cpu"]),
                    float(value["memory"]),
                    int(value["ref_count"]),
                )
                for key, value in self.vnf_instances.items()
            },
        )

    def _effective_footprint_unlocked(
        self, footprint: ResourceFootprint
    ) -> ResourceFootprint:
        if not footprint.vnf_instances:
            return footprint
        cpu: Counter[int] = Counter()
        memory: Counter[int] = Counter()
        new_instances = set()
        for requirement in footprint.vnf_instances:
            key = (requirement.node, requirement.vnf_type)
            if key in self.vnf_instances or key in new_instances:
                continue
            cpu[requirement.node] += requirement.cpu
            memory[requirement.node] += requirement.memory
            new_instances.add(key)
        return ResourceFootprint(
            dict(cpu), dict(memory), dict(footprint.bandwidth), footprint.vnf_instances
        )

    def _feasible_unlocked(self, footprint: ResourceFootprint) -> Tuple[bool, str]:
        footprint = self._effective_footprint_unlocked(footprint)
        for node, amount in footprint.cpu.items():
            if self.cpu_used[node] + amount > self.cpu_capacity.get(node, 0.0) + 1e-9:
                return False, "cpu"
        for node, amount in footprint.memory.items():
            if self.memory_used[node] + amount > self.memory_capacity.get(node, 0.0) + 1e-9:
                return False, "memory"
        for edge, amount in footprint.bandwidth.items():
            if self.bandwidth_used[edge] + amount > self.bandwidth_capacity.get(edge, 0.0) + 1e-9:
                return False, "bandwidth"
        return True, ""

    def _reserve_unlocked(
        self,
        request_id: int,
        footprint: ResourceFootprint,
        *,
        increment_version: bool = True,
    ) -> None:
        effective = self._effective_footprint_unlocked(footprint)
        self.cpu_used.update(effective.cpu)
        self.memory_used.update(effective.memory)
        self.bandwidth_used.update(footprint.bandwidth)
        self.allocations[int(request_id)] = footprint
        bindings = []
        for requirement in footprint.vnf_instances:
            key = (requirement.node, requirement.vnf_type)
            instance = self.vnf_instances.get(key)
            if instance is None:
                instance = {
                    "cpu": float(requirement.cpu),
                    "memory": float(requirement.memory),
                    "ref_count": 0.0,
                }
                self.vnf_instances[key] = instance
            instance["ref_count"] += 1.0
            bindings.append(key)
        if bindings:
            self.allocation_bindings[int(request_id)] = bindings
        if increment_version:
            self.version += 1

    def _release_unlocked(self, request_id: int) -> Optional[ResourceFootprint]:
        footprint = self.allocations.pop(int(request_id), None)
        if footprint is None:
            return None
        self.bandwidth_used.subtract(footprint.bandwidth)
        bindings = self.allocation_bindings.pop(int(request_id), [])
        if bindings:
            for key in bindings:
                instance = self.vnf_instances.get(key)
                if instance is None:
                    continue
                instance["ref_count"] -= 1.0
                if instance["ref_count"] <= 0.0:
                    node = key[0]
                    self.cpu_used[node] -= float(instance["cpu"])
                    self.memory_used[node] -= float(instance["memory"])
                    self.vnf_instances.pop(key, None)
        else:
            self.cpu_used.subtract(footprint.cpu)
            self.memory_used.subtract(footprint.memory)
        return footprint

    def _reserve_temporary_unlocked(
        self, footprint: ResourceFootprint
    ) -> ResourceFootprint:
        """Reserve overlap resources without creating an active allocation."""

        effective = self._effective_footprint_unlocked(footprint)
        reserved = ResourceFootprint(
            dict(effective.cpu),
            dict(effective.memory),
            dict(footprint.bandwidth),
        )
        self.cpu_used.update(reserved.cpu)
        self.memory_used.update(reserved.memory)
        self.bandwidth_used.update(reserved.bandwidth)
        return reserved

    def _release_temporary_unlocked(self, footprint: ResourceFootprint) -> None:
        self.cpu_used.subtract(footprint.cpu)
        self.memory_used.subtract(footprint.memory)
        self.bandwidth_used.subtract(footprint.bandwidth)

    def _abort_prepared_unlocked(self, token: str) -> Optional[Dict[str, Any]]:
        prepared = self.prepared_replacements.pop(str(token), None)
        if prepared is None:
            return None
        self._release_temporary_unlocked(prepared["reserved_footprint"])
        self.prepared_replacement_by_request.pop(int(prepared["request_id"]), None)
        return prepared

    def commit_ranked(
        self,
        request_ids: Sequence[int],
        candidates: Sequence[Sequence[Optional[ResourceFootprint]]],
        rankings: Sequence[Sequence[int]],
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Commit a batch under one lock, falling back in ranking order.

        ``None`` is an explicit policy-reject candidate.  A version mismatch is
        reported but does not bypass validation; every candidate is checked
        against the current authoritative ledger before reservation.
        """

        if not (len(request_ids) == len(candidates) == len(rankings)):
            raise ValueError("request_ids, candidates, and rankings must align")
        with self._lock:
            start_version = self.version
            results = []
            for request_id, request_candidates, ranking in zip(request_ids, candidates, rankings):
                request_id = int(request_id)
                if request_id in self.allocations:
                    results.append({
                        "request_id": request_id,
                        "accepted": False,
                        "candidate_index": -1,
                        "reason": "duplicate_request",
                        "attempts": 0,
                    })
                    continue
                selected = -1
                reason = "no_feasible_candidate"
                attempts = 0
                for candidate_index in ranking:
                    candidate_index = int(candidate_index)
                    if not 0 <= candidate_index < len(request_candidates):
                        continue
                    attempts += 1
                    footprint = request_candidates[candidate_index]
                    if footprint is None:
                        reason = "policy_reject"
                        break
                    feasible, failure = self._feasible_unlocked(footprint)
                    if not feasible:
                        reason = f"insufficient_{failure}"
                        continue
                    self._reserve_unlocked(request_id, footprint)
                    selected = candidate_index
                    reason = ""
                    break
                results.append({
                    "request_id": request_id,
                    "accepted": selected >= 0,
                    "candidate_index": selected,
                    "reason": reason,
                    "attempts": attempts,
                })
            return {
                "expected_version": expected_version,
                "start_version": start_version,
                "version_mismatch": (
                    expected_version is not None and int(expected_version) != start_version
                ),
                "end_version": self.version,
                "accepted": sum(int(item["accepted"]) for item in results),
                "results": results,
            }

    def commit_exact(
        self,
        request_ids: Sequence[int],
        candidates: Sequence[Sequence[Optional[ResourceFootprint]]],
        actions: Sequence[int],
        *,
        expected_version: int,
    ) -> Dict[str, Any]:
        """Atomically commit one already-decoded, jointly feasible batch.

        Unlike :meth:`commit_ranked`, this method never repairs or partially
        applies a stale proposal.  A version mismatch, duplicate request,
        invalid action, or jointly infeasible action vector leaves the ledger
        unchanged so the caller can re-decode against a fresh snapshot.
        """

        if not (len(request_ids) == len(candidates) == len(actions)):
            raise ValueError("request_ids, candidates, and actions must align")
        request_ids = [int(request_id) for request_id in request_ids]
        actions = [int(action) for action in actions]
        with self._lock:
            start_version = self.version
            if int(expected_version) != start_version:
                return {
                    "committed": False,
                    "reason": "version_mismatch",
                    "expected_version": int(expected_version),
                    "start_version": start_version,
                    "end_version": self.version,
                    "accepted": 0,
                    "results": [],
                }
            if len(set(request_ids)) != len(request_ids) or any(
                request_id in self.allocations for request_id in request_ids
            ):
                return {
                    "committed": False,
                    "reason": "duplicate_request",
                    "expected_version": int(expected_version),
                    "start_version": start_version,
                    "end_version": self.version,
                    "accepted": 0,
                    "results": [],
                }

            selected: List[Optional[ResourceFootprint]] = []
            for request_candidates, action in zip(candidates, actions):
                if not 0 <= action < len(request_candidates):
                    return {
                        "committed": False,
                        "reason": "invalid_action",
                        "expected_version": int(expected_version),
                        "start_version": start_version,
                        "end_version": self.version,
                        "accepted": 0,
                        "results": [],
                    }
                selected.append(request_candidates[action])

            # Local import avoids coupling the policy-independent decoder to
            # the ledger module during import initialization.
            from core.marl.joint_candidate_decoder import joint_footprints_feasible

            feasible, failure = joint_footprints_feasible(
                selected, self._snapshot_unlocked()
            )
            if not feasible:
                return {
                    "committed": False,
                    "reason": f"joint_infeasible_{failure}",
                    "expected_version": int(expected_version),
                    "start_version": start_version,
                    "end_version": self.version,
                    "accepted": 0,
                    "results": [],
                }

            results = []
            for request_id, action, footprint in zip(
                request_ids, actions, selected
            ):
                if footprint is None:
                    results.append({
                        "request_id": request_id,
                        "accepted": False,
                        "candidate_index": action,
                        "reason": "policy_reject",
                        "attempts": 1,
                    })
                    continue
                self._reserve_unlocked(request_id, footprint)
                results.append({
                    "request_id": request_id,
                    "accepted": True,
                    "candidate_index": action,
                    "reason": "",
                    "attempts": 1,
                })
            return {
                "committed": True,
                "reason": "",
                "expected_version": int(expected_version),
                "start_version": start_version,
                "end_version": self.version,
                "accepted": sum(int(item["accepted"]) for item in results),
                "results": results,
            }

    def release(self, request_id: int) -> bool:
        with self._lock:
            request_id = int(request_id)
            token = self.prepared_replacement_by_request.get(request_id)
            if token is not None:
                self._abort_prepared_unlocked(token)
            footprint = self._release_unlocked(request_id)
            if footprint is None:
                return False
            self.version += 1
            return True

    def prepare_replacement(
        self,
        request_id: int,
        temporary_footprint: ResourceFootprint,
        *,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reserve migration overlap resources while retaining the live plan."""

        request_id = int(request_id)
        with self._lock:
            start_version = self.version
            base = {
                "request_id": request_id,
                "expected_version": expected_version,
                "start_version": start_version,
            }
            if expected_version is not None and int(expected_version) != start_version:
                return {
                    **base,
                    "prepared": False,
                    "token": None,
                    "reason": "version_mismatch",
                    "end_version": self.version,
                }
            if request_id not in self.allocations:
                return {
                    **base,
                    "prepared": False,
                    "token": None,
                    "reason": "request_not_allocated",
                    "end_version": self.version,
                }
            if request_id in self.prepared_replacement_by_request:
                return {
                    **base,
                    "prepared": False,
                    "token": self.prepared_replacement_by_request[request_id],
                    "reason": "replacement_already_prepared",
                    "end_version": self.version,
                }
            feasible, failure = self._feasible_unlocked(temporary_footprint)
            if not feasible:
                return {
                    **base,
                    "prepared": False,
                    "token": None,
                    "reason": f"insufficient_{failure}",
                    "end_version": self.version,
                }
            token = f"replacement-{request_id}-{self._next_preparation_id}"
            self._next_preparation_id += 1
            reserved = self._reserve_temporary_unlocked(temporary_footprint)
            self.prepared_replacements[token] = {
                "request_id": request_id,
                "temporary_footprint": temporary_footprint,
                "reserved_footprint": reserved,
            }
            self.prepared_replacement_by_request[request_id] = token
            self.version += 1
            return {
                **base,
                "prepared": True,
                "token": token,
                "reason": "",
                "end_version": self.version,
            }

    def commit_prepared_replacement(
        self,
        request_id: int,
        token: str,
        new_footprint: ResourceFootprint,
    ) -> Dict[str, Any]:
        """Atomically exchange a prepared overlap and live plan for a new plan."""

        request_id, token = int(request_id), str(token)
        with self._lock:
            start_version = self.version
            prepared = self.prepared_replacements.get(token)
            base = {
                "request_id": request_id,
                "token": token,
                "start_version": start_version,
            }
            if prepared is None:
                return {
                    **base,
                    "replaced": False,
                    "reason": "preparation_not_found",
                    "end_version": self.version,
                }
            if int(prepared["request_id"]) != request_id:
                return {
                    **base,
                    "replaced": False,
                    "reason": "request_mismatch",
                    "end_version": self.version,
                }

            reserved = prepared["reserved_footprint"]
            self._release_temporary_unlocked(reserved)
            old = self._release_unlocked(request_id)
            if old is None:
                self._reserve_temporary_unlocked(reserved)
                return {
                    **base,
                    "replaced": False,
                    "reason": "request_not_allocated",
                    "end_version": self.version,
                }
            feasible, failure = self._feasible_unlocked(new_footprint)
            if not feasible:
                self._reserve_unlocked(request_id, old, increment_version=False)
                restored = self._reserve_temporary_unlocked(
                    prepared["temporary_footprint"]
                )
                prepared["reserved_footprint"] = restored
                return {
                    **base,
                    "replaced": False,
                    "reason": f"insufficient_{failure}",
                    "end_version": self.version,
                }
            self._reserve_unlocked(
                request_id, new_footprint, increment_version=False
            )
            self.prepared_replacements.pop(token, None)
            self.prepared_replacement_by_request.pop(request_id, None)
            self.version += 1
            return {
                **base,
                "replaced": True,
                "reason": "",
                "end_version": self.version,
            }

    def abort_prepared_replacement(self, token: str) -> Dict[str, Any]:
        """Release one migration overlap reservation without changing the live plan."""

        token = str(token)
        with self._lock:
            start_version = self.version
            prepared = self._abort_prepared_unlocked(token)
            if prepared is None:
                return {
                    "aborted": False,
                    "token": token,
                    "request_id": None,
                    "reason": "preparation_not_found",
                    "start_version": start_version,
                    "end_version": self.version,
                }
            self.version += 1
            return {
                "aborted": True,
                "token": token,
                "request_id": int(prepared["request_id"]),
                "reason": "",
                "start_version": start_version,
                "end_version": self.version,
            }

    def replace(
        self,
        request_id: int,
        footprint: ResourceFootprint,
        *,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Atomically replace one active request footprint or leave it unchanged."""

        request_id = int(request_id)
        with self._lock:
            start_version = self.version
            if request_id in self.prepared_replacement_by_request:
                return {
                    "replaced": False,
                    "reason": "replacement_prepared",
                    "expected_version": expected_version,
                    "start_version": start_version,
                    "end_version": self.version,
                }
            if expected_version is not None and int(expected_version) != start_version:
                return {
                    "replaced": False,
                    "reason": "version_mismatch",
                    "expected_version": int(expected_version),
                    "start_version": start_version,
                    "end_version": self.version,
                }
            old = self._release_unlocked(request_id)
            if old is None:
                return {
                    "replaced": False,
                    "reason": "request_not_allocated",
                    "expected_version": expected_version,
                    "start_version": start_version,
                    "end_version": self.version,
                }
            feasible, failure = self._feasible_unlocked(footprint)
            if not feasible:
                self._reserve_unlocked(
                    request_id, old, increment_version=False
                )
                return {
                    "replaced": False,
                    "reason": f"insufficient_{failure}",
                    "expected_version": expected_version,
                    "start_version": start_version,
                    "end_version": self.version,
                }
            self._reserve_unlocked(
                request_id, footprint, increment_version=False
            )
            self.version += 1
            return {
                "replaced": True,
                "reason": "",
                "expected_version": expected_version,
                "start_version": start_version,
                "end_version": self.version,
            }

    def _validate_migration_footprint_unlocked(
        self, footprint: ResourceFootprint
    ) -> str:
        """Return a stable reason for malformed or out-of-topology input."""

        if not isinstance(footprint, ResourceFootprint):
            return "invalid_footprint_type"
        if footprint.invalid_reason:
            return str(footprint.invalid_reason)
        try:
            for node in footprint.cpu:
                if int(node) not in self.cpu_capacity or node != int(node):
                    return "unknown_cpu_node"
            for node in footprint.memory:
                if int(node) not in self.memory_capacity or node != int(node):
                    return "unknown_memory_node"
        except (TypeError, ValueError, OverflowError):
            return "invalid_node_id"
        if any(edge not in self.bandwidth_capacity for edge in footprint.bandwidth):
            return "unknown_bandwidth_edge"
        for requirement in footprint.vnf_instances:
            try:
                node = int(requirement.node)
                key = (node, int(requirement.vnf_type))
                cpu = float(requirement.cpu)
                memory = float(requirement.memory)
            except (TypeError, ValueError, OverflowError):
                return "invalid_vnf_requirement"
            if node not in self.cpu_capacity or node not in self.memory_capacity:
                return "unknown_vnf_node"
            if key in self.vnf_instances:
                instance = self.vnf_instances[key]
                if (
                    abs(float(instance["cpu"]) - cpu) > 1e-9
                    or abs(float(instance["memory"]) - memory) > 1e-9
                ):
                    return "vnf_instance_mismatch"
        return ""

    def apply_migration(
        self,
        request_id: int,
        new_footprint: ResourceFootprint,
        *,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Synchronously apply one migration as an atomic ledger transaction.

        The active allocation is exchanged for ``new_footprint`` under one
        lock.  Shared VNF instances retain their reference-count semantics,
        while source CPU/memory, target CPU/memory, and directed bandwidth are
        updated together.  Every validation failure leaves all ledger state,
        including the version, byte-for-byte unchanged.
        """

        request_id = int(request_id)
        with self._lock:
            start_version = self.version
            base = {
                "request_id": request_id,
                "expected_version": expected_version,
                "start_version": start_version,
            }
            if expected_version is not None and int(expected_version) != start_version:
                return {
                    **base,
                    "applied": False,
                    "replaced": False,
                    "reason": "version_mismatch",
                    "end_version": self.version,
                }
            invalid = self._validate_migration_footprint_unlocked(new_footprint)
            if invalid:
                return {
                    **base,
                    "applied": False,
                    "replaced": False,
                    "reason": invalid,
                    "end_version": self.version,
                }
            token = self.prepared_replacement_by_request.get(request_id)
            if token is not None:
                return {
                    **base,
                    "applied": False,
                    "replaced": False,
                    "reason": "replacement_prepared",
                    "token": token,
                    "end_version": self.version,
                }
            old = self.allocations.get(request_id)
            if old is None:
                return {
                    **base,
                    "applied": False,
                    "replaced": False,
                    "reason": "request_not_allocated",
                    "end_version": self.version,
                }
            if old == new_footprint:
                return {
                    **base,
                    "applied": False,
                    "replaced": False,
                    "reason": "duplicate_request",
                    "end_version": self.version,
                }

            # Keep a complete in-memory transaction image.  The normal
            # release/reserve helpers are then reused, preserving all shared
            # instance and directed-edge accounting rules.  On any failure,
            # restore the image before returning while still holding the lock.
            state = (
                self.cpu_used.copy(),
                self.memory_used.copy(),
                self.bandwidth_used.copy(),
                dict(self.allocations),
                {key: list(value) for key, value in self.allocation_bindings.items()},
                copy.deepcopy(self.vnf_instances),
                self.version,
            )
            try:
                self._release_unlocked(request_id)
                feasible, failure = self._feasible_unlocked(new_footprint)
                if not feasible:
                    (
                        self.cpu_used,
                        self.memory_used,
                        self.bandwidth_used,
                        self.allocations,
                        self.allocation_bindings,
                        self.vnf_instances,
                        self.version,
                    ) = state
                    return {
                        **base,
                        "applied": False,
                        "replaced": False,
                        "reason": f"insufficient_{failure}",
                        "end_version": self.version,
                    }
                self._reserve_unlocked(request_id, new_footprint, increment_version=False)
                self.version += 1
                return {
                    **base,
                    "applied": True,
                    "replaced": True,
                    "reason": "",
                    "end_version": self.version,
                }
            except Exception as exc:
                (
                    self.cpu_used,
                    self.memory_used,
                    self.bandwidth_used,
                    self.allocations,
                    self.allocation_bindings,
                    self.vnf_instances,
                    self.version,
                ) = state
                return {
                    **base,
                    "applied": False,
                    "replaced": False,
                    "reason": "transaction_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "end_version": self.version,
                }


def ranked_actions(q_values: torch.Tensor, action_mask: torch.Tensor) -> List[List[int]]:
    """Convert per-candidate Q values to valid descending rankings."""

    masked = q_values.masked_fill(~action_mask.bool(), torch.finfo(q_values.dtype).min)
    order = masked.argsort(dim=-1, descending=True)
    rankings: List[List[int]] = []
    for agent_index in range(order.shape[0]):
        rankings.append([
            int(index)
            for index in order[agent_index].tolist()
            if bool(action_mask[agent_index, index])
        ])
    return rankings
