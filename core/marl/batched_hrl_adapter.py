"""Batch HRL scoring for complete-plan candidates.

The legacy online planner performs an environment rollout for every request.
This module provides the first stateless production bridge: it reuses the
loaded HRL encoder and policy heads, builds request-conditioned graph tensors
for a whole micro-batch, and ranks already constructed complete plans.  It
does not advance the legacy environment or mutate a resource ledger.  A pure
candidate generator remains responsible for constructing valid plans, while
the central decoder remains responsible for joint feasibility.

This is intentionally a scoring adapter, not a claim that a rollout has been
vectorized.  The distinction is recorded in ``metadata()`` and in the output
experiment scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import networkx as nx
import numpy as np
import torch

from core.hrl.batched_policy import BatchedHRLPolicy
from core.marl.batch_deployment_wqmix import ResourceSnapshot
from core.marl.online_parallel_pipeline import (
    CandidateGeneration,
    CandidateFactory,
    complete_plan_candidate_factory,
)
from core.marl.deployment_topk import CompletePlanCandidateGenerator


def _module_input_dim(module: Any, default: int) -> int:
    for name in ("topo_input_norm", "input_norm"):
        layer = getattr(module, name, None)
        shape = getattr(layer, "normalized_shape", None)
        if shape:
            return int(shape[0])
    return int(default)


def _request_vector(request: Mapping[str, Any], width: int) -> list[float]:
    """Return the same three request values used by the legacy encoder."""

    cpu = request.get("cpu_origin", request.get("cpu", [])) or []
    memory = request.get("memory_origin", request.get("memory", [])) or []
    values = [
        float(request.get("bw_origin", request.get("bw", 0.0))),
        float(np.mean(list(map(float, cpu)))) if cpu else 0.0,
        float(np.mean(list(map(float, memory)))) if memory else 0.0,
    ]
    if width <= len(values):
        return values[:width]
    return values + [0.0] * (width - len(values))


@dataclass(frozen=True)
class BatchedGraphState:
    node_embeddings: torch.Tensor
    graph_embeddings: torch.Tensor
    dc_indices: tuple[int, ...]


class BatchedTopologyCollator:
    """Build a padded, request-conditioned graph batch from one snapshot."""

    def __init__(
        self,
        profile: Mapping[str, Any],
        encoder: Any,
        *,
        cpu_capacity: float = 55.0,
        memory_capacity: float = 45.0,
    ) -> None:
        self.profile = profile
        self.encoder = encoder
        self.device = next(encoder.parameters()).device
        self.node_dim = _module_input_dim(encoder, 28)
        self.req_dim = int(
            getattr(getattr(encoder, "req_fc", None), "in_features", 0) or 0
        )
        self.cpu_capacity = max(float(cpu_capacity), 1e-9)
        self.memory_capacity = max(float(memory_capacity), 1e-9)
        self.dc_nodes = tuple(sorted(int(value) for value in profile["dc_nodes_1based"]))
        self.node_count = max(
            [int(value) for edge in profile.get("edges", []) for value in (edge["u"], edge["v"])]
            + list(self.dc_nodes)
            + [1]
        )
        graph = nx.Graph()
        for raw in profile.get("edges", []):
            graph.add_edge(int(raw["u"]), int(raw["v"]))
        self.graph = graph
        self.degree = dict(graph.degree())
        self.hops = dict(nx.all_pairs_shortest_path_length(graph))
        self.edge_rows: list[tuple[int, int, float, float]] = []
        default_bw = float(profile.get("default_bandwidth_mbps", 90.0))
        default_delay = float(profile.get("default_delay_ms", 1.0))
        for raw in profile.get("edges", []):
            u, v = int(raw["u"]) - 1, int(raw["v"]) - 1
            bw = float(raw.get("bandwidth_mbps", default_bw))
            delay = float(raw.get("delay_ms", default_delay))
            self.edge_rows.extend(((u, v, bw, delay), (v, u, bw, delay)))
        self.edge_index = torch.tensor(
            [[row[0] for row in self.edge_rows], [row[1] for row in self.edge_rows]],
            dtype=torch.long,
            device=self.device,
        )
        self.base_edge_attr = torch.tensor(
            [
                [1.0, 0.0, edge_delay / max(default_delay, 1e-9), 0.0, 0.0]
                for _, _, _, edge_delay in self.edge_rows
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _node_features(
        self,
        request: Mapping[str, Any],
        snapshot: ResourceSnapshot,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.zeros(
            (self.node_count, self.node_dim), dtype=torch.float32, device=self.device
        )
        source = int(request.get("source_dpid", int(request.get("source", 0)) + 1))
        destinations = {
            int(value)
            for value in (
                request.get("destination_dpids")
                or [int(value) + 1 for value in request.get("dest", [])]
            )
        }
        req_bw = float(request.get("bw_origin", 0.0))
        cpu = request.get("cpu_origin", request.get("cpu", [])) or []
        memory = request.get("memory_origin", request.get("memory", [])) or []
        avg_cpu = float(np.mean(list(map(float, cpu)))) if cpu else 0.0
        avg_memory = float(np.mean(list(map(float, memory)))) if memory else 0.0
        max_degree = max(self.degree.values(), default=1)
        for index in range(self.node_count):
            dpid = index + 1
            cpu_remaining = float(snapshot.cpu_remaining.get(dpid, self.cpu_capacity))
            memory_remaining = float(snapshot.memory_remaining.get(dpid, self.memory_capacity))
            is_dc = dpid in self.dc_nodes
            fit_factor = float(
                is_dc
                and cpu_remaining + 1e-9 >= float(cpu[0])
                and memory_remaining + 1e-9 >= float(memory[0])
                if cpu and memory
                else is_dc
            )
            distances = [
                float(self.hops[dpid][target])
                for target in destinations
                if target in self.hops.get(dpid, {})
            ]
            row = [0.0] * self.node_dim
            # Indices 0-19 follow the authoritative model.yaml layout.
            values = {
                0: cpu_remaining / 100.0,
                1: memory_remaining / 100.0,
                2: 1.0 if fit_factor else -1.0,
                3: 0.5,
                4: 0.5,
                14: 0.0,  # no current tree in a stateless pre-commit state
                15: 0.0,  # no receiver has been committed in this candidate pass
                16: 1.0 if dpid in destinations else 0.0,
                17: 0.0,  # VNF depth is candidate-specific and added by the ranker
                18: 1.0 if is_dc else 0.0,
                19: 0.0,  # progress starts at zero for a complete-plan candidate
                20: min(distances, default=0.0) / max(self.node_count, 1),
                21: float(np.mean(distances)) / max(self.node_count, 1) if distances else 0.0,
                22: max(distances, default=0.0) / max(self.node_count, 1),
                23: min(distances, default=0.0) / max(self.node_count, 1),
                24: min(distances, default=0.0) / max(self.node_count, 1),
                25: float(np.mean(distances)) / max(self.node_count, 1) if distances else 0.0,
                26: 1.0 if distances else 0.0,
                27: 1.0 if distances else 0.0,
            }
            for feature_index, value in values.items():
                if feature_index < self.node_dim:
                    row[feature_index] = float(value)
            # Source and degree are useful only in models with spare feature
            # slots; do not overwrite the fixed legacy indices above.
            if self.node_dim > 5:
                row[5] = 1.0 if dpid == source else 0.0
            if self.node_dim > 6:
                row[6] = float(self.degree.get(dpid, 0)) / max(float(max_degree), 1.0)
            if self.node_dim > 7:
                row[7] = req_bw / max(float(self.profile.get("default_bandwidth_mbps", 90.0)), 1e-9)
            if self.node_dim > 8:
                row[8] = avg_cpu / self.cpu_capacity
            if self.node_dim > 9:
                row[9] = avg_memory / self.memory_capacity
            features[index] = torch.tensor(row, device=self.device)
        dest_mask = torch.zeros(self.node_count, dtype=torch.bool, device=self.device)
        for dpid in destinations:
            if 1 <= dpid <= self.node_count:
                dest_mask[dpid - 1] = True
        return features, dest_mask

    @torch.inference_mode()
    def collate(self, requests: Sequence[Mapping[str, Any]], snapshot: ResourceSnapshot) -> BatchedGraphState:
        if not requests:
            raise ValueError("requests must not be empty")
        node_rows = []
        dest_rows = []
        req_rows = []
        for request in requests:
            x, dest_mask = self._node_features(request, snapshot)
            node_rows.append(x)
            dest_rows.append(dest_mask)
            req_rows.append(_request_vector(request, self.req_dim))
        x = torch.cat(node_rows, dim=0)
        batch = torch.arange(len(requests), device=self.device).repeat_interleave(self.node_count)
        offsets = torch.arange(len(requests), device=self.device).repeat_interleave(self.edge_index.shape[1]) * self.node_count
        edge_index = self.edge_index.repeat(1, len(requests)) + offsets.repeat(2, 1)
        edge_attr_rows = []
        for request in requests:
            for u, v, capacity, delay in self.edge_rows:
                remaining = float(snapshot.bandwidth_remaining.get((u + 1, v + 1), capacity))
                ratio = max(0.0, min(1.0, remaining / max(capacity, 1e-9)))
                edge_attr_rows.append([ratio, 1.0 - ratio, delay / max(float(self.profile.get("default_delay_ms", 1.0)), 1e-9), 0.0, 0.0])
        edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32, device=self.device)
        dest_mask = torch.cat(dest_rows, dim=0)
        req_vec = torch.tensor(req_rows, dtype=torch.float32, device=self.device) if self.req_dim else None
        encoded = self.encoder(
            x,
            edge_index,
            edge_attr,
            batch=batch,
            tree_edge_index=None,
            dest_mask=dest_mask,
            req_vec=req_vec,
        )
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        node_embeddings = encoded.reshape(len(requests), self.node_count, -1)
        graph_embeddings = node_embeddings.mean(dim=1)
        return BatchedGraphState(node_embeddings, graph_embeddings, tuple(node - 1 for node in self.dc_nodes))


class BatchedHRLPolicyRanker:
    """Rank complete plans with one batched high/low HRL policy pass."""

    def __init__(
        self,
        planner: Any,
        profile: Mapping[str, Any],
        *,
        cpu_capacity: float = 55.0,
        memory_capacity: float = 45.0,
    ) -> None:
        high_agent = getattr(getattr(planner, "coordinator", None), "high_agent", None)
        low_agent = getattr(getattr(planner, "coordinator", None), "low_agent", None)
        self.high_policy = getattr(high_agent, "high_policy", None)
        self.low_policy = getattr(low_agent, "low_policy", None)
        self.encoder = getattr(high_agent, "encoder", None)
        if self.high_policy is None or self.low_policy is None or self.encoder is None:
            raise TypeError("loaded planner does not expose encoder, high_policy and low_policy")
        self.policy = BatchedHRLPolicy(self.high_policy, self.low_policy, device=next(self.encoder.parameters()).device)
        self.collator = BatchedTopologyCollator(
            profile,
            self.encoder,
            cpu_capacity=cpu_capacity,
            memory_capacity=memory_capacity,
        )
        self.last_batch = 0
        self.last_error: Optional[str] = None

    @staticmethod
    def _plan_edges(plan: Mapping[str, Any]) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for segment in plan.get("segments") or []:
            path = [int(value) - 1 for value in segment.get("path") or []]
            edges.extend(zip(path, path[1:]))
        multicast = plan.get("multicast") or {}
        for path in (multicast.get("paths") or {}).values():
            path = [int(value) - 1 for value in path]
            edges.extend(zip(path, path[1:]))
        return edges

    @torch.inference_mode()
    def _low_scores(
        self,
        state: BatchedGraphState,
        goals: torch.Tensor,
        requests: Sequence[Mapping[str, Any]],
        generations: Sequence[CandidateGeneration],
    ) -> list[list[float]]:
        rows: list[tuple[int, int, int, int]] = []
        for request_index, generation in enumerate(generations):
            for candidate_index, payload in enumerate(generation.payloads):
                if payload is None:
                    continue
                for current, target in self._plan_edges(payload):
                    if 0 <= current < state.node_embeddings.shape[1] and 0 <= target < state.node_embeddings.shape[1]:
                        rows.append((request_index, candidate_index, current, target))
        scores = [[0.0 for _ in generation.payloads] for generation in generations]
        if not rows:
            return scores
        indices = torch.tensor([[target] for _, _, _, target in rows], dtype=torch.long, device=state.node_embeddings.device)
        current = torch.tensor([current for _, _, current, _ in rows], dtype=torch.long, device=state.node_embeddings.device)
        state_rows = state.node_embeddings[torch.tensor([request_index for request_index, *_ in rows], device=state.node_embeddings.device)]
        goal_rows = goals[torch.tensor([request_index for request_index, *_ in rows], device=state.node_embeddings.device)]
        output = self.policy.forward_low(
            state_rows,
            goal_rows,
            indices,
            current,
            candidate_local_features=torch.zeros((len(rows), 1, 6), device=state.node_embeddings.device),
            candidate_mask=torch.ones((len(rows), 1), dtype=torch.bool, device=state.node_embeddings.device),
        )
        for (request_index, candidate_index, _, _), value in zip(rows, output.candidate_scores[:, 0].tolist()):
            scores[request_index][candidate_index] += float(value)
        counts: dict[tuple[int, int], int] = {}
        for request_index, candidate_index, _, _ in rows:
            counts[(request_index, candidate_index)] = counts.get((request_index, candidate_index), 0) + 1
        for key, count in counts.items():
            scores[key[0]][key[1]] /= max(count, 1)
        return scores

    @torch.inference_mode()
    def __call__(
        self,
        requests: Sequence[Mapping[str, Any]],
        generations: Sequence[CandidateGeneration],
        snapshot: ResourceSnapshot,
        masks: Sequence[Sequence[bool]],
    ) -> Sequence[Sequence[int]]:
        if len(requests) != len(generations):
            raise ValueError("request and candidate counts differ")
        try:
            state = self.collator.collate(requests, snapshot)
            k = len(state.dc_indices)
            candidate_embeddings = state.node_embeddings[:, list(state.dc_indices), :]
            local = torch.zeros((len(requests), k, 9), device=state.node_embeddings.device)
            high = self.policy.forward_high(
                state.graph_embeddings,
                candidate_node_embeddings=candidate_embeddings,
                candidate_local_features=local,
                candidate_mask=torch.ones((len(requests), k), dtype=torch.bool, device=state.node_embeddings.device),
            )
            low = self._low_scores(state, high.goal_embeddings, requests, generations)
            dc_to_score = [dict(zip(self.collator.dc_nodes, row)) for row in high.candidate_scores.tolist()]
            rankings: list[list[int]] = []
            for request_index, generation in enumerate(generations):
                values: list[tuple[float, int]] = []
                for candidate_index, payload in enumerate(generation.payloads):
                    if payload is None:
                        continue
                    chain = [int(value) for value in payload.get("chain_nodes", [])]
                    place = float(np.mean([dc_to_score[request_index].get(node, -1e6) for node in chain])) if chain else -1e6
                    values.append((place + 0.25 * float(low[request_index][candidate_index]), candidate_index))
                values.sort(key=lambda item: (-item[0], item[1]))
                rankings.append([index for _, index in values] + [len(generation.payloads) - 1])
            self.last_batch = len(requests)
            self.last_error = None
            return rankings
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            # A failed policy pass must never disable the hard-feasible path.
            return [list(range(len(generation.payloads))) for generation in generations]

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "batched_hrl_policy_scoring",
            "rollout_vectorized": False,
            "encoder": type(self.encoder).__name__,
            "high_policy": type(self.high_policy).__name__,
            "low_policy": type(self.low_policy).__name__,
            "last_batch": int(self.last_batch),
            "last_error": self.last_error,
        }


class BatchedHRLCandidateAdapter:
    """Pure complete-plan generator plus batched HRL policy ranker."""

    def __init__(
        self,
        planner: Any,
        profile: Mapping[str, Any],
        *,
        max_candidates: int = 8,
        placement_beam: int = 4,
        placement_chains: int = 4,
        pool_limit: int = 16,
        cpu_capacity: float = 55.0,
        memory_capacity: float = 45.0,
    ) -> None:
        self.planner = planner
        self.generator = CompletePlanCandidateGenerator(
            profile,
            max_candidates=max_candidates,
            placement_beam=placement_beam,
            placement_chains=placement_chains,
            pool_limit=pool_limit,
        )
        self.ranker = BatchedHRLPolicyRanker(
            planner,
            profile,
            cpu_capacity=cpu_capacity,
            memory_capacity=memory_capacity,
        )
        self._factory: CandidateFactory = complete_plan_candidate_factory(self.generator)

    def prewarm_paths(self, max_paths: int = 8) -> None:
        self.generator.prewarm_paths(max_paths=max_paths)

    def candidate_factory(self, request: Mapping[str, Any], snapshot: ResourceSnapshot) -> CandidateGeneration:
        return self._factory(request, snapshot)

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "pure_complete_candidates_ranked_by_batched_hrl",
            "generator": type(self.generator).__name__,
            "max_candidates": int(self.generator.max_candidates),
            "ranker": self.ranker.metadata(),
        }

    def close(self) -> None:
        close = getattr(self.planner, "close", None)
        if callable(close):
            close()


__all__ = [
    "BatchedGraphState",
    "BatchedTopologyCollator",
    "BatchedHRLPolicyRanker",
    "BatchedHRLCandidateAdapter",
]
