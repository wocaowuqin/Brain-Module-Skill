"""Heuristic SFT reconfiguration manager for research content 2.

This module is the stage-2 bridge between the existing MSFT-HIRL deployment
system and the later MARL layer. It deliberately starts with deterministic,
auditable heuristics:

- detect overloaded nodes and links,
- select Top-K risky active multicast SFTs,
- provide baseline decisions,
- safely execute small local migration/rerouting actions.

The MARL environment can later reuse these methods as action executors and
baseline comparators.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, asdict
import hashlib
import heapq
import json
import logging
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


Edge = Tuple[int, int]


@dataclass
class Hotspot:
    kind: str
    key: Any
    utilization: float
    capacity: float
    available: float


@dataclass
class SftRisk:
    req_id: int
    score: float
    link_hot_score: float
    node_hot_score: float
    delay_estimate: float
    migration_count: int
    reconfig_count: int


@dataclass
class DelayBreakdown:
    propagation_ms: float
    processing_ms: float
    queueing_ms: float
    reconfiguration_ms: float

    @property
    def total_ms(self) -> float:
        return (
            self.propagation_ms
            + self.processing_ms
            + self.queueing_ms
            + self.reconfiguration_ms
        )

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["total_ms"] = self.total_ms
        return data


@dataclass
class ReconfigAction:
    baseline: str
    req_id: Optional[int]
    action_type: str
    target: Optional[Any] = None
    reason: str = ""
    estimated_gain: float = 0.0
    applied: bool = False
    success: bool = False
    status_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReconfigurationManager:
    """Low-disturbance SFT reconfiguration baseline manager."""

    def __init__(
        self,
        env_or_resource_mgr,
        node_util_threshold: float = 0.80,
        link_util_threshold: float = 0.80,
        delay_threshold: Optional[float] = None,
        max_path_hops: int = 6,
        processing_arrival_rate: float = 0.05,
        queueing_delay_weight: float = 1.0,
        migration_delay_penalty: float = 0.20,
        reroute_delay_penalty: float = 0.10,
        max_tree_edge_growth: int = 2,
        max_reroute_delay_ratio: float = 1.25,
        max_reroute_delay_increase_ms: float = 2.0,
        path_utilization_weight: float = 2.0,
        path_delay_weight: float = 0.05,
        max_projected_link_utilization: float = 0.90,
        max_reroute_candidates: int = 8,
        max_migration_target_utilization: float = 0.75,
        min_residual_connectivity_ratio: float = 0.0,
    ):
        self.env = env_or_resource_mgr if hasattr(env_or_resource_mgr, "resource_mgr") else None
        self.rm = getattr(env_or_resource_mgr, "resource_mgr", env_or_resource_mgr)
        self.node_util_threshold = float(node_util_threshold)
        self.link_util_threshold = float(link_util_threshold)
        self.delay_threshold = delay_threshold
        self.max_path_hops = int(max_path_hops)
        self.processing_arrival_rate = float(processing_arrival_rate)
        self.queueing_delay_weight = float(queueing_delay_weight)
        self.migration_delay_penalty = float(migration_delay_penalty)
        self.reroute_delay_penalty = float(reroute_delay_penalty)
        self.max_tree_edge_growth = int(max_tree_edge_growth)
        self.max_reroute_delay_ratio = float(max_reroute_delay_ratio)
        self.max_reroute_delay_increase_ms = float(max_reroute_delay_increase_ms)
        self.path_utilization_weight = float(path_utilization_weight)
        self.path_delay_weight = float(path_delay_weight)
        self.max_projected_link_utilization = float(max_projected_link_utilization)
        self.max_reroute_candidates = int(max_reroute_candidates)
        self.max_migration_target_utilization = float(max_migration_target_utilization)
        self.min_residual_connectivity_ratio = float(min_residual_connectivity_ratio)
        self._reroute_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Inventory and metrics
    # ------------------------------------------------------------------
    def active_sft_records(self):
        return [
            rec for rec in self.rm.request_table.values()
            if rec.state == "ACTIVE" and rec.tree_edges
        ]

    def node_hotspots(self) -> List[Hotspot]:
        pool = self.rm.pool
        hotspots: List[Hotspot] = []
        for node in range(self.rm.n):
            cpu_cap = float(pool.cpu_cap[node])
            mem_cap = float(pool.mem_cap[node])
            cpu_avail = float(pool.get_available_cpu(node))
            mem_avail = float(pool.get_available_memory(node))
            cpu_util = 1.0 - cpu_avail / max(cpu_cap, 1e-9)
            mem_util = 1.0 - mem_avail / max(mem_cap, 1e-9)
            util = max(cpu_util, mem_util)
            if util >= self.node_util_threshold:
                hotspots.append(Hotspot("node", node, util, max(cpu_cap, mem_cap), min(cpu_avail, mem_avail)))
        return sorted(hotspots, key=lambda h: h.utilization, reverse=True)

    def link_hotspots(self) -> List[Hotspot]:
        pool = self.rm.pool
        hotspots: List[Hotspot] = []
        for edge in pool.iter_bandwidth_keys():
            cap = float(pool.bw_cap.get(edge, 0.0))
            avail = float(pool.get_available_bandwidth(*edge))
            util = 1.0 - avail / max(cap, 1e-9)
            if util >= self.link_util_threshold:
                hotspots.append(Hotspot("link", edge, util, cap, avail))
        return sorted(hotspots, key=lambda h: h.utilization, reverse=True)

    def estimate_sft_delay_components(self, rec) -> DelayBreakdown:
        """Estimate SFT delay as propagation, processing, queueing, and reconfig cost."""
        propagation = 0.0
        queueing = 0.0
        topo = getattr(self.rm, "topo", None)
        for u, v in rec.tree_edges:
            base_delay = self._edge_delay(u, v, topo)
            propagation += base_delay
            queueing += self._edge_queueing_delay(u, v, base_delay)

        processing = 0.0
        for detail in rec.placement_detail.values():
            node = int(detail.get("node", 0))
            cpu = max(float(detail.get("cpu_used", 1.0)), 1e-6)
            try:
                avail = max(float(self.rm.pool.get_available_cpu(node)), cpu)
            except Exception:
                avail = cpu
            service_rate = max(avail / cpu, 1e-6)
            processing += 1.0 / max(service_rate - self.processing_arrival_rate, 1e-6)

        reconfiguration = (
            float(getattr(rec, "migration_count", 0)) * self.migration_delay_penalty
            + float(getattr(rec, "reconfig_count", 0)) * self.reroute_delay_penalty
        )
        return DelayBreakdown(
            propagation_ms=float(propagation),
            processing_ms=float(processing),
            queueing_ms=float(queueing),
            reconfiguration_ms=float(reconfiguration),
        )

    def estimate_sft_delay(self, rec) -> float:
        """Backward-compatible total delay estimate."""
        return self.estimate_sft_delay_components(rec).total_ms

    def score_sft_risk(self, rec, node_hot: Optional[Dict[int, float]] = None,
                       link_hot: Optional[Dict[Edge, float]] = None) -> SftRisk:
        if node_hot is None:
            node_hot = {h.key: h.utilization for h in self.node_hotspots()}
        if link_hot is None:
            link_hot = {h.key: h.utilization for h in self.link_hotspots()}

        link_score = sum(link_hot.get(edge, 0.0) for edge in rec.tree_edges)
        node_score = sum(node_hot.get(node, 0.0) for node in rec.placement_by_vnf.values())
        delay = self.estimate_sft_delay(rec)
        delay_score = 0.0
        if self.delay_threshold is not None:
            delay_score = max(0.0, delay - float(self.delay_threshold)) / max(float(self.delay_threshold), 1e-9)
        score = link_score + node_score + delay_score + 0.05 * rec.migration_count + 0.03 * rec.reconfig_count
        return SftRisk(
            req_id=rec.req_id,
            score=float(score),
            link_hot_score=float(link_score),
            node_hot_score=float(node_score),
            delay_estimate=float(delay),
            migration_count=int(rec.migration_count),
            reconfig_count=int(rec.reconfig_count),
        )

    def select_topk_risky_sfts(self, k: int = 3) -> List[SftRisk]:
        node_hot = {h.key: h.utilization for h in self.node_hotspots()}
        link_hot = {h.key: h.utilization for h in self.link_hotspots()}
        risks = [self.score_sft_risk(rec, node_hot, link_hot) for rec in self.active_sft_records()]
        risks.sort(key=lambda r: r.score, reverse=True)
        return risks[: max(0, int(k))]

    def metrics_snapshot(self) -> Dict[str, Any]:
        active = self.active_sft_records()
        total_edges = sum(len(r.tree_edges) for r in active)
        total_migrations = sum(r.migration_count for r in active)
        total_reconfigs = sum(r.reconfig_count for r in active)
        breakdowns = [self.estimate_sft_delay_components(r) for r in active]
        delays = [d.total_ms for d in breakdowns]
        avg_prop = self._avg(d.propagation_ms for d in breakdowns)
        avg_proc = self._avg(d.processing_ms for d in breakdowns)
        avg_queue = self._avg(d.queueing_ms for d in breakdowns)
        avg_reconfig = self._avg(d.reconfiguration_ms for d in breakdowns)
        return {
            "active_sfts": len(active),
            "node_hotspots": len(self.node_hotspots()),
            "link_hotspots": len(self.link_hotspots()),
            "total_tree_edges": total_edges,
            "total_migrations": total_migrations,
            "total_reconfigs": total_reconfigs,
            "avg_delay_estimate": sum(delays) / len(delays) if delays else 0.0,
            "avg_delay_total_ms": sum(delays) / len(delays) if delays else 0.0,
            "avg_propagation_delay_ms": avg_prop,
            "avg_processing_delay_ms": avg_proc,
            "avg_queueing_delay_ms": avg_queue,
            "avg_reconfiguration_delay_ms": avg_reconfig,
            "max_delay_total_ms": max(delays) if delays else 0.0,
        }

    # ------------------------------------------------------------------
    # Baselines and action execution
    # ------------------------------------------------------------------
    def plan_no_reconfig(self) -> ReconfigAction:
        return ReconfigAction(
            baseline="No-Reconfig",
            req_id=None,
            action_type="noop",
            reason="keep existing SFTs unchanged",
        )

    def plan_full_redeploy(self, req_id: Optional[int] = None) -> ReconfigAction:
        risks = self.select_topk_risky_sfts(1) if req_id is None else []
        chosen = int(req_id) if req_id is not None else (risks[0].req_id if risks else None)
        return ReconfigAction(
            baseline="Full-Redeploy",
            req_id=chosen,
            action_type="redeploy_request",
            target=chosen,
            reason="planned baseline; actual redeploy should call the MSFT-HIRL deployment pipeline",
        )

    def plan_greedy_migrate(self, req_id: Optional[int] = None) -> ReconfigAction:
        rec = self._get_record_for_action(req_id)
        if rec is None:
            return ReconfigAction("Greedy-Migrate", None, "noop", reason="no active SFT")

        node_hot = {h.key: h.utilization for h in self.node_hotspots()}
        tree_nodes = {int(rec.source)}
        for u, v in rec.tree_edges:
            tree_nodes.update((int(u), int(v)))
        ranked_bindings = sorted(
            rec.vnf_bindings,
            key=lambda b: node_hot.get(b.node, 0.0),
            reverse=True,
        )
        for binding in ranked_bindings:
            if int(binding.node) not in node_hot:
                continue
            inst = self.rm.instance_table.get(binding.inst_id)
            if binding.reused or inst is None or inst.state != "ACTIVE" or inst.ref_count != 1:
                continue
            vnf_idx = self._binding_vnf_index(rec, binding)
            if vnf_idx is None:
                continue
            target = self._best_migration_target(
                binding.node,
                binding.vnf_type,
                binding.cpu,
                binding.mem,
                allowed_nodes=tree_nodes,
            )
            if target is not None:
                projected_target = self._projected_node_utilization(target, binding.cpu, binding.mem)
                projected_source = self._projected_node_utilization(
                    binding.node, -binding.cpu, -binding.mem
                )
                before_peak = max(
                    self._node_utilization(binding.node), self._node_utilization(target)
                )
                after_peak = max(projected_source, projected_target)
                gain = before_peak - after_peak
                if gain <= 0.0:
                    continue
                return ReconfigAction(
                    baseline="Greedy-Migrate",
                    req_id=rec.req_id,
                    action_type="migrate_vnf",
                    target={
                        "vnf_type": binding.vnf_type,
                        "old_node": binding.node,
                        "new_node": target,
                        "cpu": binding.cpu,
                        "mem": binding.mem,
                        "inst_id": binding.inst_id,
                        "vnf_idx": int(vnf_idx),
                        "state_fingerprint": self.request_state_fingerprint(rec.req_id),
                        "projected_target_utilization": float(projected_target),
                    },
                    reason="move a VNF away from the hottest hosting node",
                    estimated_gain=float(gain),
                )
        return ReconfigAction("Greedy-Migrate", rec.req_id, "noop", reason="no feasible migration target")

    def apply_greedy_migrate(self, req_id: Optional[int] = None) -> ReconfigAction:
        action = self.plan_greedy_migrate(req_id)
        if action.action_type != "migrate_vnf" or not action.target:
            return action
        return self.apply_migration_proposal(action)

    def apply_migration_proposal(self, proposal: Any) -> ReconfigAction:
        """Revalidate and atomically apply the exact migration selected by a policy."""
        payload = proposal.to_dict() if isinstance(proposal, ReconfigAction) else dict(proposal or {})
        target = dict(payload.get("target") or {})
        req_id = payload.get("req_id")
        result = ReconfigAction(
            baseline=str(payload.get("baseline", "Greedy-Migrate")),
            req_id=int(req_id) if req_id is not None else None,
            action_type=str(payload.get("action_type", "noop")),
            target=target or None,
            reason=str(payload.get("reason", "")),
            estimated_gain=float(payload.get("estimated_gain", 0.0) or 0.0),
            applied=True,
            success=False,
        )
        if result.req_id is None or result.action_type != "migrate_vnf" or not target:
            result.reason = "invalid migration proposal"
            result.status_code = "INVALID_PROPOSAL"
            return result
        expected = target.get("state_fingerprint")
        current = self.request_state_fingerprint(result.req_id)
        if expected and expected != current:
            result.reason = "migration proposal state is stale"
            result.status_code = "STALE_PLAN"
            return result
        valid, status, reason = self._validate_migration_candidate(result.req_id, target)
        if not valid:
            result.reason = reason
            result.status_code = status
            return result
        success, status, reason = self._apply_vnf_migration(result.req_id, target)
        result.success = success
        result.status_code = status
        result.reason = reason
        return result

    def plan_greedy_reroute(self, req_id: Optional[int] = None) -> ReconfigAction:
        action, _ = self.plan_greedy_reroute_with_diagnostics(req_id=req_id)
        return action

    def plan_greedy_reroute_with_diagnostics(
        self, req_id: Optional[int] = None
    ) -> Tuple[ReconfigAction, Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "tree_edges": 0,
            "hot_edges": 0,
            "anchors": 0,
            "candidate_paths": 0,
            "valid_candidates": 0,
            "nonpositive_gain": 0,
            "no_path_anchors": 0,
            "validation_rejections": {},
        }
        rejection_counts: Counter[str] = Counter()
        rec = self._get_record_for_action(req_id)
        if rec is None:
            diagnostics["outcome"] = "no_active_sft"
            return ReconfigAction("Greedy-Reroute", None, "noop", reason="no active SFT"), diagnostics
        diagnostics["tree_edges"] = len(rec.tree_edges)
        state_fingerprint = self.request_state_fingerprint(rec.req_id)
        diagnostics["state_fingerprint"] = state_fingerprint
        link_hot = {h.key: h.utilization for h in self.link_hotspots()}
        ranked_edges = sorted(rec.tree_edges, key=lambda e: link_hot.get(e, 0.0), reverse=True)
        for edge in ranked_edges:
            if link_hot.get(edge, 0.0) <= 0.0:
                continue
            diagnostics["hot_edges"] += 1
            remaining_edges = set(rec.tree_edges) - {edge}
            best_action = None
            child = int(edge[1])
            subtree_nodes = self._directed_descendants(rec.tree_edges, child)
            main_tree_nodes = {
                int(node) for candidate_edge in remaining_edges for node in candidate_edge
                if int(node) not in subtree_nodes
            }
            main_tree_nodes.update(
                node for node in (int(edge[0]), int(rec.source))
                if node not in subtree_nodes
            )
            anchors = sorted(main_tree_nodes, key=lambda node: (node != int(edge[0]), node))
            all_tree_nodes = {int(node) for candidate_edge in rec.tree_edges for node in candidate_edge}
            for anchor in anchors:
                diagnostics["anchors"] += 1
                forbidden_internal = all_tree_nodes - {int(anchor), child}
                paths = self._find_alternate_paths(
                    anchor, child, rec.bw,
                    avoid_edges={edge},
                    forbidden_internal_nodes=forbidden_internal,
                    existing_edges=remaining_edges,
                )
                diagnostics["candidate_paths"] += len(paths)
                if not paths:
                    diagnostics["no_path_anchors"] += 1
                for path in paths:
                    old_hops = 1
                    new_hops = max(0, len(path) - 1)
                    valid, reason, safety = self._validate_reroute_candidate(rec, edge, path, rec.bw)
                    if not valid:
                        rejection_counts[reason] += 1
                        logger.debug("reject reroute req=%s edge=%s: %s", rec.req_id, edge, reason)
                        continue
                    estimated_gain = (
                        safety["old_utilization"] - safety["max_new_utilization"]
                        - 0.02 * max(0, new_hops - old_hops)
                        - 0.01 * max(0.0, safety["new_path_delay_ms"] - safety["old_edge_delay_ms"])
                    )
                    if estimated_gain <= 0.0:
                        diagnostics["nonpositive_gain"] += 1
                        continue
                    diagnostics["valid_candidates"] += 1
                    candidate = ReconfigAction(
                        baseline="Greedy-Reroute",
                        req_id=rec.req_id,
                        action_type="reroute_edge",
                        target={
                            "candidate_id": self._candidate_id(
                                state_fingerprint, rec.req_id, edge, path
                            ),
                            "state_fingerprint": state_fingerprint,
                            "old_edge": edge,
                            "new_path": path,
                            "bw": rec.bw,
                            "safety": safety,
                        },
                        reason="reattach a hot subtree through a safe lower-pressure path",
                        estimated_gain=float(estimated_gain),
                        status_code="ACTION_AVAILABLE",
                    )
                    if best_action is None or candidate.estimated_gain > best_action.estimated_gain:
                        best_action = candidate
            if best_action is not None:
                diagnostics["validation_rejections"] = dict(rejection_counts)
                diagnostics["outcome"] = "reroute_edge"
                return best_action, diagnostics
        diagnostics["validation_rejections"] = dict(rejection_counts)
        diagnostics["outcome"] = "no_feasible_alternate_path"
        return (
            ReconfigAction("Greedy-Reroute", rec.req_id, "noop", reason="no feasible alternate path"),
            diagnostics,
        )

    @staticmethod
    def _directed_descendants(tree_edges: Iterable[Edge], root: int) -> set[int]:
        adjacency: Dict[int, List[int]] = {}
        for u, v in tree_edges:
            adjacency.setdefault(int(u), []).append(int(v))
        descendants: set[int] = set()
        stack = [int(root)]
        while stack:
            node = stack.pop()
            if node in descendants:
                continue
            descendants.add(node)
            stack.extend(adjacency.get(node, []))
        return descendants

    def apply_greedy_reroute(self, req_id: Optional[int] = None) -> ReconfigAction:
        action = self.plan_greedy_reroute(req_id)
        if action.action_type != "reroute_edge" or not action.target:
            return action
        return self.apply_reroute_proposal(action)

    def apply_reroute_proposal(self, proposal: Any) -> ReconfigAction:
        payload = proposal.to_dict() if isinstance(proposal, ReconfigAction) else dict(proposal or {})
        target = dict(payload.get("target") or {})
        req_id = payload.get("req_id")
        result = ReconfigAction(
            baseline=str(payload.get("baseline", "Greedy-Reroute")),
            req_id=int(req_id) if req_id is not None else None,
            action_type=str(payload.get("action_type", "noop")),
            target=target or None,
            reason=str(payload.get("reason", "")),
            estimated_gain=float(payload.get("estimated_gain", 0.0) or 0.0),
            applied=True,
            success=False,
            status_code=str(payload.get("status_code", "")),
        )
        if result.req_id is None or result.action_type != "reroute_edge" or not target:
            result.reason = "invalid reroute proposal"
            result.status_code = "INVALID_PROPOSAL"
            return result
        expected = target.get("state_fingerprint")
        current = self.request_state_fingerprint(result.req_id)
        if expected and expected != current:
            result.reason = "reroute proposal state is stale"
            result.status_code = "STALE_PLAN"
            return result
        record_key, rec = self._get_record_entry_for_action(result.req_id)
        if rec is None:
            result.reason = "request is not active"
            result.status_code = "NO_ACTIVE_SFT"
            return result
        try:
            old_edge = tuple(int(node) for node in target.get("old_edge", ()))
            path = [int(node) for node in target.get("new_path", [])]
            bw = float(target.get("bw", rec.bw))
        except (TypeError, ValueError):
            result.reason = "reroute proposal fields are malformed"
            result.status_code = "INVALID_PROPOSAL"
            return result
        if len(old_edge) != 2 or len(path) < 2 or bw <= 0.0:
            result.reason = "reroute proposal fields are incomplete"
            result.status_code = "INVALID_PROPOSAL"
            return result
        valid, reason, safety = self._validate_reroute_candidate(rec, old_edge, path, bw)
        if not valid:
            result.reason = reason
            result.status_code = self._validation_status_code(reason)
            return result
        target["safety"] = safety
        result.target = target
        result.success, result.status_code, result.reason = self._apply_edge_reroute(
            record_key, target
        )
        return result

    def request_state_fingerprint(self, req_id: int) -> str:
        rec = self.rm.request_table.get(req_id) or self.rm.request_table.get(str(req_id))
        if rec is None:
            return "missing"
        pool = self.rm.pool
        payload = {
            "req_id": int(rec.req_id),
            "state": str(rec.state),
            "tree_edges": sorted((int(u), int(v), float(flow)) for (u, v), flow in rec.tree_edges.items()),
            "tree_usage": sorted((int(u), int(v), int(count)) for (u, v), count in rec.tree_usage.items()),
            "placements": sorted((int(index), int(node)) for index, node in rec.placement_by_vnf.items()),
            "allocations": sorted(
                (int(item.u), int(item.v), float(item.bw)) for item in rec.edge_allocations
            ),
            "cpu_avail": [float(value) for value in pool.cpu_avail],
            "mem_avail": [float(value) for value in pool.mem_avail],
            "bw_avail": sorted(
                (int(u), int(v), float(value)) for (u, v), value in pool.bw_avail.items()
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _candidate_id(state_fingerprint: str, req_id: int, old_edge: Edge, path: List[int]) -> str:
        payload = f"{state_fingerprint}|{int(req_id)}|{tuple(old_edge)}|{tuple(path)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _validation_status_code(reason: str) -> str:
        codes = {
            "invalid edge replacement": "INVALID_REPLACEMENT",
            "candidate exceeds tree-growth limit": "TREE_GROWTH_LIMIT",
            "candidate has insufficient bandwidth": "INSUFFICIENT_BW",
            "candidate exceeds projected-utilization limit": "PROJECTED_UTIL_LIMIT",
            "candidate reduces residual network connectivity": "CONNECTIVITY_LOSS",
            "old edge bandwidth ledger is missing, duplicate, or inconsistent": "LEDGER_MISMATCH",
            "candidate does not preserve a rooted acyclic tree": "NOT_ROOTED_TREE",
            "candidate disconnects a destination or VNF placement": "CRITICAL_DISCONNECTED",
            "candidate has incomplete VNF stage placements": "VNF_STAGE_MISSING",
            "candidate violates directed VNF stage order": "VNF_STAGE_ORDER",
            "candidate places a destination before the last VNF": "DESTINATION_ORDER",
            "candidate exceeds predicted-delay safety bound": "DELAY_BOUND",
        }
        return codes.get(str(reason), "REVALIDATION_FAILED")

    def run_baseline(self, name: str, req_id: Optional[int] = None, apply: bool = False) -> ReconfigAction:
        normalized = name.strip().lower().replace("_", "-")
        if normalized in {"no-reconfig", "none", "noop"}:
            return self.plan_no_reconfig()
        if normalized in {"full-redeploy", "redeploy"}:
            action = self.plan_full_redeploy(req_id)
            action.applied = bool(apply)
            action.success = False
            return action
        if normalized in {"greedy-migrate", "migrate"}:
            return self.apply_greedy_migrate(req_id) if apply else self.plan_greedy_migrate(req_id)
        if normalized in {"greedy-reroute", "reroute"}:
            return self.apply_greedy_reroute(req_id) if apply else self.plan_greedy_reroute(req_id)
        raise ValueError(f"unknown baseline: {name}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_record_entry_for_action(self, req_id: Optional[int]):
        if req_id is not None:
            if req_id in self.rm.request_table:
                return req_id, self.rm.request_table[req_id]
            try:
                alt_key = int(req_id) if isinstance(req_id, str) else str(req_id)
            except (TypeError, ValueError):
                return None, None
            if alt_key in self.rm.request_table:
                return alt_key, self.rm.request_table[alt_key]
            return None, None
        risks = self.select_topk_risky_sfts(1)
        if not risks:
            return None, None
        return self._get_record_entry_for_action(risks[0].req_id)

    def _get_record_for_action(self, req_id: Optional[int]):
        return self._get_record_entry_for_action(req_id)[1]

    def _node_utilization(self, node: int) -> float:
        pool = self.rm.pool
        cpu = 1.0 - float(pool.get_available_cpu(node)) / max(float(pool.cpu_cap[node]), 1e-9)
        mem = 1.0 - float(pool.get_available_memory(node)) / max(float(pool.mem_cap[node]), 1e-9)
        return max(cpu, mem)

    @staticmethod
    def _avg(values: Iterable[float]) -> float:
        vals = list(values)
        return sum(vals) / len(vals) if vals else 0.0

    def _edge_delay(self, u: int, v: int, topo=None) -> float:
        if self.env is not None and hasattr(self.env, "delay_matrix"):
            try:
                delay = float(self.env.delay_matrix[int(u), int(v)])
                if delay > 0.0:
                    return delay
            except Exception:
                pass
        if topo is not None and 0 <= u < self.rm.n and 0 <= v < self.rm.n:
            try:
                return max(float(topo[u, v]), 1.0)
            except Exception:
                return 1.0
        return 1.0

    def _edge_queueing_delay(self, u: int, v: int, base_delay: float) -> float:
        try:
            cap = float(self.rm.pool.bw_cap.get((int(u), int(v)), 0.0))
            avail = float(self.rm.pool.get_available_bandwidth(int(u), int(v)))
        except Exception:
            return 0.0
        if cap <= 0.0:
            return 0.0
        utilization = min(0.999, max(0.0, 1.0 - avail / cap))
        if utilization <= 0.0:
            return 0.0
        return self.queueing_delay_weight * base_delay * utilization / max(1.0 - utilization, 1e-6)

    def _best_migration_target(
        self,
        old_node: int,
        vnf_type: int,
        cpu: float,
        mem: float,
        allowed_nodes: Optional[Iterable[int]] = None,
    ) -> Optional[int]:
        allowed = None if allowed_nodes is None else {int(node) for node in allowed_nodes}
        candidates = []
        for node in getattr(self.rm, "dc_nodes", []):
            if allowed is not None and int(node) not in allowed:
                continue
            if int(node) == int(old_node):
                continue
            if self.rm.instance_index.get((int(node), int(vnf_type))) is not None:
                continue
            if self.rm.check_node_resource(int(node), int(vnf_type), float(cpu), float(mem)):
                projected = self._projected_node_utilization(int(node), float(cpu), float(mem))
                if projected <= self.max_migration_target_utilization:
                    candidates.append((projected, int(node)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _projected_node_utilization(self, node: int, cpu_delta: float, mem_delta: float) -> float:
        pool = self.rm.pool
        cpu_cap = max(float(pool.cpu_cap[int(node)]), 1e-9)
        mem_cap = max(float(pool.mem_cap[int(node)]), 1e-9)
        cpu_avail = min(cpu_cap, float(pool.get_available_cpu(int(node))) - float(cpu_delta))
        mem_avail = min(mem_cap, float(pool.get_available_memory(int(node))) - float(mem_delta))
        return max(0.0, 1.0 - cpu_avail / cpu_cap, 1.0 - mem_avail / mem_cap)

    @staticmethod
    def _binding_vnf_index(rec, binding) -> Optional[int]:
        for raw_idx, detail in rec.placement_detail.items():
            if (
                int(detail.get("node", -1)) == int(binding.node)
                and int(detail.get("vnf_type", -1)) == int(binding.vnf_type)
                and str(detail.get("inst_id", "")) == str(binding.inst_id)
            ):
                return int(raw_idx)
        return None

    def _validate_migration_candidate(
        self, req_id: int, target: Dict[str, Any]
    ) -> Tuple[bool, str, str]:
        rec = self.rm.request_table.get(req_id)
        if rec is None or rec.state != "ACTIVE":
            return False, "NO_ACTIVE_SFT", "request is not active"
        try:
            old_node = int(target["old_node"])
            new_node = int(target["new_node"])
            vnf_type = int(target["vnf_type"])
            cpu = float(target["cpu"])
            mem = float(target["mem"])
            inst_id = str(target["inst_id"])
            vnf_idx = int(target["vnf_idx"])
        except (KeyError, TypeError, ValueError):
            return False, "INVALID_PROPOSAL", "migration proposal fields are incomplete"

        binding = None
        for candidate in rec.vnf_bindings:
            if (
                candidate.node == old_node
                and candidate.vnf_type == vnf_type
                and str(candidate.inst_id) == inst_id
            ):
                binding = candidate
                break
        if binding is None or binding.reused:
            return False, "BINDING_NOT_MIGRATABLE", "binding is missing or reuses a shared instance"
        inst = self.rm.instance_table.get(binding.inst_id)
        if inst is None or inst.state != "ACTIVE" or inst.ref_count != 1:
            return False, "INSTANCE_SHARED", "instance is missing, inactive, or shared"
        if rec.placement_by_vnf.get(vnf_idx) != old_node:
            return False, "PLACEMENT_MISMATCH", "placement index no longer points to the source node"
        tree_nodes = {int(rec.source)}
        for u, v in rec.tree_edges:
            tree_nodes.update((int(u), int(v)))
        if new_node not in tree_nodes:
            return False, "TARGET_OUTSIDE_TREE", "target is outside the current SFT tree"
        if self.rm.instance_index.get((new_node, vnf_type)) is not None:
            return False, "TARGET_INSTANCE_EXISTS", "target already hosts this VNF type"
        if not self.rm.check_node_resource(new_node, vnf_type, cpu, mem):
            return False, "INSUFFICIENT_RESOURCE", "target resources are no longer sufficient"
        projected = self._projected_node_utilization(new_node, cpu, mem)
        if projected > self.max_migration_target_utilization:
            return False, "TARGET_TOO_HOT", "target would exceed the migration utilization limit"
        return True, "VALID", "migration proposal is executable"

    def _apply_vnf_migration(
        self, req_id: int, target: Dict[str, Any]
    ) -> Tuple[bool, str, str]:
        valid, status, reason = self._validate_migration_candidate(req_id, target)
        if not valid:
            return False, status, reason
        rec = self.rm.request_table[req_id]
        old_node = int(target["old_node"])
        new_node = int(target["new_node"])
        vnf_type = int(target["vnf_type"])
        cpu = float(target["cpu"])
        mem = float(target["mem"])
        inst_id = str(target["inst_id"])
        vnf_idx = int(target["vnf_idx"])
        binding = next(item for item in rec.vnf_bindings if str(item.inst_id) == inst_id)
        inst = self.rm.instance_table[inst_id]

        pool = self.rm.pool
        snapshot = {
            "cpu_avail": copy.deepcopy(pool.cpu_avail),
            "mem_avail": copy.deepcopy(pool.mem_avail),
            "hvt_all": copy.deepcopy(self.rm.hvt_all),
            "instance_table": copy.deepcopy(self.rm.instance_table),
            "instance_index": copy.deepcopy(self.rm.instance_index),
            "vnf_bindings": copy.deepcopy(rec.vnf_bindings),
            "placement_by_vnf": copy.deepcopy(rec.placement_by_vnf),
            "placement_detail": copy.deepcopy(rec.placement_detail),
            "node_stage": copy.deepcopy(rec.node_stage),
            "migration_count": rec.migration_count,
            "last_reconfig_time": rec.last_reconfig_time,
        }

        try:
            if not self.rm.allocate_node_resource(new_node, vnf_type, cpu, mem):
                return False, "ALLOCATE_FAILED", "target allocation failed"
            self.rm.release_node_resource(old_node, vnf_type, cpu, mem)

            self.rm.instance_index.pop((old_node, vnf_type), None)
            new_inst_id = self.rm._make_inst_id(new_node, vnf_type)
            inst.inst_id = new_inst_id
            inst.node = new_node
            self.rm.instance_table.pop(inst_id, None)
            self.rm.instance_table[new_inst_id] = inst
            self.rm.instance_index[(new_node, vnf_type)] = new_inst_id

            binding.node = new_node
            binding.inst_id = new_inst_id
            rec.placement_by_vnf[vnf_idx] = new_node
            rec.placement_detail[vnf_idx]["node"] = new_node
            rec.placement_detail[vnf_idx]["inst_id"] = new_inst_id
            rec.node_stage.setdefault(new_node, rec.node_stage.get(old_node, vnf_idx))
            rec.migration_count += 1
            rec.last_reconfig_time = self._current_time()
            report = self.rm.validate_request_sft_snapshot(req_id)
            if not report.get("ok", False):
                raise RuntimeError(f"post-migration snapshot invalid: {report.get('reason')}")
            self.rm._sync_legacy_views()
            return True, "APPLIED", "migration proposal applied"
        except Exception as exc:
            pool.cpu_avail = snapshot["cpu_avail"]
            pool.mem_avail = snapshot["mem_avail"]
            self.rm.hvt_all = snapshot["hvt_all"]
            self.rm.instance_table = snapshot["instance_table"]
            self.rm.instance_index = snapshot["instance_index"]
            rec.vnf_bindings = snapshot["vnf_bindings"]
            rec.placement_by_vnf = snapshot["placement_by_vnf"]
            rec.placement_detail = snapshot["placement_detail"]
            rec.node_stage = snapshot["node_stage"]
            rec.migration_count = snapshot["migration_count"]
            rec.last_reconfig_time = snapshot["last_reconfig_time"]
            self.rm._sync_legacy_views()
            logger.warning("migration rollback req=%s: %s", req_id, exc)
            return False, "ROLLED_BACK", str(exc)

    def _find_alternate_path(self, source: int, target: int, bw: float,
                             avoid_edges: Optional[Iterable[Edge]] = None,
                             forbidden_internal_nodes: Optional[Iterable[int]] = None) -> Optional[List[int]]:
        paths = self._find_alternate_paths(
            source, target, bw, avoid_edges=avoid_edges,
            forbidden_internal_nodes=forbidden_internal_nodes,
        )
        return paths[0] if paths else None

    def _find_alternate_paths(self, source: int, target: int, bw: float,
                              avoid_edges: Optional[Iterable[Edge]] = None,
                              forbidden_internal_nodes: Optional[Iterable[int]] = None,
                              existing_edges: Optional[Iterable[Edge]] = None) -> List[List[int]]:
        avoid = set(avoid_edges or [])
        existing = set(existing_edges or [])
        forbidden = set(int(node) for node in (forbidden_internal_nodes or []))
        if source == target:
            return []
        heap = [(0, [int(source)])]
        results: List[List[int]] = []
        expansions = 0
        while heap and len(results) < max(1, self.max_reroute_candidates) and expansions < 5000:
            cost, path = heapq.heappop(heap)
            expansions += 1
            node = path[-1]
            if len(path) > self.max_path_hops + 1:
                continue
            if node == target and len(path) > 1:
                results.append(path)
                continue
            for nxt in self.rm.get_neighbors(node):
                edge = (int(node), int(nxt))
                if edge in avoid:
                    continue
                if nxt in path:
                    continue
                if int(nxt) in forbidden and int(nxt) != int(target):
                    continue
                extra_bw = 0.0 if edge in existing else bw
                if self.rm.pool.get_available_bandwidth(edge[0], edge[1]) < extra_bw - 1e-5:
                    continue
                projected_util = self._edge_utilization(edge[0], edge[1], extra_bw=extra_bw)
                edge_delay = self._edge_delay(edge[0], edge[1], getattr(self.rm, "topo", None))
                new_cost = cost + 1.0 + self.path_utilization_weight * projected_util + self.path_delay_weight * edge_delay
                heapq.heappush(heap, (new_cost, path + [int(nxt)]))
        return results

    def _edge_utilization(self, u: int, v: int, extra_bw: float = 0.0) -> float:
        cap = float(self.rm.pool.bw_cap.get((int(u), int(v)), 0.0))
        if cap <= 0.0:
            return 1.0
        avail = float(self.rm.pool.get_available_bandwidth(int(u), int(v)))
        return min(1.0, max(0.0, 1.0 - (avail - float(extra_bw)) / cap))

    def _projected_edge_delay(self, u: int, v: int, extra_bw: float = 0.0) -> float:
        base = self._edge_delay(u, v, getattr(self.rm, "topo", None))
        utilization = min(0.999, self._edge_utilization(u, v, extra_bw=extra_bw))
        queueing = self.queueing_delay_weight * base * utilization / max(1.0 - utilization, 1e-6)
        return float(base + queueing)

    def _validate_reroute_candidate(self, rec, old_edge: Edge, path: List[int], bw: float):
        try:
            if len(old_edge) != 2 or len(path) < 2 or float(bw) <= 0.0:
                return False, "invalid edge replacement", {}
            old_edge = (int(old_edge[0]), int(old_edge[1]))
            new_edges = [(int(path[i]), int(path[i + 1])) for i in range(len(path) - 1)]
        except (TypeError, ValueError, IndexError):
            return False, "invalid edge replacement", {}
        if old_edge not in rec.tree_edges or len(path) < 2 or old_edge in new_edges:
            return False, "invalid edge replacement", {}
        remaining_edges = set(rec.tree_edges) - {old_edge}
        added_edges = set(new_edges) - remaining_edges
        proposed_edges = remaining_edges | set(new_edges)
        growth = len(proposed_edges) - len(rec.tree_edges)
        if growth > self.max_tree_edge_growth:
            return False, "candidate exceeds tree-growth limit", {}
        for u, v in added_edges:
            if self.rm.pool.get_available_bandwidth(u, v) < bw - 1e-5:
                return False, "candidate has insufficient bandwidth", {}
            if self._edge_utilization(u, v, extra_bw=bw) > self.max_projected_link_utilization:
                return False, "candidate exceeds projected-utilization limit", {}
        connectivity_before = self._residual_connectivity_score()
        connectivity_adjustments = {old_edge: float(bw)}
        for edge in added_edges:
            connectivity_adjustments[edge] = -float(bw)
        connectivity_after = self._residual_connectivity_score(
            adjustments=connectivity_adjustments
        )
        if (
            self.min_residual_connectivity_ratio > 0.0
            and connectivity_after + 1e-9
            < connectivity_before * self.min_residual_connectivity_ratio
        ):
            return False, "candidate reduces residual network connectivity", {}
        old_allocations = [ea for ea in rec.edge_allocations
                           if int(ea.u) == old_edge[0] and int(ea.v) == old_edge[1]]
        if len(old_allocations) != 1 or abs(float(old_allocations[0].bw) - float(bw)) > 1e-5:
            return False, "old edge bandwidth ledger is missing, duplicate, or inconsistent", {}
        if not self._is_directed_tree(proposed_edges, int(rec.source)):
            return False, "candidate does not preserve a rooted acyclic tree", {}
        proposed_nodes = {node for edge in proposed_edges for node in edge}
        critical_nodes = set(int(node) for node in getattr(rec, "connected_dests", set()))
        critical_nodes.update(int(node) for node in rec.placement_by_vnf.values())
        if not critical_nodes.issubset(proposed_nodes):
            return False, "candidate disconnects a destination or VNF placement", {}
        ordered, order_reason, order_details = self._validate_ordered_sfc_edges(
            rec, proposed_edges
        )
        if not ordered:
            return False, order_reason, order_details
        old_delay = self._projected_edge_delay(*old_edge, extra_bw=0.0)
        new_delay = sum(self._projected_edge_delay(
            u, v, extra_bw=0.0 if (u, v) in remaining_edges else bw
        ) for u, v in new_edges)
        delay_limit = old_delay * self.max_reroute_delay_ratio + self.max_reroute_delay_increase_ms
        if new_delay > delay_limit:
            return False, "candidate exceeds predicted-delay safety bound", {}
        safety = {
            "old_utilization": self._edge_utilization(*old_edge),
            "max_new_utilization": max(self._edge_utilization(
                u, v, extra_bw=0.0 if (u, v) in remaining_edges else bw
            ) for u, v in new_edges),
            "old_edge_delay_ms": float(old_delay),
            "new_path_delay_ms": float(new_delay),
            "extra_edges": int(growth),
            "connectivity_before": float(connectivity_before),
            "connectivity_after": float(connectivity_after),
        }
        return True, "ok", safety

    def _residual_connectivity_score(
        self, adjustments: Optional[Dict[Edge, float]] = None
    ) -> float:
        """Mean all-pairs widest-path residual bandwidth under a projected update."""
        n = int(self.rm.n)
        projected = [[0.0 for _ in range(n)] for _ in range(n)]
        delta = adjustments or {}
        for raw_edge in self.rm.pool.iter_bandwidth_keys():
            u, v = int(raw_edge[0]), int(raw_edge[1])
            cap = float(self.rm.pool.bw_cap.get((u, v), 0.0))
            avail = float(self.rm.pool.get_available_bandwidth(u, v))
            avail += float(delta.get((u, v), 0.0))
            projected[u][v] = min(cap, max(0.0, avail))
        for k in range(n):
            for i in range(n):
                via_k = projected[i][k]
                if via_k <= 0.0:
                    continue
                for j in range(n):
                    candidate = min(via_k, projected[k][j])
                    if candidate > projected[i][j]:
                        projected[i][j] = candidate
        values = [projected[i][j] for i in range(n) for j in range(n) if i != j]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _is_directed_tree(edges: Iterable[Edge], root: int) -> bool:
        edge_set = {(int(u), int(v)) for u, v in edges}
        if not edge_set or any(u == v for u, v in edge_set):
            return False
        nodes = {node for edge in edge_set for node in edge}
        if root not in nodes or len(edge_set) != len(nodes) - 1:
            return False
        indegree = {node: 0 for node in nodes}
        adjacency = {node: [] for node in nodes}
        for u, v in edge_set:
            indegree[v] += 1
            adjacency[u].append(v)
        if indegree[root] != 0 or any(indegree[node] != 1 for node in nodes if node != root):
            return False
        visited = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node in visited:
                return False
            visited.add(node)
            stack.extend(adjacency[node])
        return visited == nodes

    @staticmethod
    def _directed_reachable(edges: Iterable[Edge], source: int, target: int) -> bool:
        source, target = int(source), int(target)
        if source == target:
            return True
        adjacency: Dict[int, List[int]] = {}
        for u, v in edges:
            adjacency.setdefault(int(u), []).append(int(v))
        visited = {source}
        stack = [source]
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, []):
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False

    def _validate_ordered_sfc_edges(
        self, rec, proposed_edges: Iterable[Edge]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate service order on the proposed directed bandwidth ledger."""
        edge_set = {(int(u), int(v)) for u, v in proposed_edges}
        expected_stages = set(range(len(rec.vnfs)))
        actual_stages = {int(stage) for stage in rec.placement_by_vnf}
        if actual_stages != expected_stages:
            return False, "candidate has incomplete VNF stage placements", {
                "expected_stages": sorted(expected_stages),
                "actual_stages": sorted(actual_stages),
            }

        placements = [
            int(rec.placement_by_vnf[stage]) for stage in range(len(rec.vnfs))
        ]
        chain_nodes = [int(rec.source), *placements]
        bad_segments = [
            (source, target)
            for source, target in zip(chain_nodes, chain_nodes[1:])
            if not self._directed_reachable(edge_set, source, target)
        ]
        if bad_segments:
            return False, "candidate violates directed VNF stage order", {
                "unreachable_chain_segments": bad_segments,
            }

        branch_root = placements[-1] if placements else int(rec.source)
        unreachable_dests = sorted(
            int(destination)
            for destination in rec.connected_dests
            if not self._directed_reachable(
                edge_set, branch_root, int(destination)
            )
        )
        if unreachable_dests:
            return False, "candidate places a destination before the last VNF", {
                "branch_root": branch_root,
                "unreachable_destinations": unreachable_dests,
            }
        return True, "ok", {
            "chain_nodes": chain_nodes,
            "branch_root": branch_root,
        }

    def _sync_reroute_legacy_tree(self, rec) -> None:
        """Keep a live request's compatibility tree aligned with RequestRecord."""
        owners = [self.rm]
        if self.env is not None and self.env is not self.rm:
            owners.append(self.env)
        tree_nodes = {int(rec.source)}
        for u, v in rec.tree_edges:
            tree_nodes.update((int(u), int(v)))
        for owner in owners:
            current_request = getattr(owner, "current_request", None)
            if not isinstance(current_request, dict):
                continue
            try:
                current_req_id = int(current_request.get("id"))
            except (TypeError, ValueError):
                continue
            if current_req_id != int(rec.req_id):
                continue
            current_tree = getattr(owner, "current_tree", None)
            if not isinstance(current_tree, dict):
                continue
            current_tree["tree"] = copy.deepcopy(rec.tree_edges)
            current_tree["tree_usage"] = copy.deepcopy(rec.tree_usage)
            current_tree["node_stage"] = copy.deepcopy(rec.node_stage)
            current_tree["connected_dests"] = set(rec.connected_dests)
            if hasattr(owner, "nodes_on_tree"):
                owner.nodes_on_tree = set(tree_nodes)

    def _apply_edge_reroute(
        self, req_id: Any, target: Dict[str, Any]
    ) -> Tuple[bool, str, str]:
        with self._reroute_lock:
            record_key, rec = self._get_record_entry_for_action(req_id)
            if rec is None:
                return False, "NO_ACTIVE_SFT", "request is not active"
            try:
                old_edge = tuple(int(value) for value in target["old_edge"])
                path = [int(x) for x in target["new_path"]]
                bw = float(target.get("bw", rec.bw))
            except (KeyError, TypeError, ValueError):
                return False, "INVALID_PROPOSAL", "reroute proposal fields are malformed"
            if len(old_edge) != 2 or len(path) < 2 or bw <= 0.0:
                return False, "INVALID_PROPOSAL", "reroute proposal fields are incomplete"
            new_edges = [
                (path[index], path[index + 1])
                for index in range(len(path) - 1)
            ]
            valid, reason, _ = self._validate_reroute_candidate(
                rec, old_edge, path, bw
            )
            if not valid:
                logger.warning("reject unsafe reroute req=%s: %s", req_id, reason)
                return False, self._validation_status_code(reason), reason

            remaining_edges = set(rec.tree_edges) - {old_edge}
            added_edges = list(dict.fromkeys(
                edge for edge in new_edges if edge not in remaining_edges
            ))
            pool = self.rm.pool
            legacy_owners = [self.rm]
            if self.env is not None and self.env is not self.rm:
                legacy_owners.append(self.env)
            legacy_snapshot = [
                {
                    "owner": owner,
                    "current_tree": copy.deepcopy(
                        getattr(owner, "current_tree", None)
                    ),
                    "nodes_on_tree": copy.deepcopy(
                        getattr(owner, "nodes_on_tree", None)
                    ),
                }
                for owner in legacy_owners
            ]
            debug_counters = {}
            for name in ("_dbg_alloc_bw", "_dbg_rel_bw"):
                debug_counters[name] = (
                    hasattr(pool, name), copy.deepcopy(getattr(pool, name, None))
                )
            snapshot = {
                "bw_avail": copy.deepcopy(pool.bw_avail),
                "bw_reserved": copy.deepcopy(pool.bw_reserved),
                "bw_version": getattr(pool, "_bw_version", 0),
                "bw_query_cache_version": getattr(
                    pool, "_bw_query_cache_version", -1
                ),
                "bw_query_cache": copy.deepcopy(
                    getattr(pool, "_bw_query_cache", {})
                ),
                "edge_allocations": copy.deepcopy(rec.edge_allocations),
                "tree_edges": copy.deepcopy(rec.tree_edges),
                "tree_usage": copy.deepcopy(rec.tree_usage),
                "node_stage": copy.deepcopy(rec.node_stage),
                "snapshot_time": rec.snapshot_time,
                "last_reconfig_time": rec.last_reconfig_time,
                "migration_count": rec.migration_count,
                "reconfig_count": rec.reconfig_count,
                "state": rec.state,
                "shared_vnf_instances": copy.deepcopy(
                    self.rm.shared_vnf_instances
                ),
            }

            failure_status = "ROLLED_BACK"
            try:
                for u, v in added_edges:
                    if not self.rm.commit_edge_bandwidth(record_key, u, v, bw):
                        failure_status = "ALLOCATE_FAILED"
                        raise RuntimeError(
                            f"failed to allocate reroute edge {(u, v)}"
                        )

                old_flow = rec.tree_edges.get(old_edge, 1.0)
                old_usage = rec.tree_usage.get(old_edge, 1)
                self.rm.release_bandwidth(old_edge[0], old_edge[1], bw)
                rec.edge_allocations = [
                    allocation
                    for allocation in rec.edge_allocations
                    if not (
                        int(allocation.u) == old_edge[0]
                        and int(allocation.v) == old_edge[1]
                    )
                ]
                rec.tree_edges.pop(old_edge, None)
                rec.tree_usage.pop(old_edge, None)
                for edge in added_edges:
                    rec.tree_edges[edge] = old_flow
                    rec.tree_usage[edge] = old_usage
                inherited_stage = rec.node_stage.get(old_edge[0], 0)
                for node in path[1:-1]:
                    rec.node_stage.setdefault(node, inherited_stage)
                rec.reconfig_count += 1
                rec.last_reconfig_time = self._current_time()

                report = self.rm.validate_request_sft_snapshot(record_key)
                if not report.get("ok", False):
                    raise RuntimeError(
                        "post-reroute snapshot invalid: "
                        f"{report.get('reason', 'unknown')}"
                    )
                self._sync_reroute_legacy_tree(rec)
                self.rm._sync_legacy_views()
                return True, "APPLIED", "reroute proposal applied"
            except Exception as exc:
                pool.bw_avail = snapshot["bw_avail"]
                pool.bw_reserved = snapshot["bw_reserved"]
                pool._bw_version = snapshot["bw_version"]
                pool._bw_query_cache_version = snapshot[
                    "bw_query_cache_version"
                ]
                pool._bw_query_cache = snapshot["bw_query_cache"]
                for name, (existed, value) in debug_counters.items():
                    if existed:
                        setattr(pool, name, value)
                    elif hasattr(pool, name):
                        delattr(pool, name)
                rec.edge_allocations = snapshot["edge_allocations"]
                rec.tree_edges = snapshot["tree_edges"]
                rec.tree_usage = snapshot["tree_usage"]
                rec.node_stage = snapshot["node_stage"]
                rec.snapshot_time = snapshot["snapshot_time"]
                rec.last_reconfig_time = snapshot["last_reconfig_time"]
                rec.migration_count = snapshot["migration_count"]
                rec.reconfig_count = snapshot["reconfig_count"]
                rec.state = snapshot["state"]
                self.rm.shared_vnf_instances = snapshot[
                    "shared_vnf_instances"
                ]
                for item in legacy_snapshot:
                    owner = item["owner"]
                    if item["current_tree"] is not None:
                        owner.current_tree = item["current_tree"]
                    if item["nodes_on_tree"] is not None:
                        owner.nodes_on_tree = item["nodes_on_tree"]
                logger.warning("reroute rollback req=%s: %s", req_id, exc)
                return False, failure_status, str(exc)

    def _current_time(self) -> Optional[float]:
        if self.env is not None:
            return getattr(self.env, "time_step", None)
        return None


def action_to_dict(action: ReconfigAction) -> Dict[str, Any]:
    return action.to_dict()
