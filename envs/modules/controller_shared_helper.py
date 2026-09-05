"""
envs/modules/controller_shared_helper.py
================================================
Shared env-bound helper for TA-HRL controllers/coordinator.

This helper now centralizes the reusable helper logic that was previously
scattered across HRL_Coordinator / HighLevelController / LowLevelController.

Covered areas:
- tree/request view helpers
- snapshot / reuse / active-instance counting
- agent subgoal-state cleanup
- lazy topology validation / hop-distance helpers (for high-level side)
- low-level candidate feature building and top-k pruning helpers
- high-level candidate feature building and top-k / BW / anchor helpers
"""

from __future__ import annotations

from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import logging
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class ControllerSharedHelper:
    """Shared env-bound helper used by controllers and coordinator."""

    def __init__(self, env: Any):
        self.env = env
        self._lazy_hop_cache: Dict[Tuple[int, int, int], int] = {}
        self._lazy_hop_cache_reqid: Optional[int] = None
        self._topology_graph: Optional[nx.Graph] = None
        self._all_pairs_hops: Optional[Dict[int, Dict[int, int]]] = None
        self._k_path_cache: Dict[Tuple[int, int, int], List[List[int]]] = {}

    # ------------------------------------------------------------------
    # Generic topology helpers
    # ------------------------------------------------------------------
    def is_valid_node(self, node: Any) -> bool:
        try:
            node = int(node)
            if node < 0 or node >= int(getattr(self.env, 'n', 0)):
                return False
            rm = getattr(self.env, 'resource_mgr', None)
            if rm is not None and hasattr(rm, 'get_neighbors'):
                try:
                    rm.get_neighbors(node)
                    return True
                except Exception:
                    return False
            return True
        except Exception:
            return False

    def _build_topology_graph(self) -> nx.Graph:
        G = nx.Graph()
        rm = getattr(self.env, 'resource_mgr', None)
        n = int(getattr(self.env, 'n', 0))
        if rm is not None and hasattr(rm, 'get_neighbors'):
            for u in range(n):
                try:
                    for v in rm.get_neighbors(u):
                        if self.is_valid_node(v):
                            G.add_edge(u, v)
                except Exception:
                    continue
        elif hasattr(self.env, 'topology'):
            for u in range(n):
                for v in range(n):
                    try:
                        if u != v and self.env.topology[u][v] > 0:
                            G.add_edge(u, v)
                    except Exception:
                        continue
        return G

    def get_topology_graph(self, refresh: bool = False) -> nx.Graph:
        if refresh or self._topology_graph is None:
            self._topology_graph = self._build_topology_graph()
            if refresh:
                self._k_path_cache.clear()
                self._lazy_hop_cache.clear()
            self._all_pairs_hops = {
                int(source): {int(target): int(distance) for target, distance in rows.items()}
                for source, rows in nx.all_pairs_shortest_path_length(self._topology_graph)
            }
        return self._topology_graph

    def get_hop_distance_lazy(self, node1: int, node2: int) -> int:
        if node1 == node2:
            return 0
        if not self.is_valid_node(node1) or not self.is_valid_node(node2):
            return 9999

        req_id = id(self.env.current_request) if getattr(self.env, 'current_request', None) else 0
        if self._lazy_hop_cache_reqid != req_id:
            self._lazy_hop_cache.clear()
            self._lazy_hop_cache_reqid = req_id

        key = (req_id, int(node1), int(node2))
        rev = (req_id, int(node2), int(node1))
        if key in self._lazy_hop_cache:
            return self._lazy_hop_cache[key]
        if rev in self._lazy_hop_cache:
            return self._lazy_hop_cache[rev]

        try:
            G = self.get_topology_graph()
            if G.has_node(node1) and G.has_node(node2):
                if self._all_pairs_hops is None:
                    self._all_pairs_hops = {
                        int(source): {int(target): int(distance) for target, distance in rows.items()}
                        for source, rows in nx.all_pairs_shortest_path_length(G)
                    }
                dist = int(self._all_pairs_hops.get(int(node1), {}).get(int(node2), 9999))
            else:
                dist = 9999
        except Exception:
            dist = 9999
        self._lazy_hop_cache[key] = dist
        return dist

    def prewarm_k_paths(self, k: int = 4) -> None:
        """Initialize the static topology used by the lazy K-path cache.

        Path enumeration is intentionally deferred until a concrete
        ``(source, target, k)`` tuple is requested.  Enumerating every node
        pair caused a large first-request latency on 50- and 100-node
        topologies.  Feasibility is still checked against live bandwidth in
        :meth:`get_k_path_first_hops`.
        """
        del k
        self.get_topology_graph()

    def _get_k_paths(self, source: int, target: int, k: int) -> List[List[int]]:
        source, target = int(source), int(target)
        k = max(1, int(k))
        cache_key = (source, target, k)
        cached = self._k_path_cache.get(cache_key)
        if cached is not None:
            return cached

        graph = self.get_topology_graph()
        if source == target:
            paths = [[source]]
        else:
            try:
                paths = [
                    [int(value) for value in path]
                    for path in islice(
                        nx.shortest_simple_paths(graph, source, target), k
                    )
                ]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths = []
        self._k_path_cache[cache_key] = paths
        return paths

    def get_k_path_first_hops(
        self,
        current: Optional[int],
        target: Optional[int],
        bw_req: float,
        tree_edges: Optional[Set[Tuple[int, int]]] = None,
        k: int = 4,
    ) -> Set[int]:
        """Return feasible first hops from prewarmed topology paths.

        This is a candidate filter only.  The low-level policy still scores
        and selects the returned next-hop actions.
        """
        if current is None or target is None:
            return set()
        current, target = int(current), int(target)
        if current == target:
            return set()
        self.prewarm_k_paths(k)
        rm = getattr(self.env, 'resource_mgr', None)
        pool = getattr(rm, 'pool', None)
        tree_edges = tree_edges or set()
        first_hops: Set[int] = set()
        for path in self._get_k_paths(current, target, k):
            if len(path) < 2:
                continue
            feasible = True
            for u, v in zip(path, path[1:]):
                if (u, v) in tree_edges:
                    continue
                try:
                    if pool is None or float(pool.get_available_bandwidth(u, v)) + 1e-9 < float(bw_req):
                        feasible = False
                        break
                except Exception:
                    feasible = False
                    break
            if feasible:
                first_hops.add(int(path[1]))
        return first_hops

    # ------------------------------------------------------------------
    # Tree / request-view helpers
    # ------------------------------------------------------------------
    def get_positive_tree_edge_set(self) -> Set[Tuple[int, int]]:
        if not hasattr(self.env, 'current_tree') or not self.env.current_tree:
            return set()

        tree = self.env.current_tree.get('tree', {})
        pos_edges: Set[Tuple[int, int]] = set()
        for edge_key, flow in tree.items():
            try:
                u, v = int(edge_key[0]), int(edge_key[1])
                if float(flow) > 0.0:
                    pos_edges.add((u, v))
            except Exception:
                continue
        return pos_edges

    def get_connected_dests_view(self) -> Set[int]:
        if not hasattr(self.env, 'current_request') or not self.env.current_request:
            return set()

        
        
        
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            tree_cd = self.env.current_tree.get('connected_dests', None)
            if tree_cd is not None:
                try:
                    return set(int(x) for x in tree_cd)
                except Exception:
                    return set(tree_cd)

        
        req_id = self.env.current_request.get('id')
        if req_id is not None:
            rm = getattr(self.env, 'resource_mgr', None)
            if rm is not None:
                req_table = getattr(rm, 'request_table', {})
                req_rec = req_table.get(req_id)
                if req_rec is not None:
                    try:
                        return set(int(x) for x in req_rec.connected_dests)
                    except Exception:
                        try:
                            return set(req_rec.connected_dests)
                        except Exception:
                            pass

        return set()

    def is_tree_edge(self, u: int, v: int) -> bool:
        """检查有向边 (u→v) 是否已在正树中（flow>0）。"""
        tree = (self.env.current_tree or {}).get('tree', {})
        return tree.get((u, v), 0.0) > 0.0

    def is_reverse_tree_edge(self, u: int, v: int) -> bool:
        """True when the opposite directed edge (v,u) is already in the tree."""
        tree = (self.env.current_tree or {}).get('tree', {})
        return tree.get((v, u), 0.0) > 0.0

    def get_reuse_edge_count(self) -> int:
        if not hasattr(self.env, 'current_tree') or not self.env.current_tree:
            return 0
        try:
            return sum(
                1
                for _, usage in self.env.current_tree.get('tree_usage', {}).items()
                if usage > 1
            )
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Snapshot / metrics helpers
    # ------------------------------------------------------------------
    def count_active_vnf_instances(self) -> int:
        try:
            rm = getattr(self.env, 'resource_mgr', None)
            if rm is not None and hasattr(rm, 'instance_table') and rm.instance_table:
                return sum(
                    1
                    for inst in rm.instance_table.values()
                    if getattr(inst, 'state', None) == 'ACTIVE'
                )
        except Exception:
            pass

        try:
            if hasattr(self.env, 'current_tree') and self.env.current_tree:
                return len(self.env.current_tree.get('placement', {}))
        except Exception:
            pass

        return 0

    def take_tree_snapshot(self) -> Dict[str, int]:
        snap = {
            'connected_dests': 0,
            'vnf_done': 0,
            'tree_edges': 0,
            'vnf_instances': 0,
            'reused_edges': 0,
        }
        try:
            if hasattr(self.env, 'current_tree') and self.env.current_tree:
                snap['connected_dests'] = len(self.get_connected_dests_view())
                tree_dict = self.env.current_tree.get('tree', {})
                snap['tree_edges'] = sum(1 for flow in tree_dict.values() if flow > 0.0)
                snap['vnf_instances'] = self.count_active_vnf_instances()
                snap['reused_edges'] = self.get_reuse_edge_count()
            snap['vnf_done'] = int(getattr(self.env, 'next_vnf_idx', 0))
        except Exception:
            pass
        return snap

    # ------------------------------------------------------------------
    # Agent state cleanup helper
    # ------------------------------------------------------------------
    def clear_subgoal_state(self, high_agent: Optional[Any] = None, low_agent: Optional[Any] = None) -> None:
        if high_agent is not None:
            try:
                if hasattr(high_agent, 'current_subgoal'):
                    high_agent.current_subgoal = None
                if hasattr(high_agent, 'current_goal_emb'):
                    high_agent.current_goal_emb = None
                if hasattr(high_agent, 'current_subgoal_emb'):
                    high_agent.current_subgoal_emb = None
            except Exception:
                pass

        if low_agent is not None:
            try:
                if hasattr(low_agent, 'current_subgoal'):
                    low_agent.current_subgoal = None
                if hasattr(low_agent, 'current_goal_emb'):
                    low_agent.current_goal_emb = None
                if hasattr(low_agent, 'current_subgoal_emb'):
                    low_agent.current_subgoal_emb = None
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Low-level navigation helper methods
    # ------------------------------------------------------------------
    def is_hard_feasible_neighbor(
        self,
        current: int,
        nbr: int,
        bw_req: float,
        tree_edges: Optional[Set[Tuple[int, int]]] = None,
    ) -> bool:
        try:
            if nbr is None or current is None:
                return False
            if nbr < 0 or nbr >= int(getattr(self.env, 'n', 0)):
                return False

            rm = getattr(self.env, 'resource_mgr', None)
            if rm is None:
                return False

            neighbors = rm.get_neighbors(current)
            if nbr not in neighbors:
                return False

            edge_key = (int(current), int(nbr))
            if tree_edges and edge_key in tree_edges:
                return True

            avail_bw = rm.pool.get_available_bandwidth(current, nbr)
            return float(avail_bw) >= float(bw_req)
        except Exception:
            return False

    def build_soft_nav_features(
        self,
        current: int,
        nbr: int,
        target: Optional[int],
        bw_req: float,
        tree_edges: Set[Tuple[int, int]],
        nodes_on_tree: Set[int],
        tabu_set: Set[int],
        undone_dests: Sequence[int],
        hop_distance_fn: Callable[[int, int], int],
        max_hops: int,
    ) -> List[float]:
        rm = getattr(self.env, 'resource_mgr', None)
        edge_key = (int(current), int(nbr))

        d_current = hop_distance_fn(current, target) if target is not None else 0
        d_nbr = hop_distance_fn(nbr, target) if target is not None else 0
        if target is not None and d_current < 9999 and d_nbr < 9999:
            delta_hop = float(d_current - d_nbr) / max(max_hops, 1)
        else:
            delta_hop = 0.0

        if edge_key in tree_edges:
            is_tree_edge = 1.0
        elif (int(nbr), int(current)) in tree_edges:
            # Reverse direction of an existing tree edge. It is not free reuse,
            # but it is a useful tree corridor for the scorer.
            is_tree_edge = 0.5
        else:
            is_tree_edge = 0.0

        avail_bw_ratio = 0.0
        try:
            if rm is not None:
                avail_bw = rm.pool.get_available_bandwidth(current, nbr)
                denom = max(float(bw_req) * 2.0, 1.0)
                avail_bw_ratio = float(min(avail_bw / denom, 1.0))
        except Exception:
            avail_bw_ratio = 0.0

        is_on_tree = 1.0 if nbr in nodes_on_tree else 0.0

        if undone_dests:
            hops = [hop_distance_fn(nbr, int(d)) for d in undone_dests]
            valid = [h for h in hops if h < 9999]
            avg_hop_to_undone = (sum(valid) / len(valid) / max(max_hops, 1)) if valid else 1.0
        else:
            avg_hop_to_undone = 0.0

        tabu_penalty = 1.0 if nbr in tabu_set else 0.0

        return [
            float(delta_hop),
            float(is_tree_edge),
            float(avail_bw_ratio),
            float(is_on_tree),
            float(avg_hop_to_undone),
            float(tabu_penalty),
        ]

    def get_low_level_candidates(
        self,
        hop_distance_fn: Callable[[int, int], int],
        action_mask_fn: Callable[..., np.ndarray],
        mutate_env: bool = False,
    ) -> Dict[str, Any]:
        mask = action_mask_fn(mutate_env=mutate_env)
        candidate_indices = [int(i) for i in np.where(mask > 0)[0]]
        raw_candidate_count = len(candidate_indices)

        current = getattr(self.env, 'current_node_location', None)
        phase = getattr(self.env, 'current_phase', None)
        target = None
        if phase == 'vnf_deployment':
            target = getattr(self.env, 'current_deployment_target', None)
        elif phase == 'destination_connection':
            target = getattr(self.env, 'current_target_node', None)

        bw_req = 0.0
        if getattr(self.env, 'current_request', None):
            bw_req = self.env.current_request.get('bw_origin', 0.0)

        tree_edges = self.get_positive_tree_edge_set()
        nodes_on_tree = set(getattr(self.env, 'nodes_on_tree', set()))
        tabu_set = set(getattr(self.env, 'current_path_trace', []))

        # Optional topology-only K-path pruning.  It narrows the low-level
        # action space but never selects an action itself: low_policy still
        # scores candidate_indices below.  Keep the original mask as a safe
        # fallback when all cached paths are stale/infeasible.
        k_path_filter = bool(getattr(self.env, '_k_path_candidate_filter', False))
        k_path_k = int(getattr(self.env, '_k_path_candidate_k', 4))
        if k_path_filter and current is not None and target is not None:
            try:
                path_hops = self.get_k_path_first_hops(
                    current=current,
                    target=target,
                    bw_req=float(bw_req),
                    tree_edges=tree_edges,
                    k=k_path_k,
                )
                # Preserve already-built tree corridors even when they are
                # absent from the topology-only shortest-path set.
                tree_hops = {
                    int(nbr) for nbr in candidate_indices
                    if (int(current), int(nbr)) in tree_edges
                }
                allowed = path_hops | tree_hops
                if allowed:
                    filtered_indices = [
                        int(nbr) for nbr in candidate_indices if int(nbr) in allowed
                    ]
                    if filtered_indices:
                        candidate_indices = filtered_indices
            except Exception as exc:
                logger.debug("[KPath] candidate pruning skipped: %s", exc)

        if k_path_filter:
            stats = getattr(self.env, '_k_path_candidate_stats', None)
            if not isinstance(stats, dict):
                stats = {
                    'calls': 0,
                    'raw_candidates': 0,
                    'filtered_candidates': 0,
                    'reduced_calls': 0,
                    'empty_fallback_calls': 0,
                }
                self.env._k_path_candidate_stats = stats
            stats['calls'] += 1
            stats['raw_candidates'] += raw_candidate_count
            stats['filtered_candidates'] += len(candidate_indices)
            stats['reduced_calls'] += int(len(candidate_indices) < raw_candidate_count)
            stats['empty_fallback_calls'] += int(raw_candidate_count > 0 and not candidate_indices)

        all_dests = []
        if getattr(self.env, 'current_request', None):
            all_dests = [int(d) for d in self.env.current_request.get('dest', [])]
        connected_dests = self.get_connected_dests_view()
        undone_dests = [d for d in all_dests if d not in connected_dests]

        max_hops = max(int(getattr(self.env, 'n', 1)), 1)
        K = len(candidate_indices)
        features = np.zeros((K, 6), dtype=np.float32)

        for i, nbr in enumerate(candidate_indices):
            features[i] = np.asarray(
                self.build_soft_nav_features(
                    current=current,
                    nbr=nbr,
                    target=target,
                    bw_req=bw_req,
                    tree_edges=tree_edges,
                    nodes_on_tree=nodes_on_tree,
                    tabu_set=tabu_set,
                    undone_dests=undone_dests,
                    hop_distance_fn=hop_distance_fn,
                    max_hops=max_hops,
                ),
                dtype=np.float32,
            )

        return {
            'indices': candidate_indices,
            'mask': np.ones(K, dtype=np.float32),
            'current_node': current,
            'target_node': target,
            'features': features,
            'candidate_mode': 'k_path_filtered' if k_path_filter else 'neighbor_mask',
            'candidate_k': k_path_k if k_path_filter else None,
        }

    def rank_next_hops(
        self,
        current_node: int,
        candidates: Iterable[int],
        target: Optional[int],
        bw_req: float,
        tree_edges: Set[Tuple[int, int]],
        hop_distance_fn: Callable[[int, int], int],
    ) -> List[int]:
        rm = getattr(self.env, 'resource_mgr', None)
        scored = []
        phase = getattr(self.env, 'current_phase', None)
        remaining_dests = 99
        try:
            if phase == 'destination_connection' and getattr(self.env, 'current_request', None):
                all_dests = set(int(d) for d in self.env.current_request.get('dest', []))
                connected = self.get_connected_dests_view()
                remaining_dests = len(all_dests - connected)
        except Exception:
            remaining_dests = 99

        for nbr in candidates:
            nbr = int(nbr)
            score = 0.0
            edge_key = (int(current_node), nbr)
            d_before = 9999
            d_after = 9999
            if target is not None:
                try:
                    d_before = hop_distance_fn(current_node, target)
                    d_after = hop_distance_fn(nbr, target)
                except Exception:
                    d_before = 9999
                    d_after = 9999

            is_tree_edge = edge_key in tree_edges
            is_reverse_tree_corridor = (nbr, int(current_node)) in tree_edges

            if is_tree_edge:
                if phase == 'destination_connection' and remaining_dests <= 1:
                    score += 1.0 if d_after <= d_before else -3.5
                else:
                    score += 4.0
            elif is_reverse_tree_corridor:
                # Directed reverse edge is not free reuse. Give only a small
                # corridor bonus, so it does not beat a real forward tree edge.
                score += 0.4
            else:
                score -= 1.5

            try:
                if rm is not None:
                    avail_bw = rm.pool.get_available_bandwidth(current_node, nbr)
                    if bw_req > 0:
                        score += 2.0 * min(float(avail_bw) / max(1e-6, float(bw_req)), 2.0)
            except Exception:
                pass

            if target is not None:
                if d_before < 9999 and d_after < 9999:
                    hop_weight = 6.0 if (
                        phase == 'destination_connection' and remaining_dests <= 1
                    ) else 1.5
                    score += hop_weight * float(d_before - d_after)

            if phase == 'destination_connection' and remaining_dests > 1:
                try:
                    nodes_on_tree = set(getattr(self.env, 'nodes_on_tree', set()))
                    if nbr in nodes_on_tree:
                        score += 0.8
                except Exception:
                    pass

            scored.append((score, nbr))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [int(nbr) for _, nbr in scored]

    def apply_low_topk_mask(
        self,
        mask,
        current_node: int,
        target: Optional[int],
        bw_req: float,
        tree_edges: Set[Tuple[int, int]],
        hop_distance_fn: Callable[[int, int], int],
        low_topk: int = 5,
        phase: Optional[str] = None,
    ):
        candidates = [int(i) for i in np.where(mask > 0)[0]]
        if len(candidates) <= int(low_topk):
            return mask

        sparse_threshold = 6 if phase == 'destination_connection' else 4
        if len(candidates) <= sparse_threshold:
            return mask

        if target is not None:
            hop_to_tgt = hop_distance_fn(current_node, target)
            bypass_threshold = 4 if phase == 'destination_connection' else 2
            if hop_to_tgt <= bypass_threshold:
                return mask

        ranked = self.rank_next_hops(
            current_node=current_node,
            candidates=candidates,
            target=target,
            bw_req=bw_req,
            tree_edges=tree_edges,
            hop_distance_fn=hop_distance_fn,
        )
        keep = set(ranked[: int(low_topk)])
        topk_mask = np.zeros_like(mask)
        for idx in keep:
            topk_mask[idx] = 1.0
        if 0 <= int(current_node) < len(mask) and mask[current_node] > 0:
            topk_mask[current_node] = mask[current_node]
        return topk_mask

    # ------------------------------------------------------------------
    # High-level helper methods
    # ------------------------------------------------------------------
    def build_high_level_candidates(
        self,
        mask: np.ndarray,
        hop_distance_fn: Callable[[int, int], int],
    ) -> Dict[str, Any]:
        indices = [int(i) for i in np.where(mask > 0)[0]]
        K = len(indices)
        if K == 0:
            return {'indices': [], 'features': np.zeros((0, 7), dtype=np.float32)}

        rm = getattr(self.env, 'resource_mgr', None)
        if rm is None:
            return {'indices': indices, 'features': np.zeros((K, 7), dtype=np.float32)}

        n = int(getattr(self.env, 'n', 0))
        dc_nodes = set(getattr(self.env, 'dc_nodes', set()))
        nodes_on_tree = set(getattr(self.env, 'nodes_on_tree', set()))
        C_cap = max(1.0, float(getattr(rm, 'C_cap', 100.0)))
        M_cap = max(1.0, float(getattr(rm, 'M_cap', 100.0)))
        max_hops = max(n, 1)

        all_dests = []
        if getattr(self.env, 'current_request', None):
            all_dests = [int(d) for d in self.env.current_request.get('dest', [])]
        connected_dests = self.get_connected_dests_view() if getattr(self.env, 'current_tree', None) else set()
        undone_dests = [d for d in all_dests if d not in connected_dests]
        tree_edges = self.get_positive_tree_edge_set()

        G = None
        if connected_dests:
            try:
                G = self.get_topology_graph()
            except Exception:
                G = None

        features = np.zeros((K, 7), dtype=np.float32)
        for i, node in enumerate(indices):
            features[i, 0] = 1.0 if node in dc_nodes else 0.0
            try:
                features[i, 1] = min(rm.pool.get_available_cpu(node) / C_cap, 1.0)
            except Exception:
                features[i, 1] = 0.0
            try:
                features[i, 2] = min(rm.pool.get_available_memory(node) / M_cap, 1.0)
            except Exception:
                features[i, 2] = 0.0

            if nodes_on_tree:
                d_tree = min(hop_distance_fn(node, t) for t in nodes_on_tree)
                features[i, 3] = min(d_tree / max_hops, 1.0)
            else:
                features[i, 3] = 1.0

            if undone_dests:
                hops = [hop_distance_fn(node, d) for d in undone_dests]
                valid = [h for h in hops if h < 9999]
                features[i, 4] = (sum(valid) / len(valid) / max_hops) if valid else 1.0
            else:
                features[i, 4] = 0.0

            features[i, 5] = 1.0 if node in nodes_on_tree else 0.0

            if connected_dests and G is not None:
                try:
                    reuse_total, len_total = 0, 0
                    for cd in connected_dests:
                        try:
                            path = nx.shortest_path(G, node, cd)
                            for j in range(len(path) - 1):
                                ek = (path[j], path[j + 1])
                                len_total += 1
                                if ek in tree_edges:
                                    reuse_total += 1
                        except Exception:
                            pass
                    features[i, 6] = reuse_total / max(1, len_total)
                except Exception:
                    features[i, 6] = 0.0
            else:
                features[i, 6] = 0.0

        logger.debug(
            f"[HighCandidates] phase={getattr(self.env, 'current_phase', '?')} "
            f"K={K} indices={indices[:8]}{'...' if K > 8 else ''}"
        )
        return {'indices': indices, 'features': features}

    def score_high_candidates(self, valid_indices: Iterable[int], start_node: int) -> List[int]:
        """
        与 HRL_Coordinator._score_high_candidates 保持同一打分逻辑。
        destination_connection: bw_bonus(上限2.0) - hop_w * hops（动态权重）
        vnf_deployment        : cpu/mem slack - 1.0 * hops/n（线性惩罚）
        """
        rm = getattr(self.env, 'resource_mgr', None)
        if rm is None:
            return [int(i) for i in valid_indices]

        req = getattr(self.env, 'current_request', None) or {}
        bw_need = float(req.get('bw_origin', 0.0))
        phase = getattr(self.env, 'current_phase', 'idle')
        n_nodes = max(int(getattr(self.env, 'n', 1)), 1)

        _all_dests  = set(int(d) for d in req.get('dest', []))
        _done_dests = self.get_connected_dests_view()
        _undone_dests = sorted(int(d) for d in (_all_dests - _done_dests))
        _remaining  = len(_undone_dests)
        tree_edges = self.get_positive_tree_edge_set()

        # Directed BW graph for route-aware high-level ranking. Existing tree
        # edges are reusable, while new directed edges must still have BW.
        G = nx.DiGraph()
        for u in range(int(getattr(rm, 'n', getattr(self.env, 'n', 0)))):
            try:
                nbrs = rm.get_neighbors(u)
            except Exception:
                continue
            for v in nbrs:
                ek = (int(u), int(v))
                try:
                    avail = float(rm.pool.get_available_bandwidth(u, v))
                    cap = float(getattr(rm.pool, 'bw_cap', {}).get(ek, 1.0))
                    bw_ok = avail >= bw_need
                except Exception:
                    avail = 0.0
                    cap = 1.0
                    bw_ok = False

                if ek in tree_edges or bw_ok:
                    util = 1.0 - avail / max(cap, 1.0)
                    util = max(0.0, min(1.0, util))
                    if ek in tree_edges:
                        weight = 0.05 + 0.10 * util
                    elif (ek[1], ek[0]) in tree_edges:
                        weight = 0.80 + 0.60 * util
                    else:
                        weight = 1.25 + 1.25 * util
                    G.add_edge(ek[0], ek[1], bw=avail, weight=weight)

        def path_stats(src, dst):
            if src == dst:
                return {
                    'reachable': True,
                    'bottleneck': float('inf'),
                    'hops': 0,
                    'reused': 0,
                    'new_edges': 0,
                    'cost': 0.0,
                }
            try:
                path = nx.shortest_path(G, int(src), int(dst), weight='weight')
                hops = len(path) - 1
                bottleneck = min(G[path[i]][path[i + 1]].get('bw', 0.0) for i in range(hops)) if hops > 0 else float('inf')
                reused = sum(1 for i in range(hops) if (path[i], path[i + 1]) in tree_edges)
                cost = float(nx.path_weight(G, path, weight='weight')) if hops > 0 else 0.0
                return {
                    'reachable': True,
                    'bottleneck': bottleneck,
                    'hops': hops,
                    'reused': reused,
                    'new_edges': max(0, hops - reused),
                    'cost': cost,
                }
            except Exception:
                return {
                    'reachable': False,
                    'bottleneck': 0.0,
                    'hops': 999,
                    'reused': 0,
                    'new_edges': 999,
                    'cost': float('inf'),
                }

        scores = []
        for idx in valid_indices:
            node = int(idx)
            stats_to_node = path_stats(start_node, node)
            if phase == 'destination_connection':
                bottleneck = stats_to_node['bottleneck']
                hops = stats_to_node['hops']
                bw_ratio = bottleneck / max(1e-6, bw_need) if bw_need > 0 else 2.0
                bw_bonus = 2.0 * min(bw_ratio, 2.0)
                if _remaining <= 1:
                    hop_w = 4.0
                elif _remaining <= 2:
                    hop_w = 3.0
                else:
                    hop_w = 2.0
                score = (
                    bw_bonus
                    - hop_w * hops
                    - 1.5 * stats_to_node['new_edges']
                    - 0.8 * stats_to_node['cost']
                    + 1.2 * stats_to_node['reused']
                )
            else:
                vnf_list = req.get('vnf', [])
                vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
                cpu_need = mem_need = 0.0
                if vnf_idx < len(vnf_list):
                    cpu_list = req.get('cpu_origin', req.get('cpu', []))
                    mem_list = req.get('memory_origin', req.get('memory', []))
                    cpu_need = float(cpu_list[vnf_idx]) if vnf_idx < len(cpu_list) else 0.0
                    mem_need = float(mem_list[vnf_idx]) if vnf_idx < len(mem_list) else 0.0
                cpu_avail = rm.pool.get_available_cpu(node)
                mem_avail = rm.pool.get_available_memory(node)
                cpu_slack = (cpu_avail - cpu_need) / max(1.0, cpu_need if cpu_need > 0 else cpu_avail)
                mem_slack = (mem_avail - mem_need) / max(1.0, mem_need if mem_need > 0 else mem_avail)
                if not stats_to_node['reachable']:
                    score = -1e6 + 0.1 * cpu_slack + 0.1 * mem_slack
                    scores.append((score, node))
                    continue

                dest_stats = [path_stats(node, d) for d in _undone_dests if int(d) != node]
                reachable_dest_stats = [s for s in dest_stats if s['reachable']]
                if reachable_dest_stats:
                    avg_dest_cost = sum(s['cost'] for s in reachable_dest_stats) / len(reachable_dest_stats)
                    avg_dest_hops = sum(s['hops'] for s in reachable_dest_stats) / len(reachable_dest_stats)
                    avg_dest_new = sum(s['new_edges'] for s in reachable_dest_stats) / len(reachable_dest_stats)
                    dest_reuse = sum(s['reused'] for s in reachable_dest_stats) / max(
                        1.0, sum(s['hops'] for s in reachable_dest_stats)
                    )
                    dest_reach_ratio = len(reachable_dest_stats) / max(1, len(dest_stats))
                elif dest_stats:
                    avg_dest_cost = float(n_nodes)
                    avg_dest_hops = float(n_nodes)
                    avg_dest_new = float(n_nodes)
                    dest_reuse = 0.0
                    dest_reach_ratio = 0.0
                else:
                    avg_dest_cost = 0.0
                    avg_dest_hops = 0.0
                    avg_dest_new = 0.0
                    dest_reuse = 0.0
                    dest_reach_ratio = 1.0

                is_last_vnf = bool(vnf_list) and int(vnf_idx) >= len(vnf_list) - 1
                future_w = 2.4 if is_last_vnf else 1.5
                score = (
                    0.9 * cpu_slack
                    + 0.9 * mem_slack
                    - 2.4 * stats_to_node['cost']
                    - 1.4 * stats_to_node['new_edges']
                    + 1.8 * stats_to_node['reused']
                    - future_w * avg_dest_cost
                    - 0.8 * avg_dest_new
                    - 0.25 * (stats_to_node['hops'] + avg_dest_hops) / max(1, n_nodes)
                    + 2.0 * dest_reuse
                    + 3.0 * dest_reach_ratio
                )
            scores.append((score, node))

        scores.sort(key=lambda x: x[0], reverse=True)
        if phase == 'vnf_deployment' and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[HighRouteScore] start={start_node} bw={bw_need:.1f} "
                f"top5={[(n, round(float(s), 3)) for s, n in scores[:5]]}"
            )
        return [node for _, node in scores]

    def apply_high_topk_mask(self, high_mask, start_node: int, high_topk: int):
        valid_indices = np.where(high_mask > 0)[0]
        if len(valid_indices) <= int(high_topk):
            return high_mask
        ranked = self.score_high_candidates(valid_indices, start_node)
        keep = set(ranked[: int(high_topk)])
        topk_mask = np.zeros_like(high_mask)
        for idx in keep:
            topk_mask[idx] = 1.0
        return topk_mask

    def select_bw_feasible_target(
        self,
        target_node: int,
        start_node: int,
        bw_need: float,
        valid_indices: Iterable[int],
    ) -> int:
        rm = getattr(self.env, 'resource_mgr', None)
        if rm is None:
            return target_node

        G = nx.DiGraph()
        for u in range(rm.n):
            for v in rm.get_neighbors(u):
                avail = rm.pool.get_available_bandwidth(u, v)
                G.add_edge(u, v, bw=avail)

        def path_stats(src, dst):
            if src == dst:
                return float('inf'), 0
            try:
                path = nx.shortest_path(G, src, dst)
                hops = len(path) - 1
                bottleneck = min(G[path[i]][path[i + 1]]['bw'] for i in range(hops))
                return bottleneck, hops
            except Exception:
                return 0.0, 999

        cur_bw, cur_hops = path_stats(start_node, target_node)
        if cur_bw >= bw_need:
            return target_node

        cur_score = cur_bw / max(1, cur_hops)
        best_node = target_node
        best_score = cur_score
        for idx in valid_indices:
            node = int(idx)
            if node == target_node:
                continue
            bw, hops = path_stats(start_node, node)
            if bw < bw_need:
                continue
            score = bw / max(1, hops)
            if score > best_score:
                best_score = score
                best_node = node

        if best_node != target_node:
            logger.debug(
                f"[Shared][BW过滤v3] target={target_node} BW={cur_bw:.1f}<{bw_need:.1f}, "
                f"换为 {best_node} (score={best_score:.2f})"
            )
        return best_node

    def choose_best_anchor(
        self,
        target_goal: int,
        hop_distance_fn: Callable[[int, int], int],
        failed_anchors: set = None,
    ) -> int:
        chain = getattr(self.env, 'chain_nodes', [])
        fallback = chain[-1] if chain else getattr(self.env, 'current_node_location', None)
        if failed_anchors is None:
            failed_anchors = set()

        try:
            nodes_on_tree = set(getattr(self.env, 'nodes_on_tree', set()))
            logger.debug(
                f"[Anchor.choose] target={target_goal} "
                f"nodes_on_tree={len(nodes_on_tree)} fallback={fallback}"
            )
            if not nodes_on_tree:
                logger.debug(
                    f"[Anchor.choose] nodes_on_tree 为空，退回 fallback={fallback} "
                    f"(chain={chain})"
                )
                return fallback

            tree_edges = self.get_positive_tree_edge_set()
            rm = getattr(self.env, 'resource_mgr', None)
            if rm is None:
                return fallback
            max_hops = max(int(getattr(self.env, 'n', 1)), 1)

            best_score = -9999.0
            best_anchor = fallback
            scored_candidates = []

            connected_dests = self.get_connected_dests_view() if getattr(self.env, 'current_tree', None) else set()
            is_first_dest = (len(connected_dests) == 0)

            all_dests = []
            if getattr(self.env, 'current_request', None):
                all_dests = [int(d) for d in self.env.current_request.get('dest', [])]
            undone_dests = [d for d in all_dests if d not in connected_dests]

            total_vnf = len(self.env.current_request.get('vnf', [])) if getattr(self.env, 'current_request', None) else 0
            node_stage = self.env.current_tree.get('node_stage', {}) if getattr(self.env, 'current_tree', None) else {}

            full_stage_nodes = set()
            if total_vnf > 0:
                full_stage_nodes = {n for n in nodes_on_tree if node_stage.get(n, 0) >= total_vnf}
                full_stage_nodes.add(fallback)

            root_component_nodes = set(nodes_on_tree)
            try:
                g_pos = nx.Graph()
                g_pos.add_edges_from(list(tree_edges))
                if fallback is not None and g_pos.has_node(fallback):
                    root_component_nodes = set(nx.node_connected_component(g_pos, fallback))
                else:
                    root_component_nodes = {fallback} if fallback is not None else set()
            except Exception:
                root_component_nodes = {fallback} if fallback is not None else set(nodes_on_tree)

            if root_component_nodes:
                before_component_filter = len(full_stage_nodes)
                full_stage_nodes = {n for n in full_stage_nodes if n in root_component_nodes}
                if fallback is not None:
                    full_stage_nodes.add(fallback)
                if before_component_filter != len(full_stage_nodes):
                    logger.info(
                        f"[AnchorComponentFilter] target={target_goal} "
                        f"kept={len(full_stage_nodes)}/{before_component_filter} "
                        f"fallback={fallback} root_component_size={len(root_component_nodes)}"
                    )

            if full_stage_nodes:
                anchor_pool = full_stage_nodes
                full_stage_filtered = True
                logger.debug(
                    f"[Anchor.choose] full_stage_filter: "
                    f"{len(full_stage_nodes)}/{len(nodes_on_tree)} 节点通过 "
                    f"(total_vnf={total_vnf}, fallback={fallback} always included)"
                )
            else:
                anchor_pool = {fallback}
                full_stage_filtered = False
                logger.debug(
                    f"[Anchor.choose] full_stage_filter: 无全阶段节点且 total_vnf=0, "
                    f"锁死 fallback={fallback}"
                )

            bw_req = 0.0
            if getattr(self.env, 'current_request', None):
                bw_req = float(self.env.current_request.get('bw_origin', 0.0) or 0.0)

            def _bw_feasible_path_stats(src: int, dst: int) -> Tuple[int, int, int, float]:
                """Return (hop_count, reused_edges, new_edges, weighted_cost)."""
                G_bw = nx.DiGraph()
                for u in range(int(getattr(self.env, 'n', 0))):
                    try:
                        nbrs = rm.get_neighbors(u)
                    except Exception:
                        continue
                    for v in nbrs:
                        ek = (u, v)
                        try:
                            avail = rm.pool.get_available_bandwidth(u, v)
                            bw_ok = avail >= bw_req
                        except Exception:
                            avail = 0.0
                            bw_ok = False
                        if ek in tree_edges or bw_ok:
                            try:
                                cap = rm.pool.bw_cap.get((u, v), 1.0)
                                util = 1.0 - float(avail) / max(float(cap), 1.0)
                            except Exception:
                                util = 0.0
                            if ek in tree_edges:
                                weight = 0.05 + 0.10 * util
                            elif (v, u) in tree_edges:
                                # Reverse direction is not free reuse, but it still follows
                                # the existing tree corridor and is usually less detouring.
                                weight = 0.75 + 0.60 * util
                            else:
                                weight = 1.20 + 1.20 * util
                            G_bw.add_edge(u, v, weight=weight)
                if src not in G_bw or dst not in G_bw:
                    return 9999, 0, 9999, float('inf')
                try:
                    p = nx.shortest_path(G_bw, src, dst, weight='weight')
                except Exception:
                    return 9999, 0, 9999, float('inf')
                reused = sum(1 for j in range(len(p) - 1) if (p[j], p[j + 1]) in tree_edges)
                hops = max(0, len(p) - 1)
                new_edges = max(0, hops - reused)
                try:
                    cost = float(nx.path_weight(G_bw, p, weight='weight'))
                except Exception:
                    cost = float(hops)
                return hops, reused, new_edges, cost

            G = self.get_topology_graph()
            for anchor in anchor_pool:
                
                if anchor in failed_anchors:
                    logger.debug(f"[Anchor.choose] 跳过失败anchor={anchor} target={target_goal}")
                    continue
                hop_to_target = hop_distance_fn(anchor, target_goal)
                if hop_to_target >= 9999:
                    continue

                bw_hop_to_target, bw_reuse, bw_new_edges, bw_path_cost = _bw_feasible_path_stats(anchor, target_goal)
                if bw_hop_to_target >= 9999:
                    logger.debug(
                        f"[Anchor.choose] skip anchor={anchor} target={target_goal}: no BW-feasible path"
                    )
                    continue

                reuse = bw_reuse
                try:
                    path = nx.shortest_path(G, anchor, target_goal)
                    for j in range(len(path) - 1):
                        ek = (path[j], path[j + 1])
                        if ek in tree_edges:
                            reuse += 1
                except Exception:
                    reuse = 0

                slack = 0.0
                try:
                    c_cap = max(1.0, rm.C_cap)
                    m_cap = max(1.0, rm.M_cap)
                    c_avail = rm.pool.get_available_cpu(anchor)
                    m_avail = rm.pool.get_available_memory(anchor)
                    slack = 0.5 * (c_avail / c_cap + m_avail / m_cap)
                except Exception:
                    slack = 0.0

                detour_hop = max(0, bw_hop_to_target - hop_to_target)
                is_last_dest = (len(undone_dests) <= 1)

                if is_last_dest:
                    cur_loc = getattr(self.env, 'current_node_location', None)
                    hop_cur_to_anchor = hop_distance_fn(cur_loc, anchor) if cur_loc is not None else 0
                    score = (
                        -3.5 * bw_hop_to_target
                        -3.0 * bw_new_edges
                        -1.0 * bw_path_cost
                        + 1.2 * reuse
                        + 0.3 * slack
                        - 2.0 * detour_hop
                        - 0.4 * hop_cur_to_anchor
                    )
                elif is_first_dest:
                    avg_hop_to_others = 0.0
                    if len(undone_dests) > 1:
                        other_dests = [d for d in undone_dests if d != target_goal]
                        if other_dests:
                            hops_others = [hop_distance_fn(anchor, d) for d in other_dests]
                            valid_others = [h for h in hops_others if h < 9999]
                            avg_hop_to_others = (
                                sum(valid_others) / len(valid_others)
                                if valid_others else max_hops
                            )
                    score = (
                        -2.5 * bw_hop_to_target
                        -3.5 * bw_new_edges
                        -1.0 * bw_path_cost
                        + 2.2 * reuse
                        + 0.5 * slack
                        - 1.5 * detour_hop
                        - 1.0 * (avg_hop_to_others / max(max_hops, 1))
                    )
                else:
                    cur_loc = getattr(self.env, 'current_node_location', None)
                    hop_cur_to_anchor = hop_distance_fn(cur_loc, anchor) if cur_loc is not None else 0
                    score = (
                        -2.0 * bw_hop_to_target
                        -4.5 * bw_new_edges
                        -1.0 * bw_path_cost
                        + 2.5 * reuse
                        + 0.5 * slack
                        - 1.2 * detour_hop
                        - 0.5 * hop_cur_to_anchor
                    )

                scored_candidates.append((score, anchor, bw_hop_to_target, reuse, bw_new_edges))
                if score > best_score:
                    best_score = score
                    best_anchor = anchor

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            top3 = [
                (a, f"score={s:.2f} hop={h} reuse={r} new={ne}")
                for s, a, h, r, ne in scored_candidates[:3]
            ]
            try:
                self.env._last_anchor_top3 = top3
                self.env._last_anchor_choice = {
                    'target': int(target_goal),
                    'best_anchor': int(best_anchor) if best_anchor is not None else None,
                    'best_score': round(float(best_score), 4),
                    'pool_size': len(anchor_pool),
                    'candidate_count': len(scored_candidates),
                    'failed_anchors': sorted(int(x) for x in failed_anchors if x is not None),
                    'is_first_dest': bool(is_first_dest),
                    'top3': top3,
                }
            except Exception:
                pass

            
            
            if not scored_candidates:
                logger.debug(
                    f"[Anchor.choose] target={target_goal} 모든 anchor가 실패집합에 포함 "
                    f"failed={failed_anchors}, None 반환"
                )
                return None
            logger.debug(
                f"[Anchor.choose] target={target_goal} "
                f"best_anchor={best_anchor} score={best_score:.2f} "
                f"fallback={fallback} candidates={len(scored_candidates)} "
                f"top3={top3}"
            )
            if is_first_dest or (scored_candidates and scored_candidates[0][2] >= 4):
                logger.info(
                    f"[AnchorChoice] target={target_goal} best_anchor={best_anchor} "
                    f"first_dest={int(is_first_dest)} pool={len(anchor_pool)} "
                    f"candidates={len(scored_candidates)} failed={sorted(failed_anchors)} "
                    f"top3={top3}"
                )
            try:
                self.env._last_anchor_pool_size = len(anchor_pool)
            except Exception:
                pass
            logger.debug(
                f"[AnchorStats] target={target_goal} "
                f"best_anchor={best_anchor} "
                f"pool_size={len(anchor_pool)} "
                f"full_stage_filtered={full_stage_filtered} "
                f"total_vnf={total_vnf} "
                f"anchor_stage={node_stage.get(best_anchor, 0)} "
                f"is_first_dest={is_first_dest}"
            )
            return best_anchor
        except Exception as e:
            logger.debug(f"[Anchor.choose] 计算异常，退回 fallback={fallback}: {e}")
            return fallback
