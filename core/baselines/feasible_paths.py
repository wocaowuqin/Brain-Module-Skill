"""Bandwidth-aware K-shortest path selection for flat baselines.

The RL policy still selects the destination/placement node.  This helper only
replaces the old single static-shortest-path execution rule with a bounded,
deterministic feasibility check over cached simple paths.  It therefore keeps
the baseline action space unchanged while avoiding false ``link_blocked``
failures when an alternate path has enough residual bandwidth.
"""

from __future__ import annotations

from itertools import islice
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np


class FeasiblePathOracle:
    """Cache up to ``k`` hop-shortest paths and select a BW-feasible one."""

    def __init__(self, adjacency: np.ndarray, k: int = 8):
        matrix = np.asarray(adjacency, dtype=float)
        self.k = max(1, int(k))
        self.graph = nx.from_numpy_array(
            (matrix > 0).astype(np.int8), create_using=nx.DiGraph
        )
        self._cache: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}

    def paths(self, src: int, dst: int) -> tuple[tuple[int, ...], ...]:
        src, dst = int(src), int(dst)
        if src == dst:
            return ((src,),)
        key = (src, dst)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if src not in self.graph or dst not in self.graph:
            self._cache[key] = ()
            return ()
        try:
            iterator = nx.shortest_simple_paths(self.graph, src, dst)
            value = tuple(tuple(int(node) for node in path)
                          for path in islice(iterator, self.k))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            value = ()
        self._cache[key] = value
        return value

    @staticmethod
    def _edge_is_reused(used_edges: Iterable[tuple[int, int]], u: int, v: int) -> bool:
        used = used_edges if isinstance(used_edges, set) else set(used_edges)
        # Baseline executors allocate directed edges and only reuse the exact
        # orientation.  Treating (v,u) as reused would make the mask disagree
        # with the commit path and produce false link_blocked outcomes.
        return (u, v) in used

    def is_feasible(self, path: Sequence[int], bw_req: float, resource_manager,
                    used_edges: Iterable[tuple[int, int]] = ()) -> bool:
        if len(path) <= 1:
            return True
        for u, v in zip(path, path[1:]):
            if self._edge_is_reused(used_edges, int(u), int(v)):
                continue
            try:
                available = float(resource_manager.pool.get_available_bandwidth(int(u), int(v)))
            except Exception:
                return False
            if available < float(bw_req) - 1e-5:
                return False
        return True

    def select(self, src: int, dst: int, bw_req: float, resource_manager,
               used_edges: Iterable[tuple[int, int]] = ()) -> Optional[list[int]]:
        for path in self.paths(src, dst):
            if self.is_feasible(path, bw_req, resource_manager, used_edges):
                return list(path)
        return None
