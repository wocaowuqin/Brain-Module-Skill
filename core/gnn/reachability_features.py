"""
reachability_features.py
节点↔目的集合的关系特征：跳数距离 + 瓶颈可用带宽。
纯函数，无类耦合，只依赖 numpy 与拓扑/带宽快照。

设计原则（重要）：
  这些特征必须在【状态构建时】算好并写进 Data.x，存进 buffer。
  绝不能在 encoder 里现算——replay 时资源已经变了，现算会用错带宽快照，
  且 state / next_state 必须各自用各自时刻的快照分别计算。
"""
import numpy as np


def precompute_all_pairs_hops(topo: np.ndarray) -> np.ndarray:
    """静态全源最短跳数。init 时调用一次。
    返回 D[i, j] = i 到 j 的最短跳数；不可达=该图直径+1（有限大数，便于归一化）。"""
    n = topo.shape[0]
    adj = (topo > 0)
    INF = n + 1
    D = np.full((n, n), INF, dtype=np.float32)
    for s in range(n):
        D[s, s] = 0.0
        frontier = np.array([s])
        dist = 0
        visited = np.zeros(n, dtype=bool); visited[s] = True
        while frontier.size > 0:
            dist += 1
            nxt = adj[frontier].any(axis=0) & (~visited)
            idx = np.where(nxt)[0]
            if idx.size == 0:
                break
            D[s, idx] = dist
            visited[idx] = True
            frontier = idx
    return D


def compute_widest_paths(avail_bw: np.ndarray) -> np.ndarray:
    """动态全源最宽路径（max-min 瓶颈）。每步调用。
    avail_bw[u, v] = 有向边 (u,v) 当前可用带宽，无边=0。
    返回 W[i, j] = i→j 所有路径中“最小边带宽”的最大值（瓶颈可用带宽）。
    向量化 Floyd-Warshall 变体，O(n) 次 O(n^2) numpy 运算。"""
    n = avail_bw.shape[0]
    W = avail_bw.astype(np.float32).copy()
    np.fill_diagonal(W, np.inf)
    for k in range(n):
        
        through_k = np.minimum(W[:, k:k+1], W[k:k+1, :])
        np.maximum(W, through_k, out=W)
    np.fill_diagonal(W, 0.0)
    return W


def node_to_destset_features(hops_all: np.ndarray,
                             widest: np.ndarray,
                             remaining_dests,
                             max_hop: float,
                             bw_cap: float) -> np.ndarray:
    """把 [n,n] 的两张关系表，针对【剩余目的集合】聚合成每节点特征。
    返回 [n, 4]：
      [0] 到最近剩余目的的跳数(归一化)      —— 多近能够到
      [1] 到剩余目的的平均跳数(归一化)      —— 整体方向远近
      [2] 到剩余目的的最大瓶颈带宽(归一化)   —— 最好的那条路有多宽
      [3] 到剩余目的的最小瓶颈带宽(归一化)   —— 最差目的有多难到（多播短板）
    无剩余目的时返回全 0。"""
    n = hops_all.shape[0]
    feat = np.zeros((n, 4), dtype=np.float32)
    dests = list(remaining_dests)
    if len(dests) == 0:
        return feat
    H = hops_all[:, dests]           # [n, |D|]
    B = widest[:, dests]             # [n, |D|]
    feat[:, 0] = H.min(axis=1) / max(1.0, max_hop)
    feat[:, 1] = H.mean(axis=1) / max(1.0, max_hop)
    feat[:, 2] = B.max(axis=1) / max(1.0, bw_cap)
    feat[:, 3] = B.min(axis=1) / max(1.0, bw_cap)
    return feat