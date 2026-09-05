#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSFCE 求解器 - 合并版
包含以下原始模块：
  - config.py          → SolverConfig, parse_mat_request
  - metrics.py         → MetricsCollector
  - cache_manager.py   → CacheManager, LinkCache
  - validators.py      → validate_request, validate_state, check_resource_availability
  - resource_manager.py→ ResourceManager
  - placement.py       → VNFPlacementStrategy, OptimizedPlacementStrategy
  - path_engine.py     → PathEngine  (standalone, used by MSFCE_Solver)
  - tree_builder.py    → _TreePathEngine (internal BFS engine), TreeBuilder
  - solver.py          → MSFCE_Solver
"""

import copy
import csv
import logging
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import scipy.io as sio

logger = logging.getLogger(__name__)


# =============================================================================
# config.py
# =============================================================================

@dataclass
class SolverConfig:
    """集中式配置管理"""
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.4
    candidate_set_size: int = 8
    lookahead_depth: int = 1
    k_path: int = 5
    max_cache_size: int = 5000
    max_iterations: int = 500
    max_time_seconds: float = 60.0
    max_candidates: int = 30
    otv_link_weight: float = 0.2
    otv_node_weight: float = 0.8
    otv_norm_link: float = 90.0
    otv_norm_node: float = 8.0

    def __post_init__(self):
        """参数验证"""
        if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1 and 0 <= self.gamma <= 1):
            raise ValueError("Alpha, beta, gamma must be between 0 and 1")
        if abs(self.alpha + self.beta + self.gamma - 1.0) > 1e-6:
            logger.warning("Score weights do not sum to 1.0")


def parse_mat_request(req_obj) -> Dict:
    """解析请求（兼容 Python Dict 和 MATLAB 格式）"""
    if isinstance(req_obj, dict):
        return req_obj

    try:
        return {
            'id': int(req_obj['id'][0, 0]),
            'source': int(req_obj['source'][0, 0]),
            'dest': [int(d) for d in req_obj['dest'].flatten()],
            'vnf': [int(v) for v in req_obj['vnf'].flatten()],
            'bw_origin': float(req_obj['bw_origin'][0, 0]),
            'cpu_origin': [float(c) for c in req_obj['cpu_origin'].flatten()],
            'memory_origin': [float(m) for m in req_obj['memory_origin'].flatten()],
            'arrival_time': int(req_obj.get('arrival_time', [[0]])[0, 0]),
            'leave_time': int(req_obj.get('leave_time', [[0]])[0, 0]),
        }
    except Exception:
        return {
            'id': int(req_obj[0][0][0]),
            'source': int(req_obj[0][1][0]),
            'dest': [int(x) for x in req_obj[0][2].flatten()],
            'vnf': [int(x) for x in req_obj[0][3].flatten()],
            'cpu_origin': [float(x) for x in req_obj[0][4].flatten()],
            'memory_origin': [float(x) for x in req_obj[0][5].flatten()],
            'bw_origin': float(req_obj[0][6][0][0])
        }


# =============================================================================
# metrics.py
# =============================================================================

class MetricsCollector:
    """性能指标收集器"""

    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'failure_reasons': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'errors': 0,
        }

    def record_request(self, accepted: bool, processing_time: float):
        """记录请求处理结果"""
        self.metrics['total_requests'] += 1
        if accepted:
            self.metrics['accepted'] += 1
        else:
            self.metrics['rejected'] += 1
        self.metrics['processing_times'].append(processing_time)

    def record_cache_access(self, hit: bool):
        """记录缓存访问"""
        if hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1

    def record_failure(self, reason: str):
        """记录失败原因"""
        if reason not in self.metrics['failure_reasons']:
            self.metrics['failure_reasons'][reason] = 0
        self.metrics['failure_reasons'][reason] += 1

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.metrics.copy()

        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['p95_processing_time'] = np.percentile(stats['processing_times'], 95)
            stats['p99_processing_time'] = np.percentile(stats['processing_times'], 99)
        else:
            stats['avg_processing_time'] = 0
            stats['p95_processing_time'] = 0
            stats['p99_processing_time'] = 0

        total_cache = stats['cache_hits'] + stats['cache_misses']
        stats['cache_hit_rate'] = stats['cache_hits'] / max(1, total_cache)

        total_req = stats['total_requests']
        stats['acceptance_rate'] = stats['accepted'] / max(1, total_req)

        return stats

    def reset(self):
        """重置所有指标"""
        self.__init__()


# =============================================================================
# cache_manager.py
# =============================================================================

class CacheManager:
    """缓存管理器"""

    def __init__(self, max_cache_size: int = 5000):
        self.max_cache_size = max_cache_size
        self._path_eval_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_path_eval(self, cache_key: Tuple) -> Optional[Tuple]:
        """获取缓存的路径评分"""
        if cache_key in self._path_eval_cache:
            value = self._path_eval_cache.pop(cache_key)
            self._path_eval_cache[cache_key] = value
            self.cache_hits += 1
            return value
        else:
            self.cache_misses += 1
            return None

    def set_path_eval(self, cache_key: Tuple, value: Tuple):
        """设置路径评分缓存"""
        self._path_eval_cache[cache_key] = value
        if len(self._path_eval_cache) > self.max_cache_size:
            self._path_eval_cache.popitem(last=False)

    def clear_path_eval_cache(self):
        """清空路径评分缓存"""
        self._path_eval_cache.clear()

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total)
        return {
            'path_eval_cache_size': len(self._path_eval_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def reset_stats(self):
        """重置统计信息"""
        self.cache_hits = 0
        self.cache_misses = 0


class LinkCache:
    """链路查找缓存"""

    def __init__(self):
        self._link_cache: Dict[Tuple[int, int], int] = {}

    def build_from_link_map(self, link_map: Dict):
        """从link_map构建缓存"""
        self._link_cache.clear()
        for edge, lid in link_map.items():
            self._link_cache[edge] = lid
        logger.info(f" Link lookup table: {len(self._link_cache)} entries")

    def get_link_id(self, u: int, v: int) -> Optional[int]:
        """获取链路ID（支持双向查询）"""
        return self._link_cache.get((u, v)) or self._link_cache.get((v, u))

    def __getitem__(self, edge: Tuple[int, int]) -> Optional[int]:
        return self._link_cache.get(edge)

    def get(self, edge: Tuple[int, int], default=None):
        return self._link_cache.get(edge, default)


# =============================================================================
# validators.py
# =============================================================================

def validate_request(request: Dict) -> bool:
    """验证请求格式"""
    required_fields = ['id', 'source', 'dest', 'vnf', 'bw_origin', 'cpu_origin', 'memory_origin']
    for field in required_fields:
        if field not in request:
            return False
    if not request['dest']:
        return False
    if len(request['vnf']) != len(request['cpu_origin']):
        return False
    if len(request['vnf']) != len(request['memory_origin']):
        return False
    return True


def validate_state(state: Dict, node_num: int, link_num: int) -> bool:
    """验证网络状态"""
    if 'cpu' not in state or 'mem' not in state or 'bw' not in state:
        return False
    if len(state['cpu']) != node_num:
        return False
    if len(state['mem']) != node_num:
        return False
    if len(state['bw']) != link_num:
        return False
    return True


def check_resource_availability(
        cpu_delta: np.ndarray,
        mem_delta: np.ndarray,
        rem_cpu: np.ndarray,
        rem_mem: np.ndarray,
        bw_req: float,
        rem_bw: np.ndarray,
        links_used: list,
        tolerance: float = 1e-7
) -> Tuple[bool, str]:
    """
    检查资源可用性（向量化版本）

    Returns:
        (is_feasible, reason)
    """
    violations = (cpu_delta > rem_cpu + tolerance) | (mem_delta > rem_mem + tolerance)
    if np.any(violations):
        return False, "CPU_or_MEM_violation"

    if links_used and bw_req > tolerance:
        unique_links = set(links_used)
        valid_indices = [lid - 1 for lid in unique_links if 0 < lid <= len(rem_bw)]
        if valid_indices and np.any(bw_req > rem_bw[valid_indices] + tolerance):
            return False, "BW_violation"

    return True, ""


# =============================================================================
# resource_manager.py
# =============================================================================

class ResourceManager:
    """
    完整的资源管理器

    修复记录:
    1.  [CRITICAL] 修复 check_global_feasibility 缩进错误，使其能被外部调用
    2.  [CRITICAL] 修复 _encode_request 返回全0向量的问题，增加特征维度
    3.  [LOGIC] check_global_feasibility 改为宽松逻辑（单路径可达即放行）
    4.  [LOGIC] apply_tree_deployment 增加 phase 参数，支持 Phase1 宽松模式
    """

    def __init__(self, *args, **kwargs):
        """
        初始化资源管理器
        支持:
        1. (topo, capacities, dc_nodes) - 新版
        2. (node_num, link_num, type_num, ...) - 旧版
        """
        self.init_mode = "unknown"
        self.node_index_base = kwargs.get('node_index_base', 1)
        self._lock = threading.RLock()

        if len(args) >= 4 and isinstance(args[0], int):
            self._init_legacy(*args, **kwargs)
        elif len(args) >= 2:
            self._init_modern(*args, **kwargs)
        else:
            if 'topo' in kwargs:
                self._init_modern(**kwargs)
            else:
                raise TypeError("Invalid arguments for ResourceManager init")

    def _init_legacy(self, node_num, link_num, type_num, cap_cpu, cap_mem=80.0, cap_bw=80.0, **kwargs):
        self.init_mode = "legacy"
        logger.info("[RM] Using legacy initialization")

        self.n = int(node_num)
        self.topo = np.ones((self.n, self.n), dtype=np.float32)
        np.fill_diagonal(self.topo, 0)

        self.L_provided = int(link_num)
        self.K_vnf = int(type_num)
        self.num_graph_edges = int(np.sum(self.topo > 0))
        self.C_cap = float(cap_cpu)
        self.M_cap = float(cap_mem)
        self.B_cap = float(cap_bw)
        self.dc_nodes = list(range(min(10, self.n)))

        self.link_map = self._create_link_map(self.topo)
        max_lid = max(self.link_map.values()) if self.link_map else 0
        self.L = max(self.L_provided, max_lid)

        self._init_common()

    def _init_modern(self, topo, capacities, dc_nodes, link_map=None, **kwargs):
        self.init_mode = "modern"
        logger.info("[RM] Using modern initialization")

        self.topo = topo
        self.n = topo.shape[0]
        self.num_graph_edges = int(np.sum(topo > 0))

        if link_map:
            self.link_map = link_map
        else:
            self.link_map = self._create_link_map(topo)

        self.L = max(self.link_map.values()) if self.link_map else 0

        self.C_cap = float(capacities.get('cpu', 80.0))
        self.M_cap = float(capacities.get('memory', 60.0))
        self.B_cap = float(capacities.get('bandwidth', 80.0))
        self.K_vnf = 8
        self.dc_nodes = list(dc_nodes)

        self._init_common()

    def _init_common(self):
        self.C = np.full(self.n, self.C_cap, dtype=np.float32)
        self.M = np.full(self.n, self.M_cap, dtype=np.float32)

        if self.L > 0:
            self.B = np.full(self.L, self.B_cap, dtype=np.float32)
        else:
            self.B = np.array([], dtype=np.float32)

        self.link_ref_count = np.zeros(self.L, dtype=np.int32)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=np.float32)

        self.nodes = {'cpu': self.C, 'memory': self.M}
        self.links = {'bandwidth': {}}

        for (u, v), lid in self.link_map.items():
            idx = lid - 1
            if 0 <= idx < self.L:
                self.links['bandwidth'][(u, v)] = self.B[idx]

        self.initial_C = self.C.copy()
        self.initial_M = self.M.copy()
        self.initial_B = self.B.copy()

        self.vnf_instances = []
        self.active_requests = {}
        self.vnf_sharing_map = defaultdict(set)

        try:
            self.shortest_dist = self._build_shortest_dist_matrix()
            self.edge_index = self._build_edge_index()
        except Exception as e:
            logger.warning(f"GNN结构初始化部分失败: {e}")
            self.edge_index = np.zeros((2, 0))

        self.node_feat_dim = 6 + self.K_vnf + 3
        self.edge_feat_dim = 5
        self.request_dim = 24

        self.initial_state_template = {
            'cpu': self.C.copy(),
            'mem': self.M.copy(),
            'bw': self.B.copy(),
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'bw_ref_count': np.zeros(self.L, dtype=np.int32)
        }
        logger.info(f" ResourceManager Ready: Nodes={self.n}, Links={self.L}")

    def _build_bw_feasible_subgraph(self, bw_req):
        """构建一个仅包含满足带宽需求链路的临时 NetworkX 图"""
        G = nx.Graph()
        for (u, v), lid in self.link_map.items():
            idx = lid - 1
            if self.B[idx] >= bw_req - 1e-5:
                G.add_edge(u, v)
        return G

    def check_global_feasibility(self, request, state):
        """
        检查全局可行性 (修正版)
        1. 自动转换 1-based 节点 ID 到 0-based 内部索引
        2. 在连通子图中检查 source -> dests 的可达性
        """
        raw_source = request.get('source')
        if raw_source is None:
            return False
        source = self._normalize_node(raw_source, from_external=True)

        raw_dests = request.get('destinations') or request.get('dest')
        if not raw_dests:
            return False
        dests = [self._normalize_node(d, from_external=True) for d in raw_dests]

        if source < 0 or source >= self.n:
            return False

        bw_req = request.get('bw_origin', 0)
        G_bw = self._build_bw_feasible_subgraph(bw_req)

        if not G_bw.has_node(source):
            return False

        for d in dests:
            if 0 <= d < self.n and G_bw.has_node(d):
                if nx.has_path(G_bw, source, d):
                    return True

        return False

    def create_initial_state(self):
        return copy.deepcopy(self.initial_state_template)

    def normalize_state(self, state):
        normalized = {}
        for key in ['cpu', 'mem', 'bw', 'hvt']:
            normalized[key] = state.get(key, self.initial_state_template[key].copy())
        normalized['bw_ref_count'] = state.get('bw_ref_count',
                                               self.initial_state_template['bw_ref_count'].copy())
        return normalized

    def reset(self):
        with self._lock:
            self.C[:] = self.initial_C
            self.M[:] = self.initial_M
            self.B[:] = self.initial_B
            self.hvt_all.fill(0)
            self.link_ref_count.fill(0)
            self.vnf_instances.clear()
            self.active_requests.clear()
            self.vnf_sharing_map.clear()

            for (u, v), lid in self.link_map.items():
                idx = lid - 1
                if 0 <= idx < self.L:
                    self.links['bandwidth'][(u, v)] = self.B[idx]

    def check_node_resources(self, node: int, cpu_req: float, mem_req: float,
                             external_index: bool = False) -> bool:
        internal_node = self._normalize_node(node, external_index)
        if internal_node < 0 or internal_node >= self.n:
            return False
        return (self.C[internal_node] >= cpu_req - 1e-5) and (self.M[internal_node] >= mem_req - 1e-5)

    def check_link_bandwidth(self, u: int, v: int, bw_req: float) -> bool:
        lid = self.link_map.get((u, v))
        if lid is None:
            return False
        idx = lid - 1
        if idx < 0 or idx >= self.L:
            return False
        return self.B[idx] >= bw_req - 1e-5

    def update_resources(self, node_id: int, cpu_delta: float, mem_delta: float,
                         strict: bool = True, external_index: bool = False) -> bool:
        with self._lock:
            internal_node = self._normalize_node(node_id, external_index)
            if internal_node < 0 or internal_node >= self.n:
                return False

            new_cpu = self.C[internal_node] + cpu_delta
            new_mem = self.M[internal_node] + mem_delta

            if strict:
                if new_cpu < -1e-5 or new_mem < -1e-5:
                    return False
                if new_cpu > self.C_cap + 1e-5:
                    return False

            self.C[internal_node] = np.clip(new_cpu, 0.0, self.C_cap)
            self.M[internal_node] = np.clip(new_mem, 0.0, self.M_cap)
            return True

    def allocate_link_bandwidth(self, u: int, v: int, bw_req: float) -> bool:
        with self._lock:
            if (u, v) not in self.link_map:
                return False
            lid = self.link_map[(u, v)]
            idx = lid - 1
            if idx < 0 or idx >= len(self.B):
                return False
            if self.B[idx] < bw_req - 1e-5:
                return False
            self.B[idx] = max(0.0, self.B[idx] - bw_req)
            self.link_ref_count[idx] += 1
            self.links['bandwidth'][(u, v)] = self.B[idx]
            return True

    def release_link_bandwidth(self, u: int, v: int, bw_req: float) -> bool:
        with self._lock:
            if (u, v) not in self.link_map:
                return False
            lid = self.link_map[(u, v)]
            idx = lid - 1
            if 0 <= idx < self.L:
                self.B[idx] = min(self.B_cap, self.B[idx] + bw_req)
                self.link_ref_count[idx] = max(0, self.link_ref_count[idx] - 1)
                self.links['bandwidth'][(u, v)] = self.B[idx]
                return True
            return False

    
    def consume_bandwidth(self, u, v, bw):
        return self.allocate_link_bandwidth(u, v, bw)

    def release_bandwidth(self, u, v, bw):
        return self.release_link_bandwidth(u, v, bw)

    def allocate_node_resources(self, n, c, m, vt=None):
        return self.update_resources(n, -c, -m, strict=True)

    def release_node_resources(self, n, c, m, vt=None):
        return self.update_resources(n, c, m, strict=False)

    def check_tree_bandwidth(self, tree, bw_req):
        """部署前带宽预检查（按物理链路聚合）"""
        link_demand = defaultdict(float)

        for edge, _ in tree.items():
            u, v = self._parse_edge(edge)
            if u is None or v is None:
                continue
            if (u, v) not in self.link_map:
                return False
            lid = self.link_map[(u, v)]
            idx = lid - 1
            if idx < 0 or idx >= self.L:
                return False
            link_demand[idx] += bw_req

        for idx, demand in link_demand.items():
            if self.B[idx] < demand - 1e-5:
                return False

        return True

    def apply_tree_deployment(self, plan: Dict, request: Dict, phase: str = "phase2") -> bool:
        """
        部署树到网络状态

        修复点：增加 phase 参数，支持 Phase 1 宽松模式
        """
        with self._lock:
            req_id = request.get('id', -1)

            if not self.apply_deployment(plan, request):
                return False

            if phase == "phase1":
                return True

            tree = plan.get('tree', {})
            bw_req = request.get('bw_origin', 0)

            if not self.check_tree_bandwidth(tree, bw_req):
                self._rollback_ops([
                    v for v in self.vnf_instances if v.get('req_id') == req_id
                ])
                return False

            link_demand = defaultdict(float)
            for edge, _ in tree.items():
                u, v = self._parse_edge(edge)
                if u is None or v is None:
                    continue
                link_demand[(u, v)] += bw_req

            deployed_links = []
            success = True

            for (u, v), demand in link_demand.items():
                if not self.allocate_link_bandwidth(u, v, demand):
                    success = False
                    break
                deployed_links.append((u, v, demand))

            if not success:
                for u, v, bw in deployed_links:
                    self.release_link_bandwidth(u, v, bw)
                self._rollback_ops([
                    v for v in self.vnf_instances if v.get('req_id') == req_id
                ])
                return False

            if req_id != -1:
                if req_id not in self.active_requests:
                    self.active_requests[req_id] = {}
                self.active_requests[req_id]['links'] = deployed_links
            return True

    def apply_deployment(self, plan: Dict, request: Dict) -> bool:
        with self._lock:
            parsed_ops = self._parse_deployment_ops(plan, request)
            if parsed_ops is None:
                return False

            for op in parsed_ops:
                if not self.check_node_resources(op['node'], op['cpu'], op['mem']):
                    return False

            executed_ops = []
            req_id = request.get('id', -1)

            for op in parsed_ops:
                node, c, m, vt = op['node'], op['cpu'], op['mem'], op['vnf_type']
                if not self.update_resources(node, -c, -m, strict=True):
                    self._rollback_ops(executed_ops)
                    return False

                hvt_inc = False
                if 0 <= vt < self.K_vnf:
                    self.hvt_all[node, vt] += 1.0
                    hvt_inc = True

                record = {
                    'req_id': req_id, 'node': node, 'cpu': c, 'memory': m,
                    'vnf_type': vt, 'hvt_inc': hvt_inc
                }
                executed_ops.append(record)
                self.vnf_instances.append(record)
            return True

    def _parse_deployment_ops(self, plan, request):
        placement = plan.get('placement', {})
        vnf_types = request.get('vnf', [])
        cpu_reqs = request.get('cpu_origin', [])
        mem_reqs = request.get('memory_origin', [])

        ops = []
        for key, node_id in placement.items():
            try:
                v_idx = -1
                if 'vnf_' in key and '_type_' in key:
                    v_idx = int(key.split('_')[1])
                elif key.startswith('vnf_'):
                    v_idx = int(key.split('_')[1])
                elif key.isdigit():
                    v_idx = int(key)

                if 0 <= v_idx < len(vnf_types):
                    v_type = vnf_types[v_idx]
                    c_req = cpu_reqs[v_idx]
                    m_req = mem_reqs[v_idx]
                    internal_node = self._normalize_node(node_id, from_external=True)
                    ops.append({'node': internal_node, 'cpu': c_req, 'mem': m_req, 'vnf_type': v_type})
            except Exception:
                continue
        return ops

    def _rollback_ops(self, executed_ops):
        for op in reversed(executed_ops):
            self.update_resources(op['node'], op['cpu'], op['memory'], strict=False)
            if op['hvt_inc']:
                vt = op['vnf_type']
                self.hvt_all[op['node'], vt] = max(0, self.hvt_all[op['node'], vt] - 1.0)
            if op in self.vnf_instances:
                self.vnf_instances.remove(op)

    def remove_request(self, req_id: int) -> bool:
        with self._lock:
            has_records = req_id in self.active_requests or any(
                v.get('req_id') == req_id for v in self.vnf_instances
            )
            if not has_records:
                return False

            if req_id in self.active_requests:
                links = self.active_requests[req_id].get('links', [])
                for u, v, bw in links:
                    self.release_link_bandwidth(u, v, bw)
                del self.active_requests[req_id]

            to_remove = [v for v in self.vnf_instances if v.get('req_id') == req_id]
            self._rollback_ops(to_remove)
            return True

    def _normalize_node(self, node: int, from_external: bool = False) -> int:
        if from_external and self.node_index_base == 1:
            return node - 1
        return node

    def _parse_edge(self, edge):
        try:
            if isinstance(edge, str):
                return tuple(map(int, edge.strip("()").replace(" ", "").split(",")))
            return int(edge[0]), int(edge[1])
        except Exception:
            return None, None

    def _create_link_map(self, topo):
        lm = {}
        lid = 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if topo[i, j] > 0:
                    lm[(i, j)] = lid
                    lm[(j, i)] = lid
                    lid += 1
        return lm

    def _build_shortest_dist_matrix(self):
        try:
            G = nx.from_numpy_array(self.topo)
            return nx.floyd_warshall_numpy(G)
        except Exception:
            return np.full((self.n, self.n), 999.0)

    def _build_edge_index(self):
        rows, cols = np.nonzero(self.topo)
        return np.array([rows, cols], dtype=np.int64)

    def get_gnn_state(self, current_request=None, **kwargs):
        x = self._build_node_features(current_request)
        req = self._encode_request(current_request) if current_request else np.zeros(self.request_dim)
        return {'x': x.astype(np.float32), 'edge_index': self.edge_index, 'request': req.astype(np.float32)}

    def _build_node_features(self, req):
        feats = []
        for i in range(self.n):
            base = [
                self.C[i] / self.C_cap,
                self.M[i] / self.M_cap,
                1.0 if i in self.dc_nodes else 0.0,
                0.0, 0.0, 0.0
            ]
            feats.append(np.concatenate([base, self.hvt_all[i], [0, 0, 0]]))
        return np.array(feats)

    def _encode_request(self, req):
        """
        修复点：填充实际特征，避免返回全0向量
        """
        vec = np.zeros(self.request_dim, dtype=np.float32)
        if not req:
            return vec

        vec[0] = req.get('source', 0) / self.n

        dests = req.get('destinations') or req.get('dest', [])
        vec[1] = len(dests) / self.n

        vec[2] = req.get('bw_origin', 0) / self.B_cap

        cpus = req.get('cpu_origin', [])
        mems = req.get('memory_origin', [])

        count = len(cpus) if len(cpus) > 0 else 1
        vec[3] = sum(cpus) / (self.C_cap * count)
        vec[4] = sum(mems) / (self.M_cap * count)

        return vec


# =============================================================================
# placement.py
# =============================================================================

class VNFPlacementStrategy:
    """VNF放置策略基类"""

    def __init__(self, node_num: int, type_num: int, node_index_base: int = 1):
        self.node_num = node_num
        self.type_num = type_num
        self.node_index_base = node_index_base
        logger.debug(f"[Placement] 初始化: 节点={node_num}, 类型={type_num}, 索引基值={node_index_base}")

    def place_vnf_chain(
            self,
            chain: List[int],
            cpu_reqs: List[float],
            mem_reqs: List[float],
            candidate_nodes: List[int],
            existing_hvt: np.ndarray,
            cpu_delta: np.ndarray,
            mem_delta: np.ndarray,
            vnf_delta: np.ndarray,
            state: Optional[Dict] = None,
            enable_debug: bool = False,
            **kwargs: Any
    ) -> Optional[Dict]:
        raise NotImplementedError

    def _to_internal(self, node: int) -> int:
        """外部节点ID → 内部索引 (0-based)"""
        if self.node_index_base == 1:
            return node - 1
        return node

    def _to_external(self, node: int) -> int:
        """内部索引 (0-based) → 外部节点ID"""
        if self.node_index_base == 1:
            return node + 1
        return node


class OptimizedPlacementStrategy(VNFPlacementStrategy):
    """
    高性能放置策略 - 带宽优化版

    核心特性：
    1. 容量感知三级策略（充裕/适中/紧张）
    2. 距离感知优化（减少树复杂度）
    3. 改进的资源预过滤
    """

    def place_vnf_chain(
            self,
            chain: List[int],
            cpu_reqs: List[float],
            mem_reqs: List[float],
            candidate_nodes: List[int],
            existing_hvt: np.ndarray,
            cpu_delta: np.ndarray,
            mem_delta: np.ndarray,
            vnf_delta: np.ndarray,
            state: Optional[Dict] = None,
            enable_debug: bool = False,
            **kwargs: Any
    ) -> Optional[Dict]:
        strategy_type = kwargs.get('strategy_type', 'capacity_aware')
        source_node = kwargs.get('source_node', None)
        distance_matrix = kwargs.get('distance_matrix', None)
        debug = enable_debug

        if debug:
            print(f"\n{'=' * 60}")
            print(f"[Placement] 开始放置VNF链: {chain}")
            print(f"  候选节点(外部): {candidate_nodes}")
            print(f"  策略类型: {strategy_type}")

        
        c_indices = []
        c_external = []
        for node_ext in candidate_nodes:
            node_int = self._to_internal(node_ext)
            if 0 <= node_int < self.node_num:
                c_indices.append(node_int)
                c_external.append(node_ext)
            elif debug:
                print(f"   忽略无效节点: 外部{node_ext} → 内部{node_int}")

        if not c_indices:
            if debug:
                print("   无有效候选节点")
            return None

        c_indices = np.array(c_indices)

        
        utilization = 0.5

        if state is not None and 'cpu' in state and len(state['cpu']) == self.node_num:
            cpu_used = state.get('cpu_used', np.zeros(self.node_num))
            mem_used = state.get('mem_used', np.zeros(self.node_num))
            cpu_capacity = state['cpu']
            mem_capacity = state['mem']

            cpu_remaining = cpu_capacity[c_indices] - cpu_used[c_indices] - cpu_delta[c_indices]
            mem_remaining = mem_capacity[c_indices] - mem_used[c_indices] - mem_delta[c_indices]

            avg_cpu_remaining = np.mean(cpu_remaining)
            avg_mem_remaining = np.mean(mem_remaining)
            cpu_cap = np.mean(cpu_capacity) if len(cpu_capacity) > 0 else 80.0
            mem_cap = np.mean(mem_capacity) if len(mem_capacity) > 0 else 60.0

            cpu_util = 1.0 - (avg_cpu_remaining / cpu_cap) if cpu_cap > 0 else 0.5
            mem_util = 1.0 - (avg_mem_remaining / mem_cap) if mem_cap > 0 else 0.5
            utilization = (cpu_util + mem_util) / 2

            if debug:
                print(f"   资源利用率: CPU={cpu_util * 100:.1f}%, MEM={mem_util * 100:.1f}%, 综合={utilization * 100:.1f}%")

            total_cpu = sum(cpu_reqs)
            total_mem = sum(mem_reqs)

            feasible_mask = (cpu_remaining >= total_cpu * 0.3) & (mem_remaining >= total_mem * 0.3)

            if debug and np.sum(feasible_mask) < len(c_indices):
                filtered = len(c_indices) - np.sum(feasible_mask)
                print(f"   资源预过滤: 移除{filtered}个节点")

            c_indices = c_indices[feasible_mask]
            c_external = [c_external[i] for i in range(len(feasible_mask)) if feasible_mask[i]]

            if len(c_indices) == 0:
                if debug:
                    print("   资源预过滤后无候选节点")
                return None

            curr_cpu = cpu_remaining[feasible_mask]
            curr_mem = mem_remaining[feasible_mask]
        else:
            curr_cpu = 1000.0 - cpu_delta[c_indices]
            curr_mem = 1000.0 - mem_delta[c_indices]

        
        cpu_reqs_np = np.array(cpu_reqs)
        mem_reqs_np = np.array(mem_reqs)

        if strategy_type == "cpu_heavy_first":
            order = np.argsort(-cpu_reqs_np)
        elif strategy_type == "mem_heavy_first":
            order = np.argsort(-mem_reqs_np)
        else:
            combined_req = cpu_reqs_np + mem_reqs_np
            order = np.argsort(-combined_req)

        if debug:
            print(f"  VNF处理顺序: {order}")

        placement = {}

        
        for vnf_idx in order:
            vnf_type = chain[vnf_idx]
            req_c = cpu_reqs[vnf_idx]
            req_m = mem_reqs[vnf_idx]
            vnf_t = vnf_type - 1

            if debug:
                print(f"\n   VNF{vnf_idx}(类型{vnf_type}): CPU={req_c:.1f}, MEM={req_m:.1f}")

            
            reuse_mask = existing_hvt[c_indices, vnf_t] > 0

            if np.any(reuse_mask):
                reuse_nodes = c_indices[reuse_mask]
                if len(reuse_nodes) > 1:
                    load_scores = curr_cpu[reuse_mask] + curr_mem[reuse_mask]
                    best_idx = np.argmax(load_scores)
                else:
                    best_idx = 0

                chosen_node_int = reuse_nodes[best_idx]
                chosen_node_ext = self._to_external(chosen_node_int)
                placement[(chosen_node_ext, vnf_type)] = chosen_node_ext

                if debug:
                    print(f"     复用: 节点{chosen_node_ext}")
                continue

            
            not_occupied_mask = vnf_delta[c_indices, vnf_t] == 0
            res_mask = (curr_cpu >= req_c - 1e-7) & (curr_mem >= req_m - 1e-7)
            valid_mask = not_occupied_mask & res_mask

            if not np.any(valid_mask):
                if debug:
                    print(f"     放置失败: 无合适节点")
                return None

            
            valid_nodes = c_indices[valid_mask]
            valid_cpu = curr_cpu[valid_mask]
            valid_mem = curr_mem[valid_mask]

            if strategy_type == "capacity_aware":
                if utilization < 0.3:
                    scores = -(valid_cpu + valid_mem)
                    best_idx = np.argmax(scores)
                    if debug:
                        print(f"     策略: 集中放置（利用率{utilization * 100:.1f}%）")
                elif utilization > 0.7:
                    scores = valid_cpu + valid_mem
                    best_idx = np.argmax(scores)
                    if debug:
                        print(f"     策略: 负载均衡（利用率{utilization * 100:.1f}%）")
                else:
                    resource_scores = valid_cpu + valid_mem

                    if source_node is not None and distance_matrix is not None:
                        source_int = self._to_internal(source_node)
                        if 0 <= source_int < self.node_num and source_int < len(distance_matrix):
                            distances = np.array([
                                distance_matrix[source_int, node_int]
                                if node_int < len(distance_matrix[source_int])
                                else 999
                                for node_int in valid_nodes
                            ])
                            max_dist = np.max(distances) if np.max(distances) > 0 else 1.0
                            distance_scores = 1.0 - (distances / max_dist)
                            scores = resource_scores * 0.6 + distance_scores * 0.4
                            if debug:
                                print(f"     策略: 混合（资源60% + 距离40%）")
                        else:
                            scores = resource_scores
                    else:
                        scores = resource_scores
                        if debug:
                            print(f"     策略: 资源优先（利用率{utilization * 100:.1f}%）")

                    best_idx = np.argmax(scores)

            elif strategy_type == "fragmentation":
                scores = valid_cpu + valid_mem
                best_idx = np.argmin(scores)
            else:
                scores = valid_cpu + valid_mem
                best_idx = np.argmax(scores)

            chosen_node_int = valid_nodes[best_idx]
            chosen_node_ext = self._to_external(chosen_node_int)

            pos_in_candidates = np.where(c_indices == chosen_node_int)[0][0]

            
            cpu_delta[chosen_node_int] += req_c
            mem_delta[chosen_node_int] += req_m
            vnf_delta[chosen_node_int, vnf_t] = 1

            curr_cpu[pos_in_candidates] -= req_c
            curr_mem[pos_in_candidates] -= req_m

            placement[(chosen_node_ext, vnf_type)] = chosen_node_ext

            if debug:
                print(f"     新放置: 节点{chosen_node_ext}")

        if debug:
            unique_nodes = len(set(placement.values()))
            concentration = (1 - unique_nodes / len(chain)) * 100 if len(chain) > 0 else 0
            print(f"\n   完成: 使用{unique_nodes}个节点, 集中度{concentration:.1f}%")
            print(f"{'=' * 60}")

        return placement


# =============================================================================
# path_engine.py  (standalone, used by MSFCE_Solver)
# =============================================================================

class PathEngine:
    """路径计算和查询引擎"""

    def __init__(
            self,
            path_db_file: Path,
            node_num: int,
            k_path: int,
            link_cache,
            topology_matrix: Optional[np.ndarray] = None,
    ):
        self.path_db_file = path_db_file
        self.node_num = node_num
        self.k_path = k_path
        self._link_cache = link_cache

        self._path_cache: Dict[Tuple[int, int, int], Tuple[List[int], int, List[int]]] = {}
        self._missing_paths: Set[Tuple[int, int, int]] = set()
        self._distance_matrix: Optional[np.ndarray] = None
        self.topo = None if topology_matrix is None else np.asarray(topology_matrix)

        self._load_path_db()
        self._precompute_distance_matrix()

    def _load_path_db(self):
        """加载路径数据库"""
        if not Path(self.path_db_file).exists():
            raise FileNotFoundError(f"Path DB missing: {self.path_db_file}")

        try:
            mat = sio.loadmat(self.path_db_file)
            self.path_db = mat['Paths']
            logger.info(f"Loaded Path DB from {self.path_db_file}")
        except Exception as e:
            raise RuntimeError(f"Path DB load failed: {e}")

    def _load_path_from_db(self, src: int, dst: int, k: int) -> Tuple[List[int], int, List[int]]:
        """从PathDB加载单条路径"""
        try:
            pinfo = self.path_db[src - 1, dst - 1]

            if 'paths' not in pinfo.dtype.names:
                return [], 0, []

            raw_paths = pinfo['paths']
            if raw_paths.size == 0:
                return [], 0, []

            idx = k - 1
            path_arr = None

            if raw_paths.dtype == 'O':
                flat_data = raw_paths.flatten()
                if idx < len(flat_data):
                    path_arr = flat_data[idx]
            elif raw_paths.ndim == 2:
                if idx < raw_paths.shape[0]:
                    path_arr = raw_paths[idx]
            elif raw_paths.ndim == 1 and idx == 0:
                path_arr = raw_paths

            if path_arr is None:
                return [], 0, []

            path_arr_flat = np.array(path_arr).flatten()

            dist_k = -1
            if 'pathsdistance' in pinfo.dtype.names:
                raw_dists = pinfo['pathsdistance'].flatten()
                if idx < len(raw_dists):
                    dist_k = int(raw_dists[idx])

            if dist_k >= 0 and (dist_k + 1) <= len(path_arr_flat):
                path_segment = path_arr_flat[:dist_k + 1]
            else:
                path_segment = path_arr_flat

            path_nodes = [int(x) for x in path_segment if int(x) > 0]

            if len(path_nodes) < 2:
                return [], 0, []

            links = self._compute_links_fast(path_nodes)
            return path_nodes, len(path_nodes) - 1, links

        except Exception:
            return [], 0, []

    def _compute_links_fast(self, path_nodes: List[int]) -> List[int]:
        """快速计算路径的链路ID列表"""
        links = []
        if len(path_nodes) <= 1:
            return links

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            link_id = self._link_cache.get((u, v)) or self._link_cache.get((v, u))
            if link_id is not None:
                links.append(link_id)

        return links

    def _precompute_distance_matrix(self):
        """Build hop distances without eagerly decoding every K-path entry."""
        n = self.node_num
        self._distance_matrix = np.full((n, n), 9999, dtype=int)
        np.fill_diagonal(self._distance_matrix, 0)

        if self.topo is not None and self.topo.shape == (n, n):
            try:
                import scipy.sparse as sp
                from scipy.sparse.csgraph import shortest_path

                distances = shortest_path(
                    csgraph=sp.csr_matrix(self.topo > 0),
                    directed=False,
                    unweighted=True,
                    return_predecessors=False,
                )
                distances[np.isinf(distances)] = 9999
                self._distance_matrix = distances.astype(int)
                logger.info(" Distance matrix computed from static topology")
                return
            except Exception as exc:
                logger.warning("Topology distance calculation failed; using path DB: %s", exc)

        computed_count = 0
        for src in range(1, n + 1):
            for dst in range(1, n + 1):
                if src == dst:
                    continue
                nodes, dist, _ = self._load_path_from_db(src, dst, 1)
                if nodes:
                    self._distance_matrix[src - 1, dst - 1] = dist
                    computed_count += 1

        logger.info(f" Distance matrix: {computed_count}/{n * (n - 1)} entries")

    def get_path_info(self, src: int, dst: int, k: int) -> Tuple[List[int], int, List[int]]:
        """Return a static path, decoding and caching it on first use."""
        if src == dst:
            return [src], 0, []

        if not (1 <= src <= self.node_num and 1 <= dst <= self.node_num):
            logger.warning(f"Invalid nodes: src={src}, dst={dst}")
            return [], 0, []

        if not (1 <= k <= self.k_path):
            return [], 0, []

        cache_key = (src, dst, k)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        if cache_key in self._missing_paths:
            return [], 0, []

        path_info = self._load_path_from_db(src, dst, k)
        if path_info[0]:
            self._path_cache[cache_key] = path_info
            if k == 1 and self._distance_matrix is not None:
                self._distance_matrix[src - 1, dst - 1] = path_info[1]
            return path_info

        self._missing_paths.add(cache_key)
        return [], 0, []

    def get_shortest_distance(self, src: int, dst: int) -> int:
        """快速获取最短距离（O(1)时间复杂度）"""
        if self._distance_matrix is None:
            return 9999
        if 1 <= src <= self.node_num and 1 <= dst <= self.node_num:
            return int(self._distance_matrix[src - 1, dst - 1])
        return 9999

    def get_max_hops(self, src: int, dst: int) -> int:
        """获取最大跳数"""
        try:
            nodes, dist, _ = self.get_path_info(src, dst, self.k_path)
            if nodes:
                return dist
            shortest = self.get_shortest_distance(src, dst)
            return shortest * 2 if shortest < 9999 else 10
        except Exception:
            return 10

    def validate_cache(self) -> bool:
        """Validate the backing DB; path coverage is populated lazily."""
        shape = getattr(self.path_db, 'shape', ())
        valid = len(shape) >= 2 and shape[0] >= self.node_num and shape[1] >= self.node_num
        if valid:
            logger.info(" Lazy path cache ready: %d loaded entries", len(self._path_cache))
        else:
            logger.warning(" Path DB shape %s is smaller than %dx%d", shape, self.node_num, self.node_num)
        return valid


# =============================================================================
# tree_builder.py
# =============================================================================

class _TreePathEngine:
    """
    局部路径引擎，供 TreeBuilder 内部使用。
    (与顶层 PathEngine 独立，基于拓扑矩阵做 BFS 和距离计算)
    """

    def __init__(
        self,
        topo: np.ndarray,
        dist_matrix: Optional[np.ndarray] = None,
        link_cache: Optional[Dict] = None,
    ):
        self.topo = topo
        self.n = topo.shape[0]
        self._link_cache = self._normalize_link_cache(link_cache)
        if not self._link_cache:
            self._link_cache = self._build_link_cache(topo)

        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        else:
            self.dist_matrix = self._compute_hop_distances()

    def _normalize_link_cache(self, link_cache: Optional[Dict]) -> Dict:
        if isinstance(link_cache, dict):
            return dict(link_cache)
        if hasattr(link_cache, "cache") and isinstance(link_cache.cache, dict):
            return dict(link_cache.cache)
        if hasattr(link_cache, "_cache") and isinstance(link_cache._cache, dict):
            return dict(link_cache._cache)
        return {}

    def _build_link_cache(self, topo: np.ndarray) -> Dict[Tuple[int, int], int]:
        cache: Dict[Tuple[int, int], int] = {}
        lid = 1
        rows, cols = np.where(np.triu(topo) > 0)
        for u, v in zip(rows, cols):
            cache[(u + 1, v + 1)] = lid
            cache[(v + 1, u + 1)] = lid
            lid += 1
        return cache

    def _compute_hop_distances(self) -> np.ndarray:
        try:
            import scipy.sparse as sp
            from scipy.sparse.csgraph import shortest_path
            graph = sp.csr_matrix(self.topo)
            d = shortest_path(csgraph=graph, directed=False, unweighted=True,
                              return_predecessors=False)
            if isinstance(d, tuple):
                d = d[0]
            d[np.isinf(d)] = 9999
            return d.astype(int)
        except Exception:
            return np.full((self.n, self.n), 9999, dtype=int)

    def get_link_id(self, u: int, v: int) -> Optional[int]:
        return self._link_cache.get((u, v)) or self._link_cache.get((v, u))

    def compute_links(self, path_nodes: List[int]) -> List[int]:
        links: List[int] = []
        for i in range(len(path_nodes) - 1):
            lid = self.get_link_id(path_nodes[i], path_nodes[i + 1])
            if lid is not None:
                links.append(lid)
        return links

    def bfs_shortest_path(self, src: int, dst: int) -> Optional[List[int]]:
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            if node < 1 or node > self.n:
                continue
            neighbors = np.where(self.topo[node - 1] > 0)[0] + 1
            for nxt in neighbors:
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                if nxt == dst:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
        return None

    def get_paths_from_tree_nodes(
        self,
        source_nodes: Set[int],
        target: int,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for src in sorted(source_nodes):
            path_nodes = self.bfs_shortest_path(src, target)
            if not path_nodes:
                continue
            edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
            links = self.compute_links(path_nodes)
            dist = 9999
            if self.dist_matrix is not None and 1 <= src <= self.n and 1 <= target <= self.n:
                dist = int(self.dist_matrix[src - 1, target - 1])
            results.append({
                "source": src,
                "target": target,
                "nodes": path_nodes,
                "edges": edges,
                "links": links,
                "hops": len(path_nodes) - 1,
                "distance": dist,
            })
        results.sort(key=lambda x: (x["distance"], x["hops"], x["source"]))
        return results[:k]


class TreeBuilder:
    """
    Tree Construction Algorithm

    目标：
    1. 不再让低层"随便走"，而是对高层选中的 target 做硬约束
    2. 对每个未覆盖目的节点，枚举"树上锚点 -> 目标"的可行最短路径
    3. 用更接近 MATLAB 专家风格的目标函数选扩展
    4. 保留原工程接口：construct_tree(request, network_state) -> (tree, traj, failed_dests)
    """

    def __init__(
        self,
        node_num: int,
        link_num: int,
        type_num: int,
        config,
        path_engine,
        resource_manager,
        placement_strategy=None,
    ):
        self.node_num = node_num
        self.link_num = link_num
        self.type_num = type_num
        self.config = config
        self.resource_manager = resource_manager

        self.stats = {
            "trees_evaluated": 0,
            "trees_pruned_pareto": 0,
            "trees_pruned_hash": 0,
            "trees_pruned_beam": 0,
        }

        topo_data = None
        link_cache_dict = None
        dist_matrix = None

        if hasattr(path_engine, "topo") and isinstance(path_engine.topo, np.ndarray):
            topo_data = path_engine.topo

        if hasattr(path_engine, "_link_cache"):
            link_cache_dict = self._extract_dict_safe(path_engine._link_cache)
        elif hasattr(path_engine, "link_map"):
            link_cache_dict = self._extract_dict_safe(path_engine.link_map)

        if hasattr(path_engine, "dist_matrix"):
            dist_matrix = path_engine.dist_matrix
        elif hasattr(path_engine, "_distance_matrix"):
            dist_matrix = path_engine._distance_matrix

        if topo_data is None:
            logger.warning("PathEngine missing topo, fallback reconstruction.")
            if link_cache_dict:
                topo_data = np.zeros((node_num, node_num), dtype=np.uint8)
                for key in link_cache_dict:
                    if isinstance(key, tuple) and len(key) == 2:
                        u, v = key
                        if 1 <= u <= node_num and 1 <= v <= node_num:
                            topo_data[u - 1, v - 1] = 1
                            topo_data[v - 1, u - 1] = 1
            elif dist_matrix is not None:
                topo_data = np.where(dist_matrix == 1, 1, 0).astype(np.uint8)
            elif isinstance(path_engine, np.ndarray):
                topo_data = path_engine.astype(np.uint8)

        if topo_data is None:
            logger.error("Failed to extract topology. Use empty topology.")
            topo_data = np.zeros((node_num, node_num), dtype=np.uint8)

        self._tree_path_engine = _TreePathEngine(topo_data, dist_matrix, link_cache_dict)
        self.dist_matrix = self._tree_path_engine.dist_matrix
        self.link_lookup = self._tree_path_engine._link_cache

        if placement_strategy is not None:
            self.placement_strategy = placement_strategy
        else:
            self.placement_strategy = OptimizedPlacementStrategy(node_num, type_num)

        self.alpha = float(getattr(config, "alpha", 1.0))
        self.beta = float(getattr(config, "beta", 1.0))
        self.gamma = float(getattr(config, "gamma", 1.0))
        self.beam_size = int(getattr(config, "candidate_set_size", 8))

        self.k_paths = int(getattr(config, "k_path", 5))
        self.max_paths_per_dest = int(getattr(config, "expert_max_paths_per_dest", 5))
        self.force_target_completion = bool(getattr(config, "force_target_completion", True))
        self.first_dest_bonus = float(getattr(config, "expert_first_dest_bonus", 0.0))
        self.reuse_bonus = float(getattr(config, "expert_reuse_bonus", 0.35))
        self.new_link_penalty = float(getattr(config, "expert_new_link_penalty", 1.0))
        self.detour_penalty = float(getattr(config, "expert_detour_penalty", 0.2))
        self.vnf_bonus = float(getattr(config, "expert_vnf_bonus", 0.15))

    # ------------------------------------------------------------------
    # basic utils
    # ------------------------------------------------------------------

    def _extract_dict_safe(self, obj) -> Dict:
        if isinstance(obj, dict):
            return dict(obj)
        if hasattr(obj, "cache") and isinstance(obj.cache, dict):
            return dict(obj.cache)
        if hasattr(obj, "_cache") and isinstance(obj._cache, dict):
            return dict(obj._cache)
        return {}

    def _fast_state_copy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        copied = state.copy()
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                copied[k] = v.copy()
        return copied

    def _init_tree_struct(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tree": np.zeros(self.link_num, dtype=np.uint8),
            "hvt": np.zeros((self.node_num, self.type_num), dtype=int),
            "nodes": {request["source"]},
            "paths_map": {},
            "covered_dests": set(),
            "link_count": 0,
            "node_count": 1,
            "structure_hash": 0,
            "sum_cpu": 0.0,
            "sum_mem": 0.0,
            "sum_bw": 0.0,
            "traj": [],
        }

    def _compute_links_fast(self, nodes: List[int]) -> List[int]:
        return self._tree_path_engine.compute_links(nodes)

    def _shortest_hop(self, src: int, dst: int) -> int:
        if self.dist_matrix is None:
            return 9999
        if not (1 <= src <= self.node_num and 1 <= dst <= self.node_num):
            return 9999
        return int(self.dist_matrix[src - 1, dst - 1])

    def _select_next_dest(
        self,
        source_nodes: Set[int],
        unadded: Set[int],
        forced_first_dest_idx: Optional[int],
        request: Dict[str, Any],
        step_idx: int,
    ) -> Optional[int]:
        if not unadded:
            return None

        dest_list = list(request["dest"])

        if step_idx == 0 and forced_first_dest_idx is not None:
            if 0 <= forced_first_dest_idx < len(dest_list):
                forced_dest = dest_list[forced_first_dest_idx]
                if forced_dest in unadded:
                    return forced_dest

        unadded_list = sorted(list(unadded))
        if not source_nodes or self.dist_matrix is None:
            return unadded_list[0]

        best_dest = None
        best_score = None
        for d in unadded_list:
            hop = min(self._shortest_hop(s, d) for s in source_nodes)
            score = (hop, d)
            if best_score is None or score < best_score:
                best_score = score
                best_dest = d
        return best_dest

    # ------------------------------------------------------------------
    # resource & tree update
    # ------------------------------------------------------------------

    def _check_resource_feasibility_safe(
        self,
        state: Dict[str, Any],
        delta: Dict[str, np.ndarray],
        request: Dict[str, Any],
        links_to_deduct: List[int],
    ) -> bool:
        if np.any(state["cpu"] < delta["cpu"] - 1e-5):
            return False
        if np.any(state["mem"] < delta["mem"] - 1e-5):
            return False

        bw_req = float(request.get("bw_origin", 0.0))
        if bw_req > 0:
            for lid in links_to_deduct:
                idx = lid - 1
                if 0 <= idx < len(state["bw"]):
                    if state["bw"][idx] < bw_req - 1e-5:
                        return False
        return True

    def _apply_path_to_tree(
        self,
        tree_struct: Dict[str, Any],
        info: Dict[str, Any],
        request: Dict[str, Any],
        state: Dict[str, Any],
        real_deploy: bool = False,
        resource_delta: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[bool, int, int]:
        """return: (success, new_link_count, reuse_link_count)"""
        nodes = info["nodes"]
        path_links = info.get("links")
        if path_links is None:
            path_links = self._compute_links_fast(nodes)

        links_to_deduct: List[int] = []
        new_link_indices: Set[int] = set()
        reuse_link_count = 0

        for lid in path_links:
            idx = lid - 1
            if not (0 <= idx < self.link_num):
                continue
            if tree_struct["tree"][idx] == 0 and idx not in new_link_indices:
                new_link_indices.add(idx)
                if real_deploy:
                    links_to_deduct.append(lid)
            else:
                reuse_link_count += 1

        if real_deploy and resource_delta is not None:
            if not self._check_resource_feasibility_safe(
                state, resource_delta, request, links_to_deduct
            ):
                return False, 0, 0

        for idx in new_link_indices:
            tree_struct["tree"][idx] = 1

        tree_struct["nodes"].update(nodes)
        tree_struct["paths_map"][nodes[-1]] = list(nodes)

        if "hvt" in info:
            tree_struct["hvt"] = np.maximum(tree_struct["hvt"], info["hvt"])

        tree_struct["link_count"] += len(new_link_indices)
        tree_struct["node_count"] = len(tree_struct["nodes"])

        if real_deploy and resource_delta is not None:
            state["cpu"] = np.maximum(state["cpu"] - resource_delta["cpu"], 0.0)
            state["mem"] = np.maximum(state["mem"] - resource_delta["mem"], 0.0)

            bw_req = float(request.get("bw_origin", 0.0))
            if bw_req > 0:
                for lid in links_to_deduct:
                    idx = lid - 1
                    if 0 <= idx < len(state["bw"]):
                        state["bw"][idx] = max(0.0, state["bw"][idx] - bw_req)

        return True, len(new_link_indices), reuse_link_count

    # ------------------------------------------------------------------
    # matlab-like scoring
    # ------------------------------------------------------------------

    def _count_new_vnf_instances(self, vnf_delta: np.ndarray) -> int:
        return int(np.count_nonzero(np.sum(vnf_delta, axis=1) > 0))

    def _score_candidate(
        self,
        tree_before: Dict[str, Any],
        tree_after: Dict[str, Any],
        path: Dict[str, Any],
        request: Dict[str, Any],
        cpu_delta: np.ndarray,
        mem_delta: np.ndarray,
        vnf_delta: np.ndarray,
        new_links: int,
        reuse_links: int,
        is_first_dest: bool,
    ) -> float:
        bw_req = float(request.get("bw_origin", 0.0))
        d_cpu = float(np.sum(cpu_delta))
        d_mem = float(np.sum(mem_delta))
        d_bw = float(new_links * bw_req)

        added_vnf_nodes = self._count_new_vnf_instances(vnf_delta)
        hops = int(path["hops"])
        shortest_hop = max(1, int(path["distance"]))
        detour_ratio = float(hops) / float(shortest_hop)

        score = 0.0
        score -= self.alpha * (d_cpu + d_mem)
        score -= self.gamma * d_bw
        score -= self.new_link_penalty * new_links
        score += self.reuse_bonus * reuse_links
        score += self.vnf_bonus * added_vnf_nodes

        if detour_ratio > 1.0:
            score -= self.detour_penalty * (detour_ratio - 1.0) * max(1, shortest_hop)

        if is_first_dest:
            score += self.first_dest_bonus

        if path["source"] == path["target"]:
            score += 5.0
        elif hops == 1:
            score += 2.0
        elif hops == 2:
            score += 0.8

        score -= 0.02 * tree_after["node_count"]
        score -= 0.05 * tree_after["link_count"]

        return float(score)

    # ------------------------------------------------------------------
    # placement + path evaluation
    # ------------------------------------------------------------------

    def _try_expand_to_target(
        self,
        base_tree: Dict[str, Any],
        base_state: Dict[str, Any],
        request: Dict[str, Any],
        target_dest: int,
        is_first_dest: bool,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        枚举"树上锚点 -> target"的路径，返回所有可行候选：
        [(new_tree, new_state, score), ...]
        """
        candidates: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []

        paths = self._tree_path_engine.get_paths_from_tree_nodes(
            base_tree["nodes"],
            target_dest,
            k=max(1, self.max_paths_per_dest),
        )
        if not paths:
            return candidates

        for path in paths:
            self.stats["trees_evaluated"] += 1

            branch_tree = {
                "tree": base_tree["tree"].copy(),
                "hvt": base_tree["hvt"].copy(),
                "nodes": base_tree["nodes"].copy(),
                "paths_map": dict(base_tree["paths_map"]),
                "covered_dests": base_tree["covered_dests"].copy(),
                "link_count": int(base_tree["link_count"]),
                "node_count": int(base_tree["node_count"]),
                "structure_hash": int(base_tree.get("structure_hash", 0)),
                "sum_cpu": float(base_tree.get("sum_cpu", 0.0)),
                "sum_mem": float(base_tree.get("sum_mem", 0.0)),
                "sum_bw": float(base_tree.get("sum_bw", 0.0)),
                "traj": list(base_tree.get("traj", [])),
            }
            branch_state = self._fast_state_copy(base_state)

            existing_hvt = branch_tree["hvt"].copy()
            cpu_delta = np.zeros(self.node_num, dtype=np.float64)
            mem_delta = np.zeros(self.node_num, dtype=np.float64)
            vnf_delta = np.zeros((self.node_num, self.type_num), dtype=np.int32)

            vnf_types = request.get("vnf", [])
            cpu_reqs = request.get("cpu_origin", [])
            mem_reqs = request.get("memory_origin", [])

            try:
                placement_res = self.placement_strategy.place_vnf_chain(
                    vnf_types,
                    cpu_reqs,
                    mem_reqs,
                    path["nodes"],
                    existing_hvt,
                    cpu_delta,
                    mem_delta,
                    vnf_delta,
                    branch_state,
                )
            except Exception as e:
                logger.debug(f"place_vnf_chain failed: {e}")
                placement_res = None

            if not placement_res:
                continue

            resource_delta = {
                "cpu": cpu_delta,
                "mem": mem_delta,
                "vnf": vnf_delta,
            }
            info = {
                "nodes": path["nodes"],
                "links": path["links"],
                "hvt": vnf_delta,
            }

            ok, new_links, reuse_links = self._apply_path_to_tree(
                branch_tree,
                info,
                request,
                branch_state,
                real_deploy=True,
                resource_delta=resource_delta,
            )
            if not ok:
                continue

            
            if path["nodes"][-1] != target_dest:
                continue

            branch_tree["covered_dests"].add(target_dest)

            d_cpu = float(np.sum(cpu_delta))
            d_mem = float(np.sum(mem_delta))
            d_bw = float(new_links * float(request.get("bw_origin", 0.0)))

            branch_tree["sum_cpu"] += d_cpu
            branch_tree["sum_mem"] += d_mem
            branch_tree["sum_bw"] += d_bw

            try:
                d_idx = request["dest"].index(target_dest)
                action_data = {
                    "target_dest": target_dest,
                    "anchor": path["source"],
                    "path": list(path["nodes"]),
                    "links": list(path["links"]),
                    "placement": placement_res,
                }
                branch_tree["traj"].append((d_idx, action_data, resource_delta))
            except ValueError:
                pass

            for u, v in path["edges"]:
                edge_hash = hash(tuple(sorted((u, v))))
                branch_tree["structure_hash"] = (
                    branch_tree["structure_hash"] * 1315423911
                ) ^ edge_hash

            score = self._score_candidate(
                tree_before=base_tree,
                tree_after=branch_tree,
                path=path,
                request=request,
                cpu_delta=cpu_delta,
                mem_delta=mem_delta,
                vnf_delta=vnf_delta,
                new_links=new_links,
                reuse_links=reuse_links,
                is_first_dest=is_first_dest,
            )

            candidates.append((branch_tree, branch_state, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # construct tree (public interface)
    # ------------------------------------------------------------------

    def construct_tree(
        self,
        request: Dict[str, Any],
        network_state: Dict[str, Any],
        forced_first_dest_idx: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List, List[int]]:
        self.stats = {k: 0 for k in self.stats}

        destinations = set(request["dest"])
        if not destinations:
            tree = self._init_tree_struct(request)
            tree["added_dest_indices"] = []
            return tree, [], []

        initial_tree = self._init_tree_struct(request)
        initial_state = self._fast_state_copy(network_state)

        candidate_beam: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = [
            (initial_tree, initial_state, 0.0)
        ]

        seen_trees: Set[Tuple[int, Tuple[int, ...]]] = set()

        for step in range(len(destinations)):
            next_beam: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
            is_first_dest = (step == 0)

            for tree, state, cum_score in candidate_beam:
                unadded = destinations - tree["covered_dests"]
                if not unadded:
                    next_beam.append((tree, state, cum_score))
                    continue

                target_dest = self._select_next_dest(
                    tree["nodes"],
                    unadded,
                    forced_first_dest_idx,
                    request,
                    step,
                )
                if target_dest is None:
                    continue

                expanded = self._try_expand_to_target(
                    base_tree=tree,
                    base_state=state,
                    request=request,
                    target_dest=target_dest,
                    is_first_dest=is_first_dest,
                )

                if not expanded:
                    if self.force_target_completion:
                        continue
                    else:
                        next_beam.append((tree, state, cum_score))
                        continue

                for new_tree, new_state, local_score in expanded:
                    sig = (
                        int(new_tree["structure_hash"]),
                        tuple(sorted(new_tree["covered_dests"])),
                    )
                    if sig in seen_trees:
                        self.stats["trees_pruned_hash"] += 1
                        continue
                    seen_trees.add(sig)
                    next_beam.append((new_tree, new_state, cum_score + local_score))

            if not next_beam:
                break

            next_beam.sort(
                key=lambda x: (
                    -len(x[0]["covered_dests"]),
                    -x[2],
                    x[0]["link_count"],
                    x[0]["node_count"],
                )
            )

            if len(next_beam) > self.beam_size:
                self.stats["trees_pruned_beam"] += len(next_beam) - self.beam_size
                next_beam = next_beam[: self.beam_size]

            candidate_beam = next_beam

        if not candidate_beam:
            return None, [], list(request["dest"])

        candidate_beam.sort(
            key=lambda x: (
                -len(x[0]["covered_dests"]),
                -x[2],
                x[0]["link_count"],
                x[0]["node_count"],
            )
        )
        best_tree, _, _ = candidate_beam[0]

        final_tree: Dict[str, Any] = best_tree
        final_tree["added_dest_indices"] = [
            request["dest"].index(d)
            for d in sorted(final_tree["covered_dests"])
            if d in request["dest"]
        ]

        failed_dests = list(destinations - final_tree["covered_dests"])
        return final_tree, final_tree["traj"], failed_dests

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()


# =============================================================================
# solver.py  (MSFCE_Solver)
# =============================================================================

class MSFCE_Solver:
    """MSFCE专家算法求解器（合并版）"""

    def __init__(
            self,
            path_db_file: Path,
            topology_matrix: np.ndarray,
            dc_nodes: List[int],
            capacities: Dict,
            config: Optional[SolverConfig] = None
    ):
        logger.info("=" * 60)
        logger.info("Initializing MSFCE Solver (Merged Version)")
        logger.info("=" * 60)

        self.config = config or SolverConfig()
        self._recall_failed_req_ids: Set[int] = set()

        self.node_num = int(topology_matrix.shape[0])
        self.type_num = 8
        self.link_num, self.link_map = self._create_link_map(topology_matrix)

        if dc_nodes and min(dc_nodes) == 0:
            logger.info("Converting DC nodes from 0-based to 1-based")
            self.DC = {n + 1 for n in dc_nodes}
        else:
            self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)

        self.cap_cpu = float(capacities['cpu'])
        self.cap_mem = float(capacities['memory'])
        self.cap_bw = float(capacities['bandwidth'])

        self.k_path = int(self.config.k_path)

        logger.info("Initializing modules...")

        
        self.link_cache = LinkCache()
        self.link_cache.build_from_link_map(self.link_map)

        
        logger.info("Initializing path engine...")
        self.path_engine = PathEngine(
            path_db_file,
            self.node_num,
            self.k_path,
            self.link_cache,
            topology_matrix=topology_matrix,
        )

        self.path_engine.validate_cache()

        
        self.cache_manager = CacheManager(self.config.max_cache_size)

        
        self.resource_manager = ResourceManager(
            topo=topology_matrix,
            capacities={
                'cpu': self.cap_cpu,
                'memory': self.cap_mem,
                'bandwidth': self.cap_bw,
            },
            dc_nodes=list(self.DC),
            link_map=self.link_map,
        )

        
        self.placement_strategy = OptimizedPlacementStrategy(self.node_num, self.type_num)

        
        self.tree_builder = TreeBuilder(
            self.node_num, self.link_num, self.type_num, self.config,
            self.path_engine, self.resource_manager, self.placement_strategy
        )
        self.tree_builder.DC = self.DC
        self.tree_builder.dc_num = self.dc_num

        
        self.metrics_collector = MetricsCollector()

        
        self.metrics = self.metrics_collector.metrics
        self.initial_state_template = self.resource_manager.initial_state_template

        logger.info("=" * 60)
        logger.info(" MSFCE Solver initialized successfully")
        logger.info(f"  Nodes: {self.node_num}, Links: {self.link_num}")
        logger.info(f"  DC nodes: {len(self.DC)}, VNF types: {self.type_num}")
        logger.info(f"  K-path: {self.k_path}")
        logger.info("=" * 60)

    def _create_link_map(self, topo: np.ndarray) -> Tuple[int, Dict]:
        """构建链路映射"""
        link_map = {}
        lid = 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

    def solve_request_for_expert(
            self,
            request: Dict,
            network_state: Optional[Dict] = None
    ) -> Tuple[Optional[Dict], List]:
        """
        求解单个请求

        Returns:
            (tree, trajectory)
        """
        start_time = time.time()

        try:
            if network_state is not None:
                current_state = {}
                for k, v in network_state.items():
                    if isinstance(v, np.ndarray):
                        current_state[k] = v.astype(np.float64).copy()
                    else:
                        current_state[k] = copy.deepcopy(v)
            else:
                current_state = self.resource_manager.create_initial_state()

            req_internal = copy.deepcopy(request)
            
            
            req_internal['source'] = req_internal['source'] + 1
            req_internal['dest'] = [d + 1 for d in req_internal['dest']]

            current_state['request'] = req_internal

            if not self.resource_manager.check_global_feasibility(req_internal, current_state):
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(False, processing_time)
                return None, []

            tree, traj, failed_dests = self.tree_builder.construct_tree(
                req_internal, current_state
            )

            processing_time = time.time() - start_time

            if tree is not None:
                self.metrics_collector.record_request(True, processing_time)
                return tree, traj
            else:
                self.metrics_collector.record_request(False, processing_time)
                return None, []

        except Exception as e:
            logger.exception(f"Error in solve_request_for_expert: {e}")
            self.metrics_collector.record_request(False, time.time() - start_time)
            return None, []

    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return self.metrics_collector.get_stats()

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        stats = self.cache_manager.get_stats()
        stats['path_cache_entries'] = len(self.path_engine._path_cache)
        return stats

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("SOLVER STATISTICS")
        print("=" * 70)

        metrics = self.get_metrics()
        print(f"Total Requests:  {metrics['total_requests']}")
        print(f"Accepted:        {metrics['accepted']}")
        print(f"Rejected:        {metrics['rejected']}")
        print(f"Acceptance Rate: {metrics['acceptance_rate']:.2%}")

        if metrics.get('avg_processing_time'):
            print(f"Avg Time:        {metrics['avg_processing_time'] * 1000:.2f} ms")
            print(f"P95 Time:        {metrics['p95_processing_time'] * 1000:.2f} ms")

        cache_stats = self.get_cache_stats()
        print(f"\nCache Hit Rate:  {cache_stats['hit_rate']:.2%}")
        print(f"Path Cache Size: {cache_stats['path_cache_entries']}")
        print("=" * 70 + "\n")

    def clear_cache(self):
        """清空路径评分缓存"""
        self.cache_manager.clear_path_eval_cache()
        logger.info("Path evaluation cache cleared")

    def export_metrics(self, path: Optional[Path] = None):
        """导出性能指标到 CSV"""
        if path is None:
            path = Path('expert_metrics.csv')

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Requests', self.metrics['total_requests']])
            writer.writerow(['Accepted', self.metrics['accepted']])
            writer.writerow(['Rejected', self.metrics['rejected']])

            accept_rate = self.metrics['accepted'] / max(1, self.metrics['total_requests'])
            writer.writerow(['Accept Rate', f"{accept_rate:.2%}"])

            writer.writerow([])
            writer.writerow(['Failure Reason', 'Count'])
            for reason, count in self.metrics.get('failure_reasons', {}).items():
                writer.writerow([reason, count])

            if self.metrics.get('processing_times'):
                writer.writerow([])
                writer.writerow(['Avg Processing Time (s)',
                                 np.mean(self.metrics['processing_times'])])

        logger.info(f"Metrics exported to {path}")

    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        report = {
            'total_requests': self.metrics['total_requests'],
            'acceptance_rate': self.metrics['accepted'] / max(1, self.metrics['total_requests']),
            'cache_hit_rate': self.metrics['cache_hits'] /
                              max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
            'failure_reasons': self.metrics.get('failure_reasons', {}),
        }

        if self.metrics.get('processing_times'):
            times = self.metrics['processing_times']
            report.update({
                'avg_processing_time': float(np.mean(times)),
                'max_processing_time': float(max(times)),
                'min_processing_time': float(min(times)),
            })

        return report

    def get_detailed_performance_report(self) -> Dict:
        """详细性能报告"""
        report = self.get_performance_report()

        cache_stats = self.get_cache_stats()
        report['cache_efficiency'] = {
            'cache_size': cache_stats['path_eval_cache_size'],
            'cache_max_size': self.config.max_cache_size,
            'cache_utilization': cache_stats['path_eval_cache_size'] / max(1, self.config.max_cache_size)
        }

        if self.metrics.get('processing_times'):
            times = self.metrics['processing_times']
            report['recent_performance'] = {
                'last_10_avg': float(np.mean(times[-10:])) if len(times) >= 10 else 0.0,
                'trend': 'improving' if len(times) > 1 and times[-1] < times[0] else 'stable'
            }

        return report

    def validate_cache(self) -> bool:
        """验证缓存完整性"""
        return self.path_engine.validate_cache()

    def _normalize_state(self, state: Dict) -> Dict:
        """规范化网络状态（兼容方法）"""
        return self.resource_manager.normalize_state(state)
