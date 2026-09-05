import torch
"""
envs/modules/AllResourceManager.py（原 fused_resource_manager.py）
====================================
融合版资源管理器 - TA-HRL v4 (delayed-commit 模式)
====================================

【模块定位】
本模块管理多播 SFC 网络中所有物理资源的分配、释放与生命周期，是整个系统的资源账本。
采用 delayed-commit 模式：CPU/MEM 在 VNF 部署时即时扣减，BW 在请求成功后一次性提交。

【核心组件】
  SharedResourcePool         物理资源原子操作（CPU/MEM/BW 的 allocate/release/check）
  RequestLifecycleManager    请求生命周期管理（注册、到期释放、全局清理）
  RequestHandler             业务辅助方法（_try_deploy、_archive_request）
  FusedResourceManager       外观类，组合上述组件，对外提供统一接口

  TransactionManager 已删除：
    事务式预留/提交/回滚模型与 delayed-commit 主链不兼容，已完全废弃。

【资源记账语义（delayed-commit 模式）】
  CPU/MEM  →  _try_deploy() 部署 VNF 时即时扣，_archive_episode_fail() 失败时回滚
  BW       →  搜索阶段不扣，_commit_episode_bandwidth() 成功后对 flow>0 树边一次性提交
             _release_request() 到期释放时也只释放 flow>0 的边

【主要函数索引（FusedResourceManager）】
  allocate_node_resource()       扣 CPU/MEM，更新 hvt_all 部署计数
  release_node_resource()        还 CPU/MEM，归零 hvt_all 计数
  allocate_bandwidth()           扣链路 BW（仅 commit 时调用）
  release_bandwidth()            还链路 BW（lifecycle 到期时调用）
  get_available_bandwidth()      查询链路可用 BW（mask 计算 + planner 构图时调用）
  check_node_resource()          检查节点 CPU/MEM 是否满足需求（mask 生成时调用）
  _try_deploy()                  代理到 RequestHandler._try_deploy()
  _archive_request()             代理到 RequestHandler._archive_request()
  get_neighbors()                返回节点的拓扑邻居列表
  build_dynamic_edge_attr()      构建动态边属性（供 GNN encoder 使用）
  episode_reset()                episode 级轻量重置（清 current_tree/request，不动 lifecycle）
  reset()                        全局重置（清所有资源和 lifecycle）

【与其他模块的依赖关系】
  → LowLevelController     通过 env.resource_mgr 调用 allocate/release/check 系列
  → HighLevelController    通过 env.resource_mgr 查询 CPU/MEM/BW 余量，生成 action mask
  → HRL_Coordinator        通过 env.resource_mgr 做 BW 感知目标打分
  → RequestLifecycleManager  成功请求注册进来，lifecycle 到期后自动触发 _release_request
  → TimeSlotManager        调用 check_and_release_expired() 驱动到期释放
"""

import numpy as np
import time
import logging
import threading
import copy
from collections import deque
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==================== 新内核数据结构（单一真相） ====================

@dataclass
class DeployResult:
    """VNF 部署结果，替代旧的 bool 返回值"""
    ok: bool
    reused: bool = False
    new_instance: bool = False
    inst_id: Optional[str] = None
    reason: str = ""
    cpu_used: float = 0.0
    mem_used: float = 0.0


@dataclass
class VNFBinding:
    """请求级别：记录该请求绑定了哪个VNF实例"""
    req_id: int
    node: int
    vnf_type: int
    inst_id: str
    reused: bool
    cpu: float
    mem: float


@dataclass
class EdgeAllocation:
    """请求级别：记录该请求占用了哪条边的带宽"""
    req_id: int
    u: int
    v: int
    bw: float


@dataclass
class RequestRecord:
    """请求级状态（单一真相）"""
    req_id: int
    source: int
    dests: List[int]
    vnfs: List[int]
    bw: float
    state: str = "PENDING"   # PENDING / ACTIVE / FAILED / EXPIRED / RELEASED
    connected_dests: Set[int] = field(default_factory=set)
    vnf_bindings: List[VNFBinding] = field(default_factory=list)
    edge_allocations: List[EdgeAllocation] = field(default_factory=list)
    # Persistent SFT snapshot for runtime reconfiguration/migration.
    # Resource accounting stays in vnf_bindings/edge_allocations; these fields
    # preserve the logical multicast service function tree after deployment.
    tree_edges: Dict[Tuple[int, int], float] = field(default_factory=dict)
    placement_by_vnf: Dict[int, int] = field(default_factory=dict)
    placement_detail: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    node_stage: Dict[int, int] = field(default_factory=dict)
    tree_usage: Dict[Tuple[int, int], int] = field(default_factory=dict)
    snapshot_time: Optional[float] = None
    last_reconfig_time: Optional[float] = None
    migration_count: int = 0
    reconfig_count: int = 0


@dataclass
class VNFInstanceRecord:
    """实例级状态（单一真相）"""
    inst_id: str
    node: int
    vnf_type: int
    cpu: float
    mem: float
    ref_count: int = 0
    active_req_ids: Set[int] = field(default_factory=set)
    state: str = "ACTIVE"    # ACTIVE / DELETED


# ==================== 共享资源池 ====================
class SharedResourcePool:
    """共享资源池 - 底层物理资源管理（原子操作）"""

    def __init__(self, topology: np.ndarray, capacities: Dict):
        self.n = topology.shape[0]
        self.topology = topology
        self.bandwidth_model = str(
            capacities.get('bandwidth_model',
                           capacities.get('link_bandwidth_model', 'directed'))
        ).lower()

        # 物理容量
        self.cpu_cap = np.full(self.n, capacities.get('cpu', 100.0), dtype=float)
        self.mem_cap = np.full(self.n, capacities.get('memory', 80.0), dtype=float)

        # 当前可用资源
        self.cpu_avail = self.cpu_cap.copy()
        self.mem_avail = self.mem_cap.copy()

        # 预留标记
        self.cpu_reserved = np.zeros(self.n, dtype=float)
        self.mem_reserved = np.zeros(self.n, dtype=float)

        # 链路相关
        self.link_map = {}          # (u,v) -> edge_id
        self.bw_cap = {}            # (u,v) -> 带宽容量
        self.bw_avail = {}          # (u,v) -> 当前可用带宽
        self.bw_reserved = {}       # (u,v) -> 预留带宽
        self.link_locks = {}        # (u,v) -> 锁

        # Bandwidth is queried repeatedly while constructing one HRL state and
        # its action mask.  Resource updates invalidate this small per-version
        # cache; reads within the same decision avoid reacquiring the link lock
        # for every feature and neighbor check.
        self._bw_version = 0
        self._bw_query_cache_version = -1
        self._bw_query_cache = {}

        # 节点锁
        self.node_locks = [threading.RLock() for _ in range(self.n)]

        self._init_links(capacities.get('bandwidth', 100.0))
        logger.debug(f"[SharedPool] 初始化: {self.n} 节点, {len(self.link_map)} 链路")

    def _shared_undirected_bw(self) -> bool:
        return self.bandwidth_model in {
            'shared_undirected', 'undirected', 'physical', 'shared'
        }

    def _bw_key(self, u: int, v: int):
        if not self._shared_undirected_bw():
            return (u, v)
        a, b = (u, v) if u <= v else (v, u)
        if (a, b) in self.bw_avail:
            return (a, b)
        if (b, a) in self.bw_avail:
            return (b, a)
        return (u, v)

    def _sync_reverse_bw(self, key):
        if not self._shared_undirected_bw():
            return
        u, v = key
        rev = (v, u)
        if rev in self.bw_avail:
            self.bw_avail[rev] = self.bw_avail[key]
        if rev in self.bw_reserved:
            self.bw_reserved[rev] = self.bw_reserved[key]

    def iter_bandwidth_keys(self):
        if not self._shared_undirected_bw():
            return list(self.link_map.keys())
        keys = []
        seen = set()
        for u, v in self.link_map:
            phy = tuple(sorted((u, v)))
            if phy in seen:
                continue
            seen.add(phy)
            keys.append(self._bw_key(u, v))
        return keys

    def _init_links(self, bw_cap: float):
        edge_id = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.topology[i, j] > 0:
                    key = (i, j)
                    self.link_map[key] = edge_id
                    self.bw_cap[key] = bw_cap
                    self.bw_avail[key] = bw_cap
                    self.bw_reserved[key] = 0.0
                    self.link_locks[key] = threading.RLock()
                    edge_id += 1
        self.L = len(self.link_map)

    # ---------- 节点资源操作 ----------
    def allocate_cpu(self, node: int, amount: float) -> bool:
        with self.node_locks[node]:
            if self.cpu_avail[node] >= amount - 1e-5:
                self.cpu_avail[node] -= amount
                return True
            return False

    def allocate_memory(self, node: int, amount: float) -> bool:
        with self.node_locks[node]:
            if self.mem_avail[node] >= amount - 1e-5:
                self.mem_avail[node] -= amount
                return True
            return False

    def release_cpu(self, node: int, amount: float):
        with self.node_locks[node]:
            self.cpu_avail[node] = min(self.cpu_cap[node], self.cpu_avail[node] + amount)

    def release_memory(self, node: int, amount: float):
        with self.node_locks[node]:
            self.mem_avail[node] = min(self.mem_cap[node], self.mem_avail[node] + amount)

    def reserve_cpu(self, node: int, amount: float) -> bool:
        with self.node_locks[node]:
            if self.cpu_avail[node] >= amount - 1e-5:
                self.cpu_avail[node] -= amount
                self.cpu_reserved[node] += amount
                return True
            return False

    def reserve_memory(self, node: int, amount: float) -> bool:
        with self.node_locks[node]:
            if self.mem_avail[node] >= amount - 1e-5:
                self.mem_avail[node] -= amount
                self.mem_reserved[node] += amount
                return True
            return False

    def commit_reservation(self, node: int, cpu_amount: float, mem_amount: float):
        with self.node_locks[node]:
            self.cpu_reserved[node] = max(0, self.cpu_reserved[node] - cpu_amount)
            self.mem_reserved[node] = max(0, self.mem_reserved[node] - mem_amount)

    def cancel_reservation(self, node: int, cpu_amount: float, mem_amount: float):
        with self.node_locks[node]:
            if cpu_amount > 0:
                self.cpu_avail[node] = min(self.cpu_cap[node], self.cpu_avail[node] + cpu_amount)
                self.cpu_reserved[node] = max(0, self.cpu_reserved[node] - cpu_amount)
            if mem_amount > 0:
                self.mem_avail[node] = min(self.mem_cap[node], self.mem_avail[node] + mem_amount)
                self.mem_reserved[node] = max(0, self.mem_reserved[node] - mem_amount)

    # ---------- 链路资源操作 ----------
    def allocate_bandwidth(self, u: int, v: int, amount: float) -> bool:
        key = self._bw_key(u, v)
        if key not in self.bw_avail:
            return False
        with self.link_locks[key]:
            if self.bw_avail[key] >= amount - 1e-5:
                self.bw_avail[key] -= amount
                self._sync_reverse_bw(key)
                self._bw_version += 1
                # [LEAK追踪] 累计分配BW总量
                self._dbg_alloc_bw = getattr(self, '_dbg_alloc_bw', 0.0) + amount
                return True
            return False

    def release_bandwidth(self, u: int, v: int, amount: float):
        key = self._bw_key(u, v)
        if key not in self.bw_avail:
            return
        with self.link_locks[key]:
            self.bw_avail[key] = min(self.bw_cap[key], self.bw_avail[key] + amount)
            self._sync_reverse_bw(key)
            self._bw_version += 1
            # [LEAK追踪] 累计归还BW总量
            self._dbg_rel_bw = getattr(self, '_dbg_rel_bw', 0.0) + amount

    def reserve_bandwidth(self, u: int, v: int, amount: float) -> bool:
        key = self._bw_key(u, v)
        if key not in self.bw_avail:
            return False
        with self.link_locks[key]:
            if self.bw_avail[key] >= amount - 1e-5:
                self.bw_avail[key] -= amount
                self.bw_reserved[key] += amount
                self._sync_reverse_bw(key)
                self._bw_version += 1
                return True
            return False

    def commit_link_reservation(self, u: int, v: int, amount: float):
        key = self._bw_key(u, v)
        if key not in self.bw_reserved:
            return
        with self.link_locks[key]:
            self.bw_reserved[key] = max(0, self.bw_reserved[key] - amount)
            self._sync_reverse_bw(key)

    def cancel_link_reservation(self, u: int, v: int, amount: float):
        key = self._bw_key(u, v)
        if key not in self.bw_avail:
            return
        with self.link_locks[key]:
            self.bw_avail[key] = min(self.bw_cap[key], self.bw_avail[key] + amount)
            self.bw_reserved[key] = max(0, self.bw_reserved[key] - amount)
            self._sync_reverse_bw(key)
            self._bw_version += 1

    # ---------- 查询接口 ----------
    def get_available_cpu(self, node: int) -> float:
        with self.node_locks[node]:
            return max(0, self.cpu_avail[node])

    def get_available_memory(self, node: int) -> float:
        with self.node_locks[node]:
            return max(0, self.mem_avail[node])

    def get_available_bandwidth(self, u: int, v: int) -> float:
        key = self._bw_key(u, v)
        if key not in self.bw_avail:
            return 0.0
        if self._bw_query_cache_version != self._bw_version:
            self._bw_query_cache.clear()
            self._bw_query_cache_version = self._bw_version
        cached = self._bw_query_cache.get(key)
        if cached is not None:
            return cached
        with self.link_locks[key]:
            value = max(0, self.bw_avail[key])
        self._bw_query_cache[key] = value
        return value

    def get_edge_id(self, u: int, v: int) -> Optional[int]:
        return self.link_map.get((u, v))

    def get_edge_key(self, edge_id: int) -> Optional[tuple]:
        for key, eid in self.link_map.items():
            if eid == edge_id:
                return key
        return None

    def reset(self, hard: bool = False):
        if hard:
            self.cpu_avail = self.cpu_cap.copy()
            self.mem_avail = self.mem_cap.copy()
            for key in self.bw_avail:
                self.bw_avail[key] = self.bw_cap[key]
            for key in list(self.bw_avail):
                self._sync_reverse_bw(key)
            self._bw_version += 1
            self._bw_query_cache.clear()
            self._bw_query_cache_version = self._bw_version
        self.cpu_reserved.fill(0)
        self.mem_reserved.fill(0)
        for key in self.bw_reserved:
            self.bw_reserved[key] = 0.0


# ==================== 事务管理器 ====================
# TransactionManager 已删除：
# 这是"预留→提交→回滚"事务式资源管理模型，与当前主链不兼容。
# 当前主链采用 delayed-commit 模式：
#   - CPU/MEM：_try_deploy() 即时扣，_archive_episode_fail() 失败时回滚
#   - BW：_commit_episode_bandwidth() 成功后对 flow>0 树边一次性提交
# 事务机制（reserve/commit/rollback）已完全由上述两套逻辑取代，不再需要。


# ==================== 请求生命周期管理器 ====================
class RequestLifecycleManager:
    """
    请求生命周期管理器 - 纯仿真时间版
    - 完全依赖外部传入的 current_time（仿真时间）
    - 移除了 cleanup_interval 节流，每个时间步都会检查
    - register_request 必须传入 arrival_time 和 lifetime
    - 所有释放操作都基于记录的资源量
    """

    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.active_requests: Dict[str, dict] = {}
        self.expired_requests: Dict[str, dict] = {}
        self.lock = threading.RLock()
        self.stats = {
            'total_registered': 0,
            'total_expired': 0,
            'total_failed': 0,
            'total_cpu_released': 0.0,
            'total_mem_released': 0.0,
            'total_bw_released': 0.0,
        }

    def register_request(self, request: dict, resources_allocated: dict) -> bool:
        """
        注册请求及其占用的资源
        要求 request 必须包含 'id', 'arrival_time', 'lifetime'
        """
        req_id = request.get('id')
        if req_id is None:
            logger.error("[Lifecycle] 请求缺少ID")
            return False

        arrival = request.get('arrival_time')
        lifetime = request.get('lifetime')
        if arrival is None or lifetime is None:
            logger.error("[Lifecycle] 请求缺少 arrival_time 或 lifetime")
            return False

        expire_time = arrival + lifetime

        with self.lock:
            if req_id in self.active_requests:
                logger.warning(f"[Lifecycle] 请求 {req_id} 已存在，跳过")
                return False

            self.active_requests[req_id] = {
                'request': copy.deepcopy(request),
                'resources': copy.deepcopy(resources_allocated),
                'arrival_time': arrival,
                'lifetime': lifetime,
                'expire_time': expire_time,
                'status': 'active'
            }
            self.stats['total_registered'] += 1
            logger.debug(f"[Lifecycle] 注册请求 {req_id}, 过期时间 {expire_time:.2f}")
            return True

    def rollback_registration(self, req_id) -> bool:
        """Undo lifecycle metadata added by a failed success finalization."""
        with self.lock:
            if req_id not in self.active_requests:
                return False
            del self.active_requests[req_id]
            self.stats['total_registered'] = max(
                0, int(self.stats.get('total_registered', 0)) - 1
            )
            self.stats['total_failed'] = int(self.stats.get('total_failed', 0)) + 1
            logger.debug(f"[Lifecycle] 回滚请求 {req_id} 的注册元数据")
            return True

    def check_and_release_expired(self, current_time: float) -> List[str]:
        """
        检查并释放所有过期的请求
        current_time: 当前仿真时间
        返回已释放的请求ID列表
        """
        expired = []
        with self.lock:
            if self.active_requests:
                expires = {rid: info['expire_time']
                           for rid, info in self.active_requests.items()}
                logger.debug(f"[Lifecycle] t={current_time:.3f} | 活跃请求: "
                             f"{', '.join(f'{r}→exp{e:.2f}' for r,e in expires.items())}")
            for req_id, info in list(self.active_requests.items()):
                if current_time > info['expire_time']:
                    expired.append(req_id)
            for req_id in expired:
                self._release_request(req_id, current_time)
        return expired

    def _release_request(self, req_id: str, current_time: float):
        """
        释放单个请求的资源（完全新内核版）。

        主路径：调用 release_request_record(req_id) — 它负责：
          - BW 释放（via edge_allocations，不走 resources['tree']）
          - CPU/MEM 释放（via _release_vnf_binding → ref_count）
        旧路径 fallback：只有在 request_table 中完全找不到记录时才触发，
          仅用 release_vnf_ref 释放 CPU/MEM；BW 已通过新内核管理，不再重复释放。
        active_requests 本身只保留时序元数据（到达/到期时间）；
        resources['placement'] / resources['tree'] 不再作为释放来源。
        """
        if req_id not in self.active_requests:
            return
        info = self.active_requests[req_id]
        rm = self.resource_manager

        # ── 新内核统一释放链（主路径）────────────────────────────────
        released = rm.release_request_record(req_id, rollback=False)
        if released:
            logger.debug(f"[Lifecycle-NewKernel] 请求 {req_id} → release_request_record OK")
        else:
            # ── 旧兼容路径 fallback（request_table 无记录时）────────
            # 仅释放 CPU/MEM（BW 未在新内核登记，说明该请求在切换前完成）
            logger.debug(f"[Lifecycle-Fallback] 请求 {req_id} 不在 request_table，走兼容路径")
            resources = info.get('resources', {})
            placement = resources.get('placement', {})
            for key, alloc in placement.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    node, vnf_idx = key[0], key[1]
                    vnf_list = info['request'].get('vnf', [])
                    vnf_type = alloc.get(
                        'vnf_type',
                        vnf_list[vnf_idx] if vnf_idx < len(vnf_list) else vnf_idx
                    )
                    rm.release_vnf_ref(node, vnf_type)
                    logger.debug(f"[CPU-EXPIRE-Fallback] req={req_id} node={node} vnf_t={vnf_type}")

        # ── 更新时序元数据 ────────────────────────────────────────────
        self.stats['total_expired'] += 1
        info['status'] = 'expired'
        info['release_time'] = current_time
        info['actual_lifetime'] = current_time - info['arrival_time']
        self.expired_requests[req_id] = info
        del self.active_requests[req_id]
        logger.debug(f"[Lifecycle] 请求 {req_id} 已释放 (t={current_time:.1f})")

    # force_release() 已删除：
    # 强制释放接口。当前失败路径统一由 _archive_episode_fail() 回滚 CPU/MEM，
    # 不再需要外部强制触发 lifecycle 释放。

    def cleanup_all(self, current_time: float):
        """
        清理所有活跃请求（用于环境重置）
        current_time: 当前仿真时间（通常为重置时刻）
        """
        with self.lock:
            req_ids = list(self.active_requests.keys())
            for req_id in req_ids:
                self._release_request(req_id, current_time)
            logger.debug(f"[Lifecycle] 清理所有请求，共 {len(req_ids)} 个")

    # get_stats() 已删除：
    # 返回 stats 字典的工具接口。当前训练诊断直接通过 RESOURCE 日志读取
    # pool/lc 的 CPU/MEM/BW 对比，不依赖此接口。


# ==================== 请求处理器（辅助方法） ====================
class RequestHandler:
    """
    请求处理器 - 负责与请求相关的业务逻辑
    包括部署、归档、状态查询等辅助方法
    """

    def __init__(self, resource_mgr):
        self.rm = resource_mgr  # FusedResourceManager 实例

    def _try_deploy(self, node: int, allow_reuse: bool = False):
        """
        降级为薄封装：直接委托给 FusedResourceManager._try_deploy（新内核主链）。
        不再自行操作 shared_vnf_instances / pool / placement。
        """
        return self.rm._try_deploy(node, allow_reuse=allow_reuse)

    def _archive_request(self, success=False, already_rolled_back=False):
        """
        归档请求。
        成功路径：只更新统计计数；资源由 lifecycle 到期时通过 release_request_record 释放。
        失败路径：调用 release_request_record(req_id, rollback=True) 走新内核统一回滚链：
          - 清空 vnf_bindings / edge_allocations / connected_dests
          - 置请求状态为 FAILED
          - CPU/MEM 仅在 ref_count==0 时释放；BW delayed-commit 未扣无需回滚
        """
        if self.rm.current_request is None:
            return
        # ── 统一使用业务 req_id，不使用 id(对象) ──────────────────────
        req_id = self.rm.current_request.get('id')
        if req_id is None:
            logger.error("[_archive_request] current_request 缺少 'id'，跳过归档")
            return

        if success:
            self.rm.total_requests_accepted += 1
            if hasattr(self.rm, 'served_dest_count'):
                self.rm.served_dest_count += len(self.rm.current_request.get('dest', []))
            logger.debug(f'[Archive] 请求 {req_id} 成功归档')
        else:
            if not already_rolled_back:
                logger.debug(f'[Archive] 请求 {req_id} 失败 → release_request_record(rollback=True)')
                # ── 新内核统一回滚链（主路径）────────────────────────────
                released = self.rm.release_request_record(req_id, rollback=True)
                if not released:
                    # request_table 无记录时（过渡期兼容）走旧 placement 路径
                    logger.debug(f'[Archive-Fallback] req={req_id} 不在 request_table，走兼容路径')
                    placement = self.rm.current_tree.get('placement', {})
                    for key, alloc in placement.items():
                        if isinstance(key, tuple) and len(key) >= 2:
                            node, vnf_idx = key[0], key[1]
                            vnf_list = self.rm.current_request.get('vnf', [])
                            vnf_t = alloc.get('vnf_type',
                                              vnf_list[vnf_idx] if vnf_idx < len(vnf_list) else vnf_idx)
                            self.rm.release_vnf_ref(node, vnf_t)
                            logger.debug(f'[CPU-ROLLBACK-Fallback] req={req_id} node={node} vnf_t={vnf_t}')
            logger.debug(f'[Archive] 请求 {req_id} 失败归档')

    # complete_current_request() 已删除：
    # 旧版请求完成后手动清理接口。当前主链通过 episode_reset() 统一清理
    # current_tree / current_request / next_vnf_idx 等状态，不再需要此接口。

    # apply_deployment() 已删除：
    # 基于 hvt_branch 矩阵的批量 VNF 部署接口，属于旧版离线部署方案。
    # 当前主链通过 _try_deploy() 逐个 VNF 即时部署，不需要批量接口。

    # apply_tree_deployment() 已删除：
    # 旧版树部署接口，调用 apply_deployment + 批量扣 BW。
    # 当前主链采用 delayed-commit 模式，BW 由 _commit_episode_bandwidth() 提交。

    # get_network_state_dict() 已删除：
    # 导出全网 CPU/MEM/BW 快照的工具接口。当前状态构建由
    # HighLevelController.get_high_level_state_graph() 直接组装 PyG Data 对象，
    # LowLevelController.get_state() 输出 21 维特征向量，不依赖此接口。



# ==================== 融合版资源管理器（外观类） ====================
class FusedResourceManager:
    """
    融合版资源管理器 - 外观类
    组合核心组件，提供统一接口，保持与原有代码兼容
    """

    def __init__(self, topo: np.ndarray, capacities: Dict, dc_nodes: List[int], link_map: Optional[Dict] = None):
        self.topo = topo
        self.n = topo.shape[0]
        self.dc_nodes = dc_nodes
        self.link_map = link_map
        self.tools = {}  # 保留兼容性

        # 核心组件
        self.pool = SharedResourcePool(topo, capacities)
        # transaction_mgr 已删除：TransactionManager 事务链废弃，不再初始化
        self.request_manager = RequestLifecycleManager(self)
        self.handler = RequestHandler(self)  # 业务辅助方法

        # 兼容性字段
        self.C_cap = capacities.get('cpu', 100.0)
        self.M_cap = capacities.get('memory', 80.0)
        self.B_cap = capacities.get('bandwidth', 100.0)

        # VNF相关
        self.K_vnf = 8
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.vnf_instances = []
        # 共享VNF实例表：key=(node, vnf_type)
        # ref_count 统计当前被多少个 placement 引用；为 0 时才真正释放物理CPU/MEM
        self.shared_vnf_instances = {}
        self.vnf_share_cap = capacities.get('vnf_share_cap', None)

        # ===== 新内核：单一真相（RequestRecord + VNFInstanceRecord）=====
        # 与旧的 shared_vnf_instances / placement / lifecycle 并存，
        # 逐步替换，新逻辑优先读新内核，旧逻辑兼容视图保持同步
        self.request_table: Dict[int, RequestRecord] = {}
        self.instance_table: Dict[str, VNFInstanceRecord] = {}
        self.instance_index: Dict[Tuple[int, int], str] = {}  # (node, vnf_type) -> inst_id

        # 构建边索引 + 边特征
        self._build_edge_index()
        self._build_edge_attr()

        # [Reach] 预计算静态全源最短跳数（多跳可达性特征用，init 一次性）
        from core.gnn.reachability_features import precompute_all_pairs_hops
        self._hops_all = precompute_all_pairs_hops(self.topo)
        self._max_hop  = float(self._hops_all[self._hops_all < self.n + 1].max()) if self.n > 1 else 1.0

        # 状态维度（兼容GNN）
        self.dim_request = 10
        self.dim_network = self.n * 2 + self.pool.L + self.n * self.K_vnf
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request
        self.node_feat_dim = 6 + self.K_vnf + 3 + 3  # [SDG-HRL] +vnf_depth, +progress, +phase_flag → 20
        self.edge_feat_dim = 5
        self.request_dim = 24

        # 当前请求相关（由外部设置）
        self.current_request = None
        self.current_tree = {
            'tree': {},
            'placement': {},
            'connected_dests': set(),
            'new_instance_count': 0,
            'reused_vnf_count': 0,
            'node_stage': {},
        }
        self.current_phase = 'idle'
        self.next_vnf_idx = 0
        self.nodes_on_tree = set()
        self.total_requests_accepted = 0
        self.served_dest_count = 0

        logger.debug(f"[FusedRM] 初始化完成: {self.n}节点, {self.pool.L}链路 (外观版)")

    def _build_edge_index(self):
        rows, cols = np.where(self.topo > 0)
        self.edge_index = np.array([rows, cols], dtype=np.int64)
        self.edge_hops = np.array([float(self.topo[u, v]) for u, v in zip(rows, cols)], dtype=np.float32)
        self.edge_to_phys = {}
        self.phys_to_graph_edges = {}
        for idx, (u, v) in enumerate(zip(rows, cols)):
            eid = self.pool.get_edge_id(u, v)
            if eid is not None:
                self.edge_to_phys[(u, v)] = eid
                if eid not in self.phys_to_graph_edges:
                    self.phys_to_graph_edges[eid] = []
                self.phys_to_graph_edges[eid].append(idx)

    def _build_edge_attr(self):
        """
        构建静态边特征矩阵（初始化时调用一次）
        edge_attr: [E, 5]
          [0] bw_remaining     — 归一化可用带宽（动态，初始=1.0）
          [1] bw_utilization   — 带宽利用率（动态，初始=0.0）
          [2] hop_weight_norm  — 拓扑跳数权重归一化（静态）
          [3] is_tree_edge     — 是否被当前树占用（动态，初始=0.0）
          [4] reserved         — 预留带宽比例（动态，初始=0.0）
        """
        rows, cols = self.edge_index[0], self.edge_index[1]
        E = len(rows)
        max_bw  = max(1.0, max(self.pool.bw_cap.values()) if self.pool.bw_cap else 1.0)
        max_hop = max(1.0, float(self.edge_hops.max()) if len(self.edge_hops) > 0 else 1.0)

        attr = np.zeros((E, 5), dtype=np.float32)
        for idx, (u, v) in enumerate(zip(rows, cols)):
            cap   = self.pool.bw_cap.get((u, v), max_bw)
            avail = self.pool.bw_avail.get((u, v), cap)
            # 维度与 shared_encoder 期望一致:
            # [0] bw_remaining (avail/cap)
            # [1] bw_utilization (1 - avail/cap)
            # [2] hop_weight_norm
            # [3] is_tree_edge (动态更新，初始0)
            # [4] reserved (bw_reserved/cap)
            util = 1.0 - avail / max(1.0, cap)
            attr[idx, 0] = avail / max(1.0, cap)         # bw_remaining
            attr[idx, 1] = util                          # bw_utilization
            attr[idx, 2] = self.edge_hops[idx] / max_hop # hop_weight_norm
            attr[idx, 3] = 0.0                           # is_tree_edge (动态)
            attr[idx, 4] = 0.0                           # reserved (动态)
        self.edge_attr = attr
        self._edge_attr_max_bw  = max_bw
        self._edge_attr_max_hop = max_hop
        logger.debug(f"[FusedRM] edge_attr 构建完成: shape={attr.shape}")

    def build_dynamic_edge_attr(self):
        """
        [SDG-HRL] 动态刷新 edge_attr（每step调用，更新带宽占用和树使用情况）
        返回 torch.Tensor [E, 5]，供 get_state() 直接使用
        """
        rows, cols = self.edge_index[0], self.edge_index[1]
        E   = len(rows)
        attr = self.edge_attr.copy()  # 在静态基础上更新动态部分

        # 当前树占用的边集合
        tree_edges = set()
        if hasattr(self, 'current_tree') and self.current_tree:
            for (u, v), flow in self.current_tree.get('tree', {}).items():
                if flow > 0.0:
                    tree_edges.add((u, v))

        for idx, (u, v) in enumerate(zip(rows, cols)):
            cap   = self.pool.bw_cap.get((u, v), self._edge_attr_max_bw)
            avail = self.pool.get_available_bandwidth(u, v)
            util  = 1.0 - avail / max(1.0, cap)
            attr[idx, 0] = avail / max(1.0, cap)         # bw_remaining
            attr[idx, 1] = util                          # bw_utilization
            attr[idx, 3] = 1.0 if (u, v) in tree_edges else 0.0  # is_tree_edge
            # 🆕 补全 reserved 维度（之前始终为0，GNN看不到预留信息）
            reserved = self.pool.bw_reserved.get((u, v), 0.0)
            attr[idx, 4] = reserved / max(1.0, cap)      # reserved ratio

        return torch.from_numpy(attr).float()

    def get_reach_feats(self, remaining_dests):
        """[Reach] 返回 [n,4] 多跳可达性特征：到剩余目的集合的
        [min跳, mean跳, max瓶颈带宽, min瓶颈带宽]（均已归一化）。
        必须在【状态构建时】调用，结果写进 Data.x；
        切勿在 encoder 里现算——replay 时带宽快照已变。
        state / next_state 各用各自时刻的快照分别计算。"""
        from core.gnn.reachability_features import compute_widest_paths, node_to_destset_features
        rd_raw = [int(d) for d in remaining_dests]
        invalid = [d for d in rd_raw if d < 0 or d >= self.n]
        if invalid:
            logger.warning("[Reach] ignoring invalid destination ids: %s", invalid[:8])
        rd = [d for d in rd_raw if 0 <= d < self.n]
        if not rd:
            return np.zeros((self.n, 4), dtype=np.float32)
        if getattr(self, '_hops_all', None) is None or self._hops_all.shape != (self.n, self.n):
            from core.gnn.reachability_features import precompute_all_pairs_hops
            self._hops_all = precompute_all_pairs_hops(self.topo)
            finite_hops = self._hops_all[self._hops_all < self.n + 1]
            self._max_hop = float(finite_hops.max()) if finite_hops.size else 1.0
        avail = np.zeros((self.n, self.n), dtype=np.float32)
        for (u, v), b in self.pool.bw_avail.items():
            if 0 <= u < self.n and 0 <= v < self.n:
                avail[u, v] = max(0.0, b)
        widest = compute_widest_paths(avail)
        return node_to_destset_features(
            self._hops_all, widest, rd,
            max_hop=self._max_hop, bw_cap=self.B_cap)

    # ---------- 事务接口 ----------
    # ---- 事务链已删除（begin_transaction / reserve_node_resource / reserve_link_resource /
    #      commit_transaction / rollback_transaction）：
    #      这五个是 TransactionManager 的包装代理，随 TransactionManager 一起废弃。
    #      当前主链不使用事务式资源管理。 ----

    # ---------- 直接分配接口 ----------
    def allocate_node_resource(self, node: int, vnf_type: int,
                               cpu_need: float, mem_need: float = 0.0) -> bool:
        if node < 0 or node >= self.n:
            return False
        if not self.pool.allocate_cpu(node, cpu_need):
            return False
        if mem_need > 0 and not self.pool.allocate_memory(node, mem_need):
            self.pool.release_cpu(node, cpu_need)
            return False
        if 0 <= vnf_type < self.K_vnf:
            self.hvt_all[node, vnf_type] += 1
        return True

    def allocate_link_resource(self, u: int, v: int, bw_need: float) -> bool:
        return self.pool.allocate_bandwidth(u, v, bw_need)

    def allocate_bandwidth(self, u: int, v: int, bw: float) -> bool:
        return self.allocate_link_resource(u, v, bw)

    def release_node_resource(self, node: int, vnf_type: int, cpu_val: float, mem_val: float):
        if node < 0 or node >= self.n:
            return
        if 0 <= vnf_type < self.K_vnf and self.hvt_all[node, vnf_type] > 0:
            self.hvt_all[node, vnf_type] = max(0, self.hvt_all[node, vnf_type] - 1)
        if cpu_val > 0:
            self.pool.release_cpu(node, cpu_val)
        if mem_val > 0:
            self.pool.release_memory(node, mem_val)

    # ===================================================================
    # 新内核方法（单一真相）
    # ===================================================================

    def _make_inst_id(self, node: int, vnf_type: int) -> str:
        return f"{node}:{vnf_type}"

    def _sync_legacy_views(self) -> None:
        """把 instance_table ACTIVE 实例同步回 shared_vnf_instances（兼容旧代码读取）"""
        self.shared_vnf_instances = {
            (inst.node, inst.vnf_type): {
                'ref_count': inst.ref_count,
                'cpu_used': inst.cpu,
                'mem_used': inst.mem,
            }
            for inst in self.instance_table.values()
            if inst.state == "ACTIVE"
        }

    def commit_edge_bandwidth(self, req_id, u: int, v: int, bw: float) -> bool:
        """
        delayed-commit BW 提交（原子操作）：
        1. allocate_bandwidth — 扣减物理 BW
        2. EdgeAllocation(req_id, u, v, bw) 追加到 request_table[req_id].edge_allocations
        edge_allocations 是 BW 释放的唯一真相；release_request_record 通过它单路径还 BW。
        由 LowLevelController._commit_episode_bandwidth() 调用。
        """
        if not self.allocate_bandwidth(u, v, bw):
            logger.warning(f"[commit_edge_bw] 扣BW失败 ({u},{v}) bw={bw:.1f}")
            return False

        # 主路径：直接查 request_table
        req = self.request_table.get(req_id)
        if req is not None:
            if req.state not in {"PENDING", "ACTIVE"}:
                logger.warning(
                    "[commit_edge_bw] req_id=%s state=%s; rolling back (%s,%s)",
                    req_id, req.state, u, v,
                )
                self.pool.release_bandwidth(u, v, bw)
                return False
            req.edge_allocations.append(EdgeAllocation(req_id=req.req_id, u=u, v=v, bw=bw))
            return True

        # 备路径：str ↔ int 转换后再查一次
        try:
            alt = int(req_id) if isinstance(req_id, str) else str(req_id)
            req = self.request_table.get(alt)
            if req is not None:
                if req.state not in {"PENDING", "ACTIVE"}:
                    logger.warning(
                        "[commit_edge_bw] req_id=%s state=%s; rolling back (%s,%s)",
                        alt, req.state, u, v,
                    )
                    self.pool.release_bandwidth(u, v, bw)
                    return False
                req.edge_allocations.append(EdgeAllocation(req_id=req.req_id, u=u, v=v, bw=bw))
                return True
        except (ValueError, TypeError):
            pass

        # BW 已扣但 request_table 无记录 — 立即归还防止永久泄漏
        # 原来 return True 会导致 BW 永久锁死（无 edge_allocations 记录，无法通过
        # release_request_record 归还），是 BW=21% 却疯狂失败的根本原因之一
        logger.warning(
            f"[commit_edge_bw] ⚠️ req_id={req_id} 不在 request_table，"
            f"BW已扣 ({u},{v}) bw={bw:.1f}，立即归还防止泄漏"
        )
        self.pool.release_bandwidth(u, v, bw)
        return False

    def register_request_record(self, req_id: int, source: int = 0,
                                dests: Optional[List[int]] = None,
                                vnfs: Optional[List[int]] = None,
                                bw: float = 0.0) -> bool:
        """注册请求到新内核（不分配任何资源）"""
        existing = self.request_table.get(req_id)
        if existing is not None and existing.state not in {'FAILED', 'RELEASED'}:
            return False
        self.request_table[req_id] = RequestRecord(
            req_id=req_id,
            source=source,
            dests=list(dests or []),
            vnfs=list(vnfs or []),
            bw=float(bw),
        )
        return True

    def bind_vnf_to_request(self, req_id: int, node: int, vnf_type: int,
                            deploy_res: DeployResult,
                            req_cpu: float, req_mem: float) -> None:
        """VNF 部署成功后把绑定关系写入 RequestRecord"""
        req = self.request_table.get(req_id)
        if req is None or not deploy_res.ok or deploy_res.inst_id is None:
            return
        req.vnf_bindings.append(VNFBinding(
            req_id=req_id, node=node, vnf_type=vnf_type,
            inst_id=deploy_res.inst_id,
            reused=deploy_res.reused,
            cpu=req_cpu, mem=req_mem,
        ))

    def mark_dest_connected(self, req_id: int, target: int) -> None:
        """验证通过后才调用，标记目的地已连接"""
        req = self.request_table.get(req_id)
        if req is not None:
            req.connected_dests.add(target)

    def canonicalize_request_sft(self, req_id, current_tree: Optional[dict] = None,
                                 allow_topology_repair: bool = True) -> Dict[str, Any]:
        """Build a rooted directed tree from all recorded deployment edges.

        HRL records traversed connector edges with flow values 0 and -1. Those
        edges are structural evidence even though they were previously omitted
        from bandwidth commit. This routine roots the complete recorded graph at
        the request source, prunes non-critical leaves, and can attach missing
        critical nodes through residual-bandwidth-feasible topology paths.
        """
        req = self.request_table.get(req_id)
        if req is None:
            try:
                alt_key = int(req_id) if isinstance(req_id, str) else str(req_id)
                req = self.request_table.get(alt_key)
            except (ValueError, TypeError):
                req = None
        if req is None:
            return {'ok': False, 'reason': 'missing_request'}

        tree = current_tree or {}
        explicit_spine_paths = list(tree.get('spine_paths', []) or [])
        explicit_branch_paths = dict(tree.get('branch_paths', {}) or {})

        def _edge_key(edge):
            if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                return int(edge[0]), int(edge[1])
            raise ValueError(f"invalid edge key: {edge!r}")

        recorded_edges: Set[Tuple[int, int]] = set()
        recorded_adj: Dict[int, Set[int]] = {}
        for edge in (tree.get('tree', {}) or {}):
            try:
                u, v = _edge_key(edge)
            except Exception:
                continue
            if u == v:
                continue
            recorded_edges.add((min(u, v), max(u, v)))
            recorded_adj.setdefault(u, set()).add(v)
            recorded_adj.setdefault(v, set()).add(u)

        placement_by_stage: Dict[int, int] = {}
        for key, value in (tree.get('placement', {}) or {}).items():
            try:
                if isinstance(key, (tuple, list)) and len(key) >= 2:
                    node = int(key[0])
                    stage = int(key[1])
                else:
                    node = int(value.get('node'))
                    stage = int(value.get('vnf_idx', len(placement_by_stage)))
                placement_by_stage[stage] = node
            except Exception:
                continue

        source = int(req.source)
        expected_stages = set(range(len(req.vnfs)))
        if set(placement_by_stage) != expected_stages:
            return {
                'ok': False,
                'reason': 'missing_vnf_stage',
                'expected_stages': sorted(expected_stages),
                'actual_stages': sorted(placement_by_stage),
            }
        ordered_placements = [
            int(placement_by_stage[stage]) for stage in range(len(req.vnfs))
        ]
        critical_nodes = {source}
        critical_nodes.update(int(node) for node in req.dests)
        critical_nodes.update(int(node) for node in req.connected_dests)
        critical_nodes.update(ordered_placements)

        required_bw = max(0.0, float(req.bw))
        allocated_edges = {
            (int(allocation.u), int(allocation.v))
            for allocation in req.edge_allocations
        }

        repaired_edges: Set[Tuple[int, int]] = set()
        parent: Dict[int, Optional[int]] = {source: None}

        def _edge_is_feasible(u: int, v: int) -> bool:
            if (u, v) in allocated_edges:
                return True
            try:
                available = float(self.pool.get_available_bandwidth(u, v))
            except Exception:
                return False
            return available + 1e-5 >= required_bw

        def _find_path(roots: Set[int], target: int, blocked: Set[int],
                       recorded_only: bool) -> Optional[List[int]]:
            target = int(target)
            roots = {int(node) for node in roots}
            if target in roots:
                return [target]
            predecessor: Dict[int, Optional[int]] = {
                node: None for node in sorted(roots)
            }
            frontier = deque(sorted(roots))
            while frontier:
                node = frontier.popleft()
                neighbors = (
                    recorded_adj.get(node, ())
                    if recorded_only else self.get_neighbors(node)
                )
                for neighbor in sorted(int(item) for item in neighbors):
                    if neighbor in predecessor:
                        continue
                    if neighbor in blocked and neighbor != target:
                        continue
                    if not _edge_is_feasible(node, neighbor):
                        continue
                    predecessor[neighbor] = node
                    if neighbor == target:
                        path = [target]
                        while predecessor[path[-1]] is not None:
                            path.append(int(predecessor[path[-1]]))
                        path.reverse()
                        return path
                    frontier.append(neighbor)
            return None

        def _ordered_extension(roots: Set[int], target: int,
                               blocked: Set[int]) -> Optional[List[int]]:
            path = _find_path(roots, target, blocked, recorded_only=True)
            if path is not None or not allow_topology_repair:
                return path
            return _find_path(roots, target, blocked, recorded_only=False)

        def _validated_explicit_path(candidate, roots: Set[int], target: int,
                                     blocked: Set[int]) -> Optional[List[int]]:
            try:
                path = [int(node) for node in candidate]
            except Exception:
                return None
            if not path or path[0] not in roots or path[-1] != int(target):
                return None
            if len(path) != len(set(path)):
                return None
            if any(node in blocked for node in path[1:-1]):
                return None
            for u, v in zip(path, path[1:]):
                if v not in recorded_adj.get(u, ()):
                    return None
                if not _edge_is_feasible(u, v):
                    return None
            return path

        # Build source -> VNF0 -> ... -> last VNF without entering a
        # destination or re-entering an earlier spine node.
        previous_stage = source
        destinations = {int(node) for node in req.dests}
        for stage, placement in enumerate(ordered_placements):
            blocked = set(parent).difference({previous_stage, placement})
            blocked.update(destinations.difference({placement}))
            explicit = (
                explicit_spine_paths[stage]
                if stage < len(explicit_spine_paths) else None
            )
            path = _validated_explicit_path(
                explicit, {previous_stage}, placement, blocked
            ) if explicit is not None else None
            if path is None:
                path = _ordered_extension({previous_stage}, placement, blocked)
            if path is None:
                return {
                    'ok': False,
                    'reason': 'vnf_stage_order_unreachable',
                    'stage': stage,
                    'start': previous_stage,
                    'target': placement,
                }
            for u, v in zip(path, path[1:]):
                if v in parent and parent[v] != u:
                    return {
                        'ok': False,
                        'reason': 'vnf_stage_second_parent',
                        'stage': stage,
                        'node': v,
                    }
                parent[v] = u
                if (min(u, v), max(u, v)) not in recorded_edges:
                    repaired_edges.add((u, v))
            previous_stage = placement

        # All destination branches start at the last VNF or a node already
        # downstream of it. They cannot attach to the pre-chain spine.
        branch_root = ordered_placements[-1] if ordered_placements else source
        pre_chain_nodes = set(parent).difference({branch_root})
        post_chain_nodes = {branch_root}
        for destination in sorted(destinations):
            if destination in post_chain_nodes:
                continue
            explicit = explicit_branch_paths.get(
                destination, explicit_branch_paths.get(str(destination))
            )
            path = _validated_explicit_path(
                explicit, set(post_chain_nodes), destination,
                set(pre_chain_nodes)
            ) if explicit is not None else None
            if path is None:
                path = _ordered_extension(
                    set(post_chain_nodes), destination, set(pre_chain_nodes)
                )
            if path is None:
                explicit_checks = None
                if explicit is not None:
                    try:
                        explicit_nodes = [int(node) for node in explicit]
                        explicit_checks = {
                            'start_ok': bool(
                                explicit_nodes and explicit_nodes[0] in post_chain_nodes
                            ),
                            'target_ok': bool(
                                explicit_nodes and explicit_nodes[-1] == destination
                            ),
                            'unique': len(explicit_nodes) == len(set(explicit_nodes)),
                            'blocked_internal': sorted(
                                set(explicit_nodes[1:-1]).intersection(pre_chain_nodes)
                            ),
                            'edges': [
                                {
                                    'edge': (u, v),
                                    'recorded': v in recorded_adj.get(u, ()),
                                    'allocated': (u, v) in allocated_edges,
                                    'available': float(
                                        self.pool.get_available_bandwidth(u, v)
                                    ),
                                    'required': required_bw,
                                    'feasible': _edge_is_feasible(u, v),
                                }
                                for u, v in zip(
                                    explicit_nodes, explicit_nodes[1:]
                                )
                            ],
                        }
                    except Exception as exc:
                        explicit_checks = {'error': str(exc)}
                return {
                    'ok': False,
                    'reason': 'destination_after_chain_unreachable',
                    'branch_root': branch_root,
                    'destination': destination,
                    'explicit_branch': explicit,
                    'explicit_spine_paths': explicit_spine_paths,
                    'post_chain_nodes': sorted(post_chain_nodes),
                    'pre_chain_nodes': sorted(pre_chain_nodes),
                    'recorded_edges': sorted(recorded_edges),
                    'explicit_checks': explicit_checks,
                }
            for u, v in zip(path, path[1:]):
                if v in parent:
                    continue
                parent[v] = u
                post_chain_nodes.add(v)
                if (min(u, v), max(u, v)) not in recorded_edges:
                    repaired_edges.add((u, v))

        unreachable = sorted(critical_nodes.difference(parent))
        if unreachable:
            return {
                'ok': False,
                'reason': 'critical_nodes_unreachable',
                'unreachable_nodes': unreachable,
                'critical_nodes': sorted(critical_nodes),
            }

        canonical_edges = {
            (int(parent[node]), int(node)): 1.0
            for node in parent
            if node != source
        }
        if len(canonical_edges) != max(0, len(parent) - 1):
            return {'ok': False, 'reason': 'canonical_tree_invariant_failed'}

        return {
            'ok': True,
            'reason': 'ok',
            'tree_edges': canonical_edges,
            'critical_nodes': sorted(critical_nodes),
            'repaired_edges': sorted(repaired_edges),
            'used_topology_repair': bool(repaired_edges),
        }

    def snapshot_request_sft(self, req_id, current_tree: Optional[dict] = None,
                             snapshot_time: Optional[float] = None,
                             already_canonical: bool = False) -> bool:
        """
        Persist the logical SFT built for a successful request.

        The resource ledger remains edge_allocations/vnf_bindings. This method
        stores the structural view needed by runtime reconfiguration: active tree
        edges, VNF-index placement, node stage, and tree-edge usage counters.
        """
        req = self.request_table.get(req_id)
        if req is None:
            try:
                alt_key = int(req_id) if isinstance(req_id, str) else str(req_id)
                req = self.request_table.get(alt_key)
                if req is not None:
                    req_id = alt_key
            except (ValueError, TypeError):
                pass
        if req is None:
            logger.warning(f"[SFT-Snapshot] req={req_id} not found in request_table")
            return False

        tree = current_tree or {}
        if already_canonical:
            canonical_edges = {}
            for edge, flow in (tree.get('tree', {}) or {}).items():
                try:
                    u, v = int(edge[0]), int(edge[1])
                    if u != v:
                        canonical_edges[(u, v)] = float(flow)
                except Exception:
                    continue
            canonical = {
                'ok': bool(canonical_edges),
                'reason': 'ok' if canonical_edges else 'missing_canonical_tree',
                'tree_edges': canonical_edges,
            }
        else:
            canonical = self.canonicalize_request_sft(
                req_id, tree, allow_topology_repair=not bool(req.edge_allocations)
            )
        if not canonical.get('ok'):
            logger.warning(
                f"[SFT-Snapshot] req={req_id} canonicalization failed: "
                f"{canonical.get('reason')}"
            )
            return False

        def _edge_key(edge):
            if isinstance(edge, tuple) and len(edge) >= 2:
                return int(edge[0]), int(edge[1])
            if isinstance(edge, list) and len(edge) >= 2:
                return int(edge[0]), int(edge[1])
            raise ValueError(f"invalid edge key: {edge!r}")

        tree_edges = dict(canonical['tree_edges'])

        placement_by_vnf = {}
        placement_detail = {}
        for key, value in (tree.get('placement', {}) or {}).items():
            try:
                if isinstance(key, tuple) and len(key) >= 2:
                    node = int(key[0])
                    vnf_idx = int(key[1])
                else:
                    node = int(value.get('node'))
                    vnf_idx = int(value.get('vnf_idx', len(placement_by_vnf)))
                placement_by_vnf[vnf_idx] = node
                placement_detail[vnf_idx] = {
                    'node': node,
                    'vnf_type': int(value.get('vnf_type', -1)),
                    'cpu_used': float(value.get('cpu_used', 0.0)),
                    'mem_used': float(value.get('mem_used', 0.0)),
                    'reused': bool(value.get('reused', False)),
                    'inst_id': value.get('inst_id'),
                }
            except Exception:
                continue

        node_stage = {}
        for node, stage in (tree.get('node_stage', {}) or {}).items():
            try:
                node_stage[int(node)] = int(stage)
            except Exception:
                continue

        tree_usage = {}
        for edge, count in (tree.get('tree_usage', {}) or {}).items():
            try:
                tree_usage[_edge_key(edge)] = int(count)
            except Exception:
                continue

        previous = (
            req.tree_edges, req.placement_by_vnf, req.placement_detail,
            req.node_stage, req.tree_usage, req.snapshot_time, req.state,
        )
        req.tree_edges = tree_edges
        req.placement_by_vnf = placement_by_vnf
        req.placement_detail = placement_detail
        req.node_stage = node_stage
        req.tree_usage = {
            edge: max(1, tree_usage.get(edge, tree_usage.get((edge[1], edge[0]), 1)))
            for edge in tree_edges
        }
        req.snapshot_time = snapshot_time

        report = self.validate_request_sft_snapshot(req_id)
        if not report['ok']:
            (
                req.tree_edges, req.placement_by_vnf, req.placement_detail,
                req.node_stage, req.tree_usage, req.snapshot_time, req.state,
            ) = previous
            logger.warning(
                f"[SFT-Snapshot] req={req_id} validation failed: {report['reason']}"
                f" snapshot_only={report.get('snapshot_only_edges', [])}"
                f" allocation_only={report.get('allocation_only_edges', [])}"
                f" snapshot_edges={report.get('snapshot_edges', [])}"
                f" ledger_edges={report.get('allocation_edges', [])}"
            )
            return False
        req.state = "ACTIVE"

        logger.debug(
            f"[SFT-Snapshot] req={req_id} edges={len(tree_edges)} "
            f"vnfs={len(placement_by_vnf)} dests={len(req.connected_dests)}"
        )
        return True

    def validate_request_sft_snapshot(self, req_id) -> Dict[str, Any]:
        """Return a compact structural validation report for one SFT snapshot."""
        req = self.request_table.get(req_id)
        if req is None:
            try:
                alt_key = int(req_id) if isinstance(req_id, str) else str(req_id)
                req = self.request_table.get(alt_key)
            except (ValueError, TypeError):
                req = None
        if req is None:
            return {'ok': False, 'reason': 'missing_request'}

        missing = []
        if not req.tree_edges:
            missing.append('tree_edges')
        if not req.placement_by_vnf and req.vnfs:
            missing.append('placement_by_vnf')
        expected_dests = {int(node) for node in req.dests}
        connected_dests = {int(node) for node in req.connected_dests}
        if connected_dests != expected_dests:
            missing.append('connected_dest_mismatch')
        if not req.vnf_bindings and req.vnfs:
            missing.append('vnf_bindings')
        if not req.edge_allocations and req.tree_edges:
            missing.append('edge_allocations')

        structural_errors = []
        source = int(req.source)
        outgoing: Dict[int, Set[int]] = {}
        indegree: Dict[int, int] = {source: 0}
        nodes = {source}
        for edge in req.tree_edges:
            try:
                u, v = int(edge[0]), int(edge[1])
            except Exception:
                structural_errors.append('invalid_edge')
                continue
            if u == v:
                structural_errors.append('self_loop')
                continue
            nodes.update((u, v))
            outgoing.setdefault(u, set()).add(v)
            indegree[v] = indegree.get(v, 0) + 1
            indegree.setdefault(u, indegree.get(u, 0))

        if indegree.get(source, 0) != 0:
            structural_errors.append('root_has_parent')
        if any(indegree.get(node, 0) != 1 for node in nodes if node != source):
            structural_errors.append('invalid_parent_count')

        reachable = {source}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in outgoing.get(node, ()):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        if reachable != nodes:
            structural_errors.append('not_root_reachable')
        if len(req.tree_edges) != max(0, len(nodes) - 1):
            structural_errors.append('not_a_tree')

        critical_nodes = {source}
        critical_nodes.update(int(node) for node in req.connected_dests)
        critical_nodes.update(int(node) for node in req.dests)
        critical_nodes.update(int(node) for node in req.placement_by_vnf.values())
        unreachable_critical = sorted(critical_nodes.difference(reachable))
        if unreachable_critical:
            structural_errors.append('critical_nodes_unreachable')

        expected_stages = set(range(len(req.vnfs)))
        actual_stages = {int(stage) for stage in req.placement_by_vnf}
        if actual_stages != expected_stages:
            structural_errors.append('missing_vnf_stage')

        placement_binding_mismatch = False
        if len(req.vnf_bindings) != len(req.vnfs):
            placement_binding_mismatch = bool(req.vnfs)
        elif actual_stages == expected_stages:
            for stage, binding in enumerate(req.vnf_bindings):
                placement_node = int(req.placement_by_vnf[stage])
                expected_vnf_type = int(req.vnfs[stage])
                detail = req.placement_detail.get(stage, {})
                if (
                    int(binding.node) != placement_node
                    or int(binding.vnf_type) != expected_vnf_type
                    or (
                        detail
                        and (
                            int(detail.get('node', placement_node)) != placement_node
                            or int(detail.get('vnf_type', expected_vnf_type))
                            != expected_vnf_type
                            or detail.get('inst_id') != binding.inst_id
                        )
                    )
                ):
                    placement_binding_mismatch = True
                    break
        if placement_binding_mismatch:
            structural_errors.append('placement_binding_mismatch')

        def _directed_reachable(start: int, target: int) -> bool:
            start, target = int(start), int(target)
            if start == target:
                return True
            seen = {start}
            frontier = deque([start])
            while frontier:
                node = frontier.popleft()
                for neighbor in outgoing.get(node, ()):
                    if neighbor == target:
                        return True
                    if neighbor not in seen:
                        seen.add(neighbor)
                        frontier.append(neighbor)
            return False

        unreachable_chain_segments = []
        unreachable_destinations_after_chain = []
        if actual_stages == expected_stages:
            placements = [
                int(req.placement_by_vnf[stage]) for stage in range(len(req.vnfs))
            ]
            chain_terminals = [source, *placements]
            unreachable_chain_segments = [
                (start, target)
                for start, target in zip(chain_terminals, chain_terminals[1:])
                if not _directed_reachable(start, target)
            ]
            if unreachable_chain_segments:
                structural_errors.append('vnf_stage_order_violation')

            branch_root = placements[-1] if placements else source
            unreachable_destinations_after_chain = sorted(
                destination
                for destination in expected_dests
                if not _directed_reachable(branch_root, destination)
            )
            if unreachable_destinations_after_chain:
                structural_errors.append('destination_before_last_vnf')

        allocation_edges = [(int(ea.u), int(ea.v)) for ea in req.edge_allocations]
        snapshot_edge_set = {
            (int(edge[0]), int(edge[1])) for edge in req.tree_edges
        }
        allocation_edge_set = set(allocation_edges)
        ledger_match = (
            len(allocation_edges) == len(req.tree_edges)
            and allocation_edge_set == snapshot_edge_set
            and all(abs(float(ea.bw) - float(req.bw)) <= 1e-5 for ea in req.edge_allocations)
        )
        if req.tree_edges and not ledger_match:
            structural_errors.append('tree_ledger_mismatch')

        if structural_errors:
            missing.extend(error for error in structural_errors if error not in missing)

        return {
            'ok': len(missing) == 0,
            'reason': 'ok' if not missing else ','.join(missing),
            'req_id': req.req_id,
            'state': req.state,
            'tree_edges': len(req.tree_edges),
            'placement_by_vnf': len(req.placement_by_vnf),
            'connected_dests': len(req.connected_dests),
            'total_dests': len(req.dests),
            'vnf_bindings': len(req.vnf_bindings),
            'edge_allocations': len(req.edge_allocations),
            'root_reachable_nodes': len(reachable),
            'critical_nodes': len(critical_nodes),
            'unreachable_critical_nodes': unreachable_critical,
            'ledger_match': ledger_match,
            'snapshot_edges': sorted(snapshot_edge_set),
            'allocation_edges': sorted(allocation_edges),
            'snapshot_only_edges': sorted(snapshot_edge_set - allocation_edge_set),
            'allocation_only_edges': sorted(allocation_edge_set - snapshot_edge_set),
            'ordered_sfc': not (
                unreachable_chain_segments or unreachable_destinations_after_chain
            ),
            'unreachable_chain_segments': unreachable_chain_segments,
            'unreachable_destinations_after_chain': (
                unreachable_destinations_after_chain
            ),
            'structural_errors': structural_errors,
            'migration_count': req.migration_count,
            'reconfig_count': req.reconfig_count,
        }

    def try_deploy_new_kernel(self, node: int, vnf_type: int,
                              req_cpu: float, req_mem: float,
                              allow_reuse: bool = False,
                              req_id: Optional[int] = None) -> DeployResult:
        """
        新内核部署逻辑（单一真相版）：
        - 复用判断直接基于 instance_table（不再路由到 shared_vnf_instances）
        - allow_reuse=True 且活跃实例未超共享上限 → 复用
        - allow_reuse=False 且已有活跃实例 → FAIL（多实例并存未实现）
        - 无活跃实例 → 新建
        成功新建必须返回有效 inst_id（None 为错误状态，会记录 error log）。
        """
        key = (node, vnf_type)
        inst_id = self.instance_index.get(key)

        # 1) 已有活跃实例 ────────────────────────────────────────────────
        if inst_id is not None:
            inst = self.instance_table.get(inst_id)
            if inst is not None and inst.state == "ACTIVE" and inst.ref_count > 0:
                # ── 复用判断直接读 instance_table（不经 shared_vnf_instances）──
                cap_ok = (self.vnf_share_cap is None or inst.ref_count < self.vnf_share_cap)
                if allow_reuse and cap_ok:
                    inst.ref_count += 1
                    if req_id is not None:
                        inst.active_req_ids.add(req_id)
                    self._sync_legacy_views()
                    logger.debug(f"[VNF-Reuse] 节点={node} vnf={vnf_type} ref={inst.ref_count}")
                    return DeployResult(
                        ok=True, reused=True, new_instance=False,
                        inst_id=inst_id,
                        reason="reused_existing_instance",
                        cpu_used=0.0, mem_used=0.0
                    )
                elif allow_reuse and not cap_ok:
                    logger.warning(
                        f"[VNF-Reject] 节点={node} vnf={vnf_type} "
                        f"ref={inst.ref_count} 已达共享上限 {self.vnf_share_cap}"
                    )
                    return DeployResult(
                        ok=False, reused=False, new_instance=False,
                        inst_id=inst_id, reason="reuse_cap_exceeded"
                    )
                else:
                    # allow_reuse=False：不允许多实例并存
                    logger.warning(
                        f"[VNF-Reject] 节点={node} vnf={vnf_type} "
                        f"已有活跃实例(ref={inst.ref_count})，allow_reuse=False，拒绝部署"
                    )
                    return DeployResult(
                        ok=False, reused=False, new_instance=False,
                        inst_id=inst_id, reason="active_instance_exists_reuse_not_allowed"
                    )

        # 2) 无活跃实例：新建 ────────────────────────────────────────────
        if (self.pool.get_available_cpu(node) < req_cpu - 1e-5 or
                self.pool.get_available_memory(node) < req_mem - 1e-5):
            logger.warning(
                f"[VNF-Fail] 节点{node}资源不足 "
                f"CPU={self.pool.get_available_cpu(node):.1f}<{req_cpu:.1f} "
                f"MEM={self.pool.get_available_memory(node):.1f}<{req_mem:.1f}"
            )
            return DeployResult(ok=False, reason="resource_insufficient")

        if not self.allocate_node_resource(node, vnf_type, req_cpu, req_mem):
            return DeployResult(ok=False, reason="allocate_failed")

        new_inst_id = self._make_inst_id(node, vnf_type)
        new_inst = VNFInstanceRecord(
            inst_id=new_inst_id, node=node, vnf_type=vnf_type,
            cpu=req_cpu, mem=req_mem,
            ref_count=1,
            active_req_ids={req_id} if req_id is not None else set(),
            state="ACTIVE",
        )
        self.instance_table[new_inst_id] = new_inst
        self.instance_index[key] = new_inst_id
        self._sync_legacy_views()

        logger.debug(f"[VNF-New] 节点={node} vnf={vnf_type} cpu={req_cpu:.1f} mem={req_mem:.1f}")
        result = DeployResult(
            ok=True, reused=False, new_instance=True,
            inst_id=new_inst_id,
            reason="new_instance_created",
            cpu_used=req_cpu, mem_used=req_mem
        )
        # 防御性检查：新建成功后 inst_id 绝不应为 None
        if result.inst_id is None:
            logger.error(
                f"[try_deploy_new_kernel] ❌ 新建成功但 inst_id=None "
                f"node={node} vnf={vnf_type} — 这是内部错误"
            )
        return result

    def release_request_record(self, req_id, rollback: bool = False) -> bool:
        """
        新内核统一释放路径（正常到期 / 失败回滚共用）。
        - BW：仅通过 edge_allocations 释放，不走旧 resources['tree']
        - CPU/MEM：通过 _release_vnf_binding → release_vnf_ref，ref_count=0 时才释放
        - 释放后将 connected_dests / vnf_bindings / edge_allocations 清空
        - 请求状态置为 RELEASED 或 FAILED
        req_id 支持 str / int 两种形式（统一转 str 查找）
        """
        # 支持 str/int 混用（lifecycle 传 str，_try_deploy 传 current_request['id']）
        req = self.request_table.get(req_id)
        if req is None:
            # 尝试 str↔int 转换后再查一次
            try:
                alt_key = int(req_id) if isinstance(req_id, str) else str(req_id)
                req = self.request_table.get(alt_key)
                if req is not None:
                    req_id = alt_key
            except (ValueError, TypeError):
                pass
        if req is None:
            logger.debug(f"[release_request_record] req_id={req_id} 不在 request_table，跳过")
            return False

        bw_released = 0.0
        cpu_released = 0.0

        # ── 1) 带宽：仅通过 edge_allocations（唯一 BW 真相）──────────
        for ea in req.edge_allocations:
            self.pool.release_bandwidth(ea.u, ea.v, ea.bw)
            bw_released += ea.bw
        req.edge_allocations.clear()

        # ── 2) VNF 绑定：逆序释放，ref_count→0 时才还 CPU/MEM ────────
        for vb in reversed(req.vnf_bindings):
            self._release_vnf_binding(vb)
        req.vnf_bindings.clear()

        # ── 3) 清理目的地记录 ─────────────────────────────────────────
        req.connected_dests.clear()

        # 失败事务不能保留一个看似可迁移/可重配置的 SFT 结构快照。
        # 正常生命周期到期仍保留快照，便于离线审计历史部署。
        if rollback:
            req.tree_edges.clear()
            req.placement_by_vnf.clear()
            req.placement_detail.clear()
            req.node_stage.clear()
            req.tree_usage.clear()
            req.snapshot_time = None

        # ── 4) 更新状态并同步兼容视图 ────────────────────────────────
        req.state = "FAILED" if rollback else "RELEASED"
        self._sync_legacy_views()

        logger.debug(
            f"[release_request_record] req={req_id} "
            f"rollback={int(rollback)} bw_rel={bw_released:.1f}"
        )
        return True

    def _release_vnf_binding(self, vb: VNFBinding) -> None:
        """释放单个 VNF 绑定（新内核内部使用）"""
        if vb.inst_id is None:
            # cutover 后不应再出现 inst_id=None 的 binding（alongside 分支已禁用）
            # 保留为兜底，同时记录警告
            logger.warning(
                f"[_release_vnf_binding] inst_id=None (node={vb.node} vnf={vb.vnf_type})"
                f" — alongside分支已禁用，此路径不应出现，直接还资源"
            )
            self.release_node_resource(vb.node, vb.vnf_type, vb.cpu, vb.mem)
            return
        inst = self.instance_table.get(vb.inst_id)
        if inst is None or inst.state != "ACTIVE":
            # 实例已不在表中（可能已由其他绑定提前释放）
            if not vb.reused:
                self.release_node_resource(vb.node, vb.vnf_type, vb.cpu, vb.mem)
            return
        inst.active_req_ids.discard(vb.req_id)
        if inst.ref_count > 0:
            inst.ref_count -= 1
        if inst.ref_count == 0:
            self.release_node_resource(vb.node, vb.vnf_type, inst.cpu, inst.mem)
            inst.state = "DELETED"
            self.instance_table.pop(vb.inst_id, None)
            self.instance_index.pop((vb.node, vb.vnf_type), None)

    def audit_state(self) -> Dict[str, Any]:
        """诊断：输出新内核的当前状态"""
        active_reqs = [r.req_id for r in self.request_table.values()
                       if r.state in {"PENDING", "ACTIVE"}]
        active_insts = [(i.node, i.vnf_type, i.ref_count)
                        for i in self.instance_table.values()
                        if i.state == "ACTIVE"]
        return {
            "active_requests": active_reqs,
            "active_instances": active_insts,
            "request_count": len(self.request_table),
            "instance_count": len(active_insts),
        }

    # ===================================================================
    # 以下为原有方法（兼容旧代码）
    # ===================================================================

    def can_reuse_vnf_instance(self, node: int, vnf_type: int) -> bool:
        """
        复用判断：优先查 instance_table（新内核真相），
        fallback 到 shared_vnf_instances（兼容视图，仅在 instance_index 找不到时使用）。
        """
        inst_id = self.instance_index.get((node, vnf_type))
        if inst_id is not None:
            inst = self.instance_table.get(inst_id)
            if inst is None or inst.state != "ACTIVE" or inst.ref_count <= 0:
                return False
            if self.vnf_share_cap is not None and inst.ref_count >= self.vnf_share_cap:
                return False
            return True
        # fallback：instance_index 无记录时查兼容视图
        inst_dict = self.shared_vnf_instances.get((node, vnf_type))
        if not inst_dict or inst_dict.get('ref_count', 0) <= 0:
            return False
        if self.vnf_share_cap is not None and inst_dict.get('ref_count', 0) >= self.vnf_share_cap:
            return False
        return True

    def try_reuse_or_deploy(self, node: int, vnf_type: int,
                            cpu_need: float, mem_need: float = 0.0) -> Dict[str, Any]:
        """
        最小VNF复用接口（新内核委托版）。
        不再直接操作 shared_vnf_instances；全部通过 try_deploy_new_kernel 走统一路径。
        req_id 从 current_request['id'] 读取，无请求上下文时传 None。
        返回: {success, reused, instance_key, cpu_used, mem_used}
        """
        req_id = None
        if self.current_request is not None:
            req_id = self.current_request.get('id')

        res = self.try_deploy_new_kernel(
            node=node, vnf_type=vnf_type,
            req_cpu=cpu_need, req_mem=mem_need,
            allow_reuse=True,   # try_reuse_or_deploy 语义是"尽量复用"
            req_id=req_id,
        )
        return {
            'success': res.ok,
            'reused': res.reused,
            'instance_key': (node, vnf_type),
            'cpu_used': res.cpu_used,
            'mem_used': res.mem_used,
        }

    def release_vnf_ref(self, node: int, vnf_type: int, req_id: Optional[int] = None):
        """
        唯一统一的 VNF 释放入口（新内核实现）。
        规则：ref_count 降到 0 时才真正调用 release_node_resource。
        同时兼容旧路径：如果 instance_index 找不到，回退读 shared_vnf_instances。
        """
        inst_id = self.instance_index.get((node, vnf_type))

        if inst_id is not None:
            # ── 新内核路径 ──────────────────────────────────────────────
            inst = self.instance_table.get(inst_id)
            if inst is None or inst.state != "ACTIVE":
                # 实例已不存在，清理兼容视图
                self.shared_vnf_instances.pop((node, vnf_type), None)
                return
            if req_id is not None:
                inst.active_req_ids.discard(req_id)
            if inst.ref_count > 0:
                inst.ref_count -= 1
            if inst.ref_count == 0:
                # 最后一个引用离开，才真正释放物理 CPU/MEM
                self.release_node_resource(node, vnf_type, inst.cpu, inst.mem)
                inst.state = "DELETED"
                self.instance_table.pop(inst_id, None)
                self.instance_index.pop((node, vnf_type), None)
            self._sync_legacy_views()
        else:
            # ── 旧兼容路径（兜底，逐步淘汰）──────────────────────────────
            inst_key = (node, vnf_type)
            inst_dict = self.shared_vnf_instances.get(inst_key)
            if not inst_dict:
                return
            inst_dict['ref_count'] = max(0, int(inst_dict.get('ref_count', 0)) - 1)
            if inst_dict['ref_count'] == 0:
                # 只有最后引用离开才释放
                cpu_used = float(inst_dict.get('cpu_used', 0.0))
                mem_used = float(inst_dict.get('mem_used', 0.0))
                self.release_node_resource(node, vnf_type, cpu_used, mem_used)
                self.shared_vnf_instances.pop(inst_key, None)

    def release_vnf_ref_new(self, node: int, vnf_type: int,
                            req_id: Optional[int] = None) -> bool:
        """已合并到 release_vnf_ref，保留此名作为兼容别名"""
        self.release_vnf_ref(node, vnf_type, req_id=req_id)
        return True

    def release_link_resource(self, u: int, v: int, bw_val: float):
        self.pool.release_bandwidth(u, v, bw_val)

    def release_bandwidth(self, u: int, v: int, bw: float):
        self.release_link_resource(u, v, bw)

    def get_available_bandwidth(self, u: int, v: int) -> float:
        return self.pool.get_available_bandwidth(u, v)

    def probe_vnf_deploy(self, node: int, vnf_type: int,
                         cpu_need: float, mem_need: float) -> dict:
        """
        单一真相探针：统一回答在该节点部署该 VNF 类型是否可行。
        供高层 mask、低层预检查、低层部署三处共用，消除口径不一致。
        返回:
          ok           : bool  — 是否可行
          reuse        : bool  — 应走复用路径（不消耗新 CPU/MEM）
          new_instance : bool  — 应走新建路径
          reason       : str   — 决策原因
        """
        if node not in self.dc_nodes:
            return {'ok': False, 'reuse': False, 'new_instance': False,
                    'reason': 'not_dc'}

        inst_id = self.instance_index.get((node, vnf_type))
        if inst_id is not None:
            inst = self.instance_table.get(inst_id)
            if inst is not None and inst.state == 'ACTIVE' and inst.ref_count > 0:
                cap = self.vnf_share_cap
                can_reuse = (cap is None or inst.ref_count < cap)
                if can_reuse:
                    return {'ok': True, 'reuse': True, 'new_instance': False,
                            'reason': 'reusable_active_instance'}
                else:
                    # 活跃实例存在但共享上限已满，不允许再新建同类型
                    return {'ok': False, 'reuse': False, 'new_instance': False,
                            'reason': 'active_instance_exists_reuse_not_allowed'}

        # 无活跃实例 → 检查新建资源
        cpu_ok = self.pool.get_available_cpu(node) >= cpu_need - 1e-5
        mem_ok = self.pool.get_available_memory(node) >= mem_need - 1e-5
        if cpu_ok and mem_ok:
            return {'ok': True, 'reuse': False, 'new_instance': True,
                    'reason': 'fresh_deploy'}
        return {'ok': False, 'reuse': False, 'new_instance': False,
                'reason': 'insufficient_resource'}

    def check_node_resource(self, node: int, vnf_type: int = 0,
                            cpu_need: float = 0.0, mem_need: float = 0.0) -> bool:
        """统一走 probe_vnf_deploy，不再依赖 nodes_on_tree 判断复用。"""
        return bool(self.probe_vnf_deploy(node, vnf_type, cpu_need, mem_need)['ok'])

    def check_link_resource(self, u: int, v: int, bw_need: float) -> bool:
        return self.pool.get_available_bandwidth(u, v) >= bw_need - 1e-5

    # ---------- 业务辅助方法 ----------
    def _try_deploy(self, node: int, allow_reuse: bool = False):
        """
        主链部署入口（新内核完全切换版）。
        - req_id 统一使用 current_request['id']，不再使用 id(self.current_request)
        - 首次部署时自动调用 register_request_record()
        - 部署成功后调用 bind_vnf_to_request() 写 RequestRecord.vnf_bindings
        - current_tree['placement'] 只作为控制器兼容视图
        - 返回 dict（ok/reused/new_instance/reason/cpu_used/mem_used）不变
        """
        if self.current_request is None:
            return {'ok': False, 'reused': False, 'new_instance': False,
                    'reason': 'no_request', 'cpu_used': 0.0, 'mem_used': 0.0}
        if self.current_phase != 'vnf_deployment':
            return {'ok': True,  'reused': False, 'new_instance': False,
                    'reason': 'phase_skip', 'cpu_used': 0.0, 'mem_used': 0.0}

        vnf_list = self.current_request.get('vnf', [])
        if not vnf_list:
            return {'ok': True,  'reused': False, 'new_instance': False,
                    'reason': 'no_vnf_list', 'cpu_used': 0.0, 'mem_used': 0.0}

        idx = self.next_vnf_idx
        if idx >= len(vnf_list):
            return {'ok': True,  'reused': False, 'new_instance': False,
                    'reason': 'vnf_done', 'cpu_used': 0.0, 'mem_used': 0.0}

        vnf_type = vnf_list[idx]
        cpu_reqs = self.current_request.get('cpu_origin', []) or self.current_request.get('vnf_cpu', [])
        mem_reqs = self.current_request.get('memory_origin', []) or self.current_request.get('vnf_mem', [])
        cpu_need = float(cpu_reqs[idx]) if idx < len(cpu_reqs) else 10.0
        mem_need = float(mem_reqs[idx]) if idx < len(mem_reqs) else 10.0

        if node not in self.dc_nodes:
            logger.warning(f"[_try_deploy] 节点{node}不是DC节点")
            return {'ok': False, 'reused': False, 'new_instance': False,
                    'reason': 'not_dc', 'cpu_used': 0.0, 'mem_used': 0.0}

        # ── 统一业务 req_id：使用 current_request['id']，而非 id(对象) ──
        req_id = self.current_request.get('id')
        if req_id is None:
            logger.error("[_try_deploy] current_request 缺少 'id' 字段，终止部署")
            return {'ok': False, 'reused': False, 'new_instance': False,
                    'reason': 'missing_req_id', 'cpu_used': 0.0, 'mem_used': 0.0}

        # ── 首次部署 或 上一轮失败/释放的旧记录 → 重新注册 ──────────
        # 旧记录 state=FAILED/RELEASED 时 edge_allocations 已被 clear，
        # 但 req_id 仍在 request_table，导致跳过注册，新 episode 的 BW
        # commit 写入旧记录后无法被正确 lifecycle 管理
        _existing = self.request_table.get(req_id)
        if _existing is None or _existing.state in ('FAILED', 'RELEASED'):
            self.register_request_record(
                req_id=req_id,
                source=self.current_request.get('source', 0),
                dests=self.current_request.get('dest', []),
                vnfs=vnf_list,
                bw=float(self.current_request.get('bw_origin', 0.0)),
            )

        # ── 新内核部署（唯一真相） ──────────────────────────────────────
        res = self.try_deploy_new_kernel(
            node=node, vnf_type=vnf_type,
            req_cpu=cpu_need, req_mem=mem_need,
            allow_reuse=allow_reuse,
            req_id=req_id,
        )

        if not res.ok:
            return {'ok': False, 'reused': False, 'new_instance': False,
                    'reason': res.reason, 'cpu_used': 0.0, 'mem_used': 0.0}

        # ── 防御性检查：成功部署后 inst_id 不应为 None ──────────────────
        if res.inst_id is None:
            logger.error(
                f"[_try_deploy] ❌ 部署成功但 inst_id=None "
                f"req={req_id} node={node} vnf_type={vnf_type} reason={res.reason}"
            )

        # ── 写 RequestRecord.vnf_bindings（唯一会计记录）──────────────
        self.bind_vnf_to_request(
            req_id=req_id, node=node, vnf_type=vnf_type,
            deploy_res=res, req_cpu=cpu_need, req_mem=mem_need,
        )

        # ── 写 current_tree['placement'] 兼容视图（控制器读取用）────────
        self.current_tree.setdefault('placement', {})
        self.current_tree.setdefault('new_instance_count', 0)
        self.current_tree.setdefault('reused_vnf_count', 0)
        self.current_tree['placement'][(node, idx)] = {
            'node': node,
            'vnf_type': vnf_type,
            'cpu_used': res.cpu_used,
            'mem_used': res.mem_used,
            'reused': res.reused,
            'inst_id': res.inst_id,
            'shared_ref_only': res.reused,
        }
        if res.reused:
            self.current_tree['reused_vnf_count'] += 1
        else:
            self.current_tree['new_instance_count'] += 1

        logger.debug(
            f"[CPU-ALLOC] req={req_id} vnf_idx={idx} node={node} "
            f"reused={int(res.reused)} cpu={cpu_need:.1f} mem={mem_need:.1f}"
        )
        return {
            'ok': True,
            'reused': res.reused,
            'new_instance': res.new_instance,
            'reason': res.reason,
            'cpu_used': res.cpu_used,
            'mem_used': res.mem_used,
        }

    def _archive_request(self, success=False, already_rolled_back=False):
        self.handler._archive_request(success, already_rolled_back)

    # ---- 旧部署链已删除（complete_current_request / apply_deployment /
    #      apply_tree_deployment / get_network_state_dict）：
    #      均为 RequestHandler 旧接口的代理，随 RequestHandler 对应方法一起废弃。
    #      当前请求清理由 episode_reset() 负责，部署由 _try_deploy() 即时完成。 ----

    # ---- 工具/兼容接口已删除（get_available_resources / has_link /
    #      get_link_cost / get_node_features）：
    #      get_available_resources：全网快照，当前状态由 get_state()/get_high_level_state_graph() 构建。
    #      has_link：调用方直接用 get_neighbors() 即可判断连通性。
    #      get_link_cost：固定返回 1.0，无实际意义。
    #      get_node_features：旧版 GNN 接口，当前特征由 HighLevelController 直接组装。 ----

    # ---------- 查询接口 ----------
    def get_neighbors(self, node: int) -> List[int]:
        if node < 0 or node >= self.n:
            return []
        return np.where(self.topo[node] > 0)[0].tolist()

    def episode_reset(self):
        """
        🔥 Episode 级轻量重置（在线模拟专用）

        只清理未提交的事务预留（防止上一个 Episode 的残留锁住资源），
        保留：
          - lifecycle 中的活跃请求（让其自然过期）
          - hvt_all 和 vnf_instances（lifecycle 释放时需要这两个表）
          - pool 的 avail（CPU/MEM/BW 真实占用量）

        调用时机：每个 Episode（请求）结束后，准备处理下一个请求前。
        不调用时机：仿真开始/完全重置（用 reset(hard=True)）。
        """
        # 只清 reserved（未提交事务的预留残留），不动 avail 和 lifecycle
        self.pool.reset(hard=False)
        # [BugFix] 清空 current_tree 的 placement 和 tree，防止上一轮失败的
        # VNF 部署残留在 placement 里，被下一轮成功时的 _collect_allocated_resources
        # 一起注册进 lifecycle，导致 lc_cpu/lc_mem > pool（幽灵VNF）。
        self.current_tree = {
            'tree': {},
            'placement': {},
            'connected_dests': set(),
            'new_instance_count': 0,
            'reused_vnf_count': 0,
            'node_stage': {},
        }
        self.current_request = None
        self.next_vnf_idx = 0
        self.nodes_on_tree = set()
        # 新内核：episode 重置时清理已完结的请求记录（保留 ACTIVE）
        self.request_table = {
            k: v for k, v in self.request_table.items()
            if v.state in {"PENDING", "ACTIVE"}
        }
        logger.debug("[FusedRM] Episode软重置完成 (lifecycle/hvt/instances保留, tree已清空)")

    def reset(self, hard: bool = False, current_time: float = 0.0):
        """
        重置入口，自动根据模式选择行为：
          online_mode=True  + hard=False → episode_reset()（保留lifecycle）
          online_mode=False 或 hard=True → 完整重置（cleanup_all）
        """
        online = (hasattr(self, 'env') and
                  getattr(self.env, 'online_mode', False))

        if online and not hard:
            self.episode_reset()
        else:
            self.pool.reset(hard)
            self.hvt_all.fill(0)
            self.vnf_instances.clear()
            self.shared_vnf_instances.clear()
            if hasattr(self, 'env') and hasattr(self.env, 'current_time'):
                ct = self.env.current_time
            else:
                ct = current_time
            self.request_manager.cleanup_all(current_time=ct)
            logger.debug(f"[FusedRM] 完整重置完成 (hard={hard})")
