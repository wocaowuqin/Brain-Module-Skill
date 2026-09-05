"""
envs/modules/low_level_controller.py
====================================
低层执行控制器 - TA-HRL v4 升级版 (Tree-Aware & Steiner Routing)
====================================

【模块定位】
本模块是 TA-HRL（Tree-Aware Hierarchical Reinforcement Learning）系统的低层执行控制器。
在两层 HRL 架构中，高层（HRL_Coordinator + HighLevelController）负责选择子目标（目标 DC
节点或目的地节点），低层（本模块）负责在当前网络状态下一步步走到该子目标，并沿途构建
多播 SFC 树。低层的每一个 step 对应网络中一次节点间移动或一次 VNF 部署决策。

【核心职责】
  1. VNF 部署（vnf_deployment 阶段）
     - 移动到高层指定的 DC 节点，调用 _try_deploy() 部署当前 VNF
     - 部署成功后推进 next_vnf_idx，直到所有 VNF 依次部署完毕
     - 失败时触发重试或截断，并通过 _archive_episode_fail() 归还 CPU/MEM

  2. 目的地连接（destination_connection 阶段）
     - 从最后一个 VNF 节点出发，走到高层指定的目的地节点
     - 沿途将经过的边记录进 current_tree['tree']（flow 语义见下）
     - 所有目的地全部连通后，调用 _commit_episode_bandwidth() 一次性提交 BW

  3. 树边构建与 BW 语义（delayed-commit 模式）
     - 搜索阶段：_handle_movement() 只记录树边，不扣 BW（避免试探路径污染资源池）
     - 成功阶段：_commit_episode_bandwidth() 只对 flow > 0.0 的最终树边一次性扣 BW
     - 失败阶段：_archive_episode_fail() 只回滚 CPU/MEM，BW 无需回滚（从未扣过）
     - flow 值含义：
         1.0  → 正常多播树边，BW 在成功时提交
         0.0  → 过程记录边（skip边/dest出边），不计入最终树，不扣 BW
        -1.0  → branch/spine 记录边，仅用于结构追踪，不扣 BW

  4. 动作掩码生成（get_low_level_action_mask）
     - 带宽基础过滤：只允许走 BW 足够的链路（树边无条件放行）
     - 方向约束：禁止远离目标的移动（有前进方向可走时才施加）
     - 禁忌表：Path Tabu 防止来回震荡；软释放保证死路时有出口
     - top-k 收缩：对候选邻居按树边复用 > BW 富余 > 距离目标排序，只保留 top-k
     - 硬防环：写入 flow=1.0 前检查两端是否在正树中已连通，防止闭环

  5. 候选邻居暴露（get_low_level_candidates）  ← v4.2 新增
     - 在 get_low_level_action_mask 基础上，额外计算每个候选邻居的局部特征
     - 供 low_policy 的 score_candidates() 做逐点评分，替代全图 logits 输出
     - 特征包含：delta_hop、is_tree_edge、avail_bw_ratio、is_on_tree、
                 avg_hop_to_undone_dests、tabu_penalty

  6. 路径规划辅助（compute_bw_aware_path）
     - 基于 networkx 构建 BW 感知图，优先复用树边（权重 0.1），只走 BW 足够的新边
     - BW 不足时降级为 Widest Path（最大化瓶颈 BW），兜底保障可达性

  7. 状态特征构建（get_state）
     - 输出 24 维特征向量 + dest_mask，供低层 policy 网络使用
     - 特征包含：位置、VNF 进度、资源余量、树连通状态、目标距离等

【主要函数索引】
  step_low_level()                   主入口，路由到对应阶段处理函数
  _handle_vnf_deployment()           VNF 部署阶段：移动 + 部署 + 超时/失败处理
  _handle_destination_connection()   目的地连接阶段：走到目的地 + 全部连通时触发成功归档
  _handle_movement()                 通用移动逻辑：更新位置、记录树边、计算奖励
  get_low_level_action_mask()        生成合法动作掩码（含方向约束、禁忌、top-k、防环）
  get_low_level_candidates()         暴露结构化候选邻居 + 局部特征（v4.2 新增）
  compute_bw_aware_path()            BW 感知最短路径规划，带 Widest Path 降级
  get_state()                        构建低层状态特征向量
  _commit_episode_bandwidth()        成功后对 flow>0 树边一次性提交 BW
  _archive_episode_fail()            失败时回滚 CPU/MEM，不回滚 BW
  _add_request_to_lifecycle_manager()  将成功请求注册进生命周期管理器
  _would_create_cycle()              防环检查：两端均在正树中才做连通性判断

【与其他模块的依赖关系】
  → HRL_Coordinator        调用 step_low_level() 驱动低层执行；提供 planner 路径
  → HighLevelController    设置 current_target_node / current_deployment_target 子目标
  → AllResourceManager     CPU/MEM 扣减（allocate_node_resource）与释放（release_node_resource）
                           BW 扣减（allocate_bandwidth，仅在 commit 时调用）
  → RequestLifecycleManager  成功请求注册，生命周期到期后自动释放资源
  → RewardCritic           统一奖励入口 _r()，替代所有硬编码奖励数值

架构级修复与升级:
  [1] TA-HRL v4: 注入 hop_to_tree 距离感知，引导多播树边复用。
  [2] TA-HRL v4: 注入 dest_mask 目标感知，引导全局最优 Steiner 分叉点。
  [3] DAG Mask & Tabu List: 严格距离掩码禁止反向游走，动态禁忌表防止三角死锁。
  [4] Reward Gradient: +2.0(靠近目标), -3.0(远离目标), -3.0(死路退回), -1.0(等距振荡), -5.0(连续无进展)

带宽孤岛修复 (v4.1):
  [修改1] compute_bw_aware_path: 增加 Widest Path 降级兜底，严格BW路径失败时改走带宽最宽路径。
  [修改2] _handle_destination_connection: 带宽孤岛不再直接Episode失败，
          先尝试绕路，绕路无解时截断子目标让高层重调度，连续5次失败才终止Episode。
  [修改3] get_low_level_action_mask 死路软释放: 优先复用树边（不消耗新带宽），
          树边也不通才放开有剩余带宽的新边，减少无效带宽消耗。

候选邻居逐点评分 (v4.2):
  [新增] get_low_level_candidates(): 在 mask 基础上暴露结构化候选 + 6 维局部特征。
         供 GoalConditionedLowLevelPolicy.score_candidates() 使用，替代全图 logits。
         调用方在获取 mask 同时调用此方法，将返回值传入 policy.select_action()。
"""

import numpy as np
import torch
import logging
import networkx as nx
from torch_geometric.data import Data
import copy

try:
    from .controller_shared_helper import ControllerSharedHelper
except ImportError:
    try:
        from controller_shared_helper import ControllerSharedHelper
    except ImportError:
        ControllerSharedHelper = None

logger = logging.getLogger(__name__)


class LowLevelController:
    """低层执行控制器 - TA-HRL v4 顶级架构优化版"""

    def __init__(self, env):
        self.env = env

        if not hasattr(self.env, 'resource_mgr'):
            logger.error("❌ LowLevelController: 未找到 resource_mgr")
            raise RuntimeError("resource_mgr 必须配置")

        if not hasattr(self.env, 'request_manager'):
            if hasattr(self.env.resource_mgr, 'request_manager'):
                self.env.request_manager = self.env.resource_mgr.request_manager
            else:
                try:
                    req_mgr_class = self.env.resource_mgr.__class__.__module__
                    import sys
                    mod = sys.modules[req_mgr_class]
                    if hasattr(mod, 'RequestLifecycleManager'):
                        RM_Class = getattr(mod, 'RequestLifecycleManager')
                        self.env.request_manager = RM_Class(self.env.resource_mgr)
                        self.env.resource_mgr.request_manager = self.env.request_manager
                except Exception as e:
                    logger.error(f"❌ [Init] 无法创建 request_manager: {e}")

        # 预计算最短路径距离矩阵（一次性）
        # top-k 候选收缩（与HRL_Coordinator配合）
        # [Task4] 默认值从3提高到5，减少对可行绕路的裁剪
        self._low_topk = self.env.config.get('low_topk', 5) if hasattr(self.env, 'config') else 5

        # 共享辅助器（与 HighLevelController 复用同一套工具方法）
        if ControllerSharedHelper is not None:
            self.shared = ControllerSharedHelper(env)
        else:
            self.shared = None
            logger.warning("[LLC] ControllerSharedHelper 不可用，降级为内置实现")

        logger.debug("✅ [LowLevelController] 初始化完成（TA-HRL v4.2 架构）")

    def _r(self, phase: str, **kwargs) -> float:
        """统一奖励入口，委托给env.reward_critic.get_reward()"""
        rc = getattr(self.env, 'reward_critic', None)
        if rc is None:
            return 0.0
        try:
            return float(rc.get_reward(phase, **kwargs))
        except Exception as _e:
            logger.debug(f"[RewardCritic] get_reward({phase}) 异常: {_e}")
            return 0.0

    # ==================================================================
    # Hop Distance 缓存
    # ==================================================================
    def _get_hop_distance(self, u, v):
        return self.shared.get_hop_distance_lazy(u, v)

    def compute_bw_aware_path(self, source, target, excluded_edges=None):
        if source == target:
            return [source]

        if excluded_edges is None:
            excluded_edges = set()

        bw_req = 0.0
        if self.env.current_request:
            bw_req = self.env.current_request.get('bw_origin', 0.0)

        tree_edges = self.shared.get_positive_tree_edge_set()
        # Reuse positive tree edges of the current request even if their
        # residual physical BW is below bw_req. New edges still need BW.
        # Directed bandwidth keeps (u, v) and (v, u) independent.
        G = nx.DiGraph()
        for u in range(self.env.n):
            for v in self.env.resource_mgr.get_neighbors(u):
                if (u, v) in excluded_edges:
                    continue
                avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(u, v)
                is_tree_edge = (u, v) in tree_edges
                is_reverse_tree_edge = (v, u) in tree_edges
                if avail_bw < bw_req and not is_tree_edge:
                    continue
                # [Fix-UtilWeight] 在 weight 里加入方向利用率，引导 planner 绕开饱和方向
                _cap_uv = self.env.resource_mgr.pool.bw_cap.get((u, v), 1.0)
                _util_uv = 1.0 - avail_bw / max(_cap_uv, 1.0)  # 0=空闲, 1=饱和
                if is_tree_edge:
                    G.add_edge(u, v, weight=0.1 + 0.5 * _util_uv)
                elif is_reverse_tree_edge:
                    # A reverse tree corridor is structurally useful, but it
                    # is a new directed allocation and was checked above.
                    G.add_edge(u, v, weight=0.45 + 1.0 * _util_uv)
                else:
                    G.add_edge(u, v, weight=1.0 + 3.0 * _util_uv)

        if source not in G or target not in G:
            return None

        try:
            return nx.shortest_path(G, source, target, weight='weight')
        except nx.NetworkXNoPath:
            # 降级：Widest Path 兜底（最大最小带宽路径）
            G_fallback = nx.DiGraph()
            for u in range(self.env.n):
                for v in self.env.resource_mgr.get_neighbors(u):
                    if (u, v) in excluded_edges:
                        continue
                    avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(u, v)
                    is_tree_edge = (u, v) in tree_edges
                    is_reverse_tree_edge = (v, u) in tree_edges
                    if avail_bw < bw_req and not is_tree_edge:
                        continue
                    # [Fix-UtilWeight-Fallback] fallback 路径同步加入利用率感知
                    _cap_fb = self.env.resource_mgr.pool.bw_cap.get((u, v), 1.0)
                    _util_fb = 1.0 - avail_bw / max(_cap_fb, 1.0)
                    if is_tree_edge:
                        G_fallback.add_edge(u, v, weight=0.01 + 0.1 * _util_fb)
                    elif is_reverse_tree_edge:
                        G_fallback.add_edge(u, v, weight=0.2 + 0.3 * _util_fb)
                    else:
                        G_fallback.add_edge(u, v, weight=1.0 / (avail_bw + 0.01) + _util_fb)
            try:
                path = nx.shortest_path(G_fallback, source, target, weight='weight')
                logger.debug(f"[WidestPath] 严格BW路径失败，降级找到带宽最宽路径: {source}→{target} = {path}")
                return path
            except Exception:
                return None
        except Exception as e:
            return None

    def compute_progress_path(self, source, target, excluded_edges=None):
        """Return a committable destination extension path.

        Existing directed tree edges may be reused. New edges may leave any
        full-stage tree node, including an already connected receiver, but may
        not re-enter the tree.  A receiver switch can replicate one output to
        its host and forward another output without violating SFC order.
        """
        if source == target:
            return [source]
        source = int(source)
        target = int(target)
        excluded_edges = set(excluded_edges or ())
        bw_req = float((self.env.current_request or {}).get('bw_origin', 0.0))
        tree_edges = self.shared.get_positive_tree_edge_set()
        tree_nodes = {node for edge in tree_edges for node in edge}
        graph = nx.DiGraph()
        for u in range(self.env.n):
            for v in self.env.resource_mgr.get_neighbors(u):
                if (u, v) in excluded_edges:
                    continue
                if self._would_create_cycle(u, v):
                    continue
                available = self.env.resource_mgr.pool.get_available_bandwidth(u, v)
                is_tree_edge = (u, v) in tree_edges
                if not is_tree_edge and available + 1e-9 < bw_req:
                    continue
                if not is_tree_edge and v in tree_nodes:
                    continue
                capacity = self.env.resource_mgr.pool.bw_cap.get((u, v), 1.0)
                utilization = max(0.0, min(1.0, 1.0 - available / max(1.0, capacity)))
                graph.add_edge(
                    u,
                    v,
                    weight=99.0 + utilization if is_tree_edge else 100.0 + utilization,
                )
        try:
            return nx.shortest_path(graph, source, target, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def plan_destination_completion(self, beam_width=64):
        """Plan a complete, ordered multicast extension for pending receivers.

        Each expansion attaches one receiver to any full-stage tree node using
        only new nodes.  This preserves the directed-tree single-parent rule;
        a completed receiver may also be an internal multicast replication
        point.  Beam search bounds runtime while
        considering different receiver orders, which a one-step reachability
        mask cannot do.
        """
        beam_width = max(1, int(getattr(
            self.env, '_online_destination_beam_width', beam_width
        ) or beam_width))
        request = getattr(self.env, 'current_request', None) or {}
        all_dests = {int(node) for node in request.get('dest', [])}
        connected = set(self.shared.get_connected_dests_view())
        pending = sorted(all_dests - connected)
        if not pending:
            return []

        tree_edges = set(self.shared.get_positive_tree_edge_set())
        tree_nodes = {node for edge in tree_edges for node in edge}
        chain = list(getattr(self.env, 'chain_nodes', []) or [])
        root = int(chain[-1]) if chain else None
        if root is None:
            return None
        tree_nodes.add(root)

        total_vnf = len(request.get('vnf', []) or [])
        node_stage = (getattr(self.env, 'current_tree', None) or {}).get(
            'node_stage', {}
        )
        full_stage = {
            int(node) for node in tree_nodes
            if int(node_stage.get(node, 0)) >= total_vnf
        }
        full_stage.add(root)
        bw_req = float(request.get('bw_origin', 0.0))
        pool = self.env.resource_mgr.pool
        super_source = '__destination_plan_root__'
        failed_anchors_by_target = getattr(
            self.env, '_failed_anchors_for_target', {}
        )

        def find_attachment(target, remaining, state_tree, state_full, state_connected):
            failed_anchors = {
                int(node) for node in failed_anchors_by_target.get(
                    int(target), set()
                )
                if node is not None
            }
            anchors = set(state_full) - failed_anchors
            if not anchors:
                return None
            blocked_terminals = set(remaining) - {int(target)}
            graph = nx.DiGraph()
            for anchor in anchors:
                graph.add_edge(super_source, int(anchor), weight=0.0)
            for u in range(self.env.n):
                for v in self.env.resource_mgr.get_neighbors(u):
                    u, v = int(u), int(v)
                    if v in state_tree:
                        continue
                    if u in state_tree and u not in anchors:
                        continue
                    if v in blocked_terminals:
                        continue
                    available = float(pool.get_available_bandwidth(u, v))
                    if available + 1e-9 < bw_req:
                        continue
                    capacity = float(pool.bw_cap.get((u, v), 1.0))
                    utilization = max(
                        0.0,
                        min(1.0, 1.0 - available / max(1.0, capacity)),
                    )
                    graph.add_edge(u, v, weight=1.0 + utilization)
            try:
                path = nx.shortest_path(
                    graph, super_source, int(target), weight='weight'
                )[1:]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
            if not path:
                return None
            anchor = int(path[0])
            return {
                'target': int(target),
                'anchor': anchor,
                'path': [int(node) for node in path],
                'cost': max(0, len(path) - 1),
            }

        states = [{
            'tree': set(tree_nodes),
            'full': set(full_stage),
            'connected': set(connected),
            'remaining': tuple(pending),
            'plan': [],
            'cost': 0,
        }]
        for _ in range(len(pending)):
            expanded = []
            for state in states:
                for target in state['remaining']:
                    attachment = find_attachment(
                        target,
                        state['remaining'],
                        state['tree'],
                        state['full'],
                        state['connected'],
                    )
                    if attachment is None:
                        continue
                    path_nodes = set(attachment['path'])
                    expanded.append({
                        'tree': state['tree'] | path_nodes,
                        'full': state['full'] | path_nodes,
                        'connected': state['connected'] | {int(target)},
                        'remaining': tuple(
                            node for node in state['remaining']
                            if int(node) != int(target)
                        ),
                        'plan': state['plan'] + [attachment],
                        'cost': state['cost'] + attachment['cost'],
                    })
            if not expanded:
                return None
            expanded.sort(key=lambda item: (item['cost'], item['remaining']))
            states = expanded[:max(1, int(beam_width))]
        return states[0]['plan'] if states else None

    def validate_destination_plan_step(self, step):
        """Revalidate one cached receiver attachment against the live ledger."""
        if not isinstance(step, dict):
            return False, 'missing_step'
        try:
            target = int(step['target'])
            anchor = int(step['anchor'])
            path = [int(node) for node in step.get('path', [])]
        except (KeyError, TypeError, ValueError):
            return False, 'malformed_step'

        request = getattr(self.env, 'current_request', None) or {}
        destinations = {int(node) for node in request.get('dest', [])}
        connected = set(self.shared.get_connected_dests_view())
        if target not in destinations or target in connected:
            return False, 'target_not_pending'
        if not path or path[0] != anchor or path[-1] != target:
            return False, 'path_shape'
        if len(path) != len(set(path)):
            return False, 'path_cycle'
        failed_anchors = getattr(
            self.env, '_failed_anchors_for_target', {}
        ).get(target, set())
        if anchor in failed_anchors:
            return False, 'failed_anchor'

        total_vnf = len(request.get('vnf', []) or [])
        node_stage = (getattr(self.env, 'current_tree', None) or {}).get(
            'node_stage', {}
        )
        if int(node_stage.get(anchor, 0)) < total_vnf:
            return False, 'anchor_not_full_stage'

        pending_others = destinations - connected - {target}
        if any(node in pending_others for node in path[1:-1]):
            return False, 'crosses_pending_receiver'

        tree_edges = set(self.shared.get_positive_tree_edge_set())
        tree_nodes = {node for edge in tree_edges for node in edge}
        tree_nodes.add(anchor)
        bw_req = float(request.get('bw_origin', 0.0))
        for index, (u, v) in enumerate(zip(path, path[1:])):
            if v not in self.env.resource_mgr.get_neighbors(u):
                return False, f'non_neighbor:{u}->{v}'
            if (u, v) in tree_edges:
                continue
            # A new branch may leave its first tree anchor, but must not enter
            # another existing tree node and acquire a second parent.
            if index > 0 and u in tree_nodes:
                return False, f'leaves_unplanned_tree_node:{u}'
            if v in tree_nodes:
                return False, f'tree_reentry:{v}'
            available = float(
                self.env.resource_mgr.pool.get_available_bandwidth(u, v)
            )
            if available + 1e-9 < bw_req:
                return False, f'bandwidth:{u}->{v}'
        return True, 'ok'

    # ==================================================================
    # 核心步进
    # ==================================================================
    def step_low_level(self, action):
        if self.env.current_request is None:
            return self.get_state(), self._r('penalty', type='internal_error'), True, False, {
                'fail': True, 'reason': 'no_req'
            }

        if not hasattr(self.env, 'subgoal_step_count'):
            self.env.subgoal_step_count = 0
        if not hasattr(self.env, '_dest_anchor_mismatch'):
            self.env._dest_anchor_mismatch = False
        if not hasattr(self.env, '_dest_step0_checked'):
            self.env._dest_step0_checked = False
        if not hasattr(self.env, '_dest_cycle_armed'):
            self.env._dest_cycle_armed = False

        # 全局失败计数：拆成两个独立计数器，各自语义/阈值不同
        # _timeout_count : 步数超时 + VNF资源不足，阈值=3
        # _island_count  : 带宽孤岛（目标所有接入链路已满），阈值=5
        if not hasattr(self.env, '_timeout_count'):
            self.env._timeout_count = 0
        if not hasattr(self.env, '_island_count'):
            self.env._island_count = 0
        # 向后兼容：_consecutive_timeout_count 指向 _timeout_count
        self.env._consecutive_timeout_count = self.env._timeout_count
        if not hasattr(self.env, '_bw_fail_count'):
            self.env._bw_fail_count = 0
        if not hasattr(self.env, '_deadlock_step_count'):
            self.env._deadlock_step_count = 0

        # 仅在子目标刚切换（_need_reset_to_last_vnf标志）时执行一次回位
        # [Task2] 优先回 current_anchor_node，没有时才 fallback 到 last_vnf
        if self.env.current_phase == 'destination_connection' and \
                getattr(self.env, '_need_reset_to_last_vnf', False):
            _anchor = getattr(self.env, 'current_anchor_node', None)
            _chain_nodes = getattr(self.env, 'chain_nodes', [])
            _last_vnf = _chain_nodes[-1] if _chain_nodes else None
            _reset_target = _anchor if _anchor is not None else _last_vnf
            if _reset_target is not None and self.env.current_node_location != _reset_target:
                self.env.current_node_location = _reset_target
                self.env.current_subgoal_full_path = [int(_reset_target)]
                _src = 'anchor' if _anchor is not None else 'last_vnf'
                logger.debug(f"🔄 [Low] 瞬移回位至{_src}={_reset_target} "
                             f"(anchor={_anchor}, last_vnf={_last_vnf})")
                if hasattr(self.env, 'current_path_trace'):
                    self.env.current_path_trace = [_reset_target]
            self.env._need_reset_to_last_vnf = False

        self.env.subgoal_step_count += 1

        if (
            self.env.current_phase == 'destination_connection'
            and getattr(self.env, '_dest_cycle_armed', False)
            and getattr(self.env, '_dest_step0_checked', False)
            and getattr(self.env, '_dest_anchor_mismatch', False)
        ):
            _anchor_mm = getattr(self.env, 'current_anchor_node', None)
            _target_mm = getattr(self.env, 'current_target_node', None)
            _cur_mm = getattr(self.env, 'current_node_location', None)
            logger.error(
                f"[AnchorMismatchAbort] target={_target_mm} anchor={_anchor_mm} cur={_cur_mm} "
                f"path_trace={getattr(self.env, 'current_path_trace', None)}"
            )
            self.env._dest_cycle_armed = False
            return self.get_state(), self._r('penalty', type='invalid_action'), False, True, {
                'anchor_mismatch': True,
                'reason': 'anchor_mismatch',
                'target': _target_mm,
                'anchor': _anchor_mm,
                'current': _cur_mm,
            }

        current_node = self.env.current_node_location
        target_action = int(action)
        is_stay = (target_action == current_node)

        at_target = False
        if is_stay:
            if self.env.current_phase == 'vnf_deployment':
                at_target = (current_node == getattr(self.env, 'current_deployment_target', None))
            elif self.env.current_phase == 'destination_connection':
                at_target = (current_node == getattr(self.env, 'current_target_node', None))

        _phase_now_sl = getattr(self.env, 'current_phase', None)
        _max_steps_now = getattr(self.env, 'max_subgoal_steps', 25)
        if _phase_now_sl == 'destination_connection':
            # dest 阶段步数上限：first dest 宁早截断重选，也别把树拉歪
            _all_d  = len(self.env.current_request.get('dest', [])) \
                      if self.env.current_request else 0
            _done_d = len(self.env.current_tree.get('connected_dests', set())) \
                      if self.env.current_tree else 0
            _remaining_d = _all_d - _done_d
            if _remaining_d == _all_d:
                # [Fix-F] first dest：步数改为动态，不再硬限25步
                # 原来25步对大网络/远anchor不够，改为至少35步
                _max_steps_now = max(_max_steps_now, 35)
            elif _remaining_d <= 1:
                _max_steps_now = max(_max_steps_now, 55)  # 最后1个dest宽裕
            else:
                _max_steps_now = max(_max_steps_now, 45)  # 其余dest
        elif _phase_now_sl == 'vnf_deployment':
            # VNF 阶段：每个VNF最多20步，防止长途绕行污染树
            _max_steps_now = min(_max_steps_now, 20)
        if not at_target and self.env.subgoal_step_count > _max_steps_now:
            self.env.subgoal_step_count = 0
            if _phase_now_sl == 'destination_connection':
                self.env._dest_cycle_armed = False
                self.env._dest_step0_checked = False
                self.env._dest_anchor_mismatch = False
            self.env._need_reset_to_last_vnf = False
            self.env._timeout_count += 1
            self.env._consecutive_timeout_count = self.env._timeout_count  # 同步别名

            _ph = getattr(self.env, 'current_phase', '?')
            _tgt = (getattr(self.env, 'current_deployment_target', None)
                    if _ph == 'vnf_deployment'
                    else getattr(self.env, 'current_target_node', None))
            _dist = self.shared.get_hop_distance_lazy(current_node, _tgt) if _tgt is not None else '?'
            _conn = len(self.env.current_tree.get('connected_dests', set())) if self.env.current_tree else 0
            _alld = len(self.env.current_request.get('dest', [])) if self.env.current_request else 0
            _max_steps = _max_steps_now

            _trace = getattr(self.env, 'current_path_trace', [])
            _trace_set = set(_trace)
            _nbrs = self.env.resource_mgr.get_neighbors(current_node)
            _mask = [n for n in _nbrs if n not in _trace_set]
            # 打印每个邻居的距离和被屏蔽原因
            _nbr_detail = []
            _d_cur = self.shared.get_hop_distance_lazy(current_node, _tgt) if _tgt is not None else 99
            _tree_edges = self.shared.get_positive_tree_edge_set()
            for _n in _nbrs:
                _dn = self.shared.get_hop_distance_lazy(_n, _tgt) if _tgt is not None else 99
                _ek = (current_node, _n)
                _reasons = []
                if _n in _trace_set: _reasons.append('tabu')
                # [Fix-Bidir-G] 诊断日志：原来\u53ea\u68c0\u67e5\u524d\u5411 key，\u53cd\u5411\u6811\u8fb9\u4f1a\u88ab\u8bef\u62a5 constraint_A
                if _dn > _d_cur and _ek not in _tree_edges and (_n, current_node) not in _tree_edges: _reasons.append('constraint_A')
                _nbr_detail.append(f"{_n}(d={_dn},{'|'.join(_reasons) if _reasons else 'OK'})")
            # 区分失败原因：走满40步 vs 物理资源拦截（trace很短）
            _steps_used = len(_trace)
            if _steps_used >= _max_steps * 0.8:
                _fail_type = "⏰ 迷宫超时"
                _res_info = ""
            else:
                # 诊断是哪种资源不足
                _res_parts = []
                _bw_req = self.env.current_request.get('bw_origin', 0.0) if self.env.current_request else 0.0
                _vnf_list = self.env.current_request.get('vnf', []) if self.env.current_request else []
                _vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
                _cpu_reqs = self.env.current_request.get('cpu_origin', []) if self.env.current_request else []
                _mem_reqs = self.env.current_request.get('memory_origin', []) or \
                            (self.env.current_request.get('vnf_mem', []) if self.env.current_request else [])
                _req_cpu = _cpu_reqs[_vnf_idx] if _vnf_idx < len(_cpu_reqs) else 0.0
                _req_mem = _mem_reqs[_vnf_idx] if _vnf_idx < len(_mem_reqs) else 0.0
                # 检查target节点的CPU/MEM
                if _tgt is not None:
                    try:
                        _avail_cpu = self.env.resource_mgr.pool.get_available_cpu(_tgt)
                        _avail_mem = self.env.resource_mgr.pool.get_available_memory(_tgt)
                        if _req_cpu > 0 and _avail_cpu < _req_cpu:
                            _res_parts.append(f"CPU不足({_avail_cpu:.0f}<{_req_cpu:.0f})")
                        if _req_mem > 0 and _avail_mem < _req_mem:
                            _res_parts.append(f"MEM不足({_avail_mem:.0f}<{_req_mem:.0f})")
                        # 检查所有邻居链路的BW，找出不足的
                        _bw_blocked = []
                        for _n in _nbrs:
                            _avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, _n)
                            if _bw_req > 0 and _avail_bw < _bw_req:
                                _bw_blocked.append(f"{current_node}-{_n}({_avail_bw:.1f})")
                        if _bw_blocked:
                            _res_parts.append(f"BW不足需要{_bw_req:.1f}:{','.join(_bw_blocked)}")
                    except Exception:
                        pass
                # 如果还是未知，检查target所有接入链路BW和节点资源
                if not _res_parts and _tgt is not None:
                    try:
                        _tgt_nbrs = self.env.resource_mgr.get_neighbors(_tgt)
                        _tgt_bw_blocked = []
                        for _tn in _tgt_nbrs:
                            _abw = self.env.resource_mgr.pool.get_available_bandwidth(_tn, _tgt)
                            if _bw_req > 0 and _abw < _bw_req:
                                _tgt_bw_blocked.append(f"{_tn}-{_tgt}({_abw:.1f})")
                        if len(_tgt_bw_blocked) == len(_tgt_nbrs):
                            _res_parts.append(f"target={_tgt}带宽孤岛(需{_bw_req:.1f})")
                        elif _tgt_bw_blocked:
                            _res_parts.append(f"target={_tgt}部分链路BW不足:{','.join(_tgt_bw_blocked)}")
                        # 检查当前节点到所有邻居的BW
                        _cur_bw_blocked = []
                        for _cn in _nbrs:
                            _abw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, _cn)
                            if _bw_req > 0 and _abw < _bw_req:
                                _cur_bw_blocked.append(f"{current_node}-{_cn}({_abw:.1f})")
                        if _cur_bw_blocked:
                            _res_parts.append(f"当前节点出链路BW不足:{','.join(_cur_bw_blocked)}")
                    except Exception as _e:
                        _res_parts.append(f"资源查询异常:{_e}")
                _res_info = f" [{', '.join(_res_parts) if _res_parts else '原因待查'}]"
                _fail_type = "💥 资源拦截"
            logger.debug(
                f"{_fail_type}{_res_info} at Node {current_node} "
                f"(连续: {self.env._consecutive_timeout_count}次, steps_used={_steps_used}/{_max_steps}) | "
                f"phase={_ph} target={_tgt} hop_dist={_dist} "
                f"dest_progress={_conn}/{_alld} | "
                f"tabu_free={_mask} nbr_detail={_nbr_detail}"
            )

            # [Fix-VNF-Timeout] 阈值从 3 提高到 5：
            # VNF 阶段每次 timeout 会切换目标 DC，给高层更多重选机会，
            # 避免在资源充裕时因为路径问题过早终止整个 episode。
            if self.env._timeout_count >= 5:
                _vt = len(self.env.current_request.get('vnf', [])) if self.env.current_request else 0
                _vd = getattr(self.env, 'next_vnf_idx', 0)
                _dt = len(self.env.current_request.get('dest', [])) if self.env.current_request else 0
                _dc_conn = len(self.env.current_tree.get('connected_dests', set())) if self.env.current_tree else 0

                logger.debug(
                    f"❌ [Low] 连续超时{self.env._timeout_count}次，Episode失败 | "
                    f"VNF={_vd}/{_vt} Dest={_dc_conn}/{_dt} | "
                    f"phase={_ph} cur={current_node} target={_tgt} hop_dist={_dist}"
                )
                _is_vnf_phase = (_ph == 'vnf_deployment')
                self.env._timeout_count = 0
                self.env._consecutive_timeout_count = 0
                self._archive_episode_fail()
                return self.get_state(), self._r('timeout', in_vnf_phase=_is_vnf_phase), True, False, {
                    'fail': True, 'reason': 'consecutive_timeout'
                }

            # VNF 阶段 timeout 时，把当前目标 DC 加入本 episode 封禁集，
            # 让高层下次不再重选同一个卡死的节点
            if _ph == 'vnf_deployment' and _tgt is not None:
                if not hasattr(self.env, '_episode_deploy_failed'):
                    self.env._episode_deploy_failed = set()
                self.env._episode_deploy_failed.add(int(_tgt))
                logger.debug(f"[VNF-Timeout-Ban] 目标DC={_tgt} 加入本episode封禁集")
            _is_vnf_phase = (getattr(self.env, 'current_phase', None) == 'vnf_deployment')
            return self.get_state(), self._r('timeout', in_vnf_phase=_is_vnf_phase), False, True, {'timeout': True}

        if self.env.current_phase == 'vnf_deployment':
            return self._handle_vnf_deployment(current_node, target_action, is_stay)
        elif self.env.current_phase == 'destination_connection':
            # [Task A] 确认 anchor 没有被中间逻辑清掉
            _anchor_check = getattr(self.env, 'current_anchor_node', None)
            _chain_check = getattr(self.env, 'chain_nodes', [])
            _last_vnf_check = _chain_check[-1] if _chain_check else None
            if not hasattr(self, '_anchor_log_last') or self._anchor_log_last != _anchor_check:
                logger.debug(
                    f"[AnchorConfirm] dest入口 anchor={_anchor_check} "
                    f"last_vnf={_last_vnf_check} cur={current_node}"
                )
                self._anchor_log_last = _anchor_check
            return self._handle_destination_connection(current_node, target_action, is_stay)

        return self.get_state(), self._r('penalty', type='internal_error'), True, False, {
            'fail': True, 'reason': 'unknown_phase'
        }

    # ==================================================================
    # VNF部署
    # ==================================================================
    def _handle_vnf_deployment(self, current_node, target_action, is_stay):
            target_goal = getattr(self.env, 'current_deployment_target', None)

            # [Fix 5] VNF目标节点资源预检查：避免一路走到资源不足DC再fail
            if target_goal is not None and current_node != target_goal and self.env.current_request:
                _vnf_idx  = getattr(self.env, 'next_vnf_idx', 0)
                _vnf_list = self.env.current_request.get('vnf', [])
                if _vnf_idx < len(_vnf_list):
                    _vnf_type = _vnf_list[_vnf_idx]
                    _cpu_list = self.env.current_request.get('cpu_origin', []) or \
                                self.env.current_request.get('vnf_cpu', [])
                    _mem_list = self.env.current_request.get('memory_origin', []) or \
                                self.env.current_request.get('vnf_mem', [])
                    _req_cpu = float(_cpu_list[_vnf_idx]) if _vnf_idx < len(_cpu_list) else 0.0
                    _req_mem = float(_mem_list[_vnf_idx]) if _vnf_idx < len(_mem_list) else 0.0
                    # [统一探针] probe_vnf_deploy 是单一真相，取代旧的 nodes_on_tree + check_node_resource
                    _probe = self.env.resource_mgr.probe_vnf_deploy(
                        target_goal, _vnf_type, _req_cpu, _req_mem
                    )
                    if not _probe['ok']:
                        logger.warning(
                            f"[VNF-TargetBlock] 目标DC不可部署，拒绝继续前往 | "
                            f"target={target_goal} vnf_type={_vnf_type} "
                            f"req_cpu={_req_cpu:.1f} req_mem={_req_mem:.1f} "
                            f"avail_cpu={self.env.resource_mgr.pool.get_available_cpu(target_goal):.1f} "
                            f"avail_mem={self.env.resource_mgr.pool.get_available_memory(target_goal):.1f} "
                            f"reuse={int(_probe['reuse'])} reason={_probe['reason']}"
                        )
                        self._reset_vnf_phase_only()
                        return self.get_state(), self._r('vnf_deploy', success=False), False, True, {
                            'deploy_fail': True,
                            'reason': _probe['reason'],
                            'target': target_goal
                        }

            if not is_stay:
                return self._handle_movement(current_node, target_action, target_goal)

            if target_goal is not None and current_node == target_goal:
                if hasattr(self.env, 'resource_mgr'):
                    self.env.resource_mgr.current_request = self.env.current_request
                    self.env.resource_mgr.current_tree = self.env.current_tree
                    self.env.resource_mgr.current_phase = self.env.current_phase
                    self.env.resource_mgr.next_vnf_idx = self.env.next_vnf_idx

                # [统一探针] allow_reuse 由 probe_vnf_deploy 决定，语义对齐高层 mask
                _vnf_idx_d = getattr(self.env, 'next_vnf_idx', 0)
                _vnf_list_d = self.env.current_request.get('vnf', []) if self.env.current_request else []
                _vnf_type_d = _vnf_list_d[_vnf_idx_d] if _vnf_idx_d < len(_vnf_list_d) else -1
                _cpu_list_d = self.env.current_request.get('cpu_origin', []) or \
                              self.env.current_request.get('vnf_cpu', []) \
                              if self.env.current_request else []
                _mem_list_d = self.env.current_request.get('memory_origin', []) or \
                              self.env.current_request.get('vnf_mem', []) \
                              if self.env.current_request else []
                _req_cpu_d = float(_cpu_list_d[_vnf_idx_d]) if _vnf_idx_d < len(_cpu_list_d) else 0.0
                _req_mem_d = float(_mem_list_d[_vnf_idx_d]) if _vnf_idx_d < len(_mem_list_d) else 0.0
                _probe_deploy = self.env.resource_mgr.probe_vnf_deploy(
                    target_goal, _vnf_type_d, _req_cpu_d, _req_mem_d
                )
                _allow_reuse = bool(_probe_deploy['reuse'])

                deploy_res = {}
                if hasattr(self.env.resource_mgr, '_try_deploy'):
                    deploy_res = self.env.resource_mgr._try_deploy(target_goal, allow_reuse=_allow_reuse)
                elif hasattr(self.env, '_try_deploy'):
                    deploy_res = self.env._try_deploy(target_goal)
                # 兼容旧接口（返回 bool）
                if isinstance(deploy_res, bool):
                    deploy_res = {'ok': deploy_res, 'reused': False, 'new_instance': not deploy_res, 'reason': ''}
                deploy_success = bool(deploy_res.get('ok', False))

                if deploy_success:
                    self.env._timeout_count = 0
                    self.env._island_count = 0
                    self.env._consecutive_timeout_count = 0
                    if not hasattr(self.env, 'chain_nodes'):
                        self.env.chain_nodes = []
                    self.env.chain_nodes.append(current_node)

                    # [改动2] 维护 node_stage：记录每个节点完成了几个 VNF
                    try:
                        if self.env.current_tree is None:
                            self.env.current_tree = {}
                        self.env.current_tree.setdefault('node_stage', {})
                        _deployed_stage = self.env.next_vnf_idx + 1
                        _prev_stage = self.env.current_tree['node_stage'].get(current_node, 0)
                        self.env.current_tree['node_stage'][current_node] = max(_prev_stage, _deployed_stage)
                        logger.debug(
                            f"[NodeStage] node={current_node} "
                            f"prev_stage={_prev_stage} "
                            f"new_stage={self.env.current_tree['node_stage'][current_node]} "
                            f"vnf_idx={self.env.next_vnf_idx}"
                        )
                    except Exception as _ns_e:
                        logger.debug(f"[NodeStage] 更新失败: {_ns_e}")

                    try:
                        import networkx as _nx
                        _sfc = getattr(self.env, 'current_sfc', None)
                        if _sfc is not None:
                            _prev = (self.env.current_request.get('source')
                                     if not _sfc['chain_nodes']
                                     else _sfc['chain_nodes'][-1])
                            _G_topo = _nx.DiGraph()
                            _bw = float(
                                self.env.current_request.get('bw_origin', 0.0)
                            )
                            _tree_edges = self.shared.get_positive_tree_edge_set()
                            for _u in range(self.env.n):
                                for _v in self.env.resource_mgr.get_neighbors(_u):
                                    if (
                                        (_u, _v) in _tree_edges
                                        or self.env.resource_mgr.pool.get_available_bandwidth(
                                            _u, _v
                                        ) >= _bw
                                    ):
                                        _G_topo.add_edge(_u, _v)
                            _executed_seg = list(getattr(
                                self.env, 'current_subgoal_full_path', []
                            ) or [])
                            if (_executed_seg and _executed_seg[0] == _prev
                                    and _executed_seg[-1] == current_node):
                                _seg = _executed_seg
                            elif _prev != current_node:
                                _seg = _nx.shortest_path(_G_topo, _prev, current_node)
                            else:
                                _seg = [current_node]
                            _sfc['spine_paths'].append(_seg)
                            _sfc['chain_nodes'].append(current_node)
                            logger.debug(f"[SFC-DAG] spine段: {_prev}→{current_node} = {_seg}")
                    except Exception as _e:
                        logger.warning(f"[SFC-DAG] spine记录失败: {_e}")

                    self.env.next_vnf_idx += 1
                    vnf_list = self.env.current_request.get('vnf', [])
                    current_count = self.env.next_vnf_idx
                    _dr = deploy_res.get('reused', False)
                    _dni = deploy_res.get('new_instance', False)

                    if current_count >= len(vnf_list):
                        self.env.sfc_upstream_nodes = set(self.env.chain_nodes[:-1])
                        info = {'phase': 'vnf_complete', 'all_vnf_deployed': True,
                                'deployed_count': current_count, 'total_vnf': len(vnf_list),
                                'deploy_reused': _dr, 'deploy_new_instance': _dni}
                        self._reset_vnf_phase_only()
                        return self.get_state(), self._r('vnf_deploy', success=True, all_complete=True), False, True, info
                    else:
                        info = {'vnf_deployed': True, 'vnf_idx': self.env.next_vnf_idx - 1,
                                'deployed_count': current_count, 'total_vnf': len(vnf_list),
                                'deploy_reused': _dr, 'deploy_new_instance': _dni}
                        return self.get_state(), self._r('vnf_deploy', success=True, all_complete=False), False, True, info
                else:
                    try:
                        avail_cpu = self.env.resource_mgr.pool.get_available_cpu(target_goal)
                        avail_mem = self.env.resource_mgr.pool.get_available_memory(target_goal)
                    except Exception:
                        avail_cpu, avail_mem = 0.0, 0.0

                    req_cpu, req_mem = 0.0, 0.0
                    if self.env.current_request:
                        vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
                        cpu_reqs = self.env.current_request.get('cpu_origin', [])
                        mem_reqs = self.env.current_request.get('memory_origin', [])
                        if vnf_idx < len(cpu_reqs): req_cpu = cpu_reqs[vnf_idx]
                        if vnf_idx < len(mem_reqs): req_mem = mem_reqs[vnf_idx]

                    self.env._timeout_count = getattr(self.env, '_timeout_count', 0) + 1
                    self.env._consecutive_timeout_count = self.env._timeout_count

                    # 全局DC资源快照
                    _dc_snapshot = []
                    try:
                        for _dn in sorted(getattr(self.env, 'dc_nodes', [])):
                            _dc_snapshot.append(
                                f"n{_dn}:cpu={self.env.resource_mgr.pool.get_available_cpu(_dn):.0f}"
                                f"/mem={self.env.resource_mgr.pool.get_available_memory(_dn):.0f}"
                            )
                    except Exception:
                        pass
                    logger.warning(
                        f"❌ [VNF-FAIL] 节点={target_goal} reason={deploy_res.get('reason','')} | "
                        f"可用CPU={avail_cpu:.1f} 需要={req_cpu:.1f} | "
                        f"可用MEM={avail_mem:.1f} 需要={req_mem:.1f} | "
                        f"连续失败={self.env._timeout_count} | DC资源={_dc_snapshot}"
                    )

                    info = {'deploy_fail': True, 'vnf_idx': self.env.next_vnf_idx,
                            'reason': deploy_res.get('reason', 'resource_insufficient'),
                            'avail_cpu': avail_cpu, 'avail_mem': avail_mem}
                    self._reset_vnf_phase_only()
                    if self.env._timeout_count >= 3:
                        self.env._timeout_count = 0
                        self.env._consecutive_timeout_count = 0
                        self._archive_episode_fail()
                        return self.get_state(), self._r('vnf_deploy', success=False), True, False, {**info, 'fail': True}
                    return self.get_state(), self._r('vnf_deploy', success=False), False, True, info
            else:
                return self.get_state(), self._r('timeout', in_vnf_phase=True), False, False, {'warning': 'wait_for_stay'}

    # ==================================================================
    # 目的地连接
    # ==================================================================
    def _handle_destination_connection(self, current_node, target_action, is_stay):
        target_goal = getattr(self.env, 'current_target_node', None)

        # 🚀 [修改2] 带宽孤岛检测：目标节点所有接入链路已满时，尝试绕路而非直接失败
        if target_goal is not None and current_node != target_goal:
            _bw_req = self.env.current_request.get('bw_origin', 0.0)
            _tree_edges = self.env.current_tree.get('tree', {})
            _target_alive = False
            _reachable_via_tree = False
            for _n in self.env.resource_mgr.get_neighbors(target_goal):
                # [Directed-BW] 检查 _n→target_goal 方向是否在树中
                _ek = (_n, target_goal)
                _in_tree = _ek in _tree_edges
                if _in_tree:
                    _target_alive = True
                    _reachable_via_tree = True
                    break
                if self.env.resource_mgr.pool.get_available_bandwidth(_n, target_goal) >= _bw_req:
                    _target_alive = True
                    break
            if not _target_alive:
                # [Monitor 10] BW 孤岛详细日志：打印目标节点各邻居 BW 值
                try:
                    _nbrs_bw = {
                        _nb: self.env.resource_mgr.pool.get_available_bandwidth(_nb, target_goal)
                        for _nb in self.env.resource_mgr.get_neighbors(target_goal)
                    }
                    logger.debug(
                        f"[BWIsland] target={target_goal} bw_req={_bw_req} "
                        f"neighbors_bw={_nbrs_bw} "
                        f"island_count={getattr(self.env, '_island_count', 0)}"
                    )
                except Exception:
                    pass
                # 尝试通过 compute_bw_aware_path 找带宽最宽的绕路（Widest Path 兜底）
                _fallback_path = self.compute_bw_aware_path(current_node, target_goal)
                if _fallback_path and len(_fallback_path) > 1:
                    # 找到绕路方案，不快速失败，继续让Agent移动
                    logger.debug(f"🔀 [Low] 带宽孤岛: target={target_goal} 直连已满，绕路方案: {_fallback_path}")
                else:
                    logger.debug(f"🏝️ [Low] 带宽孤岛: target={target_goal} 所有接入链路已满，快速失败")
                    # 🆕 不直接让整个Episode失败，改为跳过当前目标让高层重新调度
                    # ⚠️ [计数器拆分] 孤岛用独立的_island_count，阈值5，不与超时混用
                    self.env._island_count += 1
                    if self.env._island_count >= 5:
                        self.env._island_count = 0
                        self._archive_episode_fail()
                        return self.get_state(), self._r('penalty', type='invalid_link'), True, False, {
                            'fail': True, 'reason': 'bandwidth_exhausted'
                        }
                    # 截断当前子目标，让高层重选（Truncated=True）
                    return self.get_state(), self._r('penalty', type='bandwidth_island_retry'), False, True, {
                        'bandwidth_island': True,
                        'skipped_target': target_goal,
                        'reason': 'bandwidth_exhausted'
                    }

        # ── [DestFix v4.2] 每次切换到新dest目标时，优先从 anchor 出发，没有才回 last_vnf ──
        _chain = getattr(self.env, 'chain_nodes', [])
        _anchor = getattr(self.env, 'current_anchor_node', None)
        _last_vnf_node = _chain[-1] if _chain else None

        # [改动3] 使用 node_stage 做硬断言：anchor 必须是 full-stage 节点
        # 发现非法时立即修正 env.current_anchor_node，防止每步重复报警
        _total_vnf = len(self.env.current_request.get('vnf', [])) \
            if self.env.current_request else 0
        _node_stage = self.env.current_tree.get('node_stage', {}) \
            if self.env.current_tree else {}

        if _anchor is not None and _total_vnf > 0:
            _anchor_stage = _node_stage.get(_anchor, 0)
            if _anchor_stage < _total_vnf:
                logger.warning(
                    f"[AnchorInvalid] anchor={_anchor} stage={_anchor_stage} "
                    f"< total_vnf={_total_vnf}, fallback_to_last_vnf={_last_vnf_node}"
                )
                # 关键：同步修正 env，后续步不再重复报警
                self.env.current_anchor_node = _last_vnf_node
                _anchor = None  # 局部降级

        # 注意：不再用 chain_nodes 集合做二次检查。
        # chain_nodes 只记录 VNF 部署节点，branch 传播的 full-stage 节点不在其中。
        # node_stage 是唯一合法性依据。

        _reset_origin = _anchor if _anchor is not None else _last_vnf_node
        if _reset_origin is not None:
            _prev_target = getattr(self.env, '_last_dest_target', None)
            if _prev_target != target_goal:
                self.env.current_node_location = _reset_origin
                self.env.current_subgoal_full_path = [int(_reset_origin)]
                self.env._last_dest_target = target_goal
                self.env.current_path_trace = [_reset_origin]  # 完全清空禁忌表
                _origin_src = 'anchor' if _anchor is not None else 'last_vnf'
                logger.debug(
                    f"[DestFix] 目标{_prev_target}→{target_goal} | "
                    f"anchor={_anchor} last_vnf={_last_vnf_node} "
                    f"实际起点={_reset_origin}({_origin_src})")
                current_node = _reset_origin
                is_stay = (int(target_action) == current_node)
                # [Monitor3] AnchorCheck
                _conn_now = len(self.env.current_tree.get('connected_dests', set())) \
                    if self.env.current_tree else 0
                _dest_total = len(self.env.current_request.get('dest', [])) \
                    if self.env.current_request else 0
                logger.debug(
                    f"[AnchorCheck] target={target_goal} "
                    f"anchor={_anchor} last_vnf={_last_vnf_node} "
                    f"connected={_conn_now}/{_dest_total}"
                )
                # [Task 2] 第一个 dest 专项监控
                if _conn_now == 0:
                    _nodes_on_tree_now = len(getattr(self.env, 'nodes_on_tree', set()))
                    _req_id = self.env.current_request.get('id', '?') \
                        if self.env.current_request else '?'
                    logger.debug(
                        f"[firstDestCheck] req={_req_id} target={target_goal} "
                        f"anchor={_anchor} last_vnf={_last_vnf_node} "
                        f"nodes_on_tree={_nodes_on_tree_now} "
                        f"connected=0/{_dest_total}"
                    )
                # [改动3] 改用集合差统计：记录本轮 dest 开始前的正树边集合
                self.env._dest_start_pos_edges = self.shared.get_positive_tree_edge_set()
                self.env._dest_start_reuse = self.shared.get_reuse_edge_count()
                # [Task 2] 记录是否为首个 dest，供 DestDone 打 firstDestCost
                self.env._is_first_dest_cycle = (_conn_now == 0)
        # ──────────────────────────────────────────────────────────────────

        # ── 【修复5】最后一跳强制完成 ─────────────────────────────────────
        # 若 target 是当前邻居且带宽合法，直接把 target_action 改写为 target，
        # 不论策略实际输出了什么。这是执行高层命令，不是"引导"。
        if (
            target_goal is not None
            and current_node != target_goal
            and not is_stay
            and not list(getattr(
                self.env, '_active_destination_path', None
            ) or [])
        ):
            _bw_req_fc = self.env.current_request.get('bw_origin', 0.0) \
                if self.env.current_request else 0.0
            _neighbors_fc = self.env.resource_mgr.get_neighbors(current_node)
            _tree_edges_fc = self.shared.get_positive_tree_edge_set()
            if target_goal in _neighbors_fc:
                _legal_fc = (
                    self.shared.is_tree_edge(current_node, target_goal)
                    or self.env.resource_mgr.pool.get_available_bandwidth(
                        current_node, target_goal) >= _bw_req_fc
                )
                if _legal_fc and int(target_action) != target_goal:
                    logger.debug(
                        f"[LastHopForce] cur={current_node} target={target_goal} "
                        f"original_action={target_action} → forced to {target_goal}"
                    )
                    target_action = target_goal
                    is_stay = (target_goal == current_node)
        # ─────────────────────────────────────────────────────────────────

        if not is_stay:
            return self._handle_movement(current_node, target_action, target_goal)

        if current_node == target_goal:
            if 'connected_dests' not in self.env.current_tree:
                self.env.current_tree['connected_dests'] = set()

            if target_goal not in self.env.current_tree['connected_dests']:
                # [Fix] connected_dests.add 移到 branch 验证之后
                # 原来提前加入导致：branch 验证失败截断时目标已被计为"已连接"
                _dest_connected_ok = False
                _branch_root = None
                _bseg = []

                try:
                    import networkx as _nx
                    _sfc = getattr(self.env, 'current_sfc', None)
                    if _sfc is not None and _sfc['chain_nodes']:
                        _last_vnf = _sfc['chain_nodes'][-1]
                        if 'branch_roots' not in _sfc:
                            _sfc['branch_roots'] = {}

                        # [Fix A] branch root 用 node_stage 判断合法性，不用 chain_nodes。
                        _anchor_node = getattr(self.env, 'current_anchor_node', None)
                        _total_vnf_br = len(self.env.current_request.get('vnf', [])) \
                            if self.env.current_request else 0
                        _node_stage_br = self.env.current_tree.get('node_stage', {}) \
                            if self.env.current_tree else {}

                        _anchor_is_valid = (
                            _anchor_node is not None
                            and (_total_vnf_br == 0
                                 or _node_stage_br.get(_anchor_node, 0) >= _total_vnf_br)
                        )
                        if _anchor_is_valid:
                            _branch_root = _anchor_node
                        else:
                            _branch_root = _last_vnf
                            if _anchor_node is not None:
                                logger.debug(
                                    f"[BranchRoot] anchor={_anchor_node} "
                                    f"stage={_node_stage_br.get(_anchor_node,0)}/{_total_vnf_br} "
                                    f"→ fallback to last_vnf={_last_vnf}"
                                )
                        _sfc['branch_roots'][target_goal] = _branch_root

                        _bw = self.env.current_request.get('bw_origin', 0.0)
                        _G_topo = _nx.DiGraph()
                        for _u in range(self.env.n):
                            for _v in self.env.resource_mgr.get_neighbors(_u):
                                if (
                                    self.shared.is_tree_edge(_u, _v)
                                    or self.env.resource_mgr.pool.get_available_bandwidth(
                                        _u, _v
                                    ) >= _bw
                                ):
                                    _G_topo.add_edge(_u, _v)

                        # [清单 5] branch_path 优先只用已遍历过的树边，不走全拓扑捷径
                        _G_actual = _nx.DiGraph()
                        _G_actual.add_edges_from(list(self.shared.get_positive_tree_edge_set()))

                        # Prefer the complete path actually executed by the
                        # low-level policy during this destination cycle.
                        _executed_branch = list(getattr(
                            self.env, 'current_subgoal_full_path', []
                        ) or [])

                        # A policy may traverse an already committed directed
                        # prefix more than once before reaching the target.  The
                        # canonical branch evidence is the loop-erased walk:
                        # every retained hop was physically executed, while no
                        # synthetic topology edge is introduced.
                        _loop_erased_branch = []
                        _loop_erased_pos = {}
                        for _node in _executed_branch:
                            if _node in _loop_erased_pos:
                                _keep = _loop_erased_pos[_node]
                                for _removed in _loop_erased_branch[_keep + 1:]:
                                    _loop_erased_pos.pop(_removed, None)
                                _loop_erased_branch = _loop_erased_branch[:_keep + 1]
                            else:
                                _loop_erased_pos[_node] = len(_loop_erased_branch)
                                _loop_erased_branch.append(_node)
                        _bseg = None
                        _branch_source = None

                        def _valid_executed_branch(path):
                            if not path:
                                return False
                            if path[0] != _branch_root or path[-1] != target_goal:
                                return False
                            if len(path) != len(set(path)):
                                return False
                            for _u, _v in zip(path, path[1:]):
                                if _v not in self.env.resource_mgr.get_neighbors(_u):
                                    return False
                            return True

                        if _valid_executed_branch(_executed_branch):
                            _bseg = _executed_branch
                            _branch_source = 'executed'
                        elif _valid_executed_branch(_loop_erased_branch):
                            _bseg = _loop_erased_branch
                            _branch_source = 'executed_loop_erased'

                        # Fallback to structural evidence for legacy callers.
                        try:
                            if _bseg is None:
                                _bseg = _nx.shortest_path(
                                    _G_actual, _branch_root, target_goal
                                )
                                _branch_source = 'actual_tree'
                        except (_nx.NetworkXNoPath, _nx.NodeNotFound):
                            pass

                        # fallback: last_vnf → dest（用实际树）
                        if _bseg is None and _branch_root != _last_vnf:
                            try:
                                _bseg = _nx.shortest_path(_G_actual, _last_vnf, target_goal)
                                _branch_root = _last_vnf
                                _branch_source = 'last_vnf_actual'
                            except (_nx.NetworkXNoPath, _nx.NodeNotFound):
                                pass

                        # Reaching a destination must be backed by edges that
                        # were actually traversed.  Synthesizing a full-topology
                        # path here changes the action after execution and can
                        # insert non-physical or unallocated edges into the SFT.
                        if _bseg is None:
                            raise ValueError(
                                f'no_executed_branch:{_branch_root}->{target_goal}; '
                                f'executed={_executed_branch}'
                            )

                        if not _valid_executed_branch(_bseg):
                            raise ValueError(
                                f'invalid_executed_branch:{_branch_root}->{target_goal}; '
                                f'source={_branch_source}; path={_bseg}'
                            )

                        _sfc['branch_paths'][target_goal] = _bseg
                        self.env._last_effective_anchor = _branch_root
                        # [改动4] branch_path 可视化 + root_stage 验证
                        _chain_set_log = set(getattr(self.env, 'chain_nodes', []))
                        _bseg_covers_vnf = [n for n in _bseg if n in _chain_set_log]
                        _via = _branch_source
                        _node_stage_map = self.env.current_tree.get('node_stage', {}) \
                            if self.env.current_tree else {}
                        _total_vnf_log = len(self.env.current_request.get('vnf', [])) \
                            if self.env.current_request else 0
                        _root_stage = _node_stage_map.get(_branch_root, 0)
                        logger.debug(
                            f"[BranchPathCheck] target={target_goal} "
                            f"root={_branch_root} root_stage={_root_stage}/{_total_vnf_log} "
                            f"path_len={len(_bseg)} path_nodes={_bseg} via={_via} "
                            f"covers_vnf={_bseg_covers_vnf}"
                        )
                        logger.debug(
                            f"[BranchVNFCheck] target={target_goal} "
                            f"branch_root={_branch_root} "
                            f"covers_vnf={_bseg_covers_vnf if _bseg_covers_vnf else 'empty(OK if propagated)'}"
                        )
                        _anchor_on_tree = _G_actual.has_node(_branch_root)
                        _root_is_vnf_deployed = _branch_root in _chain_set_log
                        if not _bseg_covers_vnf and _chain_set_log and not _anchor_on_tree:
                            logger.warning(
                                f"[BranchVNFCheck] ⚠️ branch root 不在树上且未经过 VNF! "
                                f"target={target_goal} root={_branch_root} "
                                f"root_stage={_root_stage}/{_total_vnf_log} "
                                f"path={_bseg} chain_nodes={list(_chain_set_log)}"
                            )
                        elif not _bseg_covers_vnf and _chain_set_log and _anchor_on_tree:
                            logger.debug(
                                f"[BranchVNFCheck] propagated-anchor branch (正常): "
                                f"root={_branch_root} on_tree=True "
                                f"root_is_vnf={_root_is_vnf_deployed}"
                            )

                        # branch_path 边写入 tree（仅记录，不即时扣BW）
                        for _j in range(len(_bseg) - 1):
                            # [Directed-BW] branch路径按走向方向记录有向键
                            _ek = (_bseg[_j], _bseg[_j+1])
                            if 'tree' not in self.env.current_tree:
                                self.env.current_tree['tree'] = {}
                            if _ek not in self.env.current_tree['tree']:
                                self.env.current_tree['tree'][_ek] = -1.0

                        # 只有 branch 路径构建成功才提交连接状态
                        _dest_connected_ok = True

                except Exception as _e:
                    logger.warning(f"[SFC-DAG] branch记录失败: {_e}")
                    self.env.current_target_node = None
                    self.env._dest_cycle_armed = False
                    self.env._dest_step0_checked = False
                    self.env._dest_anchor_mismatch = False
                    return self.get_state(), self._r(
                        'penalty', type='invalid_link'
                    ), False, True, {
                        'branch_path_invalid': True,
                        'reason': 'branch_path_invalid',
                        'target': target_goal,
                    }

                if _dest_connected_ok:
                    self.env.current_tree['connected_dests'].add(target_goal)
                    # [Fix-E] dest成功连通时部分重置timeout计数
                    # 原来dest成功后计数不变，导致前面的截断积累会让后续dest更快触发终止
                    # 修复：成功连通一个dest后减半计数（不清零，保留历史记忆）
                    self.env._timeout_count = max(0, self.env._timeout_count - 2)
                    self.env._consecutive_timeout_count = self.env._timeout_count
                    # ── 同步写入新内核 RequestRecord.connected_dests ──────
                    req_id = self.env.current_request.get('id') if self.env.current_request else None
                    if req_id is not None and hasattr(self.env.resource_mgr, 'mark_dest_connected'):
                        self.env.resource_mgr.mark_dest_connected(req_id, target_goal)

                # [改动4b] full-stage 沿成功 branch 传播
                try:
                    if _branch_root is not None and _bseg:
                        self._propagate_full_stage_on_branch(_branch_root, _bseg)
                except Exception as _pe:
                    logger.debug(f"[StagePropagate] 失败: {_pe}")

                # [改动4c] DestDone：改用集合差统计本轮真正新增正树边
                try:
                    _after_pos_edges = self.shared.get_positive_tree_edge_set()
                    _before_pos_edges = getattr(self.env, '_dest_start_pos_edges', set())
                    _new_edge_set = _after_pos_edges - _before_pos_edges
                    _delta_edges = len(_new_edge_set)

                    _after_reuse = self.shared.get_reuse_edge_count()
                    _delta_reuse = _after_reuse - getattr(self.env, '_dest_start_reuse', _after_reuse)

                    _effective_anchor = _branch_root  # 实际执行起点
                    logger.debug(
                        f"[DestDone] target={target_goal} "
                        f"anchor={_effective_anchor} "
                        f"steps={getattr(self.env, 'subgoal_step_count', 0)} "
                        f"new_edges={_delta_edges} reuse_edges={_delta_reuse}"
                    )
                    if getattr(self.env, '_is_first_dest_cycle', False):
                        logger.debug(
                            f"[firstDestCost] target={target_goal} "
                            f"anchor={_effective_anchor} "
                            f"steps={getattr(self.env, 'subgoal_step_count', 0)} "
                            f"new_edges={_delta_edges} reuse_edges={_delta_reuse}"
                        )
                        # [Log 2] FirstDestDetail — 计算 detour_ratio
                        try:
                            _shortest_hop = self.shared.get_hop_distance_lazy(_effective_anchor, target_goal)
                            _steps_fd = getattr(self.env, 'subgoal_step_count', 0)
                            _detour_ratio = (_steps_fd / max(1, _shortest_hop)
                                             if _shortest_hop > 0 and _shortest_hop < 9999
                                             else -1.0)
                            logger.debug(
                                f"[FirstDestDetail] "
                                f"req={self.env.current_request.get('id','?')} "
                                f"target={target_goal} anchor={_effective_anchor} "
                                f"root_stage={_root_stage}/{_total_vnf_log} "
                                f"steps={_steps_fd} "
                                f"new_edges={_delta_edges} reuse_edges={_delta_reuse} "
                                f"path_len={len(_bseg)} "
                                f"shortest_hop={_shortest_hop} "
                                f"detour_ratio={_detour_ratio:.2f}"
                            )
                            # 把关键字段写回 env，供 EarlyDiagEpisode / HRL_Coordinator 读取
                            self.env._first_dest_steps = _steps_fd
                            self.env._first_dest_new_edges = _delta_edges
                            self.env._first_dest_reuse_edges = _delta_reuse
                            self.env._first_dest_shortest_hop = _shortest_hop
                            self.env._first_dest_detour_ratio = _detour_ratio
                        except Exception as _fdd_e:
                            logger.debug(f"[FirstDestDetail] 统计失败: {_fdd_e}")
                        self.env._is_first_dest_cycle = False
                    # 更新基准，供下一个 dest 使用
                    self.env._dest_start_pos_edges = _after_pos_edges
                    self.env._dest_start_reuse = _after_reuse
                except Exception as _de:
                    logger.debug(f"[DestDone] 统计失败: {_de}")

            steps_used = getattr(self.env, 'subgoal_step_count', 50)
            _tgt_node = getattr(self.env, 'current_target_node', None)
            if _tgt_node is not None:
                _min_hops = self.shared.get_hop_distance_lazy(self.env.current_node_location, _tgt_node) + 1
            else:
                _min_hops = 1
            step_reward = max(20.0 - max(0, steps_used - _min_hops) * 1.5, -5.0)

            try:
                all_dests = set(int(x) for x in self.env.current_request.get('dest', []))
                connected = set(int(x) for x in self.env.current_tree.get('connected_dests', set()))
            except:
                all_dests, connected = set(), set()

            if all_dests.issubset(connected) and len(all_dests) > 0:
                logger.debug(f"✅ [Episode完成] 多播树建树成功！ connected={len(connected)}/{len(all_dests)}")

                # 🔥 排查白嫖Bug：检查树边数是否异常累积
                _active_edges = {e: f for e, f in self.env.current_tree.get('tree', {}).items() if f > 0.0}
                logger.debug(f"[BugCheck] 请求{self.env.current_request.get('id','?')} 建树完成，"
                            f"flow>0边数={len(_active_edges)} 总树边数={len(self.env.current_tree.get('tree', {}))}")

                # 🔥 排查原地打包：检查Spine实际跳数
                _sfc = getattr(self.env, 'current_sfc', None)
                _spine_hops = sum(len(p)-1 for p in _sfc.get('spine_paths', [])) if _sfc else 0
                _branch_hops = sum(len(p)-1 for p in _sfc.get('branch_paths', {}).values()) if _sfc else 0
                logger.debug(f"[BugCheck] Spine跳数={_spine_hops} Branch跳数={_branch_hops} "
                            f"chain_nodes={getattr(self.env, 'chain_nodes', [])}")

                self.env._consecutive_timeout_count = 0
                self.env._timeout_count = 0
                self.env._island_count = 0
                self.env._bw_fail_count = 0
                # 成功只能在资源账本、SFT 快照和 lifecycle 全部提交后归档计数。
                if not self._finalize_episode_success():
                    _finalize_reason = getattr(
                        self.env, '_success_finalize_reason', 'success_finalize_failed'
                    )
                    logger.warning(
                        f"[Episode完成] req={self.env.current_request.get('id', '?')} "
                        f"成功收尾失败({_finalize_reason})，部署已回滚"
                    )
                    return self.get_state(), self._r('penalty', type='invalid_link'), False, True, {
                        'deployment_failed': True,
                        'reason': _finalize_reason,
                    }

                # [改动5 / Monitor A] EpisodeStruct：使用辅助方法统一口径
                try:
                    _flow_edges  = len(self.shared.get_positive_tree_edge_set())
                    _reuse_edges = self.shared.get_reuse_edge_count()
                    # sharing_eff：唯一复用边占比，≤ 1.0
                    _sharing_eff = min(1.0, _reuse_edges / max(1, _flow_edges))
                    # reuse_density：额外复用次数 / 唯一正树边，体现共享深度，可 > 1
                    _tree_usage = self.env.current_tree.get('tree_usage', {}) \
                        if self.env.current_tree else {}
                    _extra_reuse = sum(max(0, v - 1) for _, v in _tree_usage.items())
                    _reuse_density = _extra_reuse / max(1, _flow_edges)
                    logger.debug(
                        f"[EpisodeStruct] req={self.env.current_request.get('id','?')} "
                        f"flow_edges={_flow_edges} reuse_edges={_reuse_edges} "
                        f"sharing_eff={_sharing_eff:.3f} "
                        f"reuse_density={_reuse_density:.3f}"
                    )
                except Exception as _se:
                    logger.debug(f"[EpisodeStruct] 统计失败: {_se}")

                # ── [带宽消耗统计] Episode成功完成时打印 ──────────────────────────
                try:
                    _bw_req = self.env.current_request.get('bw_origin', 0.0)
                    _tree_edges = self.env.current_tree.get('tree', {})
                    # ⚠️ [统计修复] branch_path边写入tree时flow=0.0（仅记录路径，未实际分配BW）
                    # 只统计flow>0的边，避免高估带宽消耗
                    _active_edges = {e: f for e, f in _tree_edges.items() if f > 0.0}
                    _n_edges = len(_active_edges)
                    _total_bw_consumed = _bw_req * _n_edges

                    # 统计每条边的剩余带宽（仅实际分配的边）
                    _edge_details = []
                    for (_u, _v), _ratio in _active_edges.items():
                        try:
                            _avail = self.env.resource_mgr.pool.get_available_bandwidth(_u, _v)
                            _edge_details.append(f"({_u}-{_v}: 剩余{_avail:.1f})")
                        except Exception:
                            _edge_details.append(f"({_u}-{_v}: 查询失败)")

                    # 全局带宽利用率
                    _total_avail = 0.0
                    _total_cap = 0.0
                    _default_cap = getattr(self.env.resource_mgr, 'BW_cap',
                                          getattr(self.env.resource_mgr, 'bw_cap_default', 100.0))
                    for _u in range(self.env.n):
                        for _v in self.env.resource_mgr.get_neighbors(_u):
                            if _u < _v:
                                try:
                                    # [Directed-BW] 双向链路各自独立，统计时需合并两个方向
                                    _avail_uv = self.env.resource_mgr.pool.get_available_bandwidth(_u, _v)
                                    _avail_vu = self.env.resource_mgr.pool.get_available_bandwidth(_v, _u)
                                    _cap_uv = self.env.resource_mgr.pool.bw_cap.get((_u, _v), _default_cap)
                                    _cap_vu = self.env.resource_mgr.pool.bw_cap.get((_v, _u), _default_cap)
                                    _total_avail += _avail_uv + _avail_vu
                                    _total_cap   += _cap_uv   + _cap_vu
                                except Exception:
                                    pass
                    _global_util = (1.0 - _total_avail / max(1.0, _total_cap)) * 100.0

                #     logger.debug(
                #         f"📊 [BW统计] 请求bw={_bw_req:.1f} | 树边数={_n_edges} | "
                #         f"本次请求消耗带宽≈{_total_bw_consumed:.1f} | "
                #         f"全局BW利用率={_global_util:.1f}% | "
                #         f"树边剩余BW: {' '.join(_edge_details)}"
                #     )
                except Exception as _bw_e:
                    logger.warning(f"⚠️ [BW统计] 统计失败: {_bw_e}")
                # ──────────────────────────────────────────────────────────────

                _sfc = getattr(self.env, 'current_sfc', None)
                _chain = getattr(self.env, 'chain_nodes', [])
                self.env._dest_cycle_armed = False
                self.env._dest_step0_checked = False
                self.env._dest_anchor_mismatch = False
                return self.get_state(), self._r('tree_connect',
                    is_complete=True, connected_count=len(connected), total_dests=len(all_dests)), True, False, {
                    'episode_complete': True, 'all_destinations_connected': True,
                    'success': True, 'connected_count': len(connected),
                    'chain_nodes': _chain
                }
            else:
                self.env.subgoal_step_count = 0
                self.env.current_target_node = None
                self.env._dest_cycle_armed = False
                self.env._dest_step0_checked = False
                self.env._dest_anchor_mismatch = False

                if hasattr(self.env, 'chain_nodes') and len(self.env.chain_nodes) > 0:
                    self.env._need_reset_to_last_vnf = True
                if hasattr(self.env, 'current_path_trace'):
                    # [Task2] 切换dest目标时清空禁忌表，以 anchor 为起点（没有则用 last_vnf）
                    _anchor_node = getattr(self.env, 'current_anchor_node', None)
                    _fallback_origin = self.env.chain_nodes[-1] if self.env.chain_nodes else None
                    _origin = _anchor_node if _anchor_node is not None else _fallback_origin
                    if _origin is not None:
                        self.env.current_path_trace = [_origin]
                    else:
                        self.env.current_path_trace = []
# 同时清除_last_dest_target，确保DestFix在新目标上强制瞬移
                self.env._last_dest_target = None
                self.env.hard_tabu_list = set()  # 👈 【新增】子目标终止时清空强禁忌表

                return self.get_state(), self._r('tree_connect',
                    is_complete=False, connected_count=len(connected), total_dests=len(all_dests)), False, True, {'dest_connected': True}
        else:
            return self.get_state(), self._r('timeout', in_vnf_phase=False), False, False, {'warning': 'wait_for_stay'}

    # ==================================================================
    # 移动处理（带共享边奖励逻辑 & 陡峭梯度）
    # ==================================================================
    def _handle_movement(self, current_node, target_action, target_goal):
        next_node = int(target_action)

        if next_node == current_node:
            if current_node != target_goal:
                neighbors = self.env.resource_mgr.get_neighbors(current_node)
                bw_req = self.env.current_request.get('bw_origin', 0.0)
                _tree_fm = (self.env.current_tree or {}).get('tree', {})
                valid_neighbors = [
                    n for n in neighbors
                    if self.env.resource_mgr.pool.get_available_bandwidth(current_node, n) >= bw_req
                    or _tree_fm.get((current_node, n), 0.0) > 0.0
                ]
                if not valid_neighbors:
                    self._archive_episode_fail()
                    return self.get_state(), self._r('penalty', type='invalid_link'), True, False, {
                        'fail': True, 'reason': 'trapped'
                    }
                return self.get_state(), self._r('penalty', type='invalid_action'), False, False, {'warning': 'stay'}

        # The mask is advisory to the policy; the executor remains the final
        # authority.  Never allow a stale/misaligned mask to turn an arbitrary
        # node id into a physical hop.
        directed_neighbors = {
            int(node) for node in self.env.resource_mgr.get_neighbors(current_node)
        }
        if next_node not in directed_neighbors:
            logger.warning(
                f"[DirectedEdgeReject] non-neighbor action "
                f"{current_node}->{next_node}; neighbors={sorted(directed_neighbors)}"
            )
            return self.get_state(), self._r(
                'penalty', type='invalid_link'
            ), False, False, {
                'error': 'non_neighbor_action',
                'reason': 'invalid_directed_edge',
                'edge': (current_node, next_node),
            }

        bw_req = self.env.current_request.get('bw_origin', 0.0)
        # [Directed-BW] 有向键：(current_node, next_node)，双向链路方向独立
        edge_key = (current_node, next_node)
        _tree = self.env.current_tree.get('tree', {})
        _tree_val     = _tree.get(edge_key,     None)

        # Directed-BW: (u→v) and (v→u) have independent capacities.
        # flow<=0 records are only structural placeholders; when physically traversed,
        # the directed edge must be upgraded to flow=1.0 and charged on commit.
        is_new_edge = _tree_val is None or float(_tree_val) <= 0.0

        if is_new_edge:
            try:
                avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, next_node)
                has_bw = avail_bw >= bw_req
            except:
                has_bw = False
            if not has_bw:
                self.env._bw_fail_count = getattr(self.env, '_bw_fail_count', 0) + 1
                if self.env._bw_fail_count >= 10:
                    self.env._bw_fail_count = 0
                    self._archive_episode_fail()
                    return self.get_state(), self._r('penalty', type='invalid_link'), True, False, {
                        'fail': True, 'reason': 'bandwidth_exhausted'
                    }
                return self.get_state(), self._r('penalty', type='invalid_link'), False, False, {'error': 'no_bandwidth'}

        _phase = getattr(self.env, 'current_phase', None)
        _is_vnf_phase = (_phase == 'vnf_deployment')

        # 先计算距离，供后续奖励使用
        dist_before = None
        dist_after = None
        if target_goal is not None:
            try:
                dist_before = self.shared.get_hop_distance_lazy(current_node, target_goal)
                dist_after  = self.shared.get_hop_distance_lazy(next_node,    target_goal)
            except Exception:
                pass

        if is_new_edge:
            action_type = "NewPath"
            if _is_vnf_phase:
                # vnf_move: to_dc, valid_link, guidance_val(靠近/远离目标的跳数变化)
                _guidance = 0.0
                if dist_before is not None and dist_after is not None and dist_before < 9999 and dist_after < 9999:
                    _guidance = float(dist_before - dist_after) * self.env.reward_critic.params.guidance_closer_rate \
                        if hasattr(self.env, 'reward_critic') and self.env.reward_critic else float(dist_before - dist_after)
                _to_dc = (next_node in getattr(self.env, 'dc_nodes', set()))
                reward = self._r('vnf_move', to_dc=_to_dc, valid_link=True, guidance_val=_guidance)
            else:
                # tree移动: 目标dest, 链路有效性, 前/后最短距离
                _to_dest = (next_node == target_goal)
                _db = dist_before if (dist_before is not None and dist_before < 9999) else 999
                _da = dist_after  if (dist_after  is not None and dist_after  < 9999) else 999
                reward = self._r('tree_move', to_dest=_to_dest, valid_link=True,
                                 min_dist_before=_db, min_dist_after=_da)
            # 热点边额外惩罚
            try:
                _cap = self.env.resource_mgr.pool.bw_cap.get(
                    (current_node, next_node), 100.0)
                _avail = self.env.resource_mgr.pool.get_available_bandwidth(current_node, next_node)
                _util = 1.0 - _avail / max(1.0, _cap)
                if _util > 0.7:
                    reward += self._r('hotspot', util=_util)
                    action_type = f"NewPath_Hotspot({_util:.0%})"
            except Exception:
                pass
            # [压树长] 新增树边额外成本，与move_cost叠加，让Agent倾向复用已有边
            reward += 0.5 * self._r('new_edge')  # 减半：避免与 move_cost 双重惩罚
        else:
            # 复用边：只扣步数成本，无额外奖励
            reward = self._r('penalty', type='move_cost')
            action_type = "Reuse"

        # 等距振荡惩罚（guidance_idle_penalty）
        if dist_before is not None and dist_after is not None and dist_after == dist_before and next_node != current_node:
            reward += self._r('penalty', type='guidance_idle_penalty')
            action_type = "Oscillation"

        # ── 【修复4】destination 阶段偏离 subgoal 强负奖励 ──────────────────
        # 存在更近候选但没有选择，或 target 就在邻居里却没走 → 额外惩罚。
        if (
            _phase == 'destination_connection'
            and target_goal is not None
            and dist_before is not None
            and dist_after is not None
            and dist_before < 9999
        ):
            _neighbors_now = self.env.resource_mgr.get_neighbors(current_node)
            _closer_existed = any(
                self.shared.get_hop_distance_lazy(n, target_goal) < dist_before
                for n in _neighbors_now
            )
            _target_was_neighbor = (target_goal in _neighbors_now)

            if _target_was_neighbor and next_node != target_goal:
                # 最严重：目标就在一跳内却没走
                reward -= 8.0
                action_type = "DeviateFromTarget_Neighbor"
                if np.random.rand() < 0.05:
                    logger.debug(
                        f"[DeviatePenalty-Hard] target={target_goal} was neighbor "
                        f"cur={current_node} chose={next_node} penalty=-8.0"
                    )
            elif _closer_existed and dist_after >= dist_before:
                # 次严重：存在更近候选却没有缩短距离
                reward -= 3.0
                action_type = "DeviateFromTarget_NotCloser"
                if np.random.rand() < 0.02:
                    logger.debug(
                        f"[DeviatePenalty-Mid] target={target_goal} closer_existed "
                        f"cur={current_node} chose={next_node} "
                        f"d_before={dist_before} d_after={dist_after} penalty=-3.0"
                    )
        if not hasattr(self.env, '_last_target_goal'):
            self.env._last_target_goal = None
            self.env._stuck_steps = 0
            self.env._last_dist_to_target = None
        if target_goal != self.env._last_target_goal:
            self.env._stuck_steps = 0
            self.env._last_dist_to_target = None
            self.env._last_target_goal = target_goal
            self.env.hard_tabu_list = set()
        if target_goal is not None and dist_after is not None:
            if self.env._last_dist_to_target is not None:
                if dist_after >= self.env._last_dist_to_target:
                    self.env._stuck_steps += 1
                else:
                    self.env._stuck_steps = 0
            self.env._last_dist_to_target = dist_after
            if self.env._stuck_steps >= 5:
                reward += self._r('penalty', type='wrong_position')
                self.env._stuck_steps = 0
                action_type = "StuckPenalty"
        else:
            self.env._stuck_steps = 0
            self.env._last_dist_to_target = None

        if not hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        # 回访惩罚（freq_penalty_rate）
        if next_node in self.env.current_path_trace:
            reward += self._r('penalty', type='freq_penalty_rate')
            action_type = "Fallback_Revisit"

        if is_new_edge:
            # [delayed-commit] 搜索阶段不扣BW，只记录边到current_tree
            # BW统一在_commit_episode_bandwidth()成功后一次性提交
            # [Directed-BW] tree 键已经是有向 (current_node, next_node)，无需 directed_edges
            if 'tree' not in self.env.current_tree:
                self.env.current_tree['tree'] = {}
            _phase = getattr(self.env, 'current_phase', None)
            _dests = set()
            if self.env.current_request:
                _dests = set(int(d) for d in self.env.current_request.get('dest', []))
            _skip_edge = (_phase == 'vnf_deployment' and next_node in _dests)

            _total_vnf = len(self.env.current_request.get('vnf', [])) \
                if self.env.current_request else 0
            _node_stage = self.env.current_tree.get('node_stage', {}) \
                if self.env.current_tree else {}
            _full_stage_fanout = (
                _phase == 'destination_connection'
                and current_node in _dests
                and _node_stage.get(current_node, 0) >= _total_vnf
            )
            _skip_dest_outgoing = (
                _phase == 'destination_connection'
                and current_node in _dests
                and not _full_stage_fanout
            )
            _skip_vnf_through_dest = (
                _phase == 'vnf_deployment' and
                current_node in _dests
            )

            if not _skip_edge and not _skip_dest_outgoing and not _skip_vnf_through_dest:
                # 硬防环：加入前检查两端是否在flow>0正树里已连通
                if self._would_create_cycle(current_node, next_node):
                    logger.debug(f"[CycleBlock] 边({current_node},{next_node})会成环，拒绝写入flow=1.0")
                    return self.get_state(), self._r('penalty', type='invalid_link'), False, False, {
                        'error': 'would_create_cycle', 'reason': 'cycle_blocked'
                    }
                self.env.current_tree['tree'][edge_key] = 1.0
                self.env.nodes_on_tree.add(current_node)
                self.env.nodes_on_tree.add(next_node)
            elif _skip_dest_outgoing or _skip_vnf_through_dest:
                self.env.current_tree['tree'][edge_key] = 0.0
                logger.debug(f"[DestFix] dest节点{current_node}出边仅记录，不即时扣BW({current_node}→{next_node})")
            else:
                self.env.current_tree['tree'][edge_key] = 0.0
                logger.debug(f"[SFC修复] spine边({current_node},{next_node})经过dest节点，仅记录，不即时扣BW")

        if 'tree_usage' not in self.env.current_tree:
            self.env.current_tree['tree_usage'] = {}
        self.env.current_tree['tree_usage'][edge_key] = self.env.current_tree['tree_usage'].get(edge_key, 0) + 1

        # Commit the physical position only after every structural gate above
        # accepts the edge.  Moving earlier leaves location and path evidence
        # inconsistent when cycle validation returns without appending a step.
        self.env.current_node_location = next_node
        self.env.current_path_trace.append(next_node)
        if (
            not hasattr(self.env, 'current_subgoal_full_path')
            or not self.env.current_subgoal_full_path
        ):
            self.env.current_subgoal_full_path = [current_node]
        elif int(self.env.current_subgoal_full_path[-1]) != int(current_node):
            logger.warning(
                f"[PathTraceResync] trace_tail="
                f"{self.env.current_subgoal_full_path[-1]} actual={current_node}; "
                f"reset before appending {next_node}"
            )
            self.env.current_subgoal_full_path = [current_node]
        self.env.current_subgoal_full_path.append(next_node)
        _phase_now = getattr(self.env, 'current_phase', None)
        if _phase_now == 'destination_connection':
            # [Fix-Tabu-Len] 降低tabu上限：10→6，最后1个dest时→4
            # 原来10步tabu在单路径网络中会把整条路封死（路径上只有6-8个节点）
            # 降低后tabu更快轮转，减少封死单路径的概率
            _remaining_tabu = 99
            if self.env.current_request and self.env.current_tree:
                _all_d_t = len(self.env.current_request.get('dest', []))
                _done_d_t = len(self.env.current_tree.get('connected_dests', set()))
                _remaining_tabu = _all_d_t - _done_d_t
            _max_tabu = 4 if _remaining_tabu <= 1 else 6
        else:
            _max_tabu = 4   # VNF 阶段：短 tabu，防成环但不压死探索
        if len(self.env.current_path_trace) > _max_tabu:
            self.env.current_path_trace = self.env.current_path_trace[-_max_tabu:]

        # [SoftTabu] 同步维护有向边历史，供 mask 里区分"反向抖动边"与"合理回访"
        if not hasattr(self.env, 'recent_edge_trace'):
            self.env.recent_edge_trace = []
        self.env.recent_edge_trace.append((current_node, next_node))
        if len(self.env.recent_edge_trace) > 8:
            self.env.recent_edge_trace = self.env.recent_edge_trace[-8:]

        return self.get_state(), reward, False, False, {'moved': True, 'type': action_type}

    # ==================================================================
    # 动作掩码 (DAG 掩码 + 死路软释放)
    # ==================================================================
    def get_low_level_action_mask(self, mutate_env=True):
        mask = np.zeros(self.env.n, dtype=np.float32)
        current = self.env.current_node_location
        neighbors = self.env.resource_mgr.get_neighbors(current)

        bw_req = 0.0
        if self.env.current_request:
            bw_req = self.env.current_request.get('bw_origin', 0.0)

        tree_edges = self.shared.get_positive_tree_edge_set()
        phase = getattr(self.env, 'current_phase', None)
        _raw_nbrs = len(neighbors)

        # A destination branch must start after the final VNF and may never
        # re-enter the source-to-final-VNF spine.  Keep this as a hard guard:
        # several recovery paths below rebuild the mask from scratch.
        _dest_order_blocked = set()
        if phase == 'destination_connection':
            _sfc_for_mask = getattr(self.env, 'current_sfc', None) or {}
            for _path in _sfc_for_mask.get('spine_paths', []) or []:
                _dest_order_blocked.update(int(node) for node in (_path or []))
            _chain_for_mask = list(
                _sfc_for_mask.get('chain_nodes', [])
                or getattr(self.env, 'chain_nodes', [])
                or []
            )
            if _chain_for_mask:
                _dest_order_blocked.discard(int(_chain_for_mask[-1]))

        def _apply_destination_order_guard(candidate_mask):
            if phase != 'destination_connection':
                return candidate_mask
            for node in _dest_order_blocked:
                if 0 <= node < len(candidate_mask):
                    candidate_mask[node] = 0.0
            return candidate_mask

        # A VNF-stage path may only extend the existing directed SFC spine.
        # Destinations are not legal transit nodes before the final VNF, and
        # re-entering an older tree node would create a cycle or second parent.
        _vnf_order_blocked = set()
        if phase == 'vnf_deployment':
            if self.env.current_request:
                _vnf_order_blocked.update(
                    int(node) for node in self.env.current_request.get('dest', [])
                )
            for u, v in tree_edges:
                _vnf_order_blocked.update((int(u), int(v)))
            _vnf_order_blocked.discard(int(current))

        def _apply_vnf_order_guard(candidate_mask):
            if phase != 'vnf_deployment':
                return candidate_mask
            stage_target = getattr(self.env, 'current_deployment_target', None)
            for node in _vnf_order_blocked:
                if stage_target is None or int(node) != int(stage_target):
                    if 0 <= int(node) < len(candidate_mask):
                        candidate_mask[int(node)] = 0.0
            return candidate_mask

        # 1. 带宽基础筛选
        _tree_dict_mask = (self.env.current_tree or {}).get('tree', {})
        _hard_tabu = getattr(self.env, 'hard_tabu_list', set())  # 👈 【新增】获取当前强禁忌表

        for nbr in neighbors:
            if nbr in _hard_tabu:
                continue  # 👈 【新增】如果邻居在永久黑名单中，绝对不开放

            if self.shared.is_tree_edge(current, nbr):
                mask[nbr] = 1.0  # 正向树边：无条件开放
            else:
                avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                if avail_bw >= bw_req:
                    mask[nbr] = 1.0
        _apply_vnf_order_guard(mask)
        _apply_destination_order_guard(mask)
        _bw_ok = int(np.sum(mask > 0))

        target = None
        if phase == 'vnf_deployment':
            target = getattr(self.env, 'current_deployment_target', None)
        elif phase == 'destination_connection':
            target = getattr(self.env, 'current_target_node', None)

        # 仅记录“是否应强制执行高层目标”，不要在这里提前 return。
        # 否则会绕过后面的 phase-specific 约束和状态维护。
        force_target_now = False
        if target is not None and target != current:
            target_is_neighbor = target in neighbors
            target_is_legal = False

            if target_is_neighbor:
                if self.shared.is_tree_edge(current, target):
                    target_is_legal = True
                else:
                    try:
                        _avail_bw_tgt = self.env.resource_mgr.pool.get_available_bandwidth(current, target)
                        target_is_legal = (_avail_bw_tgt >= bw_req)
                    except Exception:
                        target_is_legal = False

            if (target_is_neighbor and target_is_legal
                    and 0 <= int(target) < len(mask) and mask[int(target)] > 0):
                force_target_now = True

        # Stage-1: destination cycle 第0步只做一次 anchor 对齐核验
        # 只有 mutate_env=True 的主调用才允许落日志和改标志；
        # get_low_level_candidates()/get_state() 的只读调用不能重复触发。
        if (
                mutate_env
                and phase == 'destination_connection'
                and target is not None
                and getattr(self.env, '_dest_cycle_armed', False)
                and not getattr(self.env, '_dest_step0_checked', False)
        ):
            _anchor_s1 = getattr(self.env, 'current_anchor_node', None)
            logger.debug(
                f"[DestMaskStep0] target={target} anchor={_anchor_s1} "
                f"cur={current} path_trace={getattr(self.env, 'current_path_trace', None)}"
            )
            self.env._dest_step0_checked = True
            # [Fix-AnchorMismatch] v4.2 之后 anchor 不再强制等于 last_vnf，
            # Coordinator 在 cycle 开始时已通过 choose_best_anchor 设置 anchor
            # 并把 current_node_location 强制对齐到 anchor。
            # anchor==target（hop=0，从自身出发）是完全合法的，不应触发 mismatch。
            # 只有 anchor != target 且 current 既不在 anchor 也不在 target 时，
            # 才是真正的状态不同步 mismatch。
            _real_mismatch = (
                _anchor_s1 is not None
                and current != _anchor_s1
                and current != target
                and _anchor_s1 != target
            )
            self.env._dest_anchor_mismatch = _real_mismatch
            if _real_mismatch:
                logger.error(
                    f"[AnchorMismatch] target={target} anchor={_anchor_s1} cur={current} "
                    f"path_trace={getattr(self.env, 'current_path_trace', None)}"
                )
            else:
                logger.debug(
                    f"[AnchorOK] target={target} anchor={_anchor_s1} cur={current} "
                    f"anchor==target={_anchor_s1 == target}"
                )

        # ── 【修复1&3】destination 阶段：target 邻接且合法时立即硬返回 ──────────
        # 优先级高于方向约束、tabu、top-k。
        # 目标就在一跳之内且带宽合法 → 策略不应有其他选择。
        if force_target_now and phase == 'destination_connection':
            strict_mask = np.zeros(self.env.n, dtype=np.float32)
            strict_mask[target] = 1.0
            _apply_destination_order_guard(strict_mask)
            if np.sum(strict_mask) == 0:
                strict_mask[current] = 1.0
            self.env._last_mask_topk_count = 1
            self.env._last_mask_alive_count = 1
            if np.random.rand() < 0.05:
                logger.debug(
                    f"[ForceHighTarget] phase={phase} cur={current} target={target} "
                    f"forced=1 bw_req={bw_req:.2f}"
                )
            return strict_mask

        # 2. VNF阶段：对DC邻居节点用 probe_vnf_deploy 统一预检
        if phase == 'vnf_deployment' and self.env.current_request:
            _vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
            _vnf_list = self.env.current_request.get('vnf', [])
            _cpu_reqs = self.env.current_request.get('cpu_origin', []) or \
                        self.env.current_request.get('vnf_cpu', [])
            _mem_reqs = self.env.current_request.get('memory_origin', []) or \
                        self.env.current_request.get('vnf_mem', [])
            _req_cpu = _cpu_reqs[_vnf_idx] if _vnf_idx < len(_cpu_reqs) else 0.0
            _req_mem = _mem_reqs[_vnf_idx] if _vnf_idx < len(_mem_reqs) else 0.0
            _vnf_type = _vnf_list[_vnf_idx] if _vnf_idx < len(_vnf_list) else -1
            _dc_nodes = set(getattr(self.env, 'dc_nodes', []))

            for _nbr in [int(i) for i in np.where(mask > 0)[0]]:
                # Passing through a DC does not deploy the VNF there.  Resource
                # feasibility belongs only to the high-level placement target;
                # probing every transit DC incorrectly removed otherwise valid
                # spine paths whenever an intermediate DC was short on CPU/MEM.
                if _nbr not in _dc_nodes or target is None or _nbr != int(target):
                    continue
                _probe = self.env.resource_mgr.probe_vnf_deploy(
                    _nbr, _vnf_type, _req_cpu, _req_mem
                )
                if not _probe['ok']:
                    mask[_nbr] = 0.0

        if target is not None and current == target:
            if phase == 'destination_connection':
                self.env._dest_cycle_armed = False
            mask[:] = 0.0
            mask[current] = 1.0
            return mask
        else:
            mask[current] = 0.0

        # ============================================================
        # VNF 阶段：严格方向约束（d_next <= d_current）+ 短 tabu；不做 top-k
        # ============================================================
        if phase == 'vnf_deployment':
            d_current = self.shared.get_hop_distance_lazy(current, target) if target is not None else 9999

            # ① 严格方向约束：只允许不远离目标的方向（d_next <= d_current）
            if target is not None and current != target:
                _forward_exist = [
                    nbr for nbr in neighbors
                    if mask[nbr] > 0 and self.shared.get_hop_distance_lazy(nbr, target) <= d_current
                ]
                if _forward_exist:  # 有前进方向才施加约束，否则放行避免死锁
                    for nbr in neighbors:
                        if mask[nbr] <= 0:
                            continue
                        d_next = self.shared.get_hop_distance_lazy(nbr, target)
                        # [Relax-L5] VNF方向约束：允许+1跳绕路
                        if d_next > d_current + 1 and not self.shared.is_tree_edge(current, nbr):
                            mask[nbr] = 0.0
            _dir_ok = int(np.sum(mask > 0))

            # ② 短 tabu：只有存在其他非 tabu 候选时才屏蔽 tabu 节点
            current_path_set = set(getattr(self.env, 'current_path_trace', []))
            _non_tabu_alive = [
                nbr for nbr in neighbors
                if mask[nbr] > 0 and nbr not in current_path_set
            ]
            if _non_tabu_alive:
                for nbr in neighbors:
                    if mask[nbr] > 0 and nbr in current_path_set and nbr != target:
                        mask[nbr] = 0.0
            # [Fix-VNF-Tabu] 若方向约束后只剩 1 个候选且它在 tabu 里，
            # 直接保留（不清 tabu），避免软释放→走→进 tabu 的无效振荡循环。
            # _non_tabu_alive 为空说明所有候选都在 tabu 里，mask 此时可能还有值
            # （tabu 没有屏蔽它们，因为 _non_tabu_alive 为空），直接通过即可。
            _tabu_ok = int(np.sum(mask > 0))

            # [Fix-VNF-DeadLock] alive==1 且唯一候选在 tabu 中时，
            # 说明方向约束 + tabu 组合把 agent 逼入单步死胡同。
            # 此时扩大方向约束到 d_next <= d_current+1，纳入同距邻居，
            # 给 tabu 释放逻辑更多非 tabu 候选可选，避免反复在同一节点空转。
            if _tabu_ok == 1 and target is not None:
                _sole = int(np.argmax(mask))
                if _sole in current_path_set:
                    # 唯一候选是 tabu 节点：放宽方向约束到 d_next <= d_current+1
                    _expanded = []
                    for nbr in neighbors:
                        avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                        if self.shared.is_tree_edge(current, nbr) or avail_bw >= bw_req:
                            d_next = self.shared.get_hop_distance_lazy(nbr, target)
                            if d_next <= d_current + 1:
                                _expanded.append(nbr)
                    if len(_expanded) > 1:
                        # 重建 mask：把放宽后的候选全部打开
                        mask[:] = 0.0
                        for nbr in _expanded:
                            mask[nbr] = 1.0
                        mask[current] = 0.0
                        # 重新做 tabu：有非 tabu 候选时才屏蔽 tabu 节点
                        _non_tabu_exp = [n for n in _expanded if n not in current_path_set]
                        if _non_tabu_exp:
                            for nbr in _expanded:
                                if nbr in current_path_set and nbr != target:
                                    mask[nbr] = 0.0
                        _tabu_ok = int(np.sum(mask > 0))
                        logger.debug(
                            f"[VNF-DeadLock-Expand] cur={current} target={target} "
                            f"d_current={d_current} sole_tabu={_sole} "
                            f"expanded={_expanded} alive_after={_tabu_ok}"
                        )

            # ③ 软释放：被筛到全 0 时才放回候选并清 tabu
            # [Fix-VNF-Tabu] 若 mask 已有值（tabu 逻辑已保留唯一候选），
            # 不清空 tabu，避免走一步又立刻被重新封死的振荡。
            if np.sum(mask) == 0:
                if mutate_env:
                    self.env.current_path_trace = [current]
                for nbr in neighbors:
                    avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                    if self.shared.is_tree_edge(current, nbr) or avail_bw >= bw_req:
                        if target is None:
                            mask[nbr] = 1.0
                        else:
                            d_next = self.shared.get_hop_distance_lazy(nbr, target)
                            if d_next <= d_current:
                                mask[nbr] = 1.0
                if np.sum(mask) == 0:
                    for nbr in neighbors:
                        avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                        if self.shared.is_tree_edge(current, nbr) or avail_bw >= bw_req:
                            mask[nbr] = 1.0

            _apply_vnf_order_guard(mask)

            if np.sum(mask) == 0:
                mask[current] = 1.0

            _vnf_alive = int(np.sum(mask > 0))

            # 在最终返回前统一强制收口
            if force_target_now:
                forced_mask = np.zeros_like(mask)
                forced_mask[target] = 1.0
                self.env._last_mask_topk_count = 1
                self.env._last_mask_alive_count = 1
                if np.random.rand() < 0.02:
                    logger.debug(
                        f"[ForceHighTarget] phase={phase} cur={current} target={target} "
                        f"forced=1 bw_req={bw_req:.2f}"
                    )
                return forced_mask

            self.env._last_mask_topk_count = _vnf_alive
            self.env._last_mask_alive_count = _vnf_alive

            if np.random.rand() < 0.01:
                logger.debug(
                    f"[MaskDiag] phase={phase} cur={current} target={target} "
                    f"raw={_raw_nbrs} bw={_bw_ok} dir={_dir_ok} "
                    f"tabu={_tabu_ok} topk=SKIP alive={_vnf_alive}"
                )

            return mask

        # ============================================================
        # 以下仅用于 destination_connection 阶段
        # ============================================================
        _all_dests = set(self.env.current_request.get('dest', [])) \
            if self.env.current_request else set()
        _done_dests = set(self.env.current_tree.get('connected_dests', set())) \
            if self.env.current_tree else set()
        _remaining = len(_all_dests - _done_dests)

        # 1) 方向约束：先判断是否需要 BW 绕路，再决定门槛松紧
        # 核心思路：若 BW 感知最短路 >> 拓扑最短路，说明直连路段 BW 已耗尽，
        # agent 必须走"拓扑上更远"的绕路节点，此时硬门槛会封死所有有效路径。
        # 解决：计算 BW 感知 detour ratio，据此动态放宽 extra_hop 允许量。
        if target is not None and current != target:
            d_current = self.shared.get_hop_distance_lazy(current, target)
            _alive_bw = int(np.sum(mask > 0))

            # 计算 BW 感知绕路程度：BW 路径跳数 vs 拓扑最短跳数
            # 优先读 Coordinator 在 DestCycleStart 时预计算好的缓存值，
            # 只有缓存不可用时才回退到实时计算（慢，每步调用一次 BW 路径）。
            _cached_extra = getattr(self.env, '_dest_detour_tolerance', None)
            if _cached_extra is not None:
                _bw_extra_hop = int(_cached_extra)
            else:
                # 实时计算（fallback）
                _bw_path_len = d_current
                try:
                    _bw_path = self.compute_bw_aware_path(current, target)
                    if _bw_path and len(_bw_path) > 1:
                        _bw_path_len = len(_bw_path) - 1
                except Exception:
                    pass
                _detour_ratio = _bw_path_len / max(1, d_current)
                if _detour_ratio <= 1.0:
                    _bw_extra_hop = 0
                elif _detour_ratio <= 1.5:
                    _bw_extra_hop = 1
                elif _detour_ratio <= 2.5:
                    _bw_extra_hop = 2
                else:
                    _bw_extra_hop = 3

            # 计算各候选到目标的距离
            _closer = [nbr for nbr in neighbors if mask[nbr] > 0
                       and self.shared.get_hop_distance_lazy(nbr, target) < d_current]
            _equal = [nbr for nbr in neighbors if mask[nbr] > 0
                      and self.shared.get_hop_distance_lazy(nbr, target) == d_current]

            if _closer and _bw_extra_hop == 0:
                # [Relax-L1] 严格模式：保留更近+等距候选
                _closer_or_equal = set(_closer) | set(_equal)
                _before_count = int(np.sum(mask))
                for nbr in neighbors:
                    if mask[nbr] > 0 and nbr not in _closer_or_equal:
                        mask[nbr] = 0.0
                # [Relax-L1b] alive<=2时撤销方向约束，避免两节点振荡
                _after_count = int(np.sum(mask))
                if _before_count != _after_count and np.random.rand() < 0.01:
                    logger.debug(
                        f"[DirConstraint-Hard1] cur={current}→target={target} "
                        f"only_closer={_closer} before={_before_count} after={_after_count}"
                    )
            elif (_equal and not _closer) and _bw_extra_hop == 0:
                # [Relax-L7] 等距模式：保留等距+后退1跳候选（原来只保留等距）
                _equal_or_back1 = {
                    nbr for nbr in neighbors if mask[nbr] > 0
                                                and self.shared.get_hop_distance_lazy(nbr, target) <= d_current + 1
                }
                _before_count = int(np.sum(mask))
                for nbr in neighbors:
                    if mask[nbr] > 0 and nbr not in _equal_or_back1:
                        if not self.shared.is_tree_edge(current, nbr):
                            mask[nbr] = 0.0
                _after_count = int(np.sum(mask))
                if _before_count != _after_count and np.random.rand() < 0.01:
                    logger.debug(
                        f"[DirConstraint-Hard2] cur={current}→target={target} "
                        f"only_equal={_equal} before={_before_count} after={_after_count}"
                    )
                # [Relax-L7b] alive<=2时撤销方向约束
                if int(np.sum(mask)) <= 2:
                    mask[:] = 0.0
                    _td7 = (self.env.current_tree or {}).get('tree', {})
                    for nbr in neighbors:
                        if self.shared.is_tree_edge(current, nbr) \
                                or self.env.resource_mgr.pool.get_available_bandwidth(current, nbr) >= bw_req:
                            mask[nbr] = 1.0
                    mask[current] = 0.0
            else:
                # 宽松模式（需要绕路 或 全部候选都更远）：允许 extra_hop 范围内的节点
                _is_first_dest = (
                        getattr(self.env, '_is_first_dest_cycle', False)
                        and phase == 'destination_connection'
                )
                # [Fix] first_dest 同样需要 BW 绕路容忍——BW 约束与树是否建立无关。
                # 只有在 first_dest 且 BW 不需要绕路（_bw_extra_hop==0）时才用最小 base_extra，
                # 避免 first_dest 把主干拉得太歪；但 BW 确实需要绕路时必须放宽。
                if _is_first_dest:
                    # first_dest 优先保持方向性（不额外 detour），但 BW 绕路需求仍然生效
                    _base_extra = 0
                else:
                    _base_extra = 4 if (_remaining <= 2 or _alive_bw <= 2) else 3  # [Relax-L2b]
                _allow_extra_hop = max(_bw_extra_hop, _base_extra)

                _forward_free = [
                    nbr for nbr in neighbors
                    if mask[nbr] > 0
                       and self.shared.get_hop_distance_lazy(nbr, target) <= d_current + _allow_extra_hop
                ]
                if _forward_free:
                    for nbr in neighbors:
                        if mask[nbr] <= 0:
                            continue
                        d_next = self.shared.get_hop_distance_lazy(nbr, target)
                        if d_next > d_current + _allow_extra_hop and not self.shared.is_tree_edge(current, nbr):
                            mask[nbr] = 0.0
                if np.random.rand() < 0.02:
                    logger.debug(
                        f"[DirConstraint-Relaxed] cur={current}→target={target} "
                        f"detour_ratio={_detour_ratio:.2f} bw_path={_bw_path_len} "
                        f"topo={d_current} extra_hop={_allow_extra_hop}"
                    )
        _dir_ok = int(np.sum(mask > 0))

        # Prefer compact BW-feasible progress. If a short front is still
        # available, avoid opening long detours that consume downstream BW.
        if phase == 'destination_connection' and target is not None and current != target:
            _alive_compact = [nbr for nbr in neighbors if mask[nbr] > 0]
            if len(_alive_compact) > 1:
                _dist_items = [
                    (nbr, self.shared.get_hop_distance_lazy(nbr, target))
                    for nbr in _alive_compact
                ]
                _best_next_d = min(d for _, d in _dist_items)
                _compact_margin = 0 if _remaining > 2 else 1
                _compact_keep = {
                    nbr for nbr, d in _dist_items
                    if d <= _best_next_d + _compact_margin
                    or self.shared.is_tree_edge(current, nbr)
                }
                if 0 < len(_compact_keep) < len(_alive_compact):
                    _before_compact = len(_alive_compact)
                    for nbr in _alive_compact:
                        if nbr not in _compact_keep:
                            mask[nbr] = 0.0
                    _dir_ok = int(np.sum(mask > 0))
                    if np.random.rand() < 0.03:
                        logger.debug(
                            f"[CompactPathMask] cur={current} target={target} "
                            f"best_next_d={_best_next_d} margin={_compact_margin} "
                            f"before={_before_compact} after={_dir_ok} "
                            f"keep={sorted(_compact_keep)}"
                        )

        # 2) 绕路感知软 tabu（[SoftTabu]）
        # 原逻辑："节点在 current_path_trace 里 → 尽量禁掉"
        # 新逻辑："只有在没有绕路必要，且回访不改善可达性时才硬禁；
        #         需要绕路时（_bw_extra_hop>=1）只禁反向抖动边，保留可接受回访"
        current_path_set = set(getattr(self.env, 'current_path_trace', []))
        # 反向边集合：recent_edge_trace 里出现过 nbr→current 的边 = 明显回退
        _recent_edges = set(getattr(self.env, 'recent_edge_trace', []))
        _reverse_tabu = {nbr for nbr in neighbors if (nbr, current) in _recent_edges}

        if target is not None:
            _best_d = min(
                (self.shared.get_hop_distance_lazy(nbr, target) for nbr in neighbors),
                default=99
            )
            _best_nbrs = {
                nbr for nbr in neighbors
                if self.shared.get_hop_distance_lazy(nbr, target) == _best_d
            }
        else:
            _best_d = 99
            _best_nbrs = set()

        # 🚀 [Fix-TabuLoop] 统一、鲁棒的软禁忌逻辑，彻底斩断死循环
        # 核心原则：如果存在任何尚未走过的有效路径（非 tabu 且不远离目标的节点），
        # 绝不走回头路（tabu）。只有在完全没有新路可选时，才允许走回头路以尝试脱困。
        for nbr in neighbors:
            if mask[nbr] <= 0 or nbr == target:
                continue
            if nbr not in current_path_set:
                continue

            # 对于已经走过的节点（tabu 节点）：
            # 1. 寻找当前掩码中是否还有【没走过】的节点作为替代品
            _non_tabu_options = [n for n in neighbors if mask[n] > 0 and n not in current_path_set]

            if _non_tabu_options:
                # 只要有没走过的新路，一律封杀回头路！
                mask[nbr] = 0.0
            else:
                # 如果没有新路了（全都是 tabu 节点），允许回头。
                # 但反向抖动边（u→v→u，刚从那里过来）优先级最低，如果还有其他走过的旧路（比如成环），
                # 尽量走其他旧路，别原路退回，除非这是唯一选择。
                if nbr in _reverse_tabu:
                    _other_tabu_options = [n for n in neighbors if mask[n] > 0 and n != nbr]
                    if _other_tabu_options:
                        mask[nbr] = 0.0

        _tabu_ok = int(np.sum(mask > 0))

        # [Fix-Dest-DeadLock] alive==1 且唯一候选在 tabu 中时，
        # 方向约束 + tabu 组合把 agent 逼入单步死胡同。
        # 扩大方向约束到 d_next <= d_current + max(_bw_extra_hop+1, 2)，
        # 纳入更多非 tabu 邻居，避免在同一节点反复空转。
        if _tabu_ok == 1 and target is not None:
            _sole_d = int(np.argmax(mask))
            if _sole_d in current_path_set:
                # _bw_extra_hop 在 target is not None and current != target 块内赋值；
                # current==target 경우는 위에서 이미 return되므로 실제로 None이 될 수 없지만
                # 안전을 위해 locals()로 보호
                _safe_bw_extra = locals().get('_bw_extra_hop', 0)
                _expand_extra = max(_safe_bw_extra + 1, 2)
                _expanded_d = []
                for nbr in neighbors:
                    avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                    if self.shared.is_tree_edge(current, nbr) or avail_bw >= bw_req:
                        d_next = self.shared.get_hop_distance_lazy(nbr, target)
                        if d_next <= d_current + _expand_extra:
                            _expanded_d.append(nbr)
                if len(_expanded_d) > 1:
                    mask[:] = 0.0
                    for nbr in _expanded_d:
                        mask[nbr] = 1.0
                    mask[current] = 0.0
                    # 重新做 tabu：有非 tabu 候选才屏蔽 tabu 节点
                    _non_tabu_exp_d = [n for n in _expanded_d if n not in current_path_set]
                    if _non_tabu_exp_d:
                        for nbr in _expanded_d:
                            if nbr in current_path_set and nbr != target:
                                mask[nbr] = 0.0
                    _tabu_ok = int(np.sum(mask > 0))
                    logger.debug(
                        f"[Dest-DeadLock-Expand] cur={current} target={target} "
                        f"d_current={d_current} sole_tabu={_sole_d} "
                        f"expand_extra={_expand_extra} expanded={_expanded_d} "
                        f"alive_after={_tabu_ok}"
                    )

        # 🚀 [Fix-BacktrackTrap] 全局防震荡 Trap 机制
        # Trap-A: 所有候选（或唯一候选）只能让你离目标更远（或退回原地）
        # Trap-B: 所有可行的候选路线都已经走过了（深陷迷宫循环）
        _alive_count = int(np.sum(mask))
        if _alive_count > 0 and target is not None and mutate_env:
            _candidates = [n for n in neighbors if mask[n] > 0]
            _d_cur = self.shared.get_hop_distance_lazy(current, target)
            _all_backtrack = all(self.shared.get_hop_distance_lazy(n, target) > _d_cur for n in _candidates)
            _current_path_set = set(getattr(self.env, 'current_path_trace', []))
            _all_in_tabu = all(n in _current_path_set for n in _candidates)
            _tabu_sz = len(_current_path_set)

            _remaining_trap = 99
            if self.env.current_request and self.env.current_tree:
                _all_dt = len(self.env.current_request.get('dest', []))
                _done_dt = len(self.env.current_tree.get('connected_dests', set()))
                _remaining_trap = _all_dt - _done_dt
            _max_tabu_now = 3 if _remaining_trap <= 1 else 5

            # 触发阻断条件：所有选项都是后退且积累了历史，或者所有选项都是走过的老路
            _should_clear = (
                (_alive_count == 1 and _all_backtrack and _tabu_sz >= 2) or
                (_all_in_tabu and _tabu_sz >= _max_tabu_now - 1)
            )

            if _should_clear:
                _sole_alive = _candidates[0] if _alive_count == 1 else _candidates[-1]
                _d_sole = self.shared.get_hop_distance_lazy(_sole_alive, target)
                _is_backtrack = _all_backtrack
                # 👈 【新增区块开始】将当前死胡同节点加入强禁忌表
                if not hasattr(self.env, 'hard_tabu_list'):
                    self.env.hard_tabu_list = set()
                self.env.hard_tabu_list.add(current)
                _hard_tabu = self.env.hard_tabu_list  # 👈 同步更新本函数内的局部变量
                # 👈 【新增区块结束】

                self.env.current_path_trace = [current]
                mask[:] = 0.0
                _tree_d2 = (self.env.current_tree or {}).get('tree', {})
                for nbr in neighbors:
                    if nbr in _hard_tabu:  # 👈 【新增】回溯开放时，强禁忌表里的死路坚决不放开
                        continue

                    if self.shared.is_tree_edge(current, nbr) \
                            or self.env.resource_mgr.pool.get_available_bandwidth(current, nbr) >= bw_req:
                        mask[nbr] = 1.0
                mask[current] = 0.0
                _trap_type = 'A(backtrack)' if _is_backtrack else 'B(tabu-block)'
                # 最后1个dest时提升到INFO
                _bt_log = logger.debug if (_remaining_trap <= 1) else logger.debug
                _bt_log(
                    f"[BacktrackTrap-{_trap_type}] cur={current} sole={_sole_alive} "
                    f"d_cur={_d_cur} d_sole={_d_sole} tabu_sz={_tabu_sz} → 清tabu 开放nbrs={[n for n in neighbors if mask[n] > 0]}"
                )

        # 3) 死路软释放
        if np.sum(mask) == 0:
            if mutate_env:
                self.env.current_path_trace = [current]
            for nbr in neighbors:
                if nbr in _hard_tabu: continue  # 👈 【新增】死路软释放时，不要放出已经被拉黑的节点
                if self.shared.is_tree_edge(current, nbr):
                    mask[nbr] = 1.0
            if np.sum(mask) == 0:
                for nbr in neighbors:
                    if nbr in _hard_tabu: continue  # 👈 【新增】
                    # Hard capacity invariant: a recovery action must remain
                    # committable. Opening a merely non-zero link here creates
                    # plans that can never pass the final atomic BW commit.
                    if self.env.resource_mgr.pool.get_available_bandwidth(current, nbr) >= bw_req:
                        mask[nbr] = 1.0

        _apply_destination_order_guard(mask)
        if np.sum(mask) == 0:
            mask[current] = 1.0

        if (
            phase == 'destination_connection'
            and target is not None
            and current != target
            and (
                getattr(self.env, '_planner_destinations_enabled', False)
                or
                int(target) in getattr(self.env, '_dest_recovery_targets', set())
                or getattr(self.env, '_timeout_count', 0) > 0
            )
        ):
            recovery_path = self.compute_progress_path(current, target)
            if recovery_path and len(recovery_path) > 1:
                next_hop = int(recovery_path[1])
                if next_hop in neighbors and next_hop not in _dest_order_blocked:
                    recovery_mask = np.zeros_like(mask)
                    recovery_mask[next_hop] = 1.0
                    self.env._last_mask_topk_count = 1
                    self.env._last_mask_alive_count = 1
                    logger.info(
                        f"[DestRecoveryPath] current={current} target={target} "
                        f"next={next_hop} path={recovery_path}"
                    )
                    return recovery_mask

        # 4) first dest 用更小的 top-k(3)；最后两个目的地/候选很少时跳过
        _is_first_dest_mask = (
                getattr(self.env, '_is_first_dest_cycle', False)
                and phase == 'destination_connection'
        )
        _alive_before_topk = int(np.sum(mask > 0))
        if _remaining <= 2 or _alive_before_topk <= 3:
            _topk_ok = _alive_before_topk

            # 在最终返回前统一强制收口
            if force_target_now:
                forced_mask = np.zeros_like(mask)
                forced_mask[target] = 1.0
                _apply_destination_order_guard(forced_mask)
                if np.sum(forced_mask) == 0:
                    forced_mask[current] = 1.0
                self.env._last_mask_topk_count = 1
                self.env._last_mask_alive_count = 1
                if np.random.rand() < 0.02:
                    logger.debug(
                        f"[ForceHighTarget] phase={phase} cur={current} target={target} "
                        f"forced=1 bw_req={bw_req:.2f}"
                    )
                return forced_mask

            self.env._last_mask_topk_count = _topk_ok
            self.env._last_mask_alive_count = _topk_ok

            _tabu_size = len(set(getattr(self.env, 'current_path_trace', [])))
            # [Diag] 最后1个dest时提高日志采样率到20%，便于诊断
            _log_rate = 0.20 if _remaining <= 1 else 0.01
            if np.random.rand() < _log_rate:
                logger.debug(
                    f"[MaskDiag] phase={phase} cur={current} target={target} "
                    f"raw={_raw_nbrs} bw={_bw_ok} dir={_dir_ok} "
                    f"tabu={_tabu_ok} topk=SKIP(rem={_remaining},alive={_alive_before_topk}) "
                    f"tabu_list_sz={_tabu_size}"
                )
            _apply_destination_order_guard(mask)
            if np.sum(mask) == 0:
                mask[current] = 1.0
            return mask

        # 5) 其余情况做 top-k；first dest 收紧到 k=3
        _orig_topk = self._low_topk
        if _is_first_dest_mask:
            self._low_topk = 3
        mask = self.shared.apply_low_topk_mask(
            mask,
            current,
            target,
            bw_req,
            tree_edges,
            hop_distance_fn=self.shared.get_hop_distance_lazy,
            low_topk=self._low_topk,
            phase=phase,
        )
        self._low_topk = _orig_topk
        _apply_destination_order_guard(mask)
        _topk_ok = int(np.sum(mask > 0))

        # 在最终返回前统一强制收口
        if force_target_now:
            forced_mask = np.zeros_like(mask)
            forced_mask[target] = 1.0
            _apply_destination_order_guard(forced_mask)
            if np.sum(forced_mask) == 0:
                forced_mask[current] = 1.0
            self.env._last_mask_topk_count = 1
            self.env._last_mask_alive_count = 1
            if np.random.rand() < 0.02:
                logger.debug(
                    f"[ForceHighTarget] phase={phase} cur={current} target={target} "
                    f"forced=1 bw_req={bw_req:.2f}"
                )
            return forced_mask

        self.env._last_mask_topk_count = _topk_ok
        self.env._last_mask_alive_count = _topk_ok

        _tabu_size = len(set(getattr(self.env, 'current_path_trace', [])))
        if np.random.rand() < 0.01:
            logger.debug(
                f"[MaskDiag] phase={phase} cur={current} target={target} "
                f"raw={_raw_nbrs} bw={_bw_ok} dir={_dir_ok} "
                f"tabu={_tabu_ok} topk={_topk_ok} tabu_list_sz={_tabu_size}"
            )
        if np.random.rand() < 0.001:
            _active = [int(i) for i in np.where(mask > 0)[0]]
            logger.debug(
                f"[LowMaskViz] phase={phase} cur={current} target={target} "
                f"active_nodes={_active} topk={self._low_topk} "
                f"tabu_list={getattr(self.env, 'current_path_trace', [])[-5:]}"
            )
        if _topk_ok == 0:
            logger.warning(
                f"[MaskDiag] ⚠️ 零候选! phase={phase} cur={current} target={target} "
                f"raw={_raw_nbrs} bw={_bw_ok} dir={_dir_ok} tabu={_tabu_ok}"
            )

        _apply_destination_order_guard(mask)
        if np.sum(mask) == 0:
            mask[current] = 1.0
        return mask
    # ==================================================================
    # 候选邻居暴露（v4.2 新增）
    # 供 GoalConditionedLowLevelPolicy.score_candidates() 做逐点评分
    # ==================================================================
    def get_low_level_candidates(self):
        """
        在 get_low_level_action_mask() 的基础上，额外计算每个候选邻居的结构化局部特征。
        供 low_policy 的 score_candidates() 使用，替代全图 logits 输出。

        返回 dict：
            indices      : List[int]          候选节点 ID（已经过 top-k 裁剪）
            mask         : np.ndarray[K]      全 1（已过滤合法候选）
            current_node : int
            target_node  : int or None
            features     : np.ndarray[K, 6]  每个候选的局部特征
        """
        return self.shared.get_low_level_candidates(
            hop_distance_fn=self.shared.get_hop_distance_lazy,
            action_mask_fn=self.get_low_level_action_mask,
        )

    def _calculate_tree_metrics(self) -> dict:
        """
        计算当前树的结构指标，供 HighLevelController 在 episode 完成时调整奖励。

        返回:
            tree_n_edges   : 正流量树边总数
            reused_edges   : 被复用（tree_usage > 1）的边数
            redundancy     : reused_edges / max(1, tree_n_edges)，衡量冗余度
            connected_dests: 已连通目的地数量
        """
        if not getattr(self.env, 'current_tree', None):
            return {'tree_n_edges': 0, 'reused_edges': 0, 'redundancy': 0.0, 'connected_dests': 0}

        tree_dict  = self.env.current_tree.get('tree', {})
        usage_dict = self.env.current_tree.get('tree_usage', {})

        tree_n_edges = sum(1 for f in tree_dict.values() if float(f) > 0.0)
        reused_edges = sum(1 for v in usage_dict.values() if v > 1)
        redundancy   = reused_edges / max(1, tree_n_edges)

        connected_dests = 0
        req_id = (self.env.current_request or {}).get('id')
        if req_id is not None:
            rm = getattr(self.env, 'resource_mgr', None)
            if rm is not None:
                rec = getattr(rm, 'request_table', {}).get(req_id)
                if rec is not None:
                    connected_dests = len(rec.connected_dests)
        if connected_dests == 0:
            connected_dests = len(self.env.current_tree.get('connected_dests', set()))

        return {
            'tree_n_edges':    tree_n_edges,
            'reused_edges':    reused_edges,
            'redundancy':      redundancy,
            'connected_dests': connected_dests,
        }


    def _propagate_full_stage_on_branch(self, branch_root, branch_path):
        """
        若 branch_root 已是 full-stage，则把 branch_path 上所有节点标记为 full-stage。
        语义：这些节点都已在完整 SFC 之后承载流量，可作为后续安全 anchor。
        """
        if not getattr(self.env, 'current_tree', None):
            return
        total_vnf = len(self.env.current_request.get('vnf', [])) \
            if self.env.current_request else 0
        if total_vnf <= 0:
            return
        self.env.current_tree.setdefault('node_stage', {})
        node_stage = self.env.current_tree['node_stage']
        root_stage = node_stage.get(branch_root, 0)
        if root_stage < total_vnf:
            return
        changed = []
        for node in branch_path:
            prev = node_stage.get(node, 0)
            if prev < total_vnf:
                node_stage[node] = total_vnf
                changed.append((node, prev, total_vnf))
        if changed:
            logger.debug(
                f"[StagePropagate] root={branch_root} root_stage={root_stage}/{total_vnf} "
                f"updated={changed} path={branch_path}"
            )

    def _would_create_cycle(self, u, v):
        """
        检查把边(u,v)加入flow>0正树后是否会成环。
        只有两端都已在树上时才可能成环；任一端是新节点则不可能成环。
        """
        tree_edges = (self.env.current_tree or {}).get('tree', {})
        # 收集flow>0的树上节点集合
        tree_nodes = set()
        for (a, b), flow in tree_edges.items():
            if flow > 0.0:
                tree_nodes.add(a)
                tree_nodes.add(b)
        # 任一端不在树上，加边不会成环
        if u not in tree_nodes or v not in tree_nodes:
            return False
        # Treat opposite directions as the same physical link for topology
        # cycle detection.  Ignore the direct physical link being considered
        # so the reverse direction of an SFC spine edge remains legal.
        G = nx.Graph()
        for (a, b), flow in tree_edges.items():
            if flow > 0.0:
                if {int(a), int(b)} == {int(u), int(v)}:
                    continue
                G.add_edge(a, b)
        try:
            return G.has_node(u) and G.has_node(v) and nx.has_path(G, u, v)
        except Exception:
            return False


    # ==================================================================
    def get_state(self):
        rm = self.env.resource_mgr
        K_vnf    = rm.K_vnf
        C_cap    = max(1, rm.C_cap)
        M_cap    = max(1, rm.M_cap)
        n        = self.env.n

        # [Task5] 状态日志：确认 bw_need 与 bw_origin 一致
        if not hasattr(self, '_bw_state_log_count'): self._bw_state_log_count = 0
        self._bw_state_log_count += 1
        if self._bw_state_log_count % 500 == 1 and self.env.current_request:
            _bw_origin = self.env.current_request.get('bw_origin', None)
            _bw_legacy = self.env.current_request.get('bw', None)
            logger.debug(
                f"[BWCheck] bw_origin={_bw_origin} bw(legacy)={_bw_legacy} "
                f"→ 状态特征使用bw_origin={_bw_origin}"
            )

        current_vnf_demand = 0.0
        if self.env.current_request:
            vnf_list  = self.env.current_request.get('vnf', [])
            idx       = getattr(self.env, 'next_vnf_idx', 0)
            if idx < len(vnf_list):
                cpu_reqs = self.env.current_request.get('cpu_origin', [10.0])
                current_vnf_demand = cpu_reqs[idx] if idx < len(cpu_reqs) else 10.0

        target_node = None
        if self.env.current_phase == 'vnf_deployment':
            target_node = getattr(self.env, 'current_deployment_target', None)
        elif self.env.current_phase == 'destination_connection':
            target_node = getattr(self.env, 'current_target_node', None)
        target_node_int = int(target_node) if target_node is not None else -1

        current_node = getattr(self.env, 'current_node_location', -1)
        _dc = getattr(self.env, 'dc_nodes', None) or getattr(getattr(self.env, 'resource_mgr', None), 'dc_nodes', None) or getattr(getattr(getattr(self.env, 'resource_mgr', None), 'pool', None), 'dc_nodes', None)
        dc_nodes = set(_dc) if _dc else set()

        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())
        connected_dests = (self.env.current_tree.get('connected_dests', set()) if self.env.current_tree else set())
        hvt_all = rm.hvt_all

        # 🚀 目标感知掩码 (Dest Mask)
        try:
            all_dests = set(int(x) for x in self.env.current_request.get('dest', []))
            remaining_dests = all_dests - connected_dests
        except Exception:
            remaining_dests = set()

        dest_mask = torch.zeros(n, dtype=torch.bool)
        for d in remaining_dests:
            if 0 <= d < n:
                dest_mask[int(d)] = True

        max_hops = max(1, n - 1)
        def _hop(u, v):
            if u < 0 or v < 0 or u == v: return 0.0
            try: return self.shared.get_hop_distance_lazy(u, v) / max_hops
            except Exception: return 1.0

        vnf_list_total = (self.env.current_request.get('vnf', []) if self.env.current_request else [])
        total_vnf = max(1, len(vnf_list_total))
        cur_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        vnf_depth_norm = min(1.0, cur_vnf_idx / total_vnf)

        subgoal_steps = getattr(self.env, 'subgoal_step_count', 0)
        subgoal_horizon = getattr(self.env, 'subgoal_horizon', 40)
        progress_ratio = min(1.0, subgoal_steps / max(1, subgoal_horizon))

        phase = getattr(self.env, 'current_phase', 'other')
        if phase == 'vnf_deployment': phase_flag = 0.0
        elif phase == 'destination_connection': phase_flag = 1.0
        else: phase_flag = 0.5

        # 🚀 24 维状态特征矩阵 (21维原有 + 3维邻边BW感知)
        features = np.zeros((n, 28), dtype=np.float32)  # [Reach] 24->28
        for node in range(n):
            avail_cpu  = rm.pool.get_available_cpu(node)
            avail_mem  = rm.pool.get_available_memory(node)
            fit_factor = 1.0 if avail_cpu >= current_vnf_demand else -1.0

            features[node, 0] = avail_cpu / C_cap
            features[node, 1] = avail_mem / M_cap
            features[node, 2] = fit_factor
            features[node, 3] = 1.0 if node in dc_nodes else 0.0
            features[node, 4] = 1.0 if node == current_node else 0.0
            features[node, 5] = _hop(node, target_node_int)

            if 0 <= node < hvt_all.shape[0]:
                features[node, 6:6 + K_vnf] = hvt_all[node, :K_vnf].astype(np.float32)

            features[node, 6 + K_vnf]     = 1.0 if node in nodes_on_tree else 0.0
            features[node, 6 + K_vnf + 1] = 1.0 if node in connected_dests else 0.0
            features[node, 6 + K_vnf + 2] = 1.0 if node == target_node_int else 0.0
            features[node, 6 + K_vnf + 3] = vnf_depth_norm
            features[node, 6 + K_vnf + 4] = progress_ratio
            features[node, 6 + K_vnf + 5] = phase_flag

            # 🚀 核心创新: hop_to_tree (支持消融：env._ablation_hop=True 时清零)
            if not getattr(self.env, '_ablation_hop', False):
                if len(nodes_on_tree) > 0:
                    features[node, 20] = min([self.shared.get_hop_distance_lazy(node, t) for t in nodes_on_tree]) / max_hops
                else:
                    features[node, 20] = 1.0
            # else: 保持 0.0（np.zeros 已初始化）

            # 🚀 dim21-23：BW感知特征
            # dim21 = 最差入向BW比例（邻居→node，反映可达性瓶颈）
            # dim22 = 均值入向BW比例（邻居→node，反映整体可达水平）
            # dim23 = 出向平均利用率（node→邻居，反映出发方向拥塞程度）
            # [Fix-OutUtil] 将 dim23 从二值可达信号改为出向平均利用率：
            #   原来：1.0/-1.0 表示"能否从邻居到达"，但 mask 已保证可达，信息冗余
            #   现在：出向利用率 [0,1]，0=全空闲，1=全饱和，让低层感知方向拥塞
            _nbrs_node = rm.get_neighbors(node)
            _in_bws = [
                rm.pool.get_available_bandwidth(nbr, node)
                for nbr in _nbrs_node
            ]
            if _in_bws:
                _bw_cap = max(1.0, getattr(rm, 'BW_cap', 100.0))
                _bw_need = (self.env.current_request.get('bw_origin',
                             self.env.current_request.get('bw', 0.0))
                            if self.env.current_request else 0.0)
                _min_in_bw = min(_in_bws)
                _avg_in_bw = sum(_in_bws) / len(_in_bws)
                features[node, 21] = _min_in_bw / _bw_cap       # 最差入向BW（可达性瓶颈）
                features[node, 22] = _avg_in_bw / _bw_cap       # 均值入向BW
                # [Fix-OutUtil] dim23：出向平均利用率
                _out_bws  = [rm.pool.get_available_bandwidth(node, nbr) for nbr in _nbrs_node]
                _out_caps = [max(rm.pool.bw_cap.get((node, nbr), 1.0), 1.0) for nbr in _nbrs_node]
                _avg_out_util = 1.0 - sum(_out_bws) / sum(_out_caps)
                features[node, 23] = float(_avg_out_util)        # 出向平均利用率（0=空闲,1=饱和）
            # else: 孤立节点保持 0.0

        # [Reach] dim24-27：多跳可达性（到剩余目的集合的瓶颈带宽/跳数）
        #   消融开关 _ablation_reach=True 时清零（仿照已有 _ablation_hop）
        if not getattr(self.env, '_ablation_reach', False):
            try:
                features[:, 24:28] = rm.get_reach_feats(remaining_dests)
            except Exception as _e:
                logger.warning(f"[Reach] 低层可达性特征计算失败，置0: {_e}")
        # else: 保持 0.0（np.zeros 已初始化）

        # [Ablation-MinMLP] Build a stricter w/o GNN baseline.  Plain MLP should
        # not receive explicit topology/tree/destination/reachability shortcuts.
        if getattr(self.env, '_minimal_mlp_state', False):
            graph_flag_start = 6 + K_vnf
            features[:, 5] = 0.0                         # hop_to_target
            features[:, graph_flag_start:graph_flag_start + 3] = 0.0  # on_tree / connected_dest / is_target
            features[:, 20:28] = 0.0                     # hop_to_tree / BW stats / reachability
            dest_mask.zero_()

        x_tensor = torch.from_numpy(features).float()
        low_mask  = self.get_low_level_action_mask()

        if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'build_dynamic_edge_attr'):
            edge_attr_tensor = self.env.resource_mgr.build_dynamic_edge_attr()
        elif hasattr(self.env, 'edge_attr') and self.env.edge_attr is not None:
            edge_attr_tensor = self.env.edge_attr
        else:
            edge_attr_tensor = None

        tree_edge_index = None
        tree_dict = getattr(self.env, 'current_tree', None)
        if tree_dict:
            tree_edges_raw = tree_dict.get('tree', {})
            if tree_edges_raw:
                rows, cols = [], []
                for (u, v), flow in tree_edges_raw.items():
                    if flow > 0.0:
                        rows.append(u)
                        cols.append(v)
                tree_edge_index = torch.tensor([rows, cols], dtype=torch.long)

        if getattr(self.env, '_minimal_mlp_state', False):
            edge_attr_tensor = None
            tree_edge_index = None

        try:
            reach_abs_sum = float(x_tensor[:, 24:28].abs().sum().item()) if x_tensor.size(1) >= 28 else None
            tree_edges = int(tree_edge_index.size(1)) if tree_edge_index is not None else None
            edge_attr_shape = tuple(edge_attr_tensor.shape) if edge_attr_tensor is not None else None
            diag_key = (
                int(dest_mask.sum().item()) > 0,
                tree_edges is not None and tree_edges > 0,
                reach_abs_sum is not None and reach_abs_sum > 0.0,
            )
            seen = getattr(self.env, '_gnn_state_debug_seen_keys', set())
            if diag_key not in seen and len(seen) < 6:
                logger.info(
                    "[GNN-DIAG state] "
                    f"x={tuple(x_tensor.shape)} edge_attr={edge_attr_shape} "
                    f"dest_mask_sum={int(dest_mask.sum().item())} "
                    f"tree_edge_index_edges={tree_edges} "
                    f"reach_feat_abs_sum={reach_abs_sum} "
                    f"ablation_reach={getattr(self.env, '_ablation_reach', False)}"
                )
                seen.add(diag_key)
                self.env._gnn_state_debug_seen_keys = seen
        except Exception as exc:
            logger.warning(f"[GNN-DIAG state] failed: {exc}")

        # 🚀 返回 PyG Data (加入了 dest_mask)
        return Data(
            x=x_tensor,
            edge_index=self.env.edge_index if hasattr(self.env, 'edge_index') else None,
            edge_attr=edge_attr_tensor,
            tree_edge_index=tree_edge_index,
            action_mask=torch.from_numpy(low_mask).bool().unsqueeze(0),
            dest_mask=None if getattr(self.env, '_minimal_mlp_state', False) else dest_mask
        )

    # ==================================================================
    # 辅助方法
    # ==================================================================
    def _commit_episode_bandwidth(self):
        """
        [delayed-commit] 请求成功后先构建规范根树，再一次性扣带宽。
        通过 resource_mgr.commit_edge_bandwidth() 同时完成两件事：
          1. allocate_bandwidth — 扣减物理带宽
          2. 将 EdgeAllocation 写入 RequestRecord.edge_allocations — 供 release_request_record 单路径释放带宽
        flow=0/-1 的遍历边可作为规范树的连接证据，但只有规范树边会提交。
        提交前做全树带宽预检；任一提交或快照校验失败都会统一回滚。
        [Directed-BW] 规范树键是有向 (u,v)，直接按键方向扣带宽。
        """
        if not self.env.current_request or not self.env.current_tree:
            return False
        req_id = self.env.current_request.get('id')
        bw = self.env.current_request.get('bw_origin', 0.0)
        canonical_input = dict(self.env.current_tree)
        current_sfc = getattr(self.env, 'current_sfc', None) or {}
        canonical_input['spine_paths'] = [
            list(path) for path in current_sfc.get('spine_paths', [])
        ]
        canonical_input['branch_paths'] = {
            int(target): list(path)
            for target, path in current_sfc.get('branch_paths', {}).items()
        }
        # Snapshot validation canonicalizes the tree a second time. Persist
        # the same stage-aware evidence so crossing edges cannot produce a
        # different (but still rooted) tree after bandwidth was committed.
        self.env.current_tree['spine_paths'] = [
            list(path) for path in canonical_input['spine_paths']
        ]
        self.env.current_tree['branch_paths'] = {
            int(target): list(path)
            for target, path in canonical_input['branch_paths'].items()
        }
        canonical = self.env.resource_mgr.canonicalize_request_sft(
            req_id, canonical_input, allow_topology_repair=True
        )
        if not canonical.get('ok'):
            logger.warning(
                f"[BW-Commit] req={req_id} SFT规范化失败: {canonical.get('reason')} "
                f"unreachable={canonical.get('unreachable_nodes', [])} "
                f"detail={canonical}"
            )
            self._archive_episode_fail()
            return False

        tree = dict(canonical['tree_edges'])
        self.env.current_tree['tree'] = tree
        old_usage = self.env.current_tree.get('tree_usage', {}) or {}
        self.env.current_tree['tree_usage'] = {
            edge: max(1, int(old_usage.get(edge, old_usage.get((edge[1], edge[0]), 1))))
            for edge in tree
        }

        for u, v in tree:
            if self.env.resource_mgr.get_available_bandwidth(u, v) + 1e-5 < float(bw):
                logger.warning(f"[BW-Commit] req={req_id} 预检失败 ({u},{v}) bw={bw}")
                self._archive_episode_fail()
                return False

        committed = 0
        for edge_key, flow in tree.items():
            if flow > 0.0:
                # [Directed-BW] tree 键已是有向 (u,v)，直接扣对应方向带宽
                u, v = edge_key[0], edge_key[1]
                if not self.env.resource_mgr.commit_edge_bandwidth(req_id, u, v, bw):
                    self._archive_episode_fail()
                    return False
                committed += 1
        if hasattr(self.env.resource_mgr, 'snapshot_request_sft'):
            snap_ok = self.env.resource_mgr.snapshot_request_sft(
                req_id,
                current_tree=self.env.current_tree,
                snapshot_time=getattr(self.env, 'time_step', None),
                already_canonical=True,
            )
            if not snap_ok:
                self._archive_episode_fail()
                return False
        if canonical.get('used_topology_repair'):
            logger.info(
                f"[BW-Commit] req={req_id} 补齐连接边={canonical.get('repaired_edges', [])}"
            )
        logger.debug(f"[BW-Commit] req={req_id} committed_edges={committed} bw={bw}")
        return True

    def _finalize_episode_success(self) -> bool:
        """Commit one successful request in a fixed, rollback-safe order.

        The order is part of the accounting contract: bandwidth commit, SFT
        snapshot validation, lifecycle registration, then success accounting.
        """
        self.env._success_finalize_reason = 'success_finalize_failed'
        if not self.env.current_request:
            self.env._success_finalize_reason = 'missing_request'
            return False

        req_id = self.env.current_request.get('id')
        if req_id is None:
            self.env._success_finalize_reason = 'missing_req_id'
            return False

        if not self._commit_episode_bandwidth():
            self.env._success_finalize_reason = 'bandwidth_or_snapshot_commit_failed'
            return False

        validator = getattr(self.env.resource_mgr, 'validate_request_sft_snapshot', None)
        if not callable(validator):
            self.env._success_finalize_reason = 'sft_validator_unavailable'
            self._archive_episode_fail()
            return False

        try:
            validation = validator(req_id)
        except Exception as exc:
            logger.error(f"[SuccessFinalizeFail] req={req_id} SFT校验异常: {exc}")
            validation = {'ok': False, 'reason': f'validation_exception:{exc}'}
        self.env._last_sft_validation_report = validation
        if not validation.get('ok', False):
            self.env._success_finalize_reason = (
                f"invalid_sft_snapshot:{validation.get('reason', 'unknown')}"
            )
            self._archive_episode_fail()
            return False

        request_manager = getattr(self.env, 'request_manager', None)
        active_before = (
            req_id in request_manager.active_requests
            if request_manager is not None else False
        )

        def rollback_lifecycle_if_added() -> None:
            if request_manager is None or active_before:
                return
            rollback_registration = getattr(
                request_manager, 'rollback_registration', None
            )
            if not callable(rollback_registration):
                logger.error(
                    f"[SuccessFinalizeFail] req={req_id} 无法撤销 lifecycle 注册"
                )
                return
            try:
                rollback_registration(req_id)
            except Exception as exc:
                logger.error(
                    f"[SuccessFinalizeFail] req={req_id} lifecycle 回滚异常: {exc}"
                )

        if not self._add_request_to_lifecycle_manager():
            self.env._success_finalize_reason = 'lifecycle_registration_failed'
            # register_request() 应当是原子的；这里仍处理“已插入后异常”的防御路径。
            rollback_lifecycle_if_added()
            self._archive_episode_fail()
            return False

        if not self._archive_episode_success_only():
            self.env._success_finalize_reason = 'success_accounting_failed'
            rollback_lifecycle_if_added()
            self._archive_episode_fail()
            return False

        self.env._success_finalize_reason = 'ok'
        return True

    def _archive_episode_success_only(self) -> bool:
        resource_mgr = getattr(self.env, 'resource_mgr', None)
        if resource_mgr is not None and hasattr(resource_mgr, '_archive_request'):
            accepted_before = getattr(resource_mgr, 'total_requests_accepted', None)
            served_before = getattr(resource_mgr, 'served_dest_count', None)
            try:
                result = resource_mgr._archive_request(
                    success=True, already_rolled_back=False
                )
            except Exception as exc:
                if accepted_before is not None:
                    resource_mgr.total_requests_accepted = accepted_before
                if served_before is not None:
                    resource_mgr.served_dest_count = served_before
                logger.error(f"[SuccessArchive] 成功计数失败: {exc}")
                return False

            if result is False:
                if accepted_before is not None:
                    resource_mgr.total_requests_accepted = accepted_before
                if served_before is not None:
                    resource_mgr.served_dest_count = served_before
                logger.error("[SuccessArchive] _archive_request 拒绝成功归档")
                return False
            if (
                accepted_before is not None
                and resource_mgr.total_requests_accepted != accepted_before + 1
            ):
                resource_mgr.total_requests_accepted = accepted_before
                if served_before is not None:
                    resource_mgr.served_dest_count = served_before
                logger.error("[SuccessArchive] 接受计数未按预期增加，已撤销计数")
                return False
            return True
        logger.error("[SuccessArchive] resource_mgr 缺少 _archive_request，拒绝虚假成功")
        return False

    def _archive_episode_fail(self):
        """
        delayed-commit 模式失败回滚。
        主路径：通过 release_request_record(req_id, rollback=True) 走新内核统一释放链：
          - CPU/MEM：_release_vnf_binding → ref_count=0 才释放
          - BW：delayed-commit 搜索阶段未扣，edge_allocations 为空，无需回滚
        旧路径 fallback：只有在 request_table 中找不到记录时才启用（仅 release_vnf_ref）。
        """
        # 防止重复进入导致二次释放
        if getattr(self.env, '_bw_already_rolled_back', False):
            logger.debug("[FailRollback] 已回滚过，跳过重复回滚")
            return
        self.env._bw_already_rolled_back = True

        if not self.env.current_request:
            return

        req_id = self.env.current_request.get('id')

        # ── 主路径：新内核统一回滚链 ─────────────────────────────────
        if req_id is not None:
            released = self.env.resource_mgr.release_request_record(req_id, rollback=True)
            if released:
                logger.debug(f"[FailRollback] req={req_id} → release_request_record(rollback=True) OK")
                return

        # ── 旧路径 fallback（request_table 无记录时）─────────────────
        logger.debug(f"[FailRollback] req={req_id} 不在 request_table，走 placement 兼容路径")
        rolled = 0
        try:
            if self.env.current_tree:
                placement = self.env.current_tree.get('placement', {})
                vnf_list = self.env.current_request.get('vnf', [])
                for key, alloc in list(placement.items()):
                    if not (isinstance(key, tuple) and len(key) >= 2):
                        continue
                    node, vnf_idx = int(key[0]), int(key[1])
                    vnf_t = vnf_list[vnf_idx] if vnf_idx < len(vnf_list) else vnf_idx
                    try:
                        # 统一走 release_vnf_ref（ref_count 管理，不区分 reused）
                        self.env.resource_mgr.release_vnf_ref(node, vnf_t)
                        rolled += 1
                    except Exception as e:
                        logger.warning(f"[FailRollback] 回滚失败 node={node} vnf_t={vnf_t}: {e}")
        except Exception as e:
            logger.warning(f"[FailRollback] CPU/MEM回滚异常: {e}")

        logger.debug(f"[FailRollback] fallback路径回滚 {rolled} 个VNF，BW不回滚")

    def _manual_save_resources_only(self):
        # [BugFix] 不写 resources_allocated，防止污染下一轮注册。
        pass

    def _add_request_to_lifecycle_manager(self):
        """
        向 lifecycle 管理器注册请求——仅记录到达时间、存活时长等时序元数据。
        CPU/MEM/BW 的实际账务由 AllResourceManager 新内核统一管理
        （request_table / edge_allocations），resources_allocated 不再是释放的依据。
        资源释放路径：release_request_record(req_id)，参见 AllResourceManager。
        """
        if not hasattr(self.env, 'request_manager') or not self.env.request_manager:
            return False
        if self.env.current_request is None:
            return False
        req_id = self.env.current_request.get('id')
        if req_id is None:
            logger.warning("[Lifecycle注册] current_request 缺少 'id'，跳过注册")
            return False
        if req_id in self.env.request_manager.active_requests:
            return True

        # 仅传入元数据载荷 — placement/tree 仅供统计展示，
        # 资源释放由新内核通过 release_request_record(req_id) 统一处理。
        resources_meta = {
            'placement': {},   # 兼容结构占位；实际账务由新内核统一管理
            'tree':      {},   # 同上 — 带宽释放仅通过 edge_allocations
        }
        logger.debug(f"[Lifecycle注册] req={req_id} (仅元数据注册，资源释放由新内核统一处理)")
        try:
            ok = self.env.request_manager.register_request(
                request=self.env.current_request,
                resources_allocated=resources_meta,
            )
            if ok:
                self.env.current_request.pop('resources_allocated', None)
            return ok
        except Exception as e:
            logger.warning(f"[Lifecycle注册] 注册失败: {e}")
            return False

    def _reset_vnf_phase_only(self):
        self.env.current_deployment_target = None
        self.env.subgoal_step_count = 0
        # ⚠️ [问题2修复] subgoal切换时必须清零两个独立计数器
        # 否则跨subgoal的失败会累加，导致不同目标上的失败触发误判
        self.env._timeout_count = 0
        self.env._island_count = 0
        self.env._consecutive_timeout_count = 0  # 别名同步
        self.env._deadlock_step_count = 0        # [Fix-DeadLock] 切换子目标时清死锁计数
