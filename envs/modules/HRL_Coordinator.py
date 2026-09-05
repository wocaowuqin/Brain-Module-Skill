"""
envs/modules/HRL_Coordinator.py
====================================
分层强化学习协调器 - TA-HRL v4
====================================

【模块定位】
本模块是 TA-HRL 系统的核心调度层，位于高层策略（HRLAgent / HighLevelController）与低层
执行器（LowLevelController）之间。它不直接做网络决策，而是负责：
  - 驱动高层选目标 → 驱动低层执行 → 汇总奖励 → 存储经验 → 推进 episode

【两层 HRL 执行流程】
  每个 episode 由若干个 high-low cycle 组成，每个 cycle 的执行顺序如下：

  run_episode()
    └─ while not done:
         └─ run_high_low_cycle()
               ├─ 1. 获取高层 action mask（过滤不可达 + 资源不足节点）
               ├─ 2. top-k 收缩（按 BW + 跳数启发式排序，保留最优 k 个候选）
               ├─ 3. 高层 agent 选子目标（actual_high_action = target_node）
               ├─ 4. BW 感知目标替换（_select_bw_feasible_target）
               ├─ 5. 设置子目标并执行高层 step（set_high_level_goal / step_high_level）
               ├─ 6. 低层执行循环（max_low_steps 步）
               │      ├─ planner 主导：按概率用 compute_bw_aware_path 给定路径
               │      ├─ planner 受阻时重规划（最多 3 次，用当前位置为起点）
               │      └─ planner 失效时：低层 policy 接管做局部修正
               ├─ 7. 低层经验存储（只存 policy 接管的 transition）
               └─ 8. 高层经验存储 + 更新统计

【Planner-Policy 协同机制（方案2）】
  - ep < 500：planner 概率 0.7，低层主要跟 planner 走，policy 学局部绕路
  - ep < 2000：planner 概率 0.4，policy 承担更多决策
  - ep ≥ 2000：planner 概率 0.15，policy 主导，planner 仅在严重受阻时兜底
  - planner 路径走完或 BW 受阻时自动重规划，用当前位置而非原起点重算
  - 只有 policy 接管的 transition 写入 low_memory，避免 replay 被 planner 样本淹没

【高层目标筛选三级过滤】
  1. action mask：屏蔽不可达节点 + 资源不足 DC + 当次 episode 已失败目标
  2. top-k 收缩：_score_high_candidates() 按 BW 富余 + 跳数打分，dest 阶段权重 BW，
     VNF 阶段权重 CPU/MEM slack，只保留 top-k（ep<2000 取 5，ep≥2000 取 3）
  3. BW 感知替换：_select_bw_feasible_target() 检查路径瓶颈 BW，不足时换更优目标

【主要函数索引】
  run_episode()                   episode 主入口，驱动完整请求处理流程
  run_high_low_cycle()            单次高-低循环：高层选目标 → 低层执行 → 汇总
  _score_high_candidates()        对高层候选目标按 BW+跳数启发式打分
  _apply_high_topk_mask()         把 action mask 收缩到 top-k 得分最高的候选
  _select_bw_feasible_target()    BW 感知目标替换（v3：瓶颈 BW < need 立即换）
  _fallback_success_check()       成功判断兜底（高层 controller 不可用时）
  _update_stats()                 统计 subgoal 成功/失败，包含 BW/CPU/timeout 等原因
  _store_transition()             统一经验存储入口，兼容多种 agent 接口

【与其他模块的依赖关系】
  → HRLAgent（high_agent）        高层策略，select_action() + store_transition_high()
  → HRLAgent（low_agent）         低层策略，select_action() + store_transition_low()
  → HighLevelController           set_high_level_goal() / step_high_level() / get_high_level_action_mask()
  → LowLevelController            step_low_level() / get_low_level_action_mask() / compute_bw_aware_path()
  → SFCEnv                        get_high_level_state_graph() / get_state() / current_request 等状态字段
  → AllResourceManager            通过 env.resource_mgr 间接访问，用于 BW 感知打分

【关键设计决策记录】
  - actual_high_action = int(target_node)：BW 过滤后统一，保证 set_high_level_goal 和
    step_high_level 使用同一节点 ID（HighLevelController 把 action_idx 直接当节点 ID 用）
  - _unreachable_targets 在 episode 开始重置，cycle 末尾不清空，跨 cycle 有效
  - 高层 embedding 在 _update_high_level() 中 detach()，防止梯度污染 encoder
  - 高层奖励 = 0.1 × low_total_reward，不额外叠加 milestone/completion bonus

v4.2 升级（候选评分 + anchor + 树级奖励）：
  [1] 低层动作选择改为候选邻居逐点评分：
      调用 low_level_controller.get_low_level_candidates()，
      把 candidate_indices / current_node_idx / candidate_local_feats 传入
      low_policy.select_action()，走路径1而非全图 logits。
  [2] destination 阶段从 last_vnf 出发，并允许低层复用合法的
      downstream tree edge；任意下游 anchor 仍是实验功能，默认禁用。
  [3] low_topk 放宽到 5；距离目标 <= 2 时不做 top-k 收缩。
  [4] max_subgoal_steps 默认 35（原 25）。
  [5] destination 阶段方向约束放宽：d_next >= d_current + 2 才屏蔽（原 > d_current）。
  [6] 修正 bw 字段：get_state() 内改用 bw_origin。
  [7] 高层奖励改为树级增量奖励：
      delta_connected * 8 + delta_vnf * 6 - delta_edges * 1.5
      - delta_inst * 2 + delta_reuse * 1
"""

import numpy as np
import logging
import copy
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class HRL_Coordinator:
    _SOFT_DEST_FAILURE_REASONS = frozenset({
        'timeout',
        'stuck',
        'consecutive_timeout',
        'deadlock_single_candidate',
        'oscillation_truncated',
        'no_progress',
        'cycle_blocked',
    })

    def __init__(self, env, high_agent, low_agent, config=None):
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        # config 有些入口不会显式传给 Coordinator，因此这里合并 env.config 作为兜底。
        _env_config = getattr(env, 'config', {})
        if not isinstance(_env_config, dict):
            _env_config = {}
        _arg_config = config or {}
        self.config = {**_env_config, **_arg_config}

        # 消融：single_dqn 旁路高层策略，低层 goal_emb 固定全零。
        # 读取顺序尽量鲁棒：CLI/config/env/agent 任一处标记 single_dqn 都生效。
        _ablation_variant = (
            ('single_dqn' if getattr(high_agent, '_single_dqn', False) else None)
            or ('single_dqn' if getattr(low_agent, '_single_dqn', False) else None)
            or getattr(env, '_ablation_variant', None)
            or getattr(env, 'ablation_variant', None)
            or getattr(high_agent, '_ablation_variant', None)
            or getattr(low_agent, '_ablation_variant', None)
            or self.config.get('ablation_variant')
            or self.config.get('ablation', {}).get('variant')
            or self.config.get('hrl', {}).get('ablation_variant')
            or 'full'
        )
        self._ablation_single_dqn = (_ablation_variant == 'single_dqn')
        self.env._ablation_single_dqn = self._ablation_single_dqn
        logger.info(
            f"[Ablation] Coordinator variant={_ablation_variant} "
            f"single_dqn={self._ablation_single_dqn}"
        )

        if hasattr(env, 'resource_mgr'):
            self.resource_mgr = env.resource_mgr
        else:
            self.resource_mgr = None

        self.max_low_steps = self.config.get('max_low_steps', 50)
        self.deployment_executor = self.config.get('deployment_executor', 'policy')
        if self.deployment_executor not in {'policy', 'bw_planner'}:
            raise ValueError(
                f"unsupported deployment_executor={self.deployment_executor!r}; "
                "expected 'policy' or 'bw_planner'"
            )
        # Keep the low-level policy in control, but prevent an untrained policy
        # from repeatedly choosing a stay/backtrack action when the constrained
        # path executor has a legal progress step. The executed action is still
        # stored in replay, so training and online execution share semantics.
        self.planner_safety_guard = bool(
            self.config.get('planner_safety_guard', True)
        )
        # v4.2: max_subgoal_steps 提高到 35，给 destination 阶段更多步数
        self.env.max_subgoal_steps = self.config.get('max_subgoal_steps', 35)
        self.stats = defaultdict(int)
        self.current_episode = 0
        self.resources_released = False

        if not hasattr(self.env, '_dest_anchor_mismatch'):
            self.env._dest_anchor_mismatch = False
        if not hasattr(self.env, '_dest_step0_checked'):
            self.env._dest_step0_checked = False
        if not hasattr(self.env, '_dest_cycle_armed'):
            self.env._dest_cycle_armed = False

        self._episode_deploy_failed = set()
        # Low-level and coordinator failure bans are one phase-scoped set.
        # Keeping two independent sets made stale VNF failures survive into
        # later VNF stages and destination selection.
        self.env._episode_deploy_failed = self._episode_deploy_failed
        self._failure_memory_scope = None

        # top-k 候选收缩：先启发式排序，只让policy在最优k个候选里选
        # high_topk 在 run_high_low_cycle() 里按 episode 动态调整（前期5，后期3）
        # v4.2: low_topk 放宽到 5
        self.high_topk = self.config.get('high_topk', 5)  # 初始值，运行时会被动态覆盖

        # v4.2: 同步把 LowLevelController._low_topk 设为 5
        _llc = getattr(self.env, 'low_level_controller', None)
        if _llc is not None:
            _llc._low_topk = self.config.get('low_topk', 5)

        # 注入 ControllerSharedHelper
        try:
            from .controller_shared_helper import ControllerSharedHelper
        except ImportError:
            from controller_shared_helper import ControllerSharedHelper
        self.shared = ControllerSharedHelper(env)

        # Per-request algorithm timings. These are reset by run_episode() and
        # intentionally exclude Mininet/Ryu deployment and probe execution.
        self._decision_timings = {'high': [], 'low': []}

    def _reset_decision_timings(self):
        self._decision_timings = {'high': [], 'low': []}

    def _sync_failure_memory_scope(self):
        """Clear temporary action bans when the service stage changes."""
        request = getattr(self.env, 'current_request', None) or {}
        request_id = request.get('id')
        vnfs = request.get('vnf', []) or []
        vnf_idx = int(getattr(self.env, 'next_vnf_idx', 0))
        if vnf_idx < len(vnfs):
            scope = (request_id, 'vnf', vnf_idx)
        else:
            scope = (request_id, 'destination')

        previous_scope = getattr(self, '_failure_memory_scope', None)
        if scope != previous_scope:
            previous_request_id = (
                previous_scope[0]
                if isinstance(previous_scope, tuple) and previous_scope
                else None
            )
            if previous_request_id != request_id or scope[1] == 'destination':
                self.env._vnf_completion_schedule = None
            if previous_request_id != request_id:
                self.env._destination_completion_schedule = None
            if hasattr(self, '_unreachable_targets'):
                self._unreachable_targets.clear()
            self._episode_deploy_failed.clear()
            self.env._episode_deploy_failed = self._episode_deploy_failed
            # Low-level route bans are subgoal-stage scoped. Keeping the
            # previous stage's tabu/path history blocks legal first moves in
            # the next VNF or destination routing problem.
            self.env.hard_tabu_list = set()
            self.env.current_path_trace = []
            self.env.recent_edge_trace = []
            self.env._last_target_goal = None
            self.env._stuck_steps = 0
            self.env._last_dist_to_target = None
            self._failure_memory_scope = scope
        elif getattr(self.env, '_episode_deploy_failed', None) is not self._episode_deploy_failed:
            # Repair legacy callers that replaced the env set instead of
            # mutating the coordinator-owned set.
            self._episode_deploy_failed.update(
                getattr(self.env, '_episode_deploy_failed', set()) or set()
            )
            self.env._episode_deploy_failed = self._episode_deploy_failed
        return scope

    def _record_decision_timing(self, level, started_ns, phase, source='policy'):
        duration_ms = (time.perf_counter_ns() - int(started_ns)) / 1_000_000.0
        self._decision_timings.setdefault(level, []).append({
            'duration_ms': float(duration_ms),
            'phase': str(phase or 'unknown'),
            'source': str(source or 'policy'),
        })

    @staticmethod
    def _summarize_decision_timings(rows, phase=None):
        values = [
            float(row['duration_ms'])
            for row in rows
            if phase is None or row.get('phase') == phase
        ]
        if not values:
            return {
                'count': 0,
                'total_ms': 0.0,
                'mean_ms': None,
                'p50_ms': None,
                'p95_ms': None,
                'max_ms': None,
            }
        array = np.asarray(values, dtype=np.float64)
        return {
            'count': len(values),
            'total_ms': float(array.sum()),
            'mean_ms': float(array.mean()),
            'p50_ms': float(np.percentile(array, 50)),
            'p95_ms': float(np.percentile(array, 95)),
            'max_ms': float(array.max()),
        }

    def _decision_timing_summary(self, request_total_ms):
        high_rows = self._decision_timings.get('high', [])
        low_rows = self._decision_timings.get('low', [])
        high = self._summarize_decision_timings(high_rows)
        low = self._summarize_decision_timings(low_rows)
        decision_total_ms = float(high['total_ms'] + low['total_ms'])
        return {
            'scope': 'HRL algorithm only; excludes Mininet/Ryu/VNF execution',
            'high_level': high,
            'high_vnf_placement': self._summarize_decision_timings(
                high_rows, 'vnf_deployment'
            ),
            'high_destination_connection': self._summarize_decision_timings(
                high_rows, 'destination_connection'
            ),
            'low_level': low,
            'low_vnf_routing': self._summarize_decision_timings(
                low_rows, 'vnf_deployment'
            ),
            'low_destination_routing': self._summarize_decision_timings(
                low_rows, 'destination_connection'
            ),
            'decision_total_ms': decision_total_ms,
            'request_total_ms': float(request_total_ms),
            'non_action_algorithm_ms': max(
                0.0, float(request_total_ms) - decision_total_ms
            ),
        }

    def _candidate_feats(self, cand_info):
        if cand_info is None:
            return None
        feats = cand_info.get('features')
        if feats is not None and getattr(self.env, '_ablation_candidate_feats', False):
            return np.zeros_like(feats)
        return feats

    @staticmethod
    def _align_low_candidates_with_mask(cand_info, action_mask, current_node):
        """Make policy candidates match the final executable low-level mask.

        The controller builds candidates from the hard topology/resource mask.
        The coordinator may subsequently narrow that mask with VNF or
        destination path evidence.  Reusing the pre-narrowing Top-K set can
        leave no common action and force the policy into its full-logit
        fallback.  Keep aligned features where possible; when the final mask
        contains only actions pruned from Top-K, rebuild from the mask and let
        the scorer use zero local features.
        """
        mask = np.asarray(action_mask).reshape(-1)
        open_indices = [int(i) for i in np.where(mask > 0)[0]]
        if not open_indices:
            return None

        if not cand_info or not cand_info.get('indices'):
            return {
                'indices': open_indices,
                'current_node': int(current_node),
                'features': None,
                'realigned_from_mask': True,
            }

        original = [int(i) for i in cand_info.get('indices', [])]
        open_set = set(open_indices)
        kept_positions = [
            position for position, node in enumerate(original)
            if node in open_set
        ]
        if not kept_positions:
            return {
                'indices': open_indices,
                'current_node': int(cand_info.get('current_node', current_node)),
                'features': None,
                'realigned_from_mask': True,
            }

        aligned = dict(cand_info)
        aligned['indices'] = [original[position] for position in kept_positions]
        features = cand_info.get('features')
        if features is not None:
            try:
                aligned['features'] = features[kept_positions]
            except (TypeError, IndexError):
                aligned['features'] = [features[position] for position in kept_positions]
        aligned['realigned_from_mask'] = len(kept_positions) != len(original)
        return aligned

    def _single_candidate_progress_signature(self, target_node):
        """Describe observable destination-routing progress for deadlock detection."""
        current_node = getattr(self.env, 'current_node_location', None)
        target_distance = None
        controller = getattr(self.env, 'low_level_controller', None)
        if controller is not None and current_node is not None and target_node is not None:
            try:
                target_distance = controller._get_hop_distance(current_node, target_node)
            except Exception:
                target_distance = None

        try:
            tree_edges = len(self.shared.get_positive_tree_edge_set())
        except Exception:
            tree = (getattr(self.env, 'current_tree', None) or {}).get('tree', {})
            tree_edges = sum(1 for flow in tree.values() if float(flow) > 0.0)
        try:
            connected_dests = len(self.shared.get_connected_dests_view())
        except Exception:
            connected_dests = len(
                (getattr(self.env, 'current_tree', None) or {}).get(
                    'connected_dests', set()
                )
            )
        return (
            current_node,
            target_node,
            target_distance,
            tree_edges,
            connected_dests,
        )

    def _update_single_candidate_deadlock(self, target_node):
        signature = self._single_candidate_progress_signature(target_node)
        previous = getattr(self.env, '_deadlock_progress_signature', None)
        progressed = previous is None or signature != previous
        count = 1 if progressed else getattr(self.env, '_deadlock_step_count', 0) + 1
        self.env._deadlock_progress_signature = signature
        self.env._deadlock_step_count = count
        return count, progressed, signature

    def _reset_single_candidate_deadlock(self):
        self.env._deadlock_step_count = 0
        self.env._deadlock_progress_signature = None

    def _bw_planner_action(self, llc, blocked_edges):
        """Return one physically feasible planner step for the active subgoal."""
        if llc is None or not hasattr(llc, 'compute_bw_aware_path'):
            return None
        phase = getattr(self.env, 'current_phase', None)
        if phase == 'vnf_deployment':
            target = getattr(self.env, 'current_deployment_target', None)
        elif phase == 'destination_connection':
            target = getattr(self.env, 'current_target_node', None)
        else:
            return None
        current = getattr(self.env, 'current_node_location', None)
        if current is None or target is None:
            return None

        path = None
        if hasattr(llc, 'compute_progress_path'):
            path = llc.compute_progress_path(
                current, target, excluded_edges=blocked_edges
            )
        if not path:
            path = llc.compute_bw_aware_path(
                current, target, excluded_edges=blocked_edges
            )
        if not path or path[0] != current:
            return None
        if len(path) == 1:
            return int(current)

        next_hop = int(path[1])
        if next_hop not in self.env.resource_mgr.get_neighbors(current):
            return None
        tree = (getattr(self.env, 'current_tree', None) or {}).get('tree', {})
        is_tree_edge = float(tree.get((current, next_hop), 0.0)) > 0.0
        bw_need = float((self.env.current_request or {}).get('bw_origin', 0.0))
        bw_available = self.env.resource_mgr.pool.get_available_bandwidth(current, next_hop)
        if not is_tree_edge and bw_available + 1e-9 < bw_need:
            return None
        return next_hop

    def _apply_planner_safety_guard(
            self, policy_action, planner_action, low_mask, llc, target_node):
        """Replace only a demonstrably non-progressing policy action."""
        policy_action = int(policy_action)
        if not self.planner_safety_guard or planner_action is None:
            return policy_action, False

        planner_action = int(planner_action)
        current = getattr(self.env, 'current_node_location', None)
        if current is None or target_node is None or int(current) == int(target_node):
            return policy_action, False
        if not (0 <= planner_action < len(low_mask)) or low_mask[planner_action] <= 0:
            return policy_action, False
        if planner_action == policy_action:
            return policy_action, False

        recent = list(getattr(self, '_pos_history', [])[-3:])
        policy_repeats = policy_action == int(current) or policy_action in recent
        policy_progress = None
        planner_progress = None
        if llc is not None and hasattr(llc, '_get_hop_distance'):
            try:
                current_hop = llc._get_hop_distance(int(current), int(target_node))
                policy_hop = llc._get_hop_distance(policy_action, int(target_node))
                planner_hop = llc._get_hop_distance(planner_action, int(target_node))
                policy_progress = policy_hop < current_hop
                planner_progress = planner_hop < current_hop
            except Exception:
                pass

        should_guard = policy_repeats or (
            policy_progress is False and planner_progress is True
        )
        if not should_guard:
            return policy_action, False

        logger.debug(
            f"[PlannerSafetyGuard] phase={getattr(self.env, 'current_phase', None)} "
            f"cur={current} target={target_node} policy={policy_action} "
            f"planner={planner_action} repeat={int(policy_repeats)} "
            f"policy_progress={policy_progress} planner_progress={planner_progress}"
        )
        return planner_action, True

    def _apply_vnf_completion_guard(
            self, policy_action, planner_action, low_mask, llc, target_node):
        """Keep only low-level moves that preserve end-to-end completion."""
        if getattr(self.env, 'current_phase', None) != 'vnf_deployment':
            return int(policy_action), False
        high_controller = getattr(self.env, 'high_level_controller', None)
        if high_controller is None or not hasattr(
                high_controller, '_can_complete_after_placement'):
            return int(policy_action), False
        request = getattr(self.env, 'current_request', None) or {}
        current = getattr(self.env, 'current_node_location', None)
        if current is None or target_node is None:
            return int(policy_action), False
        current_idx = int(getattr(self.env, 'next_vnf_idx', 0))
        destinations = {int(node) for node in request.get('dest', [])}
        bw_req = float(request.get('bw_origin', 0.0))

        def is_safe(action):
            action = int(action)
            if not (0 <= action < len(low_mask)) or low_mask[action] <= 0:
                return False
            if action == int(current) and int(current) != int(target_node):
                return False
            return high_controller._can_complete_after_placement(
                current_idx,
                int(target_node),
                bw_req,
                destinations,
                previous_override=action,
                prefix_occupied={int(current)},
            )

        policy_action = int(policy_action)
        if is_safe(policy_action):
            return policy_action, False

        alternatives = []
        if planner_action is not None:
            alternatives.append(int(planner_action))
        valid = [int(index) for index in np.where(low_mask > 0)[0]]
        if llc is not None and hasattr(llc, '_get_hop_distance'):
            valid.sort(key=lambda node: llc._get_hop_distance(node, target_node))
        alternatives.extend(valid)
        seen = {policy_action}
        for action in alternatives:
            if action in seen:
                continue
            seen.add(action)
            if is_safe(action):
                logger.debug(
                    f"[VNFCompletionGuard] idx={current_idx} cur={current} "
                    f"target={target_node} policy={policy_action} safe={action}"
                )
                return action, True
        return policy_action, False

    def _filter_vnf_completion_mask(self, low_mask, target_node):
        """Remove VNF moves that make the remaining ordered SFT impossible.

        The previous post-selection guard could still execute the unsafe policy
        action when no replacement survived.  Filtering before policy scoring
        keeps training and execution semantics aligned and prevents a partial
        VNF spine from poisoning all later high-level masks.
        """
        if getattr(self.env, 'current_phase', None) != 'vnf_deployment':
            return low_mask, False
        high_controller = getattr(self.env, 'high_level_controller', None)
        if high_controller is None or not hasattr(
                high_controller, '_can_complete_after_placement'):
            return low_mask, False

        request = getattr(self.env, 'current_request', None) or {}
        current = getattr(self.env, 'current_node_location', None)
        if current is None or target_node is None:
            return low_mask, False
        current_idx = int(getattr(self.env, 'next_vnf_idx', 0))
        destinations = {int(node) for node in request.get('dest', [])}
        bw_req = float(request.get('bw_origin', 0.0))
        safe_mask = np.zeros_like(low_mask)

        for action in np.where(low_mask > 0)[0]:
            action = int(action)
            if action == int(current) and int(current) != int(target_node):
                continue
            if high_controller._can_complete_after_placement(
                    current_idx,
                    int(target_node),
                    bw_req,
                    destinations,
                    previous_override=action,
                    prefix_occupied={int(current)},
            ):
                safe_mask[action] = low_mask[action]

        changed = not np.array_equal(safe_mask, low_mask)
        if changed:
            logger.debug(
                f"[VNFCompletionMask] idx={current_idx} cur={current} "
                f"target={target_node} before={int(np.sum(low_mask > 0))} "
                f"after={int(np.sum(safe_mask > 0))}"
            )
        return safe_mask, changed

    def _take_tree_snapshot(self):
        """
        v4.2: 采集当前树状态快照，供 _compute_high_reward 计算增量。
        返回 dict，安全地捕获所有可能为空的字段。
        """
        snap = {
            'connected_dests': 0,
            'vnf_done': 0,
            'tree_edges': 0,
            'vnf_instances': 0,
            'reused_edges': 0,
        }
        try:
            if self.env.current_tree:
                snap['connected_dests'] = len(
                    self.shared.get_connected_dests_view())
                tree_dict = self.env.current_tree.get('tree', {})
                snap['tree_edges'] = sum(1 for f in tree_dict.values() if f > 0.0)
                # [Fix4] placement 길이 대신 instance_table의 실제 ACTIVE 인스턴스 수
                # placement는 요청 내 배치 기록 → 복용 시에도 증가 → delta_inst 오염
                _rm_snap = getattr(self.env, 'resource_mgr', None)
                if (_rm_snap is not None and
                        hasattr(_rm_snap, 'instance_table') and
                        _rm_snap.instance_table):
                    snap['vnf_instances'] = sum(
                        1 for inst in _rm_snap.instance_table.values()
                        if getattr(inst, 'state', None) == 'ACTIVE'
                    )
                else:
                    snap['vnf_instances'] = len(
                        self.env.current_tree.get('placement', {}))
                snap['reused_edges'] = sum(
                    1 for v in self.env.current_tree.get('tree_usage', {}).values()
                    if v > 1)
                # [Fix-5] include VNF instance reuse so delta_reuse covers both edge & VNF reuse
                snap['reused_edges'] += self.env.current_tree.get('reused_vnf_count', 0)
            snap['vnf_done'] = getattr(self.env, 'next_vnf_idx', 0)
        except Exception:
            pass
        return snap

    def _compute_high_reward(self, low_total_reward, low_step=0):
        """
        v4.2: 树级增量奖励。
        每个 high-low cycle 结束后，根据树结构的显式变化量计算高层奖励。

        奖励项：
            +8.0 * delta_connected_dests    新连通目的地数
            +6.0 * delta_vnf_done           新完成 VNF 数
            -1.5 * delta_tree_edges         新增树边（越少越好，鼓励复用）
            -2.0 * delta_vnf_instances      新增 VNF 实例（越少越好，鼓励共享）
            +1.0 * delta_reused_edges       新增被复用的树边（共享奖励）
            -0.1 * max(0, low_step - 5)     VNF子目标超过5步的导航代价（引导选近节点）

        兜底：若增量全为 0，退化为 0.05 * low_total_reward（弱信号保证梯度）
        """
        snap_after = self._take_tree_snapshot()
        snap_before = getattr(self, '_snap_before', snap_after)

        delta_connected = snap_after['connected_dests'] - snap_before['connected_dests']
        delta_vnf       = snap_after['vnf_done']        - snap_before['vnf_done']
        delta_edges     = snap_after['tree_edges']      - snap_before['tree_edges']
        delta_inst      = snap_after['vnf_instances']   - snap_before['vnf_instances']
        delta_reuse     = snap_after['reused_edges']    - snap_before['reused_edges']

        # 只取非负增量（避免因 episode 重置导致负值污染）
        delta_connected = max(0, delta_connected)
        delta_vnf       = max(0, delta_vnf)
        delta_edges     = max(0, delta_edges)
        delta_inst      = max(0, delta_inst)
        delta_reuse     = max(0, delta_reuse)

        # VNF 서브골 완료 시 저층 탐색 스텝 패널티
        # 근거: 고층은 전체 그래프를 보지만 Q값이 거리 비용을 학습하지 못함
        # 5스텝 초과분에 -0.1씩 패널티 → 가까운 DC를 선택하도록 유도
        _step_penalty = -0.10 * max(0, low_step - 5) if delta_vnf > 0 else 0.0

        # [Fix-C] 提升共享奖励系数（强化稀疏梯度信号）
        # delta_inst  : -2.0 → -3.0（新建实例惩罚加重，驱动高层优先选可复用DC节点）
        # delta_reuse : +1.0 → +2.5（复用边奖励增强，量级接近 delta_vnf 的40%）
        high_reward = (
            8.0 * delta_connected
            + 6.0 * delta_vnf
            - 1.5 * delta_edges
            - 3.0 * delta_inst
            + 2.5 * delta_reuse
            + _step_penalty
        )

        # 首个 dest 惩罚：用 detour_ratio 替代纯边数惩罚
        # - detour_ratio ≤ 1.5 或 shortest_hop < 2：基本不罚（走法合理）
        # - detour_ratio > 1.5 且路径有意义：线性惩罚绕路超出量
        _first_dest_detour_penalty = 0.0
        if snap_before['connected_dests'] == 0 and delta_connected >= 1:
            _detour_ratio  = float(getattr(self.env, '_first_dest_detour_ratio', 1.0) or 1.0)
            _shortest_hop  = int(getattr(self.env, '_first_dest_shortest_hop', 1) or 1)
            if _shortest_hop >= 2 and _detour_ratio > 1.5:
                _first_dest_detour_penalty = -2.5 * (_detour_ratio - 1.5)
            high_reward += _first_dest_detour_penalty
            logger.debug(
                f"[FirstDestPenalty] delta_edges={delta_edges} "
                f"detour_ratio={_detour_ratio:.2f} shortest_hop={_shortest_hop} "
                f"detour_penalty={_first_dest_detour_penalty:.2f}"
            )

        # 兜底：无结构变化时加性弱信号，不覆盖已有的结构惩罚
        # [Fix P0] 原来是 = (覆盖)，会把"长了很多边但没完成目标"的负分洗成弱正信号
        if delta_connected == 0 and delta_vnf == 0:
            high_reward += 0.05 * low_total_reward

        # [Item 3] reuse_per_new_edge: 本轮每条新增边带来的复用增益（可 > 1，不是比例）
        # 区别于 sharing_eff（唯一复用边占比，≤1）和 reuse_density（额外复用次数密度）
        _reuse_per_new_edge = delta_reuse / max(1, delta_edges) if delta_edges > 0 else 0.0
        _is_first_dest_log = (snap_after['connected_dests'] == 1 and delta_connected == 1)
        if delta_connected > 0 or delta_edges > 0:
            logger.debug(
                f"[TreeReuse] dest={snap_after['connected_dests']} "
                f"reuse_per_new_edge={_reuse_per_new_edge:.2f} "
                f"delta_reuse={delta_reuse} delta_edges={delta_edges} "
                f"first_dest={_is_first_dest_log}"
            )

        logger.debug(
            f"[HighReward] Δconn={delta_connected} Δvnf={delta_vnf} "
            f"Δedges={delta_edges} Δinst={delta_inst} Δreuse={delta_reuse} "
            f"reuse_per_new_edge={_reuse_per_new_edge:.2f} first_dest={_is_first_dest_log} "
            f"first_penalty={_first_dest_detour_penalty:.2f} "
            f"→ reward={high_reward:.2f} (low={low_total_reward:.2f}) "
            f"step_pen={_step_penalty:.2f} "
            f"dest={snap_after['connected_dests']} vnf={snap_after['vnf_done']}"
        )
        return high_reward

    def _check_bw_reachability(self, source, target, bw_req):
        """
        检查在当前剩余带宽图上，source -> target 是否仍存在可达路径。
        返回:
            {
                'reachable_bw': bool,
                'shortest_bw_hop': int,   # 不可达时为 -1
                'bottleneck_bw': float,   # 不可达时为 0.0
            }
        """
        try:
            import networkx as nx

            rm = getattr(self, 'resource_mgr', None)
            if rm is None or source is None or target is None:
                return {
                    'reachable_bw': False,
                    'shortest_bw_hop': -1,
                    'bottleneck_bw': 0.0,
                }

            G = nx.DiGraph()
            for u in range(rm.n):
                for v in rm.get_neighbors(u):
                    try:
                        avail_bw = rm.pool.get_available_bandwidth(u, v)
                    except Exception:
                        avail_bw = 0.0
                    _tree_b = (self.env.current_tree or {}).get('tree', {}) if hasattr(self, 'env') else {}
                    if (_tree_b.get((u, v), 0.0) > 0.0 or avail_bw >= bw_req):
                        G.add_edge(u, v, bw=max(float(avail_bw), bw_req))  # 树边用 bw_req 占位

            if source not in G or target not in G:
                return {
                    'reachable_bw': False,
                    'shortest_bw_hop': -1,
                    'bottleneck_bw': 0.0,
                }

            path = nx.shortest_path(G, source, target)
            hops = max(0, len(path) - 1)
            bottleneck = min(G[path[i]][path[i + 1]]['bw'] for i in range(hops)) if hops > 0 else float('inf')

            return {
                'reachable_bw': True,
                'shortest_bw_hop': hops,
                'bottleneck_bw': float(bottleneck),
            }

        except Exception:
            return {
                'reachable_bw': False,
                'shortest_bw_hop': -1,
                'bottleneck_bw': 0.0,
            }

    def _check_target_incident_bw(self, target, bw_req):
        """
        检查目标节点相邻边中，满足带宽需求的边数量。
        用于判断 timeout 时是否是“目标口子被堵死”。
        """
        try:
            rm = getattr(self, 'resource_mgr', None)
            if rm is None or target is None:
                return {'target_incident_ok': 0, 'target_incident_total': 0}

            nbrs = rm.get_neighbors(target)
            ok = 0
            _tree_c = (self.env.current_tree or {}).get('tree', {}) if hasattr(self, 'env') else {}
            for n in nbrs:
                try:
                    _in_bw_ok = rm.pool.get_available_bandwidth(n, target) >= bw_req
                    _in_tree = _tree_c.get((n, target), 0.0) > 0.0
                    if _in_bw_ok or _in_tree:
                        ok += 1
                except Exception:
                    pass
            return {'target_incident_ok': ok, 'target_incident_total': len(nbrs)}
        except Exception:
            return {'target_incident_ok': 0, 'target_incident_total': 0}







    def run_high_low_cycle(self, high_obs, training=True):
        if hasattr(self, 'resources_released'):
            self.resources_released = False

        # Any override belongs only to the previous immutable high-decision
        # state.  Low-level execution may have changed the tree and ledger.
        self.env._cycle_high_mask_override = None

        # v4.2: 记录 cycle 开始时的树状态快照，用于树级增量奖励
        self._snap_before = self._take_tree_snapshot()

        self.env._dest_anchor_mismatch = False
        self.env._dest_step0_checked = False
        self.env._dest_cycle_armed = False

        request = getattr(self.env, 'current_request', None) or {}
        vnf_count = len(request.get('vnf', []) or [])
        if (
            getattr(self.env, 'next_vnf_idx', 0) >= vnf_count
            and getattr(self.env, 'current_phase', None) == 'vnf_deployment'
        ):
            self.env.current_phase = 'destination_connection'
            self.env.current_deployment_target = None
        self._sync_failure_memory_scope()

        # Reuse the complete VNF plan proven at the first placement decision.
        # The schedule is only a certificate: every stage is revalidated against
        # the current hard resource and directed-edge ledger.  Validate it before
        # building the full high-level mask: the old order paid for the recursive
        # all-DC completion scan and then immediately replaced its result.
        _scheduled_vnf_entry = None
        high_mask = None
        if getattr(self.env, 'current_phase', None) == 'vnf_deployment':
            _schedule = getattr(self.env, '_vnf_completion_schedule', None)
            _request_id = request.get('id')
            _current_idx = int(getattr(self.env, 'next_vnf_idx', 0))
            if (
                isinstance(_schedule, dict)
                and _schedule.get('request_id') == _request_id
            ):
                _scheduled_vnf_entry = next((
                    dict(item) for item in _schedule.get('stages', [])
                    if int(item.get('stage_idx', -1)) == _current_idx
                ), None)
            if _scheduled_vnf_entry is not None:
                _scheduled_node = int(_scheduled_vnf_entry['node'])
                _scheduled_path = [
                    int(node) for node in _scheduled_vnf_entry.get('path', [])
                ]
                _chain_now = list(getattr(self.env, 'chain_nodes', []) or [])
                _scheduled_start = (
                    int(_chain_now[-1])
                    if _chain_now else int(request.get('source'))
                )
                _vnfs_now = list(request.get('vnf', []) or [])
                _cpu_now = request.get('cpu_origin', []) or request.get('vnf_cpu', [])
                _mem_now = request.get('memory_origin', []) or request.get('vnf_mem', [])
                _req_cpu_now = float(_cpu_now[_current_idx])
                _req_mem_now = float(_mem_now[_current_idx])
                _probe_now = self.env.resource_mgr.probe_vnf_deploy(
                    _scheduled_node,
                    _vnfs_now[_current_idx],
                    _req_cpu_now,
                    _req_mem_now,
                )
                _path_shape_ok = (
                    bool(_scheduled_path)
                    and _scheduled_path[0] == _scheduled_start
                    and _scheduled_path[-1] == _scheduled_node
                )
                _path_edges_ok = _path_shape_ok
                _positive_edges = self.shared.get_positive_tree_edge_set()
                _bw_now = float(request.get('bw_origin', 0.0))
                _destinations_now = {int(node) for node in request.get('dest', [])}
                _existing_nodes = {
                    int(node) for edge in _positive_edges for node in edge
                }
                for _path_pos, (_u, _v) in enumerate(zip(
                        _scheduled_path, _scheduled_path[1:])):
                    if _v not in self.env.resource_mgr.get_neighbors(_u):
                        _path_edges_ok = False
                        break
                    if _v in _destinations_now and _v != _scheduled_node:
                        _path_edges_ok = False
                        break
                    if _v in _existing_nodes and _v != _scheduled_start:
                        _path_edges_ok = False
                        break
                    if (
                        (_u, _v) not in _positive_edges
                        and self.env.resource_mgr.pool.get_available_bandwidth(_u, _v)
                        + 1e-9 < _bw_now
                    ):
                        _path_edges_ok = False
                        break
                if (
                    _probe_now.get('ok', False)
                    and _path_edges_ok
                    and 0 <= _scheduled_node < int(self.env.n)
                ):
                    _schedule_mask = np.zeros(int(self.env.n), dtype=np.float32)
                    _schedule_mask[_scheduled_node] = 1.0
                    high_mask = _schedule_mask
                    self.env._scheduled_high_mask_override = _scheduled_node
                    _hlc_schedule = getattr(
                        self.env, 'high_level_controller', None
                    )
                    if _hlc_schedule is not None:
                        _hlc_schedule._completion_path_evidence = {
                            _scheduled_node: list(_scheduled_path)
                        }
                    logger.debug(
                        "[VNFCompletionSchedule] req=%s idx=%s node=%s path=%s",
                        _request_id, _current_idx, _scheduled_node,
                        _scheduled_path,
                    )
                else:
                    logger.warning(
                        "[VNFCompletionScheduleInvalid] req=%s idx=%s node=%s "
                        "probe=%s path_ok=%s path=%s",
                        _request_id, _current_idx, _scheduled_node,
                        _probe_now.get('reason'), int(_path_edges_ok),
                        _scheduled_path,
                    )
                    self.env._vnf_completion_schedule = None
                    self.env._scheduled_high_mask_override = None
                    _scheduled_vnf_entry = None

        if high_mask is None:
            high_mask = self.env.get_high_level_action_mask()

        if (
            getattr(self.env, 'current_phase', None) == 'destination_connection'
            and self.config.get('destination_joint_completion_shield', True)
        ):
            _llc_joint = getattr(self.env, 'low_level_controller', None)
            _request_id = request.get('id')
            _connected_now = set(self.shared.get_connected_dests_view())
            _schedule = getattr(
                self.env, '_destination_completion_schedule', None
            )
            _next_plan_step = None
            if (
                isinstance(_schedule, dict)
                and _schedule.get('request_id') == _request_id
            ):
                _next_plan_step = next((
                    dict(item) for item in _schedule.get('steps', [])
                    if int(item.get('target', -1)) not in _connected_now
                ), None)
                if _next_plan_step is not None:
                    _valid, _invalid_reason = (
                        _llc_joint.validate_destination_plan_step(
                            _next_plan_step
                        )
                    )
                    if not _valid:
                        logger.info(
                            "[DestinationScheduleInvalid] req=%s target=%s "
                            "anchor=%s reason=%s path=%s",
                            _request_id,
                            _next_plan_step.get('target'),
                            _next_plan_step.get('anchor'),
                            _invalid_reason,
                            _next_plan_step.get('path'),
                        )
                        self.env._destination_completion_schedule = None
                        _next_plan_step = None

            if _next_plan_step is None:
                _joint_plan = (
                    _llc_joint.plan_destination_completion()
                    if _llc_joint is not None
                    and hasattr(_llc_joint, 'plan_destination_completion')
                    else None
                )
                if _joint_plan:
                    self.env._destination_completion_schedule = {
                        'request_id': _request_id,
                        'steps': copy.deepcopy(list(_joint_plan)),
                    }
                    _next_plan_step = dict(_joint_plan[0])
                    logger.debug(
                        "[DestinationScheduleCreated] req=%s steps=%s",
                        _request_id,
                        [
                            {
                                'target': item.get('target'),
                                'anchor': item.get('anchor'),
                                'path': item.get('path'),
                            }
                            for item in _joint_plan
                        ],
                    )

            if _next_plan_step is not None:
                _planned_target = int(_next_plan_step['target'])
                _joint_mask = np.zeros_like(high_mask)
                if (
                    0 <= _planned_target < len(high_mask)
                    and high_mask[_planned_target] > 0
                ):
                    _joint_mask[_planned_target] = 1.0
                    high_mask = _joint_mask
                    self.env._active_destination_plan_step = _next_plan_step
                    # The decoder used the latest resource snapshot and found
                    # an explicit committable path.  A prior soft routing ban
                    # for the same target is stale after the tree has changed.
                    self._unreachable_targets.discard(_planned_target)
                else:
                    self.env._active_destination_plan_step = None
            else:
                self.env._active_destination_plan_step = None
                _req_diag = getattr(self.env, 'current_request', None) or {}
                _tree_diag = getattr(self.env, 'current_tree', None) or {}
                logger.warning(
                    "[JointDestinationPlanEmpty] req=%s chain=%s pending=%s "
                    "positive_edges=%s node_stage=%s bw=%.3f",
                    _req_diag.get('id'),
                    list(getattr(self.env, 'chain_nodes', []) or []),
                    sorted(
                        set(_req_diag.get('dest', []) or [])
                        - set(self.shared.get_connected_dests_view())
                    ),
                    sorted(self.shared.get_positive_tree_edge_set()),
                    dict(_tree_diag.get('node_stage', {}) or {}),
                    float(_req_diag.get('bw_origin', 0.0)),
                )
                # Do not fall back to independent receiver reachability.  If a
                # complete joint tree cannot be decoded, every partial commit
                # can only consume bandwidth and make the terminal failure less
                # recoverable.
                high_mask = np.zeros_like(high_mask)

        # [Fix-C] 最后1个dest强制解封：必须在unreachable清零之前执行
        # 若最后1个dest在封禁名单里，先解封，再让后续mask逻辑正常处理
        # [Fix-C-v2] 解封前检查 _failed_anchors_for_target：
        #   若已失败的anchor数 >= _MAX_RETRIES，说明真的无路可走，不再解封
        #   避免 Fix-C 与失败记忆机制形成死循环（封禁→解封→封禁→解封...）
        if self.env.current_request and self.env.current_tree:
            _all_d_fc = set(self.env.current_request.get('dest', []))
            _done_d_fc = self.shared.get_connected_dests_view()
            _pending_fc = _all_d_fc - _done_d_fc
            if len(_pending_fc) == 1:
                _last_dest = next(iter(_pending_fc))
                if _last_dest in self._unreachable_targets:
                    # 检查是否已穷尽所有可用anchor
                    _failed_anchors_fc = getattr(
                        self.env, '_failed_anchors_for_target', {}
                    ).get(_last_dest, set())
                    _nodes_on_tree_fc = set(getattr(self.env, 'nodes_on_tree', set()))
                    _total_vnf_fc = len(self.env.current_request.get('vnf', []))                         if self.env.current_request else 0
                    _full_stage_fc = {
                        n for n in _nodes_on_tree_fc
                        if self.env.current_tree.get('node_stage', {}).get(n, 0) >= _total_vnf_fc
                    } if _total_vnf_fc > 0 else _nodes_on_tree_fc
                    _MAX_RETRIES_FC = min(max(1, len(_full_stage_fc)), 5)
                    _available_fc = _full_stage_fc - _failed_anchors_fc
                    # 还有可用anchor → 解封让低层继续尝试
                    # 已无可用anchor → 保持封禁，让episode尽快终止
                    if len(_available_fc) > 0 and len(_failed_anchors_fc) < _MAX_RETRIES_FC:
                        self._unreachable_targets.discard(_last_dest)
                        logger.debug(
                            f"[Fix-C] 最后1个dest={_last_dest}解封，"
                            f"剩余可用anchor={len(_available_fc)}/{len(_full_stage_fc)}"
                        )
                    else:
                        logger.debug(
                            f"[Fix-C] 最后1个dest={_last_dest}已穷尽anchor"
                            f"({len(_failed_anchors_fc)}/{_MAX_RETRIES_FC})，保持封禁"
                        )

        if hasattr(self, '_unreachable_targets') and self._unreachable_targets:
            for idx in self._unreachable_targets:
                if idx < len(high_mask):
                    high_mask[idx] = 0

        if self._episode_deploy_failed:
            for idx in self._episode_deploy_failed:
                if idx < len(high_mask):
                    high_mask[idx] = 0

        if sum(high_mask) == 0:
            _req_mask_diag = getattr(self.env, 'current_request', None) or {}
            logger.warning(
                "[NoHighActionsDiag] req=%s phase=%s vnf_idx=%s "
                "joint_first=%s unreachable=%s deploy_failed=%s connected=%s",
                _req_mask_diag.get('id'),
                getattr(self.env, 'current_phase', None),
                getattr(self.env, 'next_vnf_idx', None),
                (
                    _joint_plan[0] if '_joint_plan' in locals() and _joint_plan
                    else None
                ),
                sorted(getattr(self, '_unreachable_targets', set()) or set()),
                sorted(getattr(self, '_episode_deploy_failed', set()) or set()),
                sorted(self.shared.get_connected_dests_view()),
            )
            return 0.0, True, {'error': 'no_high_actions'}

        # Candidate construction and step_high_level() run before low-level
        # mutation, so they must see this exact already-filtered hard mask.
        # Avoid rebuilding the recursive all-DC completion lookahead twice.
        self.env._cycle_high_mask_override = np.asarray(
            high_mask, dtype=np.float32
        ).copy()

        _high_decision_started_ns = time.perf_counter_ns()
        _high_decision_phase = getattr(self.env, 'current_phase', 'unknown')

        if self._ablation_single_dqn:
            # single_dqn 消融：不调用 high_agent / high_policy / 候选 scorer。
            # 环境仍需要一个合法 target 来驱动低层，所以只从当前 high mask 中随机取一个合法节点。
            valid_indices = np.where(high_mask > 0)[0]
            target_node = int(np.random.choice(valid_indices))
            agent_info = {
                'subgoal': target_node,
                'start_node': None,
                'ablation_variant': 'single_dqn',
            }
            actual_high_action = target_node

            # 后续 destination 首个目标日志会引用这些变量；ablation 分支显式置空，避免 NameError。
            _high_cand_info = None
            _high_policy = None
            _node_embs = None
            _graph_emb_1d = None

            # 关键：清掉可能残留的高层语义 embedding，保证低层 goal_emb 恒为 0。
            try:
                import torch as _torch
                goal_dim = getattr(self.low_agent, 'goal_dim', 64)
                device = getattr(self.low_agent, 'device', _torch.device('cpu'))
                if not isinstance(device, _torch.device):
                    device = _torch.device(device)
                zero_goal = _torch.zeros(1, goal_dim, device=device)
                self.low_agent.current_subgoal_emb = zero_goal
                _low_policy = getattr(self.low_agent, 'low_policy', None)
                if _low_policy is not None and hasattr(_low_policy, 'current_subgoal_emb'):
                    _low_policy.current_subgoal_emb = zero_goal
            except Exception as _sde:
                logger.debug(f"[single_dqn] 设置零 goal_emb 失败: {_sde}")

            if hasattr(self.high_agent, '_last_high_action_meta'):
                self.high_agent._last_high_action_meta = None
            logger.info(f"[single_dqn] bypass high policy, random target={target_node}")
        else:
            # [Fix P1] high_topk 必须在 apply_high_topk_mask 之前设置，否则滞后一个 cycle 生效
            self.high_topk = 5 if self.current_episode < 2000 else 3

            # 双向带宽修复：destination_connection 阶段不要基于 last_vnf 先做 top-k 预裁剪。
            # Destination candidates are not top-k pruned from last_vnf here;
            # the low-level route still needs the full pending-destination set.
            _chain_pre = getattr(self.env, 'chain_nodes', [])
            _vnf_list_pre = self.env.current_request.get('vnf', []) if self.env.current_request else []
            _vnf_done_pre = getattr(self.env, 'next_vnf_idx', 0) >= len(_vnf_list_pre) if _vnf_list_pre else False
            if not _vnf_done_pre:
                _start_for_topk = self.env.current_node_location
                high_mask = self.shared.apply_high_topk_mask(high_mask, _start_for_topk, self.high_topk)

            # [Task D] 优先走高层候选评分路径
            _hlc = getattr(self.env, 'high_level_controller', None)
            _high_cand_info = None
            if _hlc is not None and hasattr(_hlc, 'get_high_level_candidates'):
                try:
                    _high_cand_info = _hlc.get_high_level_candidates()
                    # 同步应用 unreachable 过滤到候选列表
                    if _high_cand_info and _high_cand_info['indices']:
                        _filtered_indices = [
                            idx for idx in _high_cand_info['indices']
                            if high_mask[idx] > 0
                        ]
                        if _filtered_indices:
                            _keep = set(_filtered_indices)
                            _keep_mask = [i for i, idx in enumerate(_high_cand_info['indices']) if idx in _keep]
                            _high_cand_info = {
                                'indices': _filtered_indices,
                                'features': _high_cand_info['features'][_keep_mask]
                            }
                        else:
                            _high_cand_info = None
                except Exception as _hce:
                    logger.debug(f"[Coord] get_high_level_candidates 异常: {_hce}")
                    _high_cand_info = None

            target_node = None

            # [Fix-NameError] 在路径1 if块外预初始化，防止路径2时日志块引用未定义变量
            _high_policy = None
            _node_embs = None
            _graph_emb_1d = None

            if _high_cand_info is not None and len(_high_cand_info['indices']) > 0:
                # ── 路径1：高层候选逐点评分 ──────────────────────────────
                _high_policy = getattr(self.high_agent, 'high_policy', None)
                if _high_policy is None:
                    _high_policy = getattr(self.high_agent, 'policy', None)
                # high_agent 本身可能就是 HighLevelPolicy（直接包装场景）
                if _high_policy is None and hasattr(self.high_agent, 'score_goal_candidates'):
                    _high_policy = self.high_agent
                # 尝试从 agent 内部找 HighLevelPolicy 实例
                if _high_policy is None:
                    for _attr in vars(self.high_agent).values():
                        if hasattr(_attr, 'score_goal_candidates'):
                            _high_policy = _attr
                            break

                if _high_policy is not None and hasattr(_high_policy, 'score_goal_candidates'):
                    try:
                        import torch as _torch
                        _epsilon_high = getattr(self.high_agent, 'epsilon_high', 0.1)

                        # [Fix-Eps] 首个 dest 选择用更低 epsilon
                        # 目的：首个 dest 极为关键（决定树的主干方向），不应受探索噪声影响
                        _connected_for_eps = len(self.shared.get_connected_dests_view()) \
                            if self.env.current_tree else 0
                        if _connected_for_eps == 0:  # 还没有连通任何 dest = 首个选择
                            _epsilon_high = min(_epsilon_high, 0.05)  # 首 dest: 最多 5% 探索

                        # 获取节点级嵌入（来自 encoder 或 state graph）
                        _node_embs = None
                        _encoder = getattr(self.high_agent, 'encoder', None)
                        if _encoder is not None:
                            try:
                                _s = high_obs[0] if isinstance(high_obs, tuple) else high_obs
                                if hasattr(_s, 'x') and hasattr(_s, 'edge_index'):
                                    _device = next(_encoder.parameters()).device
                                    _ei = _s.edge_index.to(_device)
                                    _ea = getattr(_s, 'edge_attr', None)
                                    if _ea is None:
                                        _ea = _torch.zeros(_ei.shape[1], 5, device=_device)
                                    else:
                                        _ea = _ea.to(_device)
                                        if _ea.dim() == 1: _ea = _ea.unsqueeze(1)
                                        if _ea.shape[1] < 5:
                                            _ea = _torch.cat([_ea, _torch.zeros(_ea.shape[0], 5 - _ea.shape[1], device=_device)], dim=1)
                                    _b = _torch.zeros(_s.x.size(0), dtype=_torch.long, device=_device)
                                    _tei = getattr(_s, 'tree_edge_index', None)
                                    if _tei is not None:
                                        _tei = _tei.to(_device)
                                    _dest = getattr(_s, 'dest_mask', None)
                                    if _dest is not None:
                                        _dest = _dest.to(_device)
                                    _req = getattr(_s, 'req_vec', None)
                                    if _req is not None:
                                        _req = _req.to(_device)
                                    elif getattr(_encoder, 'req_fc', None) is not None:
                                        _cur_req = getattr(self.env, 'current_request', None)
                                        if _cur_req:
                                            _bw = float(_cur_req.get('bw_origin', _cur_req.get('bw', 0.0)))
                                            _cpu = _cur_req.get('cpu_origin', _cur_req.get('cpu', []))
                                            _mem = _cur_req.get('memory_origin', _cur_req.get('memory', []))
                                            _avg_cpu = float(np.mean(_cpu)) if len(_cpu) > 0 else 0.0
                                            _avg_mem = float(np.mean(_mem)) if len(_mem) > 0 else 0.0
                                            _req = _torch.tensor([[_bw, _avg_cpu, _avg_mem]], dtype=_torch.float32, device=_device)
                                    _node_embs = _encoder(_s.x.to(_device), _ei, _ea, batch=_b,
                                                          tree_edge_index=_tei,
                                                          dest_mask=_dest,
                                                          req_vec=_req)
                                    if _node_embs.dim() == 2:
                                        _node_embs = _node_embs.unsqueeze(0)  # [1, N, H]
                            except Exception as _ee:
                                logger.debug(f"[HighCand] encoder失败: {_ee}")

                        if _node_embs is None:
                            # fallback: 用 graph_emb 广播
                            _graph_emb = self._get_high_graph_emb(high_obs)
                            _N = self.env.n
                            _H = _graph_emb.size(-1)
                            _node_embs = _graph_emb.unsqueeze(1).expand(1, _N, _H)

                        _graph_emb_1d = self._get_high_graph_emb(high_obs)

                        _high_feats = self._candidate_feats(_high_cand_info)

                        goal_idx_t, goal_emb = _high_policy.select_goal(
                            _graph_emb_1d,
                            valid_goals_mask=high_mask,
                            epsilon=_epsilon_high,
                            candidate_indices=_high_cand_info['indices'],
                            candidate_node_embs=_node_embs,
                            candidate_local_feats=_high_feats,
                        )
                        target_node = int(goal_idx_t.item())
                        logger.debug(
                            f"[HighCand] 路径1: K={len(_high_cand_info['indices'])} "
                            f"→ target_node={target_node}"
                        )
                    except Exception as _hpe:
                        logger.debug(f"[HighCand] score_goal_candidates 失败，退回路径2: {_hpe}")
                        # 尝试把候选信息传给 high_agent.select_action（若支持 kwargs）
                        try:
                            _r = self.high_agent.select_action(
                                high_obs, action_mask=high_mask,
                                candidate_indices=_high_cand_info['indices'],
                                candidate_local_feats=self._candidate_feats(_high_cand_info),
                            )
                            if _r is not None:
                                _idx, _remap, _ai = _r
                                target_node = int(_remap) if _remap is not None else int(_idx)
                        except Exception:
                            target_node = None

            if target_node is None:
                # ── 路径2：全图 logits（向后兼容） ─────────────────────────
                high_action_idx, high_action_remapped, agent_info = self.high_agent.select_action(
                    high_obs, action_mask=high_mask
                )
                actual_high_action = high_action_remapped if high_action_remapped is not None else high_action_idx
                valid_indices = np.where(high_mask > 0)[0]
                if agent_info and 'subgoal' in agent_info and agent_info['subgoal'] is not None:
                    candidate = int(agent_info['subgoal'])
                    if candidate < len(high_mask) and high_mask[candidate] > 0:
                        target_node = candidate
                    elif len(valid_indices) > 0:
                        idx_pick = high_action_idx if high_action_idx < len(valid_indices) else 0
                        target_node = int(valid_indices[idx_pick])
                if target_node is None:
                    if actual_high_action < len(high_mask) and high_mask[actual_high_action] > 0:
                        target_node = actual_high_action
                    elif high_action_idx < len(valid_indices):
                        target_node = int(valid_indices[high_action_idx])
                    elif len(valid_indices) > 0:
                        target_node = int(valid_indices[0])
                    else:
                        target_node = actual_high_action
            else:
                # 路径1成功：构造兼容的 agent_info
                agent_info = {'subgoal': target_node, 'start_node': None}
                actual_high_action = target_node
                valid_indices = np.where(high_mask > 0)[0]
                # [Fix-B3] 路径1绕过了 agent.select_action()，_last_high_action_meta 不会被
                # agent_action._select_subgoal() 写入，导致高层 replay 里 used_candidate_path
                # 全为 False，高层 scorer 永远训练不到。这里手动补写。
                if _high_cand_info is not None:
                    _local_goal_fix = (
                        _high_cand_info['indices'].index(target_node)
                        if target_node in _high_cand_info['indices'] else None
                    )
                    self.high_agent._last_high_action_meta = {
                        'used_candidate_path':   _local_goal_fix is not None,
                        'candidate_indices':     list(_high_cand_info['indices']),
                        'candidate_local_feats': self._candidate_feats(_high_cand_info),
                        'local_goal_idx':        _local_goal_fix,
                        'action_mask':           high_mask.copy(),
                    }

        self._record_decision_timing(
            'high',
            _high_decision_started_ns,
            _high_decision_phase,
            'single_dqn' if self._ablation_single_dqn else 'policy',
        )

        self.env._active_vnf_completion_path = None
        if getattr(self.env, 'current_phase', None) == 'vnf_deployment':
            _hlc_path = getattr(self.env, 'high_level_controller', None)
            _path_evidence = getattr(
                _hlc_path, '_completion_path_evidence', {}
            ) if _hlc_path is not None else {}
            _selected_path = _path_evidence.get(int(target_node))
            if _selected_path:
                self.env._active_vnf_completion_path = list(_selected_path)
            _plan_evidence = getattr(
                _hlc_path, '_completion_plan_evidence', {}
            ) if _hlc_path is not None else {}
            _selected_plan = _plan_evidence.get(int(target_node))
            if _selected_plan:
                self.env._vnf_completion_schedule = {
                    'request_id': request.get('id'),
                    'stages': copy.deepcopy(list(_selected_plan)),
                }

        # ===== VNF阶段：对最终 target_node 做硬校验（防止路径1绕过mask） =====
        if self.env.current_request is not None:
            _vnf_list_rc = self.env.current_request.get('vnf', [])
            _vnf_idx_rc  = getattr(self.env, 'next_vnf_idx', 0)
            if _vnf_idx_rc < len(_vnf_list_rc):
                _vnf_type_rc = _vnf_list_rc[_vnf_idx_rc]
                _cpu_list_rc = self.env.current_request.get('cpu_origin', []) or \
                               self.env.current_request.get('vnf_cpu', [])
                _mem_list_rc = self.env.current_request.get('memory_origin', []) or \
                               self.env.current_request.get('vnf_mem', [])
                _req_cpu_rc  = float(_cpu_list_rc[_vnf_idx_rc]) if _vnf_idx_rc < len(_cpu_list_rc) else 10.0
                _req_mem_rc  = float(_mem_list_rc[_vnf_idx_rc]) if _vnf_idx_rc < len(_mem_list_rc) else 10.0
                # [统一探针] HighTargetRecheck 走 probe_vnf_deploy，与高层 mask 口径一致
                _probe_rc = self.env.resource_mgr.probe_vnf_deploy(
                    target_node, _vnf_type_rc, _req_cpu_rc, _req_mem_rc
                )
                _resource_ok_rc = bool(_probe_rc['ok'])
                _can_reuse_rc   = bool(_probe_rc['reuse'])
                _mask_ok_rc = (0 <= target_node < len(high_mask) and high_mask[target_node] > 0)

                if not _mask_ok_rc or not _resource_ok_rc:
                    logger.warning(
                        f"[Coord][HighTargetRecheck] reject target={target_node} | "
                        f"mask_ok={int(_mask_ok_rc)} resource_ok={int(_resource_ok_rc)} "
                        f"req_cpu={_req_cpu_rc:.1f} req_mem={_req_mem_rc:.1f} "
                        f"avail_cpu={self.env.resource_mgr.pool.get_available_cpu(target_node):.1f} "
                        f"avail_mem={self.env.resource_mgr.pool.get_available_memory(target_node):.1f} "
                        f"can_reuse={int(_can_reuse_rc)} reason={_probe_rc['reason']}"
                    )
                    # [Fix P0] 不再 silent rerank，直接截断让高层重选
                    # 原来改 target_node 会破坏 credit assignment（选了A执行B）
                    self.env._cycle_high_mask_override = None
                    return 0.0, False, {
                        'subgoal_truncated': True,
                        'reason': 'high_target_recheck_failed',
                        'target': int(target_node)
                    }

        start_node = agent_info.get('start_node') if isinstance(agent_info, dict) else None
        if start_node is None:
            _chain = getattr(self.env, 'chain_nodes', [])
            _vnf_list = []
            if self.env.current_request:
                _vnf_list = self.env.current_request.get('vnf', [])
            _vnf_done = getattr(self.env, 'next_vnf_idx', 0) >= len(_vnf_list) if _vnf_list else False
            if _vnf_done and len(_chain) > 0:

                # [Task 2] 首个 dest 专项：打印高层候选排序
                _connected_now = len(
                    self.shared.get_connected_dests_view()
                ) if self.env.current_tree else 0
                if _connected_now == 0 and _high_cand_info is not None:
                    # [Fix3] 진짜 scorer 점수 기반 top3 출력 (이전엔 candidate 목록 순서 출력)
                    _top3_cands = _high_cand_info['indices'][:3]  # fallback
                    _top3_scored = _top3_cands
                    _top3_scores_v = []
                    if (_high_policy is not None and
                            hasattr(_high_policy, 'score_goal_candidates') and
                            _node_embs is not None and
                            len(_high_cand_info['indices']) > 0):
                        try:
                            import torch as _t3
                            _scores3 = _high_policy.score_goal_candidates(
                                graph_emb=_graph_emb_1d,
                                candidate_indices=_high_cand_info['indices'],
                                candidate_node_embs=_node_embs,
                                candidate_local_feats=self._candidate_feats(_high_cand_info),
                            )
                            _topk3 = min(3, _scores3.size(1))
                            _tv, _ti = _scores3.topk(_topk3, dim=1)
                            _top3_scored = [_high_cand_info['indices'][i]
                                            for i in _ti[0].tolist()]
                            _top3_scores_v = [round(v, 3) for v in _tv[0].tolist()]
                        except Exception:
                            pass
                    logger.debug(
                        f"[firstDestRank] target_selected={target_node} "
                        f"top3_goal_candidates={_top3_cands} "
                        f"top3_scored={_top3_scored} scores={_top3_scores_v} "
                        f"total_candidates={len(_high_cand_info['indices'])}"
                    )
                # Strict SFC order: every destination cycle starts at a node
                # that has already traversed every VNF stage.
                _failed_anchors_now = getattr(
                    self.env, '_failed_anchors_for_target', {}
                ).get(target_node, set())
                _planned_step = getattr(
                    self.env, '_active_destination_plan_step', None
                )
                if (
                    isinstance(_planned_step, dict)
                    and int(_planned_step.get('target', -1)) == int(target_node)
                ):
                    best_anchor = int(_planned_step['anchor'])
                    self.env._active_destination_path = list(
                        _planned_step.get('path', [])
                    )
                else:
                    best_anchor = _chain[-1] if _chain else None
                    self.env._active_destination_path = None

                # [Fix-NoAnchor] 所有anchor都已失败 → 封禁target，跳过此次cycle
                if best_anchor is None:
                    self._unreachable_targets.add(target_node)
                    logger.debug(
                        f"[NoAnchor] target={target_node} 所有anchor已失败，立即封禁"
                    )
                    self.env._cycle_high_mask_override = None
                    return 0.0, False, {'fail': True, 'reason': 'no_anchor', 'subgoal_truncated': True}

                # Coordinator-level sanity check before mutating the env.
                _c_total_vnf = len(self.env.current_request.get('vnf', [])) \
                    if self.env.current_request else 0
                _c_node_stage = self.env.current_tree.get('node_stage', {}) \
                    if self.env.current_tree else {}
                _c_last_vnf = _chain[-1] if _chain else self.env.current_node_location
                if (_c_total_vnf > 0 and best_anchor is not None
                        and _c_node_stage.get(best_anchor, 0) < _c_total_vnf):
                    logger.warning(
                        f"[AnchorSanitize] invalid best_anchor={best_anchor} "
                        f"stage={_c_node_stage.get(best_anchor,0)} < total_vnf={_c_total_vnf}, "
                        f"force_fallback={_c_last_vnf}"
                    )
                    best_anchor = _c_last_vnf

                # [Task7] 写入合法的 anchor 到 env
                _cur_before_low = getattr(self.env, 'current_node_location', None)
                self.env.current_anchor_node = best_anchor
                self.env.current_node_location = best_anchor
                self.env.current_subgoal_full_path = [int(best_anchor)]
                self.env._dest_anchor_mismatch = False
                self.env._dest_step0_checked = False
                self.env._dest_cycle_armed = True
                # [Fix-DeadLock] 每次切换 dest 目标时必须清零死锁计数器，
                # 否则上一个 dest 累积的计数会在新 dest 第一步就误触发截断
                self.env._deadlock_step_count = 0
                # [Fix-Oscillation] 同步重置位置振荡检测状态
                self._pos_history = []
                self._osc_escape_count = 0
                # 同步清空禁忌表，以 anchor 为起点重新开始
                if hasattr(self.env, 'current_path_trace'):
                    self.env.current_path_trace = [best_anchor]
                # 同步 _last_dest_target 以触发 DestFix 的回位逻辑
                self.env._last_dest_target = None
                start_node = best_anchor
                # [Diag-CycleStart] 每次dest cycle开始时打印关键信息
                _llc_diag = getattr(self.env, 'low_level_controller', None)
                _hop_a2t = _llc_diag._get_hop_distance(best_anchor, target_node) \
                    if _llc_diag else '?'
                _failed_a = getattr(self.env, '_failed_anchors_for_target', {}).get(target_node, set())
                _nbrs_target = list(self.env.resource_mgr.get_neighbors(target_node)) \
                    if hasattr(self.env, 'resource_mgr') else []
                # [Diag+] 打印每个target邻居的入向BW
                _bw_req_cs = float(self.env.current_request.get('bw_origin', 0)) \
                    if self.env.current_request else 0
                _pool_cs = getattr(self.env.resource_mgr, 'pool', None)
                _nbr_bw_info = {
                    nbr: round(_pool_cs.get_available_bandwidth(nbr, target_node), 1)
                    for nbr in _nbrs_target if _pool_cs
                } if _pool_cs else {}
                logger.info(
                    f"[CycleStart] target={target_node} anchor={best_anchor} "
                    f"hop_a2t={_hop_a2t} "
                    f"target_nbrs={_nbrs_target} "
                    f"nbr_bw(need={_bw_req_cs})={_nbr_bw_info} "
                    f"failed_anchors={_failed_a} "
                    f"connected={len(self.shared.get_connected_dests_view())}/{len(self.env.current_request.get('dest',[]))}"
                )
                _connected_dests = len(
                    self.shared.get_connected_dests_view()
                ) if self.env.current_tree else 0
                _nodes_on_tree = len(getattr(self.env, 'nodes_on_tree', set()))
                logger.debug(
                    f"[DestCycleStart] target={target_node} anchor={best_anchor} "
                    f"cur_before_low={_cur_before_low} path_trace={getattr(self.env, 'current_path_trace', None)}"
                )
                logger.debug(
                    f"[Anchor] selected_anchor={best_anchor} target_goal={target_node} "
                    f"last_vnf={_chain[-1]} connected_dests={_connected_dests} "
                    f"nodes_on_tree={_nodes_on_tree} "
                    f"anchor_stage={_c_node_stage.get(best_anchor,0)}/{_c_total_vnf}"
                )
            else:
                start_node = self.env.current_node_location
                # VNF阶段：清除anchor，避免低层误读旧值
                self.env.current_anchor_node = None
                self.env._dest_cycle_armed = False
                self.env._dest_step0_checked = False
                self.env._dest_anchor_mismatch = False

        # 弱化启发式后置替换：让高层完全依赖于mask的前置过滤进行学习
        # 不再把选中的节点又强制换掉
        actual_high_action = int(target_node)

        self.env.set_high_level_goal(actual_high_action, target_node, start_node)
        _, _, done, truncated, high_info = self.env.step_high_level(actual_high_action)
        self.env._cycle_high_mask_override = None
        self.env._scheduled_high_mask_override = None

        if done:
            return 0.0, True, {'episode_done': True}

        if truncated:
            self._unreachable_targets.add(int(target_node))
            if hasattr(self.high_agent, 'current_subgoal'):
                self.high_agent.current_subgoal = None
            if hasattr(self.high_agent, 'subgoal_steps'):
                horizon = getattr(self.high_agent, 'subgoal_horizon', 10)
                self.high_agent.subgoal_steps = horizon + 1
            if hasattr(self.high_agent, 'subgoal_step_count'):
                self.high_agent.subgoal_step_count = 999
            # [Fix P0] 不清空整个 _unreachable_targets；失败记忆在 episode 结束前保留
            # 原来 clear() 会让同一 episode 内已超时/卡死的目标重新被选中
            _high_reason = (
                high_info.get('reason')
                or high_info.get('warning')
                or 'high_subgoal_truncated'
            ) if isinstance(high_info, dict) else 'high_subgoal_truncated'
            return 0.0, False, {
                'subgoal_truncated': True,
                'reason': str(_high_reason),
            }

        self.env.subgoal_step_count = 0
        if getattr(self.env, 'current_phase', None) != 'destination_connection':
            self.env._dest_cycle_armed = False
            self.env._dest_step0_checked = False
            self.env._dest_anchor_mismatch = False
            if hasattr(self.env, 'current_path_trace'):
                self.env.current_path_trace = []
        else:
            # destination cycle 第0步必须保留 [anchor]，不能在这里被清空；
            # 否则 step0 核验会看到空 trace，低层也失去从 anchor 出发的短禁忌表。
            if hasattr(self.env, 'current_path_trace'):
                _anchor_keep = getattr(self.env, 'current_anchor_node', None)
                if _anchor_keep is not None:
                    self.env.current_path_trace = [_anchor_keep]

        low_state = self.env.get_state()
        low_step = 0
        low_total_reward = 0.0
        episode_done = False
        info = {}
        low_level_stalled = False
        subgoal_achieved = False
        start_pos_before_low = self.env.current_node_location

        planned_path = None
        path_idx = 0
        blocked_edges = set()
        replan_count = 0
        no_progress_count = 0

        # [Monitor B] 记录本次 cycle 的阶段和步数，用于成功/失败分析
        _cycle_phase = getattr(self.env, 'current_phase', 'unknown')
        _cycle_transaction = None
        if _cycle_phase in {'vnf_deployment', 'destination_connection'}:
            _cycle_transaction = {
                'current_tree': copy.deepcopy(getattr(self.env, 'current_tree', None)),
                'current_sfc': copy.deepcopy(getattr(self.env, 'current_sfc', None)),
                'nodes_on_tree': set(getattr(self.env, 'nodes_on_tree', set())),
                'current_node_location': getattr(self.env, 'current_node_location', None),
                'current_anchor_node': getattr(self.env, 'current_anchor_node', None),
                'chain_nodes': list(getattr(self.env, 'chain_nodes', [])),
            }
        if not hasattr(self, '_phase_steps'):
            self._phase_steps = {'vnf_deployment': [], 'destination_connection': [], 'other': []}
        if not hasattr(self, '_fail_dest_log'):
            self._fail_dest_log = []

        # Full planner execution is an explicit evaluation mode. During
        # policy execution only the narrow safety guard below may override a
        # demonstrably non-progressing action.
        use_planner_decision = (
            not training and self.deployment_executor == 'bw_planner'
        )
        planned_path = None

        # [Log 5] 初始化 planner 诊断计数器（每个 cycle 重置）
        _planner_steps_this_cycle = 0
        _policy_steps_this_cycle  = 0
        _replan_count_this_cycle  = 0
        _subgoal_achieved_this_cycle = False
        self._reset_single_candidate_deadlock()

        while low_step < self.max_low_steps and not episode_done:
            used_planner = False

            # 每步初始化 _cand_info
            _cand_info = None

            # policy 决策（planner已停用，全部走policy路径）
            llc = getattr(self.env, 'low_level_controller', None)
            # Keep the hard environment mask separate from the completion
            # lookahead.  The high-level decoder already produced an explicit
            # path for the selected VNF target on this unchanged resource
            # snapshot.  Re-running the recursive lookahead after every partial
            # edge used to reject that same path because the newly committed
            # prefix was interpreted as an unrelated occupied spine.
            hard_low_mask = self.env.get_low_level_action_mask()
            low_mask = hard_low_mask.copy()
            _completion_mask_changed = False
            _used_vnf_path_evidence = False
            if (
                _cycle_phase == 'vnf_deployment'
                and self.config.get('vnf_path_shield', True)
            ):
                _current_vnf_node = getattr(
                    self.env, 'current_node_location', None
                )
                _vnf_path = list(getattr(
                    self.env, '_active_vnf_completion_path', None
                ) or [])
                if (
                    _vnf_path
                    and int(_vnf_path[-1]) == int(target_node)
                    and _current_vnf_node in _vnf_path
                ):
                    _vnf_path_idx = _vnf_path.index(_current_vnf_node)
                    _vnf_next = (
                        int(_vnf_path[_vnf_path_idx + 1])
                        if _vnf_path_idx + 1 < len(_vnf_path)
                        else int(_current_vnf_node)
                    )
                    _vnf_shielded_mask = np.zeros_like(low_mask)
                    if (
                        0 <= _vnf_next < len(hard_low_mask)
                        and hard_low_mask[_vnf_next] > 0
                    ):
                        _vnf_shielded_mask[_vnf_next] = 1.0
                        low_mask = _vnf_shielded_mask
                        _used_vnf_path_evidence = True
                    else:
                        # The certificate cannot become valid later in this
                        # unchanged cycle.  Repeating the same deterministic
                        # check up to max_low_steps produced 5-40 second tails.
                        # Roll back this subgoal and let the high level select a
                        # different placement candidate on the stable snapshot.
                        logger.warning(
                            "[VNFPathEvidenceRejected] req=%s idx=%s cur=%s "
                            "target=%s next=%s path=%s hard_open=%s; "
                            "invalidate_and_reselect=1",
                            (getattr(self.env, 'current_request', None) or {}).get('id'),
                            getattr(self.env, 'next_vnf_idx', None),
                            _current_vnf_node,
                            target_node,
                            _vnf_next,
                            _vnf_path,
                            [int(i) for i in np.where(hard_low_mask > 0)[0]],
                        )
                        self.env._active_vnf_completion_path = None
                        self.env._vnf_completion_schedule = None
                        self.env._scheduled_high_mask_override = None
                        info = {
                            'deploy_fail': True,
                            'timeout': True,
                            'reason': 'vnf_path_certificate_invalid',
                            'target': target_node,
                        }
                        low_level_stalled = True
                        break
            if (
                _cycle_phase == 'vnf_deployment'
                and not _used_vnf_path_evidence
            ):
                low_mask, _completion_mask_changed = (
                    self._filter_vnf_completion_mask(low_mask, target_node)
                )
            if (
                _cycle_phase == 'vnf_deployment'
                and np.sum(low_mask) == 0
            ):
                info = {
                    'deploy_fail': True,
                    'timeout': True,
                    'reason': 'vnf_completion_dead_end',
                    'target': target_node,
                }
                low_level_stalled = True
                logger.debug(
                    f"[VNFCompletionDeadEnd] cur="
                    f"{getattr(self.env, 'current_node_location', None)} "
                    f"target={target_node}; rollback subgoal"
                )
                break
            if (
                _cycle_phase == 'destination_connection'
                and self.config.get('destination_path_shield', True)
            ):
                _current_for_shield = getattr(
                    self.env, 'current_node_location', None
                )
                _active_path = list(getattr(
                    self.env, '_active_destination_path', None
                ) or [])
                _shield_action = None
                if (
                    _active_path
                    and int(_active_path[-1]) == int(target_node)
                    and _current_for_shield in _active_path
                ):
                    _path_index = _active_path.index(_current_for_shield)
                    if _path_index + 1 < len(_active_path):
                        _shield_action = int(_active_path[_path_index + 1])
                    else:
                        _shield_action = int(_current_for_shield)
                if _shield_action is None:
                    _shield_action = self._bw_planner_action(llc, blocked_edges)
                if _shield_action is None and _current_for_shield != target_node:
                    info = {
                        'timeout': True,
                        'fail': True,
                        'reason': 'destination_no_safe_path',
                        'target': target_node,
                    }
                    low_level_stalled = True
                    break
                if _shield_action is not None:
                    _shielded_mask = np.zeros_like(low_mask)
                    _shielded_mask[int(_shield_action)] = 1.0
                    low_mask = _shielded_mask
            # [Fix-DeadLock] mask 已更新，_last_mask_alive_count 此时是本步真实值。
            # 在 dest 阶段，若连续 _DEADLOCK_THRESH 步 alive_count <= 1（agent 被困），
            # 立即截断当前子目标，让高层重选 anchor/dest，避免浪费 40+ 步。
            # 阈值 12：留足 BranchPathCheck 兜底的触发窗口，又远小于 max_subgoal_steps(35-55)。
            # [Fix-D] 最后1个dest时提高deadlock阈值（12→20）
            _pending_dl = 0
            if self.env.current_request and self.env.current_tree:
                _all_d_dl = len(self.env.current_request.get('dest', []))
                _done_d_dl = len(self.env.current_tree.get('connected_dests', set()))
                _pending_dl = _all_d_dl - _done_d_dl
            _DEADLOCK_THRESH = 20 if _pending_dl <= 1 else 12
            if _cycle_phase == 'destination_connection':
                _alive_now = getattr(self.env, '_last_mask_alive_count', 99)
                _at_goal = (self.env.current_node_location ==
                            getattr(self.env, 'current_target_node', None))
                if _alive_now <= 1 and not _at_goal:
                    _tgt_dl = getattr(self.env, 'current_target_node', None)
                    _deadlock_count, _, _ = self._update_single_candidate_deadlock(
                        _tgt_dl
                    )
                    if _deadlock_count >= _DEADLOCK_THRESH:
                        self._reset_single_candidate_deadlock()
                        _cur_dl = getattr(self.env, 'current_node_location', None)
                        logger.debug(
                            f"[DeadLock] dest阶段连续{_deadlock_count}步mask_alive=1，"
                            f"cur={_cur_dl} target={_tgt_dl} → 提前截断重选"
                        )
                        self.env._timeout_count = getattr(self.env, '_timeout_count', 0) + 1
                        self.env._consecutive_timeout_count = self.env._timeout_count
                        self.env._dest_cycle_armed = False
                        self.env._dest_step0_checked = False
                        self.env._dest_anchor_mismatch = False
                        # 记入 fail log，方便诊断
                        info = {
                            'timeout': True,
                            'fail': True,
                            'reason': 'deadlock_single_candidate',
                        }
                        low_level_stalled = True
                        break
                else:
                    self._reset_single_candidate_deadlock()
            else:
                self._reset_single_candidate_deadlock()
            if (
                _cycle_phase == 'destination_connection'
                and getattr(self.env, '_dest_step0_checked', False)
                and getattr(self.env, '_dest_anchor_mismatch', False)
            ):
                info = {
                    'anchor_mismatch': True,
                    'reason': 'anchor_mismatch',
                    'target': getattr(self.env, 'current_target_node', None),
                    'anchor': getattr(self.env, 'current_anchor_node', None),
                    'current': getattr(self.env, 'current_node_location', None),
                }
                low_level_stalled = True
                break
            if sum(low_mask) == 0:
                low_level_stalled = True
                break

            _low_decision_started_ns = time.perf_counter_ns()
            _planner_action = None
            if use_planner_decision:
                _planner_action = self._bw_planner_action(llc, blocked_edges)
            if _planner_action is None:
                _policy_steps_this_cycle += 1
            else:
                _planner_steps_this_cycle += 1

            if llc is not None and hasattr(llc, 'get_low_level_candidates'):
                try:
                    _cand_info = llc.get_low_level_candidates()
                except Exception as _ce:
                    logger.debug(f"[Coord] get_low_level_candidates 异常: {_ce}")

            _cand_info = self._align_low_candidates_with_mask(
                _cand_info,
                low_mask,
                getattr(self.env, 'current_node_location', 0),
            )

            if _cand_info is not None and len(_cand_info['indices']) > 0:
                # 路径1：候选邻居逐点评分
                _low_policy = getattr(self.low_agent, 'low_policy', None)
                if _low_policy is not None and hasattr(_low_policy, 'select_action'):
                    _state_emb, _goal_emb = self._encode_low_state_for_policy(low_state)
                    import torch as _torch
                    _mask_t = _torch.from_numpy(low_mask).bool().unsqueeze(0)
                    _epsilon = getattr(self.low_agent, 'epsilon_low', 0.0)
                    _low_feats = self._candidate_feats(_cand_info)

                    _action_t, _ = _low_policy.select_action(
                        _state_emb,
                        goal_emb=_goal_emb,
                        action_mask=_mask_t,
                        epsilon=_epsilon,
                        candidate_indices=_cand_info['indices'],
                        current_node_idx=_cand_info['current_node'],
                        candidate_local_feats=_low_feats,
                    )
                    low_action = int(_action_t.item()) if hasattr(_action_t, 'item') else int(_action_t)
                    logger.debug(
                        f"[low_select] candidate_path "
                        f"cands={len(_cand_info['indices'])} action={low_action}"
                    )
                else:
                    # low_policy 不可达，降级
                    _, low_action, _ = self.low_agent.select_action(
                        low_state, action_mask=low_mask
                    )
                    logger.debug(f"[low_select] full_logits_fallback (low_policy不可达)")
            else:
                # 路径2：降级到全图 logits（无候选时兜底）
                _, low_action, _ = self.low_agent.select_action(
                    low_state, action_mask=low_mask
                )
                logger.debug(f"[low_select] full_logits_fallback (无候选)")

            if _planner_action is not None:
                low_action = int(_planner_action)
                used_planner = True
                logger.debug(
                    f"[low_select] bw_planner cur={self.env.current_node_location} "
                    f"action={low_action} phase={_cycle_phase}"
                )
            elif self.planner_safety_guard:
                _guard_action = self._bw_planner_action(llc, blocked_edges)
                low_action, _guarded = self._apply_planner_safety_guard(
                    low_action,
                    _guard_action,
                    low_mask,
                    llc,
                    target_node,
                )
                if _guarded:
                    used_planner = True
                    _planner_steps_this_cycle += 1
                    _policy_steps_this_cycle = max(0, _policy_steps_this_cycle - 1)

            _completion_hint = (
                _planner_action
                if _planner_action is not None
                else self._bw_planner_action(llc, blocked_edges)
            )
            _was_guarded = used_planner
            low_action, _completion_guarded = self._apply_vnf_completion_guard(
                low_action,
                _completion_hint,
                low_mask,
                llc,
                target_node,
            )
            if _completion_guarded and not _was_guarded:
                used_planner = True
                _planner_steps_this_cycle += 1
                _policy_steps_this_cycle = max(0, _policy_steps_this_cycle - 1)

            self._record_decision_timing(
                'low',
                _low_decision_started_ns,
                _cycle_phase,
                'bw_planner' if used_planner else 'policy',
            )

            pos_before = self.env.current_node_location
            next_state, reward, done, truncated_low, info = self.env.step_low_level(low_action)
            pos_after = self.env.current_node_location

            is_success = info.get('dest_connected') or info.get('vnf_deployed') \
                         or info.get('all_vnf_deployed') or info.get('episode_complete')

            if pos_after == pos_before and not is_success:
                no_progress_count += 1
                failed_edge = info.get('edge')
                if failed_edge:
                    # [Directed-BW] 阻塞边记录有向键
                    if isinstance(failed_edge, (list, tuple)) and len(failed_edge) == 2:
                        blocked_edges.add(tuple(failed_edge))
            else:
                no_progress_count = 0

            # [Fix-Oscillation] 位置振荡检测：追踪近 _OSC_WIN 步的位置历史
            # 若近 _OSC_WIN 步内只出现 <= _OSC_UNIQ 个不同节点，认为 agent 在来回振荡，
            # 强制清空 tabu 表，给低层一次逃脱机会。
            # 若连续两次逃脱尝试失败（_osc_escape_count >= 2），截断本子目标让高层重选。
            _OSC_WIN  = 10   # 观察窗口步数
            _OSC_UNIQ = 2    # 窗口内不同节点数阈值（<= 2 视为振荡）
            if not hasattr(self, '_pos_history'):
                self._pos_history = []
                self._osc_escape_count = 0
            self._pos_history.append(pos_after)
            if len(self._pos_history) > _OSC_WIN:
                self._pos_history = self._pos_history[-_OSC_WIN:]

            # [Fix-B] 振荡检测：最后1个dest时提高容忍度（阈值从2次改为4次）
            _pending_osc = 0
            if self.env.current_request and self.env.current_tree:
                _all_d_osc = len(self.env.current_request.get('dest', []))
                _done_d_osc = len(self.env.current_tree.get('connected_dests', set()))
                _pending_osc = _all_d_osc - _done_d_osc
            _osc_escape_thresh = 4 if _pending_osc <= 1 else 2
            if (
                _cycle_phase == 'destination_connection'
                and not is_success
                and len(self._pos_history) >= _OSC_WIN
                and len(set(self._pos_history)) <= _OSC_UNIQ
            ):
                self._osc_escape_count += 1
                _tgt_osc = getattr(self.env, 'current_target_node', None)
                _cur_osc = self.env.current_node_location
                # 最后1个dest时升为INFO
                _osc_log = logger.info if _pending_osc <= 1 else logger.debug
                _osc_log(
                    f"[Oscillation] dest阶段{_OSC_WIN}步内仅访问节点"
                    f"{set(self._pos_history)}，cur={_cur_osc} target={_tgt_osc} "
                    f"escape_attempt={self._osc_escape_count} pending={_pending_osc}"
                )
                if self._osc_escape_count >= _osc_escape_thresh:
                    # 两次逃脱均失败，截断子目标
                    self._osc_escape_count = 0
                    self._pos_history = []
                    self.env._deadlock_step_count = 0
                    self.env._timeout_count = getattr(self.env, '_timeout_count', 0) + 1
                    self.env._consecutive_timeout_count = self.env._timeout_count
                    self.env._dest_cycle_armed = False
                    self.env._dest_step0_checked = False
                    self.env._dest_anchor_mismatch = False
                    info = {
                        'timeout': True, 'fail': True,
                        'reason': 'oscillation_truncated',
                    }
                    low_level_stalled = True
                    break
                else:
                    # Do not teleport between tree nodes.  A relocation without
                    # traversing and charging a physical path corrupts the SFT
                    # execution trace.  Give the policy one clean retry in place;
                    # the next oscillation is truncated and transactionally rolled
                    # back by the branch above.
                    if hasattr(self.env, 'current_path_trace'):
                        self.env.current_path_trace = [self.env.current_node_location]
                    self._pos_history = []
            elif is_success:
                self._osc_escape_count = 0
                self._pos_history = []

            # 路径受阻时标记不可达（planner已停用，直接走policy）
            _bw_blocked = info.get('error') in ('no_bandwidth', 'resource_failure')
            if _bw_blocked:
                replan_count += 1
                # [Fix-G] delayed-commit模式下BW未实际扣减，replan封禁容易误判
                # 提高阈值：3次 → 6次，减少误封可达目标
                if replan_count > 6:
                    self._unreachable_targets.add(target_node)

            # [Fix-H] no_progress阈值：最后1个dest时宽松（5→8步）
            _np_thresh = 5
            if self.env.current_request and self.env.current_tree:
                _pend_np = len(set(self.env.current_request.get('dest',[])) -
                               self.shared.get_connected_dests_view())
                if _pend_np <= 1:
                    _np_thresh = 8
            if no_progress_count >= _np_thresh:
                self._unreachable_targets.add(target_node)
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                if hasattr(self.high_agent, 'subgoal_steps'):
                    self.high_agent.subgoal_steps = getattr(self.high_agent, 'subgoal_horizon', 10) + 1
                info['stuck'] = True
                info['fail'] = True
                info.setdefault('reason', 'stuck')
                break

            # Store the action that was actually executed. Guarded actions
            # retain candidate metadata and therefore train the scorer instead
            # of creating empty-candidate planner samples.
            if training:
                # ── [P0 Fix] 统一构建含候选信息的 transition dict ─────────────
                # 无论 store_transition 接口如何，我们都把完整 dict 写进 low_memory
                # 这是修复 use_score_candidates=0/N 的根本手段

                # 1. 计算 next 状态的候选信息
                _next_cand_info = None
                if not done and not episode_done:
                    _llc_next = getattr(self.env, 'low_level_controller', None)
                    if _llc_next is not None and hasattr(_llc_next, 'get_low_level_candidates'):
                        try:
                            _next_cand_info = _llc_next.get_low_level_candidates()
                        except Exception:
                            pass

                # 2. 计算 local_action_idx（候选集内下标，不是全图节点ID）
                _local_action_idx = None
                _cand_info_for_train = _cand_info
                _anchor_mismatch_sample = bool(isinstance(info, dict) and info.get('anchor_mismatch', False))

                _used_cand_path = (not _anchor_mismatch_sample and
                                   _cand_info_for_train is not None and
                                   _cand_info_for_train.get('indices') and
                                   len(_cand_info_for_train['indices']) > 0)
                if _used_cand_path:
                    try:
                        _local_action_idx = _cand_info_for_train['indices'].index(low_action)
                    except ValueError:
                        # action 不在候选集（mask fallback 或边界情况），
                        # 保留候选信息但标记为非候选路径样本
                        _local_action_idx = None
                        _used_cand_path = False
                        logger.debug(
                            f"[CandTrain] low_action={low_action} 不在候选集"
                            f"{_cand_info_for_train['indices']}，降级为fallback样本"
                        )

                # 3. 构建完整 transition dict
                _transition = {
                    'state':       low_state,
                    'action':      int(low_action),          # 全图节点 ID（兜底用）
                    'reward':      float(reward),
                    'next_state':  next_state,
                    'done':        bool(done),

                    # candidate scorer 训练所需字段
                    'candidate_indices':          (_cand_info_for_train['indices']   if _used_cand_path else None),
                    'current_node_idx':           (_cand_info_for_train['current_node'] if _used_cand_path else None),
                    'candidate_local_feats':      (self._candidate_feats(_cand_info_for_train) if _used_cand_path else None),
                    'local_action_idx':           _local_action_idx,       # 候选集内下标
                    'used_candidate_path':        bool(_used_cand_path and _local_action_idx is not None),

                    # next state 候选（用于 Double DQN target）
                    'next_candidate_indices':     (_next_cand_info['indices']  if _next_cand_info else None),
                    'next_candidate_local_feats': (self._candidate_feats(_next_cand_info) if _next_cand_info else None),
                    'next_current_node_idx':      (_next_cand_info['current_node'] if _next_cand_info else None),
                }

                # 4. 写入 replay buffer：优先直接操作 memory，兜底走旧接口
                _stored = False
                # 方式A：直接写 low_memory（支持 deque / list / PER）
                _low_mem = getattr(self.low_agent, 'low_memory', None)
                if _low_mem is not None:
                    try:
                        if hasattr(_low_mem, 'push'):
                            _low_mem.push(_transition)
                            _stored = True
                        elif hasattr(_low_mem, 'add'):
                            _low_mem.add(_transition)
                            _stored = True
                        elif hasattr(_low_mem, 'store'):
                            _low_mem.store(_transition)
                            _stored = True
                        elif hasattr(_low_mem, 'append'):
                            _low_mem.append(_transition)
                            _stored = True
                    except Exception as _me:
                        logger.debug(f"[Replay] low_memory 直写失败: {_me}")

                # 方式B：store_transition_ext（自定义扩展接口）
                if not _stored and hasattr(self.low_agent, 'store_transition_ext'):
                    try:
                        self.low_agent.store_transition_ext(**_transition)
                        _stored = True
                    except Exception:
                        pass

                # 方式C：store_transition with extra kwargs
                if not _stored and hasattr(self.low_agent, 'store_transition'):
                    try:
                        self.low_agent.store_transition(
                            low_state, int(low_action), float(reward), next_state, bool(done),
                            candidate_indices=_transition['candidate_indices'],
                            current_node_idx=_transition['current_node_idx'],
                            candidate_local_feats=_transition['candidate_local_feats'],
                            local_action_idx=_transition['local_action_idx'],
                            used_candidate_path=_transition['used_candidate_path'],
                            next_candidate_indices=_transition['next_candidate_indices'],
                            next_candidate_local_feats=_transition['next_candidate_local_feats'],
                            next_current_node_idx=_transition['next_current_node_idx'],
                        )
                        _stored = True
                    except TypeError:
                        # 旧接口不支持 kwargs，最终兜底
                        self._store_transition(
                            self.low_agent, low_state, int(low_action),
                            float(reward), next_state, bool(done)
                        )
                        _stored = True

                if not _stored:
                    self._store_transition(
                        self.low_agent, low_state, int(low_action),
                        float(reward), next_state, bool(done)
                    )

                # [Fix-B4] coordinator 直写 low_memory 绕过了 agent_memory.store_transition_low()，
                # 导致 _ep_transitions 永远为空，success_memory / elite_buffer 无法积累样本。
                # 这里补充维护：写入后同步追加到 _ep_transitions，并推进 steps_done 计数。
                try:
                    import copy as _cp4
                    if not hasattr(self.low_agent, '_ep_transitions'):
                        self.low_agent._ep_transitions = []
                    self.low_agent._ep_transitions.append(_cp4.deepcopy(_transition))
                    self.low_agent.steps_done = getattr(self.low_agent, 'steps_done', 0) + 1
                    # success_memory / elite_buffer：episode done 时触发
                    if bool(done):
                        _ep_ts = self.low_agent._ep_transitions
                        _ep_r  = sum(t['reward'] for t in _ep_ts)
                        _ep_n  = len(_ep_ts)
                        if _ep_r > 60.0 and _ep_n < 40:
                            _sc_mem = getattr(self.low_agent, 'success_memory', None)
                            if _sc_mem is not None:
                                _sc_mem.extend(_cp4.deepcopy(_ep_ts))
                        _eli = getattr(self.low_agent, 'elite_buffer', None)
                        if _eli is not None:
                            _best = getattr(self.low_agent, '_best_reward', -1e9)
                            self.low_agent._best_reward = max(_best, _ep_r)
                            if _ep_r >= self.low_agent._best_reward * 0.8:
                                _eli.add_episode(_cp4.deepcopy(_ep_ts), _ep_r)
                        self.low_agent._ep_transitions = []
                except Exception as _b4e:
                    logger.debug(f"[Fix-B4] _ep_transitions 同步失败: {_b4e}")

            low_total_reward += reward
            low_state = next_state
            low_step += 1

            if info.get('subgoal_done', False):
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                break

            if info.get('goal_reached', False) or info.get('current_goal_satisfied', False):
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                break

            if info.get('dest_connected', False) or info.get('vnf_deployed', False) \
                    or info.get('all_vnf_deployed', False):
                if _cycle_phase == 'destination_connection':
                    getattr(self.env, '_dest_recovery_targets', set()).discard(int(target_node))
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                break

            if info.get('episode_complete', False) or info.get('all_destinations_connected', False):
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                episode_done = True
                break

            if done:
                episode_done = True
            if truncated_low:
                if info.get('deploy_fail'):
                    self._episode_deploy_failed.add(target_node)
                if info.get('deployment_failed'):
                    episode_done = True
                break

        actual_end_pos = self.env.current_node_location
        if low_step >= self.max_low_steps and not subgoal_achieved:
            if hasattr(self.high_agent, 'current_subgoal'):
                self.high_agent.current_subgoal = None
            if hasattr(self.high_agent, 'subgoal_steps'):
                self.high_agent.subgoal_steps = getattr(self.high_agent, 'subgoal_horizon', 10) + 1

            self._unreachable_targets.add(target_node)
            info['timeout'] = True
            info['fail'] = True
            info.setdefault('reason', 'timeout')

            if actual_end_pos == start_pos_before_low:
                info['stuck'] = True

        if (
            not subgoal_achieved
            and _cycle_phase == 'vnf_deployment'
            and info.get('reason') in {
                'vnf_completion_dead_end',
                'vnf_path_certificate_invalid',
            }
            and _cycle_transaction is not None
        ):
            self.env.current_tree = copy.deepcopy(
                _cycle_transaction['current_tree']
            )
            self.env.current_sfc = copy.deepcopy(
                _cycle_transaction['current_sfc']
            )
            self.env.nodes_on_tree = set(
                _cycle_transaction['nodes_on_tree']
            )
            self.env.current_node_location = (
                _cycle_transaction['current_node_location']
            )
            self.env.chain_nodes = list(_cycle_transaction['chain_nodes'])
            self.env.current_subgoal_full_path = [
                int(self.env.current_node_location)
            ] if self.env.current_node_location is not None else []
            self.env.current_path_trace = [self.env.current_node_location] \
                if self.env.current_node_location is not None else []
            _rm_tx = getattr(self.env, 'resource_mgr', None)
            if _rm_tx is not None:
                _rm_tx.current_tree = self.env.current_tree
            # The policy and completion decoder are deterministic on this
            # rolled-back snapshot.  Retrying the same target repeats the same
            # dead path, so exclude only that target and immediately explore
            # the remaining legal high-level candidates.
            self._episode_deploy_failed.add(int(target_node))
            self.env._episode_deploy_failed = self._episode_deploy_failed
            self.env._active_vnf_completion_path = None
            self.env._vnf_completion_schedule = None
            self.env._scheduled_high_mask_override = None
            if hasattr(self.high_agent, 'current_subgoal'):
                self.high_agent.current_subgoal = None
            logger.debug(
                f"[VNFTxnRollback] target={target_node} "
                f"reason={info.get('reason')}"
            )

        # [Diag-CycleFail] cycle结束时，若失败则打印详情
        if not subgoal_achieved and _cycle_phase == 'destination_connection':
            _fa2 = getattr(self.env, '_failed_anchors_for_target', {}).get(target_node, set())
            _reason_cf = info.get('reason') or info.get('error') or ('timeout' if info.get('timeout') else 'unknown')
            _anchor_cf = (
                _cycle_transaction.get('current_anchor_node')
                if _cycle_transaction is not None
                else getattr(self.env, 'current_anchor_node', None)
            )
            _llc_cf = getattr(self.env, 'low_level_controller', None)
            _hop_cf = _llc_cf._get_hop_distance(_anchor_cf, target_node) if _llc_cf and _anchor_cf else '?'
            # 分析target邻居的BW可达性
            _pool_cf = getattr(self.env.resource_mgr, 'pool', None)
            _bw_req_cf = float(self.env.current_request.get('bw_origin', 0)) \
                if self.env.current_request else 0
            _nbrs_cf = list(self.env.resource_mgr.get_neighbors(target_node)) \
                if hasattr(self.env, 'resource_mgr') else []
            _nbr_analysis = {}
            _tree_cf = (self.env.current_tree or {}).get('tree', {})
            for _nb in _nbrs_cf:
                _bw_in = round(_pool_cf.get_available_bandwidth(_nb, target_node), 1) if _pool_cf else -1
                _has_tree = _tree_cf.get((_nb, target_node), 0.0) > 0.0
                _nb_on_tree = _nb in set(getattr(self.env, 'nodes_on_tree', set()))
                _nbr_analysis[_nb] = f'bw={_bw_in},tree={int(_nb_on_tree)},in_tree_edge={int(_has_tree)}'
            logger.info(
                f"[CycleFail] target={target_node} anchor={_anchor_cf} "
                f"reason={_reason_cf} low_steps={low_step} hop_a2t={_hop_cf} "
                f"failed_anchors_so_far={_fa2} "
                f"nbr_analysis(need_bw={_bw_req_cf})={_nbr_analysis}"
            )

            if _cycle_transaction is not None:
                _failed_tree = (getattr(self.env, 'current_tree', None) or {}).get('tree', {})
                _stable_tree = (_cycle_transaction['current_tree'] or {}).get('tree', {})
                _discarded_edges = sorted(
                    edge for edge, flow in _failed_tree.items()
                    if float(flow) > 0.0 and float(_stable_tree.get(edge, 0.0)) <= 0.0
                )
                self.env.current_tree = copy.deepcopy(_cycle_transaction['current_tree'])
                self.env.current_sfc = copy.deepcopy(_cycle_transaction['current_sfc'])
                self.env.nodes_on_tree = set(_cycle_transaction['nodes_on_tree'])
                self.env.current_node_location = _cycle_transaction['current_node_location']
                self.env.chain_nodes = list(_cycle_transaction['chain_nodes'])
                self.env.current_subgoal_full_path = [
                    int(self.env.current_node_location)
                ] if self.env.current_node_location is not None else []
                self.env.current_path_trace = [self.env.current_node_location] \
                    if self.env.current_node_location is not None else []
                _rm_tx = getattr(self.env, 'resource_mgr', None)
                if _rm_tx is not None:
                    _rm_tx.current_tree = self.env.current_tree
                logger.info(
                    f"[DestTxnRollback] target={target_node} anchor={_anchor_cf} "
                    f"discarded_edges={_discarded_edges}"
                )

            _active_failed_step = getattr(
                self.env, '_active_destination_plan_step', None
            )
            if (
                isinstance(_active_failed_step, dict)
                and int(_active_failed_step.get('target', -1)) == int(target_node)
            ):
                # Every later step in this schedule was decoded from the
                # speculative tree produced by the failed step. Rebuild the
                # whole schedule from the rolled-back ledger.
                self.env._destination_completion_schedule = None
                self.env._active_destination_plan_step = None
                self.env._active_destination_path = None
                logger.info(
                    "[DestinationScheduleInvalidated] target=%s anchor=%s "
                    "reason=%s",
                    target_node, _anchor_cf, _reason_cf,
                )

        # [Monitor B] 记录本次 cycle 的阶段步数
        _phase_key = _cycle_phase if _cycle_phase in self._phase_steps else 'other'
        self._phase_steps[_phase_key].append(low_step)

        # [Monitor C] 失败时记录卡在哪个 dest
        if not subgoal_achieved and _cycle_phase == 'destination_connection':
            _vnf_done_c = getattr(self.env, 'next_vnf_idx', 0)
            _vnf_total_c = len(self.env.current_request.get('vnf', [])) \
                if self.env.current_request else 0
            _dest_conn_c = len(self.shared.get_connected_dests_view()) \
                if self.env.current_tree else 0
            _dest_total_c = len(self.env.current_request.get('dest', [])) \
                if self.env.current_request else 0

            # [Item 2] 完整失败原因推断：直接读 reason/error 字段，兼容多种 key
            _fail_reason_c = (
                info.get('reason') or
                info.get('error') or
                info.get('fail_reason') or
                ''
            )
            if not _fail_reason_c:
                if info.get('timeout', False):
                    _fail_reason_c = 'timeout'
                elif info.get('bandwidth_island', False) or info.get('bandwidth_exhausted', False):
                    _fail_reason_c = 'bandwidth_exhausted'
                elif info.get('deploy_fail', False) or info.get('vnf_fail', False):
                    _fail_reason_c = 'resource_insufficient'
                elif info.get('stuck', False):
                    _fail_reason_c = 'stuck'
                elif info.get('fail', False):
                    _fail_reason_c = 'dest_fail'
                else:
                    _fail_reason_c = 'unknown'

            if (
                _fail_reason_c in self._SOFT_DEST_FAILURE_REASONS
                and getattr(self.env, '_safe_dest_recovery_enabled', False)
            ):
                if not hasattr(self.env, '_dest_recovery_targets'):
                    self.env._dest_recovery_targets = set()
                self.env._dest_recovery_targets.add(int(target_node))

            self._fail_dest_log.append({
                'failed_target_dest': target_node,
                'failed_anchor': _anchor_cf,
                'failed_phase': _cycle_phase,
                'dest_done': _dest_conn_c,
                'dest_total': _dest_total_c,
                'vnf_done': _vnf_done_c,
                'vnf_total': _vnf_total_c,
                'reason': _fail_reason_c,
                'low_steps': low_step,
            })
            logger.debug(
                f"[FailDest] target={target_node} "
                f"anchor={_anchor_cf} "
                f"phase={_cycle_phase} reason={_fail_reason_c} "
                f"dest={_dest_conn_c}/{_dest_total_c} vnf={_vnf_done_c}/{_vnf_total_c} "
                f"steps={low_step}"
            )

            # [Fix-A] dest 阶段失败封禁策略：最后1个dest失败时不封禁
            # 原来：任何失败都封禁 → 最后1个dest第一次失败就永久封禁 → episode必然失败
            # 修复：只剩1个pending dest时，timeout/stuck不封禁，让高层换anchor重试
            if _cycle_phase == 'destination_connection' and _fail_reason_c not in (
                    'bandwidth_island', 'bandwidth_exhausted'):
                _pending_after_fail = _dest_total_c - _dest_conn_c
                _is_last_dest = (_pending_after_fail <= 1)
                _is_soft_fail = (
                    _fail_reason_c in self._SOFT_DEST_FAILURE_REASONS
                )
                # [Fix-FailedAnchor] 任意dest失败都记录失败anchor，不限于最后1个
                # 原来只记录 _is_last_dest，导致非末尾dest高层反复选同一个死路anchor
                if _is_soft_fail:
                    if not hasattr(self.env, '_failed_anchors_for_target'):
                        self.env._failed_anchors_for_target = {}
                    _cur_failed_anchor = _anchor_cf
                    if _cur_failed_anchor is not None:
                        self.env._failed_anchors_for_target.setdefault(
                            target_node, set()
                        ).add(int(_cur_failed_anchor))
                    else:
                        self.env._failed_anchors_for_target.setdefault(
                            target_node, set()
                        )
                    _failed_cnt = len(self.env._failed_anchors_for_target[target_node])
                    logger.debug(
                        f"[FailedAnchor] target={target_node} anchor={_cur_failed_anchor} "
                        f"累计失败={_failed_cnt} is_last={_is_last_dest}"
                    )

                if _is_soft_fail:
                    # A soft failure invalidates one anchor, not the target.
                    # Retry any destination while another full-stage anchor
                    # remains; limiting this to the final destination caused
                    # earlier receivers to be banned after one bad branch.
                    _nodes_on_tree_r = set(getattr(self.env, 'nodes_on_tree', set()))
                    _total_vnf = len(self.env.current_request.get('vnf', [])) if self.env.current_request else 0
                    _full_stage_nodes = {
                        n for n in _nodes_on_tree_r
                        if self.env.current_tree.get('node_stage', {}).get(n, 0) >= _total_vnf
                    }
                    _available_anchors = _full_stage_nodes - self.env._failed_anchors_for_target.get(target_node, set())
                    _MAX_RETRIES = min(max(1, len(_full_stage_nodes)), 5)
                    _failed_cnt_last = len(self.env._failed_anchors_for_target.get(target_node, set()))
                    _should_ban = (_failed_cnt_last >= _MAX_RETRIES) or (len(_available_anchors) == 0)
                    if _should_ban:
                        self._unreachable_targets.add(target_node)
                        logger.debug(
                            f"[FailDest] target={target_node} 已尝试{_failed_cnt_last}个anchor "
                            f"可用anchor={len(_available_anchors)}，封禁"
                        )
                    else:
                        self._unreachable_targets.discard(target_node)
                        logger.debug(
                            f"[FailDest] target={target_node} 第{_failed_cnt}次软失败，"
                            f"换anchor重试，剩余可用anchor={len(_available_anchors)}"
                        )
                else:
                    self._unreachable_targets.add(target_node)
                    logger.debug(
                        f"[FailDest] target={target_node} 已加入本 episode 临时封禁 "
                        f"(reason={_fail_reason_c})"
                    )

            # [Log 3] DestFailDetail — dest 阶段详细失败信息
            if _cycle_phase == 'destination_connection':
                _bw_req_fd = float(self.env.current_request.get('bw_origin', -1)) \
                    if self.env.current_request else -1
                # mask 候选数：读低层最近一次 mask 存的值（get_low_level_action_mask 里设置）
                _llc_fd = getattr(self.env, 'low_level_controller', None)
                _mask_alive = getattr(self.env, '_last_mask_topk_count', -1)
                _pool_size = getattr(self.env, '_last_anchor_pool_size', -1)
                _cur_node = getattr(self.env, 'current_node_location', None)
                _reach = self._check_bw_reachability(
                    source=_cur_node,
                    target=target_node,
                    bw_req=max(0.0, _bw_req_fd),
                )
                _tgt_inc = self._check_target_incident_bw(
                    target=target_node,
                    bw_req=max(0.0, _bw_req_fd),
                )
                logger.debug(
                    f"[TimeoutReachability] "
                    f"target={target_node} current={_cur_node} "
                    f"reachable_bw={int(_reach['reachable_bw'])} "
                    f"shortest_bw_hop={_reach['shortest_bw_hop']} "
                    f"bottleneck_bw={_reach['bottleneck_bw']:.2f} "
                    f"bw_req={_bw_req_fd:.2f}"
                )
                logger.debug(
                    f"[TimeoutTargetBW] target={target_node} "
                    f"ok_edges={_tgt_inc['target_incident_ok']}/{_tgt_inc['target_incident_total']} "
                    f"bw_req={_bw_req_fd:.2f}"
                )
                logger.debug(
                    f"[DestFailDetail] ep={self.current_episode} "
                    f"req={self.env.current_request.get('id','?') if self.env.current_request else '?'} "
                    f"target={target_node} "
                    f"anchor={getattr(self.env,'current_anchor_node',None)} "
                    f"reason={_fail_reason_c} "
                    f"dest={_dest_conn_c}/{_dest_total_c} "
                    f"steps={low_step} "
                    f"pool_size={_pool_size} "
                    f"mask_alive_last={_mask_alive} "
                    f"bw_req={_bw_req_fd} "
                    f"current_node={_cur_node}"
                )

            # [Fix G] dest timeout 후 anchor 재설정
            # 현재 위치가 목표에서 멀리 떨어진 경우, 마지막 VNF 위치로 돌아가서 재시도
            if _fail_reason_c == 'timeout':
                _connected_now = self.shared.get_connected_dests_view()
                _all_d = set(self.env.current_request.get('dest', []))\
                         if self.env.current_request else set()
                _remaining_now = _all_d - _connected_now
                _cur_node = getattr(self.env, 'current_node_location', None)
                _last_vnf = getattr(self.env, 'chain_nodes', [])
                _last_vnf = _last_vnf[-1] if _last_vnf else None
                # 현재 위치가 마지막 VNF로부터 3홉 이상 멀어진 경우 강제 복귀
                if (_cur_node is not None and _last_vnf is not None
                        and _cur_node != _last_vnf and len(_remaining_now) <= 2):
                    _llc_g = getattr(self.env, 'low_level_controller', None)
                    _dist_g = _llc_g._get_hop_distance(_cur_node, _last_vnf) \
                              if _llc_g is not None else 0
                    if _dist_g >= 3:
                        # 현재 경로 trace 초기화 → 다음 dest 시도 시 anchor에서 깔끔하게 출발
                        self.env.current_path_trace = []
                        logger.debug(
                            f"[Fix-G] timeout후 anchor 재초기화: "
                            f"cur={_cur_node} last_vnf={_last_vnf} dist={_dist_g}"
                        )

        # [Log 5] PlannerDiag — 每 cycle 统计 planner vs policy 使用情况（5% 采样）
        import random as _r5
        if _r5.random() < 0.05:
            logger.debug(
                f"[PlannerDiag] ep={self.current_episode} "
                f"req={self.env.current_request.get('id','?') if self.env.current_request else '?'} "
                f"phase={_cycle_phase} "
                f"planner_used={int(use_planner_decision)} "
                f"planner_steps={_planner_steps_this_cycle} "
                f"policy_steps={_policy_steps_this_cycle} "
                f"total_steps={low_step} "
                f"success={int(subgoal_achieved)}"
            )

        high_done = episode_done

        # v4.2: 树级增量奖励，替代原 0.1 * low_total_reward 弱耦合
        # 在 cycle 开始前记录快照（由 run_high_low_cycle 入口处采集，见下方 _snap_before）
        high_reward = self._compute_high_reward(low_total_reward, low_step=low_step)
        if subgoal_achieved and low_step > 0:
            _efficiency = max(0.3, 1.0 - (low_step - 1) / self.max_low_steps)
            high_reward = high_reward * _efficiency if high_reward > 0 else high_reward
        # [Fix-Reward] 失败 subgoal의 reward는 반드시 음수
        if not subgoal_achieved:
            _fail_reason_r = ''
            if isinstance(info, dict):
                _fail_reason_r = info.get('reason', '') or info.get('error', '') or ''
            _is_hard_fail = ('timeout' in str(_fail_reason_r) or
                             bool(info.get('fail') if isinstance(info, dict) else False))
            if _cycle_phase == 'destination_connection':
                _cap = -3.0 if _is_hard_fail else -1.5  # 扩大失败惩罚
                high_reward = min(high_reward, _cap)
            elif _cycle_phase == 'vnf_deployment':
                high_reward = min(high_reward, -2.0)  # 扩大失败惩罚

        if not high_done and self.env.current_request:
            if hasattr(self.env, 'high_level_controller') and hasattr(self.env.high_level_controller,
                                                                      '_is_all_tasks_completed'):
                try:
                    completed, status = self.env.high_level_controller._is_all_tasks_completed()
                    if completed:
                        high_done = True
                        episode_done = True
                except Exception:
                    pass

        if training and not self._ablation_single_dqn:
            high_next_state = None if high_done else self.env.get_high_level_state_graph()

            # [Fix1] 高层 replay 带 candidate metadata，让 _update_high_level 走 scorer 路径
            _high_action_meta = getattr(self.high_agent, '_last_high_action_meta', None)
            _next_high_action_meta = None
            if not high_done:
                _hlc_next = getattr(self.env, 'high_level_controller', None)
                if _hlc_next is not None and hasattr(_hlc_next, 'get_high_level_candidates'):
                    try:
                        _next_cand = _hlc_next.get_high_level_candidates()
                        if _next_cand is not None:
                            _next_high_action_meta = {
                                'candidate_indices':     _next_cand.get('indices'),
                                'candidate_local_feats': self._candidate_feats(_next_cand),
                                'action_mask': None,
                            }
                    except Exception:
                        pass

            if hasattr(self.high_agent, 'store_transition_high'):
                try:
                    self.high_agent.store_transition_high(
                        high_obs, actual_high_action,
                        high_reward, high_next_state, high_done,
                        action_meta=_high_action_meta,
                        next_action_meta=_next_high_action_meta,
                    )
                except TypeError:
                    # fallback: 구버전 서명 호환
                    self.high_agent.store_transition_high(
                        high_obs, actual_high_action,
                        high_reward, high_next_state, high_done
                    )
            else:
                self._store_transition(
                    self.high_agent, high_obs, actual_high_action,
                    high_reward, high_next_state, high_done
                )

        self._update_stats(high_done, info if isinstance(info, dict) else {})

        return high_reward, high_done, {
            'high_action': actual_high_action,
            'target_node': target_node,
            'low_steps': low_step,
            'high_reward': high_reward,
            'info': info
        }

    def run_episode(self, training=True, max_steps=100):
        _request_algorithm_started_ns = time.perf_counter_ns()
        self._reset_decision_timings()
        self.current_episode += 1
        self.resources_released = False

        # [BugFix] env.reset()返回(state, info)二元组，必须解包
        # 同时高层应使用get_high_level_state_graph()，而非get_state()（后者是低层状态）
        _, _ = self.env.reset()
        high_obs = self.env.get_high_level_state_graph()

        self._unreachable_targets = set()
        self.env._vnf_completion_schedule = None
        self.env._destination_completion_schedule = None
        self.env._active_destination_plan_step = None
        self.env._active_destination_path = None
        # [Fix-LastDestRetry] 记录最后1个dest已尝试过的anchor，防止无限重试
        self.env._failed_anchors_for_target = {}
        self.env._dest_recovery_targets = set()
        self._episode_deploy_failed = set()  # 每 episode 重置，不跨 episode 封禁节点
        self.env._episode_deploy_failed = self._episode_deploy_failed
        self._failure_memory_scope = None
        self._last_tree_redundancy = 0.0

        # [EarlyDiagEpisode] 每 episode 重置诊断字段
        self.env._first_dest_steps = -1
        self.env._first_dest_new_edges = -1
        self.env._first_dest_reuse_edges = -1

        episode_done = False
        total_reward = 0.0
        total_steps = 0
        no_progress_cycles = 0
        terminal_failure_reason = ''
        last_connected_count = 0
        last_vnf_count = 0
        last_tree_edge_count = 0   # [Fix P1] 加入树边增长作为进展指标
        MAX_NO_PROGRESS = 5
        info = {}  # 防止while未进入时UnboundLocalError

        while not episode_done and total_steps < max_steps:
            cycle_reward, done, info = self.run_high_low_cycle(
                high_obs, training=training
            )

            total_reward += cycle_reward
            total_steps += 1
            episode_done = done

            _cycle_info = info.get('info', {}) if isinstance(info, dict) else {}
            if _cycle_info.get('deployment_failed'):
                terminal_failure_reason = str(
                    _cycle_info.get('reason') or 'deployment_failed'
                )
                episode_done = True

            cur_vnf = getattr(self.env, 'next_vnf_idx', 0)
            cur_conn = len(self.shared.get_connected_dests_view()) if self.env.current_tree else 0
            cur_tree_edges = len(self.shared.get_positive_tree_edge_set())

            # [Fix P1] 进展判据：任务进度 OR 树边扩展；只有真正硬停滞才计数
            _cycle_diag = info.get('info', {}) if isinstance(info, dict) else {}
            _cur_reason = (
                info.get('reason', '')
                or info.get('error', '')
                or (_cycle_diag.get('reason', '') if isinstance(_cycle_diag, dict) else '')
                or (_cycle_diag.get('error', '') if isinstance(_cycle_diag, dict) else '')
            ) if isinstance(info, dict) else ''
            _has_hard_stall = (_cur_reason in {
                'timeout', 'consecutive_timeout', 'no_valid_low_action',
                'high_target_recheck_failed', 'selected_completed_target',
                'high_subgoal_truncated', 'vnf_completion_dead_end',
                'vnf_path_certificate_invalid',
            })

            if cur_vnf > last_vnf_count or cur_conn > last_connected_count or cur_tree_edges > last_tree_edge_count:
                no_progress_cycles = 0
                last_vnf_count = cur_vnf
                last_connected_count = cur_conn
                last_tree_edge_count = cur_tree_edges
            else:
                # 只在真正硬停滞时累计，避免误杀"慢但在推进"的episode
                no_progress_cycles += 1 if _has_hard_stall else 0
            if not episode_done and no_progress_cycles >= MAX_NO_PROGRESS:
                logger.warning(f'[Coord] 连续{no_progress_cycles}个cycle无进展，终止(VNF={cur_vnf} Conn={cur_conn})')
                # 用reward_critic的超时惩罚，根据当前阶段决定惩罚幅度
                _in_vnf = (getattr(self.env, 'current_phase', None) == 'vnf_deployment')
                _conn = len(self.shared.get_connected_dests_view()) if self.env.current_tree else 0
                _total = len(self.env.current_request.get('dest', [])) if self.env.current_request else 1
                _rc = getattr(self.env, 'reward_critic', None)
                _np_penalty = float(_rc.get_reward('timeout', in_vnf_phase=_in_vnf,
                                                   connected_count=_conn, total_dests=max(1, _total))) if _rc else -20.0
                total_reward += _np_penalty
                episode_done = True
                info['fail'] = True
                info['reason'] = 'no_progress'
                # [Fix] no_progress 终止时写入 _fail_dest_log，
                # 让 _final_reason 能从 _fail_dest_log 继承，避免 Unknown 掩盖真实原因
                if not hasattr(self, '_fail_dest_log'):
                    self._fail_dest_log = []
                _phase_np = getattr(self.env, 'current_phase', 'unknown')
                _conn_np  = len(self.shared.get_connected_dests_view()) if self.env.current_tree else 0
                _total_np = len(self.env.current_request.get('dest', [])) if self.env.current_request else 0
                self._fail_dest_log.append({
                    'failed_target_dest': None,
                    'failed_anchor': None,
                    'failed_phase': _phase_np,
                    'dest_done': _conn_np,
                    'dest_total': _total_np,
                    'reason': 'no_progress',
                    'low_steps': 0,
                })

            if not episode_done:
                high_obs = self.env.get_high_level_state_graph()

        vnf_success = False
        dest_success = False
        episode_success = False
        completion_status = "未知"
        sft_validation = {'ok': False, 'reason': 'not_checked'}

        # ── [Fix] 在任何 rollback 之前先捕获本 episode 的连通进度峰值快照 ──
        # _archive_episode_fail → release_request_record(rollback=True) 会
        # 清空 request_table[req_id].connected_dests，导致之后读到 0。
        # 必须在这里（rollback 发生之前）把峰值记录下来，供 EarlyDiagEpisode 使用。
        _dest_done_peak = 0
        _vnf_done_peak  = getattr(self.env, 'next_vnf_idx', 0)
        try:
            if self.env.current_tree:
                _dest_done_peak = len(self.shared.get_connected_dests_view())
        except Exception:
            pass

        try:
            if hasattr(self.env, 'current_request') and self.env.current_request:
                vnf_list = self.env.current_request.get('vnf', [])
                dest_list = self.env.current_request.get('dest', [])

                logical_completed = False
                logical_status = "未完成"
                if hasattr(self.env, 'high_level_controller') and \
                        hasattr(self.env.high_level_controller, '_is_all_tasks_completed'):
                    try:
                        logical_completed, logical_status = (
                            self.env.high_level_controller._is_all_tasks_completed()
                        )
                    except Exception as e:
                        logical_completed = self._fallback_success_check()
                        logical_status = "回退检测"
                else:
                    logical_completed = self._fallback_success_check()
                    logical_status = "回退检测"

                if hasattr(self.env, 'next_vnf_idx'):
                    vnf_progress = self.env.next_vnf_idx
                    vnf_success = vnf_progress >= len(vnf_list)

                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                    connected_dests = self.shared.get_connected_dests_view()
                    dest_success = len(connected_dests) >= len(dest_list)

                req_id = self.env.current_request.get('id')
                resource_mgr = getattr(self.env, 'resource_mgr', None)
                if req_id is None:
                    sft_validation = {'ok': False, 'reason': 'missing_req_id'}
                elif not hasattr(resource_mgr, 'validate_request_sft_snapshot'):
                    sft_validation = {'ok': False, 'reason': 'sft_validator_unavailable'}
                else:
                    try:
                        sft_validation = resource_mgr.validate_request_sft_snapshot(req_id)
                    except Exception as exc:
                        sft_validation = {
                            'ok': False,
                            'reason': f'validation_exception:{exc}',
                        }

                # Final success is ledger-backed. Logical progress remains a
                # diagnostic only; a valid snapshot proves VNF placement,
                # destination reachability and exact BW ledger correspondence.
                episode_success = bool(sft_validation.get('ok', False))
                completion_status = (
                    'SFT快照校验通过'
                    if episode_success
                    else f"SFT快照校验失败:{sft_validation.get('reason', 'unknown')}"
                )
                if logical_completed and not episode_success:
                    logger.warning(
                        f"[CoordinatorSuccessReject] req={req_id} "
                        f"logical={logical_status} validation={sft_validation.get('reason')}"
                    )

                if terminal_failure_reason:
                    episode_success = False
                    completion_status = terminal_failure_reason

        except Exception as e:
            episode_success = False

        if episode_success:
            self.resources_released = True
        else:
            # [BugFix] 检查 _bw_already_rolled_back 防止双重回滚。
            # low_level_controller._archive_episode_fail() 在失败时已经回滚了资源
            # 并设置了此标志，若不检查会导致 CPU/MEM/BW 被二次归还到 pool，
            # pool_used 虚低，lc >> pool 的幽灵差值根因。
            already_rolled = getattr(self.env, '_bw_already_rolled_back', False)
            if not already_rolled:
                try:
                    if (hasattr(self.env, 'resource_mgr') and
                            hasattr(self.env.resource_mgr, '_archive_request')):
                        self.env.resource_mgr._archive_request(
                            success=False, already_rolled_back=False)
                except Exception:
                    pass
            # 每个 episode 结束时重置标志，供下一轮使用
            self.env._bw_already_rolled_back = False

        self._reset_episode_stats()

        # ── [VIS/EXPORT] 成功时只导出已校验的 RequestRecord 权威快照 ────────
        tree_snapshot = None
        authoritative_record = None
        try:
            if episode_success:
                _rm = getattr(self.env, 'resource_mgr', None)
                _table = getattr(_rm, 'request_table', {}) if _rm is not None else {}
                authoritative_record = _table.get(req_id)
                if authoritative_record is None:
                    _alt = str(req_id) if not isinstance(req_id, str) else int(req_id)
                    authoritative_record = _table.get(_alt)
            if authoritative_record is not None:
                tree_snapshot = {
                    'tree': dict(authoritative_record.tree_edges),
                    'placement': copy.deepcopy(authoritative_record.placement_detail),
                    'connected_dests': set(authoritative_record.connected_dests),
                }
            elif hasattr(self.env, 'current_tree') and self.env.current_tree:
                _t = self.env.current_tree
                tree_snapshot = {
                    'tree': dict(_t.get('tree', {})),
                    'placement': copy.deepcopy(_t.get('placement', {})),
                    'connected_dests': set(_t.get('connected_dests', set())),
                }
        except Exception:
            pass
        req_snapshot = None
        try:
            if hasattr(self.env, 'current_request') and self.env.current_request:
                req_snapshot = dict(self.env.current_request)
        except Exception:
            pass
        if authoritative_record is not None:
            chain_snapshot = [
                int(authoritative_record.placement_by_vnf[index])
                for index in range(len(authoritative_record.vnfs))
            ]
        else:
            chain_snapshot = list(getattr(self.env, 'chain_nodes', []))
        # 收集 current_sfc（分层DAG）快照
        sfc_snapshot = None
        try:
            _sfc = getattr(self.env, 'current_sfc', None)
            if _sfc:
                sfc_snapshot = {
                    'chain_nodes': list(_sfc.get('chain_nodes', [])),
                    'spine_paths': [list(p) for p in _sfc.get('spine_paths', [])],
                    'branch_paths': {k: list(v) for k, v in _sfc.get('branch_paths', {}).items()},
                    'branch_roots': {k: v for k, v in _sfc.get('branch_roots', {}).items()},
                }
        except Exception:
            pass
        # ──────────────────────────────────────────────────────────────────

        # [Fix] episode终结时驱动ε衰减（基于episode而非steps）
        if training and hasattr(self.high_agent, 'on_episode_end'):
            self.high_agent.on_episode_end()

        # [BugFix] 强制清理_ep_transitions，防止截断/失败episode跨episode泄漏
        if training and hasattr(self.low_agent, 'finalize_episode_memory'):
            self.low_agent.finalize_episode_memory()

        # reason透传：外层info可能是cycle dict，真正失败原因在info['info']里
        _inner_info = info.get('info', {}) if isinstance(info, dict) else {}
        _final_reason = (
            info.get('reason', '') or
            info.get('error', '') or
            _inner_info.get('reason', '') or
            _inner_info.get('error', '')
        )
        if terminal_failure_reason:
            _final_reason = terminal_failure_reason

        if (
            not _final_reason
            and not episode_success
            and sft_validation.get('reason') not in {None, '', 'not_checked'}
        ):
            _final_reason = f"sft_validation:{sft_validation['reason']}"

        # [Fix C] 若 _final_reason 为空但 episode 失败，从 _fail_dest_log 继承最后一次失败原因
        if not _final_reason and not episode_success:
            _fail_log_c = getattr(self, '_fail_dest_log', [])
            if _fail_log_c:
                _final_reason = _fail_log_c[-1].get('reason', 'unknown')
            else:
                _final_reason = 'unknown'

        # [Task9] Episode监控：失败原因分布 + 进度 + sharing指标
        try:
            if not hasattr(self, '_ep_stats_accum'):
                self._ep_stats_accum = {
                    'total': 0, 'success': 0,
                    'fail_timeout': 0, 'fail_bw': 0, 'fail_vnf': 0, 'fail_other': 0,
                    'sum_low_steps': 0, 'sum_tree_edges': 0, 'sum_reused_edges': 0,
                    'sum_sharing_ratio': 0.0,
                }
            _acc = self._ep_stats_accum
            _acc['total'] += 1
            if episode_success:
                _acc['success'] += 1
            else:
                if 'timeout' in _final_reason or 'consecutive_timeout' in _final_reason:
                    _acc['fail_timeout'] += 1
                elif 'bandwidth' in _final_reason:
                    _acc['fail_bw'] += 1
                elif 'resource' in _final_reason or 'deploy' in _final_reason:
                    _acc['fail_vnf'] += 1
                else:
                    _acc['fail_other'] += 1
            _acc['sum_low_steps'] += total_steps
            _tree_edges = 0
            _reused_edges = 0
            _sharing_ratio = 0.0
            if self.env.current_tree:
                _td = self.env.current_tree.get('tree', {})
                _tree_edges = sum(1 for f in _td.values() if f > 0.0)
                _reused_edges = sum(
                    1 for v in self.env.current_tree.get('tree_usage', {}).values() if v > 1)
                _total_dests = len(self.env.current_request.get('dest', [])) \
                    if self.env.current_request else 1
                _sharing_ratio = _reused_edges / max(1, _tree_edges) if _tree_edges > 0 else 0.0
            _acc['sum_tree_edges'] += _tree_edges
            _acc['sum_reused_edges'] += _reused_edges
            _acc['sum_sharing_ratio'] += _sharing_ratio

            # 每50个episode输出一次汇总
            _N = _acc['total']
            if _N % 50 == 0:
                _suc_rate = _acc['success'] / max(1, _N)
                _vnf_prog = getattr(self.env, 'next_vnf_idx', 0)
                _vnf_total = len(self.env.current_request.get('vnf', [])) \
                    if self.env.current_request else 1
                _dest_conn = len(self.shared.get_connected_dests_view()) \
                    if self.env.current_tree else 0
                _dest_total = len(self.env.current_request.get('dest', [])) \
                    if self.env.current_request else 1

                # [Monitor B] 各阶段平均步数
                _vnf_steps  = self._phase_steps.get('vnf_deployment', [])
                _dest_steps = self._phase_steps.get('destination_connection', [])
                _avg_vnf_steps  = sum(_vnf_steps)  / max(1, len(_vnf_steps))
                _avg_dest_steps = sum(_dest_steps) / max(1, len(_dest_steps))

                # [Monitor C] 失败 dest 分布：统计最后 _fail_dest_log 里 dest_done 分布
                _fail_log = getattr(self, '_fail_dest_log', [])
                _fail_by_dest = {}
                for _fl in _fail_log:
                    _k = f"{_fl['dest_done']}/{_fl['dest_total']}"
                    _fail_by_dest[_k] = _fail_by_dest.get(_k, 0) + 1

                logger.debug(
                    f"[EpStats N={_N}] "
                    f"success={_suc_rate:.2%} | "
                    f"fail: timeout={_acc['fail_timeout']} bw={_acc['fail_bw']} "
                    f"vnf={_acc['fail_vnf']} other={_acc['fail_other']} | "
                    f"最后ep: vnf={_vnf_prog}/{_vnf_total} dest={_dest_conn}/{_dest_total} | "
                    f"avg_steps={_acc['sum_low_steps']/_N:.1f} "
                    f"avg_vnf_steps={_avg_vnf_steps:.1f} avg_dest_steps={_avg_dest_steps:.1f} | "
                    f"avg_tree_edges={_acc['sum_tree_edges']/_N:.1f} "
                    f"avg_reused={_acc['sum_reused_edges']/_N:.1f} "
                    f"avg_sharing={_acc['sum_sharing_ratio']/_N:.2f} | "
                    f"fail_dest_dist={_fail_by_dest}"
                )
                # 重置累计器，窗口统计
                self._ep_stats_accum = {
                    'total': 0, 'success': 0,
                    'fail_timeout': 0, 'fail_bw': 0, 'fail_vnf': 0, 'fail_other': 0,
                    'sum_low_steps': 0, 'sum_tree_edges': 0, 'sum_reused_edges': 0,
                    'sum_sharing_ratio': 0.0,
                }
                self._phase_steps = {'vnf_deployment': [], 'destination_connection': [], 'other': []}
                self._fail_dest_log = []
        except Exception as _stat_e:
            logger.debug(f"[EpStats] 统计异常: {_stat_e}")

        # ── [EarlyDiagEpisode] per-episode 诊断摘要 ───────────────────────
        try:
            _req_id = self.env.current_request.get('id', '?') \
                if self.env.current_request else '?'
            _vnf_total = len(self.env.current_request.get('vnf', [])) \
                if self.env.current_request else 0
            _vnf_done = getattr(self.env, 'next_vnf_idx', 0)
            _dest_total = len(self.env.current_request.get('dest', [])) \
                if self.env.current_request else 0
            # 使用 rollback 前捕获的峰值，避免 release_request_record 已清空
            _dest_done = _dest_done_peak
            _flow_edges = 0
            _shr_eff = 0.0
            if self.env.current_tree:
                _te = self.env.current_tree.get('tree', {})
                _tu = self.env.current_tree.get('tree_usage', {})
                _flow_edges = sum(1 for f in _te.values() if f > 0.0)
                _reuse_e = sum(1 for v in _tu.values() if v > 1)
                _shr_eff = min(1.0, _reuse_e / max(1, _flow_edges))
            _dest_steps = self._phase_steps.get('destination_connection', [])
            _vnf_steps  = self._phase_steps.get('vnf_deployment', [])
            _avg_dest_steps = sum(_dest_steps)/max(1,len(_dest_steps)) if _dest_steps else -1
            _avg_vnf_steps  = sum(_vnf_steps)/max(1,len(_vnf_steps))   if _vnf_steps  else -1
            _fail_log = getattr(self, '_fail_dest_log', [])
            _timeout_cnt = sum(1 for f in _fail_log if 'timeout' in str(f.get('reason','')))
            _stuck_cnt   = sum(1 for f in _fail_log if 'stuck' in str(f.get('reason','')))
            logger.debug(
                f"[EarlyDiagEpisode] ep={self.current_episode} req={_req_id} "
                f"success={int(episode_success)} fail_reason={_final_reason or 'none'} "
                f"vnf={_vnf_done}/{_vnf_total} dest={_dest_done}/{_dest_total} "
                f"first_dest_steps={getattr(self.env,'_first_dest_steps',-1)} "
                f"first_dest_new_edges={getattr(self.env,'_first_dest_new_edges',-1)} "
                f"first_dest_reuse={getattr(self.env,'_first_dest_reuse_edges',-1)} "
                f"dest_timeout={_timeout_cnt} dest_stuck={_stuck_cnt} "
                f"avg_vnf_steps={_avg_vnf_steps:.1f} avg_dest_steps={_avg_dest_steps:.1f} "
                f"flow_edges={_flow_edges} sharing_eff={_shr_eff:.3f}"
            )
        except Exception as _ede:
            logger.debug(f"[EarlyDiagEpisode] 统计异常: {_ede}")

        _request_algorithm_ms = (
            time.perf_counter_ns() - _request_algorithm_started_ns
        ) / 1_000_000.0
        _algorithm_timing = self._decision_timing_summary(_request_algorithm_ms)

        return total_reward, {
            'steps': total_steps,
            'success': episode_success,
            'reward': total_reward,
            'reason': _final_reason if not episode_success else '',
            'fail': not episode_success,
            'subgoals_ok': self.stats.get('subgoals_ok', 0),
            'subgoals_fail': self.stats.get('subgoals_fail', 0),
            'vnf_success': vnf_success,
            'dest_success': dest_success,
            'completion_status': completion_status,
            'tree_snapshot': tree_snapshot,
            'req_snapshot': req_snapshot,
            'chain_nodes': chain_snapshot,
            'sfc_snapshot': sfc_snapshot,
            'sft_validation': sft_validation,
            'algorithm_timing': _algorithm_timing,
        }

    def _encode_low_state_for_policy(self, low_state):
        """
        v4.2: 把 low_state（PyG Data 对象）编码成 (state_emb, goal_emb)，
        供直接调 low_policy.select_action() 时使用。

        复用 low_agent 内部已有的 encoder / _get_state_embedding 逻辑，
        保证与正常训练推断路径完全一致。

        返回：
            state_emb : Tensor [1, N, H]
            goal_emb  : Tensor [1, goal_dim] 或 None
        """
        import torch
        device = getattr(self.low_agent, 'device',
                         torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # ── 1. state_emb：优先用 encoder，其次用 _extract_state_embedding ──
        state_emb = None
        encoder = getattr(self.low_agent, 'encoder', None)
        if encoder is not None:
            try:
                s = low_state[0] if isinstance(low_state, tuple) else low_state
                if hasattr(s, 'x') and hasattr(s, 'edge_index'):
                    ei = s.edge_index.to(device) if s.edge_index is not None else None
                    if ei is not None:
                        n_nodes = s.x.size(0)
                        # edge_attr 修复（和 agent_train._fix_ea 一致）
                        ea = getattr(s, 'edge_attr', None)
                        fdim = 5
                        if ea is None:
                            ea = torch.zeros(ei.shape[1], fdim, device=device)
                        else:
                            ea = ea.to(device)
                            if ea.dim() == 1: ea = ea.unsqueeze(1)
                            if ea.shape[0] != ei.shape[1]:
                                ea = torch.zeros(ei.shape[1], fdim, device=device)
                            if ea.shape[1] < fdim:
                                ea = torch.cat(
                                    [ea, torch.zeros(ea.shape[0], fdim - ea.shape[1], device=device)], dim=1)
                        batch_idx = torch.zeros(n_nodes, dtype=torch.long, device=device)
                        tei = getattr(s, 'tree_edge_index', None)
                        if tei is not None: tei = tei.to(device)
                        dest_mask = getattr(s, 'dest_mask', None)
                        if dest_mask is not None:
                            dest_mask = dest_mask.to(device)
                        req_vec = getattr(s, 'req_vec', None)
                        if req_vec is not None:
                            req_vec = req_vec.to(device)
                        elif getattr(encoder, 'req_fc', None) is not None:
                            req = getattr(self.env, 'current_request', None)
                            if req:
                                bw = float(req.get('bw_origin', req.get('bw', 0.0)))
                                cpu_list = req.get('cpu_origin', req.get('cpu', []))
                                mem_list = req.get('memory_origin', req.get('memory', []))
                                avg_cpu = float(np.mean(cpu_list)) if len(cpu_list) > 0 else 0.0
                                avg_mem = float(np.mean(mem_list)) if len(mem_list) > 0 else 0.0
                                req_vec = torch.tensor([[bw, avg_cpu, avg_mem]], dtype=torch.float32, device=device)
                        out = encoder(s.x.to(device), ei, ea, batch=batch_idx,
                                      tree_edge_index=tei,
                                      dest_mask=dest_mask,
                                      req_vec=req_vec)
                        # 保证 [1, N, H]
                        if out.size(0) == n_nodes:
                            state_emb = out.unsqueeze(0)
                        else:
                            state_emb = out.mean(0, keepdim=True).unsqueeze(0).expand(1, n_nodes, -1)
            except Exception as _e:
                logger.debug(f"[EncodeState] encoder 失败: {_e}")

        if state_emb is None:
            # fallback: 用 agent 自带的 embedding 方法
            try:
                raw = self.low_agent._extract_state_embedding(low_state).to(device)
                n_nodes = getattr(self.env, 'n', 28)
                hidden = getattr(self.low_agent, 'hidden_dim', 128)
                # raw: [1, H] → broadcast 到 [1, N, H]
                state_emb = raw.unsqueeze(1).expand(1, n_nodes, hidden)
            except Exception as _e2:
                logger.debug(f"[EncodeState] fallback 失败: {_e2}")
                n_nodes = getattr(self.env, 'n', 28)
                hidden = getattr(self.low_agent, 'hidden_dim', 128)
                state_emb = torch.zeros(1, n_nodes, hidden, device=device)

        # ── 2. goal_emb：从 low_agent 获取当前子目标 embedding ──
        goal_emb = None
        try:
            goal_dim = getattr(self.low_agent, 'goal_dim', 64)
            if getattr(self, '_ablation_single_dqn', False):
                goal_emb = torch.zeros(1, goal_dim, device=device)
                return state_emb, goal_emb
            # HRLAgent 通常把当前子目标 embedding 存在 current_subgoal_emb
            _g = getattr(self.low_agent, 'current_subgoal_emb', None)
            if _g is not None:
                goal_emb = _g.to(device)
                if goal_emb.dim() == 1:
                    goal_emb = goal_emb.unsqueeze(0)
            else:
                goal_emb = torch.zeros(1, goal_dim, device=device)
        except Exception:
            goal_dim = getattr(self.low_agent, 'goal_dim', 64)
            goal_emb = torch.zeros(1, goal_dim, device=device)

        return state_emb, goal_emb

    def _get_high_graph_emb(self, high_obs):
        """
        Task D 辅助：从 high_obs（PyG Data）提取全图嵌入 [1, H]，
        供 score_goal_candidates() 作为 state context 使用。
        """
        import torch as _torch
        device = getattr(self.high_agent, 'device',
                         _torch.device('cuda' if _torch.cuda.is_available() else 'cpu'))
        encoder = getattr(self.high_agent, 'encoder', None)
        if encoder is not None:
            try:
                s = high_obs[0] if isinstance(high_obs, tuple) else high_obs
                if hasattr(s, 'x') and hasattr(s, 'edge_index'):
                    ei = s.edge_index.to(device)
                    ea = getattr(s, 'edge_attr', None)
                    if ea is None:
                        ea = _torch.zeros(ei.shape[1], 5, device=device)
                    else:
                        ea = ea.to(device)
                        if ea.dim() == 1: ea = ea.unsqueeze(1)
                        if ea.shape[1] < 5:
                            ea = _torch.cat([ea, _torch.zeros(ea.shape[0], 5 - ea.shape[1], device=device)], dim=1)
                    b = _torch.zeros(s.x.size(0), dtype=_torch.long, device=device)
                    # [Fix-TreeEdge] 补传 tree_edge_index，与低层 encoder 调用接口对齐
                    _tei_h = getattr(s, 'tree_edge_index', None)
                    if _tei_h is not None:
                        _tei_h = _tei_h.to(device)
                    _dest_h = getattr(s, 'dest_mask', None)
                    if _dest_h is not None:
                        _dest_h = _dest_h.to(device)
                    _req_h = getattr(s, 'req_vec', None)
                    if _req_h is not None:
                        _req_h = _req_h.to(device)
                    elif getattr(encoder, 'req_fc', None) is not None:
                        req = getattr(self.env, 'current_request', None)
                        if req:
                            bw = float(req.get('bw_origin', req.get('bw', 0.0)))
                            cpu_list = req.get('cpu_origin', req.get('cpu', []))
                            mem_list = req.get('memory_origin', req.get('memory', []))
                            avg_cpu = float(np.mean(cpu_list)) if len(cpu_list) > 0 else 0.0
                            avg_mem = float(np.mean(mem_list)) if len(mem_list) > 0 else 0.0
                            _req_h = _torch.tensor([[bw, avg_cpu, avg_mem]], dtype=_torch.float32, device=device)
                    try:
                        out = encoder(s.x.to(device), ei, ea, batch=b,
                                      tree_edge_index=_tei_h,
                                      dest_mask=_dest_h,
                                      req_vec=_req_h)
                    except TypeError:
                        # encoder 不接受 tree_edge_index 参数时降级
                        out = encoder(s.x.to(device), ei, ea, batch=b)
                    return out.mean(0, keepdim=True)  # [1, H]
            except Exception:
                pass
        try:
            raw = self.high_agent._extract_state_embedding(high_obs).to(device)
            return raw
        except Exception:
            hidden = getattr(self.high_agent, 'hidden_dim', 128)
            return _torch.zeros(1, hidden, device=device)

    def _fallback_success_check(self):
        if not hasattr(self.env, 'current_request') or not self.env.current_request:
            return True
        try:
            vnf_list = self.env.current_request.get('vnf', [])
            dest_list = self.env.current_request.get('dest', [])

            if hasattr(self.env, 'next_vnf_idx'):
                vnf_success = self.env.next_vnf_idx >= len(vnf_list)
            elif hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'next_vnf_idx'):
                vnf_success = self.env.resource_mgr.next_vnf_idx >= len(vnf_list)
            else:
                vnf_success = False

            if hasattr(self.env, 'current_tree') and self.env.current_tree:
                connected_dests = self.shared.get_connected_dests_view()
                dest_success = len(connected_dests) >= len(dest_list)
            else:
                dest_success = False

            return vnf_success and dest_success
        except:
            return False

    def _store_transition(self, agent, state, action, reward, next_state, done):
        if hasattr(agent, 'store_transition'):
            agent.store_transition(state, action, reward, next_state, done)
        elif hasattr(agent, 'memory') and hasattr(agent.memory, 'store'):
            agent.memory.store(state, action, reward, next_state, done)
        elif hasattr(agent, 'store'):
            agent.store(state, action, reward, next_state, done)

    def _update_stats(self, high_done, info):
        if info:
            if info.get('vnf_deployed', False) or info.get('dest_connected', False):
                self.stats['subgoals_ok'] += 1
            elif (
                    info.get('deploy_fail', False)
                    or info.get('timeout', False)
                    or info.get('fail', False)
                    or info.get('reason') in (
                            'bandwidth_exhausted', 'no_progress', 'trapped',
                            'consecutive_timeout', 'resource_insufficient', 'bandwidth_island'
                    )
            ):
                self.stats['subgoals_fail'] += 1
        if high_done:
            self.stats['episodes_completed'] += 1

    def _reset_episode_stats(self):
        self.stats['subgoals_ok'] = 0
        self.stats['subgoals_fail'] = 0

    def get_stats(self):
        return dict(self.stats)
