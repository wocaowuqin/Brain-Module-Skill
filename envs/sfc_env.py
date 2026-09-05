# envs/sfc_env.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
envs/sfc_env.py
====================================
SFC_HIRL_Env - 多播感知 SFC 编排环境（TA-HRL v4）
====================================

【模块定位】
本模块是 TA-HRL 系统的 Gym 环境主类，是高层/低层控制器与物理网络之间的唯一交互接口。
它不包含策略逻辑，只负责：
  - 维护网络状态（current_tree、current_request、phase、位置等）
  - 接收高层/低层动作，委托给对应控制器执行，返回 (next_state, reward, done, info)
  - 管理请求的生命周期（加载、分配、reset）
  - 构建状态特征（低层 21 维向量、高层 PyG 图）

【环境交互流程】
  env.reset()
    └─ 加载下一条请求，初始化 current_tree / phase / 位置等状态变量

  env.step_high_level(action)       # 委托给 HighLevelController
    └─ 解析高层动作为目标节点，更新 phase/subgoal，返回 high_info

  env.step_low_level(action)        # 委托给 LowLevelController
    └─ 执行一步移动/部署，更新位置/树/资源，返回 (state, reward, done, truncated, info)

  env.get_state()                   # 低层状态，21维特征向量
  env.get_high_level_state_graph()  # 高层状态，PyG Data 图对象

【两阶段执行模型】
  vnf_deployment 阶段：
    - current_deployment_target 指定当前要部署的 VNF 所在 DC 节点
    - 低层走到该 DC 节点后触发 _try_deploy()，成功则 next_vnf_idx++
    - 所有 VNF 部署完成后 phase 切换为 destination_connection

  destination_connection 阶段：
    - current_target_node 指定当前要连通的目的地节点
    - 低层从 last_vnf 出发，走到目的地后记录连通
    - 所有目的地连通后触发 _commit_episode_bandwidth()，episode 成功结束

【辅助类】
  SimpleTopologyManager    轻量拓扑工具（get_neighbors / get_node_degree / get_node_betweenness），
                           env 初始化时若无独立 TopologyManager 则用此兜底
  ExpertWrapper            包装 MSFCE_Solver，供 BackupPolicy 使用；
                           find_any_path() 已删除（路径规划已迁移到 compute_bw_aware_path）
  SimpleDataLoader         数据加载兼容层，当前主链直接使用 DataLoader(self.config)

【主要函数索引（SFC_HIRL_Env）】
  reset()                       加载新请求，重置状态
  reset_request()               请求级重置，推进时间槽并触发资源到期释放
  step_high_level()             委托 HighLevelController 执行高层 step
  step_low_level()              委托 LowLevelController 执行低层 step
  set_high_level_goal()         设置子目标（委托 HighLevelController）
  get_high_level_action_mask()  获取高层 action mask（委托 HighLevelController）
  get_high_level_state_graph()  获取高层 PyG 图状态（委托 HighLevelController）
  get_low_level_action_mask()   获取低层 action mask（委托 LowLevelController）
  get_state()                   获取低层状态特征向量（委托 LowLevelController）
  load_dataset()                从 pkl 文件或 phase 名加载请求数据集
  load_requests()               加载并校准请求（自动修正 1-based 索引）
  get_resource_utilization()    当前 CPU 利用率统计
  get_e2e_delay()               端到端时延（链路传播时延 + M/M/1 VNF处理时延）
  _init_infrastructure()        初始化网络拓扑、时延矩阵和资源管理器
  _init_core_modules()          初始化专家系统（MSFCE_Solver + ExpertWrapper）
  _init_rl_components()         初始化 DataLoader / RewardCritic（含 yaml→params 扁平化）
  _init_state_variables()       初始化 episode 状态变量
  _init_gym_spaces()            定义 Gym action/observation space

【与其他模块的依赖关系】
  → HighLevelController    step_high_level / get_high_level_action_mask / get_high_level_state_graph
  → LowLevelController     step_low_level / get_low_level_action_mask / get_state / compute_bw_aware_path
  → AllResourceManager     通过 self.resource_mgr 管理 CPU/MEM/BW 资源
  → RewardCritic           奖励计算（self.reward_critic.get_reward()）
  → DataLoader             请求数据集加载（self.data_loader）
  → TimeSlotManager        驱动请求到期释放和时钟推进
  → SFCToolkit             工具箱（self.tools），供控制器调用
"""
import os
import logging
import random
import time

import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import gymnasium as gym


from envs.modules.AllResourceManager import FusedResourceManager as ResourceManager
from envs.modules.data_loader import DataLoader
from envs.modules.event_handler import EventHandler
from envs.modules.tools import SFCToolkit
from envs.modules.low_level_controller import LowLevelController
from envs.modules.high_level_controller import HighLevelController
from envs.modules.TimeSlotManager import TimeSlotManager
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimpleTopologyManager:
    """
    增强版简化拓扑管理器
    补全 GNN 特征提取所需的度数和介数计算接口
    """

    def __init__(self, topo):
        self.topo = topo
        self.n = topo.shape[0]
        self.original_topo = topo.copy()
        self.degrees = np.sum(self.topo > 0, axis=1)

    def reset(self):
        self.topo = self.original_topo.copy()
        self.degrees = np.sum(self.topo > 0, axis=1)

    def get_neighbors(self, node):
        return np.where(self.topo[node] > 0)[0].tolist()

    def get_node_degree(self, node):
        return float(self.degrees[node])

    def get_node_betweenness(self, node):
        
        return float(self.degrees[node] / max(1, self.n))


class ExpertWrapper:
    """包装 MSFCE_Solver，适配 BackupPolicy"""

    def __init__(self, msfce_solver):
        self.solver = msfce_solver
        self.node_num = getattr(msfce_solver, 'node_num', 28)
        self.DC = getattr(msfce_solver, 'DC', [])

    
    
    


class SimpleDataLoader:
    """
    简化的数据加载器（兼容层）
    当前主链使用 DataLoader(self.config)，此类仅作接口兼容保留。
    """

    def __init__(self, config):
        self.config = config
        self.requests = []
        self.events = []
        self.total_steps = 0
        self.req_map = {}

    def reset(self):
        self.req_map = {r['id']: r for r in self.requests}

    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        import pickle
        success = False
        loaded_requests = []
        if os.path.exists(phase_or_req_file) and phase_or_req_file.endswith('.pkl'):
            try:
                with open(phase_or_req_file, 'rb') as f:
                    loaded_requests = pickle.load(f)
                if hasattr(self, 'data_loader'):
                    self.data_loader.requests = loaded_requests
                    self.data_loader.total_steps = len(loaded_requests)
                success = True
            except Exception as e:
                return False
        else:
            if hasattr(self, 'data_loader'):
                success = self.data_loader.load_dataset(phase_or_req_file)
                loaded_requests = getattr(self.data_loader, 'requests', [])
            else:
                return False
        if success and loaded_requests:
            self.load_requests(loaded_requests)
        return success


class SFC_HIRL_Env(gym.Env):

    def __init__(self, config, use_gnn=True):
        self.config = config
        self.use_gnn = use_gnn

        self._init_infrastructure()
        self._init_core_modules()
        self._init_rl_components()
        self._init_gym_spaces()

        
        self.current_episode = 0
        self.current_step = 0
        self.total_reward = 0.0

        
        self.next_vnf_idx = 0

        
        self.max_subgoal_steps = config.get('max_low_steps', 25)
        self.subgoal_step_count = 0

        
        self.current_subgoal_node = None
        self.last_high_action_idx = None

        
        self.current_phase = None
        self.current_deployment_target = None
        self.current_target_node = None
        self.current_vnf_to_deploy = None

        self._init_state_variables()

        self.branch_states = {}
        self.current_branch_id = None
        self.branch_counter = 0
        self.vnf_deployment_history = {}
        self.step_count = 0

        self.online_mode = self.config.get('environment', {}).get('online_mode', True)
        self.simulation_done = False
        self.slot_queue = []
        self.requests_by_slot = {}
        self.active_requests_by_slot = {}
        self.leave_heap = []

        logger.info(f"ok 环境基础参数: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")
        logger.info(f"ok HRL 控制参数: Max Subgoal Steps={self.max_subgoal_steps}")

        visualization_config = self.config.get('visualization', {})
        self.enable_visualization = bool(
            visualization_config.get('enable_render', False)
        )
        visualization_root = os.path.expanduser(str(
            visualization_config.get(
                'output_dir', 'artifacts/runs/hrl/visualization'
            )
        ))
        if not os.path.isabs(visualization_root):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            visualization_root = os.path.join(project_root, visualization_root)
        self.visualization_dir = os.path.normpath(visualization_root)
        if self.enable_visualization:
            try:
                os.makedirs(os.path.join(self.visualization_dir, 'success'), exist_ok=True)
                os.makedirs(os.path.join(self.visualization_dir, 'fail'), exist_ok=True)
                logger.info("ok 可视化目录已就绪: %s", self.visualization_dir)
            except Exception as e:
                logger.warning(f"   可视化初始化部分失败: {e}")
                self.enable_visualization = False

        self.tools = SFCToolkit(self)
        logger.info("ok 工具箱 已初始化")

        self.time_slot_mgr = TimeSlotManager(self, self.config)
        logger.info("ok TimeSlotManager 已初始化")

        self.low_level_controller = LowLevelController(self)
        logger.info("ok LowLevelController 已初始化")

        self.high_level_controller = HighLevelController(self)
        logger.info("ok HighLevelController 已初始化")

    def _init_infrastructure(self):
        topo = self.config.get('topology', {}).get('matrix')
        if topo is None:
            n = self.config.get('environment', {}).get('num_nodes', 28)
            topo = np.ones((n, n), dtype=np.float32)
            np.fill_diagonal(topo, 0)
        self.topo = np.asarray(topo, dtype=np.float32)
        self.n = self.topo.shape[0]
        self.K_vnf = self.config.get('vnf', {}).get('n_types', 8)
        self.L = int(np.sum(self.topo > 0))

        capacities = self.config.get('capacities', {'cpu': 100.0, 'memory': 80.0, 'bandwidth': 100.0})
        raw_dc_nodes = self.config.get('topology', {}).get('dc_nodes', list(range(10)))
        self.dc_nodes = [x - 1 for x in raw_dc_nodes]
        print(f"ok [Index Fix] DC Nodes converted: {self.dc_nodes}")

        self.resource_mgr = ResourceManager(self.topo, capacities, self.dc_nodes)
        self.resource_mgr.env = self
        self.request_manager = self.resource_mgr.request_manager
        self.topology_mgr = SimpleTopologyManager(self.topo)

        
        rng = np.random.default_rng(seed=42)
        self.delay_matrix = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.topo[i, j] > 0:
                    d = float(rng.uniform(2.0, 4.0))
                    self.delay_matrix[i, j] = d
                    self.delay_matrix[j, i] = d
        logger.info(f"ok 链路时延矩阵已初始化: "
                    f"min={self.delay_matrix[self.delay_matrix>0].min():.2f}ms "
                    f"max={self.delay_matrix[self.delay_matrix>0].max():.2f}ms")
        logger.info(f"ok 环境参数: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")

        import torch
        if hasattr(self.resource_mgr, 'edge_index'):
            self.edge_index = torch.from_numpy(self.resource_mgr.edge_index).long()
        self.edge_attr = None

    def _init_core_modules(self):
        try:
            from core.expert.expert_msfce.core.solver import MSFCE_Solver
            from core.expert.expert_msfce.core.solver import SolverConfig
            topology_name = self.config.get('topo', self.config.get('env', {}).get('topo'))
            path_db_names = {
                'us_backbone': 'US_Backbone_path.mat',
                '13node': 'path_db_13node.mat',
                '23node': 'path_db_23node.mat',
                '50node': 'path_db_50node.mat',
            }
            if topology_name not in path_db_names:
                raise FileNotFoundError(f"No expert path database configured for topology {topology_name!r}")
            path_db_file = Path(__file__).resolve().parents[1] / "topo" / path_db_names[topology_name]
            capacities = self.config.get('capacities', {})
            msfce_solver = MSFCE_Solver(
                path_db_file=path_db_file,
                topology_matrix=self.topo,
                dc_nodes=self.dc_nodes,
                capacities=capacities,
                config=SolverConfig()
            )
            self.expert = ExpertWrapper(msfce_solver)
        except (ImportError, FileNotFoundError) as e:
            logger.error(f"     无法初始化专家模块: {e}")
            self.expert = None

    def _init_rl_components(self):
        self.data_loader = DataLoader(self.config)
        self.event_handler = EventHandler(resource_manager=self.resource_mgr)

        from core.reward.reward_critic import RewardCritic

        def _flatten_reward_cfg(cfg: dict) -> dict:
            vnf     = cfg.get('vnf',     {})
            tree    = cfg.get('tree',    {})
            penalty = cfg.get('penalty', {})
            timeout = cfg.get('timeout', {})
            return {
                'vnf_deploy_success':    vnf.get('deploy_success',     100.0),
                'vnf_all_complete':      vnf.get('all_complete',        100.0),
                'vnf_deploy_failed':     vnf.get('deploy_failed',      -100.0),
                'move_to_dc_bonus':      vnf.get('move_to_dc_bonus',     15.0),
                'move_cost':             vnf.get('move_cost',             -1.0),
                'connection_base':            tree.get('connection_base',            50.0),
                'connection_exponential':     tree.get('connection_exponential',      1.0),
                'connection_progress_bonus':  tree.get('connection_progress_bonus',  50.0),
                'dest_reached_bonus':         tree.get('dest_reached_bonus',          15.0),
                'full_completion_bonus':      tree.get('full_completion_bonus',      100.0),
                'guidance_closer_rate':       tree.get('guidance_closer_rate',        15.0),
                'guidance_farther_penalty':   tree.get('guidance_farther_penalty',   -10.0),
                'guidance_idle_penalty':      tree.get('guidance_idle_penalty',       -3.0),
                'invalid_link':           penalty.get('invalid_link',           -30.0),
                'invalid_action':         penalty.get('invalid_action',         -50.0),
                'wrong_position':         penalty.get('wrong_position',         -20.0),
                'freq_penalty_threshold': penalty.get('freq_penalty_threshold',     3),
                'freq_penalty_rate':      penalty.get('freq_penalty_rate',       -10.0),
                'timeout_high_progress_threshold': timeout.get('high_progress_threshold', 0.9),
                'timeout_bonus_rate':              timeout.get('bonus_rate',               30.0),
                'timeout_penalty_rate':    abs(timeout.get('penalty_rate',              300.0)),
                'vnf_timeout_penalty':         timeout.get('vnf_timeout_penalty',     -200.0),
                'internal_error':          cfg.get('internal_error',           -10.0),
                'bandwidth_island_retry':  cfg.get('bandwidth_island_retry',   -10.0),
                'hotspot_penalty_rate':    cfg.get('hotspot_penalty_rate',      -2.0),
                'new_edge_cost':           tree.get('new_edge_cost',            -3.0),
                'reward_scale': cfg.get('reward_scale', 1.0),
            }

        raw_reward_cfg = self.config.get('reward', {})
        flat_reward_params = _flatten_reward_cfg(raw_reward_cfg)
        self.reward_critic = RewardCritic(training_phase=3, params=flat_reward_params)
        logger.info(f"ok RewardCritic 参数已扁平化: {sorted(flat_reward_params.keys())}")

    def _init_state_variables(self):
        self.step_counter = 0
        self.total_reward = 0
        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self.node_visit_counts = {}
        self.current_node_location = 0
        self.current_vnf_index = 0
        self.nodes_on_tree = set()

        env_config = self.config.get('environment', {})
        self.nb_high_level_goals = env_config.get('nb_high_level_goals', 10)
        self.NB_LOW_LEVEL_ACTIONS = self.n
        self._n_actions = self.n

        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set(),
            'tree_usage': {},
            'node_stage': {},
            'reused_vnf_count': 0,
        }
        self.current_request = None
        self._prev_dist = None
        self.failed_deploy_attempts = set()

        self.curr_ep_node_allocs = []
        self.curr_ep_link_allocs = []
        self._current_req_record = {}

        self.chain_nodes = []
        self.sfc_upstream_nodes = set()
        self._need_reset_to_last_vnf = False

        self.current_sfc = {
            'chain_nodes': [],
            'spine_paths': [],
            'branch_paths': {}
        }

        self.branch_states = {}
        self.current_branch_id = None
        self.branch_counter = 0

        self.online_mode = self.config.get('environment', {}).get('online_mode', True)
        self.simulation_done = False
        self.current_slot_index = 0
        self.slot_queue = []
        self.all_requests = []
        self.requests_by_slot = {}
        self.max_slot_index = 0
        self.active_requests_by_slot = {}
        self.leave_heap = []

        self.delta_t = self.config.get('data_generation', {}).get('time_slot_delta', 0.01)
        self.processing_delay = 0.0 if self.online_mode else 0.002
        self.time_step = 0.0
        self.current_time_slot = 0
        self.decision_step = 0

        dynamic_cfg = self.config.get('dynamic_env', {})
        self.dynamic_env = dynamic_cfg.get('enabled', True)

        self.global_request_index = 0
        self._request_index = 0
        self.served_dest_count = 0

        p3_cfg = self.config.get('phase3', {})
        env_cfg = self.config.get('env', {})
        self.max_steps = p3_cfg.get('max_steps_per_episode', env_cfg.get('max_steps', 1000))

    def _init_gym_spaces(self):
        if self.use_gnn:
            try:
                from core.gnn.feature_builder import GNNFeatureBuilder
                self.feature_builder = GNNFeatureBuilder(self.config)
            except Exception as e:
                logger.warning(f"   FeatureBuilder 初始化失败: {e}")
                self.feature_builder = None
        else:
            self.feature_builder = None

        self.observation_space = gym.spaces.Dict({
            'x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n, 17), dtype=np.float32),
            'edge_index': gym.spaces.Box(low=0, high=self.n, shape=(2, self.n * self.n), dtype=np.int64),
        })
        self.action_space = gym.spaces.Discrete(self.n)

    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        import pickle
        success = False
        loaded_requests = []
        self.dataset_file = phase_or_req_file
        self.dataset_dir = os.path.dirname(os.path.abspath(phase_or_req_file)) if isinstance(phase_or_req_file, str) else ""

        if os.path.exists(phase_or_req_file) and phase_or_req_file.endswith('.pkl'):
            print(f" [Env] 检测到直接文件路径: {phase_or_req_file}")
            try:
                with open(phase_or_req_file, 'rb') as f:
                    loaded_requests = pickle.load(f)
                if hasattr(self, 'data_loader'):
                    self.data_loader.requests = loaded_requests
                    self.data_loader.total_steps = len(loaded_requests)
                success = True
                print(f"ok [Env] 直接文件加载成功: {len(loaded_requests)} 条")
            except Exception as e:
                print(f"     [Env] 直接文件加载失败: {e}")
                return False
        else:
            print(f"  [Env] 调用 data_loader 加载 phase: {phase_or_req_file}")
            if hasattr(self, 'data_loader'):
                success = self.data_loader.load_dataset(phase_or_req_file)
                loaded_requests = getattr(self.data_loader, 'requests', [])
            else:
                print("     [Env] data_loader 未初始化")
                return False

        if success and loaded_requests:
            print(f"  [Env] 正在构建在线仿真索引 (Requests: {len(loaded_requests)})...")
            self.load_requests(loaded_requests)
        else:
            print("   [Env] 数据加载报告成功，但请求列表为空！")

        return success

    def load_requests(self, requests, requests_by_slot=None):
        if not requests:
            print("   [Env] 请求列表为空")
            return

        def _scan_request_bounds(req_list):
            max_node = 0
            invalid_nodes = []
            max_vnf = 0
            for idx, req in enumerate(req_list):
                src = int(req.get('source', 0))
                dests = [int(d) for d in req.get('dest', [])]
                vnfs = [int(v) for v in req.get('vnf', [])]
                node_ids = [src] + dests
                if node_ids:
                    max_node = max(max_node, max(node_ids))
                for node_id in node_ids:
                    if node_id < 0 or node_id >= self.n:
                        invalid_nodes.append((idx, node_id))
                        if len(invalid_nodes) >= 8:
                            break
                if vnfs:
                    max_vnf = max(max_vnf, max(vnfs))
            return max_node, max_vnf, invalid_nodes

        max_node_in_reqs = 0
        max_vnf_type = 0
        for r in requests:
            s = r.get('source', 0)
            dests = r.get('dest', [])
            vnfs = r.get('vnf', [])
            curr_max_node = max(s, max(dests) if dests else 0)
            max_node_in_reqs = max(max_node_in_reqs, curr_max_node)
            if vnfs:
                max_vnf_type = max(max_vnf_type, max(vnfs))

        print(f"      [数据检查] 请求中最大节点ID: {max_node_in_reqs} (环境 N={self.n})")
        print(f"      [数据检查] 请求中最大VNF类型: {max_vnf_type} (环境 K={self.K_vnf})")

        if max_node_in_reqs >= self.n:
            print(f"       [自动修复] 执行 1-based -> 0-based 节点索引转换...")
            for r in requests:
                r['source'] = r['source'] - 1
                r['dest'] = [d - 1 for d in r['dest']]
                if r['source'] < 0 or r['source'] >= self.n:
                    r['source'] = 0

        max_node_after, _, invalid_nodes = _scan_request_bounds(requests)
        if invalid_nodes:
            sample = invalid_nodes[:5]
            hint = ""
            if self.n == 28 and max_node_after >= 28:
                hint = " This looks like a 50-node dataset loaded with the 28-node us_backbone topology. Use --topo 50node and the matching path database."
            raise ValueError(
                f"Request/topology node-id mismatch: topology has n={self.n}, "
                f"max request node id after index normalization is {max_node_after}, "
                f"invalid samples={sample}.{hint}"
            )

        if max_vnf_type >= self.K_vnf:
            print(f"       [自动修复] 执行 1-based -> 0-based VNF 索引转换...")
            for r in requests:
                r['vnf'] = [v - 1 for v in r['vnf']]

        self.all_requests = requests
        self.global_request_index = 0

        if hasattr(self, 'data_loader'):
            self.data_loader.requests = requests
            self.data_loader.total_steps = len(requests)
            if hasattr(self.data_loader, 'reset'):
                self.data_loader.reset()

        requests_by_slot = {}
        for req in requests:
            arr_time = float(req.get('arrival_time', 0))
            slot = req.get('time_slot', int(arr_time / self.delta_t))
            if slot not in requests_by_slot:
                requests_by_slot[slot] = []
            requests_by_slot[slot].append(req)

        self.requests_by_slot = requests_by_slot
        self.max_slot_index = max(requests_by_slot.keys()) if requests_by_slot else 0
        logger.info(f"ok 数据加载完成 (已校准): {len(requests)} 条")

        if self.online_mode:
            self.current_slot_index = 0
            self.slot_queue = []
            self.simulation_done = False

        if hasattr(self, 'time_slot_mgr') and self.time_slot_mgr is not None:
            self.time_slot_mgr.load(requests)

    

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if hasattr(self, 'action_space'):
                self.action_space.seed(seed)

        self.current_episode += 1
        self.current_step = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.next_vnf_idx = 0
        self.current_phase = None
        self.current_deployment_target = None
        self.current_vnf_to_deploy = None
        self.current_target_node = None
        self.subgoal_step_count = 0
        self.last_high_action_idx = None
        self.current_subgoal_node = None

        options = options or {}
        self._node_visit_count = {}
        self._recent_positions = []
        self._vnf_complete_steps = 0

        if hasattr(self, 'resource_mgr') and self.resource_mgr:
            self.resource_mgr.reset()

        self.nodes_on_tree = set()
        self.current_tree = {
            'tree': {},
            'placement': {},
            'connected_dests': set(),
            'hvt': np.zeros((self.n, self.K_vnf)) if hasattr(self, 'K_vnf') else np.zeros((self.n, 10)),
            'tree_usage': {},
            'node_stage': {},
            'reused_vnf_count': 0,
        }
        self.current_placements = {}

        
        
        self.current_anchor_node = None
        self._dest_cycle_armed = False
        self._dest_step0_checked = False
        self._dest_anchor_mismatch = False
        self._last_dest_target = None
        self._deadlock_step_count = 0
        self._timeout_count = 0
        self._consecutive_timeout_count = 0
        self._bw_already_rolled_back = False
        self._bw_fail_count = 0
        self._island_count = 0
        self._stuck_steps = 0
        self._last_mask_alive_count = 99
        self._last_mask_topk_count = 0
        self._last_anchor_pool_size = 0
        self._last_dist_to_target = None
        self._last_target_goal = None
        self._last_effective_anchor = None
        self._is_first_dest_cycle = True
        self._failed_anchors_for_target = {}
        self._first_dest_steps = -1
        self._first_dest_new_edges = -1
        self._first_dest_reuse_edges = -1
        self._first_dest_detour_ratio = 1.0
        self._first_dest_shortest_hop = 1

        self.branch_states = {}
        self.current_branch_id = None
        self.curr_ep_node_allocs = []
        self.curr_ep_link_allocs = []
        self.chain_nodes = []
        self.sfc_upstream_nodes = set()
        self._need_reset_to_last_vnf = False
        self.current_sfc = {'chain_nodes': [], 'spine_paths': [], 'branch_paths': {}}

        if self.online_mode:
            req_raw = self.time_slot_mgr.get_next_request()
        else:
            req_raw, _ = self.reset_request()

        if req_raw is not None:
            if hasattr(req_raw, 'to_dict'):
                req = req_raw.to_dict()
            elif hasattr(req_raw, '__dict__') and not isinstance(req_raw, dict):
                req = req_raw.__dict__
            else:
                req = req_raw
        else:
            req = None

        if req is None and self.online_mode:
            if options.get('hard_reset', False):
                logger.warning("   [Reset] hard_reset后仍无请求，继续使用空请求")
            else:
                logger.info("  [Reset] 无请求，尝试hard_reset")
                return self.reset(seed, options={'hard_reset': True})

        self.current_request = req

        if req:
            self.current_node_location = req.get('source', 0)
            # Initialize the phase before the coordinator asks for the first
            # high-level mask.  Leaving this as None makes the first cycle
            # ambiguous: the mask builder evaluates VNF placement while the
            # coordinator diagnostics still report phase=None, and a valid
            # request can be rejected as having no high-level action.
            vnf_list = req.get('vnf', []) or []
            if self.next_vnf_idx < len(vnf_list):
                self.current_phase = 'vnf_deployment'
                self.current_deployment_target = None
                self.current_target_node = None
            else:
                self.current_phase = 'destination_connection'
                self.current_deployment_target = None
                self.current_target_node = None
        else:
            logger.warning("   [Reset] 没有可用的请求，使用默认配置")
            self.current_node_location = 0

        import torch
        if hasattr(self.resource_mgr, 'edge_index'):
            self.edge_index = torch.from_numpy(self.resource_mgr.edge_index).long()
        if hasattr(self.resource_mgr, 'build_edge_attr'):
            self.edge_attr = self.resource_mgr.build_edge_attr(
                current_tree=getattr(self, 'current_tree', None))
        else:
            self.edge_attr = None

        initial_state = self.get_state()
        info = {
            'request': req,
            'action_mask': self.get_low_level_action_mask(),
            'decision_steps': 0,
        }
        return initial_state, info

    def reset_request(self):
        if not hasattr(self, 'all_requests') or not self.all_requests:
            return None, self.get_state()

        if not hasattr(self, 'global_request_index'):
            self.global_request_index = 0

        if self.global_request_index >= len(self.all_requests):
            if not hasattr(self, '_request_cycle_offset'):
                self._request_cycle_offset = 0.0
            if self.all_requests:
                cycle_end = max(
                    float(r.get('arrival_time', 0.0)) + float(r.get('lifetime', 5.0))
                    for r in self.all_requests
                ) * 1.10
            else:
                cycle_end = 600.0
            self._request_cycle_offset += cycle_end
            logger.info(f"[ResetReq] 数据集循环，时间偏移累计 +{cycle_end:.1f}s → "
                        f"总偏移={self._request_cycle_offset:.1f}s")
            if hasattr(self.resource_mgr, 'request_manager'):
                self.resource_mgr.request_manager.check_and_release_expired(
                    self._request_cycle_offset)
            self.global_request_index = 0

        req = dict(self.all_requests[self.global_request_index])

        offset = getattr(self, '_request_cycle_offset', 0.0)
        if offset > 0:
            raw_arrival = float(req.get('arrival_time', 0.0))
            raw_lifetime = float(req.get('lifetime', 5.0))
            req['arrival_time'] = raw_arrival + offset
            req['expire_time'] = raw_arrival + offset + raw_lifetime

        new_arrival_time = float(req.get('arrival_time', self.time_step))
        new_time_slot = req.get('time_slot', 0)
        old_time_slot = getattr(self, 'current_time_slot', None)

        if not hasattr(self, 'current_time_slot'):
            self.current_time_slot = new_time_slot
            old_time_slot = new_time_slot

        if old_time_slot is not None and new_time_slot != old_time_slot:
            self.time_step = new_arrival_time
            self.current_time_slot = new_time_slot
            if hasattr(self.resource_mgr, 'request_manager'):
                self.resource_mgr.request_manager.check_and_release_expired(self.time_step)
        else:
            self.time_step = new_arrival_time
            self.current_time_slot = new_time_slot

        self.global_request_index += 1
        return req, self.get_state()

    

    def set_high_level_goal(self, high_action_idx, target_node_id, start_node_id=None):
        """    [委托] 设定高层目标"""
        return self.high_level_controller.set_high_level_goal(
            high_action_idx, target_node_id, start_node_id=start_node_id)

    def step_high_level(self, action):
        """    [委托] 执行高层步骤"""
        return self.high_level_controller.step_high_level(action)

    def get_high_level_action_mask(self):
        """    [委托] 获取高层动作掩码"""
        return self.high_level_controller.get_high_level_action_mask()

    def get_high_level_state_graph(self):
        """    [委托] 获取高层图状态"""
        return self.high_level_controller.get_high_level_state_graph()

    

    def step_low_level(self, action):
        """ [委托] 低层步进函数"""
        return self.low_level_controller.step_low_level(action)

    def get_low_level_action_mask(self):
        """获取低层动作掩码 - 委托给 LowLevelController"""
        if hasattr(self, 'low_level_controller'):
            return self.low_level_controller.get_low_level_action_mask()
        return np.ones(self.n, dtype=np.float32)

    def get_state(self):
        """获取环境状态 - 委托给 LowLevelController，并保证 req_vec 字段存在"""
        import torch

        if hasattr(self, 'low_level_controller'):
            state = self.low_level_controller.get_state()
        else:
            from torch_geometric.data import Data
            state = Data(x=torch.zeros((self.n, 17)))

        
        
        
        _REQ_VEC_CANDIDATES = (
            "req_vec", "request_vec", "req_feat",
            "request_feat", "req_feature", "request_feature", "req",
        )

        def _has_key(s, k):
            if isinstance(s, dict):
                return k in s and s[k] is not None
            return hasattr(s, k) and getattr(s, k) is not None

        def _set_key(s, k, v):
            if isinstance(s, dict):
                s[k] = v
            else:
                setattr(s, k, v)

        has_req_vec = any(_has_key(state, k) for k in _REQ_VEC_CANDIDATES)

        if not has_req_vec:
            
            req_obj = getattr(self, "current_request", None) or {}
            n = self.n
            bw_cap = getattr(self, "B_cap", getattr(self, "cap_bw", 90.0))
            cpu_cap = getattr(self, "C_cap", getattr(self, "cap_cpu", 80.0))
            src = req_obj.get("source", 0)
            dests = req_obj.get("dest", [])
            bw = float(req_obj.get("bw_origin", req_obj.get("bandwidth", 0.0)))
            cpus = req_obj.get("cpu_origin", req_obj.get("cpu", []))
            mems = req_obj.get("memory_origin", req_obj.get("memory", []))
            vnfs = req_obj.get("vnf", [])
            avg_cpu = float(sum(cpus)) / max(len(cpus), 1) if cpus else 0.0
            avg_mem = float(sum(mems)) / max(len(mems), 1) if mems else 0.0
            
            
            
            req_vec = torch.tensor([[
                bw / max(bw_cap, 1.0),
                avg_cpu / max(cpu_cap, 1.0),
                avg_mem / max(cpu_cap, 1.0),
            ]], dtype=torch.float32)
            _set_key(state, "req_vec", req_vec)

        elif not _has_key(state, "req_vec"):
            
            for k in _REQ_VEC_CANDIDATES:
                if _has_key(state, k):
                    val = state[k] if isinstance(state, dict) else getattr(state, k)
                    _set_key(state, "req_vec", val)
                    break

        
        if not _has_key(state, "edge_attr"):
            edge_index = None
            if _has_key(state, "edge_index"):
                edge_index = state["edge_index"] if isinstance(state, dict) else state.edge_index
            num_edges = edge_index.shape[1] if (edge_index is not None and edge_index.dim() == 2) else 0
            _set_key(state, "edge_attr", torch.zeros(num_edges, 5, dtype=torch.float32))

        return state

    

    def get_resource_utilization(self):
        """资源利用率 — 基于 pool 实际数据"""
        try:
            total_used = 0.0
            for i in range(self.n):
                avail = self.resource_mgr.pool.get_available_cpu(i)
                total_used += (100.0 - avail)
            return total_used / (self.n * 100.0)
        except Exception:
            return 0.0

    def get_e2e_delay(self):
        """
        端到端时延 = 链路传播时延 + VNF 处理时延（M/M/1 模型）
        返回：(e2e_total_ms, link_delay_ms, vnf_delay_ms)
        """
        try:
            tree_edges = self.current_tree.get('tree', {})
            placement = self.current_tree.get('placement', {})
            request = self.current_request or {}

            link_delay = 0.0
            if tree_edges and hasattr(self, 'delay_matrix'):
                delays = [float(self.delay_matrix[u, v])
                          for (u, v), _ in tree_edges.items()
                          if 0 <= u < self.n and 0 <= v < self.n]
                link_delay = float(np.sum(delays)) if delays else 0.0

            lam = 0.05
            vnf_delay = 0.0
            cpu_reqs = request.get('cpu_origin', [])
            for key, alloc in placement.items():
                if not (isinstance(key, tuple) and len(key) >= 2):
                    continue
                node, vnf_idx = key[0], key[1]
                r_fi = cpu_reqs[vnf_idx] if vnf_idx < len(cpu_reqs) else alloc.get('cpu_used', 1.0)
                r_fi = max(float(r_fi), 1e-5)
                try:
                    c_avail = self.resource_mgr.pool.get_available_cpu(node)
                except Exception:
                    c_avail = 80.0
                c_avail = max(float(c_avail), r_fi)
                mu = c_avail / r_fi
                vnf_delay += 1.0 / max(mu - lam, 1e-5)

            return float(link_delay + vnf_delay), float(link_delay), float(vnf_delay)
        except Exception as ex:
            logger.debug(f"get_e2e_delay 异常: {ex}")
            return 0.0, 0.0, 0.0
