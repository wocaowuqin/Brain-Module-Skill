
import copy
import os
import pickle
import logging
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np
from torch_geometric.data import Data
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class Phase1ExpertCollector:
    """
    Phase 1 专家数据收集器（时间槽版本 - 最终修复版）

     修复问题：
    1. max_episodes 现在指的是"成功样本数"而不是"处理的请求数"
    2. 时间槽变化时正确释放过期资源（通过 RequestLifecycleManager）
    3. 自建图状态，不再依赖 resource_mgr.get_graph_state
    4. 修正负载估计方法，避免使用不存在的 B/C 属性
    5. **构造准确的资源分配记录**：从 vnf_instances 中提取 CPU/MEM 用量，确保节点资源被正确释放
    """

    def __init__(self, env, expert_solver, output_dir: str, max_episodes: int = 15000,
                 save_every: int = 500, use_timeslot: bool = True):
        self.env = env
        self.expert = expert_solver
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_success_samples = max_episodes
        self.save_every = save_every

        self.success_samples = []
        self.fail_contexts = []
        self.stats = {
            "requests": 0,
            "success_requests": 0,
            "failed_requests": 0,
            "paths_collected": 0,
            "vnf_deploy_samples": 0,
            "dest_select_samples": 0,
            "path_step_samples": 0,
        }

        self.use_timeslot = use_timeslot
        self.timeslot_stats = {
            'total_time_slots': 0,
            'requests_per_slot': [],
            'current_time_slot': 0
        }

    def _estimate_load(self):
        """基于 resource_mgr.pool 计算平均 CPU 和带宽利用率"""
        rm = self.env.resource_mgr
        n = rm.n
        L = rm.pool.L

        total_cpu_avail = 0.0
        for i in range(n):
            total_cpu_avail += rm.pool.get_available_cpu(i)
        cpu_util = 1.0 - total_cpu_avail / (n * max(rm.C_cap, 1.0))

        total_bw_avail = 0.0
        for key in rm.pool.link_map.keys():
            total_bw_avail += rm.pool.get_available_bandwidth(*key)
        bw_util = 1.0 - total_bw_avail / (L * max(rm.B_cap, 1.0))

        return 0.5 * bw_util + 0.5 * cpu_util

    def _sanitize_request(self, req):
        if isinstance(req, dict):
            return req.copy()
        if hasattr(req, '__dict__'):
            return req.__dict__.copy()
        try:
            return dict(req)
        except:
            return {
                'id': getattr(req, 'id', -1),
                'source': getattr(req, 'source', 0),
                'dest': getattr(req, 'dest', []),
                'vnf': getattr(req, 'vnf', []),
                'bandwidth': getattr(req, 'bandwidth', 1.0),
                'ttl': getattr(req, 'ttl', 100),
                'time_slot': getattr(req, 'time_slot', 0),
                'duration': getattr(req, 'duration', 100),
                'leave_time_slot': getattr(req, 'leave_time_slot', 100)
            }

    def _convert_request_indices(self, raw_req):
        req = self._sanitize_request(raw_req)

        src = req.get("source", 0)
        if isinstance(src, (list, np.ndarray)):
            src = src.item()
        if src > 0:
            src = src - 1
        req['source'] = int(src)

        new_dests = []
        raw_dests = req.get("dest", [])
        if hasattr(raw_dests, 'flatten'):
            raw_dests = raw_dests.flatten()
        for d in raw_dests:
            d_val = int(d)
            if d_val > 0:
                d_val = d_val - 1
            new_dests.append(d_val)
        req['dest'] = new_dests

        new_vnfs = req.get('vnf', [])
        if hasattr(new_vnfs, 'flatten'):
            new_vnfs = new_vnfs.flatten()
        req['vnf'] = [int(v) for v in new_vnfs]

        if 'bandwidth' not in req or req['bandwidth'] is None:
            req['bandwidth'] = req.get('bw_origin', 3.0)

        
        if 'bw_origin' not in req or req['bw_origin'] is None:
            req['bw_origin'] = req.get('bandwidth', 3.0)

        if 'cpu' not in req or req['cpu'] is None:
            req['cpu'] = req.get('cpu_origin', [1.0] * len(req['vnf']))

        if 'cpu_origin' not in req or req['cpu_origin'] is None:
            req['cpu_origin'] = req.get('cpu', [1.0] * len(req['vnf']))

        if 'memory' not in req or req['memory'] is None:
            req['memory'] = req.get('memory_origin', [1.0] * len(req['vnf']))

        if 'memory_origin' not in req or req['memory_origin'] is None:
            req['memory_origin'] = req.get('memory', [1.0] * len(req['vnf']))

        return req

    def _try_auto_load_timeslot_data(self):
        logger.info(" 尝试自动加载时间槽数据...")
        if hasattr(self.env, 'config'):
            config = self.env.config
            data_dir = Path(config.get('paths', {}).get('input_dir', 'data/input_dir'))
        else:
            data_dir = Path('data/input_dir')

        requests_file = data_dir / 'phase1_requests.pkl'
        requests_by_slot_file = data_dir / 'phase1_requests_by_slot.pkl'

        logger.info(f"   检查文件: {requests_file}")
        logger.info(f"   检查文件: {requests_by_slot_file}")

        if not requests_file.exists() or not requests_by_slot_file.exists():
            return False

        try:
            with open(requests_file, 'rb') as f:
                requests = pickle.load(f)
            with open(requests_by_slot_file, 'rb') as f:
                requests_by_slot = pickle.load(f)

            logger.info(f"    文件加载成功: {len(requests)} 请求, {len(requests_by_slot)} 时间槽")

            if hasattr(self.env, 'load_requests'):
                self.env.load_requests(requests, requests_by_slot)
            else:
                self.env.all_requests = requests
                self.env.requests_by_slot = requests_by_slot

            return True
        except Exception as e:
            logger.error(f"    自动加载失败: {e}")
            return False

    def load_timeslot_data(self):
        if not self.use_timeslot:
            return False
        try:
            if (hasattr(self.env, 'all_requests') and self.env.all_requests and
                    hasattr(self.env, 'requests_by_slot') and self.env.requests_by_slot):
                logger.info(f" 环境已加载时间槽数据: {len(self.env.all_requests)} 请求")
                return True
            else:
                if self._try_auto_load_timeslot_data():
                    return True
                else:
                    self.use_timeslot = False
                    return False
        except Exception as e:
            self.use_timeslot = False
            return False

    def collect(self):
        logger.info(" Starting Phase 1: Expert Data Collection")
        logger.info(f"   目标样本数: {self.max_success_samples}")

        self.load_timeslot_data()
        self.env.reset()

        requests = None
        events = None

        if hasattr(self.env, 'all_requests') and self.env.all_requests:
            requests = self.env.all_requests
        elif hasattr(self.env, 'data_loader'):
            if hasattr(self.env.data_loader, 'requests'):
                requests = self.env.data_loader.requests
            if hasattr(self.env.data_loader, 'events'):
                events = self.env.data_loader.events

        if not requests:
            logger.error(" No requests found!")
            return self.stats

        
        if events is None:
            events = self._try_load_events()

        if events is not None:
            logger.info(f" 使用事件驱动模式（真实负载模拟）")
            return self._collect_from_events(events, requests)
        else:
            logger.warning(" 未找到 events 文件，退化为顺序模式（空载状态，IL质量较低）")
            logger.warning("   建议先运行 Generate_all_events.py 生成 phase1_events.pkl")
            return self._collect_from_requests(requests)

    def _try_load_events(self):
        """从数据目录加载 phase1_events.pkl，支持多路径搜索"""
        import pickle
        from pathlib import Path

        
        candidate_dirs = []
        dataset_dir = getattr(self.env, "dataset_dir", "")
        if dataset_dir:
            candidate_dirs.append(Path(dataset_dir))

        
        if hasattr(self.env, 'config'):
            cfg = self.env.config
            for key in ['input_dir', 'data_dir']:
                v = (cfg.get('paths', {}) or {}).get(key)
                if v:
                    candidate_dirs.append(Path(v))

        if not candidate_dirs:
            for base in [Path('data'), Path('../data'),
                         Path(__file__).parent.parent / 'data']:
                if base.exists():
                    for sub in sorted(base.iterdir()):
                        if sub.is_dir():
                            candidate_dirs.append(sub)

        seen_dirs = set()
        for d in candidate_dirs:
            d = Path(d)
            try:
                d_key = str(d.resolve())
            except Exception:
                d_key = str(d)
            if d_key in seen_dirs:
                continue
            seen_dirs.add(d_key)
            for fname in ['phase1_events.pkl', 'phase1_events_by_slot.pkl']:
                fpath = d / fname
                if not fpath.exists():
                    continue
                try:
                    with open(fpath, 'rb') as f:
                        events = pickle.load(f)
                    if isinstance(events, dict):
                        
                        max_slot = max(events.keys()) if events else 0
                        events_list = []
                        for slot in range(max_slot + 1):
                            slot_evs = events.get(slot, [])
                            arrive_reqs = [e['request'] for e in slot_evs
                                           if e.get('type') == 'arrive']
                            leave_ids   = [e['request'].get('id') for e in slot_evs
                                           if e.get('type') == 'leave']
                            events_list.append({'arrive_requests': arrive_reqs,
                                                'leave_event': leave_ids})
                        events = events_list
                    elif isinstance(events, list) and events and 'type' in events[0]:
                        
                        
                        slot_map = {}
                        for e in events:
                            slot = e.get('time_slot', 0)
                            if slot not in slot_map:
                                slot_map[slot] = {'arrive_requests': [], 'leave_event': []}
                            if e.get('type') == 'arrive':
                                slot_map[slot]['arrive_requests'].append(e['request'])
                            else:
                                slot_map[slot]['leave_event'].append(e['request'].get('id'))
                        max_slot = max(slot_map.keys()) if slot_map else 0
                        events = [slot_map.get(s, {'arrive_requests': [], 'leave_event': []})
                                  for s in range(max_slot + 1)]
                    logger.info(f" 加载事件文件: {fpath} ({len(events)} 个时间槽)")
                    return events
                except Exception as e:
                    logger.warning(f" 加载 {fpath} 失败: {e}")
        return None

    def _collect_from_events(self, events, requests):
        pbar = tqdm(desc="Collecting HRL Data", ncols=120)
        req_map = {r.get('id'): r for r in (requests or []) if r.get('id')}

        for t, event in enumerate(events):
            
            leave_list = event.get("leave", event.get("leave_event", []))
            for leave_req_id in leave_list:
                try:
                    self.env.event_handler.unregister_service(leave_req_id)
                except:
                    pass

            
            
            
            arrive_reqs = event.get("arrive_requests", None)
            if arrive_reqs is not None:
                
                for req in arrive_reqs:
                    self._process_single_request(req, pbar)
                    if len(self.success_samples) >= self.max_success_samples:
                        logger.info(f"\n 达到目标样本数: {len(self.success_samples)}")
                        break
            else:
                
                arrive_list = event.get("arrive", event.get("arrive_event", []))
                for req_id in arrive_list:
                    req = req_map.get(req_id)
                    if req is None:
                        if isinstance(req_id, int) and 0 < req_id <= len(requests or []):
                            req = requests[req_id - 1]
                    if req is None:
                        continue
                    self._process_single_request(req, pbar)
                    if len(self.success_samples) >= self.max_success_samples:
                        logger.info(f"\n 达到目标样本数: {len(self.success_samples)}")
                        break

            if len(self.success_samples) >= self.max_success_samples:
                break

        pbar.close()
        self._save_final()
        return self.stats

    def _collect_from_requests(self, requests):
        pbar = tqdm(desc="Collecting HRL Data (Time Slot)", ncols=120)
        for raw_req in requests:
            self._process_single_request(raw_req, pbar)
            if len(self.success_samples) >= self.max_success_samples:
                logger.info(f"\n 达到目标样本数: {len(self.success_samples)}")
                break
        pbar.close()
        self._save_final()
        return self.stats

    def _build_graph_state(
            self,
            request,
            nodes_on_tree,
            current_tree,
            served_dest_count,
            current_node=None,
            phase='destination_connection',
            target_node=None,
            next_vnf_idx=None,
            chain_nodes=None,
    ):
        """
        构造当前step对应的图状态，而不是整条path共用一个state
        """
        original_request = self.env.current_request
        original_phase = getattr(self.env, "current_phase", None)
        original_node = getattr(self.env, "current_node_location", None)
        original_target = getattr(self.env, "current_target_node", None)
        original_deploy_target = getattr(self.env, "current_deployment_target", None)
        original_tree = self.env.current_tree
        original_next_vnf_idx = getattr(self.env, "next_vnf_idx", None)
        original_chain_nodes = copy.deepcopy(getattr(self.env, "chain_nodes", []))
        original_nodes_on_tree = copy.deepcopy(getattr(self.env, "nodes_on_tree", set()))

        try:
            self.env.current_request = request
            self.env.current_tree = current_tree
            if next_vnf_idx is not None:
                self.env.next_vnf_idx = int(next_vnf_idx)
            if chain_nodes is not None:
                self.env.chain_nodes = [int(v) for v in chain_nodes]
            self.env.nodes_on_tree = set(int(v) for v in nodes_on_tree)

            if current_node is None:
                current_node = request.get("source", 0)

            self.env.current_node_location = int(current_node)
            self.env.current_phase = phase

            if phase == "destination_connection":
                self.env.current_target_node = int(target_node) if target_node is not None else None
                self.env.current_deployment_target = None
            elif phase == "vnf_deployment":
                self.env.current_deployment_target = int(target_node) if target_node is not None else None
                self.env.current_target_node = None
            else:
                self.env.current_target_node = None
                self.env.current_deployment_target = None

            
            # --------------------------------------------------
            state = self.env.get_state()

            
            def _get(s, *keys, default=None):
                """按优先顺序尝试多个键名，兼容 dict / Data 对象"""
                for k in keys:
                    # dict-style
                    if isinstance(s, dict):
                        if k in s and s[k] is not None:
                            return s[k]
                    # Data / object attribute-style
                    elif hasattr(s, k) and getattr(s, k) is not None:
                        return getattr(s, k)
                return default

            import torch

            x = _get(state, "x")
            if x is None:
                raise KeyError("state 中缺少 'x' 字段")

            edge_index = _get(state, "edge_index")
            if edge_index is None:
                raise KeyError("state 中缺少 'edge_index' 字段")

            
            edge_attr = _get(state, "edge_attr")
            if edge_attr is None:
                num_edges = edge_index.shape[1] if edge_index.dim() == 2 else 0
                edge_attr = torch.zeros(num_edges, 5, dtype=torch.float32)

            
            req_vec = _get(
                state,
                "req_vec", "request_vec", "req_feat", "request_feat",
                "req_feature", "request_feature", "req",
            )
            if req_vec is None:
                
                req_obj = request
                n = self.env.n if hasattr(self.env, "n") else 28
                bw_cap = getattr(self.env, "B_cap", getattr(self.env, "cap_bw", 90.0))
                cpu_cap = getattr(self.env, "C_cap", getattr(self.env, "cap_cpu", 80.0))
                src = req_obj.get("source", 0)
                dests = req_obj.get("dest", [])
                bw = req_obj.get("bw_origin", req_obj.get("bandwidth", 0.0))
                cpus = req_obj.get("cpu_origin", req_obj.get("cpu", []))
                mems = req_obj.get("memory_origin", req_obj.get("memory", []))
                vnfs = req_obj.get("vnf", [])
                avg_cpu = float(sum(cpus)) / max(len(cpus), 1) if cpus else 0.0
                avg_mem = float(sum(mems)) / max(len(mems), 1) if mems else 0.0
                req_vec = torch.tensor([[
                    src / max(n, 1),
                    len(dests) / max(n, 1),
                    float(bw) / max(bw_cap, 1.0),
                    avg_cpu / max(cpu_cap, 1.0),
                    avg_mem / max(cpu_cap, 1.0),
                    len(vnfs) / 8.0,
                ]], dtype=torch.float32)

            tree_edge_index = _get(state, "tree_edge_index")
            dest_mask = _get(state, "dest_mask")
            # --------------------------------------------------

            action_mask = None
            try:
                if hasattr(self.env, "low_level_controller"):
                    mask_np = self.env.low_level_controller.get_low_level_action_mask(
                        mutate_env=False
                    )
                    action_mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
            except Exception as e:
                logger.warning(
                    f"构造action_mask失败 req={request.get('id', 'unknown')} "
                    f"current={current_node} phase={phase} target={target_node}: {e}"
                )
                action_mask = None

            return x, edge_index, edge_attr, req_vec, tree_edge_index, dest_mask, action_mask

        finally:
            self.env.current_request = original_request
            self.env.current_phase = original_phase
            self.env.current_node_location = original_node
            self.env.current_target_node = original_target
            self.env.current_deployment_target = original_deploy_target
            self.env.current_tree = original_tree
            if original_next_vnf_idx is not None:
                self.env.next_vnf_idx = original_next_vnf_idx
            self.env.chain_nodes = original_chain_nodes
            self.env.nodes_on_tree = original_nodes_on_tree

    def _extract_vnf_deploy_plan(self, req, traj):
        """
        Build chain-order VNF deployment labels from expert_msfce placement dicts.
        expert_msfce uses 1-based node ids; Phase1 stores 0-based labels.
        """
        vnf_list = [int(v) for v in req.get("vnf", [])]
        plan = [None] * len(vnf_list)
        if not vnf_list:
            return []

        for _, action_data, _ in traj:
            if not isinstance(action_data, dict):
                continue
            placement = action_data.get("placement", {}) or {}
            if not isinstance(placement, dict):
                continue
            for key, node_ext in placement.items():
                try:
                    if isinstance(key, tuple) and len(key) >= 2:
                        vnf_type = int(key[1])
                    else:
                        vnf_type = None
                    node_0 = int(node_ext) - 1
                except Exception:
                    continue
                if vnf_type is None:
                    continue
                for idx, chain_type in enumerate(vnf_list):
                    if plan[idx] is None and int(chain_type) == vnf_type:
                        plan[idx] = node_0
                        break

        cleaned = []
        last_node = int(req.get("source", 0))
        for idx, node in enumerate(plan):
            if node is None:
                continue
            cleaned.append({
                "vnf_idx": int(idx),
                "vnf_type": int(vnf_list[idx]),
                "node": int(node),
                "current_node": int(last_node),
            })
            last_node = int(node)
        return cleaned

    def _make_state_data(self, x, edge_index, edge_attr, req_vec, tree_edge_index, dest_mask, action_mask):
        return Data(
            x=x.cpu(),
            edge_index=edge_index.cpu(),
            edge_attr=edge_attr.cpu(),
            req_vec=req_vec.cpu(),
            tree_edge_index=tree_edge_index.cpu() if tree_edge_index is not None else None,
            dest_mask=dest_mask.cpu() if dest_mask is not None else None,
            action_mask=action_mask.cpu() if action_mask is not None else None,
        )

    def _process_single_request(self, raw_req, pbar):
        """
        把专家结果转成 step-level imitation samples
        核心修复：
          1. 内部调用 expert solver，不再依赖外部传入 expert_result
          2. solver 返回 (tree, traj)，从中提取 traj_paths
          3. 每个step单独构造state
          4. current_node_location 跟着 path 逐步推进
          5. 为每个step保存 low_label / action_mask
        """
        self.stats["requests"] += 1

        
        try:
            req = self._convert_request_indices(raw_req)
        except Exception as e:
            logger.warning(f"请求预处理失败: {e}")
            self.stats["failed_requests"] += 1
            pbar.update(1)
            return

        
        try:
            tree, traj = self.expert.solve_request_for_expert(req)
        except Exception as e:
            logger.warning(f"expert求解失败 req={req.get('id', 'unknown')}: {e}")
            self.stats["failed_requests"] += 1
            pbar.update(1)
            return

        if tree is None or not traj:
            self.stats["failed_requests"] += 1
            pbar.update(1)
            return

        
        
        
        traj_paths = []
        try:
            for dest_idx, action_data, _ in traj:
                path_nodes = action_data.get("path", [])
                target_dest = action_data.get("target_dest",
                                              path_nodes[-1] if path_nodes else None)
                if path_nodes and len(path_nodes) >= 2:
                    
                    path_0 = [n - 1 for n in path_nodes]
                    subgoal_0 = (target_dest - 1) if target_dest is not None else path_0[-1]
                    traj_paths.append((path_0, subgoal_0, subgoal_0))
        except Exception as e:
            logger.warning(f"traj解析失败 req={req.get('id', 'unknown')}: {e}")
            self.stats["failed_requests"] += 1
            pbar.update(1)
            return

        if not traj_paths:
            self.stats["failed_requests"] += 1
            pbar.update(1)
            return

        
        try:
            clean_req = copy.deepcopy(req)
            current_tree_for_state = {
                "tree": {},
                "connected_dests": set(),
                "placement": {},
                "node_stage": {},
            }

            nodes_on_tree_so_far = set([req.get("source", 0)])
            served_dest_count = 0
            dest_list = set(req.get("dest", []))
            chain_nodes_so_far = []

            vnf_deploy_plan = self._extract_vnf_deploy_plan(req, traj)
            for deploy in vnf_deploy_plan:
                deploy_node = int(deploy["node"])
                vnf_idx = int(deploy["vnf_idx"])
                vnf_type = int(deploy["vnf_type"])
                cur_node_for_deploy = int(deploy.get("current_node", req.get("source", 0)))
                local_current_tree = {
                    "tree": copy.deepcopy(current_tree_for_state.get("tree", {})),
                    "connected_dests": set(current_tree_for_state.get("connected_dests", set())),
                    "placement": copy.deepcopy(current_tree_for_state.get("placement", {})),
                    "node_stage": copy.deepcopy(current_tree_for_state.get("node_stage", {})),
                }
                try:
                    x, edge_index, edge_attr, req_vec, tree_edge_index, dest_mask, action_mask = \
                        self._build_graph_state(
                            request=req,
                            nodes_on_tree=nodes_on_tree_so_far,
                            current_tree=local_current_tree,
                            served_dest_count=served_dest_count,
                            current_node=cur_node_for_deploy,
                            phase="vnf_deployment",
                            target_node=None,
                            next_vnf_idx=vnf_idx,
                            chain_nodes=chain_nodes_so_far,
                        )
                    state_to_save = self._make_state_data(
                        x, edge_index, edge_attr, req_vec, tree_edge_index, dest_mask, action_mask
                    )
                    self.success_samples.append({
                        "state": state_to_save,
                        "request": clean_req,
                        "action": {
                            "sample_type": "vnf_deployment",
                            "high_label": deploy_node,
                            "low_label": deploy_node,
                            "current_node": cur_node_for_deploy,
                            "subgoal_node": deploy_node,
                            "vnf_idx": vnf_idx,
                            "vnf_type": vnf_type,
                            "high_only": True,
                            "train_high": True,
                            "train_low": False,
                        },
                        "cost": 0.0,
                        "load": self._estimate_load(),
                        "hrl_info": {
                            "sample_type": "vnf_deployment",
                            "vnf_idx": vnf_idx,
                            "vnf_type": vnf_type,
                            "target_node": deploy_node,
                        },
                    })
                    self.stats["vnf_deploy_samples"] += 1
                except Exception as e:
                    logger.warning(
                        f"vnf deployment state build failed req={req.get('id', 'unknown')} "
                        f"vnf_idx={vnf_idx} node={deploy_node}: {e}"
                    )

                current_tree_for_state["placement"][(deploy_node, vnf_idx)] = {
                    "node": deploy_node,
                    "vnf_idx": vnf_idx,
                    "vnf_type": vnf_type,
                }
                current_tree_for_state["node_stage"][deploy_node] = max(
                    current_tree_for_state["node_stage"].get(deploy_node, 0),
                    vnf_idx + 1,
                )
                chain_nodes_so_far.append(deploy_node)
                nodes_on_tree_so_far.add(deploy_node)

            for path_idx, (path_0, high_label, subgoal_node) in enumerate(traj_paths):
                if not path_0 or len(path_0) < 2:
                    continue

                
                try:
                    local_current_tree_for_high = {
                        "tree": copy.deepcopy(current_tree_for_state.get("tree", {})),
                        "connected_dests": set(current_tree_for_state.get("connected_dests", set())),
                        "placement": copy.deepcopy(current_tree_for_state.get("placement", {})),
                        "node_stage": copy.deepcopy(current_tree_for_state.get("node_stage", {})),
                    }
                    x, edge_index, edge_attr, req_vec, tree_edge_index, dest_mask, action_mask = \
                        self._build_graph_state(
                            request=req,
                            nodes_on_tree=nodes_on_tree_so_far,
                            current_tree=local_current_tree_for_high,
                            served_dest_count=served_dest_count,
                            current_node=int(path_0[0]),
                            phase="destination_connection",
                            target_node=None,
                            next_vnf_idx=len(req.get("vnf", [])),
                            chain_nodes=chain_nodes_so_far,
                        )
                    state_to_save = self._make_state_data(
                        x, edge_index, edge_attr, req_vec, tree_edge_index, dest_mask, action_mask
                    )
                    self.success_samples.append({
                        "state": state_to_save,
                        "request": clean_req,
                        "action": {
                            "sample_type": "dest_selection",
                            "high_label": int(high_label),
                            "low_label": int(path_0[1]),
                            "current_node": int(path_0[0]),
                            "subgoal_node": int(subgoal_node),
                            "path_index": int(path_idx),
                            "high_only": True,
                            "train_high": True,
                            "train_low": False,
                        },
                        "cost": 0.0,
                        "load": self._estimate_load(),
                        "hrl_info": {
                            "sample_type": "dest_selection",
                            "subgoal": int(subgoal_node),
                            "path_index": int(path_idx),
                        },
                    })
                    self.stats["dest_select_samples"] += 1
                except Exception as e:
                    logger.warning(
                        f"dest selection state build failed req={req.get('id', 'unknown')} "
                        f"path_idx={path_idx} target={subgoal_node}: {e}"
                    )

                local_nodes_on_tree = set(nodes_on_tree_so_far)
                local_connected_dests = set(current_tree_for_state.get("connected_dests", set()))
                local_tree = copy.deepcopy(current_tree_for_state.get("tree", {}))

                for step_idx in range(len(path_0) - 1):
                    curr_node = int(path_0[step_idx])
                    next_node = int(path_0[step_idx + 1])

                    local_current_tree = {
                        "tree": local_tree,
                        "connected_dests": local_connected_dests,
                        "placement": copy.deepcopy(current_tree_for_state.get("placement", {})),
                        "node_stage": copy.deepcopy(current_tree_for_state.get("node_stage", {})),
                    }

                    try:
                        x, edge_index, edge_attr, req_vec, tree_edge_index, dest_mask, action_mask = \
                            self._build_graph_state(
                                request=req,
                                nodes_on_tree=local_nodes_on_tree,
                                current_tree=local_current_tree,
                                served_dest_count=served_dest_count,
                                current_node=curr_node,
                                phase="destination_connection",
                                target_node=subgoal_node,
                            )

                        state_to_save = Data(
                            x=x.cpu(),
                            edge_index=edge_index.cpu(),
                            edge_attr=edge_attr.cpu(),
                            req_vec=req_vec.cpu(),
                            tree_edge_index=tree_edge_index.cpu()
                            if tree_edge_index is not None else None,
                            dest_mask=dest_mask.cpu()
                            if dest_mask is not None else None,
                            action_mask=action_mask.cpu()
                            if action_mask is not None else None,
                        )

                    except Exception as e:
                        logger.warning(
                            f"step状态构造失败 req={req.get('id', 'unknown')} "
                            f"path_idx={path_idx} step_idx={step_idx}: {e}"
                        )
                        continue

                    sample_data = {
                        "state": state_to_save,
                        "request": clean_req,
                        "action": {
                            "sample_type": "path_step",
                            "path": [curr_node, next_node],
                            "high_label": int(high_label),
                            "low_label": int(next_node),
                            "current_node": int(curr_node),
                            "subgoal_node": int(subgoal_node),
                            "is_dest_path": subgoal_node in dest_list,
                            "step_idx": int(step_idx),
                            "path_index": int(path_idx),
                            "full_path": [int(v) for v in path_0],
                            "low_only": True,
                            "train_high": False,
                            "train_low": True,
                        },
                        "cost": 0.0,
                        "load": self._estimate_load(),
                        "hrl_info": {
                            "sample_type": "path_step",
                            "subgoal": int(subgoal_node),
                            "full_path": [int(v) for v in path_0],
                            "path_index": int(path_idx),
                            "step_idx": int(step_idx),
                        },
                    }

                    if self.use_timeslot:
                        sample_data["timeslot_info"] = {
                            "time_slot": req.get("time_slot", 0),
                            "duration": req.get("duration", 100),
                            "leave_time_slot": req.get("leave_time_slot", 100),
                        }

                    self.success_samples.append(sample_data)
                    self.stats["paths_collected"] += 1
                    self.stats["path_step_samples"] += 1

                    
                    local_nodes_on_tree.add(curr_node)
                    local_nodes_on_tree.add(next_node)
                    local_tree[(curr_node, next_node)] = 1.0
                    local_tree[(next_node, curr_node)] = 1.0

                
                for node in path_0:
                    nodes_on_tree_so_far.add(int(node))

                for i in range(len(path_0) - 1):
                    u, v = int(path_0[i]), int(path_0[i + 1])
                    current_tree_for_state["tree"][(u, v)] = 1.0
                    current_tree_for_state["tree"][(v, u)] = 1.0

                if subgoal_node in dest_list:
                    current_tree_for_state["connected_dests"].add(int(subgoal_node))
                    served_dest_count += 1

            self.stats["success_requests"] += 1

        except Exception as e:
            logger.exception(f"_process_single_request失败 req={req.get('id', 'unknown')}: {e}")
            self.stats["failed_requests"] += 1

        pbar.update(1)

    def _save_final(self):
        path = os.path.join(self.output_dir, "expert_data_final.pkl")
        try:
            data_to_save = {
                "success": self.success_samples,
                "stats": self.stats
            }
            if self.use_timeslot:
                data_to_save["timeslot_stats"] = self.timeslot_stats

            with open(path, "wb") as f:
                pickle.dump(data_to_save, f)

            logger.info(f" Saved {len(self.success_samples)} expert samples to {path}")
            logger.info(f"   成功请求: {self.stats['success_requests']} 个")
            logger.info(f"   失败请求: {self.stats['failed_requests']} 个")
            logger.info(f"   收集路径: {self.stats['paths_collected']} 条")
            logger.info(f"   样本总数: {len(self.success_samples)} 个")

            if self.use_timeslot:
                logger.info(f"⏰ 时间槽统计:")
                logger.info(f"   总时间槽: {self.timeslot_stats['total_time_slots']}")
        except Exception as e:
            logger.error(f" Save failed: {e}")
            import traceback
            traceback.print_exc()
