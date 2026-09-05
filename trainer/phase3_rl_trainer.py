# core/trainer/phase3_rl_trainer.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 RL Trainer - TA-HRL v4 compact logging version.

Features:
1. Tracks key metrics: success rate, resource utilization, tree length, and sharing ratio.
2. Supports lightweight GAT verification and encoder diagnostics.
3. Uses the loaded dataset size as the training horizon when available.
"""
import logging
import os
import csv
import json
from typing import Optional

import numpy as np
import random
import pickle
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch

from trainer.training_analyzer import TrainingAnalyzer

logger = logging.getLogger(__name__)


class Phase3RLTrainer:
    """Phase 3: RL Trainer with HRL Coordinator (Clean Logs)"""

    def __init__(self, env, agent, output_dir, config, coordinator):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

        if coordinator is None:
            raise ValueError("Coordinator is required")
        self.coordinator = coordinator


        # Training parameters.
        phase3_cfg = config.get("phase3", {})
        self.save_freq = phase3_cfg.get("save_every", 500)
        # ``max_steps`` is the whole-training budget (3,000,000 in phase3).
        # Using it here allowed one deadlocked request to run for hours.
        self.max_steps_per_episode = int(
            phase3_cfg.get(
                "max_steps_per_episode",
                config.get("environment", {}).get("max_steps_per_episode", 600),
            )
        )

        # Prefer dataset size from the environment; fall back to config if unavailable.
        self._cfg_max_episodes = phase3_cfg.get("episodes", 1000)
        dataset_size = self._get_dataset_size()
        if dataset_size is not None:
            self.max_episodes = dataset_size
            logger.info(f"Dynamic episodes from dataset size: {self.max_episodes}")
        else:
            self.max_episodes = self._cfg_max_episodes
            logger.info(f"Using configured max_episodes: {self.max_episodes}")

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))

        # Training statistics.
        self.stats = {
            "rewards": [],
            "episode_lengths": [],
            "success_rate": [],
            "resource_utilization": [],
            "tree_lengths": [],
            "tree_flow_sums": [],
            "tree_actual_bws": [],
            "sharing_ratios": [],    # Edge sharing ratio.
            "tree_bias_vals": [],    # tree_bias trajectory.
        }

        self.start_episode = 0
        self._resume_path = None
        self._env_state_restored = False

        # Cumulative resource costs for successful requests.
        self._cum_cpu = 0.0   # Cumulative CPU cost over successful requests.
        self._cum_bw  = 0.0   # Cumulative bandwidth cost over successful requests.
        self._cum_mem = 0.0   # Cumulative memory cost over successful requests.

        logger.info("Trainer initialized (TA-HRL v4 compact)")
        self.analyzer = TrainingAnalyzer(output_dir=str(self.output_dir))

        # Dataset output initialization.
        self.dataset_dir = self.output_dir / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_csv_path = self.dataset_dir / "episodes.csv"
        self._dataset_csv_file = open(self._dataset_csv_path, 'w', newline='', encoding='utf-8')
        self._dataset_csv_writer = csv.DictWriter(self._dataset_csv_file, fieldnames=[
            'episode', 'success', 'fail_reason',
            'total_reward', 'steps',
            'vnf_done', 'vnf_total', 'dest_done', 'dest_total',
            'first_dest_steps', 'first_dest_new_edges', 'first_dest_reuse_edges',
            'first_dest_fail_target', 'first_dest_fail_anchor', 'first_dest_fail_reason',
            'dest_fail_count', 'dest_fail_log_tail',
            'tree_len', 'directed_tree_edges', 'undirected_tree_edges',
            'bidirectional_edge_pairs', 'reverse_tree_edge_moves',
            'sum_flow', 'actual_bw_per_req', 'sharing_ratio',
            'cpu_util_pct', 'bw_util_pct', 'res_util',
            'cpu_used_abs', 'mem_used_abs', 'bw_used_abs',
            'cpu_cumsum', 'mem_cumsum', 'bw_cumsum',
            'cpu_resource_comp', 'mem_resource_comp', 'bw_resource_comp',
            'accept_rate', 'block_rate',
            'epsilon_high', 'epsilon_low',
            'high_loss', 'low_loss',
            'request_id', 'arrival_time',
        ])
        self._dataset_csv_writer.writeheader()
        self._dataset_csv_file.flush()
        self._dataset_records = []   # In-memory list used for JSON/Pickle export at the end.
        logger.info(f"Dataset output directory: {self.dataset_dir}")

        self._first_dest_debug_path = self.dataset_dir / "first_dest_debug.csv"
        self._first_dest_debug_file = open(self._first_dest_debug_path, 'w', newline='', encoding='utf-8')
        self._first_dest_debug_writer = csv.DictWriter(self._first_dest_debug_file, fieldnames=[
            'episode', 'request_id', 'success', 'fail_reason',
            'vnf_done', 'vnf_total', 'dest_done', 'dest_total',
            'first_dest_steps', 'first_dest_new_edges', 'first_dest_reuse_edges',
            'first_dest_fail_target', 'first_dest_fail_anchor', 'first_dest_fail_reason',
            'fd_target', 'fd_anchor', 'fd_reason', 'fd_steps',
            'fd_pool_size', 'fd_mask_alive_last', 'fd_bw_req', 'fd_current_node',
            'fd_reachable_bw', 'fd_shortest_bw_hop', 'fd_bottleneck_bw',
            'fd_target_incident_ok', 'fd_target_incident_total',
            'fd_anchor_choice',
            'dest_fail_count', 'dest_fail_log_tail',
        ])
        self._first_dest_debug_writer.writeheader()
        self._first_dest_debug_file.flush()
        logger.info(f"first-dest debug output: {self._first_dest_debug_path}")

        # Link bandwidth status output, aligned with Matlab Bandwidth_status.mat.
        # Each episode writes one row with remaining directed-link bandwidth.
        # The column order is fixed after the first episode from link_map edge_id order.
        self._link_status_path = self.dataset_dir / "link_bandwidth_status.csv"
        self._link_status_file = open(self._link_status_path, 'w', newline='', encoding='utf-8')
        self._link_status_writer = None   # Initialized after link column order is known.
        self._link_order = None           # [(u,v), ...] fixed by edge_id order after first episode.
        logger.info(f"link bandwidth status output: {self._link_status_path}")

        # Load Phase2 IL pretrained weights when configured.
        # Only network weights are loaded; optimizer state is intentionally ignored.
        # Phase2 uses IL learning rate, Phase3 uses RL learning rate, so optimizer states are incompatible.
        _ckpt_dir = (config.get('path', config.get('paths', {})).get('ckpt_dir') or
                     str(Path(output_dir).parent))
        skip_il = bool(config.get('phase3', {}).get('no_il') or config.get('no_il'))
        if skip_il:
            logger.info("Phase3 noIL ablation: skip loading Phase2 IL checkpoint")
        else:
            il_ckpt_path = (
                config.get('phase3', {}).get('il_checkpoint') or
                config.get('il_checkpoint') or
                str(Path(_ckpt_dir) / 'il_model_best.pth')
            )
            if not Path(il_ckpt_path).exists():
                _fallback = str(Path(_ckpt_dir) / 'il_model_final.pth')
                if Path(_fallback).exists():
                    il_ckpt_path = _fallback
            self._load_il_checkpoint(il_ckpt_path)

    def _get_dataset_size(self) -> Optional[int]:
        """Return dataset size discovered from env/time_slot_mgr, or None if unavailable."""
        # Prefer common dataset/request attribute names.
        for attr in ('dataset', 'requests', '_requests', 'all_requests',
                     'request_list', 'req_list', 'data'):
            val = getattr(self.env, attr, None)
            if val is not None and hasattr(val, '__len__') and len(val) > 0:
                logger.info(f"_get_dataset_size: env.{attr} = {len(val)}")
                return len(val)
        # TimeSlotManager
        if hasattr(self.env, 'time_slot_mgr') and self.env.time_slot_mgr is not None:
            tsm = self.env.time_slot_mgr
            for attr in ('requests', '_requests', 'all_requests', 'request_list'):
                val = getattr(tsm, attr, None)
                if val is not None and hasattr(val, '__len__') and len(val) > 0:
                    logger.info(f"_get_dataset_size: time_slot_mgr.{attr} = {len(val)}")
                    return len(val)
        return None

    def set_dataset_size(self, n: int) -> None:
        """Set dataset size after data loading."""
        self.max_episodes = n
        logger.info(f"set_dataset_size: max_episodes forced to {n}")

    def _sync_epsilon_decay_to_episodes(self, num_episodes: int) -> None:
        """Keep epsilon decay tied to the final training horizon."""
        try:
            n = int(num_episodes)
        except Exception:
            return
        if n <= 0 or self.agent is None:
            return
        decay_ep = max(1, int(n * 0.8))
        try:
            self.cfg.setdefault('training', {}).setdefault('epsilon', {})['decay_episodes'] = decay_ep
        except Exception:
            pass
        self.agent.epsilon_decay_episodes = decay_ep
        self.agent.epsilon_decay = float(decay_ep)
        logger.info(
            "epsilon decay synced to final horizon: "
            f"num_episodes={n}, decay_episodes={decay_ep} (80%)"
        )

    def run(self):
        """Run the phase3 RL training loop."""
        # Diagnose whether encoder parameters are included in optimizer_low.
        logger.info("=== Encoder diagnostics ===")
        logger.info(f"encoder is None: {self.agent.encoder is None}")
        if self.agent.encoder is not None:
            enc_params = list(self.agent.encoder.parameters())
            logger.info(f"encoder parameter count: {len(enc_params)}")
            if hasattr(self.agent.encoder, 'tree_bias'):
                logger.info(f"tree_bias requires_grad: {self.agent.encoder.tree_bias.requires_grad}")
                logger.info(f"tree_bias initial value: {self.agent.encoder.tree_bias.item():.6f}")
            else:
                logger.info("tree_bias: not present for this encoder variant")
            # Check whether optimizer_low contains encoder parameters.
            opt_param_ids = set()
            for pg in self.agent.optimizer_low.param_groups:
                for p in pg['params']:
                    opt_param_ids.add(id(p))
            enc_in_opt = sum(1 for p in enc_params if id(p) in opt_param_ids)
            logger.info(f"encoder params in optimizer_low: {enc_in_opt}/{len(enc_params)}")
            logger.info(f"optimizer_low param groups: {len(self.agent.optimizer_low.param_groups)}")

        # Confirm final episode count dynamically.
        # Re-read in run() in case dataset loading happened after __init__.
        dataset_size = self._get_dataset_size()
        if dataset_size is not None:
            num_episodes = dataset_size
            logger.info(f"Training episodes from dataset size: {num_episodes}")
        else:
            num_episodes = self.cfg.get('num_episodes', self.max_episodes)
            logger.info(f"Using dataset size as training horizon: {num_episodes}")

        self._sync_epsilon_decay_to_episodes(num_episodes)

        logger.info("\n" + "=" * 40)
        logger.info(f"Start training ({num_episodes} eps)")
        logger.info("=" * 40)

        if not getattr(self, '_env_state_restored', False):
            self.env.reset()
        start_episode = int(getattr(self, 'start_episode', 0) or 0)
        if start_episode > 0 and not getattr(self, '_env_state_restored', False):
            try:
                tsm = getattr(self.env, 'time_slot_mgr', None)
                if tsm is not None and hasattr(tsm, 'global_request_index'):
                    all_requests = getattr(tsm, 'all_requests', []) or []
                    tsm.global_request_index = min(start_episode, len(all_requests))
                    logger.info(f"[Resume] time_slot_mgr.global_request_index -> {tsm.global_request_index}")
            except Exception as _resume_env_e:
                logger.warning(f"[Resume] failed to advance request pointer: {_resume_env_e}")

        success_count = 0
        if start_episode > 0 and self.stats.get('success_rate'):
            try:
                success_count = int(round(float(self.stats['success_rate'][-1]) * start_episode))
            except Exception:
                success_count = 0

        pbar = tqdm(range(start_episode, num_episodes), desc="Training", unit="ep", file=__import__('sys').stderr)
        trees_data = []

        # Interval cumulative resource statistics.
        _interval     = 50          # Interval length.
        _cum_cpu      = 0.0         # Interval cumulative CPU cost.
        _cum_bw       = 0.0         # Interval cumulative bandwidth cost.
        _cum_mem      = 0.0         # Interval cumulative memory cost.
        _cum_success  = 0           # Interval successful request count.
        _interval_rows = []         # Rows later written to resource_interval.csv.
        # CSV output path.
        import csv as _csv_mod
        _interval_csv = self.dataset_dir / "resource_interval.csv"
        _interval_file = open(_interval_csv, 'w', encoding='utf-8', newline='')
        _interval_writer = _csv_mod.writer(_interval_file)
        _interval_writer.writerow([
            'episode', 'success_count',
            'cum_cpu', 'cum_bw', 'cum_mem',
            'cpu_per_req', 'bw_per_req', 'mem_per_req',
        ])
        _interval_file.flush()

        for episode in pbar:
            total_reward, info = self.coordinator.run_episode(
                max_steps=self.max_steps_per_episode
            )

            # Snapshot current tree for visualization/analysis.
            if info.get('req_snapshot'):
                trees_data.append({
                    'ep':           episode + 1,
                    'success':      info.get('success', False),
                    'req':          info['req_snapshot'],
                    'tree':         info.get('tree_snapshot'),
                    'chain':        info.get('chain_nodes', []),
                    'sfc_snapshot': info.get('sfc_snapshot'),
                })

            # Collect training statistics.
            self.stats['rewards'].append(total_reward)
            self.stats['episode_lengths'].append(info.get('steps', 0))

            try:
                res_util = self.env.get_resource_utilization()
            except Exception:
                res_util = 0.0
            self.stats['resource_utilization'].append(res_util)

            # Tree length and success count.
            tree_len = 0
            if info.get('success', False):
                success_count += 1
                try:
                    _tree_dict_len = (getattr(self.env, 'current_tree', None) or {}).get('tree', {})
                    _pos_edges_len = {
                        tuple(sorted((int(e[0]), int(e[1]))))
                        for e, f in _tree_dict_len.items()
                        if float(f) > 0.0
                    }
                    if _pos_edges_len:
                        tree_len = len(_pos_edges_len)
                    else:
                        _sfc_snap = info.get('sfc_snapshot') or {}
                        _all_edges = set()
                        for _seg in _sfc_snap.get('spine_paths', []):
                            for _i in range(len(_seg)-1):
                                _all_edges.add(tuple(sorted((_seg[_i], _seg[_i+1]))))
                        for _bp in _sfc_snap.get('branch_paths', {}).values():
                            for _i in range(len(_bp)-1):
                                _all_edges.add(tuple(sorted((_bp[_i], _bp[_i+1]))))
                        tree_len = len(_all_edges)
                except Exception:
                    tree_len = len(self.env.current_tree.get('tree', {}))
            self.stats['tree_lengths'].append(tree_len)

            # sum_flow / actual_bw use the same accounting path as bandwidth records.
            try:
                _tree_dict = self.env.current_tree.get('tree', {})
                _req_now   = getattr(self.env, 'current_request', None) or {}
                _bw_orig   = float(_req_now.get('bw_origin', 0.0))
                # Only committed tree edges (flow > 0) consume bandwidth.
                _sum_flow  = sum(1.0 for f in _tree_dict.values() if f > 0.0)
                _actual_bw = _bw_orig * _sum_flow
                self.stats['tree_flow_sums'].append(_sum_flow)
                self.stats['tree_actual_bws'].append(_actual_bw)
            except Exception:
                self.stats['tree_flow_sums'].append(0.0)
                self.stats['tree_actual_bws'].append(0.0)

            # Edge sharing ratio.
            sharing_ratio = 0.0
            if info.get('success', False):
                try:
                    _sfc = info.get('sfc_snapshot') or {}
                    _all_edges_list = []
                    for _seg in _sfc.get('spine_paths', []):
                        for _i in range(len(_seg) - 1):
                            _all_edges_list.append(tuple(sorted((_seg[_i], _seg[_i+1]))))
                    for _bp in _sfc.get('branch_paths', {}).values():
                        for _i in range(len(_bp) - 1):
                            _all_edges_list.append(tuple(sorted((_bp[_i], _bp[_i+1]))))
                    if _all_edges_list:
                        _unique = len(set(_all_edges_list))
                        _total  = len(_all_edges_list)
                        sharing_ratio = 1.0 - (_unique / _total) if _total > 0 else 0.0
                except Exception:
                    sharing_ratio = 0.0
            self.stats['sharing_ratios'].append(sharing_ratio)

            success_rate = success_count / (episode + 1)
            self.stats['success_rate'].append(success_rate)

            # Train agent.
            losses = {}
            if hasattr(self.agent, 'update_policies'):
                try:
                    losses = self.agent.update_policies() or {}
                except Exception as e:
                    logger.debug(f"agent.update_policies() failed: {e}")

            self.analyzer.record(
                episode=episode,
                info=info,
                res_util=res_util,
                env=self.env,
                coordinator=self.coordinator,
                agent=self.agent,
            )

            # Record link bandwidth status.
            try:
                dc_nodes = getattr(self.env, 'dc_nodes', list(range(self.env.n)))
                cpu_used = sum(max(0.0, self.env.resource_mgr.pool.cpu_cap[i] - self.env.resource_mgr.pool.get_available_cpu(i)) for i in dc_nodes)
                cpu_cap = sum(self.env.resource_mgr.pool.cpu_cap[i] for i in dc_nodes)
                avg_cpu_avail = cpu_used / max(cpu_cap, 1.0) * 100.0
            except Exception:
                avg_cpu_avail = 0.0

            try:
                bw_used_total = 0.0
                bw_cap_total  = 0.0
                pool = self.env.resource_mgr.pool
                bw_keys = (pool.iter_bandwidth_keys()
                           if hasattr(pool, 'iter_bandwidth_keys') else list(pool.link_map.keys()))
                for u, v in bw_keys:
                    cap   = pool.bw_cap.get((u, v), 0.0)
                    avail = pool.get_available_bandwidth(u, v)
                    bw_used_total += max(0.0, cap - avail)
                    bw_cap_total  += cap
                avg_bw_avail = bw_used_total / max(bw_cap_total, 1.0) * 100.0
            except Exception:
                avg_bw_avail = 0.0
                bw_used_total = 0.0

            # Record link bandwidth status once per episode.
            try:
                pool = self.env.resource_mgr.pool
                # Initialize link column order once, sorted by edge_id, then write the CSV header.
                if self._link_order is None and pool.link_map:
                    self._link_order = [
                        k for k, _ in sorted(pool.link_map.items(), key=lambda x: x[1])
                    ]
                    _headers = ['episode'] + [f'link_{u}_{v}' for u, v in self._link_order]
                    self._link_status_writer = csv.DictWriter(
                        self._link_status_file, fieldnames=_headers
                    )
                    self._link_status_writer.writeheader()
                    logger.info(f"[LinkStatus] link order initialized: {len(self._link_order)} directed links")

                # Write header once after link order is known.
                if self._link_status_writer is not None and self._link_order:
                    _row_bw = {'episode': episode}
                    for u, v in self._link_order:
                        _row_bw[f'link_{u}_{v}'] = round(
                            pool.bw_avail.get((u, v), 0.0), 2
                        )
                    self._link_status_writer.writerow(_row_bw)
                    # Flush periodically to reduce data loss on interruption.
                    if episode % 100 == 0:
                        self._link_status_file.flush()
            except Exception as _le:
                logger.debug(f"link bandwidth status record failed at ep={episode}: {_le}")

            # Absolute memory consumption.
            try:
                dc_nodes = getattr(self.env, 'dc_nodes', list(range(self.env.n)))
                mem_used_abs = sum(
                    max(0.0, self.env.resource_mgr.pool.mem_cap[i]
                        - self.env.resource_mgr.pool.get_available_memory(i))
                    for i in dc_nodes
                )
            except Exception:
                mem_used_abs = 0.0

            # Update progress bar.
            eps_low = getattr(self.agent, 'epsilon_low', None)
            high_loss = losses.get('high_loss', 0.0)
            low_loss = losses.get('low_loss', 0.0)
            postfix = {
                'Suc': f"{success_rate:.1%}",
                'Rwd': f"{total_reward:.1f}",
                'CPU': f"{avg_cpu_avail:.1f}%",
                'BW': f"{avg_bw_avail:.1f}%",
                'HLoss': f"{high_loss:.3f}",
                'LLoss': f"{low_loss:.3f}",
            }
            if eps_low is not None:
                postfix['eps'] = f"{eps_low:.3f}"
            pbar.set_postfix(postfix)

            # TensorBoard records.
            self.writer.add_scalar('Episode/Reward', total_reward, episode)
            self.writer.add_scalar('Episode/SuccessRate', success_rate, episode)
            self.writer.add_scalar('Episode/ResourceUtil', res_util, episode)
            self.writer.add_scalar('Resource/AvgCPU', avg_cpu_avail, episode)
            self.writer.add_scalar('Resource/AvgBW', avg_bw_avail, episode)
            if high_loss > 0: self.writer.add_scalar('Loss/HighLevel', high_loss, episode)
            if low_loss > 0: self.writer.add_scalar('Loss/LowLevel', low_loss, episode)
            if tree_len > 0: self.writer.add_scalar('Episode/TreeLength', tree_len, episode)

            try:
                _enc = getattr(self.agent, 'encoder', None)
                if _enc is not None and hasattr(_enc, 'tree_bias'):
                    _tb_raw = _enc.tree_bias.item()
                    # Convert tree_bias to an effective sigmoid weight.
                    _tb_eff = torch.sigmoid(torch.tensor(_tb_raw)).item()
                    self.stats['tree_bias_vals'].append(_tb_eff)
                    self.writer.add_scalar('TA_HRL/tree_bias_effective', _tb_eff, episode)
                    self.writer.add_scalar('TA_HRL/tree_bias_raw', _tb_raw, episode)
            except Exception:
                pass

            self.writer.add_scalar('TA_HRL/sharing_ratio', sharing_ratio, episode)

            # Episode-level dataset record, one row per episode.
            try:
                _eps_h = getattr(self.agent, 'epsilon_high', None)
                _eps_l = getattr(self.agent, 'epsilon_low',  None)
                _req   = getattr(self.env, 'current_request', None) or {}
                _tree  = getattr(self.env, 'current_tree', None) or {}
                _vnf_done  = getattr(self.env, 'next_vnf_idx', 0)
                _vnf_total = len(_req.get('vnf', []))
                # Prefer the canonical connected_dests from request_table[req_id].
                _req_id = _req.get('id')
                _dest_done = 0
                try:
                    _rm_ref = getattr(self.env, 'resource_mgr', None)
                    if _rm_ref is not None and _req_id is not None:
                        _rr = _rm_ref.request_table.get(_req_id)
                        if _rr is not None:
                            _dest_done = len(_rr.connected_dests)
                        else:
                            _dest_done = len(_tree.get('connected_dests', set()))
                    else:
                        _dest_done = len(_tree.get('connected_dests', set()))
                except Exception:
                    _dest_done = len(_tree.get('connected_dests', set()))
                _dest_total= len(_req.get('dest', []))
                # Accumulate resource cost only for successful requests.
                _is_success = bool(info.get('success', False))
                # Aligned with MATLAB: resource_comp = resource_used * duration.
                # duration = leave_time_step - arrive_time_step
                # resource_comp = resource * duration (seconds), aligned with MATLAB accounting.
                _duration = float(_req.get('lifetime', 0))
                # Bandwidth cost uses physical-link accounting, aligned with MATLAB sum(tree.set(:)).
                # (u,v) and (v,u) refer to the same physical link here; committed flow edges are counted once.
                # Per-request resource demand used for cumulative cost accounting.
                _req_cpu = sum(float(v) for v in _req.get('cpu_origin', []))
                _req_mem = sum(float(v) for v in _req.get('memory_origin', []))
                _bw_orig_cur = float(_req.get('bw_origin', 0.0))

                _directed_tree_edges = 0
                _undirected_tree_edges = 0
                _bidirectional_edge_pairs = 0
                _reverse_tree_edge_moves = int(
                    getattr(self.env, '_reverse_tree_edge_moves', 0) or 0)
                try:
                    _tree_now = (_tree or {}).get('tree', {})
                    _pos_edges = {
                        (int(e[0]), int(e[1]))
                        for e, f in _tree_now.items()
                        if float(f) > 0.0
                    }
                    _directed_tree_edges = len(_pos_edges)
                    _phy_edges = {tuple(sorted(e)) for e in _pos_edges}
                    _undirected_tree_edges = len(_phy_edges)
                    _bidirectional_edge_pairs = sum(
                        1
                        for u, v in _phy_edges
                        if (u, v) in _pos_edges and (v, u) in _pos_edges
                    )
                except Exception:
                    pass

                # BW cost follows the same accounting path as delayed commit:
                # every flow>0 directed edge consumes bw_origin in directed mode.
                _bw_inc = _bw_orig_cur * _directed_tree_edges if _is_success else 0.0
                if _is_success:
                    self._cum_cpu += _req_cpu * _duration
                    self._cum_mem += _req_mem * _duration
                    self._cum_bw  += _bw_inc  * _duration
                _cpu_resource_comp = _req_cpu * _duration if _is_success else 0.0
                _mem_resource_comp = _req_mem * _duration if _is_success else 0.0
                _bw_resource_comp  = _bw_inc  * _duration if _is_success else 0.0

                _row = {
                    'episode':      episode,
                    'success':      int(_is_success),
                    'fail_reason':  info.get('reason', '') or info.get('error', ''),
                    'total_reward': round(float(total_reward), 4),
                    'steps':        info.get('steps', 0),
                    'vnf_done':     _vnf_done,
                    'vnf_total':    _vnf_total,
                    'dest_done':    _dest_done,
                    'dest_total':   _dest_total,
                    'first_dest_steps': info.get('first_dest_steps', -1),
                    'first_dest_new_edges': info.get('first_dest_new_edges', -1),
                    'first_dest_reuse_edges': info.get('first_dest_reuse_edges', -1),
                    'first_dest_fail_target': info.get('first_dest_fail_target', ''),
                    'first_dest_fail_anchor': info.get('first_dest_fail_anchor', ''),
                    'first_dest_fail_reason': info.get('first_dest_fail_reason', ''),
                    'dest_fail_count': info.get('dest_fail_count', 0),
                    'dest_fail_log_tail': json.dumps(info.get('dest_fail_log_tail', []), ensure_ascii=False),
                    'tree_len':     tree_len,
                    'directed_tree_edges': _directed_tree_edges,
                    'undirected_tree_edges': _undirected_tree_edges,
                    'bidirectional_edge_pairs': _bidirectional_edge_pairs,
                    'reverse_tree_edge_moves': _reverse_tree_edge_moves,
                    'sum_flow':      round(float(self.stats['tree_flow_sums'][-1]) if self.stats['tree_flow_sums'] else 0.0, 2),
                    'actual_bw_per_req': round(float(self.stats['tree_actual_bws'][-1]) if self.stats['tree_actual_bws'] else 0.0, 2),
                    'sharing_ratio': round(float(sharing_ratio), 4),
                    'cpu_util_pct': round(float(avg_cpu_avail), 2),
                    'bw_util_pct':  round(float(avg_bw_avail), 2),
                    'res_util':     round(float(res_util), 4),
                    'cpu_used_abs': round(float(cpu_used), 4),
                    'mem_used_abs': round(float(mem_used_abs), 4),
                    'bw_used_abs':  round(float(bw_used_total), 4),
                    'cpu_cumsum':   round(float(self._cum_cpu), 4),
                    'mem_cumsum':   round(float(self._cum_mem), 4),
                    'bw_cumsum':    round(float(self._cum_bw),  4),
                    'cpu_resource_comp': round(_cpu_resource_comp, 4),
                    'mem_resource_comp': round(_mem_resource_comp, 4),
                    'bw_resource_comp':  round(_bw_resource_comp,  4),
                    'accept_rate':  round(float(success_rate), 4),
                    'block_rate':   round(1.0 - float(success_rate), 4),
                    'epsilon_high': round(float(_eps_h), 4) if _eps_h is not None else '',
                    'epsilon_low':  round(float(_eps_l), 4) if _eps_l is not None else '',
                    'high_loss':    round(float(losses.get('high_loss', 0.0)), 6),
                    'low_loss':     round(float(losses.get('low_loss',  0.0)), 6),
                    'request_id':   _req.get('id', ''),
                    'arrival_time': _req.get('arrival_time', ''),
                }
                self._dataset_csv_writer.writerow(_row)
                self._dataset_records.append(_row)
                try:
                    _fd = info.get('first_dest_fail_detail', {}) or {}
                    self._first_dest_debug_writer.writerow({
                        'episode': episode,
                        'request_id': _req.get('id', ''),
                        'success': int(_is_success),
                        'fail_reason': info.get('reason', '') or info.get('error', ''),
                        'vnf_done': _vnf_done,
                        'vnf_total': _vnf_total,
                        'dest_done': _dest_done,
                        'dest_total': _dest_total,
                        'first_dest_steps': info.get('first_dest_steps', -1),
                        'first_dest_new_edges': info.get('first_dest_new_edges', -1),
                        'first_dest_reuse_edges': info.get('first_dest_reuse_edges', -1),
                        'first_dest_fail_target': info.get('first_dest_fail_target', ''),
                        'first_dest_fail_anchor': info.get('first_dest_fail_anchor', ''),
                        'first_dest_fail_reason': info.get('first_dest_fail_reason', ''),
                        'fd_target': _fd.get('target', ''),
                        'fd_anchor': _fd.get('anchor', ''),
                        'fd_reason': _fd.get('reason', ''),
                        'fd_steps': _fd.get('steps', ''),
                        'fd_pool_size': _fd.get('pool_size', ''),
                        'fd_mask_alive_last': _fd.get('mask_alive_last', ''),
                        'fd_bw_req': _fd.get('bw_req', ''),
                        'fd_current_node': _fd.get('current_node', ''),
                        'fd_reachable_bw': _fd.get('reachable_bw', ''),
                        'fd_shortest_bw_hop': _fd.get('shortest_bw_hop', ''),
                        'fd_bottleneck_bw': _fd.get('bottleneck_bw', ''),
                        'fd_target_incident_ok': _fd.get('target_incident_ok', ''),
                        'fd_target_incident_total': _fd.get('target_incident_total', ''),
                        'fd_anchor_choice': json.dumps(_fd.get('anchor_choice', {}), ensure_ascii=False),
                        'dest_fail_count': info.get('dest_fail_count', 0),
                        'dest_fail_log_tail': json.dumps(info.get('dest_fail_log_tail', []), ensure_ascii=False),
                    })
                    self._first_dest_debug_file.flush()
                except Exception as _fde:
                    logger.debug(f"first_dest_debug record failed at ep={episode}: {_fde}")
                try:
                    self.env._reverse_tree_edge_moves = 0
                except Exception:
                    pass
                self._dataset_csv_file.flush()
            except Exception as _de:
                logger.warning(f"[DatasetRecordFail] ep={episode}: {_de}", exc_info=True)

            # Accumulate interval resource usage for each episode.
            try:
                _last_rec = self._dataset_records[-1] if self._dataset_records else {}
                if _last_rec.get('success'):
                    _cum_cpu = float(self._cum_cpu)
                    _cum_bw  = float(self._cum_bw)
                    _cum_mem = float(self._cum_mem)
                    _cum_success += 1
            except Exception:
                pass

            # Periodic checkpoint saving.
            if (episode + 1) % _interval == 0 or (episode + 1) == num_episodes:
                _n = max(_cum_success, 1)
                _interval_writer.writerow([
                    episode + 1, _cum_success,
                    f'{_cum_cpu:.3f}', f'{_cum_bw:.3f}', f'{_cum_mem:.3f}',
                    f'{_cum_cpu/_n:.3f}', f'{_cum_bw/_n:.3f}', f'{_cum_mem/_n:.3f}',
                ])
                _interval_file.flush()
                logger.info(
                    f"[Interval] ep={episode+1} suc={_cum_success} | "
                    f"cum: cpu={_cum_cpu:.1f} bw={_cum_bw:.1f} mem={_cum_mem:.1f} | "
                    f"per_req: cpu={_cum_cpu/_n:.3f} bw={_cum_bw/_n:.3f} mem={_cum_mem/_n:.3f}"
                )

            if episode > 0 and episode % 10 == 0:
                # Periodic resource consistency diagnostics.
                try:
                    pool     = self.env.resource_mgr.pool
                    req_mgr  = self.env.resource_mgr.request_manager
                    n_active = len(req_mgr.active_requests)
                    dc_nodes = getattr(self.env, 'dc_nodes', list(range(self.env.n)))

                    # BW pool usage = committed allocation + reserved-but-uncommitted amount.
                    # rsv>0 means there is reserved but uncommitted bandwidth.
                    _bw_keys = (pool.iter_bandwidth_keys()
                                if hasattr(pool, 'iter_bandwidth_keys') else list(pool.link_map.keys()))
                    _bw_rsv  = sum(pool.bw_reserved.get(e, 0.0) for e in _bw_keys)
                    _bw_pool = sum(pool.bw_cap[e] - pool.bw_avail[e] for e in _bw_keys)
                    _bw_alloc = _bw_pool - _bw_rsv   # committed allocation.
                    _bw_cap  = sum(pool.bw_cap[e] for e in _bw_keys)
                    _max_lk  = max((pool.bw_cap[e]-pool.bw_avail[e])/max(pool.bw_cap[e],1e-5)
                                   for e in _bw_keys) * 100.0
                    # The canonical source for bandwidth usage is request_table edge_allocations.
                    # Lifecycle resources keep temporal metadata and are not the source of truth for BW.
                    _lc_bw = 0.0
                    try:
                        _rm = self.env.resource_mgr
                        for _rr in _rm.request_table.values():
                            if _rr.state in {'PENDING', 'ACTIVE'}:
                                _lc_bw += sum(ea.bw for ea in _rr.edge_allocations)
                    except Exception:
                        # Fallback for legacy structures.
                        _lc_bw = sum(
                            rinfo['request'].get('bw_origin', 0.0)
                            for rinfo in req_mgr.active_requests.values()
                            for f in rinfo['resources'].get('tree', {}).values() if f > 0.0)
                    _lc_bw_new = _lc_bw  # alias for log

                    # CPU / MEM
                    _cpu_pool = sum(max(0.0, pool.cpu_cap[i] - pool.get_available_cpu(i))
                                    for i in dc_nodes)
                    _cpu_cap  = sum(pool.cpu_cap[i] for i in dc_nodes)
                    # The canonical source for CPU/MEM usage is instance_table.
                    # shared_vnf_instances is retained only as a compatibility view.
                    _rm = self.env.resource_mgr
                    try:
                        _lc_cpu = sum(
                            float(inst.cpu)
                            for inst in _rm.instance_table.values()
                            if inst.state == 'ACTIVE' and inst.ref_count > 0
                        )
                        _lc_mem = sum(
                            float(inst.mem)
                            for inst in _rm.instance_table.values()
                            if inst.state == 'ACTIVE' and inst.ref_count > 0
                        )
                    except Exception:
                        # Fallback for legacy shared_vnf_instances.
                        if hasattr(_rm, 'shared_vnf_instances'):
                            _lc_cpu = sum(float(inst.get('cpu_used', 0.0))
                                          for inst in _rm.shared_vnf_instances.values()
                                          if int(inst.get('ref_count', 0)) > 0)
                            _lc_mem = sum(float(inst.get('mem_used', 0.0))
                                          for inst in _rm.shared_vnf_instances.values()
                                          if int(inst.get('ref_count', 0)) > 0)
                        else:
                            _lc_cpu = _lc_mem = 0.0
                    _mem_pool = sum(max(0.0, pool.mem_cap[i] - pool.get_available_memory(i))
                                    for i in dc_nodes)
                    _mem_cap  = sum(pool.mem_cap[i] for i in dc_nodes)

                    # LEAK means pool usage is larger than lifecycle records.
                    _bw_leak  = max(0.0, _bw_pool  - _lc_bw)
                    _cpu_leak = max(0.0, _cpu_pool - _lc_cpu)
                    _mem_leak = max(0.0, _mem_pool - _lc_mem)
                    _bw_ghost = max(0.0, _lc_bw - _bw_pool)

                    # CPU reconciliation: real_used vs expected_used from active lifecycle records.
                    _cpu_recon = 'OK' if _cpu_leak < 5 else f'LEAK={_cpu_leak:.0f}'
                    _mem_recon = 'OK' if _mem_leak < 5 else f'LEAK={_mem_leak:.0f}'

                    logger.info(
                        f"[RESOURCE] ep={episode} active={n_active} | "
                        f"BW pool={_bw_pool:.0f}/{_bw_cap:.0f}({_bw_pool/_bw_cap*100:.1f}%) "
                        f"alloc={_bw_alloc:.0f} rsv={_bw_rsv:.0f} "
                        f"lc={_lc_bw:.0f} max={_max_lk:.1f}% "
                        f"{'LEAK='+str(round(_bw_leak,0)) if _bw_leak>1 else 'OK'}"
                        f"{' ghost='+str(round(_bw_ghost,0)) if _bw_ghost>1 else ''} | "
                        f"CPU pool={_cpu_pool:.0f}/{_cpu_cap:.0f}({_cpu_pool/_cpu_cap*100:.1f}%) "
                        f"lc={_lc_cpu:.0f} {_cpu_recon} | "
                        f"MEM pool={_mem_pool:.0f}/{_mem_cap:.0f}({_mem_pool/_mem_cap*100:.1f}%) "
                        f"lc={_lc_mem:.0f} {_mem_recon}"
                    )

                    # BW direction diagnostic: periodically print hot directed links.
                    if episode % 100 == 0:
                        try:
                            _checked = set()
                            _hot_links = []
                            for (u, v) in pool.link_map:
                                if (u, v) in _checked or (v, u) in _checked:
                                    continue
                                _checked.add((u, v))
                                _cap_uv = pool.bw_cap.get((u, v), 0)
                                _cap_vu = pool.bw_cap.get((v, u), 0)
                                _avail_uv = pool.bw_avail.get((u, v), _cap_uv)
                                _avail_vu = pool.bw_avail.get((v, u), _cap_vu)
                                _used_uv = (_cap_uv - _avail_uv) / max(_cap_uv, 1) * 100
                                _used_vu = (_cap_vu - _avail_vu) / max(_cap_vu, 1) * 100
                                if getattr(pool, '_shared_undirected_bw', lambda: False)():
                                    _used_vu = _used_uv
                                _max_dir = max(_used_uv, _used_vu)
                                if _max_dir > 30:  # Print only hot links above 30% utilization.
                                    _hot_links.append((_max_dir, u, v, _used_uv, _used_vu))
                            _hot_links.sort(reverse=True)
                            for _md, _u, _v, _uuv, _uvu in _hot_links[:10]:
                                _asym = abs(_uuv - _uvu)
                                _flag = " direction_asym" if _asym > 30 else ""
                                logger.info(
                                    f"[BWDir] ({_u:>2},{_v:>2}) "
                                    f"forward={_uuv:5.1f}% reverse={_uvu:5.1f}%"
                                    f"{_flag}"
                                )
                        except Exception as _bwd_e:
                            logger.debug(f"[BWDir] diagnostic failed: {_bwd_e}")
                except Exception as _e:
                    logger.debug(f"RESOURCE diagnostic failed: {_e}")

            if episode > 0 and episode % 10 == 0:
                recent_trees = [l for l in self.stats['tree_lengths'][-10:] if l > 0]
                avg_tree_len = np.mean(recent_trees) if recent_trees else 0.0

                eps_high = getattr(self.agent, 'epsilon_high', None)
                eps_low  = getattr(self.agent, 'epsilon_low',  None)
                steps    = getattr(self.agent, 'steps_done',   None)
                eps_str  = f" | eps_h={eps_high:.3f} eps_l={eps_low:.3f} steps={steps}" if eps_high is not None else ""

                recent_sharing = [s for s in self.stats['sharing_ratios'][-10:] if s > 0]
                avg_sharing = np.mean(recent_sharing) if recent_sharing else 0.0

                _tb_str = f" | tree_bias(eff)={self.stats['tree_bias_vals'][-1]:.4f}" if self.stats[
                    'tree_bias_vals'] else ""
                logger.info(
                    f"Ep {episode}: Rate={success_rate:.2%} | "
                    f"Rwd={total_reward:.1f} | Util={res_util:.2f} | "
                    f"CPU={avg_cpu_avail:.1f}% BW={avg_bw_avail:.1f}% | "
                    f"HLoss={high_loss:.4f} LLoss={low_loss:.4f} | "
                    f"TreeLen={avg_tree_len:.1f} "
                    f"FlowSum={np.mean(self.stats['tree_flow_sums'][-10:]) if self.stats['tree_flow_sums'] else 0.0:.1f} "
                    f"Sharing={avg_sharing:.3f}"
                    f"{_tb_str}{eps_str}"
                )

            if episode > 0 and episode % self.save_freq == 0:
                self._save_checkpoint(episode)

            # on_episode_end() is already called inside HRL_Coordinator.run_episode().
            # Epsilon decay is handled once at episode end to avoid double updates.

        logger.info("\n" + "=" * 40)
        logger.info("Training finished")
        logger.info("=" * 40)

        self._save_final_model(num_episodes)

        valid_tree_lens = [l for l in self.stats['tree_lengths'] if l > 0]
        avg_tree_final = np.mean(valid_tree_lens) if valid_tree_lens else 0

        valid_sharing = [s for s in self.stats['sharing_ratios'] if s > 0]
        avg_sharing_final = np.mean(valid_sharing) if valid_sharing else 0.0
        tb_start = self.stats['tree_bias_vals'][0]  if self.stats['tree_bias_vals'] else 0.5
        tb_end   = self.stats['tree_bias_vals'][-1] if self.stats['tree_bias_vals'] else 0.5

        print("\nFinal training statistics")
        print(f"   final success rate:      {success_rate:.2%}")
        print(f"   average reward:          {np.mean(self.stats['rewards']):.2f}")
        print(f"   average resource util:   {np.mean(self.stats['resource_utilization']):.2f}")
        print(f"   average tree length:     {avg_tree_final:.2f}")
        print(f"   average sharing ratio:   {avg_sharing_final:.3f}")
        print(f"   tree_bias effective weight: {tb_start:.4f} -> {tb_end:.4f}  (>0.5 means the model used tree structure more actively)")
        print("=" * 40)

        self.analyzer.report()

        # Finalize and save complete datasets.
        try:
            self._dataset_csv_file.flush()
            self._dataset_csv_file.close()
            try:
                self._first_dest_debug_file.flush()
                self._first_dest_debug_file.close()
                logger.info(f"first-dest debug saved: {self._first_dest_debug_path}")
            except Exception:
                pass
            try:
                _interval_file.close()
                logger.info(f"resource interval saved: {_interval_csv}")
            except Exception:
                pass
            logger.info(f"CSV dataset saved: {self._dataset_csv_path}")

            # JSON export for cross-language reading.
            json_path = self.dataset_dir / "episodes.json"
            with open(json_path, 'w', encoding='utf-8') as _jf:
                json.dump(self._dataset_records, _jf, ensure_ascii=False, indent=2)
            logger.info(f"JSON dataset saved: {json_path}")

            # Pickle export keeps full Python precision for direct pandas loading.
            pkl_path = self.dataset_dir / "episodes.pkl"
            with open(pkl_path, 'wb') as _pf:
                pickle.dump(self._dataset_records, _pf)
            logger.info(f"Pickle dataset saved: {pkl_path}")

            # Finalize link bandwidth status outputs.
            try:
                self._link_status_file.flush()
                self._link_status_file.close()
                logger.info(f"link bandwidth status saved: {self._link_status_path}")

                # Also save a NumPy matrix with shape [N_episodes, num_directed_links].
                import pandas as pd
                _df = pd.read_csv(self._link_status_path)
                _bw_matrix = _df.drop(columns=['episode']).values  # [N, 90]
                _npy_path = self.dataset_dir / "link_bandwidth_status.npy"
                np.save(str(_npy_path), _bw_matrix)
                logger.info(
                    f"link bandwidth matrix saved: {_npy_path}  shape={_bw_matrix.shape}"
                    f" (matrix shape=[{_bw_matrix.shape[0]}, 90])"
                )
            except Exception as _lse:
                logger.warning(f"link bandwidth finalization failed: {_lse}")

            # Dataset summary.
            n_total   = len(self._dataset_records)
            n_success = sum(r['success'] for r in self._dataset_records)
            logger.info(f"Dataset summary: total={n_total}, success={n_success} ({n_success/max(n_total,1):.1%})")

            # Deployment cost summary.
            try:
                import math as _math
                def _mean(vals):
                    v = [x for x in vals if x is not None and _math.isfinite(float(x))]
                    return sum(float(x) for x in v) / len(v) if v else 0.0

                _suc_recs = [r for r in self._dataset_records if r.get('success')]

                _acc       = n_success / max(n_total, 1) * 100
                _blk       = (n_total - n_success) / max(n_total, 1) * 100

                # Average utilization over all episodes.
                _cpu_util  = _mean([r.get('cpu_util_pct', 0) for r in self._dataset_records])
                _bw_util   = _mean([r.get('bw_util_pct',  0) for r in self._dataset_records])
                _res_util  = _mean([r.get('res_util',     0) for r in self._dataset_records])

                # Per-request resource usage, successful requests only.
                _cpu_used  = _mean([r.get('cpu_used_abs', 0) for r in _suc_recs])
                _mem_used  = _mean([r.get('mem_used_abs', 0) for r in _suc_recs])
                _bw_used   = _mean([r.get('bw_used_abs',  0) for r in _suc_recs])

                # Cumulative resource usage from the last record.
                _last = self._dataset_records[-1] if self._dataset_records else {}
                _cpu_cum = float(_last.get('cpu_cumsum', 0) or 0)
                _mem_cum = float(_last.get('mem_cumsum', 0) or 0)
                _bw_cum  = float(_last.get('bw_cumsum',  0) or 0)

                # Tree quality metrics, successful requests only.
                _tree_len  = _mean([r.get('tree_len',      0) for r in _suc_recs])
                _dir_tree  = _mean([r.get('directed_tree_edges', 0) for r in _suc_recs])
                _undir_tree= _mean([r.get('undirected_tree_edges', 0) for r in _suc_recs])
                _bidir_pair= _mean([r.get('bidirectional_edge_pairs', 0) for r in _suc_recs])
                _rev_moves = _mean([r.get('reverse_tree_edge_moves', 0) for r in _suc_recs])
                _sharing   = _mean([r.get('sharing_ratio', 0) for r in _suc_recs])
                _bw_per_req= _mean([r.get('actual_bw_per_req', 0) for r in _suc_recs])

                # Failure reasons.
                _fail_cnt = {}
                for r in self._dataset_records:
                    if not r.get('success'):
                        reason = str(r.get('fail_reason', '') or 'unknown').strip() or 'unknown'
                        _fail_cnt[reason] = _fail_cnt.get(reason, 0) + 1

                _cost_items = [
                    ('acc',           f'{_acc:.3f}%'),
                    # Store blocking rate as a plain numeric percentage
                    # (e.g. 12.500), without a literal '%' suffix.  This
                    # keeps CSV values directly usable by MATLAB/Excel.
                    ('block_rate',    f'{_blk:.3f}'),
                    ('cpu_util',      f'{_cpu_util:.3f}%'),
                    ('bw_util',       f'{_bw_util:.3f}%'),
                    ('res_util',      f'{_res_util:.3f}'),
                    ('cpu_used',      f'{_cpu_used:.3f}'),
                    ('mem_used',      f'{_mem_used:.3f}'),
                    ('bw_used',       f'{_bw_used:.3f}'),
                    ('cpu_cumsum',    f'{_cpu_cum:.1f}'),
                    ('mem_cumsum',    f'{_mem_cum:.1f}'),
                    ('bw_cumsum',     f'{_bw_cum:.1f}'),
                    ('tree_len',      f'{_tree_len:.2f}'),
                    ('directed_tree_edges', f'{_dir_tree:.2f}'),
                    ('undirected_tree_edges', f'{_undir_tree:.2f}'),
                    ('bidirectional_edge_pairs', f'{_bidir_pair:.2f}'),
                    ('reverse_tree_edge_moves', f'{_rev_moves:.2f}'),
                    ('sharing_ratio', f'{_sharing:.3f}'),
                    ('bw_per_req',    f'{_bw_per_req:.3f}'),
                ]
                for _r, _c in sorted(_fail_cnt.items(), key=lambda x: -x[1]):
                    _cost_items.append((f'fail_{_r}', str(_c)))

                logger.info("=" * 60)
                logger.info("  Deployment cost summary")
                for _k, _v in _cost_items:
                    logger.info(f"  {_k}={_v}")
                logger.info("=" * 60)

                # Save one-row cost summary CSV for experiment comparison.
                _cost_path = self.dataset_dir / "cost_summary.csv"
                import csv as _csv
                with open(_cost_path, 'w', encoding='utf-8', newline='') as _cf:
                    _cw = _csv.writer(_cf)
                    _cw.writerow(['metric', 'value'])
                    _cw.writerow(['total_episodes', n_total])
                    _cw.writerow(['success', n_success])
                    for _k, _v in _cost_items:
                        _cw.writerow([_k, _v])
                logger.info(f"cost summary saved: {_cost_path}")

            except Exception as _ce:
                logger.warning(f"deployment cost summary failed: {_ce}")

        except Exception as _se:
            logger.warning(f"Dataset finalization failed: {_se}")

    def _load_il_checkpoint(self, ckpt_path: str):
        """Load Phase2 IL pretrained weights into agent."""
        # Mismatched layers are skipped and compatible layers are reused.

        if not ckpt_path or not Path(ckpt_path).exists():
            logger.warning(
                f"Phase2 checkpoint not found: {ckpt_path}\n"
                "Phase3 starts from random initialization."
            )
            return

        def _filtered_load(model, state_dict, name):
            current = model.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in current and v.shape == current[k].shape}
            skipped = [k for k in state_dict if k not in filtered]
            model.load_state_dict(filtered, strict=False)
            logger.info(f"   {name}: kept={len(filtered)}, skipped={len(skipped)} {skipped}")

        try:
            ckpt = torch.load(ckpt_path, map_location=self.agent.device)
            if not self._is_il_checkpoint_compatible(ckpt, ckpt_path):
                return

            ckpt_n_goals   = ckpt.get('n_goals',   'unknown')
            ckpt_n_actions = ckpt.get('n_actions', 'unknown')
            curr_n_goals   = self.agent.high_policy.num_goals
            curr_n_actions = self.agent.low_policy.action_dim
            logger.info(f"   ckpt dimensions:    n_goals={ckpt_n_goals}, n_actions={ckpt_n_actions}")
            logger.info(f"   current dimensions: n_goals={curr_n_goals}, n_actions={curr_n_actions}")
            if ckpt_n_goals != 'unknown' and ckpt_n_goals != curr_n_goals:
                logger.warning(f"high_policy output dim mismatch: ckpt={ckpt_n_goals}, current={curr_n_goals}")

            if 'high_policy' in ckpt and hasattr(self.agent, 'high_policy'):
                _filtered_load(self.agent.high_policy, ckpt['high_policy'], 'high_policy')
                if hasattr(self.agent, 'target_high_policy'):
                    _filtered_load(self.agent.target_high_policy, ckpt['high_policy'], 'target_high_policy')

            if 'low_policy' in ckpt and hasattr(self.agent, 'low_policy'):
                _filtered_load(self.agent.low_policy, ckpt['low_policy'], 'low_policy')
                if hasattr(self.agent, 'target_low_policy'):
                    _filtered_load(self.agent.target_low_policy, ckpt['low_policy'], 'target_low_policy')

            if 'encoder' in ckpt and hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                _filtered_load(self.agent.encoder, ckpt['encoder'], 'encoder')
                if hasattr(self.agent.encoder, 'tree_bias'):
                    with torch.no_grad():
                        self.agent.encoder.tree_bias.fill_(0.0)
                    logger.info("   tree_bias reset to 0.0")

            logger.info(f"Phase2 IL weights loaded: {ckpt_path}")

        except Exception as e:
            logger.error(f"Phase2 checkpoint load failed: {e}; Phase3 starts from random initialization.")

    def _infer_checkpoint_encoder_class(self, ckpt):
        enc = None
        if isinstance(ckpt, dict):
            enc = ckpt.get('encoder')
            if enc is None and isinstance(ckpt.get('agent_state'), dict):
                enc = ckpt['agent_state'].get('encoder')
        if not isinstance(enc, dict) or not enc:
            return None

        keys = [str(k).lower() for k in enc.keys()]
        if any(k.startswith('gat1.') or k.startswith('gat2.') or '.gat' in k for k in keys):
            return 'GATEncoder'
        if any(
            k.startswith('topo_transformer')
            or k.startswith('tree_transformer')
            or k.startswith('dest_attn')
            or k.startswith('topo_edge_encoder')
            or k.startswith('edge_encoder')
            or k.startswith('fusion_gate')
            for k in keys
        ):
            return 'TreeTransformerEncoder'
        if any(k.startswith('mlp.') or k.startswith('topo_gcn') for k in keys):
            return 'AblationEncoder'
        return None

    def _current_ablation_variant(self):
        return (
            getattr(self.agent, '_ablation_variant', None)
            or self.cfg.get('ablation_variant')
            or self.cfg.get('hrl', {}).get('ablation_variant')
            or self.cfg.get('phase3', {}).get('ablation_variant')
            or 'full'
        )

    def _is_il_checkpoint_compatible(self, ckpt, ckpt_path):
        allow_cross = bool(
            self.cfg.get('allow_cross_variant_il')
            or self.cfg.get('phase3', {}).get('allow_cross_variant_il')
        )
        if allow_cross:
            logger.warning(
                "Phase3 IL cross-variant loading is enabled; skip encoder-class guard."
            )
            return True

        current_variant = self._current_ablation_variant()
        current_encoder = (
            self.agent.encoder.__class__.__name__
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None
            else None
        )
        ckpt_variant = ckpt.get('ablation_variant') if isinstance(ckpt, dict) else None
        ckpt_encoder = (
            ckpt.get('encoder_class') if isinstance(ckpt, dict) else None
        ) or self._infer_checkpoint_encoder_class(ckpt)

        logger.info(
            "Phase3 IL compatibility: current_variant=%s current_encoder=%s "
            "ckpt_variant=%s ckpt_encoder=%s",
            current_variant, current_encoder, ckpt_variant, ckpt_encoder,
        )

        if current_encoder and ckpt_encoder and current_encoder != ckpt_encoder:
            logger.warning(
                "Skip Phase2 IL checkpoint because encoder mismatch: "
                "current=%s checkpoint=%s path=%s. "
                "Train a variant-specific Phase2 IL checkpoint or use --no_il.",
                current_encoder, ckpt_encoder, ckpt_path,
            )
            return False

        if ckpt_variant and current_variant and ckpt_variant != current_variant:
            if not (current_variant == 'single_dqn' and ckpt_variant == 'full'):
                logger.warning(
                    "Skip Phase2 IL checkpoint because ablation variant mismatch: "
                    "current=%s checkpoint=%s path=%s.",
                    current_variant, ckpt_variant, ckpt_path,
                )
                return False

        return True

    def _get_agent_state(self):
        """Extract HRLAgent state; HRLAgent itself is not an nn.Module."""
        state = {}
        if hasattr(self.agent, 'high_policy') and hasattr(self.agent.high_policy, 'state_dict'):
            state['high_policy'] = self.agent.high_policy.state_dict()
        if hasattr(self.agent, 'target_high_policy') and hasattr(self.agent.target_high_policy, 'state_dict'):
            state['target_high_policy'] = self.agent.target_high_policy.state_dict()
        if hasattr(self.agent, 'low_policy') and hasattr(self.agent.low_policy, 'state_dict'):
            state['low_policy'] = self.agent.low_policy.state_dict()
        if hasattr(self.agent, 'target_low_policy') and hasattr(self.agent.target_low_policy, 'state_dict'):
            state['target_low_policy'] = self.agent.target_low_policy.state_dict()
        if hasattr(self.agent, 'encoder') and hasattr(self.agent.encoder, 'state_dict'):
            state['encoder'] = self.agent.encoder.state_dict()
        if hasattr(self.agent, 'target_encoder') and self.agent.target_encoder is not None and hasattr(self.agent.target_encoder, 'state_dict'):
            state['target_encoder'] = self.agent.target_encoder.state_dict()
        if hasattr(self.agent, 'optimizer_high'):
            try:
                state['optimizer_high'] = self.agent.optimizer_high.state_dict()
            except Exception:
                pass
        if hasattr(self.agent, 'optimizer_low'):
            try:
                state['optimizer_low'] = self.agent.optimizer_low.state_dict()
            except Exception:
                pass
        for name in ('epsilon_high', 'epsilon_low', 'steps_done', 'total_episodes'):
            if hasattr(self.agent, name):
                state[name] = getattr(self.agent, name)
        return state if state else None

    def load_checkpoint(self, ckpt_path):
        import os
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=getattr(self.agent, 'device', 'cpu'))
        agent_state = ckpt.get('agent_state', ckpt)
        if not isinstance(agent_state, dict):
            raise ValueError(f"invalid checkpoint agent_state: {ckpt_path}")

        def _load_module(attr, key, strict=True):
            module = getattr(self.agent, attr, None)
            state = agent_state.get(key)
            if module is not None and state is not None and hasattr(module, 'load_state_dict'):
                if key == 'encoder':
                    module.load_state_dict(state, strict=False)
                else:
                    module.load_state_dict(state, strict=strict)

        _load_module('high_policy', 'high_policy')
        _load_module('target_high_policy', 'target_high_policy')
        _load_module('low_policy', 'low_policy')
        _load_module('target_low_policy', 'target_low_policy')
        _load_module('encoder', 'encoder', strict=False)
        _load_module('target_encoder', 'target_encoder', strict=False)

        if getattr(self.agent, 'target_high_policy', None) is not None and 'target_high_policy' not in agent_state:
            self.agent.target_high_policy.load_state_dict(self.agent.high_policy.state_dict())
        if getattr(self.agent, 'target_low_policy', None) is not None and 'target_low_policy' not in agent_state:
            self.agent.target_low_policy.load_state_dict(self.agent.low_policy.state_dict())
        if getattr(self.agent, 'target_encoder', None) is not None and 'target_encoder' not in agent_state:
            self.agent.target_encoder.load_state_dict(self.agent.encoder.state_dict())

        for opt_attr, opt_key in (('optimizer_high', 'optimizer_high'), ('optimizer_low', 'optimizer_low')):
            opt = getattr(self.agent, opt_attr, None)
            state = agent_state.get(opt_key)
            if opt is not None and state is not None:
                try:
                    opt.load_state_dict(state)
                except Exception as exc:
                    logger.warning(f"[Resume] skip {opt_key}: {exc}")

        ep = int(ckpt.get('episode', -1))
        self.start_episode = max(0, ep + 1)
        self._resume_path = str(ckpt_path)
        self.stats.update(ckpt.get('stats') or {})
        self._restore_trainer_state(ckpt.get('trainer_state') or {})

        if 'total_episodes' in agent_state:
            self.agent.total_episodes = int(agent_state['total_episodes'])
        elif hasattr(self.agent, 'total_episodes'):
            self.agent.total_episodes = self.start_episode

        for name in ('epsilon_high', 'epsilon_low', 'steps_done'):
            if name in agent_state and hasattr(self.agent, name):
                setattr(self.agent, name, agent_state[name])

        if ('epsilon_high' not in agent_state or 'epsilon_low' not in agent_state) and hasattr(self.agent, '_update_epsilon'):
            self.agent._update_epsilon()

        self._restore_replay_state(ckpt.get('replay_state') or {})
        self._restore_env_state(ckpt.get('env_state') or {})
        self._restore_rng_state(ckpt.get('rng_state') or {})

        logger.info(
            f"[Resume] loaded checkpoint={ckpt_path} start_episode={self.start_episode} "
            f"epsilon_high={getattr(self.agent, 'epsilon_high', None)} "
            f"epsilon_low={getattr(self.agent, 'epsilon_low', None)} "
            f"steps_done={getattr(self.agent, 'steps_done', None)}"
        )

    def _snapshot_buffer(self, buf):
        import copy
        from collections import deque
        if buf is None:
            return None
        if buf.__class__.__name__ == 'PrioritizedReplayBuffer':
            return {
                'kind': 'prioritized',
                'capacity': getattr(buf, 'capacity', None),
                'alpha': getattr(buf, 'alpha', None),
                'buffer': copy.deepcopy(getattr(buf, 'buffer', [])),
                'priorities': copy.deepcopy(getattr(buf, 'priorities', None)),
                'pos': getattr(buf, 'pos', 0),
            }
        if isinstance(buf, deque):
            return {
                'kind': 'deque',
                'maxlen': buf.maxlen,
                'data': list(copy.deepcopy(buf)),
            }
        if buf.__class__.__name__ == 'EliteBuffer':
            return {
                'kind': 'elite',
                'capacity': getattr(buf, 'capacity', None),
                'buffer': copy.deepcopy(getattr(buf, 'buffer', [])),
            }
        try:
            return {'kind': 'object', 'object': copy.deepcopy(buf)}
        except Exception as exc:
            logger.warning(f"[Checkpoint] skip replay buffer {buf.__class__.__name__}: {exc}")
            return None

    def _restore_buffer(self, attr, state):
        from collections import deque
        if not state:
            return
        kind = state.get('kind')
        cur = getattr(self.agent, attr, None)
        if kind == 'prioritized':
            if cur is None or cur.__class__.__name__ != 'PrioritizedReplayBuffer':
                try:
                    from core.hrl.prioritized_buffer import PrioritizedReplayBuffer
                except Exception:
                    from prioritized_buffer import PrioritizedReplayBuffer
                cur = PrioritizedReplayBuffer(state.get('capacity') or len(state.get('buffer', [])) or 1,
                                              alpha=state.get('alpha') or 0.6)
                setattr(self.agent, attr, cur)
            cur.capacity = state.get('capacity', cur.capacity)
            cur.alpha = state.get('alpha', cur.alpha)
            cur.buffer = state.get('buffer', [])
            cur.priorities = state.get('priorities', cur.priorities)
            cur.pos = state.get('pos', 0)
        elif kind == 'deque':
            setattr(self.agent, attr, deque(state.get('data', []), maxlen=state.get('maxlen')))
        elif kind == 'elite':
            if cur is None or cur.__class__.__name__ != 'EliteBuffer':
                try:
                    from core.hrl.elite_buffer import EliteBuffer
                except Exception:
                    from elite_buffer import EliteBuffer
                cur = EliteBuffer(capacity=state.get('capacity') or 5000)
                setattr(self.agent, attr, cur)
            cur.capacity = state.get('capacity', cur.capacity)
            cur.buffer = state.get('buffer', [])
        elif kind == 'object' and 'object' in state:
            setattr(self.agent, attr, state['object'])

    def _get_replay_state(self):
        state = {}
        for attr in ('high_memory', 'low_memory', 'success_memory', 'elite_buffer'):
            if hasattr(self.agent, attr):
                state[attr] = self._snapshot_buffer(getattr(self.agent, attr))
        for attr in (
            '_ep_transitions', 'update_count', 'high_loss_history', 'low_loss_history',
            '_best_reward', '_train_step_no_cand', '_cand_diag_count', '_zero_cand_grad_streak',
        ):
            if hasattr(self.agent, attr):
                try:
                    import copy
                    state[attr] = copy.deepcopy(getattr(self.agent, attr))
                except Exception:
                    pass
        return state

    def _restore_replay_state(self, state):
        if not state:
            return
        for attr in ('high_memory', 'low_memory', 'success_memory', 'elite_buffer'):
            self._restore_buffer(attr, state.get(attr))
        for attr in (
            '_ep_transitions', 'update_count', 'high_loss_history', 'low_loss_history',
            '_best_reward', '_train_step_no_cand', '_cand_diag_count', '_zero_cand_grad_streak',
        ):
            if attr in state:
                setattr(self.agent, attr, state[attr])
        logger.info(
            f"[Resume] replay restored: high={len(getattr(self.agent, 'high_memory', []))} "
            f"low={len(getattr(self.agent, 'low_memory', []))} "
            f"success={len(getattr(self.agent, 'success_memory', []))}"
        )

    def _get_env_state(self):
        import copy
        env = self.env
        state = {
            'env_attrs': {},
            'time_slot_mgr': {},
            'resource_mgr': {},
            'coordinator': {},
        }
        env_attrs = (
            'current_episode', 'current_step', 'total_reward', 'step_count',
            'next_vnf_idx', 'current_phase', 'current_deployment_target',
            'current_vnf_to_deploy', 'current_target_node', 'subgoal_step_count',
            'last_high_action_idx', 'current_subgoal_node', 'nodes_on_tree',
            'current_tree', 'current_request', 'current_placements',
            'current_anchor_node', 'branch_states', 'current_branch_id',
            'curr_ep_node_allocs', 'curr_ep_link_allocs', 'chain_nodes',
            'sfc_upstream_nodes', 'current_sfc', 'current_node_location',
            'time_step', 'current_time', 'global_request_index',
            'total_requests_seen', 'total_requests_accepted', 'served_dest_count',
        )
        for attr in env_attrs:
            if hasattr(env, attr):
                try:
                    state['env_attrs'][attr] = copy.deepcopy(getattr(env, attr))
                except Exception:
                    pass

        tsm = getattr(env, 'time_slot_mgr', None)
        if tsm is not None:
            for attr in (
                'time_step', 'current_time_slot', 'current_slot_index', 'max_slot_index',
                'global_request_index', 'simulation_done', '_time_offset',
            ):
                if hasattr(tsm, attr):
                    state['time_slot_mgr'][attr] = copy.deepcopy(getattr(tsm, attr))

        rm = getattr(env, 'resource_mgr', None)
        if rm is not None:
            rm_state = state['resource_mgr']
            pool = getattr(rm, 'pool', None)
            if pool is not None:
                rm_state['pool'] = {}
                for attr in ('cpu_avail', 'mem_avail', 'bw_avail', 'cpu_reserved', 'mem_reserved', 'bw_reserved'):
                    if hasattr(pool, attr):
                        rm_state['pool'][attr] = copy.deepcopy(getattr(pool, attr))
            req_mgr = getattr(rm, 'request_manager', None)
            if req_mgr is not None:
                rm_state['request_manager'] = {}
                for attr in ('active_requests', 'expired_requests', 'stats'):
                    if hasattr(req_mgr, attr):
                        rm_state['request_manager'][attr] = copy.deepcopy(getattr(req_mgr, attr))
            for attr in (
                'hvt_all', 'vnf_instances', 'shared_vnf_instances',
                'request_table', 'instance_table', 'instance_index',
                'current_request', 'current_tree', 'current_phase',
                'next_vnf_idx', 'nodes_on_tree', 'total_requests_accepted',
                'served_dest_count',
            ):
                if hasattr(rm, attr):
                    rm_state[attr] = copy.deepcopy(getattr(rm, attr))

        coord = getattr(self, 'coordinator', None)
        if coord is not None:
            for attr in ('current_episode', 'resources_released', 'stats', '_ep_stats_accum'):
                if hasattr(coord, attr):
                    try:
                        state['coordinator'][attr] = copy.deepcopy(getattr(coord, attr))
                    except Exception:
                        pass
        return state

    def _restore_env_state(self, state):
        if not state:
            return
        env = self.env
        for attr, value in (state.get('env_attrs') or {}).items():
            setattr(env, attr, value)

        tsm = getattr(env, 'time_slot_mgr', None)
        if tsm is not None:
            for attr, value in (state.get('time_slot_mgr') or {}).items():
                setattr(tsm, attr, value)

        rm = getattr(env, 'resource_mgr', None)
        rm_state = state.get('resource_mgr') or {}
        if rm is not None and rm_state:
            pool = getattr(rm, 'pool', None)
            for attr, value in (rm_state.get('pool') or {}).items():
                if pool is not None:
                    setattr(pool, attr, value)
            req_mgr = getattr(rm, 'request_manager', None)
            for attr, value in (rm_state.get('request_manager') or {}).items():
                if req_mgr is not None:
                    setattr(req_mgr, attr, value)
            for attr, value in rm_state.items():
                if attr not in ('pool', 'request_manager'):
                    setattr(rm, attr, value)

        coord = getattr(self, 'coordinator', None)
        if coord is not None:
            for attr, value in (state.get('coordinator') or {}).items():
                setattr(coord, attr, value)
        self._env_state_restored = True
        active = 0
        try:
            active = len(self.env.resource_mgr.request_manager.active_requests)
        except Exception:
            pass
        logger.info(f"[Resume] env restored: active_requests={active}")

    def _get_trainer_state(self):
        return {
            'cum_cpu': self._cum_cpu,
            'cum_bw': self._cum_bw,
            'cum_mem': self._cum_mem,
            'dataset_records': getattr(self, '_dataset_records', []),
            'link_order': getattr(self, '_link_order', None),
        }

    def _restore_trainer_state(self, state):
        if not state:
            return
        self._cum_cpu = float(state.get('cum_cpu', self._cum_cpu))
        self._cum_bw = float(state.get('cum_bw', self._cum_bw))
        self._cum_mem = float(state.get('cum_mem', self._cum_mem))
        if 'dataset_records' in state:
            self._dataset_records = state['dataset_records']
        if 'link_order' in state:
            self._link_order = state['link_order']

    def _get_rng_state(self):
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        if not state:
            return
        try:
            if 'python' in state:
                random.setstate(state['python'])
            if 'numpy' in state:
                np.random.set_state(state['numpy'])
            if 'torch' in state:
                torch.set_rng_state(state['torch'])
            if 'torch_cuda' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['torch_cuda'])
        except Exception as exc:
            logger.warning(f"[Resume] rng state restore failed: {exc}")

    def _save_checkpoint(self, episode):
        save_path = self.output_dir / f"checkpoint_ep{episode}.pth"
        try:
            payload = {
                'episode': episode,
                'agent_state': self._get_agent_state(),
                'replay_state': self._get_replay_state(),
                'env_state': self._get_env_state(),
                'trainer_state': self._get_trainer_state(),
                'rng_state': self._get_rng_state(),
                'config': self.cfg,
                'ablation_variant': self._current_ablation_variant(),
                'encoder_class': (
                    self.agent.encoder.__class__.__name__
                    if hasattr(self.agent, 'encoder') and self.agent.encoder is not None
                    else None
                ),
                'stats': self.stats
            }
            tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, save_path)
            logger.info(f"[Checkpoint] saved full state: {save_path}")
        except Exception as exc:
            logger.warning(f"[Checkpoint] save failed: {exc}")

    def _save_final_model(self, episode):
        final_path = self.output_dir / "final_model.pth"
        try:
            payload = {
                'episode': episode,
                'agent_state': self._get_agent_state(),
                'replay_state': self._get_replay_state(),
                'env_state': self._get_env_state(),
                'trainer_state': self._get_trainer_state(),
                'rng_state': self._get_rng_state(),
                'config': self.cfg,
                'ablation_variant': self._current_ablation_variant(),
                'encoder_class': (
                    self.agent.encoder.__class__.__name__
                    if hasattr(self.agent, 'encoder') and self.agent.encoder is not None
                    else None
                ),
                'stats': self.stats
            }
            tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, final_path)
            logger.info(f"model saved: {final_path}")
        except Exception as e:
            logger.warning(f"save failed: {e}")


