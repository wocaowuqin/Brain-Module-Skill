#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRLAgentMemory — 经验存储（High/Low/Success buffer）

补丁说明（相对原版）：
  [P1a] store_transition_low: goal_emb 存入 replay 前加 .detach().clone()，
        避免将带历史计算图的 Tensor 存进 buffer 导致内存泄漏。
  [P1b] store_transition_low: 新增 unconnected_dests 字段，
        在存 transition 时同步保存当前 env.unconnected_dests 的快照（list），
        供 agent_train.py 训练时重建 dest_mask，修复训练/推断 dest_mask 不一致问题。
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict
import copy

logger = logging.getLogger(__name__)


class HRLAgentMemory:
    """
    负责：
    - store_transition_high: 存储高层经验
    - store_transition_low:  存储低层经验 + success_memory + 内在奖励
    - store_transition:      向后兼容接口
    """

    def store_transition_high(
            self, state, goal, reward, next_state, done,
            action_meta=None, next_action_meta=None, aux_label=None,
    ):
        """存储 High-Level 经验（含 candidate metadata）"""
        goal_idx = max(0, min(int(goal), self.n_goals - 1)) if isinstance(goal, (int, np.integer)) else 0
        scaled_reward = max(-20.0, min(150.0, float(reward)))

        _req = None
        try:
            if self.env is not None and hasattr(self.env, 'current_request'):
                _req = copy.deepcopy(self.env.current_request)
        except Exception:
            pass

        action_meta      = action_meta      or {}
        next_action_meta = next_action_meta or {}

        transition_high = {
            'state':      copy.deepcopy(state),
            'goal':       goal_idx,
            'reward':     scaled_reward,
            'next_state': copy.deepcopy(next_state),
            'done':       done,
            'req':        _req,
            
            'used_candidate_path':   bool(action_meta.get('used_candidate_path', False)),
            'local_goal_idx':        action_meta.get('local_goal_idx', None),
            'candidate_indices':     action_meta.get('candidate_indices', None),
            'candidate_local_feats': action_meta.get('candidate_local_feats', None),
            'action_mask':           action_meta.get('action_mask', None),
            
            'next_candidate_indices':     next_action_meta.get('candidate_indices', None),
            'next_candidate_local_feats': next_action_meta.get('candidate_local_feats', None),
            'next_action_mask':           next_action_meta.get('action_mask', None),
            
            'aux_label':                  aux_label,
        }

        if getattr(self, '_use_per', False):
            self.high_memory.add(transition_high)
        else:
            self.high_memory.append(transition_high)


    def store_transition_low(
            self, state, action, reward, next_state, done,
            action_meta=None, next_action_meta=None,
    ):
        """存储Low-Level经验（含 candidate metadata）"""
        scaled_reward = max(-20.0, min(150.0, float(reward)))

        if self.config.get('hrl', {}).get('use_intrinsic_reward', False):
            try:
                with torch.no_grad():
                    se = self._extract_state_embedding(state)
                    nse = self._extract_state_embedding(next_state)
                    err = F.mse_loss(se, nse).item()
                    scaled_reward += min(0.3, err * 0.3)
            except Exception:
                pass

        safe_state = copy.deepcopy(state)
        safe_next_state = copy.deepcopy(next_state)

        _req = None
        try:
            if self.env is not None and hasattr(self.env, 'current_request'):
                _req = copy.deepcopy(self.env.current_request)
        except Exception:
            pass

        
        _unconnected_dests = None
        try:
            if self.env is not None and hasattr(self.env, 'unconnected_dests'):
                _ud = self.env.unconnected_dests
                _unconnected_dests = list(_ud) if _ud else None
        except Exception:
            pass

        action_meta = action_meta or {}
        next_action_meta = next_action_meta or {}

        transition = {
            'state':      safe_state,
            'action':     int(action),
            'reward':     scaled_reward,
            'next_state': safe_next_state,
            'done':       done,
            
            'goal_emb':   (self.current_goal_emb.detach().clone()
                           if self.current_goal_emb is not None else None),
            'req':        _req,
            
            'unconnected_dests': _unconnected_dests,
            
            'used_candidate_path':  bool(action_meta.get('used_candidate_path', False)),
            'local_action_idx':     action_meta.get('local_action_idx', None),
            'candidate_indices':    action_meta.get('candidate_indices', None),
            'candidate_local_feats':action_meta.get('candidate_local_feats', None),
            'current_node_idx':     action_meta.get('current_node_idx', None),
            'action_mask':          action_meta.get('action_mask', None),
            
            'next_candidate_indices':    next_action_meta.get('candidate_indices', None),
            'next_candidate_local_feats':next_action_meta.get('candidate_local_feats', None),
            'next_current_node_idx':     next_action_meta.get('current_node_idx', None),
            'next_action_mask':          next_action_meta.get('action_mask', None),
        }

        if getattr(self, '_use_per', False):
            self.low_memory.add(transition)
        else:
            self.low_memory.append(transition)

        self._ep_transitions.append(copy.deepcopy(transition))
        ep_steps = len(self._ep_transitions)

        if done:
            ep_reward = sum(t['reward'] for t in self._ep_transitions)
            
            if ep_reward > 60.0 and ep_steps < 40:
                self.success_memory.extend(copy.deepcopy(self._ep_transitions))
            if getattr(self, 'elite_buffer', None) is not None:
                self._best_reward = max(self._best_reward, ep_reward)
                if ep_reward >= self._best_reward * 0.8:
                    self.elite_buffer.add_episode(copy.deepcopy(self._ep_transitions), ep_reward)
            self._ep_transitions = []

        self.steps_done += 1

        _buf_len = len(self.low_memory)
        _buf_max = getattr(self.low_memory, 'maxlen',
                           getattr(self.low_memory, 'capacity', 0)) or 0
        if 0 < _buf_len < _buf_max and _buf_len % 10000 == 0:
            logger.info(f" Low Buffer: {_buf_len}/{_buf_max}")
        elif _buf_max > 0 and _buf_len >= _buf_max and not getattr(self, '_low_buf_full_logged', False):
            logger.info(f" Low Buffer已满: {_buf_len}/{_buf_max}")
            self._low_buf_full_logged = True

    def store_transition(
            self, state, action, reward, next_state, done,
            goal=None, next_valid_actions=None,
            action_meta=None, next_action_meta=None,
            high_action_meta=None, next_high_action_meta=None,
    ):
        """向后兼容接口"""
        if isinstance(action, (list, tuple)) and len(action) == 2:
            high_action, low_action = action
            if goal is not None:
                self.store_transition_high(
                    state, goal, reward, next_state, done,
                    action_meta=high_action_meta,
                    next_action_meta=next_high_action_meta,
                )
            self.store_transition_low(
                state, low_action, reward, next_state, done,
                action_meta=action_meta,
                next_action_meta=next_action_meta,
            )
        else:
            self.store_transition_low(
                state, action, reward, next_state, done,
                action_meta=action_meta,
                next_action_meta=next_action_meta,
            )

    def finalize_episode_memory(self):
        """每个episode结束时显式调用，防止_ep_transitions跨episode泄漏。
        适用于episode通过截断/失败/上层终止结束时没有done=True的情况。"""
        if not self._ep_transitions:
            return
        if getattr(self, 'elite_buffer', None) is not None:
            ep_reward = sum(t['reward'] for t in self._ep_transitions)
            self._best_reward = max(self._best_reward, ep_reward)
            if ep_reward >= self._best_reward * 0.8:
                self.elite_buffer.add_episode(self._ep_transitions, ep_reward)
        self._ep_transitions = []