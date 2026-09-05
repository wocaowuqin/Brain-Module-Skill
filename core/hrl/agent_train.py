#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRLAgentTrain — 策略更新（Double DQN）+ epsilon衰减 + soft update

补丁说明（相对原版）：
  [P2] _update_high_level: 计算 state_node_embs / next_state_node_embs 时，
       改为调用 _get_local_embedding(x['state'], req=x.get('req'))，
       与 state_tensor 使用同一历史 req，修复 graph_emb 和 node_embs 语义错位问题。
  [P3] _update_low_level 内的 _build_dest_mask_from_state:
       新增 transition 参数，优先从 transition['unconnected_dests'] 读取保存的
       目标节点列表（由 agent_memory.py 补丁1b 写入），修复训练/推断 dest_mask 不一致问题。
       调用方 _encode() 也同步传入 transition=transition。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HRLAgentTrain:
    """
    负责：
    - update / update_policies: 训练调度
    - _update_high_level: High-Level Double DQN
    - _update_low_level:  Low-Level Double DQN + success mix + Q监控
    - _soft_update_target_networks / _hard_update_target_networks
    - _update_epsilon / update_epsilon
    - _log_training_stats
    """

    def update(self) -> float:
        """向后兼容的update接口"""
        losses = self.update_policies()
        return losses.get('high_loss', 0.0) + losses.get('low_loss', 0.0)

    def update_policies(self) -> Dict[str, float]:
        losses = {}
        self.update_count += 1
        self._last_high_td = None
        self._last_low_td = None
        
        if self.update_count % 5 == 0:
            logger.info(f"[Buffer] high={len(self.high_memory)} low={len(self.low_memory)} "
                        f"update_count={self.update_count} "
                        f"高层门槛={max(1, self.batch_size // 8)} 低层门槛={max(16, self.batch_size // 4)}")
        
        ll_updated = False
        if len(self.low_memory) >= max(16, self.batch_size // 4):
            ll = self._update_low_level()
            losses['low_loss'] = ll
            if ll > 0:
                self.low_loss_history.append(ll)
                ll_updated = True

        
        
        hl_updated = False
        if self.update_count % 3 == 0 and len(self.high_memory) >= max(1, self.batch_size // 8):
            hl = self._update_high_level()
            losses['high_loss'] = hl
            if hl > 0:
                self.high_loss_history.append(hl)
                hl_updated = True

        
        if self.update_count % 3 == 0 and getattr(self.encoder, 'use_aux_head', False):
            _al = self._update_aux()
            if _al > 0:
                losses['aux_loss'] = _al

        losses['total_loss'] = losses.get('high_loss', 0) + losses.get('low_loss', 0)

        
        
        if hl_updated:
            for tp, p in zip(self.target_high_policy.parameters(), self.high_policy.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        if ll_updated:
            for tp, p in zip(self.target_low_policy.parameters(), self.low_policy.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            
            if getattr(self, 'target_encoder', None) is not None and self.encoder is not None:
                for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        
        

        if self.update_count % 100 == 0:
            self._log_training_stats()

        self._record_train_stats(losses)
        return losses

    
    def _encode_with_aux(self, state, req=None):
        """带梯度编码 state，返回 (node_emb[N,H], aux_pred[N] or None)。
        与 _get_local_embedding 同构，但不加 no_grad、并请求 return_aux。"""
        real_state = state[0] if isinstance(state, tuple) else state
        _x  = real_state.x.to(self.device)
        _ei = real_state.edge_index.to(self.device)
        _ea = getattr(real_state, 'edge_attr', None)
        _ea = (_ea.to(self.device) if _ea is not None
               else torch.zeros(_ei.shape[1], 5, device=self.device))
        _b  = torch.zeros(_x.size(0), dtype=torch.long, device=self.device)
        _tei = getattr(real_state, 'tree_edge_index', None)
        if _tei is not None:
            _tei = _tei.to(self.device)
        if req is not None and getattr(self.encoder, 'req_fc', None) is not None:
            bw      = float(req.get('bw_origin', req.get('bw', 0.0)))
            cpu_lst = req.get('cpu_origin', req.get('cpu', []))
            mem_lst = req.get('memory_origin', req.get('memory', []))
            avg_cpu = float(np.mean(cpu_lst)) if len(cpu_lst) > 0 else 0.0
            avg_mem = float(np.mean(mem_lst)) if len(mem_lst) > 0 else 0.0
            _req = torch.tensor([[bw, avg_cpu, avg_mem]], dtype=torch.float32, device=self.device)
        else:
            _req = self._build_req_vec()
        _dest = self._build_dest_mask(state, _x.size(0))
        return self.encoder(_x, _ei, _ea, batch=_b, tree_edge_index=_tei, dest_mask=_dest,
                            req_vec=_req, return_aux=True)

    def _update_aux(self) -> float:
        """辅助头更新：用放置节点预测增量路由代价。
        无 aux_label 标签时自动跳过(返回0)，因此 coordinator 未提供标签前，
        本方法对训练零影响——可安全先跑 A/B/C 耦合消融。"""
        if not getattr(self.encoder, 'use_aux_head', False):
            return 0.0
        if len(self.high_memory) < max(8, self.batch_size // 4):
            return 0.0
        try:
            
            if getattr(self, '_use_per', False):
                batch, _, _ = self.high_memory.sample(self.batch_size, beta=0.4)
                if not batch:
                    return 0.0
            else:
                pool = list(self.high_memory)
                batch = random.sample(pool, min(self.batch_size, len(pool)))
            batch = [t for t in batch
                     if t.get('aux_label') is not None
                     and t.get('candidate_indices') is not None
                     and t.get('local_goal_idx') is not None]
            if len(batch) < 8:
                return 0.0
            preds, labels = [], []
            for t in batch:
                _, aux_pred = self._encode_with_aux(t['state'], req=t.get('req'))
                if aux_pred is None:
                    continue
                ci = t['candidate_indices']; li = int(t['local_goal_idx'])
                if li < 0 or li >= len(ci):
                    continue
                placed = int(ci[li])
                if placed < 0 or placed >= aux_pred.shape[0]:
                    continue
                preds.append(aux_pred[placed]); labels.append(float(t['aux_label']))
            if len(preds) < 8:
                return 0.0
            aux_loss = F.smooth_l1_loss(
                torch.stack(preds),
                torch.tensor(labels, device=self.device, dtype=torch.float32))
            if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                return 0.0
            self.optimizer_low.zero_grad()
            aux_loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad_norm)
            self.optimizer_low.step()
            return float(aux_loss.item())
        except Exception as _e:
            logger.warning(f"[_update_aux] 跳过本次辅助更新: {_e}")
            return 0.0

    

    def _update_high_level(self) -> float:
        """高层 candidate-based DDQN 更新。
        有候选集时走 score_goal_candidates()，否则 fallback 到全图 Q。"""
        if getattr(self, '_single_dqn', False):
            return 0.0
        
        
        if len(self.high_memory) < max(1, self.batch_size // 8):
            return 0.0

        try:
            _per_idx_h = None
            _per_w_h = None
            if getattr(self, '_use_per', False):
                batch, _per_idx_h, _per_w_h = self.high_memory.sample(self.batch_size, beta=0.4)
                if not batch:
                    return 0.0
            else:
                batch = random.sample(list(self.high_memory), self.batch_size)

            state_tensor = torch.cat([
                self._get_graph_embedding(x['state'], req=x.get('req'))
                for x in batch
            ]).to(self.device)

            next_state_tensor = torch.cat([
                self._get_graph_embedding(x['next_state'], req=x.get('req'))
                for x in batch
            ]).to(self.device)

            rewards = torch.tensor(
                [x['reward'] for x in batch], device=self.device
            ).float().unsqueeze(1)
            dones = torch.tensor(
                [x['done'] for x in batch], device=self.device
            ).float().unsqueeze(1)
            goals = torch.tensor(
                [x['goal'] for x in batch], device=self.device
            ).long().unsqueeze(1)

            
            
            
            state_node_embs = next_state_node_embs = None
            if getattr(self, 'encoder', None) is not None:
                try:
                    state_node_embs = torch.cat([
                        self._get_local_embedding(x['state'], req=x.get('req'))
                        for x in batch
                    ]).to(self.device)   # [B, N, H]
                    next_state_node_embs = torch.cat([
                        self._get_local_embedding(x['next_state'], req=x.get('req'))
                        for x in batch
                    ]).to(self.device)
                except Exception:
                    state_node_embs = next_state_node_embs = None

            _has_scorer = hasattr(self.high_policy, 'score_goal_candidates')

            
            curr_q_list   = []
            used_cand_cnt = 0

            for i, tr in enumerate(batch):
                cand_idx   = tr.get('candidate_indices')
                cand_feats = tr.get('candidate_local_feats')
                local_goal = tr.get('local_goal_idx')
                used_cand  = bool(tr.get('used_candidate_path', False))

                valid_cand = (
                    _has_scorer
                    and state_node_embs is not None
                    and used_cand
                    and cand_idx is not None
                    and len(cand_idx) > 0
                    and isinstance(local_goal, int)
                    and 0 <= local_goal < len(cand_idx)
                )

                if valid_cand:
                    used_cand_cnt += 1
                    feat_t = (torch.as_tensor(cand_feats, dtype=torch.float32, device=self.device)
                              if cand_feats is not None else None)
                    scores = self.high_policy.score_goal_candidates(
                        graph_emb=state_tensor[i:i+1],
                        candidate_indices=list(cand_idx),
                        candidate_node_embs=state_node_embs[i:i+1],
                        candidate_local_feats=feat_t,
                    )   # [1, K]
                    local_goal_t = torch.tensor([[int(local_goal)]], device=self.device)
                    curr_q_list.append(scores.gather(1, local_goal_t))
                else:
                    q_values, _, _ = self.high_policy(
                        state_tensor[i:i+1], return_subgoal=False
                    )
                    curr_q_list.append(q_values.gather(1, goals[i:i+1]))

            curr_q = torch.cat(curr_q_list, dim=0)

            
            with torch.no_grad():
                next_q_list = []

                for i, tr in enumerate(batch):
                    next_cand_idx   = tr.get('next_candidate_indices')
                    next_cand_feats = tr.get('next_candidate_local_feats')

                    valid_next = (
                        _has_scorer
                        and next_state_node_embs is not None
                        and next_cand_idx is not None
                        and len(next_cand_idx) > 0
                    )

                    if valid_next:
                        feat_t = (torch.as_tensor(next_cand_feats, dtype=torch.float32, device=self.device)
                                  if next_cand_feats is not None else None)
                        
                        scores_on = self.high_policy.score_goal_candidates(
                            graph_emb=next_state_tensor[i:i+1],
                            candidate_indices=list(next_cand_idx),
                            candidate_node_embs=next_state_node_embs[i:i+1],
                            candidate_local_feats=feat_t,
                        )
                        best_local = scores_on.argmax(dim=1, keepdim=True)
                        
                        scores_tg = self.target_high_policy.score_goal_candidates(
                            graph_emb=next_state_tensor[i:i+1],
                            candidate_indices=list(next_cand_idx),
                            candidate_node_embs=next_state_node_embs[i:i+1],
                            candidate_local_feats=feat_t,
                        )
                        next_q_list.append(scores_tg.gather(1, best_local))
                    else:
                        next_q_online, _, _ = self.high_policy(
                            next_state_tensor[i:i+1], return_subgoal=False
                        )
                        next_actions = next_q_online.argmax(dim=1, keepdim=True)
                        next_q_target, _, _ = self.target_high_policy(
                            next_state_tensor[i:i+1], return_subgoal=False
                        )
                        next_q_list.append(next_q_target.gather(1, next_actions))

                next_q   = torch.cat(next_q_list, dim=0)
                target_q = rewards + (1 - dones) * self.gamma * next_q

            
            
            _scale = 3.0
            elementwise_loss_h = F.smooth_l1_loss(
                curr_q / _scale, target_q / _scale, reduction='none')
            if getattr(self, '_use_per', False) and _per_w_h is not None:
                _w_h = torch.as_tensor(_per_w_h, dtype=torch.float32,
                                       device=self.device).view(-1, 1)
                if _w_h.shape[0] == elementwise_loss_h.shape[0]:
                    loss = (elementwise_loss_h * _w_h).mean()
                else:
                    loss = elementwise_loss_h.mean()
            else:
                loss = elementwise_loss_h.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(" High-Level Loss NaN/Inf，跳过")
                return 0.0

            self.optimizer_high.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.high_policy.parameters(), self.clip_grad_norm)
            self.optimizer_high.step()

            
            self._last_high_td = None
            if getattr(self, '_use_per', False) and _per_idx_h is not None:
                _td_h = (curr_q - target_q).abs().detach().cpu().numpy().flatten()
                _n_td = min(len(_per_idx_h), len(_td_h))
                if _n_td > 0:
                    self.high_memory.update_priorities(_per_idx_h[:_n_td], _td_h[:_n_td])
                    self._last_high_td = float(_td_h[:_n_td].mean())

            if not hasattr(self, '_high_train_step'):
                self._high_train_step = 0
            self._high_train_step += 1
            if self._high_train_step % 50 == 0:
                logger.info(
                    f"[HighTrainDiag] step={self._high_train_step} "
                    f"cand_ratio={used_cand_cnt / max(1, len(batch)):.3f} "
                    f"used_score_goal_candidates={used_cand_cnt}/{len(batch)} "
                    f"loss={loss.item():.4f}"
                )
            return loss.item()

        except Exception as e:
            logger.error(f"[Update High Level] Error: {e}")
            import traceback; traceback.print_exc()
            return 0.0

    

    def _update_low_level(self) -> float:
        if len(self.low_memory) < max(16, self.batch_size // 4):
            return 0.0
        
        if not hasattr(self, '_train_step'):              self._train_step = 0
        if not hasattr(self, '_train_step_no_cand'):      self._train_step_no_cand = 0
        if not hasattr(self, '_zero_cand_grad_streak'):   self._zero_cand_grad_streak = 0
        if not hasattr(self, '_cand_diag_count'):         self._cand_diag_count = 0
        if not hasattr(self, '_use_score_candidates_count_last'): self._use_score_candidates_count_last = 0
        try:
            
            _per_indices = None
            _per_weights = None
            if getattr(self, '_use_per', False) and len(self.low_memory) >= self.batch_size // 2:
                per_size = int(self.batch_size * 0.6)
                suc_size = int(self.batch_size * 0.2)
                eli_size = self.batch_size - per_size - suc_size
                per_batch, _per_indices, _per_weights = self.low_memory.sample(per_size, beta=0.4)
                suc_batch = (random.sample(self.success_memory, min(suc_size, len(self.success_memory)))
                             if len(self.success_memory) >= suc_size else [])
                eli_batch = (self.elite_buffer.sample(eli_size)
                             if getattr(self, 'elite_buffer', None) and len(self.elite_buffer) > 0 else [])
                batch = per_batch + suc_batch + eli_batch
                if len(batch) < self.batch_size // 2:
                    batch = per_batch  # fallback
                
                
            else:
                success_size = self.batch_size // 4
                if len(self.success_memory) >= success_size:
                    success_batch = random.sample(self.success_memory, success_size)
                    
                    if getattr(self, "_use_per", False):
                        normal_batch, _, _ = self.low_memory.sample(
                            self.batch_size - success_size, beta=0.4)
                    else:
                        normal_batch = random.sample(list(self.low_memory), self.batch_size - success_size)
                    batch = success_batch + normal_batch
                    random.shuffle(batch)
                else:
                    
                    if getattr(self, "_use_per", False):
                        batch, _, _ = self.low_memory.sample(self.batch_size, beta=0.4)
                    else:
                        
                        batch = random.sample(list(self.low_memory), self.batch_size)

            
            if self.encoder is not None:
                _topo_ei = (self.env.edge_index.to(self.device)
                            if self.env is not None
                            and hasattr(self.env, 'edge_index')
                            and self.env.edge_index is not None else None)

                def _fix_ea(ei, state_obj, fdim=5):
                    n = ei.shape[1]
                    ea = getattr(state_obj, 'edge_attr', None)
                    if ea is None: return torch.zeros(n, fdim, device=self.device)
                    ea = ea.to(self.device)
                    if ea.dim() == 1: ea = ea.unsqueeze(1)
                    if ea.shape[0] != n: return torch.zeros(n, fdim, device=self.device)
                    if ea.shape[1] < fdim:
                        ea = torch.cat([ea, torch.zeros(n, fdim - ea.shape[1], device=self.device)], dim=1)
                    return ea

                
                _N = (self.env.n if self.env is not None and hasattr(self.env, 'n')
                      else 28)

                
                _encoder_has_req = (self.encoder is not None
                                    and getattr(self.encoder, 'req_fc', None) is not None)

                def _build_req_vec_from_transition(transition):
                    """从transition的saved req字段重建req_vec，保持训练与推断一致。"""
                    if not _encoder_has_req:
                        return None
                    try:
                        req = transition.get('req')
                        if req is None:
                            return None
                        bw      = float(req.get('bw_origin', req.get('bw', 0.0)))
                        cpu_lst = req.get('cpu_origin', req.get('cpu', []))
                        mem_lst = req.get('memory_origin', req.get('memory', []))
                        avg_cpu = float(np.mean(cpu_lst)) if len(cpu_lst) > 0 else 0.0
                        avg_mem = float(np.mean(mem_lst)) if len(mem_lst) > 0 else 0.0
                        return torch.tensor([[bw, avg_cpu, avg_mem]],
                                            dtype=torch.float32, device=self.device)
                    except Exception:
                        return None

                
                
                def _build_dest_mask_from_state(s, n_nodes, transition=None):
                    if self.env is not None and getattr(self.env, '_minimal_mlp_state', False):
                        return None
                    """从 transition 或 state 对象还原 dest 掩码。"""
                    try:
                        dests = None
                        
                        if transition is not None:
                            dests = transition.get('unconnected_dests')
                        
                        if not dests:
                            dests = getattr(s, 'unconnected_dests', None)
                        if dests:
                            mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
                            for d in dests:
                                if isinstance(d, int) and 0 <= d < n_nodes:
                                    mask[d] = True
                            return mask if mask.any() else None
                    except Exception:
                        pass
                    return None

                def _encode(state_obj, detach=False, transition=None, encoder_model=None):
                    s = state_obj[0] if isinstance(state_obj, tuple) else state_obj
                    enc = encoder_model if encoder_model is not None else self.encoder
                    try:
                        if hasattr(s, 'x') and hasattr(s, 'edge_index'):
                            ei = s.edge_index.to(self.device) if s.edge_index is not None else _topo_ei
                            if ei is not None:
                                ea   = _fix_ea(ei, s)
                                b    = torch.zeros(s.x.size(0), dtype=torch.long, device=self.device)
                                _tei = getattr(s, 'tree_edge_index', None)
                                if _tei is not None: _tei = _tei.to(self.device)
                                _req  = (_build_req_vec_from_transition(transition)
                                         if transition is not None else None)
                                
                                _dest = _build_dest_mask_from_state(
                                    s, s.x.size(0), transition=transition)
                                out = enc(s.x.to(self.device), ei, ea, batch=b,
                                          tree_edge_index=_tei,
                                          dest_mask=_dest,
                                          req_vec=_req)
                                if detach: out = out.detach()
                                if out.size(0) == _N:
                                    return out.unsqueeze(0)
                                return out.mean(dim=0, keepdim=True).unsqueeze(0).expand(1, _N, -1)
                    except Exception:
                        pass
                    return torch.zeros(1, _N, self.hidden_dim, device=self.device)

                # online encoder: state / target_encoder: next_state
                _online_enc = self.encoder
                _target_enc = getattr(self, 'target_encoder', None) or self.encoder
                state_tensor      = torch.cat([
                    _encode(x['state'],      detach=False, transition=x, encoder_model=_online_enc)
                    for x in batch])
                next_state_tensor = torch.cat([
                    _encode(x['next_state'], detach=True,  transition=x, encoder_model=_target_enc)
                    for x in batch]).detach()
                # state_tensor: [B, N, H]
            else:
                
                _N_fb = (self.env.n if self.env is not None and hasattr(self.env, 'n') else 28)
                def _enc_fb(s):
                    e = self._extract_state_embedding(s).to(self.device)  # [1, H]
                    return e.unsqueeze(1).expand(1, _N_fb, -1)            # [1, N, H]
                state_tensor      = torch.cat([_enc_fb(x['state'])      for x in batch])
                next_state_tensor = torch.cat([_enc_fb(x['next_state']) for x in batch]).detach()

            actions = torch.tensor([x['action']  for x in batch], device=self.device).long().unsqueeze(1)
            rewards = torch.tensor([x['reward']  for x in batch], device=self.device).float().unsqueeze(1)
            dones   = torch.tensor([x['done']    for x in batch], device=self.device).float().unsqueeze(1)

            
            valid_mask = (actions >= 0).squeeze()
            if valid_mask.sum() == 0: return 0.0
            state_tensor      = state_tensor[valid_mask]
            next_state_tensor = next_state_tensor[valid_mask]
            actions = actions[valid_mask]
            rewards = torch.clamp(rewards[valid_mask], -20.0, 150.0)
            dones   = dones[valid_mask]

            # Goal embedding
            valid_idx = torch.nonzero(valid_mask).squeeze().cpu().tolist()
            if not isinstance(valid_idx, list): valid_idx = [valid_idx]
            goal_embs = []
            for idx in valid_idx:
                g = batch[idx].get('goal_emb')
                if g is None:
                    g = torch.zeros(1, self.goal_dim, device=self.device)
                else:
                    g = g.to(self.device)
                    if g.dim() == 1: g = g.unsqueeze(0)
                    if g.size(1) != self.goal_dim:
                        g = (g[:, :self.goal_dim] if g.size(1) > self.goal_dim
                             else torch.cat([g, torch.zeros(g.size(0), self.goal_dim - g.size(1), device=self.device)], 1))
                goal_embs.append(g)
            goal_tensor = torch.cat(goal_embs).to(self.device)

            
            
            _used_cand_flags = []
            _local_idx_list  = []
            for idx in valid_idx:
                tr = batch[idx]
                _ucp = tr.get('used_candidate_path', False)
                _lai = tr.get('local_action_idx', None)
                _cands = tr.get('candidate_indices')
                
                _valid_cand = (
                    bool(_ucp)
                    and _lai is not None
                    and isinstance(_lai, int)
                    and _lai >= 0
                    and _cands is not None
                    and len(_cands) > 0
                    and _lai < len(_cands)
                )
                _used_cand_flags.append(_valid_cand)
                _local_idx_list.append(_lai if _valid_cand else None)

            _n_cand_samples = sum(_used_cand_flags)
            _cand_nonempty_ratio = _n_cand_samples / max(len(valid_idx), 1)

            
            self._cand_diag_count += 1
            if self._cand_diag_count % 200 == 0:
                logger.info(
                    f"[CandDiag] candidate路径样本={_n_cand_samples}/{len(valid_idx)} "
                    f"ratio={_cand_nonempty_ratio:.2f} "
                    f"batch_size={len(batch)}"
                )

            
            if _n_cand_samples == 0:
                self._train_step_no_cand += 1
                if self._train_step_no_cand >= 50 and self._train_step_no_cand % 50 == 0:
                    logger.warning(
                        f"[TrainAlert]  连续 {self._train_step_no_cand} 步训练未使用 score_candidates! "
                        f"use_score_candidates=0/{len(valid_idx)} — "
                        f"请检查 replay buffer 是否存入了 used_candidate_path=True 的样本"
                    )
            else:
                self._train_step_no_cand = 0

            
            
            _has_scorer = hasattr(self.low_policy, 'score_candidates')
            curr_q_values = None

            
            curr_q_list = []
            _actual_cand_count = 0
            for i, idx in enumerate(valid_idx):
                tr = batch[idx]
                _is_cand = _used_cand_flags[i]
                cand_idx   = tr.get('candidate_indices')
                cand_feats = tr.get('candidate_local_feats')
                cur_node   = tr.get('current_node_idx')

                if _has_scorer and _is_cand and cand_idx is not None and len(cand_idx) > 0:
                    _actual_cand_count += 1
                    feat_t = None
                    if cand_feats is not None:
                        feat_t = torch.as_tensor(
                            cand_feats, dtype=torch.float32, device=self.device)
                        if feat_t.dim() == 2:
                            feat_t = feat_t.unsqueeze(0)  # [1, K, F]
                    scores = self.low_policy.score_candidates(
                        state_tensor[i:i+1],
                        goal_tensor[i:i+1],
                        list(cand_idx),
                        int(cur_node) if cur_node is not None else 0,
                        feat_t,
                    )  # [1, K]
                    local_act = _local_idx_list[i]
                    K = scores.size(1)
                    local_act = max(0, min(int(local_act), K - 1))
                    local_act_t = torch.tensor([[local_act]], device=self.device)
                    curr_q_list.append(scores.gather(1, local_act_t))
                else:
                    
                    out_fb = self.low_policy(state_tensor[i:i+1], goal_tensor[i:i+1])
                    q_fb = out_fb[0] if isinstance(out_fb, tuple) else out_fb
                    curr_q_list.append(q_fb.gather(1, actions[i:i+1]))

            curr_q = torch.cat(curr_q_list, dim=0)  # [B, 1]
            _use_score_candidates_count = _actual_cand_count

            
            if self._train_step > 200 and _use_score_candidates_count == 0:
                logger.warning(
                    f"[TrainAlert] step={self._train_step} "
                    f"本轮 low update 完全没用到 candidate_scorer "
                    f"(0/{len(valid_idx)}) — "
                    f"candidate样本数={_n_cand_samples} ratio={_cand_nonempty_ratio:.2f}"
                )

            if torch.isnan(curr_q).any() or torch.isinf(curr_q).any():
                logger.error(" Q值NaN/Inf，触发重置")
                self.reset_network_parameters()
                return 0.0

            
            with torch.no_grad():
                next_q_list = []
                for i, idx in enumerate(valid_idx):
                    tr = batch[idx]
                    next_cand_idx   = tr.get('next_candidate_indices')
                    next_cand_feats = tr.get('next_candidate_local_feats')
                    _next_cur_node  = tr.get('next_current_node_idx')
                    _next_cur_node_idx = int(_next_cur_node) if _next_cur_node is not None else 0

                    if (_has_scorer
                            and next_cand_idx is not None
                            and len(next_cand_idx) > 0):
                        nfeat_t = None
                        if next_cand_feats is not None:
                            nfeat_t = torch.as_tensor(
                                next_cand_feats, dtype=torch.float32, device=self.device)
                            if nfeat_t.dim() == 2:
                                nfeat_t = nfeat_t.unsqueeze(0)
                        
                        scores_on = self.low_policy.score_candidates(
                            next_state_tensor[i:i+1], goal_tensor[i:i+1],
                            list(next_cand_idx), _next_cur_node_idx, nfeat_t,
                        )
                        best_local = scores_on.argmax(dim=1, keepdim=True)
                        scores_tg = self.target_low_policy.score_candidates(
                            next_state_tensor[i:i+1], goal_tensor[i:i+1],
                            list(next_cand_idx), _next_cur_node_idx, nfeat_t,
                        )
                        next_q_list.append(scores_tg.gather(1, best_local))
                    else:
                        
                        m = tr.get('next_action_mask')
                        if m is None:
                            m_t = torch.ones(1, self.n_actions, device=self.device)
                        else:
                            m_t = torch.as_tensor(
                                m, dtype=torch.float32, device=self.device)
                            if m_t.dim() == 1: m_t = m_t.unsqueeze(0)
                        out_on = self.low_policy(
                            next_state_tensor[i:i+1], goal_tensor[i:i+1], m_t)
                        q_on = out_on[0] if isinstance(out_on, tuple) else out_on
                        best_act = q_on.argmax(dim=1, keepdim=True)
                        out_tg = self.target_low_policy(
                            next_state_tensor[i:i+1], goal_tensor[i:i+1], m_t)
                        q_tg = out_tg[0] if isinstance(out_tg, tuple) else out_tg
                        next_q_list.append(q_tg.gather(1, best_act))

                next_q   = torch.cat(next_q_list, dim=0)
                target_q = rewards + (1 - dones) * self.gamma * next_q

            
            
            
            _reward_scale = 10.0
            curr_q_scaled   = curr_q   / _reward_scale
            target_q_scaled = target_q / _reward_scale
            elementwise_loss = F.smooth_l1_loss(
                curr_q_scaled, target_q_scaled, reduction='none')
            if getattr(self, '_use_per', False) and _per_weights is not None:
                
                
                _w_full = torch.ones(len(batch), 1, device=self.device)
                for _i, _bi in enumerate(_per_weights):
                    _w_full[_i, 0] = float(_bi)
                
                _w_valid = _w_full[valid_mask]
                loss = (elementwise_loss * _w_valid).mean()
            else:
                loss = elementwise_loss.mean()
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(" Low-Level Loss NaN/Inf，跳过")
                return 0.0

            self.optimizer_low.zero_grad()
            loss.backward()

            
            self._train_step += 1
            if self._train_step % 50 == 0:
                with torch.no_grad():
                    _td_raw    = (curr_q - target_q).abs().mean().item()
                    _td_scaled = (curr_q / 30.0 - target_q / 30.0).abs().mean().item()
                _cand_scorer_ltd = getattr(self.low_policy, 'candidate_scorer', None)
                def _gn(m):
                    if m is None: return 0.0
                    t = sum(p.grad.data.norm(2).item()**2 for p in m.parameters()
                            if p.grad is not None)
                    return t**0.5
                _cg = _gn(_cand_scorer_ltd)

                
                _cand_ratio = _use_score_candidates_count / max(1, len(valid_idx))

                
                _k_vals = []
                for idx in valid_idx:
                    tr = batch[idx]
                    ci = tr.get('candidate_indices')
                    if ci is not None:
                        _k_vals.append(len(ci))
                    else:
                        _k_vals.append(0)
                _k_gt1 = sum(1 for k in _k_vals if k > 1)
                _k_mean = sum(_k_vals) / max(1, len(_k_vals))
                _k_eq1  = sum(1 for k in _k_vals if k == 1)
                _k_eq0  = sum(1 for k in _k_vals if k == 0)

                
                _fallback_cnt = len(valid_idx) - _use_score_candidates_count

                logger.info(
                    f"[LowTrainDiag] step={self._train_step} "
                    f"loss={loss.item():.4f} "
                    f"td_raw={_td_raw:.4f} "
                    f"cand_ratio={_cand_ratio:.3f} "
                    f"K_gt1={_k_gt1}/{len(valid_idx)} "
                    f"K_mean={_k_mean:.1f} "
                    f"K_eq0={_k_eq0} K_eq1={_k_eq1} "
                    f"fallback={_fallback_cnt} "
                    f"cand_grad={_cg:.6f}"
                )

                
                if not hasattr(self, '_cov_window'):
                    self._cov_window = {'cand': 0, 'total': 0, 'k_sum': 0, 'k_gt1': 0, 'steps': 0}
                self._cov_window['cand']   += _use_score_candidates_count
                self._cov_window['total']  += len(valid_idx)
                self._cov_window['k_sum']  += sum(_k_vals)
                self._cov_window['k_gt1']  += _k_gt1
                self._cov_window['steps']  += 1
                if self._train_step % 500 == 0:
                    _w = self._cov_window
                    logger.info(
                        f"[CoverageWindow] steps={_w['steps']} "
                        f"avg_cand_ratio={_w['cand']/max(1,_w['total']):.3f} "
                        f"avg_K={_w['k_sum']/max(1,_w['total']):.2f} "
                        f"K_gt1_ratio={_w['k_gt1']/max(1,_w['total']):.3f} "
                        f"→ {' scorer训练不足' if _w['cand']/max(1,_w['total']) < 0.3 else ' scorer覆盖正常'}"
                    )
                    self._cov_window = {'cand': 0, 'total': 0, 'k_sum': 0, 'k_gt1': 0, 'steps': 0}

            if getattr(self, '_use_per', False) and _per_indices is not None:
                with torch.no_grad():
                    _td = (curr_q - target_q).abs().detach().cpu().numpy().flatten()
                    _n  = min(len(_per_indices), len(_td))
                    self.low_memory.update_priorities(_per_indices[:_n], _td[:_n])
                    self._last_low_td = float(np.mean(_td[:_n])) if _n > 0 else None

            
            try:
                if self.encoder is not None and hasattr(self.encoder, 'tree_bias'):
                    logger.debug(f"[tree_bias] grad={self.encoder.tree_bias.grad} "
                                 f"val={self.encoder.tree_bias.item():.6f}")
            except Exception as _e:
                logger.debug(f"[tree_bias] error: {_e}")

            
            def _grad_norm(module):
                total, count = 0.0, 0
                for p in module.parameters():
                    if p.grad is not None:
                        total += p.grad.data.norm(2).item() ** 2
                        count += 1
                return (total ** 0.5) if count > 0 else 0.0

            if self._train_step % 100 == 0:
                _cand_scorer = getattr(self.low_policy, 'candidate_scorer', None)
                _actor       = getattr(self.low_policy, 'actor', None)
                _cand_grad   = _grad_norm(_cand_scorer) if _cand_scorer is not None else -1.0
                _actor_grad  = _grad_norm(_actor)       if _actor       is not None else -1.0
                _use_cand_cnt = getattr(self, '_use_score_candidates_count_last', 0)
                _batch_sz = len(batch)
                _ratio = _use_cand_cnt / max(1, _batch_sz)
                logger.info(
                    f"[LowTrainCheck] step={self._train_step} "
                    f"cand_grad={_cand_grad:.6f} actor_grad={_actor_grad:.6f} "
                    f"use_score_candidates={_use_cand_cnt}/{_batch_sz} "
                    f"candidate_ratio={_ratio:.2f} "
                    f"loss={loss.item():.4f}"
                )
                
                if _cand_grad < 1e-8:
                    self._zero_cand_grad_streak += 1
                    if self._zero_cand_grad_streak >= 3:
                        logger.warning(
                            f"[TrainAlert]  cand_grad 连续 {self._zero_cand_grad_streak*100} step 为 0! "
                            f"candidate_ratio={_ratio:.2f} — 请检查 replay 是否存入候选信息"
                        )
                else:
                    self._zero_cand_grad_streak = 0

            
            self._use_score_candidates_count_last = _use_score_candidates_count

            
            
            _actor_module = getattr(self.low_policy, 'actor', None)
            if (_actor_module is not None
                    and self._train_step > 500
                    and _cand_nonempty_ratio > 0.5):
                for _p in _actor_module.parameters():
                    if _p.grad is not None:
                        _p.grad.data.mul_(0.1)

            
            self.gradient_norms.append(
                sum(p.grad.norm().item() for p in self.low_policy.parameters() if p.grad is not None))
            _low_params = list(self.low_policy.parameters())
            _enc_params = list(self.encoder.parameters()) if self.encoder is not None else []
            nn.utils.clip_grad_norm_(_low_params + _enc_params, self.clip_grad_norm)
            self.optimizer_low.step()

            
            with torch.no_grad():
                if not hasattr(self, '_q_stats'): self._q_stats = {'count': 0}
                self._q_stats['count'] += 1
                if self._q_stats['count'] % 200 == 0:
                    _allq_max = curr_q.max().item()
                    if curr_q_values is not None:
                        try: _allq_max = curr_q_values.max().item()
                        except Exception: pass
                    logger.info(
                        f"[Q-Monitor] step={self._q_stats['count']} | "
                        f"CurrQ: mean={curr_q.mean():.2f} std={curr_q.std():.2f} "
                        f"min={curr_q.min():.2f} max={curr_q.max():.2f} | "
                        f"TargetQ mean={target_q.mean():.2f} | "
                        f"AllQ_max={_allq_max:.2f} | Loss={loss.item():.4f}"
                    )

            return loss.item()
        except Exception as e:
            logger.error(f"[Update Low Level] Error: {e}")
            import traceback; traceback.print_exc()
            return 0.0

    

    def _soft_update_target_networks(self):
        """软更新（tau=0.005），比hard update更稳定"""
        for tp, p in zip(self.target_high_policy.parameters(), self.high_policy.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.target_low_policy.parameters(), self.low_policy.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _hard_update_target_networks(self):
        """硬更新（保留备用，训练中不调用）"""
        self.target_high_policy.load_state_dict(self.high_policy.state_dict())
        self.target_low_policy.load_state_dict(self.low_policy.state_dict())

    

    def _update_epsilon(self):
        
        total_ep = getattr(self, 'total_episodes', 0)
        decay_episodes = getattr(self, 'epsilon_decay_episodes', 200)
        progress = min(total_ep / max(1, decay_episodes), 1.0)
        self.epsilon_high = self.epsilon_high_start + (self.epsilon_high_end - self.epsilon_high_start) * progress
        self.epsilon_low  = self.epsilon_low_start  + (self.epsilon_low_end  - self.epsilon_low_start)  * progress

    def clear_replay_buffers(self):
        """
        [改动3] 清空所有 replay buffer，用于从头积累含 candidate 信息的新样本。
        在代码改动后重新训练前调用，避免旧样本污染 scorer 训练。
        """
        cleared = []
        if hasattr(self, 'low_memory') and self.low_memory is not None:
            try:
                if hasattr(self.low_memory, 'clear'):
                    self.low_memory.clear()
                elif hasattr(self.low_memory, '_storage'):
                    self.low_memory._storage.clear()
                else:
                    import collections
                    self.low_memory = collections.deque(maxlen=getattr(self.low_memory, 'maxlen', 10000))
                cleared.append('low_memory')
            except Exception as _e:
                logger.warning(f"[ClearReplay] low_memory 清空失败: {_e}")
        if hasattr(self, 'success_memory') and self.success_memory is not None:
            try:
                if hasattr(self.success_memory, 'clear'):
                    self.success_memory.clear()
                cleared.append('success_memory')
            except Exception as _e:
                logger.warning(f"[ClearReplay] success_memory 清空失败: {_e}")
        if hasattr(self, 'elite_buffer') and self.elite_buffer is not None:
            try:
                if hasattr(self.elite_buffer, 'clear'):
                    self.elite_buffer.clear()
                cleared.append('elite_buffer')
            except Exception as _e:
                logger.warning(f"[ClearReplay] elite_buffer 清空失败: {_e}")
        if hasattr(self, 'high_memory') and self.high_memory is not None:
            try:
                if hasattr(self.high_memory, 'clear'):
                    self.high_memory.clear()
                cleared.append('high_memory')
            except Exception as _e:
                logger.warning(f"[ClearReplay] high_memory 清空失败: {_e}")
        logger.info(f"[ClearReplay]  已清空: {cleared}")
        
        self._ep_transitions          = []
        self._best_reward             = -1e9
        self._train_step_no_cand      = getattr(self, "_train_step_no_cand", 0) * 0
        self._cand_diag_count         = getattr(self, "_cand_diag_count", 0) * 0
        self._zero_cand_grad_streak   = getattr(self, "_zero_cand_grad_streak", 0) * 0

    def on_episode_end(self):
        """每个episode结束时调用，驱动ε衰减"""
        self.total_episodes = getattr(self, 'total_episodes', 0) + 1
        self._update_epsilon()

    def update_epsilon(self):
        """向后兼容（保留，但不再是主要驱动）"""
        pass

    

    def _log_training_stats(self):
        if self.high_loss_history and self.low_loss_history:
            logger.debug(
                f" 训练统计: HighLoss={np.mean(self.high_loss_history):.4f} "
                f"LowLoss={np.mean(self.low_loss_history):.4f} "
                f"GradNorm={np.mean(self.gradient_norms) if self.gradient_norms else 0:.2f} "
                f"ε_low={self.epsilon_low:.3f}"
            )

    def _record_train_stats(self, losses):
        """[RouteB-Stats] 每次更新落盘 loss / TD-error，便于绘制收敛/稳定性曲线。"""
        try:
            import os
            if getattr(self, '_loss_log_fh', None) is None:
                _path = 'loss_log.csv'
                try:
                    _cfg = getattr(self, 'config', None)
                    if isinstance(_cfg, dict):
                        _path = _cfg.get('hrl', {}).get('loss_log_path', _path)
                except Exception:
                    pass
                os.makedirs(os.path.dirname(os.path.abspath(_path)), exist_ok=True)
                _new = not os.path.exists(_path)
                self._loss_log_fh = open(_path, 'a', encoding='utf-8')
                self._loss_log_path = _path
                if _new:
                    self._loss_log_fh.write(
                        'update_count,high_loss,low_loss,total_loss,'
                        'high_td,low_td,epsilon_low,epsilon_high,grad_norm\n')
            def _f(x):
                return '' if x is None else x
            _gn = (float(np.mean(self.gradient_norms))
                   if getattr(self, 'gradient_norms', None) else 0.0)
            self._loss_log_fh.write(
                f"{self.update_count},"
                f"{_f(losses.get('high_loss'))},"
                f"{_f(losses.get('low_loss'))},"
                f"{losses.get('total_loss', 0.0)},"
                f"{_f(getattr(self, '_last_high_td', None))},"
                f"{_f(getattr(self, '_last_low_td', None))},"
                f"{_f(getattr(self, 'epsilon_low', None))},"
                f"{_f(getattr(self, 'epsilon_high', None))},"
                f"{_gn}\n")
            if self.update_count % 50 == 0:
                self._loss_log_fh.flush()
        except Exception as _e:
            logger.debug(f"[LossLog] skip: {_e}")
