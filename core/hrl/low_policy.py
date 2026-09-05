#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-Level Policy (TA-HRL v4.2 候选邻居评分版)
职责：给定 subgoal (destination / target)，结合 Tree-Aware Encoder 输出的全图节点特征，
使用 Goal-Conditioned Attention 机制选择最优下一跳节点。

v4.2 核心升级：候选邻居逐点评分
  - 新增 candidate_scorer：只对 controller 筛出的合法候选邻居逐点打分，
    替代原先对全图所有节点直接出 logits 的方式。
  - 新增 score_candidates()：接收 candidate_indices + local_feats，
    拼接 [cand_emb, goal, cur_emb, diff, local_feats] 后用 candidate_scorer 打分。
  - select_action() 优先走候选评分路径（路径1），
    若 candidate_indices 未传入则退回全图 logits（路径2，向后兼容）。

调用方式（推荐）：
    cand_info = low_level_controller.get_low_level_candidates()
    action, value = low_policy.select_action(
        state_emb=state_emb,
        goal_emb=goal_emb,
        action_mask=action_mask,
        epsilon=epsilon,
        candidate_indices=cand_info['indices'],
        current_node_idx=cand_info['current_node'],
        candidate_local_feats=cand_info['features'],   # np.ndarray[K, 6]
    )
"""

import numpy as np
import torch
import torch.nn as nn
import logging
import random
from typing import Optional, List

logger = logging.getLogger(__name__)


class GoalConditionedLowLevelPolicy(nn.Module):
    """
    Goal-Conditioned Low-Level Policy (TA-HRL v4.2)

    核心机制（双路径）：
    路径1（推荐）：候选邻居逐点评分
        用 score_candidates() 对 controller 预筛候选打分，
        特征 = [cand_emb, goal, cur_emb, diff, local_feats(6)]，
        彻底脱离"全图节点 ID 偏好"。

    路径2（兼容）：全图注意力 logits
        保留原 goal_attention + actor，用于 candidate_indices 未传入的场景。
    """

    def __init__(self, config):
        super().__init__()

        
        use_cuda = config.get('use_cuda', False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        
        self.state_dim = config.get('state_dim', 128)
        self.goal_dim = config.get('goal_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 128)

        
        env_cfg = config.get('environment', config.get('env', {}))
        self.action_dim = env_cfg.get('nb_low_level_actions', 50)
        dropout = config.get('dropout', 0.1)

        
        self.state_projection = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        self.goal_projection = nn.Sequential(
            nn.Linear(self.goal_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        
        
        #      = hidden_dim * 4 + 6
        self.candidate_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 4 + 6, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        
        self.goal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    # ==================================================================
    
    # ==================================================================
    def forward(self, state_emb: torch.Tensor,
                goal_emb: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None) -> tuple:
        """
        全图注意力前向传播（路径2，向后兼容）。
        :param state_emb:   [B, N, state_dim]
        :param goal_emb:    [B, goal_dim]
        :param action_mask: [B, N] 合法性掩码
        :return: (logits [B, action_dim], value [B, 1])
        """
        B = state_emb.size(0)

        
        state_proj = self.state_projection(state_emb)   # [B, N, H]
        if goal_emb is not None:
            goal_proj = self.goal_projection(goal_emb)  # [B, H]
        else:
            goal_proj = torch.zeros(B, self.hidden_dim, device=state_emb.device)

        
        query = goal_proj.unsqueeze(1)                  # [B, 1, H]
        attn_out, _ = self.goal_attention(query=query, key=state_proj, value=state_proj)

        
        attn_context = self.attn_norm(attn_out.squeeze(1) + goal_proj)  # [B, H]

        
        fused = torch.cat([attn_context, goal_proj], dim=-1)            # [B, H*2]

        
        logits = self.actor(fused)

        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        
        value = self.critic(fused)

        return logits, value

    # ==================================================================
    
    # ==================================================================
    def score_candidates(
        self,
        state_emb: torch.Tensor,
        goal_emb: Optional[torch.Tensor],
        candidate_indices: List[int],
        current_idx: int,
        candidate_local_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对候选邻居逐点打分，替代全图 logits。

        Args:
            state_emb:             [B, N, state_dim]  全图节点特征
            goal_emb:              [B, goal_dim]       子目标特征（可为 None）
            candidate_indices:     List[int], 长度 K   候选邻居节点 ID
            current_idx:           int                 当前节点 ID
            candidate_local_feats: [K, 6] 或 [B, K, 6] or None
                                   controller.get_low_level_candidates() 提供的局部特征
                                   含义：[delta_hop, is_tree_edge, bw_ratio,
                                          is_on_tree, avg_hop_to_undone, tabu_penalty]

        Returns:
            scores: [B, K]  每个候选的得分（未经 softmax）
        """
        B = state_emb.size(0)
        K = len(candidate_indices)

        if K == 0:
            return torch.zeros(B, 0, device=state_emb.device)

        
        state_proj = self.state_projection(state_emb)   # [B, N, H]
        if goal_emb is not None:
            goal_proj = self.goal_projection(goal_emb)  # [B, H]
        else:
            goal_proj = torch.zeros(B, self.hidden_dim, device=state_emb.device)

        
        cand_idx_t = torch.tensor(candidate_indices, dtype=torch.long, device=state_emb.device)
        cand_emb = state_proj[:, cand_idx_t, :]         # [B, K, H]

        
        cur_emb = state_proj[:, current_idx, :].unsqueeze(1).expand(-1, K, -1)  # [B, K, H]

        
        goal_expand = goal_proj.unsqueeze(1).expand(-1, K, -1)                  # [B, K, H]

        
        diff = cand_emb - cur_emb                                               # [B, K, H]

        
        feat = torch.cat([cand_emb, goal_expand, cur_emb, diff], dim=-1)

        
        if candidate_local_feats is not None:
            if isinstance(candidate_local_feats, np.ndarray):
                local_f = torch.from_numpy(candidate_local_feats).float().to(state_emb.device)
            else:
                local_f = candidate_local_feats.float().to(state_emb.device)
            
            if local_f.dim() == 2:
                local_f = local_f.unsqueeze(0).expand(B, -1, -1)
            feat = torch.cat([feat, local_f], dim=-1)   # [B, K, H*4+6]
        else:
            
            pad = torch.zeros(B, K, 6, device=state_emb.device)
            feat = torch.cat([feat, pad], dim=-1)        # [B, K, H*4+6]

        
        scores = self.candidate_scorer(feat).squeeze(-1) # [B, K]

        
        if not hasattr(self, '_scorer_call_cnt'): self._scorer_call_cnt = 0
        self._scorer_call_cnt += 1
        if self.training and self._scorer_call_cnt % 500 == 0:
            for _n, _p in self.candidate_scorer.named_parameters():
                if _p.grad is not None:
                    logger.debug(
                        f"[LowScorerGrad] {_n} "
                        f"grad_norm={_p.grad.norm():.4f} "
                        f"param_norm={_p.data.norm():.4f}"
                    )
            
            _g_total = sum(
                _p.grad.data.norm(2).item() ** 2
                for _p in self.candidate_scorer.parameters()
                if _p.grad is not None
            )
            logger.info(
                f"[LowScorerGrad] call={self._scorer_call_cnt} "
                f"total_grad_norm={_g_total**0.5:.6f}"
            )

        return scores

    # ==================================================================
    
    # ==================================================================
    def select_action(
        self,
        state_emb: torch.Tensor,
        goal_emb: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        epsilon: float = 0.0,
        
        candidate_indices: Optional[List[int]] = None,
        current_node_idx: Optional[int] = None,
        candidate_local_feats=None,
        **kwargs
    ):
        """
        动作选择统一入口，支持 epsilon-greedy 探索。

        路径1（候选邻居逐点评分，推荐）：
            当 candidate_indices 非空且 current_node_idx 已传入时激活。
            只在 controller 预筛的候选中选最高分，彻底避免全图节点 ID 偏好。

        路径2（全图 logits，向后兼容）：
            candidate_indices 为空或未传入时退回此路径。

        返回：(action: Tensor scalar, value: Tensor [1,1])
        """
        with torch.no_grad():
            
            if (candidate_indices is not None
                    and len(candidate_indices) > 0
                    and current_node_idx is not None):

                try:
                    scores = self.score_candidates(
                        state_emb, goal_emb,
                        candidate_indices, current_node_idx,
                        candidate_local_feats
                    )  # [B, K]

                    
                    _, value = self.forward(state_emb, goal_emb, action_mask)

                    K = len(candidate_indices)
                    cand_tensor = torch.tensor(
                        candidate_indices, dtype=torch.long, device=scores.device
                    )

                    valid_local_indices = list(range(K))
                    if action_mask is not None:
                        flat_mask = action_mask.reshape(-1).to(scores.device)
                        valid_local_indices = [
                            index for index, node in enumerate(candidate_indices)
                            if 0 <= int(node) < flat_mask.numel()
                            and bool(flat_mask[int(node)].item())
                        ]
                        if not valid_local_indices:
                            raise RuntimeError(
                                'candidate_indices do not intersect action_mask'
                            )
                        valid_local_mask = torch.zeros(
                            K, dtype=torch.bool, device=scores.device
                        )
                        valid_local_mask[valid_local_indices] = True
                        scores = scores.masked_fill(
                            ~valid_local_mask.unsqueeze(0), float('-inf')
                        )

                    if epsilon > 0.0 and random.random() < epsilon:
                        local_idx = random.choice(valid_local_indices)
                        _mode = 'epsilon'
                    else:
                        local_idx = int(scores.squeeze(0).argmax().item())
                        _mode = 'greedy'

                    
                    if random.random() < 0.01:
                        _scores_flat = scores.squeeze(0)
                        _topk = min(3, K)
                        _top_vals = _scores_flat.topk(_topk)
                        _top_info = [
                            (candidate_indices[ii], f'{sv:.2f}')
                            for ii, sv in zip(_top_vals.indices.tolist(), _top_vals.values.tolist())
                        ]
                        logger.info(
                            f"[EpsilonSample] layer=low mode={_mode} "
                            f"epsilon={epsilon:.3f} "
                            f"K={K} cur={current_node_idx} "
                            f"chosen={candidate_indices[local_idx]} "
                            f"top3={_top_info}"
                        )
                        _top_vals = _scores_flat.topk(_topk)
                    action = cand_tensor[local_idx]
                    return action, value

                except Exception as _ce:
                    logger.warning(f"[LowSelect] score_candidates 失败: {_ce}, 退回 logits")

            
            if random.random() < 0.01:
                logger.warning("[LowSelect] mode=full_logits_fallback")
            logits, value = self.forward(state_emb, goal_emb, action_mask)

            if action_mask is not None:
                valid_actions = (action_mask.squeeze() > 0).nonzero(as_tuple=True)[0]
            else:
                valid_actions = torch.arange(logits.size(-1), device=logits.device)

            if epsilon > 0.0 and random.random() < epsilon:
                if len(valid_actions) > 0:
                    idx = random.randint(0, len(valid_actions) - 1)
                    action = valid_actions[idx]
                else:
                    logger.warning("[LowPolicy] action_mask 全0，无合法动作，返回-1")
                    action = torch.tensor(-1, device=logits.device)
            else:
                action = torch.argmax(logits, dim=-1).squeeze()
                if len(valid_actions) > 0 and action not in valid_actions:
                    logger.warning(
                        f"[LowPolicy] 贪心选中非法动作 {action.item()}，"
                        f"强制修正为最高Q的合法动作"
                    )
                    valid_logits = logits.squeeze()[valid_actions]
                    action = valid_actions[valid_logits.argmax()]
                elif len(valid_actions) == 0:
                    logger.warning("[LowPolicy] action_mask 全0，无合法动作，返回-1")
                    action = torch.tensor(-1, device=logits.device)

            return action, value

    def reset_parameters(self):
        """重置网络参数（用于异常自愈）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)


# ============================================

# ============================================

class LowLevelPolicy(nn.Module):
    """
    低层策略网络（向后兼容版）
    底层已映射为 GoalConditionedLowLevelPolicy。
    """

    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        logger.warning(
            " 正在使用向后兼容的 LowLevelPolicy 接口，"
            "底层已映射为 GoalConditionedLowLevelPolicy"
        )

        config = {
            'state_dim': input_dim,
            'goal_dim': 64,
            'hidden_dim': hidden_dim,
            'environment': {
                'nb_low_level_actions': action_dim
            },
            'use_cuda': False,
            'dropout': 0.1
        }
        self.policy = GoalConditionedLowLevelPolicy(config)
        self.actor = self.policy.actor
        self.critic = self.policy.critic

    def forward(self, state, action_mask=None):
        """兼容旧的前向传播（没有 goal_emb 的情况）"""
        if state.dim() == 2:
            state = state.unsqueeze(1)
        logits, value = self.policy(state, None, action_mask)
        return logits, value, value

    def select_action(self, state, action_mask=None, epsilon=0.0,
                      candidate_indices=None, current_node_idx=None,
                      candidate_local_feats=None, **kwargs):
        """向后兼容接口，支持路径1/2自动切换"""
        if state.dim() == 2:
            state = state.unsqueeze(1)
        return self.policy.select_action(
            state, None, action_mask, epsilon=epsilon,
            candidate_indices=candidate_indices,
            current_node_idx=current_node_idx,
            candidate_local_feats=candidate_local_feats,
            **kwargs
        )
