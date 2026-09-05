#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Level Policy (第一步修复版 - 修复网络架构与起点选择)
"""

import torch
import torch.nn as nn
import logging
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


def deep_get(cfg, keys, default=None):
    if cfg is None: return default
    if isinstance(cfg, dict):
        for k in keys:
            if k in cfg and cfg[k] is not None: return cfg[k]
        for v in cfg.values():
            if isinstance(v, dict):
                found = deep_get(v, keys, None)
                if found is not None: return found
        return default
    for k in keys:
        if hasattr(cfg, k):
            val = getattr(cfg, k)
            if val is not None: return val
    return default


class HighLevelPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()

        use_cuda = deep_get(config, ["use_cuda"], False)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        self.hidden_dim = deep_get(config, ["hidden_dim"], 128)
        self.goal_dim = deep_get(config, ["goal_dim"], 64)
        
        env_cfg = config.get('environment', {})
        self.num_goals = env_cfg.get('nb_high_level_goals',
                         env_cfg.get('num_nodes', 28))
        self.gnn_output_dim = deep_get(config, ["gnn_output_dim"], self.hidden_dim)
        dropout = deep_get(config, ["dropout"], 0.1)

        
        self.state_projection = nn.Sequential(
            nn.Linear(self.gnn_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        
        self.q_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_goals)
        )

        
        
        self.start_selector = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        
        self.start_projection = nn.Sequential(
            nn.Linear(self.gnn_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Subgoal Generator
        self.subgoal_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.goal_dim),
            nn.Tanh()
        )

        # 5. [Task B] Goal Candidate Scorer
        
        
        
        self._goal_feat_dim = 9
        self.goal_candidate_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self._goal_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self.to(self.device)

    def forward(self, graph_emb, return_subgoal=True, return_value=False):
        z = self.state_projection(graph_emb)
        q_values = self.q_network(z)

        subgoal_emb = None
        if return_subgoal:
            subgoal_emb = self.subgoal_generator(z)
            if torch.isnan(subgoal_emb).any():
                subgoal_emb = torch.zeros_like(subgoal_emb)

        return q_values, subgoal_emb, None

    def score_goal_candidates(
        self,
        graph_emb: torch.Tensor,
        candidate_indices,
        candidate_node_embs: torch.Tensor,
        candidate_local_feats=None,
    ) -> torch.Tensor:
        """
        Task B: 对高层候选 goal 节点逐个打分。
        替代 q_network 对固定 [0..N-1] 输出 Q 的方式。

        Args:
            graph_emb          : [1, H] 或 [H]    全图状态嵌入（已经过 state_projection 之前）
            candidate_indices  : List[int], 长度 K   候选节点 ID
            candidate_node_embs: [K, H] 或 [1, N, H] 节点级嵌入（来自 encoder 输出）
            candidate_local_feats: np.ndarray[K,7] 或 Tensor[K,7], 可选

        Returns:
            scores: Tensor [1, K]   每个候选的评分
        """
        import torch
        device = graph_emb.device

        
        z_flat = graph_emb.view(-1)  # [H] or averaged
        if z_flat.size(0) != self.hidden_dim:
            
            z_flat = self.state_projection(graph_emb.view(1, -1)).squeeze(0)
        else:
            z_flat = self.state_projection(z_flat.unsqueeze(0)).squeeze(0)  # project

        K = len(candidate_indices)
        if K == 0:
            return torch.zeros(1, 0, device=device)

        
        if candidate_node_embs.dim() == 3:
            
            node_embs_2d = candidate_node_embs.squeeze(0)  # [N, H]
            cand_t = torch.tensor(candidate_indices, dtype=torch.long, device=device)
            cand_embs = node_embs_2d[cand_t]  # [K, H]
        elif candidate_node_embs.dim() == 2:
            
            cand_embs = candidate_node_embs
        else:
            cand_embs = candidate_node_embs.view(K, -1)

        if cand_embs.size(1) != self.hidden_dim:
            
            if not hasattr(self, '_cand_emb_proj') or self._cand_emb_proj.in_features != cand_embs.size(1):
                self._cand_emb_proj = nn.Linear(cand_embs.size(1), self.hidden_dim).to(device)
            cand_embs = self._cand_emb_proj(cand_embs)

        
        z_exp = z_flat.unsqueeze(0).expand(K, -1)  # [K, H]

        if candidate_local_feats is not None:
            if not isinstance(candidate_local_feats, torch.Tensor):
                feat_t = torch.tensor(candidate_local_feats, dtype=torch.float32, device=device)
            else:
                feat_t = candidate_local_feats.to(device)
            if feat_t.dim() == 1:
                feat_t = feat_t.unsqueeze(0).expand(K, -1)
            
            if feat_t.size(1) < self._goal_feat_dim:
                pad = torch.zeros(K, self._goal_feat_dim - feat_t.size(1), device=device)
                feat_t = torch.cat([feat_t, pad], dim=1)
            elif feat_t.size(1) > self._goal_feat_dim:
                feat_t = feat_t[:, :self._goal_feat_dim]
        else:
            feat_t = torch.zeros(K, self._goal_feat_dim, device=device)

        combined = torch.cat([z_exp, cand_embs, feat_t], dim=1)  # [K, 2H+feat]
        scores = self.goal_candidate_scorer(combined).squeeze(-1)  # [K]

        
        if not hasattr(self, '_scorer_log_cnt'): self._scorer_log_cnt = 0
        self._scorer_log_cnt += 1
        if self.training and self._scorer_log_cnt % 500 == 0:
            for _n, _p in self.goal_candidate_scorer.named_parameters():
                if _p.grad is not None:
                    logger.debug(
                        f"[HighScorerGrad] {_n} "
                        f"grad_norm={_p.grad.norm():.4f} "
                        f"param_norm={_p.data.norm():.4f}"
                    )

        
        if self._scorer_log_cnt % 20 == 0:
            import random as _r
            if _r.random() < 0.25:
                _k = min(3, K)
                _top_vals, _top_idx = scores.topk(_k)
                _top_nodes = [candidate_indices[i] for i in _top_idx.tolist()]
                logger.info(
                    f"[HighScorerMonitor] K={K} "
                    f"top3={list(zip(_top_nodes, [f'{v:.3f}' for v in _top_vals.tolist()]))}"
                )

        return scores.unsqueeze(0)  # [1, K]

    
    
    def select_goal(self, state_emb, valid_goals_mask, epsilon=0.1,
                    candidate_indices=None, candidate_node_embs=None,
                    candidate_local_feats=None):
        """
        统一高层目标选择入口。

        路径1（强制首选）: candidate_indices 非空时，全部走 score_goal_candidates()。
            - epsilon-greedy 在候选集内随机/贪心选择
            - 不再 fallback 到全图 q_network
        路径2（最终兜底）: 仅当候选集完全为空时才退回 q_network + mask。
        """
        import numpy as np
        import random as _random

        with torch.no_grad():
            _, goal_emb, _ = self.forward(state_emb, return_subgoal=True)

            
            if (candidate_indices is not None
                    and len(candidate_indices) > 0
                    and candidate_node_embs is not None):
                try:
                    scores = self.score_goal_candidates(
                        state_emb, candidate_indices,
                        candidate_node_embs, candidate_local_feats
                    ).squeeze(0)  # [K]

                    K = len(candidate_indices)
                    if epsilon > 0.0 and _random.random() < epsilon:
                        local_idx = _random.randint(0, K - 1)
                        _mode = 'epsilon_random'
                    else:
                        local_idx = int(scores.argmax().item())
                        _mode = 'greedy'

                    goal_node = candidate_indices[local_idx]
                    goal_idx = torch.tensor(goal_node, device=scores.device)

                    
                    if _random.random() < 0.01:
                        logger.info(
                            f"[EpsilonSample] layer=high mode={_mode} "
                            f"epsilon={epsilon:.3f} node={goal_node} "
                            f"K={K} score={scores[local_idx]:.3f} "
                            f"top3={[(candidate_indices[i], f'{scores[i]:.2f}') for i in scores.topk(min(3,K)).indices.tolist()]}"
                        )

                    if goal_idx.dim() == 0:
                        goal_idx = goal_idx.unsqueeze(0)
                    return goal_idx, goal_emb

                except Exception as _e:
                    logger.warning(f"[HighSelect] score_goal_candidates 失败: {_e}, 退回 q_network")

            
            if _random.random() < 0.01:
                logger.warning("[HighSelect] mode=full_logits_fallback (无候选或scorer异常)")

            q_values, _, _ = self.forward(state_emb, return_subgoal=False)
            q_values_flat = q_values.view(-1)

            if valid_goals_mask is not None:
                if isinstance(valid_goals_mask, np.ndarray):
                    valid_goals_mask = torch.FloatTensor(valid_goals_mask).to(q_values.device)
                mask_flat = valid_goals_mask.view(-1)
                if mask_flat.size(0) != q_values_flat.size(0):
                    min_dim = min(mask_flat.size(0), q_values_flat.size(0))
                    temp_mask = torch.zeros_like(q_values_flat)
                    temp_mask[:min_dim] = mask_flat[:min_dim]
                    mask_flat = temp_mask
                masked_q = q_values_flat.clone()
                masked_q[mask_flat == 0] = -1e9
            else:
                masked_q = q_values_flat
                mask_flat = torch.ones_like(q_values_flat)

            if np.random.rand() < epsilon:
                valid_indices = torch.nonzero(mask_flat > 0, as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    goal_idx = torch.tensor(0)
                elif valid_indices.numel() == 1:
                    goal_idx = valid_indices[0]
                else:
                    goal_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].squeeze()
            else:
                goal_idx = torch.argmax(masked_q)

            if goal_idx.dim() == 0:
                goal_idx = goal_idx.unsqueeze(0)
            return goal_idx, goal_emb

    def select_start_node(self, node_embeddings, target_emb, tree_mask, sample=True):
        """
         [修复版] 选择起点 (支持采样和梯度流)

        关键修复：
        1.  使用专门的投影网络处理节点嵌入
        2.  确保target_emb维度正确
        3.  修复mask处理逻辑
        """
        
        
        node_proj = self.start_projection(node_embeddings)

        num_nodes = node_proj.size(0)

        
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)

        
        if target_emb.size(1) != self.hidden_dim:
            
            if hasattr(self, 'state_projection'):
                target_emb = self.state_projection(target_emb)
            else:
                
                if not hasattr(self, 'target_proj_fix'):
                    self.target_proj_fix = nn.Linear(target_emb.size(1), self.hidden_dim).to(target_emb.device)
                target_emb = self.target_proj_fix(target_emb)

        
        target_expanded = target_emb.expand(num_nodes, -1)

        
        combined = torch.cat([node_proj, target_expanded], dim=1)

        
        scores = self.start_selector(combined).squeeze(-1)  # [num_nodes]

        
        if tree_mask is not None:
            
            if tree_mask.numel() == 1:
                
                scores = scores * tree_mask.item()
            elif tree_mask.size(0) == num_nodes:
                scores = scores.masked_fill(tree_mask == 0, -1e9)

        
        probs = torch.softmax(scores, dim=0)
        dist = Categorical(probs)

        
        if sample:
            start_node = dist.sample()
            log_prob = dist.log_prob(start_node)
            
            import random as _r7
            if _r7.random() < 0.01:
                logger.info(
                    f"[HighStartNode] start_node={start_node.item()} "
                    f"prob={probs[start_node].item():.3f} "
                    f"sample_mode=sample"
                )
            return start_node.item(), log_prob
        else:
            start_node = torch.argmax(probs)
            import random as _r7b
            if _r7b.random() < 0.01:
                logger.info(
                    f"[HighStartNode] start_node={start_node.item()} "
                    f"prob={probs[start_node].item():.3f} "
                    f"sample_mode=greedy"
                )
            return start_node.item(), None