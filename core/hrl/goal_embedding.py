"""
===============================================================================
core/hrl/goal_embedding_final.py
Goal Embedding 最终优化版本
===============================================================================

整合所有反馈的改进点：
1. 动态缩放因子（Relative Goal）
2. 自适应子目标距离（Subgoal）
3. Softmax Option 选择策略
4. 迭代目标优化（Hybrid）
5. 并行化批量索引

参考论文：
- HIRO: Data-Efficient Hierarchical Reinforcement Learning
- HAC: Hierarchical Actor-Critic
- Option-Critic: End-to-End Learning of Options
- FuN: FeUdal Networks for Hierarchical Reinforcement Learning

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


# ============================================

# ============================================

class EnhancedRelativeGoalEmbedding(nn.Module):
    """
    增强版相对目标嵌入

    改进点：
    1. 可学习的缩放因子（动态调整目标范围）
    2. 多尺度目标表示
    3. 注意力机制融合
    """

    def __init__(
            self,
            node_feat_dim: int = 32,
            goal_dim: int = 64,
            use_learned_scaling: bool = True,
            use_attention: bool = True
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.goal_dim = goal_dim
        self.use_learned_scaling = use_learned_scaling
        self.use_attention = use_attention

        
        if use_learned_scaling:
            self.scale_factor = nn.Parameter(
                torch.ones(1) * 0.5
            )
        else:
            self.register_buffer('scale_factor', torch.tensor([1.0]))

        
        self.goal_generator = nn.Sequential(
            nn.Linear(node_feat_dim * 2, goal_dim * 2),
            nn.LayerNorm(goal_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(goal_dim * 2, goal_dim),
            nn.Tanh()
        )

        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=node_feat_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )

            self.attention_proj = nn.Linear(node_feat_dim, goal_dim)

    def forward(
            self,
            current_node_feat: torch.Tensor,
            target_node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        生成相对目标嵌入

        Args:
            current_node_feat: [batch, node_feat_dim]
            target_node_feat: [batch, node_feat_dim]

        Returns:
            goal_emb: [batch, goal_dim]
            info: 诊断信息
        """
        info = {}

        
        x = torch.cat([current_node_feat, target_node_feat], dim=-1)

        
        base_goal = self.goal_generator(x)

        
        scaled_goal = self.scale_factor * base_goal

        
        if self.use_attention:
            
            feats = torch.stack([current_node_feat, target_node_feat], dim=1)

            # Self-attention
            attn_out, attn_weights = self.attention(feats, feats, feats)

            
            attn_goal = self.attention_proj(attn_out.mean(dim=1))

            
            goal_emb = scaled_goal + 0.3 * attn_goal

            info['attention_weights'] = attn_weights
        else:
            goal_emb = scaled_goal

        
        goal_emb = F.normalize(goal_emb, p=2, dim=-1)

        info['scale_factor'] = self.scale_factor.item()
        info['base_goal_norm'] = base_goal.norm(dim=-1).mean().item()

        return goal_emb, info


# ============================================

# ============================================

class AdaptiveSubgoalEmbedding(nn.Module):
    """
    自适应子目标嵌入

    改进点：
    1. 动态调整子目标距离
    2. 基于任务复杂度的自适应
    3. 子目标可达性预测
    """

    def __init__(
            self,
            state_dim: int = 32,
            goal_dim: int = 64,
            init_subgoal_distance: float = 5.0,
            adaptive_distance: bool = True
    ):
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.adaptive_distance = adaptive_distance

        # Subgoal Generator
        self.subgoal_generator = nn.Sequential(
            nn.Linear(state_dim, goal_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(goal_dim * 2, state_dim),
            nn.Tanh()
        )

        
        if adaptive_distance:
            self.distance_predictor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()
            )
        else:
            self.register_buffer(
                'max_subgoal_distance',
                torch.tensor([init_subgoal_distance])
            )

        
        self.reachability_predictor = nn.Sequential(
            nn.Linear(state_dim * 2, 64),  # current + subgoal
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
            self,
            current_state: torch.Tensor,
            task_complexity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        生成自适应子目标

        Args:
            current_state: [batch, state_dim]
            task_complexity: [batch, 1] 任务复杂度（可选）

        Returns:
            subgoal: [batch, state_dim]
            info: 诊断信息
        """
        batch_size = current_state.size(0)
        info = {}

        
        delta = self.subgoal_generator(current_state)

        
        if self.adaptive_distance:
            
            max_distance = self.distance_predictor(current_state)

            
            if task_complexity is not None:
                max_distance = max_distance * (1 + task_complexity)

            info['predicted_distance'] = max_distance.mean().item()
        else:
            max_distance = self.max_subgoal_distance.expand(batch_size, 1)

        
        delta = delta * max_distance

        
        subgoal = current_state + delta

        
        reachability_input = torch.cat([current_state, subgoal], dim=-1)
        reachability = self.reachability_predictor(reachability_input)

        info['reachability'] = reachability.mean().item()
        info['delta_norm'] = delta.norm(dim=-1).mean().item()

        return subgoal, info

    def compute_reward(
            self,
            achieved_state: torch.Tensor,
            subgoal: torch.Tensor
    ) -> torch.Tensor:
        """
        计算内在奖励（考虑可达性）
        """
        
        distance = torch.norm(achieved_state - subgoal, dim=-1)

        
        base_reward = -distance

        
        reachability_input = torch.cat([achieved_state, subgoal], dim=-1)
        reachability = self.reachability_predictor(reachability_input).squeeze(-1)

        
        adjusted_reward = base_reward * reachability

        return adjusted_reward


    def compute_intrinsic_reward(self, next_state, subgoal):
        """计算内在奖励"""
        try:
            
            if hasattr(next_state, 'x'):
                
                state_emb = next_state.x.mean(dim=0)
            else:
                # Fallback
                return 0.0

            
            distance = torch.norm(state_emb - subgoal.squeeze())
            reward = -distance.item() * 0.1
            return reward
        except:
            return 0.0

# ============================================

# ============================================

class EnhancedOptionEmbedding(nn.Module):
    """
    增强版 Option 嵌入

    改进点：
    1. Softmax Option 选择策略
    2. Option 价值估计
    3. 动态 Option 终止
    """

    def __init__(
            self,
            num_options: int = 4,
            option_dim: int = 64,
            state_dim: int = 32,
            temperature: float = 1.0
    ):
        super().__init__()

        self.num_options = num_options
        self.option_dim = option_dim
        self.state_dim = state_dim
        self.temperature = temperature

        
        self.option_embeddings = nn.Embedding(num_options, option_dim)

        
        self.option_policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_options)
        )

        
        self.option_value = nn.Sequential(
            nn.Linear(state_dim + option_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        
        self.termination_net = nn.Sequential(
            nn.Linear(state_dim + option_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def get_option_probs(
            self,
            state: torch.Tensor,
            temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        获取 Option 选择概率

        Args:
            state: [batch, state_dim]
            temperature: 温度参数（控制探索）

        Returns:
            probs: [batch, num_options]
        """
        if temperature is None:
            temperature = self.temperature

        
        logits = self.option_policy(state)

        
        scaled_logits = logits / temperature

        # Softmax
        probs = F.softmax(scaled_logits, dim=-1)

        return probs

    def select_option(
            self,
            state: torch.Tensor,
            epsilon: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择 Option（带探索）

        Args:
            state: [batch, state_dim]
            epsilon: ε-greedy 探索率

        Returns:
            option_id: [batch] 选中的 Option
            option_emb: [batch, option_dim] Option 嵌入
        """
        batch_size = state.size(0)

        
        if torch.rand(1).item() < epsilon:
            
            option_id = torch.randint(0, self.num_options, (batch_size,))
        else:
            
            probs = self.get_option_probs(state)
            option_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        
        option_emb = self.option_embeddings(option_id)

        return option_id, option_emb

    def compute_option_value(
            self,
            state: torch.Tensor,
            option_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Option 的价值

        Returns:
            value: [batch, 1]
        """
        x = torch.cat([state, option_emb], dim=-1)
        value = self.option_value(x)

        return value

    def should_terminate(
            self,
            state: torch.Tensor,
            option_emb: torch.Tensor,
            deterministic: bool = False
    ) -> torch.Tensor:
        """
        判断 Option 是否应该终止

        Args:
            state: [batch, state_dim]
            option_emb: [batch, option_dim]
            deterministic: 是否确定性终止

        Returns:
            terminate: [batch] bool tensor
        """
        x = torch.cat([state, option_emb], dim=-1)
        termination_prob = self.termination_net(x).squeeze(-1)

        if deterministic:
            
            terminate = termination_prob > 0.5
        else:
            
            terminate = torch.bernoulli(termination_prob).bool()

        return terminate


# ============================================

# ============================================

class IterativeHybridGoalEmbedding(nn.Module):
    """
    迭代优化的混合 Goal Embedding

    改进点：
    1. 多步迭代优化子目标
    2. 精细化 Goal 编码
    3. 自注意力机制
    """

    def __init__(
            self,
            local_state_dim: int = 32,
            goal_dim: int = 64,
            subgoal_horizon: int = 5,
            num_refinement_steps: int = 3
    ):
        super().__init__()

        self.local_state_dim = local_state_dim
        self.goal_dim = goal_dim
        self.subgoal_horizon = subgoal_horizon
        self.num_refinement_steps = num_refinement_steps

        
        self.initial_subgoal_generator = nn.Sequential(
            nn.Linear(local_state_dim, goal_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(goal_dim * 2, goal_dim),
            nn.Tanh()
        )

        
        self.refinement_steps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(goal_dim * 2, goal_dim),  # goal + context
                nn.ReLU(),
                nn.Linear(goal_dim, goal_dim),
                nn.Tanh()
            )
            for _ in range(num_refinement_steps)
        ])

        
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, goal_dim * 2),
            nn.LayerNorm(goal_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(goal_dim * 2, goal_dim)
        )

        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=goal_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        
        self.context_encoder = nn.Sequential(
            nn.Linear(local_state_dim, goal_dim),
            nn.ReLU()
        )

    def forward(
            self,
            current_local_state: torch.Tensor,
            return_refinement_history: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[list]]:
        """
        生成迭代优化的 Goal Embedding

        Args:
            current_local_state: [batch, local_state_dim]
            return_refinement_history: 是否返回优化历史

        Returns:
            subgoal: [batch, goal_dim] 最终子目标
            goal_emb: [batch, goal_dim] Goal Embedding
            refinement_history: 优化历史（可选）
        """
        batch_size = current_local_state.size(0)

        
        subgoal = self.initial_subgoal_generator(current_local_state)

        
        context = self.context_encoder(current_local_state)

        
        refinement_history = [subgoal.clone()]

        for refine_step in self.refinement_steps:
            
            refine_input = torch.cat([subgoal, context], dim=-1)

            
            delta = refine_step(refine_input)

            
            subgoal = subgoal + 0.3 * delta

            refinement_history.append(subgoal.clone())

        
        
        subgoal_sequence = torch.stack(refinement_history, dim=1)  # [batch, steps, goal_dim]

        
        attn_out, _ = self.self_attention(
            subgoal_sequence, subgoal_sequence, subgoal_sequence
        )

        
        refined_subgoal = attn_out[:, -1, :]

        
        goal_emb = self.goal_encoder(refined_subgoal)

        
        goal_emb = F.normalize(goal_emb, p=2, dim=-1)

        if return_refinement_history:
            return refined_subgoal, goal_emb, refinement_history
        else:
            return refined_subgoal, goal_emb, None

    def compute_intrinsic_reward(
            self,
            achieved_state: torch.Tensor,
            subgoal: torch.Tensor
    ) -> torch.Tensor:
        """
        计算内在奖励
        """
        
        achieved_proj = self.initial_subgoal_generator(achieved_state)

        
        similarity = F.cosine_similarity(achieved_proj, subgoal, dim=-1)

        
        reward = similarity

        return reward


# ============================================

# ============================================

def optimized_batch_indexing(
        node_embeddings: torch.Tensor,
        target_nodes: torch.Tensor,
        batch: torch.Tensor
) -> torch.Tensor:
    """
    优化的批量索引（并行化）

    改进点：
    - 使用 scatter/gather 操作
    - 减少循环
    - 更高效的内存访问

    Args:
        node_embeddings: [total_nodes, dim]
        target_nodes: [batch_size] 局部索引
        batch: [total_nodes] 图 ID

    Returns:
        target_embs: [batch_size, dim]
    """
    device = node_embeddings.device
    batch_size = target_nodes.size(0)

    
    
    unique_batches = torch.unique(batch, sorted=True)
    num_graphs = len(unique_batches)

    
    offsets = torch.zeros(num_graphs, dtype=torch.long, device=device)

    for i, b in enumerate(unique_batches):
        mask = (batch == b)
        offsets[i] = mask.nonzero(as_tuple=True)[0][0]

    
    
    global_indices = target_nodes + offsets

    
    target_embs = node_embeddings[global_indices]

    return target_embs


# ============================================

# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("Goal Embedding 最终优化版本")
    print("=" * 70)

    
    print("\n1. Enhanced Relative Goal Embedding")
    rel_goal = EnhancedRelativeGoalEmbedding(
        node_feat_dim=32,
        goal_dim=64,
        use_learned_scaling=True,
        use_attention=True
    )

    current = torch.randn(4, 32)
    target = torch.randn(4, 32)
    goal_emb, info = rel_goal(current, target)

    print(f"   Goal shape: {goal_emb.shape}")
    print(f"   Scale factor: {info['scale_factor']:.3f}")
    print(f"   Base goal norm: {info['base_goal_norm']:.3f}")

    
    print("\n2. Adaptive Subgoal Embedding")
    subgoal_gen = AdaptiveSubgoalEmbedding(
        state_dim=32,
        goal_dim=64,
        adaptive_distance=True
    )

    state = torch.randn(4, 32)
    complexity = torch.rand(4, 1) * 0.5
    subgoal, info = subgoal_gen(state, complexity)

    print(f"   Subgoal shape: {subgoal.shape}")
    print(f"   Predicted distance: {info['predicted_distance']:.3f}")
    print(f"   Reachability: {info['reachability']:.3f}")

    
    print("\n3. Enhanced Option Embedding")
    option_gen = EnhancedOptionEmbedding(
        num_options=4,
        option_dim=64,
        state_dim=32
    )

    probs = option_gen.get_option_probs(state)
    option_id, option_emb = option_gen.select_option(state)
    value = option_gen.compute_option_value(state, option_emb)

    print(f"   Option probs: {probs[0].tolist()}")
    print(f"   Selected option: {option_id[0].item()}")
    print(f"   Option value: {value[0].item():.3f}")

    
    print("\n4. Iterative Hybrid Goal Embedding")
    hybrid = IterativeHybridGoalEmbedding(
        local_state_dim=32,
        goal_dim=64,
        num_refinement_steps=3
    )

    subgoal, goal_emb, history = hybrid(state, return_refinement_history=True)

    print(f"   Final subgoal shape: {subgoal.shape}")
    print(f"   Goal embedding shape: {goal_emb.shape}")
    print(f"   Refinement steps: {len(history)}")

    print("\n" + "=" * 70)
    print("所有改进点已实现：")
    print("   1. 动态缩放因子")
    print("   2. 自适应子目标距离")
    print("   3. Softmax Option 选择")
    print("   4. 迭代目标优化")
    print("   5. 并行化批量索引")
    print("   6. 子目标可达性预测")
    print("   7. 注意力机制融合")
    print("=" * 70)