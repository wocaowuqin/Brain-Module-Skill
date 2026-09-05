"""
ablation_encoder.py
====================
消融实验用编码器变体，通过 variant 参数控制：

  variant='full'         → 完整 TA-HRL v4（双流 + dest_mask）
  variant='no_tree'      → 去掉树流，只保留拓扑流
  variant='no_dest_mask' → 去掉 dest_mask 目标感知
  variant='single_stream'→ 最简单的单流GNN基线
  variant='mlp'          → 纯MLP，完全不使用图结构

用法示例（在 agent_base.py 里）:
  from ablation_encoder import AblationEncoder
  self.encoder = AblationEncoder(variant='no_tree', node_dim=24, edge_dim=5, hidden_dim=128)
"""

import torch
import torch.nn as nn
import logging
from torch_geometric.nn import TransformerConv, global_mean_pool

logger = logging.getLogger(__name__)


class AblationEncoder(nn.Module):
    """
    统一消融实验编码器
    通过 variant 参数切换不同的消融配置，保持其他超参完全一致
    """

    VALID_VARIANTS = ('full', 'no_tree', 'no_dest_mask', 'single_stream', 'mlp')

    def __init__(self, node_dim=24, edge_dim=5, hidden_dim=128, num_heads=4,
                 variant='full', req_dim=0):
        super().__init__()
        assert variant in self.VALID_VARIANTS, \
            f"variant 必须是 {self.VALID_VARIANTS} 之一，当前: {variant}"
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.req_dim = req_dim
        self._mlp_diag_logged = False

        
        if variant != 'mlp':
            self.topo_input_norm = nn.LayerNorm(node_dim)
            self.tree_input_norm = nn.LayerNorm(node_dim)

        
        if variant != 'mlp':
            self.topo_transformer1 = TransformerConv(
                in_channels=node_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=edge_dim,
                beta=True
            )
            self.topo_transformer2 = TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=edge_dim,
                beta=True
            )

        
        if variant in ('full', 'no_dest_mask'):
            self.tree_transformer1 = TransformerConv(
                in_channels=node_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                beta=True
            )
            self.tree_transformer2 = TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                beta=True
            )
            
            
            self.tree_bias = nn.Parameter(torch.tensor(0.0))
            self.fusion_norm = nn.LayerNorm(hidden_dim)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif variant == 'mlp':
            
            
            self.mlp_layers = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        
        if variant == 'full':
            self.dest_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
            )

        
        if req_dim > 0:
            self.req_fc = nn.Linear(req_dim, hidden_dim)
            self.req_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.req_norm = nn.LayerNorm(hidden_dim)
        else:
            self.req_fc = None

    

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                tree_edge_index=None, dest_mask=None, req_vec=None):

        
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        
        if self.variant == 'mlp':
            if not self._mlp_diag_logged:
                logger.info(
                    "[MLP-DIAG] AblationEncoder variant=mlp: graph message passing is disabled; "
                    "edge_index/edge_attr/tree_edge_index/dest_mask are ignored. "
                    "supplied: edge_index=%s edge_attr=%s tree_edge_index=%s dest_mask=%s req_vec=%s",
                    edge_index is not None,
                    edge_attr is not None,
                    tree_edge_index is not None,
                    dest_mask is not None,
                    req_vec is not None,
                )
                self._mlp_diag_logged = True
            node_emb = self.mlp_layers(x)
            if self.req_fc is not None and req_vec is not None:
                if req_vec.dim() == 1:
                    req_vec = req_vec.unsqueeze(0)
                req_emb = self.req_fc(req_vec)
                req_expanded = (req_emb[batch] if batch is not None
                                else req_emb.expand(node_emb.size(0), -1))
                node_emb = self.req_norm(
                    self.req_fusion(torch.cat([node_emb, req_expanded], dim=-1))
                )
            return node_emb
        

        
        
        x_topo = self.topo_input_norm(x)
        topo_emb = self.topo_transformer1(x_topo, edge_index, edge_attr)
        topo_emb = torch.nn.functional.gelu(topo_emb)
        topo_emb = self.topo_transformer2(topo_emb, edge_index, edge_attr)

        
        if self.variant in ('full', 'no_dest_mask'):
            
            if tree_edge_index is None or tree_edge_index.size(1) == 0:
                nodes = torch.arange(x.size(0), device=x.device)
                tree_edge_index = torch.stack([nodes, nodes], dim=0)

            x_tree = self.tree_input_norm(x)
            tree_emb = self.tree_transformer1(x_tree, tree_edge_index)
            tree_emb = torch.nn.functional.gelu(tree_emb)
            tree_emb = self.tree_transformer2(tree_emb, tree_edge_index)

            gate = torch.sigmoid(self.tree_bias)
            fused = gate * tree_emb + (1.0 - gate) * topo_emb
            fused = self.fusion_norm(fused)
            node_emb = self.fusion_mlp(fused)

        else:
            
            node_emb = self.output_proj(topo_emb)

        
        if self.req_fc is not None and req_vec is not None:
            if req_vec.dim() == 1:
                req_vec = req_vec.unsqueeze(0)
            req_emb = self.req_fc(req_vec)                          # [1 or B, H]
            if batch is not None:
                req_expanded = req_emb[batch]                       # [N, H]
            else:
                req_expanded = req_emb.expand(node_emb.size(0), -1)
            node_emb = self.req_norm(
                self.req_fusion(torch.cat([node_emb, req_expanded], dim=-1))
            )

        
        if self.variant == 'full' and dest_mask is not None and dest_mask.any():
            dest_pool = node_emb[dest_mask].mean(dim=0, keepdim=True).unsqueeze(0)
            query = node_emb.unsqueeze(0)
            attn_out, _ = self.dest_cross_attn(query=query, key=dest_pool, value=dest_pool)
            node_emb = node_emb + attn_out.squeeze(0)

        return node_emb

    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None,
                            tree_edge_index=None, dest_mask=None, req_vec=None):
        node_embeddings = self.forward(x, edge_index, edge_attr, batch,
                                       tree_edge_index, dest_mask, req_vec)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(node_embeddings, batch)
