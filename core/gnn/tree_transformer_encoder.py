import torch
import torch.nn as nn
import logging
from torch_geometric.nn import TransformerConv, global_mean_pool

logger = logging.getLogger(__name__)


class TreeTransformerEncoder(nn.Module):
    """
    [v3 / 方向三] 在 v2(单拓扑流 + 目的集合注意力)基础上，新增
    “放置↔路由耦合编码”：从同一次 GNN 输出分出放置语义 h_place 与路由语义
    h_route 两个投影头，再用低秩双线性项刻画二者耦合，并可选输出辅助预测
    (在某节点放 VNF 会引入多少增量路由代价)。

    设计要点(对应讨论的六步)：
      - 只跑一遍 GNN，双头是投影而非独立消息传递流(规避树感知双流翻车史)。
      - 耦合是候选集约束在“放置/路由联合可行性”上的延伸，不是第二创新。
      - coupling_mode 三档对应 A/B/C 消融，且 B 与 C 参数自动对齐：
          'off'  = A：单流，等价 v2(无双头、无耦合)
          'dual' = B：双头 h_place/h_route，但只做线性相加组合(无交叉项)
          'full' = C：双头 + 低秩双线性耦合(有交叉项)  ← 方向三完整版
      - use_aux_head：辅助监督头，缓解耦合项只靠 RL 回报学不动的问题。
    """
    def __init__(self, node_dim=28, edge_dim=5, hidden_dim=128, num_heads=4, req_dim=0,
                 dest_attn_mode='per_dest',
                 coupling_mode='full', coupling_rank=None, use_aux_head=True):
        """
        Args:
            req_dim: 请求特征维度。>0 时启用请求融合。默认0保持向后兼容。
            dest_attn_mode: 'per_dest'(逐目的 set attention) | 'mean'(目的质心，退化对照)。
            coupling_mode:  'off'|'dual'|'full'，见类文档。消融时切此开关。
            coupling_rank:  低秩耦合的秩 r。None 时自动设为 ~2H/3，使 'dual' 与 'full'
                            参数量近似对齐(保证消融差异只来自“交叉项”本身)。
            use_aux_head:   是否启用辅助预测头(预测增量路由代价)。'off' 模式下强制关闭。
        """
        super().__init__()
        self.req_dim = req_dim
        assert dest_attn_mode in ('per_dest', 'mean'), dest_attn_mode
        self.dest_attn_mode = dest_attn_mode
        assert coupling_mode in ('off', 'dual', 'full'), coupling_mode
        self.coupling_mode = coupling_mode
        self.use_aux_head = bool(use_aux_head) and coupling_mode != 'off'

        self.topo_input_norm = nn.LayerNorm(node_dim)

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

        self.dest_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
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

        
        if coupling_mode in ('dual', 'full'):
            
            self.place_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim))
            self.route_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim))

            if coupling_mode == 'full':
                
                r = coupling_rank if coupling_rank is not None else max(8, (2 * hidden_dim) // 3)
                self.coupling_rank = r
                self.coupling_A = nn.Linear(hidden_dim, r, bias=False)
                self.coupling_B = nn.Linear(hidden_dim, r, bias=False)
                self.coupling_up = nn.Linear(r, hidden_dim)
            else:
                
                self.combine_linear = nn.Linear(hidden_dim * 2, hidden_dim)

            
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim))
            self.fuse_norm = nn.LayerNorm(hidden_dim)

            if self.use_aux_head:
                
                self.aux_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1))

    def _apply_coupling(self, node_emb):
        """方向三核心：返回 (node_emb_final, aux_pred)。aux_pred 可能为 None。"""
        h_place = self.place_proj(node_emb)
        h_route = self.route_proj(node_emb)
        if self.coupling_mode == 'full':
            
            coup = self.coupling_up(self.coupling_A(h_place) * self.coupling_B(h_route))
        else:  # 'dual'
            coup = self.combine_linear(torch.cat([h_place, h_route], dim=-1))
        fused = self.fuse(torch.cat([node_emb, coup], dim=-1))
        node_emb = self.fuse_norm(node_emb + fused)
        aux_pred = self.aux_head(h_place).squeeze(-1) if self.use_aux_head else None
        return node_emb, aux_pred

    def forward(self, x, edge_index, edge_attr=None, batch=None, tree_edge_index=None,
                dest_mask=None, req_vec=None, return_aux=False):
        
        try:
            dest_count = int(dest_mask.sum().item()) if dest_mask is not None else None
            tree_edges = int(tree_edge_index.size(1)) if tree_edge_index is not None else None
            req_shape = tuple(req_vec.shape) if req_vec is not None else None
            edge_attr_shape = tuple(edge_attr.shape) if edge_attr is not None else None
            reach_sum = None
            if x is not None and x.dim() == 2 and x.size(1) >= 28:
                reach_sum = float(x[:, 24:28].abs().sum().item())
            diag_key = (
                dest_count is not None and dest_count > 0,
                tree_edges is not None and tree_edges > 0,
                req_shape is not None,
                reach_sum is not None and reach_sum > 0.0,
            )
            seen = getattr(self, '_debug_seen_keys', set())
            if diag_key not in seen and len(seen) < 6:
                logger.info(
                    "[GNN-DIAG encoder] class=TreeTransformerEncoder "
                    f"x={tuple(x.shape)} edge_attr={edge_attr_shape} "
                    f"dest_mask_sum={dest_count} tree_edge_index_edges={tree_edges} "
                    f"req_vec_shape={req_shape} reach_feat_abs_sum={reach_sum} "
                    f"coupling_mode={self.coupling_mode} "
                    "tree_edge_index_used_directly=False"
                )
                seen.add(diag_key)
                self._debug_seen_keys = seen
        except Exception as exc:
            logger.warning(f"[GNN-DIAG encoder] failed: {exc}")

        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        x_topo = self.topo_input_norm(x)

        topo_emb = self.topo_transformer1(x_topo, edge_index, edge_attr)
        topo_emb = torch.nn.functional.gelu(topo_emb)
        topo_emb = self.topo_transformer2(topo_emb, edge_index, edge_attr)

        node_emb = self.output_norm(topo_emb)
        node_emb = self.output_mlp(node_emb)

        
        if self.req_fc is not None and req_vec is not None:
            if req_vec.dim() == 1:
                req_vec = req_vec.unsqueeze(0)
            req_emb = self.req_fc(req_vec)
            if batch is not None:
                req_expanded = req_emb[batch]
            else:
                req_expanded = req_emb.expand(node_emb.size(0), -1)
            node_emb = self.req_norm(
                self.req_fusion(torch.cat([node_emb, req_expanded], dim=-1))
            )

        if dest_mask is not None and dest_mask.any():
            
            if batch is not None and batch.max().item() > 0:
                num_graphs = batch.max().item() + 1
                attn_deltas = []
                for g_id in range(num_graphs):
                    g_mask = (batch == g_id)
                    g_dest = g_mask & dest_mask
                    g_node_emb = node_emb[g_mask]
                    if g_dest.any():
                        if self.dest_attn_mode == 'per_dest':
                            dest_kv_g = node_emb[g_dest].unsqueeze(0)
                        else:
                            dest_kv_g = node_emb[g_dest].mean(dim=0, keepdim=True).unsqueeze(0)
                        query_g = g_node_emb.unsqueeze(0)
                        attn_out_g, _ = self.dest_cross_attn(
                            query=query_g, key=dest_kv_g, value=dest_kv_g)
                        attn_deltas.append(attn_out_g.squeeze(0))
                    else:
                        attn_deltas.append(torch.zeros_like(g_node_emb))
                delta_full = torch.zeros_like(node_emb)
                for g_id in range(num_graphs):
                    g_mask = (batch == g_id)
                    delta_full[g_mask] = attn_deltas[g_id]
                node_emb = node_emb + delta_full
            else:
                if self.dest_attn_mode == 'per_dest':
                    dest_kv = node_emb[dest_mask].unsqueeze(0)
                else:
                    dest_kv = node_emb[dest_mask].mean(dim=0, keepdim=True).unsqueeze(0)
                query = node_emb.unsqueeze(0)
                attn_out, _ = self.dest_cross_attn(query=query, key=dest_kv, value=dest_kv)
                node_emb = node_emb + attn_out.squeeze(0)

        
        aux_pred = None
        if self.coupling_mode != 'off':
            node_emb, aux_pred = self._apply_coupling(node_emb)

        if return_aux:
            return node_emb, aux_pred
        return node_emb

    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None,
                            tree_edge_index=None, dest_mask=None, req_vec=None):
        node_embeddings = self.forward(x, edge_index, edge_attr, batch,
                                       tree_edge_index, dest_mask, req_vec, return_aux=False)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(node_embeddings, batch)
