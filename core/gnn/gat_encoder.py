import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATEncoder(nn.Module):
    """Plain GAT baseline for the MSFT-HIRL w/ GAT ablation."""

    def __init__(self, node_dim=28, hidden_dim=128, num_heads=4, req_dim=0,
                 dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.req_dim = req_dim
        self.use_aux_head = False

        out_per_head = hidden_dim // num_heads
        self.input_norm = nn.LayerNorm(node_dim)
        self.gat1 = GATConv(
            in_channels=node_dim,
            out_channels=out_per_head,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=out_per_head,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if req_dim > 0:
            self.req_fc = nn.Linear(req_dim, hidden_dim)
            self.req_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.req_norm = nn.LayerNorm(hidden_dim)
        else:
            self.req_fc = None

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                tree_edge_index=None, dest_mask=None, req_vec=None,
                return_aux=False):
        # Plain GAT only uses node features and graph adjacency. The extra
        # arguments are kept for interface compatibility with TreeTransformer.
        h = self.input_norm(x)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        h = self.output_norm(h)
        node_emb = self.output_mlp(h)

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

        if return_aux:
            return node_emb, None
        return node_emb

    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None,
                            tree_edge_index=None, dest_mask=None, req_vec=None):
        node_embeddings = self.forward(
            x, edge_index, edge_attr=edge_attr, batch=batch,
            tree_edge_index=tree_edge_index, dest_mask=dest_mask,
            req_vec=req_vec,
        )
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(node_embeddings, batch)
