import os
import pickle
import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch
from pathlib import Path
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import platform
from torch_geometric.loader import DataLoader as PyGDataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return False


class ExpertDataset(Dataset):
    """
    支持两种格式：
      1. 新格式：step-level sample，action里直接有 low_label
      2. 旧格式：path-level sample，兼容展开
    """
    def __init__(self, transitions):
        self.samples = []
        converted = 0
        skipped = 0
        self.format_counts = {"step": 0, "path": 0, "skipped": 0}

        for trans in transitions:
            action_data = trans.get("action", {})

            
            if isinstance(action_data, dict) and "low_label" in action_data:
                self.samples.append({
                    "state": trans.get("state"),
                    "high_label": int(action_data["high_label"]),
                    "low_label": int(action_data["low_label"]),
                    "current_node": int(action_data.get("current_node", -1)),
                    "subgoal_node": int(action_data.get("subgoal_node", action_data.get("high_label", -1))),
                    "sample_type": action_data.get("sample_type", "path_step"),
                    "train_high": bool(action_data.get("train_high", not action_data.get("low_only", False))),
                    "train_low": bool(action_data.get("train_low", not action_data.get("high_only", False))),
                    "req": trans.get("request", {}),
                })
                converted += 1
                self.format_counts["step"] += 1

            
            elif isinstance(action_data, dict) and "path" in action_data:
                converted_samples = self._convert_path_to_steps(trans)
                self.samples.extend(converted_samples)
                converted += len(converted_samples)
                self.format_counts["path"] += len(converted_samples)

            else:
                skipped += 1
                self.format_counts["skipped"] += 1

        logger.info(
            f"ExpertDataset loaded: total={len(transitions)} "
            f"usable_steps={len(self.samples)} converted={converted} skipped={skipped}"
        )

    def _convert_path_to_steps(self, trans):
        """
        兼容旧数据：
        旧数据只有一条path和一个state，这种监督信号其实不干净，
        这里只保底兼容，主流程应使用新collector生成的step-level数据。
        """
        converted = []
        action_data = trans.get("action", {})
        path = action_data.get("path", None)
        high_label = action_data.get("high_label", None)
        state = trans.get("state", None)

        if path is None or state is None or len(path) < 2:
            return converted

        for i in range(len(path) - 1):
            converted.append({
                "state": state,
                "high_label": int(high_label),
                "low_label": int(path[i + 1]),
                "current_node": int(path[i]),
                "subgoal_node": int(high_label) if high_label is not None else int(path[-1]),
                "sample_type": action_data.get("sample_type", "path_step"),
                "train_high": bool(action_data.get("train_high", True)),
                "train_low": bool(action_data.get("train_low", True)),
                "req": trans.get("request", {}),
            })

        return converted

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "state":      item["state"],
            "high_label": item["high_label"],
            "low_label":  item["low_label"],
            "current_node": item.get("current_node", -1),
            "subgoal_node": item.get("subgoal_node", item["high_label"]),
            "sample_type": item.get("sample_type", "path_step"),
            "train_high": item.get("train_high", True),
            "train_low": item.get("train_low", True),
            "req":        item.get("req", {}),
        }

class Phase2ILTrainer:
    def __init__(self, env, agent, expert_data_path: str, output_dir: str, config: dict):
        self.env = env
        self.agent = agent
        self.cfg = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "il_loss_history.csv"
        self.dataset_summary_path = self.output_dir / "il_dataset_summary.json"
        self._loss_high_sum = 0.0
        self._loss_low_sum = 0.0
        self._loss_high_cand_sum = 0.0
        self._loss_low_cand_sum = 0.0
        self._loss_count = 0

        phase2_cfg = config.get('phase2', {})
        self.epochs = phase2_cfg.get('epochs', 150)
        self.batch_size = phase2_cfg.get('batch_size', 64)
        self.validation_split = phase2_cfg.get('validation_split', 0.1)
        self.candidate_loss_weight = float(phase2_cfg.get('candidate_loss_weight', 1.0))
        self.global_loss_weight = float(phase2_cfg.get('global_loss_weight', 1.0))
        self.device = agent.device

        self.is_hrl = hasattr(agent, 'high_policy') and hasattr(agent, 'low_policy')

        if self.is_hrl:
            logger.info(" Phase 2: 检测到 HRL Agent，准备进行双层策略训练")
            self.model_high = agent.high_policy
            self.model_low = agent.low_policy

            il_lr = config.get('phase2', {}).get('lr', 3e-4)
            self.optimizer_high = torch.optim.Adam(
                agent.high_policy.parameters(), lr=il_lr
            )

            low_params = list(agent.low_policy.parameters())
            if hasattr(agent, 'encoder') and agent.encoder is not None:
                low_params += list(agent.encoder.parameters())
                agent.encoder.train()
                logger.info("    [修复] TreeTransformerEncoder 参数已加入 optimizer_low")

            self.optimizer_low = torch.optim.Adam(
                low_params, lr=il_lr
            )
            logger.info(f" Phase2: 独立创建优化器 lr={il_lr:.2e}")

            self.scheduler_high = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_high, mode='min', factor=0.5, patience=10, min_lr=1e-6
            )
            self.scheduler_low = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_low, mode='min', factor=0.5, patience=10, min_lr=1e-6
            )
        else:
            logger.warning(" Phase 2: 检测到旧版 Agent，仅训练 PolicyNet")
            self.model = agent.policy_net
            self.optimizer = agent.optimizer
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()

        self.num_workers = 0 if platform.system() == 'Windows' else 4
        self._prepare_data(expert_data_path)

        
        from collections import Counter
        n_nodes = env.n
        weights = torch.ones(n_nodes, dtype=torch.float)
        if getattr(self, 'train_loader', None) is not None:
            low_label_counts = Counter(
                s['low_label'] for s in self.train_loader.dataset.dataset.samples
            )
            total = sum(low_label_counts.values())
            for node_id, cnt in low_label_counts.items():
                if node_id < n_nodes:
                    weights[node_id] = total / (n_nodes * cnt)
            weights = weights / weights.sum() * n_nodes
            logger.info(f" 低层损失加权完成，最大权重节点: "
                        f"{weights.argmax().item()} ({weights.max().item():.2f}x)")
        self.criterion_low = nn.CrossEntropyLoss(weight=weights.to(self.device))
        
        self.n_dest = 28
        logger.info(f" Phase2: 高层固定类别数 n_dest={self.n_dest}")
        from core.gnn.tree_transformer_encoder import TreeTransformerEncoder
        from torch_geometric.nn import global_mean_pool

        node_feat_dim = config.get('node_feat_dim', 21)
        edge_feat_dim = config.get('edge_feat_dim', 5)
        hidden_dim = config.get('hrl', {}).get('hidden_dim', 128)
        num_heads = config.get('hrl', {}).get('num_heads', 4)

        if agent.encoder is None:
            gnn_enc = TreeTransformerEncoder(
                node_dim=node_feat_dim,
                edge_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(self.device)
            agent.encoder = gnn_enc
            logger.info(f" Phase2: 使用 TreeTransformerEncoder "
                        f"(node={node_feat_dim}, edge={edge_feat_dim}, hidden={hidden_dim})")
            if self.is_hrl:
                
                
                self.optimizer_low.add_param_group({'params': gnn_enc.parameters()})
                logger.info("    [修复] TreeTransformerEncoder 参数已加入 optimizer_low")

        n_nodes = env.n

        def _phase2_graph_embedding(pyg_batch, req_vec=None):
            enc = agent.encoder
            node_emb = enc(pyg_batch.x, pyg_batch.edge_index,
                           edge_attr=getattr(pyg_batch, 'edge_attr', None),
                           batch=getattr(pyg_batch, 'batch', None),
                           tree_edge_index=getattr(pyg_batch, 'tree_edge_index', None),
                           dest_mask=getattr(pyg_batch, 'dest_mask', None),
                           req_vec=req_vec)

            graph_emb = global_mean_pool(node_emb, pyg_batch.batch)

            B = graph_emb.size(0)
            H = node_emb.size(-1)
            
            node_emb_3d = node_emb.view(B, -1, H)

            return graph_emb, node_emb_3d

        self._phase2_graph_embedding = _phase2_graph_embedding

        self.early_stopping = EarlyStopping(patience=30)

    def _prepare_data(self, data_path):
        
        data_path = Path(data_path)
        if not data_path.exists():
            logger.error(f" 专家数据文件不存在: {data_path}")
            self.train_loader = None
            self.val_loader = None
            return

        with open(data_path, "rb") as f:
            raw = pickle.load(f)

        
        if isinstance(raw, dict):
            transitions = raw.get("success", raw.get("samples", raw.get("data", [])))
        elif isinstance(raw, list):
            transitions = raw
        else:
            logger.error(f" 无法解析专家数据格式: {type(raw)}")
            self.train_loader = None
            self.val_loader = None
            return

        logger.info(f" 加载专家数据: {len(transitions)} 条样本 from {data_path}")

        full_dataset = ExpertDataset(transitions)
        if len(full_dataset) == 0:
            self.train_loader = None
            return

        self._write_dataset_summary(full_dataset)

        val_size = int(len(full_dataset) * self.validation_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._collate_fn,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate_fn,
            drop_last=True
        )

    def _collate_fn(self, batch):
        states = []
        high_labels = []
        low_labels = []
        current_nodes = []
        subgoal_nodes = []
        reqs = []
        train_high_flags = []
        train_low_flags = []

        for item in batch:
            state = item.get('state')
            if state is None: continue

            states.append(state)
            high_labels.append(item['high_label'])
            low_labels.append(item['low_label'])
            current_nodes.append(item.get('current_node', -1))
            subgoal_nodes.append(item.get('subgoal_node', item['high_label']))
            train_high_flags.append(bool(item.get('train_high', True)))
            train_low_flags.append(bool(item.get('train_low', True)))
            reqs.append(item.get('req'))

        if not states: return None

        
        
        
        
        
        
        
        for s in states:
            if getattr(s, 'tree_edge_index', None) is None:
                nodes = torch.arange(s.x.size(0))
                s.tree_edge_index = torch.stack([nodes, nodes], dim=0)
            if getattr(s, 'dest_mask', None) is None:
                s.dest_mask = torch.zeros(s.x.size(0), dtype=torch.bool)

        graph_batch = Batch.from_data_list(states)
        high_labels = torch.tensor(high_labels, dtype=torch.long)
        low_labels  = torch.tensor(low_labels,  dtype=torch.long)
        current_nodes = torch.tensor(current_nodes, dtype=torch.long)
        subgoal_nodes = torch.tensor(subgoal_nodes, dtype=torch.long)
        train_high_mask = torch.tensor(train_high_flags, dtype=torch.bool)
        train_low_mask = torch.tensor(train_low_flags, dtype=torch.bool)

        
        req_vecs = []
        for r in reqs:
            if r is not None:
                bw  = float(r.get('bw_origin', r.get('bw', 0.0)))
                cpu = float(np.mean(r.get('cpu_origin', r.get('cpu', [0.0]))) if r.get('cpu_origin', r.get('cpu')) else 0.0)
                mem = float(np.mean(r.get('memory_origin', r.get('memory', [0.0]))) if r.get('memory_origin', r.get('memory')) else 0.0)
                req_vecs.append([bw, cpu, mem])
            else:
                req_vecs.append([0.0, 0.0, 0.0])
        req_tensor = torch.tensor(req_vecs, dtype=torch.float32)  # [B, 3]

        return graph_batch, high_labels, low_labels, req_tensor, current_nodes, subgoal_nodes, train_high_mask, train_low_mask

    def _write_dataset_summary(self, dataset):
        highs = [int(s["high_label"]) for s in dataset.samples]
        lows = [int(s["low_label"]) for s in dataset.samples]
        currents = [int(s.get("current_node", -1)) for s in dataset.samples]
        action_mask_missing = 0
        action_mask_label_off = 0
        type_counts = {}
        train_high_count = 0
        train_low_count = 0
        for s in dataset.samples:
            stype = str(s.get("sample_type", "unknown"))
            type_counts[stype] = type_counts.get(stype, 0) + 1
            train_high_count += int(bool(s.get("train_high", True)))
            train_low_count += int(bool(s.get("train_low", True)))
            if not bool(s.get("train_low", True)):
                continue
            st = s.get("state")
            mask = getattr(st, "action_mask", None) if st is not None else None
            if mask is None:
                action_mask_missing += 1
                continue
            try:
                row = mask.view(-1)
                label = int(s["low_label"])
                if label < 0 or label >= row.numel() or float(row[label].item()) <= 0.0:
                    action_mask_label_off += 1
            except Exception:
                action_mask_missing += 1

        summary = {
            "num_samples": len(dataset.samples),
            "format_counts": getattr(dataset, "format_counts", {}),
            "sample_type_counts": type_counts,
            "train_high_count": train_high_count,
            "train_low_count": train_low_count,
            "high_label_min": min(highs) if highs else None,
            "high_label_max": max(highs) if highs else None,
            "low_label_min": min(lows) if lows else None,
            "low_label_max": max(lows) if lows else None,
            "current_node_missing": sum(1 for v in currents if v < 0),
            "action_mask_missing": action_mask_missing,
            "action_mask_label_off": action_mask_label_off,
        }
        try:
            with open(self.dataset_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Phase2 IL dataset summary saved: {self.dataset_summary_path}")
            logger.info(f"Phase2 IL dataset summary: {summary}")
        except Exception as e:
            logger.warning(f"Phase2 IL dataset summary save failed: {e}")

    def _ensure_history_header(self):
        if self.history_path.exists():
            return
        fields = [
            "epoch", "train_loss", "val_loss",
            "train_high_loss", "train_low_loss",
            "train_high_candidate_loss", "train_low_candidate_loss",
            "val_high_loss", "val_low_loss",
            "val_high_candidate_loss", "val_low_candidate_loss",
            "train_high_acc", "train_low_acc", "val_high_acc", "val_low_acc",
            "low_mask_repair_rate",
            "lr",
        ]
        with open(self.history_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append_history(self, row):
        self._ensure_history_header()
        fields = [
            "epoch", "train_loss", "val_loss",
            "train_high_loss", "train_low_loss",
            "train_high_candidate_loss", "train_low_candidate_loss",
            "val_high_loss", "val_low_loss",
            "val_high_candidate_loss", "val_low_candidate_loss",
            "train_high_acc", "train_low_acc", "val_high_acc", "val_low_acc",
            "low_mask_repair_rate",
            "lr",
        ]
        with open(self.history_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    def _build_low_action_masks(
            self,
            states,
            node_emb_3d,
            low_labels=None,
            train_low_mask=None,
            current_nodes=None,
            subgoal_nodes=None,
    ):
        B_cur = node_emb_3d.size(0)
        action_dim = self.model_low.action_dim
        if hasattr(states, 'action_mask'):
            action_masks = states.action_mask.float().to(self.device)
            if action_masks.dim() == 1:
                action_masks = action_masks.view(B_cur, -1)
            if action_masks.size(0) != B_cur:
                action_masks = torch.ones(B_cur, action_dim, device=self.device)
            elif action_masks.size(1) != action_dim:
                if action_masks.size(1) > action_dim:
                    action_masks = action_masks[:, :action_dim]
                else:
                    pad = torch.zeros(B_cur, action_dim - action_masks.size(1), device=self.device)
                    action_masks = torch.cat([action_masks, pad], dim=1)
        else:
            logger.warning("Phase2 batch missing action_mask, using all-one low mask")
            action_masks = torch.ones(B_cur, action_dim, device=self.device)

        if low_labels is not None:
            fixed = 0
            checked = 0
            examples = []
            for i, label in enumerate(low_labels.detach().cpu().tolist()):
                if train_low_mask is not None and not bool(train_low_mask[i].item()):
                    continue
                checked += 1
                if 0 <= int(label) < action_dim and action_masks[i, int(label)] <= 0:
                    if len(examples) < 5:
                        valid = torch.nonzero(action_masks[i] > 0, as_tuple=False).view(-1)
                        examples.append({
                            "i": int(i),
                            "cur": int(current_nodes[i].item()) if current_nodes is not None else -1,
                            "target": int(subgoal_nodes[i].item()) if subgoal_nodes is not None else -1,
                            "expert_next": int(label),
                            "valid": valid[:12].detach().cpu().tolist(),
                            "valid_count": int(valid.numel()),
                        })
                    action_masks[i, int(label)] = 1.0
                    fixed += 1
            self._epoch_mask_repair_checked = getattr(self, "_epoch_mask_repair_checked", 0) + checked
            self._epoch_mask_repair_fixed = getattr(self, "_epoch_mask_repair_fixed", 0) + fixed
            if fixed:
                if not hasattr(self, "_mask_repair_log_count"):
                    self._mask_repair_log_count = 0
                self._mask_repair_log_count += 1
                if self._mask_repair_log_count <= 8 or self._mask_repair_log_count % 100 == 0:
                    logger.warning(
                        f"Phase2 repaired {fixed} low expert labels missing from action_mask "
                        f"examples={examples}"
                    )
        return action_masks

    def _candidate_losses(self, graph_emb, node_emb_3d, subgoal_emb, high_labels, low_labels,
                          current_nodes, action_masks, train_high_mask=None, train_low_mask=None):
        high_cand_losses = []
        low_cand_losses = []
        B, N, _ = node_emb_3d.shape
        all_nodes = list(range(N))

        for i in range(B):
            h_label = int(high_labels[i].item())
            use_high = True if train_high_mask is None else bool(train_high_mask[i].item())
            if use_high and 0 <= h_label < N and hasattr(self.model_high, "score_goal_candidates"):
                local_feats_h = torch.zeros(N, getattr(self.model_high, "_goal_feat_dim", 9), device=self.device)
                h_scores = self.model_high.score_goal_candidates(
                    graph_emb[i:i + 1],
                    all_nodes,
                    node_emb_3d[i:i + 1],
                    local_feats_h,
                )
                high_cand_losses.append(
                    self.criterion(h_scores, torch.tensor([h_label], dtype=torch.long, device=self.device))
                )

            l_label = int(low_labels[i].item())
            cur = int(current_nodes[i].item()) if current_nodes is not None else -1
            if cur < 0 or cur >= N:
                cur = 0
            use_low = True if train_low_mask is None else bool(train_low_mask[i].item())
            if use_low and 0 <= l_label < self.model_low.action_dim and hasattr(self.model_low, "score_candidates"):
                cand = torch.nonzero(action_masks[i] > 0, as_tuple=False).view(-1).detach().cpu().tolist()
                cand = sorted(set(int(v) for v in cand if 0 <= int(v) < N))
                if l_label not in cand:
                    cand.append(l_label)
                    cand = sorted(set(int(v) for v in cand if 0 <= int(v) < N))
                if cand:
                    local_target = cand.index(l_label)
                    local_feats_l = torch.zeros(len(cand), 6, device=self.device)
                    l_scores = self.model_low.score_candidates(
                        node_emb_3d[i:i + 1],
                        subgoal_emb[i:i + 1] if subgoal_emb is not None else None,
                        cand,
                        cur,
                        local_feats_l,
                    )
                    low_cand_losses.append(
                        self.criterion(l_scores, torch.tensor([local_target], dtype=torch.long, device=self.device))
                    )

        zero = graph_emb.new_tensor(0.0)
        high_loss = torch.stack(high_cand_losses).mean() if high_cand_losses else zero
        low_loss = torch.stack(low_cand_losses).mean() if low_cand_losses else zero
        return high_loss, low_loss

    def run(self):
        if not self.train_loader:
            logger.error(" 数据未就绪，停止训练")
            return

        logger.info(" 开始 Phase 2 模仿学习 (HRL Mode)...")
        logger.info(
            f"Phase2 IL loss weights: global={self.global_loss_weight:.3f}, "
            f"candidate={self.candidate_loss_weight:.3f}"
        )
        self._ensure_history_header()
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            train_stats = self._train_epoch(epoch)
            val_stats = self._validate_epoch(epoch)
            train_loss = train_stats["loss"]
            val_loss = val_stats["loss"]

            if self.is_hrl:
                self.scheduler_high.step(val_loss)
                self.scheduler_low.step(val_loss)
                cur_lr = self.optimizer_high.param_groups[0]['lr']
            else:
                cur_lr = self.optimizer.param_groups[0]['lr']

            self._append_history({
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "train_high_loss": train_stats["high_loss"],
                "train_low_loss": train_stats["low_loss"],
                "train_high_candidate_loss": train_stats["high_candidate_loss"],
                "train_low_candidate_loss": train_stats["low_candidate_loss"],
                "val_high_loss": val_stats["high_loss"],
                "val_low_loss": val_stats["low_loss"],
                "val_high_candidate_loss": val_stats["high_candidate_loss"],
                "val_low_candidate_loss": val_stats["low_candidate_loss"],
                "train_high_acc": train_stats["high_acc"],
                "train_low_acc": train_stats["low_acc"],
                "val_high_acc": val_stats["high_acc"],
                "val_low_acc": val_stats["low_acc"],
                "low_mask_repair_rate": train_stats.get("low_mask_repair_rate", 0.0),
                "lr": cur_lr,
            })

            if epoch % 10 == 0:
                avg_h = self._loss_high_sum / max(1, self._loss_count)
                avg_l = self._loss_low_sum / max(1, self._loss_count)
                avg_hc = self._loss_high_cand_sum / max(1, self._loss_count)
                avg_lc = self._loss_low_cand_sum / max(1, self._loss_count)
                logger.info(
                    f"Epoch {epoch:>4} | TrainL={train_loss:.4f} | ValL={val_loss:.4f} | "
                    f"High={avg_h:.4f} Low={avg_l:.4f} "
                    f"HCand={avg_hc:.4f} LCand={avg_lc:.4f} "
                    f"AccH={train_stats['high_acc']:.3f} AccL={train_stats['low_acc']:.3f} | "
                    f"MaskRepair={train_stats.get('low_mask_repair_rate', 0.0):.3%} | "
                    f"lr={cur_lr:.2e}"
                )

                self._loss_high_sum = 0
                self._loss_low_sum = 0
                self._loss_high_cand_sum = 0
                self._loss_low_cand_sum = 0
                self._loss_count = 0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best")
                logger.info(f"   新最优验证Loss={val_loss:.4f}，已保存 il_model_best.pth")

            if epoch % 50 == 0:
                self._save_checkpoint(epoch)

            if self.early_stopping(val_loss):
                logger.info(f"⏹  早停触发 (epoch={epoch}, best_val={best_val_loss:.4f})")
                break

        self._save_checkpoint("final")
        logger.info(f" Phase 2 完成 | 最佳验证Loss={best_val_loss:.4f}")
        logger.info("    建议Phase3加载 il_model_best.pth 而非 il_model_final.pth")

        logger.info(f"   Phase2 history: {self.history_path}")

    def _train_epoch(self, epoch):
        self.model_high.train()
        self.model_low.train()
        self._epoch_mask_repair_checked = 0
        self._epoch_mask_repair_fixed = 0

        total_loss = 0
        total_high = 0
        total_low = 0
        total_high_cand = 0
        total_low_cand = 0
        total_high_ok = 0
        total_low_ok = 0
        total_items = 0
        total_high_items = 0
        total_low_items = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_data in pbar:
            if isinstance(batch_data, dict):
                states = batch_data['state'].to(self.device)
                high_labels = batch_data['high_label'].to(self.device)
                low_labels = batch_data['low_label'].to(self.device)
                req_tensor = batch_data.get('req_vec')
                current_nodes = batch_data.get('current_node')
                subgoal_nodes = batch_data.get('subgoal_node')
                train_high_mask = batch_data.get('train_high_mask')
                train_low_mask = batch_data.get('train_low_mask')
            else:
                states, high_labels, low_labels, req_tensor, current_nodes, subgoal_nodes, train_high_mask, train_low_mask = batch_data
                states = states.to(self.device)
                high_labels = high_labels.to(self.device)
                low_labels = low_labels.to(self.device)
            current_nodes = current_nodes.to(self.device) if current_nodes is not None else None
            subgoal_nodes = subgoal_nodes.to(self.device) if subgoal_nodes is not None else None
            train_high_mask = train_high_mask.to(self.device) if train_high_mask is not None else torch.ones_like(high_labels, dtype=torch.bool)
            train_low_mask = train_low_mask.to(self.device) if train_low_mask is not None else torch.ones_like(low_labels, dtype=torch.bool)
            req_vec = req_tensor.to(self.device) if req_tensor is not None else None

            self.optimizer_high.zero_grad()
            self.optimizer_low.zero_grad()

            graph_emb, node_emb_3d = self._phase2_graph_embedding(states, req_vec=req_vec)

            high_logits, subgoal_emb, _ = self.model_high(graph_emb, return_subgoal=True)

            
            
            
            
            if hasattr(states, 'action_mask'):
                action_masks = states.action_mask.float().to(self.device)   # [B, 28]
                B_cur = node_emb_3d.size(0)
                action_dim = self.model_low.action_dim
                if action_masks.size(0) != B_cur:
                    action_masks = torch.ones(B_cur, action_dim, device=self.device)
                elif action_masks.size(1) != action_dim:
                    if action_masks.size(1) > action_dim:
                        action_masks = action_masks[:, :action_dim]
                    else:
                        pad = torch.zeros(B_cur, action_dim - action_masks.size(1), device=self.device)
                        action_masks = torch.cat([action_masks, pad], dim=1)
            else:
                logger.warning(" Phase2 batch 缺少 action_mask，退化为全1mask")
                action_masks = torch.ones(
                    node_emb_3d.size(0),
                    self.model_low.action_dim,
                    device=self.device
                )
            
            action_masks = self._build_low_action_masks(
                states, node_emb_3d, low_labels,
                train_low_mask=train_low_mask,
                current_nodes=current_nodes,
                subgoal_nodes=subgoal_nodes,
            )
            low_logits, _ = self.model_low(node_emb_3d, subgoal_emb, action_mask=action_masks)

            
            n_classes = high_logits.size(1)
            if high_labels.max() >= n_classes:
                logger.warning("Phase2 high label exceeds high logits dimension; clamping invalid labels")
                high_labels = high_labels.clamp(0, n_classes - 1)
            if low_labels.max() >= low_logits.size(1):
                logger.warning("Phase2 low label exceeds low logits dimension; clamping invalid labels")
                low_labels = low_labels.clamp(0, low_logits.size(1) - 1)
            loss_high = self.criterion(high_logits[train_high_mask], high_labels[train_high_mask]) \
                if bool(train_high_mask.any().item()) else graph_emb.new_tensor(0.0)
            loss_low_bc = self.criterion_low(low_logits[train_low_mask], low_labels[train_low_mask]) \
                if bool(train_low_mask.any().item()) else graph_emb.new_tensor(0.0)
            loss_high_cand, loss_low_cand = self._candidate_losses(
                graph_emb, node_emb_3d, subgoal_emb,
                high_labels, low_labels, current_nodes, action_masks,
                train_high_mask=train_high_mask, train_low_mask=train_low_mask
            )

            
            
            
            
            loss_global = loss_high * 0.5 + loss_low_bc
            loss_candidate = loss_high_cand * 0.5 + loss_low_cand
            loss = self.global_loss_weight * loss_global + self.candidate_loss_weight * loss_candidate

            loss.backward()
            if not hasattr(self, "_phase2_grad_log_count"):
                self._phase2_grad_log_count = 0
            self._phase2_grad_log_count += 1
            if self._phase2_grad_log_count <= 5 or self._phase2_grad_log_count % 200 == 0:
                def _grad_norm(module):
                    total = 0.0
                    for p in module.parameters():
                        if p.grad is not None:
                            total += float(p.grad.detach().data.norm(2).item() ** 2)
                    return total ** 0.5

                high_scorer_norm = (
                    _grad_norm(self.model_high.goal_candidate_scorer)
                    if hasattr(self.model_high, "goal_candidate_scorer") else 0.0
                )
                low_scorer_norm = (
                    _grad_norm(self.model_low.candidate_scorer)
                    if hasattr(self.model_low, "candidate_scorer") else 0.0
                )
                logger.info(
                    f"[Phase2Grad] step={self._phase2_grad_log_count} "
                    f"high_candidate_scorer={high_scorer_norm:.6f} "
                    f"low_candidate_scorer={low_scorer_norm:.6f}"
                )
            torch.nn.utils.clip_grad_norm_(self.model_high.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model_low.parameters(), 1.0)
            
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.encoder.parameters(), 1.0)
            self.optimizer_high.step()
            self.optimizer_low.step()

            self._loss_high_sum += loss_high.item()
            self._loss_low_sum += loss_low_bc.item()
            self._loss_high_cand_sum += loss_high_cand.item()
            self._loss_low_cand_sum += loss_low_cand.item()
            self._loss_count += 1
            total_loss += loss.item()
            total_high += loss_high.item()
            total_low += loss_low_bc.item()
            total_high_cand += loss_high_cand.item()
            total_low_cand += loss_low_cand.item()
            with torch.no_grad():
                if bool(train_high_mask.any().item()):
                    total_high_ok += int((high_logits.argmax(dim=1)[train_high_mask] == high_labels[train_high_mask]).sum().item())
                    total_high_items += int(train_high_mask.sum().item())
                if bool(train_low_mask.any().item()):
                    total_low_ok += int((low_logits.argmax(dim=1)[train_low_mask] == low_labels[train_low_mask]).sum().item())
                    total_low_items += int(train_low_mask.sum().item())
                total_items += int(max(train_high_mask.sum().item(), train_low_mask.sum().item()))

            pbar.set_postfix({
                'L': f"{loss.item():.3f}",
                'H': f"{loss_high.item():.3f}",
                'Low': f"{loss_low_bc.item():.3f}",
                'HC': f"{loss_high_cand.item():.3f}",
                'LC': f"{loss_low_cand.item():.3f}",
            })

        n_batches = max(1, len(self.train_loader))
        repair_rate = self._epoch_mask_repair_fixed / max(1, self._epoch_mask_repair_checked)
        return {
            "loss": total_loss / n_batches,
            "high_loss": total_high / n_batches,
            "low_loss": total_low / n_batches,
            "high_candidate_loss": total_high_cand / n_batches,
            "low_candidate_loss": total_low_cand / n_batches,
            "high_acc": total_high_ok / max(1, total_high_items),
            "low_acc": total_low_ok / max(1, total_low_items),
            "low_mask_repair_rate": repair_rate,
        }

    def _validate_epoch(self, epoch):
        total_loss = 0
        total_high = 0
        total_low = 0
        total_high_cand = 0
        total_low_cand = 0
        total_high_ok = 0
        total_low_ok = 0
        total_items = 0
        total_high_items = 0
        total_low_items = 0
        count = 0

        if self.is_hrl:
            self.model_high.eval()
            self.model_low.eval()
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                self.agent.encoder.eval()

        with torch.no_grad():
            for batch in self.val_loader:
                if not batch: continue
                states, high_labels, low_labels, req_tensor, current_nodes, subgoal_nodes, train_high_mask, train_low_mask = batch
                states = states.to(self.device)
                high_labels = high_labels.to(self.device)
                low_labels = low_labels.to(self.device)
                current_nodes = current_nodes.to(self.device)
                subgoal_nodes = subgoal_nodes.to(self.device)
                train_high_mask = train_high_mask.to(self.device)
                train_low_mask = train_low_mask.to(self.device)
                req_vec = req_tensor.to(self.device) if req_tensor is not None else None

                if self.is_hrl:
                    graph_emb, node_emb_3d = self._phase2_graph_embedding(states, req_vec=req_vec)

                    high_logits, subgoal_emb, _ = self.model_high(graph_emb, return_subgoal=True)

                    
                    _val_B   = node_emb_3d.size(0)
                    _val_dim = self.model_low.action_dim
                    if hasattr(states, 'action_mask'):
                        val_action_masks = states.action_mask.float().to(self.device)
                        if val_action_masks.size(0) != _val_B:
                            val_action_masks = torch.ones(_val_B, _val_dim, device=self.device)
                        elif val_action_masks.size(1) != _val_dim:
                            if val_action_masks.size(1) > _val_dim:
                                val_action_masks = val_action_masks[:, :_val_dim]
                            else:
                                _pad = torch.zeros(_val_B, _val_dim - val_action_masks.size(1), device=self.device)
                                val_action_masks = torch.cat([val_action_masks, _pad], dim=1)
                    else:
                        val_action_masks = torch.ones(_val_B, _val_dim, device=self.device)
                    val_action_masks = self._build_low_action_masks(
                        states, node_emb_3d, low_labels,
                        train_low_mask=train_low_mask,
                        current_nodes=current_nodes,
                        subgoal_nodes=subgoal_nodes,
                    )
                    low_logits, _ = self.model_low(node_emb_3d, subgoal_emb, action_mask=val_action_masks)

                    n_cls = high_logits.size(1)
                    if high_labels.max() >= n_cls:
                        high_labels = high_labels.clamp(0, n_cls - 1)
                    if low_labels.max() >= low_logits.size(1):
                        low_labels = low_labels.clamp(0, low_logits.size(1) - 1)
                    loss_high = self.criterion(high_logits[train_high_mask], high_labels[train_high_mask]) \
                        if bool(train_high_mask.any().item()) else graph_emb.new_tensor(0.0)
                    loss_low_bc = self.criterion_low(low_logits[train_low_mask], low_labels[train_low_mask]) \
                        if bool(train_low_mask.any().item()) else graph_emb.new_tensor(0.0)
                    loss_high_cand, loss_low_cand = self._candidate_losses(
                        graph_emb, node_emb_3d, subgoal_emb,
                        high_labels, low_labels, current_nodes, val_action_masks,
                        train_high_mask=train_high_mask, train_low_mask=train_low_mask
                    )
                    loss_global = loss_high * 0.5 + loss_low_bc
                    loss_candidate = loss_high_cand * 0.5 + loss_low_cand
                    loss = self.global_loss_weight * loss_global + self.candidate_loss_weight * loss_candidate
                else:
                    loss = torch.tensor(0.0)
                    loss_high = torch.tensor(0.0)
                    loss_low_bc = torch.tensor(0.0)
                    loss_high_cand = torch.tensor(0.0)
                    loss_low_cand = torch.tensor(0.0)

                total_loss += loss.item()
                total_high += loss_high.item()
                total_low += loss_low_bc.item()
                total_high_cand += loss_high_cand.item()
                total_low_cand += loss_low_cand.item()
                if self.is_hrl:
                    if bool(train_high_mask.any().item()):
                        total_high_ok += int((high_logits.argmax(dim=1)[train_high_mask] == high_labels[train_high_mask]).sum().item())
                        total_high_items += int(train_high_mask.sum().item())
                    if bool(train_low_mask.any().item()):
                        total_low_ok += int((low_logits.argmax(dim=1)[train_low_mask] == low_labels[train_low_mask]).sum().item())
                        total_low_items += int(train_low_mask.sum().item())
                    total_items += int(high_labels.numel())
                count += 1

        if self.is_hrl:
            self.model_high.train()
            self.model_low.train()
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                self.agent.encoder.train()

        denom = max(1, count)
        return {
            "loss": total_loss / denom,
            "high_loss": total_high / denom,
            "low_loss": total_low / denom,
            "high_candidate_loss": total_high_cand / denom,
            "low_candidate_loss": total_low_cand / denom,
            "high_acc": total_high_ok / max(1, total_high_items),
            "low_acc": total_low_ok / max(1, total_low_items),
        }

    def _save_checkpoint(self, tag):
        path = self.output_dir / f"il_model_{tag}.pth"

        save_dict = {
            'config': self.cfg,
            'ablation_variant': getattr(self.agent, '_ablation_variant', None),
            'encoder_class': (
                self.agent.encoder.__class__.__name__
                if hasattr(self.agent, 'encoder') and self.agent.encoder is not None
                else None
            ),
        }

        if self.is_hrl:
            save_dict.update({
                'high_policy': self.model_high.state_dict(),
                'low_policy': self.model_low.state_dict(),
                'optimizer_high': self.optimizer_high.state_dict(),
                'optimizer_low': self.optimizer_low.state_dict(),
                'n_goals':   self.model_high.num_goals,
                'n_actions': self.model_low.action_dim,
            })
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                save_dict['encoder'] = self.agent.encoder.state_dict()
                logger.info("    Encoder 权重已保存至 checkpoint")
        else:
            save_dict['model_state_dict'] = self.model.state_dict()

        torch.save(save_dict, path)
        logger.info(f" 模型已保存: {path}")
