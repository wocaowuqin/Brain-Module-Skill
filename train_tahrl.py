"""
main.py
====================================
TA-HRL v4 训练入口 - Goal-Conditioned HRL
====================================

【模块定位】
本文件是整个 TA-HRL 系统的训练启动入口，负责：
  - 解析命令行参数，加载 yaml 配置
  - 初始化网络拓扑、环境（SFC_HIRL_Env）、资源管理器
  - 按 phase 分支初始化 Agent 和 Trainer：
      Phase 2：GoalConditionedHRLAgent + Phase2ILTrainer（模仿学习预训练）
      Phase 3：GoalConditionedHRLAgent + HRL_Coordinator + Phase3RLTrainer（强化学习）
  - 加载数据集并启动训练

【训练流程（Phase 3 主链）】
  main()
    ├─ load_config()          加载 phase3.yaml，注入 HRL 超参数
    ├─ load_topology()        从 yaml 读取拓扑矩阵，写入 config
    ├─ SFC_HIRL_Env(config)   初始化 Gym 环境（含控制器、资源管理器、TimeSlotManager）
    ├─ inject_dynamic_dimensions()  将 env.n / K_vnf 等动态维度注入 config
    ├─ GoalConditionedHRLAgent(config, phase=2)  初始化 Agent（网络结构）
    ├─ HRL_Coordinator(env, high_agent, low_agent, config)  初始化协调器
    ├─ env.load_dataset(data_file)   加载 pkl 请求数据集
    └─ Phase3RLTrainer.run()         启动 RL 训练主循环

【主要函数索引】
  main()                        训练主入口，按 phase 分支初始化并启动训练
  set_seed()                    全局随机种子设置
  load_topology()               从 yaml 加载拓扑矩阵，支持 .mat / numpy 格式
  inject_dynamic_dimensions()   把 env 的动态维度（n、K_vnf等）写回 config
  setup_hrl_config()            注入 TA-HRL v4 防死锁超参数到 config
  get_config_path()             安全读取 config 中的嵌套字段
  ensure_paths_exist()          确保输出目录存在
  validate_config()             检查必要配置项是否存在

【与其他模块的依赖关系】
  → SFC_HIRL_Env           Gym 环境主类，含拓扑/资源/控制器初始化
  → GoalConditionedHRLAgent  双层 HRL 策略网络（high_policy + low_policy + encoder）
  → HRL_Coordinator        高-低层协调驱动，episode 主循环
  → Phase3RLTrainer        RL 训练循环，含 checkpoint / CSV 日志
  → Phase2ILTrainer        模仿学习预训练（Phase 2）
  → phase3.yaml            训练超参数配置文件
"""

import scipy.io
import argparse
import logging
import os
import sys

# Headless training always writes charts to files. Set the backend before
# Torch/TensorBoard can import pyplot; switching it later fails in Python
# environments that expose an incomplete IPython namespace package.
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import torch
import random

import yaml

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env


from core.hrl.agent import (
    GoalConditionedHRLAgent,
    create_goal_conditioned_agent
)

from trainer.phase1_collector import Phase1ExpertCollector
from trainer.phase2_il_trainer import Phase2ILTrainer
from trainer.phase3_rl_trainer import Phase3RLTrainer
from trainer.phase3_rl_trainer_no_checkpoint import Phase3RLTrainerNoCheckpoint


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_dir: str, topo: str, phase: str, seed: int):
    """把所有日志同时写入文件"""
    import time
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{phase}_{topo}_seed{seed}_{ts}.log")
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)
    logger.info(f" 日志文件: {log_file}")
    return log_file


def finalize_phase3_output_dir(run_dir, rate_tag='', variant_tag='full'):
    """Rename a finished Phase3 directory to time-rate-variant-acc-tree."""
    import csv
    import re
    import time
    from datetime import datetime

    run_dir = os.path.abspath(str(run_dir))
    dataset_dir = os.path.join(run_dir, 'dataset')
    cost_path = os.path.join(dataset_dir, 'cost_summary.csv')
    episodes_path = os.path.join(dataset_dir, 'episodes.csv')

    def _safe_tag(value):
        raw = str(value or '').strip()
        return re.sub(r'[^0-9A-Za-z._-]+', '_', raw) or 'NA'

    def _num_tag(value, digits):
        if value is None:
            return 'NA'
        try:
            return f"{float(value):.{digits}f}".rstrip('0').rstrip('.')
        except Exception:
            return _safe_tag(value)

    acc = None
    tree = None

    if os.path.exists(cost_path):
        try:
            metrics = {}
            with open(cost_path, 'r', encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    metrics[str(row.get('metric', '')).strip()] = str(row.get('value', '')).strip()
            acc_text = metrics.get('acc', '').replace('%', '').strip()
            tree_text = metrics.get('tree_len', '').strip()
            acc = float(acc_text) if acc_text else None
            tree = float(tree_text) if tree_text else None
        except Exception as exc:
            logger.warning(f"Failed to read cost_summary.csv for rename: {exc}")

    if (acc is None or tree is None) and os.path.exists(episodes_path):
        try:
            total = 0
            success = 0
            trees = []
            with open(episodes_path, 'r', encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    total += 1
                    ok = str(row.get('success', '')).strip().lower() in {'1', 'true', 'yes'}
                    success += int(ok)
                    if ok:
                        try:
                            trees.append(float(row.get('tree_len', 0) or 0))
                        except Exception:
                            pass
            acc = success / max(total, 1) * 100.0
            tree = sum(trees) / len(trees) if trees else 0.0
        except Exception as exc:
            logger.warning(f"Failed to read episodes.csv for rename: {exc}")

    if acc is None:
        logger.warning(f"Skip Phase3 output rename, metrics not found: {run_dir}")
        return run_dir

    rate_part = _safe_tag(rate_tag or 'rateNA')
    if not rate_part.startswith('rate'):
        rate_part = f"rate{rate_part}"
    variant_part = _safe_tag(variant_tag or 'full')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_name = f"{stamp}-{rate_part}-{variant_part}-acc{_num_tag(acc, 3)}-tree{_num_tag(tree, 2)}"

    if re.search(r'-acc[^\\\/]+-tree', os.path.basename(run_dir)):
        logger.info(f"Phase3 output dir already finalized: {run_dir}")
        return run_dir

    final_dir = os.path.join(os.path.dirname(run_dir), final_name)
    if os.path.abspath(final_dir) == os.path.abspath(run_dir):
        return run_dir
    if os.path.exists(final_dir):
        final_dir = os.path.join(os.path.dirname(run_dir), f"{final_name}-{int(time.time())}")

    try:
        os.rename(run_dir, final_dir)
        logger.info(f"Phase3 output dir renamed: {final_dir}")
        return final_dir
    except Exception as exc:
        logger.warning(f"Failed to rename Phase3 output dir, keep original: {run_dir} ({exc})")
        return run_dir


def set_seed(seed):
    """设置全局随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_config_path(config, key_path):
    """安全地获取配置路径"""
    possible_locations = [
        ['path', key_path],
        ['project', key_path],
        ['paths', key_path],
        [key_path],
    ]

    for location in possible_locations:
        try:
            value = config
            for key in location:
                value = value[key]
            return value
        except (KeyError, TypeError):
            continue

    default_paths = {
        'ckpt_dir': './artifacts/runs/hrl/checkpoints',
        'log_dir': './artifacts/runs/hrl/logs',
        'expert_data_dir': './artifacts/runs/hrl/expert',
        'input_dir': './data/input_dir',
    }

    if key_path in default_paths:
        logger.warning(f"  配置中未找到 {key_path}，使用默认值: {default_paths[key_path]}")
        return default_paths[key_path]

    return None


def ensure_paths_exist(config):
    """确保所有必要的目录存在"""
    path_keys = ['ckpt_dir', 'log_dir', 'expert_data_dir']

    for key in path_keys:
        path = get_config_path(config, key)
        if path:
            os.makedirs(path, exist_ok=True)
            logger.info(f" 目录准备完成: {path}")


def validate_config(config, phase):
    """验证配置完整性"""
    logger.info(" 验证配置...")

    errors = []
    warnings = []

    if 'gnn' not in config:
        warnings.append("  缺少 'gnn' 配置块（将从环境动态获取）")

    if 'env' not in config and 'environment' not in config:
        errors.append(" 缺少 'env' 或 'environment' 配置块")

    if phase not in config:
        warnings.append(f"  缺少 '{phase}' 配置块")

    
    if phase == 'phase3' and 'hrl' not in config:
        warnings.append("  缺少 'hrl' 配置块（将使用默认值）")

    if errors:
        logger.error(" 配置验证失败:")
        for err in errors:
            logger.error(f"  {err}")
        raise ValueError("配置不完整，请检查配置文件")

    if warnings:
        logger.warning("  配置警告:")
        for warn in warnings:
            logger.warning(f"  {warn}")

    logger.info(" 配置验证通过")


def load_topology(config):
    """
    Unified topology loader - 修复版

    修复：
    1. 跳过完全图矩阵
    2. 优先使用特定键名
    3. 从 Paths 构建拓扑作为后备
    """
    logger.info(" 正在加载拓扑矩阵...")

    if 'topology' not in config:
        config['topology'] = {}

    
    mat_path = config.get('topology', {}).get('file')
    logger.info(f"  拓扑文件：{mat_path}")

    if not os.path.exists(mat_path):
        logger.error(f" 拓扑文件不存在: {mat_path}")
        return False

    try:
        mat_data = scipy.io.loadmat(mat_path)
    except Exception as e:
        logger.error(f" 读取 mat 文件失败: {e}")
        return False

    
    available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    logger.info(f" MAT文件包含的键: {available_keys}")

    
    adjacency_keys = ['adjacency', 'Adjacency', 'topo', 'Topo', 'adj_matrix',
                      'graph', 'topology', 'network', 'links']

    for key in adjacency_keys:
        if key in mat_data:
            val = mat_data[key]
            if isinstance(val, np.ndarray) and val.ndim == 2:
                if val.shape[0] == val.shape[1] and np.issubdtype(val.dtype, np.number):
                    topo = (val > 0).astype(np.float32)
                    np.fill_diagonal(topo, 0)

                    
                    N = topo.shape[0]
                    expected_complete_graph_edges = N * (N - 1)
                    actual_edges = int(np.sum(topo))

                    if actual_edges == expected_complete_graph_edges:
                        logger.warning(f"  跳过 '{key}': 这是完全图 ({actual_edges}条边)")
                        continue

                    if actual_edges == 0:
                        logger.warning(f"  跳过 '{key}': 空矩阵")
                        continue

                    
                    sparsity = actual_edges / expected_complete_graph_edges
                    if sparsity > 0.8:
                        logger.warning(f"  跳过 '{key}': 太密集 (sparsity={sparsity:.2%})")
                        continue

                    logger.info(f" 使用邻接矩阵字段: '{key}'")
                    logger.info(f"   节点数: {N}")
                    logger.info(f"   物理链路数: {actual_edges // 2}")
                    logger.info(f"   平均度数: {actual_edges / N:.2f}")
                    logger.info(f"   稀疏度: {sparsity:.2%}")

                    config['topology']['matrix'] = topo
                    return True

    
    logger.info(" 在所有方阵中寻找最合适的拓扑矩阵...")

    candidates = []
    for key, val in mat_data.items():
        if key.startswith('__'):
            continue

        if isinstance(val, np.ndarray) and val.ndim == 2:
            if val.shape[0] == val.shape[1] and np.issubdtype(val.dtype, np.number):
                topo = (val > 0).astype(np.float32)
                np.fill_diagonal(topo, 0)

                N = topo.shape[0]
                actual_edges = int(np.sum(topo))

                if actual_edges == 0:
                    continue

                expected_complete = N * (N - 1)
                if actual_edges == expected_complete:
                    continue

                sparsity = actual_edges / expected_complete
                avg_degree = actual_edges / N

                candidates.append({
                    'key': key,
                    'topo': topo,
                    'edges': actual_edges // 2,
                    'sparsity': sparsity,
                    'avg_degree': avg_degree,
                    'nodes': N
                })

    if candidates:
        
        candidates.sort(key=lambda x: x['sparsity'])

        logger.info(f" 找到 {len(candidates)} 个候选矩阵:")
        for i, c in enumerate(candidates[:3]):
            logger.info(f"   {i + 1}. '{c['key']}': {c['edges']}条链路, "
                        f"度数={c['avg_degree']:.2f}, 稀疏度={c['sparsity']:.2%}")

        best = candidates[0]

        
        if 2 <= best['avg_degree'] <= 15:
            logger.info(f" 选择最稀疏的矩阵: '{best['key']}'")
            logger.info(f"   节点数: {best['nodes']}")
            logger.info(f"   物理链路数: {best['edges']}")
            logger.info(f"   平均度数: {best['avg_degree']:.2f}")

            config['topology']['matrix'] = best['topo']
            return True
        else:
            logger.warning(f"  最佳候选 '{best['key']}' 的度数异常: {best['avg_degree']:.2f}")

    
    if 'Paths' not in mat_data:
        logger.error(" 无法找到合适的邻接矩阵，且 mat 文件中不存在 Paths 结构")
        return False

    logger.info(" 从 Paths 构建拓扑（后备方案）...")
    paths_matrix = mat_data['Paths']
    N, M = paths_matrix.shape

    topo = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            if i == j:
                continue

            cell = paths_matrix[i, j]

            if not hasattr(cell, 'dtype'):
                continue
            if cell.dtype.names is None:
                continue
            if 'paths' not in cell.dtype.names:
                continue

            paths_array = cell['paths']
            if not isinstance(paths_array, np.ndarray):
                continue

            if paths_array.ndim == 1:
                paths_array = paths_array[np.newaxis, :]

            for path in paths_array:
                nodes = path[path > 0] - 1
                if len(nodes) < 2:
                    continue

                for k in range(len(nodes) - 1):
                    u, v = int(nodes[k]), int(nodes[k + 1])
                    if 0 <= u < N and 0 <= v < N:
                        topo[u, v] = 1.0
                        topo[v, u] = 1.0

    np.fill_diagonal(topo, 0)

    if np.sum(topo) == 0:
        logger.error(" Paths 解析完成，但未发现任何物理链路")
        return False

    num_edges = int(np.sum(topo) / 2)
    avg_degree = np.sum(topo) / N

    logger.info(f" 从 Paths 构建拓扑成功:")
    logger.info(f"   节点数: {N}")
    logger.info(f"   物理链路数: {num_edges}")
    logger.info(f"   平均度数: {avg_degree:.2f}")

    config['topology']['matrix'] = topo.astype(np.float32)
    return True


def inject_dynamic_dimensions(config, env):
    """从环境中获取动态维度并注入到配置"""
    logger.info(" 注入动态维度...")

    if 'gnn' not in config:
        config['gnn'] = {}

    
    try:
        _sample_state = env.get_state()
        _actual_node_dim = _sample_state.x.shape[1]
    except Exception:
        _actual_node_dim = 24  # fallback
    config['gnn']['node_feat_dim'] = _actual_node_dim
    logger.info(f"  node_feat_dim (auto from env.get_state()): {_actual_node_dim}")

    
    if 'edge_feat_dim' not in config['gnn']:
        config['gnn']['edge_feat_dim'] = env.resource_mgr.edge_feat_dim
    if 'request_feat_dim' not in config['gnn']:
        config['gnn']['request_feat_dim'] = env.resource_mgr.request_dim

    
    if 'hrl' not in config:
        config['hrl'] = {}

    
    if hasattr(env, 'observation_space'):
        if 'x' in env.observation_space:
            state_dim = env.observation_space['x'].shape[1]
            config['hrl']['state_dim'] = state_dim
            logger.info(f"  state_dim: {state_dim}")

    logger.info(f"  node_feat_dim: {config['gnn']['node_feat_dim']}")
    logger.info(f"  edge_feat_dim: {config['gnn']['edge_feat_dim']}")
    logger.info(f"  request_feat_dim: {config['gnn']['request_feat_dim']}")


def setup_hrl_config(config):
    """
     新增：设置 HRL 默认配置
    """
    if 'hrl' not in config:
        config['hrl'] = {}

    hrl_defaults = {
        'goal_dim': 64,
        'subgoal_horizon': 5,
        'intrinsic_reward_weight': 0.3,
        'max_complexity_threshold': 0.8,
        'goal_strategy': 'adaptive'  # 'relative', 'adaptive', 'hybrid'
    }

    for key, default_value in hrl_defaults.items():
        if key not in config['hrl']:
            config['hrl'][key] = default_value
            logger.info(f"  使用默认 hrl.{key}: {default_value}")

def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description='TA-HRL v4 训练启动器',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    
    parser.add_argument(
        '--phase', type=str, default='phase3',
        choices=['phase1', 'phase2', 'phase3'],
        help='训练阶段：phase1=专家数据收集 / phase2=模仿学习 / phase3=强化学习',
    )
    parser.add_argument(
        '--data', type=str,
        default='data/us_backbone_rate8/phase3_requests.pkl',
        help='训练数据集 .pkl 文件路径',
    )
    parser.add_argument(
        '--max_requests', type=int, default=0,
        help='仅使用数据集前 N 条请求训练；0 表示使用完整数据集',
    )
    parser.add_argument(
        '--max_episode_steps', type=int, default=0,
        help='单请求最多执行的高层周期数；0 表示使用 YAML 配置',
    )
    parser.add_argument(
        '--output_dir', type=str, default='artifacts/runs/hrl',
        help='Phase3 checkpoint/output base directory',
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='yaml 配置文件路径，不填则由 --phase 自动推断（phase3 → phase3.yaml）',
    )

    
    parser.add_argument('--gpu',  type=int, default=0,
                        help='GPU 编号，-1 强制使用 CPU')
    parser.add_argument('--torch_threads', type=int, default=1,
                        help='PyTorch CPU 线程数；小图建议 1，避免线程调度开销')
    parser.add_argument('--quiet', action='store_true',
                        help='降低训练期控制台日志，只保留警告和最终统计')
    parser.add_argument('--fast', action='store_true',
                        help='快速双层 HRL 配置：64维/2头编码器，减少无效探索步数')
    parser.add_argument('--seed', type=int, default=2026,
                        help='全局随机种子')

    
    parser.add_argument(
        '--goal_strategy', type=str, default='adaptive',
        choices=['adaptive', 'relative', 'hybrid'],
        help='子目标嵌入策略',
    )

    
    parser.add_argument(
        '--ablation_variant', type=str, default='full',
        choices=['full', 'no_tree', 'no_dest_mask', 'single_stream', 'single_dqn', 'mlp', 'gat', 'no_il', 'noIL'],
        help=(
            'Encoder / HRL 消融变体：\n'
            '  full          完整系统（对照组）\n'
            '  no_tree       去掉树流，只保留拓扑流\n'
            '  no_dest_mask  去掉 dest_mask 目标感知\n'
            '  single_stream 最简单单流GNN基线\n'
            '  single_dqn    去掉HRL分层，退化为单层DQN\n'
            '  mlp           完全去掉GNN图结构，换成纯MLP\n'
            '  gat           替换为普通两层GAT编码器\n'
            '  no_il         不加载Phase2模仿学习权重，直接Phase3随机初始化\n'
        ),
    )
    parser.add_argument('--ablation_hop',    action='store_true',
                        help='消融：禁用 hop-distance 感知')
    parser.add_argument('--ablation_reach', action='store_true',
                        help='消融：禁用剩余目的节点可达性特征')
    parser.add_argument('--zero_candidate_feats', action='store_true',
                        help='消融：置零高/低层候选局部路径特征')
    parser.add_argument(
        '--minimal_mlp_state',
        action='store_true',
        help='强制使用纯MLP状态；--ablation_variant mlp 时自动启用',
    )
    parser.add_argument('--ablation_no_hrl', action='store_true',
                        help='兼容旧参数：等价于 --ablation_variant single_dqn')
    parser.add_argument(
        '--candidate_ablation', type=str, default='none',
        choices=['none', 'no_ac', 'no_high_ac', 'no_low_ac'],
        help=(
            'Candidate/action-mask ablation: '
            'none=full candidate constraints; '
            'no_ac=disable high VNF candidate mask and low neighbor mask; '
            'no_high_ac=disable high VNF candidate mask only; '
            'no_low_ac=disable low-level neighbor mask only'
        ),
    )
    parser.add_argument(
        '--no_il', '--noIL',
        dest='no_il',
        action='store_true',
        help='Phase3 ablation: disable Phase2 imitation-learning warm start',
    )

    
    parser.add_argument(
        '--topo', type=str, default='us_backbone',
        choices=['us_backbone', '13node', '23node', '50node'],
        help=(
            '拓扑选择：\n'
            '  us_backbone  原始 US Backbone 28节点（默认）\n'
            '  13node       14节点小型拓扑\n'
            '  23node       24节点中型拓扑\n'
            '  50node       50节点大型拓扑'
        ),
    )

    
    parser.add_argument('--bw_cap', type=int, default=None,
                        help='链路带宽容量覆盖（90/80/70/60/50/40），None=yaml默认值')
    parser.add_argument('--cap_cpu', type=int, default=None,
                        help='节点CPU容量上限覆盖，None=yaml默认值（对应 capacities.cpu）')
    parser.add_argument('--cap_mem', type=int, default=None,
                        help='节点MEM容量上限覆盖，None=yaml默认值（对应 capacities.memory）')
    parser.add_argument(
        '--resume', type=str, default=None,
        metavar='CKPT_PATH',
        help='从指定 checkpoint 继续训练（传给 Phase3RLTrainer）',
    )
    parser.add_argument(
        '--disable-checkpoint', action='store_true',
        help='运行训练但不周期性保存 checkpoint；仍保留最终统计输出',
    )
    parser.add_argument(
        '--stable-output-dir', action='store_true',
        help='保留 --output_dir 的稳定目录名，供批量实验断点续跑和结果聚合使用',
    )

    return parser


def infer_topology_from_data_path(data_path: str):
    norm = str(data_path).replace('\\', '/').lower()
    markers = [
        ('us_backbone', 'us_backbone'),
        ('50node', '50node'),
        ('23node', '23node'),
        ('13node', '13node'),
    ]
    for marker, topo in markers:
        if marker in norm:
            return topo
    return None


def main():
    """TA-HRL v4 训练启动器入口。"""
    parser = build_arg_parser()
    args   = parser.parse_args()
    data_topo = infer_topology_from_data_path(args.data)
    if data_topo is not None and data_topo != args.topo:
        logger.error(
            "Topology/data mismatch: --topo=%s but --data looks like %s (%s). "
            "Use the matching --topo or matching data directory.",
            args.topo,
            data_topo,
            args.data,
        )
        return
    if args.ablation_variant in ('no_il', 'noIL'):
        args.ablation_variant = 'full'
        args.no_il = True
    if getattr(args, 'ablation_no_hrl', False):
        if args.ablation_variant != 'single_dqn':
            logger.warning(
                "--ablation_no_hrl 已作为旧参数兼容；将 ablation_variant=%s 映射为 single_dqn",
                args.ablation_variant,
            )
        args.ablation_variant = 'single_dqn'
    if args.fast and not args.no_il and not args.resume:
        parser.error('--fast changes model dimensions; use it with --no_il or a matching --resume checkpoint')
    logger.info("CLI args: %s", sys.argv[1:])
    logger.info(
        "Parsed ablation_variant=%s candidate_ablation=%s minimal_mlp_state=%s ablation_no_hrl=%s no_il=%s",
        args.ablation_variant,
        getattr(args, 'candidate_ablation', 'none'),
        args.minimal_mlp_state,
        getattr(args, 'ablation_no_hrl', False),
        getattr(args, 'no_il', False),
    )

    
    if args.config:
        _config_key = args.config
    else:
        _config_key = args.phase

    
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"  使用 GPU: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        logger.info("  使用 CPU")

    # The agent reads config['use_cuda']; selecting a CLI device alone is not
    # sufficient. Keep the setting explicit so CPU/GPU runs are reproducible.
    config_use_cuda = bool(device.type == 'cuda')
    try:
        torch.set_num_threads(max(1, int(args.torch_threads)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Inter-op threads may already be initialized by an imported library.
        torch.set_num_threads(max(1, int(args.torch_threads)))

    
    set_seed(args.seed)
    logger.info(f" 随机种子: {args.seed}")

    
    _log_dir = os.path.join(args.output_dir, '_logs', args.phase)
    setup_file_logging(_log_dir, args.topo, args.phase, args.seed)

    
    try:
        config = load_config(_config_key)

        config['use_cuda'] = config_use_cuda
        if args.max_episode_steps > 0:
            config.setdefault('phase3', {})['max_steps_per_episode'] = int(
                args.max_episode_steps
            )
            config.setdefault('environment', {})['max_steps_per_episode'] = int(
                args.max_episode_steps
            )
            logger.info(
                f"single-episode hard limit: {args.max_episode_steps} high-level cycles"
            )
        if args.fast:
            # Keep the same HRL semantics and observation fields, but reduce
            # the small-graph encoder and dead-end search overhead.
            config.setdefault('hrl', {})['hidden_dim'] = 64
            config['hrl']['state_dim'] = 64
            config.setdefault('gnn', {})['num_heads'] = 2
            config['hrl']['high_topk'] = min(2, int(config['hrl'].get('high_topk', 3)))
            config['hrl']['low_topk'] = min(3, int(config['hrl'].get('low_topk', 5)))
            config['max_low_steps'] = min(40, int(config.get('max_low_steps', 60)))
            config['max_subgoal_steps'] = min(25, int(config.get('max_subgoal_steps', 35)))
            config.setdefault('environment', {})['max_steps_per_episode'] = min(
                400, int(config['environment'].get('max_steps_per_episode', 600))
            )
            config.setdefault('phase3', {})['max_steps_per_episode'] = min(
                400, int(config['phase3'].get('max_steps_per_episode', 600))
            )
            logger.info('fast mode enabled: hidden_dim=64 heads=2 high_topk=2 low_topk=3')
        if args.quiet:
            for _name in (
                'envs', 'envs.sfc_env', 'envs.modules', 'core',
                'core.hrl', 'core.gnn', 'trainer',
            ):
                logging.getLogger(_name).setLevel(logging.WARNING)
            logger.info('quiet mode enabled: suppressing high-frequency training logs')

        
        if args.phase == 'phase3':
            setup_hrl_config(config)
            
            config['hrl']['goal_strategy'] = args.goal_strategy

            
            
            
            config['training'] = config.get('training', {})
            config['training'].setdefault('lr_high',          1e-5)
            config['training'].setdefault('lr_low',           1e-5)
            config['training'].setdefault('batch_size',        64)
            config['training'].setdefault('gamma',            0.95)
            config['training'].setdefault('target_update_freq', 500)
            config['training'].setdefault('buffer_size',     100000)
            
            if not isinstance(config['training'].get('epsilon'), dict):
                config['training']['epsilon'] = {
                    'initial_high': 0.30, 'final_high': 0.05,
                    'initial_low':  0.30, 'final_low':  0.05,
                    'decay_episodes': 999999,
                }
            logger.info(" RL 训练超参数已注入 (TA-HRL v4 防死锁参数)")

        logger.info(f" 配置加载成功")
    except Exception as e:
        logger.error(f" 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    
    try:
        validate_config(config, args.phase)
    except ValueError as e:
        logger.error(str(e))
        return

    
    ensure_paths_exist(config)

    
    config['topo'] = args.topo
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _topo_yaml = os.path.join(_project_root, 'configs', 'topology.yaml')
    if not os.path.exists(_topo_yaml):
        logger.error(f" topology.yaml 不存在: {_topo_yaml}")
        return
    with open(_topo_yaml, 'r', encoding='utf-8') as _f:
        _topo_registry = yaml.safe_load(_f)
    _topologies = _topo_registry.get('topologies', {})
    if args.topo not in _topologies:
        logger.error(f" 拓扑 '{args.topo}' 未定义，可用: {list(_topologies.keys())}")
        return
    _topo_info = _topologies[args.topo]

    
    config.setdefault('topology', {})['file']     = os.path.join(_project_root, _topo_info['file'])
    config.setdefault('topology', {})['dc_nodes'] = _topo_info['dc_nodes']
    config.setdefault('environment', {})['dc_nodes'] = _topo_info['dc_nodes']

    
    if not load_topology(config):
        logger.error(" 拓扑矩阵加载失败，无法继续")
        return

    
    _num_nodes = config['topology']['matrix'].shape[0]
    config['environment']['num_nodes']            = _num_nodes
    config['environment']['nb_high_level_goals']  = _num_nodes
    config['environment']['nb_low_level_actions'] = _num_nodes
    logger.info(f"  拓扑参数注入：num_nodes={_num_nodes}, dc_nodes={_topo_info['dc_nodes']}")

    # =========================================================================
    
    # =========================================================================
    
    if args.bw_cap is not None:
        config.setdefault('env', {})['link_capacity'] = float(args.bw_cap)
        config.setdefault('capacities', {})['bandwidth'] = float(args.bw_cap)
        logger.info(f" 链路带宽容量覆盖: {args.bw_cap}")

    
    if args.cap_cpu is not None:
        config.setdefault('capacities', {})['cpu'] = float(args.cap_cpu)
        logger.info(f" 节点CPU容量覆盖: {args.cap_cpu}")

    if args.cap_mem is not None:
        config.setdefault('capacities', {})['memory'] = float(args.cap_mem)
        logger.info(f" 节点MEM容量覆盖: {args.cap_mem}")

    logger.info(" Initializing Environment (Global)...")
    try:
        
        env = SFC_HIRL_Env(config, use_gnn=True)

        
        inject_dynamic_dimensions(config, env)

        logger.info(" Environment Initialized Successfully")

        
        try:
            print("\n" + "=" * 40)
            if hasattr(env.resource_mgr, 'nodes'):
                nodes = env.resource_mgr.nodes
                
                if isinstance(nodes, dict):
                    cpu_data = nodes.get('cpu', [])
                    print(f" CPU配置 (前5个): {cpu_data[:5] if len(cpu_data) > 0 else '空'}")
                    mem_data = nodes.get('memory', [])
                    print(f" MEM配置 (前5个): {mem_data[:5] if len(mem_data) > 0 else '空'}")
                
                elif hasattr(nodes, 'shape'):
                    print(f" CPU配置 (前5个): {nodes[:5, 0]}")
                
                else:
                    print(f" CPU配置 (原始): {nodes}")
            else:
                print(" resource_mgr.nodes 属性不存在")

            
            bw_cap = config.get('capacities', {}).get('bandwidth', '未知')
            print(f" 默认带宽配置: {bw_cap}")
            print("=" * 40 + "\n")

            
            try:
                import sys as _sys
                _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from check_bw_bidir import check_bw_bidir, print_bw_snapshot
                check_bw_bidir(env.resource_mgr)
            except ImportError:
                print(" check_bw_bidir.py 不在同目录，跳过双向BW验证")
            except Exception as _bw_e:
                print(f" 双向BW验证失败: {_bw_e}")
            
        except Exception as e:
            print(f" 资源打印失败: {e}")
    except Exception as e:
        logger.error(f" 环境初始化崩溃: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Phase 1: Expert Data Collection
    # =========================================================================
    if args.phase == 'phase1':
        logger.info("=" * 70)
        logger.info(" Phase 1: Expert Data Collection")
        logger.info("=" * 70)

        try:
            
            from core.expert.expert_msfce.core.solver import MSFCE_Solver
            from core.expert.expert_msfce.core.solver import SolverConfig
            from pathlib import Path

            ckpt_dir  = get_config_path(config, 'ckpt_dir')
            input_dir = config.get('path', config.get('paths', {})).get('input_dir', 'data/input_dir')
            _topo_key2 = config.get('topo', 'us_backbone')
            _pr = os.path.dirname(os.path.abspath(__file__))
            _PATH_DB_MAP = {
                'us_backbone': os.path.join(_pr, 'topo', 'US_Backbone_path.mat'),
                '13node':      os.path.join(_pr, 'topo', 'path_db_13node.mat'),
                '23node':      os.path.join(_pr, 'topo', 'path_db_23node.mat'),
                '50node':      os.path.join(_pr, 'topo', 'path_db_50node.mat'),
            }
            path_db = Path(_PATH_DB_MAP.get(_topo_key2, os.path.join(_pr, 'topo', 'US_Backbone_path.mat')))

            topo_matrix = config['topology']['matrix']
            dc_nodes = list(config['topology']['dc_nodes'])
            configured_capacities = config.get('capacities', {})
            capacities = {
                'cpu': float(configured_capacities.get('cpu', 55.0)),
                'memory': float(configured_capacities.get('memory', 45.0)),
                'bandwidth': float(configured_capacities.get('bandwidth', 90.0)),
            }

            expert_solver = MSFCE_Solver(
                path_db_file=path_db,
                topology_matrix=topo_matrix,
                dc_nodes=dc_nodes,
                capacities=capacities,
                config=SolverConfig(),
            )
            logger.info(" Expert Solver 已加载")
        except Exception as e:
            logger.error(f" Expert Solver 获取失败: {e}")
            import traceback; traceback.print_exc()
            return

        output_dir = get_config_path(config, 'expert_data_dir')
        max_episodes = config.get("phase1", {}).get("max_episodes", 15000)
        save_every = config.get("phase1", {}).get("save_every", 500)

        
        _p1_data = args.data.replace('phase3_requests', 'phase1_requests')
        logger.info(f" Phase1 数据文件: {_p1_data}")
        if not os.path.exists(_p1_data):
            logger.error(f" Phase1 数据不存在: {_p1_data}")
            logger.error("   请先运行 Generate_all_datasets.py --overwrite 生成 phase1 数据")
            return
        _p1_dir = os.path.dirname(os.path.abspath(_p1_data))
        config.setdefault('paths', {})['input_dir'] = _p1_dir
        config.setdefault('paths', {})['data_dir'] = _p1_dir
        config.setdefault('path', {})['input_dir'] = _p1_dir
        if hasattr(env, 'config'):
            env.config.setdefault('paths', {})['input_dir'] = _p1_dir
            env.config.setdefault('paths', {})['data_dir'] = _p1_dir
            env.config.setdefault('path', {})['input_dir'] = _p1_dir
        env.load_dataset(_p1_data)
        logger.info(f" Phase1 数据加载成功")

        collector = Phase1ExpertCollector(
            env=env,
            expert_solver=expert_solver,
            output_dir=output_dir,
            max_episodes=max_episodes,
            save_every=save_every,
        )

        try:
            collector.collect()
            logger.info(" Phase 1 完成")
        except Exception as e:
            logger.error(f" Phase 1 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Phase 2: Imitation Learning
    # =========================================================================
    elif args.phase == 'phase2':
        logger.info("=" * 70)
        logger.info(" Phase 2: Imitation Learning")
        logger.info("=" * 70)

        try:
            logger.info(" 初始化 Phase 2 Agent...")
            agent = GoalConditionedHRLAgent(config, phase=2)
            logger.info(" Agent 初始化成功")
            logger.info(f"   模式: Phase {agent.phase}")
            logger.info(f"   动作空间: {agent.n_actions}")
            logger.info(f"   设备: {agent.device}")

            
            if hasattr(agent, 'high_policy') and hasattr(agent, 'low_policy'):
                logger.info("    检测到 HRL Agent (双层策略网络)")
            elif hasattr(agent, 'policy_net'):
                logger.info("    检测到 Legacy Agent (单层策略网络)")
            else:
                logger.error("    无法识别 Agent 结构: 既没有 policy_net 也没有 high/low policy")
                return

        except Exception as e:
            logger.error(f" Agent 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        data_file = "expert_data_final.pkl"
        expert_data_dir = get_config_path(config, 'expert_data_dir')
        data_path = os.path.join(expert_data_dir, data_file)

        if not os.path.exists(data_path):
            logger.error(f" 专家数据不存在: {data_path}")
            logger.error("   请先运行 Phase 1 收集数据")
            return

        phase2_config = config.get('phase2', {})
        phase2_config['node_feat_dim'] = config.get('gnn', {}).get('node_feat_dim', 24)
        
        _topo_name = config.get('topo', config.get('env', {}).get('topo', 'unknown'))
        _topo_short = {'us_backbone': 'us', '13node': '13', '23node': '23', '50node': '50'}.get(_topo_name, _topo_name)
        _project_root = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(args.output_dir, 'il', _topo_short)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f" IL 模型将保存至: {output_dir}")

        try:
            
            trainer = Phase2ILTrainer(
                agent=agent,
                env=env,
                expert_data_path=data_path,
                output_dir=output_dir,
                config=phase2_config
            )
            logger.info(" Phase2 Trainer 初始化成功")
        except Exception as e:
            logger.error(f" Trainer 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            trainer.run()
            logger.info(" Phase 2 完成")
        except Exception as e:
            logger.error(f" Phase 2 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Phase 3: Goal-Conditioned RL
    # =========================================================================
    elif args.phase == 'phase3':
        logger.info("=" * 70)
        logger.info(" Phase 3: Goal-Conditioned RL Fine-tuning")
        logger.info("=" * 70)

        
        try:
            logger.info(" 初始化 Goal-Conditioned Agent...")

            
            config['ablation_variant'] = args.ablation_variant
            config['candidate_ablation'] = args.candidate_ablation
            config['no_il'] = bool(getattr(args, 'no_il', False))
            config.setdefault('phase3', {})['no_il'] = bool(getattr(args, 'no_il', False))
            env._ablation_variant = args.ablation_variant
            env.ablation_variant = args.ablation_variant
            env._candidate_ablation = args.candidate_ablation
            env.candidate_ablation = args.candidate_ablation
            minimal_mlp_state = bool(args.minimal_mlp_state or args.ablation_variant == 'mlp')
            env._minimal_mlp_state = minimal_mlp_state
            env._ablation_hop = bool(args.ablation_hop or minimal_mlp_state)
            env._ablation_reach = bool(args.ablation_reach or minimal_mlp_state)
            env._ablation_candidate_feats = bool(args.zero_candidate_feats or minimal_mlp_state)
            logger.info(
                "Ablations: candidate=%s hop=%s reach=%s candidate_feats_zero=%s minimal_mlp_state=%s no_il=%s",
                env._candidate_ablation,
                env._ablation_hop,
                env._ablation_reach,
                env._ablation_candidate_feats,
                env._minimal_mlp_state,
                config['no_il'],
            )

            agent = create_goal_conditioned_agent(
                config=config,
                phase=3,
                goal_strategy=args.goal_strategy,
                env=env,
                ablation_variant=args.ablation_variant,
            )

            
            if args.ablation_variant == 'single_dqn':
                for param in agent.high_policy.parameters():
                    param.requires_grad = False
                logger.info("[Ablation] w/o HRL active: high policy frozen and Coordinator will bypass high policy")

            logger.info(" Agent 初始化成功")

            
            
            logger.info(" Agent 初始化成功 (已自动挂载 TreeTransformerEncoder)")

            _enc = getattr(agent, 'encoder', None)
            _enc_class = _enc.__class__.__name__ if _enc is not None else 'None'
            if args.ablation_variant == 'gat' and _enc_class != 'GATEncoder':
                raise RuntimeError(
                    f'--ablation_variant gat expected GATEncoder, got {_enc_class}'
                )
            logger.info(
                f'Agent encoder: variant={args.ablation_variant}, class={_enc_class}'
            )

        except Exception as e:
            logger.error(f" Agent 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        

        # =========================================================
        
        # =========================================================
        from envs.modules.HRL_Coordinator import HRL_Coordinator

        logger.info(" 初始化 HRL Coordinator...")
        try:
            coordinator = HRL_Coordinator(
                env=env,
                high_agent=agent,
                low_agent=agent,
                config=config
            )
            logger.info(" HRL Coordinator 初始化成功")
        except Exception as e:
            logger.error(f" HRL Coordinator 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # =========================================================
        
        # =========================================================
        logger.info("=" * 70)
        logger.info(" 加载Phase3训练数据")
        logger.info("=" * 70)

        data_file = args.data
        logger.info(f" 数据文件路径: {data_file}")
        logger.info(f" 当前工作目录: {os.getcwd()}")
        logger.info(f" 文件是否存在: {os.path.exists(data_file)}")

        if os.path.exists(data_file):
            logger.info(f" 文件存在，开始加载...")
            try:
                success = env.load_dataset(data_file)
                if success:
                    logger.info(f" 数据加载成功")

                    if int(args.max_requests) > 0:
                        loaded_requests = list(getattr(env, 'all_requests', []) or [])
                        if len(loaded_requests) < int(args.max_requests):
                            raise ValueError(
                                f"--max_requests={args.max_requests} exceeds loaded dataset size "
                                f"{len(loaded_requests)}"
                            )
                        selected_requests = loaded_requests[: int(args.max_requests)]
                        # Rebuild the environment's time-slot index so the
                        # trainer horizon and lifecycle simulation use exactly
                        # the selected prefix.
                        env.load_requests(selected_requests)
                        logger.info(
                            " 已截取训练请求: %d/%d 条",
                            len(selected_requests), len(loaded_requests),
                        )

                    
                    if hasattr(env, 'all_requests') and env.all_requests is not None:
                        dataset_size = len(env.all_requests)
                    elif hasattr(env, 'dataset') and env.dataset is not None:
                        dataset_size = len(env.dataset)
                    elif hasattr(env, 'requests') and env.requests is not None:
                        dataset_size = len(env.requests)
                    else:
                        dataset_size = None
                        logger.warning(" 未找到 env.dataset 或 env.requests，decay_episodes 回退到 yaml 值")

                    if dataset_size is not None:
                        _decay_ep = int(dataset_size * 0.8)
                        config['training']['epsilon']['decay_episodes'] = _decay_ep
                        logger.info(f" decay_episodes 自适应数据集: {dataset_size} × 0.8 = {_decay_ep}")
                    else:
                        _decay_ep = config['training']['epsilon'].get('decay_episodes', 999999)
                        logger.info(f" decay_episodes 使用 yaml 值: {_decay_ep}")

                    agent.epsilon_decay_episodes = int(_decay_ep)
                    agent.epsilon_decay = float(_decay_ep)
                    agent.epsilon_high = agent.epsilon_high_start
                    agent.epsilon_low = agent.epsilon_low_start
                    agent.total_episodes = 0
                    logger.info(f" agent epsilon 已同步: "
                                f"ε_high=[{agent.epsilon_high_start}→{agent.epsilon_high_end}] "
                                f"ε_low=[{agent.epsilon_low_start}→{agent.epsilon_low_end}] "
                                f"decay_episodes={agent.epsilon_decay_episodes}")

                    
                    logger.info(" 验证数据加载...")
                    test_state = env.reset()
                    if env.current_request:
                        vnf_list = env.current_request.get('vnf', [])
                        logger.info(f" 验证成功：请求存在，VNF数量={len(vnf_list)}")
                    else:
                        logger.warning(" 验证警告：reset后无请求")
                else:
                    logger.error(f" load_dataset返回False")
                    raise RuntimeError("数据加载失败")
            except Exception as e:
                logger.error(f" 数据加载异常: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            logger.error(f" 文件不存在: {data_file}")

            
            logger.info(" 检查data/input_dir/目录内容...")
            input_dir = 'data/input_dir'
            if os.path.exists(input_dir):
                files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
                logger.info(f" 找到的.pkl文件:")
                for f in files:
                    logger.info(f"   - {f}")
            else:
                logger.error(f" 目录不存在: {input_dir}")

            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        logger.info("=" * 70)

        # =========================================================
        
        # =========================================================
        
        _topo_name_p3 = config.get('topo', config.get('env', {}).get('topo', 'unknown'))
        _topo_short_p3 = {'us_backbone': 'us', '13node': '13', '23node': '23', '50node': '50'}.get(_topo_name_p3, _topo_name_p3)
        _project_root_p3 = os.path.dirname(os.path.abspath(__file__))
        
        _il_ckpt_dir = os.path.join(args.output_dir, 'il', _topo_short_p3)
        _bundled_il_ckpt = os.path.join(
            _project_root_p3, 'ilModel', _topo_short_p3, 'il_model_best.pth'
        )
        if not getattr(args, 'no_il', False):
            os.makedirs(_il_ckpt_dir, exist_ok=True)
            _local_il_ckpt = os.path.join(_il_ckpt_dir, 'il_model_best.pth')
            if os.path.isfile(_local_il_ckpt):
                _phase3_il_ckpt = _local_il_ckpt
                logger.info(f" Phase3 将从 {_phase3_il_ckpt} 加载 IL 权重")
            elif os.path.isfile(_bundled_il_ckpt):
                _phase3_il_ckpt = _bundled_il_ckpt
                logger.info(f" Phase3 使用包内 IL 权重: {_phase3_il_ckpt}")
            else:
                _phase3_il_ckpt = _local_il_ckpt
                logger.info(f" Phase3 将从 {_phase3_il_ckpt} 加载 IL 权重")
        
        _base_ckpt = (
            getattr(args, 'output_dir', None)
            or get_config_path(config, 'ckpt_dir')
            or os.path.join(_project_root_p3, 'artifacts', 'runs', 'hrl', 'checkpoints')
        )
        
        _data_arg  = getattr(args, 'data', '') or ''
        import re as _re_rate
        _rate_m    = _re_rate.search(r'rate(\d+)', _data_arg)
        _rate_tag  = f"rate{_rate_m.group(1)}" if _rate_m else ''
        _abl_tag   = getattr(args, 'ablation_variant', 'full')
        _cand_tag  = getattr(args, 'candidate_ablation', 'none')
        _no_il_tag = 'noIL' if getattr(args, 'no_il', False) else ''
        _variant_tag = '_'.join(filter(None, [
            _abl_tag if _abl_tag != 'full' else '',
            _cand_tag if _cand_tag != 'none' else '',
            _no_il_tag,
        ])) or 'full'
        _run_tag   = '_'.join(filter(None, [_variant_tag if _variant_tag != 'full' else '', _rate_tag]))
        ckpt_dir   = os.path.join(_base_ckpt, _run_tag) if _run_tag else _base_ckpt
        os.makedirs(ckpt_dir, exist_ok=True)
        config.setdefault('hrl', {})['loss_log_path'] = os.path.join(
            ckpt_dir, 'loss_log.csv'
        )
        logger.info(f" Phase3 训练输出目录: {ckpt_dir}")
        
        config.setdefault('phase3', {})['no_il'] = bool(getattr(args, 'no_il', False))
        if getattr(args, 'no_il', False):
            config['phase3']['il_checkpoint'] = ''
            logger.info("Phase3 noIL ablation enabled: skip Phase2 IL warm start")
        else:
            config['phase3']['il_checkpoint'] = _phase3_il_ckpt
        trainer_cls = (
            Phase3RLTrainerNoCheckpoint
            if args.disable_checkpoint
            else Phase3RLTrainer
        )
        trainer = trainer_cls(
            env=env,
            agent=agent,
            output_dir=ckpt_dir,
            config=config,
            coordinator=coordinator,
        )
        
        if args.resume:
            if hasattr(trainer, 'load_checkpoint'):
                trainer.load_checkpoint(args.resume)
                logger.info(f" 已从 checkpoint 恢复: {args.resume}")
            elif hasattr(agent, 'load'):
                agent.load(args.resume)
                logger.info(f" 已加载 agent checkpoint: {args.resume}")
            else:
                logger.warning(f" --resume 指定了 {args.resume}，但 trainer/agent 均无 load 接口，已忽略")

        # =========================================================
        
        # =========================================================
        try:
            
            if dataset_size is not None and hasattr(trainer, 'set_dataset_size'):
                trainer.set_dataset_size(dataset_size)
            trainer.run()
            if args.stable_output_dir:
                logger.info(f" Phase3 保留稳定输出目录: {ckpt_dir}")
            else:
                ckpt_dir = finalize_phase3_output_dir(ckpt_dir, _rate_tag, _variant_tag)
            logger.info(" Phase 3 完成")
        except Exception as e:
            logger.error(f" Phase 3 执行失败: {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 70)
    logger.info(" 程序执行完成")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
