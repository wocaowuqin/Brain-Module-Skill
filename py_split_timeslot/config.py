#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py（多拓扑版）
====================
修改 TOPO 变量切换拓扑，运行 generate_dataset.py 生成对应数据集
"""

import os

# ── 选择拓扑（改这里）────────────────────────────────────────────────────────
TOPO = '23node'   # 'us_backbone' / '13node' / '23node' / '50node'
# ─────────────────────────────────────────────────────────────────────────────

_TOPO_PARAMS = {
    # US Backbone 28节点（原始）
    # 8个源节点，比例 28.6%，全局到达率 16 req/s
    'us_backbone': {
        'num_nodes':         28,
        'NODE_TRAFFIC_LIST': [2, 5, 6, 10, 15, 16, 21, 24],
        'data_subdir':       '时间间隔为5请求到达率为16时',
    },

    # 14节点小型拓扑
    # 4个源节点（度≤6的边缘节点），比例 28.6%，全局到达率 8 req/s
    '13node': {
        'num_nodes':         14,
        'NODE_TRAFFIC_LIST': [0, 3, 7, 11],
        'data_subdir':       '13node_rate8',
    },

    # 24节点中型拓扑
    # 7个源节点（度≤6的边缘节点），比例 29.2%，全局到达率 14 req/s
    '23node': {
        'num_nodes':         24,
        'NODE_TRAFFIC_LIST': [0, 4, 7, 13, 15, 17, 19],
        'data_subdir':       '23node_rate14',
    },

    # 50节点大型拓扑
    # 14个源节点（度≤6的边缘节点），比例 28.0%，全局到达率 28 req/s
    '50node': {
        'num_nodes':         50,
        'NODE_TRAFFIC_LIST': [0, 1, 2, 6, 7, 8, 9, 12, 14, 15, 17, 20, 26, 27],
        'data_subdir':       '50node_rate28',
    },
}

if TOPO not in _TOPO_PARAMS:
    raise ValueError(f"未知拓扑：{TOPO}，可选：{list(_TOPO_PARAMS.keys())}")

_p = _TOPO_PARAMS[TOPO]

DATA_DIR          = os.path.join('./data/input_dir', _p['data_subdir'])
NODE_TRAFFIC_LIST = _p['NODE_TRAFFIC_LIST']

# ── 时间与负载参数 ────────────────────────────────────────────────────────────
TIME_INTERVAL        = 5.0
LAMBDA_PER_INTERVAL  = 10          # 每个时间间隔每个源节点产生的请求数
LAMBDA_RATE          = LAMBDA_PER_INTERVAL / TIME_INTERVAL  # 单节点 2 req/s
TIME_SLOT_DELTA      = 0.1
MIN_LIFETIME         = 1.0
MAX_LIFETIME         = 6.0
MEAN_LIFETIME        = 1.8

# ── 业务请求参数 ──────────────────────────────────────────────────────────────
NUM_DESTINATIONS = 5
VNF_CHAIN_LENGTH = 3
VNF_TYPES        = 8
MIN_BANDWIDTH    = 4
MAX_BANDWIDTH    = 8

# ── 双阶段生成参数 ────────────────────────────────────────────────────────────
PHASE1_NUM_INTERVALS = 80
PHASE1_SEED          = 42
PHASE3_NUM_INTERVALS = 80
PHASE3_SEED          = 2026

# ── 打印当前配置 ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    n_src = len(NODE_TRAFFIC_LIST)
    print(f"当前拓扑：{TOPO}")
    print(f"  节点数：      {_p['num_nodes']}")
    print(f"  源节点数：    {n_src}个（比例 {n_src/_p['num_nodes']:.1%}）")
    print(f"  源节点列表：  {NODE_TRAFFIC_LIST}")
    print(f"  全局到达率：  {LAMBDA_RATE * n_src:.0f} req/s")
    print(f"  数据输出目录：{DATA_DIR}")
    print(f"  Phase3 总时长：{TIME_INTERVAL * PHASE3_NUM_INTERVALS:.0f}s")
    print(f"  Phase3 预计请求数：{LAMBDA_PER_INTERVAL * n_src * PHASE3_NUM_INTERVALS}")