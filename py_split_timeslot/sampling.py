#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""采样函数：寿命、带宽、CPU、内存"""

import random
import numpy as np

# ── 固定 VNF 系数表（与 vnf_catalog.py 保持一致）────────────────────────────
_FIXED_VNF_TABLE = [
    {'type': 1, 'cpu_need': 2.17571289793936,  'memory_need': 1.06847854592062},
    {'type': 2, 'cpu_need': 0.327471689918480, 'memory_need': 1.67212031605700},
    {'type': 3, 'cpu_need': 0.691000236990969, 'memory_need': 1.84641423103628},
    {'type': 4, 'cpu_need': 0.401957140875833, 'memory_need': 1.30930392240156},
    {'type': 5, 'cpu_need': 0.804261986990572, 'memory_need': 1.66085909342891},
    {'type': 6, 'cpu_need': 2.83555094177359,  'memory_need': 1.09494296372459},
    {'type': 7, 'cpu_need': 1.64532198696671,  'memory_need': 0.296237077563595},
    {'type': 8, 'cpu_need': 0.983307643923246, 'memory_need': 0.754735724747883},
]


def sample_lifetime(mean_lifetime=3.0, max_lifetime=6.0):
    """MATLAB 语义：1 + 指数分布，且截断到 <= 6。"""
    lifetime = 1.0 + np.random.exponential(mean_lifetime - 1.0)
    while lifetime > max_lifetime:
        lifetime = 1.0 + np.random.exponential(mean_lifetime - 1.0)
    return lifetime


def sample_bandwidth():
    """
    带宽加权采样：
      4, 5, 6 -> 合计 70%（各约 23.3%）
      7, 8    -> 合计 30%（各 15%）
    """
    return random.choices(
        population=[4,  5,  6,  7,  8],
        weights=   [23, 24, 23, 15, 15],
        k=1
    )[0]


def sample_cpu_needs(vnf_chain, bandwidth, all_vnf=None):
    """
    按固定 VNF 系数 x bandwidth 计算 CPU 需求，结果 clip 到 [1, 24]。
    all_vnf 参数保留以兼容旧调用方，实际使用内置 _FIXED_VNF_TABLE。
    """
    cpu_needs = []
    for vnf_id in vnf_chain:
        coeff = _FIXED_VNF_TABLE[vnf_id - 1]['cpu_need']
        val = max(1, min(24, int(round(bandwidth * coeff))))
        cpu_needs.append(val)
    return cpu_needs


def sample_memory_needs(vnf_chain, bandwidth, all_vnf=None):
    """
    按固定 VNF 系数 x bandwidth 计算内存需求，结果 clip 到 [1, 16]。
    all_vnf 参数保留以兼容旧调用方，实际使用内置 _FIXED_VNF_TABLE。
    """
    mem_needs = []
    for vnf_id in vnf_chain:
        coeff = _FIXED_VNF_TABLE[vnf_id - 1]['memory_need']
        val = max(1, min(16, int(round(bandwidth * coeff))))
        mem_needs.append(val)
    return mem_needs