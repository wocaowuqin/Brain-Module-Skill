#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSFCE Expert Algorithm - Modular Version

优化版本特性：
1.  模块化架构，职责清晰
2.  三级缓存系统：路径缓存 + 链路缓存 + 距离矩阵
3.  O(1) 路径查询和距离查询
4.  向量化资源检查
5.  完整的性能指标收集
"""

import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)


from core.expert.expert_msfce.core.solver import (
    MSFCE_Solver,
    SolverConfig,
    MetricsCollector,
)


def parse_mat_request(*args, **kwargs):
    raise NotImplementedError("parse_mat_request is not available in the merged MSFCE solver")


def validate_request(*args, **kwargs):
    return True


def validate_state(*args, **kwargs):
    return True

__version__ = "2.0.0"
__all__ = [
    'MSFCE_Solver',
    'SolverConfig',
    'parse_mat_request',
    'MetricsCollector',
    'validate_request',
    'validate_state'
]
