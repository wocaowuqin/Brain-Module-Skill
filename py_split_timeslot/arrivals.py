#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""泊松到达时间生成"""

import numpy as np


def generate_poisson_arrivals(T, lamda):
    """生成泊松到达时间序列。"""
    arrivals = []
    time_state = 0.0
    while time_state < T:
        interval = np.random.exponential(1.0 / lamda)
        time_state += interval
        if time_state < T:
            arrivals.append(time_state)
    return arrivals
