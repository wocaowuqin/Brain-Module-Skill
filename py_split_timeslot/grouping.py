#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""分组函数"""


def group_requests_by_time_slot(requests):
    """将请求按时间槽分组。"""
    grouped = {}
    for req in requests:
        slot = req['time_slot']
        if slot not in grouped:
            grouped[slot] = []
        grouped[slot].append(req)
    return grouped
