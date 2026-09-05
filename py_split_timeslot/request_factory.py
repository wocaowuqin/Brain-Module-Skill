#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单个请求构造"""

import numpy as np


def generate_single_request(req_id, source, destinations, vnf_chain, bandwidth,
                            cpu_needs, mem_needs, arrive_time, lifetime,
                            delta_t=0.1):
    """生成单个请求对象（时间槽版本）。"""
    leave_time = arrive_time + lifetime

    arrive_time_step = int(np.ceil(arrive_time))
    leave_time_step = int(np.ceil(leave_time))

    time_slot = int(arrive_time / delta_t)
    leave_time_slot = int(leave_time / delta_t)
    duration = leave_time_slot - time_slot

    return {
        'id': req_id,
        'source': source,
        'dest': destinations,
        'vnf': vnf_chain,
        'bw_origin': bandwidth,
        'cpu_origin': cpu_needs,
        'memory_origin': mem_needs,
        'arrival_time': arrive_time,
        'leave_time': leave_time,
        'lifetime': lifetime,
        'time_slot': time_slot,
        'leave_time_slot': leave_time_slot,
        'duration': duration,
        'arrive_time_step': arrive_time_step,
        'leave_time_step': leave_time_step,
    }
