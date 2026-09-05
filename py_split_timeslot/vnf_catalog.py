#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""VNF 目录生成（沿用你原 data_generator.py 的逻辑）"""

import random


# def generate_vnfs_catalog(vnf_type_num=8):
#     all_vnf = []
#     for vnf_type in range(1, vnf_type_num + 1):
#         cpu_need = random.random() * 2.75 + 0.25
#         memory_need = random.random() * 1.75 + 0.25
#         all_vnf.append({
#             'type': vnf_type,
#             'cpu_need': cpu_need,
#             'memory_need': memory_need,
#         })
#     return all_vnf
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 固定 VNF 系数，每次运行保持一致
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


def generate_vnfs_catalog(vnf_type_num=8):
    """返回固定 VNF 目录，忽略 vnf_type_num（保持接口兼容）。"""
    return [dict(v) for v in _FIXED_VNF_TABLE[:vnf_type_num]]
