#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate_all.py
===============
一键生成所有实验所需数据：
  - 拓扑:   us_backbone, 13node, 50node
  - 到达率: 8, 16, 24, 32, 40, 48
  - 阶段:   phase1（含 events）、phase3

用法：直接运行，或加 --overwrite 覆盖已有文件
"""

import os
import sys
import subprocess
import logging
from itertools import product

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── 硬编码配置 ────────────────────────────────────────────────────────────────
DATA_GEN_DIR = r'E:\pycharmworkspace\HRL-GNN for Multicast-aware SFC Orchestration\py_split_timeslot'
DATA_ROOT    = r'E:\pycharmworkspace\HRL-GNN for Multicast-aware SFC Orchestration\data'
TOPOS        = ['us_backbone', '13node', '50node']
RATES        = [8, 16, 24, 32, 40, 48]
SIM_DURATION = 400
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd, desc=""):
    logger.info(f"  ▶ {desc}")
    result = subprocess.run(cmd, cwd=DATA_GEN_DIR, timeout=600)
    if result.returncode != 0:
        logger.error(f"  ❌ 失败: {desc}")
        return False
    logger.info(f"  ✅ 完成: {desc}")
    return True


def exists(topo, rate, phase):
    path = os.path.join(DATA_ROOT, f'{topo}_rate{rate}', f'{phase}_requests.pkl')
    return os.path.exists(path)


def events_exist(topo, rate, phase):
    path = os.path.join(DATA_ROOT, f'{topo}_rate{rate}', f'{phase}_events.pkl')
    return os.path.exists(path)


def generate_requests(topo, rate, phase, overwrite):
    if not overwrite and exists(topo, rate, phase):
        logger.info(f"  ⏭️  跳过（已存在）: {topo}_rate{rate}/{phase}_requests.pkl")
        return True
    num_req = rate * SIM_DURATION
    script  = os.path.join(DATA_GEN_DIR, 'main_generate_split.py')
    cmd = [sys.executable, script,
           '--topo', topo, '--rate', str(rate),
           '--num_requests', str(num_req),
           '--phase', phase, '--seed', str(SEED)]
    return run(cmd, f"{phase} requests: topo={topo} rate={rate} n={num_req}")


def generate_events(topo, rate, overwrite):
    if not overwrite and events_exist(topo, rate, 'phase1'):
        logger.info(f"  ⏭️  跳过（已存在）: {topo}_rate{rate}/phase1_events.pkl")
        return True
    script = os.path.join(DATA_GEN_DIR, 'main_generate_events_split.py')
    cmd = [sys.executable, script, '--topo', topo, '--rate', str(rate)]
    return run(cmd, f"events: topo={topo} rate={rate}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help='覆盖已有文件')
    args = parser.parse_args()

    total     = len(TOPOS) * len(RATES)
    success   = 0
    fail_list = []

    logger.info("=" * 60)
    logger.info(f"生成计划: {len(TOPOS)}个拓扑 × {len(RATES)}个到达率 = {total}组")
    logger.info(f"阶段: phase1 + phase3（含 phase1 events）")
    logger.info(f"覆盖已有: {args.overwrite}")
    logger.info("=" * 60)

    for i, (topo, rate) in enumerate(product(TOPOS, RATES), 1):
        logger.info(f"\n[{i:02d}/{total}] topo={topo}  rate={rate}")

        ok = True

        # 1. 生成 phase1 请求
        if not generate_requests(topo, rate, 'phase1', args.overwrite):
            ok = False

        # 2. 生成 phase3 请求
        if not generate_requests(topo, rate, 'phase3', args.overwrite):
            ok = False

        # 3. 生成 events（phase1+phase3 共用同一个脚本，会同时生成两个 phase 的 events）
        if ok and not generate_events(topo, rate, args.overwrite):
            ok = False

        if ok:
            success += 1
        else:
            fail_list.append((topo, rate))

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("生成完毕汇总")
    logger.info(f"  ✅ 成功: {success}/{total}")
    logger.info(f"  ❌ 失败: {len(fail_list)}")
    for topo, rate in fail_list:
        logger.warning(f"    失败: topo={topo} rate={rate}")
    logger.info("=" * 60)

    # 验证输出
    logger.info("\n文件检查:")
    for topo, rate in product(TOPOS, RATES):
        for phase in ['phase1', 'phase3']:
            path = os.path.join(DATA_ROOT, f'{topo}_rate{rate}', f'{phase}_requests.pkl')
            ev   = os.path.join(DATA_ROOT, f'{topo}_rate{rate}', f'{phase}_events.pkl')
            r_ok = "✅" if os.path.exists(path) else "❌"
            e_ok = "✅" if os.path.exists(ev)   else "❌"
            logger.info(f"  {r_ok} requests  {e_ok} events  | {topo}_rate{rate}/{phase}")

    sys.exit(0 if not fail_list else 1)


if __name__ == '__main__':
    main()