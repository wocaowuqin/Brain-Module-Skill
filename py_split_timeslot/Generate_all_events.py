#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess

# ── 硬编码配置 ────────────────────────────────────────────────────────────────
DATA_GEN_DIR = r'E:\pycharmworkspace\HRL-GNN for Multicast-aware SFC Orchestration\py_split_timeslot'
TOPOS        = ['us_backbone', '13node', '50node']
RATES        = [8, 16, 24, 32, 40, 48]
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script = os.path.join(DATA_GEN_DIR, 'main_generate_events_split.py')
    total  = len(TOPOS) * len(RATES)
    idx    = 0

    print(f"批量生成 events 文件")
    print(f"  拓扑: {TOPOS}  到达率: {RATES}")
    print(f"  脚本: {script}")
    print("=" * 60)

    for topo in TOPOS:
        for rate in RATES:
            idx += 1
            print(f"\n[{idx:02d}/{total}] topo={topo}  rate={rate}")
            cmd = [sys.executable, script, '--topo', topo, '--rate', str(rate)]
            result = subprocess.run(cmd, cwd=DATA_GEN_DIR)
            if result.returncode != 0:
                print(f"  ❌ 失败: topo={topo} rate={rate}")
            else:
                print(f"  ✅ 完成: topo={topo} rate={rate}")

    print("\n" + "=" * 60)
    print("所有 events 生成完毕")

if __name__ == '__main__':
    main()