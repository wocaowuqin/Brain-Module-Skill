# All-variant arrival sweep

默认任务数为 `2 拓扑 x 3 到达率 x 8 算法 = 48` 个独立任务。

入口脚本：

```text
C:/Users/11353/Desktop/hrl_marl_reconfig_starter/scripts/run_all_variants_arrival_sweep.py
```

默认比较两个拓扑 `us_backbone` 和 `germany50`，每个源节点到达率为
`0.5, 1.5, 2.5 req/s`，算法为图中 8 个变体：
`MSFT-HIRL`, `MSFT-HRL`, `MSFT-ILRL`, `MSFT-HIRL-GAT`,
`MSFT-HIRL-MLP`, `MSFC-CE`, `PPO`, `A2C`。
`germany50` 使用论文中的 `topo/50node.mat`，不是另建一个拓扑。

## 运行

先生成缺失的半档数据（整数档会尽量复用旧数据）：

```powershell
$PY = 'C:\Users\11353\.conda\envs\sfc_ppo\python.exe'
& $PY scripts\generate_rl_arrival_sweep.py --rates 0.5 1.5 2.5
```

先跑 100 条验证环境和接口：

```powershell
& $PY scripts\run_all_variants_arrival_sweep.py `
  --algorithms all `
  --topologies us_backbone germany50 `
  --rates 0.5 1.5 2.5 `
  --limit-requests 100 `
  --continue-on-error
```

确认无误后运行完整数据集：

```powershell
& $PY scripts\run_all_variants_arrival_sweep.py `
  --algorithms all `
  --topologies us_backbone germany50 `
  --rates 0.5 1.5 2.5 `
  --continue-on-error
```

脚本按 `(topology, rate, algorithm)` 自动跳过已有完整结果。中途停止后再次执行即可续跑；`--force` 会重算指定配置，`--dry-run` 只打印任务，`--limit-requests N` 只使用每个数据集前 N 条。

## 结果目录

默认输出为：

```text
C:/Users/11353/Desktop/hrl_marl_reconfig_starter/artifacts/runs/all_variants_arrival_sweep/
```

每个任务单独保存：

```text
<topology>/per_node_rate_<rate>/<algorithm>/
  dataset/episodes.csv
  dataset/cost_summary.csv
  run.log                 # TA-HRL/A2C/PPO
```

TA-HRL 消融在算法目录下还会保留其真实变体子目录（如 `noIL`、
`single_dqn`、`gat`、`mlp`）；汇总中的 `output_dir` 指向实际完成目录。

根目录会生成：

```text
summary.csv
average_resource_curves.csv
<topology>_average_cpu.png
<topology>_average_mem.png
<topology>_average_bw.png
run_manifest.json
```

`summary.csv` 中的 `source` 区分当前入口结果；不会覆盖
`C:/Users/11353/Desktop/结果`。

## 口径

- `msft_hirl`、`msft_hrl`、`msft_ilrl`、`msft_hirl_gat`、`msft_hirl_mlp`
  均调用当前工作区的 `train_tahrl.py`，分别对应 `full`、`no_il`、
  `single_dqn`、`gat`、`mlp` 变体；不是旧的历史模型回放。
- `a2c`、`ppo` 调用现有 flat RL 基线入口；`msfc_ce` 调用当前迁移版求解器。
- `msfc_ce` 调用当前工作区的 Python 迁移版 `MSFCE_Solver`，按请求到达时间处理，并在 `leave_time` 释放 CPU、MEM、BW。
- 旧入口只接受历史 rate 标签，因此即使实际数据是半档 rate，命令行仍传兼容值 `--rate 8`；真正的数据由 `--data` 指定。
- 平均资源图优先使用瞬时 `*_util_pct`。旧 DDQN 没有瞬时 BW 列时，图中使用 `bw_req / 85` 的归一化替代值；原始累计消耗仍在 `summary.csv` 和 `episodes.csv` 中保留。
