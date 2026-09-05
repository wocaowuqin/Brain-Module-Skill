# HRL-MARL SFC 编排与真实执行项目

本项目研究多播服务功能树（SFT/SFC）的在线映射、并发联合调度、真实
Ryu/Mininet 执行，以及部署后的 VNF 迁移。当前代码的主链路是：

```text
双层 HRL 生成完整映射候选
        -> Top-K 候选与不可变资源快照
        -> Weighted QMIX 排序 + 有界联合可行性解码
        -> 版本化原子资源提交
        -> Ryu/OVS + 常驻 VNF/UDP Agent 执行和 SLA 探测
```

这里的“完整候选”包含 VNF 放置、各 SFC 分段路径和多播树，不是单个物理
节点或单跳动作。每个微批中的请求才是 WQMIX 的临时智能体。

## 当前边界

- 根目录的 `core/`、`envs/`、`trainer/`、`configs/` 和 `topo/` 是 HRL
  唯一实现；`hrl_training/train_tahrl.py` 只是旧命令兼容包装器。
- `scripts/run_sdn_runtime_requests.py` 当前把在线 HRL、在线 WQMIX 和预生成
  计划定义为互斥模式。因此，现有 HRL -> WQMIX 闭环是先导出/构建候选，
  再由在线 WQMIX 读取候选并执行，而不是在同一请求上同时启动两个在线规划器。
- 纯仿真中的排队/丢包/时延是模型估计；只有完成 Ryu/Mininet UDP 探测的
  结果才能称为实测严格 SLA。
- 迁移数据目前是“真实运行计划 + 合成热点快照”。它适合实现检查和基线
  联调，不能单独支撑跨 seed 的在线迁移效果结论。
- `artifacts/reference_runs/` 保存的是保留的参考权重与报告。目录名中的
  `current` 表示本项目当前选定的参考版本，不表示它刚刚完成了重新训练，也不
  自动证明任何接受率或 SLA 指标。

## 快速开始

推荐解释器：

```powershell
$PY = 'C:\Users\11353\.conda\envs\sfc_ppo\python.exe'
Set-Location 'C:\Users\11353\Desktop\hrl_marl_reconfig_starter'
```

检查 Python 依赖、输入文件和工作流配置：

```powershell
& $PY -m sfc_project doctor
& $PY -m sfc_project list
```

先查看命令，不启动实验：

```powershell
& $PY -m sfc_project run simulation-smoke
```

确认后执行：

```powershell
& $PY -m sfc_project run simulation-smoke --execute
```

`run` 默认永远是 dry run。Mininet 工作流会改变 WSL 中的 Ryu/OVS/Mininet
运行状态，还必须显式增加 `--confirm-system-changes`：

```powershell
& $PY -m sfc_project run mininet-testbed-smoke `
  --execute --confirm-system-changes
```

浅层 `doctor` 显示 Mininet preset 为 `ready`，只代表当前 Windows 端的文件、
模块和 `wsl.exe` 前置条件通过；必须运行 `mininet-testbed-smoke` 才能验证 WSL
内部的 Ryu、OVS 和 Mininet。

向 preset 追加底层脚本参数时，在 `--` 后传入：

```powershell
& $PY -m sfc_project run simulation-smoke --execute -- --max-requests 10
```

## 推荐入口

| 目标 | 入口 |
| --- | --- |
| 查看所有可运行流程 | `python -m sfc_project list` |
| 环境检查 | `python -m sfc_project doctor` |
| HRL 小规模集成检查 | `python -m sfc_project run hrl-smoke --execute` |
| 续训保留的 rate-24 HRL 权重 | `python -m sfc_project run hrl-rate24-resume --execute` |
| 部署 WQMIX 与基线评估 | `python -m sfc_project run deployment-eval --execute` |
| 迁移算法实现检查 | `python -m sfc_project run migration-check --execute` |
| 中央大脑与多智能体编排检查 | `python -m sfc_project run multiagent-orchestration-check --execute` |
| 生命周期感知纯仿真 | `python -m sfc_project run simulation-smoke --execute` |
| Ryu/Mininet 连通性检查 | `python -m sfc_project run mininet-testbed-smoke` |
| US backbone 拓扑/端口检查 | `python -m sfc_project run mininet-us-validate` |
| Cogentco 拓扑构造检查 | `python -m sfc_project run mininet-cogentco-validate` |
| SFT group REST 检查 | `python -m sfc_project run mininet-sft-group` |
| 多跳 SFT 与分支重路由 | `python -m sfc_project run mininet-sft-multihop` |
| Cogentco 多播数据面检查 | `python -m sfc_project run mininet-cogentco-data-plane` |
| 单条保留 HRL 计划实测 | `python -m sfc_project run mininet-sfc-smoke` |

完整参数和输出位置以 `configs/workflows.yaml` 为准。训练、评估和执行的详细
顺序见 [docs/WORKFLOWS.md](docs/WORKFLOWS.md)。

## 目录

| 目录 | 责任 |
| --- | --- |
| `sfc_project/` | 安全的统一命令行入口 |
| `configs/` | HRL 配置和工作流 preset |
| `core/hrl/`、`core/gnn/` | 双层 HRL 策略与图编码 |
| `core/marl/` | Top-K、资源账本、联合解码、WQMIX、迁移和 SLA 风险模型 |
| `envs/` | HRL 环境、生命周期和资源管理 |
| `trainer/` | 模仿学习与强化学习训练流程 |
| `scripts/` | 数据生成、训练、评估、检查和实验入口 |
| `sdn/` | Ryu 控制器、在线规划器、Mininet 拓扑、VNF/UDP Agent |
| `data/` | 保留的可复现实验输入；见 [data/CATALOG.md](data/CATALOG.md) |
| `artifacts/` | 参考权重、计划、报告和新实验输出 |
| `outputs/` | 仅为旧脚本保留的临时兼容目录 |
| `hrl_training/` | 旧 HRL 入口兼容层，不包含第二份实现 |

架构和代码所有权见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
当前代码状态、验证证据、限制和下一步见
[docs/HANDOFF_CURRENT.md](docs/HANDOFF_CURRENT.md)。

## 实验结果规则

报告一个结果前至少记录：请求文件及 SHA-256、seed、拓扑 profile、到达率与
持续时间、生命周期版本、候选来源、checkpoint、资源容量、带宽利用率上限、
微批宽度、解码器预算、`time-scale`、VNF/probe backend 和完整命令。

下列指标不可混写：

- 映射/规划成功率：算法能否生成完整计划。
- 资源接受率：计划能否通过 CPU、内存和有向带宽账本并完成提交。
- 执行成功率：Ryu、VNF 和接收端是否成功启动并完成请求。
- 条件严格 SLA：在已启动流量中，所有目的端是否同时满足阈值。
- 端到端严格 SLA：以全部到达请求为分母，未规划、未部署、未启动或探针不全
  均计为失败。

## 备份与恢复

重构前的完整校验备份位于：

```text
C:\Users\11353\Desktop\hrl_marl_reconfig_starter_backup_20260826_1636_files
```

不要直接把备份镜像覆盖回当前目录。先恢复到新目录、核对后再人工切换。校验
数据、删除清单和逐步恢复命令见
[docs/CLEANUP_MANIFEST.md](docs/CLEANUP_MANIFEST.md)。
