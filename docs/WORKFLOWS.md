# 工作流手册

## 1. 统一命令行

以下命令均从项目根目录执行：

```powershell
$PY = 'C:\Users\11353\.conda\envs\sfc_ppo\python.exe'
Set-Location 'C:\Users\11353\Desktop\hrl_marl_reconfig_starter'
```

查看运行条件与 preset：

```powershell
& $PY -m sfc_project doctor
& $PY -m sfc_project doctor --deep
& $PY -m sfc_project list
& $PY -m sfc_project list --json
```

`doctor` 检查 Python 版本、模块、输入文件和平台约束；`--deep` 在 Windows 上
额外查询 WSL 状态。它不代替 Ryu/OVS/Mininet 冒烟实验。

查看某个 preset 的展开命令：

```powershell
& $PY -m sfc_project run deployment-eval
```

不带 `--execute` 时不会启动子进程。执行计算型流程：

```powershell
& $PY -m sfc_project run deployment-eval --execute
```

底层参数放在单独的 `--` 后：

```powershell
& $PY -m sfc_project run simulation-smoke --execute -- --max-requests 10
```

Mininet preset 属于 `system` 风险，需要二次确认：

```powershell
& $PY -m sfc_project run mininet-sfc-smoke `
  --execute --confirm-system-changes
```

## 2. 当前 preset

| Preset | 类型 | 实际用途 | 默认输出 |
| --- | --- | --- | --- |
| `hrl-smoke` | HRL | 20 请求、CPU、无 IL、无 checkpoint 的 Phase 3 集成检查 | `artifacts/runs/hrl/smoke` |
| `hrl-rate24-resume` | HRL | 从保留的 full checkpoint 续训完整 rate-24 trace | `artifacts/runs/hrl/rate24_resume` |
| `deployment-eval` | 部署 | 在 seed 7301 v4 测试批次比较 WQMIX 与统一基线 | `artifacts/runs/deployment/eval_seed7301` |
| `migration-check` | 迁移 | 检查 heuristic、BC、IDQN、QMIX、WQMIX、MAPPO、MILP 路径 | `artifacts/runs/migration/baseline_check.json` |
| `simulation-smoke` | 仿真 | 前 100 条、生命周期释放、hard SLA admission 的纯仿真 | `artifacts/runs/simulation/seed7301_first100` |
| `mininet-testbed-smoke` | 系统 | 启动/验证 WSL Ryu、OVS、Mininet，运行 ping/iperf | 工具日志 |
| `mininet-us-validate` | 系统 | 验证 US-backbone switch 和固定端口 profile | 工具日志 |
| `mininet-cogentco-validate` | 系统 | 无 host 构造完整 Cogentco switch/端口拓扑 | 工具日志 |
| `mininet-sft-group` | 系统 | 通过 Ryu REST 检查 SFT group 安装与替换 | 工具日志 |
| `mininet-sft-multihop` | 系统 | 多跳多播 SFT 和在线分支重路由检查 | 工具日志 |
| `mininet-cogentco-data-plane` | 系统 | 完整 Cogentco 拓扑的组播切换与包传递 | `artifacts/runs/mininet/cogentco_data_plane/result.json` |
| `mininet-sfc-smoke` | 系统 | 用保留 HRL 计划真实部署并测量 1 条请求 | `artifacts/runs/mininet/sfc_smoke/result.json` |

命令参数以 `configs/workflows.yaml` 为唯一准则。表格说明不替代配置。

## 3. HRL 训练与候选导出

### 3.1 快速集成检查

```powershell
& $PY -m sfc_project run hrl-smoke --execute
```

这个 preset 的目标是检查环境、agent、资源账本和训练循环能否连通。它只有 20
条请求，且使用 `--no_il --disable-checkpoint`，不能用其接受率评价算法性能。

### 3.2 续训保留 checkpoint

先 dry run 核对 GPU、数据和输出目录：

```powershell
& $PY -m sfc_project run hrl-rate24-resume
& $PY -m sfc_project run hrl-rate24-resume --execute
```

输入 checkpoint 位于
`artifacts/reference_runs/hrl/rate24_full_current/final_model.pth`，输出写入新的
`artifacts/runs/hrl/rate24_resume`，不会覆盖参考权重。训练结果只有在完成指定
episode、保存 checkpoint 并做独立评估后才能称为新模型。

### 3.3 导出可执行 HRL 计划

先在少量请求上导出，并同时保存环境资源账本：

```powershell
& $PY scripts/export_hrl_sfc_plans.py `
  --legacy-root . `
  --checkpoint artifacts/reference_runs/hrl/rate24_full_current/final_model.pth `
  --data data/sdn_runtime_requests/seed_7071_rate24_duration100_lifetime50node/requests.pkl `
  --runtime-requests data/sdn_runtime_requests/seed_7071_rate24_duration100_lifetime50node/requests.jsonl `
  --profile sdn/topologies/us_backbone_28_bw90.json `
  --output artifacts/runs/hrl/export_seed7071/plans.jsonl `
  --summary artifacts/runs/hrl/export_seed7071/summary.json `
  --resource-csv artifacts/runs/hrl/export_seed7071/resource_ledger.csv `
  --episodes 100 `
  --seed 7071
```

`--data` 和 `--runtime-requests` 必须来自同一个 trace。训练用的
`data/hrl/us_backbone_rate24/phase3_requests.pkl` 与这里的 seed-7071 runtime
JSON 内容不同，不能混用；导出器会在推理前逐字段校验请求签名并拒绝错配。

导出后至少运行：

```powershell
& $PY scripts/check_hrl_sfc_resource_ledger.py `
  --plans artifacts/runs/hrl/export_seed7071/plans.jsonl `
  --resource-csv artifacts/runs/hrl/export_seed7071/resource_ledger.csv
```

不要把导出记录数当作映射成功数；以 summary 中完整、可验证计划数和 ledger
检查为准。

## 4. HRL 候选到 WQMIX

### 4.1 构建 Top-K

将 HRL 导出计划作为候选来源之一：

```powershell
& $PY scripts/generate_deployment_topk_v3.py `
  --requests data/sdn_runtime_requests/seed_7071_rate24_duration100_lifetime50node/requests.jsonl `
  --profile sdn/topologies/us_backbone_28_bw90.json `
  --baseline-plans artifacts/runs/hrl/export_seed7071/plans.jsonl `
  --output artifacts/runs/deployment/topk_seed7071 `
  --max-requests 100 `
  --top-k 8 `
  --max-agents 32 `
  --microbatch-ms 5 `
  --bandwidth-utilization-limit 0.8
```

Top-K 生成器会补充其他完整候选并做多样性筛选。只有 `source=legacy_hrl` 的
记录可以称为 HRL 候选；beam/greedy 候选不能改名成 HRL。

### 4.2 生成联合 Oracle 标签

```powershell
& $PY scripts/generate_deployment_oracle_labels.py `
  --data artifacts/runs/deployment/topk_seed7071 `
  --output artifacts/runs/deployment/topk_seed7071_labeled `
  --time-limit 2
```

求解状态、超时率、接受数和候选覆盖率必须随数据一起保存。单个 seed 的前 100
条只适合链路检查，不能作为最终训练/测试划分。

### 4.3 BC 预训练

主数据 `deployment_v4_tree_bw_rate24_sla80` 已按 seed 分离。一个最小训练命令
示例：

```powershell
& $PY scripts/train_deployment_bc.py `
  --train data/deployment_v4_tree_bw_rate24_sla80/train/seed_7101/mb5ms `
  --train data/deployment_v4_tree_bw_rate24_sla80/train/seed_7102/mb5ms `
  --validation data/deployment_v4_tree_bw_rate24_sla80/validation/seed_7201/mb5ms `
  --output artifacts/runs/deployment/bc_trial `
  --device cpu
```

正式训练应使用全部训练 seed，并用全部 validation seed 选 checkpoint。不要对
正式实验增加 `--allow-source-overlap`。

### 4.4 QMIX/WQMIX 微调

WQMIX 的训练环境会重放请求到达与生命周期，并使用与在线相同的联合解码和原子
提交：

```powershell
& $PY scripts/train_deployment_wqmix.py `
  --algorithm wqmix `
  --bc-checkpoint artifacts/reference_runs/deployment/deployment_bc_v4_tree_bw_rate24_sla80/bc_pretrained.pt `
  --train-trace data/deployment_v4_tree_bw_rate24_sla80/traces/seed_7101/requests.jsonl `
  --train-trace data/deployment_v4_tree_bw_rate24_sla80/traces/seed_7102/requests.jsonl `
  --output artifacts/runs/deployment/wqmix_trial `
  --decoder-top-r 4 `
  --decoder-time-budget-ms 2 `
  --device cpu
```

用 `--algorithm qmix` 运行结构一致的 QMIX 对照。只改变算法参数，保持 trace、
候选、mask、decoder、ledger、seed 和训练预算一致。

### 4.5 统一评估

```powershell
& $PY -m sfc_project run deployment-eval --execute
```

评估器共用一个 decoder 和 ledger，包含 reject-all、原始序列化动作、独立 Top-1、
random feasible、objective greedy、legacy-HRL-only、联合 greedy、预生成 MILP
Oracle，以及兼容的 BC/QMIX/WQMIX checkpoint。只有确实包含 `legacy_hrl` 候选的
样本才能计入 `legacy-HRL-only` 解释。

联合解码改动后先执行：

```powershell
& $PY scripts/check_joint_candidate_decoder.py
& $PY scripts/check_joint_candidate_transaction.py
```

## 5. 纯仿真筛选

```powershell
& $PY -m sfc_project run simulation-smoke --execute
```

默认执行 v4 seed 7301 前 100 条，Top-8，使用生命周期资源返还和 hard SLA
admission。输出可用于检查：

- 完整候选生成率；
- CPU/MEM/有向 BW 接受率；
- 资源释放是否发生；
- 模型化 delay/loss/SLA；
- 每时间槽资源和成本。

这些结果没有启动 Ryu/Mininet，不能写成“实测部署时延”“真实丢包”或“真实严格
SLA”。参数比较必须固定请求 trace、profile、队列模型和 admission mode。

## 6. Ryu/Mininet 实测

### 6.1 测试床

```powershell
& $PY -m sfc_project run mininet-testbed-smoke
& $PY -m sfc_project run mininet-testbed-smoke `
  --execute --confirm-system-changes
```

该流程会操作 WSL 的 Ryu/OVS/Mininet 服务。首次运行先确认 Ubuntu-22.04、sudo、
OVS、Mininet、Ryu 和编译工具均可用。

需要把故障定位到更小边界时，按顺序运行定向 preset。先 dry run，再带系统确认
执行：

```powershell
& $PY -m sfc_project run mininet-us-validate
& $PY -m sfc_project run mininet-cogentco-validate
& $PY -m sfc_project run mininet-sft-group
& $PY -m sfc_project run mininet-sft-multihop
& $PY -m sfc_project run mininet-cogentco-data-plane
```

- `mininet-us-validate` 检查保留 US-backbone profile 的交换机和端口映射；
- `mininet-cogentco-validate` 在不创建 host 的情况下检查 197 节点拓扑构造；
- `mininet-sft-group` 隔离验证 Ryu REST group 安装/替换；
- `mininet-sft-multihop` 验证多跳分支和 live reroute；
- `mininet-cogentco-data-plane` 才在完整 Cogentco 拓扑验证组播 cutover 和包交付。

这些流程仍属于测试床检查，不会自动产生 HRL/WQMIX 算法效果结论。

### 6.2 单请求真实 SFC

```powershell
& $PY -m sfc_project run mininet-sfc-smoke
& $PY -m sfc_project run mininet-sfc-smoke `
  --execute --confirm-system-changes
```

默认使用保留的 HRL `plans_first100.jsonl` 中第一条计划、常驻 Python VNF/probe
Agent 和 `time-scale=1`。它验证执行链，不包含 WQMIX 选择。

### 6.3 在线 WQMIX 执行

当前未把完整在线 WQMIX 长跑做成安全 preset。执行前先用 `--dry-run`，并确保
请求 trace 与候选数据 seed 一致：

```powershell
& $PY scripts/run_sdn_runtime_requests.py `
  --requests data/deployment_v4_tree_bw_rate24_sla80/traces/seed_7301/requests.jsonl `
  --profile sdn/topologies/us_backbone_28_bw90.json `
  --online-wqmix-checkpoint artifacts/reference_runs/deployment/deployment_wqmix_v4_tree_bw_rate24_sla80_ep5/wqmix_final.pt `
  --online-wqmix-data data/deployment_v4_tree_bw_rate24_sla80/test/seed_7301/mb5ms `
  --online-wqmix-microbatch-ms 5 `
  --online-wqmix-decoder-top-r 4 `
  --online-wqmix-decoder-time-budget-ms 2 `
  --output artifacts/runs/mininet/wqmix_seed7301/result.json `
  --max-requests 1 `
  --time-scale 1 `
  --dry-run
```

删除 `--dry-run` 才会启动真实执行。每次先跑 1 条，再跑 100 条，最后才跑完整
trace。禁止同时传 `--online-hrl-checkpoint`、`--sfc-plans` 或
`--sfc-candidate-plans`。

长跑至少记录：planning time、queue wait、setup latency、sender started、完整
receiver 数、offered-load compliance、每目的端 delay/jitter/loss、严格请求 SLA、
资源拒绝、过期拒绝、解码超时和清理积压。

## 7. SLA 风险预测器

当前保留模型：

```text
artifacts/reference_runs/sla/sla_risk_predictor_v2/runtime_sla_risk_predictor.pt
```

先做实现与 checkpoint 检查：

```powershell
& $PY scripts/check_runtime_sla_risk_predictor.py `
  --checkpoint artifacts/reference_runs/sla/sla_risk_predictor_v2/runtime_sla_risk_predictor.pt
```

在线只做保守重排的典型参数为：

```text
--online-wqmix-sla-predictor <checkpoint>
--online-wqmix-sla-calibration-rank-weight 0.25
--online-wqmix-sla-min-rerank-delta 0.05
--online-wqmix-sla-max-ood-score 4.0
```

`--online-wqmix-max-sla-failure-probability` 是可选硬拒绝门限，默认关闭，因为它会
提高条件 SLA 但降低接受率。训练、校准和测试必须来自不同运行 seed；不能用
同一结果既训练又报告收益。

## 8. 迁移

### 8.1 实现检查

```powershell
& $PY -m sfc_project run migration-check --execute
```

此流程验证 heuristic、BC、IDQN、QMIX、WQMIX、MAPPO 和小批 MILP 能走统一
mask/decoder。保留 smoke 数据来自真实部署计划加合成热点，仅说明代码可运行。

### 8.2 训练基线

```powershell
& $PY scripts/train_migration_baselines.py `
  --train-data data/migration_wqmix_v1_seed7071_train `
  --validation-data data/migration_wqmix_v1_seed7071_val `
  --algorithm all `
  --allow-source-overlap `
  --output artifacts/runs/migration/baselines_seed7071 `
  --device cpu
```

这里的 `--allow-source-overlap` 只因为保留数据的 nominal split 共用一个历史
runtime source。带该参数的结果必须标记为 non-reportable。正式实验需要独立
runtime seed，并移除该参数。

### 8.3 在线迁移约束

真实迁移使用 `run_sdn_runtime_requests.py` 的在线 WQMIX 部署模式，并额外提供：

- `--online-migration-wqmix-checkpoint`；
- `--online-wqmix-auto-migration` 或预测扫描参数；
- `--online-migration-max-agents` 和 `--online-migration-max-inflight`；
- 迁移 drain/idle 时间与预绑定 VNF endpoint 池。

执行器采用 prepare -> 启动/就绪新 VNF -> commit -> drain 旧路径的
make-before-break 协议，失败走 abort。正式比较至少包含 no-migration、reactive、
predictive/delay-aware greedy、BC、IDQN、QMIX、WQMIX、MAPPO 和小批 MILP，且
使用相同触发任务、候选、资源快照和执行预算。

## 9. 实验推进顺序

1. `doctor` 与所有纯 Python check。
2. 20 请求 HRL smoke，确认账本和生命周期。
3. 小样本候选 -> Oracle -> BC -> WQMIX 闭环。
4. seed 分离的 deployment 离线评估。
5. 纯仿真 100 条，再扩展完整 trace。
6. Mininet testbed smoke 和单请求 SFC。
7. 相同 seed 的 100 条真实 A/B；检查启动与探针完整性。
8. 多 seed 完整真实部署实验。
9. 独立 seed 的 SLA predictor 训练/校准/测试。
10. 最后才启用在线迁移，并与 no-migration 做配对实验。

任何一步出现账本违规、探针缺失、trace/checkpoint 不匹配或输出来源不明，都应
停止扩大实验规模，先修复可复现性。
