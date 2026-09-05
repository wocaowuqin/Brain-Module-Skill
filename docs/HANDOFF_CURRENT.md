# 当前项目交接文档

更新时间：2026-08-26

## 1. 当前结论

项目已完成目录、入口和核心状态语义重构。当前唯一主链为：

```text
双层 HRL 生成单请求完整映射
  -> Top-K 完整候选
  -> WQMIX 排序与联合可行性解码
  -> 版本化原子账本提交
  -> Ryu/Mininet 执行与严格 SLA 探测
  -> 可选的预测式 VNF 迁移
```

当前代码可用于继续训练、离线评估和分阶段接入 Mininet，但保留的 HRL
checkpoint 不是最终效果模型。它在同源 seed-7071 rate-24 请求 1 上生成了违反
有向 VNF 顺序的树，现已被权威校验器正确拒绝并完整回滚。因此下一项算法工作是
重新训练或约束微调 HRL，而不是放宽成功判定。

本轮未启动 Ryu、OVS 或 Mininet，也没有产生新的真实严格 SLA 结论。

## 2. 备份与恢复

清理前完整校验备份：

```text
C:\Users\11353\Desktop\hrl_marl_reconfig_starter_backup_20260826_1636_files
```

- 文件数：1,457,589
- 总字节：10,104,615,315
- 关键文件 SHA-256：912 个文件，0 个不匹配
- 聚合 SHA-256：
  `250BE5D0CCF85C0490E2E0088A0D46F3852CE0D34728D279A0E2472B69AB5091`

不要把备份直接覆盖到当前工作树。恢复步骤和删除清单见
`docs/CLEANUP_MANIFEST.md`。

## 3. 唯一入口与代码所有权

推荐解释器：

```powershell
$PY = 'C:\Users\11353\.conda\envs\sfc_ppo\python.exe'
Set-Location 'C:\Users\11353\Desktop\hrl_marl_reconfig_starter'
& $PY -m sfc_project doctor --deep
& $PY -m sfc_project list
```

主要目录：

| 路径 | 用途 |
| --- | --- |
| `train_tahrl.py` | 唯一 HRL 训练入口 |
| `core/hrl/`、`core/gnn/` | 双层策略、图编码和经验回放 |
| `envs/` | HRL 环境、SFT 状态、生命周期与资源账本 |
| `core/marl/` | Top-K、WQMIX、联合解码、SLA 风险和迁移学习 |
| `sdn/` | 在线规划器、Ryu、VNF/UDP Agent 和 Mininet 组件 |
| `scripts/` | 数据、训练、评估、检查和真实实验编排 |
| `configs/workflows.yaml` | 统一 CLI preset |
| `data/` | 保留输入，详见 `data/CATALOG.md` |
| `artifacts/reference_runs/` | 保留 checkpoint 和参考报告 |
| `artifacts/runs/` | 新运行输出 |
| `outputs/` | 仅保留旧脚本兼容 README，不再存正式结果 |
| `hrl_training/` | 旧训练命令兼容包装器，不是第二套实现 |

不得重新复制一套 `core/envs/trainer/configs` 到 `hrl_training/`。

## 4. 已修复的关键正确性问题

### HRL 与 SFT

- HRL PKL 与 runtime JSON 在推理前逐字段签名校验，错配立即失败。
- 成功必须满足有向
  `source -> VNF0 -> ... -> last VNF -> all destinations`。
- Coordinator 和 exporter 只使用已验证的 `RequestRecord` 与带宽账本。
- 顺序错误、快照失败或中途带宽提交失败会回滚 CPU、MEM、BW、生命周期和树。

### 重路由与迁移

- 重路由候选在提交前检查有向 VNF 顺序；目的节点不能绕过最后一个 VNF。
- 修改后再次调用权威 SFT validator；失败完整恢复带宽、树、计数和 legacy view。
- 畸形、过期和不安全 proposal 返回明确状态码，不使协调器崩溃。
- 在线迁移删除了已被新 monitor/planner 取代的旧调度类。
- 真实执行按冲突波次并发；同请求或同源 DC 的迁移不会进入同一并发波次。

### 执行与维护

- Python loopback 自检使用适合 Windows 调度粒度的速率；生产负载仍使用原生
  绝对时间发送器。
- HRL 可视化不再污染根目录，默认写入
  `artifacts/runs/hrl/visualization`。
- `doctor --deep` 可正确解码中文 Windows 的 WSL UTF-16LE 状态输出。

## 5. 2026-08-26 验证结果

### 核心门禁

- `doctor --deep`：通过，12/12 preset ready。
- 全部活动 Python 文件编译：通过。
- SFT snapshot、SFT 原子提交、有向边账本：通过。
- 重路由、精确 `destination-before-last-VNF` 反例、post-validation 回滚：通过。
- Role-MARL 实际提交路径：通过。
- Joint decoder、版本化 transaction、deployment env：通过。
- WQMIX/QMIX 网络训练与推理 smoke：通过。
- SLA predictor、校准、migration benefit gate：通过。
- 迁移 WQMIX 在线 planner、monitor 和冲突波次：通过。
- Python/原生 VNF Agent、probe agent 和迁移协议：在 WSL loopback 通过。
- 容器规划、失败回滚和本机三段 UDP forwarder：通过；未启动 Docker。

### 可复现数值及边界

1. 部署 WQMIX 保留测试集：100 请求接受 93，账本违规 0，决策 P95
   约 1.60 ms。这里的 SLA 是候选模型判定，不是 Mininet 实测。
2. 同一数据上 Oracle、Top-1、random、greedy 和 WQMIX 都接受 93；该 100 条
   数据没有证明 WQMIX 优于基线。数据中也没有 `legacy_hrl` 来源候选。
3. SLA 风险模型 held-out 样本仅 69 条，ROC-AUC 约 0.761，Brier 约 0.130。
   它可作接口和初步重排检查，正式结论需要独立多 seed 真实数据。
4. 迁移 WQMIX smoke 为 12 个记录，在线一次处理 3 个任务并选中 1 个，决策
   约 10.9 ms。这不是迁移收益或 SLA 改善结论。
5. WSL 原生迁移协议 loopback 为 100/100 包、0 丢包、状态和序号连续，服务
   中断约 0.91 ms。它不是 Mininet 数据面或大拓扑结果。
6. 当前 HRL checkpoint 的同源请求 1：accepted=0；非法 SFT 被正确拒绝，资源
   账本无泄漏。不能再引用清理前的虚假成功数。

## 6. 常用回归命令

```powershell
& $PY scripts/check_sft_snapshot.py --json
& $PY scripts/check_sft_atomic_commit.py --json
& $PY scripts/check_reconfiguration_manager.py --json
& $PY scripts/check_role_marl_pipeline.py --json
& $PY scripts/check_joint_candidate_decoder.py
& $PY scripts/check_joint_candidate_transaction.py
& $PY scripts/check_sdn_runtime_requests.py
& $PY -m sfc_project run deployment-eval --execute
& $PY -m sfc_project run migration-check --execute
```

HRL 计划导出必须使用同源的两个请求文件：

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
  --episodes 1 --seed 7071
```

## 7. 下一步实施顺序

1. 在 HRL 训练环境中把“下一阶段只能位于当前阶段有向下游”落实到 action mask
   和失败奖励，重新训练，不修改 validator 迁就旧 checkpoint。
2. 用至少 5 个训练 seed、2 个验证 seed、3 个测试 seed 评估完整映射率、规划
   时间和资源泄漏；每次导出均做 trace 签名与 ledger gate。
3. 用重新训练的 HRL 在线生成真实 Top-K，确保候选中明确记录
   `source=legacy_hrl`，再重训 BC/WQMIX。
4. 构造真正有批内资源冲突的高负载测试集；当前 100 条数据无法区分 WQMIX 与
   简单基线。
5. 按 1 条、100 条、多 seed 完整 trace 的顺序进入 Mininet。分别报告规划成功、
   资源接受、执行成功、条件严格 SLA 和端到端严格 SLA。
6. 最后才启用在线迁移，与 no-migration/reactive/predictive/WQMIX 做相同 trace
   的配对实验。

任何阶段若出现 trace 错配、账本违规、探针不全或来源不明，应停止扩大实验规模。

## 8. 禁止混写的结论

- 纯仿真 SLA 不等于 Mininet 实测 SLA。
- WSL loopback 协议时延不等于拓扑内迁移中断时间。
- 计划导出记录数不等于合法映射成功数。
- 已接受请求中的 SLA 比例不等于全部到达请求的端到端 SLA 成功率。
- smoke checkpoint 可加载不等于算法已经优于启发式或 Oracle。
