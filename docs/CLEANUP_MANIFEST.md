# 2026-08-26 重构、清理与恢复清单

## 1. 范围与原则

本次重构在没有原始 Git 仓库可回退的前提下进行，因此先制作完整外部副本，
再迁移有用产物，最后删除历史输出、重复源码和零引用组件。以下清单区分：

- 已验证的完整备份；
- 已迁移到新目录的保留内容；
- 已从当前工作树删除、但仍在备份中的内容；
- 恢复操作。

当前项目根目录：

```text
C:\Users\11353\Desktop\hrl_marl_reconfig_starter
```

## 2. 已验证完整备份

主备份：

```text
C:\Users\11353\Desktop\hrl_marl_reconfig_starter_backup_20260826_1636_files
```

备份完成后的只读校验结果：

| 校验项 | 原目录 | 备份目录 | 结果 |
| --- | ---: | ---: | --- |
| 文件数 | 1,457,589 | 1,457,589 | 一致 |
| 文件总字节 | 10,104,615,315 | 10,104,615,315 | 一致 |
| Robocopy 镜像预检待复制 | - | 0 | 一致 |
| Robocopy 不匹配 | - | 0 | 一致 |
| Robocopy 失败 | - | 0 | 一致 |
| 关键源码/配置/数据/模型 SHA-256 文件数 | 912 | 912 | 0 个不匹配 |

关键文件校验清单的聚合 SHA-256：

```text
250BE5D0CCF85C0490E2E0088A0D46F3852CE0D34728D279A0E2472B69AB5091
```

该目录是本次重构的唯一主恢复源，不得随当前项目清理一起删除。

### 非主备份临时文件

以下是备份过程中的未完成归档或测速副本，不具备上述完整校验证据：

```text
C:\Users\11353\Desktop\hrl_marl_reconfig_starter_backup_20260826_160404.tar
C:\Users\11353\Desktop\_backup_tar_speed_test_20260826.tar
C:\Users\11353\Desktop\_backup_copy_speed_test_20260826
```

不要用它们替代主备份。确认主备份可读取并完成一次恢复演练后，可由用户人工
删除这些临时项。

## 3. 保留内容的迁移

### 3.1 参考产物

清理 `outputs/` 前，49 个当前仍有复现价值的文件（约 95.23 MiB）迁入：

- `artifacts/reference_runs/hrl/rate24_full_current`：保留的 full HRL 权重、训练
  日志和分析；
- `artifacts/reference_runs/deployment/`：BC、QMIX/WQMIX 参考权重与统一评估；
- `artifacts/reference_runs/migration/`：迁移学习基线权重与 smoke 报告；
- `artifacts/reference_runs/sla/sla_risk_predictor_v2`：运行时 SLA 风险模型、
  报告和保留样本；
- `artifacts/plans/hrl_seed7071_rate24`：前 100/500 条 HRL 可执行计划与资源账本；
- `artifacts/reports/`：HRL-WQMIX 候选审计和迁移检查报告。

迁移是有选择的归档，不代表每个结果都可用于论文结论。具体限制见
`artifacts/README.md`。

核心回归期间又从上述已校验主备份中选择性恢复了一份迁移接口所需的参考报告：

```text
artifacts/reference_runs/migration/runtime_rate8_seed7071_first100.json
```

该文件为 2,714,328 B，SHA-256 为
`D11067F6E4072EF04556B745E2A45F1AF86A3D20F0F46945F2588743CCCC558D`。
它来自历史 rate-8 seed-7071 前 100 请求，只用于复现 migration WQMIX 的在线
装载、监控和决策接口；未恢复其余旧 `outputs/` 内容，也不能把它作为最终实验。

### 3.2 数据与研究资料

- rate-24 HRL 数据移动到 `data/hrl/us_backbone_rate24`；
- 文献 PDF 移动到 `docs/references/pdfs`；
- 文献抽取文本移动到 `docs/references/extracted_text`；
- 论文阅读材料移动到 `docs/research/paper_readers`；
- 论文工作稿移动到 `docs/research/thesis_drafts`。

科研文献和论文工作文件未作为“历史输出”删除。

## 4. 已清理内容

### 4.1 `outputs/`

清理前：

- 1,454,219 个文件；
- 约 4.416 GiB；
- 大量逐请求 probe、训练中间文件和重复参数扫描输出。

清理后仅保留 `outputs/README.md` 作为旧脚本兼容目录。新的运行结果应写入
`artifacts/runs/`。

全部旧输出仍可从主备份的 `outputs/` 恢复。

### 4.2 `data/` 历史目录

已删除 43 个当前默认工作流不再依赖的数据目录，共 4,954,314,286 B（约
4.61 GiB）：

```text
actionable_reroute_v2_online_full
actionable_reroute_v2_per_source_rate1_duration400
actionable_reroute_v2_per_source_rate1_test
actionable_reroute_v2_smoke
actionable_reroute_v2_thesis_rate4
actionable_reroute_v2_thesis_rate8
deployment_executor_ab_planner_100
deployment_executor_ab_policy_100
deployment_executor_planner_validation2101_full
deployment_executor_planner_validation2102_full
deployment_executor_planner_validation2103_full
deployment_topk_v3_seed7071_50node_rate25_first100_util80
deployment_topk_v3_seed7071_cogentco197_non_dc60_pernode2_rate120_first100_util80
deployment_topk_v3_seed7071_cogentco197_non_dc60_pn005_total03_first100_util80
deployment_topk_v3_seed7071_cogentco197_non_dc60_pn008333_total05_first100_util80
deployment_topk_v3_seed7071_cogentco197_non_dc60_pn0125_total075_first100_util80
deployment_topk_v3_seed7071_cogentco197_non_dc60_pn016667_total1_first100_util80
deployment_topk_v3_seed7071_cogentco197_non_dc60_rate2_first100_util80
deployment_topk_v3_seed7071_cogentco197_non_dc60_rate60_first100_util80
deployment_topk_v3_seed7071_cogentco197_rate98_first100_util80
deployment_topk_v3_seed7071_rate12_100s_top16_mb20_util90
deployment_topk_v3_seed7071_rate16_100s_top16_mb20_util90
deployment_topk_v3_seed7071_rate20_100s_top16_mb20_util90
deployment_topk_v3_seed7071_rate24_100s_top16_mb20_util90
deployment_topk_v3_seed7071_rate24_first100
deployment_topk_v3_seed7071_rate24_first100_mb20
deployment_topk_v3_seed7071_rate24_first100_top16_mb20_util100
deployment_topk_v3_seed7071_rate24_first100_top16_mb20_util90
deployment_topk_v3_seed7071_rate24_first100_util80
deployment_topk_v3_seed7071_rate8_first100_top16_mb20_util80
deployment_topk_v3_seed7071_rate8_first100_top16_mb20_util90
deployment_topk_v3_seed7071_rate8_full_top16_mb20_util90
deployment_topk_v3_smoke_seed7071_rate24
deployment_v3_multiseed_pilot
deployment_v3_multiseed_rate24
deployment_v3_multiseed_rate24_sla80
deployment_v3_tree_bw_pilot
per_source_rate1_duration400
per_source_rate1_test
qmix_sft_v1
qmix_sft_v1_full
qmix_sft_v1_thesis_rate4
qmix_sft_v1_thesis_rate8
```

当前保留 17 个顶层数据目录，见 `data/CATALOG.md`。删除项仍完整存在于主备份
的 `data/` 下。

### 4.3 重复和零引用源码

已删除的旧入口/脚本：

```text
RALL.py
runAllBw.py
runAllRes.py
runIlandnoIL.py
runModels.py
train_tahrl_before_reference_merge_20260702.py
py_split_timeslot/11.py
scripts/train_migration_wqmix.py
scripts/eval_migration_wqmix.py
scripts/generate_qmix_datasets.py
scripts/check_qmix_datasets.py
models/low_policy_student.py
envs/modules/MABPruner.py
```

旧 `model/A2C`、`model/PPO`、`model/DQN`、`model/D3QN` 中的历史实现文件已
移除；当前基线统一由 `core/marl/migration_baselines.py`、
`core/marl/migration_baseline_learners.py` 和统一 train/eval 脚本负责。

以下零引用类已移除：

```text
ParetoSet
LoadAwarePlacementStrategy
SimplePlacementStrategy
CriticDiagnostics
ResourceAllocation
```

`LowLevelPolicy` 和 `SimpleDataLoader` 暂时保留，因为旧 checkpoint 反序列化或
兼容调用仍可能依赖它们。不能仅凭静态零引用就删除 checkpoint 需要的类。

### 4.4 HRL 重复树

原 `hrl_training/` 中 57 个文件与根实现相同，另有 11 个分叉文件。逐项审计后：

- 以根 `core/`、`envs/`、`trainer/`、`configs/`、`topo/` 为唯一源码；
- 采用功能较完整的训练入口作为新根 `train_tahrl.py`，并合并根环境已有的迁移、
  SFT snapshot、统一资源账本、决策计时和在线加速逻辑；
- `hrl_training/` 只保留 `train_tahrl.py` 兼容包装器与 README；
- 删除重复的 no-checkpoint 入口，改用根入口的 `--disable-checkpoint`。

### 4.5 根目录与临时工具收尾

在统一 CLI、工作流和新文档完成后，又从活跃根目录移除了以下内容。删除前已逐项
确认它们位于第 2 节的完整校验备份中。

IDE、缓存和空目录：

```text
__pycache__/
.idea/
.run/
.codex_spreadsheet_read/
logs/
visualization/
_qa_song_docx/
qa_2_4/
qa_2_4_formula_fix/
qa_2_4_rewrite/
```

其中 `.run/` 是旧 IDE run configuration，不是项目运行时入口；当前命令由
`python -m sfc_project` 和 `configs/workflows.yaml` 管理。

旧工具、模型空壳和 scratch：

```text
eval_tools/        # 7 个旧评估包装文件，20,991 B
model/             # 历史基线目录，备份中 7 个文件，202,681 B
models/            # 已删除学生模型后的目录空壳，备份中 1 个文件，2,329 B
tmp/               # 96 个文档处理/抽取临时文件，47,061,377 B
work/              # 4 个临时工作文件，332,507 B
```

当前评估入口统一位于 `scripts/` 和 `configs/workflows.yaml`；当前 MARL 实现位于
`core/marl/`。`docs/references/`、`docs/research/` 中的文献、阅读器和论文工作稿
没有随临时目录删除。

根目录临时结果：

```text
loss_log.csv
temp_profile.pkl
temp_rate24_100.pkl
temp_rate24_full.pkl
```

旧根文档：

```text
ACTIONABLE_REROUTE_V2.md
EXPERIMENT_RESULTS.md
HRL_SFC_MININET.md
LEGACY_HRL_PORT.md
MARL_RECONFIG_STARTER.md
TASK_PROGRESS.md
UNUSED_CLASSES.md
UNUSED_CLASSES_REVIEW.md
```

这些文档记录的是清理前的路径、阶段状态或静态审计快照，已由根 `README.md`、
`docs/ARCHITECTURE.md`、`docs/WORKFLOWS.md`、本清单、`data/CATALOG.md` 和
`artifacts/README.md` 取代。`docs/HANDOFF_20260823.md` 也因引用外部 legacy
目录、旧 `outputs/` 和已删除脚本而移除；需要追查历史时从完整备份读取，不要把
它重新作为当前操作手册。

最终活跃根目录只保留项目实现、配置、输入、文档、产物、兼容输出目录以及统一
依赖清单。所有上述删除项均是工作树清理，不是从外部备份中销毁。

### 4.6 最终清理快照

截至 2026-08-26，本轮收尾完成后的工作树状态为：

- 899 个文件，总计 `429670439` B；
- 15 个顶层目录和 5 个根文件；
- `__pycache__` 目录为 0，`.pyc/.pyo` 文件为 0；
- `artifacts/runs/hrl/smoke_one` 和根目录 `visualization/` 不存在；
- `artifacts/runs/hrl/smoke_verify`、`artifacts/runs/hrl/visualization` 和第 2 节的
  外部完整备份仍存在；
- 在 `PYTHONDONTWRITEBYTECODE=1` 下执行统一 `doctor`，12/12 个 preset 均为
  ready，且验收后缓存数量仍为 0。

## 5. 重构中的针对性修正

本节用于说明为何重构后的根代码不应被旧副本整体覆盖。

### `train_tahrl.py`

- 默认 ablation 从错误的 `gat` 改为 `full`；
- 保留 `--max_requests`、`--output_dir`、`--torch_threads`、`--quiet`、
  `--fast`、`--candidate_ablation`、`--no_il` 和 `--disable-checkpoint`；
- 新输出默认指向 `artifacts/runs/hrl`；
- `--fast` 只有与 `--no_il` 或匹配的 `--resume` 一起使用才合法；
- Phase 1 从配置读取 DC 和容量，不再写死；
- 数据大小优先使用 `env.all_requests`。

### 资源与生命周期

- 合法请求 ID `0` 不再被生命周期代码误判为空；
- 可达性特征会过滤非法目的节点，并在 hop cache 维度异常时重建；
- 禁止向 `FAILED/RELEASED` 请求提交带宽，失败立即回滚；
- 同 ID 的终态记录可被新的 `PENDING` 请求记录替换；
- leave 事件统一调用 `release_request_record`，不再调用不存在的方法。

### 模型配置

TreeTransformer 的 `num_heads` 从配置读取，使 fast profile 的 2-head 设置真正
生效。

### 有向 SFT 与重路由事务

- HRL 数据 PKL 与 runtime JSON 在推理前逐字段签名校验；
- SFT 成功必须满足有向 VNF 阶段顺序和最后 VNF 到全部目的节点可达；
- exporter 只从权威 `RequestRecord` 和有向带宽账本重建计划；
- 重路由在提交前后均验证 SFT 顺序，失败完整恢复带宽、树和兼容状态；
- 精确的 `destination-before-last-VNF`、陈旧、畸形和 post-validation 失败均有回归。

### 运行入口与并发收尾

- HRL 可视化输出迁到 `artifacts/runs/hrl/visualization`，不再创建根目录空壳；
- Windows Python loopback smoke 降到调度粒度可稳定支持的速率，生产原生发送器
  和 SLA 阈值未改变；
- 删除仅由测试引用且已被在线 monitor/planner 取代的 `MigrationTaskScheduler`；
- 保留其冲突波次算法为 `migration_execution_waves()` 并接入真实迁移执行器；
- `doctor --deep` 按 UTF-16LE 解码 WSL 状态，中文输出不再乱码。

最终状态、实测门禁、当前 checkpoint 限制和下一步见
`docs/HANDOFF_CURRENT.md`。

## 6. 安全恢复步骤

### 6.1 恢复整个项目到新目录

先停止当前项目的 Python、Ryu、Mininet 和相关 WSL 服务。然后设置路径：

```powershell
$BACKUP = 'C:\Users\11353\Desktop\hrl_marl_reconfig_starter_backup_20260826_1636_files'
$RESTORE = 'C:\Users\11353\Desktop\hrl_marl_reconfig_starter_restore_20260826'
```

确认目标不存在，避免与其他目录混合：

```powershell
Test-Path -LiteralPath $BACKUP
Test-Path -LiteralPath $RESTORE
```

当第一条为 `True`、第二条为 `False` 时，复制到新目录：

```powershell
robocopy $BACKUP $RESTORE /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /XJ
```

Robocopy 返回码 `0` 到 `7` 都可能表示完成且无复制失败；必须检查输出中的
`FAILED`。再做只读镜像预检：

```powershell
robocopy $BACKUP $RESTORE /MIR /L /R:0 /W:0 /XJ
```

预检应显示待复制 0、不匹配 0、失败 0。然后比较文件数和总字节：

```powershell
$a = Get-ChildItem -LiteralPath $BACKUP -File -Recurse
$b = Get-ChildItem -LiteralPath $RESTORE -File -Recurse
$a.Count
$b.Count
($a | Measure-Object Length -Sum).Sum
($b | Measure-Object Length -Sum).Sum
```

预期两边均为 1,457,589 个文件、10,104,615,315 B。恢复副本验证完成前，不要
覆盖当前工作目录。

### 6.2 只恢复一个删除项

例如恢复一个旧数据目录到单独的审计目录：

```powershell
$SRC = Join-Path $BACKUP 'data\qmix_sft_v1'
$DST = 'C:\Users\11353\Desktop\recovered_qmix_sft_v1'
Copy-Item -LiteralPath $SRC -Destination $DST -Recurse
```

不要直接复制回当前 `data/`，除非已经确认现有配置确实需要它，并同步更新
`data/CATALOG.md` 和 `configs/workflows.yaml`。

### 6.3 回退当前项目

只有在新目录恢复和验证均完成后，才进行人工切换。推荐保留当前重构目录并给它
改名，再把验证后的恢复目录改成原项目名。不要使用 `git reset --hard` 或在未经
确认的目标上执行递归删除。

## 7. 校验证据的含义

上述文件数、字节数和 SHA-256 证明“清理前内容有完整恢复源”。它不等于重构后
所有训练、Mininet、迁移和长跑实验已经通过。重构后的功能回归应单独记录命令、
退出码、生成产物和运行日期；参考结果也必须保留其原始 provenance。
