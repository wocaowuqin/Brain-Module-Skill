# HRL-MARL 项目重构计划

## 现状判断

当前项目不是“功能太多”本身造成混乱，而是四类东西共存于同一棵目录树：

1. 活跃实现：`core/`、`envs/`、`trainer/`、`sdn/`、`bms/`。
2. 实验入口：`scripts/` 中约 119 个脚本，其中检查、训练、数据生成、真实执行混在一起。
3. 历史实现：`legacy_hrl_runtime/` 与根实现存在大量同名文件和重复代码。
4. 数据与产物：`data/`、`output/`、`outputs/`、`artifacts/`、`tmp_msft/` 和缓存目录同时存在。

另外，当前目录没有 Git 仓库；`sfc_project snapshot` 因此只能记录 `git_commit=unavailable`。在没有版本控制之前，不应该做批量移动、删除或重命名。

当前基线验证结果：`compileall` 通过，`python -m sfc_project doctor --json` 的 13 个 preset 均为 ready；`scripts/check_delivery_bundle.py` 在迁移 WQMIX 阶段失败，原因是现有 checkpoint 缺少当前 `BatchCandidateQNetwork` 的 `no_migration_head.*` 参数。这个模型/权重版本问题应在 Phase 0 单独记录并修复，不能在目录迁移时顺手掩盖。

## 目标结构

第一阶段不改 import 路径，先把职责整理清楚：

```text
hrl_marl_reconfig_starter/
├─ core/                         # 算法内核：HRL、GNN、MARL、奖励、领域对象
├─ envs/                         # Gym 环境和环境内状态管理
├─ trainer/                      # 训练循环；不放实验编排
├─ sdn/                          # Ryu/Mininet/VNF/UDP 执行适配层
├─ bms/                          # 编排协议和安全边界
├─ sfc_project/                  # 唯一用户入口：doctor/list/run/snapshot
├─ configs/
│  ├─ base.yaml
│  ├─ models/
│  ├─ envs/
│  └─ workflows.yaml
├─ scripts/
│  ├─ train/                     # train_*.py
│  ├─ evaluate/                  # eval_*.py、compare_*.py、summarize_*.py
│  ├─ data/                      # generate_*.py、prepare_*.py、build_*.py
│  ├─ checks/                    # check_*.py、test_*_topology.py
│  └─ runtime/                   # run_*、vnf_*、mininet 相关入口
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ e2e/
├─ data/
│  ├─ raw/                       # 原始输入，只读
│  ├─ processed/                 # 可复用处理结果
│  └─ fixtures/                  # 小型测试夹具
├─ artifacts/
│  ├─ reference_runs/            # 长期保留的权重和报告
│  └─ runs/                      # 可删除的实验结果
├─ legacy/                       # 只读兼容代码，最终删除
└─ docs/
```

长期目标可以再迁移到 `src/hrl_marl/` 包，但不要把“改包名”和“理清职责”放在同一个大改动里。

## 分阶段执行

### Phase 0：冻结和建立回滚点

- 在项目根目录初始化 Git，提交当前状态；大文件用 Git LFS 或继续放在外部备份。
- 保留现有主备份，不把 `artifacts/backups/` 当作源码。
- 运行并记录：`python -m compileall core envs trainer sdn scripts bms sfc_project`、现有 smoke 检查、`python -m sfc_project doctor`。
- 写一份 `docs/BASELINE.md`，记录 Python、PyTorch、拓扑、checkpoint 和可运行命令。

### Phase 1：物理隔离历史和产物（低风险）

- 将 `legacy_hrl_runtime/` 改为只读兼容区，例如 `legacy/legacy_hrl_runtime/`；只保留仍被外部调用的包装入口。
- 将 `artifacts/package_verify_*`、`artifacts/runtime_package_repair_*`、`artifacts/backups/*` 标记为归档，不再从源码导入。
- 删除可再生的 `__pycache__/`、`.pytest_cache/`、`.idea/`、`.venv/`、`tmp_msft/` 和 `.tmp_*.py`；删除前先确认它们不在发布包清单中。
- 将 `output/` 和旧 `outputs/` 统一为 `artifacts/runs/`，旧脚本只通过兼容路径读取。

这一阶段只移动文件和更新文档，不改变算法代码。

### Phase 2：收敛入口

- 保留 `python -m sfc_project` 作为唯一公开入口。
- `scripts/` 内脚本按 `train/evaluate/data/checks/runtime` 分组；每个脚本只负责解析参数和调用函数，业务逻辑移回 `core/`、`envs/`、`sdn/`。
- 先拆 `scripts/run_sdn_runtime_requests.py`：拆成 `runtime/config.py`、`runtime/planner.py`、`runtime/deployer.py`、`runtime/metrics.py`，原文件暂时保留薄包装器。
- 再拆根目录 `train_tahrl.py`：配置加载、数据准备、训练阶段、checkpoint 和报告分别成为可测试函数。

### Phase 3：建立领域边界

- 把请求、完整候选、资源快照、资源足迹、联合动作、提交结果定义为 `dataclass`/协议对象，集中放在 `core/domain/`。
- `core/marl/` 只做候选排序、联合解码和原子提交；不得直接调用 Ryu、Mininet 或文件系统。
- `envs/` 只实现环境状态转移；训练日志、实验目录和 CLI 参数不向下渗透。
- `sdn/` 只依赖领域计划和执行协议，不反向 import `scripts/`。
- 消除 `sdn -> scripts`、`envs -> trainer` 这类反向依赖；脚本只能位于最外层调用业务包。

### Phase 4：测试和删除兼容层

- `tests/unit` 覆盖账本、候选完整性、mask、解码器、迁移事务和生命周期释放。
- `tests/integration` 覆盖 HRL→Top-K→WQMIX→ledger；SDN 单独用 fake client 测试。
- `tests/e2e` 只保留少量 smoke，不在测试中启动长时间训练。
- 连续两个版本不再有外部调用后，删除 `legacy/` 和根目录兼容包装器。

## 依赖规则

允许的方向：

```text
sfc_project / scripts -> core, envs, trainer, sdn, bms
trainer                -> core, envs
envs                   -> core
sdn                    -> core
core                   -> 标准库和明确声明的第三方库
```

禁止的方向：

- `core` 导入 `scripts`、`artifacts` 或 `data` 的具体实验目录。
- `envs` 导入训练器。
- `sdn` 导入脚本中的私有函数。
- 业务模块通过 `os.chdir()`、隐式当前目录或硬编码绝对路径寻找数据。

## 第一批最值得改的文件

1. `scripts/run_sdn_runtime_requests.py`（约 8.8k 行）：最大风险和最大收益，先做薄包装拆分。
2. `envs/modules/HRL_Coordinator.py`、`envs/modules/low_level_controller.py`：把策略决策、拓扑搜索、资源检查拆开。
3. `train_tahrl.py`（约 56kB）：改成 `train_tahrl` 包的 CLI 适配器。
4. `legacy_hrl_runtime/`：冻结、加 README 和兼容测试，停止继续复制修补。
5. `scripts/`：按用途分组，但先不改模块内部逻辑。

## 每个重构提交的验收条件

- 一个职责变化对应一个小提交，提交前后至少通过 `compileall` 和相关 smoke。
- 新旧入口输出的请求数、候选数、接受/拒绝数、资源账本版本一致。
- 不把 `artifacts` 中的历史结果当作源码依赖。
- 任何移动都能用 Git 一条命令回滚；不直接删除主备份中的唯一副本。

## 建议的提交顺序

```text
chore: initialize repository and record baseline
chore: isolate generated artifacts and legacy runtime
chore: group experiment scripts by purpose
refactor: split runtime request orchestration
refactor: extract HRL domain objects and resource interfaces
test: add unit and integration regression gates
chore: remove legacy compatibility tree
```
