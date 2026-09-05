# 阶段三：WQMIX 决策适配

- `BatchCandidateQNetwork` 增加独立 `no_migration_head`，动作 0 不再与迁移目的地共享同一个输出头。
- `DynamicMigrationEnv.get_state()` 增加 `uncertainty`，按当前时间之前的 offered traffic 历史计算每个节点的滑动标准差。
- `decode_joint_candidates()` 增加可选 `remaining_lifetimes_s` 与 `migration_prepare_s` 参数；当剩余寿命不足准备时间时，自动过滤所有迁移动作，仅保留 reject/no-migration。

旧数据集和旧解码调用不传门控参数时保持兼容。尚未进行训练或实验验证。
