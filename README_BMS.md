# BMS 使用说明

当前 BMS 基础层提供 `BMSContext`、`BaseBrain`、`BaseModule`、`BaseSkill`、`SkillRegistry` 和 `ModuleRegistry`。

规则大脑仍通过现有 `core/marl/orchestration` 兼容链运行；学习大脑可使用 `core.bms.learning_brain.LearningBrainAgent`。设置 `configs/brain_config.yaml` 的 `mode` 为 `rule` 或 `learning` 作为上层启动器的配置约定。

本次先建立契约和安全边界，旧编排类保持兼容；计算逻辑迁移应逐个模块完成并通过 smoke 检查后再删除旧实现。
