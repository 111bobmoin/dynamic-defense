# dynamic_defense_ceni 集成说明

这是 `dynamic_defense_ceni` 的集成快照，用于在当前仓库中保留算法模块源码、配置、脚本、测试、必要模型和小型样本数据。

原独立仓库版本为 `v1.0.1-dynamic-defense-ceni`。

请区分两个目录的职责：

- `extensions/dynamic_defense_ceni` 是 sidecar 适配器，负责把 `dynamic_defense_ceni` 的 `reports/runtime` 输出转换成平台可消费的 `dynamic_defense.json`。
- `integrated_modules/dynamic_defense_ceni` 是算法模块快照，保留检测、策略优化、控制器导出等算法侧实现。

不要混淆 sidecar 适配器和算法模块。前者负责格式适配和状态服务，后者负责动态防御算法与运行逻辑。

`reports/` 和 `runtime/` 是运行产物，不提交到仓库；实际运行后会重新生成。
