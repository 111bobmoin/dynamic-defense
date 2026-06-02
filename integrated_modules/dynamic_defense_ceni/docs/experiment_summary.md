# 实验结果摘要

本文档记录 expanded CICIDS2017 动态防御实验、策略族级模型验证、minority 场景专项验证，以及 CENI 文件接口对接校验结果。

## 实验配置

- 数据场景：expanded CICIDS2017 有序场景，覆盖 13 个标签。
- 检测模式：`hybrid`
- 优化器：`actor_critic`
- 控制器路径：REST 动作翻译控制器
- 控制器执行模式：`stateful`
- CENI 导出文件：`/tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json`
- 校验脚本：`optimize/multi_vm/validate_defense_inputs.py`
- 最终归档路径：`artifacts/final_expanded_rest_stateful_ceni_validation/`

最终归档目录保存实验输出快照。普通 `reports/` 和 `runtime/` 文件属于运行生成产物，不应作为常规代码变更提交；只有在明确复制到 `artifacts/` 实验目录后，才作为实验归档保存。

## 最终指标

| 指标 | 数值 |
|---|---:|
| `windows` | 11 |
| `adjustment_events` | 11 |
| `detector` | `hybrid` |
| `optimizer` | `actor_critic` |
| `detection_success_rate` | 1.0 |
| `defense_success_rate` | 1.0 |
| CENI 校验 | `PASS` |

CENI 校验结果说明生成的 `dynamic_defense.json` 符合 multi-VM 控制器集成所需输入格式，`validate_defense_inputs.py` 结果为 `PASS`。

## Expanded v2 模型验证

`expanded_v2` FlowMLP 模型使用 `scripts/train_torch_flow_classifier.py` 训练，并启用 `feature_set=extended`。该模型用于提升 expanded 13 类场景的检测能力，同时不改变动态防御主流程。

训练参数：

| 参数 | 数值 |
|---|---:|
| `hidden_dim` | 128 |
| `num_layers` | 3 |
| `dropout` | 0.2 |
| `lr` | 0.001 |
| `weight_decay` | 0.0001 |
| `batch_size` | 128 |
| `class_weight` | `balanced` |
| `patience` | 25 |
| `epochs` | 120 |

独立检测指标：

| 指标 | 数值 |
|---|---:|
| `accuracy` | 0.8844 |
| `macro_f1` | 0.8461 |
| `weighted_f1` | 0.8639 |

与旧 expanded 模型对比：

| 模型 | `accuracy` |
|---|---:|
| 旧 expanded FlowMLP | 约 0.757 |
| `expanded_v2` FlowMLP | 0.8844 |

使用 `expanded_v2` 后的动态防御指标：

| 指标 | 数值 |
|---|---:|
| `windows` | 11 |
| `adjustment_events` | 11 |
| `attack_type_accuracy.exact` | 1.0 |
| `attack_type_accuracy.family` | 1.0 |
| `strategy_match_accuracy` | 1.0 |

`detector_source_counts`：

| 来源 | 数量 |
|---|---:|
| `torch` | 9 |
| `template_fallback` | 2 |

Web Attack 子类仍然是模型弱点：

| 类别 | 较弱指标 |
|---|---:|
| `Web Attack Brute Force` | recall = 0.10 |
| `Web Attack Sql Injection` | f1 约 0.46 |
| `Web Attack XSS` | f1 约 0.69 |

`expanded_v2` 模型权重和生成的 `reports/` 输出属于本地实验产物，不提交到 Git。

## Family v3 策略族模型验证

`family v3` FlowMLP 是策略族级分类器，目标是服务动态防御策略路由，不用于 CICIDS2017 细粒度子类报告。

`family v3` 标签集合：

- `BENIGN`
- `DDoS`
- `PortScan`
- `Brute Force`
- `Web Attack`
- `Heartbleed`

独立检测指标：

| 指标 | 数值 |
|---|---:|
| `accuracy` | 0.9606 |
| `macro_f1` | 0.9584 |
| `weighted_f1` | 0.9606 |
| `dropped_unknown_rows` | 0 |

主要类别指标：

| 类别 | F1 |
|---|---:|
| `DDoS` | 约 0.986 |
| `Brute Force` | 约 0.951 |
| `Web Attack` | 约 0.924 |
| `PortScan` | 约 0.920 |
| `Heartbleed` | 1.0 |
| `BENIGN` | 约 0.969 |

使用 `family v3` 后的动态防御指标：

| 指标 | 数值 |
|---|---:|
| `windows` | 11 |
| `adjustment_events` | 11 |
| `attack_type_accuracy.exact` | 0.2727 |
| `attack_type_accuracy.family` | 1.0 |
| `strategy_match_accuracy` | 1.0 |

`detector_source_counts`：

| 来源 | 数量 |
|---|---:|
| `torch` | 11 |

`attack_type_accuracy.exact` 较低是预期现象，因为该模型输出策略族标签，而真实窗口标签仍包含 `DoS Hulk`、`SSH-Patator`、`Web Attack XSS` 等 CICIDS2017 细粒度标签。对于动态防御目标，策略族级结果更关键，因为这些子类最终映射到相同防御策略。

REST/stateful 联调结果：

| 检查项 | 结果 |
|---|---:|
| `controller_execution_plan.jsonl` | 30 行 |
| 连接或运行错误 | 无 `Connection refused` / `ERROR` |
| `validate_defense_inputs.py` | `PASS` |

`family v3` 模型权重和 meta 文件已提交到 `models/`。生成的 `reports/`、`runtime/` 和 `artifacts/` 输出仍作为本地实验产物，不提交到 Git。

## p4 真实 CENI 多 VM 完整运行验证

最终版本已经在 `p4` 真实 CENI 多 VM 环境完成完整运行验证，不再只是静态写入 `dynamic_defense.json`。本次验证覆盖 PyTorch family_v3 模型推理、`hybrid` 检测、`actor_critic` 策略优化、REST/stateful controller 动作联调、CENI 文件导出、`validate_defense_inputs.py` 校验和 CENI dashboard 展示。

验证环境：

| 项目 | 结果 |
|---|---|
| host | `p4` |
| project_dir | `/home/p4/dynamic_defense_ceni` |
| python_env | `/home/p4/dynamic_defense_ceni/.venv` |
| CENI controller/dashboard | `/home/p4/optimize` |
| CENI input | `/tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json` |

`p4` 环境已安装并验证：

| 项目 | 状态 |
|---|---|
| Python | 3.8.10 |
| `pandas` | OK |
| `numpy` | OK |
| `sklearn` | OK |
| `torch` | OK，2.2.2+cpu |
| `yaml` | OK |
| `requests` | OK |
| `pytest` | 20 passed |

完整运行链路：

- PyTorch `family_v3` 模型推理。
- `hybrid` 检测。
- `actor_critic` 策略优化。
- REST/stateful controller 动作联调。
- `dynamic_defense.json` 导出。
- `validate_defense_inputs.py` 校验。
- CENI dashboard 展示。

`attack_defender.py` 运行结果：

| 字段 | 值 |
|---|---:|
| `status` | `OK` |
| `detector` | `hybrid` |
| `optimizer` | `actor_critic` |
| `windows` | 11 |
| `adjustment_events` | 11 |
| `detection_success_rate` | 1.0 |
| `defense_success_rate` | 1.0 |
| `attack_type_accuracy.exact` | 0.2727272727272727 |
| `attack_type_accuracy.family` | 1.0 |
| `strategy_match_accuracy` | 1.0 |
| `detector_source_counts` | `{"torch": 11}` |

REST/stateful 联调结果：

| 项目 | 结果 |
|---|---|
| 连接或运行错误 | 无 `Connection refused` / `ERROR` |
| `controller_execution_plan.jsonl` | 30 lines |
| `controller_execution_mode` | `stateful` |

CENI 导出结果：

| 字段 | 值 |
|---|---|
| `status` | `attack_detected` |
| `severity` | `critical` |
| `risk_score` | 75 |
| `affected_links` | `["s3-s4", "s4-s7"]` |
| `affected_nodes` | `["s3", "s4", "s7"]` |

CENI validate 结果：

```text
DEFENSE_INPUT_CHECK dynamic_defense PASS
defense_input_error_count = 0
defense_input_warning_count = 0
DEFENSE_INPUT_VALIDATE_RESULT = PASS
```

CENI dashboard 展示现象：

- 顶部“动态防御系统”卡片显示“发现攻击”。
- 大屏显示 11 个策略调整事件。
- 大屏显示风险评分 75。
- `affected_links = ["s3-s4", "s4-s7"]` 和 `affected_nodes = ["s3", "s4", "s7"]` 在拓扑中触发展示。

严格保留的唯一边界：

- 未执行真实 `tc` / `iptables` / `ovs-ofctl` 网络动作。
- 当前动作执行为 REST/stateful 计划生成与状态更新，未启用 `shell` execution。

## Minority 场景验证

minority 场景是用于窗口级验证稀少类别的补充有序场景，重点覆盖 `Heartbleed` 和 `Web Attack Sql Injection`。

包含标签：

- `BENIGN`
- `Heartbleed`
- `Web Attack Sql Injection`
- `Web Attack XSS`
- `Web Attack Brute Force`

数据规模：

| 项目 | 数值 |
|---|---:|
| 类别数 | 5 |
| 每类行数 | 200 |
| `total_rows` | 1000 |

过采样记录：

| 标签 | 原始行数 | 抽样后行数 | 是否过采样 |
|---|---:|---:|---|
| `Heartbleed` | 11 | 200 | true |
| `Web Attack Sql Injection` | 21 | 200 | true |

minority FlowMLP 的 `accuracy` 为 `0.72`。该模型仅用于稀少类别检测和策略路由的专项功能验证，不作为生产级流量分类器。

minority 动态防御结果：

| 指标 | 数值 |
|---|---:|
| `windows` | 5 |
| `adjustment_events` | 5 |
| `detection_success_rate` | 1.0 |
| `defense_success_rate` | 1.0 |
| `attack_type_accuracy.exact` | 1.0 |
| `attack_type_accuracy.family` | 1.0 |
| `strategy_match_accuracy` | 1.0 |

`detector_source_counts`：

| 来源 | 数量 |
|---|---:|
| `template_fallback` | 3 |
| `torch` | 2 |

策略覆盖：

- `Heartbleed` -> `s_heartbleed_deep_inspection`
- `Web Attack*` -> `s_web_attack_strict`

minority 运行导出的 CENI `dynamic_defense.json` 同样通过 `optimize/multi_vm/validate_defense_inputs.py` 校验。

minority CSV 文件、minority 模型权重和 minority 实验归档均为本地实验输出，不提交到 Git。如需复现实验快照，应在源码管理之外归档，或只把选定最终输出复制到 `artifacts/` 目录。

## 运行命令

从本地 CICIDS2017 CSV 文件构造 expanded 有序场景：

```bash
python scripts/make_cicids2017_subset.py \
  --raw-dir /path/to/CICIDS2017/csv \
  --rows-per-class 200
```

训练 expanded FlowMLP 模型：

```bash
python scripts/train_torch_flow_classifier.py \
  --input data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv \
  --model-out models/torch_flow_classifier_expanded.pt \
  --meta-out models/torch_flow_classifier_expanded_meta.json
```

启动 `stateful` REST 控制器：

```bash
python scripts/translating_defense_controller.py \
  --host 127.0.0.1 \
  --port 18082 \
  --execution-mode stateful
```

针对 expanded 场景运行动态防御：

```bash
python attack_defender.py \
  --input data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv \
  --build-templates \
  --window-size 200 \
  --limit 2200 \
  --detector hybrid \
  --torch-model models/torch_flow_classifier_expanded.pt \
  --torch-meta models/torch_flow_classifier_expanded_meta.json \
  --torch-threshold 0.70 \
  --optimizer actor_critic \
  --adapter rest \
  --controller-endpoint http://127.0.0.1:18082
```

导出 CENI 控制器输入：

```bash
python scripts/export_ceni_dynamic_defense_status.py \
  --network-status /tmp/optimize_multi_vm_runtime/defense_feeds/network_status.json \
  --out-json /tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json
```

在 optimize/multi_vm 侧执行校验：

```bash
python optimize/multi_vm/validate_defense_inputs.py
```

## 执行边界

最终实验使用 `ActionExecutor` 的 `stateful` 模式。该模式会更新：

- `runtime/controller_state.json`
- `reports/controller_execution_plan.jsonl`

该模式不会执行真实网络修改命令。当前实验没有真实运行 `tc`、`iptables` 或 `ovs-ofctl` 修改操作。网络动作 `rate_limit` 和 `isolate_flow` 只表现为控制器执行计划和 SDN/CENI 意图，用于验证对接链路。

这个边界是有意保留的：它在保证 CENI 集成安全的同时，验证了检测、优化、动作翻译、状态更新、执行计划生成和 CENI JSON 导出的完整流程。

## 模型说明

早期 expanded FlowMLP 检测器的 `accuracy` 约为 `0.757`，可以支持原型验证和集成测试，但不是生产级流量分类器。后续已经通过 `expanded_v2` 和 `family v3` 提升检测与策略路由效果。未来仍可从以下方向继续优化：

- 更完整的特征归一化和特征选择；
- 按 CICIDS2017 不同日期进行更严格的数据划分；
- 在固定每类行数之外使用更细致的类别重平衡方法；
- 使用更深或正则化更充分的 PyTorch 模型；
- 针对 `hybrid` 模式进行阈值校准；
- 在独立保留的 CICIDS2017 文件上评估，而不只评估有序场景。

## 数据管理原则

不要提交 CICIDS2017 原始 CSV 大文件。原始数据应保存在仓库之外，并按需生成小规模场景 CSV。生成的 `reports/` 和 `runtime/` 文件也不应提交，除非它们被明确复制到 `artifacts/` 目录作为实验归档。
