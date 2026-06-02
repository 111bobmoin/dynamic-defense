# 最终交付检查清单

本文档用于课程/课题验收前的最终自查，记录 `dynamic_defense` 项目的交付状态、环境要求、smoke test、截图清单、测试大纲对应关系和 CENI 文件协议验证流程。

## 1. 项目最终交付状态

当前版本定位为：已在 `p4` 真实 CENI 多 VM 环境完成完整运行验证的动态防御原型。严格保留的唯一边界是未执行真实 `tc` / `iptables` / `ovs-ofctl` 网络动作。

已冻结版本状态：

| 项目 | 状态 |
|---|---|
| 单元测试 | `python -m pytest -q`，结果为 20 passed |
| 检测模式 | `hybrid` |
| 策略优化器 | `actor_critic` |
| 策略族模型 | `family_v3` 已提交到 `models/` |
| CENI 文件协议导出 | `dynamic_defense.json` 已通过校验 |
| CENI 校验结果 | `DEFENSE_INPUT_VALIDATE_RESULT=PASS` |
| 网络动作执行 | stateful + execution plan，不真实执行 `tc`、`iptables`、`ovs-ofctl` |

最终 `dynamic_defense_summary.json` 核心指标：

| 字段 | 结果 |
|---|---:|
| `detector` | `hybrid` |
| `optimizer` | `actor_critic` |
| `windows` | 11 |
| `adjustment_events` | 11 |
| `detection_success_rate` | 1.0 |
| `defense_success_rate` | 1.0 |
| `attack_type_accuracy.family` | 1.0 |
| `strategy_match_accuracy` | 1.0 |
| `detector_source_counts.torch` | 11 |

## 2. GitHub 仓库与 tag 信息

GitHub 仓库：

```text
https://github.com/cphzyl1611/dynamic_defense.git
```

最终交付 tag：

```text
v1.0-dynamic-defense-ceni
```

验收时建议以 tag 版本为准：

```bash
git clone https://github.com/cphzyl1611/dynamic_defense.git
cd dynamic_defense
git fetch --tags
git checkout v1.0-dynamic-defense-ceni
```

如果在本地继续补充文档，请确认最终提交和 tag 对应关系，并在截图中展示 `git log -1 --oneline` 与 `git tag --points-at HEAD`。

## 3. 环境要求

推荐环境：

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu VM / CENI VM |
| Python | Python 3.7+ |
| PyTorch | CPU 版 `torch` 即可，不依赖 GPU |
| Python 依赖 | `pandas`、`numpy`、`scikit-learn`、`PyYAML`、`pytest` 等 |
| 网络权限 | smoke test 不需要 `sudo` |
| CENI 文件目录 | `/tmp/optimize_multi_vm_runtime/defense_feeds/` 和 `/tmp/optimize_multi_vm_runtime/defense_inputs/` |

Conda 环境示例：

```bash
conda env create -f environment.yml
conda activate dd37
```

部署前环境检查：

```bash
python scripts/check_deployment_readiness.py
python scripts/check_network_action_environment.py
```

## 4. 最终 smoke test 命令

基础测试：

```bash
python -m pytest -q
```

预期结果：

```text
20 passed
```

动态防御主流程 smoke test：

```bash
python attack_defender.py \
  --input data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv \
  --build-templates \
  --window-size 200 \
  --limit 2200 \
  --detector hybrid \
  --torch-model models/torch_flow_classifier_expanded_family_v3.pt \
  --torch-meta models/torch_flow_classifier_expanded_family_v3_meta.json \
  --torch-threshold 0.70 \
  --optimizer actor_critic
```

REST/stateful 控制器联调 smoke test，可在一个终端启动控制器：

```bash
python scripts/translating_defense_controller.py \
  --host 127.0.0.1 \
  --port 18082 \
  --execution-mode stateful
```

另一个终端运行动态防御：

```bash
python attack_defender.py \
  --input data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv \
  --build-templates \
  --window-size 200 \
  --limit 2200 \
  --detector hybrid \
  --torch-model models/torch_flow_classifier_expanded_family_v3.pt \
  --torch-meta models/torch_flow_classifier_expanded_family_v3_meta.json \
  --torch-threshold 0.70 \
  --optimizer actor_critic \
  --adapter rest \
  --controller-endpoint http://127.0.0.1:18082
```

CENI 文件协议导出：

```bash
python scripts/export_ceni_dynamic_defense_status.py \
  --network-status /tmp/optimize_multi_vm_runtime/defense_feeds/network_status.json \
  --out-json /tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json
```

CENI 输入校验：

```bash
python optimize/multi_vm/validate_defense_inputs.py
```

预期输出包含：

```text
DEFENSE_INPUT_VALIDATE_RESULT=PASS
```

## 5. 截图清单

建议验收材料至少包含以下截图：

| 序号 | 截图内容 | 目的 |
|---:|---|---|
| 1 | GitHub 仓库首页与 tag 页面 | 证明提交版本和 `v1.0-dynamic-defense-ceni` tag |
| 2 | `python -m pytest -q` 终端结果 | 证明测试通过，显示 20 passed |
| 3 | `reports/dynamic_defense_summary.json` | 展示最终动态防御核心指标 |
| 4 | `reports/dynamic_defense_events.csv` | 展示窗口级策略选择与 `strategy_id` |
| 5 | `models/torch_flow_classifier_expanded_family_v3_meta.json` | 证明 `family_v3` 策略族模型标签集合 |
| 6 | REST 控制器运行终端 | 展示动作接收、翻译和 `execution_result` |
| 7 | `runtime/controller_state.json` | 展示 stateful 执行状态 |
| 8 | `reports/controller_execution_plan.jsonl` | 展示 `rate_limit` / `isolate_flow` 只生成执行计划 |
| 9 | `/tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json` | 展示 CENI 文件协议 payload |
| 10 | `validate_defense_inputs.py` 终端输出 | 展示 `DEFENSE_INPUT_VALIDATE_RESULT=PASS` |

## 6. 每张截图应展示的关键字段

GitHub 仓库与 tag 截图应展示：

- 仓库地址 `https://github.com/cphzyl1611/dynamic_defense.git`
- tag 名称 `v1.0-dynamic-defense-ceni`
- 最新提交 hash 或 tag 对应提交

pytest 截图应展示：

- 命令 `python -m pytest -q`
- 结果 `20 passed`

`dynamic_defense_summary.json` 截图应展示：

- `detector = hybrid`
- `optimizer = actor_critic`
- `windows = 11`
- `adjustment_events = 11`
- `detection_success_rate = 1.0`
- `defense_success_rate = 1.0`
- `attack_type_accuracy.family = 1.0`
- `strategy_match_accuracy = 1.0`
- `detector_source_counts = {"torch": 11}`

`dynamic_defense_events.csv` 截图应展示：

- `window_id`
- `attack_type`
- `true_majority_label`
- `strategy_id`
- `reward`
- `detector_source`
- `torch_label`
- `torch_confidence`

`torch_flow_classifier_expanded_family_v3_meta.json` 截图应展示：

- `model_type = FlowMLP`
- `label_mode = family`
- `feature_set = extended`
- `labels = ["BENIGN", "Brute Force", "DDoS", "Heartbleed", "PortScan", "Web Attack"]`
- `accuracy` 或 `test_accuracy`

REST 控制器截图应展示：

- `POST /defense/action`
- `translated_action`
- `execution_result`
- 无 `Connection refused` / `ERROR`

`controller_state.json` 截图应展示：

- `last_monitored_window`
- `current_model`
- `thresholds`
- `log_fields`

`controller_execution_plan.jsonl` 截图应展示：

- `action_type = rate_limit`
- `action_type = isolate_flow`
- `tc` 命令计划
- OpenFlow meter / drop intent
- 没有真实执行系统命令

`dynamic_defense.json` 截图应展示：

- `status`
- `severity`
- `updated_at`
- `summary`
- `metrics`
- `alerts`
- `risk_score`
- `affected_links`
- `affected_nodes`
- `recommendation`
- `actions`
- `version`
- `source`

`validate_defense_inputs.py` 截图应展示：

- 校验目标文件 `/tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json`
- 输出 `DEFENSE_INPUT_VALIDATE_RESULT=PASS`

## 7. 测试大纲 34/35/36 对应关系

### 34 防御策略库构建

对应项目内容：

- `configs/strategies.yaml` 定义防御策略库。
- `strategy_loader.py` 将策略库导入 SQLite。
- `src/dynamic_defense/policy_store.py` 提供 `PolicyStore` / `DefensePolicy` 数据结构。
- 策略 ID 包括 `s_benign_monitor`、`s_ddos_vote_rate_limit`、`s_portscan_isolate`、`s_bruteforce_ssh_ftp`、`s_web_attack_strict`、`s_heartbleed_deep_inspection`。

建议展示：

```bash
python strategy_loader.py --config configs/strategies.yaml --db data/policies.sqlite
```

### 35 威胁特征提取匹配

对应项目内容：

- `feature_analyzer.py` 支持 CICIDS2017 特征模板匹配。
- `configs/feature_templates.yaml` 记录威胁特征模板。
- `src/dynamic_defense/torch_detector.py` 使用 CPU PyTorch `FlowMLP` 做流量分类。
- `hybrid` 检测模式结合 `torch` 与模板匹配，必要时回退到 `template_fallback`。

建议展示：

```bash
python feature_analyzer.py \
  --input data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv \
  --build-templates \
  --limit 2200
```

### 36 应对实时攻击的动态防御

对应项目内容：

- `attack_defender.py` 按窗口处理流量并选择动态防御策略。
- `src/dynamic_defense/ac_optimizer.py` 提供 PyTorch `actor_critic` 策略优化器。
- `scripts/translating_defense_controller.py` 提供 REST 动作翻译控制器。
- `src/dynamic_defense/action_executor.py` 支持 `simulated`、`stateful`、`shell` 三种执行模式。
- `reports/controller_execution_plan.jsonl` 记录 `rate_limit` 和 `isolate_flow` 的执行计划。
- CENI 文件接口通过 `scripts/export_ceni_dynamic_defense_status.py` 导出 `dynamic_defense.json`。

建议展示：

```bash
python attack_defender.py \
  --input data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv \
  --build-templates \
  --window-size 200 \
  --limit 2200 \
  --detector hybrid \
  --optimizer actor_critic \
  --adapter rest \
  --controller-endpoint http://127.0.0.1:18082
```

## 8. CENI 文件协议验证步骤

1. 准备 CENI 文件目录：

```bash
mkdir -p /tmp/optimize_multi_vm_runtime/defense_feeds
mkdir -p /tmp/optimize_multi_vm_runtime/defense_inputs
```

2. 准备或确认网络状态输入：

```text
/tmp/optimize_multi_vm_runtime/defense_feeds/network_status.json
```

3. 运行动态防御主流程，生成：

```text
reports/dynamic_defense_summary.json
reports/dynamic_defense_events.csv
runtime/controller_state.json
reports/controller_execution_plan.jsonl
```

4. 导出 CENI 输入文件：

```bash
python scripts/export_ceni_dynamic_defense_status.py \
  --network-status /tmp/optimize_multi_vm_runtime/defense_feeds/network_status.json \
  --out-json /tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json
```

5. 检查 `dynamic_defense.json` 必需字段：

```text
status
severity
updated_at
summary
message
metrics
alerts
risk_score
scenario
event_type
affected_links
affected_nodes
recommendation
actions
version
source
```

6. 在 optimize/multi_vm 侧执行校验：

```bash
python optimize/multi_vm/validate_defense_inputs.py
```

7. 确认输出：

```text
DEFENSE_INPUT_VALIDATE_RESULT=PASS
```

## 9. p4 真实 CENI 多 VM 完整运行验证

最终版本已经在 `p4` 真实 CENI 多 VM 环境完成完整运行验证，不再只是静态写入 `dynamic_defense.json`。本次验证使用平台侧已有 CENI controller/dashboard，不覆盖平台原有项目目录。

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
- 受影响链路和节点根据 `affected_links = ["s3-s4", "s4-s7"]`、`affected_nodes = ["s3", "s4", "s7"]` 在拓扑中触发展示。

严格保留的唯一边界：

- 未执行真实 `tc` / `iptables` / `ovs-ofctl` 网络动作。
- 当前动作执行为 REST/stateful 计划生成与状态更新，未启用 `shell` execution。

## 10. 已完成内容和未完成边界

已完成内容：

- 动态防御策略库构建。
- CICIDS2017 威胁特征模板匹配。
- `template` / `torch` / `hybrid` 检测流程。
- CPU PyTorch `FlowMLP` 流量分类模型。
- `family_v3` 策略族级模型训练与验证。
- PyTorch `actor_critic` 策略优化。
- REST translating controller。
- `ActionExecutor` 的 `stateful` 执行框架。
- `runtime/controller_state.json` 状态更新。
- `reports/controller_execution_plan.jsonl` 执行计划生成。
- CENI 文件协议 `dynamic_defense.json` 导出。
- `p4` 真实 CENI 多 VM 完整运行验证。
- `validate_defense_inputs.py` 校验通过，结果为 `PASS`。

严格保留的唯一边界：

- 未执行真实 `tc` / `iptables` / `ovs-ofctl` 网络动作。
- 当前动作执行为 REST/stateful 计划生成与状态更新，未启用 `shell` execution。

## 11. 老师验收可用项目总结

本项目实现了一个面向 CICIDS2017 流量场景的动态防御原型系统，完成了从防御策略库构建、威胁特征提取匹配、`FlowMLP` 检测模型、`actor_critic` 策略优化，到 REST 控制器动作翻译、stateful 执行计划生成和 CENI 文件协议导出的完整闭环。最终版本在 expanded 场景下实现 `windows = 11`、`adjustment_events = 11`、`detection_success_rate = 1.0`、`defense_success_rate = 1.0`、`attack_type_accuracy.family = 1.0`、`strategy_match_accuracy = 1.0`，并通过 `validate_defense_inputs.py` 校验，输出 `DEFENSE_INPUT_VALIDATE_RESULT=PASS`。当前版本定位为已在 `p4` 真实 CENI 多 VM 环境完成完整运行验证的动态防御原型，网络动作保持唯一安全边界：REST/stateful 计划生成与状态更新，不真实执行 `tc`、`iptables` 或 `ovs-ofctl` 配置。
