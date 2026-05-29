# CENI Sidecar 集成日志

## 1. 操作背景

本次工作基于重新 clone 的本地平台仓库：

- 仓库来源：https://github.com/111bobmoin/dynamic-defense
- 本地目录：`~/dynamic-defense`

由于原仓库已有文件不允许修改，也不允许删除，因此没有直接改动平台既有代码，而是采用只新增文件的 sidecar 扩展方式接入。

`dynamic_defense_ceni` 后续可以作为 `integrated_modules` 或 `external_modules` 接入仓库，但当前阶段先不混入原平台代码，避免混淆平台原有 `dynamic_defense` 概念和新增的 `dynamic_defense_ceni` 模块。

## 2. 环境说明

本次操作环境：

- 使用 WSL。
- 使用 Python `venv`，不使用 conda。
- venv 激活方式：

```bash
source .venv/bin/activate
```

## 3. 干净仓库原则

本次工作遵循干净仓库原则：

- 重新 clone 了一份干净仓库。
- `git status --short` 初始为空。
- 原有平台文件不能修改。
- 原有平台文件不能删除。
- 只允许新增 `extensions/dynamic_defense_ceni` 和 `tests/test_dynamic_defense_ceni_extension.py`。
- 不提交 `.venv`、`logs`、`__pycache__`、`.pytest_cache`、`/tmp` 输出、`reports`、`runtime` 等运行产物。

已完成提交：

```text
726a454 feat: add dynamic defense CENI sidecar bridge
```

该提交只新增 sidecar 扩展文件和对应测试，没有修改 `dynamic_defense_dashboard/server.py`，没有修改 `dynamic_defense_dashboard/static` 下已有文件，也没有修改 `muti3`、`graph`、`log` 下已有文件。

## 4. 新增 sidecar 文件说明

### `extensions/dynamic_defense_ceni/README.md`

说明 sidecar 扩展的用途、输入文件、默认输出路径、运行方式和校验方式。该文档用于后续维护者理解如何在不修改平台原有代码的前提下，把 `dynamic_defense_ceni` 的结果转换为平台可消费的 `dynamic_defense.json`。

### `extensions/dynamic_defense_ceni/action_logger.py`

标准库实现的 JSONL 动作日志记录器。每条日志包含：

- `timestamp`
- `action`
- `status`
- `message`
- `inputs`
- `outputs`
- `details`

日志以 UTF-8 追加写入，并自动创建父目录。

### `extensions/dynamic_defense_ceni/bridge.py`

sidecar 桥接主程序，标准库优先实现，不依赖 `torch`。

主要功能：

- 支持读取 `dynamic_defense_ceni` 项目输出：
  - `reports/dynamic_defense_summary.json`
  - `reports/dynamic_defense_events.csv`
  - `runtime/controller_state.json`
  - `reports/controller_execution_plan.jsonl`
- 支持 `--static-demo` 生成 P4 最终验证 payload。
- 支持 `--once` 单次运行。
- 支持 `--watch-interval` 周期运行。
- 默认输出目录为 `/tmp/optimize_multi_vm_runtime/defense_inputs`。
- 默认输出文件为 `dynamic_defense.json`。
- 写入 JSON 时先写 `.tmp`，再使用 replace 原子替换目标文件。
- 每次启动、读取、转换、写入、失败、退出都会写 action log。

输出 payload 包含：

- `status`
- `severity`
- `updated_at`
- `summary`
- `message`
- `metrics`
- `alerts`
- `risk_score`
- `scenario`
- `event_type`
- `affected_links`
- `affected_nodes`
- `recommendation`
- `actions`
- `version`
- `source`

### `extensions/dynamic_defense_ceni/run_bridge.sh`

Bash 启动脚本，不使用 conda。

脚本会进入仓库根目录，并使用当前仓库 `.venv`：

```bash
source .venv/bin/activate
```

如果不传参数，默认调用：

```bash
python extensions/dynamic_defense_ceni/bridge.py --static-demo --once
```

如果传入参数，则透传给 `bridge.py`。

### `extensions/dynamic_defense_ceni/validate_bridge_output.py`

标准库实现的输出校验脚本。

参数：

- `--input`：待校验的 `dynamic_defense.json`。
- `--log-file`：action log 路径，默认 `logs/dynamic_defense_ceni_actions.jsonl`。

校验项：

- `status` 存在。
- `risk_score` 在 `0-100`。
- `affected_links` 是 list。
- `affected_nodes` 是 list。
- `alerts` 是 list。
- `source = dynamic_defense_ceni`。
- `version` 存在。

校验结果输出 `PASS` 或 `FAIL`，校验动作也写入 action log。

### `tests/test_dynamic_defense_ceni_extension.py`

pytest 测试文件，使用 `tempfile` 创建临时目录。

覆盖内容：

- `--static-demo` 能生成 `dynamic_defense.json`。
- action log 能生成 JSONL。
- `validate_bridge_output.py` 能输出 `PASS`。
- 原子写入后的目标 JSON 可解析。
- 不依赖 `torch`。
- 不依赖 `dynamic_defense_dashboard/server.py`。
- 不修改任何已有文件。

## 5. 验证命令和结果

记录的验证命令如下：

```bash
source .venv/bin/activate
bash extensions/dynamic_defense_ceni/run_bridge.sh --output-dir /tmp/dynamic_defense_bridge_test --static-demo --once
python extensions/dynamic_defense_ceni/validate_bridge_output.py --input /tmp/dynamic_defense_bridge_test/dynamic_defense.json
tail -n 20 logs/dynamic_defense_ceni_actions.jsonl
bash extensions/dynamic_defense_ceni/run_bridge.sh --output-dir /tmp/optimize_multi_vm_runtime/defense_inputs --static-demo --once
```

验证结果：

- `validate_bridge_output.py` 输出 `PASS`。
- action log 中出现 `startup`、`read`、`transform`、`write`、`exit`、`validate`。
- static demo payload 中 `source = dynamic_defense_ceni`。
- static demo payload 中 `version = v1.0.1-dynamic-defense-ceni`。
- static demo payload 中 `risk_score = 75`。
- static demo payload 中 `affected_links = ["s3-s4", "s4-s7"]`。
- static demo payload 中 `metrics.detector = hybrid`。
- static demo payload 中 `metrics.optimizer = actor_critic`。
- static demo payload 中 `metrics.detector_source_counts = {"torch": 11}`。

## 6. 接入方式说明

当前接入方式不是修改平台原有代码：

- 当前不是修改 `dynamic_defense_dashboard/server.py`。
- 当前不是修改 `dynamic_defense_dashboard/static` 前端。
- 当前不是修改 `muti3`、`graph`、`log`。
- 当前是通过新增 sidecar，把 `dynamic_defense_ceni` 结果转换为 `dynamic_defense.json`。

默认 `output-dir` 是：

```text
/tmp/optimize_multi_vm_runtime/defense_inputs
```

该路径用于兼容 CENI/optimize 的 `defense_inputs` 文件协议。

本地测试可以改成：

```text
/tmp/dynamic_defense_bridge_test
```

## 7. 日志策略

bridge 的运行日志写入：

```text
logs/dynamic_defense_ceni_actions.jsonl
```

该 JSONL 日志是运行产物，不提交。

本 Markdown 文件记录人工集成过程、约束、验证命令和验证结果，可以提交。

## 8. 后续计划

第一阶段：sidecar 文件接口集成，已完成。

第二阶段：把 `dynamic_defense_ceni` 作为 `integrated_modules/dynamic_defense_ceni` 或 `external_modules/dynamic_defense_ceni` 放入仓库。

第三阶段：如果允许修改平台 dashboard，再考虑把 sidecar 输出接入 dashboard API 或前端页面。

后续始终避免混淆平台原有 `dynamic_defense` 概念和新增的 `dynamic_defense_ceni` 模块。
