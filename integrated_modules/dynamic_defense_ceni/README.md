# dynamic_defense

`dynamic_defense` 是一个面向 CICIDS2017 流量场景的动态防御原型。项目把策略库、特征匹配、PyTorch 检测、Actor-Critic 策略优化、REST 控制器翻译、状态化动作执行计划和 CENI 文件接口串成一条可验证流程，用于实验动态防御策略切换与 CENI 大屏对接。

当前实现重点是“可复现实验闭环”，不是生产 IDS/IPS。网络动作默认不真实修改系统网络，只生成 `controller_state.json` 和 `controller_execution_plan.jsonl`。

## 模块结构

```text
configs/strategies.yaml                         # 动态防御策略库
configs/feature_templates.yaml                  # CICIDS2017 特征模板配置
strategy_loader.py                              # 策略库导入 SQLite
feature_analyzer.py                             # 模板匹配与特征分析
attack_defender.py                              # 动态防御主流程
src/dynamic_defense/
  ac_optimizer.py                               # PyTorch Actor-Critic 优化器
  action_executor.py                            # stateful/simulated/shell 执行框架
  ceni_adapter.py                               # dry-run/rest/local 动作适配
  defense_engine.py                             # template/torch/hybrid 检测编排
  optimizer.py                                  # heuristic 优化器
  policy_store.py                               # PolicyStore/DefensePolicy
  torch_detector.py                             # CPU PyTorch FlowMLP 检测器
scripts/
  make_cicids2017_subset.py                     # 13 类 expanded CICIDS2017 场景抽取
  train_torch_flow_classifier.py                # FlowMLP 训练入口，支持 --input
  translating_defense_controller.py             # REST 动作翻译控制器
  export_ceni_dynamic_defense_status.py         # CENI dynamic_defense.json 导出
  check_deployment_readiness.py                 # 部署前检查
  check_network_action_environment.py           # 网络执行环境只读探测
docs/experiment_summary.md                      # 最终实验结果摘要
artifacts/                                      # 已归档实验结果
```

## 环境配置

推荐在 Linux/Ubuntu 或 CENI VM 中使用 Python 3.7+。本仓库测试使用 `dd37` Conda 环境：

```bash
conda env create -f environment.yml
conda activate dd37
```

如果使用 venv，需要安装基础依赖，并额外准备 CPU 版 PyTorch 与 scikit-learn：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn
# 按你的 Python/CUDA 环境安装 CPU 版 torch；本项目默认 device=cpu
```

部署前可运行：

```bash
python scripts/check_deployment_readiness.py
python scripts/check_network_action_environment.py
```

## 基础运行

使用内置小样本跑通策略库、特征模板和动态防御：

```bash
python scripts/make_sample_cicids.py --out data/sample_cicids.csv
python strategy_loader.py --config configs/strategies.yaml --db data/policies.sqlite
python feature_analyzer.py --input data/sample_cicids.csv --build-templates --limit 300
python attack_defender.py \
  --input data/sample_cicids.csv \
  --build-templates \
  --window-size 100 \
  --limit 700
```

hybrid 检测与 Actor-Critic optimizer：

```bash
python attack_defender.py \
  --input data/cicids2017_subset/cicids2017_scenario_ordered.csv \
  --build-templates \
  --window-size 200 \
  --limit 2000 \
  --detector hybrid \
  --torch-threshold 0.70 \
  --optimizer actor_critic
```

主要运行产物在 `reports/`、`runtime/` 和 `models/` 下。`reports/` 与 `runtime/` 是运行输出，不应作为普通代码变更提交；需要保存实验结果时放入 `artifacts/` 归档目录。

## Expanded CICIDS2017 实验

不要把 CICIDS2017 原始大文件提交到仓库。将原始 CSV 放在本地或 VM 的外部目录，然后抽取平衡的 13 类 expanded 场景：

```bash
python scripts/make_cicids2017_subset.py \
  --raw-dir /path/to/CICIDS2017/csv \
  --rows-per-class 200
```

默认输出：

```text
data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv
data/cicids2017_subset/cicids2017_expanded_summary.json
```

训练 expanded FlowMLP 检测模型：

```bash
python scripts/train_torch_flow_classifier.py \
  --input data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv \
  --model-out models/torch_flow_classifier_expanded.pt \
  --meta-out models/torch_flow_classifier_expanded_meta.json
```

运行 expanded 动态防御实验：

```bash
python strategy_loader.py --config configs/strategies.yaml --db data/policies.sqlite

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

对应 REST 控制器可在另一个终端启动：

```bash
python scripts/translating_defense_controller.py \
  --host 127.0.0.1 \
  --port 18082 \
  --execution-mode stateful
```

当前 `ActionExecutor` 的网络动作边界：

- `simulated`：只记录计划和状态，不执行命令。
- `stateful`：更新 `runtime/controller_state.json`，并写 `reports/controller_execution_plan.jsonl`。
- `shell`：预留真实执行结构；`rate_limit` / `isolate_flow` 默认 `BLOCKED_FOR_SAFETY`。

也就是说，本项目当前没有真实执行 `tc`、`iptables` 或 `ovs-ofctl` 修改命令。

## CENI 文件接口导出

CENI 控制器文件接口约定：

```text
读取: /tmp/optimize_multi_vm_runtime/defense_feeds/network_status.json
写入: /tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json
```

导出 dynamic_defense 状态：

```bash
python scripts/export_ceni_dynamic_defense_status.py \
  --network-status /tmp/optimize_multi_vm_runtime/defense_feeds/network_status.json \
  --out-json /tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json
```

脚本会读取 `reports/dynamic_defense_summary.json`、`reports/dynamic_defense_events.csv`、`runtime/controller_state.json` 和 `reports/controller_execution_plan.jsonl`，并用 `.tmp` + `os.replace()` 原子写入 CENI 输入文件。

导出后可在 optimize/multi_vm 工程侧校验：

```bash
python optimize/multi_vm/validate_defense_inputs.py
```

最终 expanded REST + stateful + CENI 校验实验结果见 [docs/experiment_summary.md](docs/experiment_summary.md)。

## 模型说明与限制

当前仓库保留多种模型和策略路由配置，面向课程/课题验收时建议区分其用途：

- `expanded_v2` FlowMLP：使用 `feature_set=extended` 的 13 类 exact 分类模型，用于展示细粒度 CICIDS2017 标签检测能力；Web Attack 子类仍存在混淆。
- `family v3` FlowMLP：策略族级分类器，输出 `BENIGN`、`DDoS`、`PortScan`、`Brute Force`、`Web Attack`、`Heartbleed` 等策略族标签，主要用于防御策略路由，不用于 CICIDS2017 细粒度子类报告。
- `actor_critic`：PyTorch Actor-Critic 策略优化器，和已有 heuristic 流程并存；默认流程仍可使用 heuristic。
- `hybrid`：结合 `torch` 和模板匹配结果，低置信度或缺失模型时可回退到 `template_fallback`。

当前网络动作执行仍是安全验证边界：`rate_limit` 和 `isolate_flow` 不真实运行 `tc`、`iptables` 或 `ovs-ofctl`，只更新 `runtime/controller_state.json` 并生成 `reports/controller_execution_plan.jsonl`。如需真实下发网络规则，应在 CENI/SDN 控制器侧补充审计、回滚和最小权限控制后再启用。
