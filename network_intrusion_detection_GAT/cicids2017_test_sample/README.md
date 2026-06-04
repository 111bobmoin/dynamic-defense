# CICIDS2017 流程测试样本

这个目录提供统一的样本生成功能，用于从给定的 `MachineLearningCVE` / CICIDS2017 数据集构造“可直接送入项目流程测试”的标准 CSV 样本。

目标不是恢复原始抓包里的真实主机关系，而是：

- 保留真实流特征分布
- 合成稳定、可重复的 `src_ip` / `dst_ip` / `timestamp` 等节点元数据
- 为 `inference.py -> node_summary.py -> repair_plan.py` 提供可重复测试输入
- 统一覆盖三种测试场景：
  - `no_anomaly`：无异常节点
  - `single_anomaly`：单异常节点
  - `multi_anomaly`：多异常节点

## 目录内容

- `generate_test_samples.py`
  - 统一入口。根据给定数据集生成一个或多个测试场景。
- `evaluate_test_sample.py`
  - 统一评估入口。根据场景类型执行对应评估。
- `generated/`
  - 默认输出目录。

## 支持的场景

### `no_anomaly`

用途：

- 验证无异常节点时 `node_summary.py` 的误报情况
- 验证在完全良性场景下 `repair_plan.py` 应输出空修复顺序或零代价结果
- 所有节点真值均为 `uncertain`

### `single_anomaly`

用途：

- 验证只有一个异常节点时，节点摘要是否能将其识别出来
- 验证 `repair_plan.py` 是否只对该异常节点给出修复排序
- 当前真值主节点类型为 `suspected_compromised_host`

### `multi_anomaly`

用途：

- 验证多异常节点混合场景下的节点角色识别效果
- 验证 `repair_plan.py` 对多异常节点的核心节点筛选与修复顺序
- 样本中包含：
  - `suspected_attacker`
  - `suspected_victim`
  - `suspected_compromised_host`
  - `uncertain`

## 生成方式

生成全部场景：

```powershell
python cicids2017_test_sample\generate_test_samples.py `
  --dataset-dir E:\dataset\MachineLearningCVE
```

只生成单个场景：

```powershell
python cicids2017_test_sample\generate_test_samples.py `
  --dataset-dir E:\dataset\MachineLearningCVE `
  --scenario no_anomaly
```

```powershell
python cicids2017_test_sample\generate_test_samples.py `
  --dataset-dir E:\dataset\MachineLearningCVE `
  --scenario single_anomaly
```

```powershell
python cicids2017_test_sample\generate_test_samples.py `
  --dataset-dir E:\dataset\MachineLearningCVE `
  --scenario multi_anomaly
```

## 默认输出结构

```text
cicids2017_test_sample/generated/
+-- suite_manifest.json
+-- no_anomaly/
|   +-- cicids2017_no_anomaly_sample.csv
|   +-- cicids2017_no_anomaly_ground_truth.csv
|   `-- sample_manifest.json
+-- single_anomaly/
|   +-- cicids2017_single_anomaly_sample.csv
|   +-- cicids2017_single_anomaly_ground_truth.csv
|   `-- sample_manifest.json
`-- multi_anomaly/
    +-- cicids2017_multi_anomaly_sample.csv
    +-- cicids2017_multi_anomaly_ground_truth.csv
    `-- sample_manifest.json
```

## 与主流程衔接

这些样本是标准 CSV 输入，因此不需要经过 `pcap_to_csv.py`，可直接送入：

- `inference.py`
- `node_summary.py`
- `repair_plan.py`

例如对 `single_anomaly` 场景：

```powershell
python inference.py `
  --model-path outputs\training\20260507_200244\model.pt `
  --input-path cicids2017_test_sample\generated\single_anomaly\cicids2017_single_anomaly_sample.csv `
  --output-path cicids2017_test_sample\generated\single_anomaly\cicids2017_single_anomaly_predictions.csv

python node_summary.py `
  --input-path cicids2017_test_sample\generated\single_anomaly\cicids2017_single_anomaly_predictions.csv `
  --output-path cicids2017_test_sample\generated\single_anomaly\cicids2017_single_anomaly_summary.csv

python repair_plan.py `
  --input-path cicids2017_test_sample\generated\single_anomaly\cicids2017_single_anomaly_summary.csv `
  --output-path cicids2017_test_sample\generated\single_anomaly\cicids2017_single_anomaly_repair_order.csv `
  --report-path cicids2017_test_sample\generated\single_anomaly\cicids2017_single_anomaly_repair_report.json
```

## 评估方式

对 `no_anomaly` 场景：

```powershell
python cicids2017_test_sample\evaluate_test_sample.py `
  --scenario no_anomaly `
  --predictions-path cicids2017_test_sample\generated\no_anomaly\cicids2017_no_anomaly_predictions.csv `
  --summary-path cicids2017_test_sample\generated\no_anomaly\cicids2017_no_anomaly_summary.csv
```

对 `single_anomaly` 场景：

```powershell
python cicids2017_test_sample\evaluate_test_sample.py `
  --scenario single_anomaly `
  --summary-path cicids2017_test_sample\generated\single_anomaly\cicids2017_single_anomaly_summary.csv
```

对 `multi_anomaly` 场景：

```powershell
python cicids2017_test_sample\evaluate_test_sample.py `
  --scenario multi_anomaly `
  --summary-path cicids2017_test_sample\generated\multi_anomaly\cicids2017_multi_anomaly_summary.csv
```
