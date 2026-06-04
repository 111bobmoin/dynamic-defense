# 输出规范

本文档定义项目当前统一的输出布局。

## 根输出目录

项目正式保留两个输出根目录：

- `outputs/results/`
- `outputs/training/`

其中：

- `outputs/results/` 存放推理链路及后处理产物
- `outputs/training/` 存放模型训练产物

## 训练输出

训练运行目录：

```text
outputs/training/<timestamp>/
```

标准文件：

- `config.json`
- `history.json`
- `metrics.json`
- `selected_features.json`
- `label_mapping.json`
- `model.pt`

## 结果输出

单次运行目录：

```text
outputs/results/<timestamp>_<input_name>/
```

推荐阶段目录：

- `pcap_csv/`
- `inference/`
- `node_summary/`
- `repair_plan/`

### 1. `pcap_csv`

存放转换后的流量 CSV。

标准 CSV 文件名：

- `<input_name>_flows.csv`

推荐 manifest：

- `pcap_to_csv_manifest.json`

### 2. `inference`

存放流级预测结果。

标准 CSV 文件名：

- `<input_name>_predictions.csv`

推荐 manifest：

- `inference_manifest.json`

### 3. `node_summary`

存放节点级异常摘要。

标准 CSV 文件名：

- `<input_name>_node_summary.csv`

推荐 manifest：

- `node_summary_manifest.json`

### 4. `repair_plan`

存放修复顺序和修复代价结果。

标准 CSV 文件名：

- `<input_name>_repair_order.csv`

推荐 manifest：

- `repair_plan_manifest.json`

## 命名规则

所有阶段 CSV 统一使用：

```text
<input_name>_<stage_suffix>.csv
```

当前使用的后缀：

- `_flows`
- `_predictions`
- `_node_summary`
- `_repair_order`

阶段 JSON 元数据优先使用 `*_manifest.json`。

## 当前 `run_pipeline.py` 范围

当前统一流水线为：

```text
pcap -> pcap_csv -> inference -> node_summary -> repair_plan
```

`run_pipeline.py` 现在负责默认全链路，包括修复规划阶段。

## 内部临时目录

运行 `pcap_to_csv.py` 或 `run_pipeline.py` 时，可能产生：

```text
outputs/_internal/cicflowmeter_tmp/
```

这是临时工作目录，不属于正式输出布局。
