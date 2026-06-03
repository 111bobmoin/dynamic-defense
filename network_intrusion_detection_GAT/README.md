# network_intrusion_detection_GAT

基于 `MachineLearningCVE` / CICIDS2017 风格流量特征的图注意力网络入侵检测项目。

项目当前覆盖三条主线：

- 用 `MachineLearningCVE` 风格 CSV 训练 `GAT / GATv2` 模型
- 用 `pcap / pcapng` 跑完整链路：`pcap_to_csv -> inference -> node_summary -> repair_plan`
- 生成三类标准化测试样本，并对节点级误报、异常节点排序和多异常召回做自动评估

当前推荐统一入口：

- 真实流量全流程：`run_pipeline.py`
- 标准样本生成：`cicids2017_test_sample/generate_test_samples.py`
- 样本全流程批量实验：`run_regenerated_experiments.py`

## 1. 项目结构

```text
.
+-- configs/
|   `-- default_multiclass_gatv2.json
+-- cicids2017_test_sample/
|   +-- generate_test_samples.py
|   +-- evaluate_test_sample.py
|   `-- README.md
+-- in_flow/
|   `-- 示例 pcap / pcapng
+-- outputs/
|   +-- training/
|   +-- results/
|   +-- experiments/
|   `-- _internal/
+-- src/
|   +-- cicflow_adapter.py
|   +-- cicids2017_test_sample.py
|   +-- data.py
|   +-- graph.py
|   +-- losses.py
|   +-- metrics.py
|   +-- model.py
|   +-- output_layout.py
|   +-- pso.py
|   +-- repair.py
|   `-- trainer.py
+-- train.py
+-- pcap_to_csv.py
+-- inference.py
+-- node_summary.py
+-- repair_plan.py
+-- run_pipeline.py
+-- run_regenerated_experiments.py
+-- OUTPUT_SPEC.md
`-- requirements.txt
```

## 2. 关键脚本职责

- `train.py`
  - 训练 `GAT / GATv2` 模型
  - 支持二分类和多分类
  - 支持可选 `PSO` 特征选择
- `pcap_to_csv.py`
  - 用 `cicflowmeter` 将 `pcap / pcapng` 转成标准 CSV
  - 可保留 `src_ip / dst_ip / src_port / protocol / timestamp` 等元数据
- `inference.py`
  - 对标准 CSV 执行流级推理
  - 输出 `predicted_label / anomaly_score / is_anomaly`
- `node_summary.py`
  - 将流级结果聚合成节点级异常摘要
  - 输出节点角色，如 `suspected_attacker / suspected_victim / suspected_compromised_host / uncertain`
- `repair_plan.py`
  - 基于节点摘要生成修复优先级和最小代价修复顺序
- `run_pipeline.py`
  - 串起 `pcap_to_csv -> inference -> node_summary -> repair_plan`
  - 把一次运行的全部产物集中到同一个结果目录
- `cicids2017_test_sample/generate_test_samples.py`
  - 基于真实 `MachineLearningCVE` 数据构造三类标准测试样本
- `cicids2017_test_sample/evaluate_test_sample.py`
  - 对三类测试样本的全流程输出做评估
- `run_regenerated_experiments.py`
  - 每轮重新生成新样本
  - 自动执行 `生成样本 -> inference -> node_summary -> repair_plan -> 评估`
  - 汇总多轮实验结果

## 3. 环境与依赖

安装依赖：

```powershell
pip install -r requirements.txt
```

`requirements.txt` 当前声明：

- `torch`
- `torch-geometric`
- `pandas`
- `numpy`
- `cicflowmeter`

注意：

- 只做 CSV 推理和样本评估时，不依赖原始 `pcap`
- 处理 `pcap / pcapng` 时，需要本地 `cicflowmeter` 可用
- 默认数据集路径是 `E:\dataset\MachineLearningCVE`

## 4. 当前默认训练配置

默认配置文件：

- `configs/default_multiclass_gatv2.json`

当前默认值：

- `task = multiclass`
- `model_name = gatv2`
- `use_residual = true`
- `graph_strategy = knn`
- `graph_metric = cosine`
- `loss_name = cross_entropy`
- `batch_size = 192`
- `epochs = 36`
- `patience = 10`
- `hidden_dim = 128`
- `heads = 4`
- `dropout = 0.1`
- `learning_rate = 0.0005`
- `weight_decay = 0.00001`
- `device = cpu`
- `use_pso = false`
- `max_rows_per_class = 10000`
- `dataset_dir = E:\dataset\MachineLearningCVE`

训练命令：

```powershell
python train.py --config configs\default_multiclass_gatv2.json
```

训练产物输出到：

```text
outputs/training/<timestamp>/
```

标准文件包括：

- `config.json`
- `history.json`
- `metrics.json`
- `selected_features.json`
- `label_mapping.json`
- `model.pt`

## 5. 当前推荐模型

当前本地推荐模型：

- `outputs/training/20260511_182846/model.pt`

对应训练摘要：

- `selected_feature_count = 68`
- `best_epoch = 14`
- `val_accuracy = 0.9603328010757206`
- `val_macro_f1 = 0.8144686434642843`
- `test_accuracy = 0.9638594721801984`
- `test_macro_f1 = 0.8134502315523816`

如果不显式传 `--model-path`：

- `inference.py` 会自动扫描 `outputs/` 下最新的 `model.pt`
- `run_pipeline.py` 也会自动扫描 `outputs/` 下最新的 `model.pt`

如果你本地同时保留多个训练目录，建议显式传模型路径，避免误选。

## 6. 当前默认后处理参数

这组参数已经作为默认值写入主流程脚本：

- `decision_threshold = 0.55`
- `anomaly_threshold = 0.65`
- `role_threshold = 0.65`
- `min_role_total_flows = 60`
- `min_role_anomaly_ratio = 0.50`
- `min_role_high_conf_flows = 8`
- `min_directional_high_conf_flows = 4`
- `core_top_ratio = 0.30`

其中：

- `decision_threshold` 控制流级 `is_anomaly`
- `anomaly_threshold` 控制高置信异常流统计
- `role_threshold` 控制节点角色赋值
- `core_top_ratio` 控制修复计划中核心节点比例

## 7. 快速开始

### 7.1 跑真实流量完整链路

推荐直接使用统一入口：

```powershell
python run_pipeline.py `
  --input-path in_flow\20261118.pcap `
  --model-path outputs\training\20260511_182846\model.pt
```

也可以对 `pcapng` 跑：

```powershell
python run_pipeline.py `
  --input-path in_flow\2026954.pcapng `
  --model-path outputs\training\20260511_182846\model.pt
```

默认输出目录：

```text
outputs/results/<timestamp>_<input_name>/
```

目录结构：

```text
outputs/results/<timestamp>_<input_name>/
+-- pcap_csv/
+-- inference/
+-- node_summary/
+-- repair_plan/
`-- manifest.json
```

### 7.2 分阶段运行

1. `pcap -> CSV`

```powershell
python pcap_to_csv.py `
  --input-path in_flow\20261118.pcap `
  --keep-metadata `
  --result-dir outputs\results\manual_run
```

2. `CSV -> 流级推理`

```powershell
python inference.py `
  --model-path outputs\training\20260511_182846\model.pt `
  --input-path outputs\results\manual_run\pcap_csv\20261118_flows.csv `
  --result-dir outputs\results\manual_run
```

3. `推理结果 -> 节点摘要`

```powershell
python node_summary.py `
  --input-path outputs\results\manual_run\inference\20261118_predictions.csv `
  --result-dir outputs\results\manual_run
```

4. `节点摘要 -> 修复计划`

```powershell
python repair_plan.py `
  --input-path outputs\results\manual_run\node_summary\20261118_node_summary.csv `
  --result-dir outputs\results\manual_run
```

### 7.3 直接对标准 CSV 推理

如果输入已经是 `MachineLearningCVE` 风格 CSV，可以跳过 `pcap_to_csv.py`：

```powershell
python inference.py `
  --model-path outputs\training\20260511_182846\model.pt `
  --input-path your_input.csv `
  --output-path your_predictions.csv
```

如果后续还要做节点级汇总，输入 CSV 需要保留这些字段：

- `src_ip`
- `dst_ip`
- `timestamp`

而 `node_summary.py` 依赖的推理结果最少需要：

- `src_ip`
- `dst_ip`
- `timestamp`
- `predicted_label`
- `anomaly_score`
- `is_anomaly`

## 8. 标准测试样本

统一样本入口在：

- `cicids2017_test_sample/`

三种标准场景：

- `no_anomaly`
  - 无异常节点
  - 用于测节点级误报
- `single_anomaly`
  - 单异常节点
  - 用于测异常节点排序是否足够靠前
- `multi_anomaly`
  - 多异常节点
  - 用于测多异常召回和修复序列覆盖率

生成全部场景：

```powershell
python cicids2017_test_sample\generate_test_samples.py `
  --dataset-dir E:\dataset\MachineLearningCVE
```

只生成某一场景：

```powershell
python cicids2017_test_sample\generate_test_samples.py `
  --dataset-dir E:\dataset\MachineLearningCVE `
  --scenario single_anomaly
```

默认输出目录：

```text
cicids2017_test_sample/generated/
```

样本生成后，可以直接送入主流程，不需要再经过 `pcap_to_csv.py`。

## 9. 样本评估

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

## 10. 批量重生成实验

`run_regenerated_experiments.py` 会在每一轮里执行：

```text
重新生成新样本 -> inference -> node_summary -> repair_plan -> 评估
```

示例：

```powershell
python run_regenerated_experiments.py `
  --runs 5 `
  --dataset-dir E:\dataset\MachineLearningCVE `
  --model-path outputs\training\20260511_182846\model.pt `
  --output-root outputs\experiments\regenerated_samples
```

如果要让每轮三场景节点数都在 `50-100` 随机变化：

```powershell
python run_regenerated_experiments.py `
  --runs 5 `
  --dataset-dir E:\dataset\MachineLearningCVE `
  --model-path outputs\training\20260511_182846\model.pt `
  --output-root outputs\experiments\regenerated_samples_large `
  --randomize-large-node-counts `
  --min-large-nodes 50 `
  --max-large-nodes 100
```

实验输出：

```text
outputs/experiments/<experiment_name>/
+-- experiment_manifest.json
+-- experiment_summary.csv
`-- run_XX_seed_<seed>/
    +-- run_summary.json
    `-- samples/
```

当前脚本内部判定的通过标准是：

- `no_anomaly`：节点级 `node_false_positive_rate_any_window < 0.05`
- `single_anomaly`：良性节点 `benign_fp < 0.05`
- `single_anomaly`：真实异常节点位于修复序列前 `20%`
- `multi_anomaly`：良性节点 `benign_fp < 0.05`
- `multi_anomaly`：真实异常节点在修复序列中的召回率 `>= 0.80`

## 11. 输出约定

正式输出根目录：

- `outputs/training/`
- `outputs/results/`
- `outputs/experiments/`

内部临时目录：

- `outputs/_internal/cicflowmeter_tmp/`

其中：

- `outputs/training/` 存放训练产物
- `outputs/results/` 存放单次真实流量或手工链路运行结果
- `outputs/experiments/` 存放多轮再生成样本实验
- `outputs/_internal/` 不是正式结果目录

更细的输出命名约定见：

- `OUTPUT_SPEC.md`

## 12. `src/` 核心模块

- `src/data.py`
  - 数据加载、清洗、标签映射和训练集划分
- `src/model.py`
  - `IntrusionGAT` 模型定义
- `src/graph.py`
  - 基于特征构图，当前主用 `kNN`
- `src/trainer.py`
  - 训练、验证、测试与特征子集搜索
- `src/pso.py`
  - 可选二进制 `PSO` 特征选择
- `src/repair.py`
  - 修复代价计算与修复顺序生成
- `src/cicflow_adapter.py`
  - 把 `cicflowmeter` 输出适配到训练 / 推理需要的字段格式
- `src/output_layout.py`
  - 统一输出目录与默认命名
- `src/cicids2017_test_sample.py`
  - 三场景样本生成与评估核心实现

## 13. 使用建议

- 真实流量优先用 `run_pipeline.py`，不要手工拼多条命令，除非你在调某个阶段
- 如果只验证模型推理，不需要 `pcap_to_csv.py`
- 如果要做节点级分析，输入必须保留 `src_ip / dst_ip / timestamp`
- 如果本地存在多个 `model.pt`，显式传 `--model-path` 更稳
- 生成样本和实验结果都属于可再生产物，清理后可随时重跑

## 14. 相关文档

- 流程测试样本说明：`cicids2017_test_sample/README.md`
- 输出布局说明：`OUTPUT_SPEC.md`
- 项目交接文档：`HANDOVER.md`
