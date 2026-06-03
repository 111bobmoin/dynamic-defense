# 项目交接文档

这份文档给下一位接手的人用。目标不是记录全部历史，而是让人打开仓库后能立刻知道：

- 项目现在能做什么
- 应该从哪个入口开始
- 哪些模型和结果当前可直接用
- 继续实验时要注意什么

## 1. 当前状态

- 项目根目录：`C:\Users\Administrator\Desktop\杂\论文\network_intrusion_detection_GAT`
- 当前仓库不是 git 仓库
- 当前代码主线已经统一，不再使用早期拆分的旧样本目录
- 当前统一测试样本入口：`cicids2017_test_sample/`
- 当前推荐真实流量入口：`run_pipeline.py`

项目当前覆盖三条主线：

- 训练：基于 `MachineLearningCVE` 风格 CSV 训练 `GAT / GATv2`
- 真实流量：`pcap / pcapng -> CSV -> inference -> node_summary -> repair_plan`
- 样本实验：重新生成三场景测试样本并自动评估

## 2. 先看哪些文件

接手时优先看这些文件：

- `README.md`
- `HANDOVER.md`
- `configs/default_multiclass_gatv2.json`
- `run_pipeline.py`
- `run_regenerated_experiments.py`
- `cicids2017_test_sample/README.md`
- `OUTPUT_SPEC.md`

如果要深入代码，再看：

- `src/data.py`
- `src/model.py`
- `src/graph.py`
- `src/trainer.py`
- `src/cicflow_adapter.py`
- `src/cicids2017_test_sample.py`
- `src/repair.py`

## 3. 关键目录与入口

关键目录：

- `configs/`
- `cicids2017_test_sample/`
- `in_flow/`
- `outputs/training/`
- `outputs/results/`
- `outputs/experiments/`
- `src/`

关键脚本：

- `train.py`
- `pcap_to_csv.py`
- `inference.py`
- `node_summary.py`
- `repair_plan.py`
- `run_pipeline.py`
- `run_regenerated_experiments.py`

入口建议：

- 训练模型：`train.py`
- 跑真实流量全流程：`run_pipeline.py`
- 跑三场景批量实验：`run_regenerated_experiments.py`

## 4. 当前推荐模型

当前最推荐直接使用的模型：

- `outputs/training/20260511_182846/model.pt`

对应摘要：

- `selected_feature_count = 68`
- `best_epoch = 14`
- `val_accuracy = 0.9603328010757206`
- `val_macro_f1 = 0.8144686434642843`
- `test_accuracy = 0.9638594721801984`
- `test_macro_f1 = 0.8134502315523816`

另一个较早训练目录也还在：

- `outputs/training/20260507_200244/`

但后续默认优先使用 `20260511_182846`。

## 5. 当前默认参数

训练默认配置文件：

- `configs/default_multiclass_gatv2.json`

当前默认后处理参数已经固化到主流程脚本：

- `decision_threshold = 0.55`
- `anomaly_threshold = 0.65`
- `role_threshold = 0.65`
- `min_role_total_flows = 60`
- `min_role_anomaly_ratio = 0.50`
- `min_role_high_conf_flows = 8`
- `min_directional_high_conf_flows = 4`
- `core_top_ratio = 0.30`

这些默认值已经接入：

- `inference.py`
- `node_summary.py`
- `run_pipeline.py`
- `run_regenerated_experiments.py`

## 6. 当前保留的输入与产物

### 6.1 输入

当前 `in_flow/` 下保留的真实输入：

- `in_flow/20261118.pcap`
- `in_flow/2026954.pcapng`

默认数据集路径：

- `E:\dataset\MachineLearningCVE`

### 6.2 训练产物

当前保留：

- `outputs/training/20260507_200244/`
- `outputs/training/20260511_182846/`

### 6.3 实验产物

此前大量中间样本和旧结果已经清理掉了。

当前明确保留的实验目录：

- `outputs/experiments/one_shot_20260512_163107/`

这是一次完整的三场景实验结果，执行的是：

- 重新生成三场景样本
- `inference -> node_summary -> repair_plan -> evaluate`

其中关键文件：

- `outputs/experiments/one_shot_20260512_163107/experiment_summary.csv`
- `outputs/experiments/one_shot_20260512_163107/experiment_manifest.json`
- `outputs/experiments/one_shot_20260512_163107/run_01_seed_2026051201/run_summary.json`

## 7. 最近一次已确认结果

最近一次完整三场景实验目录：

- `outputs/experiments/one_shot_20260512_163107/`

结果摘要：

- `no_anomaly`
  - 节点级 `node_false_positive_rate_any_window = 0.0`
  - 流级 `flow_false_positive_rate = 0.16805555555555557`
- `single_anomaly`
  - `overall_accuracy = 1.0`
  - `benign_fp = 0.0`
  - `true_anomaly_rank = 1`
  - `true_anomaly_in_top_20pct = true`
- `multi_anomaly`
  - `overall_accuracy = 1.0`
  - `benign_fp = 0.0`
  - `true_anomaly_recall_in_repair = 1.0`
- 总结
  - `all_passed = true`

这说明当前版本已经满足节点级目标，但不代表流级误报为 0。

## 8. 当前测试标准

项目近期实验主要按下面的节点级目标看：

- `no_anomaly`
  - 无异常节点时，节点级误报 `< 0.05`
- `single_anomaly`
  - 一异常节点时，良性节点误报 `< 0.05`
  - 真异常节点排序在前 `20%`
- `multi_anomaly`
  - 多异常节点时，良性节点误报 `< 0.05`
  - 至少 `80%` 的真异常节点出现在预测修复序列中

`run_regenerated_experiments.py` 当前内部也是按这组逻辑判断 `all_passed`。

## 9. 已知事实与注意事项

- `node_summary.py` 强依赖元数据列
  - 至少需要 `src_ip / dst_ip / timestamp`
- `run_pipeline.py` 内部固定使用 `keep_metadata=True`
- `inference.py` 和 `run_pipeline.py` 都支持自动选择最新 `model.pt`
- 但如果 `outputs/training/` 下有多个训练目录，显式传 `--model-path` 更稳
- 当前 `outputs/_internal/cicflowmeter_tmp/` 是内部临时目录，不是正式结果
- 样本目录和实验结果属于可再生产物，必要时可以删掉后重跑
- 终端里直接 `Get-Content` 读中文文件可能乱码，但文件本身是正常 UTF-8

## 10. 推荐恢复命令

### 10.1 跑真实流量全流程

```powershell
python run_pipeline.py `
  --input-path in_flow\20261118.pcap `
  --model-path outputs\training\20260511_182846\model.pt
```

或：

```powershell
python run_pipeline.py `
  --input-path in_flow\2026954.pcapng `
  --model-path outputs\training\20260511_182846\model.pt
```

### 10.2 跑一轮三场景完整实验

```powershell
python run_regenerated_experiments.py `
  --runs 1 `
  --dataset-dir E:\dataset\MachineLearningCVE `
  --model-path outputs\training\20260511_182846\model.pt `
  --output-root outputs\experiments\manual_one_shot
```

### 10.3 跑多轮大节点实验

```powershell
python run_regenerated_experiments.py `
  --runs 5 `
  --dataset-dir E:\dataset\MachineLearningCVE `
  --model-path outputs\training\20260511_182846\model.pt `
  --output-root outputs\experiments\manual_large_batch `
  --randomize-large-node-counts `
  --min-large-nodes 50 `
  --max-large-nodes 100
```

## 11. 如果重新开新会话

建议直接告诉下一位助手这些信息：

1. 项目目录是 `C:\Users\Administrator\Desktop\杂\论文\network_intrusion_detection_GAT`
2. 先读 `README.md` 和 `HANDOVER.md`
3. 推荐真实流量入口是 `run_pipeline.py`
4. 推荐实验入口是 `run_regenerated_experiments.py`
5. 当前稳定模型是 `outputs/training/20260511_182846/model.pt`
6. 默认数据集目录是 `E:\dataset\MachineLearningCVE`
7. `node_summary` 依赖元数据列
8. 仓库不是 git 仓库

可以直接复制这段提示词：

```text
项目目录：
C:\Users\Administrator\Desktop\杂\论文\network_intrusion_detection_GAT

请先阅读 README.md 和 HANDOVER.md。

当前推荐真实流量入口：
run_pipeline.py

当前推荐实验入口：
run_regenerated_experiments.py

当前稳定模型：
outputs/training/20260511_182846/model.pt

默认数据集：
E:\dataset\MachineLearningCVE

注意：
1. node_summary 依赖元数据列
2. 当前仓库不是 git 仓库
3. 如有多个训练目录，优先显式传 --model-path
4. 继续实验时，优先看 outputs/experiments/one_shot_20260512_163107/
```

## 12. 后续最可能继续做的事

- 继续追加更多轮新样本实验，验证稳定性
- 分析为什么流级误报不为 0，但节点级误报可以为 0
- 对新的 `pcap / pcapng` 跑完整链路
- 重新训练模型并对比新旧指标
- 继续整理文档、输出结构和论文实验表格
