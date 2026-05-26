# Codex Memory

本文档用于记录这个项目的长期记忆，面向后续接手的 Codex / AI / 开发者。

原则：

- 只记录稳定事实、关键决策、长期约定、反复踩坑点。
- 临时调试细节不要堆在这里，阶段性过程放到提交记录或单独日志。
- 修改本文件时，优先追加，不要随意覆盖已有结论；若结论失效，标明“已废弃”和替代方案。

---

## 1. 项目身份

- 项目名称：`Dynamic Defense`
- 当前形态：CENI 多虚拟机 P4/BMv2 多模态网络控制与展示项目
- 核心职责：
  - 管理多交换机节点的数据面启动、恢复、重建
  - 支持主备路径切换
  - 支持多种网络模态切换
  - 提供遥测、抓包、应用验证和可视化大屏
  - 为外部防御系统提供状态输出和输入接入面

补充说明：

- 更完整的项目说明见 `Agent.md`。
- `codex.md` 的目标不是替代 `Agent.md`，而是沉淀“后续维护必须记住的事实”。

---

## 2. 稳定事实

### 2.1 部署角色

- `controller`：集中控制节点，不运行数据面交换机
- `host1`：业务客户端
- `server1`：业务服务端
- `s1` ~ `s9`：P4/BMv2 交换机节点

### 2.2 当前推荐运行方式

- 推荐使用 direct `simple_switch`
- 主要入口：
  - `multi_vm/start_switch_direct.py`
  - `multi_vm/rebuild_direct_switches.py`
- `p4-utils` 方式保留，但不是主维护路径

### 2.3 当前关键配置文件

- `multi_vm/nodes.yaml`
- `multi_vm/topology.yaml`
- `multi_vm/paths.yaml`

这些文件一旦变化，很多控制脚本、路径切换、状态采集和前端展示都会受影响。

---

## 3. 长期约定

### 3.1 文档约定

- `Agent.md`：偏项目全景说明和接手说明
- `codex.md`：偏长期记忆、维护共识、关键决策

### 3.2 修改约定

- 涉及节点、链路、端口映射的修改时：
  - 先更新 YAML 配置
  - 再检查相关控制脚本是否依赖旧字段
  - 最后确认状态采集和前端展示没有硬编码旧值

- 涉及外部防御系统接入时：
  - 优先新增 adapter
  - 不要破坏现有状态文件输出格式
  - 保持输入输出目录和字段尽量向后兼容

### 3.3 打包约定

- 发布包由 `multi_vm/package_release.py` 生成
- 不应把历史抓包、缓存和测试输出打进发布包

---

## 4. 关键决策记录

按时间倒序追加，推荐格式如下：

```md
### YYYY-MM-DD

- 决策：
- 原因：
- 影响范围：
- 替代方案：
- 后续动作：
```

当前已知重要决策：


### 2026-05-22（前后端联调与模型详情页）

- 决策：`dynamic_defense_dashboard` 现已按 `graph(1) + log(3) + muti3(3)` 共 7 个模型做实时评估聚合，不再只展示模型工件挂接状态。
- 原因：项目需求已从“模型已接入展示”升级为“模型真实跑起来，并能查看实时检测内容和状态”。
- 影响范围：`dynamic_defense_dashboard/server.py`、`static/app.js`、`static/model-detail.html`、`static/styles.css`，以及 `log/TEST_main/DLSTM/predict_lstm.py` 的 CPU 兼容加载方式。
- 替代方案：保留原先的占位回退逻辑，但那无法满足联调测试，也无法支撑逐模型详情查看。
- 后续动作：后续若替换数据集、模型路径或详情页展示字段，必须同步检查 `/api/integration` 和 `/api/model-detail` 两条接口的兼容性。

### 2026-05-22

- 决策：新增 `codex.md` 作为项目长期记忆文件。
- 原因：避免关键维护事实只存在于对话、临时上下文或单次交接说明中。
- 影响范围：后续 AI/开发者在做结构性修改前，应先查看本文件。
- 替代方案：只维护 `Agent.md`，但那会让“稳定事实”和“全景说明”混在一起，长期可维护性较差。
- 后续动作：后续每次出现稳定约定、关键坑点、架构取舍时，都应更新这里。

---

## 5. 容易遗忘但重要的点

按主题追加，避免写成流水账。

### 5.1 运行形态

- 当前主维护目标是多 VM 直连部署，不是单机模拟优先。
- controller 是控制中心，不应把它误认为数据面交换机节点。

### 5.2 防御系统集成边界

- 目前项目负责“集成框架和状态对接面”，不负责四个外部安全系统内部算法实现。
- 外部系统状态会通过固定输入文件目录回写，再由状态采集逻辑合并。

### 5.3 配置风险

- `nodes.yaml`、`topology.yaml`、`paths.yaml` 之间存在强关联。
- 任何一个文件改名、改字段、改节点 ID，都可能引发连锁问题。


### 5.4 仪表盘与模型联调约定

- 未知威胁页当前的模型分区是稳定约定：`graph` 1 个模型、`log` 3 个模型、`muti3` 3 个模型，共 7 个模型位。
- 每个模型卡片都应可点击进入详情页；详情页依赖 `model-detail.html` 和 `/api/model-detail`，不要退回成只显示汇总分数的静态卡片。
- `/api/integration` 负责汇总状态，`/api/model-detail` 负责单模型实时样本、指标、预测分布和运行状态；新增模型时要同时接这两层。
- 目前服务对外监听 `0.0.0.0:8099`，需要兼容本机访问和主机访问，不能只绑定 `127.0.0.1`。

---

## 6. 常见坑位

后续遇到可复现问题时，按下面格式追加：

```md
### 坑位标题

- 现象：
- 根因：
- 规避方式：
- 相关文件：
```

当前预置坑位：


### Python 模块路径冲突会导致联调接口失效

- 现象：`/api/integration` 或 `/api/model-detail` 在运行 `muti3` 相关模型时出现 `No module named 'utils.model'; 'utils' is not a package`。
- 根因：向 `sys.path` 注入 `log/TEST_main/Kmeans` 后，顶层 `utils.py` 会遮蔽 `muti3/utils` 包，导致多模态模型导入失败。
- 规避方式：后端动态导入时只加入必要上级目录，避免把带有同名顶层模块的子目录直接塞进 `sys.path`。
- 相关文件：`dynamic_defense_dashboard/server.py`、`log/TEST_main/Kmeans/utils.py`、`muti3/utils/`

### Graph 模型全量加载会触发内存爆炸

- 现象：图模型联调时直接加载 `graph/processed_bgl/X_val.npz` 并转成稠密矩阵，会触发几十 GB 级内存申请并导致接口失败。
- 根因：验证特征矩阵维度过大，不能在联调接口里无采样地 `toarray()`。
- 规避方式：当前后端只对图验证集做采样评估，并同步裁剪边集合；若以后改回全量评估，需要先重做推理管线。
- 相关文件：`dynamic_defense_dashboard/server.py`、`graph/processed_bgl/X_val.npz`

### DLSTM 权重默认按 CUDA 保存，CPU 环境会直接报错

- 现象：无 GPU 环境下运行日志 `DLSTM` 评估时，模型加载阶段失败。
- 根因：权重文件按 CUDA 设备保存，原始加载代码没有指定 `map_location`。
- 规避方式：保持 `torch.load(..., map_location='cpu')` 的兼容写法；若后续改动该文件，不要回退这个行为。
- 相关文件：`log/TEST_main/DLSTM/predict_lstm.py`

### CENI 重启后地址或网口状态漂移

- 现象：节点管理地址或数据地址失效，导致控制脚本和路径验证异常。
- 根因：实验环境重启后网口配置可能未按预期恢复。
- 规避方式：优先对照 `multi_vm/nodes.yaml` 和 `multi_vm/topology.yaml` 恢复。
- 相关文件：`multi_vm/nodes.yaml`、`multi_vm/topology.yaml`

---

## 7. 当前关注项

这里写“接下来一段时间内仍然重要”的事项，完成后转移到决策记录或删除。

- 确保 `codex.md` 与 `Agent.md` 分工清晰，避免两份文档长期重复复制。
- 后续若外部防御系统真实接入，需要把正式接口约束沉淀到本文件。

---

## 8. 建议维护方式

建议在以下场景更新本文件：

- 新增或废弃关键脚本入口
- 修改部署形态或默认启动方式
- 调整状态文件协议或目录
- 发现反复出现的环境问题
- 做出会长期影响维护方式的架构决策

不建议写入本文件的内容：

- 一次性调试命令
- 临时实验结果
- 可直接从代码推导出的低价值细节

## 2026-05-25 `muti3` 评估口径
### 结论
- `muti3` 三模型的准确率展示必须对齐原始脚本：`eval_origin_lstm.py`、`eval_origin_subplace.py`、`eval_origin_ag.py`。
- 原始脚本固定评估 `muti3/Dataset/validata.csv`，不能用 `validata_sample.csv` 代替。

### 稳定规则
- `dynamic_defense_dashboard/server.py` 中 `evaluate_multi3_detail()` 必须沿用这套口径：
  - 数据读取：`utils.DataProcessing.load_and_preprocess_data2`
  - 数据集：`muti3/Dataset/validata.csv`
  - DataLoader：`batch_size=64, shuffle=True`
  - LSTM：`x_batch.unsqueeze(1)`
  - Subspace Clustering / Autoregressive：直接使用 `x_batch`
- 后端改完后必须重启 `dynamic_defense_dashboard/server.py`，前端详情页才会拿到新指标。

### 排障优先级
- 如果 `muti3` 准确率明显偏低，先查是否误回到了 `validata_sample.csv` 口径。
- `validata_sample.csv` 几乎没有 `BENIGN`，标签分布与 `validata.csv` 不同，直接用于展示原始准确率会失真。
- 不要先改前端展示；先确认后端评估口径是否仍与原始 `eval_origin_*` 脚本一致。

### 性能注意
- `muti3` 首次全量评估会较慢，这是 `validata.csv` 体量导致的正常现象。
- 不要再为详情页样本预览全量读取 20 多万行 CSV；这会显著放大首轮延迟，但不会提升指标可信度。

## 2026-05-25 会话压缩记忆
### 系统现状
- `dynamic_defense_dashboard` 已完成前后端联调，当前核心展示围绕 7 个在线模型：`graph(1) + log(3) + muti3(3)`。
- 后端统一由 `dynamic_defense_dashboard/server.py` 提供聚合与详情接口，前端主要逻辑在 `static/app.js`，整体样式在 `static/styles.css`。
- 服务对外监听 `0.0.0.0:8099`，用于本机和主机访问。

### 页面结构约定
- 首页采用企业监控后台布局：`KPI -> 拓扑图 -> 模块入口/状态/事件`，拓扑图是视觉中心。
- 未知威胁页采用：上方拓扑总览，下方三类异构组件。
- 三类组件固定顺序：
  - 攻击数据特征检测异构组件
  - 攻击逻辑检测异构组件
  - 行为图结构检测异构组件
- 未知威胁页已去掉“数据集路径”输入和伪实时数据集摘要，页面语义统一为实时检测流。
- 组件卡和详情页支持点击进入实时监控页；详情页为双列大模型卡 + 大折线图 + 最近预测。

### 模型命名约定
- `muti3` 组件中的三个模型名称统一为：`LSTM`、`Subspace Clustering`、`Autoregressive`。
- 不再在 `muti3` 详情页使用“流量检测 / 日志检测 / 图检测”这类旧命名。
- 若前端改名后页面仍显示旧字样，先检查后端原始返回字段是否仍是旧名称，再决定是否只改前端映射。

### 实时监控页约定
- 详情页折线图已改为专业监控风格：深蓝黑底、清晰坐标轴、轻网格、阈值线、异常点、紧凑图例。
- 图表尺寸必须由容器动态测量，不能写死 `width/height/viewBox`。
- 任何重叠问题优先通过正常文档流和间距解决，不使用负 margin、绝对定位硬压。
- 详情页中容易误伤的文本位有 3 处：模型标题、模型副标题、状态文案；需要统一处理时要三处一起看。

### `muti3` 评估口径
- `muti3` 三模型的准确率展示必须对齐原始脚本：`eval_origin_lstm.py`、`eval_origin_subplace.py`、`eval_origin_ag.py`。
- 原始脚本固定评估 `muti3/Dataset/validata.csv`，不能用 `validata_sample.csv` 替代。
- 当前后端 `evaluate_multi3_detail()` 应保持这套口径：
  - 数据读取：`utils.DataProcessing.load_and_preprocess_data2`
  - 数据集：`muti3/Dataset/validata.csv`
  - DataLoader：`batch_size=64, shuffle=True`
  - LSTM：`x_batch.unsqueeze(1)`
  - Subspace Clustering / Autoregressive：直接使用 `x_batch`
- 如果 `muti3` 准确率异常低，先检查是否误回到了 `validata_sample.csv` 口径，而不是先改前端展示。

### 性能与稳定性
- `muti3` 首次全量评估会慢，这是 `validata.csv` 体量导致的正常现象；后续应尽量依赖缓存。
- 不要为详情页样本预览全量读取大 CSV，这会显著拉长首轮响应时间。
- `graph` 评估要避免全量稠密化导致内存爆炸；采样策略是当前稳定方案。
- `DLSTM` 需用 CPU 兼容方式加载权重，例如 `map_location=torch.device('cpu')`。
- 之前踩过的路径冲突问题与 `sys.path` / `utils` 模块覆盖有关，再次改导入路径时要优先自查。

### 维护优先级
- 页面展示异常时，先区分：前端文案映射问题、后端原始字段问题、还是服务未重启导致的旧缓存/旧代码。
- 改后端展示字段后，通常需要重启 `dynamic_defense_dashboard/server.py` 才能让页面看到新值。
- 如果用户反馈“还是没改”，优先核对接口返回 JSON，而不是只看前端代码。



## 2026-05-25 `validata.csv` 持久化缓存与 dashboard 非阻塞约定
### 结论
- `dynamic_defense_dashboard/server.py` 现已为全量 `muti3/Dataset/validata.csv` 评估结果增加磁盘持久化缓存，缓存目录固定为 `dynamic_defense_dashboard/.runtime_cache/`。
- 首页 `dashboard`、`/api/integration`、`/api/detection` 必须优先返回“已落盘缓存 + 未完成模型 waiting 占位”，不能再因为某个全量模型未跑完而整页阻塞。
- 默认启动时不再自动预热全量 `validata.csv`；只有显式设置环境变量 `DYNAMIC_DEFENSE_WARMUP=1` 时才允许后台预热。

### 稳定规则
- 需要长期复用的模型评估结果应落盘为 JSON，当前至少包括：
  - `graph_gcn_detail.json`
  - `log_gru_detail.json`
  - `log_kmeans_detail.json`
  - `log_dlstm_detail.json`
  - `muti3_traffic_detail.json`
  - `muti3_log_detail.json`
  - `muti3_graph_detail.json`
- 磁盘缓存必须带输入文件指纹；模型文件或数据文件时间戳变更后，应自动失效并重算，不能盲目复用旧结果。
- 对默认数据集 `validata.csv`，`build_dashboard_payload()` / `build_integration_payload()` / `build_multi3_section()` 必须支持 `allow_partial=True` 的非阻塞返回路径。

### 排障优先级
- 如果用户反馈“dashboard 不显示数据”，先查是否误开启了启动预热，导致同进程 CPU 被全量评估吃满。
- 如果 `/api/dashboard` 超时，先查 `.runtime_cache/` 中已有几个 `muti3_*_detail.json`，不要先怀疑前端。
- 如果页面能显示但状态是 `waiting`，说明页面逻辑正常，问题只是剩余模型缓存未完成。

### 性能注意
- `validata.csv` 全量 `muti3` 首轮评估可能非常慢，尤其是 `LSTM` 和 `Subspace Clustering`；这是正常现象。
- `waiting` 占位数据不能反复重新扫描 20 多万行 `validata.csv` 去做摘要，否则会把“非阻塞返回”重新拖慢。
- 平时对外启动建议直接使用默认模式；需要补齐全量缓存时，再显式使用 `DYNAMIC_DEFENSE_WARMUP=1` 启动一次。
