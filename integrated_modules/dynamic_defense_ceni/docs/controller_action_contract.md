# 动态防御控制器动作接口协议

## 1. 文档目的

本文档定义动态防御引擎与外部控制器之间的 REST 接口协议。

外部控制器可以是：

- 本地模拟控制器
- SDN 控制器
- CENI 平台侧编排服务
- 后续部署在 CENI VM 上的防御控制模块

动态防御引擎负责完成攻击识别、策略匹配和防御动作生成，然后通过如下接口向控制器下发动作：

```http
POST /defense/action
```

控制器负责接收动作，并将动作翻译为具体网络操作，例如限速、隔离、流表下发、日志增强或检测模型切换。

---

## 2. 请求格式

每次请求只包含一个防御动作。

示例：

```json
{
  "strategy_id": "s_ddos_vote_rate_limit",
  "action": {
    "type": "rate_limit",
    "scope": "suspicious_src",
    "value": "20mbit"
  },
  "context": {
    "window_id": 2,
    "attack_type": "DDoS",
    "raw_matched_attack_type": "DDoS",
    "avg_match_score": 0.7485,
    "attack_present_by_label": true,
    "rows": 200
  }
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `strategy_id` | 当前触发的防御策略 ID |
| `action` | 本次需要执行的防御动作 |
| `action.type` | 动作类型，例如 `rate_limit`、`isolate_flow` |
| `context` | 当前检测窗口的上下文信息 |
| `context.window_id` | 当前窗口编号 |
| `context.attack_type` | 经过置信度门控后的有效攻击类型 |
| `context.raw_matched_attack_type` | 特征匹配器原始输出的攻击类型 |
| `context.avg_match_score` | 当前窗口平均匹配分数 |
| `context.attack_present_by_label` | 数据标签中是否存在攻击 |
| `context.rows` | 当前窗口包含的流量记录数量 |

---

## 3. 响应格式

控制器应返回 JSON 格式响应。

推荐格式：

```json
{
  "status": "OK",
  "message": "action accepted"
}
```

推荐状态值：

| 状态 | 含义 |
|---|---|
| `OK` | 动作已接收并执行或进入执行队列 |
| `IGNORED` | 动作已接收，但控制器选择忽略 |
| `FAILED` | 动作执行失败 |
| `UNSUPPORTED` | 控制器不支持该动作类型 |

---

## 4. 支持的动作类型

### 4.1 `monitor_only`

用于正常流量或低置信度攻击窗口。

该动作不改变网络转发行为，只记录当前窗口状态。

示例：

```json
{
  "type": "monitor_only",
  "scope": "baseline_window"
}
```

预期控制器行为：

- 不下发阻断或限速规则
- 记录窗口摘要
- 维持基线监控状态
- 可用于后续流量基线学习

适用场景：

```text
BENIGN 阶段
低置信度攻击判断
误报抑制场景
```

---

### 4.2 `log_enrich`

请求控制器或监控模块采集更多日志字段。

示例：

```json
{
  "type": "log_enrich",
  "fields": ["src_ip", "dst_ip", "dst_port", "flow_duration"]
}
```

预期控制器行为：

- 增强日志采集
- 记录更多流量上下文字段
- 为后续威胁特征分析提供更多信息

适用场景：

```text
PortScan 检测
UNKNOWN 威胁分析
BENIGN 窗口摘要记录
攻击溯源
```

---

### 4.3 `switch_model`

请求切换或启用指定检测模型。

示例：

```json
{
  "type": "switch_model",
  "target_model": "ensemble_lstm_ar_subspace"
}
```

预期控制器行为：

- 切换当前使用的检测模型
- 启用指定检测模型
- 或通知检测服务加载指定模型配置

当前原型中的模型类型包括：

| 模型名 | 含义 |
|---|---|
| `ensemble_lstm_ar_subspace` | LSTM、自回归、子空间模型集成检测 |
| `subspace_detector` | 子空间检测模型 |
| `lstm_sequence_detector` | 序列检测模型 |
| `web_feature_detector` | Web 攻击特征检测模型 |
| `protocol_signature_detector` | 协议特征检测模型 |

适用场景：

```text
DDoS 阶段切换到集成检测模型
PortScan 阶段切换到子空间检测模型
SSH/FTP 暴力破解阶段切换到序列检测模型
```

---

### 4.4 `raise_threshold`

请求调整某个检测指标的阈值。

示例：

```json
{
  "type": "raise_threshold",
  "metric": "flow_rate",
  "value": 0.85
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `metric` | 需要调整的指标 |
| `value` | 新阈值 |

预期控制器行为：

- 调整检测模块中的阈值参数
- 提高异常判定敏感度
- 或更新当前策略配置

示例指标：

| 指标 | 含义 |
|---|---|
| `flow_rate` | 流速相关指标 |
| `failed_login_rate` | 登录失败率 |
| `packet_rate` | 包速率 |
| `connection_rate` | 连接速率 |

适用场景：

```text
DDoS 检测增强
SSH/FTP 暴力破解检测增强
持续攻击下策略自适应优化
```

---

### 4.5 `rate_limit`

请求对可疑源、可疑流或特定链路进行限速。

示例：

```json
{
  "type": "rate_limit",
  "scope": "suspicious_src",
  "value": "20mbit"
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `scope` | 限速作用范围 |
| `value` | 限速值 |

可选作用范围：

| scope | 含义 |
|---|---|
| `suspicious_src` | 可疑源地址 |
| `suspicious_flow` | 可疑流 |
| `link` | 指定链路 |
| `service` | 指定服务 |

可能的 SDN/CENI 实现方式：

```text
OpenFlow meter
Linux tc
QoS 策略
链路带宽调整
VSR 转发策略调整
```

适用场景：

```text
DDoS
DoS
流量洪泛
高频连接请求
```

---

### 4.6 `isolate_flow`

请求隔离可疑流量。

示例：

```json
{
  "type": "isolate_flow",
  "scope": "suspicious_src_dst"
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `scope` | 隔离范围 |

可选作用范围：

| scope | 含义 |
|---|---|
| `suspicious_src` | 隔离可疑源 |
| `suspicious_dst` | 隔离可疑目标 |
| `suspicious_src_dst` | 隔离可疑源到目标的流 |
| `tls_heartbeat_flow` | 隔离 TLS heartbeat 异常流 |

可能的 SDN/CENI 实现方式：

```text
OpenFlow drop rule
ACL 阻断规则
安全组规则更新
重定向到隔离网络
重定向到清洗节点
```

适用场景：

```text
PortScan
SSH/FTP 暴力破解
Heartbleed
可疑横向移动流量
```

---

## 5. 当前本地验证结果

当前本地 REST 联调中，动态防御引擎通过 REST 接口向模拟控制器下发动作。

控制器接收路径：

```http
POST /defense/action
```

控制器日志文件：

```text
reports/controller_actions.jsonl
```

当前 REST 联调结果：

```text
动态防御窗口数：10
策略调整事件数：9
检测成功率：1.0
动态防御响应成功率：1.0
控制器接收动作数：26
```

策略动作统计：

| 策略 ID | 动作数 |
|---|---:|
| `s_benign_monitor` | 2 |
| `s_ddos_vote_rate_limit` | 6 |
| `s_portscan_isolate` | 6 |
| `s_bruteforce_ssh_ftp` | 12 |

动作类型统计：

| 动作类型 | 次数 |
|---|---:|
| `monitor_only` | 1 |
| `log_enrich` | 3 |
| `switch_model` | 8 |
| `raise_threshold` | 6 |
| `rate_limit` | 2 |
| `isolate_flow` | 6 |

---

## 6. CENI 对接方式

当前本地模拟控制器地址为：

```text
http://127.0.0.1:18080
```

在 CENI 部署时，只需要将动态防御程序中的控制器地址替换为 CENI 侧控制器地址：

```text
http://<controller-vm-ip>:<port>
```

动态防御程序执行方式示例：

```bash
python attack_defender.py \
  --input data/cicids2017_subset/cicids2017_scenario_ordered.csv \
  --build-templates \
  --window-size 200 \
  --limit 2000 \
  --adapter rest \
  --controller-endpoint http://<controller-vm-ip>:<port>
```

只要 CENI 侧控制器实现如下接口：

```http
POST /defense/action
```

动态防御引擎就不需要修改核心逻辑。

---

## 7. CENI/SDN 动作映射建议

| 动态防御动作 | SDN/CENI 侧可执行操作 |
|---|---|
| `monitor_only` | 不改变网络，仅记录监控窗口 |
| `log_enrich` | 开启增强日志采集，记录更多流量字段 |
| `switch_model` | 通知检测服务切换模型 |
| `raise_threshold` | 调整检测阈值或策略参数 |
| `rate_limit` | OpenFlow meter、Linux tc、QoS 或链路带宽调整 |
| `isolate_flow` | OpenFlow drop、ACL 阻断、重定向到隔离网络 |

---

## 8. 当前阶段结论

当前动态防御原型已经完成以下能力：

```text
1. 基于 CICIDS2017 流量特征进行威胁特征匹配；
2. 基于策略库完成防御策略选择；
3. 区分 detection_success 与 defense_success；
4. 支持 BENIGN 监控策略，降低正常流量误触发；
5. 支持 Actor-Critic-like 奖励更新机制；
6. 支持 REST 接口向外部控制器下发动作；
7. 已通过本地模拟控制器验证动作接收流程。
```

后续在 CENI 平台上部署时，需要完成：

```text
1. 创建控制器 VM；
2. 部署 REST 控制器；
3. 将 rate_limit、isolate_flow 等动作翻译为真实网络操作；
4. 将 controller_endpoint 指向 CENI 控制器地址；
5. 记录 CENI 平台侧动作执行日志。
```
