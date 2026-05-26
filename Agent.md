# Agent.md

本文面向后续接手本项目的 AI 或开发者，说明当前 CENI 多虚拟机 P4/BMv2 多模态网络项目已经完成了什么、目录里有什么、运行时依赖哪些脚本，以及维护时需要注意的边界。

## 1. 项目定位

本项目是在 CENI 类多虚拟机网络实验环境中运行的多模态网络控制与展示系统。每个网络节点是一台虚拟机，交换机节点运行 P4/BMv2 数据面程序，controller 节点通过 SSH 统一控制其他节点。

当前已经实现的核心能力：

- 在 controller 上集中启动、停止、恢复交换机数据面。
- 支持主路径 `host1 -> s1 -> s3 -> s4 -> s7 -> server1`。
- 支持备用路径 `host1 -> s1 -> s5 -> s3 -> s4 -> s7 -> server1`。
- 支持扩展备用路径经过 `s2`、`s6`、`s8`、`s9` 等节点，当前不包含 `server2`。
- 支持按整条路径或单段链路切换传输模态。
- 支持 `ipv4`、`ipv6`、`mpls`、`geonet`、`scion_v2` 五类模态。
- 支持 host1 到 server1 的 ICMP、UDP、TCP、HTTP 应用验证。
- 支持 server1 上部署 DeepSeek 代理服务，host1 通过数据面访问该服务。
- 支持从 controller 发起端口抓包，并将 pcap 文件拉回 controller。
- 支持 INT digest 采集、路径处理时延展示、链路吞吐统计、告警展示。
- 支持 Web 大屏展示 3D/2D 拓扑、链路状态、模态状态、应用入口、INT digest、告警信息。
- 支持 Web 大屏显示最近活跃在线人数，并在状态快照过旧时触发受限后台自愈刷新。
- 支持四个外部安全系统的占位集成：可信标识根服务系统、动态防御系统、主动防御系统、内生安全验证系统。
- 支持为外部安全系统导出实时网络状态文件，并读取外部系统写回的状态文件。
- 支持日志轮转和过期运行产物清理，避免长期运行撑满磁盘。

## 2. 当前维护的部署形态

当前主要维护的是 CENI 多 VM 直连部署形态：

- controller：集中控制节点，不运行数据面交换机。
- host1：业务客户端节点。
- server1：业务服务端节点，可运行 DeepSeek 代理服务。
- s1、s2、s3、s4、s5、s6、s7、s8、s9：P4/BMv2 交换机节点。

交换机推荐使用 direct simple_switch 启动方式，由 `multi_vm/start_switch_direct.py` 和 `multi_vm/rebuild_direct_switches.py` 负责启动与注入端口映射。p4-utils 启动方式仍保留，但现在主要用于兼容和必要时的编译验证。

## 3. 网络节点信息

当前配置文件位于：

- `multi_vm/nodes.yaml`
- `multi_vm/topology.yaml`
- `multi_vm/paths.yaml`

常用管理地址：

| 节点 | 管理 IP | 管理网口 | 数据网口 / 数据地址 |
| --- | --- | --- | --- |
| host1 | `192.168.10.2` | `ens8` | `ens7 / 100.0.0.100` |
| s1 | `192.168.1.2` | `ens7` | 见 `topology.yaml` |
| s2 | `192.168.2.2` | `ens8` | 见 `topology.yaml` |
| s3 | `192.168.3.2` | `ens7` | 见 `topology.yaml` |
| s4 | `192.168.4.2` | `ens8` | 见 `topology.yaml` |
| s5 | `192.168.5.2` | `ens7` | 见 `topology.yaml` |
| s6 | `192.168.6.2` | `ens7` | 见 `topology.yaml` |
| s7 | `192.168.7.2` | `ens7` | 见 `topology.yaml` |
| s8 | `192.168.8.2` | `ens8` | 见 `topology.yaml` |
| s9 | `192.168.9.2` | `ens7` | 见 `topology.yaml` |
| server1 | `192.168.11.2` | `ens9` | `ens7 / 100.0.0.4` |

主要 BMv2 端口映射：

| 链路 | 端点 A | 端点 B |
| --- | --- | --- |
| host1-s1 | host1 `ens7` | s1 `ens8`, port `1` |
| s1-s3 | s1 `ens9`, port `2` | s3 `ens8`, port `1` |
| s1-s5 | s1 `ens10`, port `3` | s5 `ens8`, port `2` |
| s2-s3 | s2 `ens10`, port `2` | s3 `ens10`, port `4` |
| s2-s5 | s2 `ens9`, port `1` | s5 `ens11`, port `3` |
| s5-s3 | s5 `ens9`, port `1` | s3 `ens9`, port `3` |
| s3-s4 | s3 `ens11`, port `2` | s4 `ens7`, port `1` |
| s4-s7 | s4 `ens10`, port `2` | s7 `ens10`, port `1` |
| s4-s6 | s4 `ens11`, port `3` | s6 `ens12`, port `5` |
| s4-s8 | s4 `ens9`, port `4` | s8 `ens10`, port `2` |
| s4-s9 | s4 `ens12`, port `5` | s9 `ens10`, port `2` |
| s5-s6 | s5 `ens10`, port `4` | s6 `ens8`, port `1` |
| s6-s7 | s6 `ens9`, port `2` | s7 `ens9`, port `3` |
| s6-s8 | s6 `ens10`, port `3` | s8 `ens7`, port `1` |
| s6-s9 | s6 `ens11`, port `4` | s9 `ens9`, port `1` |
| s7-server1 | s7 `ens8`, port `2` | server1 `ens7` |

如果 CENI 平台重启后网口 IP 丢失，优先根据 `nodes.yaml` 和 `topology.yaml` 恢复管理地址和数据地址。

## 4. 打包内容

上传包由以下命令生成：

```bash
python3 multi_vm/package_release.py --name optimize_ceni_bundle.tar.gz
```

包内根目录为 `optimize/`。核心内容包括：

- `new_topo.p4`：统一 P4 数据面程序。
- `s*-commands.txt`：各交换机基础表项，当前包含 `s1` 到 `s9` 的交换机命令文件。
- `policies/`：单 VM 回归和策略测试使用的策略文件。
- `tools/`：P4 表项生成、pcap 分析、命令文件校验等工具。
- `multi_vm/`：CENI 多 VM 部署、控制、状态采集、大屏、恢复、抓包、应用测试脚本。
- `README.md`、`使用说明.md`、`Agent.md`：项目说明、使用说明和后续 AI 接手说明。

打包时会排除：

- `dist/`
- `test_outputs/`
- `__pycache__/`
- `.pyc`
- `.pcap`
- `.pcapng`

这样可以避免把历史测试输出、抓包文件、缓存文件带入部署包。

## 5. 关键目录和文件

### 根目录

- `new_topo.p4`：核心数据面逻辑，包含多模态转发、封装/解封装、INT digest 相关字段。
- `s*-commands.txt`：交换机基础表项文件。
- `run_full_regression.sh`：单 VM 回归测试入口。
- `run_single_vm_smoke.sh`：单 VM ping smoke。
- `run_tcp_smoke.sh`：单 VM TCP smoke。
- `run_udp_int_smoke.sh`：单 VM UDP/INT smoke。
- `run_policy_capture_test.sh`：策略抓包验证。
- `run_runtime_*_test.sh`：运行时策略切换验证。

### `multi_vm/`

- `nodes.yaml`：节点账号、管理地址、管理网口、数据地址等配置。
- `topology.yaml`：链路、端口、网口、MAC 等拓扑配置。
- `paths.yaml`：主路径、`via_s5` 和扩展备用路径定义。
- `controllerctl.py`：配置检查、命令渲染、部署命令生成、端口映射渲染等控制辅助入口。
- `package_release.py`：生成上传包。
- `setup_ssh_keys.py` / `check_ssh_keys.py`：免密 SSH 配置和检查。
- `rebuild_direct_switches.py`：重建 direct simple_switch 数据面。
- `start_switch_direct.py`：在单个交换机节点启动 direct simple_switch。
- `start_switch_p4utils.py`：保留的 p4-utils 启动入口。
- `set_link_mode.py`：按路径或单段链路切换模态。
- `switch_path.py`：主路径和备用路径切换。
- `capture_port.py`：抓取单个交换机端口流量。
- `capture_path.py`：按路径抓取多端口流量。
- `run_main_path_smoke.py`：主路径 ICMP/TCP smoke。
- `run_app_service_smoke.py`：DeepSeek/本地应用服务 smoke。
- `run_udp_path_smoke.py`：UDP 路径连通和 digest 触发 smoke。
- `run_int_telemetry_probe.py`：INT digest 遥测验证。
- `collect_controller_status.py`：controller 状态采集，生成大屏状态 JSON。
- `run_status_collector_loop.sh`：循环状态采集。
- `run_int_refresh_loop.sh`：循环触发 INT digest 刷新。
- `dashboard_server.py`：Web 大屏后端。
- `dashboard/static/`：Web 大屏前端资源。
- `run_dashboard_server.sh`：单独启动大屏服务。
- `run_ceni_controller_stack.sh`：controller 一键运行脚本。
- `rotating_log_runner.py`：日志轮转包装器。
- `cleanup_runtime_artifacts.py`：过期运行产物清理脚本。
- `recover_switch.py` / `recover_switch.sh`：单交换机恢复脚本。
- `runtime_state.py`：运行时状态文件读写工具。

### 防御系统集成占位

当前只负责集成框架，不实现四个外部安全系统内部算法：

| 系统 | ID | 当前状态 | 说明 |
|---|---|---|---|
| 可信标识根服务系统 | `trusted_identity_root` | `pending_integration` | 对多模态网络内设备进行可信认证标识 |
| 动态防御系统 | `dynamic_defense` | `pending_integration` | 攻击数据特征检测、攻击逻辑检测、行为图结构检测 |
| 主动防御系统 | `active_defense` | `pending_integration` | 多模态网络自适应协同主动防御，包含拟态防御、蜜罐、入侵异常检测等 |
| 内生安全验证系统 | `intrinsic_security_validation` | `pending_integration` | 对多模态网络内生安全能力进行验证、评估和结果反馈 |

`collect_controller_status.py` 每轮采集都会在状态 JSON 中写入 `defense` 字段，并默认导出以下文件：

```text
/tmp/optimize_multi_vm_runtime/defense_feeds/network_status.json
/tmp/optimize_multi_vm_runtime/defense_feeds/node_status.csv
/tmp/optimize_multi_vm_runtime/defense_feeds/link_status.csv
```

这些文件是给外部防御系统读取的稳定对接面。后续若外部系统提供 API、数据库、消息队列或专用文件格式，应优先新增 adapter，不要破坏现有文件格式。

外部系统可将自身状态写入：

```text
/tmp/optimize_multi_vm_runtime/defense_inputs/trusted_identity_root.json
/tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json
/tmp/optimize_multi_vm_runtime/defense_inputs/active_defense.json
/tmp/optimize_multi_vm_runtime/defense_inputs/intrinsic_security_validation.json
```

推荐字段：`status`、`uptime_seconds`、`uptime_text`、`updated_at`、`summary`、`message`、`metrics`、`alerts`、`version`、`source`。`collect_controller_status.py` 会合并这些字段到 `status.json` 的 `defense.systems`，前端顶部四个系统框和右侧详情直接读取该结构。

对接协议和示例：

```text
安全系统对接协议.md
multi_vm/defense_input_examples/*.json
multi_vm/run_defense_integration_demo.py
multi_vm/validate_defense_inputs.py
```

前端会根据 `status`、`severity`、`risk_score` 和 `alerts` 推导展示等级。`warning/warn/degraded` 显示黄色联动，`attack_detected/error/offline/input_error` 或 `severity=critical` 显示红色联动。若 JSON 中包含 `affected_links`，对应链路会在 2D/3D 拓扑中高亮；若 `status=attack_detected` 且未指定链路，则默认高亮当前活动路径。

安全系统接入自测工具：

- `run_defense_integration_demo.py`：写入 demo 状态，支持 `--phase normal|warning|attack|recover|all|examples|clear`，用于演示顶部卡片、拓扑模型和链路高亮联动；`clear` 用于清除之前写入的 demo 状态，让四个外部系统回到待接入。
- `validate_defense_inputs.py`：校验外部系统 JSON，检查状态枚举、告警等级、风险分数、受影响链路/节点、文件大小和文件更新时间。

威胁场景演示框架：

```text
multi_vm/threat_scenarios.yaml
multi_vm/run_threat_scenario_demo.py
```

`threat_scenarios.yaml` 当前内置 10 类占位场景：通信劫持类、文件篡改类、未授权设备接入类、通信数据伪造类、操作系统漏洞类、中间件层漏洞类、应用层攻击类、通信欺骗类、流量攻击类、非法操作类。

`run_threat_scenario_demo.py --scenario <ID>` 会根据场景配置写入四个安全系统 JSON，并写入 `defense_inputs/current_threat_scenario.json`。状态采集会把它合并到 `status.json` 的 `defense.current_threat_scenario`，前端综合态势和右侧总览会展示当前威胁场景。`--phase sequence` 可用于“触发后自动恢复”的演示。

## 6. 推荐启动流程

在 controller 上进入项目目录：

```bash
cd /home/p4/optimize
```

检查配置：

```bash
python3 multi_vm/controllerctl.py check-config
python3 multi_vm/controllerctl.py render-port-map --path main
python3 multi_vm/controllerctl.py render-ssh-checks
```

配置免密 SSH：

```bash
python3 multi_vm/setup_ssh_keys.py --strict-host-key-checking no
python3 multi_vm/check_ssh_keys.py
```

重建交换机数据面：

```bash
python3 multi_vm/rebuild_direct_switches.py --path main
```

初始化 IPv4 主路径：

```bash
python3 multi_vm/apply_path_forwarding.py main
python3 multi_vm/set_link_mode.py --path main --all ipv4 --enable-int
```

启动 controller 后台组件：

```bash
bash multi_vm/run_ceni_controller_stack.sh restart
```

查看状态：

```bash
bash multi_vm/run_ceni_controller_stack.sh status
```

查看日志：

```bash
bash multi_vm/run_ceni_controller_stack.sh logs
```

停止后台组件：

```bash
bash multi_vm/run_ceni_controller_stack.sh stop
```

## 7. Web 大屏

后台启动后，大屏服务默认监听：

```text
http://127.0.0.1:8088/
```

如果本地无法直接访问 controller IP，使用 SSH 端口转发：

```bash
ssh -L 8088:127.0.0.1:8088 p4@<controller_public_ip_or_jump_host>
```

## 21. 大屏交互细节

- `dashboard/static/app.js` 维护前端展示状态，包含 3D/2D 切换、拓扑选择、应用面板、控制面板、告警展示和安全系统详情。
- 中间拓扑区域支持鼠标滚轮缩放。3D 模式通过缩放 Three.js group 实现，2D 模式通过调整 SVG viewBox 实现。
- 选中四个安全系统时，右侧详情页隐藏 INT Digest 区域，只展示该系统详情、该系统当前预警和该系统历史预警。
- 综合态势和安全系统历史预警来自 `status.history.samples[*].alerts` 与 `status.history.samples[*].defense`。这些字段由 `collect_controller_status.py` 在滚动历史样本中写入，保留策略仍由历史 retention 参数控制。
- 如果后续修改告警结构，需要同时检查 `compact_current_alerts()`、`compact_defense_history()`、前端 `buildAlerts()`、`buildDefenseAlerts()` 和 `historyAlertEntries()`。

## 22. 外部防御系统交付文档

面向外部防御系统团队的文档是：

```text
防御系统接入说明.md
```

该文档刻意避免解释 P4、交换机启动和 dashboard 内部实现，只描述外部系统需要读取的 `defense_feeds`、需要写回的 `defense_inputs`、JSON 字段、状态枚举、告警格式、链路/节点 ID、自测命令和常见错误。

后续如果外部系统接入规则变更，需要同步更新：

- `防御系统接入说明.md`：给外部团队看的接入说明。
- `安全系统对接协议.md`：内部协议细节。
- `multi_vm/validate_defense_inputs.py`：实际校验规则。
- `multi_vm/collect_controller_status.py`：实际读取和合并逻辑。
- `multi_vm/dashboard/static/app.js`：前端展示逻辑。

然后在本地浏览器打开：

```text
http://127.0.0.1:8088/
```

大屏当前能力：

- 默认 3D 拓扑展示，可切换 2D。
- 点击链路查看链路详情。
- 点击交换机查看交换机详情。
- 再次点击已选中的节点或链路会回到总览。
- 点击 host1 展示客户端应用。
- 点击 server1 展示服务端应用状态。
- 支持在 host1 应用中调用 server1 的 DeepSeek 服务。
- 支持在 host1 应用中发起抓包任务并下载 pcap。
- 顶部展示综合态势卡片和四个外部安全系统状态卡片；综合态势卡片包含当前路径、路径处理时延和告警摘要，点击可展开全部告警。
- 右上角展示最近活跃在线人数，例如 `在线 3人`。
- 右侧展示 INT digest、正向/反向路径数据、告警信息。
- 状态快照过旧时，后端会在有刷新能力的 controller 部署中触发一次受限后台刷新，避免页面长期停留在旧状态。
- 拓扑上方展示四个安全系统特殊节点，点击后右侧显示系统定位、预留功能、外部输入文件和实时状态文件路径。

## 8. 模态切换

支持模态：

```text
ipv4, ipv6, mpls, geonet, scion_v2
```

切换整条主路径：

```bash
python3 multi_vm/set_link_mode.py --path main --all ipv4 --enable-int
python3 multi_vm/set_link_mode.py --path main --all ipv6 --enable-int
python3 multi_vm/set_link_mode.py --path main --all mpls --enable-int
python3 multi_vm/set_link_mode.py --path main --all geonet --enable-int
python3 multi_vm/set_link_mode.py --path main --all scion_v2 --enable-int
```

切换单段链路：

```bash
python3 multi_vm/set_link_mode.py --path main --link s3-s4 --mode mpls --preserve-int
```

切换备用路径：

```bash
python3 multi_vm/switch_path.py via_s5 --require-node s5 --include-hosts
```

切回主路径：

```bash
python3 multi_vm/switch_path.py main
```

只下发路径转发表：

```bash
python3 multi_vm/apply_path_forwarding.py via_s5_s6_s9_s4
```

切换到某条扩展路径后统一设置核心链路模态：

```bash
python3 multi_vm/set_link_mode.py --path via_s5_s6_s9_s4 --all ipv4 --enable-int
```

注意：

- `--all` 会将路径上的所有可控链路切到同一模态。
- `--link` 只应改变指定链路，不应改动其他链路。
- `--enable-int` 会同步更新 INT 表项；`--preserve-int` 会保留当前 INT 表项。
- Web 大屏应用路径模态时，会先执行 `apply_path_forwarding.py`，再执行 `set_link_mode.py`，所以页面上的“路径核心链路”操作会真实切换活动路径。
- host1-s1 和 s7-server1 是接入链路，业务应用通常仍按 IPv4 地址访问。
- `scion_v2` 当前主要用于 UDP/probe 类验证，不建议用于 TCP/HTTP 应用链路。

## 9. DeepSeek 应用服务

server1 上服务默认监听：

```text
100.0.0.4:18080
```

API key 默认文件：

```bash
~/.config/optimize/deepseek_api_key
```

在 server1 上写入 key：

```bash
mkdir -p ~/.config/optimize
chmod 700 ~/.config/optimize
printf '%s' '<your_api_key>' > ~/.config/optimize/deepseek_api_key
chmod 600 ~/.config/optimize/deepseek_api_key
```

从 controller 发起 smoke：

```bash
python3 multi_vm/run_app_service_smoke.py --mode deepseek --prompt "请用一句话说明当前多模态网络已经连通。"
```

本地模拟模式：

```bash
python3 multi_vm/run_app_service_smoke.py --mode local
```

注意：

- server1 的外网访问走 server1 自己的外网网口。
- host1 访问 server1 服务走数据面路径。
- 如果当前路径包含 `scion_v2`，Web 大屏会禁止 DeepSeek 应用发送请求，并提示当前模态不适合 TCP/HTTP 应用。

## 10. 抓包能力

抓单个交换机端口：

```bash
python3 multi_vm/capture_port.py --switch s3 --interface ens11 --duration 8
```

按路径抓包：

```bash
python3 multi_vm/capture_path.py --path main --duration 8
python3 multi_vm/capture_path.py --path via_s5 --duration 8
```

Web 大屏里的 host1 抓包应用会：

1. 在指定交换机端口启动 tcpdump。
2. 从 host1 发出指定数量的数据包。
3. 停止抓包。
4. 将 pcap 拉回 controller。
5. 在页面提供下载链接。

pcap 默认保存在 controller 的测试输出或运行时输出目录中。后台清理脚本会定期删除过期文件。

## 11. INT digest 与遥测

手动运行 INT 探测：

```bash
python3 multi_vm/run_int_telemetry_probe.py --count 5
```

触发 UDP 路径流量：

```bash
python3 multi_vm/run_udp_path_smoke.py --count 5 --interval 0.5
```

后台自动刷新：

```bash
bash multi_vm/run_int_refresh_loop.sh
```

重要字段说明：

- `path_processing_latency`：当前推荐展示的路径处理时延，单位在大屏上转换为 ms。
- `previous_hop_processing_latency`：上一跳处理参考。
- `single_hop_latency` / `end_to_end_latency`：在多 VM 环境中受不同虚拟机时钟参考影响，只作为原始参考字段，不作为真实跨 VM 时延展示。
- `packet_length`：采样包长。
- `last_hop_switch_id`：上一跳交换机 ID。

## 12. 状态采集与告警

单次采集 controller 状态：

```bash
python3 multi_vm/collect_controller_status.py --output /tmp/optimize_multi_vm_runtime/status.json
```

循环采集：

```bash
bash multi_vm/run_status_collector_loop.sh
```

运行时状态目录：

```text
/tmp/optimize_multi_vm_runtime
```

大屏主要读取：

```text
/tmp/optimize_multi_vm_runtime/status.json
```

状态采集循环由 `run_status_collector_loop.sh` 调用 `collect_controller_status.py --interval ... --iterations 0`。新版采集循环遇到单轮异常会打印 `STATUS_SNAPSHOT_RESULT=FAIL` 并继续下一轮，避免后台进程因为一次 SSH/探测异常直接退出。

dashboard 后端读取到 `status.json` 过旧时，如果以 `--enable-refresh` 启动，会触发一次受限后台刷新。相关环境变量：

```text
OPTIMIZE_DASHBOARD_CLIENT_TTL_SECONDS=45
OPTIMIZE_DASHBOARD_AUTO_REFRESH_STALE_SECONDS=45
OPTIMIZE_DASHBOARD_AUTO_REFRESH_MIN_INTERVAL_SECONDS=15
```

在线人数统计按浏览器本地 client id 计算，而不是按 IP 计算；多人通过 SSH 端口转发访问时也能区分不同浏览器。

常见告警：

- 数据面不通。
- 某交换机未在线。
- 某备用交换机未启动。
- 状态采集失败。
- INT digest 长时间未刷新。

## 13. 健康检查

controller 综合健康检查：

```bash
python3 multi_vm/run_controller_healthcheck.py
```

保存输出：

```bash
python3 multi_vm/run_controller_healthcheck.py | tee /home/p4/optimize/output_dashboard_healthcheck.txt
```

健康检查通常包含：

- 配置检查。
- SSH 免密检查。
- 交换机状态检查。
- 主路径连通性。
- 应用服务检查。
- INT digest 检查。
- 大屏状态接口检查。

## 14. 单交换机恢复

如果某台交换机异常，例如 s3 进程消失或 9090 端口冲突，可以在 controller 上运行：

```bash
bash multi_vm/run_ceni_controller_stack.sh recover-switch s3
```

或直接运行：

```bash
python3 multi_vm/recover_switch.py --switch s3
```

恢复过程会尽量完成：

1. 停止目标交换机上的旧 simple_switch / p4-utils 进程。
2. 清理目标节点 Mininet 残留。
3. 重新启动目标交换机。
4. 重新注入该交换机相关表项。
5. 刷新当前路径和 INT 表项。
6. 重新采集 controller 状态。

常见交换机恢复命令：

```bash
bash multi_vm/run_ceni_controller_stack.sh recover-switch s1
bash multi_vm/run_ceni_controller_stack.sh recover-switch s2
bash multi_vm/run_ceni_controller_stack.sh recover-switch s3
bash multi_vm/run_ceni_controller_stack.sh recover-switch s4
bash multi_vm/run_ceni_controller_stack.sh recover-switch s5
bash multi_vm/run_ceni_controller_stack.sh recover-switch s6
bash multi_vm/run_ceni_controller_stack.sh recover-switch s7
bash multi_vm/run_ceni_controller_stack.sh recover-switch s8
bash multi_vm/run_ceni_controller_stack.sh recover-switch s9
```

## 15. 日志与清理

controller 一键栈使用日志轮转包装器，默认日志目录：

```text
/tmp/optimize_ceni_controller_stack/logs
```

日志轮转脚本：

```bash
python3 multi_vm/rotating_log_runner.py --help
```

过期产物清理脚本：

```bash
python3 multi_vm/cleanup_runtime_artifacts.py --help
```

一键栈启动时会自动启动清理循环。清理范围包括：

- 过期测试输出。
- 过期抓包文件。
- 过期运行时缓存。
- 过大的日志文件。
- 过期临时目录。

维护原则：

- 不要长期将 tcpdump、tee、大量调试输出放在无轮转文件里。
- 新增后台循环任务时，必须接入 `rotating_log_runner.py` 或等价轮转机制。
- 新增临时输出目录时，必须纳入 `cleanup_runtime_artifacts.py`。

## 16. 回归测试

单 VM 完整回归：

```bash
bash run_full_regression.sh
```

多 VM 常用回归：

```bash
python3 multi_vm/run_main_path_smoke.py
python3 multi_vm/run_app_service_smoke.py --mode local
python3 multi_vm/run_app_service_smoke.py --mode deepseek --prompt "请用一句话说明当前多模态网络已经连通。"
python3 multi_vm/run_udp_path_smoke.py --count 5 --interval 0.5
python3 multi_vm/run_int_telemetry_probe.py --count 5
python3 multi_vm/run_modal_regression.py
python3 multi_vm/run_controller_healthcheck.py
```

语法检查：

```bash
python3 -m py_compile multi_vm/*.py
bash -n multi_vm/*.sh
```

前端 JS 检查可在有 Node.js 的环境执行：

```bash
node --check multi_vm/dashboard/static/app.js
```

## 17. 设计决策

### 17.1 为什么用 controller 集中控制

CENI 环境中每个节点都是独立虚拟机。手动登录每台交换机修改表项、启动进程、抓包会很慢，也容易出错。因此当前项目将控制动作集中到 controller：

- 通过 SSH 操作交换机。
- 通过配置文件统一维护网口和端口。
- 通过脚本自动注入 P4 表项。
- 通过大屏展示统一状态。

### 17.2 为什么采用 direct simple_switch

direct simple_switch 更适合当前多 VM 环境：

- 可以显式绑定物理/虚拟网口到 BMv2 端口。
- 更容易控制 thrift 端口和 notification 地址。
- 更容易在 controller 上远程恢复单个交换机。
- 避免在每台交换机上手工进入 CLI 添加端口映射。

### 17.3 为什么保留 p4-utils 脚本

p4-utils 在早期单 VM 和原始项目中使用较多，部分环境仍可用于编译和兼容测试。因此保留：

- `start.py`
- `multi_vm/start_switch_p4utils.py`

但在 CENI 多 VM 正式演示中，优先使用 direct simple_switch 脚本。

### 17.4 为什么路径处理时延采用 `path_processing_latency`

多 VM 之间的时间戳没有严格统一时钟源，直接用不同虚拟机采集到的 raw 时间戳计算端到端时延会出现很大的参考值。因此大屏展示采用更稳定的路径处理累计字段：

- 页面展示 `path_processing_latency`。
- raw `single_hop_latency` / `end_to_end_latency` 只保留为参考，不作为真实跨 VM 时延。

## 18. 已知限制

- 当前维护 `main`、`via_s5` 以及若干经 `s2/s6/s8/s9` 的扩展备用路径；`server2` 尚未纳入当前项目范围。
- CENI 平台重启后，部分虚拟机网口 IP 可能丢失，需要按配置文件恢复。
- `scion_v2` 当前主要用于 UDP/probe 验证，不作为 TCP/HTTP 应用承载模态。
- 交换机节点可能没有独立 `p4c` 命令，常见做法是使用现有 P4 工具链或随项目脚本启动。
- Web 大屏当前面向实验演示环境，部署到公网服务器时需要补充鉴权、反向代理、HTTPS、访问控制。

## 19. 后续 AI 接手注意事项

后续 AI 维护本项目时，优先遵守以下原则：

- 先阅读 `multi_vm/nodes.yaml`、`multi_vm/topology.yaml`、`multi_vm/paths.yaml`，不要凭记忆改端口。
- 修改 P4 表项相关逻辑后，同步检查 `set_link_mode.py`、`switch_path.py`、`rebuild_direct_switches.py`。
- 修改大屏状态结构后，同步检查 `collect_controller_status.py`、`dashboard_server.py`、`dashboard/static/app.js`。
- 新增后台任务时，同步接入日志轮转和过期清理。
- 新增抓包或测试输出时，不要写入无限增长的固定文件。
- 修改脚本后更新 `使用说明.md`。
- 修改部署包内容后更新 `package_release.py`。
- 打包前至少运行 Python 语法检查、关键脚本语法检查、必要 smoke。
- 不要把 `test_outputs/`、`captures/`、`dist/`、`__pycache__/` 打进发布包。

## 20. 快速恢复到演示状态

推荐演示前执行：

```bash
cd /home/p4/optimize
python3 multi_vm/check_ssh_keys.py
python3 multi_vm/rebuild_direct_switches.py --path main
python3 multi_vm/apply_path_forwarding.py main
python3 multi_vm/set_link_mode.py --path main --all ipv4 --enable-int
bash multi_vm/run_ceni_controller_stack.sh restart
python3 multi_vm/run_controller_healthcheck.py
```

如果健康检查通过，再打开大屏：

```text
http://127.0.0.1:8088/
```

本地浏览器访问需要 SSH 端口转发：

```bash
ssh -L 8088:127.0.0.1:8088 p4@<controller_public_ip_or_jump_host>
```
