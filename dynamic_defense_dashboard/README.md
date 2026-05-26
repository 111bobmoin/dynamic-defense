# Dynamic Defense Dashboard

这个目录是 `Dynamic Defense` 子项目的独立前后端联调壳层。

当前目标：

- 把 `muti3` 里的三类检测能力接成统一后端接口
- 提供一个模仿上层多模态网络大屏风格的前端界面
- 抗体泛化和动态防御暂时只保留展示占位

## 启动

推荐方式：

```bash
cd /home/xiaowang/Desktop/Dynamic Defense
python3 dynamic_defense_dashboard/server.py
```

如果你希望完全沿用 `muti3` 的相对路径上下文，也可以：

```bash
cd /home/xiaowang/Desktop/Dynamic Defense/muti3
python3 ../dynamic_defense_dashboard/server.py
```

默认地址：

```text
http://127.0.0.1:8099/
```

## 说明

- 后端使用 Python 标准库 HTTP 服务，不额外依赖 Web 框架。
- `muti3` 模型评估依赖 `torch`、`pandas`、`scikit-learn`。
- 如果环境里缺少这些依赖，页面仍可打开，但检测结果会显示依赖缺失状态。
- 默认联调数据集是 `muti3/Dataset/validata_sample.csv`。

## 接口

- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/detection`

可选查询参数：

- `dataset=muti3/Dataset/validata_sample.csv`
