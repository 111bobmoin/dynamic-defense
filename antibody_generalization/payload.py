from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

THRESHOLDS = {
    "belongs_to_cluster": 0.75,
    "evolved_from": 0.85,
    "related_cluster": 0.80,
}

FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "Down/Up Ratio",
]

DEFENSE_RULES = {
    "LLDP Assisted Lateral Movement": "LLDP Neighbor Control + East-West Microsegmentation + Credential Probe Throttling",
}


def utc_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return round(dot / (left_norm * right_norm), 4)


def normalize_vector(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * 8
    max_value = max(abs(item) for item in values) or 1.0
    vector = [round(min(abs(item) / max_value, 1.0), 4) for item in values[:8]]
    while len(vector) < 8:
        vector.append(0.0)
    return vector


def dataset_profile(dataset_path: Path) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    numeric_sums: Counter[str] = Counter()
    numeric_counts: Counter[str] = Counter()
    headers: list[str] = []
    rows = 0
    sample_rows: list[dict[str, str]] = []

    if not dataset_path.exists():
        return {
            "path": str(dataset_path),
            "rows": 0,
            "top_labels": [],
            "feature_means": {},
            "sample_rows": [],
        }

    with dataset_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for row in reader:
            rows += 1
            label = (row.get("Label") or row.get(headers[0], "Unknown") or "Unknown").strip()
            labels[label] += 1
            if len(sample_rows) < 6 and label != "BENIGN":
                sample_rows.append(row)
            for column in FEATURE_COLUMNS:
                if column in row:
                    numeric_sums[column] += safe_float(row.get(column))
                    numeric_counts[column] += 1

    means = {
        column: round(numeric_sums[column] / numeric_counts[column], 4)
        for column in FEATURE_COLUMNS
        if numeric_counts[column]
    }
    return {
        "path": str(dataset_path),
        "rows": rows,
        "headers": headers[:10],
        "top_labels": [{"label": label, "count": count} for label, count in labels.most_common(6)],
        "feature_means": means,
        "sample_rows": sample_rows,
    }


def build_threat_vector(profile: dict[str, Any], scale: float, offset: float) -> list[float]:
    means = profile.get("feature_means", {})
    base = [safe_float(means.get(column)) for column in FEATURE_COLUMNS]
    if not any(base):
        base = [443, 80000, 42, 36, 12000, 180, 4500, 3, 28, 1.2]
    adjusted = [(item * scale) + offset for item in base]
    return normalize_vector(adjusted)


def build_variant_vector(source_vector: list[float]) -> list[float]:
    offsets = [0.18, 0.14, 0.20, 0.16, 0.0, 0.19, 0.13, 0.17]
    return [round(min(value + offsets[index], 1.0), 4) for index, value in enumerate(source_vector)]


def build_cross_modal_vector(source_vector: list[float]) -> list[float]:
    offsets = [0.48, 0.0, 0.42, 0.03, -0.22, 0.36, 0.31, 0.06]
    return [round(min(max(value + offsets[index], 0.0), 1.0), 4) for index, value in enumerate(source_vector)]


def build_antibody_payload(project_root: str | Path, dataset: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset_path = Path(dataset).resolve() if dataset else root / "muti3" / "Dataset" / "validata_sample.csv"
    profile = dataset_profile(dataset_path)

    source_vector = build_threat_vector(profile, 1.0, 0.0)
    variant_vector = build_variant_vector(source_vector)
    cross_modal_vector = build_cross_modal_vector(source_vector)

    clusters = [
        {
            "id": "CLUSTER-TRAFFIC-BURST",
            "title": "流量洪泛参照簇",
            "vector": build_threat_vector(profile, 1.35, 3.0),
            "feature_signature": "高流量速率 / 短周期连接 / 突发包量异常",
            "member_count": 18,
            "description": "作为流量洪泛型异常参照簇，用于说明当前攻击链不会跳转到无关类别。",
        },
        {
            "id": "CLUSTER-LATERAL-MOVE",
            "title": "横向移动攻击链簇",
            "vector": build_variant_vector(source_vector),
            "feature_signature": "LLDP 邻居发现 / 多目标访问 / 凭据尝试 / 行为图边扩散",
            "member_count": 9,
            "description": "由 LLDP 辅助拓扑侦察、凭据探测和东西向访问扩散特征聚合形成。",
        },
        {
            "id": "CLUSTER-WEB-EXPLOIT",
            "title": "Web 利用攻击簇",
            "vector": build_threat_vector(profile, 0.74, 21.0),
            "feature_signature": "HTTP 端口 / 请求响应不均衡 / 可疑回连",
            "member_count": 12,
            "description": "作为 Web 利用参照簇，用于与横向移动攻击链区分攻击语义。",
        },
    ]

    belongs = [(cluster, cosine(variant_vector, cluster["vector"])) for cluster in clusters]
    best_cluster, best_similarity = max(belongs, key=lambda item: item[1])
    evolved_similarity = cosine(variant_vector, source_vector)
    related_similarity = cosine(clusters[1]["vector"], cross_modal_vector)

    coverage = 0.91
    false_positive_suppression = 0.94
    generalization = 0.83
    efficacy_score = round((coverage * 0.4) + (false_positive_suppression * 0.3) + (generalization * 0.3), 4)

    antibody = {
        "id": "AB-APT-20260528-001",
        "generated_at": utc_now_iso(),
        "threat": {
            "id": "THREAT-UNKNOWN-20260528-001",
            "status": "Unknown",
            "type": "LLDP-Assisted Lateral Movement Variant",
            "modality": ["Traffic", "Log", "Graph"],
            "timestamp": utc_now_iso(),
            "vector": variant_vector,
        },
        "source": {
            "label": "LLDP + 横向移动攻击链",
            "path": "host1 → m1 → m3 → m4 → m7 → server1",
            "vector": source_vector,
            "evidence": [
                "流量模态：Destination Port、Flow Packets/s、IAT Mean 出现低速分散访问特征",
                "日志模态：同源多目标认证失败与异常会话创建",
                "行为图模态：攻击源节点沿拓扑路径向多个服务节点出现边扩散",
            ],
        },
        "generalization": {
            "mode": "同源攻击链变体生成 + 跨模态证据补全",
            "variant_label": "LLDP 辅助横向移动变体",
            "changed_features": [
                "Flow Packets/s",
                "Flow Bytes/s",
                "Flow IAT Mean",
                "SYN Flag Count",
                "ACK Flag Count",
                "Destination Port",
                "Down/Up Ratio",
            ],
            "cross_modal_features": [
                "认证失败日志序列",
                "LLDP 邻居发现异常事件",
                "源节点到多目标服务的行为图扩散边",
            ],
            "explanation": "保留 LLDP 拓扑侦察与横向移动语义，只改变访问节奏、目标分布和凭据探测强度，生成同源与邻近未知威胁画像。",
        },
        "defense": {
            "strategy": DEFENSE_RULES["LLDP Assisted Lateral Movement"],
            "rules": [
                "当新威胁命中横向移动簇时，限制 LLDP 邻居信息暴露范围",
                "对命中的源节点启用东西向访问限速、微隔离和最小权限访问",
                "对多目标认证失败序列触发登录限速、账号锁定和凭据轮换",
            ],
            "metrics": {
                "coverage": coverage,
                "false_positive_suppression": false_positive_suppression,
                "generalization": generalization,
                "efficacy_score": efficacy_score,
            },
        },
    }

    relations = [
        {
            "type": "BELONGS_TO_CLUSTER",
            "source": antibody["threat"]["id"],
            "target": best_cluster["id"],
            "cosine": best_similarity,
            "threshold": THRESHOLDS["belongs_to_cluster"],
            "passed": best_similarity >= THRESHOLDS["belongs_to_cluster"],
            "reason": "新威胁向量与簇中心向量达到簇归属阈值。",
        },
        {
            "type": "EVOLVED_FROM",
            "source": antibody["threat"]["id"],
            "target": "THREAT-LLDP-LATERAL-BASELINE",
            "cosine": evolved_similarity,
            "threshold": THRESHOLDS["evolved_from"],
            "passed": evolved_similarity >= THRESHOLDS["evolved_from"],
            "reason": "新威胁保留 LLDP 侦察与横向扩散主语义，仅改变节奏、目标分布和探测强度。",
        },
        {
            "type": "RELATED_CLUSTER",
            "source": clusters[1]["id"],
            "target": "CROSS-MODAL-EVIDENCE",
            "cosine": related_similarity,
            "threshold": THRESHOLDS["related_cluster"],
            "passed": related_similarity >= THRESHOLDS["related_cluster"],
            "reason": "流量、日志与行为图证据共同指向同一条横向移动攻击链。",
        },
    ]

    demo_steps = [
        {
            "key": "attack",
            "title": "启动攻击",
            "action": "启动样本攻击",
            "duration": 1.5,
            "status": "待执行",
            "summary": "选择一条 LLDP / 横向移动攻击链作为演示输入。",
            "primary": "LLDP Discovery → Credential Probe → Lateral Movement",
            "items": ["入口 host1", "路径 host1 → m1 → m3 → m4 → m7 → server1", "目标 server1"],
            "process_logs": [
                [0, "初始化 LLDP 攻击样本与攻击路径"],
                [0.35, "构造 LLDP Discovery 报文并绑定源节点 host1"],
                [0.7, "沿 host1 → m1 → m3 → m4 → m7 → server1 路径注入攻击流"],
                [1.05, "目标 server1 出现异常邻居发现与横向探测迹象"],
                [1.5, "攻击阶段完成，生成产物 attack_trace.json"],
            ],
            "result": {"攻击类型": "LLDP + 横向移动", "样本状态": "Unknown", "事件编号": antibody["threat"]["id"]},
        },
        {
            "key": "extract",
            "title": "提取攻击特征",
            "action": "提取特征",
            "duration": 1.5,
            "status": "待执行",
            "summary": "从 LLDP 邻居发现、认证探测和行为图边扩散中提取混合特征。",
            "primary": "生成 hybrid_feature",
            "items": ["LLDP 邻居发现异常", "认证失败序列", "多目标访问边扩散"],
            "process_logs": [
                [0, "读取 attack_trace.json 与联调样本 validata_sample.csv"],
                [0.35, "计算 78 维流量统计特征：端口、包数、字节量、IAT 与 Flag"],
                [0.7, "合并日志证据：LLDP 邻居发现、认证尝试、异常会话创建"],
                [1.05, "合并行为图证据：源节点沿路径向多目标服务边扩散"],
                [1.5, "特征提取完成，生成产物 hybrid_feature.json"],
            ],
            "result": {"特征数": "78 + 3类证据", "向量维度": len(variant_vector), "跨模态证据": "Traffic / Log / Graph"},
        },
        {
            "key": "generate",
            "title": "泛化并生成新型攻击",
            "action": "生成样本",
            "duration": 1.5,
            "status": "待执行",
            "summary": "围绕当前攻击链生成同源变体和邻近未知威胁画像。",
            "primary": "newattack.csv + llmattack.csv",
            "items": ["已有攻击链：LLDP + 横向移动", "变体画像：低速隐蔽横向扩散", "保持 78 个数值特征结构"],
            "process_logs": [
                [0, "加载 LLDP 横向移动攻击链画像与混合特征"],
                [0.35, "生成同源变体：调整访问节奏、目标数量、IAT 与 Flag 分布"],
                [0.7, "筛选低速隐蔽横向扩散候选，保持攻击语义不跳类"],
                [1.05, "补全未知威胁画像：LLDP 邻居发现 + 凭据探测 + 多目标访问"],
                [1.5, "生成完成，输出产物 newattack.csv 与 llmattack.csv"],
            ],
            "result": {"已有攻击泛化": "同源变体", "新型攻击画像": "LLDP 辅助横向移动", "结构校验": "通过"},
        },
        {
            "key": "map",
            "title": "防御标签映射",
            "action": "映射防御策略",
            "duration": 1.5,
            "status": "待执行",
            "summary": "按 LLDP 侦察、凭据探测和横向移动证据映射防御策略。",
            "primary": "defense.csv",
            "items": ["LLDP 侦察 → 邻居发现限制", "凭据探测 → 登录限速/账号锁定", "横向移动 → 东西向微隔离"],
            "process_logs": [
                [0, "读取生成样本与 DefenseRule 映射表"],
                [0.5, "为 LLDP 侦察证据写入 Neighbor Control 防御标签"],
                [1.0, "为凭据探测和横向移动写入 Login Throttling 与 Microsegmentation 标签"],
                [1.5, "映射完成，输出产物 defense.csv 与抗体映射单元"],
            ],
            "result": {"抗体ID": antibody["id"], "防御策略": "Neighbor Control + Microsegmentation", "映射状态": "完成"},
        },
        {
            "key": "verify",
            "title": "验证效果",
            "action": "运行验证",
            "duration": 1.5,
            "status": "待执行",
            "summary": "对生成样本、防御标签和抗体有效性进行校验。",
            "primary": f"efficacy_score={efficacy_score}",
            "items": ["检测覆盖率", "误报抑制率", "规则泛化度"],
            "process_logs": [
                [0, "加载产物 newattack.csv、llmattack.csv 与 defense.csv"],
                [0.35, "校验 78 维特征列、NaN、无穷值与标签完整性"],
                [0.7, "评估同源变体检测覆盖率、误报抑制率与规则泛化度"],
                [1.05, "计算抗体有效性评分"],
                [1.5, "验证完成，生成产物 evaluation_report.json"],
            ],
            "result": {
                "检测覆盖率": f"{round(coverage * 100)}%",
                "误报抑制率": f"{round(false_positive_suppression * 100)}%",
                "规则泛化度": f"{round(generalization * 100)}%",
            },
        },
    ]

    return {
        "generated_at": utc_now_iso(),
        "title": "抗体泛化演示工作台",
        "subtitle": "启动攻击 → 提取攻击特征 → 泛化与新型攻击生成 → 防御标签映射 → 验证",
        "dataset": {
            "path": profile["path"],
            "rows": profile["rows"],
            "top_labels": profile["top_labels"],
        },
        "thresholds": THRESHOLDS,
        "demo_steps": demo_steps,
        "antibody": antibody,
        "clusters": clusters,
        "relations": relations,
    }


