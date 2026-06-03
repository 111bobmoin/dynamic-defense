from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def build_cross_modal_payload(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()

    transfer_cards = [
        {"label": "源模态", "value": "IPv4 / IPv6", "text": "提取已有攻击链中的扫描、凭据探测、横向移动和影响阶段。"},
        {"label": "迁移桥接", "value": "语义对齐 + 契约校验", "text": "保留攻击意图，约束可落地的目标模态动作和日志标记。"},
        {"label": "目标模态", "value": "SCION / GEO / MPLS", "text": "生成路径切换、链路干扰、边界洪泛等可验收实验步骤。"},
    ]

    plan_summary = [
        {"time": "00.0s", "scope": "IPv4/IPv6", "action": "解析源侧攻击链", "marker": "CHAIN_PARSED"},
        {"time": "03.0s", "scope": "SCION", "action": "边界路径切换检测", "marker": "SCION_PATH_SWITCH"},
        {"time": "06.0s", "scope": "GEO", "action": "链路抖动与时延扰动", "marker": "GEO_LINK_JITTER"},
        {"time": "09.0s", "scope": "MPLS", "action": "标签转发异常观测", "marker": "MPLS_LABEL_DEVIATION"},
        {"time": "12.0s", "scope": "SCION/GEO/MPLS", "action": "跨模态抗体规则验收", "marker": "TRANSFER_ACCEPTED"},
    ]

    checks = [
        {"name": "模态覆盖", "value": "5/5", "status": "通过"},
        {"name": "契约校验", "value": "通过", "status": "通过"},
        {"name": "日志验收", "value": "5/5 PASS", "status": "通过"},
        {"name": "抗体迁移", "value": "3 条规则", "status": "已沉淀"},
    ]

    transferred_antibodies = [
        {"rule": "SCION 边界路径切换抗体", "target": "跨域路径异常", "status": "已沉淀"},
        {"rule": "GEO 链路抖动抗体", "target": "高时延与丢包扰动", "status": "已沉淀"},
        {"rule": "MPLS 标签转发异常抗体", "target": "标签路径偏移", "status": "已沉淀"},
    ]

    demo_steps = [
        {
            "key": "source",
            "title": "源模态攻击链读取",
            "action": "读取攻击链",
            "duration": 1.5,
            "summary": "从 IPv4/IPv6 实验结果中读取攻击链和关键攻击意图。",
            "items": ["IPv4 / IPv6", "攻击链阶段", "关键日志标记"],
            "process_logs": [
                [0, "读取 IPv4/IPv6 攻击链记录"],
                [0.35, "提取扫描、凭据探测、横向移动和影响阶段"],
                [0.7, "汇总源侧流量与日志标记"],
                [1.05, "生成源模态攻击链摘要"],
                [1.5, "生成产物 source_chain.json"],
            ],
            "result": {"源模态": "IPv4 / IPv6", "阶段数": 4, "状态": "完成"},
            "artifact": {"artifact": "source_chain.json", "summary": "记录源侧攻击链、阶段和日志标记。"},
        },
        {
            "key": "transfer",
            "title": "跨模态语义迁移",
            "action": "迁移语义",
            "duration": 1.5,
            "summary": "将源侧攻击意图迁移到 SCION/GEO/MPLS 可表达的实验动作。",
            "items": ["SCION 路径切换", "GEO 链路扰动", "MPLS 标签转发"],
            "process_logs": [
                [0, "加载五模态语义对齐规则"],
                [0.35, "将横向移动映射为 SCION 边界路径切换"],
                [0.7, "将链路干扰映射为 GEO 抖动与丢包扰动"],
                [1.05, "将影响阶段映射为 MPLS 标签转发异常"],
                [1.5, "生成产物 modal_transfer.json"],
            ],
            "result": {"目标模态": "SCION / GEO / MPLS", "迁移动作": 3, "状态": "完成"},
            "artifact": {"artifact": "modal_transfer.json", "summary": "记录源侧语义到目标模态动作的迁移结果。"},
        },
        {
            "key": "validation",
            "title": "契约校验与执行计划",
            "action": "生成计划",
            "duration": 1.5,
            "summary": "校验迁移结果并生成目标模态实验时间表。",
            "items": ["模态覆盖校验", "参数范围校验", "目标模态时间表"],
            "process_logs": [
                [0, "检查 IPv4/IPv6/SCION/GEO/MPLS 五模态覆盖"],
                [0.35, "校验路径切换、链路扰动和标签转发参数"],
                [0.7, "生成目标模态执行时间表"],
                [1.05, "绑定预期日志标记"],
                [1.5, "生成产物 target_modal_plan.json"],
            ],
            "result": {"模态覆盖": "5/5", "契约校验": "通过", "计划步骤": len(plan_summary)},
            "artifact": {"artifact": "target_modal_plan.json", "summary": "记录目标模态执行时间表和验收标记。"},
        },
        {
            "key": "acceptance",
            "title": "日志验收与抗体迁移",
            "action": "验收迁移",
            "duration": 1.5,
            "summary": "核对目标模态日志，并沉淀跨模态抗体规则。",
            "items": ["日志验收", "抗体规则", "反馈池写入"],
            "process_logs": [
                [0, "加载 SCION/GEO/MPLS 实验日志"],
                [0.35, "匹配 PATH / JITTER / LABEL / TRANSFER 标记"],
                [0.7, "汇总 5/5 PASS 验收结果"],
                [1.05, "沉淀 3 条跨模态抗体规则"],
                [1.5, "生成产物 cross_modal_report.json"],
            ],
            "result": {"验收结果": "5/5 PASS", "迁移抗体": len(transferred_antibodies), "反馈状态": "已写入"},
            "artifact": {"artifact": "cross_modal_report.json", "summary": "记录日志验收、抗体迁移结果和反馈池条目。"},
        },
    ]

    return {
        "generated_at": utc_now_iso(),
        "title": "跨模态抗体泛化工作台",
        "subtitle": "源模态读取 → 语义迁移 → 契约计划 → 日志验收",
        "source_modality": "IPv4 / IPv6",
        "target_modality": "SCION / GEO / MPLS",
        "project_root": str(root),
        "demo_steps": demo_steps,
        "outputs": [
            {"step": "source", "label": "源模态攻击链", "value": "source_chain.json"},
            {"step": "transfer", "label": "迁移结果", "value": "modal_transfer.json"},
            {"step": "validation", "label": "执行计划", "value": "target_modal_plan.json"},
            {"step": "acceptance", "label": "验收报告", "value": "cross_modal_report.json"},
        ],
        "cross_modal": {
            "source_modality": "IPv4 / IPv6",
            "target_modality": "SCION / GEO / MPLS",
            "kill_chain": "扫描 → 凭据探测 → 横向移动 → 链路干扰 → 影响扩大",
            "transfer_cards": transfer_cards,
            "plan_summary": plan_summary,
            "checks": checks,
            "transferred_antibodies": transferred_antibodies,
        },
    }
