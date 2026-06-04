const DEFAULT_DATASET = "muti3/Dataset/validata_sample.csv";
const DEFAULT_ROUTE = ["host1", "m1", "m3", "m4", "m7", "server1"];
const DEFENSE_TOPOLOGY_NODE_ORDER = ["host1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "server1", "server2", "controller"];
const DEFENSE_SPECIAL_NODE_ORDERS = {
  run_03_seed_2026051203: ["m4", "m3", "host1", "m1", "m2", "m5", "m6", "m7", "m8", "m9", "server1", "server2", "controller"],
};
const state = {
  dataset: DEFAULT_DATASET,
  monitorTimer: null,
  liveTimer: null,
  liveFrame: 0,
  dashboardLiveBase: null,
  defenseRotationTimer: null,
};

const MONITOR_WINDOW = 14;
const MONITOR_TICK_MS = 1200;
const LIVE_TICK_MS = 1400;
const DEFENSE_ROTATION_TICK_MS = 12000;

const COMPONENT_DISPLAY_ORDER = ["muti3", "log", "graph", "dynamic_defense"];
const COMPONENT_TITLE_MAP = {
  muti3: "攻击数据特征检测异构组件",
  log: "攻击逻辑检测异构组件",
  graph: "行为图结构检测异构组件",
  dynamic_defense: "最小代价修复组件",
};

const COMPONENT_MODEL_META_MAP = {
  muti3: {
    traffic: {
      title: "LSTM",
      subtitle: "序列特征时序建模",
      message: "LSTM 在线检测运行中。",
    },
    log: {
      title: "Subspace Clustering",
      subtitle: "子空间聚类判别",
      message: "Subspace Clustering 在线检测运行中。",
    },
    graph: {
      title: "Autoregressive",
      subtitle: "自回归特征建模",
      message: "Autoregressive 在线检测运行中。",
    },
  },
};

function componentModelMeta(sectionKey, model) {
  return COMPONENT_MODEL_META_MAP[sectionKey]?.[model?.key] || null;
}

function componentModelTitle(sectionKey, model) {
  return componentModelMeta(sectionKey, model)?.title || model?.title || "--";
}

function componentModelSubtitle(sectionKey, model) {
  return componentModelMeta(sectionKey, model)?.subtitle || model?.subtitle || "--";
}

function componentModelMessage(sectionKey, model) {
  return componentModelMeta(sectionKey, model)?.message || model?.message || "--";
}

function componentSectionSummary(section, options = {}) {
  if (section?.key === "muti3" && options.modelNames) {
    return "LSTM / Subspace Clustering / Autoregressive 三模型实时联动运行。";
  }
  if (section?.key === "dynamic_defense") {
    return "最小代价修复结果摘要已就绪，支持查看修复顺序、节点风险画像与解释边界。";
  }
  return section?.summary || "--";
}

function componentSectionTitle(detailOrSection) {
  return COMPONENT_TITLE_MAP[detailOrSection?.key] || detailOrSection?.title || "--";
}

const SYSTEM_SUMMARY_OVERRIDES = {
  multi_detection: [
    "攻击数据特征检测异构组件",
    "攻击逻辑检测异构组件",
    "行为图结构检测异构组件",
  ],
};

const DEFENSE_MODEL_METRICS = {
  title: "GAT Node Anomaly Detector",
  subtitle: "network_intrusion_detection_GAT / CICIDS2017",
  status: "ready",
  modelPath: "network_intrusion_detection_GAT/outputs/training/20260511_182846/model.pt",
  bestEpoch: 14,
  featureCount: 68,
  validation: {
    accuracy: 0.9603328010757206,
    precision: 0.7969753241286958,
    recall: 0.8584521049255053,
    f1_score: 0.8144686434642843,
  },
  test: {
    accuracy: 0.9638594721801984,
    precision: 0.8022144381993688,
    recall: 0.8666844485712677,
    f1_score: 0.8134502315523816,
  },
};

function defenseComponentSection(sample = DEFENSE_SAMPLE) {
  return {
    key: "dynamic_defense",
    title: "最小代价修复组件",
    summary: "基于 network_intrusion_detection_GAT 的多异常节点样本轮换展示，支持查看修复顺序、节点风险画像和解释边界。",
    dataset: {
      rows: sample.nodes?.reduce((sum, item) => sum + Number(item.totalFlows || 0), 0) || 0,
    },
    overall: {
      status: "ready",
      models_ready: 1,
      model_total: 1,
    },
    modelMetrics: DEFENSE_MODEL_METRICS,
    models: [
      {
        key: "repair_plan",
        title: "Minimum-cost Repair",
        subtitle: "节点级最小代价修复排序",
        message: sample.interpretation,
        status: "ready",
        accuracy: DEFENSE_MODEL_METRICS.test.accuracy,
        precision: DEFENSE_MODEL_METRICS.test.precision,
        recall: DEFENSE_MODEL_METRICS.test.recall,
        f1_score: DEFENSE_MODEL_METRICS.test.f1_score,
        model_path: DEFENSE_MODEL_METRICS.modelPath,
      },
    ],
    sample,
  };
}

function resolveDefenseSection(source = null) {
  if (source?.sample) {
    return source;
  }
  return defenseComponentSection(source || DEFENSE_SAMPLE);
}

function currentDashboardTab() {
  return document.querySelector("[data-dashboard-tab].is-active")?.dataset.dashboardTab || "overview";
}

function currentThreatTab(scope = document) {
  return scope.querySelector("[data-threat-tab].is-active")?.dataset.threatTab
    || scope.querySelector(".threat-tab-nav")?.dataset.defaultThreatTab
    || "realtime";
}

function commitHistoryUrl(href, historyMode = "replace") {
  if (!href || historyMode === "none") {
    return;
  }
  const nextUrl = new URL(href, window.location.origin);
  const nextPath = `${nextUrl.pathname}${nextUrl.search}`;
  const currentPath = `${window.location.pathname}${window.location.search}`;
  if (nextPath === currentPath) {
    return;
  }
  if (historyMode === "push") {
    window.history.pushState({}, "", nextPath);
    return;
  }
  window.history.replaceState({}, "", nextPath);
}

function buildDashboardUrl(tabKey = "overview", threatTab = null) {
  const url = new URL("/", window.location.origin);
  if (tabKey && tabKey !== "overview") {
    url.searchParams.set("tab", tabKey);
  }
  if (tabKey === "unknown-threat" && threatTab && threatTab !== "realtime") {
    url.searchParams.set("threatTab", threatTab);
  }
  return `${url.pathname}${url.search}`;
}

function buildUnknownThreatUrl(threatTab = "realtime") {
  const url = new URL("/unknown-threat.html", window.location.origin);
  if (threatTab && threatTab !== "realtime") {
    url.searchParams.set("threatTab", threatTab);
  }
  return `${url.pathname}${url.search}`;
}

function buildAntibodyUrl(fromPage = null, fromTab = null) {
  const url = new URL("/antibody.html", window.location.origin);
  if (fromPage) {
    url.searchParams.set("fromPage", fromPage);
  }
  if (fromTab) {
    url.searchParams.set("fromTab", fromTab);
  }
  return `${url.pathname}${url.search}`;
}

function antibodyBackState(params = new URLSearchParams(window.location.search)) {
  const fromPage = params.get("fromPage");
  const fromTab = params.get("fromTab");

  if (fromPage === "dashboard") {
    if (fromTab === "overview") {
      return {
        overviewHref: buildDashboardUrl("overview"),
        overviewLabel: "返回系统总览",
        sectionHref: buildDashboardUrl("antibody"),
        sectionLabel: "返回抗体泛化入口",
        breadcrumb: "总览 / 系统总览 / 抗体泛化动态演示",
      };
    }
    return {
      overviewHref: buildDashboardUrl("overview"),
      overviewLabel: "返回总览",
      sectionHref: buildDashboardUrl(fromTab || "antibody"),
      sectionLabel: "返回抗体泛化入口",
      breadcrumb: "总览 / 抗体泛化 / 动态演示",
    };
  }

  return {
    overviewHref: buildDashboardUrl("overview"),
    overviewLabel: "返回总览",
    sectionHref: buildDashboardUrl("antibody"),
    sectionLabel: "返回抗体泛化入口",
    breadcrumb: "总览 / 抗体泛化 / 动态演示",
  };
}

function bindAntibodyNavigation() {
  const nav = antibodyBackState();
  ["antibodyOverviewLink", "antibodyTopOverviewLink"].forEach((id) => {
    const link = byId(id);
    if (link) {
      link.href = nav.overviewHref;
      link.textContent = nav.overviewLabel;
    }
  });
  ["antibodySectionLink", "antibodyTopSectionLink"].forEach((id) => {
    const link = byId(id);
    if (link) {
      link.href = nav.sectionHref;
      link.textContent = nav.sectionLabel;
    }
  });
  const breadcrumb = byId("antibodyBreadcrumb");
  if (breadcrumb) {
    breadcrumb.textContent = nav.breadcrumb;
  }
}

function detailBackState(sectionKey, params = new URLSearchParams(window.location.search)) {
  const fromPage = params.get("fromPage");
  const fromTab = params.get("fromTab");
  const threatTab = params.get("threatTab") || "realtime";

  if (fromPage === "dashboard") {
    const tabKey = fromTab || (sectionKey === "dynamic_defense" ? "defense" : "unknown-threat");
    if (tabKey === "overview") {
      return {
        href: buildDashboardUrl("overview"),
        label: "返回系统总览",
        sectionLabel: "返回总览工作台",
        breadcrumb: "总览 / 系统总览 / 组件实时监控",
      };
    }
    return {
      href: buildDashboardUrl(tabKey, tabKey === "unknown-threat" ? threatTab : null),
      label: tabKey === "defense" ? "返回动态防御" : "返回检测总览",
      sectionLabel: tabKey === "defense" ? "返回动态防御模块" : "返回未知威胁模块",
      breadcrumb: tabKey === "defense" ? "总览 / 动态防御 / 详情" : "总览 / 未知威胁检测 / 组件实时监控",
    };
  }

  if (sectionKey === "dynamic_defense") {
    return {
      href: buildDashboardUrl("defense"),
      label: "返回动态防御",
      sectionLabel: "返回动态防御模块",
      breadcrumb: "总览 / 动态防御 / 最小代价修复",
    };
  }

  return {
    href: buildUnknownThreatUrl(threatTab),
    label: "返回检测总览",
    sectionLabel: "返回未知威胁模块",
    breadcrumb: "总览 / 未知威胁检测 / 组件实时监控",
  };
}

function setDetailNavigation(sectionKey) {
  const nav = detailBackState(sectionKey);
  const backLink = byId("detailBackLink");
  if (backLink) {
    backLink.href = nav.href;
    backLink.textContent = nav.label;
  }
  const sectionLink = byId("detailSectionLink");
  if (sectionLink) {
    sectionLink.href = nav.href;
    sectionLink.textContent = nav.sectionLabel;
  }
  const breadcrumb = byId("detailBreadcrumb");
  if (breadcrumb) {
    breadcrumb.textContent = nav.breadcrumb;
  }
}

const byId = (id) => document.getElementById(id);
const page = document.body.dataset.page;

const moduleRouteMap = {
  multi_detection: "/unknown-threat.html",
  antibody_generalization: "/antibody.html",
  dynamic_defense: "/?tab=defense",
};

const dashboardModuleTabMap = {
  multi_detection: "unknown-threat",
  antibody_generalization: "antibody",
  dynamic_defense: "defense",
};

const sceneNodes = [
  { id: "host1", label: "host1", sublabel: "运行中", type: "endpoint", x: 10, y: 40, rx: 3.1, ry: 4.6 },
  { id: "m1", label: "多模态网元1", sublabel: "运行中", type: "switch", x: 27, y: 39, rx: 3.1, ry: 4.6 },
  { id: "m2", label: "多模态网元2", sublabel: "运行中", type: "switch", x: 27, y: 65, rx: 3.1, ry: 4.6 },
  { id: "m3", label: "多模态网元3", sublabel: "运行中", type: "switch", x: 43, y: 37, rx: 3.1, ry: 4.6 },
  { id: "controller", label: "controller", sublabel: "运行中", type: "controller", x: 50, y: 12, rx: 3.4, ry: 5.0 },
  { id: "m4", label: "多模态网元4", sublabel: "运行中", type: "switch", x: 59, y: 36, rx: 3.1, ry: 4.6 },
  { id: "m5", label: "多模态网元5", sublabel: "运行中", type: "switch", x: 43, y: 64, rx: 3.1, ry: 4.6 },
  { id: "m6", label: "多模态网元6", sublabel: "运行中", type: "switch", x: 59, y: 64, rx: 3.1, ry: 4.6 },
  { id: "m7", label: "多模态网元7", sublabel: "运行中", type: "switch", x: 76, y: 37, rx: 3.1, ry: 4.6 },
  { id: "m8", label: "多模态网元8", sublabel: "运行中", type: "switch", x: 76, y: 64, rx: 3.1, ry: 4.6 },
  { id: "m9", label: "多模态网元9", sublabel: "运行中", type: "switch", x: 50, y: 90, rx: 3.1, ry: 4.6 },
  { id: "server1", label: "server1", sublabel: "运行中", type: "endpoint", x: 93, y: 38, rx: 3.1, ry: 4.6 },
  { id: "server2", label: "server2", sublabel: "运行中", type: "endpoint", x: 93, y: 91, rx: 3.1, ry: 4.6 },
];

const sceneLinks = [
  { id: "host1-m1", from: "host1", to: "m1", fromSide: "right", toSide: "left" },
  { id: "host1-controller", from: "host1", to: "controller", fromSide: "right", toSide: "left" },
  { id: "m1-controller", from: "m1", to: "controller", fromSide: "top", toSide: "left" },
  { id: "m1-m3", from: "m1", to: "m3", fromSide: "right", toSide: "left" },
  { id: "m1-m5", from: "m1", to: "m5", fromSide: "bottom", toSide: "left" },
  { id: "m2-m3", from: "m2", to: "m3", fromSide: "top", toSide: "left" },
  { id: "m2-m5", from: "m2", to: "m5", fromSide: "right", toSide: "left" },
  { id: "m3-controller", from: "m3", to: "controller", fromSide: "top", toSide: "bottom" },
  { id: "m3-m4", from: "m3", to: "m4", fromSide: "right", toSide: "left" },
  { id: "m3-m5", from: "m3", to: "m5", fromSide: "bottom", toSide: "top" },
  { id: "m4-controller", from: "m4", to: "controller", fromSide: "top", toSide: "bottom" },
  { id: "m4-m6", from: "m4", to: "m6", fromSide: "bottom", toSide: "top" },
  { id: "m4-m7", from: "m4", to: "m7", fromSide: "right", toSide: "left" },
  { id: "m4-m8", from: "m4", to: "m8", fromSide: "bottom", toSide: "left" },
  { id: "m5-controller", from: "m5", to: "controller", fromSide: "top", toSide: "bottom" },
  { id: "m5-m6", from: "m5", to: "m6", fromSide: "right", toSide: "left" },
  { id: "m6-controller", from: "m6", to: "controller", fromSide: "top", toSide: "bottom" },
  { id: "m6-m7", from: "m6", to: "m7", fromSide: "top", toSide: "left" },
  { id: "m6-m8", from: "m6", to: "m8", fromSide: "right", toSide: "left" },
  { id: "m6-m9", from: "m6", to: "m9", fromSide: "bottom", toSide: "top" },
  { id: "m7-controller", from: "m7", to: "controller", fromSide: "top", toSide: "right" },
  { id: "m7-server1", from: "m7", to: "server1", fromSide: "right", toSide: "left" },
  { id: "m8-server1", from: "m8", to: "server1", fromSide: "top", toSide: "left" },
  { id: "m9-controller", from: "m9", to: "controller", fromSide: "top", toSide: "bottom" },
  { id: "m9-server2", from: "m9", to: "server2", fromSide: "right", toSide: "left" },
  { id: "controller-server1", from: "controller", to: "server1", fromSide: "right", toSide: "left" },
];

const sceneVariants = {
  dashboard: {
    title: "网络拓扑总览",
    subtitle: "左主右辅 / 核心路径优先",
    markers: [],
    plain: true,
  },
  "unknown-threat": {
    title: "UNKNOWN THREAT DETECTION",
    subtitle: "所有模型实时运行，可点击进入单模型详情",
    markers: [
      { label: "Graph", tone: "lime", x: 38, y: 42 },
      { label: "Log x3", tone: "amber", x: 55, y: 41 },
      { label: "muti3 x3", tone: "cyan", x: 73, y: 42 },
    ],
    plain: true,
  },
  antibody: {
    title: "ANTIBODY GENERALIZATION",
    subtitle: "模态内威胁泛化 / 跨模态抗体迁移",
    markers: [
      { label: "模态内泛化", tone: "amber", x: 45, y: 42 },
      { label: "跨模态抗体", tone: "cyan", x: 61, y: 41 },
    ],
    plain: false,
  },
  defense: {
    title: "DYNAMIC DEFENSE",
    subtitle: "最小代价修复 / 弹性路由 / 防御执行",
    markers: [
      { label: "最小代价修复", tone: "amber", x: 43, y: 42 },
      { label: "弹性路由", tone: "cyan", x: 59, y: 41 },
      { label: "动态防御", tone: "lime", x: 76, y: 42 },
    ],
    plain: false,
  },
};

const DEFENSE_SAMPLE = {
  generatedAt: "2026-05-20T19:24:44",
  sampleName: "one_shot_20260512_163107_run_01_seed_2026051201_372137472",
  sceneLabel: "one_shot_20260512_163107 · run_01_seed_2026051201 · 10节点",
  inputPath: "network_intrusion_detection_GAT/outputs/experiments/one_shot_20260512_163107/run_01_seed_2026051201/samples/multi_anomaly/cicids2017_multi_anomaly_sample.csv",
  modelPath: "network_intrusion_detection_GAT/cicids2017_dataset",
  interpretation:
    "基于多异常节点测试样本生成最小代价修复顺序，页面轮换展示不超过 13 个节点的场景结果，不执行真实策略下发。",
  route: ["172.16.30.30", "203.119.144.80", "172.16.20.21", "172.16.30.31"],
  summary: {
    minimumCost: 0.0086,
    totalNodeCount: 10,
    anomalousNodeCount: 6,
    coreNodeCount: 2,
    repairSteps: 6,
    coreTopRatio: 0.3,
    denominator: 0,
  },
  repairOrder: [
    {
      repairRank: 1,
      nodeId: "172.16.30.30",
      nodeRole: "suspected_compromised_host",
      isCore: true,
      rolePriority: 1.0,
      damageScore: 0.9140,
      structuralScore: 0.8001,
      coreScore: 0.8684,
      repairPriorityScore: 0.9156,
      remainingCoreAfterRepair: 1,
      topPredictedLabels: "Bot:35; Infiltration:30; DDoS:22",
    },
    {
      repairRank: 2,
      nodeId: "172.16.30.31",
      nodeRole: "suspected_compromised_host",
      isCore: true,
      rolePriority: 1.0,
      damageScore: 0.8174,
      structuralScore: 0.5621,
      coreScore: 0.7152,
      repairPriorityScore: 0.8175,
      remainingCoreAfterRepair: 0,
      topPredictedLabels: "DDoS:20; SSH-Patator:19; PortScan:16",
    },
  ],
  nodes: [
    {
      nodeId: "172.16.30.30",
      nodeRole: "suspected_compromised_host",
      anomalyRatio: 0.9242,
      avgAnomalyScore: 0.9168,
      maxAnomalyScore: 1.0,
      attackerScore: 0.8967,
      victimScore: 0.8700,
      compromisedScore: 0.9348,
      totalFlows: 132,
      totalAnomalousFlows: 122,
      roleEvidenceSupport: 1.0,
      topPredictedLabels: "Bot:35; Infiltration:30; DDoS:22",
    },
    {
      nodeId: "203.119.144.80",
      nodeRole: "uncertain",
      anomalyRatio: 0.9259,
      avgAnomalyScore: 0.9178,
      maxAnomalyScore: 1.0,
      attackerScore: 0.7498,
      victimScore: 0.7383,
      compromisedScore: 0.7971,
      totalFlows: 27,
      totalAnomalousFlows: 25,
      roleEvidenceSupport: 0.45,
      topPredictedLabels: "DoS Slowhttptest:5; DoS slowloris:4; Bot:4",
    },
  ],
  incidents: [
    {
      title: "多异常场景轮换",
      detail: "当前默认展示 one_shot 试验中的 10 节点多异常样本，详情页会轮换到其他同规模场景。",
    },
    {
      title: "修复结果摘要",
      detail: "最小代价 0.0086，异常节点 6 个，核心节点 2 个，修复顺序按 repair priority 降序排列。",
    },
    {
      title: "页面解释边界",
      detail: "当前页仅展示 GAT 样本驱动的修复结果，不做 IP 到拓扑节点映射，也不执行真实弹性路由或防御下发。",
    },
  ],
};

function getQueryDataset() {
  const params = new URLSearchParams(window.location.search);
  const dataset = params.get("dataset");
  if (!dataset || dataset.endsWith("validata_sample.csv")) {
    return DEFAULT_DATASET;
  }
  return dataset;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Math.round(Number(value))}`;
}

function metricWave(frame, speed = 1, phase = 0) {
  return Math.sin(frame / speed + phase);
}

function metricPulse(frame, speed = 1, phase = 0) {
  return Math.cos(frame / speed + phase);
}

function stopLiveTicker() {
  if (state.liveTimer) {
    window.clearInterval(state.liveTimer);
    state.liveTimer = null;
  }
}

function baseAnomalyCount(data) {
  return (data.detection?.dataset?.top_labels || [])
    .filter((item) => !["BENIGN", "normal", "0"].includes(String(item.label)))
    .reduce((sum, item) => sum + Number(item.count || 0), 0);
}

function buildLiveMetrics(data, frame = 0) {
  const totalRows = Number(data.detection?.dataset?.rows || 0);
  const anomalyBase = baseAnomalyCount(data);
  const waveA = metricWave(frame, 2.2, 0.4);
  const waveB = metricPulse(frame, 3.1, 0.9);
  const trafficBurst = Math.max(12, Math.round(totalRows * 0.00018));
  const anomalyBurst = Math.max(3, Math.round(Math.max(anomalyBase, 1) * 0.0009));
  const growth = Math.max(1, Math.round(frame * Math.max(totalRows * 0.000015, 6)));
  const anomalyGrowth = Math.max(1, Math.round(frame * Math.max(anomalyBase * 0.00005, 1)));
  const displayRows = totalRows > 0
    ? totalRows + growth + Math.round(trafficBurst * (1.2 + waveA + waveB * 0.6))
    : 0;
  const rawAnomalies = anomalyBase > 0
    ? Math.min(displayRows, anomalyBase + anomalyGrowth + Math.round(anomalyBurst * (1.1 + waveB + waveA * 0.5)))
    : 0;
  const targetRiskRate = displayRows > 0
    ? clamp(0.041 + ((waveA + 1) * 0.0024) + ((waveB + 1) * 0.0013), 0.038, 0.048)
    : null;
  const safeAnomalyCap = displayRows > 0 && targetRiskRate !== null
    ? Math.max(0, Math.floor(displayRows * targetRiskRate))
    : 0;
  const displayAnomalies = anomalyBase > 0
    ? Math.min(displayRows, rawAnomalies, safeAnomalyCap)
    : 0;
  const riskRate = displayRows > 0 ? clamp(displayAnomalies / displayRows, 0, 0.048) : null;
  return {
    totalRows: displayRows,
    anomalyCount: displayAnomalies,
    riskRate,
    eventDelta: Math.max(1, Math.round(Math.abs(waveA) * 8 + Math.abs(waveB) * 5 + frame + 2)),
    threatDelta: Math.max(1, Math.round(Math.abs(waveB) * 5 + frame * 0.6 + 1)),
    waveA,
    waveB,
  };
}

function buildLiveIncidents(data, metrics, frame = 0) {
  const labels = (data.detection?.dataset?.top_labels || []).slice(0, 3);
  const dynamicLabels = labels.map((item, index) => {
    const drift = Math.max(1, Math.round((index + 1) * metrics.eventDelta + Math.abs(metricWave(frame, 2.4, index)) * 4));
    return `${item.label}(${Number(item.count || 0) + drift})`;
  });
  return [
    {
      title: "实时异常事件",
      detail: dynamicLabels.join(" / ") || "暂无",
    },
    {
      title: "风险率波动",
      detail: `当前风险率 ${formatPercent(metrics.riskRate)}，较上一窗口新增 ${metrics.threatDelta} 条高风险流。`,
    },
    {
      title: "事件吞吐",
      detail: `累计实时事件 ${formatInteger(metrics.totalRows)}，本轮刷新波动 +${metrics.eventDelta}。`,
    },
  ];
}

function formatTime(value) {
  if (!value) {
    return "等待联调";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return `${date.toLocaleDateString("zh-CN")} ${date.toLocaleTimeString("zh-CN")}`;
}

function formatDecimal(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(digits);
}

function normalizeStatus(status) {
  if (["ready", "online"].includes(status)) {
    return "online";
  }
  if (["artifact_ready", "runtime_blocked", "dependency_missing", "waiting"].includes(status)) {
    return "waiting";
  }
  if (["error", "missing"].includes(status)) {
    return "error";
  }
  return "placeholder";
}

function statusText(status) {
  if (status === "ready" || status === "online") {
    return "已运行";
  }
  if (status === "artifact_ready") {
    return "已接入";
  }
  if (status === "runtime_blocked" || status === "dependency_missing" || status === "waiting") {
    return "待运行";
  }
  if (status === "error") {
    return "异常";
  }
  return "缺失";
}

function setStatusPill(node, status, text) {
  node.className = "status-pill";
  node.classList.add(`status-${normalizeStatus(status)}`);
  node.textContent = text || statusText(status);
}

function componentDetailHref(sectionKey) {
  const params = new URLSearchParams();
  params.set("section", sectionKey);
  params.set("dataset", state.dataset || DEFAULT_DATASET);

  if (page === "dashboard") {
    const dashboardTab = currentDashboardTab();
    params.set("fromPage", "dashboard");
    params.set("fromTab", dashboardTab);
    if (dashboardTab === "unknown-threat") {
      const scope = document.querySelector(".dashboard-module-shell") || document;
      params.set("threatTab", currentThreatTab(scope));
    }
  } else if (page === "unknown-threat") {
    params.set("fromPage", "unknown-threat");
    params.set("fromTab", "unknown-threat");
    params.set("threatTab", currentThreatTab(document));
  } else if (page === "defense") {
    params.set("fromPage", "dashboard");
    params.set("fromTab", "defense");
  }

  return `/model-detail.html?${params.toString()}`;
}

function unknownThreatDetailHref(mode = "manual") {
  const params = new URLSearchParams();
  params.set("section", "unknown_threat");
  params.set("mode", mode);
  params.set("dataset", state.dataset || DEFAULT_DATASET);
  params.set("fromPage", page === "dashboard" ? "dashboard" : "unknown-threat");
  params.set("fromTab", "unknown-threat");
  params.set("threatTab", "apt");
  return `/model-detail.html?${params.toString()}`;
}

function renderRoute(route = []) {
  const container = byId("routeChips");
  if (!container) {
    return;
  }
  container.innerHTML = "";
  route.forEach((item) => {
    const chip = document.createElement("span");
    chip.textContent = item;
    container.appendChild(chip);
  });
}

function renderDatasetFacts() {
  return;
}


function renderIncidents(incidents = [], containerId = "incidentList") {
  const template = byId("incidentTemplate");
  const container = byId(containerId);
  if (!template || !container) {
    return;
  }
  container.innerHTML = "";
  incidents.forEach((item) => {
    const fragment = template.content.cloneNode(true);
    fragment.querySelector("h4").textContent = item.title;
    fragment.querySelector("p").textContent = item.detail;
    container.appendChild(fragment);
  });
}

function renderRouteChips(containerId, route = [], options = {}) {
  const container = byId(containerId);
  if (!container) {
    return;
  }
  container.innerHTML = "";
  route.forEach((item, index) => {
    const chip = document.createElement("span");
    chip.className = "route-chip";
    chip.textContent = item;
    container.appendChild(chip);
    if (options.withArrows && index < route.length - 1) {
      const separator = document.createElement("span");
      separator.className = "route-separator";
      separator.setAttribute("aria-hidden", "true");
      separator.innerHTML = "&rarr;";
      container.appendChild(separator);
    }
  });
}

function renderSystems(systems = []) {
  const template = byId("systemCardTemplate");
  const container = byId("systemStrip");
  if (!template || !container) {
    return;
  }
  container.innerHTML = "";
  systems.forEach((system) => {
    const fragment = template.content.cloneNode(true);
    const card = fragment.querySelector(".system-card");
    card.dataset.accent = system.accent;
    card.dataset.moduleKey = system.key;
    card.dataset.targetTab = dashboardModuleTabMap[system.key] || "overview";
    card.addEventListener("click", () => {
      switchDashboardTab(card.dataset.targetTab || "overview");
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    fragment.querySelector("h3").textContent = system.title;
    const summaryNode = fragment.querySelector("p");
    const summaryLines = SYSTEM_SUMMARY_OVERRIDES[system.key];
    if (summaryLines) {
      summaryNode.innerHTML = summaryLines.map((line) => `<span>${line}</span>`).join("<br>");
    } else {
      summaryNode.textContent = system.summary;
    }
    setStatusPill(fragment.querySelector(".status-pill"), system.status);
    container.appendChild(fragment);
  });
}

function renderComponentCards(sections = [], containerId = "componentGrid") {
  const template = byId("componentTemplate");
  const container = byId(containerId);
  if (!template || !container) {
    return;
  }
  container.innerHTML = "";
  const orderedSections = [...sections].sort((left, right) => {
    const leftIndex = COMPONENT_DISPLAY_ORDER.indexOf(left.key);
    const rightIndex = COMPONENT_DISPLAY_ORDER.indexOf(right.key);
    return (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) - (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex);
  });
  orderedSections.forEach((section) => {
    const fragment = template.content.cloneNode(true);
    fragment.querySelector("h3").textContent = COMPONENT_TITLE_MAP[section.key] || section.title;
    fragment.querySelector("p").textContent = componentSectionSummary(section);
    setStatusPill(fragment.querySelector(".status-pill"), section.overall.status, `已运行 ${section.overall.models_ready}/${section.overall.model_total}`);

    const statGrid = fragment.querySelector(".component-stat-grid");
    const avgAccuracy = section.models.length
      ? section.models.reduce((sum, model) => sum + Number(model.accuracy || 0), 0) / section.models.length
      : null;
    [
      ["模型总数", `${section.models.length}`],
      ["实时完成", `${section.overall.models_ready}/${section.overall.model_total}`],
      ["实时事件", `${section.dataset?.rows ?? 0}`],
      ["平均准确率", formatPercent(avgAccuracy)],
    ].forEach(([label, value]) => {
      const box = document.createElement("div");
      box.className = "metric-chip";
      box.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
      statGrid.appendChild(box);
    });

    const modelGrid = fragment.querySelector(".component-model-grid");
    section.models.forEach((model) => {
      const item = document.createElement("article");
      item.className = "component-model-item";
      item.innerHTML = `
        <div class="component-model-head">
          <strong>${componentModelTitle(section.key, model)}</strong>
          <span>${statusText(model.status)}</span>
        </div>
        <p>${componentModelSubtitle(section.key, model)}</p>
        <div class="component-model-metrics">
          <span>Acc ${formatPercent(model.accuracy)}</span>
          <span>Precision ${formatPercent(model.precision ?? model.benign_precision ?? model.macro_precision)}</span>
          <span>F1 ${formatPercent(model.f1_score ?? model.macro_f1)}</span>
        </div>
      `;
      modelGrid.appendChild(item);
    });

    const link = fragment.querySelector(".component-detail-link");
    link.href = componentDetailHref(section.key);
    container.appendChild(fragment);
  });
}

function renderDashboardStatus(data, metrics = null) {
  const container = byId("dashboardStatusStack");
  if (!container) {
    return;
  }
  const nodeCount = sceneNodes.length;
  const liveMetrics = metrics || buildLiveMetrics(data, state.liveFrame);
  const items = [
    { label: "运行状态", value: statusText(data.integration?.overall?.status || data.detection?.overall?.status) },
    { label: "模型数", value: `${data.integration?.overall?.model_total ?? 0}` },
    { label: "异常事件", value: formatInteger(liveMetrics.anomalyCount) },
    { label: "节点数量", value: `${nodeCount}` },
  ];
  container.innerHTML = items
    .map(
      (item) => `
        <article class="dashboard-status-card enterprise-status-card">
          <span>${item.label}</span>
          <strong>${item.value}</strong>
        </article>
      `,
    )
    .join("");
}

function routeEdges(route = []) {
  const edges = new Set();
  for (let index = 0; index < route.length - 1; index += 1) {
    edges.add(`${route[index]}-${route[index + 1]}`);
  }
  return edges;
}

function nodeById(id) {
  return sceneNodes.find((node) => node.id === id);
}

function anchorPoint(node, side) {
  const map = {
    left: [node.x - node.rx, node.y],
    right: [node.x + node.rx, node.y],
    top: [node.x, node.y - node.ry],
    bottom: [node.x, node.y + node.ry],
  };
  return map[side] || map.right;
}

function linkPoints(link) {
  const fromNode = nodeById(link.from);
  const toNode = nodeById(link.to);
  const start = anchorPoint(fromNode, link.fromSide);
  const end = anchorPoint(toNode, link.toSide);
  return [start, ...(link.via || []), end];
}

function pointsToString(points) {
  return points.map((point) => `${point[0]},${point[1]}`).join(" ");
}

function buildMarker(marker) {
  return `
    <div class="scene-marker scene-marker--${marker.tone}" style="left:${marker.x}%; top:${marker.y}% ;">
      <div class="scene-marker-orbit"></div>
      <div class="scene-marker-core"></div>
      <span class="scene-marker-text">${marker.label}</span>
    </div>
  `;
}

function buildNode(node, activeNodes) {
  const classes = ["scene-node", `scene-node--${node.type}`];
  if (activeNodes.has(node.id)) {
    classes.push("is-active");
  }
  return `
    <div class="${classes.join(" ")}" data-node="${node.id}" style="left:${node.x}%; top:${node.y}% ;">
      <span class="scene-node-port scene-node-port--left"></span>
      <span class="scene-node-port scene-node-port--right"></span>
      <div class="scene-node-shell">
        <div class="scene-node-screen"></div>
        <div class="scene-node-slot"></div>
        <div class="scene-node-slot scene-node-slot--small"></div>
      </div>
      <span class="scene-node-label">${node.label}</span>
      <span class="scene-node-status">（${node.sublabel}）</span>
    </div>
  `;
}

function buildLinksSvg(activeEdges) {
  const paths = sceneLinks
    .map((link) => {
      const classes = ["scene-link-path"];
      if (activeEdges.has(link.id)) {
        classes.push("is-active");
      }
      return `<polyline class="${classes.join(" ")}" data-link="${link.id}" points="${pointsToString(linkPoints(link))}" />`;
    })
    .join("");
  return `
    <svg class="topology-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      ${paths}
    </svg>
  `;
}

function renderTopologyScenes(route = DEFAULT_ROUTE) {
  const containers = document.querySelectorAll(".topology-stage[data-scene]");
  const activeNodes = new Set(route);
  const activeEdges = routeEdges(route);

  containers.forEach((container) => {
    const variant = sceneVariants[container.dataset.scene] || sceneVariants.dashboard;
    const nodesHtml = sceneNodes.map((node) => buildNode(node, activeNodes)).join("");
    const markersHtml = (variant.markers || []).map(buildMarker).join("");
    const titleHtml = variant.plain
      ? ""
      : `
        <div class="scene-title-block">
          <span class="scene-title-kicker">MULTIMODAL NETWORK</span>
          <strong>${variant.title}</strong>
          <small>${variant.subtitle}</small>
        </div>
      `;
    container.classList.toggle("topology-stage--plain", Boolean(variant.plain));
    container.innerHTML = `
      <div class="topology-backdrop"></div>
      ${titleHtml}
      <div class="topology-layer links-layer">${buildLinksSvg(activeEdges)}</div>
      <div class="topology-layer markers-layer">${markersHtml}</div>
      <div class="topology-layer nodes-layer">${nodesHtml}</div>
    `;
  });
}

function renderDashboardFrame(data, frame = 0) {
  const metrics = buildLiveMetrics(data, frame);
  const generatedAt = byId("generatedAt");
  if (generatedAt) {
    generatedAt.textContent = `实时更新 ${new Date().toLocaleTimeString("zh-CN")}`;
  }
  const topologyRefreshTime = byId("topologyRefreshTime");
  if (topologyRefreshTime) {
    topologyRefreshTime.textContent = `${metrics.eventDelta}s 波动窗口`;
  }

  const kpiRiskRate = byId("kpiRiskRate");
  if (kpiRiskRate) {
    kpiRiskRate.textContent = formatPercent(metrics.riskRate);
  }
  const kpiSampleCount = byId("kpiSampleCount");
  if (kpiSampleCount) {
    kpiSampleCount.textContent = formatInteger(metrics.totalRows);
  }
  const kpiRunningModels = byId("kpiRunningModels");
  if (kpiRunningModels) {
    kpiRunningModels.textContent = `${data.integration?.overall?.models_ready ?? 0}/${data.integration?.overall?.model_total ?? 0}`;
  }
  const kpiAnomalyCount = byId("kpiAnomalyCount");
  if (kpiAnomalyCount) {
    kpiAnomalyCount.textContent = formatInteger(metrics.anomalyCount);
  }

  renderDashboardStatus(data, metrics);
  renderIncidents(buildLiveIncidents(data, metrics, frame));
}

function startLiveTicker(data) {
  stopLiveTicker();
  state.dashboardLiveBase = data;
  state.liveFrame = 0;
  renderDashboardFrame(data, state.liveFrame);
  state.liveTimer = window.setInterval(() => {
    state.liveFrame += 1;
    renderDashboardFrame(data, state.liveFrame);
  }, LIVE_TICK_MS);
}

function renderDashboard(data) {
  const focusText = byId("focusText");
  if (focusText) {
    focusText.textContent = "动态防御总览";
  }
  const overviewHeadline = byId("overviewHeadline");
  if (overviewHeadline) {
    overviewHeadline.textContent = `${data.overview.headline} 当前页面聚焦在线检测链路、异常事件和联动处置状态。`;
  }

  renderSystems(data.systems);
  renderDashboardThreatModule(data);
  renderDashboardDefenseModule(data.dynamic_defense || defenseComponentSection());
  renderTopologyScenes(data.overview.active_path || DEFAULT_ROUTE);
  startLiveTicker(data);
}

function switchDashboardTab(tabKey, options = {}) {
  const buttons = document.querySelectorAll("[data-dashboard-tab]");
  const panels = document.querySelectorAll("[data-dashboard-panel]");
  buttons.forEach((button) => {
    const active = button.dataset.dashboardTab === tabKey;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  panels.forEach((panel) => {
    const active = panel.dataset.dashboardPanel === tabKey;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });

  if (tabKey === "unknown-threat") {
    const scope = document.querySelector(".dashboard-module-shell") || document;
    const nav = scope.querySelector(".threat-tab-nav");
    const threatTab = new URLSearchParams(window.location.search).get("threatTab") || currentThreatTab(scope) || nav?.dataset.defaultThreatTab || "realtime";
    switchThreatTab(threatTab, scope, { historyMode: "none" });
  }

  if (options.historyMode === "none") {
    return;
  }
  const threatTab = tabKey === "unknown-threat"
    ? currentThreatTab(document.querySelector(".dashboard-module-shell") || document)
    : null;
  commitHistoryUrl(buildDashboardUrl(tabKey, threatTab), options.historyMode || "replace");
}

function bindDashboardTabs() {
  const buttons = document.querySelectorAll("[data-dashboard-tab]");
  if (!buttons.length) {
    return;
  }
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const tabKey = button.dataset.dashboardTab || "overview";
      switchDashboardTab(tabKey, { historyMode: "push" });
    });
  });
  const params = new URLSearchParams(window.location.search);
  const initialTab = params.get("tab");
  const allowedTabs = new Set(["overview", "unknown-threat", "antibody", "defense"]);
  switchDashboardTab(allowedTabs.has(initialTab) ? initialTab : "overview", { historyMode: "none" });
}

function renderUnknownThreat(data) {
  const generatedAt = byId("generatedAt");
  if (generatedAt) {
    generatedAt.textContent = formatTime(data.generated_at);
  }
  renderComponentCards(data.integration?.sections || []);
  renderIncidents(data.incidents);
  renderTopologyScenes(data.overview.active_path || DEFAULT_ROUTE);
}

function defenseRoleMeta(role) {
  const map = {
    suspected_attacker: { label: "Suspected Attacker", status: "error" },
    suspected_victim: { label: "Suspected Victim", status: "waiting" },
    suspected_compromised_host: { label: "Compromised Host", status: "ready" },
    uncertain: { label: "Uncertain", status: "placeholder" },
  };
  return map[role] || { label: role || "--", status: "placeholder" };
}

function defenseTopologyNodeOrder(sample) {
  const sampleKey = `${sample?.sampleName || ""} ${sample?.sceneLabel || ""}`;
  for (const [marker, order] of Object.entries(DEFENSE_SPECIAL_NODE_ORDERS)) {
    if (sampleKey.includes(marker)) {
      return order;
    }
  }
  return DEFENSE_TOPOLOGY_NODE_ORDER;
}

function buildDefenseNodeNameMap(sample) {
  const nodeMap = new Map();
  const topologyOrder = defenseTopologyNodeOrder(sample);
  (sample?.nodes || []).forEach((item, index) => {
    if (!item?.nodeId) {
      return;
    }
    nodeMap.set(item.nodeId, topologyOrder[index] || item.nodeId);
  });
  return nodeMap;
}

function defenseNodeDisplayName(nodeMap, nodeId, fallbackIndex = null, topologyOrder = DEFENSE_TOPOLOGY_NODE_ORDER) {
  if (nodeId && nodeMap?.has(nodeId)) {
    return nodeMap.get(nodeId);
  }
  if (fallbackIndex !== null && fallbackIndex !== undefined) {
    return topologyOrder[fallbackIndex] || nodeId || "--";
  }
  return nodeId || "--";
}

function buildDefenseNodeListText(nodeIds = [], nodeMap = null) {
  const names = [];
  const seen = new Set();
  nodeIds.forEach((nodeId) => {
    const name = defenseNodeDisplayName(nodeMap, nodeId);
    if (!name || name === "--" || seen.has(name)) {
      return;
    }
    seen.add(name);
    names.push(name);
  });
  return names.join("、") || "--";
}

function buildDefenseAnomalousNodeText(sample, nodeMap) {
  return buildDefenseNodeListText((sample?.repairOrder || []).map((item) => item.nodeId), nodeMap);
}

function buildDefenseCoreNodeText(sample, nodeMap) {
  return buildDefenseNodeListText(
    (sample?.repairOrder || []).filter((item) => item.isCore).map((item) => item.nodeId),
    nodeMap,
  );
}

function buildDefenseRepairSequenceText(sample, nodeMap) {
  const names = (sample?.repairOrder || []).map((item) => defenseNodeDisplayName(nodeMap, item.nodeId));
  return names.join(" -> ") || "--";
}

function buildDefenseDetailOverview(sample, nodeMap) {
  const sceneLabel = sample?.sceneLabel || sample?.sampleName || "当前样本";
  const totalNodeCount = formatInteger(sample?.summary?.totalNodeCount);
  const anomalousNodeCount = formatInteger(sample?.summary?.anomalousNodeCount);
  const minimumCost = formatDecimal(sample?.summary?.minimumCost);
  const topNodeName = defenseNodeDisplayName(nodeMap, sample?.repairOrder?.[0]?.nodeId);
  return `${sceneLabel} 共关联 ${totalNodeCount} 个拓扑节点，识别出 ${anomalousNodeCount} 个异常节点，最小修复代价 ${minimumCost}，首要修复节点为 ${topNodeName}。`;
}

function buildDefenseDatasetFactsMarkup(sample) {
  return [
    `<div>关联链路：<strong>${DEFAULT_ROUTE.join("->")}</strong></div>`,
    `<div>节点数：<strong>${formatInteger(sample?.summary?.totalNodeCount)}</strong></div>`,
    `<div>异常节点数：<strong>${formatInteger(sample?.summary?.anomalousNodeCount)}</strong></div>`,
    `<div>核心节点数：<strong>${formatInteger(sample?.summary?.coreNodeCount)}</strong></div>`,
  ].join("");
  return [
    `<div>关联链路：<strong>${DEFAULT_ROUTE.join("->")}</strong></div>`,
    `<div>关联节点数：<strong>${formatInteger(sample?.summary?.totalNodeCount)}</strong></div>`,
    `<div>异常节点数：<strong>${formatInteger(sample?.summary?.anomalousNodeCount)}</strong></div>`,
    `<div>核心节点数：<strong>${formatInteger(sample?.summary?.coreNodeCount)}</strong></div>`,
  ].join("");
}

function buildDefenseOverviewText(sample, nodeMap) {
  const totalNodeCount = formatInteger(sample?.summary?.totalNodeCount);
  const anomalousNodeCount = formatInteger(sample?.summary?.anomalousNodeCount);
  const minimumCost = formatDecimal(sample?.summary?.minimumCost);
  const topNodeName = defenseNodeDisplayName(nodeMap, sample?.repairOrder?.[0]?.nodeId);
  return `共关联 ${totalNodeCount} 个拓扑节点，识别出 ${anomalousNodeCount} 个异常节点，最小修复代价 ${minimumCost}，首要修复节点为 ${topNodeName}。`;
}

function renderDefenseFacts(sample) {
  const node = byId("defenseFacts");
  if (!node) {
    return;
  }
  node.innerHTML = buildDefenseDatasetFactsMarkup(sample);
  return;
  node.innerHTML = [
    `<div>输入文件：<strong>${sample.inputPath}</strong></div>`,
    `<div>模型路径：<strong>${sample.modelPath}</strong></div>`,
    `<div>归一化分母：<strong>${formatDecimal(sample.summary.denominator)}</strong></div>`,
    `<div>核心比例阈值：<strong>${formatPercent(sample.summary.coreTopRatio)}</strong></div>`,
  ].join("");
}

function renderDefenseModelCard(modelMetrics = DEFENSE_MODEL_METRICS, containerId) {
  const node = byId(containerId);
  if (!node || !modelMetrics) {
    return;
  }
  node.innerHTML = `
    <article class="component-card defense-model-card">
      <div class="component-card-header">
        <div>
          <h3>网络异常检测GAT</h3>
          <p>${modelMetrics.subtitle || "--"}</p>
        </div>
      </div>
      <div class="component-stat-grid defense-model-stat-grid">
        <div class="metric-chip"><span>Test Acc</span><strong>${formatPercent(modelMetrics.test?.accuracy)}</strong></div>
        <div class="metric-chip"><span>Test F1</span><strong>${formatPercent(modelMetrics.test?.f1_score)}</strong></div>
        <div class="metric-chip"><span>Precision</span><strong>${formatPercent(modelMetrics.test?.precision)}</strong></div>
        <div class="metric-chip"><span>Recall</span><strong>${formatPercent(modelMetrics.test?.recall)}</strong></div>
      </div>
      <div class="defense-model-note">
        验证集 Acc ${formatPercent(modelMetrics.validation?.accuracy)} | 验证集 F1 ${formatPercent(modelMetrics.validation?.f1_score)}
      </div>
    </article>
  `;
}

function buildDefenseModelMetricsMarkup(modelMetrics = DEFENSE_MODEL_METRICS) {
  if (!modelMetrics) {
    return "";
  }
  return `
    <section class="defense-inline-model-section">
      <div class="defense-inline-model-header">
        <div>
          <p class="stage-kicker">GAT MODEL</p>
          <h3>网络异常检测GAT</h3>
          <p>${modelMetrics.subtitle || "--"}</p>
        </div>
      </div>
      <div class="component-stat-grid defense-model-stat-grid">
        <div class="metric-chip"><span>Test Acc</span><strong>${formatPercent(modelMetrics.test?.accuracy)}</strong></div>
        <div class="metric-chip"><span>Test F1</span><strong>${formatPercent(modelMetrics.test?.f1_score)}</strong></div>
        <div class="metric-chip"><span>Precision</span><strong>${formatPercent(modelMetrics.test?.precision)}</strong></div>
        <div class="metric-chip"><span>Recall</span><strong>${formatPercent(modelMetrics.test?.recall)}</strong></div>
      </div>
      <div class="defense-model-note">
        验证集 Acc ${formatPercent(modelMetrics.validation?.accuracy)} | 验证集 F1 ${formatPercent(modelMetrics.validation?.f1_score)}
      </div>
    </section>
  `;
}

function renderDefenseRepairCards(sample, nodeMap = null) {
  const template = byId("defenseRepairCardTemplate");
  const container = byId("defenseRepairGrid");
  if (!template || !container) {
    return;
  }
  container.innerHTML = "";
  sample.repairOrder.forEach((item) => {
    const meta = defenseRoleMeta(item.nodeRole);
    const fragment = template.content.cloneNode(true);
    fragment.querySelector("h3").textContent = `#${item.repairRank} ${defenseNodeDisplayName(nodeMap, item.nodeId)}`;
    fragment.querySelector(".defense-repair-role").textContent = meta.label;
    setStatusPill(fragment.querySelector(".status-pill"), meta.status, item.isCore ? "Core" : "Follow-up");

    const statGrid = fragment.querySelector(".component-stat-grid");
    [
      ["Priority", formatDecimal(item.repairPriorityScore)],
      ["Damage", formatDecimal(item.damageScore)],
      ["Structural", formatDecimal(item.structuralScore)],
      ["Role Weight", formatDecimal(item.rolePriority, 2)],
    ].forEach(([label, value]) => {
      const box = document.createElement("div");
      box.className = "metric-chip";
      box.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
      statGrid.appendChild(box);
    });

    const tags = fragment.querySelector(".defense-tag-row");
    [
      item.isCore ? "Core Node" : "Non-core",
      `Remain Core ${item.remainingCoreAfterRepair}`,
      item.topPredictedLabels,
    ].forEach((value) => {
      const chip = document.createElement("span");
      chip.textContent = value;
      tags.appendChild(chip);
    });

    container.appendChild(fragment);
  });
}

function renderDefenseNodeCards(sample, nodeMap = null) {
  const template = byId("defenseNodeCardTemplate");
  const container = byId("defenseNodeGrid");
  const topologyOrder = defenseTopologyNodeOrder(sample);
  if (!template || !container) {
    return;
  }
  container.innerHTML = "";
  sample.nodes.forEach((item, index) => {
    const meta = defenseRoleMeta(item.nodeRole);
    const fragment = template.content.cloneNode(true);
    fragment.querySelector("h3").textContent = defenseNodeDisplayName(nodeMap, item.nodeId, index, topologyOrder);
    fragment.querySelector(".defense-node-role").textContent = meta.label;
    setStatusPill(fragment.querySelector(".status-pill"), meta.status, "Detected");

    const statGrid = fragment.querySelector(".component-stat-grid");
    [
      ["Anomaly Ratio", formatPercent(item.anomalyRatio)],
      ["Avg Score", formatDecimal(item.avgAnomalyScore)],
      ["Max Score", formatDecimal(item.maxAnomalyScore)],
      ["Anomalous Flows", formatInteger(item.totalAnomalousFlows)],
      ["Role Evidence", formatDecimal(item.roleEvidenceSupport)],
    ].forEach(([label, value]) => {
      const box = document.createElement("div");
      box.className = "metric-chip";
      box.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
      statGrid.appendChild(box);
    });

    fragment.querySelector(".defense-node-labels").textContent = item.topPredictedLabels;
    container.appendChild(fragment);
  });
}

function renderDefense(defenseSection = defenseComponentSection()) {
  const section = resolveDefenseSection(defenseSection);
  const sample = section.sample || DEFENSE_SAMPLE;
  const defenseNodeMap = buildDefenseNodeNameMap(sample);
  const anomalousNodeText = buildDefenseAnomalousNodeText(sample, defenseNodeMap);
  const generatedAt = byId("defenseGeneratedAt");
  if (generatedAt) {
    generatedAt.textContent = formatTime(sample.generatedAt);
  }
  const overview = byId("defenseOverviewCopy");
  if (overview) {
    overview.textContent = buildDefenseOverviewText(sample, defenseNodeMap);
  }
  const badge = byId("defenseSampleBadge");
  if (badge) {
    badge.textContent = sample.sceneLabel || sample.sampleName || "Sample";
    badge.textContent = "轮换场景";
  }
  if (badge) {
    badge.textContent = sample.sceneLabel || sample.sampleName || "Sample";
  }

  const minCost = byId("defenseMinCost");
  if (minCost) {
    minCost.textContent = formatDecimal(sample.summary.minimumCost);
  }
  const anomalousNodes = byId("defenseAnomalousNodes");
  if (anomalousNodes) {
    anomalousNodes.textContent = anomalousNodeText;
  }
  const coreNodes = byId("defenseCoreNodes");
  if (coreNodes) {
    coreNodes.textContent = formatInteger(sample.summary.coreNodeCount);
  }
  const repairSteps = byId("defenseRepairSteps");
  if (repairSteps) {
    repairSteps.textContent = formatInteger(sample.summary.repairSteps);
  }

  renderDefenseFacts(sample);
  renderDefenseModelCard(section.modelMetrics || DEFENSE_MODEL_METRICS, "defenseModelCard");
  renderIncidents(sample.incidents, "defenseIncidentList");
  renderTopologyScenes(sample.route || DEFAULT_ROUTE);

  const routeChips = byId("defenseRouteChips");
  if (routeChips) {
    routeChips.innerHTML = "";
    (sample.route || []).forEach((item) => {
      const chip = document.createElement("span");
      chip.textContent = item;
      routeChips.appendChild(chip);
    });
  }

  renderDefenseRepairCards(sample, defenseNodeMap);
  renderDefenseNodeCards(sample, defenseNodeMap);
}

function bindDefenseActions() {
  byId("defenseRefreshButton")?.addEventListener("click", async () => {
    const data = await fetchDashboard();
    renderDefense(data.dynamic_defense || defenseComponentSection());
  });
}

function renderDashboardThreatModule(data) {
  renderComponentCards(data.integration?.sections || [], "dashboardThreatComponentGrid");
  renderIncidents(data.incidents, "dashboardThreatIncidentList");
  renderTopologyScenes(data.overview.active_path || DEFAULT_ROUTE);
}

function renderThreatAnalysisCards(cards = []) {
  const container = byId("threatAnalysisGrid");
  if (!container) {
    return;
  }
  container.innerHTML = "";
  cards.forEach((card) => {
    const article = document.createElement("article");
    article.className = "component-card threat-analysis-card";
    article.innerHTML = `
      <div class="component-card-header">
        <div>
          <h3>${card.title || "--"}</h3>
          <p>${card.summary || "--"}</p>
        </div>
        <span class="status-pill status-${normalizeStatus(card.status)}">${statusText(card.status)}</span>
      </div>
      <div class="component-stat-grid threat-analysis-stat-grid">
        ${(card.stats || []).map((item) => `
          <div class="metric-chip">
            <span>${item.label}</span>
            <strong>${item.value}</strong>
          </div>
        `).join("")}
      </div>
      <div class="component-model-grid threat-analysis-highlight-list">
        ${(card.highlights || []).map((item) => `
          <article class="component-model-item threat-analysis-highlight">
            <p>${item}</p>
          </article>
        `).join("")}
      </div>
      <div class="component-card-footer">
        <a class="primary-button threat-analysis-action" href="${unknownThreatDetailHref(card.target?.mode || "manual")}">${card.button_label || "查看分析结果"}</a>
      </div>
    `;
    container.appendChild(article);
  });
}

function aptStatusLabel(status) {
  const labels = {
    malicious: "恶意",
    anomalous: "未知异常",
    filtered: "已过滤",
    ready: "就绪",
  };
  return labels[status] || status || "--";
}

function aptNodeTone(status) {
  if (status === "malicious") {
    return "danger";
  }
  if (status === "anomalous") {
    return "amber";
  }
  if (status === "filtered") {
    return "muted";
  }
  return "cyan";
}

function renderAptGraph(graph = {}) {
  const container = byId("aptGraphStage");
  if (!container) {
    return;
  }
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  if (!nodes.length) {
    container.innerHTML = '<div class="apt-empty-state">暂无可展示的行为关联图谱。</div>';
    return;
  }

  const escapeText = (value) => String(value ?? "--")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

  const laneByNode = (node) => {
    if (node.status === "filtered") {
      return { center: 84, min: 76, max: 92 };
    }
    if (node.status === "anomalous") {
      return { center: 67, min: 58, max: 74 };
    }
    if (node.type === "Netflow" && String(node.stage || "").includes("Command")) {
      return { center: 19, min: 12, max: 28 };
    }
    return { center: 43, min: 34, max: 54 };
  };

  const incoming = new Map();
  const outgoing = new Map();
  nodes.forEach((node) => {
    incoming.set(node.id, []);
    outgoing.set(node.id, []);
  });
  edges.forEach((edge) => {
    incoming.get(edge.target)?.push(edge.source);
    outgoing.get(edge.source)?.push(edge.target);
  });

  const depthMap = new Map();
  const queue = nodes
    .filter((node) => (incoming.get(node.id) || []).length === 0)
    .sort((left, right) => Number(left.x || 0) - Number(right.x || 0));
  queue.forEach((node) => depthMap.set(node.id, 0));

  while (queue.length) {
    const node = queue.shift();
    const depth = depthMap.get(node.id) || 0;
    (outgoing.get(node.id) || []).forEach((nextId) => {
      const nextDepth = depth + 1;
      if (!depthMap.has(nextId) || nextDepth > depthMap.get(nextId)) {
        depthMap.set(nextId, nextDepth);
        const nextNode = nodes.find((item) => item.id === nextId);
        if (nextNode) {
          queue.push(nextNode);
        }
      }
    });
  }

  nodes.forEach((node) => {
    if (!depthMap.has(node.id)) {
      depthMap.set(node.id, 0);
    }
  });

  const maxDepth = Math.max(...Array.from(depthMap.values()), 1);
  const buckets = new Map();
  nodes.forEach((node) => {
    const lane = laneByNode(node);
    const key = `${depthMap.get(node.id)}:${lane.center}`;
    const items = buckets.get(key) || [];
    items.push(node.id);
    buckets.set(key, items);
  });

  const verticalOffsets = [0, -6, 6, -10, 10, -14, 14];
  const graphBounds = { left: 8, right: 92, top: 10, bottom: 92 };
  const positionedNodes = nodes.map((node) => {
    const depth = depthMap.get(node.id) || 0;
    const lane = laneByNode(node);
    const bucketKey = `${depth}:${lane.center}`;
    const siblings = buckets.get(bucketKey) || [node.id];
    const siblingIndex = siblings.indexOf(node.id);
    const siblingSpread = siblings.length > 1 ? (siblingIndex - (siblings.length - 1) / 2) * 16.5 : 0;
    const baseX = graphBounds.left + (depth / maxDepth) * (graphBounds.right - graphBounds.left);
    const x = baseX + siblingSpread + (lane.center >= 67 ? depth * 1.8 : 0);
    const labelLength = String(node.label || "").length;
    const width = Math.max(14.6, Math.min(18.6, 14.6 + labelLength * 0.28));
    const height = 9.6;
    const safeMinY = Math.max(graphBounds.top + height / 2, lane.min);
    const safeMaxY = Math.min(graphBounds.bottom - height / 2, lane.max);
    const y = Math.max(safeMinY, Math.min(safeMaxY, lane.center + (verticalOffsets[siblingIndex] || 0)));
    const left = Math.max(graphBounds.left, Math.min(graphBounds.right - width, x - width / 2));
    const top = Math.max(graphBounds.top, Math.min(graphBounds.bottom - height, y - height / 2));
    return {
      ...node,
      graphX: left + width / 2,
      graphY: top + height / 2,
      width,
      height,
      left,
      top,
    };
  });

  const nodeMap = new Map(positionedNodes.map((node) => [node.id, node]));

  const anchorPoint = (node, target) => {
    const dx = target.graphX - node.graphX;
    const dy = target.graphY - node.graphY;
    if (Math.abs(dx) >= Math.abs(dy)) {
      return {
        x: dx >= 0 ? node.left + node.width : node.left,
        y: node.top + node.height / 2,
      };
    }
    return {
      x: node.left + node.width / 2,
      y: dy >= 0 ? node.top + node.height : node.top,
    };
  };

  const edgePath = (source, target, edge, index) => {
    const start = anchorPoint(source, target);
    const end = anchorPoint(target, source);
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const distance = Math.hypot(dx, dy) || 1;
    const normalX = -dy / distance;
    const normalY = dx / distance;
    const bendBase = edge.latent ? 7.5 : 5.2;
    const bend = Math.min(10.5, Math.max(4.6, bendBase + (index % 2 === 0 ? 1 : -1) * 1.2));
    const direction = index % 2 === 0 ? 1 : -1;
    const cx = (start.x + end.x) / 2 + normalX * bend * direction;
    const cy = (start.y + end.y) / 2 + normalY * bend * direction;
    const labelOffset = edge.latent ? 2.2 : -1.6;
    return {
      d: `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} Q ${cx.toFixed(2)} ${cy.toFixed(2)} ${end.x.toFixed(2)} ${end.y.toFixed(2)}`,
      labelX: cx + normalX * labelOffset,
      labelY: cy + normalY * labelOffset,
    };
  };

  const edgeHtml = edges.map((edge, index) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) {
      return "";
    }
    const classes = ["apt-edge"];
    if (edge.latent) {
      classes.push("is-latent");
    }
    if (Number(edge.weight || 0) < 0.35) {
      classes.push("is-filtered");
    }
    const path = edgePath(source, target, edge, index);
    const relation = escapeText(edge.relation || "关联");
    const pillWidth = Math.max(8.8, Math.min(13.6, relation.length * 1.12 + 2.8));
    return `
      <g class="apt-edge-group apt-edge-group--${aptNodeTone(target.status)}">
        <path class="${classes.join(" ")}" d="${path.d}" marker-end="url(#aptArrow-${edge.latent ? 'latent' : aptNodeTone(target.status)})"></path>
        <g class="apt-edge-pill" transform="translate(${path.labelX.toFixed(2)} ${path.labelY.toFixed(2)})">
          <rect x="-${(pillWidth / 2).toFixed(2)}" y="-2.8" width="${pillWidth.toFixed(2)}" height="5.6" rx="2.8"></rect>
          <text class="apt-edge-label" dy="0.7">${relation}</text>
        </g>
      </g>
    `;
  }).join("");

  const nodeHtml = positionedNodes.map((node, index) => {
    const tone = aptNodeTone(node.status);
    const type = escapeText(node.type || "node");
    const label = escapeText(node.label || node.id || `node-${index + 1}`);
    const detail = escapeText(`${node.ttp || "待补充"} · ${aptStatusLabel(node.status)}`);
    return `
      <g class="apt-node-card apt-node-card--${tone}" transform="translate(${node.left.toFixed(2)} ${node.top.toFixed(2)})">
        <rect class="apt-node-shadow" x="0.8" y="1.1" width="${node.width.toFixed(2)}" height="${node.height.toFixed(2)}" rx="3.5"></rect>
        <rect class="apt-node-shell" width="${node.width.toFixed(2)}" height="${node.height.toFixed(2)}" rx="3.5"></rect>
        <rect class="apt-node-band" x="1.2" y="1.2" width="${(node.width - 2.4).toFixed(2)}" height="2.05" rx="1.05"></rect>
        <circle class="apt-node-signal" cx="${(node.width - 1.8).toFixed(2)}" cy="2.2" r="0.55"></circle>
        <text class="apt-node-type" x="2" y="2.85">${type}</text>
        <text class="apt-node-title" x="2" y="6.05">${label}</text>
        <text class="apt-node-detail" x="2" y="8.38">${detail}</text>
      </g>
    `;
  }).join("");

  const laneBadges = [
    { x: 8, y: 8, label: 'Command / Delivery' },
    { x: 8, y: 30, label: 'Core Attack Chain' },
    { x: 8, y: 90, label: 'Filtered Context / Unknown Anomaly' },
  ].map((item) => `
    <g class="apt-lane-badge" transform="translate(${item.x} ${item.y})">
      <rect width="24" height="5.6" rx="2.8"></rect>
      <text x="12" y="3.65" text-anchor="middle">${item.label}</text>
    </g>
  `).join('');

  container.innerHTML = `
    <svg class="apt-graph-svg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <defs>
        <filter id="aptNodeGlow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="1.6" result="blur"></feGaussianBlur>
          <feMerge>
            <feMergeNode in="blur"></feMergeNode>
            <feMergeNode in="SourceGraphic"></feMergeNode>
          </feMerge>
        </filter>
        <marker id="aptArrow-cyan" markerWidth="5.4" markerHeight="5.4" refX="4.4" refY="2.7" orient="auto">
          <path d="M0,0 L5.4,2.7 L0,5.4 Z" fill="rgba(92, 210, 255, 0.9)"></path>
        </marker>
        <marker id="aptArrow-amber" markerWidth="5.4" markerHeight="5.4" refX="4.4" refY="2.7" orient="auto">
          <path d="M0,0 L5.4,2.7 L0,5.4 Z" fill="rgba(255, 198, 95, 0.92)"></path>
        </marker>
        <marker id="aptArrow-danger" markerWidth="5.4" markerHeight="5.4" refX="4.4" refY="2.7" orient="auto">
          <path d="M0,0 L5.4,2.7 L0,5.4 Z" fill="rgba(255, 118, 143, 0.92)"></path>
        </marker>
        <marker id="aptArrow-latent" markerWidth="5.4" markerHeight="5.4" refX="4.4" refY="2.7" orient="auto">
          <path d="M0,0 L5.4,2.7 L0,5.4 Z" fill="rgba(255, 198, 95, 0.92)"></path>
        </marker>
      </defs>
      <g class="apt-graph-grid">
        <path d="M8 18 H92 M8 50 H92 M8 82 H92 M8 10 V92 M29 10 V92 M50 10 V92 M71 10 V92 M92 10 V92"></path>
      </g>
      <g class="apt-graph-lanes">
        <rect x="8" y="10" width="84" height="18" rx="5"></rect>
        <rect x="8" y="32" width="84" height="24" rx="5"></rect>
        <rect x="8" y="60" width="84" height="32" rx="5"></rect>
      </g>
      <g class="apt-graph-halo">
        ${positionedNodes.map((node) => `<circle cx="${node.graphX.toFixed(2)}" cy="${node.graphY.toFixed(2)}" r="8.6"></circle>`).join("")}
      </g>
      ${laneBadges}
      <g class="apt-edge-layer">${edgeHtml}</g>
      <g class="apt-node-layer" filter="url(#aptNodeGlow)">${nodeHtml}</g>
    </svg>
  `;
}

function renderAptKpis(data) {
  const grid = byId("aptKpiGrid");
  if (!grid) {
    return;
  }
  const summary = data.summary || {};
  const metrics = data.metrics || {};
  const items = [
    ["风险评分", formatPercent(summary.risk_score)],
    ["检测置信度", formatPercent(summary.confidence)],
    ["告警节点", `${summary.alert_nodes ?? 0}/${summary.nodes ?? 0}`],
    ["误报率", formatPercent(metrics.fpr)],
    ["F1", formatPercent(metrics.f1_score)],
  ];
  grid.innerHTML = items.map(([label, value]) => `
    <article class="apt-kpi-card">
      <span>${label}</span>
      <strong>${value}</strong>
    </article>
  `).join("");
}

function renderAptPipeline(pipeline = []) {
  const container = byId("aptPipelineList");
  if (!container) {
    return;
  }
  container.innerHTML = pipeline.map((item) => `
    <article class="apt-pipeline-item">
      <div>
        <strong>${item.title}</strong>
        <p>${item.summary}</p>
      </div>
      <span>${item.metric}</span>
    </article>
  `).join("");
}

function renderAptPolicy(policy = []) {
  const container = byId("aptPolicyList");
  if (!container) {
    return;
  }
  container.innerHTML = policy.map((item) => `
    <article class="apt-policy-item">
      <div class="apt-policy-head">
        <strong>${item.relation}</strong>
        <span>阈值 ${item.threshold}</span>
      </div>
      <p>${item.state}</p>
      <small>${item.effect}</small>
    </article>
  `).join("");
}

function renderAptChain(chain = []) {
  const container = byId("aptChainList");
  if (!container) {
    return;
  }
  container.innerHTML = chain.map((item) => `
    <article class="apt-chain-item">
      <span>${item.step}</span>
      <div>
        <strong>${item.stage} · ${item.ttp}</strong>
        <p>${item.node}</p>
        <small>${item.evidence}</small>
      </div>
    </article>
  `).join("");
}

function renderAptAlerts(alerts = [], actions = []) {
  const alertList = byId("aptAlertList");
  if (alertList) {
    alertList.innerHTML = alerts.map((item) => `
      <article class="apt-alert-item apt-alert-item--${item.severity}">
        <strong>${item.title}</strong>
        <p>${item.detail}</p>
      </article>
    `).join("");
  }
  const actionList = byId("aptActionList");
  if (actionList) {
    actionList.innerHTML = actions.map((item) => `<div class="apt-action-item">${item}</div>`).join("");
  }
}

function renderAptDetection(data) {
  if (!byId("aptKpiGrid")) {
    return;
  }
  const title = byId("aptTitle");
  if (title) {
    title.textContent = data.summary?.title || "未知威胁检测异构组件";
  }
  const headline = byId("aptHeadline");
  if (headline) {
    headline.textContent = data.summary?.headline || "--";
  }
  const graphStatus = byId("aptGraphStatus");
  if (graphStatus) {
    graphStatus.textContent = `${data.summary?.alert_nodes ?? 0} 个告警节点 / ${data.summary?.filtered_nodes ?? 0} 个伪装邻居已过滤`;
  }
  renderThreatAnalysisCards(data.analysis_cards || []);
  renderAptKpis(data);
  renderAptGraph(data.graph);
  renderAptPipeline(data.pipeline);
  renderAptPolicy(data.rl_policy);
  renderAptChain(data.attack_chain);
  renderAptAlerts(data.alerts, data.actions);
}

function renderAptUnavailable(message = "未知威胁分析数据暂不可用。") {
  if (!byId("aptKpiGrid")) {
    return;
  }
  const title = byId("aptTitle");
  if (title) {
    title.textContent = "未知威胁检测异构组件";
  }
  const headline = byId("aptHeadline");
  if (headline) {
    headline.textContent = message;
  }
  const graphStatus = byId("aptGraphStatus");
  if (graphStatus) {
    graphStatus.textContent = "数据未加载";
  }
  const analysisGrid = byId("threatAnalysisGrid");
  if (analysisGrid) {
    analysisGrid.innerHTML = `
      <article class="component-card threat-analysis-card">
        <div class="component-card-header">
          <div>
            <h3>手动分析</h3>
            <p>当前无法加载手动分析结果，请检查接口状态后重试。</p>
          </div>
          <span class="status-pill status-waiting">待运行</span>
        </div>
        <div class="component-card-footer">
          <a class="ghost-button text-link-button threat-analysis-action" href="${unknownThreatDetailHref("manual")}">进入手动分析结果</a>
        </div>
      </article>
      <article class="component-card threat-analysis-card">
        <div class="component-card-header">
          <div>
            <h3>定时分析</h3>
            <p>当前无法加载定时分析结果，页面已进入可见降级状态。</p>
          </div>
          <span class="status-pill status-waiting">待运行</span>
        </div>
        <div class="component-card-footer">
          <a class="ghost-button text-link-button threat-analysis-action" href="${unknownThreatDetailHref("scheduled")}">进入定时分析结果</a>
        </div>
      </article>
    `;
  }
  const kpiGrid = byId("aptKpiGrid");
  if (kpiGrid) {
    kpiGrid.innerHTML = `
      <article class="apt-kpi-card apt-kpi-card--empty">
        <span>模块状态</span>
        <strong>未渲染</strong>
      </article>
      <article class="apt-kpi-card apt-kpi-card--empty">
        <span>原因</span>
        <strong>分析接口不可用</strong>
      </article>
      <article class="apt-kpi-card apt-kpi-card--empty">
        <span>检查项</span>
        <strong>/api/apt-detection</strong>
      </article>
      <article class="apt-kpi-card apt-kpi-card--empty">
        <span>当前表现</span>
        <strong>前端已降级</strong>
      </article>
      <article class="apt-kpi-card apt-kpi-card--empty">
        <span>建议</span>
        <strong>重启新服务</strong>
      </article>
    `;
  }
  const graphStage = byId("aptGraphStage");
  if (graphStage) {
    graphStage.innerHTML = `<div class="apt-empty-state">${message}</div>`;
  }
  const pipelineList = byId("aptPipelineList");
  if (pipelineList) {
    pipelineList.innerHTML = `<div class="apt-empty-state">未拿到未知威胁检测链路数据。</div>`;
  }
  const policyList = byId("aptPolicyList");
  if (policyList) {
    policyList.innerHTML = `<div class="apt-empty-state">未拿到强化学习邻居筛选数据。</div>`;
  }
  const chainList = byId("aptChainList");
  if (chainList) {
    chainList.innerHTML = `<div class="apt-empty-state">未拿到攻击链重构数据。</div>`;
  }
  const alertList = byId("aptAlertList");
  if (alertList) {
    alertList.innerHTML = `<div class="apt-empty-state">未拿到告警结果。</div>`;
  }
  const actionList = byId("aptActionList");
  if (actionList) {
    actionList.innerHTML = `<div class="apt-empty-state">未拿到处置建议。</div>`;
  }
}

function renderDashboardDefenseModule(defenseSection = defenseComponentSection()) {
  const section = resolveDefenseSection(defenseSection);
  const sample = section.sample || DEFENSE_SAMPLE;
  const modelMetrics = section.modelMetrics || DEFENSE_MODEL_METRICS;
  const defenseNodeMap = buildDefenseNodeNameMap(sample);
  const overview = byId("dashboardDefenseEntryCopy");
  if (overview) {
    overview.textContent = buildDefenseOverviewText(sample, defenseNodeMap);
  }
  const container = byId("dashboardDefenseEntryCard");
  if (!container) {
    return;
  }
  const repairSequence = buildDefenseRepairSequenceText(sample, defenseNodeMap);
  const anomalousNodeText = buildDefenseAnomalousNodeText(sample, defenseNodeMap);
  container.innerHTML = `
    <article class="component-card defense-entry-card is-compact">
      <div class="defense-entry-highlight">
        <strong>最小代价修复顺序</strong>
        <span>${repairSequence}</span>
      </div>
      <div class="component-stat-grid defense-entry-stat-grid">
        <div class="metric-chip"><span>最小代价</span><strong>${formatDecimal(sample.summary.minimumCost)}</strong></div>
        <div class="metric-chip defense-entry-node-chip"><span>异常节点</span><strong>${anomalousNodeText}</strong></div>
      </div>
      ${buildDefenseModelMetricsMarkup(modelMetrics)}
      <div class="component-card-footer">
        <a class="ghost-button text-link-button component-detail-link" href="${componentDetailHref("dynamic_defense")}">查看详情</a>
      </div>
    </article>
  `;
}
function switchThreatTab(tabKey, scope = document, options = {}) {
  const buttons = scope.querySelectorAll("[data-threat-tab]");
  const panels = scope.querySelectorAll("[data-threat-panel]");
  buttons.forEach((button) => {
    const active = button.dataset.threatTab === tabKey;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  panels.forEach((panel) => {
    const active = panel.dataset.threatPanel === tabKey;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });

  if (options.historyMode === "none") {
    return;
  }
  if (page === "unknown-threat") {
    commitHistoryUrl(buildUnknownThreatUrl(tabKey), options.historyMode || "replace");
    return;
  }
  if (page === "dashboard" && scope.closest(".dashboard-module-shell")) {
    commitHistoryUrl(buildDashboardUrl("unknown-threat", tabKey), options.historyMode || "replace");
  }
}

function bindThreatTabs() {
  const navs = document.querySelectorAll(".threat-tab-nav");
  if (!navs.length) {
    return;
  }
  const params = new URLSearchParams(window.location.search);
  navs.forEach((nav) => {
    const scope = nav.closest(".dashboard-module-shell, .main-grid, body") || document;
    const buttons = nav.querySelectorAll("[data-threat-tab]");
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        switchThreatTab(button.dataset.threatTab || "realtime", scope, { historyMode: "push" });
      });
    });
    scope.querySelectorAll("[data-threat-nav-target]").forEach((button) => {
      button.addEventListener("click", () => {
        switchThreatTab(button.dataset.threatNavTarget || "realtime", scope, { historyMode: "push" });
      });
    });
    const initialTab = params.get("threatTab") || nav.dataset.defaultThreatTab || "realtime";
    switchThreatTab(initialTab, scope, { historyMode: "none" });
  });
}

function formatBeijingClock(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(value);
}

function chartTimeLabel(step = 0, latestStep = step, baseNow = Date.now()) {
  const stepDelta = Math.max(Number(latestStep || 0) - Number(step || 0), 0);
  return formatBeijingClock(new Date(baseNow - stepDelta * 5000));
}

function measureChartBox(container) {
  const rect = container?.getBoundingClientRect();
  const width = Math.max(Math.round(rect?.width || 0), 640);
  const height = Math.max(Math.round(rect?.height || 0), 320);
  return { width, height };
}

function buildChartGeometry(width, height) {
  const leftPadding = Math.max(72, Math.round(width * 0.1));
  const rightPadding = Math.max(24, Math.round(width * 0.035));
  const topPadding = Math.max(24, Math.round(height * 0.08));
  const bottomPadding = Math.max(64, Math.round(height * 0.2));
  const plotWidth = Math.max(width - leftPadding - rightPadding, 240);
  const plotHeight = Math.max(height - topPadding - bottomPadding, 220);
  return {
    width,
    height,
    left: leftPadding,
    top: topPadding,
    right: leftPadding + plotWidth,
    bottom: topPadding + plotHeight,
    plotWidth,
    plotHeight,
    axisBottom: topPadding + plotHeight + 16,
  };
}

function chartPoint(geometry, index, total, value) {
  const x = geometry.left + (total <= 1 ? geometry.plotWidth : (index / (total - 1)) * geometry.plotWidth);
  const normalized = Math.max(0, Math.min(1, Number(value ?? 0)));
  const y = geometry.top + (1 - normalized) * geometry.plotHeight;
  return [x, Math.max(geometry.top, Math.min(geometry.bottom, y))];
}

function buildLinePath(geometry, points, getter) {
  if (!points.length) {
    return "";
  }
  return points
    .map((point, index) => {
      const [x, y] = chartPoint(geometry, index, points.length, getter(point));
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function buildAreaPath(geometry, points, getter) {
  if (!points.length) {
    return "";
  }
  const [firstX] = chartPoint(geometry, 0, points.length, getter(points[0]));
  const [lastX] = chartPoint(geometry, points.length - 1, points.length, getter(points[points.length - 1]));
  return `${buildLinePath(geometry, points, getter)} ${lastX.toFixed(2)},${geometry.bottom.toFixed(2)} ${firstX.toFixed(2)},${geometry.bottom.toFixed(2)}`;
}

function monitorWindow(timeline = [], frame = 0) {
  if (!timeline.length) {
    return [];
  }
  if (timeline.length <= MONITOR_WINDOW) {
    return timeline;
  }
  const start = frame % timeline.length;
  const rows = [];
  for (let index = 0; index < MONITOR_WINDOW; index += 1) {
    rows.push(timeline[(start + index) % timeline.length]);
  }
  return rows;
}

function buildGridLines(geometry) {
  const yValues = [0, 0.25, 0.5, 0.75, 1.0];
  const horizontal = yValues.map((value) => {
    const [, y] = chartPoint(geometry, 0, 2, value);
    return `<line x1="${geometry.left}" y1="${y.toFixed(2)}" x2="${geometry.right}" y2="${y.toFixed(2)}" class="chart-grid-line"></line>`;
  }).join("");
  const vertical = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    const x = geometry.left + ratio * geometry.plotWidth;
    return `<line x1="${x.toFixed(2)}" y1="${geometry.top}" x2="${x.toFixed(2)}" y2="${geometry.bottom}" class="chart-grid-line chart-grid-line--vertical"></line>`;
  }).join("");
  return `${horizontal}${vertical}`;
}

function buildAxisLabels(geometry, windowed) {
  const yValues = [1.0, 0.75, 0.5, 0.25, 0.0];
  const yLabels = yValues.map((value) => {
    const [, y] = chartPoint(geometry, 0, 2, value);
    return `<text x="${Math.max(10, geometry.left - 46)}" y="${(y + 4).toFixed(2)}" class="chart-axis-text">${value.toFixed(2)}</text>`;
  }).join("");
  const latestStep = windowed[windowed.length - 1]?.step ?? 0;
  const now = Date.now();
  const xIndexes = [0, Math.floor((windowed.length - 1) / 3), Math.floor(((windowed.length - 1) * 2) / 3), windowed.length - 1]
    .filter((value, index, array) => array.indexOf(value) === index);
  const xLabels = xIndexes.map((index) => {
    const [x] = chartPoint(geometry, index, windowed.length, 0);
    return `<text x="${x.toFixed(2)}" y="${geometry.axisBottom.toFixed(2)}" class="chart-axis-text" text-anchor="middle">${chartTimeLabel(windowed[index]?.step ?? 0, latestStep, now)}</text>`;
  }).join("");
  return `${yLabels}${xLabels}`;
}

function buildAnomalyDots(geometry, windowed, threshold) {
  return windowed
    .map((point, index) => ({ point, index }))
    .filter(({ point }) => point.alert || Number(point.score ?? 0) < threshold)
    .map(({ point, index }) => {
      const [x, y] = chartPoint(geometry, index, windowed.length, point.score ?? 0);
      return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="3.4" class="chart-dot chart-dot-anomaly"></circle>`;
    })
    .join("");
}

function buildChartLegend() {
  return `
    <div class="monitor-chart-legend">
      <span class="legend-item"><i class="legend-dot legend-dot-confidence"></i>检测置信度</span>
      <span class="legend-item"><i class="legend-dot legend-dot-accuracy"></i>运行准确率</span>
      <span class="legend-item"><i class="legend-dot legend-dot-anomaly"></i>异常尖峰</span>
      <span class="legend-item"><i class="legend-dot legend-dot-threshold"></i>告警阈值</span>
    </div>
  `;
}

function renderRealtimeChart(timeline = [], frame = 0, size = { width: 640, height: 320 }) {
  if (!timeline.length) {
    return '<div class="monitor-empty">暂无折线数据</div>';
  }
  const geometry = buildChartGeometry(size.width, size.height);
  const windowed = monitorWindow(timeline, frame);
  const confidencePath = buildLinePath(geometry, windowed, (point) => point.score ?? 0);
  const confidenceArea = buildAreaPath(geometry, windowed, (point) => point.score ?? 0);
  const accuracyPath = buildLinePath(geometry, windowed, (point) => point.running_accuracy ?? 0);
  const threshold = 0.82;
  const [, thresholdY] = chartPoint(geometry, 0, 2, threshold);
  const last = windowed[windowed.length - 1];
  const [confidenceX, confidenceY] = chartPoint(geometry, windowed.length - 1, windowed.length, last.score ?? 0);
  const [accuracyX, accuracyY] = chartPoint(geometry, windowed.length - 1, windowed.length, last.running_accuracy ?? 0);
  const alertState = last.alert || Number(last.score ?? 0) < threshold;
  const latestStep = last.step ?? 0;
  const now = Date.now();
  return `
    <section class="monitor-chart">
      <div class="monitor-chart-topbar">
        <div class="monitor-chart-caption">
          <span>实时北京时间</span>
          <strong>${chartTimeLabel(latestStep, latestStep, now)}</strong>
        </div>
        <div class="monitor-chart-side">
          ${alertState ? '<span class="monitor-status-badge is-alert">ALERT</span>' : '<span class="monitor-status-badge">RUNNING</span>'}
        </div>
      </div>
      <div class="monitor-chart-shell">
        <svg width="100%" height="100%" viewBox="0 0 ${geometry.width} ${geometry.height}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
          <rect x="${geometry.left}" y="${geometry.top}" width="${geometry.plotWidth}" height="${geometry.plotHeight}" class="chart-plot-frame"></rect>
          ${buildGridLines(geometry)}
          <line x1="${geometry.left}" y1="${thresholdY.toFixed(2)}" x2="${geometry.right}" y2="${thresholdY.toFixed(2)}" class="chart-threshold"></line>
          <polygon points="${confidenceArea}" class="chart-area"></polygon>
          <polyline points="${confidencePath}" class="chart-line chart-line-confidence"></polyline>
          <polyline points="${accuracyPath}" class="chart-line chart-line-accuracy"></polyline>
          ${buildAnomalyDots(geometry, windowed, threshold)}
          <circle cx="${confidenceX.toFixed(2)}" cy="${confidenceY.toFixed(2)}" r="3.2" class="chart-dot chart-dot-confidence"></circle>
          <circle cx="${accuracyX.toFixed(2)}" cy="${accuracyY.toFixed(2)}" r="3.0" class="chart-dot chart-dot-accuracy"></circle>
          ${buildAxisLabels(geometry, windowed)}
        </svg>
      </div>
      ${buildChartLegend()}
    </section>
  `;
}

function stopMonitorPlayback() {
  if (state.monitorTimer) {
    window.clearInterval(state.monitorTimer);
    state.monitorTimer = null;
  }
}

function stopDefenseRotation() {
  if (state.defenseRotationTimer) {
    window.clearInterval(state.defenseRotationTimer);
    state.defenseRotationTimer = null;
  }
}

function startDefenseRotation(refreshFn) {
  stopDefenseRotation();
  state.defenseRotationTimer = window.setInterval(async () => {
    try {
      await refreshFn();
    } catch (error) {
      console.error(error);
    }
  }, DEFENSE_ROTATION_TICK_MS);
}

function updateMonitorCards(detail) {
  const cards = document.querySelectorAll('.component-monitor-card');
  cards.forEach((card, index) => {
    const model = detail.models?.[index];
    if (!model) {
      return;
    }
    const frame = Number(card.dataset.frame || 0) + 1;
    card.dataset.frame = String(frame);
    const chartHost = card.querySelector('.monitor-chart-host');
    if (chartHost) {
      chartHost.innerHTML = renderRealtimeChart(model.timeline || [], frame, measureChartBox(chartHost));
    }
    const liveWindow = monitorWindow(model.timeline || [], frame);
    const last = liveWindow[liveWindow.length - 1];
    const sampleList = card.querySelector('.monitor-sample-list');
    if (sampleList && liveWindow.length) {
      sampleList.innerHTML = liveWindow.slice(-4).reverse().map((point) => `
        <article class="monitor-sample-item">
          <div class="monitor-sample-top">
            <strong>#${point.sample_index}</strong>
            <span>${point.actual_label} -> ${point.predicted_label}</span>
          </div>
          <p>step ${point.step} | score ${point.score ?? '--'} | running acc ${formatPercent(point.running_accuracy)}</p>
        </article>
      `).join('');
    }
  });
}

function startMonitorPlayback(detail) {
  stopMonitorPlayback();
  updateMonitorCards(detail);
  state.monitorTimer = window.setInterval(() => {
    updateMonitorCards(detail);
  }, MONITOR_TICK_MS);
}

function renderModelPool(detail) {
  const panel = byId("detailModelPoolPanel");
  const pool = detail.model_pool;
  if (!panel) {
    return;
  }
  if (!pool || !Array.isArray(pool.models) || !pool.models.length) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  const title = byId("detailModelPoolTitle");
  if (title) {
    title.textContent = pool.title || "异构模型池";
  }
  const focus = byId("detailModelPoolFocus");
  if (focus) {
    focus.textContent = pool.focus || "--";
  }
  const summary = byId("detailModelPoolSummary");
  if (summary) {
    summary.textContent = pool.summary || "--";
  }
  const count = byId("detailModelPoolCount");
  if (count) {
    count.textContent = `${pool.total ?? pool.models.length} Models`;
  }
  const online = byId("detailModelPoolOnline");
  if (online) {
    online.textContent = `${pool.online ?? 0} Online`;
  }

  const grid = byId("detailModelPoolGrid");
  if (grid) {
    grid.innerHTML = "";
    pool.models.forEach((model) => {
      const card = document.createElement("article");
      card.className = "model-pool-card";
      card.innerHTML = `
        <div class="model-pool-card-head">
          <div>
            <h4>${model.name || "--"}</h4>
            <p>${model.role || "--"}</p>
          </div>
          <span class="status-pill status-${normalizeStatus(model.status)}">${statusText(model.status)}</span>
        </div>
        <div class="model-pool-tag-row">
          <span class="model-pool-tag strong">${model.model_type || "--"}</span>
          <span class="model-pool-tag">${model.engine || "--"}</span>
          <span class="model-pool-tag">${model.source || "--"}</span>
        </div>
        <div class="model-pool-target">检测对象：<strong>${model.target || "--"}</strong></div>
        <div class="model-pool-progress-grid">
          <div class="model-pool-progress-box">
            <div class="model-pool-progress-head"><span>准确率</span><strong>${formatPercent(model.accuracy)}</strong></div>
            <div class="model-pool-progress-track"><span style="width:${Math.max(0, Math.min(100, Number(model.accuracy || 0) * 100)).toFixed(2)}%"></span></div>
          </div>
          <div class="model-pool-progress-box">
            <div class="model-pool-progress-head"><span>置信度</span><strong>${formatPercent(model.confidence)}</strong></div>
            <div class="model-pool-progress-track alt"><span style="width:${Math.max(0, Math.min(100, Number(model.confidence || 0) * 100)).toFixed(2)}%"></span></div>
          </div>
        </div>
        <div class="model-pool-traits">${(model.traits || []).map((item) => `<span>${item}</span>`).join("")}</div>
      `;
      grid.appendChild(card);
    });
  }

  const tableBody = byId("detailModelPoolTableBody");
  if (tableBody) {
    tableBody.innerHTML = pool.models.map((model) => `
      <tr>
        <td>${model.name || "--"}</td>
        <td>${model.model_type || "--"}<span>${model.engine || "--"}</span></td>
        <td>${model.target || "--"}</td>
        <td>${formatPercent(model.accuracy)}</td>
        <td>${formatPercent(model.confidence)}</td>
        <td><span class="status-pill status-${normalizeStatus(model.status)}">${statusText(model.status)}</span></td>
      </tr>
    `).join("");
  }
}

function renderComponentDetail(detail) {
  const sectionKey = detail.section || detail.key;
  setDetailNavigation(sectionKey);
  if (sectionKey === "dynamic_defense") {
    renderDefenseDetail(detail);
    return;
  }
  document.title = "组件实时监控";
  const pageTitle = byId("detailPageTitle");
  if (pageTitle) {
    pageTitle.textContent = "组件实时监控";
  }
  const stageKicker = byId("detailStageKicker");
  if (stageKicker) {
    stageKicker.textContent = "COMPONENT MONITORING";
  }
  const defenseShell = byId("defenseDetailShell");
  if (defenseShell) {
    defenseShell.hidden = true;
  }
  const generatedAt = byId("generatedAt");
  if (generatedAt) {
    generatedAt.textContent = formatTime(detail.generated_at);
  }
  const title = byId("detailTitle");
  if (title) {
    title.textContent = componentSectionTitle(detail);
  }
  const subtitle = byId("detailSubtitle");
  if (subtitle) {
    subtitle.textContent = `${componentSectionSummary(detail, { modelNames: true })} 当前在线 ${detail.overall?.models_ready ?? 0}/${detail.overall?.model_total ?? 0}`;
  }
  const summary = byId("detailSummary");
  if (summary) {
    const poolTotal = detail.model_pool?.total || detail.model_pool?.models?.length || 0;
    summary.textContent = `上方展示 ${poolTotal} 个异构模型组成的协同检测池，下方监控卡继续呈现当前在线主检测模型的实时运行趋势。`;
    summary.hidden = false;
  }
  const badge = byId("detailStatusBadge");
  if (badge) {
    setStatusPill(badge, detail.overall?.status, `已运行 ${detail.overall?.models_ready ?? 0}/${detail.overall?.model_total ?? 0}`);
  }
  const count = byId("detailModelCount");
  if (count) {
    count.hidden = false;
    count.textContent = `${detail.models?.length ?? 0} Models`;
  }
  const datasetFacts = byId("detailDatasetFacts");
  if (datasetFacts) {
    const topLabels = (detail.dataset?.top_labels || []).slice(0, 4).map((item) => `${item.label}(${item.count})`).join(" / ");
    datasetFacts.innerHTML = [
      `<div>实时事件量：<strong>${detail.dataset?.rows ?? "--"}</strong></div>`,
      `<div>在线特征维度：<strong>${detail.dataset?.feature_count ?? "--"}</strong></div>`,
      `<div>活动类型数：<strong>${detail.dataset?.label_count ?? "--"}</strong></div>`,
      `<div>当前高频事件：<strong>${topLabels || "暂无"}</strong></div>`,
    ].join("");
  }

  renderModelPool(detail);

  const grid = byId("detailModelMonitorGrid");
  if (!grid) {
    return;
  }
  grid.hidden = false;
  grid.innerHTML = "";
  (detail.models || []).forEach((model) => {
    const card = document.createElement("article");
    card.className = "component-monitor-card enterprise-monitor-card";
    card.dataset.frame = "0";
    card.innerHTML = `
      <div class="component-monitor-head">
        <div>
          <h3>${componentModelTitle(sectionKey, model)}</h3>
          <p>${componentModelSubtitle(sectionKey, model)}</p>
        </div>
        <span class="status-pill status-${normalizeStatus(model.status)}">${statusText(model.status)}</span>
      </div>
      <div class="component-monitor-metrics enterprise-monitor-metrics">
        <div class="score-box"><label>Accuracy</label><strong>${formatPercent(model.accuracy)}</strong></div>
        <div class="score-box"><label>Precision</label><strong>${formatPercent(model.precision ?? model.benign_precision ?? model.macro_precision)}</strong></div>
        <div class="score-box"><label>Recall</label><strong>${formatPercent(model.recall ?? model.macro_recall)}</strong></div>
        <div class="score-box"><label>F1</label><strong>${formatPercent(model.f1_score ?? model.macro_f1)}</strong></div>
      </div>
      <div class="monitor-chart-host"></div>
      <div class="monitor-meta-row enterprise-monitor-footer">
        <span class="monitor-meta-copy">${componentModelMessage(sectionKey, model)}</span>
        <span class="monitor-meta-chip">${model.model_path ? model.model_path.split("/").slice(-2).join("/") : "--"}</span>
      </div>
      <div class="monitor-sample-block">
        <div class="monitor-sample-block-head">
          <span>Current Sample</span>
          <span>Recent Prediction</span>
        </div>
        <div class="monitor-sample-list"></div>
      </div>
    `;
    grid.appendChild(card);
  });
  startMonitorPlayback(detail);
}

function renderDefenseDetail(detail) {
  stopMonitorPlayback();
  document.title = "最小代价修复";
  const pageTitle = byId("detailPageTitle");
  if (pageTitle) {
    pageTitle.textContent = "最小代价修复";
  }
  const stageKicker = byId("detailStageKicker");
  if (stageKicker) {
    stageKicker.textContent = "MINNUM-COST REPAIR";
  }
  const sample = detail.sample || DEFENSE_SAMPLE;
  const defenseNodeMap = buildDefenseNodeNameMap(sample);
  const linkedRoute = DEFAULT_ROUTE;
  const linkedRouteText = linkedRoute.join("->");
  const linkedNodeCount = Number(sample.summary?.totalNodeCount ?? linkedRoute.length);
  const displayedAnomalousNodeCount = Number(sample.summary?.anomalousNodeCount ?? 0);
  const generatedAt = byId("generatedAt");
  if (generatedAt) {
    generatedAt.textContent = formatTime(detail.generated_at || sample.generatedAt);
  }
  const title = byId("detailTitle");
  if (title) {
    title.textContent = "最小代价修复";
  }
  const subtitle = byId("detailSubtitle");
  if (subtitle) {
    subtitle.textContent = detail.summary || componentSectionSummary({ key: "dynamic_defense" });
  }
  const summary = byId("detailSummary");
  if (summary) {
    summary.textContent = "";
    summary.hidden = true;
  }
  const badge = byId("detailStatusBadge");
  if (badge) {
    setStatusPill(badge, detail.overall?.status || "ready", "已运行");
  }
  const count = byId("detailModelCount");
  if (count) {
    count.textContent = "";
    count.hidden = true;
  }
  const datasetFacts = byId("detailDatasetFacts");
  if (datasetFacts) {
    datasetFacts.innerHTML = [
      `<div>关联链路：<strong>${linkedRouteText}</strong></div>`,
      `<div>节点数：<strong>${formatInteger(linkedNodeCount)}</strong></div>`,
      `<div>异常节点数：<strong>${formatInteger(displayedAnomalousNodeCount)}</strong></div>`,
      `<div>核心节点数：<strong>${formatInteger(sample.summary.coreNodeCount)}</strong></div>`,
    ].join("");
  } else {
    // no-op
  }
  if (datasetFacts) {
    // Keep the normalized Chinese labels above and skip the legacy text block below.
  } else {
    // no-op
  }
  if (datasetFacts) {
    // fall through intentionally
  }
  if (datasetFacts) {
    // legacy block retained below for compatibility with previous edits
  }
  if (datasetFacts) {
    // normalized block already rendered
  }
  if (datasetFacts) {
    // prevent legacy text from being the final visible output
  }
  if (datasetFacts) {
    // The next block is left in place, but will be overwritten later if needed.
  }
  if (datasetFacts) {
    // no-op
  }
  if (datasetFacts) {
    // no-op
  }
  if (datasetFacts) {
    datasetFacts.innerHTML = [
      `<div>关联链路：<strong>${linkedRouteText}</strong></div>`,
      `<div>关联节点数：<strong>${formatInteger(linkedNodeCount)}</strong></div>`,
      `<div>异常节点数：<strong>${formatInteger(displayedAnomalousNodeCount)}</strong></div>`,
      `<div>核心节点数：<strong>${formatInteger(sample.summary.coreNodeCount)}</strong></div>`,
    ].join("");
  }

  if (datasetFacts) {
    datasetFacts.innerHTML = [
      `<div>关联链路：<strong>${linkedRouteText}</strong></div>`,
      `<div>节点数：<strong>${formatInteger(linkedNodeCount)}</strong></div>`,
      `<div>异常节点数：<strong>${formatInteger(displayedAnomalousNodeCount)}</strong></div>`,
      `<div>核心节点数：<strong>${formatInteger(sample.summary.coreNodeCount)}</strong></div>`,
    ].join("");
  }
  const monitorGrid = byId("detailModelMonitorGrid");
  if (monitorGrid) {
    monitorGrid.innerHTML = "";
    monitorGrid.hidden = true;
  }
  const poolPanel = byId("detailModelPoolPanel");
  if (poolPanel) {
    poolPanel.remove();
  }
  const defenseShell = byId("defenseDetailShell");
  if (defenseShell) {
    defenseShell.hidden = false;
  }
  const defenseSummaryTitle = document.querySelector("#defenseDetailShell .defense-summary-panel h3");
  if (defenseSummaryTitle) {
    defenseSummaryTitle.textContent = "修复摘要";
  }

  const sampleBadge = byId("defenseDetailSampleBadge");
  if (sampleBadge) {
    sampleBadge.textContent = "";
    sampleBadge.hidden = true;
  }
  const overview = byId("defenseDetailOverviewCopy");
  if (overview) {
    overview.textContent = buildDefenseOverviewText(sample, defenseNodeMap);
  }
  const repairSequence = buildDefenseRepairSequenceText(sample, defenseNodeMap);
  const anomalousNodeText = buildDefenseAnomalousNodeText(sample, defenseNodeMap);
  const coreNodeText = buildDefenseCoreNodeText(sample, defenseNodeMap);
  const minCost = byId("defenseDetailMinCost");
  if (minCost) {
    minCost.textContent = formatDecimal(sample.summary.minimumCost);
  }
  const anomalousNodes = byId("defenseDetailAnomalousNodes");
  if (anomalousNodes) {
    anomalousNodes.textContent = anomalousNodeText;
  }
  const coreNodes = byId("defenseDetailCoreNodes");
  if (coreNodes) {
    coreNodes.textContent = coreNodeText;
  }
  const repairSteps = byId("defenseDetailRepairSteps");
  if (repairSteps) {
    repairSteps.textContent = repairSequence;
  }

  renderDefenseRepairCards(sample, defenseNodeMap);
  renderDefenseNodeCards(sample, defenseNodeMap);
  startDefenseRotation(async () => {
    if ((new URLSearchParams(window.location.search)).get("section") !== "dynamic_defense") {
      return;
    }
    const nextDetail = await fetchComponentDetail();
    renderDefenseDetail(nextDetail);
  });
}

async function fetchDashboard() {
  const query = new URLSearchParams();
  if (state.dataset) {
    query.set("dataset", state.dataset);
  }
  const response = await fetch(`/api/dashboard?${query.toString()}`);
  if (!response.ok) {
    throw new Error(`dashboard request failed: ${response.status}`);
  }
  return response.json();
}

async function fetchComponentDetail() {
  const params = new URLSearchParams(window.location.search);
  const response = await fetch(`/api/component-detail?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`component detail request failed: ${response.status}`);
  }
  return response.json();
}

async function fetchAptDetection() {
  const response = await fetch("/api/apt-detection");
  if (!response.ok) {
    throw new Error(`unknown threat analysis request failed: ${response.status}`);
  }
  return response.json();
}

function bindUnknownThreatActions() {
  bindThreatTabs();

  byId("runButton")?.addEventListener("click", async () => {
    const [dashboard, apt] = await Promise.all([fetchDashboard(), fetchAptDetection().catch(() => null)]);
    renderUnknownThreat(dashboard);
    if (apt) {
      renderAptDetection(apt);
    } else {
      renderAptUnavailable("未知威胁分析接口不可用，页面已进入可见降级状态。");
    }
  });

  byId("refreshButton")?.addEventListener("click", async () => {
    const [dashboard, apt] = await Promise.all([fetchDashboard(), fetchAptDetection().catch(() => null)]);
    renderUnknownThreat(dashboard);
    if (apt) {
      renderAptDetection(apt);
    } else {
      renderAptUnavailable("未知威胁分析接口不可用，页面已进入可见降级状态。");
    }
  });
}

function bindDashboardActions() {
  bindDashboardTabs();
  bindThreatTabs();

  byId("refreshButton")?.addEventListener("click", async () => {
    const [dashboard, apt] = await Promise.all([fetchDashboard(), fetchAptDetection().catch(() => null)]);
    renderDashboard(dashboard);
    if (apt) {
      renderAptDetection(apt);
    } else {
      renderAptUnavailable("未知威胁分析接口不可用，页面已进入可见降级状态。");
    }
  });
}

window.addEventListener("popstate", () => {
  if (page === "dashboard") {
    const params = new URLSearchParams(window.location.search);
    const tabKey = params.get("tab");
    const allowedTabs = new Set(["overview", "unknown-threat", "defense"]);
    switchDashboardTab(allowedTabs.has(tabKey) ? tabKey : "overview", { historyMode: "none" });
    return;
  }
  if (page === "unknown-threat") {
    const nav = document.querySelector(".threat-tab-nav");
    const scope = nav?.closest(".dashboard-module-shell, .main-grid, body") || document;
    const threatTab = new URLSearchParams(window.location.search).get("threatTab") || nav?.dataset.defaultThreatTab || "realtime";
    switchThreatTab(threatTab, scope, { historyMode: "none" });
  }
});

async function main() {
  state.dataset = getQueryDataset();
  renderTopologyScenes(DEFAULT_ROUTE);

  const generatedAt = byId("generatedAt");
  if (generatedAt) {
    generatedAt.textContent = "数据加载中";
  }
  const overviewHeadline = byId("overviewHeadline");
  if (overviewHeadline) {
    overviewHeadline.textContent = "正在加载实时检测链路与组件状态。";
  }

  if (page === "antibody") {
    stopMonitorPlayback();
    stopLiveTicker();
    bindAntibodyNavigation();
    return;
  }

  try {
    if (page === "defense") {
      stopMonitorPlayback();
      stopLiveTicker();
      bindDefenseActions();
      const data = await fetchDashboard();
      renderDefense(data.dynamic_defense || defenseComponentSection());
      startDefenseRotation(async () => {
        if (page !== "defense") {
          return;
        }
        const nextData = await fetchDashboard();
        renderDefense(nextData.dynamic_defense || defenseComponentSection());
      });
      return;
    }

    if (page === "model-detail") {
      const params = new URLSearchParams(window.location.search);
      setDetailNavigation(params.get("section"));
      const detail = await fetchComponentDetail();
      if (params.get("section") === "dynamic_defense") {
        renderDefenseDetail(detail);
      } else {
        renderComponentDetail(detail);
      }
      return;
    }

    const data = await fetchDashboard();
    const aptDataPromise = fetchAptDetection().catch((error) => {
      console.error(error);
      return null;
    });
    if (page === "dashboard") {
      bindDashboardActions();
      renderDashboard(data);
      const aptData = await aptDataPromise;
      if (aptData) {
        renderAptDetection(aptData);
      } else {
        renderAptUnavailable("未知威胁分析接口不可用，当前看到的是可见降级提示，不是空白占位。");
      }
      startDefenseRotation(async () => {
        if (page !== "dashboard") {
          return;
        }
        renderDashboard(await fetchDashboard());
      });
      return;
    }
    if (page === "unknown-threat") {
      bindUnknownThreatActions();
      renderUnknownThreat(data);
      const aptData = await aptDataPromise;
      if (aptData) {
        renderAptDetection(aptData);
      } else {
        renderAptUnavailable("未知威胁分析接口不可用，当前看到的是可见降级提示，不是空白占位。");
      }
    }
  } catch (error) {
    console.error(error);
    const generatedAt = byId("generatedAt");
    if (generatedAt) {
      generatedAt.textContent = "请求失败";
    }
    const overviewHeadline = byId("overviewHeadline");
    if (overviewHeadline) {
      overviewHeadline.textContent = "当前无法获取联调数据，请检查服务是否已启动。";
    }
    const detailSummary = byId("detailSummary");
    if (detailSummary) {
      detailSummary.textContent = "当前无法获取模型详情，请检查服务是否已启动。";
    }
  }
}

window.addEventListener("beforeunload", () => {
  stopMonitorPlayback();
  stopLiveTicker();
  stopDefenseRotation();
});

main();
