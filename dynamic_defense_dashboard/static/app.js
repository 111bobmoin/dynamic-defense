const DEFAULT_DATASET = "muti3/Dataset/validata.csv";
const DEFAULT_ROUTE = ["host1", "m1", "m3", "m4", "m7", "server1"];
const state = {
  dataset: DEFAULT_DATASET,
  monitorTimer: null,
  liveTimer: null,
  liveFrame: 0,
  dashboardLiveBase: null,
};

const MONITOR_WINDOW = 14;
const MONITOR_TICK_MS = 1200;
const LIVE_TICK_MS = 1400;

const COMPONENT_DISPLAY_ORDER = ["muti3", "log", "graph"];
const COMPONENT_TITLE_MAP = {
  muti3: "攻击数据特征检测异构组件",
  log: "攻击逻辑检测异构组件",
  graph: "行为图结构检测异构组件",
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

const byId = (id) => document.getElementById(id);
const page = document.body.dataset.page;

const moduleRouteMap = {
  multi_detection: "/unknown-threat.html",
  antibody_generalization: "/antibody.html",
  dynamic_defense: "/defense.html",
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

function getQueryDataset() {
  const params = new URLSearchParams(window.location.search);
  return params.get("dataset") || DEFAULT_DATASET;
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
    overviewHeadline.textContent = `${data.overview.headline} 当前数据集：${state.dataset.split("/").slice(-1)[0]}`;
  }

  renderSystems(data.systems);
  renderDashboardThreatModule(data);
  renderTopologyScenes(data.overview.active_path || DEFAULT_ROUTE);
  startLiveTicker(data);
}

function switchDashboardTab(tabKey) {
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
}

function bindDashboardTabs() {
  const buttons = document.querySelectorAll("[data-dashboard-tab]");
  if (!buttons.length) {
    return;
  }
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      switchDashboardTab(button.dataset.dashboardTab || "overview");
    });
  });
  switchDashboardTab("overview");
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

function renderDashboardThreatModule(data) {
  renderComponentCards(data.integration?.sections || [], "dashboardThreatComponentGrid");
  renderIncidents(data.incidents, "dashboardThreatIncidentList");
  renderTopologyScenes(data.overview.active_path || DEFAULT_ROUTE);
}

function switchThreatTab(tabKey, scope = document) {
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
}

function bindThreatTabs() {
  const navs = document.querySelectorAll(".threat-tab-nav");
  if (!navs.length) {
    return;
  }
  navs.forEach((nav) => {
    const scope = nav.closest(".dashboard-module-shell, .main-grid, body") || document;
    const buttons = nav.querySelectorAll("[data-threat-tab]");
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        switchThreatTab(button.dataset.threatTab || "realtime", scope);
      });
    });
    switchThreatTab("realtime", scope);
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
  }
  const badge = byId("detailStatusBadge");
  if (badge) {
    setStatusPill(badge, detail.overall?.status, `已运行 ${detail.overall?.models_ready ?? 0}/${detail.overall?.model_total ?? 0}`);
  }
  const count = byId("detailModelCount");
  if (count) {
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

function bindUnknownThreatActions() {
  bindThreatTabs();

  byId("runButton")?.addEventListener("click", async () => {
    renderUnknownThreat(await fetchDashboard());
  });

  byId("refreshButton")?.addEventListener("click", async () => {
    renderUnknownThreat(await fetchDashboard());
  });
}

function bindDashboardActions() {
  bindDashboardTabs();
  bindThreatTabs();

  byId("refreshButton")?.addEventListener("click", async () => {
    renderDashboard(await fetchDashboard());
  });
}

async function main() {
  state.dataset = getQueryDataset();
  renderTopologyScenes(DEFAULT_ROUTE);

  const generatedAt = byId("generatedAt");
  if (generatedAt) {
    generatedAt.textContent = "数据加载中";
  }
  const overviewHeadline = byId("overviewHeadline");
  if (overviewHeadline) {
    overviewHeadline.textContent = "正在加载联调数据，首次启动会预热模型缓存。";
  }

  if (page === "antibody" || page === "defense") {
    stopMonitorPlayback();
    stopLiveTicker();
    return;
  }

  try {
    if (page === "model-detail") {
      renderComponentDetail(await fetchComponentDetail());
      return;
    }

    const data = await fetchDashboard();
    if (page === "dashboard") {
      bindDashboardActions();
      renderDashboard(data);
      return;
    }
    if (page === "unknown-threat") {
      bindUnknownThreatActions();
      renderUnknownThreat(data);
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
});

main();
