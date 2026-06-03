(function () {
  const root = document.getElementById("antibodyDemoRoot");
  if (!root) return;

  const emptyRunState = { status: "idle", progress: 0, logs: [], completed_steps: [], step_runs: {} };
  const modeConfigs = {
    single: {
      label: "单模态泛化",
      caption: "同源攻击链变体生成与抗体验证",
      payloadUrl: "/api/antibody",
      statusUrl: "/api/antibody/status",
      startUrl: "/api/antibody/start",
      resetUrl: "/api/antibody/reset",
    },
    cross: {
      label: "跨模态泛化",
      caption: "IPv4/IPv6 到 SCION/GEO/MPLS 迁移",
      payloadUrl: "/api/antibody/cross-modal",
      statusUrl: "/api/antibody/cross-modal/status",
      startUrl: "/api/antibody/cross-modal/start",
      resetUrl: "/api/antibody/cross-modal/reset",
    },
  };

  let activeMode = "single";
  let payloads = { single: null, cross: null };
  let runStates = { single: { ...emptyRunState }, cross: { ...emptyRunState } };
  let selectedSteps = { single: 0, cross: 0 };
  let pollTimer = null;

  const statusText = {
    idle: "待启动",
    running: "运行中",
    completed: "已完成",
    error: "失败",
  };

  function config() {
    return modeConfigs[activeMode];
  }

  function payload() {
    return payloads[activeMode];
  }

  function runState() {
    return runStates[activeMode] || emptyRunState;
  }

  function selectedStepIndex() {
    return selectedSteps[activeMode] || 0;
  }

  function setSelectedStep(index) {
    selectedSteps[activeMode] = index;
  }

  function isCrossMode() {
    return activeMode === "cross";
  }

  function isStepDone(step) {
    return (runState().completed_steps || []).includes(step.key);
  }

  function isStepRunning(step) {
    return runState().status === "running" && runState().active_step === step.key;
  }

  function stepRun(step) {
    const state = runState();
    const storedRun = (state.step_runs || {})[step.key];
    if (storedRun) return storedRun;
    if (state.active_step === step.key) {
      return {
        progress: state.progress,
        logs: state.logs,
        artifact: state.artifact,
      };
    }
    return {};
  }

  function missingPreviousSteps(step) {
    const currentPayload = payload();
    const stepIndex = currentPayload.demo_steps.findIndex((item) => item.key === step.key);
    const completed = new Set(runState().completed_steps || []);
    return currentPayload.demo_steps.slice(0, Math.max(stepIndex, 0)).filter((item) => !completed.has(item.key));
  }

  function isStepLocked(step) {
    return !isStepDone(step) && missingPreviousSteps(step).length > 0;
  }

  function prerequisiteText(step) {
    const missingSteps = missingPreviousSteps(step);
    if (!missingSteps.length) return "";
    return `请先完成：${missingSteps.map((item) => item.title).join("、")}`;
  }

  function stepClass(step, index) {
    if (isStepRunning(step)) return "running";
    if (isStepDone(step)) return "done";
    if (isStepLocked(step)) return "locked";
    if (index === selectedStepIndex()) return "active";
    return "pending";
  }

  function currentStep() {
    return payload().demo_steps[selectedStepIndex()];
  }

  function renderModeSwitch() {
    return `
      <div class="antibody-mode-switch">
        ${Object.entries(modeConfigs).map(([key, item]) => `
          <button class="antibody-mode-button ${activeMode === key ? "is-active" : ""}" type="button" data-mode="${key}" ${runState().status === "running" ? "disabled" : ""}>
            <strong>${item.label}</strong>
            <span>${item.caption}</span>
          </button>
        `).join("")}
      </div>
    `;
  }

  function renderPath(pathText) {
    return String(pathText || "")
      .split("→")
      .map((node) => node.trim())
      .filter(Boolean)
      .map((node, index, nodes) => `
        <span class="antibody-path-node">${node}</span>
        ${index < nodes.length - 1 ? '<span class="antibody-path-arrow">→</span>' : ""}
      `).join("");
  }

  function clusterTitle(clusterId) {
    const cluster = (payload().clusters || []).find((item) => item.id === clusterId);
    return cluster ? cluster.title : clusterId;
  }

  function relationTargetText(relation) {
    if (relation.target === "CROSS-MODAL-EVIDENCE") return "跨模态证据一致性";
    if (relation.target === "THREAT-LLDP-LATERAL-BASELINE") return "LLDP 横向移动基线攻击";
    return clusterTitle(relation.target);
  }

  function relationTypeText(type) {
    const typeMap = {
      BELONGS_TO_CLUSTER: "簇归属",
      EVOLVED_FROM: "同源演化",
      RELATED_CLUSTER: "证据关联",
    };
    return typeMap[type] || type;
  }

  function renderStepRail() {
    return payload().demo_steps.map((step, index) => `
      <button class="antibody-step-tab is-${stepClass(step, index)}" type="button" data-step-index="${index}" ${isStepLocked(step) || runState().status === "running" ? "disabled" : ""}>
        <span>${String(index + 1).padStart(2, "0")}</span>
        <strong>${step.title}</strong>
        <small>${isStepDone(step) ? "完成" : isStepRunning(step) ? "运行中" : isStepLocked(step) ? "需前置步骤" : "待执行"}</small>
      </button>
    `).join("");
  }

  function renderResult(result = {}, visible) {
    if (!visible) return `<div class="antibody-empty-result">--</div>`;
    return Object.entries(result).map(([key, value]) => `
      <div class="antibody-result-item">
        <span>${key}</span>
        <strong>${value}</strong>
      </div>
    `).join("");
  }

  function replayLogs(step) {
    return (step.process_logs || []).map(([second, message]) => ({ time: `+${second}s`, message }));
  }

  function renderLogs(step) {
    let logs = isStepRunning(step) || isStepDone(step) ? (stepRun(step).logs || []) : [];
    if (!logs.length && isStepDone(step)) logs = replayLogs(step);
    if (!logs.length) return `<div class="antibody-log-line is-muted">--</div>`;
    return logs.map((log) => `<div class="antibody-log-line"><span>${formatLogTime(log.time)}</span>${log.message}</div>`).join("");
  }

  function formatLogTime(value) {
    if (String(value || "").startsWith("+")) return value;
    const match = String(value || "").match(/T(\d{2}:\d{2}:\d{2})/);
    return match ? match[1] : "--:--:--";
  }

  function renderArtifact(step) {
    const artifact = stepRun(step).artifact;
    if (isStepDone(step) && artifact) {
      return Object.entries(artifact).map(([key, value]) => `
        <div class="antibody-artifact-row"><span>${key}</span><strong>${Array.isArray(value) ? value.join(" / ") : value}</strong></div>
      `).join("");
    }
    if (isStepRunning(step)) return `<div class="antibody-artifact-row"><span>状态</span><strong>正在生成产物...</strong></div>`;
    return `<div class="antibody-artifact-row"><span>状态</span><strong>尚未运行</strong></div>`;
  }

  function renderStage() {
    const step = currentStep();
    const running = isStepRunning(step);
    const done = isStepDone(step);
    const prerequisite = prerequisiteText(step);
    const progress = running ? stepRun(step).progress || 0 : done ? 100 : 0;
    return `
      <section class="antibody-stage-card">
        <div class="antibody-stage-heading">
          <span>${statusText[runState().status] || "待启动"}</span>
          <h3>${step.title}</h3>
          <p>${step.summary || ""}</p>
        </div>
        ${step.items?.length ? `<div class="antibody-stage-list">${step.items.map((item) => `<span>${item}</span>`).join("")}</div>` : ""}
        ${prerequisite ? `<div class="antibody-prerequisite-tip">${prerequisite}</div>` : ""}
        <div class="antibody-progress-wrap">
          <div class="antibody-progress-meta"><span>处理进度</span><strong>${progress}%</strong></div>
          <div class="antibody-progress-bar"><i style="width:${progress}%"></i></div>
        </div>
        <div class="antibody-live-grid">
          <article>
            <h4>实时过程</h4>
            <div class="antibody-log-box">${renderLogs(step)}</div>
          </article>
          <article>
            <h4>阶段产物</h4>
            <div class="antibody-artifact-box">${renderArtifact(step)}</div>
          </article>
        </div>
        <div class="antibody-result-grid">${renderResult(step.result, done)}</div>
      </section>
    `;
  }

  function renderOutputs() {
    const completed = runState().completed_steps || [];
    const outputItems = payload().outputs || [
      { step: "attack", label: "攻击记录", value: "attack_trace.json" },
      { step: "extract", label: "混合特征", value: "hybrid_feature.json" },
      { step: "generate", label: "生成样本", value: "newattack.csv / llmattack.csv" },
      { step: "map", label: "防御标签", value: "defense.csv" },
      { step: "verify", label: "验证报告", value: "evaluation_report.json" },
    ];
    return `
      <section class="antibody-output-panel">
        <div class="antibody-output-grid ${outputItems.length === 6 ? "is-six" : ""}">
          ${outputItems.map((item) => `
            <article class="antibody-output-card">
              <span>${item.label}</span>
              <strong>${completed.includes(item.step) ? item.value : "--"}</strong>
            </article>
          `).join("")}
        </div>
      </section>
    `;
  }

  function renderRelations() {
    const verified = (runState().completed_steps || []).includes("verify");
    const pendingRelations = [
      ["簇归属", "等待验证报告生成后计算新威胁与横向移动攻击链簇的相似度。"],
      ["同源演化", "等待同源变体检测结果确认后判断是否来源于 LLDP 横向移动基线攻击。"],
      ["证据关联", "等待流量、日志和行为图验证完成后评估跨模态证据一致性。"],
    ];
    return `
      <section class="antibody-relation-panel">
        <div class="antibody-demo-panel-title">
          <span>ANTIBODY GRAPH</span>
          <h3>抗体库关系推理</h3>
          ${verified ? "<p>验证完成后展示最终关系、相似度与阈值判定。</p>" : ""}
        </div>
        <div class="antibody-relation-grid">
          ${verified ? (payload().relations || []).map((relation) => `
            <article class="antibody-relation-card ${relation.passed ? "is-passed" : "is-muted"}">
              <div class="antibody-relation-top">
                <span>${relationTypeText(relation.type)}</span>
                <strong>${Math.round((relation.cosine || 0) * 100)}%</strong>
              </div>
              <h4>${relationTargetText(relation)}</h4>
              <p>${relation.reason}</p>
              <div class="antibody-relation-bar"><i style="width:${Math.round((relation.cosine || 0) * 100)}%"></i></div>
              <small>阈值 ${Math.round((relation.threshold || 0) * 100)}% · ${relation.passed ? "通过" : "未通过"}</small>
            </article>
          `).join("") : pendingRelations.map(([title, description], index) => `
            <article class="antibody-relation-card is-pending">
              <div class="antibody-relation-top">
                <span>待推理 ${String(index + 1).padStart(2, "0")}</span>
                <strong>--</strong>
              </div>
              <h4>${title}</h4>
              <p>${description}</p>
              <div class="antibody-relation-bar"><i style="width:0%"></i></div>
              <small>等待第 5 步“验证效果”完成</small>
            </article>
          `).join("")}
        </div>
      </section>
    `;
  }

  function renderSimpleTable(headers, rows) {
    return `
      <div class="antibody-data-table" style="--cols:${headers.length}">
        <div class="antibody-data-row is-head">${headers.map((header) => `<span>${header}</span>`).join("")}</div>
        ${rows.map((row) => `<div class="antibody-data-row">${row.map((cell) => `<span>${cell}</span>`).join("")}</div>`).join("")}
      </div>
    `;
  }

  function renderCrossModalOverview() {
    const data = payload().cross_modal;
    const completed = runState().completed_steps || [];
    return `
      <section class="cross-modal-panel">
        <div class="antibody-demo-panel-title">
          <span>CROSS MODAL TRANSFER</span>
          <h3>跨模态迁移链路</h3>
          <p>${payload().source_modality} → ${payload().target_modality}</p>
        </div>
        <div class="cross-modal-flow is-compact">
          ${(data.transfer_cards || []).map((item) => `
            <article>
              <span>${item.label}</span>
              <strong>${item.value}</strong>
              <p>${item.text}</p>
            </article>
          `).join("")}
        </div>
        <div class="cross-modal-status-strip">
          <span class="${completed.includes("source") ? "is-done" : ""}">源模态</span>
          <span class="${completed.includes("transfer") ? "is-done" : ""}">迁移</span>
          <span class="${completed.includes("validation") ? "is-done" : ""}">计划</span>
          <span class="${completed.includes("acceptance") ? "is-done" : ""}">验收</span>
        </div>
      </section>
    `;
  }

  function renderCrossModalDetails() {
    const data = payload().cross_modal;
    const completed = runState().completed_steps || [];
    const planReady = completed.includes("validation");
    const accepted = completed.includes("acceptance");
    return `
      <section class="cross-modal-detail-grid is-simple">
        <article class="cross-modal-detail-card">
          <div class="antibody-demo-panel-title"><span>PLAN</span><h3>目标模态计划</h3></div>
          ${planReady ? `
            <div class="cross-modal-mini-list">
              ${(data.plan_summary || []).map((item) => `
                <div><strong>${item.scope}</strong><span>${item.action}</span><em>${item.marker}</em></div>
              `).join("")}
            </div>
          ` : `<div class="antibody-empty-result">完成第 3 步后展示目标模态计划</div>`}
        </article>
        <article class="cross-modal-detail-card">
          <div class="antibody-demo-panel-title"><span>CHECK</span><h3>验收结果</h3></div>
          ${accepted ? `
            <div class="cross-modal-check-grid">
              ${(data.checks || []).map((item) => `
                <div><span>${item.name}</span><strong>${item.value}</strong><small>${item.status}</small></div>
              `).join("")}
            </div>
          ` : `<div class="antibody-empty-result">完成第 4 步后展示验收结果</div>`}
        </article>
        <article class="cross-modal-detail-card is-wide">
          <div class="antibody-demo-panel-title"><span>ANTIBODY</span><h3>迁移抗体规则</h3></div>
          ${accepted ? `
            <div class="cross-modal-antibody-list">
              ${(data.transferred_antibodies || []).map((item) => `<span>${item.rule} · ${item.status}</span>`).join("")}
            </div>
          ` : `<div class="antibody-empty-result">等待日志验收完成</div>`}
        </article>
      </section>
    `;
  }

  function renderModeSpecificPanels() {
    if (isCrossMode()) {
      return `${renderCrossModalOverview()}${renderCrossModalDetails()}`;
    }
    return `
      <section class="antibody-path-panel compact">
        <div class="antibody-demo-panel-title"><h3>攻击路径</h3></div>
        <div class="antibody-path-flow">${renderPath(payload().antibody.source.path)}</div>
      </section>
      ${renderRelations()}
    `;
  }

  function render() {
    if (!payload()) return;
    const currentPrerequisite = prerequisiteText(currentStep());
    root.innerHTML = `
      ${renderModeSwitch()}
      <section class="antibody-demo-hero compact">
        <div>
          <h2>${payload().title || "运行控制"}</h2>
          <p>${payload().subtitle || config().caption}</p>
        </div>
        <div class="antibody-demo-actions">
          <button class="primary-action-button" type="button" data-action="start" ${runState().status === "running" || currentPrerequisite ? "disabled" : ""}>${currentPrerequisite || "启动当前步骤"}</button>
          <button class="ghost-button" type="button" data-action="reset" ${runState().status === "running" ? "disabled" : ""}>重置演示</button>
        </div>
      </section>
      <section class="antibody-runner ${isCrossMode() ? "is-cross" : ""}">
        <nav class="antibody-step-rail">${renderStepRail()}</nav>
        ${renderStage()}
      </section>
      ${renderModeSpecificPanels()}
      ${renderOutputs()}
    `;
    bindActions();
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `${url} failed: ${response.status}`);
    return data;
  }

  async function refreshStatus() {
    runStates[activeMode] = await fetchJson(config().statusUrl);
    const state = runState();
    if (state.active_step) {
      const activeIndex = payload().demo_steps.findIndex((step) => step.key === state.active_step);
      if (activeIndex >= 0) setSelectedStep(activeIndex);
    }
    render();
    if (state.status !== "running" && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => refreshStatus().catch(console.error), 300);
  }

  async function switchMode(nextMode) {
    if (nextMode === activeMode || runState().status === "running") return;
    activeMode = nextMode;
    if (!payloads[activeMode]) payloads[activeMode] = await fetchJson(config().payloadUrl);
    runStates[activeMode] = await fetchJson(config().statusUrl);
    render();
    if (runState().status === "running") startPolling();
  }

  function bindActions() {
    root.querySelectorAll("[data-mode]").forEach((button) => {
      button.addEventListener("click", () => switchMode(button.dataset.mode).catch(console.error));
    });
    root.querySelectorAll("[data-step-index]").forEach((button) => {
      button.addEventListener("click", () => {
        if (runState().status === "running") return;
        setSelectedStep(Number(button.dataset.stepIndex || 0));
        render();
      });
    });
    root.querySelector('[data-action="start"]')?.addEventListener("click", async () => {
      const step = currentStep();
      if (isStepLocked(step)) return;
      try {
        runStates[activeMode] = await fetchJson(`${config().startUrl}?step=${encodeURIComponent(step.key)}`);
        render();
        startPolling();
      } catch (error) {
        console.error(error);
        runStates[activeMode] = { ...runState(), status: "error", error: error.message };
        render();
      }
    });
    root.querySelector('[data-action="reset"]')?.addEventListener("click", async () => {
      runStates[activeMode] = await fetchJson(config().resetUrl);
      setSelectedStep(0);
      render();
    });
  }

  async function loadAntibodyDemo() {
    root.innerHTML = `<section class="antibody-demo-loading">正在准备动态演示...</section>`;
    try {
      payloads.single = await fetchJson(modeConfigs.single.payloadUrl);
      runStates.single = await fetchJson(modeConfigs.single.statusUrl);
      payloads.cross = await fetchJson(modeConfigs.cross.payloadUrl);
      runStates.cross = await fetchJson(modeConfigs.cross.statusUrl);
      render();
      if (runState().status === "running") startPolling();
    } catch (error) {
      console.error(error);
      root.innerHTML = `<section class="antibody-demo-error">抗体泛化演示加载失败，请确认后端服务已启动。</section>`;
    }
  }

  window.addEventListener("beforeunload", () => {
    if (pollTimer) clearInterval(pollTimer);
  });

  loadAntibodyDemo();
})();

