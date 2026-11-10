"use strict";

const state = {
  file: null,
  currentDocumentId: null,
  documents: [],
  jobs: [],
  aclDocument: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);
const headers = () => ({ Accept: "application/json" });

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...headers(), ...(options.headers || {}) },
  });
  let payload = null;
  const text = await response.text();
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = { detail: text }; }
  }
  if (!response.ok) {
    const error = new Error(payload?.detail || `请求失败 (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function toast(message, type = "") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  $("toastStack").appendChild(node);
  setTimeout(() => node.remove(), 4200);
}

function setStatus(id, message, type = "") {
  const node = $(id);
  node.textContent = message;
  node.className = `form-status ${type}`;
}

function splitCsv(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("zh-CN", { hour12: false });
}

function label(value) {
  return ({
    public: "公开", internal: "内部", confidential: "机密", secret: "绝密",
    unreviewed: "待复核", good: "高质量", needs_rework: "需重做", blocked: "已阻断",
    uploaded: "已上传", queued: "排队中", running: "构建中", succeeded: "已构建",
    failed: "失败", archived: "已归档", deleted: "已删除", pending: "待执行",
  })[value] || value || "—";
}

function badgeClass(value) {
  if (["succeeded", "good"].includes(value)) return "success";
  if (value === "running") return "running";
  if (["failed", "blocked"].includes(value)) return "failed";
  if (["queued", "unreviewed", "pending"].includes(value)) return "queued";
  if (value === "needs_rework") return "needs_rework";
  return "neutral";
}

function badge(value, text = null) {
  const node = document.createElement("span");
  node.className = `badge ${badgeClass(value)}`;
  node.textContent = text || label(value);
  return node;
}

async function boot() {
  try {
    const [me, capabilities] = await Promise.all([
      api("/v1/admin/me"),
      api("/v1/admin/capabilities"),
    ]);
    $("identityChip").classList.add("online");
    $("identityText").textContent = `${me.display_name || me.user_id} · ${me.tenant_id} · ${label(me.clearance)}`;
    $("capabilityBadge").textContent = `${capabilities.file_types.length} 类格式`;
    $("dropHint").textContent = capabilities.extensions.join("  ").replaceAll(".", "");
    await Promise.all([loadDocuments(), loadJobs()]);
    setInterval(loadJobs, 3000);
  } catch (error) {
    $("identityText").textContent = error.message;
    toast(error.message, "error");
    showEmpty("无法连接管理服务，请确认可信网关已注入身份头并配置代理令牌。");
  }
}

async function loadDocuments() {
  const params = new URLSearchParams({ limit: "200" });
  if ($("statusFilter").value) params.set("status", $("statusFilter").value);
  if ($("qualityFilter").value) params.set("quality_status", $("qualityFilter").value);
  try {
    const payload = await api(`/v1/admin/documents?${params}`);
    state.documents = payload.items;
    renderDocuments();
  } catch (error) {
    toast(`文档列表加载失败：${error.message}`, "error");
  }
}

function renderDocuments() {
  const list = $("documentList");
  list.replaceChildren();
  $("emptyState").classList.toggle("hidden", state.documents.length > 0);
  for (const document of state.documents) {
    list.appendChild(documentCard(document));
  }
}

function documentCard(document) {
  const card = document.createElement("article");
  card.className = "document-card";

  const main = document.createElement("div");
  main.className = "document-main";
  const titleRow = document.createElement("div");
  titleRow.className = "doc-title-row";
  const title = document.createElement("h3");
  title.textContent = document.file_name;
  title.title = document.file_name;
  titleRow.append(title, badge(document.status), badge(document.quality_status));
  main.appendChild(titleRow);

  const meta = document.createElement("div");
  meta.className = "doc-meta";
  for (const text of [
    `${document.file_type.toUpperCase()} · ${document.chunk_strategy}`,
    `${document.chunk_count} chunks`,
    `上传 ${formatDate(document.created_at)}`,
    `更新 ${formatDate(document.updated_at)}`,
  ]) {
    const item = document.createElement("span");
    item.textContent = text;
    meta.appendChild(item);
  }
  main.appendChild(meta);

  const acl = document.acl || {};
  const aclRow = document.createElement("div");
  aclRow.className = "doc-acl";
  const aclChips = [label(typeof acl.classification === "number"
    ? ["public", "internal", "confidential", "secret"][acl.classification]
    : acl.classification)];
  if (acl.owner_id) aclChips.push(`owner:${acl.owner_id}`);
  if (acl.allowed_roles?.length) aclChips.push(`roles:${acl.allowed_roles.join(",")}`);
  if (acl.allowed_groups?.length) aclChips.push(`groups:${acl.allowed_groups.join(",")}`);
  if (acl.allowed_users?.length) aclChips.push(`users:${acl.allowed_users.length}`);
  for (const value of aclChips) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = value;
    aclRow.appendChild(chip);
  }
  main.appendChild(aclRow);
  if (document.quality_note) {
    const note = document.createElement("div");
    note.className = "doc-meta";
    const span = document.createElement("span");
    span.textContent = document.quality_note;
    note.appendChild(span);
    main.appendChild(note);
  }

  const actions = document.createElement("div");
  actions.className = "doc-actions";
  actions.append(
    actionButton("预览", () => previewDocument(document.document_id)),
    actionButton("构建", () => buildDocument(document.document_id, false)),
    actionButton("强制重建", () => buildDocument(document.document_id, true)),
    actionButton("权限", () => openAclModal(document)),
  );
  const qualitySelect = document.createElement("select");
  qualitySelect.className = "compact-select";
  qualitySelect.style.minWidth = "90px";
  for (const [value, text] of [["good", "标记高质量"], ["needs_rework", "标记需重做"], ["blocked", "阻断"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    qualitySelect.appendChild(option);
  }
  qualitySelect.addEventListener("change", () => updateQuality(document, qualitySelect.value));
  const archive = actionButton("归档", () => archiveDocument(document));
  archive.classList.add("danger");
  actions.append(qualitySelect, archive);

  card.append(main, actions);
  return card;
}

function actionButton(text, handler) {
  const button = document.createElement("button");
  button.className = "button ghost";
  button.type = "button";
  button.textContent = text;
  button.addEventListener("click", handler);
  return button;
}

async function loadJobs() {
  try {
    const payload = await api("/v1/admin/jobs?limit=40");
    const previousRunning = state.jobs.some((job) => ["pending", "running"].includes(job.status));
    state.jobs = payload.items;
    renderJobs();
    if (previousRunning && !state.jobs.some((job) => ["pending", "running"].includes(job.status))) {
      await loadDocuments();
    }
  } catch {
    // 鉴权失败由 boot 提示；后台轮询保持静默。
  }
}

function renderJobs() {
  const list = $("jobList");
  list.replaceChildren();
  const active = state.jobs.filter((job) => ["pending", "running"].includes(job.status)).length;
  $("activeJobCount").textContent = `${active} 运行中`;
  if (!state.jobs.length) {
    const empty = document.createElement("div");
    empty.className = "job-message";
    empty.textContent = "暂无构建任务";
    list.appendChild(empty);
    return;
  }
  for (const job of state.jobs) {
    const row = document.createElement("div");
    row.className = "job-row";
    const left = document.createElement("div");
    left.append(badge(job.status));
    const detail = document.createElement("div");
    const name = document.createElement("strong");
    const document = state.documents.find((item) => item.document_id === job.document_id);
    name.textContent = document?.file_name || job.document_id.slice(0, 12);
    detail.appendChild(name);
    const message = document.createElement("div");
    message.className = "job-message";
    message.textContent = job.error || job.message || "—";
    detail.appendChild(message);
    const progress = document.createElement("div");
    progress.className = "job-progress";
    const bar = document.createElement("i");
    bar.style.width = `${Math.max(0, Math.min(100, job.progress || 0))}%`;
    progress.appendChild(bar);
    detail.appendChild(progress);
    const time = document.createElement("time");
    time.textContent = formatDate(job.updated_at);
    row.append(left, detail, time);
    list.appendChild(row);
  }
}

function updateChunkFields() {
  const parent = $("chunkStrategy").value === "parent_child";
  $("chunkSizeField").classList.toggle("hidden", parent);
  $("chunkOverlapField").classList.toggle("hidden", parent);
  $("parentSizeField").classList.toggle("hidden", !parent);
}

function chunkOptions() {
  if ($("chunkStrategy").value === "parent_child") {
    return {
      parent_chunk_size: Number($("parentSize").value) || 2048,
      parent_chunk_overlap: 0,
      chunk_size: 420,
      chunk_overlap: 60,
    };
  }
  return {
    chunk_size: Number($("chunkSize").value) || 600,
    chunk_overlap: Number($("chunkOverlap").value) || 0,
  };
}

async function uploadFile(event) {
  event.preventDefault();
  if (!state.file) {
    setStatus("uploadStatus", "请先选择文件", "error");
    return;
  }
  if (state.busy) return;
  state.busy = true;
  const button = $("uploadButton");
  button.disabled = true;
  setStatus("uploadStatus", "正在上传并解析…");
  try {
    const form = new FormData();
    form.append("file", state.file);
    form.append("chunk_strategy", $("chunkStrategy").value);
    form.append("chunk_options", JSON.stringify(chunkOptions()));
    form.append("classification", $("classification").value);
    form.append("allowed_roles", $("allowedRoles").value);
    form.append("allowed_groups", $("allowedGroups").value);
    form.append("allowed_users", $("allowedUsers").value);
    const document = await api("/v1/admin/documents/upload", { method: "POST", body: form });
    setStatus("uploadStatus", "上传成功，正在生成预览…", "success");
    toast("文件已上传", "success");
    await loadDocuments();
    await previewDocument(document.document_id);
  } catch (error) {
    setStatus("uploadStatus", error.message, "error");
    toast(error.message, "error");
  } finally {
    state.busy = false;
    button.disabled = false;
  }
}

async function previewDocument(documentId) {
  state.currentDocumentId = documentId;
  $("previewCard").classList.remove("hidden");
  $("previewTitle").textContent = "正在解析…";
  $("previewMetrics").replaceChildren();
  $("previewSamples").replaceChildren();
  $("previewQuality").textContent = "";
  try {
    const preview = await api(`/v1/admin/documents/${documentId}/preview`, { method: "POST" });
    const document = state.documents.find((item) => item.document_id === documentId);
    $("previewTitle").textContent = document?.file_name || "解析预览";
    renderPreviewMetrics(preview);
    renderPreviewQuality(preview.quality || {});
    renderPreviewSamples(preview.samples || []);
    $("previewCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    $("previewTitle").textContent = "预览失败";
    setStatus("buildStatus", error.message, "error");
  }
}

function renderPreviewMetrics(preview) {
  const values = [
    [preview.parsed_documents, "解析段"],
    [preview.chunk_count, "知识块"],
    [`${preview.min_chunk_length}–${preview.max_chunk_length}`, "字符范围"],
  ];
  for (const [value, name] of values) {
    const node = document.createElement("div");
    node.className = "metric";
    const strong = document.createElement("b");
    strong.textContent = value;
    const span = document.createElement("span");
    span.textContent = name;
    node.append(strong, span);
    $("previewMetrics").appendChild(node);
  }
}

function renderPreviewQuality(quality) {
  const node = $("previewQuality");
  node.classList.toggle("warn", Boolean(quality.needs_rework));
  const count = quality.low_quality_count ?? 0;
  const ratio = Math.round((quality.low_quality_ratio || 0) * 100);
  node.textContent = quality.needs_rework
    ? `质量预警：${count} 个低质量块（${ratio}%），建议调整分块参数。`
    : `质量检查通过：平均分 ${quality.average_quality_score ?? "—"}，低质量块 ${count} 个。`;
}

function renderPreviewSamples(samples) {
  const list = $("previewSamples");
  for (const sample of samples) {
    const node = document.createElement("div");
    node.className = "sample";
    const head = document.createElement("div");
    head.className = "sample-head";
    const index = document.createElement("span");
    index.textContent = `#${sample.chunk_index} · ${sample.length} 字符`;
    const location = document.createElement("span");
    location.textContent = sample.metadata?.page_number
      ? `p.${sample.metadata.page_number}`
      : (sample.metadata?.heading_path || "");
    head.append(index, location);
    const content = document.createElement("p");
    content.textContent = sample.content;
    node.append(head, content);
    list.appendChild(node);
  }
}

async function buildDocument(documentId, forceReindex = false) {
  if (!documentId) return;
  try {
    const job = await api(`/v1/admin/documents/${documentId}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force_reindex: forceReindex }),
    });
    const skipped = job.message === "skipped_unchanged";
    setStatus("buildStatus", skipped ? "内容未变化，已跳过重复构建" : "构建任务已提交", "success");
    toast(skipped ? "内容未变化，已跳过" : "构建任务已提交", "success");
    await Promise.all([loadJobs(), loadDocuments()]);
  } catch (error) {
    setStatus("buildStatus", error.message, "error");
    toast(error.message, "error");
  }
}

async function updateQuality(document, qualityStatus) {
  const note = window.prompt("质量复核说明（可留空）", document.quality_note || "");
  if (note === null) return;
  try {
    await api(`/v1/admin/documents/${document.document_id}/quality`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quality_status: qualityStatus, note }),
    });
    toast("质量状态已更新", "success");
    await loadDocuments();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function archiveDocument(document) {
  if (!window.confirm(`确定归档「${document.file_name}」？归档后查询将不可见。`)) return;
  try {
    await api(`/v1/admin/documents/${document.document_id}`, { method: "DELETE" });
    toast("文档已归档", "success");
    await loadDocuments();
  } catch (error) {
    toast(error.message, "error");
  }
}

function openAclModal(document) {
  state.aclDocument = document;
  const acl = document.acl || {};
  $("aclClassification").value = typeof acl.classification === "number"
    ? ["public", "internal", "confidential", "secret"][acl.classification]
    : (acl.classification || "internal");
  $("aclRoles").value = (acl.allowed_roles || []).join(", ");
  $("aclGroups").value = (acl.allowed_groups || []).join(", ");
  $("aclUsers").value = (acl.allowed_users || []).join(", ");
  $("aclModalBackdrop").classList.remove("hidden");
}

function closeAclModal() {
  $("aclModalBackdrop").classList.add("hidden");
  state.aclDocument = null;
}

async function saveAcl(event) {
  event.preventDefault();
  if (!state.aclDocument) return;
  try {
    await api(`/v1/admin/documents/${state.aclDocument.document_id}/acl`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classification: $("aclClassification").value,
        allowed_roles: splitCsv($("aclRoles").value),
        allowed_groups: splitCsv($("aclGroups").value),
        allowed_users: splitCsv($("aclUsers").value),
        acl_version: 1,
      }),
    });
    closeAclModal();
    toast("权限已更新", "success");
    await loadDocuments();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function rebuildByQuality() {
  try {
    const payload = await api("/v1/admin/documents/rebuild", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quality_status: "needs_rework" }),
    });
    toast(`已提交 ${payload.jobs.length} 个重建任务`, "success");
    await Promise.all([loadJobs(), loadDocuments()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function openAudit() {
  $("drawerBackdrop").classList.remove("hidden");
  $("auditDrawer").classList.add("open");
  $("auditDrawer").setAttribute("aria-hidden", "false");
  const list = $("auditList");
  list.replaceChildren();
  try {
    const payload = await api("/v1/admin/audit?limit=200");
    if (!payload.items.length) {
      const empty = document.createElement("div");
      empty.className = "job-message";
      empty.textContent = "暂无审计记录";
      list.appendChild(empty);
      return;
    }
    for (const event of payload.items) {
      const node = document.createElement("div");
      node.className = "audit-item";
      const head = document.createElement("div");
      head.className = "audit-item-head";
      const action = document.createElement("strong");
      action.textContent = event.action;
      const time = document.createElement("span");
      time.textContent = formatDate(event.created_at);
      head.append(action, time);
      const detail = document.createElement("p");
      detail.textContent = `${event.actor_id} · ${event.target_type}:${event.target_id} ${JSON.stringify(event.detail)}`;
      node.append(head, detail);
      list.appendChild(node);
    }
  } catch (error) {
    toast(error.message, "error");
  }
}

function closeAudit() {
  $("drawerBackdrop").classList.add("hidden");
  $("auditDrawer").classList.remove("open");
  $("auditDrawer").setAttribute("aria-hidden", "true");
}

function showEmpty(message) {
  $("emptyState").classList.remove("hidden");
  $("emptyState").querySelector("p").textContent = message;
}

function bindEvents() {
  $("uploadForm").addEventListener("submit", uploadFile);
  $("chunkStrategy").addEventListener("change", updateChunkFields);
  $("fileInput").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    state.file = file;
    $("dropTitle").textContent = file.name;
    $("dropHint").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · 点击可重新选择`;
    setStatus("uploadStatus", "");
  });
  const dropzone = $("dropzone");
  for (const eventName of ["dragenter", "dragover"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  }
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    $("fileInput").files = transfer.files;
    $("fileInput").dispatchEvent(new Event("change"));
  });
  $("buildButton").addEventListener("click", () => buildDocument(state.currentDocumentId, false));
  $("closePreview").addEventListener("click", () => $("previewCard").classList.add("hidden"));
  $("refreshButton").addEventListener("click", () => {
    loadDocuments();
    loadJobs();
    toast("已刷新");
  });
  $("auditButton").addEventListener("click", openAudit);
  $("closeAudit").addEventListener("click", closeAudit);
  $("drawerBackdrop").addEventListener("click", closeAudit);
  $("closeAclModal").addEventListener("click", closeAclModal);
  $("aclModalBackdrop").addEventListener("click", (event) => {
    if (event.target === $("aclModalBackdrop")) closeAclModal();
  });
  $("aclForm").addEventListener("submit", saveAcl);
  $("statusFilter").addEventListener("change", loadDocuments);
  $("qualityFilter").addEventListener("change", loadDocuments);
  $("rebuildQualityButton").addEventListener("click", rebuildByQuality);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeAudit();
      closeAclModal();
    }
  });
}

bindEvents();
updateChunkFields();
boot();
