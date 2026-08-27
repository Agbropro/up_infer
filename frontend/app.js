const state = {
  models: [],
  details: {},
  labels: [],
  files: [],
  results: [],
  page: 1,
  perPage: 10,
};

const byId = (id) => document.getElementById(id);
const modelInput = byId("model");
const fileInput = byId("files");
const runButton = byId("run-button");
const message = byId("form-message");

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Models could not be loaded");
    state.models = data;
    modelInput.innerHTML = data.map((model) =>
      `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name)}</option>`).join("");
    modelInput.disabled = false;
    await chooseModel();
  } catch (error) {
    modelInput.innerHTML = "<option>Models unavailable</option>";
    message.textContent = error.message;
  }
}

async function chooseModel() {
  const modelId = modelInput.value;
  if (!modelId) return;
  state.labels = [];
  renderLabels();
  modelInput.disabled = true;
  byId("model-task").textContent = "Loading selected model…";
  try {
    if (!state.details[modelId]) {
      const response = await fetch(`/api/models/${encodeURIComponent(modelId)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Model could not be loaded");
      state.details[modelId] = data;
    }
    if (modelInput.value !== modelId) return;
    const model = state.details[modelId];
    state.labels = [...model.labels];
    byId("model-task").textContent = `${model.task} model · ${model.labels.length} labels`;
    message.textContent = "";
  } catch (error) {
    byId("model-task").textContent = "Model unavailable";
    message.textContent = error.message;
  } finally {
    modelInput.disabled = false;
    renderLabels();
    updateButton();
  }
}

function renderLabels() {
  const holder = byId("labels");
  byId("label-count").textContent = `(${state.labels.length})`;
  if (!state.labels.length) {
    holder.innerHTML = '<p class="empty-labels">Restore or select at least one label.</p>';
    updateButton();
    return;
  }
  holder.innerHTML = state.labels.map((label) =>
    `<span class="label-chip">${escapeHtml(label)}<button type="button" data-label="${escapeHtml(label)}" aria-label="Remove ${escapeHtml(label)}">×</button></span>`
  ).join("");
  updateButton();
}

function addFiles(fileList) {
  const incoming = [...fileList].filter((file) => file.type.startsWith("image/"));
  const known = new Set(state.files.map((file) => `${file.name}:${file.size}:${file.lastModified}`));
  incoming.forEach((file) => {
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    if (!known.has(key) && state.files.length < 20) state.files.push(file);
  });
  renderFiles();
}

function renderFiles() {
  byId("file-list").innerHTML = state.files.map((file, index) =>
    `<div class="file-row"><span>${escapeHtml(file.name)} · ${(file.size / 1048576).toFixed(1)} MB</span><button type="button" data-file="${index}" aria-label="Remove ${escapeHtml(file.name)}">Remove</button></div>`
  ).join("");
  updateButton();
}

function updateButton() {
  runButton.disabled = !state.files.length || !state.labels.length || modelInput.disabled;
}

function setLoading(loading) {
  runButton.disabled = loading;
  runButton.classList.toggle("loading", loading);
  runButton.querySelector("span").textContent = loading ? "Running inference" : "Run inference";
  if (!loading) updateButton();
}

async function submitForm(event) {
  event.preventDefault();
  message.textContent = "";
  setLoading(true);
  const body = new FormData();
  state.files.forEach((file) => body.append("files", file));
  body.append("model", modelInput.value);
  body.append("labels", JSON.stringify(state.labels));
  body.append("confidence", byId("confidence").value);
  body.append("iou", byId("iou").value);
  try {
    const response = await fetch("/api/infer", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Inference failed");
    state.results = data.images;
    state.page = 1;
    byId("results-title").textContent = `${data.model} · ${data.images.length} image${data.images.length === 1 ? "" : "s"}`;
    byId("results").hidden = false;
    renderResults();
    byId("results").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    message.textContent = error.message;
  } finally {
    setLoading(false);
  }
}

function renderResults() {
  const start = (state.page - 1) * state.perPage;
  const visible = state.results.slice(start, start + state.perPage);
  byId("result-grid").innerHTML = visible.map((item, index) => {
    const scores = item.predictions.length ? item.predictions.map((score) => `
      <div class="score" style="--class-color:${score.color};--score:${score.confidence * 100}%">
        <div><span>${escapeHtml(score.label)}</span><b>${(score.confidence * 100).toFixed(1)}%</b></div>
        <div class="score-bar"><i></i></div>
      </div>`).join("") : '<p class="no-predictions">No selected objects passed the thresholds.</p>';
    return `<article class="result-card">
      <div class="result-image"><button type="button" data-zoom="${start + index}" aria-label="Zoom inferred result for ${escapeHtml(item.name)}"><img src="${item.image}" alt="Inferred result for ${escapeHtml(item.name)}"></button></div>
      <div class="result-data"><h3 title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</h3><small>${item.width} × ${item.height}</small><div class="score-list">${scores}</div></div>
    </article>`;
  }).join("");
  const pages = Math.max(1, Math.ceil(state.results.length / state.perPage));
  byId("page-status").textContent = `Page ${state.page} of ${pages}`;
  byId("page-prev").disabled = state.page === 1;
  byId("page-next").disabled = state.page === pages;
}

modelInput.addEventListener("change", chooseModel);
byId("labels").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-label]");
  if (!button) return;
  state.labels = state.labels.filter((label) => label !== button.dataset.label);
  renderLabels();
});
byId("restore-labels").addEventListener("click", () => {
  const model = state.details[modelInput.value];
  state.labels = model ? [...model.labels] : [];
  renderLabels();
});
fileInput.addEventListener("change", () => addFiles(fileInput.files));
byId("file-list").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-file]");
  if (!button) return;
  state.files.splice(Number(button.dataset.file), 1);
  renderFiles();
});

const dropZone = byId("drop-zone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

["confidence", "iou"].forEach((id) => byId(id).addEventListener("input", () => {
  byId(`${id}-value`).textContent = Number(byId(id).value).toFixed(2);
}));
byId("infer-form").addEventListener("submit", submitForm);
byId("columns").addEventListener("change", (event) => {
  byId("result-grid").style.setProperty("--columns", event.target.value);
});
byId("page-prev").addEventListener("click", () => { state.page -= 1; renderResults(); });
byId("page-next").addEventListener("click", () => { state.page += 1; renderResults(); });

const imageDialog = byId("image-dialog");
byId("result-grid").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-zoom]");
  if (!button) return;
  const item = state.results[Number(button.dataset.zoom)];
  byId("zoomed-image").src = item.image;
  byId("zoomed-image").alt = `Zoomed inferred result for ${item.name}`;
  byId("image-caption").textContent = `${item.name} · ${item.width} × ${item.height}`;
  imageDialog.showModal();
});
byId("image-close").addEventListener("click", () => imageDialog.close());
imageDialog.addEventListener("click", (event) => {
  if (event.target === imageDialog) imageDialog.close();
});

const dialog = byId("help-dialog");
byId("help-open").addEventListener("click", () => dialog.showModal());
["help-close", "help-done"].forEach((id) => byId(id).addEventListener("click", () => dialog.close()));
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

loadModels();
