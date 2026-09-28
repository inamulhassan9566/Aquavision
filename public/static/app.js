// AQUAVISION: AI Maritime Intelligence Platform Frontend Controller

const API_BASE = window.AQUAVISION_API_URL || localStorage.getItem('aquavision_backend_url') || window.location.origin;

// State
let currentSelectedFile = null;
let sampleChips = [];

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initUpload();
  loadSampleChips();
  loadIncidents();
  loadModelInfo();
});

// Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-target');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }

      if (targetId === 'incidents-panel') {
        loadIncidents();
      } else if (targetId === 'model-panel') {
        loadModelInfo();
      }
    });
  });
}

// Upload & Drag-and-Drop
function initUpload() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const uploadBtn = document.getElementById('upload-btn');
  const analyzeBtn = document.getElementById('analyze-btn');

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  analyzeBtn.addEventListener('click', () => {
    if (currentSelectedFile) {
      executeInference(currentSelectedFile);
    }
  });
}

function handleFileSelected(file) {
  currentSelectedFile = file;
  const fileNameDisplay = document.getElementById('selected-filename');
  fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  fileNameDisplay.style.display = 'block';

  // Preview original image in panel 1
  const reader = new FileReader();
  reader.onload = (e) => {
    const origPreview = document.getElementById('preview-original');
    origPreview.src = e.target.result;
    document.getElementById('analyze-btn').disabled = false;
  };
  reader.readAsDataURL(file);
}

// Preset Sample Chips (1-Click Evaluation)
async function loadSampleChips() {
  try {
    const res = await fetch(`${API_BASE}/samples`);
    if (!res.ok) return;
    sampleChips = await res.json();
    const container = document.getElementById('sample-chips-container');
    container.innerHTML = '';

    sampleChips.forEach(chip => {
      const chipEl = document.createElement('div');
      const isOil = chip.class_id === 1;
      chipEl.className = `sample-chip ${isOil ? 'oil' : 'no-oil'}`;
      chipEl.innerHTML = `
        <img src="${chip.url}" alt="${chip.filename}">
        <span>${isOil ? '🔴 Oil Slick' : '🟢 Clean Sea'}</span>
      `;
      chipEl.addEventListener('click', async () => {
        // Fetch chip as Blob and run
        const imgRes = await fetch(chip.url);
        const blob = await imgRes.blob();
        const file = new File([blob], chip.filename, { type: 'image/jpeg' });
        handleFileSelected(file);
        executeInference(file);
      });
      container.appendChild(chipEl);
    });
  } catch (err) {
    console.warn('Could not load sample chips:', err);
  }
}

// Inference Execution
async function executeInference(file) {
  const statusContainer = document.getElementById('analysis-status');
  const resultCard = document.getElementById('detection-card');
  const analyzeBtn = document.getElementById('analyze-btn');

  analyzeBtn.disabled = true;
  statusContainer.style.display = 'flex';
  resultCard.style.display = 'none';

  const startTime = performance.now();
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/predict-with-explanation`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned error ${res.status}`);
    }

    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    renderDetectionResult(data, elapsed);
  } catch (err) {
    alert(`Inference failed: ${err.message}`);
  } finally {
    statusContainer.style.display = 'none';
    analyzeBtn.disabled = false;
  }
}

// Render Results
function renderDetectionResult(data, latencyMs) {
  const resultCard = document.getElementById('detection-card');
  const labelEl = document.getElementById('detection-label');
  const confBadge = document.getElementById('confidence-badge');
  const oilProbText = document.getElementById('oil-prob-text');
  const oilProbFill = document.getElementById('oil-prob-fill');
  const noOilProbText = document.getElementById('nooil-prob-text');
  const noOilProbFill = document.getElementById('nooil-prob-fill');
  const explanationEl = document.getElementById('model-explanation-text');
  const latencyEl = document.getElementById('meta-latency');
  const modelVerEl = document.getElementById('meta-model-version');

  const isOil = data.class_id === 1;
  const confPct = (data.confidence * 100).toFixed(1);
  const oilPct = (data.probabilities.oil_spill * 100).toFixed(1);
  const noOilPct = (data.probabilities.no_oil * 100).toFixed(1);

  resultCard.className = `detection-card ${isOil ? 'oil' : 'no-oil'}`;
  resultCard.style.display = 'block';

  labelEl.innerHTML = isOil 
    ? '🔴 POTENTIAL OIL SPILL DETECTED' 
    : '🟢 NO OIL SPILL DETECTED';

  confBadge.textContent = `Confidence: ${confPct}%`;
  oilProbText.textContent = `${oilPct}%`;
  oilProbFill.style.width = `${oilPct}%`;
  noOilProbText.textContent = `${noOilPct}%`;
  noOilProbFill.style.width = `${noOilPct}%`;

  explanationEl.textContent = data.explanation;
  latencyEl.textContent = `${latencyMs}ms`;
  modelVerEl.textContent = data.model_version;

  // Visual attention inspector panels
  if (data.original_image_url) {
    document.getElementById('preview-original').src = `${API_BASE}${data.original_image_url}`;
  }
  if (data.gradcam_image_url) {
    document.getElementById('preview-gradcam').src = `${API_BASE}${data.gradcam_image_url}`;
  }
  if (data.overlay_image_url) {
    document.getElementById('preview-overlay').src = `${API_BASE}${data.overlay_image_url}`;
  }
}

// Load Incident History
async function loadIncidents() {
  const tbody = document.getElementById('incidents-tbody');
  tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color: var(--text-muted);">Loading incidents from SQLite database...</td></tr>';

  try {
    const res = await fetch(`${API_BASE}/incidents`);
    if (!res.ok) throw new Error('Failed to fetch incidents');
    const incidents = await res.json();

    if (incidents.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color: var(--text-dim);">No incidents recorded yet. Analyze an image above to log the first incident.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    incidents.forEach(inc => {
      const isOil = inc.class_id === 1;
      const row = document.createElement('tr');
      row.style.cursor = 'pointer';
      const dateFormatted = new Date(inc.created_at).toLocaleString();

      row.innerHTML = `
        <td>#${inc.id}</td>
        <td>${dateFormatted}</td>
        <td><img src="${API_BASE}${inc.original_image_url}" class="thumb-img" alt="SAR"></td>
        <td><span class="badge ${isOil ? 'badge-oil' : 'badge-no-oil'}">${inc.prediction}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 700;">${(inc.confidence * 100).toFixed(1)}%</td>
        <td style="font-family: var(--font-mono); color: var(--text-muted);">${inc.model_version}</td>
        <td><span style="color: var(--accent-cyan-light); font-size: 0.8rem; font-weight: 600;">View Details &rarr;</span></td>
      `;

      row.addEventListener('click', () => openIncidentModal(inc));
      tbody.appendChild(row);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--alert-red);">Error loading incidents: ${err.message}</td></tr>`;
  }
}

// Incident Modal Inspector
function openIncidentModal(inc) {
  const modal = document.getElementById('incident-modal');
  const title = document.getElementById('modal-incident-title');
  const meta = document.getElementById('modal-incident-meta');
  const origImg = document.getElementById('modal-orig-img');
  const camImg = document.getElementById('modal-cam-img');
  const overlayImg = document.getElementById('modal-overlay-img');
  const desc = document.getElementById('modal-incident-desc');

  const isOil = inc.class_id === 1;
  title.innerHTML = `Incident #${inc.id} — <span style="color: ${isOil ? 'var(--alert-red)' : 'var(--alert-green)'};">${inc.prediction}</span>`;
  meta.textContent = `Analyzed: ${new Date(inc.created_at).toLocaleString()} | Model: ${inc.model_version} | Confidence: ${(inc.confidence * 100).toFixed(1)}%`;

  origImg.src = `${API_BASE}${inc.original_image_url}`;
  camImg.src = inc.gradcam_image_url ? `${API_BASE}${inc.gradcam_image_url}` : '';
  overlayImg.src = inc.overlay_image_url ? `${API_BASE}${inc.overlay_image_url}` : '';

  desc.innerHTML = `
    <strong>Model Decision Attribution:</strong> ${inc.explanation || 'Visual patterns processed.'}<br>
    <span style="color: var(--text-dim); font-size: 0.75rem; margin-top: 0.4rem; display: block;">
      Geospatial Coordinates & AIS Attribution: ${inc.vessel_attribution_status || 'Pending Phase 2 AIS module'} (No fabricated coordinates).
    </span>
  `;

  modal.classList.add('active');
}

function closeIncidentModal() {
  document.getElementById('incident-modal').classList.remove('active');
}

// Load Model Analytics & Real Reports
async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/model-info`);
    if (!res.ok) return;
    const info = await res.json();

    const tm = info.test_metrics;
    document.getElementById('kpi-accuracy').textContent = `${(tm.accuracy * 100).toFixed(2)}%`;
    document.getElementById('kpi-f1').textContent = tm.f1_score.toFixed(4);
    document.getElementById('kpi-precision').textContent = `${(tm.precision * 100).toFixed(2)}%`;
    document.getElementById('kpi-recall').textContent = `${(tm.recall * 100).toFixed(2)}%`;
    document.getElementById('kpi-auc').textContent = tm.roc_auc.toFixed(4);
    document.getElementById('kpi-specificity').textContent = `${(tm.specificity * 100).toFixed(2)}%`;

    // Counts
    const ds = info.dataset_statistics;
    document.getElementById('stat-total-chips').textContent = ds.total_curated_chips.toLocaleString();
    document.getElementById('stat-train-chips').textContent = ds.training_samples.toLocaleString();
    document.getElementById('stat-val-chips').textContent = ds.validation_samples.toLocaleString();
    document.getElementById('stat-test-chips').textContent = ds.test_samples.toLocaleString();
    document.getElementById('stat-model-name').textContent = info.model_architecture.toUpperCase();

    // Confusion Matrix Breakdown
    const cm = tm.confusion_matrix;
    document.getElementById('cm-tn').textContent = cm.true_negative;
    document.getElementById('cm-fp').textContent = cm.false_positive;
    document.getElementById('cm-fn').textContent = cm.false_negative;
    document.getElementById('cm-tp').textContent = cm.true_positive;
  } catch (err) {
    console.warn('Could not load model info:', err);
  }
}
