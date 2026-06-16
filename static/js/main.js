/* ═══════════════════════════════════════════════════════════════
   TriGuard-DF — main.js
═══════════════════════════════════════════════════════════════ */
'use strict';

/* ── 1. CONSTANTS ── */
const FACTS = [
  "Deepfakes can alter micro-expressions invisible to the human eye.",
  "The term 'deepfake' was coined in 2017 on Reddit.",
  "TriGuard-DF uses three independent AI branches for higher accuracy.",
  "rPPG signals detect blood flow changes that deepfakes cannot replicate.",
  "Frequency domain analysis reveals GAN compression artifacts.",
  "EfficientNet-B4 processes spatial facial features across multiple frames.",
  "Deepfake videos often have inconsistent blinking patterns.",
  "The physiological branch checks heart rate signals across 6 facial ROIs.",
  "Cross-modal fusion combines spatial, spectral and physiological signals.",
  "State-of-the-art deepfake detectors achieve over 95% AUC on benchmarks.",
  "Temporal attention pooling weights the most informative video frames.",
  "GAN-generated faces often fail coherence checks across facial regions.",
  "Results are session-only — nothing is stored on the server.",
  "Batch mode processes videos in a queue so results appear live.",
  "The model was trained on 8,059 video clips across real and fake categories.",
];

const STEPS = [
  { id: 'step-1', text: 'Reading video frames...',            icon: '📽️' },
  { id: 'step-2', text: 'Detecting faces...',                 icon: '👤' },
  { id: 'step-3', text: 'Analyzing frequency patterns...',    icon: '📡' },
  { id: 'step-4', text: 'Measuring physiological signals...', icon: '💓' },
  { id: 'step-5', text: 'Running fusion model...',            icon: '🧠' },
];

const STEP_TIMINGS = [1500, 3500, 6000, 9000, 12000];

const state = {
  mode:         'single',
  singleFile:   null,
  batchFiles:   [],
  batchId:      null,
  batchResults: [],
  batchTotal:   0,
  pollTimer:    null,
  stepTimers:   [],
  factTimer:    null,
  currentFact:  0,
  isAnalyzing:  false,
  lastResult:   null,
};

/* ── 2. DOM REFS ── */
const $ = id => document.getElementById(id);

const DOM = {
  modelOverlay:        $('model-loading-overlay'),
  overlayStatus:       $('overlay-status-text'),
  app:                 $('app'),
  statusDot:           $('status-dot'),
  statusLabel:         $('status-label'),
  hackerImg:           $('hacker-img'),
  imgFlash:            $('img-flash'),
  modeSingle:          $('mode-single'),
  modeBatch:           $('mode-batch'),
  singlePanel:         $('single-panel'),
  batchPanel:          $('batch-panel'),
  singleDropzone:      $('single-dropzone'),
  singleFileInput:     $('single-file-input'),
  singlePreview:       $('single-preview'),
  previewVideo:        $('preview-video'),
  previewFilename:     $('preview-filename'),
  previewFilesize:     $('preview-filesize'),
  btnAnalyzeSingle:    $('btn-analyze-single'),
  batchDropzone:       $('batch-dropzone'),
  batchFileInput:      $('batch-file-input'),
  batchListWrap:       $('batch-list-wrap'),
  batchFileList:       $('batch-file-list'),
  batchCount:          $('batch-count'),
  btnAnalyzeBatch:     $('btn-analyze-batch'),
  analysisOverlay:     $('analysis-overlay'),
  batchProgressWrap:   $('batch-progress-wrap'),
  batchProgressFill:   $('batch-progress-fill'),
  batchProgressText:   $('batch-progress-text'),
  factsText:           $('facts-text'),
  singleResultSection: $('single-result-section'),
  singleResultCard:    $('single-result-card'),
  resultBadge:         $('result-badge'),
  resultIcon:          $('result-icon'),
  resultLabel:         $('result-label'),
  resultFilename:      $('result-filename'),
  probRingCircle:      $('prob-ring-circle'),
  probPercent:         $('prob-percent'),
  fakeBarFill:         $('fake-bar-fill'),
  realBarFill:         $('real-bar-fill'),
  fakeVal:             $('fake-val'),
  realVal:             $('real-val'),
  statConfidence:      $('stat-confidence'),
  metaTime:            $('meta-time'),
  metaDevice:          $('meta-device'),
  metaTimestamp:       $('meta-timestamp'),
  confettiCanvas:      $('confetti-canvas'),
  batchResultSection:  $('batch-result-section'),
  summaryTotal:        $('summary-total'),
  summaryReal:         $('summary-real'),
  summaryFake:         $('summary-fake'),
  summaryTime:         $('summary-time'),
  fakeRateWrap:        $('fake-rate-bar-wrap'),
  fakeRateFill:        $('fake-rate-fill'),
  fakeRatePct:         $('fake-rate-pct'),
  batchCardsGrid:      $('batch-cards-grid'),
  batchDownloadWrap:   $('batch-download-wrap'),
  btnDlCsv:            $('btn-dl-csv'),
  btnDlJson:           $('btn-dl-json'),
  toastContainer:      $('toast-container'),
};

/* ── 3. PARTICLE NETWORK ── */
(function initParticles() {
  const canvas = $('particle-canvas');
  const ctx    = canvas.getContext('2d');
  let W, H, particles;

  const CFG = {
    count:    80,
    maxDist:  140,
    speed:    0.4,
    color:    'rgba(0,212,255,',
  };

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeParticle() {
    return {
      x:     Math.random() * W,
      y:     Math.random() * H,
      vx:    (Math.random() - 0.5) * CFG.speed,
      vy:    (Math.random() - 0.5) * CFG.speed,
      r:     Math.random() * 2 + 1,
      pulse: Math.random() * Math.PI * 2,
    };
  }

  function draw() {
    ctx.fillStyle = '#050510';
    ctx.fillRect(0, 0, W, H);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx; p.y += p.vy;
      p.pulse += 0.02;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;

      const r = p.r + Math.sin(p.pulse) * 0.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = CFG.color + '0.8)';
      ctx.fill();

      for (let j = i + 1; j < particles.length; j++) {
        const q  = particles[j];
        const dx = p.x - q.x, dy = p.y - q.y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < CFG.maxDist) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(q.x, q.y);
          ctx.strokeStyle = CFG.color + (1 - d/CFG.maxDist) * 0.35 + ')';
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  resize();
  particles = Array.from({ length: CFG.count }, makeParticle);
  draw();
})();

/* ── 4. HEALTH CHECK ── */
async function checkModelHealth() {
  const msgs = [
    'Loading model weights...',
    'Initializing neural network...',
    'Warming up inference engine...',
    'Preparing face detection...',
    'Almost ready...',
  ];
  let mi = 0;
  const msgTimer = setInterval(() => {
    mi = (mi + 1) % msgs.length;
    if (DOM.overlayStatus) DOM.overlayStatus.textContent = msgs[mi];
  }, 2200);

  let attempts = 0;
  while (attempts < 60) {
    try {
      const res  = await fetch('/health');
      const data = await res.json();

      if (data.status === 'ready') {
        clearInterval(msgTimer);
        setStatusPill('ready', `Ready · ${data.device.toUpperCase()}`);
        DOM.modelOverlay.classList.add('hidden');
        DOM.app.classList.remove('app-hidden');
        DOM.app.classList.add('app-visible');
        showToast('✅ TriGuard-DF is ready!', 'success');
        return;
      }
      if (data.status === 'error') {
        clearInterval(msgTimer);
        if (DOM.overlayStatus)
          DOM.overlayStatus.textContent = 'Error: ' + data.error;
        setStatusPill('error', 'Model Error');
        showToast('❌ Model failed: ' + data.error, 'error', 8000);
        return;
      }
    } catch (_) {}
    await sleep(2000);
    attempts++;
  }

  clearInterval(msgTimer);
  if (DOM.overlayStatus)
    DOM.overlayStatus.textContent = 'Timeout — please refresh.';
}

function setStatusPill(s, label) {
  DOM.statusDot.className    = 'status-dot ' + s;
  DOM.statusLabel.textContent = label;
}

/* ── 5. MODE SWITCH ── */
function switchMode(mode) {
  state.mode = mode;
  DOM.modeSingle.classList.toggle('active', mode === 'single');
  DOM.modeBatch.classList.toggle('active',  mode === 'batch');
  DOM.singlePanel.style.display = mode === 'single' ? 'block' : 'none';
  DOM.batchPanel.style.display  = mode === 'batch'  ? 'block' : 'none';
  DOM.singleResultSection.style.display = 'none';
  DOM.batchResultSection.style.display  = 'none';
}

/* ── 6. DRAG & DROP ── */
function handleDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('drag-over');
}
function handleDragLeave(e) {
  e.currentTarget.classList.remove('drag-over');
}
function handleSingleDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  triggerRipple(e.currentTarget);
  const files = [...e.dataTransfer.files].filter(isValidVideo);
  if (!files.length) {
    showToast('⚠️ Drop a valid video (.mp4 .avi .mov)', 'warn');
    return;
  }
  setSingleFile(files[0]);
}
function handleBatchDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  triggerRipple(e.currentTarget);
  const files = [...e.dataTransfer.files].filter(isValidVideo);
  if (!files.length) {
    showToast('⚠️ Drop valid video files (.mp4 .avi .mov)', 'warn');
    return;
  }
  addBatchFiles(files);
}
function handleSingleFileSelect(input) {
  const files = [...input.files].filter(isValidVideo);
  if (files.length) setSingleFile(files[0]);
  input.value = '';
}
function handleBatchFileSelect(input) {
  addBatchFiles([...input.files].filter(isValidVideo));
  input.value = '';
}

/* ── 7. SINGLE FILE ── */
function setSingleFile(file) {
  state.singleFile = file;
  DOM.previewVideo.src            = URL.createObjectURL(file);
  DOM.previewFilename.textContent = file.name;
  DOM.previewFilesize.textContent = formatBytes(file.size);
  DOM.singlePreview.style.display = 'block';
  DOM.singleResultSection.style.display = 'none';
  showToast(`📁 ${file.name} ready`, 'info');
}
function clearSingle() {
  state.singleFile = null;
  DOM.previewVideo.src            = '';
  DOM.singlePreview.style.display = 'none';
  DOM.singleResultSection.style.display = 'none';
  resetLogoState();
}

/* ── 8. BATCH FILES ── */
function addBatchFiles(newFiles) {
  const existing = new Set(state.batchFiles.map(f => f.name + f.size));
  const toAdd    = newFiles.filter(f => !existing.has(f.name + f.size));
  if (!toAdd.length) {
    showToast('⚠️ Files already in queue.', 'warn');
    return;
  }
  state.batchFiles.push(...toAdd);
  renderBatchList();
  DOM.batchListWrap.style.display = 'block';
  showToast(`📦 ${toAdd.length} file(s) added`, 'info');
}
function removeBatchFile(index) {
  state.batchFiles.splice(index, 1);
  renderBatchList();
  if (!state.batchFiles.length)
    DOM.batchListWrap.style.display = 'none';
}
function clearBatch() {
  state.batchFiles = [];
  DOM.batchListWrap.style.display      = 'none';
  DOM.batchResultSection.style.display = 'none';
  DOM.batchFileList.innerHTML          = '';
  resetLogoState();
}
function renderBatchList() {
  const n = state.batchFiles.length;
  DOM.batchCount.textContent = `${n} video${n !== 1 ? 's' : ''} selected`;
  DOM.batchFileList.innerHTML = state.batchFiles.map((f, i) => `
    <div class="batch-file-item">
      <span class="file-item-icon">🎬</span>
      <div class="file-item-info">
        <div class="file-item-name" title="${escHtml(f.name)}">${escHtml(f.name)}</div>
        <div class="file-item-size">${formatBytes(f.size)}</div>
      </div>
      <button class="file-item-remove"
              onclick="removeBatchFile(${i})">✕</button>
    </div>
  `).join('');
}

/* ── 9. SINGLE ANALYSIS ── */
async function analyzeSingle() {
  if (!state.singleFile) {
    showToast('⚠️ Select a video first.', 'warn'); return;
  }
  if (state.isAnalyzing) return;

  state.isAnalyzing = true;
  DOM.btnAnalyzeSingle.disabled = true;

  showAnalysisOverlay(false);

  const fd = new FormData();
  fd.append('file', state.singleFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    hideAnalysisOverlay();

    if (data.prediction === 'ERROR') {
      showToast('❌ ' + data.error, 'error', 6000);
      return;
    }
    state.lastResult = data;
    renderSingleResult(data);

  } catch (err) {
    hideAnalysisOverlay();
    showToast('❌ Network error: ' + err.message, 'error', 6000);
  } finally {
    state.isAnalyzing = false;
    DOM.btnAnalyzeSingle.disabled = false;
  }
}

/* ── 10. BATCH ANALYSIS ── */
async function analyzeBatch() {
  if (!state.batchFiles.length) {
    showToast('⚠️ Add videos first.', 'warn'); return;
  }
  if (state.isAnalyzing) return;

  state.isAnalyzing  = true;
  state.batchResults = [];
  state.batchTotal   = state.batchFiles.length;
  DOM.btnAnalyzeBatch.disabled     = true;
  DOM.batchCardsGrid.innerHTML     = '';
  DOM.batchDownloadWrap.style.display = 'none';
  DOM.summaryTotal.textContent     = state.batchTotal;
  DOM.summaryReal.textContent      = '0';
  DOM.summaryFake.textContent      = '0';
  DOM.summaryTime.textContent      = '—';
  DOM.fakeRateWrap.style.display   = 'none';
  DOM.batchResultSection.style.display = 'block';

  showAnalysisOverlay(true);

  const fd = new FormData();
  state.batchFiles.forEach(f => fd.append('files[]', f));

  try {
    const res  = await fetch('/batch/start', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.error) {
      hideAnalysisOverlay();
      showToast('❌ ' + data.error, 'error', 6000);
      state.isAnalyzing = false;
      DOM.btnAnalyzeBatch.disabled = false;
      return;
    }
    state.batchId = data.batch_id;
    startBatchPolling();

  } catch (err) {
    hideAnalysisOverlay();
    showToast('❌ ' + err.message, 'error', 6000);
    state.isAnalyzing = false;
    DOM.btnAnalyzeBatch.disabled = false;
  }
}

/* ── 11. BATCH POLLING ── */
function startBatchPolling() {
  let lastDone = 0;
  state.pollTimer = setInterval(async () => {
    try {
      const res  = await fetch(`/batch/status/${state.batchId}`);
      const data = await res.json();

      if (data.error) {
        stopBatchPolling(); hideAnalysisOverlay();
        showToast('❌ ' + data.error, 'error');
        state.isAnalyzing = false;
        DOM.btnAnalyzeBatch.disabled = false;
        return;
      }

      const pct = data.total > 0 ? (data.done / data.total) * 100 : 0;
      DOM.batchProgressFill.style.width = pct + '%';
      DOM.batchProgressText.textContent =
        `${data.done} / ${data.total} videos analyzed`;

      data.results.slice(lastDone).forEach(r => {
        state.batchResults.push(r);
        appendBatchCard(r, state.batchResults.length - 1);
        updateBatchSummary();
      });
      lastDone = data.done;

      if (data.done < data.total) resetStepAnimation();

      if (data.status === 'done') {
        stopBatchPolling();
        hideAnalysisOverlay();
        onBatchComplete();
      }
    } catch (err) {
      console.error('Poll error:', err);
    }
  }, 2000);
}
function stopBatchPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}
function onBatchComplete() {
  state.isAnalyzing = false;
  DOM.btnAnalyzeBatch.disabled = false;
  DOM.batchDownloadWrap.style.display = 'block';
  DOM.fakeRateWrap.style.display      = 'block';
  DOM.btnDlCsv.onclick  = () => downloadBatch('csv');
  DOM.btnDlJson.onclick = () => downloadBatch('json');
  updateBatchSummary();
  resetLogoState();
  showToast(`✅ Batch complete! ${state.batchResults.length} analyzed.`, 'success', 5000);
}

/* ── 12. ANALYSIS OVERLAY ── */
function showAnalysisOverlay(isBatch) {
  DOM.analysisOverlay.style.display = 'flex';
  DOM.batchProgressWrap.style.display = isBatch ? 'block' : 'none';

  STEPS.forEach(s => {
    const el = $(s.id);
    if (!el) return;
    el.className = 'step';
    el.querySelector('.step-icon').textContent = '⏳';
    el.querySelector('.step-text').textContent = s.text;
  });

  startStepAnimation();
  startFactsRotation();
  setLogoState('loading');
}
function hideAnalysisOverlay() {
  DOM.analysisOverlay.style.display = 'none';
  state.stepTimers.forEach(t => clearTimeout(t));
  state.stepTimers = [];
  stopFactsRotation();
}
function startStepAnimation() {
  state.stepTimers.forEach(t => clearTimeout(t));
  state.stepTimers = [];

  STEPS.forEach((s, i) => {
    const t = setTimeout(() => {
      const el = $(s.id);
      if (!el) return;
      if (i > 0) {
        const prev = $(STEPS[i-1].id);
        if (prev) {
          prev.className = 'step done';
          prev.querySelector('.step-icon').textContent = '✅';
        }
      }
      el.className = 'step active';
      el.querySelector('.step-icon').textContent = s.icon;
    }, STEP_TIMINGS[i]);
    state.stepTimers.push(t);
  });

  const tLast = setTimeout(() => {
    const last = $(STEPS[STEPS.length-1].id);
    if (last) {
      last.className = 'step done';
      last.querySelector('.step-icon').textContent = '✅';
    }
  }, STEP_TIMINGS[STEP_TIMINGS.length-1] + 2000);
  state.stepTimers.push(tLast);
}
function resetStepAnimation() {
  state.stepTimers.forEach(t => clearTimeout(t));
  state.stepTimers = [];
  startStepAnimation();
}

/* ── 13. FACTS ROTATION ── */
function startFactsRotation() {
  DOM.factsText.textContent = FACTS[state.currentFact];
  state.factTimer = setInterval(() => {
    DOM.factsText.classList.add('fading');
    setTimeout(() => {
      state.currentFact = (state.currentFact + 1) % FACTS.length;
      DOM.factsText.textContent = FACTS[state.currentFact];
      DOM.factsText.classList.remove('fading');
    }, 500);
  }, 4000);
}
function stopFactsRotation() {
  if (state.factTimer) { clearInterval(state.factTimer); state.factTimer = null; }
}

/* ── 14. SINGLE RESULT ── */
function renderSingleResult(data) {
  const isFake = data.prediction === 'FAKE';
  const pFake  = Math.round(data.probability_fake * 100);
  const pReal  = Math.round(data.probability_real * 100);
  const conf   = Math.round(data.confidence * 100);

  DOM.singleResultSection.style.display = 'block';
  DOM.singleResultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  DOM.singleResultCard.className =
    'result-card ' + (isFake ? 'result-fake' : 'result-real');

  DOM.resultBadge.className =
    'result-badge ' + (isFake ? 'badge-fake' : 'badge-real');
  DOM.resultIcon.textContent   = isFake ? '⚠️' : '✅';
  DOM.resultLabel.textContent  = isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC VIDEO';
  DOM.resultFilename.textContent = data.video;

  // Ring animation
  const circ = 314;
  const offset = circ * (1 - data.probability_fake);
  DOM.probRingCircle.style.strokeDashoffset = circ;
  setTimeout(() => {
    DOM.probRingCircle.style.stroke = isFake
      ? 'var(--fake-color)' : 'var(--real-color)';
    DOM.probRingCircle.style.strokeDashoffset = offset;
  }, 100);

  animateCount(DOM.probPercent, 0, pFake, 1200, v => v + '%');

  setTimeout(() => {
    DOM.fakeBarFill.style.width = pFake + '%';
    DOM.realBarFill.style.width = pReal + '%';
  }, 200);

  animateCount(DOM.fakeVal, 0, pFake, 1200, v => v + '%');
  animateCount(DOM.realVal, 0, pReal, 1200, v => v + '%');

  DOM.statConfidence.textContent = conf + '% ' + (isFake ? 'FAKE' : 'REAL');
  DOM.statConfidence.style.color = isFake
    ? 'var(--fake-color)' : 'var(--real-color)';

  DOM.metaTime.textContent      = data.inference_time_sec + 's';
  DOM.metaDevice.textContent    = data.device.toUpperCase();
  DOM.metaTimestamp.textContent = formatTimestamp(data.timestamp);

  setLogoState(isFake ? 'fake' : 'real');

  if (isFake) {
    setTimeout(() => {
      DOM.singleResultCard.classList.add('glitch-active');
      setTimeout(() =>
        DOM.singleResultCard.classList.remove('glitch-active'), 600);
    }, 400);
    showToast('⚠️ Deepfake detected!', 'error', 5000);
  } else {
    setTimeout(() => launchConfetti(), 600);
    showToast('✅ Video appears authentic!', 'success', 5000);
  }
}

/* ── 15. BATCH RESULT CARDS ── */
function appendBatchCard(result, index) {
  const isFake  = result.prediction === 'FAKE';
  const isError = result.prediction === 'ERROR';
  const pFake   = Math.round((result.probability_fake || 0) * 100);

  const cardClass  = 'batch-result-card ' +
    (isError ? 'card-error' : isFake ? 'card-fake' : 'card-real');
  const badgeClass = isError ? 'badge-error-sm' :
    isFake ? 'badge-fake-sm' : 'badge-real-sm';
  const icon  = isError ? '❌' : isFake ? '⚠️' : '✅';
  const badge = isError ? 'ERROR' : isFake ? 'FAKE' : 'REAL';
  const probColor = isError ? 'var(--text-muted)' :
    isFake ? 'var(--fake-color)' : 'var(--real-color)';

  const card = document.createElement('div');
  card.className = cardClass;
  card.style.animationDelay = (index * 0.05) + 's';
  card.innerHTML = `
    <span class="batch-card-icon">${icon}</span>
    <div class="batch-card-info">
      <div class="batch-card-name"
           title="${escHtml(result.video)}">${escHtml(result.video)}</div>
      <div class="batch-card-meta">
        ${isError
          ? escHtml(result.error || 'Unknown error')
          : `${result.inference_time_sec}s &nbsp;·&nbsp; ${result.device.toUpperCase()}`}
      </div>
    </div>
    <div class="batch-mini-bar">
      <div class="batch-mini-fill ${isFake ? 'fill-fake' : 'fill-real'}"
           id="mini-fill-${index}" style="width:0%"></div>
    </div>
    <div class="batch-card-prob">
      <div class="batch-prob-num" style="color:${probColor}">
        ${isError ? '—' : pFake + '%'}
      </div>
      <div class="batch-prob-lbl">fake</div>
    </div>
    <span class="batch-card-badge ${badgeClass}">${badge}</span>
  `;
  DOM.batchCardsGrid.appendChild(card);
  setTimeout(() => {
    const fill = $(`mini-fill-${index}`);
    if (fill) fill.style.width = pFake + '%';
  }, 120);
}

function updateBatchSummary() {
  const r = state.batchResults;
  const real = r.filter(x => x.prediction === 'REAL').length;
  const fake = r.filter(x => x.prediction === 'FAKE').length;
  DOM.summaryReal.textContent = real;
  DOM.summaryFake.textContent = fake;
  const times = r.filter(x => x.inference_time_sec > 0)
                  .map(x => x.inference_time_sec);
  if (times.length) {
    const avg = (times.reduce((a,b)=>a+b,0)/times.length).toFixed(1);
    DOM.summaryTime.textContent = avg + 's';
  }
  if (r.length > 0) {
    const rate = (fake / r.length) * 100;
    DOM.fakeRateFill.style.width = rate + '%';
    DOM.fakeRatePct.textContent  = Math.round(rate) + '%';
  }
}

/* ── 16. DOWNLOADS ── */
async function downloadBatch(format) {
  if (!state.batchId) {
    showToast('⚠️ No batch to download.', 'warn'); return;
  }
  try {
    const res = await fetch(`/batch/download/${state.batchId}?format=${format}`);
    if (!res.ok) { showToast('❌ Download failed.', 'error'); return; }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    const ts   = new Date().toISOString().slice(0,19).replace(/:/g,'-');
    a.href = url; a.download = `triguard_results_${ts}.${format}`;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`📥 Downloaded as ${format.toUpperCase()}`, 'success');
  } catch (err) {
    showToast('❌ ' + err.message, 'error');
  }
}

function downloadSingleResult() {
  if (!state.lastResult) return;
  const blob = new Blob(
    [JSON.stringify({ generated: new Date().toISOString(),
                      result: state.lastResult }, null, 2)],
    { type: 'application/json' }
  );
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  const ts  = new Date().toISOString().slice(0,19).replace(/:/g,'-');
  a.href = url; a.download = `triguard_result_${ts}.json`;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast('📥 Result downloaded', 'success');
}

/* ── 17. RESET ── */
function resetToUpload() {
  state.singleFile = null; state.batchFiles = [];
  state.batchId = null; state.batchResults = [];
  state.lastResult = null;
  stopBatchPolling();
  DOM.singlePreview.style.display       = 'none';
  DOM.singleResultSection.style.display = 'none';
  DOM.batchResultSection.style.display  = 'none';
  DOM.batchListWrap.style.display       = 'none';
  DOM.batchFileList.innerHTML           = '';
  DOM.batchCardsGrid.innerHTML          = '';
  DOM.previewVideo.src                  = '';
  resetLogoState();
  window.scrollTo({ top: 0, behavior: 'smooth' });
  showToast('🔄 Ready for new analysis', 'info');
}

/* ── 18. LOGO STATE ── */
function setLogoState(stateName) {
  const img   = DOM.hackerImg;
  const flash = DOM.imgFlash;
  if (!img) return;

  img.classList.remove('state-real', 'state-fake', 'state-loading');
  if (flash) flash.classList.remove('flash-real', 'flash-fake');

  if (stateName === 'real') {
    img.classList.add('state-real');
    if (flash) {
      flash.classList.add('flash-real');
      setTimeout(() => flash.classList.remove('flash-real'), 1000);
    }
  } else if (stateName === 'fake') {
    img.classList.add('state-fake');
    if (flash) {
      flash.classList.add('flash-fake');
      setTimeout(() => flash.classList.remove('flash-fake'), 1000);
    }
  } else if (stateName === 'loading') {
    img.classList.add('state-loading');
  }
}
function resetLogoState() {
  const img = DOM.hackerImg;
  if (!img) return;
  img.classList.remove('state-real', 'state-fake', 'state-loading');
  if (DOM.imgFlash)
    DOM.imgFlash.classList.remove('flash-real', 'flash-fake');
}

/* ── 19. CONFETTI ── */
function launchConfetti() {
  const canvas = DOM.confettiCanvas;
  const ctx    = canvas.getContext('2d');
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;

  const colors = [
    '#00ff88','#00d4ff','#ffffff',
    '#88ffcc','#00ffcc','#44ff88',
  ];
  const pieces = Array.from({ length: 130 }, () => ({
    x:       Math.random() * canvas.width,
    y:       Math.random() * canvas.height - canvas.height,
    w:       Math.random() * 10 + 5,
    h:       Math.random() * 5 + 3,
    color:   colors[Math.floor(Math.random() * colors.length)],
    speed:   Math.random() * 4 + 2,
    angle:   Math.random() * Math.PI * 2,
    spin:    (Math.random() - 0.5) * 0.2,
    drift:   (Math.random() - 0.5) * 2,
    opacity: 1,
  }));

  let elapsed = 0;
  let frame;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    elapsed++;
    let alive = false;
    pieces.forEach(p => {
      p.y += p.speed; p.x += p.drift;
      p.angle += p.spin;
      p.opacity = Math.max(0, 1 - elapsed / 190);
      if (p.y < canvas.height + 20) alive = true;
      ctx.save();
      ctx.globalAlpha = p.opacity;
      ctx.translate(p.x + p.w/2, p.y + p.h/2);
      ctx.rotate(p.angle);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.w/2, -p.h/2, p.w, p.h);
      ctx.restore();
    });
    if (alive && elapsed < 210) {
      frame = requestAnimationFrame(draw);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      cancelAnimationFrame(frame);
    }
  }
  draw();
}

/* ── 20. TOASTS ── */
function showToast(message, type = 'info', duration = 3500) {
  const icons = { success:'✅', error:'❌', warn:'⚠️', info:'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
    <span class="toast-msg">${escHtml(message)}</span>
  `;
  DOM.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('hiding');
    setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 350);
  }, duration);
}

/* ── 21. RIPPLE ── */
function triggerRipple(zone) {
  const r = zone.querySelector('.drop-ripple');
  if (!r) return;
  r.classList.remove('ripple-go');
  void r.offsetWidth;
  r.classList.add('ripple-go');
  setTimeout(() => r.classList.remove('ripple-go'), 700);
}

/* ── 22. LOGO HOVER TILT ── */
(function initTilt() {
  const wrap = $('hero-logo');
  if (!wrap) return;
  wrap.addEventListener('mousemove', e => {
    const rect = wrap.getBoundingClientRect();
    const dx = (e.clientX - rect.left - rect.width/2)  / (rect.width/2);
    const dy = (e.clientY - rect.top  - rect.height/2) / (rect.height/2);
    wrap.style.transform =
      `perspective(600px) rotateX(${dy * -10}deg) rotateY(${dx * 10}deg)`;
  });
  wrap.addEventListener('mouseleave', () => {
    wrap.style.transform = 'perspective(600px) rotateX(0) rotateY(0)';
  });
})();

/* ── 23. UTILITIES ── */
function isValidVideo(f) {
  return ['mp4','avi','mov'].includes(f.name.split('.').pop().toLowerCase());
}
function formatBytes(b) {
  if (b < 1024)        return b + ' B';
  if (b < 1024*1024)   return (b/1024).toFixed(1) + ' KB';
  return (b/(1024*1024)).toFixed(1) + ' MB';
}
function formatTimestamp(iso) {
  try {
    return new Date(iso).toLocaleTimeString([],
      { hour:'2-digit', minute:'2-digit' });
  } catch { return '—'; }
}
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
    .replace(/'/g,'&#039;');
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function animateCount(el, from, to, dur, fmt) {
  const start = performance.now();
  function tick(now) {
    const p = Math.min((now - start) / dur, 1);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt ? fmt(Math.round(from + (to-from)*e))
                         : Math.round(from + (to-from)*e);
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ── 24. INIT ── */
document.addEventListener('DOMContentLoaded', () => {
  checkModelHealth();
  switchMode('single');
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && state.isAnalyzing)
      showToast('⏳ Analysis in progress...', 'info', 2000);
  });
  window.addEventListener('resize', () => {
    if (DOM.confettiCanvas) {
      DOM.confettiCanvas.width  = window.innerWidth;
      DOM.confettiCanvas.height = window.innerHeight;
    }
  });
});