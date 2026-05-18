/* ============================================================
   optrA Resistance Analyser  –  Frontend Logic
   app.js
   ============================================================ */
'use strict';

// ── Config ───────────────────────────────────────────────────
const API_BASE      = 'http://localhost:5000';
const HISTORY_KEY   = 'optra_history';
const MAX_HISTORY   = 30;

// ── DOM refs ─────────────────────────────────────────────────
const seqInput       = document.getElementById('seq-input');
const charCount      = document.getElementById('char-count');
const btnAnalyze     = document.getElementById('btn-analyze');
const btnTestSeq     = document.getElementById('btn-test-seq');
const btnClear       = document.getElementById('btn-clear');
const btnRedo        = document.getElementById('btn-redo');
const btnClearHist   = document.getElementById('btn-clear-history');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText    = document.getElementById('loading-text');
const toast          = document.getElementById('toast');
const historyList    = document.getElementById('history-list');
const historyEmpty   = document.getElementById('history-empty');

// verdict
const verdictBanner   = document.getElementById('verdict-banner');
const verdictIcon     = document.getElementById('verdict-icon');
const verdictTitle    = document.getElementById('verdict-title');
const verdictSubtitle = document.getElementById('verdict-subtitle');

// confidence
const confValue = document.getElementById('conf-value');
const confBar   = document.getElementById('conf-bar');

// grids
const featGrid  = document.getElementById('feat-grid');
const abGrid    = document.getElementById('ab-grid');
const lociGrid  = document.getElementById('loci-grid');


// ═══════════════════════════════════════════════════════════════
//  UTILITY
// ═══════════════════════════════════════════════════════════════

function showToast(msg, duration = 4000) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), duration);
}

function setLoading(on, msg = 'Running GNN analysis…') {
  loadingText.textContent = msg;
  loadingOverlay.classList.toggle('active', on);
  btnAnalyze.disabled = on;
  btnTestSeq.disabled = on;
}

function sanitiseSeq(raw) {
  return raw.toUpperCase().replace(/[^ATGCN]/g, '');
}

let toastTimer = null;

// ── Character counter ─────────────────────────────────────────
seqInput.addEventListener('input', () => {
  const clean = sanitiseSeq(seqInput.value);
  charCount.textContent = clean.length.toLocaleString();
});


// ═══════════════════════════════════════════════════════════════
//  HISTORY  (localStorage)
// ═══════════════════════════════════════════════════════════════

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch { return []; }
}

function saveHistory(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY)));
}

function addHistoryEntry(sequence, result) {
  const history = loadHistory();
  history.unshift({
    sequence,
    is_resistant: result.is_resistant,
    confidence:   result.confidence,
    timestamp:    new Date().toISOString(),
  });
  saveHistory(history);
  renderHistory();
}

function renderHistory() {
  const history = loadHistory();
  historyList.innerHTML = '';

  if (history.length === 0) {
    historyList.appendChild(historyEmpty);
    return;
  }

  history.forEach((entry, idx) => {
    const li = document.createElement('li');
    li.className = 'history-item';
    li.dataset.idx = idx;

    const seqShort = entry.sequence.length > 40
      ? entry.sequence.slice(0, 40) + '…'
      : entry.sequence;

    const labelClass = entry.is_resistant ? 'resistant' : 'non-resistant';
    const labelText  = entry.is_resistant ? 'Resistant'  : 'Non-Resistant';

    const date = new Date(entry.timestamp);
    const dateStr = date.toLocaleDateString(undefined, { month:'short', day:'numeric' })
                  + ' ' + date.toLocaleTimeString(undefined, { hour:'2-digit', minute:'2-digit' });

    li.innerHTML = `
      <div class="history-seq">${seqShort}</div>
      <div class="history-meta">
        <span class="history-label ${labelClass}">${labelText}</span>
        <span class="history-conf">${entry.confidence.toFixed(1)}%</span>
      </div>
      <div class="history-date">${dateStr}</div>
    `;

    li.addEventListener('click', () => {
      seqInput.value = entry.sequence;
      charCount.textContent = sanitiseSeq(entry.sequence).length.toLocaleString();
      seqInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      seqInput.focus();
    });

    historyList.appendChild(li);
  });
}

btnClearHist.addEventListener('click', () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

// Init history on load
renderHistory();

// ═══════════════════════════════════════════════════════════════
//  FETCH HELPERS
// ═══════════════════════════════════════════════════════════════

async function fetchTestSequence() {
  btnTestSeq.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/api/test-sequence`);
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    seqInput.value = data.sequence;
    charCount.textContent = sanitiseSeq(data.sequence).length.toLocaleString();
    seqInput.focus();
  } catch (err) {
    showToast(`Could not load test sequence: ${err.message}`);
  } finally {
    btnTestSeq.disabled = false;
  }
}

async function analyseSequence(sequence) {
  setLoading(true, 'Running GNN analysis…');

  // Cycle loading messages to show progress
  const msgs = [
    'Building de Bruijn graph…',
    'Running GCN encoder…',
    'Scoring resistance loci…',
    'Identifying antibiotic types…',
    'Preparing visualisation…',
  ];
  let mIdx = 0;
  const msgInterval = setInterval(() => {
    mIdx = (mIdx + 1) % msgs.length;
    loadingText.textContent = msgs[mIdx];
  }, 1800);

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ sequence }),
    });

    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `Server error ${res.status}`);

    return data;
  } finally {
    clearInterval(msgInterval);
    setLoading(false);
  }
}

// ═══════════════════════════════════════════════════════════════
//  RENDER RESULTS
// ═══════════════════════════════════════════════════════════════

function renderResults(result) {
  // Scroll to results smoothly
  resultsSection.classList.add('visible');
  setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);

  renderVerdict(result);
  renderConfidence(result.confidence);
  renderFeatures(result.features);
  renderAntibiotics(result.antibiotic_types || []);
  renderLoci(result.loci || []);
}

// ── Verdict ───────────────────────────────────────────────────
function renderVerdict(result) {
  const resistant = result.is_resistant;
  verdictBanner.className = `verdict-banner ${resistant ? 'resistant' : 'non-resistant'}`;
  verdictIcon.textContent  = resistant ? '⚠️' : '✅';
  verdictTitle.textContent = resistant
    ? 'Antibiotic Resistance Detected'
    : 'No Significant Resistance Detected';
  verdictSubtitle.textContent = resistant
    ? `This sequence shows characteristics consistent with antibiotic resistance. `
      + `Confidence: ${result.confidence.toFixed(1)}%. `
      + `Sequence length: ${(result.sequence_length || 0).toLocaleString()} bp.`
    : `This sequence does not exhibit strong resistance markers. `
      + `Confidence: ${result.confidence.toFixed(1)}%. `
      + `Sequence length: ${(result.sequence_length || 0).toLocaleString()} bp.`;
}

// ── Confidence bar ────────────────────────────────────────────
function renderConfidence(pct) {
  confValue.textContent = pct.toFixed(1) + '%';
  // Animate after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confBar.style.width = Math.min(pct, 100) + '%';
    });
  });
}

// ── Feature summary ───────────────────────────────────────────
function renderFeatures(features) {
  if (!features) { featGrid.innerHTML = '<p style="color:var(--text-light);font-size:.82rem;">No feature data.</p>'; return; }

  const items = [
    { name: 'GC Content',        value: features.gc_percent?.toFixed(1) + '%' },
    { name: 'HGT Risk',          value: (features.hgt_risk * 100)?.toFixed(1) + '%' },
    { name: 'Mutation Signals',  value: features.mutation_count ?? '—' },
    { name: 'IS Elements',       value: features.is_elements ?? '—' },
    { name: 'Promoter Strength', value: (features.promoter_strength * 100)?.toFixed(0) + '%' },
    { name: 'Evolution Index',   value: features.evolution_index?.toFixed(3) },
  ];

  featGrid.innerHTML = items.map(it => `
    <div class="feat-item">
      <div class="feat-name">${it.name}</div>
      <div class="feat-value">${it.value}</div>
    </div>
  `).join('');
}

// ── Antibiotic types ──────────────────────────────────────────
function renderAntibiotics(types) {
  if (!types.length) {
    abGrid.innerHTML = '<p style="color:var(--text-light);font-size:.82rem;">No resistance types identified.</p>';
    return;
  }

  abGrid.innerHTML = types.map(ab => `
    <div class="ab-badge">
      <div class="ab-name">${ab.name}</div>
      <div class="ab-class">${ab.class}</div>
      <div class="ab-conf-bar-track">
        <div class="ab-conf-bar-fill" data-conf="${(ab.confidence * 100).toFixed(1)}"></div>
      </div>
      <div class="ab-conf-text">${(ab.confidence * 100).toFixed(1)}% confidence</div>
    </div>
  `).join('');

  // Animate bars after render
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.ab-conf-bar-fill').forEach(el => {
        el.style.width = el.dataset.conf + '%';
      });
    });
  });
}

// ── Loci cards + D3 graphs ─────────────────────────────────────
function renderLoci(loci) {
  if (!loci.length) {
    lociGrid.innerHTML = '<p style="color:var(--text-light);font-size:.82rem;padding:8px 0;">No loci identified.</p>';
    return;
  }

  lociGrid.innerHTML = loci.map((locus, i) => `
    <div class="locus-card">
      <div class="locus-card-header">
        <span class="locus-rank">LOCUS #${locus.rank}</span>
        <span class="locus-score-badge">Score: ${locus.score.toFixed(4)}</span>
      </div>
      <div class="locus-card-body">
        <div class="locus-hexamer">${locus.hexamer}</div>
        <div class="locus-pos">Position in sequence: ${locus.position}</div>
        <div class="graph-container" id="graph-${i}"></div>
      </div>
    </div>
  `).join('');

  // Draw each D3 subgraph after DOM is ready
  loci.forEach((locus, i) => {
    drawLocusGraph(`graph-${i}`, locus);
  });
}


// ═══════════════════════════════════════════════════════════════
//  D3  –  EULERIAN SUBGRAPH VISUALISATION
// ═══════════════════════════════════════════════════════════════

function drawLocusGraph(containerId, locus) {
  const container = document.getElementById(containerId);
  if (!container || !locus.nodes || !locus.edges) return;

  const W = container.clientWidth  || 280;
  const H = container.clientHeight || 180;

  // Clear any existing SVG
  d3.select(container).selectAll('svg').remove();

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // ── Arrow marker ──────────────────────────────────────────────
  svg.append('defs').append('marker')
    .attr('id', `arrow-${containerId}`)
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 22)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#40916C');

  svg.append('defs').append('marker')
    .attr('id', `arrow-locus-${containerId}`)
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 22)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#D62839');

  // ── Straight horizontal layout (Eulerian path) ─────────────────
  const nodes = locus.nodes;
  const N     = nodes.length;
  const padX  = 36;
  const stepX = N > 1 ? (W - 2 * padX) / (N - 1) : 0;
  const midY  = H / 2;

  const posMap = {};
  nodes.forEach((n, i) => {
    posMap[n.id] = { x: padX + i * stepX, y: midY };
  });

  // GC colour scale: green shades
  const gcColor = d3.scaleSequential(d3.interpolate('#B7E4C7', '#1B4332'))
    .domain([0, 1]);

  // ── Edges ──────────────────────────────────────────────────────
  const edgeGroup = svg.append('g').attr('class', 'edges');

  locus.edges.forEach(edge => {
    const src = posMap[edge.source];
    const tgt = posMap[edge.target];
    if (!src || !tgt) return;

    const isLocus = edge.is_locus;
    const strokeW = isLocus ? 3 : 1.5 + edge.score * 2;
    const color   = isLocus ? '#D62839'
                            : d3.interpolateRdYlGn(1 - edge.score);

    // Slight vertical arc to avoid overlapping label
    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const dr = Math.sqrt(dx * dx + dy * dy) * 0.6;

    edgeGroup.append('path')
      .attr('d', `M${src.x},${src.y} A${dr},${dr} 0 0,1 ${tgt.x},${tgt.y}`)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', strokeW)
      .attr('stroke-dasharray', isLocus ? '5,3' : 'none')
      .attr('marker-end', `url(#${isLocus ? 'arrow-locus-' : 'arrow-'}${containerId})`)
      .attr('opacity', isLocus ? 1 : 0.75);

    // Edge score label (only for non-trivial scores)
    if (edge.score > 0.05) {
      const mx = (src.x + tgt.x) / 2;
      const my = midY - 18;
      edgeGroup.append('text')
        .attr('x', mx)
        .attr('y', my)
        .attr('text-anchor', 'middle')
        .attr('font-size', '8px')
        .attr('fill', isLocus ? '#D62839' : '#40916C')
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('font-weight', isLocus ? '700' : '400')
        .text(edge.score.toFixed(2));
    }
  });

  // ── Nodes ──────────────────────────────────────────────────────
  const nodeGroup = svg.append('g').attr('class', 'nodes');

  nodes.forEach(n => {
    const pos = posMap[n.id];
    if (!pos) return;

    const g = nodeGroup.append('g')
      .attr('transform', `translate(${pos.x},${pos.y})`);

    // Circle
    g.append('circle')
      .attr('r', 14)
      .attr('fill', gcColor(n.gc))
      .attr('stroke', '#1B4332')
      .attr('stroke-width', 1.5);

    // Pentamer label (first 3 bases to fit)
    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '6.5px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-weight', '600')
      .attr('fill', n.gc > 0.5 ? '#fff' : '#0d2b1e')
      .text(n.id.slice(0, 3));

    // Tooltip on hover
    g.append('title').text(`${n.id}  (GC: ${(n.gc * 100).toFixed(0)}%)`);
  });

  // ── Legend: locus edge indicator ──────────────────────────────
  const lg = svg.append('g').attr('transform', `translate(6,${H - 20})`);
  lg.append('line')
    .attr('x1', 0).attr('y1', 6)
    .attr('x2', 18).attr('y2', 6)
    .attr('stroke', '#D62839').attr('stroke-width', 2)
    .attr('stroke-dasharray', '4,2');
  lg.append('text')
    .attr('x', 22).attr('y', 10)
    .attr('font-size', '7px')
    .attr('fill', '#D62839')
    .attr('font-family', 'Poppins, sans-serif')
    .text('Resistance locus');
}


// ═══════════════════════════════════════════════════════════════
//  EVENT HANDLERS
// ═══════════════════════════════════════════════════════════════

btnTestSeq.addEventListener('click', fetchTestSequence);

btnClear.addEventListener('click', () => {
  seqInput.value = '';
  charCount.textContent = '0';
  seqInput.focus();
  resultsSection.classList.remove('visible');
});

btnRedo.addEventListener('click', () => {
  resultsSection.classList.remove('visible');
  seqInput.value = '';
  charCount.textContent = '0';
  window.scrollTo({ top: 0, behavior: 'smooth' });
  setTimeout(() => seqInput.focus(), 400);
});

btnAnalyze.addEventListener('click', runAnalysis);

// Allow Ctrl+Enter to trigger analysis
seqInput.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runAnalysis();
});

async function runAnalysis() {
  const raw = seqInput.value.trim();
  if (!raw) {
    showToast('Please enter a DNA sequence first.');
    seqInput.focus();
    return;
  }

  const seq = sanitiseSeq(raw);
  if (seq.length < 6) {
    showToast('Sequence is too short (minimum 6 valid bases required).');
    return;
  }

  try {
    const result = await analyseSequence(seq);
    renderResults(result);
    addHistoryEntry(seq, result);
  } catch (err) {
    showToast(`Analysis failed: ${err.message}`);
    console.error(err);
  }
}


// ═══════════════════════════════════════════════════════════════
//  STARTUP CHECK
// ═══════════════════════════════════════════════════════════════

async function checkServerHealth() {
  try {
    const res  = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    if (!data.models_ready) {
      showToast('Server is still loading models — analysis may take a moment…', 6000);
    }
  } catch {
    showToast('⚠️  Cannot reach analysis server at localhost:5000. Is server.py running?', 8000);
  }
}

// Run health check on page load
checkServerHealth();