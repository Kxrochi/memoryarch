const API = 'http://localhost:8000';
let sseSource = null, simRunning = false, lastState = null;

// Log store
const logRows = [];
let corrections = 0;

function updateLog(d) {
  logRows.push({
    cycle: d.cycle,
    time: new Date().toTimeString().slice(0,8),
    ipr: d.ipr,
    entropy: d.entropy,
    syntropy: d.syntropy,
    fidelity: d.fidelity,
    source: d.quantis_active ? 'QRNG' : 'PRNG',
    controller: d.controller,
    memory: d.memory_stores,
    correction: !!d.correction_triggered,
  });
  if (d.correction_triggered) corrections++;

  // Table
  const body = document.getElementById('logBody');
  const cls = d.ipr < .3 ? 'lock' : d.ipr < .6 ? 'marg' : 'drift';
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${d.cycle}</td>
    <td>${new Date().toTimeString().slice(0,8)}</td>
    <td class="${cls}">${d.ipr.toFixed(4)}</td>
    <td>${d.entropy.toFixed(4)}</td>
    <td>${d.syntropy.toFixed(4)}</td>
    <td>${d.fidelity.toFixed(5)}</td>
    <td>${d.quantis_active ? 'QRNG' : 'PRNG'}</td>
    <td>${d.controller}</td>
    <td>${(d.memory_stores||[]).join(', ')||'—'}</td>
    <td>${d.correction_triggered ? '<span class="corr">✓</span>' : '—'}</td>
  `;
  body.prepend(row); // newest on top

  document.getElementById('logEmpty').style.display = 'none';
  document.getElementById('logTable').style.display = '';
  document.getElementById('logCount').textContent = logRows.length + ' cycles recorded';

  updateSummary();
}

function updateSummary() {
  if (!logRows.length) return;
  const n = logRows.length;
  const avg = key => (logRows.reduce((s,r) => s + r[key], 0) / n);
  const avgIPR = avg('ipr');
  const avgS   = avg('entropy');
  const avgSig = avg('syntropy');
  const avgFid = avg('fidelity');
  const lockPct = Math.round(logRows.filter(r => r.ipr < 0.3).length / n * 100);
  const dsCount = logRows.filter(r => r.controller === 'deepseek-v3').length;
  const domCtrl = dsCount > n/2 ? 'DeepSeek-V3' : 'Llama 3.3';

  document.getElementById('sumCycles').textContent = n;
  document.getElementById('sumIPR').textContent    = avgIPR.toFixed(4);
  document.getElementById('sumS').textContent      = avgS.toFixed(4);
  document.getElementById('sumSig').textContent    = avgSig.toFixed(4);
  document.getElementById('sumFid').textContent    = avgFid.toFixed(5);
  document.getElementById('sumCorr').textContent   = corrections;
  document.getElementById('sumLock').textContent   = lockPct + '%';
  document.getElementById('sumCtrl').textContent   = domCtrl;

  // Inference
  let inf = '';
  if (avgIPR < 0.3) {
    inf = `<strong>Stable MBL phase.</strong> Average IPR ${avgIPR.toFixed(3)} is well below the 0.3 threshold — the system is spending ${lockPct}% of cycles in syntropy lock. Consistent with the paper's reported +38% MBL stability gain under ${logRows[logRows.length-1].source} entropy injection.`;
  } else if (avgIPR < 0.6) {
    inf = `<strong>Marginal stability.</strong> Average IPR ${avgIPR.toFixed(3)} sits between 0.3–0.6. ${corrections} topological correction${corrections!==1?'s':''} have been dispatched. ${lockPct}% of cycles were in lock — consider reviewing correction thresholds or entropy source quality.`;
  } else {
    inf = `<strong>High drift detected.</strong> Average IPR ${avgIPR.toFixed(3)} exceeds 0.6. The system is spending most time outside syntropy lock (only ${lockPct}% locked). ${corrections} corrections fired. This may indicate the simulation needs tuning or the entropy source is insufficient.`;
  }
  if (corrections === 0 && n > 5) inf += ' No corrections have been triggered yet.';
  document.getElementById('sumInference').innerHTML = inf;
}

// Clock is just for the toolbar — removed from header to reduce clutter

// Health check
function setChip(id, state) {
  document.getElementById(id).className = 'chip ' + state;
}
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2000) });
    const d = await r.json();
    setChip('chipQ', d.quantis   ? 'ok' : 'err');
    setChip('chipP', d.pyqpanda  ? 'ok' : 'err');
    setChip('chipO', d.ollama    ? 'ok' : 'err');
    document.getElementById('connStatus').textContent = 'Backend connected';
    document.getElementById('connStatus').style.color = 'var(--green)';
    document.getElementById('vSource').textContent = d.quantis ? 'QRNG' : 'PRNG';
    document.getElementById('vSourceSub').textContent = d.quantis ? 'Quantis USB-4M hardware' : 'Fallback — Quantis not detected';
    document.getElementById('vSource').style.color = d.quantis ? 'var(--green)' : 'var(--amber)';
  } catch {
    ['chipQ','chipP','chipO'].forEach(id => setChip(id, 'unk'));
    document.getElementById('connStatus').textContent = 'Backend not reachable · uvicorn backend:app --host 0.0.0.0 --port 8000';
    document.getElementById('connStatus').style.color = 'var(--red)';
  }
}
checkHealth();
setInterval(checkHealth, 8000);

// Apply live state
function applyState(d) {
  lastState = d;
  updateLog(d);

  // Toolbar
  document.getElementById('tCycle').textContent    = d.cycle;
  document.getElementById('tFidelity').textContent = d.fidelity.toFixed(5);
  document.getElementById('tDsq').textContent      = d.ds_squared.toFixed(6);

  // Live values
  document.getElementById('vEntropy').textContent  = d.entropy.toFixed(4);
  document.getElementById('vSyntropy').textContent = d.syntropy.toFixed(4);
  document.getElementById('vFidelity').textContent = d.fidelity.toFixed(5);

  // IPR
  document.getElementById('iprNum').textContent = d.ipr.toFixed(4);
  const fill = document.getElementById('iprFill');
  fill.style.width      = Math.round(d.ipr * 100) + '%';
  fill.style.background = d.ipr < .3 ? 'var(--green)' : d.ipr < .6 ? 'var(--amber)' : 'var(--red)';

  const cls = d.ipr < .3 ? 'lock' : d.ipr < .6 ? 'marg' : 'drift';
  const txt = d.ipr < .3 ? 'Syntropy locked' : d.ipr < .6 ? 'Marginal — correction pending' : 'Entropy drift — Phase Reset triggered';
  document.getElementById('statusPill').className   = 'status-pill ' + cls;
  document.getElementById('statusPill').textContent = txt;
  document.getElementById('tStatus').className      = 't-status ' + cls;
  document.getElementById('tStatus').textContent    = txt;

  // Correction notice
  const cn = document.getElementById('corrNotice');
  if (d.correction_triggered) {
    cn.classList.add('show');
    setTimeout(() => cn.classList.remove('show'), 3000);
  }

  // Controllers
  const isDS = d.controller === 'deepseek-v3';
  const dsBadge = document.getElementById('dsBadge');
  const llBadge = document.getElementById('llBadge');
  dsBadge.className   = 'ctrl-badge ' + (isDS  ? 'active' : 'idle');
  dsBadge.textContent = isDS  ? 'Active' : 'Standby';
  llBadge.className   = 'ctrl-badge ' + (!isDS ? 'active' : 'idle');
  llBadge.textContent = !isDS ? 'Active' : 'Standby';

  // Memory stores
  const storeMap = { Ep: 'episodic', Se: 'semantic', Pr: 'procedural' };
  const activeClass = { Ep: 'active-ep', Se: 'active-se', Pr: 'active-pr' };
  ['Ep', 'Se', 'Pr'].forEach(k => {
    const el = document.getElementById('mem' + k);
    el.className = 'mem-item' + (d.memory_stores.includes(storeMap[k]) ? ' ' + activeClass[k] : '');
  });
}

// SSE
function connectSSE() {
  if (sseSource) sseSource.close();
  sseSource = new EventSource(`${API}/stream`);
  sseSource.onmessage = e => { try { applyState(JSON.parse(e.data)); } catch {} };
  sseSource.onerror = () => {
    document.getElementById('connStatus').textContent = 'Stream interrupted — retrying...';
    document.getElementById('connStatus').style.color = 'var(--amber)';
  };
}

// Start / Stop
async function startSim() {
  try {
    await fetch(`${API}/start`, { method: 'POST' });
    simRunning = true;
    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnStop').disabled  = false;
    connectSSE();
  } catch {
    alert('Cannot reach backend.\n\nRun:\n  uvicorn backend:app --host 0.0.0.0 --port 8000');
  }
}
async function stopSim() {
  try { await fetch(`${API}/stop`, { method: 'POST' }); } catch {}
  simRunning = false;
  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnStop').disabled  = true;
  if (sseSource) { sseSource.close(); sseSource = null; }
  document.getElementById('tStatus').className   = 't-status idle';
  document.getElementById('tStatus').textContent = 'Stopped';
}

// Chat
const TAGS = {
  episodic:   '<span class="mtag ep">Episodic</span>',
  semantic:   '<span class="mtag se">Semantic</span>',
  procedural: '<span class="mtag pr">Procedural</span>',
};
function addMsg(cls, model, text, stores) {
  const c = document.getElementById('chatMsgs');
  const d = document.createElement('div');
  d.className = 'msg bot ' + cls;
  const tags = stores ? stores.map(s => TAGS[s] || '').join('') : '';
  d.innerHTML = `<div class="msg-who">${model}</div>`
    + `<div class="msg-b">${text}</div>`
    + (tags ? `<div class="msg-stores">${tags}</div>` : '');
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
}
function addUser(t) {
  const c = document.getElementById('chatMsgs');
  const d = document.createElement('div');
  d.className = 'msg user';
  d.innerHTML = `<div class="msg-who">You</div><div class="msg-b">${t}</div>`;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
}
async function sendChat() {
  const inp = document.getElementById('chatIn');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  addUser(msg);

  let model = document.getElementById('modelSel').value;
  if (model === 'auto') model = lastState ? lastState.controller : 'deepseek-v3';

  const tr = document.getElementById('typingRow');
  tr.classList.remove('hidden');
  try {
    const r = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, model }),
    });
    const d = await r.json();
    tr.classList.add('hidden');
    addMsg(model.startsWith('deep') ? 'ds' : 'll', model, d.reply, d.memory_stores);
  } catch {
    tr.classList.add('hidden');
    addMsg('sys', 'System', 'Backend not reachable.', []);
  }
}
