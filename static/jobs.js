// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/jobs.js - what is being computed now (engine/jobs.py, GET /api/jobs) + "Przerwij".
// A running job shows a centred progress window that blocks the rest of the UI (so nothing can break the pipeline);
// "Zwiń do paska" turns it into a small bar above the timeline (browse frames meanwhile) – buttons that would start
// another computation or switch the project stay locked with a hint (the backend refuses them anyway – HTTP 409).
(function () {
  const $ = (id) => document.getElementById(id);
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  const css = document.createElement('style');
  css.textContent = `
    body.job-busy .job-locked{opacity:.45;cursor:not-allowed!important}
    #jobModal{position:fixed;inset:0;z-index:1000;display:none;align-items:center;justify-content:center;background:rgba(5,8,14,.55)}
    #jobCard{width:min(520px,calc(100vw - 32px));padding:18px 20px;border-radius:12px;background:var(--bg-panel,#1f1f22);
      border:1px solid var(--bg-hover,#303035);box-shadow:0 10px 40px rgba(0,0,0,.5);color:var(--text-main,#e6e6e8);font-size:13px}
    #jobCard .t{font-size:15px;font-weight:600;margin-bottom:4px} #jobCard .s{color:var(--text-muted,#9a9aa0);min-height:18px}
    .jobTrack{height:8px;background:var(--bg-hover,#303035);border-radius:4px;overflow:hidden;margin:12px 0 6px}
    .jobFill{height:100%;width:0;background:var(--accent-primary,#e8a33d);transition:width .3s}
    @keyframes jobInd{0%{margin-left:-30%}100%{margin-left:100%}} .jobFill.ind{width:30%!important;animation:jobInd 1.4s linear infinite}
    #jobBar{position:fixed;left:50%;transform:translateX(-50%);width:min(900px,calc(100vw - 32px));z-index:900;display:none;
      align-items:center;gap:10px;padding:6px 12px;border-radius:8px;background:var(--bg-panel,#1f1f22);
      border:1px solid var(--bg-hover,#303035);box-shadow:0 4px 16px rgba(0,0,0,.4);color:var(--text-main,#e6e6e8);font-size:12px}
    .jobDet{margin-top:10px;padding:8px 10px;border-radius:8px;background:rgba(255,255,255,.03);font-size:12px;display:none}
    .jobDet .r{display:flex;align-items:center;gap:8px;margin:3px 0} .jobDet .l{width:64px;color:var(--text-muted,#9a9aa0)}
    .jobDet .jobTrack{flex:1;margin:0;height:6px} .jobDet .v{width:150px;text-align:right;color:var(--text-muted,#9a9aa0)}
    .jobDet .warn{color:#fbbf24} .jobDet .err{color:#f87171} .jobDet pre{white-space:pre-wrap;max-height:120px;overflow:auto;font-size:11px}
    #jobBarDet{position:fixed;left:50%;transform:translateX(-50%);width:min(900px,calc(100vw - 32px));z-index:899;display:none;
      background:var(--bg-panel,#1f1f22);border:1px solid var(--bg-hover,#303035);border-radius:8px}
    #jobToast{position:fixed;left:50%;transform:translateX(-50%);z-index:950;display:none;padding:8px 14px;border-radius:8px;
      background:#3b2a0a;border:1px solid #f0703a;color:#fde68a;font-size:12px;max-width:80vw}`;
  document.head.appendChild(css);

  const modal = document.createElement('div');
  modal.id = 'jobModal';
  modal.innerHTML = `<div id="jobCard"><div class="t" id="jmTitle"></div><div class="s" id="jmStep"></div>
    <div class="jobTrack"><div class="jobFill" id="jmFill"></div></div>
    <div class="s" id="jmMeta"></div>
    <div class="jobDet" id="jmDet"></div>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:14px">
      <button class="btn btn-ghost" id="jmDetBtn" title="GPU / VRAM / RAM, co liczy Comfy, krok i ETA" style="margin-right:auto">Szczegóły ▾</button>
      <button class="btn btn-secondary" id="jmMin" title="Okno znika, liczenie trwa; postęp na pasku nad osią czasu [Esc]">Zwiń do paska</button>
      <button class="btn" id="jmCancel" style="background:#7f1d1d;color:#fff">Przerwij</button></div></div>`;
  const bar = document.createElement('div');
  bar.id = 'jobBar';
  bar.innerHTML = `<span id="jobDot" style="width:8px;height:8px;border-radius:50%;background:#e8a33d;flex:none"></span>
    <span id="jobText" style="flex:none;max-width:45%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
    <div class="jobTrack" style="flex:1;min-width:60px;height:6px;margin:0"><div class="jobFill" id="jobFill"></div></div>
    <span id="jobMeta" style="flex:none;color:var(--text-muted,#9a9aa0)"></span>
    <button class="btn" id="jobDetBtn" style="flex:none;padding:2px 8px;font-size:12px" title="GPU / VRAM / RAM, co liczy Comfy">Szczegóły ▾</button>
    <button class="btn" id="jobExpand" style="flex:none;padding:2px 8px;font-size:12px" title="Pokaż okno postępu">⤢</button>
    <button class="btn" id="jobCancel" style="flex:none;padding:2px 10px;font-size:12px">Przerwij</button>`;
  const toast = document.createElement('div');
  toast.id = 'jobToast';
  const barDet = document.createElement('div');
  barDet.id = 'jobBarDet'; barDet.innerHTML = '<div class="jobDet" id="jobDet" style="display:block;margin:0"></div>';
  document.body.append(modal, bar, toast, barDet);

  // ---- "Szczegóły" (Z19): GET /api/jobs/details every second while open; open/closed remembered (localStorage, try/catch)
  let detOpen = false;
  try { detOpen = localStorage.getItem('md_job_details') === '1'; } catch (e) { /* storage blocked */ }
  let detTimer = null;
  const gb = (b) => (b == null ? '?' : (b / 2 ** 30).toFixed(1));
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  function barRow(label, frac, text, cls) {
    const pct = Math.max(0, Math.min(100, (frac || 0) * 100));
    const col = cls === 'err' ? '#ef4444' : pct > 90 ? '#f0703a' : '#e8a33d';
    return `<div class="r"><span class="l">${label}</span><div class="jobTrack"><div class="jobFill" style="width:${pct}%;background:${col}"></div></div><span class="v">${text}</span></div>`;
  }
  function detHtml(d) {
    const out = [];
    const g = d.gpu && d.gpu[0];
    const ci = (d.job && d.job.comfy_info) || {};
    const oom = /pamięci GPU/.test(ci.error || '');
    if (g) out.push(barRow('GPU', (g.util_pct || 0) / 100, `${g.util_pct ?? '?'} % · ${g.temp_c ?? '?'} °C`));
    const dev = d.comfy && d.comfy.devices && d.comfy.devices[0];
    if (g && g.mem_total_mb) out.push(barRow('VRAM', g.mem_used_mb / g.mem_total_mb, `${(g.mem_used_mb / 1024).toFixed(1)} / ${(g.mem_total_mb / 1024).toFixed(1)} GB`, oom ? 'err' : ''));
    else if (dev && dev.vram_total) out.push(barRow('VRAM', 1 - dev.vram_free / dev.vram_total, `${gb(dev.vram_total - dev.vram_free)} / ${gb(dev.vram_total)} GB`, oom ? 'err' : ''));
    const ram = d.comfy && d.comfy.ram;
    if (ram && ram.total) out.push(barRow('RAM', 1 - ram.free / ram.total, `${gb(ram.total - ram.free)} / ${gb(ram.total)} GB`));
    let cs;
    if (!d.comfy || !d.comfy.alive) cs = '<span class="err">Comfy: nie odpowiada</span>';
    else if (ci.state === 'ours') cs = 'Comfy: liczy NASZE zadanie';
    else if (ci.state === 'foreign') cs = `<span class="warn">Comfy: liczy CUDZE zadanie, nasze czeka (${ci.ahead} przed nami)</span>`;
    else if (ci.state === 'queued') cs = `Comfy: nasze w kolejce (${ci.ahead} przed nami)`;
    else if ((d.comfy.queue || {}).foreign_running && d.comfy.queue.foreign_running.length) cs = '<span class="warn">Comfy: liczy cudze zadanie</span>';
    else cs = 'Comfy: czeka';
    const q = (d.comfy && d.comfy.queue) || null;
    out.push(`<div class="r">${cs}${q ? ` · kolejka: ${q.running.length} liczy, ${q.pending.length} czeka` : ''}</div>`);
    if (d.job) {
      const j = d.job;
      out.push(`<div class="r">${esc(j.step || j.label)}${j.total ? ` · ${j.done}/${j.total}` : ''}${j.eta_s != null ? ` · ETA ${mmss(j.eta_s)}` : ''}</div>`);
      if (ci.error) out.push(`<div class="r err">${esc(ci.error)}</div>${ci.raw_error ? `<details><summary>surowy komunikat</summary><pre>${esc(JSON.stringify(ci.raw_error, null, 1))}</pre></details>` : ''}`);
    }
    return out.join('');
  }
  async function detPoll() {
    clearTimeout(detTimer);
    if (!detOpen || !busy) { $('jmDet').style.display = 'none'; barDet.style.display = 'none'; return; }
    try {
      const d = await (await fetch('/api/jobs/details')).json();
      const html = detHtml(d);
      $('jmDet').innerHTML = html; $('jobDet').innerHTML = html;
    } catch (e) { /* backend gone */ }
    const minimizedNow = busy && (busy.quiet || minimized[busy.id]);
    $('jmDet').style.display = minimizedNow ? 'none' : 'block';
    barDet.style.display = minimizedNow ? 'block' : 'none';
    detTimer = setTimeout(detPoll, 1000);
  }
  function setDet(open) {
    detOpen = open;
    try { localStorage.setItem('md_job_details', open ? '1' : '0'); } catch (e) { /* storage blocked */ }
    $('jmDetBtn').textContent = open ? 'Szczegóły ▴' : 'Szczegóły ▾'; $('jobDetBtn').textContent = $('jmDetBtn').textContent;
    detPoll();
  }
  $('jmDetBtn').addEventListener('click', () => setDet(!detOpen));
  $('jobDetBtn').addEventListener('click', () => setDet(!detOpen));
  $('jmDetBtn').textContent = detOpen ? 'Szczegóły ▴' : 'Szczegóły ▾'; $('jobDetBtn').textContent = $('jmDetBtn').textContent;

  let busy = null;          // running job (last one) or null
  let minimized = {};       // job id -> true when the user collapsed the window
  let lastShownEnd = 0;
  let toastTimer = null;

  function place() {
    const foot = document.querySelector('.app-footer');
    const h = foot ? foot.getBoundingClientRect().height : 0;
    bar.style.bottom = `${Math.round(h + 8)}px`;
    barDet.style.bottom = `${Math.round(h + 46)}px`;
    toast.style.bottom = `${Math.round(h + 52)}px`;
  }
  window.addEventListener('resize', place);
  place();
  function showToast(t) {
    toast.textContent = t; toast.style.display = 'block';
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 4000);
  }

  async function cancel(btn) {
    if (!(await window.viewer.uiConfirm('Przerwać bieżące liczenie?\nSkończone klatki zostają, niedokończona wersja nie zostanie aktywowana.', { ok: 'Przerwij' }))) return;
    btn.disabled = true;
    try { await fetch('/api/jobs/cancel', { method: 'POST' }); } catch (e) { /* backend gone */ }
  }
  $('jmCancel').addEventListener('click', () => cancel($('jmCancel')));
  $('jobCancel').addEventListener('click', () => cancel($('jobCancel')));
  $('jmMin').addEventListener('click', () => { if (busy) minimized[busy.id] = true; render([busy].filter(Boolean)); });
  $('jobExpand').addEventListener('click', () => { if (busy) delete minimized[busy.id]; render([busy].filter(Boolean)); });

  // Buttons that start a computation or switch the project: locked while something is being computed.
  const HEAVY = ['#p0Proxy', '#confirmOpenProjectBtn', '#p0Measure', '#p0Apply',
    '#btnMeasureClean', '#btnRunClean', '#btnRunCameraTrack', '#btnDepthRun', '#st1Detect', '#st2Crops',
    '#st3Run', '#st3Rerun', '#st4Final', '#z4ApplyShot', '#z4CpAuto', '#st5Export', '#st5ExportOne', '#st5ExportWhole', '#st5QRun', '.doctor-install', '#btnSavePlate',
    // newer stage buttons and the floating palettes (2026-10-16: "Run MatAnyone" on the palette did nothing visible during VideoMaMa)
    '#st1All', '#ppAll', '#st3Ma', '#epCompute', '#vmRun', '#lx6Run', '#exAnalyze', '#exCompute', '#zpApply', '#b2Apply'].join(',');
  document.addEventListener('click', (e) => {
    const b = busy && e.target.closest && e.target.closest(HEAVY);
    if (!b) return;
    e.preventDefault(); e.stopImmediatePropagation();
    showToast(`Trwa: ${busy.label} – poczekaj na koniec albo kliknij „Przerwij”.`);
  }, true);
  // while the window is open, keyboard shortcuts must not reach the editor (Esc = collapse)
  document.addEventListener('keydown', (e) => {
    if (modal.style.display !== 'flex') return;
    if (e.key === 'Escape' && busy) { minimized[busy.id] = true; render([busy]); }
    e.stopImmediatePropagation(); e.preventDefault();
  }, true);
  function markLocked() {
    document.body.classList.toggle('job-busy', !!busy);
    document.querySelectorAll(HEAVY).forEach((el) => {
      el.classList.toggle('job-locked', !!busy);
      if (busy) { if (!el.dataset.tTitle) el.dataset.tTitle = el.title || ' '; el.title = `Trwa: ${busy.label}`; }
      else if (el.dataset.tTitle) { el.title = el.dataset.tTitle.trim(); delete el.dataset.tTitle; }
    });
  }

  function fill(el, j) {
    el.classList.toggle('ind', !j.total);
    el.style.width = j.total ? `${Math.min(100, (100 * j.done) / j.total)}%` : '';
  }
  function meta(j) {
    return [j.total ? `klatki ${j.done}/${j.total}` : '', j.comfy || '', mmss(j.elapsed_s)].filter(Boolean).join(' · ');
  }
  function render(run) {
    const j = run.length ? run[run.length - 1] : null;
    if (j && j.quiet) {   // short job (SAM3 on one frame): small bar only, no blocking window
      $('jobDot').style.background = '#a855f7';
      $('jobText').textContent = `${j.label}${j.step ? ' · ' + j.step : ''}`;
      fill($('jobFill'), j); $('jobMeta').textContent = meta(j);
      $('jobCancel').style.display = ''; $('jobExpand').style.display = 'none';
      modal.style.display = 'none'; bar.style.display = 'flex';
      return;
    }
    if (j) {
      const stopping = j.step === 'przerywanie…';
      const more = run.length > 1 ? ` (+${run.length - 1})` : '';
      // centred window
      $('jmTitle').textContent = j.label + more;
      $('jmStep').textContent = j.step || 'start…';
      fill($('jmFill'), j);
      $('jmMeta').textContent = meta(j);
      $('jmCancel').disabled = stopping; $('jobCancel').disabled = stopping;
      // bar
      $('jobDot').style.background = stopping ? '#f0703a' : '#e8a33d';
      $('jobText').textContent = `${j.label}${j.step ? ' · ' + j.step : ''}${more}`;
      fill($('jobFill'), j);
      $('jobMeta').textContent = meta(j);
      $('jobCancel').style.display = ''; $('jobExpand').style.display = '';
      const min = !!minimized[j.id];
      modal.style.display = min ? 'none' : 'flex';
      bar.style.display = min ? 'flex' : 'none';
      return;
    }
    modal.style.display = 'none';
  }

  async function poll() {
    let jobs = [];
    try { jobs = (await (await fetch('/api/jobs')).json()).jobs || []; } catch (e) { setTimeout(poll, 3000); return; }
    const run = jobs.filter((j) => j.state === 'running');
    if (busy && !run.length) window.dispatchEvent(new CustomEvent('matte-job-finished', { detail: busy }));
    const wasBusy = !!busy;
    busy = run.length ? run[run.length - 1] : null;
    if (!!busy !== wasBusy) detPoll();
    if (busy && busy.quiet) { render(run); document.body.classList.remove('job-busy'); setTimeout(poll, 700); return; }
    markLocked();
    if (run.length) {
      render(run);
    } else {
      modal.style.display = 'none';
      const last = jobs.filter((j) => j.ended).sort((a, b) => b.ended - a.ended)[0];
      if (last && last.ended !== lastShownEnd && Date.now() / 1000 - last.ended < 8) {
        const txt = { done: '✔ gotowe', cancelled: '■ przerwane', error: '✖ błąd' }[last.state] || last.state;
        $('jobDot').style.background = { done: '#10b981', cancelled: '#f0703a', error: '#ef4444' }[last.state] || '#6e6e75';
        $('jobText').textContent = `${last.label} · ${txt}${last.error ? ': ' + last.error : ''}`;
        $('jobFill').classList.remove('ind'); $('jobFill').style.width = '100%';
        $('jobMeta').textContent = mmss(last.elapsed_s);
        $('jobCancel').style.display = 'none'; $('jobExpand').style.display = 'none';
        bar.style.display = 'flex';
        setTimeout(() => { lastShownEnd = last.ended; }, 6000);
      } else {
        bar.style.display = 'none';
      }
    }
    setTimeout(poll, run.length ? 1000 : 2000);
  }
  poll();
})();
