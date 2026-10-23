// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/comfy_watch.js - lost connection to ComfyUI (user request 2026-10-23): a banner that stays up while ComfyUI is
// gone, with "Przeładuj ComfyUI" (POST /api/comfy/reload: a hung local ComfyUI is closed and started again).
// GET /api/comfy/health every 4 s: up | down (refused) | busy (accepts, no answer). Shown only after ComfyUI has
// answered once in this session (a ComfyUI that never started is the Doctor's job); hidden again when it answers.
(function () {
  const POLL_MS = 4000;
  const DOWN_POLLS = 2;     // ~8 s refused before the banner (one missed answer is not a lost connection)
  const BUSY_S = 120;       // accepting but silent this long -> probably hung

  const box = document.createElement('div');
  box.id = 'comfyLost';
  box.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);'
    + 'z-index:10000;display:none;align-items:center;gap:12px;max-width:min(760px,calc(100vw - 32px));padding:10px 14px;'
    + 'background:var(--bg-card,#26262a);border:1px solid var(--accent-warning,#f0703a);border-left-width:4px;'
    + 'border-radius:3px;box-shadow:0 6px 24px rgba(0,0,0,.5);font-size:13px;color:var(--text-main,#e6e6e8)';
  box.innerHTML = `<span id="comfyLostDot" style="width:9px;height:9px;border-radius:50%;background:var(--accent-warning,#f0703a);flex:none"></span>
    <span style="flex:1;min-width:0"><b id="comfyLostTitle"></b><br><span id="comfyLostText" style="color:var(--text-muted,#9a9aa0)"></span></span>
    <button class="btn" id="comfyLostBtn" style="flex:none;padding:5px 14px;font-weight:700;border:0;background:var(--accent-warning,#f0703a);color:var(--on-accent,#1c1c1e)">🔁 Przeładuj ComfyUI</button>`;
  document.body.append(box);
  const $ = (id) => document.getElementById(id);

  let wasUp = false;
  let downPolls = 0;
  let busySince = null;
  let shown = false;
  let reloading = null;     // time the reload was clicked
  let hideT = null;

  function show(title, text, { btn = true, ok = false } = {}) {
    clearTimeout(hideT);
    shown = true;
    $('comfyLostTitle').textContent = title;
    $('comfyLostText').textContent = text;
    $('comfyLostBtn').style.display = btn ? '' : 'none';
    const col = ok ? 'var(--accent-success,#10b981)' : 'var(--accent-warning,#f0703a)';
    box.style.borderColor = col; $('comfyLostDot').style.background = col;
    box.style.display = 'flex';
  }

  async function poll() {
    let h = null;
    try { h = await (await fetch('/api/comfy/health')).json(); } catch (e) { /* MatteDesk itself gone */ }
    if (h) handle(h);
    setTimeout(poll, POLL_MS);
  }

  // footer indicator ("ComfyUI :8001 [Online]"): app.js sets it once at start, kept current from here
  function footer(h) {
    const t = document.getElementById('comfyStatusText');
    const dot = document.querySelector('.status-dot');
    if (!t || !dot) return;
    const port = h.port ? `:${h.port}` : '';
    if (h.launch === 'starting' && h.comfy !== 'up') t.textContent = 'ComfyUI: uruchamiam…';
    else if (h.comfy === 'up') t.textContent = `ComfyUI ${port} [Online]`;
    else if (h.comfy === 'busy') t.textContent = `ComfyUI ${port} [nie odpowiada]`;
    else t.textContent = `ComfyUI: Offline ${h.port ? `(${h.port})` : ''}`;
    dot.classList.toggle('online', h.comfy === 'up');
  }

  async function handle(h) {
    footer(h);
    if (h.comfy === 'up') {
      downPolls = 0; busySince = null;
      if (shown) {
        shown = false; reloading = null;
        show('ComfyUI znów działa', 'Liczenie przerwane przez utratę połączenia – uruchom je jeszcze raz.', { btn: false, ok: true });
        shown = false;
        hideT = setTimeout(() => { box.style.display = 'none'; }, 6000);
        if (window.viewer && window.viewer.checkDoctorStatus) window.viewer.checkDoctorStatus();
      }
      wasUp = true;
      return;
    }
    if (h.comfy === 'down') { downPolls += 1; busySince = null; } else { downPolls = 0; busySince = busySince || Date.now(); }
    if (h.launch === 'starting' && (shown || reloading)) {
      show('Uruchamiam ComfyUI…', `${Math.round((Date.now() - (reloading || Date.now())) / 1000)} s – zwykle 30–90 s.`, { btn: false });
      return;
    }
    if (reloading && h.launch === 'failed') {
      reloading = null;
      let err = '';
      try { err = (await (await fetch('/api/comfy/launch')).json()).error || ''; } catch (e) { /* ignore */ }
      show('Nie udało się uruchomić ComfyUI', err || 'Sprawdź okno ComfyUI.');
      return;
    }
    if (!wasUp && !reloading) return;
    const busyS = busySince ? (Date.now() - busySince) / 1000 : 0;
    if (h.comfy === 'down' && downPolls >= DOWN_POLLS) {
      show('Połączenie z ComfyUI zerwane', h.local ? 'ComfyUI nie odpowiada (zamknięty albo padł). Przeładuj go – potem policz ponownie.'
        : 'ComfyUI na drugim komputerze nie odpowiada – zrestartuj go tam.', { btn: h.local });
    } else if (h.comfy === 'busy' && busyS >= BUSY_S) {
      show('ComfyUI nie odpowiada', `Od ${Math.round(busyS)} s przyjmuje połączenie, ale nie odpowiada (zawieszony?). `
        + (h.local ? 'Przeładuj go, jeśli to nie wczytywanie dużych danych.' : 'Zrestartuj go na drugim komputerze.'), { btn: h.local });
    }
  }

  $('comfyLostBtn').addEventListener('click', async () => {
    const v = window.viewer;
    const ask = 'Zawieszony ComfyUI (jeśli jeszcze działa) zostanie zamknięty i uruchomiony od nowa. To, co liczył, przepadnie – po starcie policz ponownie.';
    if (v && v.uiConfirm && !(await v.uiConfirm(ask, { title: 'Przeładować ComfyUI?', ok: '🔁 Przeładuj' }))) return;
    $('comfyLostBtn').disabled = true;
    reloading = Date.now();
    try {
      const r = await fetch('/api/comfy/reload', { method: 'POST' });
      const d = await r.json();
      if (!r.ok) {
        reloading = null;
        show('Nie udało się przeładować ComfyUI', d.detail || '');
      } else if (d.state === 'failed') {
        reloading = null;
        show('Nie udało się uruchomić ComfyUI', d.error || '');
      } else {
        show('Uruchamiam ComfyUI…', d.killed ? `Zamknięto zawieszony proces ${d.killed}.` : 'Zwykle 30–90 s.', { btn: false });
      }
    } catch (e) {
      reloading = null;
      show('Nie udało się przeładować ComfyUI', String(e));
    }
    $('comfyLostBtn').disabled = false;
  });

  poll();
})();
