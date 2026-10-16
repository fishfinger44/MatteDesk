// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/depth.js – "Głębia" panel (Stage E1e, SPEC 4e). Uses /api/depth/*; modal styles shared with app.js.
(function () {
  function el(html) {
    const t = document.createElement('template');
    t.innerHTML = html.trim();
    return t.content.firstChild;
  }

  const modal = el(`
  <div class="modal-backdrop" id="depthModal" style="display:none;">
    <div class="modal-card modal-lg">
      <div class="modal-header">
        <h3>🌗 Głębia (DepthCrafter)</h3>
        <button class="modal-close-btn" id="closeDepthModalBtn">✕</button>
      </div>
      <div class="modal-body">
        <p class="modal-desc">Twój workflow DepthCrafter (20 kroków / 0.5 / okno 60 / zakładka 20) – narzędzie zmienia tylko ścieżki i zakres klatek.
        Wynik: dysparycja względna (bliżej = więcej), stabilizacja odcinków ze statyczną kamerą, EXR <code>depth.disparity</code> i Group <code>Depth_&lt;shot&gt;.nk</code>.
        Głębia służy tylko do podpowiedzi (kolejność obiektów, przeskoki trackera) – nigdy jako krawędź maski.</p>
        <div class="align-status-card">
          <span class="badge badge-info" id="depthBadge">Nie policzono</span>
          <p class="align-summary-text" id="depthSummary"></p>
        </div>
        <div class="progress-bar-container" id="depthProgressBar" style="display:none;margin-top:12px;">
          <div class="progress-bar-fill" id="depthProgressFill" style="width:0%"></div>
          <span class="progress-text" id="depthProgressText">0%</span>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" id="btnDepthNk" disabled>📄 Group do Nuke'a</button>
        <button class="btn btn-primary" id="btnDepthRun">▶️ Policz głębię</button>
      </div>
    </div>
  </div>`);

  function show(rep) {
    const badge = document.getElementById('depthBadge');
    const sum = document.getElementById('depthSummary');
    if (!rep) {
      badge.className = 'badge badge-info';
      badge.textContent = 'Nie policzono';
      sum.textContent = '';
      document.getElementById('btnDepthNk').disabled = true;
      return;
    }
    const r = rep.qa_raw || {}, s = rep.qa_stabilized || {};
    const drop = r.accel_static ? Math.round(100 * (1 - s.accel_static / r.accel_static)) : 0;
    badge.className = 'badge badge-success';
    badge.textContent = `Gotowe (${rep.method})`;
    const order = (rep.depth_order_hint || []).filter(o => o.visible).map(o => `${o.depth_order}: ${o.name}`).join(', ');
    sum.textContent = `Klatki ${rep.first_frame}–${rep.last_frame}. Odcinki statyczne: ${(rep.static_segments || []).map(s => s.join('–')).join(', ') || 'brak'}. ` +
      `Migotanie statyczne ↓ ${drop}%. ` + (order ? `Propozycja kolejności: ${order}. ` : '') + (rep.notes || []).join(' ');
    document.getElementById('btnDepthNk').disabled = false;
  }

  async function refresh() {
    try {
      const d = await (await fetch('/api/depth/report')).json();
      show(d.exists ? d.report : null);
    } catch (e) { console.error(e); }
  }

  async function run() {
    const btn = document.getElementById('btnDepthRun');
    btn.disabled = true;
    btn.textContent = '⏳ Liczę…';
    try {
      const res = await fetch('/api/depth/compute', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      const d = await res.json();
      if (res.ok) show(d.report); else alert(`Głębia: ${d.detail}`);
    } catch (e) {
      alert(`Błąd połączenia: ${e}`);
    } finally {
      btn.disabled = false;
      btn.textContent = '▶️ Policz głębię';
      document.getElementById('depthProgressBar').style.display = 'none';
    }
  }

  async function exportNk() {
    const d = await (await fetch('/api/depth/export_nk', { method: 'POST' })).json();
    alert(d.success ? `Zapisano: ${d.nk_path}` : `Błąd: ${d.detail}`);
  }

  window.addEventListener('DOMContentLoaded', () => {
    document.body.appendChild(modal);
    window.openDepthModal = () => {
      if (window.viewer && !window.viewer.projectLoaded) { alert('Najpierw otwórz projekt ujęcia'); return; }
      modal.style.display = 'flex';
      refresh();
    };
    document.getElementById('closeDepthModalBtn').addEventListener('click', () => { modal.style.display = 'none'; });
    document.getElementById('btnDepthRun').addEventListener('click', run);
    document.getElementById('btnDepthNk').addEventListener('click', exportNk);
    // progress from the shared WebSocket stream
    const ws = new WebSocket(`ws://${location.host}/ws/events`);
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type !== 'depth_progress') return;
      const pct = m.total ? Math.round(100 * m.done / m.total) : 0;
      document.getElementById('depthProgressBar').style.display = 'flex';
      document.getElementById('depthProgressFill').style.width = `${pct}%`;
      document.getElementById('depthProgressText').textContent = `${m.step} (${pct}%)`;
    };
  });
})();
