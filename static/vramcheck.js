// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/vramcheck.js – GPU memory forecast BEFORE a VideoMaMa run / frame preview (user request 2026-10-26: VideoMaMa at
// 2048 px on a 12 GB card spilled into RAM and failed after ~9 min without a word). A click on „▶ Policz VideoMaMa” or its
// „👁 Podgląd klatek” first asks GET /api/object/{o}/vram_forecast (engine/vram_forecast.py); over the card -> a window:
// set the suggested resolution and compute / compute anyway / cancel. Unknown forecast, no answer, an error: the click
// goes through as before (the forecast never blocks). Capture-phase listener, so objects.js stays untouched.
(function () {
  const BUTTONS = { vmRun: null, vmStagePv: 'frame' };   // id -> the forecast is of the work area (null) or of the preview window
  let passing = false;
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  async function ask(btn) {
    const obj = window.MD_OBJ && window.MD_OBJ.name;
    if (!obj) return 'go';
    const frame = BUTTONS[btn.id] === 'frame' && window.viewer ? window.viewer.currentFrame : null;
    let r;
    try {
      r = await window.UI.api('GET', `/api/object/${encodeURIComponent(obj)}/vram_forecast?stage=videomama${frame != null ? `&frame=${frame}` : ''}`);
    } catch (e) { return 'go'; }
    if (!r || r.state !== 'over') return 'go';
    const sug = r.suggest_side && r.suggest_side < r.max_side ? r.suggest_side : null;
    let anyway = false;
    const p = window.UI.dialog({
      title: '⚠ Za mało pamięci GPU', tone: 'warn',
      html: (r.text || []).map((t) => `<div style="margin-bottom:6px">${esc(t)}</div>`).join('') +
        `<div style="margin-top:10px"><button type="button" class="btn btn-ghost btn-sm" data-vf="anyway" title="Bez zmiany ustawień – z przelewem do RAM">Licz mimo to</button></div>`,
      ok: sug ? `Ustaw ${sug} px i licz` : 'Licz mimo to', cancel: 'Anuluj',
    });
    const box = [...document.querySelectorAll('.md-dialog')].pop();
    const any = box && box.querySelector('[data-vf="anyway"]');
    if (any) {
      if (!sug) any.style.display = 'none';
      any.addEventListener('click', () => { anyway = true; const c = box.querySelector('button[data-r="0"]'); if (c) c.click(); });
    }
    const ok = await p;
    if (anyway) return 'go';
    if (!ok) return 'stop';
    if (sug) {
      try {
        await window.UI.api('PUT', `/api/object/${encodeURIComponent(obj)}/videomama`, { max_side: sug });
        window.UI.toast(`VideoMaMa: rozdzielczość ${sug} px`, 3000);
        if (window.MD_OBJ.reload) await window.MD_OBJ.reload();
      } catch (e) { window.UI.toast(`Nie zapisano rozdzielczości: ${e.message}`, 5000); return 'stop'; }
    }
    return 'go';
  }

  document.addEventListener('click', async (e) => {
    if (passing) return;
    const btn = e.target && e.target.closest && e.target.closest(Object.keys(BUTTONS).map((id) => `#${id}`).join(','));
    if (!btn || btn.disabled || document.body.classList.contains('job-busy')) return;   // busy: jobs.js says what runs
    e.preventDefault(); e.stopImmediatePropagation();
    const go = await ask(btn);
    if (go !== 'go') return;
    passing = true;
    try { btn.click(); } finally { passing = false; }
  }, true);
  window.MD_VRAM = { ask };
})();
