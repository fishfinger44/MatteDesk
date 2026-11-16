// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/cleanup_prompt.js – the description of Inpaint (C1), user 2026-11-12: warnings while typing and „Co dostaje
// który model” (POST /api/cleanup/prompt/check – engine/prompt_rules.py, the run's own prompt builders) + „✨ Popraw
// opis” (local Qwen3-VL-4B, engine/prompt_assist.py): no model on disk = the install popup like the other C1 models;
// the proposal is shown first, „Wstaw” puts it into the description field (saved as typed text).
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const P = { t: null, seq: 0, busy: false };
  const C = () => window.MD_CLEANUP;
  const S = () => (C() && C().info && C().info.settings) || null;
  const api = (m, u, b) => window.UI.api(m, u, b);

  function lintHtml(list) {
    return (list || []).map((w) => `<div class="cu-lint-i ${w.level === 'warn' ? 'warn' : 'info'}">${w.level === 'warn' ? '⚠' : 'ⓘ'} ${esc(w.msg)}</div>`).join('');
  }
  async function check() {
    const s = S(); const box = $('cuPromptLint'); if (!s || !box) return;
    if ($('cuPromptBox') && $('cuPromptBox').style.display === 'none') return;
    const seq = ++P.seq;
    let d;
    try {
      d = await api('POST', '/api/cleanup/prompt/check', { prompt: $('cuPrompt').value, prompt_mode: s.prompt_mode || 'remove',
        h3_prompt: $('cuH3Prompt') ? $('cuH3Prompt').value : null });
    } catch (e) { return; }
    if (seq !== P.seq) return;
    box.innerHTML = lintHtml(d.lint) + (d.clip_lint && d.clip_lint.length ? `<div class="cu-lint-sub">Prompt klipu:</div>${lintHtml(d.clip_lint)}` : '');
    const items = (d.final && d.final.items) || [];
    $('cuFinalBody').innerHTML = items.length ? items.map((i) => `<div class="cu-final-i"><div class="cu-final-t">${esc(i.title)}</div><pre>${esc(i.text)}</pre></div>`).join('')
      : '<div class="muted">Ten silnik nie dostaje opisu.</div>';
  }
  const later = () => { clearTimeout(P.t); P.t = setTimeout(check, 450); };

  async function installPopup(st) {
    const v = window.viewer; if (!v || !v.uiDialog) return;
    const vr = st.vram && st.vram.level !== 'ok' ? `<p class="warn">⚠ Ta karta (${esc(st.vram.card_gb)} GB): ${esc(st.vram.text)}</p>` : '';
    const ok = await v.uiDialog({ title: `Zainstalować ${esc(st.label)}?`, ok: 'Zainstaluj', cancel: 'Anuluj', tone: vr ? 'warn' : 'info',
      html: `${vr}<p>„✨ Popraw opis” potrzebuje modelu ${esc(st.label)} – ${esc(st.what)}.</p><p>Pobiera ${esc(st.download)} z Hugging Face; przerwane pobieranie wznawia się. Model ładuje się tylko na czas pracy i znika z karty po 3 min bezczynności.</p><p>Licencja: ${esc(st.license)}.</p>` });
    if (!ok || !v.installWithProgress) return;
    const r = await v.installWithProgress(st.install);
    const res = (r && r.data) || {};
    window.UI.toast(r && r.ok && res.success !== false ? `${st.label}: zainstalowany – kliknij „✨ Popraw opis” jeszcze raz`
      : `${st.label}: instalacja nie powiodła się – ${res.detail || (res.log || []).slice(-1)[0] || 'szczegóły w Doktorze'}`, 9000);
  }

  // `o` (optional, 2026-11-15): another description field – {ta: its textarea, btn: its button, frame, clip: false};
  // default = the Inpaint palette's description (+ the transition clip prompt)
  async function assist(o) {
    const s = S(); if (!s || P.busy) return;
    o = o && o.ta ? o : null;
    const ta0 = o ? o.ta : $('cuPrompt');
    let st;
    try { st = await api('GET', '/api/cleanup/prompt/assist'); } catch (e) { window.UI.toast(`Asystent opisu: ${e.message || e}`, 6000); return; }
    if (!st.available) { await installPopup(st); return; }
    const A = window.MD_CU_ANCHORS;
    const transition = !o && !!(A && A.transition && A.transition() && s.anchor === 'frame');
    const b = o ? o.btn : $('cuAssist'); P.busy = true; if (b) b.disabled = true; const label = b ? b.textContent : ''; if (b) b.textContent = '⏳ Asystent…';
    let d;
    try {
      d = await api('POST', '/api/cleanup/prompt/assist', { text: ta0.value, mode: s.prompt_mode || 'remove', transition,
        frame: o && o.frame != null ? o.frame : (window.viewer ? window.viewer.currentFrame : null) });
    } catch (e) { window.UI.toast(`Asystent opisu: ${e.message || e}`, 9000); return; }
    finally { P.busy = false; if (b) { b.disabled = false; b.textContent = label; } }
    const v = window.viewer;
    const secs = d.seconds ? ` · ${(+d.seconds.load || 0) > 0 ? `ładowanie ${(+d.seconds.load).toFixed(0)} s + ` : ''}${(+d.seconds.gen || 0).toFixed(0)} s` : '';
    const ok = await v.uiDialog({ title: '✨ Propozycja opisu', ok: 'Wstaw', cancel: 'Anuluj', tone: 'info',
      html: `<p class="muted">Twój tekst:</p><pre class="cu-assist-pre">${esc(ta0.value || '(pusty)')}</pre>
        <p class="muted">Propozycja (${esc(st.label)}${esc(secs)}):</p><pre class="cu-assist-pre new">${esc(d.description || '(pusta – zostaje samo polecenie usunięcia)')}</pre>
        ${d.clip && !o ? `<p class="muted">Prompt klipu (przejście):</p><pre class="cu-assist-pre new">${esc(d.clip)}</pre>` : ''}
        ${(d.notes || []).length ? `<ul class="cu-assist-notes">${d.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : ''}
        ${lintHtml(d.lint)}<p class="muted">„Wstaw” zastępuje ${o ? 'opis kotwicy' : 'opis w palecie'} (Ctrl+Z w polu cofa).</p>` });
    if (!ok) return;
    const ta = ta0;
    ta.focus(); ta.select();
    if (!document.execCommand || !document.execCommand('insertText', false, d.description)) ta.value = d.description;   // undoable
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    if (o) return;
    if (d.clip && $('cuH3Prompt')) { const h = $('cuH3Prompt'); h.value = d.clip; h.dispatchEvent(new Event('input', { bubbles: true })); }
    check();
  }

  function init() {
    if (!$('cuPrompt')) { setTimeout(init, 300); return; }
    $('cuPrompt').addEventListener('input', later);
    if ($('cuH3Prompt')) $('cuH3Prompt').addEventListener('input', later);
    $('cuPromptMode').addEventListener('click', () => setTimeout(check, 350));   // after the mode is saved
    $('cuAssist').addEventListener('click', () => assist());
    $('cuFinal').addEventListener('toggle', () => { if ($('cuFinal').open) check(); });
  }
  window.MD_CU_PROMPT = { check, later, assist };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
