// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/ui.js – the one API call, toast and dialog of the page (audit 2026-11-02 step 7). Loaded before the other scripts.
//   UI.api(method, url, body)  -> parsed JSON; a non-OK answer throws an Error with .status and .detail (the server's)
//   UI.read(response)          -> { ok, data, error } of a fetch() whose caller wants to handle the failure itself
//   UI.toast(text, ms)         -> the short message above the timeline / job bar
//   UI.dialog({...})           -> Promise<boolean>: a message window (OK / Anuluj), tone info | ok | warn | error | danger
//   UI.form({...})             -> Promise<value | null>: a window with fields, its own validation and a server call
//   UI.exportMenu(btn, {get, what, off}) -> the „⇪ Export” menu: Nuke clipboard / .nk file / After Effects (later)
//   UI.copyText(text)          -> Promise<bool>: text to the clipboard (async API, else execCommand)
//   UI.confirmDelete({...})    -> Promise<false | value>: the warning before a PERMANENT delete (red, Enter never deletes)
//   UI.bytes(n)                -> "1,2 GB" (Polish decimal comma, as engine/deletion.py human)
// The colours are the tokens of index.css (:root); the look of the window is .md-dialog / .md-modal in index.css.
window.UI = (() => {
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // A non-JSON body (e.g. a plain-text 500) or a non-OK status gives a readable `error` (HTTP status + the server's
  // `detail` / text) instead of a JSON.parse SyntaxError.
  async function read(res) {
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
    if (res.ok && data !== null) return { ok: true, data, error: '' };
    let msg = data && data.detail !== undefined ? data.detail : (data && data.error) || text.trim().slice(0, 300) || res.statusText;
    if (msg && typeof msg === 'object') msg = msg.detail || JSON.stringify(msg);
    return { ok: false, data: data || {}, error: `HTTP ${res.status}: ${msg}` };
  }

  let wdMissingAt = 0;
  async function api(method, url, body) {
    const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const det = d.detail;
      if (det && det.code === 'work_dir_missing' && Date.now() - wdMissingAt > 15000) {   // project folder vanished: one toast, not one per request
        wdMissingAt = Date.now(); toast(det.detail, 10000);
      }
      const e = new Error(det && typeof det === 'object' ? (det.detail || JSON.stringify(det)) : (det || r.statusText));
      e.status = r.status; e.detail = det; throw e;
    }
    return d;
  }

  let toastTimer = null;
  function toast(msg, ms = 3500) {
    console.log('[MatteDesk]', msg);
    let t = document.getElementById('mdToast');
    if (!t) { t = document.createElement('div'); t.id = 'mdToast'; t.className = 'md-toast'; document.body.appendChild(t); }
    // low on the screen, just above the timeline / job bar where the eye is (user 2026-10-22: the top one was missed)
    const foot = document.querySelector('.app-footer');
    const jobBar = document.getElementById('jobBar');
    const barUp = jobBar && getComputedStyle(jobBar).display !== 'none';
    t.style.bottom = `${Math.round((foot ? foot.getBoundingClientRect().height : 0) + (barUp ? 56 : 16))}px`;
    const err = /^\s*(Błąd|Error|⚠|✗|❌)/.test(String(msg));
    t.classList.toggle('md-toast-err', err);
    t.textContent = msg; t.style.display = 'block';
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.style.display = 'none'; }, err ? Math.max(ms, 7000) : ms);
  }

  // Ctrl / Meta / Alt combinations and F-keys pass through every window (F5, F12, Ctrl+L, Ctrl+Z of the sliders)
  const browserKey = (e) => e.ctrlKey || e.metaKey || e.altKey || /^F\d{1,2}$/.test(e.key);

  // the window: backdrop, card, title, body, buttons; `onKey(e)` runs for Esc / Enter, other plain keys stop here (no
  // viewer shortcuts behind an open window). `close()` takes the key handler away with it.
  function shell({ title, html, ok, cancel, tone, okClass, okTitle, cancelTitle, report, onKey, onBackdrop }) {
    const bg = document.createElement('div');
    bg.className = 'md-dialog';
    bg.innerHTML = `<div role="dialog" class="md-modal md-tone-${esc(tone)}">
        <div class="md-modal-title">${esc(title)}</div>
        <div class="md-modal-body">${html}</div>
        <div class="md-modal-btns">
          ${report ? '<button type="button" class="btn btn-secondary md-modal-report" data-report="1" title="Wyślij autorowi opis tego błędu z diagnostyką (przed wysłaniem zobaczysz całość)">🐞 Zgłoś problem</button>' : ''}
          ${cancel ? `<button type="button" class="btn btn-secondary" data-r="0" title="${esc(cancelTitle || 'Zamknij (Esc)')}">${esc(cancel)}</button>` : ''}
          <button type="button" class="btn ${okClass || 'btn-primary md-tone-btn'}" data-r="1"${okTitle ? ` title="${esc(okTitle)}"` : ''}>${esc(ok)}</button>
        </div></div>`;
    const key = (e) => {
      if (browserKey(e)) return;
      e.stopPropagation();
      if (e.key === 'Escape' || e.key === 'Enter') { e.preventDefault(); onKey(e); }
    };
    const close = () => { document.removeEventListener('keydown', key, true); bg.remove(); };
    bg.addEventListener('mousedown', (e) => { if (e.target === bg) onBackdrop(); });
    document.addEventListener('keydown', key, true);
    document.body.appendChild(bg);
    return { bg, close, okBtn: bg.querySelector('button[data-r="1"]'), cancelBtn: bg.querySelector('button[data-r="0"]') };
  }

  function dialog({ title, html = '', ok = 'OK', cancel = null, tone = 'info' }) {
    return new Promise((resolve) => {
      const done = (v) => { w.close(); resolve(v); };
      const w = shell({ title, html, ok, cancel, tone, report: tone === 'error' && !!window.mdReport,
                        onKey: (e) => done(e.key === 'Enter'), onBackdrop: () => { if (cancel) done(false); } });
      w.okBtn.addEventListener('click', () => done(true));
      if (w.cancelBtn) w.cancelBtn.addEventListener('click', () => done(false));
      const rep = w.bg.querySelector('button[data-report]');
      if (rep) rep.addEventListener('click', () => {   // the error's title + text go with the report (engine/report.py context)
        const text = (w.bg.querySelector('.md-modal-body') || {}).innerText || '';
        done(false);
        window.mdReport.open({ source: 'error_dialog', error_title: String(title || ''), error_text: text });
      });
      w.okBtn.focus();
    });
  }

  // `check(root)` -> '' / null (null: nothing to say, still not ready) or the reason; `submit(root)` -> the value the promise
  // resolves with (undefined = true); what it throws is shown in the window, which stays open. Esc / Anuluj -> null.
  // `focus`: the id of the element to focus. `enter: false`: Enter does not confirm (a destructive window).
  function form({ title, html, ok = 'OK', cancel = 'Anuluj', tone = 'info', okClass, okTitle, cancelTitle, check, submit, focus, enter = true, select = false }) {
    return new Promise((resolve) => {
      const done = (v) => { w.close(); resolve(v); };
      const err = () => w.bg.querySelector('.md-modal-err');
      const valid = () => {
        const msg = check ? check(w.bg) : '';
        err().textContent = msg || '';
        w.okBtn.disabled = msg !== '';
        return msg === '';
      };
      const go = async () => {
        if (w.okBtn.disabled || !valid()) return;
        w.okBtn.disabled = true; err().textContent = '';
        try { const v = submit ? await submit(w.bg) : true; done(v === undefined ? true : v); }
        catch (e) { err().textContent = e.message || String(e); w.okBtn.disabled = false; }
      };
      const w = shell({ title, html: `${html}<div class="md-modal-err"></div>`, ok, cancel, tone, okClass, okTitle, cancelTitle,
                        onKey: (e) => { if (e.key === 'Escape') done(null); else if (enter) go(); }, onBackdrop: () => done(null) });
      w.okBtn.addEventListener('click', go);
      w.cancelBtn.addEventListener('click', () => done(null));
      w.bg.addEventListener('input', valid);
      valid();
      const f = focus && document.getElementById(focus);
      (f || w.cancelBtn).focus();
      if (f && select) f.select();
    });
  }

  // Bytes as text with a Polish decimal comma (the same steps as engine/deletion.py human): 0 B, 3,4 kB, 12,0 MB, 1,2 GB
  function bytes(n) {
    let v = Math.max(0, Number(n) || 0);
    if (v < 1024) return `${Math.round(v)} B`;
    for (const u of ['kB', 'MB', 'GB', 'TB']) {
      v /= 1024;
      if (v < 1024 || u === 'TB') return `${v.toFixed(1).replace('.', ',')} ${u}`;
    }
    return `${v} B`;
  }

  // The confirmation before a PERMANENT delete (user decision 2026-11-13: nothing goes to a project trash any more – it
  // is deleted from disk for good). `what` (html) names what goes, `size` (bytes, when known) the space it frees,
  // `extra` (html) more lines (e.g. a checkbox – `submit(root)` reads it, its value is what the promise resolves with).
  // Red window, the focus on „Anuluj”, Enter never deletes, Esc = cancel. Resolves false (cancelled) or true / submit's value.
  async function confirmDelete({ title, what, size = null, ok = 'Usuń na stałe', extra = '', submit, okTitle }) {
    const freed = size != null && Number(size) > 0 ? ` Zwolni się <b>${esc(bytes(size))}</b>.` : '';
    const v = await form({ title, tone: 'danger', ok, okClass: 'btn-danger', enter: false, okTitle,
      cancelTitle: 'Zamknij bez usuwania (Esc)', submit,
      html: `<div>${what}</div>
        <div class="md-delete-warn"><b>Usunięte z dysku na stałe – nie da się cofnąć.</b> Nic nie trafia do kosza.${freed}</div>${extra}` });
    return v === null ? false : v;
  }

  // text to the clipboard (Nuke scripts: Ctrl+V in the node graph); the old execCommand path when the async API is
  // refused (no focus / permission). true = copied
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) {
      const t = document.createElement('textarea'); t.value = text; t.style.position = 'fixed'; t.style.opacity = '0';
      document.body.appendChild(t); t.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
      t.remove();
      return ok;
    }
  }
  // the „⇪ Export” button of a palette section (user decision 2026-11-16: every section has one; a click = the target
  // format). items: [{label, title?, disabled?, run: async () => {}}]; one menu open at a time, Esc / click outside closes
  let openMenu = null;
  function closeMenu() { if (openMenu) { openMenu.remove(); openMenu = null; } }
  function menu(anchor, items) {
    const was = openMenu && openMenu.anchor === anchor;
    closeMenu();
    if (was) return;                              // a second click on the same button closes it
    const m = document.createElement('div');
    m.className = 'md-menu'; m.anchor = anchor;
    for (const it of items) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'md-menu-item'; b.textContent = it.label;
      if (it.title) b.title = it.title;
      b.disabled = !!it.disabled;
      b.addEventListener('click', (e) => { e.stopPropagation(); closeMenu(); if (it.run) it.run(); });
      m.appendChild(b);
    }
    document.body.appendChild(m);
    const r = anchor.getBoundingClientRect();
    const left = Math.min(r.left, window.innerWidth - m.offsetWidth - 8);
    const top = r.bottom + m.offsetHeight + 4 > window.innerHeight ? r.top - m.offsetHeight - 4 : r.bottom + 4;
    m.style.left = `${Math.max(8, left)}px`; m.style.top = `${Math.max(8, top)}px`;
    openMenu = m;
  }
  document.addEventListener('mousedown', (e) => { if (openMenu && !openMenu.contains(e.target) && e.target !== openMenu.anchor) closeMenu(); }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });
  // the menu of a „⇪ Export” button: Nuke clipboard / Nuke .nk file / After Effects (later). `get` = async () =>
  // {text, path} of the node(s); `off` = why it cannot be exported (items disabled with that reason)
  function exportMenu(anchor, { get, what, off = '', status }) {
    const say = status || ((t, k) => toast(t, k));
    menu(anchor, [
      { label: "⧉ Nuke – do schowka (Ctrl+V)", title: off || `${what} do schowka – w Nuke'u Ctrl+V w grafie węzłów (zapisany też jako plik .nk)`, disabled: !!off,
        run: async () => { try { const r = await get(); const ok = await copyText(r.text);
          say(ok ? `${what} w schowku – w Nuke'u Ctrl+V w grafie węzłów (plik: ${r.path})` : `Schowek niedostępny – plik: ${r.path}`, ok ? 'ok' : 'warn');
        } catch (e) { say(`${what} → Nuke: ${e.message || e}`, 'err'); } } },
      { label: '💾 Nuke – plik .nk', title: off || `${what} jako plik .nk (File → Import Script albo przeciągnij do Nuke'a)`, disabled: !!off,
        run: async () => { try { const r = await get(); say(`${what} – plik .nk: ${r.path}`, 'ok'); } catch (e) { say(`${what} → Nuke: ${e.message || e}`, 'err'); } } },
      { label: 'After Effects – wkrótce', title: 'Eksport do After Effects jeszcze nie jest zrobiony', disabled: true },
    ]);
  }
  return { esc, read, api, toast, dialog, form, confirmDelete, bytes, copyText, menu, exportMenu };
})();
