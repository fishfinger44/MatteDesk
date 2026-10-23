// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// "Zgłoś problem" (engine/report.py): description + optional steps / e-mail / window screenshot -> the server builds the
// anonymised package -> the user sees ALL of it -> consent -> sent to the report proxy (GitHub issue of the author).
// Opened from the header button (folds into the "⋯" menu on narrow windows), from the "O programie" window and from
// every error window (uiDialog tone 'error': button "🐞 Zgłoś problem" with the error's title and text).
// Screenshot: html2canvas loaded from cdnjs only when the user ticks the box (MIT; the app itself stays offline).
(function () {
  const H2C = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
  const MAX_W = 1600;   // screenshot width cap (keeps the PNG well under the 3 MB limit of the server)
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  let h2cLoading = null;

  function loadH2C() {
    if (window.html2canvas) return Promise.resolve(window.html2canvas);
    if (!h2cLoading) {
      h2cLoading = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = H2C; s.crossOrigin = 'anonymous';
        s.onload = () => (window.html2canvas ? resolve(window.html2canvas) : reject(new Error('html2canvas')));
        s.onerror = () => { h2cLoading = null; reject(new Error('Nie udało się pobrać html2canvas (brak internetu?)')); };
        document.head.appendChild(s);
      });
    }
    return h2cLoading;
  }

  async function grabWindow() {
    const h2c = await loadH2C();
    const canvas = await h2c(document.body, { backgroundColor: '#111113', logging: false, useCORS: true,
      width: window.innerWidth, height: window.innerHeight, windowWidth: window.innerWidth, windowHeight: window.innerHeight });
    let c = canvas;
    if (c.width > MAX_W) {
      const s = MAX_W / c.width;
      const small = document.createElement('canvas');
      small.width = MAX_W; small.height = Math.round(c.height * s);
      small.getContext('2d').drawImage(c, 0, 0, small.width, small.height);
      c = small;
    }
    return c.toDataURL('image/png');
  }

  async function postJSON(url, body) {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    let data = null;
    try { data = await r.json(); } catch (e) { data = null; }
    if (!r.ok) throw new Error((data && data.detail) || `HTTP ${r.status}`);
    return data;
  }

  function open(context = {}) {
    if (document.getElementById('mdReport')) return;
    const bg = document.createElement('div');
    bg.id = 'mdReport';
    bg.className = 'md-dialog';
    bg.style.cssText = 'position:fixed;inset:0;z-index:26000;background:rgba(10,10,12,.7);display:flex;align-items:center;justify-content:center;';
    const errBox = context.error_title || context.error_text
      ? `<div class="rp-err" style="margin-bottom:10px;padding:8px 10px;border-left:3px solid #ef4444;background:rgba(239,68,68,.08);font-size:12px;">
           <b>Dołączony błąd:</b> ${esc(context.error_title || '')}<br><span style="color:#a8a8ae;">${esc((context.error_text || '').slice(0, 400))}</span></div>` : '';
    bg.innerHTML = `<style>#mdReport [hidden] { display: none !important; } #mdReport textarea, #mdReport input[type=email] { background: #101012; color: #dcdcdf; border: 1px solid #3c3c42; border-radius: 4px; padding: 6px 8px; font: inherit; resize: vertical; }</style><div role="dialog" aria-label="Zgłoś problem" style="width:min(720px,94vw);max-height:90vh;display:flex;flex-direction:column;background:#19191b;color:#dcdcdf;border:1px solid #3c3c42;border-top:4px solid #e8a33d;border-radius:10px;box-shadow:0 20px 50px rgba(0,0,0,.6);">
        <div style="padding:14px 18px 4px;font-size:16px;font-weight:700;">🐞 Zgłoś problem</div>
        <div style="padding:4px 18px 0;font-size:12px;color:#9a9aa0;">Zgłoszenie trafia do autora MatteDesk (prywatne repozytorium na GitHubie). Ścieżki, nazwy ujęć i obiektów, nazwa użytkownika i komputera są zastępowane znacznikami – przed wysłaniem zobaczysz wszystko, co wyjdzie z komputera.</div>
        <div style="padding:10px 18px;overflow:auto;font-size:13px;display:flex;flex-direction:column;gap:8px;">
          ${errBox}
          <label>Co się stało? <span style="color:#ef4444;">*</span>
            <textarea id="rpWhat" rows="4" style="width:100%;box-sizing:border-box;margin-top:4px;" placeholder="Np. po kliknięciu „Przelicz finalnie” okno postępu zniknęło i nic się nie policzyło"></textarea></label>
          <label>Kroki, żeby to powtórzyć (opcjonalnie)
            <textarea id="rpSteps" rows="3" style="width:100%;box-sizing:border-box;margin-top:4px;" placeholder="1. Otwórz projekt… 2. Wybierz obiekt… 3. …"></textarea></label>
          <label>E-mail do odpowiedzi (opcjonalnie)
            <input id="rpEmail" type="email" style="width:100%;box-sizing:border-box;margin-top:4px;" placeholder="tylko jeśli chcesz odpowiedź"></label>
          <label style="display:flex;gap:6px;align-items:center;"><input id="rpShot" type="checkbox"> Dołącz zrzut okna MatteDesk (bez tego okienka)</label>
          <div id="rpShotInfo" style="font-size:12px;color:#9a9aa0;"></div>
          <div id="rpPreview" hidden style="display:flex;flex-direction:column;gap:6px;">
            <b>Tak wygląda całe zgłoszenie:</b>
            <pre id="rpText" style="max-height:260px;overflow:auto;background:#101012;border:1px solid #2a2a2e;padding:8px;font-size:11px;white-space:pre-wrap;word-break:break-word;margin:0;"></pre>
            <img id="rpImg" alt="Zrzut okna" hidden style="max-width:100%;border:1px solid #2a2a2e;">
            <label style="display:flex;gap:6px;align-items:flex-start;"><input id="rpConsent" type="checkbox" style="margin-top:3px;">
              <span>Zgadzam się na wysłanie powyższych danych autorowi MatteDesk w celu naprawienia problemu.</span></label>
          </div>
        </div>
        <div style="padding:12px 18px;border-top:1px solid #2a2a2e;display:flex;gap:8px;justify-content:flex-end;align-items:center;flex-wrap:wrap;">
          <div id="rpMsg" aria-live="polite" style="font-size:12px;flex:1 1 240px;"></div>
          <button type="button" class="btn btn-secondary" id="rpCopy" hidden title="Kopiuje całe zgłoszenie (tekst) do schowka – np. do wklejenia na GitHubie">Kopiuj zgłoszenie</button>
          <button type="button" class="btn btn-secondary" id="rpCancel">Zamknij</button>
          <button type="button" class="btn btn-secondary" id="rpShow">Pokaż, co zostanie wysłane</button>
          <button type="button" class="btn btn-primary" id="rpSend" disabled style="background:#e8a33d;border-color:#e8a33d;">Wyślij</button>
        </div></div>`;
    document.body.appendChild(bg);
    const $ = (id) => bg.querySelector('#' + id);
    const msg = (t, tone = '') => { $('rpMsg').innerHTML = t ? `<span style="color:${{ err: '#ef4444', ok: '#10b981' }[tone] || '#9a9aa0'};">${t}</span>` : ''; };
    let shot = null, preview = null, busy = false;

    const invalidate = () => { preview = null; $('rpPreview').hidden = true; $('rpConsent').checked = false; $('rpCopy').hidden = true; sync(); };
    const sync = () => { $('rpSend').disabled = busy || !preview || !preview.configured || !$('rpConsent').checked; };
    ['rpWhat', 'rpSteps', 'rpEmail'].forEach((id) => $(id).addEventListener('input', invalidate));
    $('rpConsent').addEventListener('change', sync);

    $('rpShot').addEventListener('change', async () => {
      invalidate();
      shot = null; $('rpShotInfo').textContent = '';
      if (!$('rpShot').checked) return;
      bg.style.visibility = 'hidden';   // the screenshot shows the app, not this window
      try {
        shot = await grabWindow();
        $('rpShotInfo').textContent = `Zrzut gotowy (${Math.round(shot.length * 0.75 / 1024)} KB) – zobaczysz go w podglądzie.`;
      } catch (e) {
        $('rpShot').checked = false;
        $('rpShotInfo').textContent = `Nie udało się zrobić zrzutu: ${e.message}. Zgłoszenie pójdzie bez niego.`;
      } finally { bg.style.visibility = ''; }
    });

    $('rpShow').addEventListener('click', async () => {
      if (busy) return;
      busy = true; sync(); msg('Zbieram diagnostykę (wersje, GPU, Doktor, ostatnie zadania, log)…');
      try {
        preview = await postJSON('/api/report/preview', { what: $('rpWhat').value, steps: $('rpSteps').value,
          email: $('rpEmail').value, screenshot: shot, context });
        $('rpText').textContent = preview.text;
        $('rpImg').hidden = !shot; if (shot) $('rpImg').src = shot;
        $('rpPreview').hidden = false; $('rpCopy').hidden = false;
        msg(preview.configured ? 'Sprawdź treść, zaznacz zgodę i kliknij „Wyślij”.'
          : 'Wysyłanie nie jest jeszcze skonfigurowane w tej wersji – skopiuj zgłoszenie i wklej je na GitHubie (link w „O programie”).', preview.configured ? '' : 'err');
      } catch (e) { preview = null; msg(esc(e.message), 'err'); }
      busy = false; sync();
    });

    $('rpCopy').addEventListener('click', async () => {
      if (!preview) return;
      try { await navigator.clipboard.writeText(preview.text); msg('Skopiowano do schowka.', 'ok'); }
      catch (e) { msg('Schowek niedostępny – zaznacz tekst podglądu i skopiuj ręcznie.', 'err'); }
    });

    $('rpSend').addEventListener('click', async () => {
      if (!preview || !$('rpConsent').checked || busy) return;
      busy = true; sync(); msg('Wysyłam…');
      try {
        const r = await postJSON('/api/report/send', { id: preview.id, consent: true });
        // sent: this window closes and a plain confirmation with OK takes its place (user request 2026-10-23)
        close();
        const text = `Zgłoszenie nr ${esc(r.number ?? '?')} zostało wysłane do autora MatteDesk. Dziękuję!`;
        if (window.viewer && window.viewer.uiDialog) window.viewer.uiDialog({ title: '✅ Zgłoszenie wysłane', tone: 'ok', html: text });
        else alert(text.replace(/<[^>]+>/g, ''));
        return;
      } catch (e) { msg(esc(e.message), 'err'); }
      busy = false; sync();
    });

    const close = () => { document.removeEventListener('keydown', onKey, true); bg.remove(); };
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } };
    document.addEventListener('keydown', onKey, true);
    $('rpCancel').addEventListener('click', close);
    $('rpWhat').focus();
  }

  window.mdReport = { open };
  const wire = () => {
    const b = document.getElementById('reportBtn');
    if (b) b.addEventListener('click', () => open({ source: 'menu' }));
    const a = document.getElementById('aboutReportBtn');
    if (a) a.addEventListener('click', () => { const m = document.getElementById('aboutModal'); if (m) m.style.display = 'none'; open({ source: 'about' }); });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire); else wire();
})();
