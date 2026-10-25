// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// "O programie" window: author, licence, links. Opened from the version tag in the header.
(function () {
  // Fill in before the public release; an empty link is hidden.
  const LINKS = {
    repo: 'https://github.com/fishfinger44/MatteDesk',
    support: 'https://ko-fi.com/fishfinger',
  };

  const modal = document.getElementById('aboutModal');
  const tag = document.getElementById('appVersion');
  if (!modal || !tag) return;

  const show = (on) => { modal.style.display = on ? 'flex' : 'none'; };
  // the whole badge (logo + name + version) opens the window, not just the version tag
  const badge = tag.closest('.logo-badge') || tag;
  badge.classList.add('about-open');
  badge.setAttribute('role', 'button');
  badge.tabIndex = 0;
  const openAbout = () => {
    document.getElementById('aboutVersion').textContent = tag.textContent;
    show(true);
  };
  badge.addEventListener('click', openAbout);
  badge.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openAbout(); } });
  document.getElementById('closeAboutModalBtn').addEventListener('click', () => show(false));
  document.getElementById('confirmAboutBtn').addEventListener('click', () => show(false));
  modal.addEventListener('click', (e) => { if (e.target === modal) show(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modal.style.display !== 'none') show(false); });

  // "Sprawdź aktualizacje": forced check (ignores the once-a-day cache and a skipped version); a newer version opens
  // the update window of app.js (short notes, changelog link, Update / Later)
  const upBtn = document.getElementById('aboutUpdateBtn');
  const upStatus = document.getElementById('aboutUpdateStatus');
  if (upBtn) upBtn.addEventListener('click', async () => {
    if (!window.viewer || !window.viewer.checkUpdate) return;
    upBtn.disabled = true;
    upStatus.textContent = 'Sprawdzam…';
    let u = null;
    try { u = await (await fetch('/api/update?force=1')).json(); } catch (e) { u = null; }
    upBtn.disabled = false;
    if (u && u.available) { show(false); window.viewer.checkUpdate(true, u); }
    const v = window.viewer;
    if (!u || u.error) {
      upStatus.textContent = 'Nie udało się sprawdzić (brak internetu?)';
      v.uiDialog({ title: 'Nie udało się sprawdzić aktualizacji', tone: 'warn',
        html: 'Brak połączenia z GitHubem (internet, zapora albo limit zapytań). Spróbuj później.' });
    } else if (!u.available) {
      upStatus.textContent = `Masz najnowszą wersję (${u.current}).`;
      v.uiDialog({ title: 'Masz najnowszą wersję', tone: 'ok',
        html: `MatteDesk ${v.esc(u.current)} – nowszej wersji nie ma.` });
    } else upStatus.textContent = `Dostępna wersja ${u.version}.`;
  });

  for (const [key, url] of Object.entries(LINKS)) {
    const a = document.getElementById(`aboutLink_${key}`);
    if (!a) continue;
    if (url) a.href = url; else a.closest('li').style.display = 'none';
  }
})();
