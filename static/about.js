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
  tag.style.cursor = 'pointer';
  tag.addEventListener('click', () => {
    document.getElementById('aboutVersion').textContent = tag.textContent;
    show(true);
  });
  document.getElementById('closeAboutModalBtn').addEventListener('click', () => show(false));
  document.getElementById('confirmAboutBtn').addEventListener('click', () => show(false));
  modal.addEventListener('click', (e) => { if (e.target === modal) show(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modal.style.display !== 'none') show(false); });

  for (const [key, url] of Object.entries(LINKS)) {
    const a = document.getElementById(`aboutLink_${key}`);
    if (!a) continue;
    if (url) a.href = url; else a.closest('li').style.display = 'none';
  }
})();
