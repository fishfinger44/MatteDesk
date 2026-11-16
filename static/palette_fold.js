// Collapse chevron of every floating palette – ONE implementation (user decision 2026-11-11: "the same script for every
// floating palette"). objects.js makeFloating() attaches it to each palette it makes floating (MD.makeFloating is what
// every module uses); a palette that re-renders its head (Objects, #otherLayers) calls attach() again after rendering.
// Folded = only the head stays (class md-folded, CSS in index.css); state per palette in localStorage md_fold_<key>.
(function () {
  const lsKey = (k) => `md_fold_${k}`;
  // state saved under older keys before the shared script (Cleanup C1 palettes, 2026-11-11 morning)
  const LEGACY = { cuPal: 'md_cu_fold_inpaint', cuAnchorPal: 'md_cu_fold_anchors' };
  function read(key, initial) {
    try {
      let v = localStorage.getItem(lsKey(key));
      if (v === null && LEGACY[key]) v = localStorage.getItem(LEGACY[key]);
      return v === null ? !!initial : v === '1';
    } catch (e) { return !!initial; }   // storage blocked
  }
  function write(key, f) { try { localStorage.setItem(lsKey(key), f ? '1' : '0'); } catch (e) { /* storage blocked */ } }
  // MatteDesk look: thin chevron, muted grey, amber on hover (index.css .md-fold)
  const CHEVRON = '<svg viewBox="0 0 10 10" width="10" height="10" aria-hidden="true"><path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const isFolded = (box) => !!(box && box.classList.contains('md-folded'));
  function show(box, b, f) {
    box.classList.toggle('md-folded', f);   // the chevron turns by CSS (.md-folded .md-fold svg)
    b.title = f ? 'Rozwiń paletę' : 'Zwiń paletę';
  }
  // box = the palette, headSel = its drag head (direct child), key = storage key (default: the palette's id);
  // opts.initial = folded when nothing is saved yet. Idempotent: a second call only re-syncs the button.
  function attach(box, headSel, key, opts) {
    const o = opts || {};
    const k = key || (box && box.id);
    const head = box && box.querySelector(headSel);
    if (!head || !k) return null;
    head.classList.add('md-fold-head');
    let b = head.querySelector(':scope > .md-fold');
    if (!b) {
      b = document.createElement('button');
      b.type = 'button'; b.className = 'md-fold';
      b.innerHTML = CHEVRON;
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        const f = !isFolded(box);
        show(box, b, f); write(k, f);
        if (o.onChange) o.onChange(f);
      });
      head.appendChild(b);
    }
    show(box, b, read(k, o.initial));
    return b;
  }
  window.MD_FOLD = { attach, isFolded };
})();
