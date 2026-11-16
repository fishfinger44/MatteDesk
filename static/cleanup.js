// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/cleanup.js – the „Czyszczenie” mode (MVP, user decision 2026-11-06; engine/cleanup.py, api/cleanup.py): a switch
// at the top „✂ Wycinanie / 🧹 Czyszczenie”. Cleaning hides the stage bar and the stage panels and shows its own panel:
// objects to remove (each with the version of its mask), grow / soft of the mask, holdout of the other objects, the fill
// (clear plate first, LaMa the rest), a mask preview of the current frame and „▶ Policz czyszczenie”. The result is
// a version in <work>/inpaint/v### and the plate „cleaned” – the viewer, the timeline and the A/B / wipe / difference of
// the header work on it as on any plate. Own file so objects.js stays untouched.
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const KEY = 'md_mode';
  const C = { mode: 'cut', info: null, saveT: null, liveOn: false, live: null };
  try { C.liveOn = localStorage.getItem('md_cu_live') === '1'; } catch (e) { /* storage blocked */ }
  let mode = 'cut';
  try { mode = localStorage.getItem(KEY) === 'clean' ? 'clean' : 'cut'; } catch (e) { /* storage blocked */ }

  function api(m, u, b) { return window.UI.api(m, u, b); }

  function switchHtml() {
    return `<div class="mode-switch" id="modeSwitch" role="group" aria-label="Tryb">
      <button type="button" class="btn btn-xs" data-mode="cut" title="Wycinanie: etapy obiektu → alpha (eksport tylko alfy)">✂ Wycinanie</button>
      <button type="button" class="btn btn-xs" data-mode="clean" title="Czyszczenie: usuwanie obiektów z plate'u (rig, uchwyt, niechciany element) – łata z clear plate'u / LaMa, osobne wersje w <work>/inpaint">🧹 Czyszczenie</button>
    </div>`;
  }

  // the RIGHT panel (user 2026-11-11): what the step does, the cleaned thing with its mask, the versions – the tools
  // are on the LEFT floating palette „🧹 Inpaint” (paletteHtml), like the stage palettes of the cut-out
  function panelHtml() {
    return `<div class="stage-pane" id="cleanupPanel" style="display:none">
      <div class="stage-title">🧹 Inpaint – usuń / zastąp</div>
      <div class="cu-intro">
        <div class="pp-hint cu-intro-lead"><b>Co robi:</b> <span>usuwa obiekt z ujęcia albo zastępuje go czymś z opisu.</span> <b>Najlepsze do:</b> <span>rigów, uchwytów, wymiany elementu, przemiany (morfing).</span></div>
        <ol class="cu-steps">
          <li>Wybierz obiekt poniżej (maska z Wycinania).</li>
          <li>W palecie <b>🧹 Inpaint</b> (po lewej): model, <b>Usuń / Zastąp</b>, opis.</li>
          <li><b>▶ Policz Inpaint</b> → plate „cleaned” (A/B) + łata RGBA do Nuke'a (Merge over).</li>
          <li>Paleta <b>🧩 Złożenie</b> (po prawej): pas łaty, ochrona innych obiektów, ziarno – bez liczenia modelu.</li>
        </ol>
        <div class="cu-guide" id="cuGuide"></div>
      </div>
      <div class="zp-sub">Obiekt</div>
      <div id="cuObjects" class="cu-objects"></div>
      <div class="pp-hint warn" id="cuWarn" style="display:none"></div>
      <div id="cuAnchors"></div>
      <div class="zp-sub">Wersje</div>
      <div id="cuVersions" class="muted pp-hint"></div>
    </div>`;
  }

  // the LEFT floating palette „🧹 Inpaint” (user 2026-11-11): fill, model, description, seed, the generation margin,
  // the mask preview and the run – shown only in Cleanup step C1 (index.css + c1Shown); the band, the holdout and the
  // grain moved to the RIGHT palette „🧩 Złożenie” (compPaletteHtml, 2026-11-15)
  function paletteHtml() {
    return `<div class="pts-palette cu-pal" id="cuPal" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> 🧹 Inpaint</div>
      <div class="cu-sec-h">Model</div>
      <label class="cu-input" title="Wejście czyszczenia (2026-11-15): na czym liczy Inpaint – oryginalny plate Beauty (domyślnie) albo wynik wcześniejszej wersji Inpaint (np. usuń kolczyki z wersji z podmienioną twarzą). Piksele poza łatą też idą z wejścia; klatki, których ta wersja nie ma, z Beauty. Maski obiektów są z Wycinania na oryginale">Wejście<select id="cuInput" class="form-input"></select></label>
      <div class="cu-row2">
        <label title="Skąd łata: auto = clear plate tam, gdzie jest (i pasuje do beauty), reszta model; tylko clear plate; tylko model">Źródło<select id="cuFill" class="form-input">
          <option value="auto">auto (clear plate → model)</option><option value="cleanplate">tylko clear plate</option><option value="lama">tylko model</option></select></label>
        <label title="Model, który wypełnia to, czego nie pokrywa clear plate – porównanie modeli w panelu po prawej">Model<select id="cuModel" class="form-input"></select></label>
      </div>
      <div id="cuModelInfo"></div>
      <div class="cu-sec-h" id="cuDoHead" style="display:none">Co zrobić</div>
      <div class="cu-pmode" id="cuPromptMode" style="display:none" role="group">
        <button type="button" class="btn btn-sm" data-pm="remove" title="Usuń z ujęcia: obiekt znika, model dorysowuje tło. Opis = jak ma wyglądać tło w tym miejscu (puste = opis ogólny)">🧽 Usuń z ujęcia</button><button type="button" class="btn btn-sm" data-pm="replace" title="Zastąp obiekt: w masce pojawia się nowa rzecz z opisu. Nowa rzecz mieści się w masce – większy kształt = większy „Zapas generowania” (model rysuje w nim), a pas łaty decyduje, ile z tego wchodzi w obraz. Opisz wygląd, nie przemianę („Transform into…” MiniMax-H3 zrobi jako zdarzenie w połowie klipu)">✨ Zastąp obiekt</button></div>
      <div class="cu-prompt-box" id="cuPromptBox" style="display:none">
        <div class="cu-prompt-h"><span title="Opis (prompt) – po angielsku; „✨ Popraw opis” przetłumaczy z polskiego">✎ Opis (po angielsku)</span><button type="button" class="btn btn-ghost btn-xs" id="cuAssist" title="Popraw opis lokalnym asystentem (Qwen3-VL-4B): przetłumaczy na angielski i przepisze według zasad modeli, patrząc na klatkę z zaznaczonym obiektem. Możesz pisać po polsku. Nic nie zapisuje się bez „Wstaw”">✨ Popraw opis</button></div>
        <div class="pp-hint muted" id="cuPromptHint" style="display:none"></div>
        <textarea id="cuPrompt" class="form-input" rows="5" style="display:none" title="Opis – po angielsku; puste = opis ogólny" placeholder="Co ma być w miejscu obiektu (po angielsku), np. „plain white wall, soft daylight”"></textarea>
        <div class="cu-lint" id="cuPromptLint"></div>
        <details class="cu-final" id="cuFinal"><summary title="Dokładne teksty, które program wyśle do każdego modelu – Twój opis w opakowaniu programu">Co dostaje który model</summary><div id="cuFinalBody"></div></details>
      </div>
      <label id="cuH3Row" class="cu-h3row" style="display:none" title="Prompt klipu wideo przy przejściu (kotwica własna albo oryginał) – dostaje go wybrany silnik (MiniMax-H3 albo Wan 2.2 Fun VACE); opisz, co się dzieje w czasie, np. „her natural face slowly turns into white porcelain with gold filigree”"><span>Prompt klipu <span id="cuH3Eng">MiniMax-H3</span> (przejście)</span><textarea id="cuH3Prompt" class="form-input" rows="3" placeholder="np. „her natural skin slowly turns into polished white porcelain with gold filigree”"></textarea></label>
      <label class="zone-slider" id="cuSeedRow" style="display:none" title="Seed silnika wideo (MiniMax-H3 / Wan 2.2 Fun VACE) i kotwic Qwen-Image-2.1 – ten sam seed + te same ustawienia = ten sam wynik; 🎲 = nowy losowy">Seed<input type="number" id="cuSeed" class="form-input" min="0" step="1"><button type="button" class="btn btn-ghost btn-xs" id="cuSeedRnd" title="Nowy losowy seed (kotwice policzą się od nowa)">🎲</button></label>
      <label class="zone-slider" id="cuStrengthRow" style="display:none" title="MiniMax-H3, „Zastąp”: od czego model zaczyna w masce. 1,0 = od dziury (czysty szum – model nie widzi, co było pod maską, mimikę zgaduje). Mniej = startuje z zaszumionego plate'u: trzyma kształt, pozę i mimikę twarzy, zmienia wygląd (np. faktura skóry). Za mało = zostaje stary wygląd. „Usuń z ujęcia” zawsze liczy od dziury – model nie może widzieć obiektu, który usuwa">Siła zmiany<input type="range" id="cuStrength" min="0.3" max="1" step="0.05"><span id="cuStrengthVal" class="muted"></span></label>
      <div class="cu-vace-row" id="cuVaceRow" style="display:none" title="Wan 2.2 Fun VACE: rozdzielczość klipu i liczba kroków">
        <select id="cuVaceRes" class="form-input" title="Rozdzielczość liczenia Wan 2.2 Fun VACE: 720p = najwięcej szczegółu; 480p = ok. 2–2,5× szybciej i mniej pamięci"><option value="720">720p</option><option value="480">480p</option></select>
        <select id="cuVaceSpeed" class="form-input" title="Szybko = LoRA Lightning: 4 kroki przy CFG 1 (ok. 5–10× szybciej, trochę mniej szczegółu); dokładnie = 20 kroków przy CFG 3,5"><option value="fast">szybko – 4 kroki</option><option value="quality">dokładnie – 20 kroków</option></select>
      </div>
      <div class="pp-hint warn" id="cuVaceNote" style="display:none"></div>
      <label class="edge-px cu-gen" title="Zapas generowania: o ile px wokół usuwanego obiektu model wideo ma narysować nowy obraz (maska modelu = obiekt poszerzony o tyle, twardo, bez punktów pasa i wtopienia, także na chronione obiekty – złożenie przywraca je z ich alfą; silnik dokłada swój rozrost). Liczy się przed „▶ Policz Inpaint” – po liczeniu zmienia go tylko nowe liczenie. Większy zapas = więcej materiału dla pasa łaty">Zapas generowania <input type="range" id="cuGenR" min="0" max="200" step="1" title="Zapas generowania (px) – zapis po puszczeniu"><input type="number" id="cuGen" class="form-input edge-num" min="0" max="300" step="1" title="Zapas generowania (px) – wpisz i Enter"></label>
      <div class="muted pp-hint cu-sources" id="cuSources"></div>
      <div class="pp-btns cu-run-row">
        <button type="button" class="btn btn-secondary btn-sm" id="cuPreview" title="Maska usuwania bieżącej klatki (czerwone = łata) – ta sama, której użyje „Policz”">👁 Maska klatki</button>
        <button type="button" class="btn btn-primary btn-sm" id="cuRun" title="Czyszczenie na zakresie pracy (oś czasu) → nowa wersja w <work>/inpaint i plate „cleaned”">▶ Policz Inpaint</button>
      </div>
      <canvas id="cuMaskCv" style="display:none;width:100%;max-height:300px;object-fit:contain;margin-top:6px;border-radius:4px"></canvas>
    </div>`;
  }


  // the RIGHT floating palette „🧩 Złożenie” (user decision 2026-11-15): what composes a finished version – the patch
  // band („Pas łaty”, live preview, „Zastosuj (nowa wersja)”), the protection of the kept objects (each with its mask,
  // overlay, „chroń”, „→ Wycinanie”) and the grain; moved here from „🧹 Inpaint” (the model run stays there)
  function compPaletteHtml() {
    return `<div class="pts-palette cu-pal cu-comp-pal" id="cuCompPal" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> 🧩 Złożenie</div>
      <div class="pp-sec cu-band">
        <div class="cu-sec-h" title="Pas łaty – co z wygenerowanego materiału wchodzi w obraz. Ten sam system pasa co w etapach wycinania, tylko strona zewnętrzna: łata musi przykryć cały obiekt, więc nie ma wcięcia. Pas jest własny dla czyszczonego obiektu (niezależny od pasa etapów wycinania). Sięga najwyżej do granicy tego, co narysował model (zapas generowania + rozrost silnika)">Pas łaty</div>
        <div class="cu-band-row">
          <label class="chk" title="Linie pasa na podglądzie (krawędź pasa i koniec wtopienia) i punkty pasa"><input type="checkbox" id="cuBandPv" title="Podgląd pasa wł. / wył."> 👁 podgląd pasa</label>
          <button type="button" class="btn btn-ghost btn-xs" id="cuBandAdd" title="Dodaj punkt pasa: klik przy obrysie obiektu = nowy punkt (klucz tej klatki); przeciągnij żółty wąs = jak daleko łata sięga na zewnątrz w tym miejscu. Między kluczami punkt jedzie za obiektem. Esc = koniec">⊕ Punkt</button>
          <button type="button" class="btn btn-ghost btn-xs" id="cuBandRemove" title="Usuń punkt pasa: klik na punkt (albo PPM na punkcie). Esc = koniec">⊖ Punkt</button>
          <button type="button" class="btn btn-ghost btn-xs" id="cuBandClear" style="display:none" title="Usuń klucze pasa w tej klatce">🔑✕</button>
        </div>
        <label class="edge-px" title="Pas na zewnątrz = Erode / Dilate: plus – o ile px poza obrys obiektu łata jest pełna (rozmycie ruchu, miękka krawędź, poświata); minus – łata zwężona do środka obiektu, jego własna krawędź zostaje z plate'u (Zastąp: np. włosy na brzegu twarzy). Przerywana linia = krawędź pasa. Punkty pasa poszerzają / zwężają go wzdłuż obrysu (tylko przy plusie), suwak skaluje cały pas">Pas na zewnątrz <input type="range" id="cuGrowR" min="-100" max="120" step="1" title="Pas na zewnątrz (px) – podgląd na żywo, zapis po puszczeniu"><input type="number" id="cuGrow" class="form-input edge-num" min="-100" max="200" step="1" title="Pas na zewnątrz (px) – wpisz i Enter"></label>
        <label class="edge-px" title="Wtopienie zewnętrzne: na ilu px ZA pasem łata miękko przechodzi w plate (cieńsza przerywana linia = koniec łaty). 0 = twarda krawędź">Wtopienie zewn. <input type="range" id="cuSoftR" min="0" max="80" step="1" title="Wtopienie zewnętrzne (px) – podgląd na żywo, zapis po puszczeniu"><input type="number" id="cuSoft" class="form-input edge-num" min="0" max="100" step="1" title="Wtopienie zewnętrzne (px) – wpisz i Enter"></label>
        <div class="muted pp-hint" id="cuBandInfo"></div>
        <div class="pp-hint warn" id="cuBandLimit" style="display:none"></div>
        <div class="pp-hint warn" id="cuGrowWarn" style="display:none"></div>
        <div class="cu-recomp" id="cuRecompRow" style="display:none">
          <div class="pp-hint" id="cuRecompInfo"></div>
          <div class="cu-band-row">
            <label class="chk" title="Podgląd złożenia bieżącej klatki z obecnym pasem (suwaki i punkty pasa, jeszcze niezapisane też) – liczony na serwerze z zapisanego obrazu modelu, bez liczenia modelu"><input type="checkbox" id="cuLive" title="Podgląd złożenia na żywo wł. / wył."> 👁 na żywo</label>
            <label class="chk" title="Sam obraz modelu tej wersji w bieżącej klatce – to, co model narysował, zanim pas łaty i ochrona go złożyły z plate'em (całe jego płótno, rozciągnięte na plate jak przy złożeniu). Pozwala odróżnić, czy coś jest w obrazie modelu, czy powstaje przy złożeniu"><input type="checkbox" id="cuModelPv" title="Obraz modelu wł. / wył."> 🎞 obraz modelu</label>
            <button type="button" class="btn btn-primary btn-xs" id="cuRecompose" title="Złóż pokazaną wersję ponownie z obecnym pasem łaty – nowa wersja (pokazana zostaje), bez liczenia modelu">Zastosuj (nowa wersja)</button>
          </div>
        </div>
      </div>
      <div class="cu-sec-h">Ochrona</div>
      <label class="chk" title="Pozostałe obiekty projektu zostają z oryginału: łata wchodzi pod nie tylko tam, gdzie są przezroczyste (ich alfa – miękka krawędź włosów zostaje miękka). Linie pasa pokazują pas samego czyszczonego obiektu, przed tym wycięciem"><input type="checkbox" id="cuHoldout"> chroń pozostałe obiekty</label>
      <label class="chk" title="Gdy obiekt chroniony wygląda na zgubiony (jego maska nagle wielokrotnie większa albo leży na krawędzi usuwanego obiektu), w tych klatkach łata go nie omija. Domyślnie wyłączone: wtedy tylko ostrzeżenie"><input type="checkbox" id="cuSkipLost"> pomiń ochronę zgubionych</label>
      <div id="cuKept" class="cu-kept"></div>
      <div class="cu-sec-h" title="Ziarno łaty modelu wideo – dokładane przy składaniu">Ziarno</div>
      <div class="cu-grain" title="Ziarno łaty: obraz modelu wideo (MiniMax-H3, Wan 2.2 Fun VACE, LTX-2.5) jest gładszy od plate'u – wtedy dokładamy łacie ziarno o widmie plate'u, nowe w każdej klatce (ten sam seed = ten sam wynik). Mierzone obok łaty, tam gdzie model i plate pokazują to samo tło">
        <label title="auto = ziarno tylko gdy plate jest wyraźnie bardziej zaszumiony od łaty (odszumiony plate = bez zmian); wł. = zawsze dokładaj brakujące ziarno; wył. = nigdy">Ziarno łaty<select id="cuGrain" class="form-input">
          <option value="auto">auto</option><option value="on">wł.</option><option value="off">wył.</option></select></label>
        <label class="zone-slider" title="Mnoży brakujące ziarno: 1 = jak plate, mniej = delikatniej, więcej = mocniej">Siła ziarna <span id="cuGrainStrengthV" class="muted"></span><input type="range" id="cuGrainStrength" min="0" max="2" step="0.05"></label>
        <div class="muted pp-hint" id="cuGrainInfo"></div>
      </div>
    </div>`;
  }

  const posStore = (key) => ({
    get: () => { try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem(key, JSON.stringify(p)); else localStorage.removeItem(key); } catch (e) { /* storage blocked */ } },
  });
  const palPos = posStore('md_cu_palette');
  const compPos = posStore('md_cu_comp_palette');
  // Cleanup step C1 is on screen (the palettes and the band editor follow it)
  const c1Shown = () => mode === 'clean' && (!window.MD_CLEANUP3D || window.MD_CLEANUP3D.step === 'remove');
  function palVis() {
    const pal = $('cuPal'); if (!pal) return;
    const show = c1Shown();
    for (const [box, pos] of [[pal, palPos], [$('cuCompPal'), compPos]]) {
      if (!box) continue;
      box.style.display = show ? '' : 'none';
      if (show && window.MD && window.MD.placeFloating) window.MD.placeFloating(box, $('canvasWrapper'), pos.get());
    }
    if (!show && C.bandTool) C.bandTool = null;
    bandUi();
    if (window.MD_CU_ANCHORS) window.MD_CU_ANCHORS.redraw();
  }

  // ---- the C1 band (engine/inpaint_band.py; user 2026-11-11): „Pas na zewnątrz” = grow_px, „Wtopienie zewn.” = soft_px,
  // band points = the one editor static/bandpoints.js (owner „c1”, /api/cleanup/band/…), lines drawn by MD_BANDPTS
  const bandPv = () => { try { return localStorage.getItem('md_cu_bandpv') !== '0'; } catch (e) { return true; } };
  const bandOwner = () => (c1Shown() && C.info && target() && (objOf(target()) || {}).mask && objOf(target()).mask.run ? 'c1' : null);
  const bandVals = () => ({ grow_px: +$('cuGrow').value || 0, soft_px: +$('cuSoft').value || 0 });
  function setBandTool(t) {
    C.bandTool = C.bandTool === t ? null : t;
    if (C.bandTool && !bandPv()) { $('cuBandPv').checked = true; try { localStorage.setItem('md_cu_bandpv', '1'); } catch (e) { /* storage blocked */ } }
    bandUi();
    const B = window.MD_BANDPTS; if (B && B.ready) B.onTool();
  }
  function bandUi() {
    if (!$('cuBandAdd')) return;
    if (!bandOwner() && C.bandTool) C.bandTool = null;
    $('cuBandPv').checked = bandPv();
    $('cuBandAdd').classList.toggle('active', C.bandTool === 'band_add');
    $('cuBandRemove').classList.toggle('active', C.bandTool === 'band_remove');
    $('cuBandAdd').disabled = $('cuBandRemove').disabled = !bandOwner();
    const B = window.MD_BANDPTS;
    if (B && B.ready) { B.info(); B.renderLane(); }
    if (!bandOwner()) { $('cuBandInfo').textContent = target() ? 'Pas: obiekt bez maski' : 'Pas: najpierw wybierz obiekt w palecie Obiekty'; $('cuBandClear').style.display = 'none'; }
  }
  function drawBand(ctx, img) {   // overlay hook of the viewer (image space): the band lines; the points: objects.js → MD_BANDPTS.draw
    const B = window.MD_BANDPTS; const v = window.viewer;
    if (!B || !B.ready || !bandOwner() || !bandPv() || !v) return;
    B.drawLines(ctx, img, v.currentFrame, bandVals(), { live: !!C.bandDrag });
  }
  function bandSlider(rid, nid, field) {
    const r = $(rid), n = $(nid);
    r.addEventListener('input', () => {
      if (!C.bandDrag) C.bandDrag = { out0: +n.value || 0 };
      n.value = r.value;
      const B = window.MD_BANDPTS;
      if (B && B.ready && field === 'grow_px') B.setLive(C.bandDrag.out0, 0, +r.value || 0, 0);
      else if (window.viewer) window.viewer.render();
      bandLimitUi();
      if (liveOn()) liveRequest();
    });
    const commit = (val) => {
      C.bandDrag = null;
      const B = window.MD_BANDPTS; if (B && B.ready) { B.setLive(0, 0, 0, 0); B.dirty(); }
      bandKeySet(field, val);      // a key of the current frame (user 2026-11-15), not one value for the shot
      bandLimitUi();
      if (window.viewer) window.viewer.render();
    };
    r.addEventListener('change', () => { n.value = r.value; commit(+r.value || 0); });
    n.addEventListener('change', () => { r.value = n.value; commit(+n.value || 0); });
  }
  async function reloadBand() {   // a band point edit (static/bandpoints.js): the preview lines again
    C.liveBump = (C.liveBump || 0) + 1;   // … and the live composite (the points are saved, the preview reads them)
    if (window.viewer) window.viewer.render();
  }
  // ---- the band edits the composite of the version shown (user decision 2026-11-13): a version that kept its model
  // frames (model/) is composited again with the current band – live preview of the current frame (GET
  // /api/cleanup/versions/{v}/preview/{f}, drawn over the frame by an overlay hook) and „Zastosuj (nowa wersja)” = ⟲
  const shownRec = () => {
    const v = ((C.info && C.info.versions) || []).find((x) => x.shown);
    return v && v.has_model && (!target() || (v.remove || []).includes(target())) ? v : null;
  };
  // the overlays of the composite are the plate „cleaned”'s (user 2026-11-15: Beauty picked, the live patch stayed over it)
  const onCleaned = () => !!(window.viewer && window.viewer.activePlate === 'cleaned');
  const liveOn = () => !!(C.liveOn && c1Shown() && shownRec() && onCleaned());
  function liveKey(f) {
    const v = shownRec();
    return v ? `${v.version}|${f}|${+$('cuGrow').value || 0}|${+$('cuSoft').value || 0}|${C.liveBump || 0}` : null;
  }
  function recompUi() {
    const row = $('cuRecompRow'); if (!row) return;
    const v = shownRec();
    row.style.display = v ? '' : 'none';
    $('cuLive').checked = !!C.liveOn;
    if (v) $('cuRecompInfo').textContent = C.liveErr ? `Podgląd na żywo: ${C.liveErr}`
      : `Pas łaty składa teraz wersję ${v.version} (na plate „cleaned”): suwaki i punkty pasa zmieniają, ile jej obrazu modelu wchodzi w kadr. „Zastosuj” = nowa wersja bez liczenia modelu.`;
  }
  function liveRequest() { clearTimeout(C.liveT); C.liveT = setTimeout(liveFetch, 180); }
  async function liveFetch() {
    const v = shownRec(); const vw = window.viewer;
    if (!liveOn() || !v || !vw) return;
    const f = vw.currentFrame; const fr = v.frames || [];
    if (fr.length === 2 && (f < fr[0] || f > fr[1])) { if (C.live) { C.live = null; vw.render(); } return; }
    const key = liveKey(f);
    if (C.live && C.live.key === key) return;
    if (C.liveBusy) { C.liveAgain = true; return; }
    C.liveBusy = true;
    try {
      const img = await loadImg(`/api/cleanup/versions/${encodeURIComponent(v.version)}/preview/${f}?grow_px=${+$('cuGrow').value || 0}&soft_px=${+$('cuSoft').value || 0}&t=${Date.now()}`);
      C.live = { key, f, img }; C.liveErr = '';
    } catch (e) { C.live = null; C.liveErr = 'nie udało się policzyć klatki'; }
    finally {
      C.liveBusy = false; recompUi(); vw.render();
      if (C.liveAgain) { C.liveAgain = false; liveRequest(); }
    }
  }
  // „🎞 obraz modelu” (user 2026-11-15): the version's model picture of the frame over everything but the band lines
  // (GET /api/cleanup/versions/{v}/model/{f}); one image cached per version + frame
  const modelPvOn = () => !!(C.modelPv && c1Shown() && shownRec() && onCleaned());
  async function modelPvFetch(v, f) {
    const key = `${v.version}|${f}`;
    if (C.mpvBusy === key) return;
    C.mpvBusy = key;
    try { C.mpv = { key, img: await loadImg(`/api/cleanup/versions/${encodeURIComponent(v.version)}/model/${f}?t=${Date.now()}`) }; }
    catch (e) { C.mpv = { key, img: null }; }
    finally { if (C.mpvBusy === key) C.mpvBusy = null; if (window.viewer) window.viewer.render(); }
  }
  // the composite overlays respect the viewer's comparison (user 2026-11-15: wipe showed the patch on both sides):
  // wipe = only the „cleaned” side (right of the line), difference / edges / C held = not drawn; returns false to skip
  function cmpClip(ctx, img) {
    const c = (window.viewer && window.viewer.compare) || {};
    if (c.holdC || c.mode === 'diff' || c.mode === 'edges' || c.mode === 'comp') return false;
    if (c.mode === 'wipe') { const x = img.width * (+c.wipe || 0); ctx.beginPath(); ctx.rect(x, 0, img.width - x, img.height); ctx.clip(); }
    return true;
  }
  function drawModelPv(ctx, img) {
    const vw = window.viewer; if (!modelPvOn() || !vw) return;
    const v = shownRec(); const f = vw.currentFrame; const fr = v.frames || [];
    if (fr.length === 2 && (f < fr[0] || f > fr[1])) return;
    const key = `${v.version}|${f}`;
    if (!C.mpv || C.mpv.key !== key) { modelPvFetch(v, f); if (!C.mpv || !C.mpv.img) return; }
    if (C.mpv.img) { ctx.save(); if (cmpClip(ctx, img)) ctx.drawImage(C.mpv.img, 0, 0, img.width, img.height); ctx.restore(); }
  }
  function drawLive(ctx, img) {   // overlay hook (image space): the live composite over the frame, under the band lines
    const vw = window.viewer;
    if (!liveOn() || !vw) return;
    const f = vw.currentFrame;
    if (!C.live || C.live.key !== liveKey(f)) liveRequest();
    if (C.live && C.live.f === f && C.live.img) { ctx.save(); if (cmpClip(ctx, img)) ctx.drawImage(C.live.img, 0, 0, img.width, img.height); ctx.restore(); }
  }
  // the composite band vs what the model draws (user decision 2026-11-13): „Zapas generowania” + the engine's grow
  // (GET /api/cleanup engine_reach) = the limit; a band reaching further is cut there (engine/cleanup.drawn_area)
  // ---- keys of the two band sliders (user decision 2026-11-15, like the band keys of the alpha stages): a slider change
  // = a key of the current frame, linear between keys, held outside them. The keys are the BAND SYSTEM's (engine/bandsys
  // keyed_at, static/bandpoints.js keyedAt / setSliderKey / lane / 🔑✕ / PPM / Ctrl+Z); here only the C1 storage
  // (PUT / DELETE /api/cleanup/band_keys) is plugged in.
  const bandKeys = () => ((C.info && C.info.settings && C.info.settings.band_keys) || {});
  function bandAt(f) {
    const s = (C.info && C.info.settings) || {};
    const B = window.MD_BANDPTS;
    if (!B || !B.keyedAt) return [+s.grow_px || 0, +s.soft_px || 0];
    const v = B.keyedAt(bandKeys(), f, { grow_px: +s.grow_px || 0, soft_px: +s.soft_px || 0 });
    return [v.grow_px, v.soft_px];
  }
  function syncBandSliders() {
    if (C.bandDrag || !C.info || !$('cuGrow')) return;
    const v = window.viewer; const [g, sf] = bandAt(v ? v.currentFrame : 0);
    const gi = Math.round(g * 10) / 10, si = Math.round(sf * 10) / 10;
    if (document.activeElement !== $('cuGrow')) $('cuGrow').value = gi;
    if (document.activeElement !== $('cuSoft')) $('cuSoft').value = si;
    $('cuGrowR').value = gi; $('cuSoftR').value = si;
  }
  function renderBandKeysLane() { const B = window.MD_BANDPTS; if (B && B.ready) B.renderLane(); }
  function bandKeysAfter(r) {
    if (C.info && r) { C.info.settings.band_keys = r.band_keys || {}; C.info.settings.grow_px = r.grow_px; C.info.settings.soft_px = r.soft_px; }
    syncBandSliders(); bandLimitUi();
    const B = window.MD_BANDPTS; if (B && B.ready) { B.dirty(); B.renderLane(); }
    if (window.MD && window.MD.updateDelKey) window.MD.updateDelKey();
    compositeChanged();
    return r;
  }
  const BAND_KEY_STORE = {
    label: 'pas / wtopienie', names: { grow_px: 'pas', soft_px: 'wtopienie' },
    keys: () => bandKeys(),
    set: async (f, field, value) => bandKeysAfter(await api('PUT', `/api/cleanup/band_keys/${f}`, { [field]: value })),
    del: async (f) => bandKeysAfter(await api('DELETE', `/api/cleanup/band_keys/${f}`)),
    restore: async (keys) => bandKeysAfter(await api('PUT', '/api/cleanup/band_keys', { band_keys: keys })),
  };
  (function plug(tries) {
    const B = window.MD_BANDPTS;
    if (!B || !B.setSliderKeys) { if (tries < 40) setTimeout(() => plug(tries + 1), 250); return; }
    B.setSliderKeys('c1', BAND_KEY_STORE);
  })(0);
  async function bandKeySet(field, val) {
    const v = window.viewer; const B = window.MD_BANDPTS; if (!v || !C.info) return;
    const f = v.currentFrame;
    try {
      if (B && B.ready && B.stage() === 'c1') await B.setSliderKey(f, field, val);
      else await BAND_KEY_STORE.set(f, field, val);   // the band editor not active yet: the key all the same
    } catch (e) { window.UI.toast(`Klucz pasa F${f}: ${e.message}`, 6000); }
  }
  window.addEventListener('md-frame', () => syncBandSliders());
  window.addEventListener('md-plate', () => {   // the plate picked: the version list's highlight and the overlays follow
    if (C.info && $('cuVersions')) $('cuVersions').innerHTML = versionsHtml(C.info);
    recompUi(); if (window.viewer) window.viewer.render();
  });

  function bandLimitUi() {
    const d = C.info; const el = $('cuBandLimit'); if (!d || !el) return;
    const s = d.settings;
    // erode (user 2026-11-15): with Usuń the object's own rim stays from the plate; the band points rest while eroding
    const g = +$('cuGrow').value || 0; const gw = $('cuGrowWarn');
    if (gw) {
      const hasPts = !!(s.band_points || []).length;
      // a negative band: the points act locally (2026-11-15) – at a point the patch reaches its whisker from the outline
      const txt = g < 0 ? [s.prompt_mode === 'replace' ? '' : `Pas ${g} px przy „Usuń”: brzeg obiektu (${-g} px) zostanie z plate'u – widoczna obwódka usuwanego obiektu.`,
        hasPts ? 'Pas ujemny: punkty działają lokalnie – przy punkcie łata sięga długość wąsa od obrysu, dalej od punktów = suwak.' : ''].filter(Boolean).join(' ') : '';
      gw.style.display = txt ? '' : 'none'; gw.textContent = txt;
    }
    const eng = (d.engine_reach || {})[s.model] || 0;
    const lim = (+$('cuGen').value || 0) + eng;
    const reach = (+$('cuGrow').value || 0) + (+$('cuSoft').value || 0);
    const cut = reach > lim + 0.5;
    el.style.display = cut ? '' : 'none';
    el.textContent = cut ? `Pas (${Math.round(reach)} px) sięga dalej niż model rysuje (zapas ${Math.round(+$('cuGen').value || 0)} + silnik ${Math.round(eng)} px) – łata zostanie ucięta na tej granicy. Zwiększ „Zapas generowania” przed liczeniem.` : '';
  }

  // ---- „Ochrona” (user decision 2026-11-15): the kept objects of the cleaned thing (GET /api/cleanup `kept` – the same
  // list api/cleanup._alpha_loaders protects), each with its mask, an overlay of EXACTLY the alpha the holdout uses
  // (GET /api/cleanup/holdout/{obj}/{f}: R = alpha, G = where it cuts the patch), „chroń” (`holdout_off`) and a jump
  // to cutting. Real case _fotage_testy: „Hair” / „Earrings” overlapped the cheek, the cut was invisible in the UI.
  const keptMaskLabel = (m) => {
    if (!m || !m.run) return `brak maski${m && m.why ? ` – ${m.why}` : ''}`;
    if (m.source === 'final_touch') return 'Final Touch';
    if (m.source === 'core' || m.run === 'core') return 'rdzeń 1+2';
    if (m.source === 'active' || /^v\d{3}$/.test(m.source || '')) return m.run;
    return `${srcLabel(m.source)}${/^v\d{3}$/.test(m.run) ? ` (${m.run})` : ''}`;
  };
  const keptColor = (n) => { try { return (window.MD && window.MD.objColor && window.MD.objColor(n)) || '#ffcc00'; } catch (e) { return '#ffcc00'; } };
  const keptOn = (k) => { const s = (C.info && C.info.settings) || {}; return !!(s.holdout && k.mask && k.mask.run && !(s.holdout_off || []).includes(k.name)); };
  C.ho = { on: new Set(), img: {}, busy: {}, bump: 0 };
  function renderKept() {
    const box = $('cuKept'); if (!box || !C.info) return;
    const s = C.info.settings; const kept = C.info.kept || [];
    if (!s.holdout) { box.innerHTML = '<div class="muted pp-hint">Ochrona wyłączona – łata może przykryć każdy obiekt.</div>'; return; }
    if (!kept.length) { box.innerHTML = '<div class="muted pp-hint">Brak innych obiektów do ochrony.</div>'; return; }
    box.innerHTML = kept.map((k) => {
      const on = keptOn(k); const ov = C.ho.on.has(k.name); const st = C.ho.img[k.name];
      const cut = st && st.cutPx != null ? (st.cutPx ? `${on ? 'tnie' : 'ciąłby'} łatę: ${st.cutPx} px` : 'nie tnie łaty') : '';
      // the row of the Obiekty palette (user 2026-11-15): checkbox (= chroń) · eye 👁 / ◌ (= overlay) · colour · name
      const has = !!(k.mask && k.mask.run);
      return `<div class="cu-kept-row${on ? '' : ' off'}" data-kept="${esc(k.name)}">
        <input type="checkbox" data-kept-on${on ? ' checked' : ''}${has ? '' : ' disabled'} title="${esc(on ? `Chroniony: łata omija „${k.name}” (wchodzi pod niego tylko tam, gdzie jest przezroczysty). Odznacz = łata może go przykryć (sprawdzenie zgubionych też go nie widzi). Model i tak rysuje pod chronionymi obiektami – złożenie przywraca je z ich alfą` : `Niechroniony: łata może przykryć „${k.name}”. Zaznacz = łata znów go omija`)}">
        <button type="button" class="cu-kept-eye${ov ? ' on' : ''}" data-kept-eye title="Pokaż / ukryj nakładkę „${esc(k.name)}”: dokładnie ta alfa, którą ochrona wycina z łaty w bieżącej klatce (kolor obiektu); kreskowane z obwódką = miejsce, gdzie faktycznie tnie łatę"${has ? '' : ' disabled'}>${ov ? '👁' : '◌'}</button>
        <span class="cu-kept-sw" style="background:${esc(keptColor(k.name))}"></span>
        <span class="cu-kept-txt"><span class="cu-kept-name" title="${esc(k.name)}">${esc(k.name)}</span><span class="cu-kept-mask muted" title="Maska, której używa ochrona (ta sama, którą wybierasz przy obiekcie)">${esc(keptMaskLabel(k.mask))}${cut ? ` · ${esc(cut)}` : ''}</span></span>
        <button type="button" class="btn btn-ghost btn-xs" data-kept-cut title="Przejdź do Wycinania z tym obiektem – popraw jego maskę (np. Final Touch); po powrocie „Złóż ponownie” i podgląd biorą nową maskę">→ Wycinanie</button>
      </div>`;
    }).join('');
  }
  function keptKey(n, f) { return `${n}|${f}|${+$('cuGrow').value || 0}|${+$('cuSoft').value || 0}|${C.ho.bump}`; }
  // the overlay of one kept object: alpha tinted in its colour, the cut hatched with an outline (stronger when protected)
  function tintHoldout(png, col, on) {
    const cv = document.createElement('canvas'); cv.width = png.width; cv.height = png.height;
    const cx = cv.getContext('2d'); cx.drawImage(png, 0, 0);
    const id = cx.getImageData(0, 0, cv.width, cv.height); const d = id.data; const W = cv.width, H = cv.height;
    const hex = String(col).replace('#', ''); const cr = parseInt(hex.slice(0, 2), 16) || 255, cg = parseInt(hex.slice(2, 4), 16) || 204, cb = parseInt(hex.slice(4, 6), 16) || 0;
    const cutA = new Uint8Array(W * H);
    for (let i = 0, j = 0; i < d.length; i += 4, j++) cutA[j] = d[i + 1] > 5 ? 1 : 0;   // = the server's HO_CUT_MIN (0.02)
    const kCut = on ? 0.85 : 0.35;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const j = y * W + x; const i = j * 4; const a = d[i]; const c = d[i + 1];
        if (cutA[j]) {
          const edge = (x > 0 && !cutA[j - 1]) || (x < W - 1 && !cutA[j + 1]) || (y > 0 && !cutA[j - W]) || (y < H - 1 && !cutA[j + W]);
          if (edge) { d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; d[i + 3] = on ? 255 : 140; continue; }
          const stripe = ((x + y) >> 2) & 1;
          d[i] = stripe ? 255 : cr; d[i + 1] = stripe ? 255 : cg; d[i + 2] = stripe ? 255 : cb; d[i + 3] = Math.round(Math.max(c, 160) * kCut);
        } else { d[i] = cr; d[i + 1] = cg; d[i + 2] = cb; d[i + 3] = Math.round(a * 0.45); }
      }
    }
    cx.putImageData(id, 0, 0);
    return cv;
  }
  async function keptFetch(n, f) {
    if (C.ho.busy[n]) { C.ho.busy[n] = 'again'; return; }
    C.ho.busy[n] = true;
    const key = keptKey(n, f);
    try {
      const r = await fetch(`/api/cleanup/holdout/${encodeURIComponent(n)}/${f}?grow_px=${+$('cuGrow').value || 0}&soft_px=${+$('cuSoft').value || 0}&t=${Date.now()}`, { cache: 'no-store' });
      if (!r.ok) throw new Error(String(r.status));
      const on = r.headers.get('X-Holdout-On') === '1';
      const cutPx = +(r.headers.get('X-Holdout-Cut-Px') || 0);
      const url = URL.createObjectURL(await r.blob());
      try { const png = await loadImg(url); C.ho.img[n] = { key, f, cv: tintHoldout(png, keptColor(n), on), cutPx, on }; } finally { URL.revokeObjectURL(url); }
    } catch (e) { C.ho.img[n] = { key, f, cv: null, cutPx: null, err: true }; }
    finally {
      const again = C.ho.busy[n] === 'again'; C.ho.busy[n] = false;
      renderKept(); if (window.viewer) window.viewer.render();
      if (again) keptRequest();
    }
  }
  function keptRequest() { clearTimeout(C.ho.t); C.ho.t = setTimeout(() => { const v = window.viewer; if (!v) return; for (const n of C.ho.on) { const st = C.ho.img[n]; if (!st || st.key !== keptKey(n, v.currentFrame)) keptFetch(n, v.currentFrame); } }, 180); }
  function drawHoldout(ctx, img) {   // overlay hook (image space): the kept objects' holdout alphas + their cut, over the live composite
    const vw = window.viewer;
    if (!vw || !c1Shown() || !C.info || !C.ho.on.size) return;
    const f = vw.currentFrame; let stale = false;
    for (const n of C.ho.on) {
      const st = C.ho.img[n];
      if (!st || st.key !== keptKey(n, f)) stale = true;
      if (st && st.f === f && st.cv) ctx.drawImage(st.cv, 0, 0, img.width, img.height);
    }
    if (stale) keptRequest();
  }
  async function keptToggle(n, on) {   // „chroń” per object -> holdout_off (per cleaned thing); the composite changes
    if (!C.info) return;
    const off = new Set(C.info.settings.holdout_off || []);
    if (on) off.delete(n); else off.add(n);
    C.info.settings.holdout_off = [...off].sort();
    renderKept();
    save({ holdout_off: C.info.settings.holdout_off });
    try { await flush(); } catch (e) { window.UI.toast(`Ochrona: ${e.message}`, 5000); }
    compositeChanged();
  }
  function compositeChanged() {   // a composite option changed: the live preview and the overlays are asked again
    C.liveBump = (C.liveBump || 0) + 1; C.ho.bump += 1;
    if (window.viewer) window.viewer.render();
  }
  function keptToCut(n) {   // „→ Wycinanie”: cutting mode with that object selected (fix its mask, e.g. Final Touch)
    setMode('cut');
    if (window.MD && window.MD.selectObject) window.MD.selectObject(n);
  }

  // ONE thing is cleaned (user decision 2026-11-10): picked in the shared Objects palette – a whole group or one object
  // outside any group (a member means its group; api/cleanup.cleanup_target). The panel shows the picked thing: its
  // mask (a group = its members' „Do grupy z:”, an object = its own choice here), its versions, the run.
  const SRC_LABEL = { core: 'Core (maska 1+2)', matanyone: 'MatAnyone2', videomama: 'VideoMaMa', vitmatte: 'ViTMatte / BiRefNet',
    ltx: 'LTX-2.5', final_touch: 'Final Touch', active: 'aktywna wersja', group: 'złożenie grupy' };
  const srcLabel = (s) => SRC_LABEL[s] || s;
  const target = () => ((C.info && C.info.settings.remove) || [])[0] || null;
  const objOf = (n) => ((C.info && C.info.objects) || []).find((o) => o.name === n) || null;


  function targetHtml(d) {
    const n = target(); const o = n && objOf(n);
    const picked = C.picked && C.picked !== n && objOf(C.picked);
    const memberNote = picked && picked.group === n
      ? `<div class="pp-hint muted">„${esc(picked.name)}” jest w grupie „${esc(n)}” – czyści się całą grupę. Żeby czyścić sam „${esc(picked.name)}”, wyciągnij go z grupy (przeciągnij poza grupę w palecie).</div>` : '';
    if (!o) return '<div class="muted">Kliknij w palecie Obiekty obiekt albo grupę do usunięcia (jedna rzecz naraz: cała grupa albo obiekt spoza grupy).</div>';
    const m = o.mask || {};
    const bad = !m.run ? `<div class="pp-hint warn">⚠ Brak maski: ${esc(m.why || 'nic nie policzono')} – nie można czyścić.</div>`
      : (m.valid ? '' : '<div class="pp-hint warn">↻ Maska nieaktualna (wynik etapu sprzed zmian) – użyta tak, jak jest.</div>');
    if (o.kind === 'group') {
      const members = d.objects.filter((x) => x.group === o.name).map((x) => `${esc(x.name)}: ${esc(srcLabel((x.mask || {}).source))}`).join(' · ');
      return `<div class="cu-target"><b>🧹 🔗 ${esc(o.name)}</b> – cała grupa</div>
        <div class="pp-hint muted">Maska = złożenie członków, każdy z maską wybraną przy nim w palecie („Do grupy z:”): ${members || '—'}</div>${bad}${memberNote}`;
    }
    return `<div class="cu-target"><b>🧹 ${esc(o.name)}</b></div>
      ${m.run ? `<div class="pp-hint muted" title="Maskę obiektu wybierasz przy nim w palecie Obiekty – ta sama w grupie i w Czyszczeniu">Usuwana maska: ${esc(srcLabel(m.source))}${m.run && m.run !== 'core' && m.source !== m.run ? ` (${esc(m.run)})` : ''} – zmień w palecie Obiekty</div>` : ''}${bad}`;
  }

  // versions in the style of a stage's version list (user 2026-11-11): a card per version – model, range, anchors,
  // description; click = the plate „cleaned” shows it AND its model / description / anchors come back; 🗑 = delete for good
  const MODEL_NAME = { lama: 'LaMa', h3: 'MiniMax-H3', vace: 'Wan 2.2 Fun VACE', void: 'VOID', qwen21: 'Qwen-Image-2.1' };
  // the version the plate „cleaned” shows (user 2026-11-12: the HUD names it, like the mask of a stage):
  // {text: "v016 · MiniMax-H3 · F1–50", version, frames: [a, b]} or null
  function shownLabel() {
    const v = ((C.info && C.info.versions) || []).find((x) => x.shown);
    if (!v) return null;
    const fr = v.frames || [];
    return { version: v.version, frames: fr, text: `${v.version} · ${MODEL_NAME[v.model] || v.model || '?'}${fr.length ? ` · F${fr[0]}–${fr[1]}` : ''}` };
  }
  function versionsHtml(d) {
    const n = target();
    const all = (d.versions || []).slice().reverse();
    const mine = n ? all.filter((v) => (v.remove || []).includes(n)) : all;
    const rest = all.length - mine.length;
    const row = (v) => {
      const shown = v.shown && onCleaned();   // Beauty / another plate on screen: no version is shown (2026-11-15)
      const fr = v.frames || [];
      const anc = (v.anchors || []).length ? ` · ⚓ ${(v.anchors || []).map((f) => `F${f}`).join(' ')}` : '';
      const upTxt = (v.upscaled_from ? ` · ⤴ z ${v.upscaled_from}` : v.recomposed_from ? ` · ⟲ z ${v.recomposed_from}` : '')
        + (v.input_plate ? ` · na ${v.input_plate}` : '');   // made on top of an earlier version („Wejście”)
      // „⟲ Złóż ponownie” (2026-11-13): only versions that kept the model's frames (model/)
      const rec = v.has_model ? `<button type="button" class="btn btn-ghost btn-xs cmp-run-act" data-cu-recomp="${esc(v.version)}" title="${esc(`Złóż ponownie ${v.version} bez liczenia modelu: jej obraz modelu z bieżącym pasem łaty (pas, wtopienie, punkty), ochroną, wyrównaniem poziomu i ziarnem – nowa wersja (ta zostaje)`)}">⟲ Złóż ponownie</button>` : '';
      const up = d.upscale || {};
      const why = [shown ? 'na plate „cleaned”' : '', v.prompt ? `„${v.prompt.length > 60 ? `${v.prompt.slice(0, 60)}…` : v.prompt}”` : ''].filter(Boolean).join(' · ');
      return `<div class="row-item cmp-run cu-ver${shown ? ' cur' : ''}" data-cu-show="${esc(v.version)}" title="${esc(shown ? `${v.version} jest na plate „cleaned”` : `Klik = ${v.version} na plate „cleaned” + jej model, opis i kotwice wracają do panelu`)}">
        <span class="cmp-run-txt"><span class="cmp-run-name">${shown ? '▶ ' : ''}${esc(v.version)} · ${esc(MODEL_NAME[v.model] || v.model || '?')} · F${esc(fr[0])}–${esc(fr[1])}${esc(anc)}${esc(upTxt)}</span>${why ? `<span class="cmp-run-why">${esc(why)}</span>` : ''}</span>
        ${rec}<button type="button" class="btn btn-ghost btn-xs cmp-run-act" data-cu-up="${esc(v.version)}" title="${esc(up.available ? `Upscale ${v.version} przez SeedVR2: wypełnienie podbite do rozdzielczości plate'u jako jeden klip, podmienione tylko w masce – nowa wersja (ta zostaje)` : `SeedVR2 niedostępny: ${up.reason || 'brak'} – Doktor → Zainstaluj SeedVR2`)}"${up.available ? '' : ' disabled'}>⤴</button>
        <button type="button" class="btn btn-ghost btn-xs cmp-run-del" data-cu-del="${esc(v.version)}" title="Usuń wersję ${esc(v.version)} z dysku na stałe (przed usunięciem zapytam)">🗑</button></div>`;
    };
    return `<details class="cmp-runs" open><summary title="Wersje czyszczenia tego obiektu; klik = na plate „cleaned” i jej ustawienia (model, opis, kotwice) wracają do panelu; 🗑 = usuń z dysku na stałe (z ostrzeżeniem)">Wersje czyszczenia (klik = pokaż + kotwice, 🗑 = usuń)${n ? ` – ${esc(n)}` : ''}</summary>`
      + (mine.length ? mine.map(row).join('') : `<div class="muted">${n ? `„${esc(n)}” – jeszcze nic nie policzono.` : 'Jeszcze nic nie policzono.'}</div>`)
      + '</details>'
      + (rest > 0 ? `<div class="muted">+ ${rest} wersji innych obiektów (wybierz je w palecie)</div>` : '')
      + `<div class="muted">${esc(d.out_dir)}</div>`;
  }

  // fill models (user 2026-11-10): LaMa / Qwen-Image-2.1 / VOID – not installed = a short card with the GPU verdict of
  // this card (engine/vram_fit) and the Doctor's install; prompt models get the description field
  const VRAM_HEAD = { ok: 'mieści się', spill: 'przelewa się do RAM-u', unstable: 'niestabilnie', no: 'nie zmieści się' };
  const VRAM_COL = { ok: 'var(--accent-success)', spill: '#f0a03a', unstable: '#f87171', no: '#f87171' };
  const curModel = (d) => (d.model_info || []).find((m) => m.id === d.settings.model) || (d.model_info || [])[0] || null;
  function vramLine(vr) {
    if (!vr) return '';
    return `<div style="color:${VRAM_COL[vr.level] || 'inherit'}" title="${esc(`${vr.text}${vr.why ? ` (${vr.why})` : ''}`)}">${vr.level === 'ok' ? '✔' : '⚠'} Ta karta ${esc(vr.card_gb)} GB: ${esc(VRAM_HEAD[vr.level] || vr.level)}${vr.basis === 'estimate' ? ' (szacunek)' : ''}</div>`;
  }
  // the model comparison of the right panel (user 2026-11-12): what each is best for, speed, GPU memory – memory from
  // engine/vram_fit.NEEDS (estimates unless measured), speed relative; an engine missing here shows its server text
  const MODEL_GUIDE = {
    lama: { best: 'małe i średnie obszary, bez opisu', speed: 3, mem: 1, memT: 'mało (< 2 GB)', short: 'Szybki, bez opisu – małe obszary.' },
    h3: { best: 'najlepsza spójność w czasie; Zastąp, przemiana, kotwice wyglądu', speed: 1, mem: 3, memT: 'bardzo dużo (model 19,5 GB + koder; < 32 GB VRAM → z RAM-u, ≥ 64 GB RAM)', short: 'Najlepsza jakość w czasie, Zastąp i przemiana – wolny, bardzo dużo pamięci.' },
    vace: { best: 'Usuń / Zastąp z kotwicami na kartach 12–16 GB', speed: 2, mem: 2, memT: 'średnio (ekspert 10,6 GB; ok od 16 GB, 12 GB = 480p)', short: 'Usuń / Zastąp z kotwicami na kartach 12–16 GB; 480p i „szybko” przyspieszają.' },
    void: { best: 'samo usuwanie z opisem tła, całe ujęcie naraz', speed: 2, mem: 2, memT: 'dużo (szczyt 16,4 GB; mniej = część z RAM-u)', short: 'Tylko usuwa – całe ujęcie naraz, z opisem tła.' },
    // LTX-2.5 (engine/vram_fit ltx_fill, estimate): 22B int8 21,5 GB + Gemma 4 12B int8 15,4 GB loaded separately, peak ~22,5 GB
    ltx: { best: 'cały klip naraz (dłuższy bok 1024), Usuń / Zastąp z opisem sceny, kotwice jako klatki kluczowe', speed: 2, mem: 3, memT: 'bardzo dużo (model 21,5 GB + koder 15,4 GB osobno, szczyt ok. 22,5 GB; < 24 GB VRAM → część z RAM-u)', short: 'Cały klip naraz, opis sceny, kotwice jako klatki kluczowe – bardzo dużo pamięci.' },
  };
  const dots = (n, ch) => `<span class="cu-dots" aria-label="${n}/3">${ch.repeat(n)}<span class="off">${ch.repeat(3 - n)}</span></span>`;
  function guideHtml(d, cur) {
    const rows = (d.model_info || []).filter((x) => x.id !== 'qwen21');
    if (!rows.length) return '';
    return `<div class="cu-guide-h">Który model?</div>
      <div class="cu-models">
      ${rows.map((x) => { const g = MODEL_GUIDE[x.id]; const inC1 = (d.model_info || []).some((y) => y.id === x.id); const st = x.state === 'comfy_restart' ? '🔁 Zrestartuj ComfyUI' : x.state === 'comfy_old' ? '⟳ Zaktualizuj ComfyUI' : x.install ? `⬇ Zainstaluj${x.download ? ` (${esc(x.download)})` : ''}` : '';
        return `<div class="cu-mcard${x.id === cur ? ' cur' : ''}${x.available ? '' : ' na'}" data-cu-pick="${esc(x.id)}" title="${esc(x.available ? `Klik = licz ${x.label}` : `${x.label} nie jest zainstalowany${x.reason ? ` – ${x.reason}` : ''}`)}" style="cursor:${inC1 ? 'pointer' : 'default'}">
        <div class="cu-mcard-top"><b>${esc(x.label)}</b>${x.available ? '' : '<span class="muted"> · nie zainstalowany</span>'}
          ${g ? `<span class="cu-mcard-r"><span title="Względna szybkość liczenia">Szybkość ${dots(g.speed, '⚡')}</span><span title="Pamięć karty graficznej (VRAM): ${esc(g.memT)}">Pamięć GPU ${dots(g.mem, '▮')}</span></span>` : ''}</div>
        <div class="cu-mcard-b">${esc(g ? g.best : x.what)}</div>${!x.available && st ? `<button type="button" class="btn btn-primary btn-xs" data-cu-minstall="${esc(x.id)}">${st}</button>` : ''}</div>`; }).join('')}
      </div>
      <div class="muted cu-guide-f">Klik w kartę = ten model. ⚡ = szybciej, ▮ = więcej pamięci GPU (szczegóły po najechaniu). Ocena dla Twojej karty – pod wyborem modelu w palecie.</div>`;
  }
  function modelHtml(m) {
    if (!m) return '';
    if (m.available) return vramLine(m.vram);   // the description is in the right panel's model cards (2026-11-15)
    if (m.state === 'comfy_restart') {   // ComfyUI already updated, the running one not restarted yet
      return `<div class="c3-tool-card"><div><b>${esc(m.label)}</b> – ComfyUI zaktualizowany, czeka na restart</div>
        <div class="muted">Działający ComfyUI ma jeszcze stare węzły w pamięci – po restarcie ${esc(m.label)} będzie dostępny.</div>
        <button type="button" class="btn btn-primary btn-xs" data-cu-install="restart_comfy">🔁 Zrestartuj ComfyUI</button></div>`;
    }
    if (m.state === 'comfy_old') {   // the model is on disk, this ComfyUI lacks its node (2026-11-10)
      return `<div class="c3-tool-card"><div><b>${esc(m.label)}</b> – zainstalowany, wymaga nowszego ComfyUI</div>
        <div class="pp-hint warn">${esc(m.reason || '')}</div>${vramLine(m.vram)}
        ${m.install ? `<button type="button" class="btn btn-primary btn-xs" data-cu-install="${esc(m.install)}">⟳ Zaktualizuj ComfyUI</button>
          <div class="muted">Potem restart ComfyUI (Doktor → „Zrestartuj ComfyUI”); „Cofnij aktualizację ComfyUI” w Doktorze wraca do poprzedniej wersji.</div>`
          : '<div class="muted">ComfyUI Desktop: zaktualizuj go w aplikacji.</div>'}</div>`;
    }
    return `<div class="c3-tool-card" title="${esc(m.reason || '')}"><div><b>${esc(m.label)}</b> – nie zainstalowany</div>
      <div class="muted">${esc(m.what)} · pobiera ${esc(m.download)}</div>${vramLine(m.vram)}
      ${m.install ? `<button type="button" class="btn btn-primary btn-xs" data-cu-install="${esc(m.install)}">⬇ Zainstaluj ${esc(m.label)} (${esc(m.download)})</button>`
        : `<div class="pp-hint warn">${esc(m.reason || 'niedostępny')}</div>`}</div>`;
  }
  async function installModel(which) {
    const d = C.info; const m = which || (d && curModel(d)); const v = window.viewer;
    if (!m || !v || !v.installWithProgress) return;
    if (m.state === 'comfy_restart') {
      const ok = await v.uiDialog({ title: 'Zrestartować ComfyUI?', ok: '🔁 Zrestartuj', cancel: 'Anuluj', tone: 'warn',
        html: '<p>ComfyUI wstanie ponownie (zwykle 30–90 s) i załaduje nowe węzły. Niezapisane zmiany w oknie ComfyUI przepadną.</p>' });
      if (!ok) return;
      const r = await v.installWithProgress('restart_comfy');
      const res = (r && r.data) || {};
      window.UI.toast(r && r.ok && res.success !== false ? 'ComfyUI zrestartowany' : `Restart ComfyUI nie powiódł się – ${res.detail || (res.log || []).slice(-1)[0] || 'Doktor'}`, 8000);
      await load();
      return;
    }
    if (m.state === 'comfy_old') {   // the ComfyUI update (engine/comfy_update.py), not a model download
      const ok = await v.uiDialog({ title: 'Zaktualizować ComfyUI?', ok: '⟳ Zaktualizuj', cancel: 'Anuluj', tone: 'warn',
        html: `<p>${esc(m.reason || '')}</p><p>git pull (tylko szybkie przewinięcie), zależności z przypiętym PyTorchem; obecna wersja zapamiętana do cofnięcia. Potem restart ComfyUI.</p>` });
      if (!ok) return;
      const r = await v.installWithProgress('comfy_update');
      const res = (r && r.data) || {};
      window.UI.toast(r && r.ok && res.success !== false ? 'ComfyUI zaktualizowany – zrestartuj ComfyUI (Doktor)' : `Aktualizacja ComfyUI nie powiodła się – ${res.detail || (res.log || []).slice(-1)[0] || 'szczegóły w Doktorze'}`, 10000);
      await load();
      return;
    }
    const vr = m.vram && m.vram.level !== 'ok' ? `<p style="color:${VRAM_COL[m.vram.level]}">⚠ Ta karta (${esc(m.vram.card_gb)} GB): ${esc(VRAM_HEAD[m.vram.level])} – ${esc(m.vram.text)}</p>` : '';
    const ok = await v.uiDialog({ title: `Zainstalować ${m.label}?`, ok: 'Zainstaluj', cancel: 'Anuluj', tone: vr ? 'warn' : 'info',
      html: `${vr}<p>${esc(m.what)}. Pobiera ${esc(m.download)}; przerwane pobieranie wznawia się.</p>${m.license ? `<p>Licencja: ${esc(m.license)} – pobierając, akceptujesz ją.</p>` : ''}` });
    if (!ok) return;
    const r = await v.installWithProgress(m.install);
    const res = (r && r.data) || {};
    window.UI.toast(r && r.ok && res.success !== false ? `${m.label}: zainstalowany` : `${m.label}: instalacja nie powiodła się – ${res.detail || (res.log || []).slice(-1)[0] || 'szczegóły w Doktorze'}`, 9000);
    await load();
  }

  // the object bar of Cleanup (user 2026-11-13): like „Obiekt” of the cutting stages, in the same place, without ＋ / 🗑 –
  // the things that can be cleaned (a group, or an object outside any group; a member stands for its group)
  // „Wejście” (user 2026-11-15): Beauty or any computed Inpaint version (newest first), as the plate to work on
  function renderInput(d) {
    const sel = $('cuInput'); if (!sel) return;
    const cur = d.settings.input_plate || '';
    const vs = (d.versions || []).slice().reverse();
    const opt = (v) => { const fr = v.frames || []; return `<option value="${esc(v.version)}"${v.version === cur ? ' selected' : ''}>${esc(v.version)} · ${esc(MODEL_NAME[v.model] || v.model || '?')} · ${esc((v.remove || []).join(', '))}${fr.length ? ` · F${fr[0]}–${fr[1]}` : ''}</option>`; };
    const missing = cur && !vs.some((v) => v.version === cur) ? `<option value="${esc(cur)}" selected>${esc(cur)} – brak (usunięta)</option>` : '';
    if (document.activeElement !== sel) sel.innerHTML = `<option value=""${cur ? '' : ' selected'}>Beauty (oryginalny plate)</option>${missing}${vs.map(opt).join('')}`;
  }
  function targetBarHtml() {
    return `<div class="stage-object cu-target-box" id="cuTargetBox" style="display:none" title="Co czyścisz: grupę albo obiekt spoza grupy (jedna rzecz naraz) – to samo co klik w wiersz palety Obiekty">
      <label class="obj-label" for="cuTargetSel">Obiekt</label><select id="cuTargetSel" class="form-input"></select></div>`;
  }
  function renderTargetBar() {
    const sel = $('cuTargetSel'); if (!sel) return;
    const objs = (C.info && C.info.objects) || [];
    const can = objs.filter((o) => o.kind === 'group' || !o.group);
    const cur = target();
    sel.innerHTML = can.length
      ? (cur ? '' : '<option value="" selected>— wybierz —</option>') + can.map((o) => {
        const m = o.mask || {};
        const tag = o.kind === 'group' ? ' (grupa)' : '';
        const warn = m.run ? (m.valid ? '' : ' ↻') : ' – brak maski';
        return `<option value="${esc(o.name)}"${o.name === cur ? ' selected' : ''}>${esc(o.name)}${tag}${warn}</option>`;
      }).join('')
      : '<option value="">— brak obiektów —</option>';
    if (document.activeElement !== sel) sel.value = cur || '';
  }
  function targetBarVis() {
    const box = $('cuTargetBox'); if (box) box.style.display = mode === 'clean' ? '' : 'none';
    renderBandKeysLane();   // the band key lane only in C1
  }

  function render() {
    const d = C.info; if (!d) return;
    renderTargetBar();
    const s = d.settings;
    renderInput(d);
    $('cuObjects').innerHTML = d.objects.length ? targetHtml(d) : '<div class="muted">Brak obiektów – najpierw wytnij obiekt (tryb Wycinanie, etap 1).</div>';
    const o = objOf(target());
    const m = curModel(d);
    $('cuModel').innerHTML = (d.model_info || []).map((x) => `<option value="${esc(x.id)}"${x.id === s.model ? ' selected' : ''}>${esc(x.label)}${x.prompt ? ' (z opisem)' : ''}${x.available ? '' : x.state === 'comfy_restart' ? ' – zrestartuj ComfyUI' : x.state === 'comfy_old' ? ' – wymaga nowszego ComfyUI' : ' – nie zainstalowany'}</option>`).join('');
    $('cuModelInfo').innerHTML = modelHtml(m);
    $('cuGuide').innerHTML = guideHtml(d, m && m.id);
    const pr = $('cuPrompt');
    pr.style.display = m && m.prompt ? '' : 'none';
    $('cuPromptBox').style.display = $('cuDoHead').style.display = m && m.prompt ? '' : 'none';
    if (document.activeElement !== pr) pr.value = s.prompt || '';
    // Usuń / Zastąp (user 2026-11-10): only the instruction around the description changes (engine/cleanup3d/regen.
    // qwen_prompt); VOID only removes
    const pm = s.prompt_mode === 'replace' ? 'replace' : 'remove';
    const pmBox = $('cuPromptMode');
    pmBox.style.display = m && m.prompt ? '' : 'none';
    pmBox.querySelectorAll('[data-pm]').forEach((b) => {
      b.classList.toggle('btn-primary', b.dataset.pm === pm);
      b.disabled = b.dataset.pm === 'replace' && m && m.id === 'void';
    });
    const replaceVoid = pm === 'replace' && m && m.id === 'void';
    const hint = $('cuPromptHint');
    hint.style.display = m && m.prompt ? '' : 'none';
    hint.textContent = replaceVoid ? 'VOID tylko usuwa obiekt – do „Zastąp” wybierz Qwen-Image-2.1.'
      : pm === 'replace' ? 'Nazwij obiekt i czym go zastąpić – opisz wygląd, nie przemianę.'
      : 'Opisz tło, które ma zostać (puste = opis ogólny).';
    pr.placeholder = pm === 'replace' ? 'np. „the woman\'s face with a porcelain mask” (po angielsku)' : 'Co ma być w miejscu obiektu (po angielsku), np. „plain white wall, soft daylight”';
    // anchors (MiniMax-H3 / Wan 2.2 Fun VACE – the engines with `anchors` in model_info): the on / off select and the
    // Qwen-Image-2.1 card sit in the head of the anchor palette (static/cleanup_anchors.js, user 2026-11-11)
    const isH3 = !!(m && m.anchors);
    const am = d.anchor_model;
    const anc = s.anchor || 'none';
    const anchorOk = !!(am && am.available);
    $('cuSeedRow').style.display = isH3 ? '' : 'none';
    // „Siła zmiany” (user 2026-11-13): only MiniMax-H3 Replace starts from the plate; Remove keeps the hole
    const showStr = !!(m && m.id === 'h3' && pm === 'replace');
    $('cuStrengthRow').style.display = showStr ? '' : 'none';
    const sv = s.h3_strength != null ? +s.h3_strength : 1;
    if (document.activeElement !== $('cuStrength')) $('cuStrength').value = sv;
    $('cuStrengthVal').textContent = sv >= 1 ? '1,00 (dziura)' : sv.toFixed(2).replace('.', ',');
    const isVace = !!(m && m.id === 'vace');   // Wan 2.2 Fun VACE: resolution + speed (Lightning LoRA, 2026-11-11)
    $('cuVaceRow').style.display = isVace ? '' : 'none';
    $('cuVaceRes').value = String(s.vace_res || 720);
    $('cuVaceSpeed').value = s.vace_speed === 'quality' ? 'quality' : 'fast';
    const noLightning = isVace && s.vace_speed !== 'quality' && !m.lightning;
    // 2026-11-13 (user: „wybieram 4, liczy się 20”): without the two Lightning LoRAs the fast option silently ran 20
    // steps – the option says so now and the note installs them in place
    const fastOpt = $('cuVaceSpeed').querySelector('option[value="fast"]');
    if (fastOpt) fastOpt.textContent = isVace && !m.lightning ? 'szybko – 4 kroki (brak LoRA → liczy 20)' : 'szybko – 4 kroki';
    $('cuVaceNote').style.display = noLightning ? '' : 'none';
    $('cuVaceNote').innerHTML = noLightning ? 'Brak LoRA Lightning – Wan 2.2 Fun VACE liczy teraz 20 kroków, nie 4. <button type="button" class="btn btn-primary btn-xs" id="cuLightningInstall" title="Pobiera LoRA lightx2v 4-step (high + low, 2 × 1,23 GB) z Comfy-Org/Wan_2.2_ComfyUI_Repackaged do models/loras; restart ComfyUI niepotrzebny">⬇ Zainstaluj LoRA Lightning (2 × 1,23 GB)</button>' : '';
    if (document.activeElement !== $('cuSeed')) $('cuSeed').value = s.seed ?? 42;
    if (window.MD_CU_ANCHORS) window.MD_CU_ANCHORS.render({ on: isH3 && anc !== 'none' && !!(o && o.mask && o.mask.run), h3: isH3, anchor: anc, anchorModel: am, qwenOk: anchorOk,
      // live: save() / flush() swap C.info.settings for the server's copy – a snapshot here kept the anchor
      // checkboxes ticked after unticking (user 2026-11-13)
      get settings() { return (C.info && C.info.settings) || s; }, engine: m ? m.label : '' });
    syncH3();
    const anchorBlocks = isH3 && anc !== 'none' && !anchorOk;
    $('cuRun').disabled = !(o && o.mask && o.mask.run) || !(m && m.available) || replaceVoid || anchorBlocks;
    $('cuRun').title = m && !m.available ? `${m.label} nie jest zainstalowany – zainstaluj go albo wybierz inny model`
      : anchorBlocks ? 'Kotwice Qwen-Image-2.1 niedostępne – wyłącz kotwice albo przygotuj Qwen-Image-2.1'
        : 'Czyszczenie na zakresie pracy (oś czasu) → nowa wersja w <work>/inpaint i plate „cleaned”';
    $('cuPreview').disabled = !(o && o.mask && o.mask.run);
    syncBandSliders(); renderBandKeysLane();
    if (document.activeElement !== $('cuGen') && document.activeElement !== $('cuGenR')) { $('cuGen').value = s.gen_margin_px ?? 24; $('cuGenR').value = Math.min(200, s.gen_margin_px ?? 24); }
    bandLimitUi();
    recompUi();
    $('cuHoldout').checked = s.holdout; $('cuFill').value = s.fill;
    $('cuSkipLost').checked = !!s.holdout_skip_lost; $('cuSkipLost').disabled = !s.holdout;
    renderKept();
    renderGrain(d);
    const src = d.sources || {};
    $('cuSources').textContent = `clear plate: ${src.clean || 'brak'} · auto clean plate: ${src.auto || 'brak'} · LaMa: ${src.lama ? (src.lama_full ? 'gotowa (pełna rozdzielczość)' : 'gotowa') : 'brak (ComfyUI / Doktor)'}${src.lama ? '' : ' – wypełnienie OpenCV'}`;
    $('cuVersions').innerHTML = versionsHtml(d);
    if (window.viewer && window.viewer.updateHudMode) window.viewer.updateHudMode();   // „cleaned · v016 …” in the HUD
    if (window.MD && window.MD.renderOthersPanel) window.MD.renderOthersPanel();   // 🧹 at the cleaned row
    if (window.MD_CU_PROMPT) window.MD_CU_PROMPT.later();   // the description's warnings + what each model gets
    // kept objects that looked lost in the last version (engine/cleanup.py check_kept) – shown until the next run
    const last = (d.versions || []).slice(-1)[0];
    const warn = (last && last.warnings) || [];
    $('cuWarn').style.display = warn.length ? '' : 'none';
    $('cuWarn').innerHTML = warn.map((w) => `<div>⚠ ${esc(last.version)}: ${esc(w)}</div>`).join('');
    palVis();
  }
  // grain match of the video model's fill (engine/cleanup.Cleaner._grain, user 2026-11-13): the setting + what the last
  // version of the cleaned thing measured (equivalent sigma in 8-bit levels, engine/cleanup3d/grain.summary)
  function renderGrain(d) {
    const s = d.settings;
    if (document.activeElement !== $('cuGrain')) $('cuGrain').value = s.grain || 'auto';
    if (document.activeElement !== $('cuGrainStrength')) $('cuGrainStrength').value = s.grain_strength ?? 1;
    $('cuGrainStrength').disabled = $('cuGrain').value === 'off';
    $('cuGrainStrengthV').textContent = `×${(+$('cuGrainStrength').value).toFixed(2)}`;
    const n = target();
    const v = (d.versions || []).filter((x) => !n || (x.remove || []).includes(n)).slice(-1)[0];
    const g = v && v.grain;
    // run.json grain: plate / patch = noise per channel R, G, B (0..1; engine/cleanup3d/grain.summary) – an array, a
    // number in older runs (2026-11-15: the array went through `* 255` = „plate NaN, łata NaN”)
    const lv = (x) => {
      if (x == null) return '–';
      const a = (Array.isArray(x) ? x : [x]).map(Number).filter((n) => Number.isFinite(n));
      return a.length ? a.map((n) => (n * 255).toFixed(2)).join(' / ') : '–';
    };
    const mean = (x) => { const a = (Array.isArray(x) ? x : [x]).map(Number).filter(Number.isFinite); return a.length ? a.reduce((p, q) => p + q, 0) / a.length : null; };
    const why = g && g.measured && !g.added && g.mode === 'auto' && mean(g.plate) != null && mean(g.patch) != null && mean(g.plate) <= mean(g.patch)
      ? ' (auto: łata nie jest gładsza od plate\'u – nie trzeba)' : '';
    $('cuGrainInfo').textContent = !g ? 'Pomiar szumu plate\'u i łaty pojawi się po „Policz”.'
      : !g.measured ? `${v.version}: za mało płaskich miejsc obok łaty, żeby zmierzyć szum – ziarno nie dodane.`
        : `${v.version} – szum (poziomy 8-bit, R / G / B): plate ${lv(g.plate)}, łata ${lv(g.patch)}; ziarno dodane w ${g.added} z ${g.frames} klatek${why}.`;
  }
  // the clip prompt of a TRANSITION (own / original anchors) for the chosen video engine (MiniMax-H3 / Wan 2.2 Fun VACE –
    // both read `h3_prompt`): only then, in the Inpaint palette, named after the engine
  function syncH3() {
    const row = $('cuH3Row'); if (!row || !C.info) return;
    const s = C.info.settings; const m = curModel(C.info);
    const A = window.MD_CU_ANCHORS;
    const tr = !!(m && m.anchors && s.anchor === 'frame' && A && A.transition && A.transition());
    if (m && m.anchors && $('cuH3Eng')) $('cuH3Eng').textContent = m.label;
    row.style.display = tr ? '' : 'none';
    const ta = $('cuH3Prompt');
    if (tr && document.activeElement !== ta) ta.value = (A.h3Prompt && A.h3Prompt()) ?? s.h3_prompt ?? '';
  }

  async function load() {
    const seq = (C.loadSeq = (C.loadSeq || 0) + 1);   // a pick right after the mode switch: only the newest answer renders
    let d;
    try { d = await api('GET', '/api/cleanup'); } catch (e) {
      if (seq !== C.loadSeq) return;
      C.info = null; $('cuObjects').innerHTML = `<div class="muted">${esc(e.message)}</div>`; return;
    }
    if (seq !== C.loadSeq) return;
    C.info = d;
    // the masks may have changed in cutting (user decision 2026-11-15, point 3): the live preview and the holdout
    // overlays are asked again (the server's caches follow the masks' files); overlays of objects no longer kept go
    for (const n of [...C.ho.on]) if (!(d.kept || []).some((k) => k.name === n)) C.ho.on.delete(n);
    C.liveBump = (C.liveBump || 0) + 1; C.ho.bump += 1;
    render();
    if (window.MD_CU_ANCHORS) window.MD_CU_ANCHORS.reload();
  }


  function save(patch) {
    clearTimeout(C.saveT);
    C.pending = { ...(C.pending || {}), ...patch };
    C.saveT = setTimeout(async () => {
      const b = C.pending; C.pending = null;
      try { const r = await api('PUT', '/api/cleanup', b); if (C.info) { C.info.settings = r.settings; if ('model' in b) render(); } } catch (e) { window.UI.toast(`Czyszczenie: ${e.message}`, 5000); }
    }, 250);
  }
  async function flush() { if (C.pending) { clearTimeout(C.saveT); const b = C.pending; C.pending = null; const r = await api('PUT', '/api/cleanup', b); if (C.info) C.info.settings = r.settings; } }

  async function previewMask() {
    const v = window.viewer; if (!v) return;
    await flush();
    const f = v.currentFrame;
    const cv = $('cuMaskCv');
    try {
      const g = +$('cuGrow').value || 0, sf = +$('cuSoft').value || 0;
      const [plate, mask, lines] = await Promise.all([loadImg(`/api/frame/${f}?plate=beauty`), loadImg(`/api/cleanup/mask/${f}?t=${Date.now()}`),
        api('GET', `/api/cleanup/band/${f}/lines?grow_px=${g}&soft_px=${sf}`).catch(() => null)]);
      cv.width = mask.width; cv.height = mask.height; cv.style.display = '';
      const ctx = cv.getContext('2d');
      ctx.drawImage(plate, 0, 0, cv.width, cv.height);
      const t = document.createElement('canvas'); t.width = cv.width; t.height = cv.height;
      const tc = t.getContext('2d'); tc.drawImage(mask, 0, 0);
      const id = tc.getImageData(0, 0, t.width, t.height);
      // R = the patch (red), G = what the model draws – the generation margin + the engine's grow (cyan, faint; 2026-11-15)
      for (let i = 0; i < id.data.length; i += 4) {
        const m = id.data[i], gm = id.data[i + 1];
        if (m > 0) { id.data[i] = 255; id.data[i + 1] = 40; id.data[i + 2] = 40; id.data[i + 3] = Math.max(Math.round(m * 0.6), Math.round(gm * 0.22)); }
        else { id.data[i] = 60; id.data[i + 1] = 200; id.data[i + 2] = 255; id.data[i + 3] = Math.round(gm * 0.22); }
      }
      tc.putImageData(id, 0, 0);
      ctx.drawImage(t, 0, 0);
      // the band lines over it (user 2026-11-11): dashed = the band's edge, thin dashes = where the soft edge ends –
      // the reach of „Wtopienie zewn.”; the plate px of the lines scaled to this canvas
      if (lines && !lines.error) {
        const pd = window.viewer && window.viewer.projectData;
        const k = cv.width / ((pd && pd.plate_w) || cv.width);   // plate px -> this canvas (the proxy)
        // the ONE band drawing of every stage (MD_BANDPTS.strokeLines), one screen px = canvas px / display width
        if (window.MD_BANDPTS) window.MD_BANDPTS.strokeLines(ctx, k, lines, { px: Math.max(1, cv.width / 540) });
        // the edge of what the model draws (generation margin + the engine's grow, 2026-11-13): solid cyan line
        ctx.save(); ctx.strokeStyle = 'rgba(80,220,255,0.95)'; ctx.lineWidth = Math.max(1, cv.width / 540); ctx.setLineDash([]);
        for (const c of lines.limit_contours || []) {
          if (!c.length) continue;
          ctx.beginPath(); ctx.moveTo(c[0][0] * k, c[0][1] * k);
          for (const [x, y] of c.slice(1)) ctx.lineTo(x * k, y * k);
          ctx.closePath(); ctx.stroke();
        }
        ctx.restore();
      }
      cv.dataset.frame = String(f);
      cv.title = `Maska F${f} – klik = duży podgląd (zoom, przesuwanie); czerwień = łata, przerywane linie = krawędź pasa i koniec wtopienia, niebieska ciągła i jasnoniebieskie tło = to, co narysuje model (pas generowania + rozrost silnika)${lines && lines.cut ? ' (pas ucięty na niej)' : ''}`;
    } catch (e) { window.UI.toast(`Maska F${f}: ${e.message || e}`, 6000); }
  }
  function loadImg(u) {
    return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error(`nie wczytano ${u}`)); i.src = u; });
  }

  async function pick(name) {
    if (!name) return;
    C.picked = name;   // a member picked: the panel says its group is cleaned
    try {
      await flush();   // an edit still waiting belongs to the thing shown so far (its settings are per cleaned thing)
      const r = await api('PUT', '/api/cleanup', { remove: [name] });
      if (C.info) C.info.settings = r.settings;
      // user 2026-11-15: another thing picked = its newest version on the preview, none = the original plate
      const v = window.viewer;
      if (v && r.plates) {
        v.updatePlateSelector(r.plates);
        if (r.version) { if (v.bumpPlate) v.bumpPlate('cleaned'); else if (v.frameCache) v.frameCache.clear(); v.setActivePlate('cleaned'); }
        else if (r.plates.beauty) v.setActivePlate('beauty');
        C.live = null;
      }
    } catch (e) { window.UI.toast(`Czyszczenie: ${e.message}`, 5000); }
    if (window.MD_BANDPTS && window.MD_BANDPTS.ready) window.MD_BANDPTS.reset();   // its own band (per cleaned thing)
    await load();
  }
  function afterPlates(r) {
    const v = window.viewer;
    if (v && r && r.plates) { v.updatePlateSelector(r.plates); if (v.bumpPlate) v.bumpPlate('cleaned'); else if (v.frameCache) v.frameCache.clear(); if (r.plates[r.plate || 'cleaned']) v.setActivePlate('cleaned'); }
  }
  async function showVersion(ver) {
    try {
      await flush();
      const r = await api('POST', `/api/cleanup/versions/${encodeURIComponent(ver)}/show`);
      afterPlates(r);
      if (window.MD_BANDPTS && window.MD_BANDPTS.ready) window.MD_BANDPTS.reset();
      const na = ((r.settings || {}).anchors || []).length;
      window.UI.toast(`Plate „cleaned” = ${ver}${na ? ` · kotwice ${ver} (${na}) w palecie` : ''}`, 3500);
    }
    catch (e) { window.UI.toast(`${ver}: ${e.message}`, 6000); }
    await load();
  }
  async function upscaleVersion(ver) {   // SeedVR2 over the version's fill -> a new version (api/cleanup.py)
    try {
      const r = await api('POST', `/api/cleanup/versions/${encodeURIComponent(ver)}/upscale`);
      afterPlates(r);
      window.UI.toast(`Upscale ${ver} → ${r.version} (SeedVR2, ${r.n_frames} kl., ${r.seconds} s)`, 6000);
    } catch (e) { window.UI.toast(`Upscale ${ver}: ${e.message}`, 9000); }
    await load();
  }
  async function recomposeVersion(ver) {   // „⟲ Złóż ponownie”: the version's model frames + the current band (api/cleanup.py)
    try {
      await flush();
      const r = await api('POST', `/api/cleanup/versions/${encodeURIComponent(ver)}/recompose`);
      afterPlates(r);
      C.live = null;
      window.UI.toast(`Złożono ponownie ${ver} → ${r.version} (${r.n_frames} kl., ${r.seconds} s, bez modelu)${(r.band_cut || []).length ? ' · pas ucięty na granicy obszaru modelu' : ''}`, 7000);
    } catch (e) { window.UI.toast(`Złóż ponownie ${ver}: ${e.message}`, 9000); }
    await load();
  }
  async function deleteVersion(ver) {
    const ok = await window.UI.confirmDelete({ title: `Usunąć wersję ${ver}?`, ok: 'Usuń wersję na stałe',
      what: `Wersja czyszczenia ${esc(ver)} – wszystkie jej klatki. Gdy plate „cleaned” pokazuje tę wersję, przejdzie na najnowszą pozostałą.` });
    if (!ok) return;
    try {
      const r = await api('DELETE', `/api/cleanup/versions/${encodeURIComponent(ver)}`);
      afterPlates({ ...r, plate: 'cleaned' });
      window.UI.toast(`Wersja ${ver} usunięta z dysku (zwolniono ${window.UI.bytes(r.freed_bytes)})`, 5000);
    }
    catch (e) { window.UI.toast(`${ver}: ${e.message}`, 6000); }
    await load();
  }

  async function runCleanup() {
    await flush();
    const btn = $('cuRun'); btn.disabled = true;
    try {
      const r = await api('POST', '/api/cleanup/run');
      const c = r.counts || {};
      window.UI.toast(`Czyszczenie ${r.version}: ${r.n_frames} kl., ${r.seconds} s · clear plate ${c['clear plate'] || 0} px, LaMa ${c.LaMa || 0} px, OpenCV ${c.OpenCV || 0} px${(r.notes || []).length ? ' · ' + r.notes.join('; ') : ''}`, 10000);
      const v = window.viewer;
      if (v && r.plates) { v.updatePlateSelector(r.plates); if (v.bumpPlate) v.bumpPlate(r.plate || 'cleaned'); else if (v.frameCache) v.frameCache.clear(); v.setActivePlate(r.plate); }
      await load();   // reloads the anchors too (the run may have made generations)
    } catch (e) { window.UI.toast(`Czyszczenie: ${e.message}`, 8000); } finally { btn.disabled = false; }
  }

  function apply() {
    document.querySelectorAll('#modeSwitch [data-mode]').forEach((b) => b.classList.toggle('btn-primary', b.dataset.mode === mode));
    const clean = mode === 'clean';
    document.body.classList.toggle('mode-clean', clean);
    // Cleanup 3D (static/cleanup3d.js, docs/PLAN_CLEANUP_3D.md §2): the stage bar stays and shows the steps C1–C6; this
    // panel = step C1; the .c3-pane panes are shown / hidden by cleanup3d.js itself
    const c3 = window.MD_CLEANUP3D;
    const bar = $('stageBar'); if (bar) bar.style.display = clean && !c3 ? 'none' : '';
    // The panes of the cutting stages are hidden by ONE stylesheet rule (index.css: body.mode-clean #stagePanel > …),
    // never by their inline style: objects.js writes that style on its own events (setStage after the project loads,
    // a stage change), which showed a stage pane next to the Cleanup one after a page reload (browser check of wave 1)
    // and left stale values to restore. So „✂ Wycinanie” gets exactly what objects.js set last.
    const cu = $('cleanupPanel'); if (cu) cu.style.display = clean && (!c3 || c3.step === 'remove') ? '' : 'none';
    if (c3) c3.onMode(clean);
    targetBarVis();
    // the cutting view (Alpha / Comp) must not stay on the canvas in Cleanup: objects.js setView forces the plate in
    // mode-clean and restoreUserView brings the user's cutting view back (2026-11-10)
    const md = window.MD;
    if (md && md.setView) { if (clean) md.setView('none', true, false); else if (md.restoreUserView) md.restoreUserView(); }
    if (md && md.renderOthersPanel) md.renderOthersPanel();   // the palette's highlight: cleaned thing ↔ cutting selection
    if (c3 && !C.c3hooked && c3.onChange) { C.c3hooked = true; c3.onChange(() => palVis()); }   // C1 ↔ C2–C6
    palVis();                                                  // the Inpaint / anchor palettes only in step C1
    // the plate follows the mode (user 2026-11-10): Cleanup shows the plate „cleaned” (when computed), Cut-out the beauty
    followPlate();
    if (clean) load().then(() => {   // nothing picked yet: the object selected in the palette
      // only when the settings really say nothing is cleaned: a reload in Cleanup ran this before the project was open
      // (GET /api/cleanup failed, C.info null) and re-picked the cut-out selection over the saved target (2026-11-11)
      if (C.info && !target() && md && md.S && md.S.obj) pick(md.S.obj);
    });
    else if (window.MD && window.MD.renderStageBar) window.MD.renderStageBar();
  }
  // the plate follows the mode once per mode change; a page reload in Cleanup does it when the project arrives
  // (md-project; user report 2026-11-11: after a reload Cleanup showed the beauty with the SAM 3.1 mask)
  function followPlate() {
    const v = window.viewer;
    if (!(v && v.projectLoaded && v.setActivePlate) || mode === C.lastMode) return;
    const has = (n) => !!(v.plateSelect && [...v.plateSelect.options].some((o) => o.value === n));
    const clean = mode === 'clean';
    if (clean && has('cleaned')) v.setActivePlate('cleaned');
    else if (!clean && v.activePlate !== 'beauty' && has('beauty')) v.setActivePlate('beauty');
    C.lastMode = mode;
  }
  window.addEventListener('md-project', () => {
    C.lastMode = null;   // another project: its plate follows the current mode again
    setTimeout(() => {
      followPlate(); const md = window.MD; if (mode === 'clean' && md && md.setView) md.setView('none', true, false);
      // the project is open now: its saved cleanup settings (and the saved target) – a load before it had failed
      if (mode === 'clean') load().then(() => { if (C.info && !target() && md && md.S && md.S.obj) pick(md.S.obj); });
    }, 0);
  });
  function setMode(m) {
    mode = m === 'clean' ? 'clean' : 'cut';
    try { localStorage.setItem(KEY, mode); } catch (e) { /* storage blocked */ }
    apply();
  }

  function init() {
    const left = document.querySelector('.header-left');
    const panel = $('stagePanel');
    if (!left || !panel || $('modeSwitch')) return;
    left.insertAdjacentHTML('beforeend', switchHtml());
    panel.insertAdjacentHTML('beforeend', panelHtml());
    const sbar = $('stageBar');
    if (sbar) {
      sbar.insertAdjacentHTML('afterbegin', targetBarHtml());
      $('cuTargetSel').addEventListener('change', (e) => { if (e.target.value) pick(e.target.value); });
    }
    const cw = $('canvasWrapper');
    if (cw) {
      cw.insertAdjacentHTML('beforeend', paletteHtml());
      const pal = $('cuPal');
      if (window.MD && window.MD.makeFloating) window.MD.makeFloating(pal, cw, '.pp-head', palPos);
      cw.insertAdjacentHTML('beforeend', compPaletteHtml());
      if (window.MD && window.MD.makeFloating) window.MD.makeFloating($('cuCompPal'), cw, '.pp-head', compPos);
    }
    $('modeSwitch').addEventListener('click', (e) => { const b = e.target.closest('[data-mode]'); if (b) setMode(b.dataset.mode); });
    $('cuGuide').addEventListener('click', async (e) => {   // the model cards (2026-11-15): pick / install
      const d = C.info; if (!d) return;
      const ib = e.target.closest('[data-cu-minstall]');
      if (ib) { const m = (d.model_info || []).find((x) => x.id === ib.dataset.cuMinstall); if (m) { ib.disabled = true; installModel(m).finally(() => { ib.disabled = false; }); } return; }
      const card = e.target.closest('[data-cu-pick]'); if (!card) return;
      const id = card.dataset.cuPick; const m = (d.model_info || []).find((x) => x.id === id);
      if (!m || !m.available || d.settings.model === id) return;
      d.settings.model = id; render(); save({ model: id }); await flush();
      if (window.MD_CU_ANCHORS) window.MD_CU_ANCHORS.reload();
    });
    $('cuVersions').addEventListener('click', (e) => {
      const del = e.target.closest('[data-cu-del]'); const show = e.target.closest('[data-cu-show]');
      const up = e.target.closest('[data-cu-up]');
      const rc = e.target.closest('[data-cu-recomp]');
      if (rc) { e.stopPropagation(); recomposeVersion(rc.dataset.cuRecomp); return; }
      if (up) { e.stopPropagation(); upscaleVersion(up.dataset.cuUp); return; }
      if (del) { e.stopPropagation(); deleteVersion(del.dataset.cuDel); }
      else if (show && !show.classList.contains('cur')) showVersion(show.dataset.cuShow);
    });
    // the shared Objects palette picks what is cleaned (static/objects.js selectObject -> md-object-pick)
    window.addEventListener('md-object-pick', (e) => { if (mode === 'clean') pick(e.detail && e.detail.name); });
    bandSlider('cuGrowR', 'cuGrow', 'grow_px');
    bandSlider('cuSoftR', 'cuSoft', 'soft_px');
    // „Zapas generowania” (2026-11-13): the model's mask before the run – no live lines, only the band's limit note
    $('cuLive').addEventListener('change', (e) => {
      C.liveOn = e.target.checked; C.liveErr = '';
      try { localStorage.setItem('md_cu_live', C.liveOn ? '1' : '0'); } catch (er) { /* storage blocked */ }
      recompUi(); if (window.viewer) window.viewer.render();
    });
    $('cuModelPv').addEventListener('change', (e) => { C.modelPv = e.target.checked; if (window.viewer) window.viewer.render(); });
    $('cuRecompose').addEventListener('click', () => { const v = shownRec(); if (v) recomposeVersion(v.version); });
    $('cuGenR').addEventListener('input', () => { $('cuGen').value = $('cuGenR').value; bandLimitUi(); });
    const genCommit = (val) => { if (C.info) C.info.settings.gen_margin_px = val; bandLimitUi(); save({ gen_margin_px: val }); };
    $('cuGenR').addEventListener('change', () => genCommit(+$('cuGenR').value || 0));
    $('cuGen').addEventListener('change', () => { $('cuGenR').value = Math.min(200, +$('cuGen').value || 0); genCommit(+$('cuGen').value || 0); });
    $('cuBandPv').addEventListener('change', (e) => {
      try { localStorage.setItem('md_cu_bandpv', e.target.checked ? '1' : '0'); } catch (er) { /* storage blocked */ }
      if (!e.target.checked && C.bandTool) { C.bandTool = null; bandUi(); if (window.MD_BANDPTS) window.MD_BANDPTS.onTool(); }
      if (window.viewer) window.viewer.render();
    });
    $('cuBandAdd').addEventListener('click', () => setBandTool('band_add'));
    $('cuBandRemove').addEventListener('click', () => setBandTool('band_remove'));
    $('cuBandClear').addEventListener('click', () => { const v = window.viewer; if (window.MD_BANDPTS && v) window.MD_BANDPTS.deleteKeyAt(v.currentFrame); });
    document.addEventListener('keydown', (e) => {   // Esc ends the band tool, Ctrl+Z undoes a band edit (as C5)
      if (!bandOwner()) return;
      const tag = (e.target && e.target.tagName) || '';
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;
      if (e.key === 'Escape' && C.bandTool) { C.bandTool = null; bandUi(); if (window.MD_BANDPTS) window.MD_BANDPTS.onTool(); }
      else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === 'z' || e.key === 'Z') && window.MD_BANDPTS && window.MD_BANDPTS.ready && C.bandTool) {
        e.preventDefault(); e.stopPropagation(); window.MD_BANDPTS.undo();
      } else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === 'z' || e.key === 'Z') && window.MD_BANDPTS && window.MD_BANDPTS.ready) {
        e.preventDefault(); e.stopPropagation(); window.MD_BANDPTS.undo();   // the band's undo: slider keys too (2026-11-15)
      }
    }, true);
    (function hook(tries) {   // the band lines inside the viewer's pan / zoom
      const v = window.viewer;
      if (!v || !Array.isArray(v.overlayHooks)) { if (tries < 40) setTimeout(() => hook(tries + 1), 300); return; }
      if (!v.overlayHooks.includes(drawLive)) v.overlayHooks.push(drawLive);   // under the band lines
      if (!v.overlayHooks.includes(drawModelPv)) v.overlayHooks.push(drawModelPv);   // the model picture over the composite
      if (!v.overlayHooks.includes(drawHoldout)) v.overlayHooks.push(drawHoldout);   // the kept objects' overlays
      if (!v.overlayHooks.includes(drawBand)) v.overlayHooks.push(drawBand);
    })(0);
    $('cuHoldout').addEventListener('change', async (e) => {
      $('cuSkipLost').disabled = !e.target.checked;
      if (C.info) C.info.settings.holdout = e.target.checked;
      renderKept(); save({ holdout: e.target.checked });
      try { await flush(); } catch (er) { /* toast from save */ }
      compositeChanged();
    });
    $('cuSkipLost').addEventListener('change', (e) => save({ holdout_skip_lost: e.target.checked }));
    $('cuKept').addEventListener('click', (e) => {
      const row = e.target.closest('[data-kept]'); if (!row) return;
      const n = row.dataset.kept;
      if (e.target.closest('[data-kept-eye]')) {
        if (C.ho.on.has(n)) C.ho.on.delete(n); else C.ho.on.add(n);
        renderKept(); if (window.viewer) window.viewer.render();
      } else if (e.target.closest('[data-kept-cut]')) keptToCut(n);
    });
    $('cuKept').addEventListener('change', (e) => {
      const cb = e.target.closest('[data-kept-on]'); const row = e.target.closest('[data-kept]');
      if (cb && row) keptToggle(row.dataset.kept, cb.checked);
    });
    $('cuFill').addEventListener('change', (e) => save({ fill: e.target.value }));
    $('cuInput').addEventListener('change', async (e) => {   // „Wejście” (2026-11-15): the anchors / preview read it too
      if (C.info) C.info.settings.input_plate = e.target.value;
      save({ input_plate: e.target.value }); await flush();
      if (window.MD_CU_ANCHORS) window.MD_CU_ANCHORS.reload();
      window.UI.toast(e.target.value ? `Wejście: ${e.target.value} – Inpaint liczy na tej wersji` : 'Wejście: Beauty (oryginał)', 3500);
    });
    $('cuGrain').addEventListener('change', (e) => { if (C.info) { C.info.settings.grain = e.target.value; renderGrain(C.info); } save({ grain: e.target.value }); });
    $('cuGrainStrength').addEventListener('input', (e) => { $('cuGrainStrengthV').textContent = `×${(+e.target.value).toFixed(2)}`; });
    $('cuGrainStrength').addEventListener('change', (e) => { if (C.info) C.info.settings.grain_strength = +e.target.value; save({ grain_strength: +e.target.value }); });
    $('cuModel').addEventListener('change', async (e) => { if (C.info) C.info.settings.model = e.target.value; render(); save({ model: e.target.value }); await flush(); if (window.MD_CU_ANCHORS) window.MD_CU_ANCHORS.reload(); });
    $('cuPromptMode').addEventListener('click', (e) => {
      const b = e.target.closest('[data-pm]'); if (!b || !C.info) return;
      C.info.settings.prompt_mode = b.dataset.pm; render(); save({ prompt_mode: b.dataset.pm });
    });
    $('cuPrompt').addEventListener('input', (e) => { if (C.info) C.info.settings.prompt = e.target.value; save({ prompt: e.target.value }); if (window.MD_CU_ANCHORS) window.MD_CU_ANCHORS.promptChanged(e.target.value); });
    $('cuModelInfo').addEventListener('click', (e) => {
      const b = e.target.closest('[data-cu-install]');
      if (b) { b.disabled = true; installModel().finally(() => { b.disabled = false; }); }
    });
    $('cuH3Prompt').addEventListener('input', (e) => window.MD_CLEANUP.setH3Prompt(e.target.value));
    $('cuVaceRes').addEventListener('change', (e) => { if (C.info) C.info.settings.vace_res = +e.target.value; save({ vace_res: +e.target.value }); });
    $('cuVaceNote').addEventListener('click', (e) => {
      const b = e.target.closest('#cuLightningInstall'); if (!b) return;
      b.disabled = true;
      installModel({ label: 'LoRA Lightning (Wan 2.2 Fun VACE)', install: 'vace_lightning', license: 'Apache-2.0',
        what: 'Przyspiesza Wan 2.2 Fun VACE ok. 5–10 razy: 4 kroki przy CFG 1 zamiast 20 przy CFG 3,5, kosztem trochę szczegółu',
        download: '2 × 1,23 GB (lightx2v 4-step T2V v1.1 high + low) z Comfy-Org/Wan_2.2_ComfyUI_Repackaged do models/loras' })
        .finally(() => { b.disabled = false; });
    });
    $('cuVaceSpeed').addEventListener('change', (e) => { if (C.info) C.info.settings.vace_speed = e.target.value; render(); save({ vace_speed: e.target.value }); });
    $('cuStrength').addEventListener('input', (e) => { const v = +e.target.value; if (C.info) C.info.settings.h3_strength = v; $('cuStrengthVal').textContent = v >= 1 ? '1,00 (dziura)' : v.toFixed(2).replace('.', ','); });
    $('cuStrength').addEventListener('change', (e) => save({ h3_strength: +e.target.value }));
    $('cuSeed').addEventListener('change', (e) => { if (C.info) C.info.settings.seed = +e.target.value || 0; save({ seed: +e.target.value || 0 }); });
    $('cuSeedRnd').addEventListener('click', () => { const sd = Math.floor(Math.random() * 2147483647); $('cuSeed').value = sd; if (C.info) C.info.settings.seed = sd; save({ seed: sd }); });
    $('cuPreview').addEventListener('click', () => previewMask());
    $('cuMaskCv').addEventListener('click', () => {   // the big preview like the anchors' (static/pop_view.js)
      const cv = $('cuMaskCv');
      if (window.MD_POPVIEW && cv.width) window.MD_POPVIEW.open({ title: `Maska klatki F${cv.dataset.frame || ''}`, src: cv.toDataURL('image/png'),
        note: 'Czerwień = łata (im mocniej, tym pełniej); przerywana linia = krawędź pasa, cienka = koniec wtopienia; jasnoniebieskie tło z niebieską ciągłą krawędzią = to, co narysuje model (pas generowania + rozrost silnika; także pod chronionymi obiektami)' });
    });
    $('cuRun').addEventListener('click', () => runCleanup());
    // „🧹 Usuń ten obiekt” shortcut: other scripts may call MD_CLEANUP.removeObject(name)
    apply();
  }
  window.MD_CLEANUP = {
    get mode() { return mode; }, get target() { return target(); }, setMode, reload: load, flush,
    shownLabel,   // the version on the plate „cleaned” for the viewer's HUD (app.js updateHudMode)
    get info() { return C.info; },
    setPrompt(v) { if (!C.info) return; C.info.settings.prompt = v; if (document.activeElement !== $('cuPrompt')) $('cuPrompt').value = v; save({ prompt: v }); },
    setH3Prompt(v) { if (!C.info) return; C.info.settings.h3_prompt = v; save({ h3_prompt: v }); },
    // how linked anchors are made (the anchor palette, engine/anchor_warp.py)
    setLink(patch) { if (!C.info) return; Object.assign(C.info.settings, patch); save(patch); },
    // the anchor palette (static/cleanup_anchors.js): its head's on / off select and the Qwen-Image-2.1 card
    async setAnchor(v) { if (!C.info) return; C.info.settings.anchor = v; render(); save({ anchor: v }); await flush(); if (window.MD_CU_ANCHORS) window.MD_CU_ANCHORS.reload(); },
    installModel: (m) => installModel(m), modelHtml: (m) => modelHtml(m), syncH3: () => syncH3(),
    get c1Shown() { return c1Shown(); },
    // static/bandpoints.js owner „c1”: the band of the cleaned thing
    bandOwner, bandTool: () => C.bandTool || null, bandPv, reloadBand,
    refresh: () => { if ($('modeSwitch')) apply(); },   // re-apply the mode (static/cleanup3d.js after its own init)
    async removeObject(name) {
      setMode('clean');
      await pick(name);   // one thing at a time: this object (its group when it is a member)
    },
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
