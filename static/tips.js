// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/tips.js - tooltips of the stage panels (Z24): what an option does + when to use it, in one place.
// Texts proposed in PLAN_OPUS_5_5.md (Z24) – the user corrects them here. Applied as native `title` to elements by id
// (and to <option>s by select id + value) whenever the panels change (MutationObserver), never overwriting a title
// that the code sets dynamically for a state (e.g. "Trwa: …" while a job runs).
(function () {
  const TIPS = {
    // bar
    objSelect: 'Obiekt, nad którym pracujesz; numer = kolejność głębi (0 = najbliżej kamery)',
    // stage 0
    st0Next: 'Przejdź do etapu 1 (aktywne, gdy plate’y są gotowe)',
    p0Manage: "Dodaj / podmień plate'y ujęcia (zapis w project.json, manifest się nie zmienia)",
    p0Aux: 'Plate do porównania z beauty: clear plate (tło) albo backlit (sylwetka)',
    p0Wipe: 'Położenie linii podziału: po lewej od linii prawa strona pary (plate albo wycięcie na tle), po prawej oglądany plate',
    p0Gain: 'Wzmocnienie różnicy – większe = widać mniejsze przesunięcia',
    p0Aligned: 'Pokaż plate pomocniczy po korekcie dopasowania (podgląd, nic nie zapisuje)',
    p0Measure: 'Zmierz, o ile plate pomocniczy jest przesunięty względem beauty (cały zakres – może trwać minuty)',
    p0Device: 'Gdzie liczyć wyrównanie: Auto (GPU w Pythonie ComfyUI, gdy działa i ComfyUI nic nie liczy; inaczej CPU), GPU (błąd, gdy niedostępne), CPU (wszystkie rdzenie). Wynik pokazuje, gdzie liczono.',
    p0CleanOpen: "Deflicker / odszumianie plate'u",
    // stage 0 – kontrola klatek (Z26)
    p0QaPlate: "Plate do kontroli klatek (✔ sprawdzony bez problemów / naprawiony, ⚠ czeka na decyzję albo zapis)",
    p0QaRun: "Sprawdź każdą klatkę plate'u: brak pliku / błąd odczytu / NaN, skok ekspozycji, skok obrazu (uszkodzona / przesunięta klatka), duplikat. 4K EXR ≈ 0,3 s/klatkę; nic nie zapisuje poza raportem",
    p0QaSource: "Plate używa naprawionej wersji _fix: sprawdź ponownie oryginał, żeby zmienić decyzje naprawy (potem zapis nadpisze _fix)",
    p0QaChart: 'Odchylenie każdej klatki od sąsiadów (1 = próg alarmu, linia przerywana): czerwone = oflagowane, pomarańczowe = ponad progiem, ale płynny trend / prawdziwy ruch kamery. Kliknij: skok do klatki',
    p0QaAccept: 'Zatwierdź proponowaną metodę dla wszystkich klatek bez decyzji (brak / skok obrazu → średnia albo przepływ, sam skok ekspozycji → wzmocnienie)',
    p0QaWrite: "Zapisz <plate>_fix obok oryginału: naprawione klatki w kodowaniu źródła, reszta = twarde dowiązania (bez kopiowania GB). Przed zapisem pokazuje rozmiar i czas; project.json wskaże _fix, manifest bez zmian; obiekty liczone z tego plate'u → do przeliczenia",
    p0Denoise: "Odszumianie czasowe plate'u do maskowania (project.json → mask_plate_denoise): potrzebne do precyzyjnych krawędzi (włosy, motion blur, ziarno EXR); zbędne przy miękkich / wtapianych maskach na czystym materiale 8-bit; kosztuje ~1,5 s/klatkę przy kadrze 1440 px. auto = wył. (modele dostają plate wybrany w kroku Plates bez dodatkowego odszumiania; wł. tylko z Twojego wyboru – plate odszumiony gdzie indziej nie jest ruszany)",
    p0CameraOpen: "Ruch kamery 2D: stabilizacja / match-move do Nuke'a",
    p0DepthOpen: 'Mapa głębi (DepthCrafter) do kolejności obiektów i ZDefocus',
    // stage 1 SAM3 (layout 2026-10-15: 1 SAM3 · 2 CoTracker · 3 MatAnyone · 4 ViTMatte · 5 Eksperymenty · 6 Eksport)
    st1Prompt: "Krótka nazwa obiektu po angielsku dla SAM 3.1, np. ‘felt puppet’ – używana przy ‘Wykryj kandydatów’",
    st1PartOf: 'Obiekt jest częścią innego (np. suknia lalki): osobna maska, eksport sumuje części do całości',
    st1Create: 'Dodaj nowy obiekt do ujęcia',
    st1PtsClear: 'Usuń wszystkie punkty tej klatki (usuwa też jej klucz z punktów; Ctrl+Z cofa)',
    st1Detect: 'SAM 3.1 szuka obiektu z promptu na całej klatce i pokazuje kandydatów do wyboru',
    st1All: 'Liczy tylko to, co nieaktualne po ostatnim przeliczeniu: odcinki ze zmienionymi kluczami (reszta z cache). Za pierwszym razem – całe ujęcie (kadry automatycznie, gdy trzeba + rdzeń SAM 3.1)',
    st3Rerun: 'Przelicz rdzeń SAM 3.1 tylko w przedziałach zmienionych przez klucze (reszta z cache)',
    st2Crops: 'Policz kwadratowe kadry wokół obiektu dla SAM 3.1 od nowa (zwykle robi to samo „▶▶ Przelicz nieaktualne”)',
    st3KeySave: 'Zapisz bieżące kreski / wielokąty jako klucz tej klatki (zwykle zapisuje się samo)',
    // stage 2 CoTracker – brush corrections (static/brush.js)
    b2Track: 'Poprawki przenoszone lokalnie punktami CoTracker3 (nadążają za ruchem i odkształceniem); wyłączone = sztywny kształt jadący za obiektem',
    b2EmptyKey: 'Pusty klucz: poprawki kończą się w tej klatce (obszar zanika od poprzedniego klucza do tej klatki)',
    b2DelKey: 'Usuń klucze poprawek tej klatki – obszar w tej klatce znów z propagacji [Delete]',
    b2Apply: 'Policz maskę 1+2 (rdzeń SAM 3.1 + poprawki) na całym ujęciu; widok Rdzeń pokaże wynik, ta sama maska idzie do MatAnyone2 i do eksportu „rdzeń (1+2)”',
    // stage 3 MatAnyone (edge settings shared with stage 5)
    st3Ma: 'MatAnyone2 na masce 1+2 bez stref ViTMatte → nowa wersja = baza etapu 4; najlepsze do naturalnej miękkiej krawędzi (ruch, nieostrość)',
    o4Out: 'Pas na zewnątrz: minimum px poza krawędzią maski 1+2, które liczy MatAnyone2 (przy rozmyciu automat daje więcej); w polu obok dowolna wartość',
    o4In: 'Wcięcie w rdzeń: minimum px w głąb maski 1+2 oddane MatAnyone2; dalej alfa = 1; w polu obok dowolna wartość',
    o4Smooth: 'Wygładzanie w czasie: mocne = mniej migotania, ale ryzyko smug przy szybkim ruchu',
    o4MaScale: 'Rozdzielczość, w której liczy MatAnyone2 (tryb krawędzi MatAnyone2): niższa = szybciej i mniej VRAM, ale bardziej miękkie / rozmyte krawędzie włosów; kontekst czasowy (pamięć klatek) bez zmian. auto = próba 9 klatek od 1.0 (kadr > 2048 px od 0.75, > 2600 px od 0.5), niżej gdy VRAM > 11 GB albo gdy ComfyUI przelewa pamięć karty do RAM komputera; przy długich ujęciach pełny przebieg może i tak przelać się do RAM i drastycznie zwolnić – wtedy wybierz 0.75 / 0.5. Zmiana = nowa wersja przy „Przelicz finalnie”',
    o4Lupa: 'Dodatkowe liczenie w powiększeniu dla cienkich części (rogi, czułki): auto / wył. / w ramce z ostatniego wielokąta',
    o4Backlit: 'Pozwól użyć backlitu jako źródła krawędzi (gdy jest wczytany)',
    st4Final: 'Dawne „Przelicz finalnie” – w etapie 5 zastąpione „▶ Przelicz z tej bazy” (paleta Eksperymenty)',
    // stage 4 ViTMatte – base + comparison (objects.js), zone panel (static/zones.js, plan step 5)
    st4BaseSel: 'Baza ViTMatte: MatAnyone2 (etap 3, miękka krawędź poza strefą) albo maska 1+2 (twarda maska SAM 3.1 + poprawki – etap dostępny zaraz po etapach 1 + 2). Każda baza ma własny wynik do porównania',
    st4CmpA: 'Wersja A porównania (domyślnie ViTMatte na masce 1+2)',
    st4CmpB: 'Wersja B porównania (domyślnie ViTMatte na MatAnyone2); klawisz B przełącza A ↔ B',
    z4New: 'Odznacz aktywną strefę: następna kreska (G / R / P) tworzy NOWĄ strefę z wybranym presetem',
    z4Prompt: 'Opis strefy: co wyciąć, czego pozbyć się z tła (np. „włosy, bez złotej linii”) – zapisany w strefie; reguły promptu i opis sceny (VLM) w kolejnych krokach',
    z4Engine: 'Silnik mattingu w strefie: MatAnyone2 (domyślnie, na kadrze strefy przy „Zastosuj”; podgląd = podpowiedź klucza), auto (dobór per fragment), clean (klucz różnicowy od clean plate’u), chroma (jednolite tło), knn (pstre, ostre tło)',
    z4Holdout: 'Maski obiektów z przodu (finalna alpha / rdzeń) wymuszają tło w strefie; bez nich – żółta flaga QA „strefa bez holdoutu”',
    z4CpAuto: 'Clean plate z ujęcia: mediana pikseli nieprzykrytych rdzeniami (+ LaMa / OpenCV na dziurach) → plate „cleanplate_auto” w etapie 0; ciężkie zadanie z paskiem i „Przerwij”',
    z4CpFile: 'Masz clear plate (tło bez obiektu): dodaj go w oknie plate’ów (zapis w project.json, manifest bez zmian)',
    z4CpNone: 'Pracuj bez clean plate’u: silnik clean niedostępny, zostają chroma / knn / MatAnyone2 (zapamiętane w przeglądarce)',
    z4Refresh: 'Policz podgląd strefy w tej klatce jeszcze raz (podgląd liczy się sam po każdej kresce i zmianie ustawień, debounce 600 ms)',
    z4Loupe: 'Lupa 3× przy kursorze na plate z nakładką alphy strefy – do oceny kosmyków [L]',
    z4ApplyShot: 'Przelicz strefy ViTMatte na całym ujęciu (cache per klatka) na wersji MatAnyone2 z etapu 3 → nowa wersja z QA; etap 4 zaliczony. Postęp i „Przerwij” na pasku zadań',
    z4KeyDel: 'Usuń klucz aktywnej strefy w tej klatce (PPM na rombie robi to samo); obszar w tej klatce znów z propagacji. Ostatniego klucza nie da się usunąć',
    z4Mode: 'Dodaj albo usuń piksele w narysowanym obszarze',
    z4Param: 'Próg jasności (0–1) albo odległości koloru (ΔE)',
    z4Soft: 'Miękkie przejście zamiast ostrego cięcia na progu',
    z4Softness: 'Szerokość miękkiego przejścia',
    z4Static: 'Strefa stoi w miejscu (nie jedzie za obiektem) – np. element dekoracji',
    z4Preview: 'Pokaż efekt strefy w tej klatce (nic nie zapisuje)',
    z4Apply: "Zapisz strefę progową dla całego ujęcia (śledzi obiekt); zadziała przy ‘Przelicz finalnie’",
    z4Pick: 'Pipeta: kliknij na obrazie kolor, który strefa ma zaznaczyć (mediana 5×5 px plate’u do maskowania)',
    // stage 6 Eksport
    e6Source: 'Źródło maski: rdzeń (1+2) – zaraz po etapie 1; MatAnyone2 / ViTMatte – wynik etapu 3 / 4; wersja z listy (etap 5); aktywna – jak dotąd',
    e6Run: 'Wersja wybranego obiektu do eksportu „Ten obiekt” / „Całość” (lista wersji z etapu 5)',
    e5Dir: 'Katalog eksportu (project.json → export_dir); zmiana przyciskiem „📁 Wybierz…”, „domyślny” przywraca <work>/matte_desk/export',
    e5Multi: 'Jeden plik EXR na klatkę z kanałami wszystkich obiektów',
    e5Format: 'Format masek (project.json → export_format): auto = PNG 8-bit dla źródeł 8-bit (MP4 / PNG / JPG), EXR dla EXR; multi-EXR tylko przy EXR',
    e5Core: 'Dodaj też rdzeń SAM 3.1 (twarda maska) do eksportu',
    e5Depth: 'Dodaj kanał głębi',
    st5Export: 'Eksport masek wszystkich obiektów z wybranego źródła (z przodu odejmowane od tych z tyłu; obiekt bez wyniku z tego źródła – z niższego etapu, ostrzeżenie)',
    st5LibAdd: 'Utwórz obiekt z zapisanego w bibliotece (prompt + ustawienia)',
    st5QRun: 'Policz kolejkę zadań (np. na noc)',
    st5RevFile: "Wczytaj zaznaczenia z ‘Przeglądu klatek’ jako klucze",
  };
  const OPTION_TIPS = {
    st4Edge: {
      auto: 'Wybiera sam (backlit, gdy obiekt zasłania światło na X2; inaczej MatAnyone2)',
      backlit: "Krawędź z plate'u backlit – najdokładniejsza, gdy obiekt jest na nim",
      matanyone: 'Miękka krawędź z modelu (włosy, futro, rozmycie)',
      core: 'Twarda krawędź z rdzenia SAM 3.1 – obiekty sztywne',
      ltx: 'Półprzezroczyste (wymaga workflowu)',
    },
    z4Engine: {
      matanyone: 'MatAnyone2 na kadrze strefy z kluczem = rdzeń ∪ obszar (rozmyte / ruchome włosy); liczy się przy „Zastosuj”, podgląd = podpowiedź klucza',
      auto: 'Mapa wag per fragment: jednolite tło → chroma, pstre → knn, clean gdzie clean plate pewny',
      clean: 'Klucz różnicowy względem clean plate’u (linia w tle = 0 bez kreski −); wymaga clean plate’u',
      chroma: 'Odległość koloru od tła z pierścienia wokół strefy – lokalnie jednolite tło',
      knn: 'pymatting KNN – pstre, ostre tło; 2–5 s / klatkę na CPU',
      vitmatte: 'ViTMatte-base per klatka na kadrze strefy (eksperyment 2026-10-13, porównanie z MatAnyone2; wygładzanie w czasie i wykończenie: sekcje w karcie strefy); wymaga GPU wolnego od Comfy',
    },
    z4Crit: {
      dark: 'Tylko ciemniejsze niż próg',
      bright: 'Tylko jaśniejsze niż próg',
      color: 'Tylko podobne do wybranego koloru (pipeta)',
      all: 'Cały obszar',
    },
  };
  const NEXT_TIP = 'Przejdź do następnego etapu (aktywne, gdy ten etap jest gotowy)';

  // the page may be translated at run time (static/i18n.js): compare and set the translated text, or the two would fight
  const T = (s) => (window.mdT ? window.mdT(s) : s);

  function applyTips(root) {
    const scope = root || document;
    for (const [id, text] of Object.entries(TIPS)) {
      const el = document.getElementById(id);
      if (el && !el.dataset.tTitle && el.title !== T(text)) el.title = T(text);
    }
    for (const [id, opts] of Object.entries(OPTION_TIPS)) {
      const sel = document.getElementById(id);
      if (!sel) continue;
      for (const o of sel.options) if (opts[o.value]) o.title = o.title && o.title.length > T(opts[o.value]).length ? o.title : T(opts[o.value]);
    }
    scope.querySelectorAll('[id^="st"][id$="Next"]').forEach((b) => { if (!b.title) b.title = T(NEXT_TIP); });
  }
  window.MD_TIPS = { TIPS, OPTION_TIPS, applyTips };
  let pending = false;
  const obs = new MutationObserver(() => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; applyTips(); });
  });
  function start() {
    applyTips();
    const sp = document.getElementById('stagePanel') || document.body;
    obs.observe(sp, { childList: true, subtree: true });
    const bar = document.getElementById('stageBar');
    if (bar) obs.observe(bar, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
