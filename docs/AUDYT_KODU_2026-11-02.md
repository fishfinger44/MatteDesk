# Audyt kodu MatteDesk – 2026-11-02 (v0.50, main @ 6d063ee)

Zakres: `app.py` (7531 linii, 193 route'y), `engine/` (95 modułów, ~40 k linii), `static/` (16,5 k linii JS/CSS),
`tests/` (169 plików, 1005 testów), `tools/`, konfiguracja, higiena repo. Tylko odczyt – nic nie zmieniono.
Metryki: ruff 0.17 (E/F + B/SIM/ARG), ast (długość funkcji), diff plików podejrzanych o duplikację, grep martwego kodu.
Oznaczenia: **BUG** = realny defekt, **NIESP** = niespójność (to samo robione na kilka sposobów), **PORZ** = porządek.
C = potwierdzone w kodzie przeze mnie / agenta, P = prawdopodobne (nie prześledzone do końca).

## 0. Ocena ogólna

Kod jest **spójny tam, gdzie były świadome decyzje** (pas = `bandsys`/`objband`, GPU = `gpu_manager`, atomowy zapis
`ObjectStore`/`project.save`, dekoratory `long_job`/`object_edit`, bramka modeli, testy statyczne JS). Rozjeżdża się
**warstwę niżej, w kodzie, który narastał przyrostowo**: app.py ma 3–4 sposoby na te same rzeczy (klatki, defaults,
run.json, proxy, eksport), pipeline.py to god-object z 10 odpowiedzialnościami, 5 workerów GPU ma skopiowany protokół,
frontend ma 4 rodzaje modali / 3 toasty / 4 style wywołań API. Realnych bugów jest kilkanaście, większość punktowa.

## 1. Bugi do poprawy w pierwszej kolejności

**Stan 2026-11-02 (krok 1 z planu, 7 commitów f77aa15..55e6f81):** zrobione #2 (`subtract_front` = max), #3 (`StateFileError`, 409),
#7 (CORS), #8 (`zones.zone_file`, `_plate_name_or_400`), #12 (GPU → `real`, leniwy worker), #13 (`??`), #14 (jeden WebSocket, zdarzenie `md-ws`),
#16 (klawisze przeglądarki przez okna zadań), #19 (skróty bez Ctrl/Meta). Testy: `test_export_subtract_front`, `test_state_files_corrupt`,
`test_api_hardening`, 4 strażniki w `test_static_js`. Sprawdzone w UI na ujęciu `_fotage_testy` (md-ws → status etapu, Ctrl+P/V bez skutku).
**Krok 2** (`f955cb2`): #4 #21 – `engine/jsonio.py` (`write_json`, `read_state`, `StateFileError` przeniesiony; bajty plików bez zmian), uszkodzona strefa = 409 z nazwą pliku w każdym czytniku.
**Krok 3** (`1a0f20f`, `396b1e6`, `df76323`): #5 – klucz klatki strefy ma treść wymuszonego tła i holdoutu (`mask_token`; istniejący cache stref przeliczy się raz); #6 – dziury w `core_fix/` raportowane (`core_fix_fallback` w outputs.json + ostrzeżenie API + log), piksele bez zmian; brak `core_fix/` = bez flagi; `Dres` z `_fotage_testy` nie ma `core_fix/`, jego ostrzeżenia to szum `imread` (usunięty).
**Krok 5** (`f6ec5ba`): `export_mattes` → `_export_by_source`, `EXPORT_*` w `engine/export.py`; `test_export_merge` = zamrożona stara gałąź kontra nowa, plik w plik. UI na `_fotage_testy`: grupa WomanFace_grp (rdzeń, F1–50) i Hair (eksport zwykły) OK.
**Krok 6** (`eda86d2`, `2a51b4d`): #11 – `engine/worker_proto.py` (`serve`, `WorkerProcess`, `check_worker`), 5 workerów, protokół na drucie bez zmian (`test_wire_*` przechodzą też na starych pętlach), „Przerwij” zatrzymuje forward ViTMatte / BiRefNet / Sapiens2 w ≤ 0,25 s; realne GPU sprawdzone tylko testami `not real` (workery FAKE).
**Krok 7** (`6998764`, `cc8fa9a`, `a1897a5`): #18 – `static/ui.js` (`UI.api/read/toast/dialog/form`), zastąpione `prompt()`, `.md-modal-bg`, `#jobErrModal`, `#jobToast`, surowe `(await fetch()).json()`, kolory równe tokenom, `--footer-height` raz; `loadStages` bez duplikatu, `Promise.all` w `loadObject`, `md-compare-ref`; strażniki w `test_static_js`. Zostały: `.modal-backdrop` (Doktor / kreator), `#comfyLost` (trwały pasek, nie toast), wywołania `fetch` + `readApiResponse` w app.js (sprawdzają `ok`). UI: okna, klawisze, toast, zmiana obiektu, porównanie OK.
**Krok 9** (`d7ce58b`..`a78e40e`): `experiments/` (25 skryptów), `docs/archive/` (7 planów), `local_results/` poza gitem (pliki na stacji), pytest do requirements-dev, `version` z pyproject, ruff F401 / F811 / RUF100 poza app.py, tests, experiments. Zostawione świadomie: `websockets` (audyt się mylił: uvicorn potrzebuje go do `/ws`); kod z testami (`qa_gray_rows`, `stmap_for_frame`, `fit_rbf_deformation_field`, `run_stage3_detail_mc`, `denoise_temporal_mc`, tryb krawędzi `ltx` w `refine`) – usunięcie wymaga decyzji o testach; martwy kod app.py dla kroku 8; `INSTALL_2026-*.md` czekają na zgodę użytkownika (Doktor generuje nowe w katalogu repo).
**Krok 4** (`214b362`): #1 – `ObjectPipeline.from_project(proj, client, defaults, display, ltx_workflow, work_frames)` w engine; `app._pipeline_for(proj, work_frames)` = jedyny budowniczy w app.py (UI: obszar roboczy, kolejka: całe ujęcie); kolejka nocna liczy z backlitem / mapami gain / clean plate'em / shutterem jak UI i respektuje bramkę modelu SAM 3.1; `defaults.json` czytane tylko przez `load_defaults` (wyłącznik mplate: `engine.mplate.global_enabled`). Test kontraktowy `tests/test_pipeline_from_project.py`.
**Krok 8** (`3f9e5d0`..): `api/` – plate_qa (5 route'ów, dotąd rejestrowane PO `if __name__` – `python app.py` startował uvicorn bez nich), groups (10), depth (3), camera (3), plate_clean (4), plate_align (4) = 29 route'ów, app.py 7472 → ~6260 linii; `make(*, zależności) -> (router, exports)`, include w miejscu sekcji (kolejność route'ów bez zmian – migawka `tests/data/app_routes_2026_11_02.json`, `tests/test_app_routers.py`). Zostawione w app.py świadomie: Final Touch (14 zależności, helpery `_ft_*` podmieniane w testach i używane przez eksport / MergeMatte), „Zgłoś problem” (dzieli `ComfyFlavorRequest` z konfiguracją Comfy), reszta obiektów / stref / pasa (gęsto powiązane helpery) – kandydaci na kolejne sesje tym samym wzorcem.
**#10** (commit „Final Touch edits under object_edit”, 2026-11-03): cztery edycje Final Touch mają `@object_edit` (409 w trakcie zadania na obiekcie / eksportu), `req.model_dump()`; test `test_final_touch_edits_refused_while_the_object_is_computed`.
**#22** (commit „assemble_full composes every crop segment”, 2026-11-03): ≥ 3 odcinki na klatce składane kolejno (przenikanie z dotychczasową mieszanką), 1–2 odcinki bit w bit jak wcześniej; `tests/test_assemble_full.py`.
**#17** (commit „Outline zone retries after a passing refusal”, 2026-11-03): 409 / sieć = `Z.autoWait` – ponowna próba po końcu zadania (`matte-job-finished`), przy wejściu w etap, przy przeładowaniu obiektu; inny błąd = `autoFailed` jak dotąd; nic nie ponawia z samego renderu. `reset()` czyści `loadedObj`, żeby pusta lista w trakcie przeładowania nie tworzyła strefy. UI na `_fotage_testy` (409 podstawione w `fetch`): 1 POST, brak pętli, po końcu zadania strefa powstaje; prawdziwe 409 „Objects in front” też czeka. Harness jsdom `tests/js/zones_harness.mjs` nieaktualny od 2026-10 (pada w l. 206, na stacji pominięty) – osobny temat.
**#9** (commit „Plate clean and camera work off the event loop”, 2026-11-03): `measure_plate_clean` = zwykły `def` (wątek long_job), `run_plate_clean` / `track_camera_motion` = cienki `async` + całość w `run_in_executor`; `JOBS.check()` per klatka w pętlach dekodowania i po metrykach pomiaru („Przerwij” dotąd odpowiadał, ale pomiar liczył dalej i kończył się jako „gotowe”; samych metryk nie da się przerwać w połowie, na 16 klatkach z plate'u czeka się ok. 2 s). `tests/test_event_loop_free.py`: `/api/jobs` < 0,5 s w trakcie (czerwone przed: 1,55 s). UI na `_fotage_testy`, pomiar beauty 16 klatek: `/api/jobs` mediana ~12 ms, max 444 ms po rozgrzaniu (844 ms w pierwszym przebiegu po starcie serwera), `cancel` 7 ms.
**#15** (commit „One project open applies once”, 2026-11-03): każde otwarcie ma `open_id` (odpowiedź POST, zdarzenie WS, `/api/project/current`), `applyOpened` pomija drugie wykonanie w dowolnej kolejności, bezpośredni `loadCurrentProject()` (stage0.js po zmianie plate'ów) nadal stosuje; `tests/test_project_open_id.py`, `test_project_opened_applied_once_2026_11_03`. UI na `_fotage_testy`: otwarcie, zmiana projektu i powrót = po 1 `applyProject` / `md-project` (wcześniej 2), plate'y, paleta i klatka bez zmian.
**#20** (commit „Caches and object.json updates safe next to a job”, 2026-11-03): `engine/lru.py` `LockedLRU` zamiast własnych kopii w `pipeline._stab_gray`, `groups._CACHE`, `app._LINES_CACHE`, `app._FT_BASE_CACHE`, `STATE["display_cache"]` (zmiana projektu nadal odcina stary cache: spóźniony wątek pisze do starego obiektu); `ObjectStore.lock(name)` = RLock na folder obiektu, działa między instancjami `ObjectStore`, pod nim `update()` i `set_object_life`; `tests/test_cache_threads.py`. Wartości w cache bez zmian (to tylko cache); limity jak dotąd, `display_cache` i `_stab_gray` zamiast FIFO / czyszczenia usuwają najstarsze użycie.
**Harness jsdom** (commit „Revive the jsdom zone panel harness”, 2026-11-03): pomijany, bo na stacji nie było jsdom (Node jest); po instalacji padał od pierwszego kroku, bo opisywał rysowanie stref ukryte od 2026-10-21. Teraz `tests/js/package.json` + lock (`npm ci`), przepisana część stref (automatyczna strefa „cały obrys”, 409 → ponowienie, paleta, „Zastosuj” / obiekty z przodu, jedna strefa na obiekt), `brush.js` zgodny z warstwami z 2026-10-30. Asercje ukrytego rysowania usunięte (są w historii gita) – ten kod nie ma testu; trzy mutacje `zones.js` / `brush.js` harness wykrywa.
**Spoza tabeli** (commit „Project switch asks for no object of the old project”, 2026-11-03): po zmianie projektu prefetch pytał nowy projekt o obiekty starego (20 × 500 w logu), bo `applyProject` szedł do klatki przed `md-project`. Teraz najpierw `md-project`, lista obiektów pusta do `refreshList`; brak obiektu w API = 404 (`ObjectNotFound`), bez tracebacku. UI na `_fotage_testy`: 0 × 500, 0 × 404, widok bez zmian. Testy `test_first_read_migration_survives_concurrent_readers` i `test_zone_panel_flow` padły raz w przebiegu testmon (353 passed, 2 failed), osobno 30/30 i 8/8 + ponowny testmon zielone: niestabilne pod obciążeniem, przyczyny nie ustalono.
Wszystkie punkty sekcji 1 zamknięte.

| # | Gdzie | Co | Skutek |
|---|---|---|---|
| 1 | `app.py` `_queue_worker` (~6690) vs `_pipeline()` (2701) | Kolejka nocna buduje `ObjectPipeline` **bez** `backlit_pattern`, `gain_pattern`, `cleanplate_pattern`, `shutter_deg`, `work_frames`, bez `_require_stage` | „Policz wszystko” na noc daje inny wynik niż „Przelicz” z UI (krawędź bez backlitu, bez map gain). C |
| 2 | `engine/export.py:147` `subtract_front` | Odejmuje fronty **sekwencyjnie** (`back − Σ front`); docstring modułu i spec: `back − max(front)`. Kontekst eksportu zawiera grupę ORAZ członków | Obiekt za grupą traci alfę 2× na miękkich krawędziach → ciemna obwódka w eksporcie. C |
| 3 | `engine/project.py:~84`, `engine/outputs.py:19` | Uszkodzony `project.json` / `outputs.json` → `data = {}` → następny `save()` **nadpisuje** pustym stanem | Utrata `plates_added`, `export_dir`, `plate_changed`, listy outputów po padzie zasilania. C |
| 4 | `engine/refine.py:~817` + `pipeline.py` (run.json, core.json, core_fix.json, edge.json, qa.json), `zones.py` (`z_##.json`), `groups.py` | `write_text` bez tmp+replace, potem kilka read-modify-write | Ucięty JSON przy padzie = wersja znika z paska / 500 w eksporcie. Kontrast: `ObjectStore._write` robi to dobrze. C |
| 5 | `engine/zones.py` (hash klatki strefy ViTMatte / MatAnyone2) | Klucz cache nie zawiera **holdoutu obiektów z przodu** (`hm`), a on wchodzi do trimapy; `bgm.sum()` zamiast treści | Nowa wersja frontu → „Zastosuj” zwraca starą alfę z cache. C |
| 6 | `engine/export.py:41` `core_fix_loader` | Brak `core_fix/c_####.png` → cichy fallback na surowy rdzeń SAM 3.1 (core_fix bywa tylko dla obszaru pracy) | Eksport „rdzeń 1+2” miesza klatki z poprawkami i bez, bez ostrzeżenia. C |
| 7 | `app.py:128` CORS | `allow_origins=["*"]` + `allow_credentials=True` | Dowolna strona w przeglądarce może wołać `127.0.0.1:8765/api/...` (DELETE obiektu, instalacje). C |
| 8 | `app.py:5394` `delete_zone` (+ `save_zone_ep`, `plate` w `_proxy_path`) | `zone_id` z URL prosto do ścieżki + `unlink()`, bez reguły nazwy jak `ObjectStore._NAME` | Na Windows `..\..\x` w path-paramie przechodzi przez `[^/]+`. C (ścieżka) / P (wykorzystanie) |
| 9 | `app.py` `measure_plate_clean`, `track_camera_motion`, `run_plate_clean` | Dekodowanie EXR / zapis N EXR w `async def` bez executora (ruff widzi tylko `open()` 1769/2114) | Na kilka–kilkadziesiąt sekund nie działa `/api/jobs`, „Przerwij”, WS. C |
| 10 | `app.py` Final Touch `put_final_touch_base/finish/frames`, `delete_final_touch_frame` | Jedyne edycje obiektu bez `@object_edit`; `req.dict()` zamiast `model_dump()` | Edycja FT w trakcie eksportu ze źródła `final_touch` miesza klatki. C/P |
| 11 | `engine/vitmatte.py:258` `_request` | Brak `cancel_check` (CoTracker3 i align sondują co 1 s / 0,25 s) | „Przerwij” nie przerywa forwardu ViTMatte / BiRefNet / Sapiens2 – czeka do `timeout_s` 600 s. C |
| 12 | `tests/test_align_gpu.py`, `test_motion_blur.py:90` | Testy GPU przez `skipif(not _gpu_ready())`, nie `real`; `_gpu_ready()` odpala worker CUDA **przy kolekcji** | `pytest -m "not real"` na stacji koliduje z Comfy / realnym liczeniem. C |
| 13 | `static/app.js:925` | `m.first \|\| 1`, `m.last \|\| 100` | Ujęcie od klatki 0 (kreator pozwala) → klatka 0 nieosiągalna w UI. Potrzebne `??`. C |
| 14 | `static/objects.js:4469`, `static/depth.js:98` | Drugi i trzeci „goły” `new WebSocket(ws://…)` bez reconnectu obok `initWebSocket` w app.js | Po restarcie serwera `object_progress` przestaje dochodzić do odświeżenia strony. C |
| 15 | `static/app.js` `openOrCreateProject` + WS `project_opened` | `applyProject` wykonywane 2× przy otwarciu projektu | 2× czyszczenie cache klatek, 2× `refreshList`/`loadObject`, 2× refresh stage0. C |
| 16 | `static/jobs.js:170,194,201` | Modal zadania robi `preventDefault` na każdy klawisz poza Esc | Przy 20-min liczeniu nie działa F5, F12, Ctrl+L, Ctrl+Z suwaków. C |
| 17 | `static/zones.js:~741` `ensureOutlineZone` | POST `/zones/new` z funkcji renderującej; po 1 błędzie (np. 409 w trakcie zadania) `autoFailed` na stałe | Pas „cały obrys” nie powstanie sam do przeładowania strony. C |
| 18 | `static/app.js:~2447`, Doktor | `innerHTML` z `p.name` / `data.detail` / `it.message` bez `esc()` | Nazwa plate'u z `<` rozwala kartę. C |
| 19 | `static/objects.js:~4313` | Skróty `V G K L P N Enter` bez `!e.ctrlKey` (R/E/O/M mają) | Ctrl+P włącza wielokąt, Ctrl+V zmienia tło compu. C |
| 20 | `engine/objects.py:89` `update()`, `pipeline._stab_gray`, `groups._CACHE`, `app` `_LINES_CACHE`/`_FT_BASE_CACHE`/`display_cache` | get+write / `pop(next(iter))` bez locka, a endpointy sync lecą w puli wątków równolegle z zadaniem | Zgubiony zapis suwaka po końcu liczenia, sporadyczny `KeyError` → 500. P |
| 21 | `engine/zones.py:100` vs `engine/stages.py:104` | `load_zones` pada na złym JSON strefy, `zone_dicts` go pomija | Pasek mówi „aktualne”, każdy przebieg kończy się wyjątkiem. C |
| 22 | `engine/refine.py:~1441` `assemble_full` | ≥ 3 odcinki kadru na klatce → brane 2 pierwsze (`[:2]`) | Trzeci odcinek `manual` cicho pomijany. P |

## 2. Niespójności (to samo na kilka sposobów)

**app.py**
- Klatki ujęcia: `_shot_frames()` vs ręczne `range(first, last+1)` (×2) vs `get("last", 100)` vs `get("last", 1)`; `apply_edge_fix` liczy na `_shot_frames`, `finalize`/`apply_zone` na `_work_frames`.
- `defaults.json`: 4 niezależne `json.loads` w app.py + własne w `pipeline.py:3319`, mimo `engine.project.load_defaults()`; `_pipeline()` czyta je i buduje nowy `ObjectPipeline` + `ComfyClient` przy **każdym** wywołaniu (33 miejsca, też per klatka podglądu).
- `run.json` aktywnej wersji czytane ręcznie w 8 funkcjach; `stages._run_json` (prywatna) importowana w `_group_payload`.
- Eksport: `export_mattes` vs `_export_by_source` – ~60 zduplikowanych linii z drobnymi różnicami (`multichannel`, teksty ostrzeżeń, walidacja `scope`).
- Proxy na żądanie: `_proxy_path`, `_load_proxy_display` (bez `plate="bg"`), `_full_display_u8` – trzy implementacje.
- Kształt błędu: `detail` str / `{code,detail}` / `{detail,missing_front}` / `{message,missing}` (`group_frame`) / 200 `{"success": False}` (`plate_fix_run`). Kody: brak wersji 404 vs 400; zła nazwa obiektu → 500 zamiast 400.
- `update_project_plate` = `@long_job`, `delete_project_plate` = `@when_idle`, `set_work_area`/`set_export_dir` – mieszanka.
- 7 lokalnych `on_progress` kopiujących `_ws_progress()`; różne typy zdarzeń WS i sygnatury `JOBS.progress`.
- 299 lokalnych `from engine.x import` bez przymusu cyklu (nic w engine nie importuje `app`); `MissingFront` ×8, `mark_dependents` ×6.
- Teksty EN w UI: „No project loaded” ×3, „Proxy frame … not found”, „Choose a file”, etykiety w `get_project_plates`.
- `api_deflicker.make_router(globals())` vs `merge_matte_api.make_router(jawne callbacki)`.
- `EXPORT_SOURCES/LABEL/FALLBACK`, `BAND_STAGES`, `EDGE_MODES` – reguły domenowe w app.py.

**engine**
- Pięć czytników maski PNG o innym zachowaniu (resize / cache / fallback / `None > 127` → TypeError).
- Trzy definicje „wcięcia strefy” (`objband._own_slider_values` = domyślne 2, `canon_sliders` = 0, `zones.outline_in_px` = int).
- Strefa „cały obrys” bez punktów: pas morfologią (`dilate/erode`), etap 3: `band_area(sdf)` – ±0,5 px różnicy; `test_band_unify` pilnuje zgodności z dawnym kodem, nie między etapami.
- `core_fix_sig` haszuje cały JSON strefy (z `prompt`, `prompt_history`) → zmiana opisu unieważnia maskę 1+2 i etapy 3–7; `zone_signature` te pola wyklucza.
- Staleness z 3 źródeł: mtime, sygnatury, zegar ścienny (`plate_changed.at`) – kopia projektu = wszystko „nieaktualne”.
- Stałe zdublowane: `fg_erode_px` 2 ×3 obok `BAND_DEFAULTS.in_px`; `OUTLINE_OUT_PX` obok `BAND_DEFAULTS.out_px`; `last` domyślnie 1 vs 100.
- Nazewnictwo: maska 1+2 = `core_fix/` = `CORE_RUN="core"` = `edge_mode:"core_fix"` (run.json) ale `"core"` (outputs.json); etap 6 „final”/„experiment”/„Eksperymenty”; CLAUDE.md „8 Eksport”, `STEPS` ma 9.
- `refine.write_run_json` importuje `comfy_client` (root) i `engine.jobs` – silnik zależy od warstwy app.
- `engine/queue.py` przesłania stdlib `queue` (workery przez to wymagają `-I`).
- Workery GPU: pętla `_serve` ×5, launcher ×4 (`_request` bez cancel, `_CoTrackerWorker` 1 s, `GpuAligner` z restartem, `_one_shot_run`); `close()` 10/5/10 s; 3 kopie `--check` z własnym cache; 3 kopie `get_runner/close_runners/atexit`; `timings` tylko w ViTMatte.
- Klient Comfy: `ComfyClient` + ≥ 8 `urllib.urlopen` na boku (doctor ×4, `align_gpu.comfy_busy` dubluje `get_queue`, comfy_launch, comfy_portable, installer); `align_gpu.py:76` fallback portu 8188 vs `DEFAULTS` 8000.
- Konfiguracja: `launch_app.py` czyta `config.local.json` z pominięciem warstw; env `MD_*` rozsiane po 6 modułach; ~12 kluczy czytanych w kodzie nieopisanych nigdzie (`on_gpu_error`, `align_workers`, `cache_dir`, `comfy_git`…).

**static**
- 4 systemy modali (`.modal-backdrop`, `uiDialog` inline, `.md-modal-bg`, `#jobModal`/`#jobErrModal`/`#comfyLost`/report), 3 toasty (`#mdToast`, `#jobToast`, `#comfyLost`), 4 style API (`MD.api`, `readApiResponse`, ~37 surowych `fetch().json()` bez `res.ok`, `postJSON`).
- `window.prompt` w „Z biblioteki” mimo własnych dialogów; `window.alert` nadpisany po wrapperze i18n.
- Kolory literalne: 170 hex w CSS poza `:root` (w tym równe zmiennym), ~190 w JS; `--footer-height` zdefiniowane 2×; override'y tego samego selektora w odległych miejscach, 10× `!important`.
- Id etapów z historycznymi numerami (`st4Final` = dzisiejszy etap 6, `st5Export` = 9, `lx6*` = 7) – `status()` buduje id z `S.stage`.
- `loadStages()` ma zduplikowaną linię renderów (wszystko 2×), `run()` woła `loadStages` w `finally` po każdym kliknięciu, `loadObject` = 9 szeregowych await + drugi GET `/api/objects`.
- Martwe: `selGroup = () => null` i gałęzie `whole*`, `groupOf`; `.dark-theme` bez reguł.
- `mm_gpu.js` / `ft_gpu.js` to NIE duplikaty (0 wspólnych linii) – wspólne tylko ~40 linii boilerplate WebGL2.

## 3. Porządek (bez ryzyka dla kodu)

- Martwy kod engine: `pipeline.qa_gray_rows`, `analysis_fix_why` (zawsze `""`), gałęzie `part_of`/`members` w pipeline (nieosiągalne po grupach), `refine._ltx_segment/ltx_batches/ltx_side/region_limit` (tryb krawędzi `ltx` z 2026-10-16), `align.stmap_for_frame`, `fit_rbf_deformation_field`, `clean.run_stage3_detail_mc`, `denoise_temporal_mc`; `_tick` ×2; `apply_zones` kopiuje ciało `_zone_tracker()`.
- Martwe w app.py: `_analysis_target`, `rerun_from(frame)` ignoruje `frame`, `asked = name`, `EDGE_MODES` z `"ltx"` tylko po to, by odrzucić, atrybuty na funkcji `_plate_display.elementwise`.
- ruff: F401 ×44, RUF100 ×61, F811 ×3 (`subprocess`, `shutil` importowane ponownie), F841 ×6, B008 ×4 (`Request()` w defaultach), `BandSettings` w adnotacjach bez importu (`pipeline.py:2478`), S110 ×26 (`except: pass` – 6 w doctor, 5 w ingest).
- Testy: twarde `D:/MD_REAL_PROJECT` w 8 testach mimo `MATTE_DESK_REAL_ROOT`; marker `real` nadawany przez grep tekstu w źródle testu (krucha reguła, 2 testy łapią się przez docstring); fixture `proj` ×8, `client` ×4; import test→test (`from test_zone_matte import` ×43); mock `system_stats` ręcznie ×8 mimo `mock_comfy.py`; `test_e3:158` skip „wyrzucony”, `test_e6:203` xfail non-strict; `test_exp_sam3_multi` testuje eksperyment wykluczony z export_public; `run_trash.py` bez testów; 16 modułów po 1 pliku testów (w tym `gpu_manager`).
- tools: 24 eksperymenty (`exp_*`, `bench_*`, `diag_e3`, `ab_zone_smooth`, `ltx_probe`) obok 12 narzędzi; `exp_z21_realtest` importuje `app`; `exp_corestab_guard_sweep` zależy od cwd i ma twarde `F:/claude_code/_fotage_testy`; 2 z twardym `D:/MD_REAL_PROJECT`. Propozycja: `experiments/` + jeden wzorzec w `export_public.py`.
- Repo: `local_results/` **śledzone** (497 plików, 23 MB, archiwum Gemini); 13 × `INSTALL_2026-*.md` (ignorowane, 92 KB); `build/`+`dist/` 1,3 GB na dysku (ignorowane, OK); `pyproject.toml` `version = "0.1.0"` vs `VERSION` 0.50 (bump i test go nie dotykają); `requirements.txt` nic nie przypięte, `websockets` nieużywane, `pytest` w runtime; `comfy_client.py`/`api_deflicker.py` w root zamiast `engine/`; 9 planów .md w root + 14 w docs, część nieodwołana (NIGHT_PLAN, CLOUD_S3, PLAN_TRANSFER…); PROGRESS.md 3719 linii; preset `"sh010"` w `final_touch.py` / `merge_matte_nk.py` i 4096×2731 w `depth.py` mimo zasady „zero SH010 w kodzie”.

## 4. Propozycja kolejności prac (wartość / koszt)

1. **Bugi punktowe, osobne małe commity** (po pół godziny każdy): #2 `subtract_front` → `clip(back − max(front))` + test kontraktowy; #3 uszkodzony JSON → wyjątek z nazwą pliku zamiast `{}`; #7 CORS do `127.0.0.1:8765`/`localhost:8765`; #8 reguła nazwy (`_NAME`) dla `zone_id`/`plate`; #13 `??`; #14 WS tylko w app.js (+ asercja w `test_static_js`); #16 przepuszczać Ctrl/Meta/F-klawisze; #19 `!e.ctrlKey`; #12 testy GPU → `real`, `_gpu_ready` leniwie.
2. **Jeden atomowy writer JSON** (`engine/jsonio.py` z tmp+replace, jak `ObjectStore._write`) użyty w `write_run_json`, `save_zone`, core/core_fix/edge/qa.json (#4) + jedna reguła dla złego pliku strefy (#21).
3. **Holdout + treść stroke'ów do hashy stref** (#5) i jawny błąd/flaga zamiast fallbacku w `core_fix_loader` (#6).
4. **`ObjectPipeline.from_project(proj, cfg)` w engine** – jedna budowa pipeline'u dla UI i kolejki (#1), centralizuje `load_defaults`, umożliwia cache per projekt zamiast per klatka podglądu.
5. **Scalenie eksportu** (`export_mattes` → `_export_by_source`, `EXPORT_*` do `engine/export.py`).
6. **`engine/worker_proto.py`** – jedna pętla serve + `WorkerProcess` z cancel_check/timeout/restart/`--check`; podpięcie 5 workerów (#11 + dryf).
7. **`static/ui.js`**: jeden `api()`, jeden toast, jeden dialog (zastępuje `prompt()`, `.md-modal-bg`, `#jobErrModal`), kolory z tokenów; porządek cyklu odświeżania objects.js (duplikat w `loadStages`, `Promise.all` w `loadObject`, `md-compare-ref` zamiast `md-plate`).
8. **Podział app.py na routery** wg istniejących markerów `# ----` z jawnymi zależnościami (jak `merge_matte_api`), wspólne `_frames()`/`_run_json()`, importy na górę, jeden kształt błędu.
9. **Sesja porządkowa**: `experiments/`, `docs/archive/`, `git rm --cached local_results/`, skasować `INSTALL_2026-*.md`, `websockets`/`pytest` z requirements, pole `version` z pyproject, martwy kod z sekcji 3, `ruff --fix` dla F401/RUF100/F811.

Nowe strażniki do `tests/test_static_js.py` (tanie): brak `new WebSocket` poza app.js; brak `window.prompt(`/`confirm(`;
`innerHTML` z `${x.name|.label|.detail}` tylko przez `esc(`.
