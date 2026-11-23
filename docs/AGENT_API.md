# AGENT_API – MatteDesk for Claude Code (and scripts)

**Division of work (user decision 2026-10-12):** "by eye" decisions – choosing the SAM 3.1 candidate, where to put a key, whether an edge is good, points / brush –
are made by the user in the UI (http://127.0.0.1:8765). The agent drives the tool through the API: it starts stages and the queue, reads progress / errors, analyses QA
and versions, takes measurements, improves the tool, hands the result over to Nuke (`nuke-mcp`). The scripts `tools/pm_stage*_run.py` and the QA tools in the shot remain
the reference / fallback for unusual cases.

## Command line: `tools/md.py`
Standard library only (`urllib`). Server from `config.json` / `config.local.json` (`backend_host`, `backend_port`, default 127.0.0.1:8765) or `--url`.

| Command | What it does |
|---|---|
| `status [--json]` | project, objects with stages and active version, current job |
| `open <katalog>` | `POST /api/project/open_or_create` (reads the manifest; writes nothing to the manifest) |
| `run <obj> [--wait]` | "Compute shot" („Licz ujęcie”) (SAM 3.1 core; intervals from cache when the inputs are unchanged) |
| `rerun <obj> [--frame F] [--wait]` | "Recompute changed intervals" („Przelicz zmienione przedziały”) |
| `finalize <obj> [--edge-mode M] [--ma-scale auto|1.0|0.75|0.5] [--stage matanyone] [--wait]` | "Compute final" („Przelicz finalnie”) (stage 5 Experiments („Eksperymenty”)) → new version `runs/v###` + QA + edge report; `--ma-scale` first saves the object setting `ma_scale_mode` (like the select in the UI; auto = 9-frame trial by VRAM, a number = fixed scale without a trial); `--stage matanyone` = stage 3: MatAnyone2 on the mask 1+2 without ViTMatte and threshold zones (base version of stage 4, `tool.stage_runs.matanyone`) |
| `core-fix <obj> [--wait]` | stage 2 "▶▶ Apply corrections" („▶▶ Zastosuj poprawki”): mask 1+2 = SAM 3.1 core + brush corrections (zones `criterion: all`, `params.source: brush`, keys propagated by CoTracker3) on the shot → `objects/<obj>/core_fix/` (`frame/F?view=core_fix`) |
| `edge-fix <obj> [--v vNNN] [--skip id,id] [--wait]` | "Apply edge correction" („Zastosuj poprawkę krawędzi”) (new version; pieces listed in `--skip` are left untouched) |
| `qa <obj> [--v vNNN] [--json]` | flicker, gate, red / yellow frames, edge pieces |
| `versions <obj> [--json]` | versions `runs/v###` (mode, time, QA, active) |
| `keys <obj>` | keys (frame, kind, author) |
| `jobs` / `cancel` | server jobs / "Abort" („Przerwij”) (only on the user's explicit instruction) |
| `export <obj> --yes` / `export --all --yes` [`--format auto\|png\|exr`] [`--source core\|matanyone\|vitmatte\|version\|active`] [`--run vNNN`] | matte export (writes – only after the user's confirmation); without `--format` = project setting (`project.json` → `export_format`, auto = 8-bit PNG for 8-bit sources, EXR for EXR). Source (2026-10-15): `core` = mask 1+2 (available right after the core; a stale one is computed at export), `matanyone` / `vitmatte` = current result of stage 3 / 4, `version` = `--run` (one object), `active` (default) = as before. Context objects (in front, parts) from the same source, or, when it does not exist, from a lower stage (warning); an object named explicitly never changes its source (409) |
| `denoise [auto\|on\|off]` | plate-to-mask denoise (4b) in the project: without a value = show it (setting → effective, reason), with a value = saved to `project.json` → `mask_plate_denoise` + list of objects to recompute; `status` also shows the effective value |
| `zone-matte <obj> --frame F --draw zone.json [--preset hair] [--matting matanyone\|vitmatte\|auto\|chroma\|knn\|cf\|hybrid\|clean] [--prompt "…"] [--zone z_01] [--no-holdout] [--wait]` | "local matting" („matting lokalny”) zone (`docs/PLAN_HAIR_TOOL.md`): drawing `{polygon, strokes, bg_strokes[, bg_static, params, engine_hints]}` in plate px in frame F → `POST /zones/{id}?preview_frame=F` (preview: px added / soft, engine shares, chroma background, seam, flags; overlays `frame/F?view=zone_alpha\|zone_trimap\|zone_engines`); 409 with `missing_front` → code 2 + list; `--wait` = then "Compute final" |
| `zones <obj>` | list of the object's zones (criterion, preset / engine, "−" strokes) + card state (computed / to recompute („do przeliczenia”), keys, version, s/frame) |
| `zone-interpret <obj> "<opis>" [--frame F [--zone z_01]]` | step 9: what a zone description means without a VLM (PL/EN rules: preset, phrase for SAM 3.1, distractors, engine multipliers, warnings); with `--frame` also "−" strokes from straight lines in the background (Hough; to be confirmed – writes nothing) |
| `zone-propose <obj> --frame F [--prompt "…"] [--min-area 400] [--accept N] [--matting …]` | step 9 "Propose zone" („Zaproponuj strefę”): SAM3_Detect with the description phrase on the object crop minus the core + soft edges outside the core → proposals (polygons, side, source); `--accept N` creates a `matte` zone from a proposal (key in F, preview) |
| `cleanplate-auto [--wait] [--frames a-b]` | automatic clean plate from the shot (`docs/PLAN_CLEANPLATE.md`): after the cores of all objects (409 with a list when missing), median of the uncovered pixels + LaMa (Comfy) / OpenCV on the holes → plate `cleanplate_auto` in the project (manifest untouched); `status` shows "Clean plate: project / auto / none" („Clean plate: projekt / auto / brak”) |
| `zone-keys <obj> <zone> [--set F --draw key.json [--prompt "…"]] [--delete F] [--region F]` | keys of a `matte` zone (plan step 4): save / delete a key (drawing in plate px), list; `--prompt` (step 7, correction loop) = iteration note saved in the key (`keys[F].prompt`, appended to `prompt_history`; `''` clears it), read by the prompt rules in that frame; `--region F` = region after propagation in the frame (contours, bbox; the first call tracks with CoTracker3, later calls come from cache) |
| `zone-apply <obj> <zone> [--stage vitmatte] [--wait]` | "Apply" („Zastosuj”): only the zones are recomputed (cache per frame / sequence) on the base alpha of the active version → new version with QA; `--stage vitmatte` = stage 4: on the MatAnyone2 version from stage 3 (must be current, otherwise 409), only brush corrections + `matte` zones (`tool.stage_runs.vitmatte`); prints recomputed / from cache; 409 `missing_front` → code 2 |
| `plate-qa [plate] [--frames a-b] [--wait]` | Z26: check of the plate frames (missing / NaN / size / exposure jump / image jump / duplicate) → `plate_qa/<plate>.json`, list of the flagged frames with a proposed method |
| `plate-fix [plate] --decisions 162:mean,170:flow [--yes] [--wait]` | Z26: writes `<katalog>_fix/` (repaired frames in the source encoding, the rest as links) and switches the plate in the project; without `--yes` only the plan (size) |
| `queue-all` | "Compute all" („Policz wszystko”): `POST /api/queue/preset/all` – objects without a valid result go into the queue, front first; start with `POST /api/queue/run` |
| `c3d status` | Cleanup 3D (`docs/PLAN_CLEANUP_3D.md`, `GET /api/cleanup3d`): steps ✔ done / ↻ stale / ○ empty (camera, cards, keys, project), camera report (reprojection, horizon), cards with their keys and active versions, engines and the camera solve with the reason when a model / COLMAP is missing |
| `c3d solve [--moving a,b] [--no-horizon] [--free-lens] [--wait]` | 3D camera of the work range (`POST /api/cleanup3d/camera/solve`, long job: COLMAP CUDA + CoTracker3 + bundle adjustment); `--moving` = objects masked out (default: the objects removed in the Cleanup settings, `''` = none); 409 with the Doctor's hint when COLMAP is missing |
| `c3d propose` | cards proposed from the solve – ground, sea level (shots with a horizon), sky – merged into `cards.json`; cards the user already has are never overwritten |
| `c3d keys <card> [--frames 1,8 \| --every N]` | list the keys of a card, or add keys (first version = the plate frame) |
| `c3d key-export <card> <frame> [--format png\|png16\|exr] [--dir D]` / `c3d key-import <card> <frame> <path>` | key out for painting (default `<work>/inpaint/keys_export/`) / painted file back as a NEW version (409: other size, unreadable, not RGB) |
| `c3d key-undo <card> <frame>` / `c3d key-delete <card> <frame> --yes` | previous version active / the key folder deleted from disk for good (no trash, 2026-11-13) |
| `c3d regen <card> <frame> [--engine lama\|qwen21_edit\|klein_restore] [--soften] [--prompt "…"] [--wait]` | regenerate the key's marked areas (LaMa, Qwen-Image-2.1 or FLUX 3 Image Klein + LoRA Restore) → new active version; 409 with the gate's reason when the model is missing, or when no area is marked |
| `c3d render [--frames a-b] [--blur auto\|off] [--feather px] [--wait]` | projection of the keys + comp → `<work>/inpaint/v###/`, plate „cleaned” in `project.json` (as the Cleanup MVP's run) |
| `c3d nuke` | step C6: Nuke script rebuilding the projection (`POST /api/cleanup3d/export_nuke`) → `<work>/inpaint/nuke/c3d_<v### or date>.nk` + `.chan` (+ STMaps, extent masks); notes say what is missing (no projection = no removal mask) |
| `api-doc [--write]` | the route table below (from `app.routes`) |

Exit codes: 0 = OK, 1 = error, **2 = server busy / refused (HTTP 409)**. `--wait` prints progress from `GET /api/jobs` (step, frames, Comfy state).

## Rules
- **The shot manifest (`shot_manifest_*.json`) is read-only.** Tool state: `<work>/matte_desk/project.json`, objects `objects/<obj>/`.
- **One heavy computation at a time** (`long_job`): a second one → 409 "Running: …" („Trwa: …”). The agent and the user work on the same server – before a heavy job the CLI
  checks `/api/jobs` and, when something is being computed, exits with code 2 (it does not abort someone else's job). Performance measurements only on an idle server.
- `object_edit` – an edit of an object is rejected (409) while that object is being computed; `when_idle` – project changes only when nothing is being computed.
- **EXR only after the user's confirmation** (`confirmed: true` / `--yes`): export, `_align`, `_deflicker`, full STMaps.
- A backend change requires a restart of the user's server – only with their consent.

## 8-bit vs EXR sources (decision 2026-10-13)
- Images for Comfy follow the plate's bit depth (`manifest bit_depth`, `engine/comfy_io.py`): **8-bit** (MP4 / PNG / JPG, sRGB) → 8-bit PNG
  (values without colour transform) + `VHS_LoadImagesPath` (masks: + `ImageToMask`), one folder per job; frames for SAM3_VideoTrack
  larger than 1008 are downscaled, masks return to the input size. **EXR** (ACES) → float32 EXR + `PlateLoadEXR`, unchanged.
  Backlit and LTX-2.5 stay on EXR.
- Frame-writing loops report progress and respond to "Abort".
- Plate-to-mask denoise (SPEC 4b, MC denoise DIS ±2 + bilateral, ~1.5 s/frame at 1440 px): project switch
  `GET/PUT /api/project/mask_plate_denoise` (`project.json` → `mask_plate_denoise`: auto / on / off; auto = on for EXR / > 8 bit, off for 8-bit).
  One setting for the core and "Compute final" (shared cache `mplate/`); `defaults.json` `mplate.enabled = false` disables it everywhere.
  The flag is part of the interval cache hash and of `core.json`; a change of the effective value = core "to recompute" (an old core without this key = computed with it on).
- Mask export format: `GET/PUT /api/project/export_format` (`project.json`), override for a single export: `format` in `POST /api/export`
  and `POST /api/object/{name}/export`.
- Plate colour profile (stage 0, 2026-11-03): `GET/PUT /api/project/color_profile` (`project.json` → `color_profile`: `auto` | a log-curve id
  `acescct` / `cineon` / `logc3` / `logc4` / `slog3` / `video`; never in the manifest). `auto` = proposal of `engine/color_profile.py` from the
  EXR header (ACES container flag, ARRI / Sony / RED metadata), the manifest colorspace, else 8-bit / video → `video`, else `cineon` – the reply
  carries `effective` {id, label, reason}, `proposal` and `options`. It is the curve MergeMatte's `log_curve` = `project` and the .nk export
  resolve to. MergeMatte (`GET/PUT /api/object/{name}/merge_matte`): `knobs.log_curve` = `project` | `none` | a curve id (the old bool `log`
  reads as `acescct` / `none`); the reply has `log_curve_effective`, `curves` (id, label, index) and `project_curve`. The Nuke group converts
  lin ↔ log with `Expression` nodes (explicit formulas, `engine/log_curves.py`), no OCIO names – it works in Nuke's default config and ACES 1.x / 2.x.
- Plate colour profile (stage 0, 2026-11-03): `GET/PUT /api/project/color_profile` (`project.json` → `color_profile`: `auto` | a log-curve id
  `acescct` / `cineon` / `logc3` / `logc4` / `slog3` / `video`; never in the manifest). `auto` = proposal of `engine/color_profile.py` from the
  EXR header (ACES container flag, ARRI / Sony / RED metadata), the manifest colorspace, else 8-bit / video → `video`, else `cineon` – the reply
  carries `effective` {id, label, reason}, `proposal` and `options`. It is the curve MergeMatte's `log_curve` = `project` and the .nk export
  resolve to. MergeMatte (`GET/PUT /api/object/{name}/merge_matte`): `knobs.log_curve` = `project` | `none` | a curve id (the old bool `log`
  reads as `acescct` / `none`); the reply has `log_curve_effective`, `curves` (id, label, index) and `project_curve`. The Nuke group converts
  lin ↔ log with `Expression` nodes (explicit formulas, `engine/log_curves.py`), no OCIO names – it works in Nuke's default config and ACES 1.x / 2.x.

## Cleanup 3D (2026-11-10, `api/cleanup3d.py`, plan `docs/PLAN_CLEANUP_3D.md`)
Steps C2–C5 of the Cleanup mode; files in `<work>/inpaint/` (`camera3d/cams.json`, `cards.json`, `cards/<card>/a_<id>.png` = brush layers
of an extent, `keys/<card>/k_####/` = key versions + `areas.json` + brush layers, `c3d_render.json` = which `v###` is the projection of the
present state). Errors: no project 400, unknown card / key 404, an engine refusal or a missing model 409 with its Polish text.
- `GET /api/cleanup3d` – `done` {camera, cards, keys, project}, `stale` {project} (the projection is older than the camera / cards / active
  key versions → ↻), `render` {version, date, stale}, `camera.report` (`reproj_px`, `horizon_px`, `worst_frames`), `cards[].keys`,
  `engines[].available / reason`, `solve.available / reason`.
- Areas (a key's regeneration areas, a card's extent, a mark on any frame) are a list of layers: curve `{id, kind: "curve", op, label, curve}`
  (`engine/curves.py` JSON, closed) or brush `{id, kind: "brush", op, label, png}`; a brush layer carries `png_data` (data URL, 8-bit,
  white = area) only when its pixels changed – without it the server keeps `a_<id>.png`. `PUT …/keys/{card}/{frame}/areas` and
  `PUT …/cards/{card}/extent` (+ `frame`; `areas: []` + `frame: null` = the whole card) replace the whole list; the pixels come back from
  `GET …/areas/{id}/png` and `GET …/extent/{id}/png`.
- `POST /api/cleanup3d/mark/{frame}` `{areas, cards | null}` – a drawing on any frame is carried through the camera and the card to the
  frame's own key or the two nearest keys of every card it touches, appended there as closed curves `m<frame>_<id>`; answer `keys: [{card, frame, added}]`.
- `GET /api/cleanup3d/view/{key|proj|cards}/{frame}[?card=&max_side=]` – PNG; `key` shows the card's key nearest to the frame (header
  `X-C3D-Key-Frame`); cached per state, the first image after a change builds the scene (seconds at 4K).
- Key modes (P13, `engine/cleanup3d/keymodes.py`): `POST …/keys/{card}/auto` `{threshold, min_gap, max_gap, max_keys, create}` –
  key frames where the projection error (plate vs neighbouring key frames projected, outside the removed objects) is above the
  threshold; saved on the card (`key_mode: auto`, `key_auto`), keys created from the plate frames; answer `frames`, `errors`, `added_keys`.
  `POST …/keys/{card}/master` `{frame | null, replace}` – ONE master key („klucz-suma”) on its own canvas (`key.canvas`, larger than the
  frame) accumulated from the clean pixels of every frame; 409 while the card has other keys unless `replace: true` (they are
  deleted from disk for good); what no frame sees = the key's brush area `unseen` (regenerate it); answer `canvas`, `seen_share`, `hole_px`,
  `replaced` (frames of the deleted keys), `freed_bytes`.
- Long jobs (one at a time, „Przerwij”): `camera/solve`, `keys/{card}/{frame}/regen`, `keys/{card}/auto`, `keys/{card}/master`, `render`. The render uses the Cleanup settings
  (`PUT /api/cleanup`: `remove`, `grow_px`, `holdout`, `fill`) – set them first; `feather_px` overrides `soft_px` for that run.
- `POST /api/cleanup3d/export_nuke` `{frames | null}` (body optional; null = every camera frame) – C6, P11 (`engine/cleanup3d/nuke.py`):
  `<work>/inpaint/nuke/c3d_<tag>.nk` + `.chan` (tag = the current projection's `v###`, else the date) + `_stmap_undistort/distort.exr`
  (only with lens distortion) + `_extent_<card>.png`; answer `{nk, chan, dir, stmaps, extents, counts, cards, frames, notes, render}`.
  Camera3 animated per frame (Nuke axes: Y up, looking down -Z; Nuke world = ours turned 180° about X), cards as Card3D, the sky as a
  Sphere, each key Read → Project3D with its own static projector Camera3 → ApplyMaterial → ScanlineRender, keys of a card crossfaded by
  Dissolves (the render's smoothstep), Merge over the plate through the alpha of the last projection's `patch/`. 409 without camera /
  cards / keys. Not in the script: matched motion blur, tone match, key coverage fallback (the render does these).

## Object stages (user decision 2026-10-15)
`GET /api/object/{o}/stages` → `steps` (UI bar): **1 SAM 3.1** (detect + crops + core) · **2 CoTracker3** (brush corrections, optional) ·
**3 MatAnyone2** (edge from the mask 1+2) · **4 ViTMatte** (zones on the result of 3) · **5 Experiments** (`final`: edge modes, threshold zones,
edge correction, versions, QA) · **6 Export** („Eksport”) (unlocked right after the core). `stages` = detailed stages (`detect, crops, core, cotracker,
matanyone, vitmatte, final, export`) with `NEEDS` rules (`engine/stages.py`) – the backend refuses an action of a locked stage (409);
`stage_runs` = versions of stages 3 / 4 (`vitmatte@core`, `vitmatte@matanyone` – a result on each base), `has.core_fix` = mask 1+2 on disk.
ViTMatte base: `PUT /api/object/{o}/settings {"vitmatte_base": "matanyone" | "core"}` (invalidates nothing; `core` = mask 1+2 as a version
`stage: core`, stage 4 without MatAnyone2). Comparing two versions: `GET /api/object/{o}/frame/{f}?view=alpha_diff&v=A&v2=B` (|A − B| ×3).
Group (object + parts): stages 3 / 4 on the main object compute the whole including the seam (`finalize stage=matanyone` on a part → computes the parent;
response `object`, `asked`, `group`, `group_outside_crop_px`); the whole group in the preview `frame/{f}?view=full&src=core|core_fix|alpha[&v=]`;
edge band preview before computing `GET /api/object/{o}/edge_preview/{f}?band_scale=&shift_px=` (255 = mask, 128 = band).
"Alpha motion blur" („Motion blur alfy”) (2026-10-16, stages 3 / 4 / 5 – palette): frame preview `GET /api/object/{o}/motion_blur/{f}?v=&strength=&shutter=`
(PNG like view=alpha, headers X-MD-MB-Max-Px / Moving-Px / Run), applying it `POST /api/object/{o}/motion_blur {"run", "strength",
"shutter_deg"}` → new version with `run.json.motion_blur`; the stage result (stage_runs) stays the stage result; GPU per `mb_device`.

## Typical flow (new shot → export)
1. `open <katalog>` (or the wizard in the UI) → stage 0 in the UI (plates, proxies, alignment of the backlit / clear plate).
2. Object: UI (name, prompt) → points / candidate (user) → "▶▶ Recompute whole" („▶▶ Przelicz nieaktualne”) (`POST /api/object/{o}/compute_all`: crops + core).
3. Core review (user: SAM 3.1 keys) → `rerun <obj> --wait`; persistent errors → brush corrections in the UI (stage 2) → `core-fix <obj> --wait`.
4. `finalize <obj> --stage matanyone --wait` (stage 3) → ViTMatte zones in the UI → `zone-apply <obj> <zone> --stage vitmatte --wait` (stage 4);
   experiments: `finalize <obj> --wait` → `qa <obj>` → optionally `edge-fix <obj> --skip …` → `versions <obj>`.
5. `export --all --yes [--source …]` (after the user's consent) → `outputs.json` (contract: `docs/DATA_CONTRACT.md`) → Nuke.

## HTTP routes (`app.py` + `api/*.py` routers, included by `app.py` in place)
Kind: `long_job` = heavy, one at a time, listed in `/api/jobs` (quiet = status bar only, no window); `object_edit` = rejected while the object is being computed;
`when_idle` = only when nothing is being computed.

<!-- routes:begin (python tools/md.py api-doc --write) -->
| Metoda | Ścieżka | Rodzaj | Opis |
|---|---|---|---|
| GET | `/api/version` | – | Running version + whether the server runs older code than on disk (the UI then asks for a restart). |
| GET | `/api/jobs` | – | What is being computed now (status bar) + who holds / waits for the GPU (engine/gpu_manager: a job waiting has `gpu_wait`; background work without a job – band  |
| GET | `/api/jobs/details` | – | 'Szczegóły' of the running job (Z19): GPU (nvidia-smi; null without it), Comfy RAM / VRAM / queue (foreign prompts), our step with frames and ETA, Comfy state o |
| POST | `/api/jobs/cancel` | – | 'Przerwij': stop the running job(s) at the next safe point; our Comfy prompt is removed / interrupted. |
| GET | `/api/doctor` | – |  |
| POST | `/api/report/preview` | – | The whole package, anonymised, as it will be sent – kept under `id` for /api/report/send. |
| POST | `/api/report/send` | – | Sends exactly the previewed package (only with the user's consent) to the report proxy. |
| GET | `/api/config/comfy` | – | Doctor: which ComfyUI is used (Desktop / from git), its folders, port and autostart. |
| POST | `/api/config/comfy` | when_idle | Doctor's switch: ComfyUI Desktop <-> ComfyUI from git |
| GET | `/api/config/weights` | – | Doctor: where MatteDesk keeps the weights it downloads outside ComfyUI (Hugging Face cache) – the folder, why it is that one, its size and the free space of its |
| POST | `/api/config/weights` | when_idle | Set the folder (config.local `hf_home`), move what is already downloaded (hub/, xet/) when asked – a move, nothing deleted; items that already exist in the new  |
| GET | `/api/models/removable` | – | Doctor's 🗑: optional models whose weights are on disk, with every path and its size (engine/model_trash.py). |
| POST | `/api/models/remove` | when_idle | Removes ONLY the weights of one optional model (the user confirmed the listed files in the Doctor). |
| GET | `/api/update` | – | Newer MatteDesk in the public GitHub releases? (at most once a day unless force; engine/updates.py) |
| POST | `/api/update/skip` | – |  |
| POST | `/api/update/settings` | – |  |
| POST | `/api/config/app_browser` | – | Where the desktop shortcut opens MatteDesk (tools/launch_app.py reads it at the next start). |
| GET | `/api/update/progress` | – |  |
| POST | `/api/update/install` | when_idle | Installer copy only: download the new setup, start it, then stop this server (the setup replaces the files). |
| GET | `/api/comfy/launch` | – | State of the ComfyUI start by MatteDesk: idle / starting / running / failed (+ how / error). |
| POST | `/api/comfy/start` | – | Doctor button "Uruchom ComfyUI": start the chosen ComfyUI (Desktop app / ComfyUI from git) in the background. |
| GET | `/api/comfy/health` | – | Banner 'ComfyUI nie odpowiada': up / down / busy + the launch state (polled every few seconds by comfy_watch.js). |
| POST | `/api/comfy/reload` | – | Banner button 'Przeładuj ComfyUI': close a hung local ComfyUI on our port and start it again (no idle check – a job waiting on a dead ComfyUI fails by itself). |
| GET | `/api/install/progress` | – | Live state of the running "Zainstaluj" (step, file, done/total bytes) - the Doctor polls it for the progress bar. |
| POST | `/api/install/{what}` | when_idle | Doctor's "Zainstaluj" button (user-triggered): our Comfy node packages, third-party packs, weights, CoTracker3. |
| POST | `/api/project/open` | when_idle |  |
| POST | `/api/project/new` | when_idle |  |
| POST | `/api/dialog/pick_folder` | – |  |
| POST | `/api/dialog/pick_file` | – | Native file picker (a single image / video: target background BG, MP4 plates); "" when cancelled. |
| POST | `/api/project/inspect_working_dir` | – |  |
| POST | `/api/project/open_or_create` | when_idle |  |
| GET | `/api/project/recent` | – | Recently opened shot folders (config.local.json, newest first) for the project dialog. |
| GET | `/api/project/current` | – |  |
| POST | `/api/project/generate_proxies` | long_job |  |
| GET | `/api/project/plates` | – |  |
| POST | `/api/project/plate/{name}/content` | – | Stage 0, the content check of a backlit / clear plate (NEXT_STEPS 5): „↻ zmierz ponownie” drops the stored result (measured again in the background), „użyj mimo |
| POST | `/api/project/plate/inspect` | – |  |
| POST | `/api/project/plate/update` | long_job |  |
| DELETE | `/api/project/plate/{plate_name}` | when_idle |  |
| GET | `/api/frame/{frame_num}` | – | Proxy JPEG of a frame |
| GET | `/api/frame/{frame_num}/tile` | – | Z8: full-resolution crop [x, y, w, h] (plate px) of a plate frame for zoom > 100 % of the proxy – the same display conversion as the proxy |
| GET | `/api/project/stages` | – | Stage 0 (Plate'y) of the shot: what exists for the stage bar / panel – plates & proxies, alignment reports of the auxiliary plates (backlit / cleanplate: the on |
| POST | `/api/plate/align/measure` | long_job |  |
| POST | `/api/plate/align/export_nk` | – |  |
| POST | `/api/plate/align/apply` | long_job |  |
| GET | `/api/plate/align/report` | – |  |
| GET | `/api/plate/align/residual/{frame}` | – | Heat map PNG (BGRA, 0 px = blue … 1 px and above = red, no measurement = transparent) of the residual of the aligned plate on frame `frame` – drawn over the pla |
| GET | `/api/plate/align/residual_stats/{frame}` | – |  |
| GET | `/api/plate/clean/exr_dir` | – | Default EXR export folder of a plate (the palette shows it before the run). |
| POST | `/api/plate/clean/measure` | long_job |  |
| POST | `/api/plate/clean/run` | long_job |  |
| GET | `/api/plate/clean/report` | – |  |
| POST | `/api/camera/track` | long_job |  |
| GET | `/api/camera/report` | – |  |
| POST | `/api/camera/export_nk` | – |  |
| POST | `/api/depth/compute` | long_job |  |
| GET | `/api/depth/report` | – |  |
| POST | `/api/depth/export_nk` | – |  |
| POST | `/api/object/{name}/preview_frame` | long_job (quiet) | SAM3_Detect on the crop of one frame (SPEC 6 pkt 2, < 3 s): quick loop before computing the shot. |
| POST | `/api/object/{name}/points/{frame}` | long_job (quiet), object_edit | Stage 1 (Wykryj): the points of a frame ARE its key |
| GET | `/api/object/{name}/edge_modes` | – | 'Krawędź' list (SPEC 5.5): Backlit only when a backlit plate exists |
| PUT | `/api/object/{name}/edge_mode` | object_edit |  |
| GET | `/api/project/work_area` | – | Work area (AE / Nuke style, user request 2026-10-15): `range` = [first, last] or None (whole shot), `shot`, `effective` = the frames „▶▶ Przelicz nieaktualne”,  |
| PUT | `/api/project/work_area` | – | Stored in project.json -> work_area (the shot manifest is never written) |
| GET | `/api/object/{name}/life` | – | Object life range (zakres życia, user request 2026-10-15): `life` = visible intervals [[a, b], ...] or None (the whole shot), `shot`, `hints` = suggestions from |
| PUT | `/api/object/{name}/life` | object_edit | Stored in objects/<name>/object.json -> life (the shot manifest is never written) |
| GET | `/api/object/{name}/stages` | – | Stage strip of the objects panel: done / stale / todo / locked + the next action. |
| POST | `/api/object/{name}/compute_all` | long_job | „▶▶ Przelicz nieaktualne” (user decision 2026-10-15): crops only when missing or when a user key no longer fits its crop, then the core on the whole shot (uncha |
| POST | `/api/object/{name}/run` | long_job | Core SAM 3.1 on the whole shot from the keys (7.2); intervals with unchanged inputs come from cache. |
| POST | `/api/object/{name}/core_import` | long_job | Stage 1 alternative (user request 2026-11-20): the user's external mask sequence (EXR / PNG / TIFF alpha) as the core instead of SAM 3.1 – core/, core_soft/, cr |
| DELETE | `/api/object/{name}/core_import` | object_edit | „Wróć do SAM 3.1”: the imported core removed from disk for good (keys stay) – then „▶▶ Przelicz nieaktualne”. |
| POST | `/api/object/{name}/core_fix` | long_job | Stage 2 „▶▶ Zastosuj poprawki” (2026-10-15): the mask 1+2 (SAM 3.1 core + brush corrections propagated by CoTracker3) on every frame the core covers -> core_fix |
| POST | `/api/object/{name}/finalize` | long_job | 'Przelicz finalnie' (7.3a) -> runs/v### on the work area (whole shot without one); the core must cover it |
| GET | `/api/object/{name}/ltx` | – | Stage „LTX-2.5 Alpha-Gen”: the object's settings (defaults.json "ltx" <- object `ltx`), the stage result, models. |
| PUT | `/api/object/{name}/ltx` | object_edit |  |
| GET | `/api/object/{name}/ltx_preview/{frame}` | – | LTX-2.5 palette (2026-10-16), live: what=band – 128 = where the LTX-2.5 matte is kept, 255 = forced 1 (proxy PNG); what=alpha – the current LTX-2.5 version's ra |
| POST | `/api/object/{name}/ltx/refinish` | long_job (quiet) | „Zastosuj suwaki na wynik” (2026-10-16): the current LTX-2.5 version's raw mattes limited again with the saved region / mask settings -> a new version (no model |
| POST | `/api/object/{name}/ltx` | long_job | Stage „LTX-2.5 Alpha-Gen” (user decision 2026-10-16): LTX-2.5 IC-LoRA Alpha-Gen on the work area, guided by the mask 1+2 (other objects painted out of the guide |
| GET | `/api/object/{name}/videomama` | – | Stage „VideoMaMa” (2026-10-16): the object's settings (defaults.json "videomama" <- object `videomama`), the stage result, weights / nodes. |
| PUT | `/api/object/{name}/videomama` | object_edit | Stage VideoMaMa settings of the object (base, the band sliders, resolution, batch / overlap). |
| GET | `/api/object/{name}/videomama_preview/{frame}` | – | VideoMaMa palette, live: what=band – 128 = the band, 255 = the base deeper inside (proxy PNG); what=alpha – the current VideoMaMa version's raw matte put over t |
| POST | `/api/object/{name}/videomama/refinish` | long_job (quiet) | „✚ Suwaki na wynik”: the current VideoMaMa version's raw mattes put over the base again with the saved base and band sliders -> a new version (no model run). |
| POST | `/api/object/{name}/videomama` | long_job | Stage „VideoMaMa” (user decision 2026-10-16): VideoMaMa on the work area guided by the mask 1+2, put over the base (mask 1+2 or the stage-3 result) -> runs/v### |
| POST | `/api/object/{name}/core_base` | long_job | Stage Eksperymenty: the mask 1+2 written as a version (reused while current), so it can be picked as a base like any stage result |
| POST | `/api/object/{name}/analyze` | long_job | Stage Eksperymenty (2026-10-16): QA + edge-correction proposals of the chosen base version (measured once). |
| GET | `/api/object/{name}/edge_recount` | – | Live count of the proposals with other thresholds (saved measurements only; nothing written). |
| POST | `/api/object/{name}/experiment` | long_job | Stage Eksperymenty (2026-10-16): „Przelicz z tej bazy” – the base version's mask -> threshold zones -> optional edge correction of the base -> edge engine -> a  |
| POST | `/api/object/{name}/edge_fix` | long_job | 'Zastosuj poprawkę krawędzi' (Z27 pkt 1): the former automatic second round, as a new runs/v###. |
| GET | `/api/object/{name}/edge_preview/{frame}` | – | Stage 3 palette: preview of the MatAnyone2 edge before computing – proxy PNG, 255 = mask kept (mask 1+2 + shrink / grow), 128 = adaptive band where MatAnyone2 d |
| POST | `/api/object/{name}/stage_preview/{frame}` | long_job (quiet) | Stage 5: the preview is computed with the object's current engine and kept per engine (slot "vitmatte" / "vitmatte@birefnet", 2026-10-22) – the other engine's p |
| GET | `/api/object/{name}/stage_preview/{stage}` | – | {exists, stage, center, frames, stale, seconds} of the last preview of this stage (stage 5: of `engine`, default the object's current engine). |
| GET | `/api/object/{name}/stage_preview/{stage}/{frame}` | – | Proxy PNG of the preview alpha of `frame` (404 outside the preview window) |
| GET | `/api/object/{name}/edge_pieces/{frame}` | – | Stage-4 overlay: contour samples of the pieces to correct in this frame, in plate px (kind halo = red / shrink, cut = blue / grow). |
| GET | `/api/object/{name}/frame/{frame}` | – | Proxy-size PNG of the object in a frame: view=core / alpha / overlay (contour on the plate proxy) |
| GET | `/api/object/{name}/frame/{frame}/tile` | – | Z8 / plan step 7: full-resolution crop [x, y, w, h] (plate px) of a mask of the object in `frame` as an 8-bit PNG: view=core (core/c_####.png) / key (keys/k_### |
| POST | `/api/object/{name}/keys/{frame}` | object_edit | User key (SPEC 7.1): base (current core / SAM 3.1 preview) +/- brush strokes and polygons -> hard mask. |
| DELETE | `/api/object/{name}/keys/{frame}` | object_edit |  |
| POST | `/api/object/{name}/correct/{frame}` | long_job (quiet), object_edit | Stage 3 local correction (user request 2026-10-12): change ONLY the clicked hole / leak of the current frame mask (its key, else the SAM 3.1 core) |
| GET | `/api/object/{name}/keys` | – |  |
| POST | `/api/object/{name}/undo` | object_edit |  |
| POST | `/api/object/{name}/redo` | object_edit |  |
| POST | `/api/object/{name}/rerun_from/{frame}` | long_job | Incremental core (7.3): only intervals whose inputs changed are recomputed (cache hash), the rest is reused. |
| GET | `/api/object/{name}/zones` | – |  |
| GET | `/api/zones/preset_defaults/{preset}` | – | Default engine and params of a zone preset (defaults.json) – the zone card's "↺ z presetu". |
| POST | `/api/object/{name}/zones/interpret` | – | Plan step 9: what the zone description means without a VLM – preset, SAM 3.1 phrase (`fg_en`), distractors, routing multipliers, warnings; with `frame`: propose |
| POST | `/api/object/{name}/zones/propose` | long_job (quiet) | Plan step 9 „Zaproponuj strefę”: SAM3_Detect with the prompt's subject on the crop around the core, minus the core, plus soft edge pixels outside the core -> zo |
| POST | `/api/object/{name}/zones/{zone_id}` | object_edit | Create (zone_id='new') or update a zone (SPEC 7.7) |
| PATCH | `/api/object/{name}/zones/{zone_id}` | object_edit | Zone card (plan step 5): enable / disable, description (prompt; the previous one goes to prompt_history), preset, engine, holdout ("Licz mimo to bez holdoutu") |
| POST | `/api/object/{name}/zones/{zone_id}/preview/{frame}` | – | Automatic preview of the panel (plan step 5, debounce in the UI): the `matte` zone as saved, in `frame` (its key there when one exists) – nothing is written to  |
| PUT | `/api/object/{name}/zones/{zone_id}/keys/{frame}` | object_edit | Zone key (plan step 4): the drawing of the zone in `frame` (polygon / '+' strokes / '-' strokes) |
| DELETE | `/api/object/{name}/zones/{zone_id}/keys/{frame}` | object_edit |  |
| GET | `/api/object/{name}/zones/{zone_id}/keys` | – |  |
| GET | `/api/object/{name}/zones/{zone_id}/region/{frame}` | long_job (quiet) | The propagated region of the zone in `frame` (contours in plate px, bbox) – the starting point of a correction |
| GET | `/api/object/{name}/zones/{zone_id}/band/{frame}` | – | Band control points of the zone in `frame`: tracked position (CoTracker3, cached), visibility, the two widths (animated between keys) and whether `frame` is a k |
| PUT | `/api/object/{name}/zones/{zone_id}/band/{frame}` | object_edit | Add a band point in `frame` or set a key of an existing one there (position and / or widths) |
| DELETE | `/api/object/{name}/zones/{zone_id}/band/{point_id}` | object_edit | Remove a band point, or only its key in `frame` (?frame=N; the last key removes the point). |
| DELETE | `/api/object/{name}/zones/{zone_id}/band` | object_edit | Band keys of ALL points in `frame` (?frame=N – playback bar 🔑✕ / Delete / RMB on the triangle, 2026-10-21; a point left without keys goes too), or every band po |
| GET | `/api/object/{name}/stage_band/{stage}/params` | – | The band points of a stage as stored (+ the sliders' reference) – what Ctrl+Z restores. |
| PUT | `/api/object/{name}/stage_band/{stage}/params` | object_edit | Ctrl+Z of a band edit (static/bandpoints.js): the points and their reference exactly as given. |
| PUT | `/api/object/{name}/stage_band/{stage}/mode` | object_edit | „Odłącz → własny pas” (own: the stage gets a copy of the shared band – nothing changes in its result) or „wróć do wspólnego” (shared: the stage's own points are |
| GET | `/api/object/{name}/stage_band/{stage}/{frame}` | – | Band control points of a stage with a band (VideoMaMa) in `frame` – as the zone's band (2026-10-22). |
| PUT | `/api/object/{name}/stage_band/{stage}/{frame}` | object_edit | Add a band point of the stage in `frame` or set its key there (position / widths); invalidates the stage. |
| DELETE | `/api/object/{name}/stage_band/{stage}/{point_id}` | object_edit | Remove a band point of the stage, or only its key in `frame` (?frame=N; the last key removes the point). |
| DELETE | `/api/object/{name}/stage_band/{stage}` | object_edit | Band keys of ALL points of the stage in `frame` (?frame=N), or every band point (no frame = constant band). |
| POST | `/api/object/{name}/stage_band/{stage}/copy` | object_edit | „Kopiuj klucze pasa z …” (user decision 2026-10-22): the band points of another owner (a stage-5 whole-outline zone or another stage) replace this band's; width |
| GET | `/api/object/{name}/band` | – | The shared band of the object (its group): points, keys, reference and the mode of every band owner (stages 3 / 4 / 7, the whole-outline zones of stage 5) – sha |
| PUT | `/api/object/{name}/band_whole` | object_edit | „Wcięcie: cały rdzeń” (user decision 2026-10-29): the band covers the whole core – the model replaces it everywhere, the inset feather keeps that much core arou |
| GET | `/api/object/{name}/band/params` | – | The shared band as stored (what Ctrl+Z restores). |
| PUT | `/api/object/{name}/band/params` | object_edit | The shared band exactly as given (Ctrl+Z of „odłącz” / „ustaw jako wspólny” / a band edit). |
| DELETE | `/api/object/{name}/band` | object_edit | Keys of the shared band in `frame` (?frame=N), or the whole shared band (every stage on it = constant band). |
| POST | `/api/object/{name}/band/from` | object_edit | „Ustaw jako wspólny”: the points of a detached stage / zone become the shared band of the object (as stored, with their reference – the source's result does not |
| POST | `/api/object/{name}/band/propose` | long_job (quiet), object_edit | „Zaproponuj pas” (plan docs/PLAN_SAPIENS2.md point 1, user decisions 2026-10-22): Sapiens2-seg-0.4B body parts of the person in `frame` -> band points of the ba |
| GET | `/api/object/{name}/mask12_status` | – | Is the exact mask 1+2 of the work area ready (the band previews stop being „przybliżony” then)? A cheap poll – nothing is computed here (the batch runs once in  |
| GET | `/api/object/{name}/stage_band/{stage}/{frame}/lines` | – | Preview lines of a stage band in `frame` (plate px): band edges (dashed), where the inset / outer feather end (thin dashes) – one lines function for every band  |
| POST | `/api/object/{name}/zones/{zone_id}/apply` | long_job | „Zastosuj” (plan step 4): only the zones are recomputed (cache) on the base alpha of the active version -> a new version runs/v### (QA + edge report); 409 with  |
| GET | `/api/object/{name}/color_lab/{frame}` | – | Pipette of the "color" zone criterion (Z24): median Lab of 5 x 5 px of the plate-to-mask display image (the image the zone key is computed on) around (x, y) in  |
| DELETE | `/api/object/{name}/zones/{zone_id}` | object_edit |  |
| GET | `/api/object/{name}/runs` | – | Versions of the final result (Z2): runs/v### with run.json + qa.json summary; `active` = tool.active_run. |
| DELETE | `/api/object/{name}/runs/{run}` | object_edit | Delete a version (user request 2026-10-25, stage 5 "Version comparison"): `dry=1` = what would go (size) and what still uses it; otherwise the folder is deleted |
| PUT | `/api/object/{name}/active_run` | object_edit | 'Ustaw jako aktywną' (Z2): export and dependent objects use this version; final_valid from its qa.json. |
| GET | `/api/object/{name}/qa` | – |  |
| PUT | `/api/object/{name}/edge_engine` | object_edit | Stage 5 engine of the object (user decision 2026-10-21: arrow menu in the stage bar when both are installed, default ViTMatte) |
| PUT | `/api/object/{name}/settings` | object_edit | Object panel options (SPEC 1.2); any change invalidates the final result (new version on recompute). |
| GET | `/api/project/export_dir` | – |  |
| PUT | `/api/project/export_dir` | when_idle | Z29: export folder stored in project.json (the shot manifest is never written). |
| GET | `/api/project/export_format` | – | Mask export format (decision 2026-10-13): auto = PNG sequence for 8-bit sources, EXR for EXR sources. |
| PUT | `/api/project/export_format` | when_idle | Export format stored in project.json -> export_format (the shot manifest is never written). |
| GET | `/api/project/mask_plate_denoise` | – | Plate-to-mask denoise (SPEC 4b, stage 0): auto = on for EXR / > 8-bit sources, off for 8-bit; `effective` used by the core and the final pass; defaults.json mpl |
| PUT | `/api/project/mask_plate_denoise` | when_idle | Stored in project.json -> mask_plate_denoise (never in the shot manifest) |
| POST | `/api/export/nk` | – | 'Tylko .nk' (Z29): the Nuke script for the existing export (outputs.json), without writing EXR again. |
| GET | `/api/export/nk_text` | – | The export's Nuke script as text (2026-11-16, „⧉ Kopiuj do schowka”): regenerated like „tylko .nk” (current Final Touch nodes, MergeMatte knobs), so Ctrl+V in N |
| POST | `/api/export` | long_job | Export (SPEC 9): full-frame alpha per object (front-to-back subtraction), multi-EXR, outputs.json |
| POST | `/api/object/{name}/export` | – |  |
| GET | `/api/object/{name}/final_touch` | – | Final Touch of the object: the base (chosen / default, resolved to a version), the strokes per frame, their hash. |
| PUT | `/api/object/{name}/final_touch` | object_edit | Choose the base (DECYZJA: any computed version, changeable at any time – the corrections stay, on the new base). |
| PUT | `/api/object/{name}/final_touch/finish` | object_edit | Global finishing of the Final Touch (levels / gamma / erode / blur, user decision 2026-10-22: moved here from the ViTMatte zone sliders): applied on the base be |
| PUT | `/api/object/{name}/final_touch/nuke` | object_edit | How the object's Final Touch goes to Nuke in the export's .nk (engine/ft_nuke, 2026-11-16): baked into the alpha, the finish as nodes, or the strokes as a RotoP |
| GET | `/api/object/{name}/final_touch/nk` | – | One section of the Final Touch palette as Nuke nodes (its „⇪ Export” button, 2026-11-16): part = grade (Grade → Dilate → Blur with the current values) / paint ( |
| PUT | `/api/object/{name}/final_touch/frames/{frame}` | object_edit | Replace the strokes of one frame (the UI appends a stroke, Ctrl+Z puts the previous list back); [] = none. |
| DELETE | `/api/object/{name}/final_touch/frames/{frame}` | object_edit | Remove the corrections of one frame (🔑✕ / Delete in the stage). |
| GET | `/api/object/{name}/final_touch/layer/{frame}` | – | Overlay of the frame's correction layer C (engine/final_touch.layer) as a BGRA proxy PNG: green where the strokes add (C > 0), red where they remove (C < 0), al |
| GET | `/api/object/{name}/final_touch/frame/{frame}` | – | Proxy PNG of base + corrections in `frame` – the alpha the export writes for source „Final Touch” (same loader, 8-bit rounding like view=alpha) |
| POST | `/api/object/{name}/recompute_dependents` | long_job | 'Przelicz zależne' (7.3): objects behind `name` marked 'do przeliczenia' are recomputed (core + final). |
| POST | `/api/library` | – |  |
| GET | `/api/library` | – |  |
| POST | `/api/library/add` | – |  |
| POST | `/api/object/{name}/import_review` | object_edit | Strokes from 'Przegląd klatek' (r = remove, g = add) -> user keys (core +/- strokes). |
| POST | `/api/queue` | – |  |
| POST | `/api/project/cleanplate_auto` | long_job | „Policz z ujęcia (auto)” (docs/PLAN_CLEANPLATE.md): median of the uncovered pixels of the shot (cores of ALL objects needed -> 409 with the missing ones), holes |
| GET | `/api/project/cleanplate_auto` | – | Clean plate state for the zone card: source (projekt / auto / brak), stats of the auto plate when built. |
| POST | `/api/queue/preset/all` | – | „Policz wszystko” (plan section 6): every object of the open project without a valid final result, front to back (depth_order) – core + „Przelicz finalnie” each |
| GET | `/api/queue` | – |  |
| POST | `/api/queue/run` | – |  |
| PUT | `/api/object/{name}/interval_engine` | object_edit | Propagation engine of the key interval containing `frame` (7.2 pkt 7): SAM 3.1 or CoTracker3 points. |
| POST | `/api/groups` | when_idle | New group entity (palette Obiekty): optionally with its first members (each joins with „Do grupy z:” = Core). |
| POST | `/api/group/{group}/members` | when_idle | Drop an object into the group (palette): it joins with Core; the whole recompute becomes stale (↻). |
| DELETE | `/api/group/{group}/members/{member}` | when_idle | Detach a member |
| PUT | `/api/group/{group}/members/{member}/source` | when_idle | „Do grupy z:” of a member (core / matanyone / videomama / vitmatte / ltx / version / final_touch). |
| PUT | `/api/object/{name}/mask_source` | when_idle | The mask of an object outside a group (user 2026-11-15) – the same choice as „Do grupy z:”: the default when it joins a group and in Cleanup (removed / protecte |
| PUT | `/api/group/{group}` | when_idle | Seam of the group (↻ on the whole recompute). |
| PUT | `/api/group/{group}/members/{member}/seam` | when_idle | A member's own seam reach (user decision 2026-10-29: hair small, a dress large); null = the group's seam |
| PUT | `/api/group/{group}/members/{member}/on` | when_idle | Palette checkbox of a member (user decision 2026-10-29): switched off = out of the group's composition (alpha, preview, whole recompute input – ↻), still a memb |
| DELETE | `/api/group/{group}` | when_idle | Ungroup: members become standalone (results untouched), the group's folder is deleted for good; with whole results and without ?confirm=1 -> 409 with the warnin |
| GET | `/api/group/{group}` | – | The group entity: members in depth order with what each brings („Do grupy z:” – source, version, current or ↻, the reason when there is nothing), the seam and t |
| GET | `/api/group/{group}/frame/{frame}` | – | Proxy PNG of the basic group matte (the composition of the members' chosen alphas with the seams, nothing computed; objects in front not subtracted) |
| GET | `/api/objects` | – | Objects in depth order; `has` = results on disk (core / final alpha of the active version) – the UI shows the masks of the OTHER computed objects as extra overl |
| PUT | `/api/objects/order` | when_idle | Depth order by drag & drop (spec 6 pkt 2, Z3): depth_order = index, atomically, no gaps / duplicates |
| POST | `/api/object` | – |  |
| GET | `/api/object/{name}/delete_info` | – | What deleting the object removes for good (for the confirmation dialog): its folder and, optionally, its exported masks (outputs.json). |
| PUT | `/api/object/{name}/rename` | when_idle | Rename an object: folder objects/<name> -> objects/<new_name> and every exact reference (parts, holdouts, groups) |
| DELETE | `/api/object/{name}` | object_edit | Delete an object: its folder is deleted from disk for good (user decision 2026-11-13: no project trash), entry removed from outputs.json, objects behind it that |
| POST | `/api/object/{name}/detect` | long_job (quiet) | Instance choice (SPEC 5.2 pkt 0): SAM3_Detect on the letterboxed proxy, candidates stored as PNG. |
| GET | `/api/object/{name}/candidate/{index}` | – | Z10: thumbnail of SAM 3.1 candidate `index` – the proxy crop of its bbox with the mask as alpha (BGRA PNG, longer side <= `side` px) for the candidate list of s |
| POST | `/api/object/{name}/choose` | object_edit | Chosen candidate -> automatic key (author=auto) at full plate resolution. |
| POST | `/api/object/{name}/crops` | long_job |  |
| GET | `/api/object/{name}/crops` | – |  |
| PUT | `/api/object/{name}/crops/{seg_id}` | object_edit | Manual frame of a segment (SPEC 5.2 pkt 8): stored with manual=true, snapped square, padding recomputed. |
| POST | `/api/plate/clean/analyze` | long_job |  |
| POST | `/api/plate/clean/preview` | long_job |  |
| GET | `/api/plate/clean/preview/file` | – |  |
| GET | `/api/object/{name}/merge_matte` | – |  |
| PUT | `/api/object/{name}/merge_matte` | – |  |
| GET | `/api/object/{name}/merge_matte/nk` | – | The palette's „Nuke” button (user request 2026-10-30): the group with the object's knobs, pasteable in Nuke (`log_curve` = `project` written as the profile's cu |
| GET | `/api/object/{name}/merge_matte/frame/{frame}` | – | k = the settings hash of the client (cache buster only); the knobs are read from final_touch.json. |
| GET | `/api/object/{name}/merge_matte/edge_bg/{frame}` | – | „Tło w krawędzi” (PLAN_FINAL_TOUCH §11 step 5): the region's edge band before (plain Merge over) and after MergeMatte with the object's knobs – {before, after:  |
| GET | `/api/project/color_profile` | – | The plate colour profile of the shot: project.json `color_profile` (auto / curve id), the curve in use with its reason. |
| PUT | `/api/project/color_profile` | – | Stored in project.json -> color_profile (never in the shot manifest); `auto` removes the override. |
| GET | `/api/object/{name}/vram_forecast` | – | Forecast of the stage's peak GPU memory against the Comfy card (engine/vram_forecast.py). |
| GET | `/api/cleanup` | – |  |
| PUT | `/api/cleanup` | – |  |
| POST | `/api/cleanup/prompt/check` | – | The description's warnings + the exact texts each model gets (engine/prompt_rules; user 2026-11-12) |
| GET | `/api/cleanup/prompt/assist` | – | „✨ Popraw opis”: is the local assistant there (what the install popup shows when not). |
| POST | `/api/cleanup/prompt/assist` | long_job | The proposal of the local assistant for the typed text (nothing saved – the palette shows it, „Wstaw” saves). |
| GET | `/api/cleanup/anchors` | – |  |
| POST | `/api/cleanup/anchors` | – | A new anchor at a frame: linked (follows the main one), original (the plate frame itself) or own. |
| POST | `/api/cleanup/anchors/projection` | – | Anchors from the card keys (user request 2026-11-12): projection anchors at every frame where a card key brings a new image (+ every `step`-th frame), each = th |
| PATCH | `/api/cleanup/anchors/{aid}` | – | Chosen generation, own description, or kind: main (the old main becomes linked), linked (attach again), own (DETACH: keeps the generation it showed), original |
| DELETE | `/api/cleanup/anchors/{aid}` | – | The anchor and its generations are deleted from disk for good (user decision 2026-11-13) |
| POST | `/api/cleanup/anchors/linked/generate` | long_job | Every linked anchor not made from the main one's chosen image yet (all=1: every linked one again, e.g |
| POST | `/api/cleanup/anchors/{aid}/generate` | long_job | A new generation of the anchor (a new random seed unless given; the description shown in the panel). |
| POST | `/api/cleanup/anchors/{aid}/import` | – | The user's own image as a new generation of the anchor (e.g |
| GET | `/api/cleanup/anchors/{aid}/gens/{name}` | – | raw=1: Qwen-Image-2.1's whole picture saved next to the generation (the pasted one when there is none). |
| DELETE | `/api/cleanup/anchors/{aid}/gens/{name}` | – | A generation deleted from disk for good (user decision 2026-11-13); the chosen one deleted = the newest remaining is chosen |
| POST | `/api/cleanup/versions/{version}/show` | – | Plate „cleaned” = the comp of this version (nothing recomputed) AND its settings come back for its object: model, description, seed and its anchors (user 2026-1 |
| DELETE | `/api/cleanup/versions/{version}` | – | A version deleted from disk for good (user decision 2026-11-13: no project trash any more) |
| POST | `/api/cleanup/versions/{version}/upscale` | long_job | „⤴ Upscale (SeedVR2)” (user 2026-11-11): the version's fill restored at the plate's scale by SeedVR2 as one clip (engine/inpaint_upscale.py) and laid back throu |
| POST | `/api/cleanup/versions/{version}/recompose` | long_job | „⟲ Złóż ponownie”: a NEW version from the stored model frames of `version` with the CURRENT composite band (grow_px, soft_px, band points / keys), holdout, leve |
| GET | `/api/cleanup/versions/{version}/model/{frame}` | – | The model's own picture of `frame` (model/m_####.png – what the video model drew, before the composite) at the viewer's preview size, stretched to the plate as  |
| GET | `/api/cleanup/versions/{version}/preview/{frame}` | – | Live preview of a recomposite: frame `frame` of `version` composited with the current band (+ the palette's unsaved grow_px / soft_px), the source version's lev |
| GET | `/api/cleanup/holdout/{name}/{frame}` | – | PNG at the viewer's preview size: R = the kept object's holdout alpha (its keep loader – the mask cleanup_masks.resolve picks), G = where it cuts the current pa |
| GET | `/api/cleanup/mask/{frame}` | – |  |
| POST | `/api/cleanup/run` | long_job |  |
| PUT | `/api/cleanup/band_keys/{frame}` | – |  |
| DELETE | `/api/cleanup/band_keys/{frame}` | – |  |
| PUT | `/api/cleanup/band_keys` | – |  |
| GET | `/api/cleanup/band/params` | – | The band points as stored (+ the slider's reference) – what Ctrl+Z restores. |
| PUT | `/api/cleanup/band/params` | – |  |
| GET | `/api/cleanup/band/{frame}` | – | The band points in `frame` as the editor shows them (carried by the removed objects, snapped to their outline, widths scaled by „Pas na zewnątrz”). |
| PUT | `/api/cleanup/band/{frame}` | – | Add a band point in `frame` or set its key there (position / outer width); the mask and the run use it. |
| GET | `/api/cleanup/band/{frame}/lines` | – | Preview lines of the C1 band in `frame` (plate px): the band ring (objects' outline + where the solid patch ends) and where the soft edge ends – the geometry of |
| DELETE | `/api/cleanup/band/{point_id}` | – | Remove a band point, or only its key in `frame` (?frame=N; the last key removes the point). |
| DELETE | `/api/cleanup/band` | – | The band keys of ALL points in `frame` (?frame=N), or every point (no frame = the plain band again). |
| GET | `/api/cleanup3d` | – | State of the Cleanup 3D mode: steps done / stale, camera report, cards with keys, engines and solve availability. |
| GET | `/api/cleanup3d/camera` | – | The solved 3D camera (cams.json); 404 before the first solve. |
| POST | `/api/cleanup3d/camera/solve` | long_job | Solve the 3D camera of the work range (COLMAP + CoTracker3 + bundle adjustment); moving objects are masked out. |
| POST | `/api/cleanup3d/camera/static` | – | C2 „Kamera nieruchoma” (user 2026-11-11): a locked-off shot – no COLMAP; one identity pose for every frame of the work range (engine/cleanup3d/solve.static_came |
| GET | `/api/cleanup3d/cards` | – | The cards (order = front to back). |
| PUT | `/api/cleanup3d/cards` | – | Replace the card list (list order = card order). |
| DELETE | `/api/cleanup3d/cards/{card}` | – | Remove a card: its entry, keys and extent are deleted from disk for good (user decision 2026-11-13). |
| POST | `/api/cleanup3d/cards/propose` | – | Cards the solve suggests: the ground plane, the sea level (when the shot has a horizon) and the sky |
| PUT | `/api/cleanup3d/cards/{card}/extent` | – | Set a card's extent: areas (curves + brush layers with png_data) drawn in `frame`; empty list = the whole card. |
| GET | `/api/cleanup3d/cards/{card}/extent/{area_id}/png` | – | A brush layer of a card's extent as 8-bit PNG (white = area). |
| GET | `/api/cleanup3d/depth_sources` | – | Depth maps a depth card can use: the tool's DepthCrafter result (outputs.json) and the shot's depth plate. |
| POST | `/api/cleanup3d/cards/{card}/depth` | – | Make `card` a depth card: the depth of reference frame `frame` from `source` stored as the card's snapshot, fitted to the camera (the solve's points, else the p |
| GET | `/api/cleanup3d/keys` | – | Keys of every card (or of `card`) with their versions. |
| POST | `/api/cleanup3d/keys/{card}` | – | Add keys of a card at `frames` or every N frames of the work range; the first version is the plate frame. |
| POST | `/api/cleanup3d/keys/{card}/auto` | long_job | Key mode „automat”: key frames where the projection error of the card grows (keymodes.auto_key_frames); the settings and the mode are saved on the card, the key |
| POST | `/api/cleanup3d/keys/{card}/master` | long_job | Key mode „jeden klucz”: ONE master key of the card accumulated from the clean pixels of every frame (keymodes.build_master) on its own canvas larger than the fr |
| POST | `/api/cleanup3d/keys/{card}/seedvr2` | long_job | SeedVR2 on the card's keys (active versions with a fill) as ONE clip of one crop around the patches, pasted back over the plate (keys.patch_on_plate): a new ver |
| DELETE | `/api/cleanup3d/keys/{card}/{frame}` | – | Delete a key (all its versions) from disk for good (user decision 2026-11-13). |
| GET | `/api/cleanup3d/keys/{card}/{frame}/image` | – | PNG preview of the active (or given) version of a key. |
| POST | `/api/cleanup3d/keys/{card}/{frame}/export` | – | Write the key's active version for painting outside (png / png16 / exr) and return the path. |
| POST | `/api/cleanup3d/keys/{card}/{frame}/import` | – | A painted file of this machine as a NEW version of the key (409: other size, unreadable, not RGB). |
| POST | `/api/cleanup3d/keys/{card}/{frame}/undo` | – | Make the previous version of the key active (files stay). |
| PUT | `/api/cleanup3d/keys/{card}/{frame}/active` | – | Make a version of the key active (the C4 version list; files stay). |
| DELETE | `/api/cleanup3d/keys/{card}/{frame}/versions/{version}` | – | Delete one version of the key from disk for good (its number is never given again) |
| GET | `/api/cleanup3d/keys/{card}/{frame}/areas` | – | The regeneration areas of a key (layers in order). |
| PUT | `/api/cleanup3d/keys/{card}/{frame}/areas` | – | Replace the regeneration areas of a key; a brush layer without png_data keeps its file. |
| GET | `/api/cleanup3d/keys/{card}/{frame}/areas/{area_id}/png` | – | A brush layer of a key's areas as 8-bit PNG (white = area). |
| POST | `/api/cleanup3d/keys/{card}/{frame}/regen` | long_job | Regenerate the marked areas of a key (LaMa, Qwen-Image-2.1 or FLUX 3 Image Klein + LoRA Restore) into a new active version. |
| POST | `/api/cleanup3d/rebase_keys` | long_job | Keys that are whole frames of a model (a video inpainter repaints everything) -> the PLATE + that fill only where the removed objects of step C1 are (grown, sof |
| POST | `/api/cleanup3d/void` | long_job | VOID once over the work range (input video + quadmask from the objects of step C1), the frames kept in <work>/inpaint/void/<run>/; with `keys` the cards' existi |
| POST | `/api/cleanup3d/void/keys` | long_job | Key versions from the last VOID run (no new VOID pass): the plate + the VOID fill (upscaled to the plate) only where the objects of step C1 and their traces are |
| POST | `/api/cleanup3d/mark/{frame}` | – | Areas drawn in plate frame `frame` -> appended (as closed curves in the key's pixels) to the areas of the keys that frame is made of: the frame's own key, else  |
| POST | `/api/cleanup3d/render` | long_job | Project the keys onto the frames and compose them into the plate: a new Cleanup version, shown as plate „cleaned”. |
| GET | `/api/cleanup3d/view/{kind}/{frame}` | – | PNG preview of plate frame `frame`: `key` = the key of `card` shown at that frame (its own key, else the nearest one – header X-C3D-Key-Frame), `proj` = the key |
| PUT | `/api/cleanup3d/render_settings` | – | Save the patch blend of C5 (used by the next render; the live preview applies it in the browser). |
| GET | `/api/cleanup3d/blend/{frame}` | – | Inputs of the live blend preview of C5 at preview size (render.blend_inputs): `part=fill` = the keys projected onto the frame (plate where they do not cover), ` |
| GET | `/api/cleanup3d/band/params` | – | The patch band points as stored (+ the slider's reference) – what Ctrl+Z restores. |
| PUT | `/api/cleanup3d/band/params` | – | Ctrl+Z of a patch band edit (static/bandpoints.js): the points and their reference exactly as given. |
| GET | `/api/cleanup3d/band/{frame}` | – | The patch band points in `frame` as the editor shows them (position carried by the removal objects' motion, snapped to the object part of the patch hole; widths |
| PUT | `/api/cleanup3d/band/{frame}` | – | Add a patch band point in `frame` or set its key there (position / outer width); the next render uses it. |
| GET | `/api/cleanup3d/band/{frame}/lines` | – | The preview lines of the patch band in `frame` (engine/bandsys.area_lines via band.lines, plate px): the band edges and where the outer feather ends, on the pre |
| DELETE | `/api/cleanup3d/band/{point_id}` | – | Remove a patch band point, or only its key in `frame` (?frame=N; the last key removes the point). |
| DELETE | `/api/cleanup3d/band` | – | The patch band keys of ALL points in `frame` (?frame=N), or every point (no frame = the plain ramp again). |
| GET | `/api/cleanup3d/horizon` | – | The user's horizon lines {frame: curve} and the horizon settings of the render. |
| GET | `/api/cleanup3d/horizon/{frame}` | – | The horizon of plate frame `frame` measured on the live preview's grid: the cards' predicted horizon, the plate's line (the user's / automatic), the projection' |
| PUT | `/api/cleanup3d/horizon/{frame}` | – | Save the user's horizon line of `frame` (open curve, plate px); between drawn frames the render interpolates. |
| DELETE | `/api/cleanup3d/horizon/{frame}` | – | Remove the user's horizon line of `frame` (the automatic detector takes the frame again). |
| GET | `/api/cleanup3d/timeline` | – | Numbers of the timeline rows for the work range: per card per frame `share` (mean weight the card fills in the frame) and `miss` (part of what the card shows th |
| POST | `/api/cleanup3d/keys/{card}/{frame}/move` | – | Move a key to frame `to` (timeline drag): the new key = the target plate frame + the source key's active version projected through the 3D camera onto the target |
| PUT | `/api/cleanup3d/cards/{card}/off` | – | Turn a card off for good (left out of the render, the previews and the Nuke export – the result becomes stale) or back on |
| PUT | `/api/cleanup3d/cards/{card}/solo` | – | Solo a card or take its solo back (P16, like After Effects): when any card is soloed, only the soloed cards that are not off go into the render, the previews an |
| POST | `/api/cleanup3d/export_nuke` | – | Write a Nuke script rebuilding the projection (Camera3 + .chan, cards as Card3D / sky Sphere, keys through Project3D, lens STMaps, Merge over the plate through  |
| GET | `/api/project/shutter` | – |  |
| POST | `/api/project/shutter/measure` | long_job |  |
| DELETE | `/api/project/shutter` | – |  |
| GET | `/` | – |  |
| POST | `/api/plate/qa` | long_job (quiet) | Z26: control of damaged frames of a plate (missing / unreadable / NaN / size / exposure jump / image jump / duplicate) -> matte_desk/plate_qa/<plate>.json with  |
| GET | `/api/plate/qa/{plate}` | – |  |
| PUT | `/api/plate/qa/{plate}/decisions` | – | Z26: the user's decision per flagged frame (mean / flow / gain / keep) – stored in the report; the gate of stage 1 opens once every flag has one and the repairs |
| GET | `/api/plate/qa/{plate}/preview/{frame}` | – | Z26: before / after of a repair (JPEG, display conversion like the proxy, longest side <= `side`): method orig = the source frame (404 when missing), mean / flo |
| POST | `/api/plate/fix` | long_job | Z26: write `<plate dir>_fix/` (repaired frames in the source encoding, the rest hard links, README) and switch the plate in project.json to it (manifest untouch |
<!-- routes:end -->
