# MatteDesk – changelog

## 0.42 – 2026-10-26
- **Version comparison for stages 3–7:** one A / B / difference panel (keys A / B / D, Q = band preview), a list of versions with
  a trash button each (moved to the project trash after showing the size; a version in use cannot be deleted and says why),
  a click on a version makes it active; the picked version always reaches the viewport.
- **Alpha / Comp only with a result:** in the edge stages (3 / 4 / 5 / 7) the Alpha and Comp tabs open once the stage has its own
  result (a version, a "Preview frames" run or a picked version) – an uncomputed stage no longer shows the SAM 3.1 core as its
  cut-out; switching stages quickly no longer leaves them on from the previous stage. The view you pick rules in every stage;
  no green tint over Comp / Alpha.
- **Smooth playback in Alpha / Comp:** the preload and the background sweep load the mask that is actually drawn (the stage's
  mask), so Play no longer stutters.
- **Stages 1–2:** the Plate view shows the selected object's core in every frame like the Core view (the eye in the Objects
  palette hides it). Stage bar grouped into Core (1 + 2) and Edges (3 / 4 / 5); stage 5 panel tidied.
- **Stage 5:** one whole-outline zone per object (a second one is refused, "New zone" is disabled while one exists); "Preview
  frames" stays visible in Alpha even when the stage has an older version.
- **Band points:** an object behind another one no longer waits minutes before CoTracker3 starts (the front object's mask is
  read once per frame); the tilt of the points is kept on crops (MatAnyone2, VideoMaMa, LTX-2.5); default bands and Sapiens2
  proposals never cross the core edge.
- **GPU memory:** one release before every stage (our GPU workers + ComfyUI's models), waiting until ComfyUI has really freed
  its memory, and a warning whenever ComfyUI's GPU memory spills into system RAM.
- **Fixes:** the "Update changes" progress counts frames (one key showed "2/2"); the SAM 3.1 core is stabilised once per run
  (cached on disk); the mask 1+2 cleanup keeps holes; reading and saving an object is safe when two
  tasks touch it at once (Windows: an object could briefly look as if it had no mask).

## 0.41 – 2026-10-25
- **One edge band in every stage:** MatAnyone2 (stage 3), VideoMaMa (4), ViTMatte (5) and LTX-2.5 (7) use the same band –
  the same sliders (outer band, core inset, both feathers), the same band points and the same defaults (40 / 2 px, feathers
  8 / 8); at equal values the band is identical in every stage. Sliders and points are shared by the object with one
  "detach" for a stage that needs its own. VideoMaMa composes only through the band; LTX-2.5 "raw" stays the only mode
  without a band.
- **Band points:** tilt the whisker pair round the point, grab the whole whisker line, Shift = symmetric whiskers; the points
  hide with the band preview; tracking runs in one long-lived CoTracker3 process (weights loaded once) and an edit cancels
  the tracking of the previous point set – a 5-frame preview right after editing points went from ~6 min to ~1–2 min; the
  job bar says "Tracking band points (CoTracker3)" while it waits.
- **Whole-outline zone (stage 5):** the band is the trimap's unknown area – the model replaces the core in the whole band,
  so inward whiskers cut out background the core had taken for hair.
- **Faster, lighter edge stages:** temporal smoothing streams a sliding window with parallel optical flow (a 4K hair zone
  needed ~41 GB of RAM, now a few GB; ~1.7x faster; results bit-identical), "Abort" works during the smoothing, all
  ViTMatte tiles of a frame go to the model in one request, stage 3 is finished frame by frame.
- **Hair preset default:** ViTMatte at 1/2 resolution (504 px), fine tune 0.5, no temporal smoothing (flow_mean r3 x2 when
  switched on).
- **Final Touch:** corrections drawn on the Core overlay, no black frame between strokes, faster stroke preview; it is now the
  only motion blur (the per-stage alpha motion blur was removed).
- **Experiments:** the mask 1+2 can be the base once the core is computed.
- **UI:** "REPAIR" button (wrench) for plate cleanup, "Update changes" instead of "Recompute all", the logo badge opens About.

## 0.40 – 2026-10-24
- **Viewport overlays on the GPU:** mask tint, outlines, the Alpha view and the comp are drawn on the graphics card (the CPU path
  stays as a fallback) – smoother scrubbing and zoom on 4K shots.
- **App-styled controls:** sliders (amber fill, grey when disabled), checkboxes, radio buttons, selects, the file button and
  scrollbars match the app; native controls are dark.
- **Layers panel:** the parts of a group are indented under the main object (guide line, main object in bold).
- **Core:** a warning when an object in front has no core yet (nothing is subtracted there); trimming an object's life (no key in
  the hidden frames) keeps its core valid.
- **Edge band:** editing band points is faster; "Propose band (Sapiens2)" spaces the points with a ramp and respects inset 0.
- **"Report a problem":** the page no longer freezes on long texts.
- **Fixed: plates carried over from the previous shot.** The Open project dialog kept the backlit / clear plate paths of the
  previously opened shot when the new one had none, and they were saved into the new project. The fields now take the shot's
  values or are cleared, and opening or creating a project refuses a backlit / clear plate whose resolution differs from the beauty.
- **README:** current stage list (Final Touch, BiRefNet / ViTMatte, shared edge band, Sapiens2) and the demo video.

## 0.39 – 2026-10-23
- **Fixed: "Report a problem" → "Send" did nothing visible** before the preview / consent. Now it asks for a description
  when empty, builds the preview by itself and outlines the consent box in red; without consent the button looks dimmed.
- **Final Touch on a hard base:** with the "mask 1+2" base (0 and 1 only) the palette explains that black / white / gamma
  change only semi-transparency (here: soft brush strokes) and dims those sliders; erode, blur and motion blur act on the
  whole alpha. Pick a soft-edged base (MatAnyone2, VideoMaMa, ViTMatte / BiRefNet, LTX-2.5) to grade the whole matte.

## 0.38 – 2026-10-23
- **Fixed: "No key in the work area"** – the work area is shared by all objects (e.g. set while working on a second object),
  so "Compute all" of an object whose keys lie outside it was refused with no hint why. The dialog now lists the object's
  key frames and offers "Extend the range to F…–…" (to the nearest key), which recomputes right away. An object without
  any key gets "place SAM 3.1 points".

## 0.37 – 2026-10-23
- **Fixed: "Reload ComfyUI"** (the lost-connection banner) closes a hung ComfyUI again – in 0.36 it failed with
  "No module named 'psutil'". psutil is now bundled with the installer; a copy from git / ZIP whose environment was made
  earlier closes the process with the Windows `taskkill` instead. A process that is not ComfyUI is still never closed.

## 0.36 – 2026-10-23
- **New: "Propose band (Sapiens2)"** – Sapiens2-seg-0.4B recognises body parts in the frame and sets the band points of the
  object band: wide on hair, narrow on clothes, thin on skin (never less than the automatic width). People only; one key frame,
  CoTracker3 tracks the points, you fix them with the whiskers. Optional model (Sapiens2 License, commercial use allowed).
- **ComfyUI lost connection:** a banner in the middle of the screen with "Reload ComfyUI"; the footer shows the live ComfyUI status.
- **Band preview:** dashed lines only in stages 3 / 4 / 5 / 7, also while a slider is held.
- **Doctor – ComfyUI Desktop paths:** the ComfyUI folder, the models folder and the port of ComfyUI Desktop can be changed in
  "ComfyUI in use" (empty field = detected automatically, the detected value is shown greyed out).
- **Doctor – two ComfyUIs on one machine** (Desktop, portable, from git): the Doctor asks the running ComfyUI for its folder and
  warns when a DIFFERENT install answers on the port – on the configured port or on another one.
- **Doctor opens faster:** the checks that start ComfyUI's Python run side by side and are prepared in the background at start-up
  (first open ~24 s → under 1 s).
- **Doctor – delete model weights:** installed models get a 🗑 button (LTX-2.5, VideoMaMa, MatAnyone2, ViTMatte, Sapiens2,
  MiniMax-Remover, LaMa). It lists every file with its size first and deletes only the weights – ComfyUI nodes stay; the stage that
  needs the model is then locked. SAM 3.1, CoTracker3 and BiRefNet_HR-matting / Lucida have no bin.
- **BiRefNet_HR-matting / Lucida is now the required stage-5 edge engine** (MIT weights, included in "Install all missing");
  ViTMatte (non-commercial weights) is optional.
- **"Propose band (Sapiens2)"** is always shown under "👁 band preview" in stages 3 / 4 / 5 / 7, with a new icon; without the
  model it is greyed out with a hint to install it in the Doctor.
- **Report a problem:** after "Send" the window closes and a confirmation with OK opens (before, the status line was out of view).
- **Update window:** a copy downloaded as a source ZIP gets a link to the new ZIP and to the installer instead of "git pull";
  a git copy is told to start the new `start_v<version>.bat`.

## 0.35 – 2026-10-22
- **Final Touch (stage 8):** paint corrections over any computed alpha – add / remove / eraser with pen pressure, kept as strokes
  per frame; black / white / gamma / erode / blur and motion blur whose strength is measured on the plate; export source "Final Touch".
- **One edge band in every stage with a band** (3 MatAnyone2, 4 VideoMaMa, 5 ViTMatte / BiRefNet, 7 LTX-2.5): the sliders size the
  band, band points tracked by CoTracker3 shape it, inner / outer feathers blend into the core, preview = dashed band lines.
- **Shared object band:** one set of band points per object, read by every stage with its own sliders; "detach" gives a stage its
  own band, "set as shared" makes it the object's band. Older projects are migrated on opening.
- **Stage 5 engine:** ViTMatte or BiRefNet (arrow next to the stage); a preview of a few frames is kept per engine.
- **Deflicker:** one button with settings from a measurement of the plate, 1024 px preview, no ghosts of moving objects, choose
  the folder of the EXR output. Large plates no longer run out of memory; alignment limits its workers by free RAM.
- **Report a problem:** an anonymised report you see in full and agree to before it is sent.
- **Faster:** stage 3 band lines 1.3 s → 0.2 s per new frame (the mask 1+2 is saved right after stage 1); stepping frames in
  stage 5 no longer unloads the ComfyUI models.
- **Fixed:** the eye of the selected object hides every mask overlay, stage previews included; ← / → step frames even when a slider
  has the focus; the Alpha view is black and white.

## 0.32 – 2026-10-18
- **Video plates:** an .mp4 / .mov / .mkv / .avi / .mxf file given as a plate (project wizard, Manage plates, reopening a project)
  is decoded once into a PNG sequence in `<shot>/matte_desk/source/<name>/` (8- or 16-bit like the source) and used everywhere.
  Before, "Proxy" of a video plate failed with "SyntaxError … is not valid JSON". Projects that stored the video path are fixed at "Proxy".
- **Project wizard:** a file button 🎞️ next to every plate's folder button – pick a video, or one frame of a sequence (the whole
  sequence is used). Clear plate, backlit and depth can be videos too.
- **Update window:** the release notes are plain text and scroll (at most 40 % of the window height).

## 0.31 – 2026-10-17
- **Windows installer** (`MatteDesk-0.31-setup.exe` in Releases): one file with its own Python and all dependencies, installs for
  your user only (no admin rights), Start-menu / desktop shortcut, uninstall in Windows Settings → Apps.
- **App window:** MatteDesk opens in its own window without an address bar in your default browser (Chrome, Edge, Brave, Comet…
  as an app window; Firefox as a tab with a tray icon; Edge when nothing is detected), with no browser prompts. Switch in the Doctor.
- **"Do you already have ComfyUI?"** – Yes: found automatically (a running ComfyUI with its port and Python, ComfyUI Desktop,
  portable or from git); No: downloads the official **ComfyUI portable** (resumable, SHA-256 checked) into a folder you choose.
- **ComfyUI portable** is a supported variant everywhere: Doctor, node installs via its `python_embeded`, autostart like `run_nvidia_gpu.bat`.
- **Downloads** of all models resume after an interruption and are checked against the SHA-256 published by Hugging Face / GitHub.
- **Updates:** checked at start (at most once a day) and with "Check for updates" in About; a short list of changes, a link to this
  changelog and **Update** (downloads and runs the new setup, settings stay). Can be switched off in the Doctor.
- The source ZIP points to the installer (`INSTALL_WINDOWS.txt`, hint in `start_v….bat`); the shortcut offers to stop an older
  MatteDesk server still running.
- **"👁 Preview frames"** in stages 3 MatAnyone, 4 VideoMaMa and 5 ViTMatte: a few frames around the current one before the
  long run on the whole shot; never a stage result (no version, nothing unlocked).
- **Edge sliders, the same everywhere** (MatAnyone, VideoMaMa, ViTMatte): outer band, core inset – the engine replaces the
  hard core edge in the band – and **inset feather** (soft blend into the core). The band updates instantly while a slider
  is dragged, also with the preview switched off; sliders never start a model.
- ViTMatte and VideoMaMa open right after stages 1 + 2 (their base is chosen inside the stage); bases without a result are
  disabled. ViTMatte "whole outline" is the default zone; without exclusions its area comes straight from the core.
- "Mask: …" label over the image (which mask is shown, version and frames); "Run again" when a stage has a result;
  "Check for updates" in About; the cut-out view switches band previews off; eye on the selected object in Objects.
- Fixed: previews disappearing while scrubbing the timeline, the ViTMatte frame preview pasting the hard SAM3 core edge,
  an empty white cut-out outside a version's frames, false "different version" of our nodes, overlapping notes.
- Fixed: "Cancel" now stops MatAnyone in ComfyUI between frames (MatAnyone2-Keyframes 1.2.2; before, the node finished the
  cancelled pass and new runs failed with "ComfyUI busy"); a backlit / clear plate of another resolution than the beauty
  (e.g. from another shot) is never used (QA of every stage crashed), rejected when set and marked ⚠ in stage 0.

## 0.3 – first public release
