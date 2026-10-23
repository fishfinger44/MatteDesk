# MatteDesk – changelog

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
