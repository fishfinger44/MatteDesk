# MatteDesk – changelog

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
