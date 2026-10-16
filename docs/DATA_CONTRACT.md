# DATA_CONTRACT – MatteDesk data formats (SPEC 16)

Generated from `engine/contract.py` (`python -m engine.contract docs`) – the same schemas are used for validation.

## Rules
- Dependent tools (e.g. Camera3D + Mesh) **only read** MatteDesk data; they write into their own subfolder (`<work>/camera3d/`).
- Every JSON has a `schema` field = `matte_desk/<name>@<version>`; an incompatible change = new version + converter.
- Coordinate system: plate pixels, origin in the top-left corner, y pointing down. Conversion to Nuke: x_n = x + 0.5, y_n = plate_h − 0.5 − y (OpenCV: pixel centre at integer coordinates).
- Frame numbers = plate frame numbers everywhere.
- Masks: full frame, linear alpha 0–1; format per `project.json` → `export_format` (auto = 8-bit PNG `<obj>_####.png` for 8-bit sources – MP4 / PNG / JPG, EXR `rgb = a` half ZIP `<obj>_####.exr` for EXR; `png` / `exr` forces it). The patterns in `outputs.json` have the real extension. Multi-EXR (`<obj>.alpha`, `union.alpha`, optionally `depth.disparity` float32) only with EXR.
- Depth: `depth.disparity` = relative disparity (closer = larger, arbitrary scale) + a copy in `R`; not Z.
- MatteDesk always updates `outputs.json` on export.

## `matte_desk/project@1.0` – Shot project (`<work>/matte_desk/project.json`)

- `schema` (string, required)
- `manifest_path` (string, required)
- `work_dir` (string, required)
- `shot` (string, required)
- `colorspace` (string)
- `fps` (number)
- `first_frame` (integer, required)
- `last_frame` (integer, required)
- `plate_w` (integer, required)
- `plate_h` (integer, required)
- `plates_added` (object) – plates added in the tool (the manifest is read-only)
- `plates_hidden` (array)
- `export_dir` (string) – export folder chosen by the user (missing = `<work>/matte_desk/export`)
- `export_format` (enum) ∈ ['png', 'exr'] – export mask format; missing = auto (8-bit PNG for 8-bit sources, EXR for EXR)
- `mask_plate_denoise` (enum) ∈ ['on', 'off'] – plate-to-mask denoise (4b, core + final); missing = auto (on for EXR / > 8 bit, off for 8-bit); defaults.json mplate.enabled = false disables it everywhere

## `matte_desk/outputs@1.0` – Results registry (`<work>/matte_desk/outputs.json`)

- `schema` (string, required)
- `shot` (string)
- `frame_range` (array)
- `plate_size` (array)
- `objects` (array, required)
  - `name` (string, required)
  - `depth_order` (integer, required)
  - `matte` (string, required) – #### pattern, full frame, rgb=a
  - `core` (string/null)
  - `version` (string, required)
  - `qa_status` (enum, required) ∈ ['ok', 'flags', 'none']
  - `date` (string, required)
  - `params_hash` (string, required)
  - `edge_mode` (string/null)
- `mattes_all` (string) – multi-EXR: <obj>.alpha, union.alpha, [<parent>_full.alpha], [depth.disparity]
- `groups` (object) – objects composed of parts (part_of): <parent> -> {matte: pattern of the whole without seams, members: [parent, parts...]}; the parts sum to the whole
- `mattes_all_channels` (array) – multi-EXR layers in depth order (without `.alpha`)
- `mattes_nk` (string) – Nuke script with the mask passes (MASKI backdrop, anchors + stamps) – File > Import Script
- `export_dir` (string) – export folder (default `<work>/matte_desk/export`; chosen in project.json)
- `export_format` (enum) ∈ ['png', 'exr'] – mask format of the last export (extension of the `matte` patterns); `mattes_all` only with exr
- `camera2d` (object)
  - `json_path` (string)
  - `nk_path` (string)
- `depth` (object)
  - `path` (string, required)
  - `kind` (enum, required) ∈ ['disparity', 'z']
  - `channel` (string, required)
  - `range` (array, required)
  - `method` (string, required)

## `matte_desk/camera2d@1.0` – 2D camera motion (`<work>/matte_desk/camera/camera2d.json`)

- `schema` (string, required)
- `shot_name` (string)
- `ref_frame` (integer, required)
- `plate_w` (integer, required)
- `plate_h` (integer, required)
- `fps` (number)
- `motion_type` (string, required)
- `residual_median_px` (number)
- `residual_p95_px` (number)
- `anchor_drift_px` (number)
- `has_parallax` (boolean)
- `frames` (array, required)
  - `frame` (integer, required)
  - `homography` (array, required) – 3x3, plate pixels (OpenCV: pixel centre at integer coordinates), y pointing down, frame -> reference

## `matte_desk/object@1.0` – Object (`objects/<obj>/object.json`)

- `schema` (string, required)
- `name` (string, required)
- `depth_order` (integer, required)
- `detect` (object, required)
- `flags` (array)
- `edge_mode` (enum, required) ∈ ['auto', 'backlit', 'matanyone', 'core', 'ltx']
- `edge_mode_auto` (string)
- `part_of` (string/null) – parent, when the object is a part of another one (e.g. Figure's dress)
- `band_scale` (number)
- `shift_px` (number)
- `band_out_px` (number)
- `band_in_px` (number)
- `temporal_smoothing` (enum) ∈ ['weak', 'normal', 'strong']
- `ma_scale_mode` (enum) ∈ ['auto', '1.0', '0.75', '0.5'] – MatAnyone scale in "Compute final" („Przelicz finalnie”): auto = 9-frame trial by VRAM (1.0 → 0.75 → 0.5), a number = fixed (no trial)
- `key_band_same` (boolean) – key frames get the same edge band as the other frames (instead of ±3 px from the spec); missing = on
- `core_soft_start` (boolean) – the MatAnyone edge band starts from the SAM3 probability maps (CoreCombine core_mode=soft); missing = on
- `tool` (object, required)

## `matte_desk/crops@1.0` – Crop segments (`objects/<obj>/crops.json`)

- `schema` (string, required)
- `segments` (array, required)
  - `seg_id` (string, required)
  - `first` (integer, required)
  - `last` (integer, required)
  - `overlap_in` (integer)
  - `overlap_out` (integer)
  - `crop` (array, required) – [x, y, side, side] of the virtual square
  - `pad` (array, required)
  - `scale_to_1008` (number, required)
  - `manual` (boolean)

## `matte_desk/run@1.0` – Result version (`objects/<obj>/runs/v###/run.json`)

- `schema` (string, required)
- `run` (string, required)
- `edge_mode` (string, required)
- `ma_scale` (number)
- `ma_scale_mode` (string) – auto | 1.0 | 0.75 | 0.5 (object setting)
- `ma_node` (object) – inputs of MatAnyone2RunKeyframes >= 1.1.0 (vram_release_frac, max_mem_frames 0 = auto); empty = older node
- `segments_def` (array, required)
- `vram_log` (array)
- `hygiene` (object)
- `soft_start` (object) – soft start of the band from the SAM3 maps: frames (frames with a map in CoreCombine), maps, node (the node has core_mode), note; empty = no maps / disabled

## `matte_desk/edge_fix@1.0` – Edge measurement / correction of a version (`objects/<obj>/runs/v###/edge/edge.json`; per-frame measurements `e_####.npz`)

- `schema` (string, required)
- `mode` (string, required)
- `fix` (array, required)
- `attempts` (integer)
- `applied_from` (string)
- `skipped` (array)

## `matte_desk/key@1.0` – Key (`objects/<obj>/keys/k_####.json`)

- `schema` (string, required)
- `frame` (integer, required)
- `kind` (string, required)
- `author` (enum, required) ∈ ['user', 'auto']
- `mask_path` (string, required)

## `matte_desk/zone@1.0` – Zone (`objects/<obj>/zones/z_##.json`)

- `schema` (string, required)
- `zone_id` (string, required)
- `mode` (enum, required) ∈ ['add', 'remove']
- `criterion` (enum, required) ∈ ['dark', 'bright', 'color', 'all']
- `frame` (integer, required)
- `soft` (boolean)

## `matte_desk/depth@1.0` – Depth report (`<work>/matte_desk/depth/depth_report.json`)

- `schema` (string, required)
- `kind` (string, required)
- `method` (string, required)
- `export_pattern` (string, required)
- `value_min` (number, required)
- `value_max` (number, required)
