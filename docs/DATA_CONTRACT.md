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
  - `members` (array) – a group entity (PLAN_GROUP_COMPOSE): its members – the matte is the group's whole (a stage computed on the whole, Final Touch, or the composition of the members with seams); one layer at its front-most member
  - `final_touch` (object) – exported from Final Touch: the base version and the hash of the corrections
    - `base` (string)
    - `corrections_sig` (string)
    - `nuke` (object) – Final Touch in the .nk (engine/ft_nuke): mode bake | grade | paint; for grade / paint the column reads `base_matte` (base [+ strokes], not cut by the objects in front) and `front_matte` (max of the front layers) and rebuilds the matte with live nodes; `requested` + `why` = a node mode that fell back to bake
      - `mode` (enum) ∈ ['bake', 'grade', 'paint']
      - `requested` (string)
      - `why` (string)
      - `base_matte` (string)
      - `front_matte` (string)
- `mattes_all` (string) – multi-EXR: <obj>.alpha (objects and groups), union.alpha, [depth.disparity]
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
- `kind` (enum) ∈ ['group'] – a group entity (PLAN_GROUP_COMPOSE): no core of its own, its mask = the composition of `members`
- `members` (array) – group: its objects (one group per object, no nesting)
- `seam_px` (integer)
- `group` (string/null) – object: the group it belongs to
- `group_source` (object/null) – object in a group: „Do grupy z:” {source: core | matanyone | videomama | vitmatte | ltx | version | final_touch, run}
- `group_on` (boolean) – object in a group: false = switched off for now (out of the group's composition), default true
- `group_seam_px` (integer/null) – object in a group: its own seam reach px (null = the group's seam_px)
- `band_scale` (number)
- `shift_px` (number)
- `band_out_px` (number)
- `band_in_px` (number)
- `temporal_smoothing` (enum) ∈ ['weak', 'normal', 'strong']
- `ma_scale_mode` (enum) ∈ ['auto', '1.0', '0.75', '0.5'] – MatAnyone2 scale in "Compute final" („Przelicz finalnie”): auto = 9-frame trial by VRAM (1.0 → 0.75 → 0.5), a number = fixed (no trial)
- `key_band_same` (boolean) – key frames get the same edge band as the other frames (instead of ±3 px from the spec); missing = on
- `core_soft_start` (boolean) – the MatAnyone2 edge band starts from the SAM 3.1 probability maps (CoreCombine core_mode=soft); missing = on
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
- `soft_start` (object) – soft start of the band from the SAM 3.1 maps: frames (frames with a map in CoreCombine), maps, node (the node has core_mode), note; empty = no maps / disabled

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

## `matte_desk/camera3d@1.0` – 3D camera of the shot (`<work>/inpaint/camera3d/cams.json`)

- `schema` (string, required)
- `width` (integer, required)
- `height` (integer, required)
- `K` (array, required) – 3x3 intrinsics, plate pixels (one focal for the shot)
- `dist` (array, required) – OpenCV distortion (k1, k2, p1, p2)
- `frames` (array, required)
  - `frame` (integer, required)
  - `R` (array, required) – 3x3, X_cam = R @ X_world + t (x right, y down, z forward)
  - `t` (array, required)
- `source` (string)
- `report` (object) – reprojection px, key projection error, horizon, worst frames

## `matte_desk/cards3d@1.0` – Cards of the Cleanup projection (`<work>/inpaint/cards.json`)

- `schema` (string, required)
- `cards` (array, required)
  - `id` (string, required)
  - `name` (string)
  - `kind` (enum, required) ∈ ['plane', 'sky', 'depth']
  - `plane` (object/null) – n . X + d = 0, n toward the cameras; null for the sky; for a depth card the approximate plane of its surface (previews, Nuke export)
  - `adjust` (object) – dh along the normal, pitch / roll in degrees
  - `extent` (array) – areas (closed Bezier curves / brush layers) drawn once in extent_frame; empty = the whole card
  - `extent_frame` (integer/null)
  - `seam_px` (number)
  - `order` (integer)
  - `color` (string)
  - `key_mode` (enum) ∈ ['manual', 'every', 'auto', 'one']
  - `key_every` (integer)
  - `warps` (object) – key frame -> edge warps (src / dst curves)
  - `key_auto` (object) – key_mode auto: threshold, min_gap, max_gap, max_keys (missing = defaults)
  - `off` (boolean) – P16 timeline eye: true = the card is left out of the render, the previews and the Nuke export (default false)
  - `solo` (boolean) – P16 timeline solo, like After Effects: when any card has solo true, only the soloed cards that are not off go into the render, the previews and the Nuke export; written only when true (default false)
  - `depth` (object) – only kind depth (P18): the surface of the reference frame's depth map
    - `frame` (integer, required) – reference plate frame (the depth map's frame)
    - `source` (enum) ∈ ['computed', 'plate'] – computed = DepthCrafter of the tool (outputs.json depth), plate = the shot's depth plate
    - `conv` (enum) ∈ ['disparity', 'z'] – disparity: 1/z = scale d + offset; z: z = scale d + offset (z = camera-z depth of the reference view)
    - `scale` (number)
    - `offset` (number)
    - `mul` (number) – user slider „Skala głębi”: z = mul z0 + shift
    - `shift` (number) – user slider „Przesunięcie” (scene units)
    - `edge` (number) – relative depth jump inside a pixel quad that splits the surface (occlusion edge)
    - `file` (string/null) – snapshot of the reference depth (.npy float32, plate size), relative to <work>/inpaint/
    - `stamp` (string)
    - `fit` (object) – fit to the camera: method points | planes | none, anchors, inliers, err_rel, z_median

## `matte_desk/card_key@1.0` – Key of a card (`<work>/inpaint/keys/<card>/k_####/key.json`; versions `v###.png` / `.exr` next to it)

- `schema` (string, required)
- `card` (string, required)
- `frame` (integer, required)
- `versions` (array, required)
  - `v` (string, required)
  - `ext` (string)
  - `source` (string)
  - `date` (string)
- `active` (string/null)
- `canvas` (object/null) – the key's own pinhole view when larger than the plate frame: w, h, K

## `matte_desk/key_areas@1.0` – Areas of a key to regenerate (`<work>/inpaint/keys/<card>/k_####/areas.json`)

- `schema` (string, required)
- `card` (string)
- `frame` (integer)
- `areas` (array, required)
  - `id` (string, required)
  - `kind` (enum, required) ∈ ['curve', 'brush']
  - `op` (enum) ∈ ['add', 'remove']
  - `curve` (object) – engine/curves.py JSON, closed
  - `png` (string)
  - `label` (string)
