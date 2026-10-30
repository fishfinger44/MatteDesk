# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/contract.py - Data contract for dependent tools (SPEC 16): versioned schemas, validation, DATA_CONTRACT.md.

Every MatteDesk JSON carries `"schema": "matte_desk/<name>@<version>"`; an incompatible change = new version +
converter. Schemas are a small JSON-Schema subset (type / required / properties / items / enum / const) so the
document and the validation come from the same source. `python -m engine.contract docs` regenerates the doc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

NUM = {"type": "number"}
INT = {"type": "integer"}
STR = {"type": "string"}
BOOL = {"type": "boolean"}

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "matte_desk/project@1.0": {
        "title": "Shot project (`<work>/matte_desk/project.json`)",
        "type": "object", "required": ["schema", "manifest_path", "work_dir", "shot", "first_frame", "last_frame", "plate_w", "plate_h"],
        "properties": {"schema": STR, "manifest_path": STR, "work_dir": STR, "shot": STR, "colorspace": STR, "fps": NUM,
                       "first_frame": INT, "last_frame": INT, "plate_w": INT, "plate_h": INT,
                       "plates_added": {"type": "object", "description": "plates added in the tool (the manifest is read-only)"},
                       "plates_hidden": {"type": "array", "items": STR},
                       "export_dir": {"type": "string", "description": "export folder chosen by the user (missing = `<work>/matte_desk/export`)"},
                       "export_format": {"enum": ["png", "exr"], "description": "export mask format; missing = auto (8-bit PNG for 8-bit sources, EXR for EXR)"},
                       "mask_plate_denoise": {"enum": ["on", "off"], "description": "plate-to-mask denoise (4b, core + final); missing = auto (on for EXR / > 8 bit, off for 8-bit); defaults.json mplate.enabled = false disables it everywhere"}},
    },
    "matte_desk/outputs@1.0": {
        "title": "Results registry (`<work>/matte_desk/outputs.json`)",
        "type": "object", "required": ["schema", "objects"],
        "properties": {
            "schema": STR, "shot": STR, "frame_range": {"type": "array", "items": INT}, "plate_size": {"type": "array", "items": INT},
            "objects": {"type": "array", "items": {"type": "object", "required": ["name", "depth_order", "matte", "version", "qa_status", "date", "params_hash"],
                        "properties": {"name": STR, "depth_order": INT, "matte": {"type": "string", "description": "#### pattern, full frame, rgb=a"},
                                       "core": {"type": ["string", "null"]}, "version": STR,
                                       "qa_status": {"enum": ["ok", "flags", "none"]}, "date": STR, "params_hash": STR,
                                       "edge_mode": {"type": ["string", "null"]},
                                       "members": {"type": "array", "items": STR, "description": "a group entity (PLAN_GROUP_COMPOSE): its members – the matte is the group's whole (a stage computed on the whole, Final Touch, or the composition of the members with seams); one layer at its front-most member"}}}},
            "mattes_all": {"type": "string", "description": "multi-EXR: <obj>.alpha (objects and groups), union.alpha, [depth.disparity]"},
            "mattes_all_channels": {"type": "array", "items": STR, "description": "multi-EXR layers in depth order (without `.alpha`)"},
            "mattes_nk": {"type": "string", "description": "Nuke script with the mask passes (MASKI backdrop, anchors + stamps) – File > Import Script"},
            "export_dir": {"type": "string", "description": "export folder (default `<work>/matte_desk/export`; chosen in project.json)"},
            "export_format": {"enum": ["png", "exr"], "description": "mask format of the last export (extension of the `matte` patterns); `mattes_all` only with exr"},
            "camera2d": {"type": "object", "properties": {"json_path": STR, "nk_path": STR}},
            "depth": {"type": "object", "required": ["path", "kind", "channel", "range", "method"],
                      "properties": {"path": STR, "kind": {"enum": ["disparity", "z"]}, "channel": STR,
                                     "range": {"type": "array", "items": NUM}, "method": STR}},
        },
    },
    "matte_desk/camera2d@1.0": {
        "title": "2D camera motion (`<work>/matte_desk/camera/camera2d.json`)",
        "type": "object", "required": ["schema", "ref_frame", "plate_w", "plate_h", "frames", "motion_type"],
        "properties": {"schema": STR, "shot_name": STR, "ref_frame": INT, "plate_w": INT, "plate_h": INT, "fps": NUM,
                       "motion_type": STR, "residual_median_px": NUM, "residual_p95_px": NUM, "anchor_drift_px": NUM,
                       "has_parallax": BOOL,
                       "frames": {"type": "array", "items": {"type": "object", "required": ["frame", "homography"],
                                  "properties": {"frame": INT, "homography": {"type": "array", "description":
                                                 "3x3, plate pixels (OpenCV: pixel centre at integer coordinates), y pointing down, frame -> reference"}}}}},
    },
    "matte_desk/object@1.0": {
        "title": "Object (`objects/<obj>/object.json`)",
        "type": "object", "required": ["schema", "name", "depth_order", "detect", "edge_mode", "tool"],
        "properties": {"schema": STR, "name": STR, "depth_order": INT, "detect": {"type": "object"}, "flags": {"type": "array", "items": STR},
                       "edge_mode": {"enum": ["auto", "backlit", "matanyone", "core", "ltx"]}, "edge_mode_auto": STR,
                       "kind": {"enum": ["group"], "description": "a group entity (PLAN_GROUP_COMPOSE): no core of its own, its mask = the composition of `members`"},
                       "members": {"type": "array", "items": STR, "description": "group: its objects (one group per object, no nesting)"},
                       "seam_px": INT,
                       "group": {"type": ["string", "null"], "description": "object: the group it belongs to"},
                       "group_source": {"type": ["object", "null"], "description": "object in a group: „Do grupy z:” {source: core | matanyone | videomama | vitmatte | ltx | version | final_touch, run}"},
                       "group_on": {"type": "boolean", "description": "object in a group: false = switched off for now (out of the group's composition), default true"},
                       "group_seam_px": {"type": ["integer", "null"], "description": "object in a group: its own seam reach px (null = the group's seam_px)"},
                       "band_scale": NUM, "shift_px": NUM, "band_out_px": NUM, "band_in_px": NUM, "temporal_smoothing": {"enum": ["weak", "normal", "strong"]},
                       "ma_scale_mode": {"enum": ["auto", "1.0", "0.75", "0.5"], "description": "MatAnyone2 scale in \"Compute final\" („Przelicz finalnie”): auto = 9-frame trial by VRAM (1.0 → 0.75 → 0.5), a number = fixed (no trial)"},
                       "key_band_same": {"type": "boolean", "description": "key frames get the same edge band as the other frames (instead of ±3 px from the spec); missing = on"},
                       "core_soft_start": {"type": "boolean", "description": "the MatAnyone2 edge band starts from the SAM 3.1 probability maps (CoreCombine core_mode=soft); missing = on"},
                       "tool": {"type": "object", "required": ["active_run", "final_valid"]}},
    },
    "matte_desk/crops@1.0": {
        "title": "Crop segments (`objects/<obj>/crops.json`)",
        "type": "object", "required": ["schema", "segments"],
        "properties": {"schema": STR, "segments": {"type": "array", "items": {"type": "object",
                       "required": ["seg_id", "first", "last", "crop", "pad", "scale_to_1008"],
                       "properties": {"seg_id": STR, "first": INT, "last": INT, "overlap_in": INT, "overlap_out": INT,
                                      "crop": {"type": "array", "items": INT, "description": "[x, y, side, side] of the virtual square"},
                                      "pad": {"type": "array", "items": INT}, "scale_to_1008": NUM, "manual": BOOL}}}},
    },
    "matte_desk/run@1.0": {
        "title": "Result version (`objects/<obj>/runs/v###/run.json`)",
        "type": "object", "required": ["schema", "run", "edge_mode", "segments_def"],
        "properties": {"schema": STR, "run": STR, "edge_mode": STR, "ma_scale": NUM,
                       "ma_scale_mode": {"type": "string", "description": "auto | 1.0 | 0.75 | 0.5 (object setting)"},
                       "ma_node": {"type": "object", "description": "inputs of MatAnyone2RunKeyframes >= 1.1.0 (vram_release_frac, max_mem_frames 0 = auto); empty = older node"},
                       "segments_def": {"type": "array"},
                       "vram_log": {"type": "array", "items": STR}, "hygiene": {"type": "object"},
                       "soft_start": {"type": "object", "description": "soft start of the band from the SAM 3.1 maps: frames (frames with a map in CoreCombine), maps, node (the node has core_mode), note; empty = no maps / disabled"}},
    },
    "matte_desk/edge_fix@1.0": {
        "title": "Edge measurement / correction of a version (`objects/<obj>/runs/v###/edge/edge.json`; per-frame measurements `e_####.npz`)",
        "type": "object", "required": ["schema", "mode", "fix"],
        "properties": {"schema": STR, "mode": STR, "fix": {"type": "array", "items": STR}, "attempts": INT,
                       "applied_from": STR, "skipped": {"type": "array", "items": STR}},
    },
    "matte_desk/key@1.0": {
        "title": "Key (`objects/<obj>/keys/k_####.json`)",
        "type": "object", "required": ["schema", "frame", "kind", "author", "mask_path"],
        "properties": {"schema": STR, "frame": INT, "kind": STR, "author": {"enum": ["user", "auto"]}, "mask_path": STR},
    },
    "matte_desk/zone@1.0": {
        "title": "Zone (`objects/<obj>/zones/z_##.json`)",
        "type": "object", "required": ["schema", "zone_id", "mode", "criterion", "frame"],
        "properties": {"schema": STR, "zone_id": STR, "mode": {"enum": ["add", "remove"]},
                       "criterion": {"enum": ["dark", "bright", "color", "all"]}, "frame": INT, "soft": BOOL},
    },
    "matte_desk/depth@1.0": {
        "title": "Depth report (`<work>/matte_desk/depth/depth_report.json`)",
        "type": "object", "required": ["schema", "kind", "method", "export_pattern", "value_min", "value_max"],
        "properties": {"schema": STR, "kind": STR, "method": STR, "export_pattern": STR, "value_min": NUM, "value_max": NUM},
    },
}

_PY = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool,
       "null": type(None)}


def _check(v: Any, s: Dict[str, Any], path: str, errs: List[str]) -> None:
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} nie w {s['enum']}")
        return
    t = s.get("type")
    if t:
        ts = t if isinstance(t, list) else [t]
        ok = any(isinstance(v, _PY[x]) and not (x in ("integer", "number") and isinstance(v, bool)) for x in ts)
        if not ok:
            errs.append(f"{path}: oczekiwano {t}, jest {type(v).__name__}")
            return
    if isinstance(v, dict):
        for r in s.get("required", []):
            if r not in v:
                errs.append(f"{path}: brak pola '{r}'")
        for k, sub in s.get("properties", {}).items():
            if k in v:
                _check(v[k], sub, f"{path}.{k}", errs)
    if isinstance(v, list) and "items" in s:
        for i, it in enumerate(v):
            _check(it, s["items"], f"{path}[{i}]", errs)


def validate(data: Dict[str, Any], schema_id: Optional[str] = None) -> List[str]:
    sid = schema_id or data.get("schema")
    if sid not in SCHEMAS:
        return [f"nieznany schemat {sid!r}"]
    errs: List[str] = []
    _check(data, SCHEMAS[sid], "$", errs)
    return errs


def validate_work_dir(work_dir: Path) -> Dict[str, List[str]]:
    """All MatteDesk JSONs under <work>/matte_desk -> {path: errors} (only files with problems)."""
    out = {}
    for p in Path(work_dir).rglob("*.json"):
        if any(x in p.parts for x in ("comfy_io", "cache", "mplate")):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            out[str(p)] = [f"JSON: {e}"]
            continue
        if isinstance(d, dict) and str(d.get("schema", "")).startswith("matte_desk/"):
            e = validate(d)
            if e:
                out[str(p)] = e
    return out


def render_markdown() -> str:
    lines = ["# DATA_CONTRACT – MatteDesk data formats (SPEC 16)", "",
             "Generated from `engine/contract.py` (`python -m engine.contract docs`) – the same schemas are used for validation.", "",
             "## Rules", "- Dependent tools (e.g. Camera3D + Mesh) **only read** MatteDesk data; they write into their own subfolder (`<work>/camera3d/`).",
             "- Every JSON has a `schema` field = `matte_desk/<name>@<version>`; an incompatible change = new version + converter.",
             "- Coordinate system: plate pixels, origin in the top-left corner, y pointing down. Conversion to Nuke: "
             "x_n = x + 0.5, y_n = plate_h − 0.5 − y (OpenCV: pixel centre at integer coordinates).",
             "- Frame numbers = plate frame numbers everywhere.",
             "- Masks: full frame, linear alpha 0–1; format per `project.json` → `export_format` (auto = 8-bit PNG `<obj>_####.png` for 8-bit sources – MP4 / PNG / JPG, EXR `rgb = a` half ZIP `<obj>_####.exr` for EXR; `png` / `exr` forces it). The patterns in `outputs.json` have the real extension. Multi-EXR (`<obj>.alpha`, `union.alpha`, optionally `depth.disparity` float32) only with EXR.",
             "- Depth: `depth.disparity` = relative disparity (closer = larger, arbitrary scale) + a copy in `R`; not Z.",
             "- MatteDesk always updates `outputs.json` on export.", ""]
    def props(s, ind=""):
        out = []
        req = set(s.get("required", []))
        for k, sub in s.get("properties", {}).items():
            t = sub.get("type", "enum" if "enum" in sub else "")
            t = "/".join(t) if isinstance(t, list) else t
            extra = f" – {sub['description']}" if "description" in sub else ""
            enum = f" ∈ {sub['enum']}" if "enum" in sub else ""
            out.append(f"{ind}- `{k}` ({t}{', required' if k in req else ''}){enum}{extra}")
            if sub.get("type") == "array" and isinstance(sub.get("items"), dict) and sub["items"].get("properties"):
                out += props(sub["items"], ind + "  ")
            elif sub.get("type") == "object" and sub.get("properties"):
                out += props(sub, ind + "  ")
        return out
    for sid, s in SCHEMAS.items():
        lines += [f"## `{sid}` – {s['title']}", ""] + props(s) + [""]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "docs":
        target = Path(__file__).resolve().parent.parent / "docs" / "DATA_CONTRACT.md"
        target.parent.mkdir(exist_ok=True)
        target.write_text(render_markdown(), encoding="utf-8")
        print(target)
    elif len(sys.argv) > 2 and sys.argv[1] == "validate":
        print(json.dumps(validate_work_dir(Path(sys.argv[2])), indent=1, ensure_ascii=False))
