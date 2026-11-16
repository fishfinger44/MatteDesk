# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/ft_nuke.py - Final Touch as live Nuke nodes in the export's `.nk` (user decision 2026-11-16).

Modes per object (`MODES`):
- `bake`  – everything burnt into the exported alpha (as before; the default);
- `grade` – brush strokes burnt in, the finish as nodes: Grade (black / white / gamma) → Dilate (erode) → Blur;
- `paint` – brush strokes as one `RotoPaint` (each stroke lives on its own frame only) + the finish nodes.
The motion blur of Final Touch cannot be rebuilt with standard nodes (per-edge strength measured on the plate), so with
`mb_strength > 0` only `bake` is available (`available`).

The chain in the MASKI column of such an object reproduces engine/final_touch + engine/export in their order:
  Read <obj>_ftbase (the base [+ strokes], NOT cut by the objects in front) → Shuffle red→rgba → [RotoPaint] → [Grade]
  → [Dilate] → [Blur] → Merge `minus` (chain − Read <obj>_front: max of the raw front layers) → Clamp → anchor
(export.subtract_front = clamp(a − max(fronts)) after the finish).

Calibration (Nuke 15.0v1 batch renders, 2026-11-16; tools: scratchpad probes, numbers in LOCAL_CHECKS):
- Grade blackpoint / whitepoint / gamma with both clamps = finish levels + gamma exactly (max |Δ| 0).
- Blur `size` = 2.3548 × σ (FWHM of our Gaussian; mean |Δ| ≤ 0.0004).
- erode: Nuke has no grey min filter with a round kernel; Dilate (square) = ours inside, corners differ (mean |Δ| ≤ 0.013).
- brush: `bs` = diameter; our smoothstep falloff ≈ Nuke hardness `h` + a larger `bs` + build-up (`BRUSH_CAL`, mean |Δ|
  ≤ 0.01 of the alpha on a single stroke, max ≈ 0.18 at hardness 0). RotoPaint renders the FIRST stroke of the list on
  top, so the strokes are written in reverse order. add = white over, remove = black (rgba 0) over, erase = `src 2`
  (reveals the input = the base), per-point pen radius = pressure (z) of the control point.
  Overlapping add / remove strokes are composited over one another in Nuke, while Final Touch sums them in a correction
  layer first – the same for single strokes, approximate where strokes overlap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

MODES = ("bake", "grade", "paint")
MODE_LABEL = {"bake": "wpalone w alfę", "grade": "Grade jako węzły (kreski wpalone)",
              "paint": "Grade + pędzel jako RotoPaint"}
BLUR_SIZE_PER_SIGMA = 2.3548
# the Root layer's planar-track attributes, as Nuke 15 writes them for a fresh RotoPaint
LAYER_ATTRS = ("pt1x 0 pt1y 0 pt2x 0 pt2y 0 pt3x 0 pt3y 0 pt4x 0 pt4y 0 ptex00 0 ptex01 0 ptex02 0 ptex03 0 ptex10 0 "
               "ptex11 0 ptex12 0 ptex13 0 ptex20 0 ptex21 0 ptex22 0 ptex23 0 ptex30 0 ptex31 0 ptex32 0 ptex33 0 "
               "ptof1x 0 ptof1y 0 ptof2x 0 ptof2y 0 ptof3x 0 ptof3y 0 ptof4x 0 ptof4y 0 pterr 0 ptrefset 0 ptmot 4 ptref 0")
NODE_DY = 70                    # vertical pitch of the chain's nodes in the MASKI column
# our hardness -> (Nuke h, bs / (2 r), build-up) – fitted on r = 20 px strokes (rendered in Nuke 15.0v1)
BRUSH_CAL = ((0.0, 0.00, 1.20, 1), (0.1, 0.00, 1.25, 1), (0.2, 0.00, 1.35, 1), (0.3, 0.05, 1.45, 1),
             (0.4, 0.25, 1.40, 1), (0.5, 0.35, 1.40, 1), (0.6, 0.50, 1.30, 1), (0.7, 0.60, 1.25, 1),
             (0.8, 0.65, 1.30, 1), (0.9, 0.95, 1.05, 0), (1.0, 1.00, 1.05, 0))


def available(fin: Dict[str, Any]) -> Tuple[bool, str]:
    """(ok, why): whether the finish of an object can go to Nuke as nodes."""
    if float((fin or {}).get("mb_strength") or 0.0) > 0.0:
        return False, "motion blur Final Touch (siła z ruchu / pomiaru plate'u) nie ma odpowiednika w węzłach Nuke'a – zostaje wpalony"
    return True, ""


def brush_params(radius: float, hardness: float) -> Tuple[float, float, int]:
    """(bs, h, bu) of a Nuke paint stroke ≈ our stroke of `radius` / `hardness` (linear between the BRUSH_CAL rows)."""
    hd = min(1.0, max(0.0, float(hardness)))
    rows = BRUSH_CAL
    for (h0, n0, k0, b0), (h1, n1, k1, _b1) in zip(rows, rows[1:]):
        if h0 <= hd <= h1:
            t = (hd - h0) / (h1 - h0)
            return 2.0 * float(radius) * (k0 + t * (k1 - k0)), n0 + t * (n1 - n0), (b0 if t < 0.5 else _b1)
    return 2.0 * float(radius) * rows[-1][2], rows[-1][1], rows[-1][3]


def _f(v: float) -> str:
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def stroke_curve(s: Dict[str, Any], f: int, idx: int, h: int) -> str:
    """One Final Touch stroke as a RotoPaint `cubiccurve` living on frame f only (plate pixels, y down -> Nuke)."""
    r_def = float(s.get("radius_px") or 8)
    pts = s["p"]
    rmax = max([r_def] + [float(q[2]) for q in pts if len(q) >= 3])
    bs, nh, bu = brush_params(rmax, float(s.get("hardness", 0.6)))
    cps = " ".join("{%s %s %s}" % (_f(q[0] + 0.5), _f(h - (q[1] + 0.5)), _f((float(q[2]) if len(q) >= 3 else r_def) / rmax))
                   for q in pts)
    attrs = [f"bs {_f(bs)}", f"h {_f(nh)}", f"bu {bu}", "str 1", "ltt 2", f"ltn {int(f)}", f"ltm {int(f)}"]
    if len(pts) and any(len(q) >= 3 for q in pts):
        attrs.append("ds 1")                       # pen pressure (z) scales the brush size
    if s["mode"] == "remove":
        attrs += ["r 0", "g 0", "b 0", "a 0"]
    elif s["mode"] == "erase":
        attrs.append("src 2")                      # Nuke's eraser: reveals the input (the base)
    name = "%s_F%d_%d" % ({"add": "Add", "remove": "Remove", "erase": "Erase"}[s["mode"]], int(f), idx)
    return "{cubiccurve %s 512 catmullrom {cc {f 2080} {p %s}} {t 0} {a %s}}" % (name, cps, " ".join(attrs))


def rotopaint_lines(frames: Dict[str, Sequence[Dict[str, Any]]], w: int, h: int, name: str, x: int, y: int,
                    label: str = "") -> List[str]:
    """`RotoPaint` node (input = the stack top) with every stroke of every frame; the list is top-first (reverse)."""
    curves: List[str] = []
    for f in sorted(frames, key=lambda k: int(k)):
        for i, s in enumerate(frames[f] or []):
            curves.append(stroke_curve(s, int(f), i + 1, h))
    curves.reverse()                               # the first curve of the list renders on top
    body = "{{{v x3f99999a} {f 0} {n {layer Root {f 0} {t %s %s} {a %s} %s}}}}" % (_f(w / 2), _f(h / 2), LAYER_ATTRS,
                                                                                  " ".join(curves))
    return ["RotoPaint {", f" curves {body}", f" name {name}", f" label {_q(label)}", f" xpos {x}", f" ypos {y}", "}"]


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "/").replace('"', '\\"') + '"'


def finish_nodes(fin: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """[(class, knobs)] of the finish (levels + gamma, erode, blur) – only the steps that do something."""
    from engine.final_touch import FINISH_DEFAULT
    f = {**FINISH_DEFAULT, **(fin or {})}
    out: List[Tuple[str, Dict[str, Any]]] = []
    b, wp, g = float(f["black"]), float(f["white"]), float(f["gamma"])
    if b > 0.0 or wp < 1.0 or abs(g - 1.0) > 1e-9:
        out.append(("Grade", {"channels": "rgba", "blackpoint": _f(b), "whitepoint": _f(max(wp, b + 1e-3)),
                              "gamma": _f(g), "black_clamp": "true", "white_clamp": "true"}))
    e = float(f["erode_px"])
    if abs(e) >= 0.5:
        out.append(("Dilate", {"channels": "rgba", "size": str(-int(round(e)))}))   # Dilate: negative = erode
    bl = float(f["blur_px"])
    if bl > 0.0:
        out.append(("Blur", {"channels": "rgba", "size": _f(bl * BLUR_SIZE_PER_SIGMA)}))
    return out


def chain_rows(spec: Dict[str, Any]) -> int:
    """Rows (NODE_DY each) the chain takes in the column between the Shuffle and the anchor."""
    n = len(finish_nodes(spec.get("finish") or {})) + (1 if spec.get("mode") == "paint" and spec.get("frames") else 0)
    return n + (2 if spec.get("front") else 0)


def chain_lines(spec: Dict[str, Any], obj: str, nk_name: str, x: int, y: int, w: int, h: int, fmt: str,
                first: int, last: int, label: str) -> List[str]:
    """Nodes after the column's Shuffle (stack top): [RotoPaint] → finish nodes → [Merge from the front + Clamp]."""
    out: List[str] = []
    yy = y
    if spec.get("mode") == "paint" and spec.get("frames"):
        nfr = len([k for k, v in spec["frames"].items() if v])
        out += rotopaint_lines(spec["frames"], w, h, f"FT_paint_{nk_name}", x, yy,
                               f"Final Touch · pędzel · {nfr} kl. · {label}")
        yy += NODE_DY
    tip = {"Grade": "Final Touch · black / white / gamma", "Dilate": "Final Touch · erode (≈ kwadrat)",
           "Blur": "Final Touch · blur"}
    for cls, knobs in finish_nodes(spec.get("finish") or {}):
        out += [f"{cls} {{"] + [f" {k} {v}" for k, v in knobs.items()] + \
               [f" name FT_{cls.lower()}_{nk_name}", f" label {_q(tip[cls])}", f" xpos {x}", f" ypos {yy}", "}"]
        yy += NODE_DY
    if spec.get("front"):
        # stack after the front's Read + Shuffle: [chain, front] -> Merge2 input0 (B) = front (top), input1 (A) = the
        # chain; operation minus = A − B = chain − front (no leftover on the stack)
        maps = " ".join(f"rgba.red 0 0 rgba.{d} 0 {i}" for i, d in enumerate(("red", "green", "blue", "alpha")))
        out += ["Read {", " inputs 0", f" file {_q(spec['front'])}", f" format {fmt}", f" first {first}", f" last {last}",
                f" origfirst {first}", f" origlast {last}", " origset true", " raw true", " on_error black",
                f" name Read_{nk_name}_front", f' label "obiekty z przodu (max)"', f" xpos {x + 140}", f" ypos {yy - NODE_DY}", "}",
                "Shuffle2 {", " fromInput1 {{0} B}", " fromInput2 {{0} B}", f' mappings "4 {maps}"',
                f" name FRONT_{nk_name}", f" xpos {x + 140}", f" ypos {yy}", "}",
                "Merge2 {", " inputs 2", " operation minus", " output rgba", f" name FT_cut_front_{nk_name}",
                f' label "− obiekty z przodu"', f" xpos {x}", f" ypos {yy}", "}",
                "Clamp {", " channels rgba", f" name FT_clamp_{nk_name}", f" xpos {x}", f" ypos {yy + NODE_DY}", "}"]
    return out


def specs(work_dir, outputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """`ft` argument of engine/nuke_mattes.mattes_nk from an outputs.json dict: every exported object whose entry has
    `final_touch.nuke` with a node mode, with its CURRENT strokes / finish (engine/final_touch) and the exported base /
    front sequences."""
    from pathlib import Path
    from engine.final_touch import load as ft_load
    out: Dict[str, Dict[str, Any]] = {}
    for e in outputs.get("objects", []) or []:
        nk = (e.get("final_touch") or {}).get("nuke") or {}
        if nk.get("mode") not in ("grade", "paint") or not nk.get("base_matte"):
            continue
        d = ft_load(Path(work_dir) / "objects" / e["name"])
        out[e["name"]] = {"mode": nk["mode"], "frames": d.get("frames") or {} if nk["mode"] == "paint" else {},
                          "finish": d.get("finish") or {}, "base": nk["base_matte"], "front": nk.get("front_matte")}
    return out


PARTS = ("grade", "paint")


def paste_text(part: str, d: Dict[str, Any], w: int, h: int, obj: str) -> Tuple[str, List[str]]:
    """(clipboard text, node names) of ONE section of the Final Touch palette (its „⇪ Export” button, 2026-11-16) – like
    nodes copied in Nuke: Ctrl+V connects the first one to the selected node. `grade` = Grade → Dilate → Blur with the
    current values (all three, so they can be tweaked in Nuke; 0 = no-op), `paint` = the RotoPaint of every stroke."""
    import re as _re
    n = _re.sub(r"\W", "_", obj) or "obj"
    if part == "grade":
        from engine.final_touch import FINISH_DEFAULT
        f = {**FINISH_DEFAULT, **(d.get("finish") or {})}
        b, wp = float(f["black"]), float(f["white"])
        nodes = [("Grade", {"channels": "rgba", "blackpoint": _f(b), "whitepoint": _f(max(wp, b + 1e-3)),
                            "gamma": _f(float(f["gamma"])), "black_clamp": "true", "white_clamp": "true"}),
                 ("Dilate", {"channels": "rgba", "size": str(-int(round(float(f["erode_px"]))))}),
                 ("Blur", {"channels": "rgba", "size": _f(float(f["blur_px"]) * BLUR_SIZE_PER_SIGMA)})]
        tip = {"Grade": "Final Touch · black / white / gamma", "Dilate": "Final Touch · erode (≈ kwadrat)",
               "Blur": "Final Touch · blur"}
        lines: List[str] = []
        names = []
        for i, (cls, knobs) in enumerate(nodes):
            nm = f"FT_{cls.lower()}_{n}"
            names.append(nm)
            lines += [f"{cls} {{"] + [f" {k} {v}" for k, v in knobs.items()] + \
                     [f" name {nm}", f" label {_q(tip[cls])}", f" xpos 0", f" ypos {i * NODE_DY}", "}"]
    elif part == "paint":
        frames = {k: v for k, v in (d.get("frames") or {}).items() if v}
        nm = f"FT_paint_{n}"
        names = [nm]
        lines = rotopaint_lines(frames, w, h, nm, 0, 0, f"Final Touch · pędzel · {len(frames)} kl.")
    else:
        raise ValueError(f"part: {' | '.join(PARTS)}")
    return "set cut_paste_input [stack 0]\nversion 15.0 v1\npush $cut_paste_input\n" + "\n".join(lines) + "\n", names
