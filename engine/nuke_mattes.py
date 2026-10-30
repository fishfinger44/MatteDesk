# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/nuke_mattes.py - Nuke script that pulls the mattes out of the export (Z29): `<export>/<shot>_mattes.nk`.

File → Import Script in Nuke. Layout follows the user's Nuke rules (handoff/context_lessons/feedback_nuke_node_layout.md):
- everything inside one backdrop **MASKI**; columns in depth order (front object left), no overlapping nodes;
- multi-EXR: `Read` (raw – alpha is linear data, no OCIO) → chain of `Dot`s, one per column (a Dot on every branch, no fan
  of long pipes) → `Shuffle2` `<layer>.alpha` → rgba (all four channels) named `MATTE_<obj>` → Stamps anchor (`NoOp`,
  Stamps v1.1 knobs, title = obj) → free Wired Stamp under the anchor (`PostageStamp`, hidden input, `anchor` knob =
  anchor name, `Stamp_copy_<obj>`) – the graph uses the masks only through stamps; checked in Nuke 15 (LOCAL_CHECKS Z29);
- objects exported without the multi-EXR ("Ten obiekt" / "Całość"): one `Read` per object folder (matte in rgb, no
  alpha) → `Shuffle2` red → rgba `MATTE_<obj>` → the same anchor + copy-stamp convention;
- labels carry the version (`runs/v###`) and the export date.
Standard nodes only (Read, Dot, Shuffle2, NoOp, PostageStamp, BackdropNode). `layers` are declared with `add_layer` so the
script parses before the Read has loaded its channels.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

COL_W = 160          # column pitch (node ~80 px wide + gap)
Y_READ, Y_DOT, Y_SHUF, Y_ANCHOR, Y_STAMP = 0, 130, 190, 280, 360
SAFE = re.compile(r"[^A-Za-z0-9_]")


def _nk_name(s: str) -> str:
    n = SAFE.sub("_", s)
    return n if n and not n[0].isdigit() else f"_{n}"


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "/").replace('"', '\\"') + '"'


# Stamps v1.1 knob sets, as written by stamps.anchor() / stamps.wired() (user's SH010 scripts): `identifier` is a hidden
# Text knob, the wired stamp stores its anchor's node name in `anchor`, autolabel shows the title, onCreate / knobChanged
# hook the Stamps callbacks (wrapped in try – the script also parses without the plugin).
_ANCHOR_ON_CREATE = r'"if nuke.GUI:\n    try:\n        import stamps; stamps.anchorOnCreate()\n    except:\n        pass"'
_WIRED_ON_CREATE = r'"if nuke.GUI:\n    try:\n        import stamps; stamps.wiredOnCreate()\n    except:\n        pass\n"'
_AUTOLABEL = r'"nuke.thisNode().knob(\"title\").value()"'


def _anchor(title: str, x: int, y: int, label: str) -> List[str]:
    """Stamps anchor (NoOp, 'Anchor Stamp' tab, identifier = anchor, title) and a free Wired Stamp (PostageStamp,
    hidden input, identifier = wired, anchor = the anchor's node name) right under it – both recognised by Stamps."""
    n = _nk_name(title)
    an = f"Anchor_{n}"
    return [
        "NoOp {",
        f" name {an}",
        f" onCreate {_ANCHOR_ON_CREATE}",
        " knobChanged stamps.anchorKnobChanged()",
        f" autolabel {_AUTOLABEL}",
        f" label {_q(label)}",
        " tile_color 0xffffff01",
        " note_font_size 20",
        f" xpos {x}", f" ypos {y}",
        ' addUserKnob {20 anchor_tab l "Anchor Stamp"}',
        " addUserKnob {26 identifier -STARTLINE +HIDDEN T anchor}",
        " addUserKnob {1 title l Title:}",
        f" title {_q(title)}",
        f' addUserKnob {{26 prev_title l "" +STARTLINE +HIDDEN T {_q(title)}}}',
        f' addUserKnob {{26 prev_name l "" +STARTLINE +HIDDEN T {an}}}',
        ' addUserKnob {3 showing l "" +STARTLINE +HIDDEN}',
        " addUserKnob {1 tags l Tags}",
        " tags MatteDesk,",
        ' addUserKnob {26 line1 l "" +STARTLINE}',
        ' addUserKnob {26 stamps_label l Stamps: T " "}',
        " addUserKnob {22 createStamp l new -STARTLINE T stamps.stampCreateWired(nuke.thisNode())}",
        " addUserKnob {22 selectStamps l select -STARTLINE T stamps.wiredSelectSimilar(nuke.thisNode().name())}",
        " addUserKnob {22 reconnectStamps l reconnect -STARTLINE T stamps.anchorReconnectWired()}",
        ' addUserKnob {22 zoomNext l "zoom next" -STARTLINE T stamps.wiredZoomNext(nuke.thisNode().name())}',
        "}",
        f"set {an} [stack 0]",   # stack variable of the anchor: wired stamps elsewhere in the script push it
    ] + _wired(title, f"Stamp_copy_{n}", x, y + (Y_STAMP - Y_ANCHOR))


def _wired(title: str, name: str, x: int, y: int) -> List[str]:
    """Stamps v1.1 Wired Stamp (PostageStamp, hidden input) of the anchor of `title`; its input = the top of the stack."""
    n = _nk_name(title)
    an = f"Anchor_{n}"
    return [
        "PostageStamp {",
        f" name {name}",
        f" onCreate {_WIRED_ON_CREATE}",
        ' knobChanged "import stamps; stamps.wiredKnobChanged()"',
        f" autolabel {_AUTOLABEL}",
        f" label {_q(title)}",
        " tile_color 0x1000001",
        " note_font_size 20",
        f" xpos {x}", f" ypos {y}",
        " hide_input true",
        ' addUserKnob {20 wired_tab l "Wired Stamp"}',
        " addUserKnob {26 identifier -STARTLINE +HIDDEN T wired}",
        ' addUserKnob {3 lockCallbacks l "" +STARTLINE +HIDDEN}',
        " addUserKnob {6 toReconnect -STARTLINE +HIDDEN}",
        " addUserKnob {1 title l Title:}",
        f" title {_q(title)}",
        f' addUserKnob {{26 prev_title l "" +STARTLINE +HIDDEN T {_q(title)}}}',
        ' addUserKnob {6 postageStamp_show l "postage stamp" +STARTLINE}',
        ' addUserKnob {22 show_anchor l " show anchor " -STARTLINE T stamps.wiredShowAnchor()}',
        ' addUserKnob {22 zoom_anchor l "zoom anchor" -STARTLINE T stamps.wiredZoomAnchor()}',
        ' addUserKnob {22 selectSimilar l " select similar " -STARTLINE T stamps.wiredSelectSimilar()}',
        " addUserKnob {22 reconnect_all l \"reconnect all\" -STARTLINE T stamps.wiredReconnectAll()}",
        " addUserKnob {1 anchor l Anchor}",
        f" anchor {an}",
        ' addUserKnob {6 auto_reconnect_by_title l "auto-reconnect by title" +STARTLINE}',
        "}",
    
    ]


def mattes_nk(outputs: Dict[str, Any], shot: str, first: int, last: int, w: int, h: int,
              date: Optional[str] = None, channels: Optional[Sequence[str]] = None,
              merge: Optional[Dict[str, Any]] = None) -> str:
    """Script text from an outputs.json dict (objects[], groups{}, mattes_all). `channels` (multi-EXR layer names,
    default: objects in depth order + <parent>_full + union [+ depth]) is what the multi-EXR really contains.
    `merge` = {"plate": pattern, "colorspace": manifest colorspace, "objects": {name: MergeMatte knobs}} adds the backdrop
    MERGEMATTE under MASKI (plan PLAN_FINAL_TOUCH §11; `merge_matte_backdrop`)."""
    date = date or datetime.now().strftime("%Y-%m-%d %H:%M")
    objs = sorted(outputs.get("objects", []), key=lambda o: o.get("depth_order", 0))
    groups = outputs.get("groups", {}) or {}
    run_of = {o["name"]: o.get("version", "") for o in objs}
    multi = outputs.get("mattes_all")
    lines: List[str] = []
    nodes_xy: List[Tuple[int, int]] = []
    if channels is None:
        channels = []
        for o in objs:
            parent = next((p for p, g in groups.items() if o["name"] in g.get("members", [])), None)
            if parent and f"{parent}_full" not in channels:
                channels.append(f"{parent}_full")
            channels.append(o["name"])
        channels.append("union")
    layers = [c for c in channels if c != "depth"]
    head = [f"# MatteDesk – maski ujęcia {shot} (eksport {date}); File > Import Script", ""]
    for c in layers:
        head.append(f"add_layer {{{c} {c}.alpha}}")
    if "depth" in channels:
        head.append("add_layer {depth depth.Z depth.disparity}")
    fmt = f'"{w} {h} 0 0 {w} {h} 1 MatteDesk_{_nk_name(shot)}"'
    x0 = 0
    col = 0
    if multi:
        lines += ["Read {", " inputs 0", f" file {_q(multi)}", f" format {fmt}", f" first {first}", f" last {last}",
                  f" origfirst {first}", f" origlast {last}", " origset true", " raw true", " on_error black",
                  f" name Read_{_nk_name(shot)}_mattes", f" label {_q('multi-EXR · ' + date)}", f" xpos {x0}", f" ypos {Y_READ}", "}"]
        nodes_xy.append((x0, Y_READ))
        prev_dot = None
        for c in channels:
            x = x0 + col * COL_W
            dot = f"Dot_{_nk_name(c)}"
            if prev_dot is not None:
                lines.append(f"push ${prev_dot}")
            lines += ["Dot {", f" name {dot}", f" xpos {x + 34}", f" ypos {Y_DOT}", "}", f"set {dot} [stack 0]"]
            nodes_xy.append((x + 34, Y_DOT))
            src = "depth.Z" if c == "depth" else f"{c}.alpha"
            in1 = "depth" if c == "depth" else c
            ver = run_of.get(c) or ("+".join(run_of.get(m, "") for m in groups.get(c[:-5], {}).get("members", []))
                                     if c.endswith("_full") else "")
            label = " · ".join(x for x in (ver, date) if x)
            maps = " ".join(f"{src} 0 0 rgba.{d} 0 {i}" for i, d in enumerate(("red", "green", "blue", "alpha")))
            lines += ["Shuffle2 {", " fromInput1 {{0} B}", f" in1 {in1}", " fromInput2 {{0} B}", f' mappings "4 {maps}"',
                      f" name MATTE_{_nk_name(c)}", f" label {_q(label)}", f" xpos {x}", f" ypos {Y_SHUF}", "}"]
            nodes_xy.append((x, Y_SHUF))
            lines += _anchor(c, x, Y_ANCHOR, label)
            nodes_xy += [(x, Y_ANCHOR), (x, Y_STAMP)]
            prev_dot = dot
            col += 1
    # objects / wholes exported without the multi-EXR: their own Read
    in_multi = set(channels) if multi else set()
    singles = [(o["name"], o.get("matte"), o.get("version", "")) for o in objs if o["name"] not in in_multi and o.get("matte")]
    singles += [(f"{p}_full", g.get("matte"), "") for p, g in groups.items() if f"{p}_full" not in in_multi and g.get("matte")]
    for name, path, ver in singles:
        x = x0 + col * COL_W
        label = " · ".join(v for v in (ver, date) if v)
        lines += ["Read {", " inputs 0", f" file {_q(path)}", f" format {fmt}", f" first {first}", f" last {last}",
                  f" origfirst {first}", f" origlast {last}", " origset true", " raw true", " on_error black",
                  f" name Read_{_nk_name(name)}", f" label {_q(label)}", f" xpos {x}", f" ypos {Y_READ}", "}"]
        nodes_xy.append((x, Y_READ))
        # single-object EXR carries the matte in rgb (no alpha) -> red into all four, like the multi-EXR columns
        maps = " ".join(f"rgba.red 0 0 rgba.{d} 0 {i}" for i, d in enumerate(("red", "green", "blue", "alpha")))
        lines += ["Shuffle2 {", " fromInput1 {{0} B}", " fromInput2 {{0} B}", f' mappings "4 {maps}"',
                  f" name MATTE_{_nk_name(name)}", f" label {_q(label)}", f" xpos {x}", f" ypos {Y_SHUF}", "}"]
        nodes_xy.append((x, Y_SHUF))
        lines += _anchor(name, x, Y_ANCHOR, label)
        nodes_xy += [(x, Y_ANCHOR), (x, Y_STAMP)]
        col += 1
    if not nodes_xy:
        nodes_xy = [(0, 0)]
    xs, ys = [p[0] for p in nodes_xy], [p[1] for p in nodes_xy]
    bx, by = min(xs) - 60, min(ys) - 110
    bw, bh = max(xs) - min(xs) + 80 + 120, max(ys) - min(ys) + 40 + 170
    backdrop = ["BackdropNode {", " inputs 0", " name BD_MASKI", f' label "MASKI\\n{shot} · {date}"', " note_font_size 42",
                " tile_color 0x4c4c4cff", f" xpos {bx}", f" ypos {by}", f" bdwidth {bw}", f" bdheight {bh}", "}"]
    tail: List[str] = []
    if merge and merge.get("objects"):
        tail = merge_matte_backdrop(merge, shot, first, last, w, h, date, x0=bx + 60, y0=by + bh + 200,
                                    known=[o["name"] for o in objs] + list(channels or []))
    return "\n".join(head + backdrop + lines + tail) + "\n"


# Read colorspace of the plate in an ACES (aces_1.2) script: MergeMatte works in the ACEScg working space
NUKE_ACES_CS = {"ACES2065-1": "ACES - ACES2065-1", "ACEScg": "ACES - ACEScg", "ACEScct": "ACES - ACEScct",
                "sRGB": "Utility - sRGB - Texture", "srgb": "Utility - sRGB - Texture", "Rec709": "Utility - Rec.709 - Camera",
                "rec709": "Utility - Rec.709 - Camera", "linear": "Utility - Linear - sRGB"}
YM_READ, YM_DOT, YM_COPY, YM_PREMULT, YM_GROUP, YM_ANCHOR = 0, 130, 260, 330, 430, 560


def merge_matte_backdrop(merge: Dict[str, Any], shot: str, first: int, last: int, w: int, h: int, date: str,
                         x0: int = 0, y0: int = 600, known: Sequence[str] = ()) -> List[str]:
    """Backdrop MERGEMATTE: the plate `Read` (manifest beauty, its colorspace) → chain of `Dot`s, per object
    `Copy` alpha from a wired stamp of `Anchor_<obj>` (MASKI) → `Premult` → group `MergeMatte_<obj>` (engine/merge_matte_nk,
    knobs = the object's Final Touch settings; input BG = a `Dot` labelled „BG – podłącz”) → anchor `FUSE_<obj>` + free stamp.
    RGB always from the original plate, alpha = the exported matte; the export's alpha files are not changed."""
    from engine.merge_matte_nk import group_nk
    objs = [o for o in merge["objects"] if not known or o in known]
    if not objs:
        return []
    fmt = f'"{w} {h} 0 0 {w} {h} 1 MatteDesk_{_nk_name(shot)}"'
    cs = NUKE_ACES_CS.get(str(merge.get("colorspace") or ""))
    rl = "plate (beauty)" + ("" if cs else f" · colorspace {merge.get('colorspace') or '?'} – ustaw ręcznie")
    out = ["Read {", " inputs 0", f" file {_q(merge['plate'])}", f" format {fmt}", f" first {first}", f" last {last}",
           f" origfirst {first}", f" origlast {last}", " origset true"] + ([f" colorspace {_q(cs)}"] if cs else []) + \
          [" on_error black", f" name Read_{_nk_name(shot)}_plate", f" label {_q(rl)}", f" xpos {x0}", f" ypos {y0 + YM_READ}", "}"]
    xy = [(x0, y0 + YM_READ)]
    prev = None
    col_w = COL_W * 2   # Copy + its stamp, the group's BG Dot: two node widths per column
    for i, o in enumerate(objs):
        n = _nk_name(o)
        x = x0 + i * col_w
        if prev:
            out.append(f"push ${prev}")
        dot = f"Dot_plate_{n}"
        out += ["Dot {", f" name {dot}", f" xpos {x + 34}", f" ypos {y0 + YM_DOT}", "}", f"set {dot} [stack 0]"]
        # Copy: input0 (B) = plate, input1 (A) = the matte stamp → push A first, then B (top of the stack = input0)
        out.append(f"push $Anchor_{n}")
        out += _wired(o, f"Stamp_matte_{n}", x + 140, y0 + YM_DOT + 40)
        out.append(f"push ${dot}")
        out += ["Copy {", " inputs 2", " from0 rgba.alpha", " to0 rgba.alpha", f" name Copy_alpha_{n}",
                f" label {_q('alfa: MATTE_' + o)}", f" xpos {x}", f" ypos {y0 + YM_COPY}", "}"]
        out += ["Premult {", f" name Premult_{n}", f" xpos {x}", f" ypos {y0 + YM_PREMULT}", "}"]
        out.append(f"set FG_{n} [stack 0]")
        # MergeMatte: input0 = BG, input1 = FG → push FG, then the BG Dot (no input)
        out.append(f"push $FG_{n}")
        out += ["Dot {", " inputs 0", f" name Dot_BG_{n}", ' label "BG – podłącz"', " note_font_size 16",
                f" xpos {x + 174}", f" ypos {y0 + YM_GROUP - 50}", "}"]
        out += group_nk(merge["objects"][o], name=f"MergeMatte_{n}", xpos=x, ypos=y0 + YM_GROUP,
                        label=f"v01 · {o} · {date}").rstrip("\n").split("\n")
        out += _anchor(f"FUSE_{o}", x, y0 + YM_ANCHOR, f"MergeMatte · {date}")
        xy += [(x + 34, y0 + YM_DOT), (x + 140, y0 + YM_DOT + 40), (x, y0 + YM_COPY), (x, y0 + YM_PREMULT),
               (x + 174, y0 + YM_GROUP - 50), (x, y0 + YM_GROUP), (x, y0 + YM_ANCHOR), (x, y0 + YM_ANCHOR + Y_STAMP - Y_ANCHOR)]
        prev = dot
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    bx, by = min(xs) - 60, min(ys) - 110
    bw, bh = max(xs) - min(xs) + 80 + 120, max(ys) - min(ys) + 40 + 170
    bd = ["BackdropNode {", " inputs 0", " name BD_MERGEMATTE", f' label "MERGEMATTE\\n{shot} · {date}\\nRGB z plate\'u, alfa z MASKI"',
          " note_font_size 42", " tile_color 0x2f8f5fff", f" xpos {bx}", f" ypos {by}", f" bdwidth {bw}", f" bdheight {bh}", "}"]
    return bd + out


def nk_nodes(text: str) -> List[Dict[str, Any]]:
    """Top-level nodes of a generated script: [{class, knobs{name: raw value}}] (for tests / checks)."""
    nodes, cur, depth = [], None, 0
    for line in text.splitlines():
        s = line.strip()
        if depth == 0:
            m = re.match(r"^([A-Za-z0-9_]+) \{$", s)
            if m:
                cur = {"class": m.group(1), "knobs": {}}
                depth = 1
            continue
        if s == "}" and depth == 1:
            nodes.append(cur)
            cur, depth = None, 0
            continue
        k, _, v = s.partition(" ")
        cur["knobs"].setdefault(k, v)
    return nodes


def write_mattes_nk(path: Path, outputs: Dict[str, Any], shot: str, first: int, last: int, w: int, h: int,
                    channels: Optional[Sequence[str]] = None, merge: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mattes_nk(outputs, shot, first, last, w, h, channels=channels, merge=merge), encoding="utf-8")
    return path
