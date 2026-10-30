# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MergeMatte v01 as Nuke script text (standard nodes only, no MergeExpression) – the group MatteDesk exports to Nuke with
the object's edge-colour settings written into its knobs, and the reference the Python port (engine/merge_matte.py) follows
node by node.

MergeMatte = Merge(over) of a premultiplied FG with edge treatments: Alpha (choke, soften) → Decontaminate (decon,
decon_spread) → Darken edge (darken, darken_width) → Match edge (match, match_exposure, match_saturation, match_width) →
Light wrap (wrap, wrap_size, flare, flare_threshold, flare_softness) → Bleed colour (bleed, bleed_size) → Edge blur
(edge_blur) → log (ACEScct), mix, bypass. Inputs: 0 = BG, 1 = FG. Operations re-created from the user's SD_Fuse v01
(a rebuild of the L_Fuse light-wrap idea, Luma Pictures), knob names and order are MergeMatte's own.

`group_nk(knobs)` returns the text of a Group node (paste / Import Script); `write_group(path)` stores MergeMatte_v01.nk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

NODE_NAME = "MergeMatte"
VERSION = "v01"

# knob, label, (lo, hi) | "bool" | None (text), default, tooltip – one source for the .nk, the port and the UI palette.
# Defaults are NEUTRAL (user decision 2026-10-30, „our starting point”): every amount 0 and log off, so the group at its
# defaults = a plain linear Merge over (tests/test_merge_matte_api.py::test_defaults_are_a_plain_over)
KNOB_SPECS: List[Tuple[str, str, object, object, str]] = [
    ("t_alpha", "<b><u>Alpha</u></b>  (FG matte, before anything else)", None, None, ""),
    ("choke", "choke (+) / expand (-) [px]", (-20, 20), 0.0, "FilterErode (gaussian) of the FG alpha"),
    ("soften", "soften [px]", (0, 20), 0.0, "Blur of the FG alpha after the choke"),
    ("t_decon", "<b><u>Decontaminate</u></b>  (interior colour replaces old-background colour in the edge)", None, None, ""),
    ("decon", "amount", (0, 1), 0.0, ""),
    ("decon_spread", "spread [px]", (1, 100), 30.0, "how far the interior colour is spread into the edge"),
    ("t_darken", "<b><u>Darken edge</u></b>  (bright FG edges go down to the BG level, only over a darker BG)", None, None, ""),
    ("darken", "amount", (0, 1), 0.0, "min(comp, BG) inside the edge band"),
    ("darken_width", "width [px]", (0, 30), 8.0, ""),
    ("t_match", "<b><u>Match edge</u></b>  (grade the edge band to sit in the BG)", None, None, ""),
    ("match", "amount", (0, 1), 0.0, ""),
    ("match_exposure", "exposure [stops]", (-3, 3), 0.0, ""),
    ("match_saturation", "saturation", (0, 2), 1.0, ""),
    ("match_width", "width [px]", (0, 30), 4.0, ""),
    ("t_wrap", "<b><u>Light wrap</u></b>  (BG light wraps over the FG edge)", None, None, ""),
    ("wrap", "wrap amount", (0, 1), 0.0, "all BG values wrap over the FG"),
    ("wrap_size", "size [px]", (0, 20), 5.0, "wrap and flare share this size (x1, x2, x4, x8 blur stack)"),
    ("flare", "flare amount", (0, 1), 0.0, "only the bright parts of the BG (luminance key) wrap over the FG"),
    ("flare_threshold", "flare threshold", (0, 1), 0.6, "BG luminance where the flare key starts"),
    ("flare_softness", "flare softness", (0, 1), 0.2, "key ramp width above the threshold"),
    ("t_bleed", "<b><u>Bleed colour</u></b>  (BG hue replaces FG hue near the edge)", None, None, ""),
    ("bleed", "amount", (0, 1), 0.0, ""),
    ("bleed_size", "size [px]", (0, 10), 5.0, ""),
    ("t_edge", "<b><u>Edge blur</u></b>  (blurs the semi-transparent edge of the composite)", None, None, ""),
    ("edge_blur", "amount", (0, 10), 0.0, ""),
    ("div", "", None, None, ""),
    ("log", "Apply operations in Log (ACEScct)", "bool", False, ""),
    ("mix", "mix", (0, 1), 1.0, ""),
    ("bypass", "Regular Over", "bool", False, "bypass: plain Merge over (after Alpha / Log)"),
]
KNOB_DEFAULTS: Dict[str, float] = {n: (float(v) if r != "bool" else (1 if v else 0))
                                   for n, _, r, v, _ in KNOB_SPECS if r is not None}
KNOB_RANGES: Dict[str, Tuple[float, float]] = {n: r for n, _, r, _, _ in KNOB_SPECS if isinstance(r, tuple)}
# what a typed value may reach (user request 2026-10-30, as Nuke: the slider range is not a limit for a typed value):
# 4x the slider's end on each side away from 0 (choke -80..80, decon_spread up to 400, amounts up to 4)
KNOB_FIELD_RANGES: Dict[str, Tuple[float, float]] = {n: (lo * 4 if lo < 0 else lo, hi * 4 if hi > 0 else hi)
                                                     for n, (lo, hi) in KNOB_RANGES.items()}
# the user's SH010 look (SD_Fuse v01 values, 2026-09-30) in MergeMatte names
PRESET_SH010: Dict[str, float] = {**KNOB_DEFAULTS, "log": 1, "choke": 6.0, "soften": 2.5, "decon": 1.0, "decon_spread": 100.0,
                                  "darken": 1.0, "darken_width": 23.2, "match": 0.155, "match_exposure": -0.75,
                                  "match_width": 4.0, "wrap": 0.355, "wrap_size": 5.0, "flare": 0.3,
                                  "flare_threshold": 1.0, "flare_softness": 0.0, "edge_blur": 2.0}


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("[", "\\[").replace("]", "\\]") + '"'


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.6g}"
        return s
    return str(v)


def user_knobs(values: Mapping[str, float]) -> List[str]:
    out = [f" addUserKnob {{20 mergematte l {NODE_NAME}}}"]
    for name, label, rng, default, tip in KNOB_SPECS:
        t = f" t {_q(tip)}" if tip else ""
        if rng is None:
            out.append(f' addUserKnob {{26 {name} l "" +STARTLINE T {_q(label)}}}')
        elif rng == "bool":
            out.append(f" addUserKnob {{6 {name} l {_q(label)}{t} +STARTLINE}}")
            out.append(f" {name} {'true' if values.get(name, default) else 'false'}")
        else:
            out.append(f" addUserKnob {{7 {name} l {_q(label)}{t} R {_fmt(rng[0])} {_fmt(rng[1])}}}")
            out.append(f" {name} {_fmt(float(values.get(name, default)))}")
    out.append(' addUserKnob {41 bbox l "set bbox to" T fg_over_bg.bbox}')
    return out


class _Emit:
    """Nuke script stack: every node is pushed for its inputs explicitly, then `set` for later reference."""

    def __init__(self):
        self.lines: List[str] = []
        self.ids: Dict[str, str] = {}

    def node(self, cls: str, name: str, inputs: Sequence[str], x: int, y: int, mask: bool = False, **knobs) -> str:
        for src in reversed(inputs):
            self.lines.append(f"push ${self.ids[src]}")
        head = f"{cls} {{"
        self.lines.append(head)
        if not inputs:
            self.lines.append(" inputs 0")
        elif mask:
            self.lines.append(f" inputs {len(inputs) - 1}+1")
        elif len(inputs) != 1:
            self.lines.append(f" inputs {len(inputs)}")
        for k, v in knobs.items():
            if isinstance(v, str) and v.startswith("{{"):
                self.lines.append(f" {k} {v}")
            elif isinstance(v, str):
                self.lines.append(f" {k} {_q(v) if (' ' in v or '<' in v) else v}")
            else:
                self.lines.append(f" {k} {_fmt(v)}")
        self.lines += [f" name {name}", f" xpos {x}", f" ypos {y}", "}"]
        nid = "N" + name
        self.ids[name] = nid
        self.lines.append(f"set {nid} [stack 0]")
        return name


def _wrap_stack(e: _Emit, src: str, size_expr: str, p: str, x: int, y: int) -> str:
    """Blur(size), Blur(2 size), Blur(4 size), Blur(8 size) merged over each other (b2 over b1, b3 over that, b4 over that)."""
    e.node("Blur", f"{p}_blur1", [src], x, y, size=f"{{{{{size_expr}}}}}")
    e.node("Blur", f"{p}_blur2", [src], x + 130, y + 40, size=f"{{{{{p}_blur1.size*2}}}}")
    e.node("Blur", f"{p}_blur3", [src], x + 130, y + 80, size=f"{{{{{p}_blur2.size*2}}}}")
    e.node("Blur", f"{p}_blur4", [src], x + 130, y + 120, size=f"{{{{{p}_blur3.size*2}}}}")
    e.node("Merge2", f"{p}_add2", [f"{p}_blur1", f"{p}_blur2"], x, y + 60)
    e.node("Merge2", f"{p}_add3", [f"{p}_add2", f"{p}_blur3"], x, y + 100)
    return e.node("Merge2", f"{p}_add4", [f"{p}_add3", f"{p}_blur4"], x, y + 140)


def group_nk(knobs: Optional[Mapping[str, float]] = None, name: str = NODE_NAME, xpos: int = 0, ypos: int = 0,
             label: str = VERSION) -> str:
    """Text of the MergeMatte group with `knobs` (MergeMatte names) written in; pasteable / importable in Nuke."""
    vals = dict(KNOB_DEFAULTS)
    if knobs:
        vals.update({k: float(v) for k, v in knobs.items() if k in KNOB_DEFAULTS})
    e = _Emit()
    off = "{{parent.choke==0 && parent.soften==0}}"
    nolog = "{{1-parent.log}}"
    e.node("Input", "BG", [], 900, 0)
    e.node("Input", "FG", [], 0, 0, number=1)
    # Alpha
    e.node("Unpremult", "choke_unpremult", ["FG"], 0, 60, disable=off)
    e.node("FilterErode", "choke_erode", ["choke_unpremult"], 0, 100, channels="alpha", size="{{parent.choke}}", filter="gaussian", disable=off)
    e.node("Blur", "choke_soften", ["choke_erode"], 0, 140, channels="alpha", size="{{parent.soften}}", disable=off)
    e.node("Premult", "choke_premult", ["choke_soften"], 0, 180, disable=off)
    # Log
    e.node("Unpremult", "fg_unpremult_log", ["choke_premult"], 0, 260, disable=nolog)
    e.node("OCIOColorSpace", "fg_to_log", ["fg_unpremult_log"], 0, 300, in_colorspace="ACES - ACEScg", out_colorspace="ACES - ACEScct", disable=nolog)
    e.node("Premult", "fg_premult_log", ["fg_to_log"], 0, 340, disable=nolog)
    e.node("Dot", "Dot_fg", ["fg_premult_log"], 34, 400)
    e.node("OCIOColorSpace", "bg_to_log", ["BG"], 900, 300, in_colorspace="ACES - ACEScg", out_colorspace="ACES - ACEScct", disable=nolog)
    e.node("Dot", "Dot_bg", ["bg_to_log"], 934, 400)
    e.node("Merge2", "regular_over", ["Dot_bg", "Dot_fg"], 1200, 460)
    e.node("Dot", "Dot_regular", ["regular_over"], 1234, 2280)
    # Decontaminate on the straight FG colour
    e.node("Unpremult", "fg_unpremult", ["Dot_fg"], 0, 460)
    e.node("Dot", "Dot_decon", ["fg_unpremult"], 34, 500)
    e.node("Dot", "Dot_decon_r", ["Dot_decon"], -166, 500)
    e.node("Invert", "decon_edge_mask", ["Dot_decon"], -350, 570, channels="alpha", label="1 - alpha")
    e.node("Dot", "Dot_decon_m", ["decon_edge_mask"], -316, 650)
    e.node("Premult", "decon_premult", ["Dot_decon_r"], -200, 530)
    e.node("Blur", "decon_spread", ["decon_premult"], -200, 570, channels="rgba", size="{{parent.decon_spread}}")
    e.node("Unpremult", "decon_unpremult", ["decon_spread"], -200, 610)
    e.node("Keymix", "decon_mix", ["Dot_decon", "decon_unpremult", "Dot_decon_m"], 0, 644, channels="rgb", mix="{{parent.decon}}",
           label="interior colour into edge", disable="{{parent.decon==0}}")
    e.node("Dot", "Dot_fgd", ["decon_mix"], 34, 720)
    e.node("Dot", "Dot_fgd_alpha", ["Dot_fgd"], -116, 1520)
    # BG outside the FG
    e.node("Copy", "bg_take_fg_alpha", ["Dot_bg", "Dot_fg"], 600, 460, from0="rgba.alpha", to0="rgba.alpha")
    e.node("Invert", "invert_alpha", ["bg_take_fg_alpha"], 600, 700, channels="alpha")
    e.node("Premult", "bg_outside_fg", ["invert_alpha"], 600, 740)
    e.node("Dot", "Dot_bgout", ["bg_outside_fg"], 634, 800)
    # Light wrap: flare (luma-keyed BG) and wrap (all BG), same size
    e.node("Keyer", "flare_key", ["Dot_bg"], 900, 800, operation="luminance key",
           range="{{parent.flare_threshold} {parent.flare_threshold+parent.flare_softness} 1 1}")
    e.node("Premult", "flare_premult", ["flare_key"], 900, 840)
    e.node("Merge2", "flare_outside_fg", ["flare_premult", "Dot_bgout"], 900, 880, operation="mask")
    e.node("Dot", "Dot_flare0", ["flare_outside_fg"], 934, 920)
    _wrap_stack(e, "Dot_flare0", "parent.wrap_size", "flare", 900, 960)
    e.node("Grade", "flare_amount", ["flare_add4"], 900, 1176, channels="all", white="{{parent.flare}}")
    e.node("Dot", "Dot_flare", ["flare_amount"], 934, 1470)
    e.node("Dot", "Dot_bgout2", ["Dot_bgout"], 634, 840)
    _wrap_stack(e, "Dot_bgout2", "parent.wrap_size", "wrap", 600, 880)
    e.node("Grade", "wrap_amount", ["wrap_add4"], 600, 1160, channels="all", white="{{parent.wrap}}")
    e.node("Dot", "Dot_wrap", ["wrap_amount"], 634, 1420)
    # Bleed colour
    e.node("Dot", "Dot_bgout3", ["Dot_bgout"], 334, 840)
    _wrap_stack(e, "Dot_bgout3", "parent.bleed_size", "bleed", 300, 880)
    e.node("Clamp", "bleed_clamp", ["bleed_add4"], 300, 1120, minimum=0.001, maximum=100)
    e.node("Dot", "Dot_bleed", ["bleed_clamp"], 334, 1160)
    e.node("Dot", "Dot_bleed_mask", ["Dot_bleed"], 334, 1360)
    e.node("Saturation", "bleed_luma", ["Dot_bleed"], 420, 1190, saturation=0)
    e.node("Merge2", "bleed_chroma", ["Dot_bleed", "bleed_luma"], 300, 1230, operation="divide")
    e.node("Saturation", "fg_desaturate", ["Dot_fgd"], 150, 1230, saturation="{{1-parent.bleed}}")
    e.node("Merge2", "bleed_apply", ["fg_desaturate", "bleed_chroma"], 150, 1280, operation="multiply", mix="{{parent.bleed}}")
    e.node("Keymix", "bleed_near_edge", ["Dot_fgd", "bleed_apply", "Dot_bleed_mask"], 0, 1360)
    e.node("Merge2", "add_wrap", ["bleed_near_edge", "Dot_wrap"], 0, 1420)
    e.node("Merge2", "add_flare", ["add_wrap", "Dot_flare"], 0, 1470)
    e.node("Copy", "restore_fg_alpha", ["add_flare", "Dot_fgd_alpha"], 0, 1520, from0="rgba.alpha", to0="rgba.alpha")
    e.node("Premult", "fg_premult", ["restore_fg_alpha"], 0, 1560)
    e.node("Dot", "Dot_bg2", ["Dot_bg"], 934, 1610)
    e.node("Merge2", "fg_over_bg", ["Dot_bg2", "fg_premult"], 0, 1610)
    # Darken edge
    e.node("Blur", "darken_band_blur", ["fg_over_bg"], 400, 1650, channels="alpha", size="{{parent.darken_width}}")
    e.node("Dot", "Dot_darken_band", ["darken_band_blur"], 434, 1700)
    e.node("Merge2", "darken_band", ["darken_band_blur", "Dot_darken_band"], 400, 1740, operation="xor")
    e.node("Dot", "Dot_bg3", ["Dot_bg2"], 934, 1670)
    e.node("Merge2", "darken_min", ["fg_over_bg", "Dot_bg3"], 200, 1670, operation="min", label="min(comp, bg)")
    e.node("Keymix", "darken_mix", ["fg_over_bg", "darken_min", "darken_band"], 0, 1760, mix="{{parent.darken}}",
           label="only in edge band", disable="{{parent.darken==0}}")
    # Match edge
    e.node("Dot", "Dot_match", ["darken_mix"], 34, 1800)
    e.node("Dot", "Dot_match_r", ["Dot_match"], 234, 1800)
    e.node("Blur", "match_band_blur", ["Dot_match_r"], 400, 1830, channels="alpha", size="{{parent.match_width}}")
    e.node("Dot", "Dot_match_b", ["match_band_blur"], 434, 1870)
    e.node("Merge2", "match_band", ["match_band_blur", "Dot_match_b"], 400, 1910, operation="xor")
    e.node("Dot", "Dot_match_x", ["match_band"], 434, 1956)
    e.node("Grade", "match_exposure", ["Dot_match_r"], 200, 1830, multiply="{{pow(2,parent.match_exposure)}}")
    e.node("Saturation", "match_saturation", ["match_exposure"], 200, 1870, saturation="{{parent.match_saturation}}")
    e.node("Dot", "Dot_match_s", ["match_saturation"], 234, 1930)
    e.node("Keymix", "match_mix", ["Dot_match", "Dot_match_s", "Dot_match_x"], 0, 1950, channels="rgb", mix="{{parent.match}}",
           label="only in edge band", disable="{{parent.match==0}}")
    # Edge blur
    e.node("Dot", "Dot_comp", ["match_mix"], 34, 2080)
    e.node("Blur", "edge_band_blur", ["Dot_comp"], 200, 2110, size=3)
    e.node("Dot", "Dot_edge_band", ["edge_band_blur"], 234, 2150)
    e.node("Merge2", "edge_band_xor", ["edge_band_blur", "Dot_edge_band"], 200, 2190, operation="xor")
    e.node("Grade", "edge_band_level", ["edge_band_xor"], 200, 2230, channels="rgba", whitepoint=0.9, white_clamp=True)
    e.node("Blur", "edge_blur", ["Dot_comp", "edge_band_level"], 0, 2230, mask=True, size="{{parent.edge_blur}}")
    e.node("Switch", "bypass_switch", ["edge_blur", "Dot_regular"], 0, 2280, which="{{parent.bypass}}")
    e.node("Dot", "Dot_bg4", ["Dot_bg2"], 934, 2330)
    e.node("Dissolve", "overall_mix", ["Dot_bg4", "bypass_switch"], 0, 2330, which="{{parent.mix}}")
    e.node("OCIOColorSpace", "from_log", ["overall_mix"], 0, 2380, in_colorspace="ACES - ACEScct", out_colorspace="ACES - ACEScg", disable=nolog)
    e.node("Output", "Output1", ["from_log"], 0, 2440)
    head = ["Group {", " inputs 2", f" name {name}", " tile_color 0x2f8f5fff", f" label {_q(label)}", " note_font_size 14",
            f" xpos {xpos}", f" ypos {ypos}"] + user_knobs(vals) + ["}"]
    return "\n".join(head + e.lines + ["end_group"]) + "\n"


def paste_text(knobs: Optional[Mapping[str, float]] = None, **kw) -> str:
    """Clipboard form (like a node copied in Nuke): paste with Ctrl+V or nuke.nodePaste(file)."""
    return "set cut_paste_input [stack 0]\nversion 15.0 v1\npush 0\npush $cut_paste_input\n" + group_nk(knobs, **kw)


def write_group(path: Path, knobs: Optional[Mapping[str, float]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paste_text(knobs), "utf-8", newline="\n")
    return path


def node_names(text: str) -> List[str]:
    return [ln.strip().split()[1] for ln in text.splitlines() if ln.startswith(" name ")]


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "handoff/nuke" / f"{NODE_NAME}_{VERSION}.nk"
    print(write_group(out), len(node_names(group_nk())), "nodes")
