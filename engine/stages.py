# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/stages.py - Pipeline stages of one object (stage layout of the user decision 2026-10-15).

Fine-grained stages (`stages`, what the backend checks before an action):
  detect    – instance chosen / a key saved (points, prompt, candidate)
  crops     – crop segments -> crops.json (automatic in „▶▶ Przelicz całość”)
  core      – SAM3 core over the shot -> core/core.json, newer than the keys and crops
  cotracker – brush corrections of the mask (zones `params.source = brush`, propagated by CoTracker3) applied on the
              shot -> core_fix/ (mask 1+2); optional: without corrections it is done
  matanyone – soft edge from the mask 1+2, matte zones skipped -> tool.stage_runs.matanyone (base version)
  videomama – VideoMaMa (engine/videomama.py) guided by the mask 1+2, put over a base (mask 1+2 or the stage-3 result);
              optional, user decision 2026-10-16 (A/B test local_results/videomama_m2) -> tool.stage_runs.videomama
  vitmatte  – matte zones (ViTMatte) applied on the stage-3 base -> tool.stage_runs.vitmatte
  final     – „Przelicz finalnie” with everything (experiments: edge modes, threshold zones, edge fix) -> active run
  ltx       – LTX-2.5 Alpha-Gen (generative matte, guided by the mask 1+2; engine/ltx_alpha.py; optional, user decision
              2026-10-16: a stage of its own before the export, no longer an edge mode of the experiments) -> tool.stage_runs.ltx
  export    – mattes exported (outputs.json has a current version of the object)
Steps of the stage bar (`steps`): 1 SAM3 = detect + crops + core, 2 CoTracker, 3 MatAnyone, 4 VideoMaMa, 5 ViTMatte,
6 Eksperymenty = final, 7 LTX, 8 Eksport (stage 0 Plate'y lives in the UI).

State of a stage: "done" | "stale" (done, but its inputs changed since -> recompute) | "todo" (can run now) |
"locked" (a stage it needs (NEEDS) is not done). The backend refuses an action whose stage is "locked" (HTTP 409)
with the reason; the UI greys the button out and highlights the next action.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ORDER = ["detect", "crops", "core", "cotracker", "matanyone", "videomama", "vitmatte", "final", "ltx", "export"]
LABEL = {"detect": "Wykryj (SAM3)", "crops": "Kadry", "core": "Rdzeń SAM3", "cotracker": "CoTracker (poprawki pędzlem)",
         "matanyone": "MatAnyone (krawędź)", "videomama": "VideoMaMa (kosmyki)", "vitmatte": "ViTMatte (strefy)", "final": "Eksperymenty (Przelicz finalnie)",
         "ltx": "LTX Alpha-Gen", "export": "Eksport"}
# what a stage needs done before it can run (2026-10-15: export right after the core, stages 2 / 5 optional)
NEEDS = {"detect": [], "crops": ["detect"], "core": ["crops"], "cotracker": ["core"], "matanyone": ["core"],
         "videomama": ["cotracker"], "vitmatte": ["matanyone"], "final": ["core"], "ltx": ["core"], "export": ["core"]}
# the bar of the UI: step number -> fine-grained stages it shows
STEPS = [(1, "sam3", "SAM3", ["detect", "crops", "core"]), (2, "cotracker", "CoTracker", ["cotracker"]),
         (3, "matanyone", "MatAnyone", ["matanyone"]), (4, "videomama", "VideoMaMa", ["videomama"]),
         (5, "vitmatte", "ViTMatte", ["vitmatte"]), (6, "final", "Eksperymenty", ["final"]), (7, "ltx", "LTX", ["ltx"]),
         (8, "export", "Eksport", ["export"])]
NEXT_CHAIN = ["detect", "crops", "core", "matanyone", "export"]   # the main path ("następny krok"); 2, 4–7 optional
# object settings the edge pass depends on (stage 3 is stale after a change)
EDGE_KEYS = ("band_scale", "shift_px", "temporal_smoothing", "thin_lens", "lupa_box", "ma_scale_mode", "core_stabilize",
             "core_soft_start", "key_band_same")
BRUSH_SOURCE = "brush"     # zone params.source of the stage-2 brush corrections
# base of stage 4 (object field `vitmatte_base`, user decision 2026-10-15 – both to be compared): "core" = the mask 1+2
# (hard 0 / 1, stage 4 open right after stages 1 + 2), "matanyone" = the stage-3 version (default, the tested recipe),
# "videomama" = the stage VideoMaMa version (2026-10-16)
VITMATTE_BASES = ("matanyone", "core", "videomama")
BASE_LABEL = {"matanyone": "wynik MatAnyone, etap 3", "core": "maska 1+2, etapy 1 + 2", "videomama": "wynik VideoMaMa"}


def vitmatte_base(obj: Dict[str, Any]) -> str:
    """The base ViTMatte lays its zones on: the object's choice, else (nothing chosen yet, bug 2026-10-17: the stage
    stayed locked behind MatAnyone while the mask 1+2 was ready) the best result that exists – MatAnyone, VideoMaMa,
    otherwise the mask 1+2."""
    b = obj.get("vitmatte_base")
    if b in VITMATTE_BASES:
        return b
    sr = (obj.get("tool") or {}).get("stage_runs") or {}
    return "matanyone" if sr.get("matanyone") else ("videomama" if sr.get("videomama") else "core")


def vmm_base(obj: Dict[str, Any]) -> str:
    """Base of stage VideoMaMa (object field videomama.base): "core" (mask 1+2, default) | "matanyone"."""
    b = (obj.get("videomama") or {}).get("base") or "core"
    return b if b in ("core", "matanyone") else "core"


def needs_of(obj: Dict[str, Any]) -> Dict[str, List[str]]:
    """NEEDS of this object: ViTMatte opens right after stage 2 whatever its base (the mask 1+2 is always a possible
    base and the base selector lives inside the stage – locking it behind MatAnyone made the choice unreachable, bug
    2026-10-17; a base without a result is refused when computing, with the reason); VideoMaMa on the mask 1+2 waits
    for stage 2, on MatAnyone for stage 3."""
    return {**NEEDS, "vitmatte": ["cotracker"],
            "videomama": ["cotracker"]}   # same for VideoMaMa: its base selector is inside the stage (bug 2026-10-17)


def _sha(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def zone_kind(z: Any) -> str:
    """'brush' (stage 2 correction: add / remove on the core), 'matte' (stage 4 ViTMatte zone) or 'threshold' (the old
    dark / bright / colour / all zones of „Zaawansowane”, stage 5). `z` = Zone or its dict."""
    d = z if isinstance(z, dict) else z.__dict__
    if d.get("criterion") == "matte":
        return "matte"
    return "brush" if (d.get("params") or {}).get("source") == BRUSH_SOURCE else "threshold"


def zone_dicts(od: Path, kinds: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Enabled zones of the given kinds (raw JSON dicts, creation order)."""
    d = Path(od) / "zones"
    out = []
    for p in sorted(d.glob("z_*.json")) if d.exists() else []:
        try:
            z = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if z.get("enabled", True) and zone_kind(z) in kinds:
            out.append(z)
    return out


def core_fix_sig(store: Any, name: str) -> str:
    """Signature of the mask 1+2 (SAM3 core + brush corrections): core.json, the brush zones, core stabilisation."""
    od = store.dir(name)
    try:
        cj = (od / "core" / "core.json").read_text(encoding="utf-8")
    except OSError:
        cj = ""
    obj = store.get(name)
    return _sha({"core": cj, "zones": zone_dicts(od, ("brush",)), "stab": obj.get("core_stabilize", True)})


def group_parts(store: Any, name: str) -> List[str]:
    """Parts of `name` (objects with part_of == name), depth order – a parent with parts is a group computed as a whole
    in stages 3 / 4 (user decision 2026-10-15: linked objects are computed with their seam)."""
    return [o["name"] for o in sorted(store.list(), key=lambda o: o.get("depth_order", 0))
            if o.get("part_of") == name and o["name"] != name]


def mask_sig(store: Any, name: str) -> str:
    """Signature of the mask the edge pass starts from: the object's mask 1+2, or for a parent with parts the masks
    1+2 of every member + the seam width (the whole of the group)."""
    parts = group_parts(store, name)
    if not parts:
        return core_fix_sig(store, name)
    return _sha({"members": [[m, core_fix_sig(store, m)] for m in [name] + parts],
                 "seam": store.get(name).get("part_seam_px")})


# px mode of the edge band (2026-10-16): part of the signature only when set – objects without it keep their signature
EDGE_KEYS_OPT = ("band_out_px", "band_in_px", "band_in_feather_px")


def settings_sig(obj: Dict[str, Any]) -> str:
    d = {k: obj.get(k) for k in EDGE_KEYS}
    d.update({k: obj[k] for k in EDGE_KEYS_OPT if obj.get(k) is not None})
    return _sha(d)


def matte_sig(od: Path) -> str:
    return _sha(zone_dicts(od, ("matte",)))


def _run_json(od: Path, run: Optional[str]) -> Optional[Dict[str, Any]]:
    if not run:
        return None
    try:
        return json.loads((Path(od) / "runs" / run / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stage_run(store: Any, name: str, stage: str, base: Optional[str] = None) -> Tuple[Optional[str], bool, str]:
    """(run, valid, why) of the result of stage `stage` ("matanyone" | "core_base" | "videomama" | "vitmatte" | "ltx"):
    tool.stage_runs; valid
    = computed from the current mask 1+2 and settings. "core_base" = the mask 1+2 written as a version (base of
    ViTMatte on 1+2). "vitmatte" = the zones on the base `base` (default: the object's vitmatte_base), current zones."""
    od = store.dir(name)
    obj = store.get(name)
    if obj.get("part_of") and obj["part_of"] != name:
        # a part: stages 3 / 4 compute its group as a whole on the parent – its result is the parent's (when the
        # parent's version really is a group version with this part)
        try:
            prun, pok, pwhy = stage_run(store, obj["part_of"], stage, base)
        except FileNotFoundError:
            return None, False, ""
        prj = _run_json(store.dir(obj["part_of"]), prun)
        if prun and prj and name in (prj.get("group") or []):
            return prun, pok, pwhy
        return None, False, ""
    runs = (obj.get("tool") or {}).get("stage_runs") or {}
    if stage == "vitmatte":
        base = base or vitmatte_base(obj)
        run = runs.get(f"vitmatte@{base}") or (runs.get("vitmatte") if base == "matanyone" else None)
    else:
        run = runs.get(stage)
    rj = _run_json(od, run)
    if rj is None:
        return None, False, ""
    if stage == "vitmatte" and (rj.get("base") or "matanyone") != base:   # "vitmatte" fallback made on the other base
        return None, False, ""
    if stage == "ltx":
        from engine.ltx_alpha import object_settings, settings_sig as ltx_sig
        if rj.get("core_fix_sig") != mask_sig(store, name):
            return run, False, "rdzeń albo poprawki pędzlem zmienione od ostatniego liczenia LTX"
        if rj.get("ltx_sig") != ltx_sig(object_settings(obj)):
            return run, False, "ustawienia LTX zmienione od ostatniego liczenia"
        return run, True, ""
    if stage == "videomama":
        from engine.videomama import object_settings as vmm_settings, settings_sig as vmm_sig
        if rj.get("core_fix_sig") != mask_sig(store, name):
            return run, False, "rdzeń albo poprawki pędzlem zmienione od ostatniego liczenia VideoMaMa"
        s = vmm_settings(obj)
        if rj.get("vmm_sig") != vmm_sig(s):
            return run, False, "ustawienia VideoMaMa zmienione od tej wersji – „▶ Policz ponownie” (albo najpierw „👁 Podgląd klatek”)"
        if s["base"] == "matanyone":
            m_run, m_ok, _ = stage_run(store, name, "matanyone")
            if rj.get("applied_from") != m_run or not m_ok:
                return run, False, "baza (wynik MatAnyone, etap 3) zmieniona od tej wersji – „▶ Policz ponownie”"
        return run, True, ""
    if stage in ("matanyone", "core_base"):
        core_t = _mtime(od / "core" / "core.json") or 0.0
        if (_mtime(od / "runs" / run / "run.json") or 0.0) < core_t or rj.get("core_fix_sig") != mask_sig(store, name):
            return run, False, ("rdzeń albo poprawki pędzlem zmienione od ostatniego liczenia MatAnyone" if stage == "matanyone"
                                else "rdzeń albo poprawki pędzlem zmienione – maska 1+2 do przeliczenia")
        if rj.get("settings_sig") != settings_sig(obj):
            return run, False, "ustawienia krawędzi zmienione od ostatniego liczenia " + ("MatAnyone" if stage == "matanyone" else "maski 1+2")
        return run, True, ""
    b_run, b_ok, _ = stage_run(store, name, {"matanyone": "matanyone", "videomama": "videomama"}.get(base, "core_base"))
    if rj.get("applied_from") != b_run or not b_ok:
        return run, False, f"baza ({BASE_LABEL.get(base, base)}) zmieniona od tej wersji – „Zastosuj na ujęcie” ponownie"
    if rj.get("matte_sig") != matte_sig(od):
        return run, False, "strefy ViTMatte zmienione od ostatniego „Zastosuj na ujęcie”"
    return run, True, ""


def core_fix_state(store: Any, name: str) -> Tuple[bool, bool]:
    """(exists, valid) of core_fix/ (mask 1+2 on the shot, written by „Zastosuj poprawki”)."""
    p = store.dir(name) / "core_fix" / "core_fix.json"
    try:
        sig = json.loads(p.read_text(encoding="utf-8")).get("sig")
    except (OSError, ValueError):
        return False, False
    return True, sig == core_fix_sig(store, name)


def _mtime(p: Path) -> Optional[float]:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def keys_signature(store: Any, name: str) -> List[List[Any]]:
    """[(frame, author, time)] of the object's keys – what a core run was computed from."""
    kd = store.dir(name) / "keys"
    return [[int(f), k.get("author"), k.get("t"), round(_mtime(kd / f"k_{int(f):04d}_mask.png") or 0.0, 3)]
            for f, k in sorted(store.keys(name).items())]


def core_denoise(od: Path) -> Optional[bool]:
    """Plate-to-mask denoise the core in `od/core/core.json` was computed with (None = no core). Cores from before
    the project switch (2026-10-13) have no key: they were computed with the denoise ON (defaults mplate.enabled)."""
    cj = Path(od) / "core" / "core.json"
    if not cj.exists():
        return None
    try:
        return bool(json.loads(cj.read_text(encoding="utf-8")).get("mask_plate_denoise", True))
    except (OSError, ValueError):
        return True


def denoise_changed(store: Any, name: str, mplate: Optional[bool]) -> bool:
    """True when the object has a core computed with a different plate-to-mask denoise than `mplate` (None = unknown)."""
    if mplate is None:
        return False
    used = core_denoise(store.dir(name))
    return used is not None and used != bool(mplate)


def plate_changes(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Z26: project.json -> plate_changed {plate: {at, frames}} – a plate repaired (`_fix` written / restored) in
    stage 0. Results computed before `at` from that plate are stale: beauty -> the core (SAM3 reads it; everything
    after the core follows), an auxiliary plate (backlit / clean plate) -> the final pass (backlit edge, QA)."""
    try:
        d = json.loads((Path(work_dir) / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in (d.get("plate_changed") or {}).items() if isinstance(v, dict) and v.get("at")}


def _beauty_change_after(od: Path, work_dir: Path) -> Optional[Dict[str, Any]]:
    """The beauty repair newer than the object's core (None = the core is not older than it / no core)."""
    ch = plate_changes(work_dir).get("beauty")
    core_t = _mtime(Path(od) / "core" / "core.json")
    return ch if (ch and core_t is not None and core_t < float(ch["at"])) else None


def plate_stale_objects(store: Any, work_dir: Path, plate: str) -> List[str]:
    """Objects with a result computed from `plate` before its last repair (stage 0 -> "do przeliczenia")."""
    ch = plate_changes(work_dir).get(plate)
    if not ch:
        return []
    out = []
    for o in store.list():
        od = store.dir(o["name"])
        if plate == "beauty":
            t = _mtime(od / "core" / "core.json")
        else:
            run = (o.get("tool") or {}).get("active_run")
            t = _mtime(od / "runs" / run / "run.json") if run else None
        if t is not None and t < float(ch["at"]):
            out.append(o["name"])
    return out


def life_changed(store: Any, name: str, frame_range: Optional[Tuple[int, int]]) -> List[List[int]]:
    """Frames whose visibility (object life range, engine/workarea.py) differs from the one the core was computed with
    (core.json -> life; absent = the whole shot)."""
    from engine.workarea import life_changed_intervals, object_life
    cj = store.dir(name) / "core" / "core.json"
    try:
        old = json.loads(cj.read_text(encoding="utf-8")).get("life")
    except (OSError, ValueError):
        return []
    new = object_life(store.get(name))
    if (old or None) == new:
        return []
    keys = sorted(int(f) for f in store.keys(name))
    lo_hi = frame_range or ((keys[0], keys[-1]) if keys else None)
    if lo_hi is None:
        return []
    fl = [x for iv in (old or []) + (new or []) for x in iv]
    shot = (min([int(lo_hi[0])] + fl), max([int(lo_hi[1])] + fl))
    return life_changed_intervals(old, new, shot)


def stale_intervals(store: Any, name: str, frame_range: Optional[Tuple[int, int]] = None,
                    mplate: Optional[bool] = None) -> List[List[int]]:
    """Frame intervals the next core run will recompute (Z2): around every key added / removed / changed since the
    last run (keys_sig in core.json vs now) – from the key before to the key after it (propagation runs both ways
    between neighbouring keys). No core.json or crops newer than the core -> the whole range."""
    od = store.dir(name)
    keys = sorted(int(f) for f in store.keys(name))
    lo, hi = frame_range if frame_range else ((keys[0], keys[-1]) if keys else (0, 0))
    cj = od / "core" / "core.json"
    if not cj.exists():
        return [[lo, hi]] if keys or frame_range else []
    if (_mtime(od / "crops.json") or 0.0) > (_mtime(cj) or 0.0) or denoise_changed(store, name, mplate):
        return [[lo, hi]]
    try:
        old = json.loads(cj.read_text(encoding="utf-8")).get("keys_sig")
    except (OSError, ValueError):
        old = None
    if old is None:
        return [[lo, hi]]
    new = keys_signature(store, name)
    a = {int(e[0]): list(e[1:]) for e in old}
    b = {int(e[0]): list(e[1:]) for e in new}
    changed = sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))
    # life range changed: every frame whose visibility changed (its span between the neighbouring keys is recomputed)
    changed = sorted(set(changed) | {f for a_, b_ in life_changed(store, name, (lo, hi)) for f in range(a_, b_ + 1)})
    # Z26: frames of the beauty repaired after the core (their span between the neighbouring keys is recomputed)
    bch = _beauty_change_after(od, Path(store.root).parent) if hasattr(store, "root") else None
    if bch:
        changed = sorted(set(changed) | {int(f) for f in bch.get("frames") or [] if lo <= int(f) <= hi})
    out: List[List[int]] = []
    for c in changed:
        prev = max((k for k in keys if k < c), default=lo)
        nxt = min((k for k in keys if k > c), default=hi)
        iv = [max(lo, min(prev, c)), min(hi, max(nxt, c))]
        if out and iv[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], iv[1])
        else:
            out.append(iv)
    return out


def object_stages(store: Any, name: str, work_dir: Path, frame_range: Optional[Tuple[int, int]] = None,
                  mplate: Optional[bool] = None, work_range: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """`mplate` = the project's effective plate-to-mask denoise (None = not checked): a core computed with the other
    setting is stale (the core and the final pass share the plate-to-mask).
    `frame_range` = the shot, `work_range` = the work area (engine/workarea.py; None = the whole shot). The core /
    final stage is "done" only when its result covers the work area (else "stale" with the missing frames); a result
    covering the work area but not the whole shot stays "done" with `scope: "range"` and `missing_shot` – it never
    passes for the whole shot (export "całość" -> 409)."""
    from engine.workarea import core_coverage, fmt_intervals, need_missing, object_life, run_coverage, union
    od = store.dir(name)
    obj = store.get(name)
    life = object_life(obj)
    tool = obj.get("tool", {})
    keys = store.keys(name)
    key_t = max((_mtime(od / "keys" / f"k_{f:04d}.json") or 0.0 for f in keys), default=0.0)
    crops_t = _mtime(od / "crops.json")
    core_t = _mtime(od / "core" / "core.json")
    run = tool.get("active_run")
    run_t = _mtime(od / "runs" / run / "run.json") if run else None
    exp_ver = None    # version of the object's last export (outputs.json): a run, or "core" (mask 1+2)
    try:
        out = json.loads((Path(work_dir) / "outputs.json").read_text(encoding="utf-8"))
        exp_ver = next((e.get("version") for e in out.get("objects", []) if e.get("name") == name), None)
    except (OSError, ValueError):
        exp_ver = None

    st: Dict[str, str] = {}
    why: Dict[str, str] = {}
    st["detect"] = "done" if keys else "todo"
    if not keys:
        why["detect"] = "wskaż obiekt: „Wykryj” i wybierz kandydata (albo zapisz klucz)"
    st["crops"] = "done" if crops_t else "todo"
    core_ok = core_t is not None and core_t >= (crops_t or 0.0)
    if core_ok:
        try:
            sig = json.loads((od / "core" / "core.json").read_text(encoding="utf-8")).get("keys_sig")
        except (OSError, ValueError):
            sig = None
        core_ok = (sig == keys_signature(store, name)) if sig is not None else core_t >= key_t
    life_diff = life_changed(store, name, frame_range) if core_t else []
    core_ok = core_ok and not life_diff
    dn_changed = denoise_changed(store, name, mplate)
    plate_ch = _beauty_change_after(od, Path(work_dir))     # Z26: beauty repaired in stage 0 after this core
    core_ok = core_ok and not plate_ch
    need = tuple(work_range) if work_range else (tuple(frame_range) if frame_range else None)
    core_cov = core_coverage(od) if core_t else None
    core_miss = need_missing(need, core_cov, life) if (need and core_cov is not None) else []
    st["core"] = "done" if core_ok and not dn_changed and not core_miss else ("stale" if core_t else "todo")
    if st["core"] == "stale":
        why["core"] = (f"plate beauty naprawiony w etapie 0 po ostatnim liczeniu rdzenia (klatki "
                       f"{', '.join(str(f) for f in (plate_ch.get('frames') or [])[:8])}) – „Przelicz zmienione przedziały”"
                       if plate_ch else
                       "odszumianie plate'u do maskowania zmienione w etapie 0 (rdzeń liczony z "
                       f"{'wł.' if not mplate else 'wył.'}) – przelicz rdzeń" if dn_changed else
                       f"zakres życia obiektu zmieniony po ostatnim liczeniu ({fmt_intervals(life_diff)}) – „Przelicz zmienione "
                       "przedziały”" if life_diff else
                       "klucze lub kadry zmienione po ostatnim liczeniu – „Przelicz zmienione przedziały”" if not core_ok else
                       f"rdzeń nie obejmuje zakresu pracy – brak {fmt_intervals(core_miss)}: „▶▶ Przelicz całość”")
    run_cov = run_coverage(od, run) if run else None
    final_miss = need_missing(need, run_cov, life) if (need and run_cov is not None) else []
    final_ok = bool(run and run_t and tool.get("final_valid") and core_t and run_t >= core_t)
    aux_ch = [k for k, v in plate_changes(work_dir).items() if k != "beauty" and run_t and run_t < float(v["at"])]
    final_ok = final_ok and not aux_ch      # Z26: backlit / clean plate repaired after the final pass
    st["final"] = "done" if final_ok and not final_miss else ("stale" if run else "todo")
    if st["final"] == "stale":
        why["final"] = (f"plate {', '.join(aux_ch)} naprawiony w etapie 0 po ostatnim „Przelicz finalnie”" if aux_ch else
                        "rdzeń albo ustawienia zmienione po ostatnim „Przelicz finalnie”" + (
            f" ({tool.get('status')})" if tool.get("status") else "")) if not final_ok else             f"wynik finalny nie obejmuje zakresu pracy – brak {fmt_intervals(final_miss)}: „Przelicz finalnie”"
    shot = tuple(frame_range) if frame_range else None
    cover = {"core": {"coverage": core_cov or [], "missing_shot": need_missing(shot, core_cov, life) if (shot and core_cov is not None) else []},
             "final": {"coverage": run_cov or [], "missing_shot": need_missing(shot, run_cov, life) if (shot and run_cov is not None) else []}}
    for k in ("core", "final"):
        cover[k]["scope"] = "range" if (st[k] == "done" and cover[k]["missing_shot"]) else "all"
        if cover[k]["scope"] == "range":
            why[k] = (f"gotowe w zakresie pracy ({fmt_intervals(cover[k]['coverage'])}); całe ujęcie: brak "
                      f"{fmt_intervals(cover[k]['missing_shot'])}")
    # stage 2 (CoTracker): brush corrections on the core -> core_fix/ (optional: no corrections = done)
    brush = zone_dicts(od, ("brush",))
    fix_exists, fix_ok = core_fix_state(store, name) if core_t else (False, False)
    if not brush:
        st["cotracker"] = "done"
        why["cotracker"] = "brak poprawek pędzlem – etap opcjonalny (maska = rdzeń SAM3)"
    else:
        st["cotracker"] = "done" if fix_ok else ("stale" if fix_exists else "todo")
        if not fix_ok:
            why["cotracker"] = (f"poprawki pędzlem ({len(brush)} stref) jeszcze nie zastosowane na ujęciu – „▶▶ Zastosuj "
                                "poprawki” (MatAnyone i tak je bierze)" if not fix_exists else
                                "rdzeń albo poprawki zmienione od ostatniego „Zastosuj poprawki”")
    # stage 3 (MatAnyone on the mask 1+2, without matte zones) and stage 4 (ViTMatte zones on that base)
    ma_run, ma_ok, ma_why = stage_run(store, name, "matanyone")
    st["matanyone"] = "done" if ma_ok else ("stale" if ma_run else "todo")
    if ma_why:
        why["matanyone"] = ma_why
    # stage VideoMaMa (optional, 2026-10-16): its own version, over the mask 1+2 or the stage-3 result
    vv_run, vv_ok, vv_why = stage_run(store, name, "videomama")
    st["videomama"] = "done" if vv_ok else ("stale" if vv_run else "todo")
    why["videomama"] = vv_why or ("" if vv_run else "opcjonalny – „▶ Policz VideoMaMa” (kosmyki włosów, detal krawędzi)")
    if not vv_ok and vmm_base(obj) == "matanyone" and st["matanyone"] != "done":
        why["videomama"] = "baza VideoMaMa = wynik MatAnyone, a etap 3 nie jest policzony – najpierw MatAnyone albo baza „maska 1+2”"
    vbase = vitmatte_base(obj)
    vm_run, vm_ok, vm_why = stage_run(store, name, "vitmatte")
    st["vitmatte"] = "done" if vm_ok else ("stale" if vm_run else "todo")
    base_ok = vbase == "core" or (st["matanyone"] == "done" if vbase == "matanyone" else st["videomama"] == "done")
    if vm_why:
        why["vitmatte"] = vm_why
    elif not base_ok:   # the stage is open (2026-10-17), but the chosen base has no current result
        why["vitmatte"] = (f"baza {BASE_LABEL[vbase]} nie jest policzona – najpierw "
                           f"{'etap 3 MatAnyone' if vbase == 'matanyone' else 'etap VideoMaMa'} albo baza ViTMatte = maska 1+2")
    elif not vm_run:
        why["vitmatte"] = ("dodaj strefę ViTMatte (🖌 malowany pas albo cały obrys) i „Zastosuj na ujęcie”"
                           if not zone_dicts(od, ("matte",)) else f"„✚ Zastosuj na ujęcie” – strefy ViTMatte na bazie: {BASE_LABEL[vbase]}")
    # stage 6 LTX Alpha-Gen (optional, generative): its own version on the mask 1+2
    lx_run, lx_ok, lx_why = stage_run(store, name, "ltx")
    st["ltx"] = "done" if lx_ok else ("stale" if lx_run else "todo")
    why["ltx"] = lx_why or ("" if lx_run else "opcjonalny – „▶ Policz LTX” (model generatywny, wymaga modeli LTX-2.5)")
    # export: the last exported version must still be a current result (active run, stage run or the mask 1+2)
    cur_ok = bool(exp_ver) and ((exp_ver == run and final_ok) or (exp_ver == ma_run and ma_ok) or
                                (exp_ver == vm_run and vm_ok) or (exp_ver == lx_run and lx_ok) or
                                (exp_ver == vv_run and vv_ok) or
                                (exp_ver == "core" and fix_ok))
    st["export"] = "done" if cur_ok else ("stale" if exp_ver else "todo")

    # a stage is locked while a stage it needs (NEEDS, transitively) is not done
    needs = needs_of(obj)

    def root(s: str) -> Optional[str]:      # the first not-done stage `s` waits for (itself not locked)
        for n in needs[s]:
            if st[n] != "done":
                return root(n) or n
        return None
    for s in ORDER:
        r = root(s)
        if r is not None:
            st[s] = "locked"      # also a "done" optional stage (CoTracker without corrections) before the core
            # the lock reason wins over the stage's own note (e.g. "etap opcjonalny" of CoTracker before the core)
            why[s] = ("najpierw: „▶▶ Przelicz całość” w etapie 1 SAM3" if r in ("crops", "core") and st[r] != "stale"
                      else f"najpierw: {LABEL[r]}" + (f" – {why[r]}" if why.get(r) and st[r] == "stale" else ""))
    nxt = next((s for s in NEXT_CHAIN if st[s] != "done"), None)
    stale = stale_intervals(store, name, need, mplate) if keys else []
    if keys and core_miss and core_ok and not dn_changed:
        stale = union(stale, core_miss)   # frames of the work area the core does not cover yet
    steps = []
    for n, sid, label, members in STEPS:
        open_ = next((m for m in members if st[m] != "done"), None)
        s_state = st[open_] if open_ else "done"
        s_why = why.get(open_ or members[-1], "")
        extra = cover.get("core" if sid == "sam3" else sid, {}) if not open_ else {}
        steps.append({"n": n, "id": sid, "label": label, "state": s_state, "why": s_why, "stage": open_ or members[-1],
                      **({"scope": extra.get("scope")} if extra.get("scope") else {})})
    return {"object": name, "stages": [{"id": s, "label": LABEL[s], "state": st[s], "why": why.get(s, ""),
                                        **cover.get(s, {})} for s in ORDER],
            "steps": steps,
            "work_range": list(work_range) if work_range else None, "life": life,
            "next": nxt, "next_label": LABEL.get(nxt) if nxt else None,
            # results that exist on disk (possibly outdated) – the UI may still show them (view tabs Rdzeń / Alpha)
            "has": {"core": core_t is not None, "final": bool(run and run_t), "core_fix": fix_exists},
            "stage_runs": {"matanyone": ma_run, "videomama": vv_run, "vitmatte": vm_run, "ltx": lx_run, "active": run,
                           # both ViTMatte results (one per base) for the comparison in stage 4
                           **{f"vitmatte@{b}": stage_run(store, name, "vitmatte", b)[0] for b in VITMATTE_BASES}},
            "vitmatte_base": vbase,
            "brush_zones": len(brush), "exported": exp_ver,
            "stale_intervals": stale}


def require(store: Any, name: str, work_dir: Path, stage: str, mplate: Optional[bool] = None,
            frame_range: Optional[Tuple[int, int]] = None, work_range: Optional[Tuple[int, int]] = None) -> None:
    """Raise ValueError with the reason when `stage` cannot run yet (a previous stage is not done – for the stages
    with a frame coverage: not done on the work area / the shot when the ranges are given)."""
    info = object_stages(store, name, work_dir, frame_range=frame_range, mplate=mplate, work_range=work_range)
    nd = needs_of(store.get(name))
    need, todo = set(), list(nd[stage])
    while todo:                                   # every stage `stage` needs, transitively
        n = todo.pop()
        if n not in need:
            need.add(n)
            todo.extend(nd[n])
    for s in info["stages"]:
        if s["id"] in need and s["state"] != "done":
            raise ValueError(f"Obiekt „{name}”: najpierw „{s['label']}”" + (f" – {s['why']}" if s["why"] else ""))


def blocked_for_export(store: Any, names: List[str], work_dir: Path, mplate: Optional[bool] = None) -> List[str]:
    """Objects without a valid final result (cannot be exported)."""
    bad = []
    for n in names:
        st = {s["id"]: s["state"] for s in object_stages(store, n, work_dir, mplate=mplate)["stages"]}
        if st["final"] != "done" and not _in_group_run(store, n, work_dir, mplate):
            bad.append(n)
    return bad


def _in_group_run(store: Any, name: str, work_dir: Path, mplate: Optional[bool]) -> bool:
    """A part whose parent's valid active version is a group version containing it (exported as its share)."""
    par = store.get(name).get("part_of")
    if not par or par == name:
        return False
    try:
        prun = (store.get(par).get("tool") or {}).get("active_run")
    except FileNotFoundError:
        return False
    prj = _run_json(store.dir(par), prun)
    if not prj or name not in (prj.get("group") or []):
        return False
    pst = {s["id"]: s["state"] for s in object_stages(store, par, work_dir, mplate=mplate)["stages"]}
    return pst["final"] == "done"
