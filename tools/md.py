# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tools/md.py - MatteDesk from the command line (for Claude Code and scripts), talking to the running server.

    python tools/md.py status
    python tools/md.py open D:/projects/film/work/SH010
    python tools/md.py run figure --wait          # also: rerun <obj> [--frame F], finalize <obj> [--edge-mode M]
    python tools/md.py finalize figure --ma-scale 0.75 --wait   # MatAnyone scale auto|1.0|0.75|0.5 (object setting)
    python tools/md.py core-fix figure --wait     # stage 2: mask 1+2 (SAM3 core + brush corrections) on the shot
    python tools/md.py finalize figure --stage matanyone --wait   # stage 3: MatAnyone on the mask 1+2, no ViTMatte zones
    python tools/md.py edge-fix figure [--skip 0_3,1_2] --wait
    python tools/md.py qa figure [--v v002] [--json]
    python tools/md.py versions figure | keys figure | jobs | cancel
    python tools/md.py export figure --yes        # or: export --all --yes [--format auto|png|exr]   (mattes are written – needs --yes)
                                                  # [--source core|matanyone|vitmatte|version|active] [--run v003] (stage layout 2026-10-15)
    python tools/md.py denoise [auto|on|off]      # plate-to-mask denoise of the project (no value = show)
    python tools/md.py zone-matte osoba --frame 115 --draw zone.json [--preset hair] [--matting auto] [--prompt "…"] [--wait]
                                                  # `matte` zone (hair…) from a drawing {polygon, strokes, bg_strokes} at frame F:
                                                  # preview of that frame; --wait = also "Przelicz finalnie" and wait
    python tools/md.py zones osoba | queue-all    # zones of an object | "Policz wszystko" (queue, front to back)
    python tools/md.py zone-keys osoba z_01 [--set 140 --draw key.json] [--delete 140] [--region 120]   # zone keys / propagated region
    python tools/md.py zone-apply osoba z_01 --wait   # "Zastosuj": only the zones recomputed (cache) -> new version
                                                  # [--stage vitmatte] = stage 4: on the stage-3 MatAnyone version
    python tools/md.py cleanplate-auto [--wait] [--frames 1-120]   # auto clean plate from the shot (after the cores; LaMa / OpenCV for holes)
    python tools/md.py api-doc > docs/AGENT_API_routes.md

Only `urllib` (no new dependencies). Server = config.json / config.local.json -> backend_host / backend_port
(default 127.0.0.1:8765), or `--url`. Exit codes: 0 ok, 1 error, 2 server busy / refused (HTTP 409) – a running job
of the user is never interrupted (only `cancel` does that, explicitly). Every command prints a short Polish summary
with the raw numbers; `--json` prints the raw response.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
HEAVY = {"run", "rerun", "finalize", "edge-fix", "export", "core-fix"}
ROUTES_BEGIN, ROUTES_END = "<!-- routes:begin (python tools/md.py api-doc --write) -->", "<!-- routes:end -->"


class Http:
    """Minimal JSON client: request(method, path, body) -> (status, data). Tests inject a TestClient adapter."""

    def __init__(self, base: str, timeout: float = 7200.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Tuple[int, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw.decode("utf-8")) if raw else None)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except Exception:
                return e.code, {"detail": str(e)}
        except (urllib.error.URLError, OSError) as e:
            return 0, {"detail": f"serwer MatteDesk nie odpowiada ({self.base}): {e}"}


def default_url() -> str:
    try:
        sys.path.insert(0, str(ROOT))
        from engine.config import load_config
        c = load_config()
        return f"http://{c.get('backend_host', '127.0.0.1')}:{int(c.get('backend_port', 8765))}"
    except Exception:
        return "http://127.0.0.1:8765"


class CLI:
    def __init__(self, http: Any, out: Callable[[str], None] = print):
        self.http = http
        self.out = out

    # ------------------------------------------------------------------ helpers
    def call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
        return self.http.request(method, path, body)

    def fail(self, st: int, data: Any) -> int:
        detail = data.get("detail") if isinstance(data, dict) else data
        if isinstance(detail, dict) and detail.get("missing_front"):   # plan section 6: objects in front without masks
            self.out(f"ODMOWA (409): {detail.get('detail')}")
            self.out("  brak maski: " + ", ".join(detail["missing_front"]))
            self.out("  kolejka: " + ", ".join(f"{q['object']}:{q['action']}" for q in detail.get("queue", []))
                     + "  (md.py queue-all, potem ponów)")
            return 2
        if st == 409:
            self.out(f"ODMOWA (409): {detail}")
            return 2
        self.out(f"BŁĄD ({st or 'brak połączenia'}): {detail}")
        return 1

    def running(self) -> List[Dict[str, Any]]:
        st, d = self.call("GET", "/api/jobs")
        return [j for j in (d or {}).get("jobs", []) if j.get("state") == "running"] if st == 200 else []

    def busy_guard(self) -> Optional[int]:
        run = self.running()
        if run:
            self.out(f"Serwer liczy: {run[-1]['label']} ({run[-1].get('step') or 'start'}) – nie przerywam; "
                     "poczekaj albo użytkownik kliknie „Przerwij”")
            return 2
        return None

    def heavy(self, method: str, path: str, body: Optional[Dict[str, Any]], wait: bool) -> Tuple[int, Any]:
        """A long_job request. --wait: the request runs in a thread, progress from /api/jobs printed meanwhile."""
        res: Dict[str, Any] = {}

        def go():
            res["r"] = self.call(method, path, body)
        th = threading.Thread(target=go, daemon=True)
        th.start()
        last = ""
        while th.is_alive():
            th.join(1.0 if wait else 0.5)
            if not wait:
                break
            run = self.running()
            if run:
                j = run[-1]
                line = f"  … {j['label']}: {j.get('step') or ''} {j.get('done', 0)}/{j.get('total', 0)} {j.get('comfy') or ''}".rstrip()
                if line != last:
                    self.out(line)
                    last = line
        if not wait and th.is_alive():
            return 202, {"detail": "uruchomione – postęp: md.py jobs"}
        return res["r"]

    # ------------------------------------------------------------------ commands
    def status(self, a) -> int:
        st, cur = self.call("GET", "/api/project/current")
        if st != 200:
            return self.fail(st, cur)
        if not cur.get("loaded"):
            self.out("Projekt: brak (md.py open <katalog>)")
            return 0
        m = cur.get("manifest", {}) or {}
        self.out(f"Projekt: {m.get('shot', '?')}  {cur.get('work_dir', '')}  kl. {m.get('first')}–{m.get('last')}  "
                 f"{m.get('plate_w')}×{m.get('plate_h')}")
        s3, dn = self.call("GET", "/api/project/mask_plate_denoise")
        if s3 == 200:
            self.out("Odszumianie plate'u do maskowania: " + self._denoise_line(dn))
        st2, cp = self.call("GET", "/api/project/cleanplate_auto")
        if st2 == 200:
            auto = cp.get("auto") or {}
            self.out("Clean plate: " + cp.get("source", "?") + (f" (pewność {auto.get('confidence_mean')}, {auto.get('inpaint_backend')})" if auto else ""))
        st, objs = self.call("GET", "/api/objects")
        rows = []
        for o in (objs or {}).get("objects", []):
            s2, stg = self.call("GET", f"/api/object/{o['name']}/stages")
            line = " ".join(f"{x['label'] if 'label' in x else x['id']}:{x['state']}" for x in (stg or {}).get("stages", [])) if s2 == 200 else "?"
            rows.append({"name": o["name"], "depth_order": o.get("depth_order"), "active_run": o.get("tool", {}).get("active_run"),
                         "status": o.get("tool", {}).get("status"), "stages": stg if s2 == 200 else None})
            self.out(f"  {o.get('depth_order')}. {o['name']}: {o.get('tool', {}).get('status') or '-'} · wersja "
                     f"{o.get('tool', {}).get('active_run') or '-'} · {line}")
        run = self.running()
        self.out(f"Zadanie: {run[-1]['label']} · {run[-1].get('step') or ''}" if run else "Zadanie: brak")
        if a.json:
            self.out(json.dumps({"project": cur, "mask_plate_denoise": dn if s3 == 200 else None, "objects": rows,
                                 "running": run}, ensure_ascii=False, indent=1))
        return 0

    def open(self, a) -> int:
        st, d = self.call("POST", "/api/project/open_or_create", {"work_dir": a.dir, "plates": {}})
        if st != 200:
            return self.fail(st, d)
        self.out(f"Otwarty: {a.dir}")
        return 0

    def _obj_job(self, a, path: str, body: Optional[Dict[str, Any]], what: str) -> int:
        g = self.busy_guard()
        if g:
            return g
        t0 = time.time()
        st, d = self.heavy("POST", path, body, a.wait)
        if st == 202:
            self.out(f"{what}: {d['detail']}")
            return 0
        if st != 200:
            return self.fail(st, d)
        self.out(f"{what}: gotowe w {time.time() - t0:.1f} s" + (f" · wersja {d.get('run')}" if d.get("run") else "") +
                 (f" · przeliczone {d.get('recomputed')} · z cache {d.get('cached')}" if "recomputed" in d else ""))
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
        return 0

    def run(self, a) -> int:
        return self._obj_job(a, f"/api/object/{a.obj}/run", None, f"Licz ujęcie {a.obj}")

    def rerun(self, a) -> int:
        return self._obj_job(a, f"/api/object/{a.obj}/rerun_from/{a.frame}", None, f"Przelicz zmienione {a.obj}")

    def finalize(self, a) -> int:
        if getattr(a, "ma_scale", None):   # object setting first (invalidates the final result like the UI select)
            g = self.busy_guard()
            if g:
                return g
            st, d = self.call("PUT", f"/api/object/{a.obj}/settings", {"ma_scale_mode": a.ma_scale})
            if st != 200:
                return self.fail(st, d)
            self.out(f"Skala MatAnyone {a.obj}: {a.ma_scale}")
        body = {"edge_mode": a.edge_mode} if a.edge_mode else {}
        if getattr(a, "stage", None):
            body["stage"] = a.stage
        return self._obj_job(a, f"/api/object/{a.obj}/finalize", body,
                             f"MatAnyone (etap 3) {a.obj}" if body.get("stage") else f"Przelicz finalnie {a.obj}")

    def core_fix(self, a) -> int:
        return self._obj_job(a, f"/api/object/{a.obj}/core_fix", None, f"Zastosuj poprawki (maska 1+2) {a.obj}")

    def zone_matte(self, a) -> int:
        """`matte` zone from a drawing file (plate px at --frame): preview of that frame, optionally finalize."""
        try:
            draw = json.loads(Path(a.json_file).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            self.out(f"BŁĄD: plik rysunku: {e}")
            return 1
        body = {"mode": "add", "criterion": "matte", "frame": int(a.frame), "polygon": draw.get("polygon", []),
                "strokes": draw.get("strokes", []), "bg_strokes": draw.get("bg_strokes", []),
                "bg_static": bool(draw.get("bg_static", True)),
                "preset": a.preset, "matting": a.matting, "prompt": a.prompt or "", "holdout": not a.no_holdout,
                "params": draw.get("params", {}), "engine_hints": draw.get("engine_hints", {})}
        zid = a.zone or "new"
        g = self.busy_guard()
        if g:
            return g
        st, d = self.call("POST", f"/api/object/{a.obj}/zones/{zid}?preview_frame={int(a.frame)}", body)
        if st != 200:
            return self.fail(st, d)
        z = d.get("zone", {})
        pv = d.get("preview") or {}
        stt = pv.get("stats") or {}
        self.out(f"Strefa {z.get('zone_id')} ({z.get('preset')}, silnik {z.get('matting')}) zapisana; klucz F{a.frame}")
        if pv:
            share = stt.get("engine_share") or {k: 1.0 for k in (stt.get("times") or {}) if k in ("chroma", "knn", "cf", "hybrid")}
            self.out(f"  podgląd F{a.frame}: dodane {pv.get('added_px', 0)} px, miękkie {pv.get('soft_px', 0)} px, "
                     f"unknown {stt.get('unknown_px')} px, kadr skala {stt.get('scale')}, {stt.get('seconds')} s; "
                     "silniki " + ", ".join(f"{k} {v:.0%}" for k, v in (share or {}).items()))
            if stt.get("bg_hex"):
                self.out(f"  tło (chroma): {stt['bg_hex']} lo {stt.get('lo')} hi {stt.get('hi')}; szew {stt.get('seam_ratio')}")
            if pv.get("holdout_objects"):
                self.out("  holdout z: " + ", ".join(pv["holdout_objects"]))
            if pv.get("cleanplate"):
                self.out(f"  clean plate: {pv['cleanplate']}")
            for fl in pv.get("flags", []):
                self.out(f"  ⚠ {fl}")
            self.out(f"  nakładki: GET /api/object/{a.obj}/frame/{a.frame}?view=zone_alpha | zone_trimap | zone_engines")
        if not a.wait:
            self.out("Przeliczenie: md.py finalize " + a.obj + " --wait  (albo „Przelicz finalnie” w UI)")
            return 0
        return self._obj_job(a, f"/api/object/{a.obj}/finalize", {}, f"Przelicz finalnie {a.obj}")

    def zones(self, a) -> int:
        st, d = self.call("GET", f"/api/object/{a.obj}/zones")
        if st != 200:
            return self.fail(st, d)
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
            return 0
        zs = d.get("zones", [])
        stat = d.get("status") or {}
        if not zs:
            self.out("Brak stref")
        for z in zs:
            extra = f" preset {z.get('preset')} silnik {z.get('matting')} kreski − {len(z.get('bg_strokes') or [])}" if z.get("criterion") == "matte" else f" param {z.get('param')}"
            st = stat.get(z["zone_id"]) or {}
            rs = st.get("run_stats") or {}
            line = (f"{z['zone_id']}: {z['mode']} / {z['criterion']} @F{z['frame']}{extra}"
                    f"{'' if z.get('enabled', True) else ' (wyłączona)'}")
            if st:   # card status (plan step 5): state, keys, stats of the active version
                line += f" | {st.get('state')} · klucze {st.get('n_keys')} ({', '.join('F%d' % k for k in st.get('keys') or [])})"
                if st.get("run"):
                    line += f" · wersja {st['run']}"
                if rs.get("frames") and rs.get("seconds") is not None:
                    line += f" · {rs.get('engine')} {rs['frames']} kl. {rs['seconds'] / max(1, rs['frames']):.2f} s/kl."
            self.out(line)
        return 0

    def plate_qa(self, a) -> int:
        g = self.busy_guard()
        if g:
            return g
        body: Dict[str, Any] = {"plate": a.plate}
        if a.frames:
            lo, hi = a.frames.split("-")
            body["frames"] = [int(lo), int(hi)]
        st, d = self.heavy("POST", "/api/plate/qa", body, a.wait)
        if st == 202:
            self.out(f"Kontrola klatek {a.plate}: {d['detail']}")
            return 0
        if st != 200:
            return self.fail(st, d)
        self.out(f"Kontrola klatek {a.plate}: {d.get('n')} kl. w {d.get('seconds')} s · oflagowane {len(d.get('flagged', []))}: "
                 + ", ".join(f"{f} ({d['suggested'].get(str(f))})" for f in d.get("flagged", [])[:40]))
        return 0

    def plate_fix(self, a) -> int:
        g = self.busy_guard()
        if g:
            return g
        decisions = {}
        for item in (a.decisions or "").split(","):
            if not item.strip():
                continue
            f, m = item.split(":")
            decisions[f.strip()] = m.strip()
        st, d = self.heavy("POST", "/api/plate/fix", {"plate": a.plate, "decisions": decisions, "confirmed": bool(a.yes)}, a.wait)
        if st == 202:
            self.out(f"Naprawa {a.plate}: {d['detail']}")
            return 0
        if st != 200:
            return self.fail(st, d)
        if not d.get("success"):
            p = d.get("plan", {})
            self.out(f"Plan naprawy {a.plate}: {len(p.get('repair', {}))} kl. do zapisu (~{p.get('bytes_written_estimate', 0) / 2 ** 20:.0f} MB), "
                     f"{p.get('hardlinks')} dowiązań → {p.get('dir')}. Potwierdź: --yes")
            return 0
        self.out(f"Naprawa {a.plate}: {len(d.get('repaired', {}))} kl. zapisane, {d.get('hardlinks')} dowiązań, {d.get('copies')} kopii, "
                 f"{d.get('seconds')} s → {d.get('pattern')} (plate w projekcie przełączony)")
        return 0

    def cleanplate_auto(self, a) -> int:
        body: Dict[str, Any] = {}
        if a.frames:
            lo, hi = a.frames.split("-")
            body["frames"] = [int(lo), int(hi)]
        g = self.busy_guard()
        if g:
            return g
        t0 = time.time()
        st, d = self.heavy("POST", "/api/project/cleanplate_auto", body, a.wait)
        if st == 202:
            self.out(f"Clean plate (auto): {d['detail']}")
            return 0
        if st != 200:
            return self.fail(st, d)
        self.out(f"Clean plate (auto): gotowy w {time.time() - t0:.1f} s · próbka {len(d.get('frames_sampled', []))} kl. · "
                 f"pewność średnia {d.get('confidence_mean')} · dziury {d.get('never_uncovered_px')} px ({d.get('inpaint_backend')}) · "
                 f"obiekty {', '.join(d.get('objects', []))}")
        self.out(f"  plate „cleanplate_auto”: {d.get('pattern')}")
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
        return 0

    def zone_interpret(self, a) -> int:
        """Plan step 9: what the zone description means (rules PL/EN) + Hough "-" strokes when --frame is given."""
        body = {"prompt": a.prompt, "frame": a.frame, "zone_id": a.zone}
        st, d = self.call("POST", f"/api/object/{a.obj}/zones/interpret", body)
        if st != 200:
            return self.fail(st, d)
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
            return 0
        pl = d.get("plan", {})
        self.out(f"Prompt „{pl.get('text')}”: preset {pl.get('preset') or '– (bez zmian)'}; fg „{pl.get('fg')}” → SAM3 „{pl.get('fg_en')}”")
        if pl.get("distractors"):
            self.out("  rozpraszacze (tło): " + "; ".join(pl["distractors"]) + ("  → linia: szukam kresek „−”" if pl.get("bg_line") else ""))
        if pl.get("engine_hints"):
            self.out("  silniki ×: " + ", ".join(f"{k} {v:g}" for k, v in pl["engine_hints"].items()) + (f" (tylko {pl['forced_engine']})" if pl.get("forced_engine") else ""))
        for w in pl.get("warnings", []):
            self.out(f"  ⚠ {w}")
        for bs in d.get("bg_strokes", []):
            self.out(f"  kreska „−” (do potwierdzenia): {bs['p'][0]} → {bs['p'][1]}, {bs.get('length_px')} px, kąt {bs.get('angle')}°")
        if a.frame is not None and not d.get("bg_strokes"):
            self.out("  brak prostych linii w tle strefy (Hough)")
        return 0

    def zone_propose(self, a) -> int:
        """Plan step 9 „Zaproponuj strefę”: proposals (polygons) around the core in --frame; --accept N creates the zone."""
        g = self.busy_guard()
        if g:
            return g
        st, d = self.call("POST", f"/api/object/{a.obj}/zones/propose", {"frame": int(a.frame), "prompt": a.prompt, "min_area": int(a.min_area)})
        if st != 200:
            return self.fail(st, d)
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
            return 0
        props = d.get("proposals", [])
        self.out(f"F{a.frame}: {len(props)} propozycji (SAM3 „{d.get('fg')}”{'' if d.get('sam3') else ' – bez maski SAM3' + (': ' + d['sam3_error'] if d.get('sam3_error') else '')}; preset {d.get('preset')})")
        for pr in props:
            self.out(f"  #{pr['id']}: {pr['area_px']} px, {pr['side']}, bbox {pr['bbox']}, źródło {pr['source']}, wielokąt {len(pr['polygon'])} pkt")
        if a.accept is not None:
            pr = next((x for x in props if x["id"] == int(a.accept)), None)
            if pr is None:
                self.out(f"BŁĄD: brak propozycji #{a.accept}")
                return 1
            body = {"mode": "add", "criterion": "matte", "frame": int(a.frame), "polygon": pr["polygon"], "preset": d.get("preset") or "hair",
                    "matting": a.matting, "prompt": a.prompt or "", "holdout": True}
            st, z = self.call("POST", f"/api/object/{a.obj}/zones/new?preview_frame={int(a.frame)}", body)
            if st != 200:
                return self.fail(st, z)
            self.out(f"Strefa {z.get('zone_id') or (z.get('zone') or {}).get('zone_id')} z propozycji #{a.accept} zapisana; klucz F{a.frame}")
        return 0

    def zone_keys(self, a) -> int:
        if a.set is not None:
            try:
                draw = json.loads(Path(a.draw).read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as e:
                self.out(f"BŁĄD: plik rysunku (--draw): {e}")
                return 1
            body = {"polygon": draw.get("polygon", []), "strokes": draw.get("strokes", []), "bg_strokes": draw.get("bg_strokes", [])}
            note = a.prompt if a.prompt is not None else draw.get("prompt")
            if note is not None:
                body["prompt"] = note
            st, d = self.call("PUT", f"/api/object/{a.obj}/zones/{a.zone}/keys/{int(a.set)}", body)
            if st != 200:
                return self.fail(st, d)
            k = d.get("key") or {}
            self.out(f"Klucz strefy {a.zone} F{a.set} zapisany{(' (uwaga: ' + k['prompt'] + ')') if k.get('prompt') else ''}; klucze: {d.get('keys')}")
        if a.delete is not None:
            st, d = self.call("DELETE", f"/api/object/{a.obj}/zones/{a.zone}/keys/{int(a.delete)}")
            if st != 200:
                return self.fail(st, d)
            self.out(f"Klucz strefy {a.zone} F{a.delete} usunięty; klucze: {d.get('keys')}")
        if a.region is not None:
            g = self.busy_guard()
            if g:
                return g
            st, d = self.call("GET", f"/api/object/{a.obj}/zones/{a.zone}/region/{int(a.region)}")
            if st != 200:
                return self.fail(st, d)
            self.out(f"Obszar strefy {a.zone} F{a.region}: {d.get('area_px')} px, bbox {d.get('bbox')}, "
                     f"{len(d.get('contours', []))} kontur(ów), klucze {d.get('keys')}{' (klucz)' if d.get('is_key') else ''}")
            if a.json:
                self.out(json.dumps(d, ensure_ascii=False))
            return 0
        st, d = self.call("GET", f"/api/object/{a.obj}/zones/{a.zone}/keys")
        if st != 200:
            return self.fail(st, d)
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
            return 0
        self.out(f"Strefa {a.zone}: klucz bazowy F{d.get('base')}, klucze: " + ", ".join(
            f"F{k} ({v['polygon_pts']} pkt, +{v['strokes']}, −{v['bg_strokes']})" for k, v in (d.get("keys") or {}).items()))
        return 0

    def zone_apply(self, a) -> int:
        g = self.busy_guard()
        if g:
            return g
        t0 = time.time()
        st, d = self.heavy("POST", f"/api/object/{a.obj}/zones/{a.zone}/apply" + (f"?stage={a.stage}" if a.stage else ""), {}, a.wait)
        if st == 202:
            self.out(f"Zastosuj {a.zone}: {d['detail']}")
            return 0
        if st != 200:
            return self.fail(st, d)
        self.out(f"Zastosuj {a.zone}: gotowe w {time.time() - t0:.1f} s · wersja {d.get('run')} · przeliczone {d.get('computed')} kl. · "
                 f"z cache {d.get('cached')} kl.")
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
        return 0

    def queue_all(self, a) -> int:
        st, d = self.call("POST", "/api/queue/preset/all", {})
        if st != 200:
            return self.fail(st, d)
        self.out(f"Kolejka „Policz wszystko”: dodane {len(d.get('added', []))} ({', '.join(d.get('objects', [])) or 'nic'}); "
                 "start: POST /api/queue/run (▶ Kolejka w UI)")
        return 0

    def edge_fix(self, a) -> int:
        skip = [x for x in (a.skip or "").split(",") if x]
        return self._obj_job(a, f"/api/object/{a.obj}/edge_fix", {"run": a.v, "skip": skip}, f"Poprawka krawędzi {a.obj}")

    def qa(self, a) -> int:
        st, d = self.call("GET", f"/api/object/{a.obj}/qa" + (f"?v={a.v}" if a.v else ""))
        if st != 200:
            return self.fail(st, d)
        if not d:
            self.out(f"{a.obj}: brak wyniku finalnego (QA)")
            return 0
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
            return 0
        cols = d.get("colours", {})
        red = sorted(int(f) for f, c in cols.items() if c == "red")
        yel = sorted(int(f) for f, c in cols.items() if c == "yellow")
        e = d.get("edge", {})
        self.out(f"{a.obj} {d.get('run', '')}: migotanie (mediana) {d.get('flicker_median')} "
                 f"({'OK' if d.get('gate_ok') else 'brama NIE'} ≤ 0.004) · klatek {len(d.get('frames', []))} · "
                 f"czerwone {len(red)} {red[:12]} · żółte {len(yel)}")
        if e:
            self.out(f"  krawędź: mediana {e.get('before')} px · odcinki do poprawy {e.get('pieces_to_fix')} · "
                     f"poprawiona {e.get('corrected')}" + (f" (po: {e.get('after')} px)" if e.get("corrected") else "") +
                     (f" · przepływ min {e.get('flow_iou_min')}" if "flow_iou_min" in e else "") +
                     (f" · clear plate {e.get('cleanplate_agreement_median')}" if "cleanplate_agreement_median" in e else ""))
            for p in (e.get("pieces") or [])[:10]:
                self.out(f"    {p['id']}: F{p['first']}–{p['last']} {p['median_px']:+} px ({p['kind']})")
        if d.get("banner"):
            self.out(f"  ! {d['banner']}")
        return 0

    def versions(self, a) -> int:
        st, d = self.call("GET", f"/api/object/{a.obj}/runs")
        if st != 200:
            return self.fail(st, d)
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
            return 0
        for r in d.get("runs", []):
            self.out(f"{'*' if r.get('active') else ' '} {r['run']}: {r.get('edge_mode')} · {r.get('seconds')} s · "
                     f"{r.get('created_at')} · QA {'OK' if r.get('gate_ok') else ('-' if r.get('gate_ok') is None else 'flagi')} · "
                     f"migotanie {r.get('flicker_median')} · flag {r.get('n_flags')}")
        return 0

    def keys(self, a) -> int:
        st, d = self.call("GET", f"/api/object/{a.obj}/keys")
        if st != 200:
            return self.fail(st, d)
        ks = d.get("keys", [])
        self.out(f"{a.obj}: {len(ks)} kluczy: " + ", ".join(f"F{k['frame']}({k.get('kind', '?')}/{k.get('author', '?')})" for k in ks))
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
        return 0

    def jobs(self, a) -> int:
        st, d = self.call("GET", "/api/jobs")
        if st != 200:
            return self.fail(st, d)
        for j in d.get("jobs", []):
            self.out(f"#{j['id']} {j['state']}: {j['label']} · {j.get('step') or ''} {j.get('done', 0)}/{j.get('total', 0)} · "
                     f"{j.get('elapsed_s')} s{' · ' + j['comfy'] if j.get('comfy') else ''}{' · ' + j['error'] if j.get('error') else ''}")
        if not d.get("jobs"):
            self.out("brak zadań")
        return 0

    def cancel(self, a) -> int:
        st, d = self.call("POST", "/api/jobs/cancel")
        if st != 200:
            return self.fail(st, d)
        self.out("Przerwano bieżące zadanie" if d.get("cancelled") else "Nic nie było liczone")
        return 0

    def export(self, a) -> int:
        if not a.yes:
            self.out("Eksport zapisuje maski (EXR / PNG) – wymaga potwierdzenia użytkownika: dodaj --yes")
            return 1
        fmt = {"format": a.format} if a.format else {}   # none = the project's setting (project.json export_format)
        if a.source:
            fmt["source"] = a.source
        if a.run:
            fmt["run"] = a.run
        if a.all:
            return self._obj_job(a, "/api/export", {"objects": [], **fmt}, "Eksport (wszystko)")
        return self._obj_job(a, "/api/export", {"objects": [a.obj], **fmt}, f"Eksport {a.obj}")

    @staticmethod
    def _denoise_line(d: Dict[str, Any]) -> str:
        eff = "wł." if d.get("effective") else "wył."
        return f"{d.get('setting')} → {eff} ({d.get('reason', '')})"

    def denoise(self, a) -> int:
        """Project switch of the plate-to-mask denoise (project.json mask_plate_denoise): show, or set auto / on / off."""
        if not a.value:
            st, d = self.call("GET", "/api/project/mask_plate_denoise")
        else:
            st, d = self.call("PUT", "/api/project/mask_plate_denoise", {"setting": a.value})
        if st != 200:
            return self.fail(st, d)
        self.out("Odszumianie plate'u do maskowania: " + self._denoise_line(d))
        if d.get("stale_objects"):
            self.out(f"  do przeliczenia (rdzeń liczony z innym ustawieniem): {', '.join(d['stale_objects'])}")
        if a.json:
            self.out(json.dumps(d, ensure_ascii=False, indent=1))
        return 0

    def api_doc(self, a) -> int:
        table = api_routes_markdown()
        if not a.write:
            self.out(table)
            return 0
        doc = ROOT / "docs" / "AGENT_API.md"
        txt = doc.read_text(encoding="utf-8")
        b, e = txt.index(ROUTES_BEGIN) + len(ROUTES_BEGIN), txt.index(ROUTES_END)
        doc.write_text(txt[:b] + "\n" + table + "\n" + txt[e:], encoding="utf-8")
        self.out(f"zapisano {doc}")
        return 0


def api_routes_markdown() -> str:
    """Table of every HTTP route of app.py: method, path, kind (long_job / quiet / object_edit / when_idle), summary."""
    sys.path.insert(0, str(ROOT))
    import app as app_module
    from fastapi.routing import APIRoute
    rows = ["| Metoda | Ścieżka | Rodzaj | Opis |", "|---|---|---|---|"]
    for r in app_module.app.routes:
        if not isinstance(r, APIRoute) or not r.include_in_schema:
            continue
        e = r.endpoint
        kind = []
        lj = getattr(e, "md_long_job", None)
        if lj:
            kind.append("long_job" + (" (quiet)" if lj.get("quiet") else ""))
        if getattr(e, "md_object_edit", False):
            kind.append("object_edit")
        if getattr(e, "md_when_idle", None):
            kind.append("when_idle")
        doc = " ".join((e.__doc__ or "").split())
        doc = doc.split(". ")[0][:160].replace("|", "/")
        for m in sorted(r.methods - {"HEAD"}):
            rows.append(f"| {m} | `{r.path}` | {', '.join(kind) or '–'} | {doc} |")
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="md.py", description="MatteDesk CLI (serwer musi działać)")
    ap.add_argument("--url", default=None)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="surowa odpowiedź JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda name, **kw: _add(name, parents=[common], **kw)
    sub.add_parser("status")
    p = sub.add_parser("open"); p.add_argument("dir")
    for name in ("run", "rerun", "finalize", "edge-fix", "core-fix"):
        p = sub.add_parser(name); p.add_argument("obj"); p.add_argument("--wait", action="store_true")
        if name == "rerun":
            p.add_argument("--frame", type=int, default=0)
        if name == "finalize":
            p.add_argument("--edge-mode", default=None)
            p.add_argument("--ma-scale", choices=("auto", "1.0", "0.75", "0.5"), default=None,
                           help="skala MatAnyone (ustawienie obiektu, zapisywane przed liczeniem); auto = próba wg VRAM")
            p.add_argument("--stage", choices=("matanyone",), default=None,
                           help="matanyone = etap 3: MatAnyone2 na masce 1+2 bez stref ViTMatte (baza etapu 4)")
        if name == "edge-fix":
            p.add_argument("--v", default=None); p.add_argument("--skip", default="")
    p = sub.add_parser("qa"); p.add_argument("obj"); p.add_argument("--v", default=None)
    p = sub.add_parser("versions"); p.add_argument("obj")
    p = sub.add_parser("keys"); p.add_argument("obj")
    sub.add_parser("jobs")
    sub.add_parser("cancel")
    p = sub.add_parser("export"); p.add_argument("obj", nargs="?"); p.add_argument("--all", action="store_true")
    p.add_argument("--yes", action="store_true"); p.add_argument("--wait", action="store_true", default=True)
    p.add_argument("--format", choices=("auto", "png", "exr"), default=None,
                   help="auto = PNG dla źródeł 8-bit, EXR dla EXR; bez flagi = ustawienie projektu")
    p.add_argument("--source", choices=("active", "core", "matanyone", "vitmatte", "version"), default=None,
                   help="źródło maski: core = rdzeń (1+2), matanyone / vitmatte = wynik etapu 3 / 4, version = --run, active = jak dotąd")
    p.add_argument("--run", default=None, help="wersja przy --source version (jeden obiekt)")
    p = sub.add_parser("zone-matte"); p.add_argument("obj"); p.add_argument("--frame", type=int, required=True)
    p.add_argument("--draw", dest="json_file", required=True, help="rysunek {polygon, strokes, bg_strokes[, params, engine_hints]} w px plate'u")
    p.add_argument("--preset", default="hair", choices=("hair", "fibres", "smoke", "motion", "thin"))
    p.add_argument("--matting", default=None, choices=("matanyone", "vitmatte", "auto", "chroma", "knn", "cf", "hybrid", "clean"),
                   help="bez flagi = domyślny silnik presetu (hair: vitmatte + przepływ r3 ×2)")
    p.add_argument("--prompt", default=""); p.add_argument("--zone", default=None, help="id istniejącej strefy (aktualizacja)")
    p.add_argument("--no-holdout", action="store_true", help="„Licz mimo to bez holdoutu” (flaga QA)")
    p.add_argument("--wait", action="store_true", help="po zapisie także „Przelicz finalnie” i czekaj")
    p = sub.add_parser("zones"); p.add_argument("obj")
    sub.add_parser("queue-all")
    p = sub.add_parser("plate-qa"); p.add_argument("plate", nargs="?", default="beauty"); p.add_argument("--frames", default=None)
    p.add_argument("--wait", action="store_true")
    p = sub.add_parser("plate-fix"); p.add_argument("plate", nargs="?", default="beauty")
    p.add_argument("--decisions", default="", help="klatka:metoda,... (mean | flow | gain | keep)")
    p.add_argument("--yes", action="store_true", help="potwierdzenie zapisu (EXR / GB)"); p.add_argument("--wait", action="store_true")
    p = sub.add_parser("zone-interpret"); p.add_argument("obj"); p.add_argument("prompt")
    p.add_argument("--frame", type=int, default=None, help="z klatką: kreski „−” z linii w tle (Hough) do potwierdzenia")
    p.add_argument("--zone", default=None, help="obszar = klucz tej strefy w klatce (inaczej bbox rdzenia + 60 px)")
    p = sub.add_parser("zone-propose"); p.add_argument("obj"); p.add_argument("--frame", type=int, required=True)
    p.add_argument("--prompt", default=None, help="opis strefy (fg → SAM3_Detect na kadrze); bez = prompt obiektu")
    p.add_argument("--min-area", dest="min_area", type=int, default=400)
    p.add_argument("--accept", type=int, default=None, help="utwórz strefę z propozycji o tym numerze")
    p.add_argument("--matting", default="matanyone", choices=("matanyone", "vitmatte", "auto", "chroma", "knn", "cf", "hybrid", "clean"))
    p = sub.add_parser("zone-keys"); p.add_argument("obj"); p.add_argument("zone")
    p.add_argument("--set", type=int, default=None, help="klatka klucza do zapisu (z --draw)")
    p.add_argument("--draw", default=None, help="rysunek {polygon, strokes, bg_strokes[, prompt]} w px plate'u")
    p.add_argument("--prompt", default=None, help="uwaga iteracji do tego klucza (pętla korekty, krok 7); '' czyści")
    p.add_argument("--delete", type=int, default=None, help="klatka klucza do usunięcia")
    p.add_argument("--region", type=int, default=None, help="pokaż obszar po propagacji w klatce")
    p = sub.add_parser("zone-apply"); p.add_argument("obj"); p.add_argument("zone"); p.add_argument("--wait", action="store_true")
    p.add_argument("--stage", choices=("vitmatte",), default=None, help="vitmatte = etap 4: strefy na wersji MatAnyone z etapu 3")
    p = sub.add_parser("cleanplate-auto"); p.add_argument("--wait", action="store_true"); p.add_argument("--frames", default=None, help="a-b")
    p = sub.add_parser("denoise"); p.add_argument("value", nargs="?", choices=("auto", "on", "off"), default=None,
                                                  help="auto = wł. dla EXR / > 8 bit, wył. dla 8-bit; bez wartości = pokaż")
    p = sub.add_parser("api-doc"); p.add_argument("--write", action="store_true", help="podmień tabelę w docs/AGENT_API.md")
    return ap


def main(argv: Optional[List[str]] = None, http: Any = None, out: Callable[[str], None] = print) -> int:
    for stream in (sys.stdout, sys.stderr):   # Windows console (cp1250) cannot print the arrows / Polish summaries
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    a = build_parser().parse_args(argv)
    if a.cmd == "export" and not a.all and not a.obj:
        out("podaj obiekt albo --all")
        return 1
    cli = CLI(http or Http(a.url or default_url()), out)
    return getattr(cli, a.cmd.replace("-", "_"))(a)


if __name__ == "__main__":
    sys.exit(main())
