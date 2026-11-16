# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/run_delete.py - Delete a version of an object (user request 2026-10-25: from the stage 5 "Version comparison").

The version's folder is deleted from disk for good (user decision 2026-11-13: no project trash any more; the UI warns
first with its size – engine/deletion.py). A version that something still uses is refused: the active version, the
current result of a stage (tool.stage_runs) or anything else in the object's JSON that names it (e.g. another version
made from it, Final Touch). The numbering never goes back: `runs/.last_run` keeps the highest number ever given, so a
new version never takes the name of a deleted one (browser caches and notes name versions by it)."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

RUN_RE = re.compile(r"v\d{3}")
LAST_RUN = ".last_run"


def _walk(x: Any, run: str, path: str, out: List[str]) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            _walk(v, run, f"{path}.{k}" if path else str(k), out)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _walk(v, run, f"{path}[{i}]", out)
    elif x == run:
        out.append(path)


def run_refs(obj_dir: Path, run: str) -> List[str]:
    """Where the object still names `run` (exact string values): object.json fields and the other versions' run.json /
    qa.json, every other JSON of the object outside caches. Empty = the version can go."""
    refs: List[str] = []
    obj_dir = Path(obj_dir)
    for p in sorted(obj_dir.rglob("*.json")):
        rel = p.relative_to(obj_dir)
        if rel.parts[0] in ("cache", "zones_cache") or (rel.parts[0] == "runs" and len(rel.parts) > 1 and rel.parts[1] == run):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found: List[str] = []
        _walk(data, run, "", found)
        refs += [f"{rel.as_posix()}: {f}" for f in found]
    return refs


def run_size(run_dir: Path) -> Dict[str, int]:
    files = [p for p in Path(run_dir).rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def last_run_number(runs_dir: Path) -> int:
    """Highest version number ever given in `runs_dir` (existing folders and the deleted ones)."""
    n = max([int(p.name[1:]) for p in Path(runs_dir).glob("v[0-9][0-9][0-9]")] + [0])
    try:
        n = max(n, int((Path(runs_dir) / LAST_RUN).read_text(encoding="utf-8").strip() or 0))
    except (OSError, ValueError):
        pass
    return n


def delete_run(obj_dir: Path, run: str) -> int:
    """Delete version `run` of the object from disk for good; the caller checked run_refs first. Returns the bytes
    freed."""
    from engine import deletion as DEL
    if not RUN_RE.fullmatch(run):
        raise ValueError(f"wersja: v###, jest {run!r}")
    runs = Path(obj_dir) / "runs"
    d = runs / run
    if not d.is_dir():
        raise FileNotFoundError(f"brak wersji {run}")
    n = last_run_number(runs)
    (runs / LAST_RUN).write_text(str(n), encoding="utf-8")   # before the delete: the number stays taken
    return DEL.remove(d, inside=runs)
