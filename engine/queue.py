# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/queue.py - Overnight queue (SPEC 6.2): objects from many projects computed one after another, with ETA,
resume after a crash (finished jobs are skipped – each job's result is a run version) and a morning report.
The queue never changes keys or settings.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

QUEUE_SCHEMA = "matte_desk/queue@1.0"


class JobQueue:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"schema": QUEUE_SCHEMA, "jobs": []}

    def save(self, q: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, manifest_path: str, obj: str, action: str = "finalize") -> Dict[str, Any]:
        q = self.load()
        job = {"id": f"j{len(q['jobs']) + 1:03d}", "manifest_path": manifest_path, "object": obj, "action": action,
               "status": "pending", "added": datetime.now().isoformat(timespec="seconds"), "seconds": None,
               "result": None}
        q["jobs"].append(job)
        self.save(q)
        return job

    def eta_s(self, default_s: float = 180.0) -> float:
        q = self.load()
        done = [j["seconds"] for j in q["jobs"] if j["status"] == "done" and j["seconds"]]
        per = sum(done) / len(done) if done else default_s
        return per * sum(1 for j in q["jobs"] if j["status"] in ("pending", "running"))

    def run(self, worker: Callable[[Dict[str, Any]], Dict[str, Any]], stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """Run pending jobs (a job left 'running' by a crash is retried). Returns the report."""
        q = self.load()
        for j in q["jobs"]:
            if j["status"] == "running":
                j["status"] = "pending"
        self.save(q)
        for j in q["jobs"]:
            if stop and stop():
                break
            if j["status"] != "pending":
                continue
            j["status"] = "running"
            self.save(q)
            t0 = time.time()
            try:
                j["result"] = worker(j)
                j["status"] = "done"
            except Exception as e:  # report and continue with the next job
                j["status"], j["result"] = "error", {"error": str(e)}
            j["seconds"] = round(time.time() - t0, 1)
            self.save(q)
        return self.report()

    def report(self) -> Dict[str, Any]:
        q = self.load()
        rows = []
        for j in q["jobs"]:
            r = j.get("result") or {}
            rows.append({"id": j["id"], "object": j["object"], "manifest": j["manifest_path"], "status": j["status"],
                         "qa": r.get("qa_status"), "first_flag": r.get("first_flag"), "error": r.get("error")})
        return {"jobs": rows, "pending": sum(1 for j in q["jobs"] if j["status"] == "pending"), "eta_s": self.eta_s()}
