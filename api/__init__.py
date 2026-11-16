# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Routers split out of app.py (audit 2026-11-02 step 8). Each module: `make(*, deps) -> (APIRouter, exports)`;
app.py passes its shared pieces explicitly and includes the router where the section used to be.
Moving a section: functions verbatim into `make()`, request models at module level (FastAPI resolves the endpoints'
annotations through module globals), app functions passed as lambdas (late binding – tests monkeypatch app names).
Contract: tests/test_app_routers.py (route table + order unchanged, no import of app.py, keyword-only deps)."""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # as in app.py: must precede the first `import cv2` (EXR plates)
