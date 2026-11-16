# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/jsonio.py - The one atomic JSON writer and the one rule for reading a tool state file.

write_json: tmp + os.replace, so a power cut leaves the old file whole (audit 2026-11-02 #4). The bytes are those of
`json.dumps(data, **kw)`: signatures hash some of these files, so each caller keeps its own indent / ensure_ascii.
read_state: missing = None, unreadable = StateFileError with the file's name, the file stays for repair (409 in the API).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

_RETRIES = 8


class StateFileError(RuntimeError):
    """A tool state file (`project.json`, `outputs.json`, a zone) exists but cannot be read. The file is left as it is."""

    def __init__(self, path: Path, cause: BaseException):
        self.path = Path(path)
        super().__init__(f"Uszkodzony plik stanu {self.path.name}: {cause}. Napraw albo usuń go: {self.path}")


def write_json(path: Union[str, Path], data: Any, **dumps_kw: Any) -> None:
    p = Path(path)
    text = json.dumps(data, **dumps_kw)   # before the tmp: data that cannot be serialised leaves nothing behind
    # a tmp of its own per thread: two threads writing the same object (a request + the mask 1+2 background batch,
    # both migrating on the first read) overwrote one shared .tmp (test_zone_outline flaky, 2026-10-26)
    tmp = p.with_name(f"{p.name}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        for i in range(_RETRIES):   # Windows: the replace fails (WinError 5) while another process reads the file –
            try:                    # a second server, a viewer, an antivirus (seen 2026-10-24: a slider save lost with a 500)
                os.replace(tmp, p)
                return
            except PermissionError:
                if i == _RETRIES - 1:
                    raise
                time.sleep(0.05 * (i + 1))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json_retry(path: Union[str, Path]) -> Any:
    """JSON of a file; Windows: opening fails (errno 13) for a moment while another thread replaces the file – retried
    like the replace in write_json (2026-10-26)."""
    p = Path(path)
    for i in range(_RETRIES):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except PermissionError:
            if i == _RETRIES - 1:
                raise
            time.sleep(0.05 * (i + 1))
    raise AssertionError("unreachable")


def read_state(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """A state file: its JSON object, None when missing, StateFileError when it exists but is not a readable object
    (never `{}` + overwrite on the next save, audit 2026-11-02 #3)."""
    p = Path(path)
    try:
        data = read_json_retry(p)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise StateFileError(p, e) from e
    if not isinstance(data, dict):
        raise StateFileError(p, TypeError("not a JSON object"))
    return data
