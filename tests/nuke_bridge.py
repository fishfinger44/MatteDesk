# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The live Nuke session through the NukeMCP bridge (kleer001/nuke-mcp addon in NukeX 15, port 54321) for `real`
tests: `execute(code)` runs Python in Nuke and returns the addon's JSON reply. One client holds the slot at a time
(reference_nuke_bridge_single_client): `available()` is False when the port refuses / nothing answers – the tests skip,
nothing is killed. Bridge quirk: nested defs do not see top-level names – helpers take them as default args."""

from __future__ import annotations

import json
import socket
from typing import Any, Dict

HOST, PORT = "127.0.0.1", 54321


def call(kind: str, timeout: float = 120.0, **params) -> Dict[str, Any]:
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    try:
        f = s.makefile("rwb")
        f.readline()   # handshake line
        f.write((json.dumps({"type": kind, "params": params}) + "\n").encode())
        f.flush()
        line = f.readline()
    finally:
        s.close()
    if not line:
        raise ConnectionError("Nuke bridge: empty reply (slot held by another client?)")
    return json.loads(line)


def execute(code: str, timeout: float = 300.0) -> Dict[str, Any]:
    return call("execute_python", timeout=timeout, code=code)


def available() -> bool:
    try:
        r = execute("result = {'ok': True, 'nuke': nuke.NUKE_VERSION_STRING}", timeout=15)
    except (OSError, ValueError, ConnectionError):
        return False
    return bool(_result(r).get("ok"))


def _result(reply: Dict[str, Any]) -> Dict[str, Any]:
    """The `result` dict the executed code set (the addon returns it under `result`, as JSON or as text)."""
    if not isinstance(reply, dict):
        return {}
    r = reply.get("result", reply)
    if isinstance(r, str):
        try:
            r = json.loads(r)
        except ValueError:
            return {}
    return r if isinstance(r, dict) else {}


def result(reply: Dict[str, Any]) -> Dict[str, Any]:
    return _result(reply)
