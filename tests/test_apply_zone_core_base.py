# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""User report 2026-11-11 („Hair”, ViTMatte on the mask 1+2): „Zastosuj” failed with „rdzeń zmienił się … Przelicz
finalnie (etap 3)” – the mask 1+2 base version (v001 of 2026-10-29) was kept by its signature while the stabilised core
changed. For the mask 1+2 base the version is written again and the apply retried once; other bases still ask for
„Przelicz finalnie”."""
import pytest

from engine import pipeline as P


class _Store:
    def __init__(self, base):
        self.base = base

    def get(self, name):
        return {"tool": {}, "vitmatte_base": self.base}


def _pipe(base, fails):
    pipe = P.ObjectPipeline.__new__(P.ObjectPipeline)
    pipe.store = _Store(base)
    calls = {"apply": 0, "base": []}

    def _apply(name, frames, zone_id, cb, stage):
        calls["apply"] += 1
        if calls["apply"] <= fails:
            raise P.CoreChanged("rdzeń zmienił się")
        return "ok"
    pipe._apply_zone = _apply
    pipe.core_base_run = lambda name, frames, cb=None, force=False: calls["base"].append(force) or "v009"
    return pipe, calls


def test_mask12_base_is_rewritten_and_retried(monkeypatch):
    import engine.stages as S
    monkeypatch.setattr(S, "vitmatte_base", lambda obj: obj["vitmatte_base"])
    pipe, calls = _pipe("core", fails=1)
    assert pipe.apply_zone("Hair", [1, 2], None, None, stage="vitmatte") == "ok"
    assert calls == {"apply": 2, "base": [True]}
    pipe, calls = _pipe("core", fails=2)                         # still different after a fresh base: the error
    with pytest.raises(P.CoreChanged):
        pipe.apply_zone("Hair", [1, 2], None, None, stage="vitmatte")
    pipe, calls = _pipe("matanyone", fails=1)                    # a MatAnyone2 base: „Przelicz finalnie” as before
    with pytest.raises(P.CoreChanged):
        pipe.apply_zone("Hair", [1, 2], None, None, stage="vitmatte")
    assert calls["base"] == [] and issubclass(P.CoreChanged, ValueError)   # the API's 409 for ValueError stays
