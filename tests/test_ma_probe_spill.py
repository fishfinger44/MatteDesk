# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MatAnyone2 auto scale probe (2026-10-21, beatch crash 2026-10-18): the start scale follows the crop size, GPU memory
spilled to system RAM (Windows counter) pushes the scale down, and Comfy down / OOM on a crop names the scale to set."""

import types

import numpy as np
import pytest

from engine import refine
from engine.crop import CropSegment
from engine.sysinfo import SpillSampler, gpu_shared_gb, pid_on_port


def _seg(side):
    return CropSegment("s00", 1, 20, 0, 0, [0, 0, side, side], [0, 0, 0, 0], 1008 / side)


class StatsClient:
    def __init__(self, used_gb=6.0):
        self.used = used_gb

    def get_system_stats(self):
        return {"devices": [{"vram_total": 12 * 2 ** 30, "vram_free": int((12 - self.used) * 2 ** 30)}]}


def _fin(side_calls, shared=None, used_gb=6.0):
    """Finalizer without __init__: the probe only needs the client, the shared reader and _matanyone_segment."""
    fin = object.__new__(refine.Finalizer)
    fin.client = StatsClient(used_gb)
    fin.shared_gb = shared

    def ma(seg, frames, core, keys, start, scale, tag, soft=None):
        side_calls.append(scale)
        fin._cur_scale = scale
        return {}
    fin._matanyone_segment = ma
    return fin


def _probe(fin, side):
    frames = list(range(1, 21))
    core = {f: np.zeros((4, 4), np.float32) for f in frames}
    rep = refine.FinalizeReport(run="v001", edge_mode="matanyone", segments=["s00"])
    return fin.probe_scale(_seg(side), frames, core, {}, 1, rep), rep


@pytest.mark.parametrize("side,start", [(1008, 1.0), (2048, 1.0), (2049, 0.75), (2600, 0.75), (2601, 0.5), (2784, 0.5)])
def test_probe_start_scale_by_crop(side, start):
    assert refine.probe_start_scale(side) == start


def test_big_crop_never_probes_full_scale():
    calls = []
    scale, rep = _probe(_fin(calls), 2784)                 # beatch Woman_01: 2784 px crop
    assert calls == [0.5] and scale == 0.5
    assert rep.vram_log[0] == "kadr 2784 px: próba od ma_scale 0.5"
    calls = []
    scale, rep = _probe(_fin(calls), 2200)
    assert calls == [0.75] and scale == 0.75
    calls = []
    scale, rep = _probe(_fin(calls), 1400)                 # small crop: as before, 1.0 first, no extra log line
    assert calls == [1.0] and scale == 1.0 and len(rep.vram_log) == 1


def test_spill_to_ram_lowers_scale():
    calls = []
    # shared GB read by the sampler: idle 0.05 -> 1.0 run spills 2 GB -> 0.75 run spills 0.3 GB (under the limit)
    level = {"v": 0.05}

    def shared():
        return level["v"]
    fin = _fin(calls, shared=shared)
    orig = fin._matanyone_segment

    def ma(seg, frames, core, keys, start, scale, tag, soft=None):
        level["v"] = {1.0: 2.05, 0.75: 0.35}[scale]          # memory still held right after the prompt
        r = orig(seg, frames, core, keys, start, scale, tag, soft)
        return r
    fin._matanyone_segment = ma
    scale, rep = _probe(fin, 1400)
    assert calls == [1.0, 0.75] and scale == 0.75
    assert "przelane do RAM 2.00 GB" in rep.vram_log[0] and "przelane do RAM 0.30 GB" in rep.vram_log[1]


def test_no_counter_keeps_vram_only_probe():
    calls = []
    scale, rep = _probe(_fin(calls, shared=lambda: None, used_gb=11.5), 1400)   # counter missing, VRAM over the limit
    assert calls == [1.0, 0.75, 0.5] and scale == 0.5 and all("przelane" not in s for s in rep.vram_log)


def test_comfy_down_error_names_crop_and_scale():
    fin = object.__new__(refine.Finalizer)

    def boom(*a, **k):
        raise RuntimeError("MatAnyone2 nie powiódł się: ComfyUI nie odpowiada (zamknięty albo padł) – uruchom go i policz ponownie")
    fin._matanyone_segment_run = boom
    with pytest.raises(RuntimeError) as e:
        fin._matanyone_segment(_seg(2784), [1, 2], {}, {}, 1, 1.0, "final")
    msg = str(e.value)
    assert "nie odpowiada" in msg and "kadr 2784 px przy MatAnyone2 scale 1.0" in msg and "ustaw MatAnyone2 scale 0.75 / 0.5" in msg
    with pytest.raises(RuntimeError) as e:
        fin._matanyone_segment(_seg(2784), [1, 2], {}, {}, 1, 0.5, "final")
    assert "już przy MatAnyone2 scale 0.5" in str(e.value)

    def other(*a, **k):
        raise RuntimeError("MatAnyone2 zwrócił 3 klatek zamiast 4")
    fin._matanyone_segment_run = other
    with pytest.raises(RuntimeError) as e:
        fin._matanyone_segment(_seg(2784), [1, 2], {}, {}, 1, 1.0, "final")
    assert str(e.value) == "MatAnyone2 zwrócił 3 klatek zamiast 4"       # unrelated errors untouched


def test_oom_and_lost_get_the_hint_too():
    assert "0.75 / 0.5" in refine.comfy_failure_hint("Brak pamięci GPU w węźle 3.", 2000, 1.0)
    assert "0.5" in refine.comfy_failure_hint("ComfyUI zrestartowany – zadanie utracone, uruchom ponownie", 2000, 0.75)
    assert refine.comfy_failure_hint("Błąd w węźle 3: shape", 2000, 1.0) == ""


def test_spill_sampler_peak_minus_idle():
    vals = iter([0.1, 0.4, 1.7, 0.9])
    s = SpillSampler(lambda: next(vals, 0.9), interval_s=0.01)
    with s:
        import time
        time.sleep(0.1)
    assert s.base == 0.1 and s.peak == pytest.approx(1.7) and s.spill_gb == pytest.approx(1.6)
    assert SpillSampler(lambda: None).__enter__() is not None
    dead = SpillSampler(lambda: (_ for _ in ()).throw(OSError("no counter")), interval_s=0.01)
    with dead:
        pass
    assert dead.spill_gb is None                             # a failing counter never raises


def test_counter_parsing_with_fake_powershell():
    out = lambda txt: (lambda *a, **k: types.SimpleNamespace(stdout=txt, returncode=0))
    assert gpu_shared_gb(123, run=out(str(3 * 2 ** 30))) == pytest.approx(3.0)
    assert gpu_shared_gb(123, run=out("")) is None
    assert gpu_shared_gb(123, run=out("1610612736,5")) == pytest.approx(1.5, abs=1e-6)   # Polish decimal comma
    assert pid_on_port(8001, run=out("4242\r\n")) == 4242 and pid_on_port(8001, run=out("")) is None


def test_reader_off_for_mock_client_and_env(monkeypatch):
    assert refine._comfy_shared_reader(object()) is None                       # no port (mock client)
    monkeypatch.setenv("MD_SHARED_PROBE", "0")
    assert refine._comfy_shared_reader(types.SimpleNamespace(port=8001)) is None
