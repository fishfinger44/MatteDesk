# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Downloaded results are deleted from the local Comfy output folder (user decision 2026-11-23: output/matte_desk had
grown to 3.7 GB / 7300 files): only under output/matte_desk/<job folder with a per-run id>, only locally, only when
the file has the downloaded size; empty job folders go too, everything else stays."""

import io

import comfy_client
from comfy_client import ComfyClient


def _client(tmp_path, argv=None, host="127.0.0.1"):
    c = ComfyClient(host=host, port=1, comfy_dir=str(tmp_path / "comfy"))
    c.get_system_stats = lambda: {"system": {"argv": argv or ["main.py"]}}
    return c


def _put(root, sub, name, data=b"abcd"):
    d = root.joinpath(*sub.split("/"))
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    return d / name


def test_deletes_our_run_folder_file_and_empty_dirs(tmp_path):
    out = tmp_path / "comfy" / "output"
    f = _put(out, "matte_desk/track_1a2b3c4d/t", "t_00001_.png")
    keep = _put(out, "matte_desk/track_1a2b3c4d/t", "t_00002_.png")
    c = _client(tmp_path)
    info = {"filename": "t_00001_.png", "subfolder": "matte_desk/track_1a2b3c4d/t", "type": "output"}
    assert c.discard_output(info, 4) and not f.exists() and keep.exists()
    assert c.discard_output({**info, "filename": "t_00002_.png"}, 4)
    assert not (out / "matte_desk" / "track_1a2b3c4d").exists() and (out / "matte_desk").is_dir()


def test_keeps_everything_else(tmp_path):
    out = tmp_path / "comfy" / "output"
    c = _client(tmp_path)
    cases = [("matte_desk/void", "v.png", 4),                    # fixed prefix: Comfy may replay it from its cache
             ("other_tool/run_1a2b3c4d", "x.png", 4),             # not ours
             ("matte_desk/h3_2c268056", "wrong_size.png", 3),     # size differs from the download
             ("matte_desk/h3_2c268056/../../x_1a2b3c4d", "e.png", 4)]
    files = [_put(out, s, n) if ".." not in s else None for s, n, _ in cases]
    for (s, n, size), p in zip(cases, files):
        assert not c.discard_output({"filename": n, "subfolder": s, "type": "output"}, size)
        assert p is None or p.exists()
    assert not c.discard_output({"filename": "v.png", "subfolder": "matte_desk/h3_2c268056", "type": "temp"}, 4)


def test_remote_or_unknown_output_dir_does_nothing(tmp_path):
    f = _put(tmp_path / "comfy" / "output", "matte_desk/h3_2c268056", "a.png")
    info = {"filename": "a.png", "subfolder": "matte_desk/h3_2c268056", "type": "output"}
    assert not _client(tmp_path, host="10.0.0.5").discard_output(info, 4)
    c = ComfyClient(host="127.0.0.1", port=1)           # no comfy_dir, relative main.py
    c.get_system_stats = lambda: {"system": {"argv": ["main.py"]}}
    assert not c.discard_output(info, 4) and f.exists()


def test_output_directory_argument_wins(tmp_path):
    other = tmp_path / "shared_out"
    f = _put(other, "matte_desk/vace_0a1b2c3d", "a.png")
    c = _client(tmp_path, argv=["C:/x/main.py", "--output-directory", str(other)])
    assert c.discard_output({"filename": "a.png", "subfolder": "matte_desk/vace_0a1b2c3d"}, 4) and not f.exists()


def test_download_output_discards_unless_kept(tmp_path, monkeypatch):
    out = tmp_path / "comfy" / "output"
    monkeypatch.setattr(comfy_client.urllib.request, "urlopen", lambda url, timeout=60: io.BytesIO(b"abcd"))
    info = {"filename": "a.png", "subfolder": "matte_desk/h3_2c268056", "type": "output"}
    f = _put(out, info["subfolder"], "a.png")
    c = _client(tmp_path)
    c.keep_outputs = True
    assert c.download_output(info) == b"abcd" and f.exists()
    c.keep_outputs = False
    assert c.download_output(info) == b"abcd" and not f.exists()
    assert c.download_output(info) == b"abcd"              # nothing left to delete: still no error
