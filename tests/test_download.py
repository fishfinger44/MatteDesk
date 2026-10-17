# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/download.py: resume (.part + Range), SHA-256 check, server hashes (GitHub digest, HF X-Linked-Etag)."""
import hashlib
import io
import urllib.error
from email.message import Message

import pytest

from engine import download as dl

DATA = bytes(range(256)) * 4096          # 1 MB
SHA = hashlib.sha256(DATA).hexdigest()


class Resp(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status
        self.headers = Message()
        self.headers["Content-Length"] = str(len(body))


def server(ignore_range=False, log=None):
    def urlopen(req, timeout=0):
        rng = req.get_header("Range")
        if log is not None:
            log.append(rng)
        if rng and not ignore_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(DATA):
                raise urllib.error.HTTPError(req.full_url, 416, "range", Message(), None)
            return Resp(DATA[start:], 206)
        return Resp(DATA)
    return urlopen


def test_fresh_download_with_checksum(tmp_path):
    dst = tmp_path / "m" / "w.bin"
    seen = []
    dl.download("http://x/w.bin", dst, sha256=SHA, urlopen=server(), progress=lambda d, t: seen.append((d, t)))
    assert dst.read_bytes() == DATA and not (tmp_path / "m" / "w.bin.part").exists()
    assert seen[-1] == (len(DATA), len(DATA))


def test_resume_from_part(tmp_path):
    dst = tmp_path / "w.bin"
    (tmp_path / "w.bin.part").write_bytes(DATA[:300000])
    log = []
    dl.download("http://x/w.bin", dst, sha256=SHA, urlopen=server(log=log))
    assert log == ["bytes=300000-"] and dst.read_bytes() == DATA


def test_server_ignoring_range_starts_over(tmp_path):
    dst = tmp_path / "w.bin"
    (tmp_path / "w.bin.part").write_bytes(b"garbage" * 10)
    dl.download("http://x/w.bin", dst, sha256=SHA, urlopen=server(ignore_range=True))
    assert dst.read_bytes() == DATA


def test_complete_part_416_finishes(tmp_path):
    dst = tmp_path / "w.bin"
    (tmp_path / "w.bin.part").write_bytes(DATA)
    dl.download("http://x/w.bin", dst, sha256=SHA, urlopen=server())
    assert dst.read_bytes() == DATA


def test_checksum_mismatch_deletes_part(tmp_path):
    dst = tmp_path / "w.bin"
    with pytest.raises(RuntimeError, match="SHA-256"):
        dl.download("http://x/w.bin", dst, sha256="0" * 64, urlopen=server())
    assert not dst.exists() and not (tmp_path / "w.bin.part").exists()


def test_github_asset_digest():
    rel = {"tag_name": "v9.9.9", "assets": [
        {"name": "other.7z", "browser_download_url": "u0", "size": 1, "digest": "sha256:" + "a" * 64},
        {"name": "pkg.7z", "browser_download_url": "u1", "size": 2_000_000_000, "digest": "sha256:" + "B" * 64}]}
    a = dl.github_asset("o/r", "pkg.7z", api_get=lambda u: rel)
    assert a == {"url": "u1", "size": 2_000_000_000, "sha256": "b" * 64, "tag": "v9.9.9"}
    with pytest.raises(RuntimeError):
        dl.github_asset("o/r", "missing.7z", api_get=lambda u: rel)


def test_hf_sha256_from_redirect_headers():
    class Opener:
        def open(self, req, timeout=0):
            assert req.get_method() == "HEAD"
            h = Message()
            h["X-Linked-Etag"] = '"' + SHA + '"'
            raise urllib.error.HTTPError(req.full_url, 302, "Found", h, None)
    assert dl.hf_sha256("https://huggingface.co/a/b/resolve/main/w.bin", opener=Opener()) == SHA
    assert dl.hf_sha256("https://github.com/x/y.bin", opener=Opener()) is None
