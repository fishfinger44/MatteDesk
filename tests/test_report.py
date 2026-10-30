# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„Zgłoś problem” (engine/report.py, PLAN_2026-10-22 I10): anonymisation (no path of config.local.json, no shot /
object / user name in the package), input checks and size caps, preview -> send of exactly the previewed package
only with consent, the proxy's answers, the Doctor row, the UI wiring."""
import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_mod
from engine import report

ROOT = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

LOCAL_CFG = {"comfy_dir": "F:\\Comfy\\ComfyUI", "models_dir": "F:/Comfy/ComfyUI/models", "last_work_dir": "D:/Shoots/Klient X/work/beach_042",
             "comfy_port": 8001}
STATE = {"manifest": {"shot": "beach_042", "plates": {"beauty": "D:/Shoots/Klient X/work/beach_042/source/plate_####.png"}},
         "manifest_path": "D:/Shoots/Klient X/work/beach_042/shot_manifest_beach_042.json",
         "work_dir": "D:/Shoots/Klient X/work/beach_042/matte_desk"}
ENV = {"USERNAME": "janek", "COMPUTERNAME": "STUDIO-PC", "USERPROFILE": "C:\\Users\\janek", "LOCALAPPDATA": "C:\\Users\\janek\\AppData\\Local"}


def _anon():
    return report.secrets_of(LOCAL_CFG, STATE, ["Woman_01", "suknia"], app_dir=Path("C:/Programs/MatteDesk"), env=ENV)


def test_anonymizer_removes_machine_paths_and_names():
    a = _anon()
    txt = a.text("Błąd w D:\\Shoots\\Klient X\\work\\beach_042\\matte_desk\\objects\\Woman_01\\core\\c_0001.png "
                 "(f:/comfy/comfyui/models/sam3) – janek@STUDIO-PC, ujęcie beach_042, obiekt suknia; "
                 "C:/Users/janek/AppData/Local/Temp/x.log; inny E:\\Projekty\\tajne\\a.exr; /home/janek/x; mail jan@firma.pl")
    for leak in ("Shoots", "Klient", "beach_042", "Woman_01", "suknia", "janek", "STUDIO-PC", "Projekty", "tajne", "firma.pl", "Comfy"):
        assert leak.lower() not in txt.lower(), (leak, txt)
    assert "<shot>" in txt and "<obj2>" in txt and "<path>" in txt and "<email>" in txt   # suknia = 2nd object


def test_name_replaced_only_as_whole_word():
    a = report.Anonymizer(names={"ab": "<obj>"})
    assert a.text("ab tab ab_1 ab") == "<obj> tab ab_1 <obj>"


def test_build_package_checks_and_caps():
    a = _anon()
    with pytest.raises(ValueError, match="co najmniej"):
        report.build_package({"what": "krótko"}, version="0.34", anon=a)
    with pytest.raises(ValueError, match="e-mail"):
        report.build_package({"what": "x" * 20, "email": "zly-adres"}, version="0.34", anon=a)
    log = [f"line {i} D:/Shoots/Klient X/work/beach_042/f.exr " + "y" * 900 for i in range(300)]
    pkg = report.build_package({"what": "Etap 3 MatAnyone2 dla Woman_01 się nie liczy", "steps": "kliknij", "email": "a@b.pl"},
                               version="0.34", anon=a, jobs=[{"kind": "x", "label": "MatAnyone2 Woman_01", "state": "failed",
                                                             "error": "D:\\Shoots\\Klient X\\boom"}], log=log,
                               doctor={"items": [{"name": "ffmpeg", "status": "error", "message": "C:\\Users\\janek\\bin"}]})
    s = json.dumps(pkg, ensure_ascii=False)
    assert len(s.encode("utf-8")) <= report.MAX_PACKAGE_BYTES
    assert "Woman_01" not in s and "Shoots" not in s and "janek" not in s
    assert pkg["user"]["email"] == "a@b.pl"            # the e-mail the user typed for an answer stays
    assert pkg["schema"] == report.SCHEMA and pkg["app"]["version"] == "0.34"


def test_screenshot_must_be_png_and_small():
    assert report._screenshot("data:image/png;base64," + base64.b64encode(PNG).decode()) == PNG
    with pytest.raises(ValueError, match="PNG"):
        report._screenshot(base64.b64encode(b"GIF89a....").decode())
    with pytest.raises(ValueError, match="za duży"):
        report._screenshot(base64.b64encode(PNG + b"\x00" * report.MAX_SCREENSHOT_BYTES).decode())


class _Resp:
    def __init__(self, code, data):
        self.status_code, self._d = code, data

    def json(self):
        return self._d


def test_send_answers():
    pkg = {"app": {"version": "0.34"}}
    sent = {}

    def ok(url, json=None, timeout=None, headers=None):
        sent.update(url=url, body=json)
        return _Resp(200, {"ok": True, "number": 7})
    assert report.send(pkg, PNG, "https://r.example", post=ok) == {"ok": True, "number": 7}
    assert sent["url"] == "https://r.example/report" and base64.b64decode(sent["body"]["screenshot_png_b64"]) == PNG
    with pytest.raises(RuntimeError, match="godzinę"):
        report.send(pkg, None, "https://r", post=lambda *a, **k: _Resp(429, {}))
    with pytest.raises(RuntimeError, match="Opis nie może"):
        report.send(pkg, None, "https://r", post=lambda *a, **k: _Resp(400, {"ok": False, "error": "Opis nie może zawierać linków"}))
    with pytest.raises(RuntimeError, match="połączyć"):
        report.send(pkg, None, "https://r", post=lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(RuntimeError, match="skonfigurowane"):
        report.send(pkg, None, "")


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "load_config", lambda: {**LOCAL_CFG, "report_endpoint": "https://r.example/"})
    monkeypatch.setattr(app_mod, "EnvironmentDoctor", lambda: type("D", (), {"run_all_checks": lambda self: type(
        "R", (), {"to_dict": lambda self: {"items": [{"name": "Git", "status": "error", "message": "F:\\Comfy\\git"}]}})()})())
    for k, v in STATE.items():
        monkeypatch.setitem(app_mod.STATE, k, v)
    monkeypatch.setitem(app_mod.STATE, "work_dir", str(tmp_path / "work"))
    from engine.objects import ObjectStore
    ObjectStore(tmp_path / "work").create("Woman_01", ["woman"])
    monkeypatch.setitem(app_mod.STATE, "manifest_path", str(tmp_path / "shot_manifest_beach_042.json"))
    monkeypatch.setattr("engine.sysinfo.gpu_info", lambda *a, **k: [{"name": "RTX 3080 Ti", "mem_total_mb": 12288}])
    monkeypatch.setattr(report, "comfy_summary", lambda c: {"alive": False})
    return TestClient(app_mod.app)


def test_preview_then_send_exactly_that_package_with_consent(client, monkeypatch, tmp_path):
    shot = "data:image/png;base64," + base64.b64encode(PNG).decode()
    r = client.post("/api/report/preview", json={"what": f"Woman_01 nie liczy się w {tmp_path}/work/x", "screenshot": shot,
                                                 "context": {"source": "error_dialog", "error_title": "Błąd", "error_text": "F:/Comfy/ComfyUI padł"}})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["configured"] and p["screenshot_kb"] == 0 and "zrzut okna PNG" in p["text"]
    for leak in ("Woman_01", str(tmp_path), "F:\\Comfy", "F:/Comfy", "Klient X"):
        assert leak not in p["text"], leak
    assert p["package"]["doctor"]["items"][0]["message"] == "<path>"
    got = {}

    def fake_send(pkg, shot_b, url):
        got.update(pkg=pkg, shot=shot_b, url=url)
        return {"ok": True, "number": 12}
    monkeypatch.setattr(report, "send", fake_send)
    assert client.post("/api/report/send", json={"id": p["id"], "consent": False}).status_code == 400
    r = client.post("/api/report/send", json={"id": p["id"], "consent": True})
    assert r.json() == {"ok": True, "number": 12}
    assert got["pkg"] == p["package"] and got["shot"] == PNG and got["url"] == "https://r.example"
    assert client.post("/api/report/send", json={"id": p["id"], "consent": True}).status_code == 404   # sent once


def test_failed_send_keeps_the_package_for_a_retry(client, monkeypatch):
    p = client.post("/api/report/preview", json={"what": "Program się zawiesza przy eksporcie"}).json()

    def boom(*a):
        raise RuntimeError("Nie udało się połączyć z serwerem zgłoszeń (ConnectError).")
    monkeypatch.setattr(report, "send", boom)
    r = client.post("/api/report/send", json={"id": p["id"], "consent": True})
    assert r.status_code == 502 and "połączyć" in r.json()["detail"]
    monkeypatch.setattr(report, "send", lambda *a: {"ok": True, "number": 3})
    assert client.post("/api/report/send", json={"id": p["id"], "consent": True}).json()["number"] == 3


def test_preview_rejects_short_description(client):
    r = client.post("/api/report/preview", json={"what": "zle"})
    assert r.status_code == 400 and "co najmniej" in r.json()["detail"]


def test_doctor_row(monkeypatch):
    from doctor import EnvironmentDoctor
    d = EnvironmentDoctor()
    d.config = {"report_endpoint": ""}
    it = d.check_report_endpoint()
    assert it.status == "info" and "Wymagane teraz: NIE" in it.message
    d.config = {"report_endpoint": "https://r.example"}
    monkeypatch.setattr(report, "health", lambda url: (True, "osiągalny"))
    assert d.check_report_endpoint().status == "ok"
    monkeypatch.setattr(report, "health", lambda url: (False, "nieosiągalny (ConnectError)"))
    assert d.check_report_endpoint().status == "warning"


def test_log_tail_from_buffer_and_file(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_RING", None)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "server.log").write_text("\n".join(f"l{i}" for i in range(1000)), encoding="utf-8")
    t = report.log_tail(tmp_path)
    assert t[-1] == "l999" and len(t) <= report.MAX_LOG_LINES


def test_ui_wiring():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="reportBtn"' in html and 'id="aboutReportBtn"' in html and '<script src="/static/report.js"></script>' in html
    js = (ROOT / "static" / "report.js").read_text(encoding="utf-8")
    assert "/api/report/preview" in js and "/api/report/send" in js and "rpConsent" in js
    assert "data-report" in (ROOT / "static" / "app.js").read_text(encoding="utf-8")   # error windows offer a report
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert "report_endpoint" in cfg            # the proxy URL lives in config, never a token
    assert "token" not in json.dumps(cfg).lower()


def test_project_summary_numbered_objects_prompts_as_typed_plates():
    """User 2026-10-22: objects as <objN> with their role, SAM 3.1 prompts as typed (even when equal to the object
    name), standard plate keys kept, custom plate names numbered; texts use the same <objN>."""
    st = {**STATE, "manifest": {**STATE["manifest"], "plates": {"beauty": "D:/a/p_####.exr", "backlit": "D:/a/b_####.exr",
                                                                  "NikeLogoPass": "D:/a/n_####.exr"}}}
    a = report.secrets_of(LOCAL_CFG, st, ["Woman_01", "suknia"], app_dir=Path("C:/Programs/MatteDesk"), env=ENV)
    objs = [{"name": "Woman_01", "detect": {"type": "text", "prompts": ["Woman_01", "woman"]}, "depth_order": 0,
             "edge_mode": "auto", "tool": {"status": "core ready"}},
            {"name": "suknia", "detect": {"type": "points"}, "group": "Woman_01", "depth_order": 1}]
    pr = report.project_summary(objs, {"Woman_01": ["woman", "hair E:/tajne/x.png"], "suknia": ["dress"]},
                                st["manifest"]["plates"], a)
    o1, o2 = pr["objects"]
    assert o1["id"] == "<obj1>" and o1["prompts"] == ["Woman_01", "woman", "hair <path>"] and o1["status"] == "core ready"
    assert o2["id"] == "<obj2>" and o2["group"] == "<obj1>" and o2["prompts"] == ["dress"]
    assert pr["plates"] == ["beauty", "backlit", "<plate1>"]
    assert a.text("błąd w NikeLogoPass dla suknia") == "błąd w <plate1> dla <obj2>"
    pkg = report.build_package({"what": "Woman_01 gubi włosy na F21"}, version="0.34", anon=a, project=pr)
    assert pkg["user"]["what"].startswith("<obj1>") and pkg["project"]["objects"][0]["prompts"][0] == "Woman_01"
