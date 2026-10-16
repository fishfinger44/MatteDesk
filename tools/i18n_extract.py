# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collect the user-visible Polish strings of the UI and the server for the English catalog (static/i18n/en.json).

    python tools/i18n_extract.py            -> list strings missing from the catalog (exit 1 when there are any)
    python tools/i18n_extract.py --json F   -> write the missing ones to F as {"source": ""} for translation

Polish is the source language of the code; the browser translates to English at run time (static/i18n.js).
Sources: static/*.js (string and template literals, text between HTML tags, title/placeholder/... attributes inside
templates), static/index.html (text and attributes), Python (app.py, doctor.py, engine/*.py: string constants and
f-strings, docstrings skipped). Placeholders (`${expr}`, `{expr}` in f-strings) become {0}, {1} ...
Over-collecting is fine: entries that are not UI text get an empty translation and are ignored at run time.
"""
from __future__ import annotations

import ast
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "static" / "i18n" / "en.json"
PL_CHARS = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ„”]")
# common Polish UI words without diacritics (strings without them would otherwise be missed)
PL_WORDS = re.compile(r"\b(Zapisz|Anuluj|Zamknij|Otwórz|Projekt|projekt|Obiekt|obiekt|obiekty|Klatka|klatka|klatki|klatek|Etap|etap|"
                      r"Brak|brak|Wybierz|wybierz|Dodaj|dodaj|Usuń|Policz|Przelicz|Pokaż|pokaż|Ukryj|Wersja|wersja|"
                      r"Kliknij|kliknij|Sprawdzanie|Trwa|trwa|nie|jest|oraz|albo|tylko|przy|przez|dla|bez|potem|teraz|"
                      r"maska|maski|maskę|strefa|strefy|klucz|klucze|kluczy|plate'u|plate'y|tło|tła|Tak|Nie|"
                      r"np|w|z|na|lub|gdy|czy|jak|tej|tego|ten|razem|wszystkie|wszystko|Uruchom|uruchom|Status|ogólny|"
                      r"Wariant|Licencja|Rozmiar|Wymagany|Szczegóły|Ustawienia|Pomoc|Wynik|wynik|Podgląd|Tryb|tryb)\b"
                      # typical Polish letter groups / endings (rare in English)
                      r"|[a-z]*(sz|cz|rz|ść)[a-z]*|\b\w+(anie|enie|ania|enia|acja|acji|cja|cji|ości|ować|ują|uje|ych|ymi|owy|owa|owe|ego|iej|ami|ach|ów)\b")
ATTRS = ("title", "placeholder", "aria-label", "alt", "data-fold-name", "data-ask", "data-tip")
ATTR_RE = re.compile(r'\b(?:%s)\s*=\s*"([^"]*)"' % "|".join(re.escape(a) for a in ATTRS))
TAGS = "a|b|i|u|s|em|strong|small|code|kbd|span|div|p|br|hr|label|input|select|option|button|textarea|ul|ol|li|table|thead|tbody|tr|td|th|img|figure|figcaption|details|summary|h[1-6]|canvas|svg|path|rect|circle|sup|sub|pre|section|header|footer|nav|aside|main|fieldset|legend|progress|mark|abbr"
TAG_RE = re.compile(r"</?(?:%s)(?=[\s/>])(?:\"[^\"]*\"|'[^']*'|[^'\">])*>" % TAGS, re.I)   # real HTML tags only: "<plate>_fix" is text
PH = "\x00"   # placeholder marker during extraction
CODE_TPL = re.compile(r"addUserKnob|knobs|Group \{|\bdef |^\s*set |name \w+ xpos")   # Nuke script text, not UI


WORD = re.compile(r"[A-Za-zĄ-ż']{3,}")


def _vocab() -> set[str]:
    """Polish word forms already known from the catalog (sources) minus English ones (translations): catches Polish
    text without diacritics ("Widok", "Kontury") the regexes miss."""
    try:
        cat = json.loads(CATALOG.read_text(encoding="utf-8")).get("strings", {})
    except (OSError, ValueError):
        return set()
    pl = {w.lower() for k in cat for w in WORD.findall(k)}
    en = {w.lower() for v in cat.values() for w in WORD.findall(v or "")}
    return pl - en


VOCAB: set[str] = set()


def is_text(s: str) -> bool:
    s = s.strip()
    if not s or re.search(r"[A-Za-zĄ-ż]{2}", s) is None:
        return False
    return bool(PL_CHARS.search(s) or PL_WORDS.search(s) or any(w.lower() in VOCAB for w in WORD.findall(s)))


def norm(s: str) -> str:
    """Collapse whitespace (the DOM matcher does the same) and number the placeholders."""
    s = re.sub(r"\s+", " ", html.unescape(s)).strip()   # the DOM shows decoded entities
    n = iter(range(1000))
    return re.sub(PH, lambda m: "{%d}" % next(n), s)


def pieces(static_with_ph: str):
    """Text segments of an HTML-ish literal: between tags, plus attribute values inside the tags."""
    for m in ATTR_RE.finditer(static_with_ph):
        yield m.group(1)
    for seg in TAG_RE.split(static_with_ph):
        yield seg
    m = re.match(r"^https?://\S+(.*)$", static_with_ph, re.S)   # Doctor links: the UI shows the text after the URL
    if m:
        yield m.group(1)


# ------------------------------------------------------------------ JavaScript
REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^") | {""}
ESC = {"n": "\n", "t": " ", "r": ""}


def js_literals(src: str):
    """Static text of every string / template literal (template ${...} replaced by PH; literals inside ${...},
    regex literals and comments handled by the same scanner)."""
    out: list[str] = []
    _scan(src, 0, out, top=True)
    return out


def _scan(src: str, i: int, out: list, top: bool) -> int:
    """Scan code from i; inside ${...} (top=False) stop after the matching '}'. Returns the end index."""
    n, prev, depth = len(src), "", 0
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i); i = n if j < 0 else j; continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2); i = n if j < 0 else j + 2; continue
        if c in "'\"":
            j, buf = i + 1, []
            while j < n and src[j] != c and src[j] != "\n":
                if src[j] == "\\" and j + 1 < n:
                    buf.append(ESC.get(src[j + 1], src[j + 1])); j += 2; continue
                buf.append(src[j]); j += 1
            out.append("".join(buf)); i = j + 1; prev = "x"; continue
        if c == "`":
            i = _template(src, i + 1, out); prev = "x"; continue
        if c == "/" and prev in REGEX_PREV:
            j, cls = i + 1, False
            while j < n and src[j] != "\n":
                if src[j] == "\\": j += 2; continue
                if src[j] == "[": cls = True
                elif src[j] == "]": cls = False
                elif src[j] == "/" and not cls: break
                j += 1
            i = j + 1; prev = "x"; continue
        if not top:
            if c == "{": depth += 1
            elif c == "}":
                if depth == 0: return i + 1
                depth -= 1
        if c.isalpha() or c == "_" or c == "$":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_$"): j += 1
            word = src[i:j]
            prev = "" if word in ("return", "typeof", "case", "in", "of", "new", "delete", "void", "yield", "await") else "x"
            i = j; continue
        if not c.isspace():
            prev = c if c in REGEX_PREV else "x"
        i += 1
    return i


def _template(src: str, j: int, out: list) -> int:
    """Template literal body starting after the backtick; appends its static text (PH for ${...}); returns end index."""
    n, buf = len(src), []
    while j < n and src[j] != "`":
        if src[j] == "\\" and j + 1 < n:
            buf.append(ESC.get(src[j + 1], src[j + 1])); j += 2; continue
        if src.startswith("${", j):
            j = _scan(src, j + 2, out, top=False); buf.append(PH); continue
        buf.append(src[j]); j += 1
    out.append("".join(buf))
    return j + 1


# ------------------------------------------------------------------ HTML
class _H(HTMLParser):
    def __init__(self):
        super().__init__(); self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"): self.skip += 1
        for k, v in attrs:
            if k in ATTRS and v: self.out.append(v)

    def handle_endtag(self, tag):
        if tag in ("script", "style"): self.skip -= 1

    def handle_data(self, d):
        if not self.skip: self.out.append(d)


# ------------------------------------------------------------------ Python
def py_strings(src: str):
    tree = ast.parse(src)
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                doc_nodes.add(id(first.value))
    inner = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                inner.add(id(v))
    for node in ast.walk(tree):
        if id(node) in doc_nodes or id(node) in inner:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            yield "".join(v.value if isinstance(v, ast.Constant) else PH for v in node.values)


def collect() -> dict[str, str]:
    VOCAB.clear(); VOCAB.update(_vocab())
    found: dict[str, str] = {}

    def add(raw: str, where: str, split: bool = True):
        for seg in (pieces(raw) if split else [raw]):
            if is_text(seg.replace(PH, "")):
                key = norm(seg)
                if re.fullmatch(r"(\{\d+\}|[\W\d_])*", key) or CODE_TPL.search(key):   # CODE_TPL: .nk templates
                    continue
                found.setdefault(key, where)

    for f in sorted((REPO / "static").glob("*.js")):
        for lit in js_literals(f.read_text(encoding="utf-8")):
            add(lit, f.name)
    h = _H(); h.feed((REPO / "static" / "index.html").read_text(encoding="utf-8"))
    for t in h.out:
        add(t, "index.html", split=False)   # parser text is already decoded: "&lt;plate&gt;" is text, not a tag
    for f in [REPO / "app.py", REPO / "doctor.py", REPO / "comfy_client.py", *sorted((REPO / "engine").glob("*.py"))]:
        for s in py_strings(f.read_text(encoding="utf-8")):
            add(s, f.name)
    return found


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    found = collect()
    cat = json.loads(CATALOG.read_text(encoding="utf-8")) if CATALOG.exists() else {}
    have = set(cat.get("strings", {}))
    missing = {k: v for k, v in found.items() if k not in have}
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps({k: "" for k in missing}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(found)} strings, {len(missing)} missing -> {out}")
        return 0
    for k, v in sorted(missing.items(), key=lambda kv: kv[1]):
        print(f"[{v}] {k}")
    print(f"{len(found)} strings, {len(missing)} missing from {CATALOG.name}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
