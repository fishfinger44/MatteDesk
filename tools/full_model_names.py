# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full model names in user-visible text (user rule 2026-10-18: always `MatAnyone2`, `SAM 3.1`, `CoTracker3`, `LTX-2.5`).

Rewrites only TEXT – never identifiers:
- JS (static/*.js): contents of '…' / "…" literals and the text parts of `…` templates (not their ${…} code), comments
  untouched; a literal that is exactly 'SAM3' / 'LTX' is a stage key (STAGES key, paneTitle('SAM3')) unless it is a `label:`;
- Python (app.py, doctor.py, engine/*.py): string tokens (docstrings included), comments untouched;
- HTML (static/index.html): everything outside <script>/<style> tags' code;
- Markdown (README*.md, docs/*.md except PLAN_*): prose outside `code` spans and ``` blocks;
- the English catalog static/i18n/en.json: keys (= the Polish source text) and values with the same rules.
Identifiers survive by construction (SAM3_Detect, MatAnyone2…, ST.SAM3 in code, lower-case ids, file names).

    python tools/full_model_names.py          -> rewrite in place, print a per-file count
    python tools/full_model_names.py --check  -> list files that still need it (exit 1)
"""
from __future__ import annotations

import io
import json
import re
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RULES = [
    (re.compile(r"\bMatAnyone(?![\w])"), "MatAnyone2"),
    (re.compile(r"\bSAM ?3(?![\w.])"), "SAM 3.1"),
    (re.compile(r"\bCoTracker(?![\w])"), "CoTracker3"),
    (re.compile(r"\bLTX(?![\w-])"), "LTX-2.5"),
]
STAGE_KEYS = {"SAM3", "LTX"}          # whole JS literals that are keys (see the module doc)


def fix_text(s: str) -> str:
    for rx, new in RULES:
        s = rx.sub(new, s)
    return s


# ---------------------------------------------------------------------------------------------------- JS
_REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^")


def fix_js(src: str) -> str:
    out = []
    i, n = 0, len(src)
    stack = []            # template nesting: each entry = brace depth inside the current ${…}
    last_sig = ""         # last non-space code char (regex vs division)

    def code_char(c):
        nonlocal last_sig
        if not c.isspace():
            last_sig = c
        out.append(c)

    while i < n:
        c = src[i]
        if stack and c == "}" and stack[-1] == 0:          # end of ${…}: back into the template text
            stack.pop()
            out.append(c)
            i += 1
            i = _template_text(src, i, out, stack)
            last_sig = "`"
            continue
        if stack and c == "{":
            stack[-1] += 1
        elif stack and c == "}":
            stack[-1] -= 1
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(src[i:j])
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(src[i:j])
            i = j
            continue
        if c == "/" and (last_sig in _REGEX_PREV or last_sig == "" or src[max(0, i - 7):i].rstrip().endswith("return")):
            j = i + 1
            in_cls = False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    in_cls = True
                elif src[j] == "]":
                    in_cls = False
                elif src[j] == "/" and not in_cls:
                    break
                j += 1
            out.append(src[i:j + 1])
            i = j + 1
            last_sig = "/"
            continue
        if c in "'\"":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            body = src[i + 1:j]
            before = _tail(out, 12)
            if body in STAGE_KEYS and not re.search(r"label:\s*$", before):
                out.append(src[i:j + 1])
            else:
                out.append(c + fix_text(body) + (src[j] if j < n else ""))
            i = j + 1
            last_sig = c
            continue
        if c == "`":
            out.append(c)
            i = _template_text(src, i + 1, out, stack)
            last_sig = "`"
            continue
        code_char(c)
        i += 1
    return "".join(out)


def _tail(out, k):
    acc = ""
    for piece in reversed(out):
        acc = piece + acc
        if len(acc) >= k:
            break
    return acc[-k:]


def _template_text(src, i, out, stack):
    """Copy template text from i (fixed) up to the closing ` (included) or a ${ (included, pushes a code level)."""
    n = len(src)
    j = i
    while j < n:
        if src[j] == "\\":
            j += 2
            continue
        if src[j] == "`":
            out.append(fix_text(src[i:j]) + "`")
            return j + 1
        if src.startswith("${", j):
            out.append(fix_text(src[i:j]) + "${")
            stack.append(0)
            return j + 2
        j += 1
    out.append(fix_text(src[i:]))
    return n


# ---------------------------------------------------------------------------------------------------- Python
def fix_py(src: str) -> str:
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    lines = src.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    edits = []
    for t in toks:
        if t.type == tokenize.STRING:
            a = offs[t.start[0] - 1] + t.start[1]
            b = offs[t.end[0] - 1] + t.end[1]
            prefix = re.match(r"[rbuRBUfF]*", t.string).group(0).lower()
            new = _fix_fstring(t.string) if "f" in prefix else fix_text(t.string)
            if new != t.string:
                edits.append((a, b, new))
    for a, b, new in reversed(edits):
        src = src[:a] + new + src[b:]
    return src


def _fix_fstring(s: str) -> str:
    """f-string token: fix the literal parts, keep {expressions}."""
    parts = re.split(r"(\{[^{}]*\})", s)
    return "".join(p if p.startswith("{") and not p.startswith("{{") else fix_text(p) for p in parts)


# ---------------------------------------------------------------------------------------------------- HTML / Markdown
def fix_html(src: str) -> str:
    parts = re.split(r"(<script\b.*?</script>|<style\b.*?</style>)", src, flags=re.S)
    return "".join(p if p.startswith(("<script", "<style")) else fix_text(p) for p in parts)


def fix_md(src: str) -> str:
    parts = re.split(r"(```.*?```|`[^`\n]*`)", src, flags=re.S)
    return "".join(p if p.startswith("`") else fix_text(p) for p in parts)


def fix_catalog(src: str) -> str:
    cat = json.loads(src)
    cat["strings"] = {fix_text(k): fix_text(v) for k, v in cat["strings"].items()}
    out = json.dumps(cat, ensure_ascii=False, indent=1) + "\n"
    return out.replace("\n", "\r\n") if "\r\n" in src else out     # keep the file's line endings


def targets():
    for p in sorted((REPO / "static").glob("*.js")):
        yield p, fix_js
    yield REPO / "static" / "index.html", fix_html
    for p in [REPO / "app.py", REPO / "doctor.py", *sorted((REPO / "engine").glob("*.py"))]:
        yield p, fix_py
    for p in [REPO / "README.md", REPO / "README.pl.md", *sorted((REPO / "docs").glob("*.md"))]:
        if not p.name.startswith("PLAN_"):
            yield p, fix_md
    yield REPO / "static" / "i18n" / "en.json", fix_catalog


def main(argv):
    check = "--check" in argv
    todo = []
    for p, fn in targets():
        if not p.exists():
            continue
        with open(p, encoding="utf-8", newline="") as fh:   # keep CRLF / LF as they are
            src = fh.read()
        new = fn(src)
        if new != src:
            todo.append(p)
            if not check:
                with open(p, "w", encoding="utf-8", newline="") as fh:
                    fh.write(new)
            n = sum(1 for a, b in zip(src.splitlines(), new.splitlines()) if a != b)
            print(f"{p.relative_to(REPO)}: {n} lines")
    return 1 if (check and todo) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
