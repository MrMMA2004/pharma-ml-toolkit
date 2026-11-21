#!/usr/bin/env python3
"""R319 PAT recovery (twelfth rollback): upload/MASTER_PROMPT.pdf is gone,
so re-splice the 93-char fine-grained PAT from the surviving text extract
scratch/pdf-extract/MASTER_PROMPT.txt (fallback: upload/Prompt_Registry.pdf
via pdftotext). Each candidate is API-validated before storage; the token is
NEVER printed -- masked output only, 600-mode files, no argv/env exposure
beyond the in-process request headers."""
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PRIMARY = Path("/home/z/my-project/scratch/pdf-extract/MASTER_PROMPT.txt")
FALLBACK_PDFS = [
    Path("/home/z/my-project/upload/Prompt_Registry.pdf"),
    Path("/home/z/my-project/upload/GitHub_Links_Registry.pdf"),
]
OUTS = [
    Path("/home/z/my-project/.credentials/github_pat"),
    Path.home() / ".github_token",
]
EXPECTED = 93


def load_lines() -> list[str]:
    texts: list[str] = []
    if PRIMARY.is_file():
        texts.append(PRIMARY.read_text(encoding="utf-8", errors="replace"))
    for pdf in FALLBACK_PDFS:
        if pdf.is_file():
            try:
                texts.append(subprocess.run(
                    ["pdftotext", "-layout", str(pdf), "-"],
                    capture_output=True, text=True, check=True).stdout)
            except Exception:
                pass
    lines: list[str] = []
    for t in texts:
        lines.extend(l.strip() for l in t.splitlines() if l.strip())
    return lines


def candidates(lines: list[str]) -> list[str]:
    cands: list[str] = []
    for l in lines:
        for m in re.finditer(r"github_pat_[A-Za-z0-9_]+", l):
            cands.append(m.group(0))
    # the 76+17 line-wrap splice: prefix line + token-suffix continuation
    for i, a in enumerate(lines):
        m = re.search(r"github_pat_[A-Za-z0-9_]{20,}", a)
        if not m:
            continue
        tok = m.group(0)
        if len(tok) == EXPECTED:
            continue  # already captured whole
        for j in (i + 1, i + 2):
            if j >= len(lines):
                break
            b = lines[j]
            if re.fullmatch(r"[A-Za-z0-9_]{4,}", b):
                spliced = tok + b
                if len(spliced) >= EXPECTED:
                    cands.append(spliced[:EXPECTED])
    return list(dict.fromkeys(cands))


def mask(t: str) -> str:
    return f"{t[:11]}...{t[-4:]} (len={len(t)})"


def api_user(tok: str):
    req = urllib.request.Request(
        "https://api.github.com/user",
        headers={"Authorization": "Bearer " + tok,
                 "User-Agent": "pat-recover-r319"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=20))
    except Exception as e:
        print(f"  candidate {mask(tok)} REJECTED: {str(e)[:80]}")
        return None


def main() -> int:
    lines = load_lines()
    cands = candidates(lines)
    print(f"{len(cands)} distinct {EXPECTED}-char candidates "
          f"(sources scanned: {1 if PRIMARY.is_file() else 0} txt + pdfs)")
    if not cands:
        print("FAIL: no candidates found")
        return 1
    for c in cands:
        if len(c) != EXPECTED or not re.fullmatch(r"github_pat_[A-Za-z0-9_]+", c):
            continue
        r = api_user(c)
        if r and r.get("login"):
            for out in OUTS:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(c)
                os.chmod(out, 0o600)
            print(f"VALIDATED {mask(c)} -> login {r['login']}; "
                  f"stored at {', '.join(str(o) for o in OUTS)} (mode 600)")
            return 0
    print("FAIL: no candidate authenticates")
    return 1


if __name__ == "__main__":
    sys.exit(main())
