#!/usr/bin/env python3
"""R320 PAT recovery (the 04:30 2h-sync housekeeping firing).

upload/MASTER_PROMPT.pdf is BACK in upload/ (the snapshot restore), so
the primary source is the PDF itself via pdftotext; fallbacks: the
r319 text extract and the registry PDFs.  The token is a 93-char
fine-grained PAT delivered as a two-line splice (a prefix line holding
`github_pat_...` + the next line's token-suffix continuation).

The token is NEVER printed: masked output only, 600-mode storage at
.credentials/github_pat + ~/.github_token, no argv/env exposure beyond
the in-process urllib request headers.  Each candidate is
API-validated before storage; an already-valid stored token short-
circuits the splice (validated first, reused if it authenticates).
"""
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PDF_PRIMARY = Path("/home/z/my-project/upload/MASTER_PROMPT.pdf")
TEXT_FALLBACKS = [
    Path("/home/z/my-project/scratch/pdf-extract/MASTER_PROMPT.txt"),
    Path("/home/z/my-project/scratch/master_prompt_full.txt"),
]
PDF_FALLBACKS = [
    Path("/home/z/my-project/upload/Prompt_Registry.pdf"),
    Path("/home/z/my-project/upload/GitHub_Links_Registry.pdf"),
]
STORED = Path("/home/z/my-project/.credentials/github_pat")
OUTS = [
    Path("/home/z/my-project/.credentials/github_pat"),
    Path.home() / ".github_token",
]
EXPECTED = 93


def load_lines() -> list[str]:
    texts: list[str] = []
    if PDF_PRIMARY.is_file():
        try:
            texts.append(subprocess.run(
                ["pdftotext", "-layout", str(PDF_PRIMARY), "-"],
                capture_output=True, text=True, check=True).stdout)
            print(f"primary source: {PDF_PRIMARY.name} (pdftotext OK)")
        except Exception as e:
            print(f"primary pdftotext failed: {str(e)[:80]}")
    for txt in TEXT_FALLBACKS:
        if txt.is_file():
            texts.append(txt.read_text(encoding="utf-8", errors="replace"))
    for pdf in PDF_FALLBACKS:
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
    # the two-line splice: prefix line + token-suffix continuation
    for i, a in enumerate(lines):
        m = re.search(r"github_pat_[A-Za-z0-9_]{20,}", a)
        if not m:
            continue
        tok = m.group(0)
        if len(tok) == EXPECTED:
            continue  # already whole on one line
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
                 "User-Agent": "pat-recover-r320"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=20))
    except Exception as e:
        print(f"  candidate {mask(tok)} REJECTED: {str(e)[:80]}")
        return None


def store(tok: str) -> None:
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tok)
        os.chmod(out, 0o600)
    modes = [oct(o.stat().st_mode & 0o777) for o in OUTS]
    print(f"VALIDATED {mask(tok)} -> stored at "
          f"{', '.join(str(o) for o in OUTS)} (modes {modes})")


def main() -> int:
    # 1) an already-stored token, if it still authenticates, wins
    if STORED.is_file():
        stored = STORED.read_text().strip()
        r = api_user(stored)
        if r and r.get("login"):
            print(f"stored credential still valid: {mask(stored)} "
                  f"-> login {r['login']}")
            store(stored)  # re-assert 600 modes on both homes
            return 0
        print("stored credential REJECTED -- re-splicing from the PDF")

    # 2) splice from the PDF (primary) / fallbacks
    cands = candidates(load_lines())
    print(f"{len(cands)} distinct candidates scanned "
          f"(source: upload/MASTER_PROMPT.pdf + fallbacks)")
    for c in cands:
        if len(c) != EXPECTED or not re.fullmatch(r"github_pat_[A-Za-z0-9_]+", c):
            continue
        r = api_user(c)
        if r and r.get("login"):
            store(c)
            return 0
    print("FAIL: no candidate authenticates")
    return 1


if __name__ == "__main__":
    sys.exit(main())
