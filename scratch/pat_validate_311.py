#!/usr/bin/env python3
"""Round 311 PAT validation: re-derive the 93-char splice candidates from
upload/MASTER_PROMPT.pdf, test each against the GitHub API (the r310
'API-validated' step), and store ONLY the candidate that authenticates.
Never prints the secret -- masked output only."""
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

PDF = "/home/z/my-project/upload/MASTER_PROMPT.pdf"
OUT = Path.home() / ".github_token"

txt = subprocess.run(["pdftotext", "-layout", PDF, "-"],
                     capture_output=True, text=True, check=True).stdout
lines = [l.rstrip("\n") for l in txt.splitlines()]
TOK_RE = re.compile(r"^[A-Za-z0-9_]+$")

cands = []
for i, a in enumerate(lines):
    a_t = a.strip()
    if len(a_t) < 10 or not TOK_RE.match(a_t.replace("-", "").replace("_", "")):
        if not re.match(r"^[A-Za-z0-9_\-]+$", a_t) or len(a_t) < 10:
            continue
    for j in (i + 1, i + 2):
        if j >= len(lines):
            break
        b_t = lines[j].strip()
        if not re.match(r"^[A-Za-z0-9_\-]+$", b_t) or len(b_t) < 10:
            continue
        for spliced in (a_t + b_t, a_t + b_t[::-1]):
            if len(spliced) == 93 and TOK_RE.match(spliced):
                cands.append(spliced)
uniq = list(dict.fromkeys(cands))
print(f"{len(uniq)} distinct 93-char candidates")


def mask(s: str) -> str:
    return f"{s[:11]}...{s[-4:]} (len={len(s)})"


def api_ok(tok: str) -> dict | None:
    req = urllib.request.Request(
        "https://api.github.com/repos/MrMMA2004/pharma-ml-toolkit",
        headers={"Authorization": "Bearer " + tok,
                 "User-Agent": "pat-validate-r311"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=20))
    except Exception as e:
        print(f"  candidate {mask(tok)} REJECTED: {str(e)[:90]}")
        return None


for c in uniq:
    r = api_ok(c)
    if r and r.get("full_name") == "MrMMA2004/pharma-ml-toolkit":
        OUT.write_text(c)
        OUT.chmod(0o600)
        print(f"VALIDATED {mask(c)} -> repo {r['full_name']} "
              f"(default {r['default_branch']}, pushed {r['pushed_at']}, "
              f"visibility {r['visibility']})")
        print(f"stored the API-validated token at {OUT} "
              f"(mode={oct(OUT.stat().st_mode & 0o777)}, size={OUT.stat().st_size})")
        break
else:
    print("FAIL: no candidate authenticates")
    raise SystemExit(1)
