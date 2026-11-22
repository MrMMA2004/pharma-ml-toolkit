#!/usr/bin/env python3
"""r333 housekeeping: recover PAT from upload/MASTER_PROMPT.pdf (93-char splice),
validate against GitHub API, store 600-mode at .credentials/github_pat.
Never prints the secret — masked diagnostics only."""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PDF = Path("/home/z/my-project/upload/MASTER_PROMPT.pdf")
CRED_DIR = Path("/home/z/my-project/.credentials")
OUT = CRED_DIR / "github_pat"
HOME_TOK = Path.home() / ".github_token"

txt = subprocess.run(["pdftotext", "-layout", str(PDF), "-"],
                     capture_output=True, text=True, check=True).stdout
lines = [ln.rstrip("\n") for ln in txt.splitlines()]
TOK_RE = re.compile(r"^[A-Za-z0-9_]+$")


def mask(s: str) -> str:
    return f"{s[:7]}...{s[-4:]} (len={len(s)})" if len(s) > 20 else "<short>"


candidates = []
for i, a in enumerate(lines):
    a_t = a.strip()
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
                candidates.append((i, j, spliced))
if not candidates:
    for i, a in enumerate(lines):
        a_t = a.strip()
        if len(a_t) == 93 and TOK_RE.match(a_t):
            candidates.append((i, i, a_t))

uniq = {}
for i, j, tok in candidates:
    uniq.setdefault(tok, (i, j))
if not uniq:
    print("FAIL: no 93-char splice candidate found")
    sys.exit(1)
if len(uniq) > 1:
    print(f"WARN: {len(uniq)} distinct candidates; validating each")

valid = None
for tok, (i, j) in uniq.items():
    req = urllib.request.Request(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {tok}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "pharma-ml-toolkit-housekeeping"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            who = json.loads(r.read().decode()).get("login", "?")
        if who == "MrMMA2004":
            valid = (tok, i, j, who)
            print(f"OK: lines {i}+{j} -> {mask(tok)} validates as {who}")
            break
        else:
            print(f"lines {i}+{j}: {mask(tok)} -> login={who} (mismatch)")
    except Exception as e:
        print(f"lines {i}+{j}: {mask(tok)} -> API error {type(e).__name__}")

if not valid:
    print("FAIL: no candidate validated as MrMMA2004")
    sys.exit(1)

tok, i, j, who = valid
CRED_DIR.mkdir(parents=True, exist_ok=True)
OUT.write_text(tok)
OUT.chmod(0o600)
HOME_TOK.write_text(tok)
HOME_TOK.chmod(0o600)
print(f"STORED: {OUT} mode={oct(OUT.stat().st_mode & 0o777)} "
      f"size={OUT.stat().st_size}; mirror at ~/.github_token (600)")
