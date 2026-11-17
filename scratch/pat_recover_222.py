#!/usr/bin/env python3
"""PAT recovery from upload/MASTER_PROMPT.pdf -- pdftotext backend.
Never prints the token; only length + sha256[:12]."""
import subprocess, re, hashlib, os, sys

PDF = os.path.expanduser("~/my-project/upload/MASTER_PROMPT.pdf")
OUT = os.path.expanduser("~/my-project/.credentials/github_pat")

txt = subprocess.run(["pdftotext", "-layout", PDF, "-"], capture_output=True, text=True).stdout

# candidate token-bearing lines: long runs of base64-ish / token chars
cands = []
for line in txt.splitlines():
    s = line.strip()
    if len(s) < 30:
        continue
    # strip common layout prefixes
    s2 = re.sub(r'^[^\w]*(?=(ghp_|github_pat_|[A-Za-z0-9_-]{40,}))', '', s)
    for m in re.finditer(r'[A-Za-z0-9_-]{40,}', s):
        cands.append(m.group(0))

# also try joining continuation lines (token split across two lines)
lines = [l.strip() for l in txt.splitlines() if l.strip()]
joined = []
for i in range(len(lines) - 1):
    combo = lines[i] + lines[i+1]
    for m in re.finditer(r'[A-Za-z0-9_-]{80,}', combo):
        joined.append(m.group(0))

pool = cands + joined
best = None
for c in pool:
    # GitHub PATs are 93 chars; ghp_ classic are 41+. prefer exact 93
    if len(c) == 93:
        best = c
        break
if best is None:
    for c in pool:
        if 80 <= len(c) <= 100:
            best = c
            break

if best is None:
    # fallback: single longest candidate
    if pool:
        best = max(pool, key=len)

if not best:
    print("NO CANDIDATE FOUND")
    sys.exit(1)

with open(OUT, "w") as f:
    f.write(best)
os.chmod(OUT, 0o600)
print(f"recovered: len={len(best)} sha12={hashlib.sha256(best.encode()).hexdigest()[:12]} -> {OUT} (mode 600)")
