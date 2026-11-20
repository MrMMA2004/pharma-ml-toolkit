#!/usr/bin/env python3
"""Part12 session recovery: PAT from upload/MASTER_PROMPT.pdf -> ~/.github_token
(600) + ~/.git_askpass_gh.sh shim + live API login verification.
The token NEVER touches stdout/logs -- only its length and sha256[:12].
"""
import subprocess, re, hashlib, os, stat, sys, json, urllib.request

ROOT = os.path.expanduser("~/my-project")
PDF = os.path.join(ROOT, "upload", "MASTER_PROMPT.pdf")
TOKEN_FILE = os.path.expanduser("~/.github_token")
ASKPASS = os.path.expanduser("~/.git_askpass_gh.sh")

txt = subprocess.run(["pdftotext", "-layout", PDF, "-"],
                     capture_output=True, text=True, check=True).stdout
lines = [l.strip() for l in txt.splitlines() if l.strip()]

cands = []
for line in lines:
    for m in re.finditer(r'(?:github_pat_|ghp_)[A-Za-z0-9_]{30,}', line):
        cands.append(m.group(0))
# two-line splice (93-char target)
joined = []
for i in range(len(lines) - 1):
    combo = lines[i] + lines[i + 1]
    for m in re.finditer(r'(?:github_pat_|ghp_)[A-Za-z0-9_]{30,}', combo):
        joined.append(m.group(0))

pool = cands + joined
best = None
for c in pool:
    if len(c) == 93:
        best = c
        break
if best is None and pool:
    for c in pool:
        if 80 <= len(c) <= 100:
            best = c
            break
if best is None:
    print("NO CANDIDATE FOUND")
    sys.exit(1)

with open(TOKEN_FILE, "w") as f:
    f.write(best + "\n")
os.chmod(TOKEN_FILE, 0o600)

with open(ASKPASS, "w") as f:
    f.write('#!/bin/sh\ncat "%s"\n' % TOKEN_FILE)
os.chmod(ASKPASS, 0o700)

sha12 = hashlib.sha256(best.encode()).hexdigest()[:12]

req = urllib.request.Request(
    "https://api.github.com/user",
    headers={"Authorization": "Bearer " + best,
             "User-Agent": "part12-recovery",
             "Accept": "application/vnd.github+json"})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        who = json.loads(r.read().decode())
        print(f"recovered: len={len(best)} sha12={sha12} -> {TOKEN_FILE} (600) + askpass shim (700)")
        print(f"API verify: HTTP {r.status} login={who.get('login')} plan={who.get('plan')}")
except Exception as e:
    print(f"recovered: len={len(best)} sha12={sha12} -> {TOKEN_FILE} (600) + askpass shim (700)")
    print(f"API verify FAILED: {e}")
    sys.exit(2)
