#!/usr/bin/env python3
"""Re-extract the PAT from upload/MASTER_PROMPT.pdf (two-line splice) and
verify it byte-matches the stored 600-mode credential file.
Prints ONLY a verdict + hash prefix, never the token."""
import re, subprocess, tempfile, os, hashlib, sys

PDF = "/home/z/my-project/upload/MASTER_PROMPT.pdf"
CRED = "/home/z/my-project/.credentials/github_pat"
EXPECTED = 93

txt = tempfile.mktemp(suffix=".txt")
subprocess.run(["pdftotext", "-layout", PDF, txt], check=True)
lines = open(txt, encoding="utf-8", errors="replace").read().splitlines()
os.unlink(txt)

token = None
for i, line in enumerate(lines):
    m = re.search(r"github_pat_[A-Za-z0-9_]+", line)
    if not m:
        continue
    tok = m.group(0)
    j = i + 1
    while len(tok) < EXPECTED and j < len(lines) and j <= i + 2:
        cont = lines[j].strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", cont):
            tok += cont
            j += 1
        else:
            break
    if len(tok) == EXPECTED and re.fullmatch(r"github_pat_[A-Za-z0-9_]+", tok):
        token = tok
        print(f"PDF splice: found at line {i+1}, length {len(tok)}, "
              f"sha256 prefix {hashlib.sha256(tok.encode()).hexdigest()[:12]}")
        break

if token is None:
    print("VERDICT: PDF splice FAILED (no 93-char token reconstructed)")
    sys.exit(1)

stored = open(CRED).read().strip()
stored_h = hashlib.sha256(stored.encode()).hexdigest()[:12]
print(f"Stored cred: length {len(stored)}, sha256 prefix {stored_h}")
match = stored == token
print(f"VERDICT: {'MATCH — stored credential is current' if match else 'MISMATCH — refreshing stored credential'}")
if not match:
    os.makedirs(os.path.dirname(CRED), exist_ok=True)
    with open(CRED, "w") as f:
        f.write(token + "\n")
    os.chmod(CRED, 0o600)
    print("Stored credential refreshed (mode 600 preserved)")
sys.exit(0 if match else 2)
