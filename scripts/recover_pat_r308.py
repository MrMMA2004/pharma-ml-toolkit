#!/usr/bin/env python3
"""Recover the GitHub PAT from upload/MASTER_PROMPT.pdf (two-line splice, 93 chars).

Protocol (r306/r307 precedent): the token is split across two adjacent extracted
text lines. Re-splice, validate, store 600-mode, NEVER print the secret.
Prints masked diagnostics only.
"""
import re
import subprocess
import sys
from pathlib import Path

PDF = Path("/home/z/my-project/upload/MASTER_PROMPT.pdf")
OUT = Path.home() / ".github_token"

txt = subprocess.run(["pdftotext", "-layout", str(PDF), "-"],
                     capture_output=True, text=True, check=True).stdout
lines = [ln.rstrip("\n") for ln in txt.splitlines()]

# charset of a fine-grained/classic PAT body
TOK_RE = re.compile(r"^[A-Za-z0-9_]+$")


def mask(s: str) -> str:
    return f"{s[:11]}...{s[-4:]} (len={len(s)})" if len(s) > 20 else "<short>"


candidates = []
for i, a in enumerate(lines):
    a_t = a.strip()
    if len(a_t) < 10 or not TOK_RE.match(a_t.replace("-", "").replace("_", "")):
        # allow lines that are pure token fragments
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
    # fallback: single line already 93 chars
    for i, a in enumerate(lines):
        a_t = a.strip()
        if len(a_t) == 93 and TOK_RE.match(a_t):
            candidates.append((i, i, a_t))

uniq = {}
for i, j, tok in candidates:
    uniq.setdefault(tok, (i, j))

if not uniq:
    print("FAIL: no 93-char splice candidate found")
    # masked diagnostics of plausible fragment lines
    for i, ln in enumerate(lines):
        t = ln.strip()
        if 20 <= len(t) <= 80 and re.match(r"^[A-Za-z0-9_\-]+$", t):
            print(f"  line {i}: {mask(t)}")
    sys.exit(1)

if len(uniq) > 1:
    print(f"WARN: {len(uniq)} distinct candidates; picking first")
    for tok, (i, j) in list(uniq.items())[:3]:
        print(f"  lines {i}+{j}: {mask(tok)}")

tok, (i, j) = next(iter(uniq.items()))
OUT.write_text(tok)  # single line, no trailing newline beyond the token
OUT.chmod(0o600)
print(f"OK: token stored {mask(tok)} from lines {i}+{j}; mode={oct(OUT.stat().st_mode & 0o777)}, size={OUT.stat().st_size}")
