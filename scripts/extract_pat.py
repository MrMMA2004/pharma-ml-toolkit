#!/usr/bin/env python3
"""Reconstruct the GitHub PAT from upload/MASTER_PROMPT.pdf text dump.

The 93-char token gets line-wrapped at ~76 chars by pdftotext, so the
remainder (17 chars) lands on a continuation line. Join them, validate,
write to ~/.github_token. Never print the token itself.
"""
import re
import sys

SRC = "/tmp/mp.txt"
OUT = "/home/z/.github_token"
EXPECTED = 93

def main() -> int:
    text = open(SRC, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"github_pat_[A-Za-z0-9_]+", line)
        if not m:
            continue
        tok = m.group(0)
        # join continuation lines while token is shorter than expected
        j = i + 1
        while len(tok) < EXPECTED and j < len(lines) and j < i + 3:
            cont = lines[j].strip()
            if re.fullmatch(r"[A-Za-z0-9_]{8,}", cont):
                tok += cont
                j += 1
            else:
                break
        if len(tok) != EXPECTED:
            print(f"line {i+1}: candidate length {len(tok)} != {EXPECTED}, trying more")
            # try joining aggressively
            k = i + 1
            while len(tok) < EXPECTED and k < len(lines):
                tok += lines[k].strip()
                k += 1
            tok = tok[: len(tok)]  # no trim of extras; validate below
        if len(tok) == EXPECTED and re.fullmatch(r"github_pat_[A-Za-z0-9_]+", tok):
            open(OUT, "w").write(tok + "\n")
            import os
            os.chmod(OUT, 0o600)
            print(f"OK: {EXPECTED} chars, prefix {tok[:11]}..., written to {OUT}")
            return 0
        print(f"line {i+1}: no valid reconstruction (len {len(tok)})")
    print("NO TOKEN FOUND")
    return 1

if __name__ == "__main__":
    sys.exit(main())
