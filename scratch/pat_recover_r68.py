#!/usr/bin/env python3
"""Round-68 PAT recovery (rollback-safe): pdftotext -layout backend.

The token NEVER touches stdout/logs -- only its length and sha256[:12].
"""
import hashlib, os, re, subprocess, sys

ROOT = os.path.expanduser("~/my-project")
PDF = os.path.join(ROOT, "upload", "MASTER_PROMPT.pdf")
CREDS = os.path.join(ROOT, ".credentials")
PAT_FILE = os.path.join(CREDS, "github_pat")

def main():
    txt = subprocess.run(
        ["pdftotext", "-layout", PDF, "-"],
        capture_output=True, text=True, check=True
    ).stdout
    lines = txt.splitlines()
    prefixes = ("github_pat_", "ghp_")
    cands = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        for p in prefixes:
            m = re.match(r"^(" + re.escape(p) + r"[A-Za-z0-9_]+)", stripped)
            if m:
                tok = m.group(1)
                cands.append((i, tok))
    # concat next line when the token is split across lines (93-char target)
    merged = []
    for i, tok in cands:
        if len(tok) < 93 and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            joined = tok + re.sub(r"\s+", "", nxt)
            if 90 <= len(joined) <= 96 and not nxt.startswith(prefixes):
                merged.append((i, joined, True))
            else:
                merged.append((i, tok, False))
        else:
            merged.append((i, tok, False))
    good = [(i, t) for (i, t, did) in merged if len(t) == 93]
    if not good:
        print("FAIL: no 93-char candidate; raw candidates:")
        for i, t, did in merged:
            print(f"  line {i+1}: len={len(t)} merged={did}")
        return 1
    if len(good) > 1:
        uniq = {t for _, t in good}
        if len(uniq) > 1:
            print(f"FAIL: {len(uniq)} distinct 93-char candidates")
            return 1
    i, tok = good[0]
    os.makedirs(CREDS, exist_ok=True)
    with open(PAT_FILE, "w") as f:
        f.write(tok)
    os.chmod(PAT_FILE, 0o600)
    print(f"OK: source=MASTER_PROMPT.pdf line {i+1}"
          f" (pdftotext -layout) len={len(tok)} "
          f"sha12={hashlib.sha256(tok.encode()).hexdigest()[:12]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
