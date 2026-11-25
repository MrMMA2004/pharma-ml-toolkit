#!/usr/bin/env python3
"""Recover the GitHub PAT from upload/MASTER_PROMPT.pdf.

Two-line splice: the 93-char fine-grained token is split across two
lines of the PDF text. This script splices, validates (93 chars,
github_pat_ prefix, token charset), and stores it in a 600-mode file.
The token is NEVER printed; only length/pattern booleans are emitted.
"""
import re
import subprocess
import sys
from pathlib import Path

PDF = Path("/home/z/my-project/upload/MASTER_PROMPT.pdf")
OUT_DIR = Path("/home/z/my-project/.credentials")
OUT_FILE = OUT_DIR / "github_pat"
EXPECTED_LEN = 93
PREFIX = "github_pat_"
CHARSET = re.compile(r"^[A-Za-z0-9_]+$")


def extract_text() -> str:
    # Try pdftotext first, fall back to pypdf.
    try:
        r = subprocess.run(
            ["pdftotext", "-layout", str(PDF), "-"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    except FileNotFoundError:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(PDF))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: pdf text extraction failed: {e}", file=sys.stderr)
        sys.exit(2)


def candidate_splices(lines: list[str]):
    """Yield (token, context) pairs from two-line splices."""
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        for head_line, tail_line in ((a, b),):
            idx = head_line.find(PREFIX)
            if idx < 0:
                continue
            frag1 = head_line[idx:].strip()
            # Take leading token-charset chars from the next line.
            m = re.match(r"[A-Za-z0-9_]+", tail_line.strip())
            frag2 = m.group(0) if m else ""
            # Splice: frag1 + as much of frag2 as needed / all of it.
            for take in range(len(frag2), -1, -1):
                tok = frag1 + frag2[:take]
                yield tok, (i, take)


def main() -> None:
    text = extract_text()
    lines = text.splitlines()
    found = None
    for tok, ctx in candidate_splices(lines):
        if len(tok) == EXPECTED_LEN and tok.startswith(PREFIX) and CHARSET.match(tok):
            found = (tok, ctx)
            break
    if not found:
        # Diagnostic without leaking content: show line count and any
        # line lengths around the prefix occurrences.
        hits = [i for i, ln in enumerate(lines) if PREFIX in ln]
        print(f"FAIL: no 93-char splice found; prefix lines at {hits}")
        for h in hits:
            tail = lines[h][lines[h].find(PREFIX):]
            print(f"  line {h}: frag len {len(tail.strip())} (charset ok={bool(CHARSET.match(tail.strip()))})")
            if h + 1 < len(lines):
                nxt = lines[h + 1].strip()
                print(f"  next line: len {len(nxt)}, head charset frag len {len(re.match(r'[A-Za-z0-9_]+', nxt).group(0)) if re.match(r'[A-Za-z0-9_]+', nxt) else 0}")
        sys.exit(3)

    tok, ctx = found
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Write without echoing content.
    OUT_FILE.write_text(tok + "\n", encoding="utf-8")
    OUT_FILE.chmod(0o600)
    import os
    mode = os.stat(OUT_FILE).st_mode & 0o777
    print(f"OK: token recovered via splice {ctx}; stored at {OUT_FILE}; "
          f"length={len(tok)}; prefix_ok={tok.startswith(PREFIX)}; mode={oct(mode)}")


if __name__ == "__main__":
    main()
