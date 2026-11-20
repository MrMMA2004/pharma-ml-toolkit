#!/usr/bin/env python3
"""Extract MASTER_PROMPT part12 text with token redaction.

Any token-like string (github_pat_..., long hex/base62 runs >= 30 chars)
is replaced with a redaction marker before the text is printed. The
canonical 93-char PAT is also matched explicitly.
"""
import re
import subprocess
import sys

SRC = "/home/z/my-project/upload/MASTER_PROMPT part12.pdf"
DST = "/tmp/mp12.txt"


def redact(text: str) -> str:
    tok = open("/home/z/.github_token").read().strip()
    # explicit canonical token first (split across lines too)
    text = text.replace(tok, "[REDACTED-PAT]")
    head = tok[: 40]
    tail = tok[-25:]
    text = text.replace(head, "[REDACTED-PAT-head]")
    text = text.replace(tail, "[REDACTED-PAT-tail]")
    # generic token-shaped patterns
    text = re.sub(r"github_pat_[A-Za-z0-9_]+", "[REDACTED-PAT]", text)
    text = re.sub(r"\bghp_[A-Za-z0-9]{20,}\b", "[REDACTED-ghp]", text)
    text = re.sub(r"\b[A-Za-z0-9_]{70,}\b", "[REDACTED-LONGRUN]", text)
    text = re.sub(r"\b[0-9a-fA-F]{40,}\b", "[REDACTED-HEX]", text)
    return text


def main() -> int:
    subprocess.run(["pdftotext", SRC, DST], check=True)
    raw = open(DST, encoding="utf-8", errors="replace").read()
    safe = redact(raw)
    print(safe)
    # also write the redacted copy for reference
    open("/tmp/mp12_redacted.txt", "w", encoding="utf-8").write(safe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
