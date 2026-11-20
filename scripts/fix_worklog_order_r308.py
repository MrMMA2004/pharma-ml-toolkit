#!/usr/bin/env python3
"""Move the misplaced Round 308 worklog block to EOF (after Round 307)."""
from pathlib import Path

P = Path("/home/z/my-project/worklog.md")
text = P.read_text()

START = "**Round 308 (the 99th recorded run"
END = "installer signing (certificate-gated)."

i = text.find(START)
assert i != -1, "round 308 block start not found"
j = text.find(END, i) + len(END)
block = text[i:j]

before = text[:i]          # ends "...(certificate-gated).\n\n---\n\n"
after = text[j:]           # starts "\n\n---\n\n**Round 307"
assert before.endswith("(certificate-gated).\n\n---\n\n"), repr(before[-60:])
assert after.startswith("\n\n---\n\n**Round 307"), repr(after[:60])

mid = before + after[len("\n\n---\n\n"):]
result = mid.rstrip() + "\n\n---\n\n" + block + "\n"
P.write_text(result)
lines = result.splitlines()
print("round order tail:", [ln for ln in lines if ln.startswith("**Round 30")][-3:])
print("new tail:", lines[-2:])
