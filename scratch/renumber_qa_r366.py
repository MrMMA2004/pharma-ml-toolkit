#!/usr/bin/env python3
"""Insert the r366 verification-pass section as the new section 1 of
docs/QA/TEST_EXECUTION_REPORT.md and cascade-renumber the existing
round-section headings (1..N -> 2..N+1). The two trailing static
sections (Standard skips / Regression requirements) keep their legacy
numbers, exactly as the file already does today. Also trues the
preamble's 'Executed-this-round' round reference to r366."""
import re
import sys
from pathlib import Path

DOC = Path("/home/z/my-project/pharma-ml-toolkit/docs/QA/TEST_EXECUTION_REPORT.md")
NEW_SECTION = Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else None

if NEW_SECTION is None:
    sys.exit("usage: renumber_qa_r366.py <new-section-file.md>")

text = DOC.read_text(encoding="utf-8")
lines = text.splitlines()

# 0. true the preamble's round reference (r363 -> r366)
for i, line in enumerate(lines):
    if "Executed-this-round (r363," in line:
        lines[i] = line.replace("Executed-this-round (r363,", "Executed-this-round (r366,")
        break

STATIC_TITLES = ("Standard skips and blocked legs", "Regression requirements compliance")

def is_round_heading(line: str) -> int | None:
    m = re.match(r"^## (\d+)\. ", line)
    if not m:
        return None
    for t in STATIC_TITLES:
        if t in line:
            return None
    return int(m.group(1))

out = []
for line in lines:
    n = is_round_heading(line)
    if n is not None:
        line = re.sub(r"^## \d+\. ", f"## {n + 1}. ", line)
    out.append(line)

text = "\n".join(out) + "\n"

# insert the new section right before the (now) "## 2. Executed this
# round (r365 ..." heading — the first round heading in the file.
m = re.search(r"^## 2\. Executed this round \(r365", text, re.MULTILINE)
assert m, "the r365 section heading (now ## 2.) not found"
text = text[: m.start()] + NEW_SECTION.rstrip() + "\n\n" + text[m.start() :]

DOC.write_text(text, encoding="utf-8")
print("renumbered + inserted; headings now:")
for h in re.findall(r"^## \d+\. .{0,60}", text, re.MULTILINE)[:6]:
    print(" ", h)
