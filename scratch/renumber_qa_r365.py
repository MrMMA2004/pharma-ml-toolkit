#!/usr/bin/env python3
"""Insert the r365 continuation section as the new section 1 of
docs/QA/TEST_EXECUTION_REPORT.md and cascade-renumber the existing
round-section headings (1..N -> 2..N+1). The two trailing static
sections (Standard skips / Regression requirements) keep their legacy
numbers, exactly as the file already does today."""
import re
import sys
from pathlib import Path

DOC = Path("/home/z/my-project/pharma-ml-toolkit/docs/QA/TEST_EXECUTION_REPORT.md")
NEW_SECTION = Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else None

if NEW_SECTION is None:
    sys.exit("usage: renumber_qa_r365.py <new-section-file.md>")

text = DOC.read_text(encoding="utf-8")
lines = text.splitlines()

# 1. cascade renumber: bump the FIRST-appearing numbered round headings
#    (1..15) that sit before the trailing static pair. The trailing
#    statics keep their numbers; identify them by their titles.
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

# 2. insert the new section right before the (now) "## 2. Executed this
#    round (r364 ..." heading — i.e. the first round heading in the file.
m = re.search(r"^## 2\. Executed this round \(r364", text, re.MULTILINE)
assert m, "the r364 section heading (now ## 2.) not found"
text = text[:m.start()] + NEW_SECTION.rstrip() + "\n\n" + text[m.start():]

DOC.write_text(text, encoding="utf-8")
print("renumbered + inserted; headings now:")
for h in re.findall(r"^## \d+\. .{0,60}", text, re.MULTILINE)[:6]:
    print(" ", h)
