#!/usr/bin/env python3
"""Fetch the failing job log for run 45856387513 and show step 20's failure region.
Token used only in headers, never printed; log is grep-filtered."""
import json
import re
import subprocess
import sys

TOKEN = open("/home/z/my-project/.credentials/github_pat").read().strip()
H = ["-H", f"Authorization: Bearer {TOKEN}", "-H", "Accept: application/vnd.github+json"]
REPO = "MrMMA2004/pharma-ml-toolkit"
RUN = "45856387513"

jobs = json.loads(subprocess.run(
    ["curl", "-sS", *H, f"https://api.github.com/repos/{REPO}/actions/runs/{RUN}/jobs"],
    capture_output=True, text=True, timeout=60,
).stdout)

job_id = jobs["jobs"][0]["id"]
log = subprocess.run(
    ["curl", "-sSL", *H, f"https://api.github.com/repos/{REPO}/actions/jobs/{job_id}/logs"],
    capture_output=True, text=True, timeout=120,
).stdout

lines = log.splitlines()
# find the step-20 section and any error-ish lines
err_idx = [i for i, l in enumerate(lines) if "Validate the installer artifacts" in l and "##[group]" in l]
start = err_idx[-1] if err_idx else max(0, len(lines) - 400)
section = lines[start:]
pat = re.compile(r"error|fail|mismatch|missing|assert|not found|absent|✗|INVALID|unexpected", re.I)
hits = [l for l in section if pat.search(l) and "0 errors" not in l]
print(f"log lines: {len(lines)}, step-20 section: {len(section)}, error hits: {len(hits)}")
for l in hits[:60]:
    # strip timestamps like 2026-11-26T16:44:52.7Z
    print(re.sub(r"^\S*Z\s", "", l)[:240])
