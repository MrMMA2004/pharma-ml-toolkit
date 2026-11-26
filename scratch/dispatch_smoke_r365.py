#!/usr/bin/env python3
"""Dispatch windows-install-smoke on the published v1.2.242 release
(tag=v1.2.242, previous_tag=v1.2.241) and report the new run id.
Token used only in headers, never printed."""
import json
import subprocess
import sys
import time

TOKEN = open("/home/z/my-project/.credentials/github_pat").read().strip()
H = ["-H", f"Authorization: Bearer {TOKEN}", "-H", "Accept: application/vnd.github+json"]
REPO = "MrMMA2004/pharma-ml-toolkit"

r = subprocess.run(
    ["curl", "-sS", "-X", "POST", *H,
     "-H", "Accept: application/vnd.github+json",
     f"https://api.github.com/repos/{REPO}/actions/workflows/windows-install-smoke.yml/dispatches",
     "-d", json.dumps({
         "ref": "main",
         "inputs": {"tag": "v1.2.242", "previous_tag": "v1.2.241"},
     })],
    capture_output=True, text=True, timeout=60,
)
print("dispatch http:", r.returncode, r.stdout[:200] or "(empty body = accepted)")

# find the newly created run
for _ in range(6):
    time.sleep(10)
    runs = json.loads(subprocess.run(
        ["curl", "-sS", *H,
         f"https://api.github.com/repos/{REPO}/actions/workflows/windows-install-smoke.yml/runs?per_page=2"],
        capture_output=True, text=True, timeout=60,
    ).stdout)
    for w in runs.get("workflow_runs", []):
        print(f"run {w['id']}: status={w['status']} conclusion={w['conclusion']} created={w['created_at']}")
    if runs.get("workflow_runs"):
        break
