#!/usr/bin/env python3
"""Capture the install-smoke run 45869621772's step ladder (token masked)."""
import json
import subprocess

TOKEN = open("/home/z/my-project/.credentials/github_pat").read().strip()
H = ["-H", f"Authorization: Bearer {TOKEN}", "-H", "Accept: application/vnd.github+json"]

d = json.loads(subprocess.run(
    ["curl", "-sS", *H,
     "https://api.github.com/repos/MrMMA2004/pharma-ml-toolkit/actions/runs/45869621772/jobs"],
    capture_output=True, text=True, timeout=60,
).stdout)
for j in d.get("jobs", []):
    print(f"job: {j['name']} -> {j['conclusion']}")
    for s in j.get("steps", []):
        if s["conclusion"] in ("success", "failure", "skipped") and not s["name"].startswith("Post"):
            print(f"  {s['number']:>2}. {s['name'][:70]} [{s['conclusion']}]")
