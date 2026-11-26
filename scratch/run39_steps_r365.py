#!/usr/bin/env python3
"""Inspect jobs/steps of Windows Release run 45864554708 (token never printed)."""
import json
import subprocess

TOKEN = open("/home/z/my-project/.credentials/github_pat").read().strip()
H = ["-H", f"Authorization: Bearer {TOKEN}", "-H", "Accept: application/vnd.github+json"]
REPO = "MrMMA2004/pharma-ml-toolkit"

r = subprocess.run(
    ["curl", "-sS", *H, f"https://api.github.com/repos/{REPO}/actions/runs/45864554708/jobs"],
    capture_output=True, text=True, timeout=60,
)
data = json.loads(r.stdout)
for job in data.get("jobs", []):
    print(f"job: {job['name']}  status={job['status']}  conclusion={job['conclusion']}  started={job.get('started_at')}")
    for step in job.get("steps", []):
        mark = {"queued": " ", "in_progress": ">", "completed": ("OK " if step["conclusion"] == "success" else "XX")}.get(step["status"], "?")
        print(f"  {mark} step {step['number']:>2}: {step['name']} [{step['status']}/{step['conclusion']}]")
