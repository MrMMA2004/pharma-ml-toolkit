#!/usr/bin/env python3
"""r371: diagnose the failed windows-release run — jobs, steps, log tail.
Token read at runtime from the 600-mode file; never printed."""
import json
import sys
import urllib.request
from pathlib import Path

TOK = Path("/home/z/my-project/.credentials/github_pat").read_text().strip()
BASE = "https://api.github.com/repos/MrMMA2004/pharma-ml-toolkit"


def api(path: str, raw: bool = False):
    req = urllib.request.Request(
        f"{BASE}{path}",
        headers={"Authorization": f"Bearer {TOK}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "pharma-ml-toolkit-r371"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
        return data if raw else json.loads(data.decode())


run_id = sys.argv[1] if len(sys.argv) > 1 else "46035752426"
jobs = api(f"/actions/runs/{run_id}/jobs?per_page=20")["jobs"]
for j in jobs:
    print(f"job {j['id']} {j['name']}: {j['status']}/{j['conclusion']}")
    for s in j.get("steps", []):
        if s["conclusion"] not in ("success", "skipped", None):
            print(f"  FAILED STEP: {s['name']} -> {s['conclusion']}")
        elif s["status"] != "completed":
            print(f"  pending: {s['name']}")
    if j.get("conclusion") == "failure":
        # /actions/jobs/{id}/logs 302s to an azure blob URL that must be
        # fetched WITHOUT the Authorization header (a 401 otherwise).
        req = urllib.request.Request(
            f"{BASE}/actions/jobs/{j['id']}/logs",
            headers={"Authorization": f"Bearer {TOK}",
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "pharma-ml-toolkit-r371"})
        import urllib.error

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        try:
            opener.open(req, timeout=60)
            loc = None
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location") if e.code in (301, 302, 303, 307) else None
        if not loc:
            print("  (no log redirect location)")
        else:
            bare = urllib.request.Request(loc, headers={"User-Agent": "pharma-ml-toolkit-r371"})
            log = urllib.request.urlopen(bare, timeout=120).read().decode("utf-8", "replace")
            lines = log.splitlines()
            errs = [ln for ln in lines if ("error" in ln.lower()
                    or "##[error]" in ln or "FAILED" in ln or "failed" in ln.lower())][-30:]
            print(f"  --- log tail ({len(lines)} lines total) ---")
            for ln in errs:
                print("  " + ln[:220])
