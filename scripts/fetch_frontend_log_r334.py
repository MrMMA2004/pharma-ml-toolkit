#!/usr/bin/env python3
"""Fetch the failing Studio frontend job log (redirect-safe: drop auth on redirect)."""
import urllib.request
from pathlib import Path

TOK = Path("/home/z/my-project/.credentials/github_pat").read_text().strip()
JOB = 137312180841


class NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        req.headers.pop("Authorization", None)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


opener = urllib.request.build_opener(NoAuthRedirect)
req = urllib.request.Request(
    f"https://api.github.com/repos/MrMMA2004/pharma-ml-toolkit/actions/jobs/{JOB}/logs",
    headers={"Authorization": f"Bearer {TOK}",
             "Accept": "application/vnd.github+json",
             "User-Agent": "pharma-ml-toolkit-housekeeping"})
with opener.open(req, timeout=60) as r:
    log = r.read().decode(errors="replace")

Path("/home/z/my-project/scratch/p14_r334_frontend_fail_log.txt").write_text(log)
lines = log.splitlines()
print(f"log: {len(lines)} lines, saved to scratch/p14_r334_frontend_fail_log.txt")
# error-centric excerpt
for i, ln in enumerate(lines):
    low = ln.lower()
    if ("error" in low or "fail" in low or "✖" in ln or "exit code" in low) \
            and "0 errors" not in low and "no error" not in low:
        print(f"{i:5d}| {ln[:220]}")
