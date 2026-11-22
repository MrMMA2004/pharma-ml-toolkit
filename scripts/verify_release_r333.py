#!/usr/bin/env python3
"""r333 housekeeping: verify the v1.2.240 release + CI verdict via GitHub API.
Never prints the token."""
import json
import urllib.request
from pathlib import Path

TOK = Path("/home/z/my-project/.credentials/github_pat").read_text().strip()
BASE = "https://api.github.com/repos/MrMMA2004/pharma-ml-toolkit"


def api(path):
    req = urllib.request.Request(
        BASE + path,
        headers={"Authorization": f"Bearer {TOK}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "pharma-ml-toolkit-housekeeping"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


run = api("/actions/runs/45047077848")
print(f"RUN 45047077848: status={run['status']} conclusion={run['conclusion']} "
      f"head_sha={run['head_sha'][:8]} event={run['event']}")

rel = api("/releases/tags/v1.2.240")
print(f"RELEASE v1.2.240: name={rel.get('name')!r} draft={rel['draft']} "
      f"assets={len(rel['assets'])}")
for a in rel["assets"]:
    print(f"  asset: {a['name']}  {a['size']:,} B  downloads={a['download_count']}")

checks = api("/commits/8d92e9e0/check-runs?per_page=100")
runs = checks.get("check_runs", [])
concl = {}
for c in runs:
    concl.setdefault(c.get("conclusion") or c.get("status"), 0)
    concl[c.get("conclusion") or c.get("status")] += 1
print(f"CHECK-RUNS at 8d92e9e0: total={checks.get('total_count')} "
      f"breakdown={concl}")

branches = api("/branches?per_page=100")
names = [b["name"] for b in branches]
arena = [n for n in names if n.startswith("arena/")]
print(f"BRANCHES: {len(names)} total, arena refs: {arena or 'none'}")
print(f"MAIN HEAD: {next((b['commit']['sha'][:8] for b in branches if b['name']=='main'), '?')}")
