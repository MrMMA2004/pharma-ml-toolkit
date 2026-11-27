#!/usr/bin/env python3
"""r371 close-out poller: watch the tag-triggered workflows and the
v1.2.244 release. Reads the PAT at runtime from the 600-mode credential
file; NEVER prints the token. Masked diagnostics only."""
import json
import sys
import time
import urllib.request
from pathlib import Path

TOK = Path("/home/z/my-project/.credentials/github_pat").read_text().strip()
BASE = "https://api.github.com/repos/MrMMA2004/pharma-ml-toolkit"


def api(path: str) -> object:
    req = urllib.request.Request(
        f"{BASE}{path}",
        headers={"Authorization": f"Bearer {TOK}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "pharma-ml-toolkit-r371"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def workflow_runs(name_filter: str, per_page: int = 6) -> list:
    runs = api(f"/actions/runs?per_page={per_page * 3}").get("workflow_runs", [])
    return [r for r in runs if name_filter in (r.get("name") or "")
            or name_filter in (r.get("path") or "")][:per_page]


def main() -> int:
    ship = "1a54153a"
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"

    if mode == "release":
        rel = api("/releases/tags/v1.2.244")
        assets = rel.get("assets", [])
        print(f"release v1.2.244: draft={rel.get('draft')} "
              f"published_at={rel.get('published_at')}")
        for a in assets:
            print(f"  asset {a['name']}  {a['size']} B  "
                  f"downloads={a['download_count']}  state={a['state']}")
        if not assets:
            print("  NO ASSETS YET")
        return 0

    if mode == "runs":
        for path in ("windows-release.yml", "windows-install-smoke.yml",
                     "penetration-test.yml", "ci.yml"):
            for r in workflow_runs(path.split("/")[-1].replace(".yml", ""),
                                   per_page=2):
                print(f"{path}: id={r['id']} head_sha={r['head_sha'][:8]} "
                      f"status={r['status']} conclusion={r['conclusion']} "
                      f"event={r['event']}")
        return 0

    # poll mode: wait until the release workflow finishes (max N minutes)
    deadline = time.time() + 45 * 60
    while time.time() < deadline:
        runs = [r for r in api("/actions/runs?per_page=30").get("workflow_runs", [])
                if r.get("head_sha", "").startswith(ship)]
        rel_wf = [r for r in runs if "windows-release" in (r.get("path") or "")]
        done = [r for r in rel_wf if r["status"] == "completed"]
        if rel_wf:
            r = rel_wf[0]
            print(f"poll: windows-release status={r['status']} "
                  f"conclusion={r['conclusion']}")
            if done:
                return 0 if r["conclusion"] == "success" else 1
        else:
            print("poll: no windows-release run for the ship SHA yet")
        time.sleep(90)
    print("poll: TIMED OUT waiting for the release workflow")
    return 2


if __name__ == "__main__":
    sys.exit(main())
