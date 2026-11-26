#!/usr/bin/env python3
"""r365 continuation: poll the third Windows Release build (run 45856387513),
the release v1.2.242 assets, and the parallel CI run. Token never printed."""
import json
import subprocess
import sys

TOKEN = open("/home/z/my-project/.credentials/github_pat").read().strip()
H = ["-H", f"Authorization: Bearer {TOKEN}", "-H", "Accept: application/vnd.github+json"]
REPO = "MrMMA2004/pharma-ml-toolkit"


def api(path):
    r = subprocess.run(
        ["curl", "-sS", *H, f"https://api.github.com/repos/{REPO}/{path}"],
        capture_output=True, text=True, timeout=60,
    )
    return json.loads(r.stdout)


def main():
    out = {}
    # third release build (run 39) / fourth release build (run 40)
    run = api("actions/runs/45864554708")
    if "id" in run:
        out["run40"] = {
            "name": run.get("name"), "status": run.get("status"),
            "conclusion": run.get("conclusion"), "head_sha": run.get("head_sha", "")[:8],
            "html_url": run.get("html_url"),
        }
    else:
        out["run40"] = {"error": run.get("message", "unknown")}
    # parallel CI at HEAD
    runs = api("actions/runs?per_page=6")
    if "workflow_runs" in runs:
        out["recent_runs"] = [
            {
                "id": w["id"], "name": w.get("name"), "status": w.get("status"),
                "conclusion": w.get("conclusion"), "head_sha": w.get("head_sha", "")[:8],
                "event": w.get("event"),
            }
            for w in runs["workflow_runs"]
        ]
    # release
    rel = api("releases/tags/v1.2.242")
    if "tag_name" in rel:
        out["release"] = {
            "tag": rel.get("tag_name"), "draft": rel.get("draft"),
            "published_at": rel.get("published_at"),
            "assets": [
                {"name": a["name"], "size": a["size"]}
                for a in rel.get("assets", [])
            ],
        }
    else:
        out["release"] = {"absent_or_error": rel.get("message", "unknown")}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
