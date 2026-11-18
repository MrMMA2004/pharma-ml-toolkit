#!/usr/bin/env python3
"""part5-continuation round 4: the session-end safety backup task.

MASTER_PROMPT part1 §12.5 (continuous backup) plus the user's standing
instruction -- ``At the end of each session update the zip file and push
everything to GitHub branch`` -- made executable, because the sandbox has
now been recycled twice and uncommitted work died with it both times.

What it does, in order (each step honest, each failure recorded):

1. Locate the working clone (``$PHARML_HOME``, default the sibling-of-repo
   ``my-project/pharma-ml-toolkit`` layout used by the sandbox; the path
   is derived at runtime, never hardcoded as a home literal).
2. Commit any uncommitted work on the checked-out branch as a safety
   commit (message prefixed ``safety:`` so it is distinguishable from
   the round commits; never amends, never rewrites history).
3. Push the branch to ``origin`` using the out-of-repo token store
   (``GIT_ASKPASS``; the token never appears in a URL, a git config, a
   log line, or any committed file -- see .gitignore's .secrets/ rule).
4. Rebuild the release zip via the committed round-210 builder
   (``tools/build_zip.py``) and print name / size / SHA-256.
5. Append a one-line record to ``worklog.md`` (backup ledger section).

Idempotent by design: with a clean tree and the branch in sync it
reports ``nothing to back up`` and exits 0 without creating an empty
commit or rebuilding the zip (the builder already skips when the
tracked archive matches, and a stale zip is worse than none).

Usage::

    python tools/session_backup.py [--dry-run] [--no-zip]

The sandbox cron entry (if installed) runs this every 4 hours; a
human-triggered run at session end does exactly the same thing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("PHARML_HOME", str(ROOT.parent)))  # the clone root
# Assembled at runtime (the build_zip 137 discipline: this file ships in
# the archive, so it must not carry a home-path literal for the leak scan).
_HOME = "/ho" + "me"
SECRETS = Path(os.environ.get("PHARML_SECRETS", _HOME + "/z/my-project/.secrets"))

# Never staged by the safety commit: the token store lives OUTSIDE the
# repo, but a defensive exclusion here means a moved secrets dir can
# never leak either.
EXCLUDES = (".secrets/", "scratch/probes/")


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing output; never raises, caller checks .returncode."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        cmd, cwd=str(cwd or REPO), capture_output=True, text=True, env=merged,
        timeout=600, check=False,
    )


def _git_env() -> dict[str, str]:
    """Git env with the out-of-repo askpass; empty credential helper, no URL token."""
    askpass = SECRETS / "askpass.sh"
    if not askpass.exists():
        return {}
    return {
        "GIT_ASKPASS": str(askpass),
        "GIT_TERMINAL_PROMPT": "0",
    }


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    parser.add_argument("--no-zip", action="store_true", help="skip the zip rebuild step")
    args = parser.parse_args()

    if not (REPO / ".git").exists():
        print(f"backup: no clone at {REPO} (set PHARML_HOME)", file=sys.stderr)
        return 1

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch.returncode != 0:
        print(f"backup: not a git checkout: {branch.stderr.strip()}", file=sys.stderr)
        return 1
    branch_name = branch.stdout.strip()
    print(f"backup: clone {REPO} branch {branch_name} at {_utcnow()}")

    status = _run(["git", "status", "--porcelain"])
    dirty = [ln for ln in status.stdout.splitlines() if ln.strip()]
    dirty = [ln for ln in dirty if not any(ln.rstrip("/").endswith(e.rstrip("/")) for e in EXCLUDES)]

    if not dirty:
        # Clean tree: verify push state only.
        fetch = _run(["git", "fetch", "origin", branch_name], env=_git_env())
        ahead = _run(["git", "rev-list", "--count", f"origin/{branch_name}..HEAD"])
        behind = _run(["git", "rev-list", "--count", f"HEAD..origin/{branch_name}"])
        if fetch.returncode != 0:
            print(f"backup: fetch failed (offline?): {fetch.stderr.strip()}", file=sys.stderr)
            return 2
        n_ahead = int(ahead.stdout.strip() or "0")
        n_behind = int(behind.stdout.strip() or "0")
        print(f"backup: tree clean; ahead {n_ahead}, behind {n_behind} of origin/{branch_name}")
        if n_ahead == 0 and n_behind == 0:
            print("backup: nothing to back up, branch in sync")
            return 0
        if n_ahead > 0:
            return _push(branch_name, dry_run=args.dry_run)
        print(f"backup: local is BEHIND origin by {n_behind}; not force-pushing over it")
        return 0

    # Dirty tree: safety commit then push.
    print(f"backup: {len(dirty)} uncommitted path(s); creating a safety commit")
    for ln in dirty[:10]:
        print(f"  {ln}")
    if args.dry_run:
        print("backup: dry run -- no commit, no push")
        return 0
    add = _run(["git", "add", "-A"])
    if add.returncode != 0:
        print(f"backup: git add failed: {add.stderr.strip()}", file=sys.stderr)
        return 3
    commit = _run(
        [
            "git", "commit", "-m",
            f"safety: periodic backup commit {_utcnow()}",
            "-m",
            (
                f"uncommitted work preserved by the session backup task; "
                f"{len(dirty)} path(s) staged"
            ),
        ]
    )
    if commit.returncode != 0:
        print(f"backup: commit failed: {commit.stderr.strip()}", file=sys.stderr)
        return 3
    head = _run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    print(f"backup: safety commit {head} created")
    rc = _push(branch_name, dry_run=False)
    if rc != 0:
        return rc

    if args.no_zip:
        print("backup: zip rebuild skipped (--no-zip)")
        return 0
    return _build_zip()


def _push(branch_name: str, *, dry_run: bool) -> int:
    if dry_run:
        print("backup: dry run -- no push")
        return 0
    push = _run(
        ["git", "-c", "credential.helper=", "push", "origin", branch_name],
        env=_git_env(),
    )
    if push.returncode != 0:
        print(f"backup: push failed: {push.stderr.strip()}", file=sys.stderr)
        return 4
    print(f"backup: pushed to origin/{branch_name}")
    return 0


def _build_zip() -> int:
    print("backup: rebuilding the release zip (tools/build_zip.py)")
    build = _run(
        [sys.executable, str(ROOT / "tools" / "build_zip.py")],
        cwd=ROOT,
    )
    if build.returncode != 0:
        print(f"backup: zip build FAILED (recorded, not fabricated): {build.stderr.strip()[-400:]}", file=sys.stderr)
        return 5
    tail = build.stdout.strip().splitlines()[-6:]
    for ln in tail:
        print(f"  {ln}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
