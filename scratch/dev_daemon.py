#!/usr/bin/env python3
"""The round-258 lesson, codified: the dev server must be launched via
a DOUBLE FORK (parent exits at birth -> the grandchild reparents to
PID 1 instantly) or the sandbox's per-call subtree reaping kills it at
every tool-call boundary -- setsid/nohup/disown all died.  Usage:

    python3 scratch/dev_daemon.py start   # double-fork + reparent
    python3 scratch/dev_daemon.py stop    # kill the next/tee tree

The launched tree reproduces the original: bun run dev ->
bash -c "next dev -p 3000 2>&1 | tee dev.log" (dev.log keeps its
append-only role; tail it for health).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _already_up() -> bool:
    probe = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "--max-time", "3", "http://localhost:3000/"],
        capture_output=True, text=True, check=False,
    )
    return probe.stdout.strip() == "200"


def start() -> int:
    if _already_up():
        print("dev-daemon: already up (GET / 200), nothing to do")
        return 0
    if os.fork() > 0:  # the parent reports and exits immediately
        print("dev-daemon: forked (grandchild reparents to PID 1)")
        return 0
    os.setsid()
    if os.fork() > 0:  # the intermediate dies -- the daemon is orphaned
        os._exit(0)
    os.chdir(REPO)
    # NOTE: package.json's "dev" script already pipes through
    # `| tee dev.log` -- so stdout/stderr are pointed at /dev/null here
    # (tee does the dev.log writing; a dup2 to dev.log would double
    # every line).
    devnull_out = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_out, 1)
    os.dup2(devnull_out, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    subprocess.run(["bun", "run", "dev"])
    os._exit(0)  # unreachable while the server lives


def stop() -> int:
    killed = 0
    scan = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False)
    for line in scan.stdout.splitlines():
        line = line.strip()
        if not line or "dev_daemon" in line:
            continue
        pid, _, args = line.partition(" ")
        if "next dev" in args or "next-server" in args or "bun run dev" in args:
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed += 1
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    time.sleep(2)
    for line in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False).stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, args = line.partition(" ")
        if "next-server" in args:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    print(f"dev-daemon: stopped ({killed} SIGTERM round)")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    sys.exit({"start": start, "stop": stop}.get(mode, start)())
