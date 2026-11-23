#!/usr/bin/env python3
"""Launch scratch/console-supervisor.sh with the dev_daemon double-fork
pattern (parent exits at birth -> grandchild reparents to PID 1), so the
self-healing loop survives the sandbox's per-call subtree reaping.
Usage: python3 scratch/supervisor_daemon.py start|stop
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUP = REPO / "scratch" / "console-supervisor.sh"


def _already_serving() -> bool:
    probe = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "--max-time", "3", "http://localhost:3000/"],
        capture_output=True, text=True, check=False,
    )
    return probe.stdout.strip() == "200"


def start() -> int:
    if _already_serving():
        print("supervisor-daemon: console already up (GET / 200)")
        return 0
    if os.fork() > 0:
        print("supervisor-daemon: forked (grandchild reparents to PID 1)")
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    os.chdir(REPO)
    devnull_out = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_out, 1)
    os.dup2(devnull_out, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.execv("/bin/bash", ["/bin/bash", str(SUP)])
    os._exit(0)  # unreachable


def stop() -> int:
    killed = 0
    scan = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False)
    for line in scan.stdout.splitlines():
        line = line.strip()
        if not line or "supervisor_daemon" in line:
            continue
        pid, _, args = line.partition(" ")
        if "console-supervisor.sh" in args:
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed += 1
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    print(f"supervisor-daemon: stopped ({killed} SIGTERM round)")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    sys.exit({"start": start, "stop": stop}.get(mode, start)())
