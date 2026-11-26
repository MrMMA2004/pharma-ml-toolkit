#!/usr/bin/env python3
"""The standalone-server daemon: same double-fork pattern as
scratch/dev_daemon.py but serving the PRODUCTION standalone build
(no on-demand Turbopack compiles -- immune to the dev-cache wedging).
Usage: python3 scratch/standalone_daemon.py start|stop
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "standalone.log"


def _already_up() -> bool:
    probe = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "--max-time", "3", "http://localhost:3000/"],
        capture_output=True, text=True, check=False,
    )
    return probe.stdout.strip() == "200"


def start() -> int:
    if _already_up():
        print("standalone: already up (GET / 200)")
        return 0
    if os.fork() > 0:
        print("standalone: forked (grandchild reparents to PID 1)")
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    os.chdir(REPO)
    log = open(LOG, "ab", buffering=0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    env = dict(os.environ, NODE_ENV="production")
    subprocess.run(["bun", ".next/standalone/server.js"], env=env)
    os._exit(0)


def stop() -> int:
    killed = 0
    for _ in range(4):
        r = subprocess.run(
            ["pkill", "-f", "standalone/server.js"],
            capture_output=True,
        )
        if r.returncode != 0:
            break
        killed += 1
        time.sleep(1)
    print(f"standalone: stopped ({killed} SIGTERM round(s))")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "start":
        sys.exit(start())
    if cmd == "stop":
        sys.exit(stop())
    print("usage: standalone_daemon.py start|stop")
    sys.exit(2)
