#!/usr/bin/env python3
"""The r291 browser-verification daemons (the r289/r290 pattern).

Double-fork BOTH servers so they survive the sandbox's per-call
subtree reaping (plain `&` dies at the tool-call boundary):

1. the FastAPI backend on a FIXED loopback port (8017) with a FIXED
   session token, stdout -> scratch/e2e291-backend.log;
2. the vite dev server (5174) with PHARMAML_DEV_BACKEND_PORT=8017 +
   VITE_PHARMAML_TOKEN so the frontend's dev-same-origin lane rides
   the real backend through the vite proxy (the app boots ITSELF).

    python3 scratch/e2e291_daemons.py start|stop
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONTEND = REPO / "frontend"
BACKEND_LOG = REPO / "scratch" / "e2e291-backend.log"
VITE_LOG = REPO / "scratch" / "e2e291-vite.log"
TOKEN = "e2e-r291-session-token-0123456789"
PORT = 8017
VITE_PORT = 5174
VENV_PY = "/home/z/.venv/bin/python3"


def _forked(child: "function") -> None:  # type: ignore[name-defined]
    if os.fork() > 0:
        return  # the parent returns immediately
    os.setsid()
    if os.fork() > 0:
        os._exit(0)  # the intermediate dies — the daemon reparents to 1
    child()
    os._exit(0)


def _backend() -> None:
    os.chdir(REPO)
    log = os.open(BACKEND_LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    subprocess.run(
        [
            VENV_PY,
            "-c",
            (
                "import asyncio\n"
                "from pharmaml.api.server import serve\n"
                f"asyncio.run(serve(token={TOKEN!r}, port={PORT}))\n"
            ),
        ]
    )


def _vite() -> None:
    os.chdir(FRONTEND)
    log = os.open(VITE_LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    env = dict(os.environ)
    env["PHARMAML_DEV_BACKEND_PORT"] = str(PORT)
    env["VITE_PHARMAML_TOKEN"] = TOKEN
    subprocess.run(["bun", "run", "dev"], env=env)


def start() -> int:
    backend_up = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "--max-time", "3", f"http://127.0.0.1:{PORT}/api/v1/health"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if backend_up in ("200", "401"):
        print(f"e2e291: backend already up on {PORT}")
    else:
        _forked(_backend)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            code = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "--max-time", "3", f"http://127.0.0.1:{PORT}/api/v1/health"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            if code in ("200", "401"):
                break
            time.sleep(0.5)
        print(f"e2e291: backend daemonized on {PORT} (health {code})")
    vite_up = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "--max-time", "3", f"http://127.0.0.1:{VITE_PORT}/"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if vite_up == "200":
        print(f"e2e291: vite already up on {VITE_PORT}")
    else:
        _forked(_vite)
        deadline = time.monotonic() + 30
        code = ""
        while time.monotonic() < deadline:
            code = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "--max-time", "3", f"http://127.0.0.1:{VITE_PORT}/"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            if code == "200":
                break
            time.sleep(0.5)
        print(f"e2e291: vite daemonized on {VITE_PORT} (GET / {code})")
    return 0


def stop() -> int:
    killed = 0
    scan = subprocess.run(
        ["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False
    )
    for line in scan.stdout.splitlines():
        line = line.strip()
        if not line or "e2e291_daemons" in line:
            continue
        pid, _, args = line.partition(" ")
        if (
            "pharmaml.api.server" in args
            or "vite --port 5174" in args
            or "bun run dev" in args
        ):
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed += 1
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    time.sleep(2)
    print(f"e2e291: stopped ({killed} SIGTERM round)")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    sys.exit({"start": start, "stop": stop}.get(mode, start)())
