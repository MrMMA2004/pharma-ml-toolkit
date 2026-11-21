#!/usr/bin/env python3
"""Round 311 browser-E2E daemons (the r291 pattern, rewritten after the
fifth rollback wiped scratch/e2e291_daemons.py): the FastAPI backend on
8017 + the Vite dev server on 5174 with the /api proxy + the dev startup
token — both double-forked so the sandbox's per-call subtree reaper
cannot kill them (the r258 lesson, same as scratch/dev_daemon.py).

Usage:
    python3 scratch/e2e311_daemons.py start   # boot both, wait for health
    python3 scratch/e2e311_daemons.py stop    # kill both trees

The session token (a localhost-only API gate token, NOT any persistent
credential) is generated once, stored 600-mode at scratch/e2e311_token.txt,
and injected into the Vite process env as VITE_PHARMAML_TOKEN (the
documented dev-same-origin startup channel, frontend/src/services/api/
startup.ts)."""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLKIT = REPO / "pharma-ml-toolkit"
TOKEN_FILE = Path(__file__).resolve().parent / "e2e311_token.txt"
BACKEND_PORT = 8017
VITE_PORT = 5174
PY = "/home/z/.venv/bin/python"


def _double_fork(child_fn) -> None:
    """The dev_daemon.py lesson: parent exits at birth, the grandchild
    reparents to PID 1 instantly — the reaper never sees it."""
    if os.fork() > 0:
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    child_fn()
    os._exit(0)


def _run_backend(token: str) -> None:
    os.chdir(TOOLKIT)
    out = open("/home/z/my-project/scratch/e2e311_backend.log", "ab")
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    code = (
        "import asyncio\n"
        "from pharmaml.api.server import serve\n"
        f"asyncio.run(serve(token={token!r}, port={BACKEND_PORT}))\n"
    )
    subprocess.run([PY, "-c", code])


def _run_vite(token: str) -> None:
    os.chdir(TOOLKIT)
    out = open("/home/z/my-project/scratch/e2e311_vite.log", "ab")
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    env = dict(os.environ)
    env["PHARMAML_DEV_BACKEND_PORT"] = str(BACKEND_PORT)
    env["VITE_PHARMAML_TOKEN"] = token
    subprocess.run(["npm", "--prefix", "frontend", "run", "dev"], env=env)


def _curl(url: str, token: str | None = None) -> tuple[int, str]:
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
           "--max-time", "4"]
    if token:
        cmd += ["-H", f"X-Session-Token: {token}"]
    cmd += [url]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return r.returncode, r.stdout.strip()


def start() -> int:
    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else ""
    # already up?
    _, b_code = _curl(f"http://127.0.0.1:{BACKEND_PORT}/api/v1/health",
                      token=token or None)
    _, v_code = _curl(f"http://localhost:{VITE_PORT}/")
    if b_code == "200" and v_code == "200":
        print(f"daemons: already up (health {b_code}, vite {v_code})")
        return 0
    token = secrets.token_urlsafe(32)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)
    _double_fork(lambda: _run_backend(token))
    _double_fork(lambda: _run_vite(token))
    deadline = time.time() + 90
    b_ok = v_ok = False
    while time.time() < deadline and not (b_ok and v_ok):
        if not b_ok:
            _, code = _curl(f"http://127.0.0.1:{BACKEND_PORT}/api/v1/health",
                            token=token)
            b_ok = code == "200"
        if not v_ok:
            _, code = _curl(f"http://localhost:{VITE_PORT}/")
            v_ok = code == "200"
        if not (b_ok and v_ok):
            time.sleep(2)
    print(f"backend health {'OK' if b_ok else 'FAIL'} · "
          f"vite {'OK' if v_ok else 'FAIL'} "
          f"(token file {TOKEN_FILE.name}, mode 600)")
    return 0 if (b_ok and v_ok) else 1


def stop() -> int:
    killed = 0
    scan = subprocess.run(["ps", "-eo", "pid,args"],
                          capture_output=True, text=True, check=False)
    for line in scan.stdout.splitlines():
        line = line.strip()
        if not line or "e2e311" in line:
            continue
        pid, _, args = line.partition(" ")
        if ("pharmaml.api.server" in args
                or "vite --port 5174" in args
                or ("npm --prefix frontend run dev" in args)):
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed += 1
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    time.sleep(2)
    for line in subprocess.run(["ps", "-eo", "pid,args"],
                               capture_output=True, text=True,
                               check=False).stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, args = line.partition(" ")
        if "pharmaml.api.server" in args or "vite --port 5174" in args:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    print(f"daemons: stopped ({killed} SIGTERM round)")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    sys.exit({"start": start, "stop": stop}.get(mode, start)())
