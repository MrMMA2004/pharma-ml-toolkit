"""Verify the frozen backend sidecar on a Windows build host (part11_2 §4
step 13: "Verify the generated backend executable").

Runs the REAL frozen ``pharmaml-backend.exe`` from the PyInstaller
onedir, waits for the single ``PHARMAML-READY {port, token, pid,
version}`` handshake line (the same wire contract the Tauri host's
backend_manager.rs implements and ``tests/api/test_backend_service.py``
pins), then exercises the live server:

* ``GET /api/v1/health`` with the session token (expect 200);
* the same request WITHOUT the token (expect 401/403 — the guard
  works in the frozen bundle too);
* a graceful SIGTERM-equivalent shutdown (CTRL_BREAK on Windows is
  the subprocess-equivalent of the tested SIGTERM path where the
  shutdown envelope reaches WS clients before close).

Exit code 0 = the sidecar is verified; anything else = a loud,
nonzero failure (§4 step 25).  Usage::

    python packaging/windows/verify_sidecar.py <path-to-onedir>

Never prints the session token (§4: "Do not expose GitHub tokens or
API credentials in build logs" — the session token is redacted).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HANDSHAKE_PREFIX = "PHARMAML-READY "
STARTUP_TIMEOUT_S = 180.0


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: verify_sidecar.py <path-to-pharmaml-backend-onedir>",
            file=sys.stderr,
        )
        return 2
    onedir = Path(sys.argv[1]).resolve()
    if not onedir.is_dir():
        print(f"FAIL: the onedir {onedir} does not exist", file=sys.stderr)
        return 2
    exe = onedir / "pharmaml-backend.exe"
    if not exe.is_file():
        print(f"FAIL: {exe} does not exist", file=sys.stderr)
        return 2

    creationflags = 0
    if os.name == "nt":
        # r297: CREATE_NEW_PROCESS_GROUP, NOT CREATE_NO_WINDOW -- the
        # graceful-shutdown leg needs send_signal(CTRL_BREAK_EVENT) to
        # REACH the child, and that requires a SHARED console. The flag
        # still isolates the parent (the child is its own process group,
        # so the CTRL_BREAK does not hit the verifier itself) -- the
        # canonical Windows pattern for a graceful console child. The
        # previous CREATE_NO_WINDOW gave the sidecar its own hidden
        # console: the CTRL_BREAK was undeliverable, the leg always
        # timed out, and kill() (TerminateProcess -> exit 1) reported a
        # broken sidecar that was actually serving perfectly.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [str(exe)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        ready: dict | None = None
        while time.monotonic() < deadline:
            line = proc.stdout.readline() if proc.stdout else ""
            if line == "" and proc.poll() is not None:
                break
            line = line.strip()
            if line.startswith(HANDSHAKE_PREFIX):
                ready = json.loads(line[len(HANDSHAKE_PREFIX):])
                break
        if ready is None:
            print(
                "FAIL: no PHARMAML-READY line within the startup window",
                file=sys.stderr,
            )
            return 1
        port = int(ready["port"])
        token = str(ready["token"])
        version = str(ready["version"])
        print(f"OK handshake: port={port} pid={ready['pid']} "
              f"version={version} token=<redacted>")
        if str(ready.get("pid")) != str(proc.pid):
            print(
                f"FAIL: handshake pid {ready.get('pid')} != child pid {proc.pid}",
                file=sys.stderr,
            )
            return 1

        base = f"http://127.0.0.1:{port}"
        # 1. the authenticated health probe
        req = urllib.request.Request(
            f"{base}/api/v1/health",
            headers={"X-Session-Token": token},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            health = json.loads(response.read().decode("utf-8"))
        print(f"OK health (authenticated): {json.dumps(health)}")

        # 2. the unauthenticated probe must be refused (the guard works
        #    frozen too — part11 §15)
        try:
            urllib.request.urlopen(f"{base}/api/v1/health", timeout=10)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                print(f"OK guard: unauthenticated probe refused ({exc.code})")
            else:
                print(f"FAIL: unexpected guard status {exc.code}", file=sys.stderr)
                return 1
        else:
            print("FAIL: the unauthenticated probe was NOT refused",
                  file=sys.stderr)
            return 1

        # 3. graceful shutdown: on Windows CTRL_BREAK_EVENT reaches the
        #    child as a console signal; on POSIX SIGTERM rides the tested
        #    graceful path (the shutdown envelope first).
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            proc.terminate()
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
        if code in (0, None):
            print(f"OK shutdown: exit={code}")
        else:
            # r297: the previous shape printed "OK shutdown: exit=1" and
            # THEN failed the run -- the OK line was a lie the step log
            # carried into every diagnosis. The verdict is honest now.
            print(
                f"FAIL: the sidecar exited {code} on the graceful shutdown "
                "(the envelope-first path must exit 0)",
                file=sys.stderr,
            )
            return 1
        print("SIDECAR-VERIFIED")
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
