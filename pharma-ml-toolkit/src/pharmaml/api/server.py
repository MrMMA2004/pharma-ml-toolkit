"""The loopback uvicorn runner with the structured startup handshake
(part10 Phase Seven §10.1 steps 5-8).

Startup contract (machine-readable, **separate from diagnostic logs**):

* the server owns a socket bound to ``127.0.0.1`` with port ``0`` —
  the OS assigns a free port atomically inside ``bind()`` (no
  hardcoded port, no check-then-bind race: the "controlled port-0
  binding" the spec asks for);
* the handshake line is written as soon as ``listen()`` returns —
  the port is real and accepting connections BEFORE uvicorn even
  starts — exactly ONE machine-readable line on stdout::

      PHARMAML-READY {"port": 43551, "token": "<session-token>",
                      "pid": 1234, "version": "1.2.218"}

  The WinUI host parses it, validates the handshake, then
  authenticates every REST/WS call with the token.  Diagnostics go
  to stderr only — stdout stays machine-clean.
* ``SIGTERM``/``SIGINT`` run FastAPI's shutdown handlers (the
  JobManager drains, the bus publishes ``shutdown``) and the
  process exits — the graceful path; forced termination stays the
  documented parent-side fallback only.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from typing import Any

import uvicorn

from . import create_app
from .persistence import default_data_dir
from .security import session_security

HANDSHAKE_PREFIX = "PHARMAML-READY"


def handshake_line(port: int, token: str, pid: int,
                    version: str) -> str:
    """The single machine-readable startup line (stdout, once)."""
    payload: dict[str, Any] = {
        "port": port,
        "token": token,
        "pid": pid,
        "version": version,
    }
    return f"{HANDSHAKE_PREFIX} {json.dumps(payload)}"


def bind_loopback_socket(host: str = "127.0.0.1",
                         port: int = 0) -> socket.socket:
    """Bind + listen on a loopback socket (port 0 = OS-assigned)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(128)
    return sock


async def serve(
    token: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    stdout=None,
) -> int:
    """Bind loopback:0, emit the handshake, serve until signalled.

    The shipped host entry is **durable** (part11 Phase Eleven): it
    always passes the platform user-data directory, so the registry
    (datasets / models / priors / artifacts / job history /
    preferences) survives restarts in SQLite + the file vault.

    Returns the *actual* bound port (tests and the Tauri host both
    need it; the handshake line carries it too).
    """
    import os

    out = stdout or sys.stdout
    app = create_app(token=token, data_dir=default_data_dir())
    secure = session_security()

    sock = bind_loopback_socket(host, port)
    real_port = sock.getsockname()[1]
    # the socket is already listening: emit the handshake NOW —
    # nothing later on stdout is machine-readable
    line = handshake_line(real_port, secure.token, os.getpid(),
                          _version())
    out.write(line + "\n")
    out.flush()

    config = uvicorn.Config(
        app,
        fd=sock.fileno(),
        log_level="warning",  # diagnostics on stderr only
        access_log=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()
    sock.close()
    return real_port


def _version() -> str:
    import pharmaml

    return pharmaml.__version__


def main() -> None:  # pragma: no cover - the WinUI-hosted entry
    """Console entry: ``python -m pharmaml.api``."""
    asyncio.run(serve())


if __name__ == "__main__":  # pragma: no cover
    main()
