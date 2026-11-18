"""The PharmaML Studio backend-sidecar entry (part11/part11_2 §5).

This is the entry the Windows PyInstaller spec freezes for the Tauri
2 sidecar: the FastAPI backend alone — NOT the retired-era PySide6
GUI (``studio.py`` stays the legacy entry until the part11 §23 parity
gate retires it; the production installer must not bundle the Qt
stack with the backend sidecar).

Frozen behavior (mirrors ``python -m pharmaml.api``):

* binds loopback only, on a port-0 socket the server itself owns;
* prints the single machine-readable ``PHARMAML-READY {port, token,
  pid, version}`` line on stdout — the Tauri host's ``backend_manager``
  parses exactly this line and forwards the endpoint to the WebView
  through the protected IPC bridge;
* diagnostics go to stderr only (stdout stays machine-clean);
* SIGTERM/CTRL_CLOSE_EVENT runs the graceful shutdown (the shutdown
  envelope reaches connected WebSocket clients before the sockets
  close — pinned by ``TestGracefulShutdownPush``).

``multiprocessing.freeze_support()`` runs first: the docking pool is
spawn-context and the JobManager's process workers must not re-enter
the server startup on Windows spawn (part9 §13's rule, preserved).
"""

from __future__ import annotations

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from pharmaml.api.server import main

    main()
