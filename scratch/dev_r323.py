#!/usr/bin/env python3
"""round-323 browser-E2E daemons: the fd-clean double-forked FastAPI
backend on 8017 + the Vite dev server on 5174 (the r311/r314/r321
pattern — the dev-same-origin token channel)."""
import os
import sys

PY = "/home/z/.venv/bin/python"
BUN = "/usr/local/bin/bun"
TOOLKIT = "/home/z/my-project/pharma-ml-toolkit"
TOKEN = "r323-browser-token"


def daemon(argv, cwd, env, logpath):
    pid = os.fork()
    if pid:
        return
    os.setsid()
    if os.fork():
        os._exit(0)
    log = open(logpath, "ab", buffering=0)
    devnull = open(os.devnull, "rb")
    os.chdir(cwd)
    os.dup2(devnull.fileno(), 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.execve(argv[0], argv, env)
    os._exit(127)


env = os.environ.copy()
daemon(
    [PY, "-c",
     "import asyncio; from pharmaml.api.server import serve; "
     f"asyncio.run(serve(token='{TOKEN}', port=8017))"],
    TOOLKIT, env, "/tmp/r323_backend.log",
)
import time
time.sleep(5)
fe_env = env.copy()
fe_env["PHARMAML_DEV_BACKEND_PORT"] = "8017"
fe_env["VITE_PHARMAML_TOKEN"] = TOKEN
daemon(
    [BUN, "run", "dev"],
    f"{TOOLKIT}/frontend", fe_env, "/tmp/r323_vite.log",
)
print("launched r323 daemons: backend 8017 + vite 5174", flush=True)
os._exit(0)
