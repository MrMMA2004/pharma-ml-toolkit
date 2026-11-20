#!/usr/bin/env python3
"""round-303 vite daemon (the backend stands; vite needs the abs path)."""
import os
import sys

BUN = "/usr/local/bin/bun"
TOOLKIT_FE = "/home/z/my-project/pharma-ml-toolkit/frontend"

if os.fork():
    sys.exit(0)
os.setsid()
if os.fork():
    sys.exit(0)
log = open("/tmp/r303_vite.log", "ab")
os.chdir(TOOLKIT_FE)
env = os.environ.copy()
env["PHARMAML_DEV_BACKEND_PORT"] = "8123"
env["VITE_PHARMAML_TOKEN"] = "r303-browser-token"
os.execve(BUN, [BUN, "run", "dev"], env)
