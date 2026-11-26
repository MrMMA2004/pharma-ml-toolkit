#!/usr/bin/env bash
# Rebuild ui/native/libEGL.so.1 (the headless-fallback EGL stub).
#
# Why this exists (round 206): PySide6's QtGui links libEGL.so.1 at load
# time even when the offscreen platform plugin (software rasterization,
# no real GL context) is used. Some hardened headless sandboxes ship
# libGL but not libEGL, which makes `import PySide6.QtWidgets` die with
# "libEGL.so.1: cannot open shared object file". The stub provides the
# EGL API surface as no-op symbols so the dynamic linker is satisfied;
# the offscreen platform never actually issues GL calls through them.
# On a normal desktop (real EGL present) the stub is never loaded --
# ui/headless.py probes for the real library first.
#
# The compiled stub is committed beside this script so rollbacks and
# gcc-less environments still work; rerun this only when eglstub.c
# changes. Requires: gcc, x86-64 Linux.
set -euo pipefail
cd "$(dirname "$0")"
gcc -shared -fPIC -Wl,-soname,libEGL.so.1 -o libEGL.so.1 eglstub.c
echo "rebuilt $(pwd)/libEGL.so.1"
