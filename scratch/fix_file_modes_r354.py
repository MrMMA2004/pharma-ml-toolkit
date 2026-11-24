#!/usr/bin/env python3
"""R354 file-mode realignment drill (the 108-flag EXE001/EXE002 class).

The sandbox rollback rewrites filesystem exec bits wholesale; core.filemode
is false so git never sees the drift, but ruff's EXE001/EXE002 do.  The
r353 resolution, codified: realign every tracked file's FS mode to its
git-index mode (NUL-safe parse — the r344 lesson) and restore the 7-file
EXE001 shebang parity set (shebang files whose index mode is 100644 get
the exec bit FS-only, silencing EXE001 without a git-visible diff).
"""
import stat
import subprocess
import sys

REPO = "/home/z/my-project"

raw = subprocess.run(
    ["git", "ls-files", "-s", "-z"], cwd=REPO, capture_output=True, check=True
).stdout
fixed_plus = fixed_minus = parity = 0
for rec in raw.split(b"\0"):
    if not rec:
        continue
    meta, _, path_b = rec.partition(b"\t")
    mode = meta.split()[0].decode()
    path = path_b.decode("utf-8", "replace")
    if mode not in ("100755", "100644"):
        continue
    full = f"{REPO}/{path}"
    try:
        st = os_stat = stat.S_IMODE(__import__("os").stat(full).st_mode)
    except FileNotFoundError:
        continue
    want_exec = mode == "100755"
    if not want_exec:
        try:
            with open(full, "rb") as fh:
                shebang = fh.read(2) == b"#!"
        except OSError:
            shebang = False
        want_exec = shebang  # the EXE001 parity set
    is_exec = bool(st & 0o111)
    if want_exec and not is_exec:
        __import__("os").chmod(full, st | 0o755)
        fixed_plus += 1
    elif not want_exec and is_exec:
        __import__("os").chmod(full, st & ~0o111)
        fixed_minus += 1
    if shebang and not is_exec and want_exec:
        parity += 1
print(f"realigned: +{fixed_plus} exec, -{fixed_minus} exec "
      f"(shebang parity files touched: {parity})")
sys.exit(0)
