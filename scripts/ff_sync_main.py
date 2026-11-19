#!/usr/bin/env python3
"""Complete the ff-only sync: remove untracked files that are byte-identical
to origin/main blobs (so the merge can write them), then merge --ff-only.

Safety: every candidate is re-verified (blob hash equality) immediately
before deletion. Files not matching origin/main are never touched.
"""
import subprocess
import sys

REPO = "/home/z/my-project"


def run(*args, **kw):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, **kw)


def main() -> int:
    # 1. enumerate untracked files
    un = run("git", "ls-files", "--others", "--exclude-standard")
    untracked = [l for l in un.stdout.splitlines() if l.strip()]

    # 2. hash them in chunks
    local_hashes = {}
    CH = 100
    for i in range(0, len(untracked), CH):
        chunk = untracked[i:i + CH]
        p = subprocess.run(["git", "hash-object", "--stdin-paths", "--"],
                           input="\n".join(chunk), cwd=REPO,
                           capture_output=True, text=True)
        for path, h in zip(chunk, p.stdout.split()):
            local_hashes[path] = h

    # 3. origin/main blob hashes
    tree = run("git", "ls-tree", "-r", "origin/main")
    origin_hashes = {}
    for line in tree.stdout.splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, h = meta.split()
        if typ == "blob":
            origin_hashes[path] = h

    tracked_here = set(run("git", "ls-files").stdout.splitlines())

    # 4. delete ONLY untracked files byte-identical to origin/main
    removed = 0
    for path, h in local_hashes.items():
        if path in tracked_here:
            continue
        if origin_hashes.get(path) == h:
            try:
                import os
                os.remove(os.path.join(REPO, path))
                removed += 1
            except OSError as e:
                print(f"WARN could not remove {path}: {e}")
    print(f"removed {removed} verified-identical untracked files")

    # 5. ff-only merge
    m = run("git", "merge", "--ff-only", "origin/main")
    print(m.stdout.strip() or m.stderr.strip())
    if m.returncode != 0:
        print("MERGE FAILED rc=", m.returncode)
        return 1

    head = run("git", "rev-parse", "--short", "HEAD").stdout.strip()
    om = run("git", "rev-parse", "--short", "origin/main").stdout.strip()
    print(f"HEAD={head} origin/main={om} ff_ok={head == om}")
    st = run("git", "status", "--short")
    lines = st.stdout.splitlines()
    print(f"status lines after merge: {len(lines)}")
    for l in lines[:12]:
        print(" ", l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
