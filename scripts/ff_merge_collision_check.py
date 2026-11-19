#!/usr/bin/env python3
"""Check untracked working-tree files that origin/main also tracks.

For each untracked file: compare its blob hash with origin/main's version.
Classify as identical / differing. Also count origin files missing locally.
Never prints file contents, only counts and (for differing) paths.
"""
import subprocess
import sys

REPO = "/home/z/my-project"


def run(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True)


def main() -> int:
    un = run("git", "ls-files", "--others", "--exclude-standard")
    untracked = [l for l in un.stdout.splitlines() if l.strip()]
    print(f"untracked files: {len(untracked)}")

    # hash all untracked files (in chunks to avoid arg limits)
    local_hashes = {}
    CH = 100
    for i in range(0, len(untracked), CH):
        chunk = untracked[i:i + CH]
        inp = "\n".join(chunk)
        p = subprocess.run(["git", "hash-object", "--stdin-paths", "--"],
                           input=inp, cwd=REPO, capture_output=True, text=True)
        for path, h in zip(chunk, p.stdout.split()):
            local_hashes[path] = h

    tree = run("git", "ls-tree", "-r", "origin/main")
    origin_hashes = {}
    for line in tree.stdout.splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, h = meta.split()
        if typ == "blob":
            origin_hashes[path] = h

    tracked_here = set(run("git", "ls-files").stdout.splitlines())

    identical, differing, not_in_origin = [], [], []
    for path, h in local_hashes.items():
        if path in tracked_here:
            continue  # tracked-and-modified, not untracked
        oh = origin_hashes.get(path)
        if oh is None:
            not_in_origin.append(path)
        elif oh == h:
            identical.append(path)
        else:
            differing.append(path)

    print(f"untracked & identical to origin/main: {len(identical)}")
    print(f"untracked & DIFFERING from origin/main: {len(differing)}")
    for p in differing[:20]:
        print(f"  DIFF {p}")
    print(f"untracked & not in origin/main (safe extras): {len(not_in_origin)}")
    for p in not_in_origin[:15]:
        print(f"  extra {p}")

    # origin files that are neither tracked locally nor untracked (i.e. absent)
    tracked_local = set(tracked_here)
    absent = [p for p in origin_hashes if p not in tracked_local
              and p not in local_hashes]
    print(f"origin files absent from working tree (will be created by ff): "
          f"{len(absent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
