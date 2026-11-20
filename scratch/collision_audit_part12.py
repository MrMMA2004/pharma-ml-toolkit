#!/usr/bin/env python3
"""Collision audit before ff-only sync: compare every file in the working tree
against origin/main blob hashes (byte-level, includes untracked files)."""
import subprocess, os, sys

ROOT = os.path.expanduser("~/my-project")

def sh(*args):
    return subprocess.run(args, capture_output=True, text=True, cwd=ROOT)

# target tree: path -> blob hash
tree = {}
out = sh("git", "ls-tree", "-r", "origin/main").stdout
for line in out.splitlines():
    meta, path = line.split("\t", 1)
    parts = meta.split()
    tree[path] = parts[2]

identical, different, missing = [], [], []
for path, blob in tree.items():
    fp = os.path.join(ROOT, path)
    if not os.path.exists(fp):
        missing.append(path)
        continue
    h = sh("git", "hash-object", "--", path).stdout.strip()
    if h == blob:
        identical.append(path)
    else:
        different.append(path)

print(f"target tracked files: {len(tree)}")
print(f"  identical in worktree:    {len(identical)}")
print(f"  DIFFERENT in worktree:    {len(different)}")
print(f"  MISSING from worktree:    {len(missing)}")

for name, lst in (("DIFFERENT", different), ("MISSING", missing)):
    if lst:
        print(f"--- {name} files ---")
        for p in lst[:40]:
            print(" ", p)
        if len(lst) > 40:
            print(f"  ... and {len(lst)-40} more")

# summarize worktree files NOT in target (untracked extras)
extras = []
for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in (".git", "node_modules", ".next", ".venv")]
    for f in files:
        rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
        if rel not in tree:
            extras.append(rel)
print(f"worktree files NOT tracked in target (extras, incl. gitignored): {len(extras)}")
