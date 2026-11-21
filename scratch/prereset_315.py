#!/usr/bin/env python3
"""r315 pre-reset safety check: compare every origin/main tree file against
the disk state, categorize (identical / differing / missing / untracked),
and verify the r315-wip backup covers every differing+new WIP file."""
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/z/my-project")


def run(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, cwd=REPO, **kw)

# 1. the origin/main tree: "mode type sha\tpath"
ls = run("git", "ls-tree", "-r", "origin/main")
assert ls.returncode == 0, ls.stderr
tree = {}
for line in ls.stdout.splitlines():
    meta, path = line.split("\t", 1)
    mode, typ, sha = meta.split()
    tree[path] = sha
print(f"origin/main tree: {len(tree)} files")

# 2. hash every existing disk file the same way git does
paths = [p for p in tree if (REPO / p).exists()]
ho = subprocess.run(["git", "hash-object", "--stdin-paths"], input="\n".join(paths),
                    capture_output=True, text=True, cwd=REPO)
assert ho.returncode == 0, ho.stderr
disk_sha = dict(zip(paths, ho.stdout.splitlines()))

identical = {p for p in paths if disk_sha[p] == tree[p]}
differing = sorted(p for p in paths if disk_sha[p] != tree[p])
missing = sorted(p for p in tree if not (REPO / p).exists())
print(f"identical: {len(identical)}  differing: {len(differing)}  missing: {len(missing)}")

# 3. untracked disk files (not in the tree) -- categories
st = run("git", "status", "--porcelain")
untracked = [line[3:] for line in st.stdout.splitlines() if line.startswith("??")]
print(f"untracked-on-disk: {len(untracked)}")

# 4. the backup must cover every differing tracked file
backup = REPO / "scratch" / "r315-wip"
backed = {str(p.relative_to(backup)) for p in backup.rglob("*") if p.is_file()}
uncovered = [p for p in differing if p not in backed]
print(f"differing files NOT in the r315-wip backup: {len(uncovered)}")
for p in uncovered[:20]:
    print("  MISSING-BACKUP:", p)

print("\n-- differing (the WIP set) --")
for p in differing:
    print("  M", p)
print("\n-- missing on disk (deleted in status) --")
for p in missing:
    print("  D", p)
print("\n-- untracked, outside scratch/tool-results/node_modules --")
for p in untracked:
    if not any(p.startswith(x) for x in ("scratch/", "tool-results/", "node_modules/",
                                          ".next/", "dev.log")):
        print("  ??", p)
