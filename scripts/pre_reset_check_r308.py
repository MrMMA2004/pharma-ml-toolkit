#!/usr/bin/env python3
"""Pre-reset safety check: untracked files that share a path with origin/main
must be content-identical, otherwise `reset --hard` would destroy unique work.
Prints only counts and paths of any mismatch (max 20)."""
import subprocess

untracked = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    capture_output=True, text=True, check=True).stdout.split("\0")
untracked = [p for p in untracked if p]

tree = subprocess.run(
    ["git", "ls-tree", "-r", "--format=%(objectname) %(path)", "origin/main"],
    capture_output=True, text=True, check=True).stdout
tree_map = {}
for ln in tree.splitlines():
    oid, _, path = ln.partition(" ")
    tree_map[path] = oid

same = diff = absent = 0
mismatches = []
batch = [p for p in untracked if p in tree_map]
# hash in chunks to avoid ARG_MAX
CH = 200
for i in range(0, len(batch), CH):
    chunk = batch[i:i + CH]
    r = subprocess.run(["git", "hash-object", "--"] + chunk,
                       capture_output=True, text=True, check=True)
    for p, oid in zip(chunk, r.stdout.split()):
        if oid == tree_map[p]:
            same += 1
        else:
            diff += 1
            mismatches.append(p)
absent = len(untracked) - len(batch)

print(f"untracked total: {len(untracked)}")
print(f"  same-path as origin/main: {len(batch)}  identical: {same}  DIFFERENT: {diff}")
print(f"  not in origin/main tree (survive reset): {absent}")
if mismatches:
    print("MISMATCHED PATHS (up to 20):")
    for p in mismatches[:20]:
        print(f"  {p}")
