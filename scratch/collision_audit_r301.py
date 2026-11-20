#!/usr/bin/env python3
"""Round-301 pre-ff-sync collision audit (the round-300 procedure).

The sandbox rolled back again: HEAD=2eb0ede (482 behind origin/main=35aab254)
with the working tree carrying the newer rounds' content uncommitted.
Verify every locally-modified tracked file and every untracked file
byte-identically (blob-hash) against origin/main:
  - identical tracked-modified  -> safe to restore to HEAD (merge rewrites them)
  - identical untracked         -> safe to remove (merge writes them)
  - differing                   -> must be inspected, never silently touched
Never touches differing files. Prints only counts + differing paths.
"""
import subprocess
import os
import sys

REPO = "/home/z/my-project"


def run(*args, **kw):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, **kw)


def main() -> int:
    # origin/main blob hashes
    tree = run("git", "ls-tree", "-r", "origin/main")
    origin_hashes = {}
    for line in tree.stdout.splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, h = meta.split()
        if typ == "blob":
            origin_hashes[path] = h

    status = run("git", "status", "--short")
    modified, untracked = [], []
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        st = line[:2]
        path = line[3:].strip().strip('"')
        if st.strip() == "M" or st in ("MM", " M", "M "):
            modified.append(path)
        elif st == "??":
            untracked.append(path)
        elif st.strip() == "D":
            modified.append(path)  # deleted tracked; handled below
        else:
            # renames/copies/unmerged -- collect both sides
            modified.append(path)

    def hash_file(path):
        p = os.path.join(REPO, path)
        if not os.path.exists(p):
            return None
        r = run("git", "hash-object", p)
        return r.stdout.strip() if r.returncode == 0 else None

    identical_mod, differing_mod = [], []
    for p in modified:
        h = hash_file(p)
        if h is None:
            # locally deleted: identical only if origin/main also deletes it
            differing_mod.append((p, "locally-deleted"))
            continue
        if origin_hashes.get(p) == h:
            identical_mod.append(p)
        else:
            differing_mod.append((p, f"wt-hash {h[:8]} vs origin {str(origin_hashes.get(p))[:8]}"))

    identical_un, differing_un = [], []
    CH = 200
    for i in range(0, len(untracked), CH):
        chunk = [u for u in untracked[i:i + CH] if os.path.isfile(os.path.join(REPO, u))]
        if not chunk:
            continue
        pr = subprocess.run(["git", "hash-object", "--stdin-paths", "--"],
                            input="\n".join(chunk), cwd=REPO,
                            capture_output=True, text=True)
        for path, h in zip(chunk, pr.stdout.split()):
            if origin_hashes.get(path) == h:
                identical_un.append(path)
            else:
                differing_un.append(path)

    print(f"modified-tracked : {len(modified)} total | {len(identical_mod)} identical to origin/main | {len(differing_mod)} DIFFERING")
    for p, why in differing_mod[:40]:
        print(f"  DIFF tracked {p}: {why}")
    print(f"untracked        : {len(untracked)} total | {len(identical_un)} identical to origin/main | {len(differing_un)} not-in-origin")
    for p in differing_un[:20]:
        print(f"  (untracked-not-in-origin) {p}")

    # persist lists for the merge step
    with open("/tmp/r301_identical_tracked.txt", "w") as f:
        f.write("\n".join(identical_mod))
    with open("/tmp/r301_identical_untracked.txt", "w") as f:
        f.write("\n".join(identical_un))
    with open("/tmp/r301_differing_tracked.txt", "w") as f:
        for p, why in differing_mod:
            f.write(f"{p}\t{why}\n")
    return 0 if not differing_mod else 2


if __name__ == "__main__":
    sys.exit(main())
