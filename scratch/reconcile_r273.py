#!/usr/bin/env python3
"""r273 untracked/modified reconcile vs origin/main (rollback recovery).

For every local change vs HEAD, classify against origin/main:
- untracked local file also in origin/main: blob-identical -> delete local;
  differing -> keep (report; will block merge only if merge touches it).
- modified tracked file: local == origin/main blob -> restore clean (checkout).
- deleted tracked file present in origin/main -> restore via checkout.
The ff-only merge afterwards converges the tree to origin/main exactly.
"""
import subprocess, sys, os

ROOT = os.path.expanduser("~/my-project")
os.chdir(ROOT)

def sh(*args):
    return subprocess.run(args, capture_output=True, text=True)

def origin_blob(path):
    r = sh("git", "rev-parse", f"origin/main:{path}")
    return r.stdout.strip() if r.returncode == 0 else None

def local_blob(path):
    r = sh("git", "hash-object", path)
    return r.stdout.strip() if r.returncode == 0 else None

status = sh("git", "status", "--porcelain", "-z")
entries = [e for e in status.stdout.split("\0") if e]

untracked_same = untracked_diff = restored_mod = restored_del = 0
kept_diff = []
for raw in entries:
    code, path = raw[:2], raw[3:]
    if code == "??":
        # path may be quoted or a dir; only handle regular files
        p = path.strip('"')
        if p.endswith("/"):
            continue
        ob = origin_blob(p)
        if ob is None:
            continue  # not on origin: local-only, leave alone
        lb = local_blob(p)
        if lb == ob:
            os.remove(p)
            untracked_same += 1
        else:
            untracked_diff += 1
            kept_diff.append(p)
    elif code == " M" or code == "M ":
        p = path.strip('"')
        ob = origin_blob(p)
        lb = local_blob(p)
        if ob is not None and lb == ob:
            sh("git", "checkout", "--", p)
            restored_mod += 1
        else:
            kept_diff.append(f"MOD:{p}")
    elif code == " D" or code == "D ":
        p = path.strip('"')
        if origin_blob(p) is not None:
            sh("git", "checkout", "--", p)
            restored_del += 1
        else:
            # deleted locally, absent on origin too: nothing to do
            pass

print(f"untracked identical-removed: {untracked_same}")
print(f"untracked differing kept:    {untracked_diff}")
for p in kept_diff:
    print(f"  KEEP {p}")
print(f"modified restored:          {restored_mod}")
print(f"deleted restored:           {restored_del}")
leftover = sh("git", "status", "--porcelain")
print("--- status after reconcile ---")
print(leftover.stdout if leftover.stdout else "(clean)")
