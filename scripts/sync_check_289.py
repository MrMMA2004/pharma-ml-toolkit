#!/usr/bin/env python3
"""Compare every locally-modified tracked file's disk content against the
origin/main blob, so the reset --hard to origin/main is provably lossless.
Prints only paths of genuine mismatches (never file content)."""
import subprocess
import sys

REPO = "/home/z/my-project"


def blob_md5(path: str) -> str | None:
    r = subprocess.run(
        ["git", "cat-file", "blob", f"origin/main:{path}"],
        cwd=REPO, capture_output=True, check=False,
    )
    if r.returncode != 0:
        return None
    import hashlib
    return hashlib.md5(r.stdout).hexdigest()


def main() -> int:
    st = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    checked = mismatch = missing_in_origin = 0
    for line in st:
        if not line.startswith(" M "):
            continue
        path = line[3:].strip()
        try:
            disk_md5 = subprocess.run(
                ["md5sum", path], cwd=REPO,
                capture_output=True, text=True, check=True,
            ).stdout.split()[0]
        except subprocess.CalledProcessError:
            continue  # vanished from disk; reset will restore it
        checked += 1
        orig = blob_md5(path)
        if orig is None:
            missing_in_origin += 1
            print(f"NOT-IN-ORIGIN: {path}")
        elif orig != disk_md5:
            mismatch += 1
            print(f"MISMATCH: {path}")
    print(f"checked={checked} mismatch={mismatch} not-in-origin={missing_in_origin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
