"""Resume missing release copies with an explicit digest and independent backup.

Defaults to a read-only plan; --execute is required to create missing copies.
Never overwrite, delete, retire, run build gates, or commit. This is not a
crash-atomic transaction and assumes trusted, quiescent local filesystems.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from release_recovery_status import digest, inspect

NAME = re.compile(r"pharma-ml-toolkit-v[0-9]+\.[0-9]+\.[0-9]+\.zip\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def plan(candidate: Path, backup: Path, expected: str, destinations: list[Path]) -> dict:
    """Validate all prerequisites without modifying files or directories."""
    if not SHA.fullmatch(expected):
        raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    if not destinations:
        raise ValueError("at least one destination is required")
    # Canonicalized parents detect equivalent paths and symlink aliases; this
    # is not a defense against another process changing them after preflight.
    paths = [path.parent.resolve() / path.name for path in destinations]
    if len(set(paths)) != len(paths) or len({p.parent for p in paths}) != len(paths):
        raise ValueError("destinations must be distinct, with one path per home")
    if len({p.name for p in paths}) != 1 or not NAME.fullmatch(paths[0].name):
        raise ValueError("all destinations must have the same versioned release ZIP filename")
    if not all(p.parent.is_dir() for p in paths):
        raise ValueError("every destination parent must already exist")
    if backup.is_symlink() or not backup.is_file():
        raise ValueError("backup must be an existing regular file, not a symlink")
    backup_path = backup.resolve()
    if any(backup_path.is_relative_to(path.parent) for path in paths):
        raise ValueError("backup must be outside all publication homes")
    report = inspect(candidate, paths)
    if report["candidate_sha256"] != expected:
        raise ValueError("candidate does not match the explicitly expected SHA-256")
    if os.path.samefile(candidate, backup) or digest(backup) != expected:
        raise ValueError("backup must be distinct and match the expected SHA-256")
    for entry, path in zip(report["destinations"], paths):
        if entry["status"] not in ("matches", "missing"):
            raise ValueError(f"unsafe destination {path}: {entry['status']}")
        if entry["status"] == "matches" and os.path.samefile(backup, path):
            raise ValueError(f"backup aliases a publication copy: {path}")
    return {"candidate": str(candidate.absolute()), "backup": str(backup.absolute()),
            "expected_sha256": expected, "destinations": report["destinations"],
            "notice": "Plan only; hash/CRC do not prove provenance, safety or release readiness."}


def resume(candidate: Path, backup: Path, expected: str, destinations: list[Path], *, execute: bool = False) -> dict:
    """Optionally publish only missing destinations, retaining partial successes on failure."""
    report = plan(candidate, backup, expected, destinations)
    report["executed"] = execute
    if not execute:
        return report
    report["notice"] = (
        "Publication requested; matching hashes and CRC do not prove provenance, safety or readiness."
    )
    try:
        for record in report["destinations"]:
            if record["status"] != "missing":
                continue
            destination = Path(record["path"])
            with tempfile.TemporaryDirectory(prefix=".recovery-copy-", dir=destination.parent) as folder:
                staged = Path(folder) / destination.name
                shutil.copy2(candidate, staged)
                if digest(staged) != expected or digest(candidate) != expected or digest(backup) != expected:
                    raise OSError("candidate, backup or staged copy changed before publication")
                os.link(staged, destination)  # same directory; fails if target already exists
            record["status"] = "published"
        # A concurrent modification can still happen after this verification;
        # report no success when one is detected.
        if digest(candidate) != expected or digest(backup) != expected:
            raise OSError("candidate or backup changed after publication")
        if any(digest(Path(r["path"])) != expected for r in report["destinations"]):
            raise OSError("a destination failed final verification")
    except OSError as exc:
        raise OSError(f"recovery stopped; already published copies retained, re-inventory: {exc}") from exc
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="independently verified candidate SHA-256")
    parser.add_argument("--execute", action="store_true", help="actually publish missing copies")
    parser.add_argument("destinations", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        report = resume(args.candidate, args.backup, args.sha256, args.destinations, execute=args.execute)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"recovery failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
