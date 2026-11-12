"""Read-only inventory of an interrupted release publication.

This does not publish, delete, repair, resume, or authenticate an archive.
Provide a preserved candidate and explicit destination paths to compare.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def inspect(candidate: Path, destinations: list[Path]) -> dict:
    """Inventory paths without following symlinks or changing any file."""
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("candidate must be an existing regular file, not a symlink")
    candidate_sha = digest(candidate)
    try:
        with zipfile.ZipFile(candidate) as archive:
            bad = archive.testzip()
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        raise ValueError(f"candidate ZIP could not be verified: {exc}") from exc
    if bad is not None:
        raise ValueError(f"candidate ZIP CRC failure: {bad}")
    paths = [p.absolute() for p in destinations]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate destination paths")
    results = []
    for path in paths:
        record: dict = {"path": str(path)}
        if path.is_symlink():
            record["status"] = "symlink: manual review"
        elif not os.path.lexists(path):
            record["status"] = "missing"
        elif not path.is_file():
            record["status"] = "not a regular file: manual review"
        else:
            record["sha256"] = digest(path)
            record["same_file_as_candidate"] = os.path.samefile(candidate, path)
            record["status"] = "matches" if record["sha256"] == candidate_sha else "differs: manual review"
        results.append(record)
    return {"candidate": str(candidate.absolute()), "candidate_sha256": candidate_sha,
            "candidate_crc": "passed", "destinations": results,
            "notice": "Read-only inventory; CRC and hash matching do not establish provenance or safety."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="preserved candidate ZIP")
    parser.add_argument("destinations", type=Path, nargs="+", help="explicit final ZIP paths")
    args = parser.parse_args()
    try:
        report = inspect(args.candidate, args.destinations)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"inventory failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0 if all(item["status"] == "matches" for item in report["destinations"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
