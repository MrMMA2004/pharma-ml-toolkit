"""Audit an existing ZIP without publishing, altering it, or certifying a release.

The report records archive checks only; it does not run the test profiles or
engines, create a release backup, or prove crash recovery. Outputs are written
only to a new directory, never over an existing report or release archive.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import re
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any

import checkpoint

ENGINE_NAMES = ("autodock4.exe", "autogrid4.exe", "vina.exe")
ARCHIVE_CHECKS = ("path_and_bounds", "crc", "version_carriers", "manifest", "excluded_paths",
                  "engine_checksums", "isolated_restore", "archive_stability")


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with archive.open(info) as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_archive(path: Path) -> dict[str, Any]:
    """Perform bounded, non-mutating archive checks; never run tests or engines."""
    report: dict[str, Any] = {
        "scope": "Existing archive audit only; not a release certification",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "archive": {"name": path.name},
        "checks": {},
        "manifest": [],
        "engines": [],
        "test_profiles": {"status": "not executed by this audit"},
        "backup_and_recovery": {"status": "not verified by this audit"},
        "limitations": [
            "The archive is not necessarily the current checkout or a new release.",
            "No test profile or real engine was executed by this audit.",
            "No backup was created or recovery after interruption verified by this audit.",
            "Companion web version and full documentation/scientific consistency are not checked.",
        ],
    }
    checks = report["checks"]
    phase = "open_archive"
    try:
        report["archive"].update(size=path.stat().st_size, sha256=_digest_file(path))
        with zipfile.ZipFile(path) as archive:
            report["archive"]["entry_count"] = len(archive.infolist())
            phase = "path_and_bounds"
            prefix, plan = checkpoint._restore_plan(archive, checkpoint.RESTORE_LIMITS)
            checks[phase] = {"status": "passed", "layout": prefix or "root-level"}
            phase = "crc"
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"CRC failure on {bad}")
            checks[phase] = {"status": "passed"}
            by_name = {rel.as_posix(): info for info, rel in plan}
            phase = "version_carriers"
            version = checkpoint._validate_restore_versions(
                {name: archive.read(by_name[name]) for name in checkpoint._REQUIRED_RESTORE_FILES}, path
            )
            checks[phase] = {"status": "passed", "version": version,
                             "web_carrier": "not checked; companion source not in toolkit ZIP"}
            phase = "manifest"
            manifest = [
                {"path": name, "size": info.file_size, "sha256": _digest_member(archive, info)}
                for name, info in sorted(by_name.items())
            ]
            report["manifest"] = manifest
            report["archive"]["file_count"] = len(manifest)
            checks[phase] = {"status": "passed"}
            # Audit directories as well as files, including empty excluded directories.
            hits = []
            for info in archive.infolist():
                name = info.filename[len(prefix):] if prefix else info.filename
                parts = name.rstrip("/").split("/")
                excluded_dir = any(
                    part in checkpoint.ZIP_EXCLUDE_DIRS or part.endswith(checkpoint.ZIP_EXCLUDE_DIR_SUFFIX)
                    for part in (parts if info.is_dir() else parts[:-1])
                )
                excluded_file = not info.is_dir() and Path(parts[-1]).suffix.lower() in checkpoint.ZIP_EXCLUDE_SUFFIXES
                if excluded_dir or excluded_file:
                    hits.append(info.filename)
            checks["excluded_paths"] = {"status": "failed" if hits else "passed", "hits": sorted(hits)}
            data = [name for name in by_name if name.startswith("examples/data/")]
            report["example_data"] = {"recursive_file_count": len(data), "top_level_file_count": sum(
                "/" not in name.removeprefix("examples/data/") for name in data
            ), "documentation_consistency": "not fully checked"}
            phase = "engine_checksums"
            reference = by_name.get("engines/README.md")
            if reference is None or reference.file_size > 1024 * 1024:
                raise ValueError("engine checksum reference missing or too large")
            readme = archive.read(reference).decode("utf-8")
            for name in ENGINE_NAMES:
                engine_path = f"engines/{name}"
                info = by_name.get(engine_path)
                refs = re.findall(rf"(?im)^\s*([0-9a-f]{{32}})\s+{re.escape(name)}\s*$", readme)
                actual = _digest_member(archive, info, "md5") if info else None
                report["engines"].append({"path": engine_path, "reference_md5": refs[0] if len(refs) == 1 else None,
                    "md5": actual, "reference_matches": len(refs) == 1 and actual == refs[0].lower(),
                    "execution": "not executed"})
            checks[phase] = {"status": "passed" if all(e["reference_matches"] for e in report["engines"]) else "failed"}
        phase = "isolated_restore"
        # The destination and sentinel are created solely for this check; the
        # restore tool validates and verifies each installed file independently.
        with tempfile.TemporaryDirectory(prefix="pharmaml-release-audit-") as tmp:
            target = Path(tmp) / "tree"
            target.mkdir()
            sentinel_name = ".audit-unrelated-user-file"
            if sentinel_name in by_name:
                raise ValueError("reserved smoke-test sentinel occurs in archive")
            sentinel = target / sentinel_name
            sentinel.write_bytes(b"unrelated user data\n")
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                rc = checkpoint._restore_archive(path, target)
            if rc or sentinel.read_bytes() != b"unrelated user data\n":
                raise ValueError(f"restore smoke failed: {log.getvalue().strip()}")
            for item in manifest:
                extracted = target / item["path"]
                if not extracted.is_file() or extracted.stat().st_size != item["size"] or _digest_file(extracted) != item["sha256"]:
                    raise ValueError(f"extracted member mismatch: {item['path']}")
            checks[phase] = {"status": "passed", "files_verified": len(manifest), "unrelated_file_preserved": True}
        phase = "archive_stability"
        if path.stat().st_size != report["archive"]["size"] or _digest_file(path) != report["archive"]["sha256"]:
            raise ValueError("archive changed while it was being audited")
        checks[phase] = {"status": "passed"}
    except (OSError, ValueError, KeyError, UnicodeError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error) as exc:
        checks[phase] = {"status": "failed", "error": str(exc)}
    for name in ARCHIVE_CHECKS:
        checks.setdefault(name, {"status": "not executed"})
    report["archive_checks_passed"] = all(check["status"] == "passed" for check in checks.values())
    return report


def summary(report: dict[str, Any]) -> str:
    archive = report["archive"]
    lines = ["# Archive audit — not a release certification", "", f"Archive: `{archive['name']}`",
             f"Size: {archive.get('size', 'unknown')} bytes; SHA-256: `{archive.get('sha256', 'unknown')}`",
             f"Entries: {archive.get('entry_count', 'unknown')}; files: {archive.get('file_count', 'unknown')}", "",
             "## Checks", ""]
    for name, value in report["checks"].items():
        lines.append(f"- {name}: **{value['status']}**" + (f" — {value['error']}" if 'error' in value else ""))
    data = report.get("example_data")
    if data is not None:
        lines += ["", (f"Example-data files: {data['recursive_file_count']} recursive, "
                       f"{data['top_level_file_count']} top-level. Documentation consistency not fully checked.")]
    if report["engines"]:
        lines += ["", "## Bundled engine checksum references", ""]
        for engine in report["engines"]:
            lines.append(f"- `{engine['path']}`: MD5 `{engine['md5']}`, "
                         f"reference match: {engine['reference_matches']}; not executed")
    lines += ["", "Test profiles and engines were not executed. Backup and recovery were not verified.",
              "See `report.json` for the sorted per-file manifest, checksums and limitations.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory; must not already exist")
    args = parser.parse_args()
    result = audit_archive(args.archive)
    args.output_dir.mkdir(parents=False, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "SUMMARY.md").write_text(summary(result), encoding="utf-8")
    print(f"wrote {args.output_dir / 'report.json'} and SUMMARY.md")
    return 0 if result["archive_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
