"""Audit an existing ZIP without publishing, altering it, or certifying a release.

The report records archive checks only; it does not run the test profiles or
engines, create a release backup, or prove crash recovery. Outputs are written
only to a new directory, never over an existing report or release archive.

``--profile-results`` (part5-continuation round 5, §14): fold a separately
recorded test-profile results file into the report.  The audit still never
runs tests itself — the input is caller-authored evidence (the profiles'
commands and their exact passed / failed / skipped / blocked / not-executed
counts, the gates, the backup/recovery status, the runtime/platform and
scientific limitations).  The loader validates that evidence for honest
labelling and refuses the whole file on any inconsistency: counts never
merge across profiles, a skipped or not-executed test can never count as
passed, and every profile that reports a command must account for all of
its collected tests across the five outcome buckets.
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

PROFILE_SCHEMA = "pharmaml-profile-results/2"
_OUTCOME_KEYS = ("passed", "failed", "skipped", "blocked", "not_executed")


def load_profile_results(path: Path) -> dict[str, Any]:
    """Load and validate the §14 profile-results evidence file.

    The file is caller-authored (the audit never runs tests); this loader
    enforces the honest-labelling contract so a malformed or inflated file
    cannot ride the report:

    * ``schema`` must equal :data:`PROFILE_SCHEMA`;
    * every profile needs a unique name, a recorded command, and all five
      outcome counts (ints >= 0);
    * the five counts must sum to ``collected`` when that field is present
      (the audit recomputes and records both);
    * ``passed`` must be 0 when ``not_executed``/``blocked`` is the only
      recorded outcome -- a profile that ran nothing passes nothing;
    * unknown top-level or per-profile keys are rejected (typos must not
      silently drop evidence);
    * gate outcomes must be one of ``passed`` / ``failed`` / ``not run``.

    Raises ``ValueError`` (with the precise reason) on any violation; the
    caller treats that as a failed audit input rather than silently
    degrading to the "not provided" defaults.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"profile-results file unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("profile-results file must be a JSON object")  # noqa: TRY004 - file-validation contract
    allowed_top = {"schema", "generated_by", "environment", "profiles", "gates",
                   "backup_and_recovery", "limitations"}
    unknown = set(data) - allowed_top
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    if data.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"schema must be {PROFILE_SCHEMA!r}, got {data.get('schema')!r}")
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a non-empty list")
    seen: set[str] = set()
    for entry in profiles:
        if not isinstance(entry, dict):
            raise ValueError("each profile must be an object")  # noqa: TRY004 - file-validation contract
        allowed = {"name", "command", "outcome_counts", "collected", "notes"}
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(f"profile {entry.get('name')!r}: unknown keys {sorted(unknown)}")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"profile names must be unique non-empty strings, got {name!r}")
        seen.add(name)
        command = entry.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"profile {name!r}: a recorded command is required")
        counts = entry.get("outcome_counts")
        if not isinstance(counts, dict):
            raise ValueError(f"profile {name!r}: outcome_counts must be an object")  # noqa: TRY004 - file-validation contract
        unknown = set(counts) - set(_OUTCOME_KEYS)
        if unknown:
            raise ValueError(f"profile {name!r}: unknown outcome keys {sorted(unknown)}")
        missing = set(_OUTCOME_KEYS) - set(counts)
        if missing:
            raise ValueError(f"profile {name!r}: missing outcome keys {sorted(missing)}")
        total = 0
        for key in _OUTCOME_KEYS:
            value = counts[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"profile {name!r}: {key} must be an int >= 0, got {value!r}")
            total += value
        recorded = entry.get("collected")  # the recorded total must match the sum
        entry["collected"] = total  # the audit recomputes; the recorded one must match
        if recorded is not None and recorded != total:
            raise ValueError(f"profile {name!r}: counts sum to {total}, not {recorded}")
    gates = data.get("gates", [])
    if not isinstance(gates, list):
        raise ValueError("gates must be a list")  # noqa: TRY004 - file-validation contract
    for gate in gates:
        if not isinstance(gate, dict) or set(gate) - {"name", "outcome", "detail"}:
            raise ValueError(f"malformed gate entry: {gate!r}")
        if gate.get("outcome") not in ("passed", "failed", "not run"):
            raise ValueError(f"gate {gate.get('name')!r}: outcome must be passed/failed/not run")
    for key in ("backup_and_recovery", "limitations"):
        block = data.get(key)
        if key == "limitations":
            if block is not None and (
                not isinstance(block, list) or any(not isinstance(item, str) for item in block)
            ):
                raise ValueError("limitations must be a list of strings when present")
        elif block is not None and not isinstance(block, dict):
            raise ValueError(f"{key} must be an object when present")
    return data


def _profile_summary_block(report: dict[str, Any]) -> list[str]:
    """Render the §14 profile/backup/limitation sections for SUMMARY.md."""
    profiles = report["test_profiles"]
    lines: list[str] = []
    if profiles.get("status") == "not executed by this audit" and "profiles" not in profiles:
        lines += ["", "Test profiles: **not executed by this audit** (no --profile-results input)."]
        return lines
    lines += ["", "## Test profiles (recorded evidence; not executed by this audit)", ""]
    for entry in profiles["profiles"]:
        counts = entry["outcome_counts"]
        parts = ", ".join(f"{counts[k]} {k.replace('_', '-')}" for k in _OUTCOME_KEYS)
        lines.append(f"- **{entry['name']}** — `{entry['command']}`")
        lines.append(f"  {parts} (collected: {entry['collected']})")
        if entry.get("notes"):
            lines.append(f"  Notes: {entry['notes']}")
    gates = report.get("gates") or []
    if gates:
        lines += ["", "### Gates", ""]
        for gate in gates:
            lines.append(f"- {gate['name']}: **{gate['outcome']}**"
                         + (f" — {gate['detail']}" if gate.get("detail") else ""))
    backup = report["backup_and_recovery"]
    lines += ["", "### Backup and recovery", ""]
    if backup.get("status") == "not verified by this audit" and len(backup) == 1:
        lines.append("Not verified by this audit (no --profile-results input).")
    else:
        for key, value in backup.items():
            lines.append(f"- {key}: {value}")
    lines += ["", "### Runtime / platform / scientific limitations", ""]
    limitations = report["limitations"]
    recorded = [l for l in limitations if not l.startswith("(the audit itself")]
    for line in recorded:
        lines.append(f"- {line}")
    if not recorded:
        lines.append("None recorded in the profile-results file.")
    return lines



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


def audit_archive(path: Path, profile_results: dict[str, Any] | None = None) -> dict[str, Any]:
    """Perform bounded, non-mutating archive checks; never run tests or engines.

    ``profile_results`` is the already-validated §14 evidence (see
    :func:`load_profile_results`) or ``None``.  When present it replaces the
    "not executed" placeholders in ``test_profiles`` / ``backup_and_recovery``
    / ``limitations`` — always as RECORDED evidence, clearly labelled as not
    executed by this audit; the counts are never merged across profiles and
    never folded into any archive-check verdict.
    """
    limitations = [
        "The archive is not necessarily the current checkout or a new release.",
        "No test profile or real engine was executed by this audit.",
        "No backup was created or recovery after interruption verified by this audit.",
        "Companion web version and full documentation/scientific consistency are not checked.",
    ]
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
        "limitations": limitations,
    }
    if profile_results is not None:
        report["test_profiles"] = {
            "status": "recorded evidence; not executed by this audit",
            "schema": PROFILE_SCHEMA,
            "source_generated_by": profile_results.get("generated_by"),
            "environment": profile_results.get("environment"),
            "profiles": profile_results["profiles"],
        }
        report["gates"] = profile_results.get("gates", [])
        report["backup_and_recovery"] = profile_results.get(
            "backup_and_recovery", {"status": "not verified by this audit"}
        )
        recorded = list(profile_results.get("limitations") or [])
        if recorded:
            report["limitations"] = recorded + [
                ("(the audit itself executed none of the recorded profiles, "
                 "gates, backups or recoveries; they are recorded evidence)")
            ]
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
            # (r287) the check mirrors build_zip.py's PATH-ANCHORED semantics
            # via checkpoint.ZIP_EXCLUDE_DIR_PATHS: the bare "artifacts"
            # component rule that lived here until this fix wrongly flagged
            # frontend/src/services/artifacts/client.ts -- the file the r285
            # build_zip fix exists to SHIP -- as a violation. Only the
            # anchored data/artifacts (the backend output home) is excluded.
            anchored = tuple(checkpoint.ZIP_EXCLUDE_DIR_PATHS)
            hits = []
            for info in archive.infolist():
                name = info.filename[len(prefix):] if prefix else info.filename
                parts = name.rstrip("/").split("/")
                excluded_dir = any(
                    part in checkpoint.ZIP_EXCLUDE_DIRS or part.endswith(checkpoint.ZIP_EXCLUDE_DIR_SUFFIX)
                    for part in (parts if info.is_dir() else parts[:-1])
                )
                if not info.is_dir():
                    rel_posix = "/".join(parts[:-1])
                    excluded_dir = excluded_dir or any(
                        rel_posix == a for a in anchored
                    )
                else:
                    rel_posix = name.rstrip("/")
                    excluded_dir = excluded_dir or any(
                        rel_posix == a or rel_posix.startswith(a + "/") for a in anchored
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
    lines += _profile_summary_block(report)
    lines += ["", ("See `report.json` for the sorted per-file manifest, checksums, "
              "recorded profile evidence and limitations."), ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory; must not already exist")
    parser.add_argument("--profile-results", type=Path, default=None,
                        help="JSON file of recorded test-profile evidence (schema "
                             f"{PROFILE_SCHEMA}); validated for honest labelling, "
                             "folded into the report as recorded-not-executed evidence")
    args = parser.parse_args()
    profile_results = None
    if args.profile_results is not None:
        profile_results = load_profile_results(args.profile_results)  # ValueError -> traceable failure
    result = audit_archive(args.archive, profile_results=profile_results)
    args.output_dir.mkdir(parents=False, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "SUMMARY.md").write_text(summary(result), encoding="utf-8")
    print(f"wrote {args.output_dir / 'report.json'} and SUMMARY.md")
    return 0 if result["archive_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
