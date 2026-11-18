"""Read-only inspection of review-first Vina campaign artifacts.

This is not a resume authorization or a substitute for scientific validation.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from pharmaml.docking.campaign_launch import _validated_options
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.provenance import fingerprint_file


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"duplicate JSON key in campaign manifest: {key}")
        result[key] = value
    return result


def _check_fingerprint(value: Any) -> bool:
    return (isinstance(value, dict) and isinstance(value.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
            and isinstance(value.get("size_bytes"), int) and not isinstance(value["size_bytes"], bool)
            and value["size_bytes"] >= 0)


def _relative_name(text: Any) -> bool:
    if not isinstance(text, str) or not text or "\\" in text:
        return False
    p = PurePosixPath(text)
    return not p.is_absolute() and all(part not in ("..", ".") for part in p.parts) and p.as_posix() == text


def _inside_file(root: Path, relative: str) -> Path:
    """Build a fixed contained path and refuse symlinks at every descendant."""
    if not _relative_name(relative):
        raise DataValidationError("unsafe relative path in campaign record")
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise DataValidationError(f"campaign contains symlink at {relative}; inspection refused")
    return path


def _fingerprint_status(path: Path, expected: dict[str, Any]) -> str:
    if path.is_symlink():
        return "symlink"
    if not path.is_file():
        return "missing"
    try:
        actual = fingerprint_file(path)
    except DataValidationError:
        return "unreadable"
    return ("match" if (actual["sha256"], actual["size_bytes"]) ==
            (expected["sha256"], expected["size_bytes"]) else "changed")


def _generated_config(plan: dict[str, Any]) -> bytes:
    p = plan["preview"]
    values = p.get("center_angstrom"), p.get("size_angstrom")
    if any(not isinstance(v, list) or len(v) != 3 for v in values):
        raise DataValidationError("invalid recorded box")
    coordinates = [*values[0], *values[1]]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in coordinates):
        raise DataValidationError("invalid recorded box coordinates")
    if any(v <= 0 for v in coordinates[3:]):
        raise DataValidationError("invalid recorded box dimensions")
    options = plan.get("options")
    if not isinstance(options, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                            for k, v in options.items()):
        raise DataValidationError("invalid recorded Vina options")
    _validated_options(options)
    keys = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    text = "receptor = receptor.pdbqt\n" + "".join(f"{k} = {v:.17g}\n" for k, v in zip(keys, coordinates, strict=True))
    text += "".join(f"{name} = {value}\n" for name, value in options.items())
    return text.encode("utf-8")


def inspect_vina_campaign(directory: str | Path, *, check_originals: bool = False) -> dict[str, Any]:
    """Inspect recorded progress, snapshot/output hashes, and optional original inputs.

    Never executes an engine, changes a file, quarantines, or resumes. Artifacts
    and manifest are unsigned and checked on a trusted quiescent filesystem.
    Original input/engine paths are read ONLY when explicitly requested. A running
    process or races cannot be ruled out. Missing/unrecorded outputs are not
    considered completed, and matches do not validate docking or enable resume.
    """
    if not str(directory).strip():
        raise DataValidationError("campaign directory is required")
    root = Path(os.path.abspath(directory))
    if root.is_symlink() or not root.is_dir():
        raise DataValidationError("campaign must be an existing non-symlink directory")
    root = root.resolve()
    manifest_path = _inside_file(root, "campaign.json")
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle, object_pairs_hook=_no_duplicate_pairs)
    except (OSError, ValueError, UnicodeError) as exc:
        raise DataValidationError(f"cannot read campaign manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "pharmaml-vina-campaign/1":
        raise DataValidationError("unsupported campaign manifest")
    plan, rows, status = manifest.get("plan"), manifest.get("results"), manifest.get("status")
    if (not isinstance(plan, dict) or plan.get("schema") != "pharmaml-vina-launch-plan/1"
            or not isinstance(plan.get("preview"), dict) or not isinstance(plan.get("ligands"), list)
            or not plan["ligands"] or not isinstance(rows, list) or len(rows) > len(plan["ligands"])
            or status not in {"preparing", "running", "completed", "cancelled", "failed"}
            or not _check_fingerprint(plan.get("config")) or not _check_fingerprint(plan.get("receptor"))):
        raise DataValidationError("invalid campaign manifest structure")
    preview, ligands = plan["preview"], plan["ligands"]
    if (any(not isinstance(item, dict) or not _relative_name(item.get("relative_path"))
            or not _check_fingerprint(item) for item in ligands)
            or len({item["relative_path"] for item in ligands}) != len(ligands)):
        raise DataValidationError("invalid recorded ligand list")
    expected_config = _generated_config(plan)
    snapshots = {"config": _fingerprint_status(_inside_file(root, "inputs/reviewed_config.txt"), plan["config"]),
                 "receptor": _fingerprint_status(_inside_file(root, "inputs/receptor.pdbqt"), plan["receptor"])}
    config_path = _inside_file(root, "inputs/vina.conf")
    try:
        actual_config = config_path.read_bytes()
        snapshots["generated_config"] = "match" if actual_config.replace(b"\r\n", b"\n") == expected_config else "changed"
    except FileNotFoundError:
        snapshots["generated_config"] = "missing"
    except OSError:
        snapshots["generated_config"] = "unreadable"
    ligand_statuses = []
    expected_outputs: set[str] = set()
    for index, item in enumerate(ligands, 1):
        snapshot_path = f"inputs/ligands/{index:06d}/ligand_{index:06d}.pdbqt"
        row_status: dict[str, Any] = {"ligand": item["relative_path"],
                                     "snapshot": _fingerprint_status(_inside_file(root, snapshot_path), item),
                                     "recorded_result": "not_recorded"}
        if index <= len(rows):
            row = rows[index - 1]
            if not isinstance(row, dict) or row.get("ligand") != item["relative_path"] or type(row.get("ok")) is not bool:
                raise DataValidationError("invalid or out-of-order campaign result")
            if row["ok"]:
                pose = f"results/ligand_{index:06d}_out.pdbqt"
                log = f"results/ligand_{index:06d}.log"
                # Older Windows manifests serialize Path relative strings with backslashes.
                if (row.get("pose") not in {pose, pose.replace("/", "\\")}
                        or row.get("log") not in {log, log.replace("/", "\\")}
                        or not _check_fingerprint(row.get("pose_fingerprint"))
                        or not _check_fingerprint(row.get("log_fingerprint"))
                        or row["pose_fingerprint"]["size_bytes"] == 0 or row["log_fingerprint"]["size_bytes"] == 0):
                    raise DataValidationError("invalid recorded output paths or fingerprints")
                expected_outputs.update((pose, log))
                row_status.update({"recorded_result": "success",
                                   "pose": _fingerprint_status(_inside_file(root, pose), row["pose_fingerprint"]),
                                   "log": _fingerprint_status(_inside_file(root, log), row["log_fingerprint"])})
            else:
                if index != len(rows):
                    raise DataValidationError("failed result is not the final recorded result")
                if row.get("error") is not None and not isinstance(row["error"], str):
                    raise DataValidationError("invalid recorded failure message")
                row_status["recorded_result"] = "failed"
                row_status["recorded_error"] = row.get("error")
        ligand_statuses.append(row_status)
    successes = sum(row["recorded_result"] == "success" for row in ligand_statuses)
    failed = sum(row["recorded_result"] == "failed" for row in ligand_statuses)
    if ((status == "completed" and (successes != len(ligands) or failed))
            or (status == "failed" and failed != 1)
            or (status == "preparing" and rows)
            or (status == "cancelled" and (failed or len(rows) == len(ligands)))):
        raise DataValidationError("campaign status contradicts recorded result sequence")
    results_dir = _inside_file(root, "results")
    unrecorded = []
    unrecorded_dirs = []
    if results_dir.exists():
        if not results_dir.is_dir():
            raise DataValidationError("results path is not a directory")
        for path in sorted(results_dir.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise DataValidationError(f"campaign contains symlink at {relative}; inspection refused")
            if path.is_file() and relative not in expected_outputs:
                unrecorded.append(relative)
            elif path.is_dir():
                unrecorded_dirs.append(relative)
    originals: dict[str, Any] = {"status": "not checked"}
    if check_originals:
        if any(not isinstance(preview.get(k), str) or not os.path.isabs(preview[k])
               for k in ("config", "receptor", "ligands_dir")):
            raise DataValidationError("invalid recorded original input paths")
        engine = plan.get("engine_prefix")
        if not isinstance(engine, list) or not engine or any(not _check_fingerprint(item)
                or not isinstance(item.get("path"), str) or not os.path.isabs(item["path"]) for item in engine):
            raise DataValidationError("invalid recorded engine prefix")
        original_root = Path(preview["ligands_dir"])
        if original_root.is_symlink() or not original_root.is_dir():
            census: dict[str, Any] = {"status": "unavailable"}
        else:
            current_ligands = {path.relative_to(original_root).as_posix() for path in original_root.rglob("*.pdbqt")
                               if path.is_file() or path.is_symlink()}
            reviewed_ligands = {item["relative_path"] for item in ligands}
            census = {"status": "match" if current_ligands == reviewed_ligands else "changed",
                      "added": sorted(current_ligands - reviewed_ligands),
                      "missing": sorted(reviewed_ligands - current_ligands)}
        originals = {"status": "checked", "ligand_census": census,
                     "config": _fingerprint_status(Path(preview["config"]), plan["config"]),
                     "receptor": _fingerprint_status(Path(preview["receptor"]), plan["receptor"]),
                     "ligands": [{"ligand": item["relative_path"],
                                  "status": ("unavailable" if census["status"] == "unavailable" else
                                             _fingerprint_status(_inside_file(original_root, item["relative_path"]), item))}
                                 for item in ligands],
                     "engine_prefix": [{"path": e["path"], "status": _fingerprint_status(Path(e["path"]), e)}
                                       for e in engine]}
    verified = sum(row.get("pose") == "match" and row.get("log") == "match" for row in ligand_statuses)
    return {"schema": "pharmaml-vina-campaign-inspection/1", "recorded_status": status,
            "progress": {"total": len(ligands), "recorded_successes": successes, "recorded_failures": failed,
                         "not_recorded": len(ligands) - len(rows), "success_output_pairs_matching": verified},
            "snapshots": snapshots, "ligands": ligand_statuses, "unrecorded_result_files": unrecorded,
            "unrecorded_result_directories": unrecorded_dirs,
            "originals": originals,
            "notice": ("Read-only inspection of unsigned files; match is not proof of completion, scientific "
                       "validity, independence or safe resume. A process may still be running; quiescent "
                       "filesystem assumed. Unrecorded files are not treated as completed. No resume is available.")}
