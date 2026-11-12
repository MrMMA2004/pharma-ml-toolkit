"""Review-then-launch sequential Vina campaigns in a fresh output directory.

This is not resumable, crash-durable, or safe against concurrent writers. It
runs a user-selected external executable, which must be trusted. Successful
process exit and nonempty output do not establish scientific validity.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pharmaml.docking.campaign_preview import preview_vina_campaign
from pharmaml.docking.workflow import engine_argv, run_vina_batch
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.provenance import fingerprint_file

_SUPPORTED = {"exhaustiveness", "num_modes", "cpu", "seed", "energy_range"}


def _validated_options(options: dict[str, str]) -> dict[str, str]:
    extra = set(options) - _SUPPORTED
    if extra:
        raise DataValidationError("launch does not support config directives: " + ", ".join(sorted(extra)))
    for name, text in options.items():
        if name == "energy_range":
            import math

            try:
                value = float(text)
            except ValueError as exc:
                raise DataValidationError("energy_range must be finite and positive") from exc
            if not math.isfinite(value) or value <= 0:
                raise DataValidationError("energy_range must be finite and positive")
        else:
            if not re.fullmatch(r"[0-9]+", text) or len(text) > 10:
                raise DataValidationError(f"{name} must be a positive integer")
            value = int(text)
            if not 1 <= value <= 2_147_483_647:
                raise DataValidationError(f"{name} must be between 1 and 2147483647")
    return dict(options)


def build_vina_launch_plan(
    config_path: str | Path, ligands_dir: str | Path, output_dir: str | Path, *,
    vina_bin: str = "vina", engine_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build an unsigned input fingerprint plan without writing or executing.

    Only simple, supported Vina options are allowed; all ligand files, the
    receptor, config and engine prefix are hashed. The output must not exist.
    A plan is only a snapshot and assumes trusted, quiescent input paths.
    """
    config = Path(config_path).resolve()
    before = fingerprint_file(config)
    preview = preview_vina_campaign(config, ligands_dir, output_dir, vina_bin=vina_bin, engine_dir=engine_dir)
    if before != fingerprint_file(config):
        raise DataValidationError("config changed during review")
    output = Path(preview["output_dir"])
    if output.exists():
        raise DataValidationError("launch requires a new, non-existing output directory")
    if not output.parent.is_dir():
        raise DataValidationError("output parent directory must already exist")
    if preview["engine"]["prefix"] is None:
        raise DataValidationError("Vina engine is unavailable: " + preview["engine"].get("reason", "unknown"))
    prefix = preview["engine"]["prefix"]
    engine = []
    for item in prefix:
        file = Path(item)
        if not file.is_file():
            raise DataValidationError(f"engine prefix cannot be fingerprinted: {item}")
        engine.append({"path": str(file.resolve()), **fingerprint_file(file)})
    ligand_root = Path(preview["ligands_dir"])
    paths = sorted(p for p in ligand_root.rglob("*.pdbqt") if p.is_file())
    if len(paths) != preview["ligands"]:
        raise DataValidationError("ligand set changed during review")
    if any(p.is_symlink() for p in paths):
        raise DataValidationError("launch does not accept symlinked ligand files")
    ligands = [{"relative_path": p.relative_to(ligand_root).as_posix(), **fingerprint_file(p)} for p in paths]
    options = _validated_options(preview["other_config_directives"])
    notices = ["Sequential execution only; no resume or skip-existing.",
               "Input snapshots and partial results remain in the new output directory.",
               "Run only trusted engines and scientifically prepared inputs.",
               "Stop is cooperative between ligands, not during an engine process.",
               "No crash-atomic multi-file transaction or power-loss durability."]
    if "seed" not in options:
        notices.append("No seed specified; engine randomness is not controlled by this plan.")
    return {"schema": "pharmaml-vina-launch-plan/1", "preview": preview, "config": before,
            "receptor": fingerprint_file(preview["receptor"]), "ligands": ligands,
            "engine_prefix": engine, "options": options,
            "notices": notices}


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    # Atomic replacement of one JSON file, not a transaction with pose/logs.
    with tempfile.NamedTemporaryFile("w", dir=output, prefix=".manifest-", suffix=".tmp",
                                     encoding="utf-8", delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, output / "campaign.json")
    finally:
        temp.unlink(missing_ok=True)


def _copy_verified(source: Path, destination: Path, expected: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    copied = fingerprint_file(destination)
    if copied["sha256"] != expected["sha256"] or copied["size_bytes"] != expected["size_bytes"]:
        raise DataValidationError(f"input changed since review: {source}")


def launch_vina_campaign(
    reviewed_plan: dict[str, Any], *, on_progress: Callable[[int, int], bool | None] | None = None,
    timeout_s: float = 3600,
) -> dict[str, Any]:
    """Recheck a reviewed plan, create a fresh output directory, then run sequentially.

    Never resumes or skips. Snapshots are checked against reviewed hashes and
    retained under the output directory. Returns a manifest with completed,
    failed or cancelled status. Exceptions and crashes can leave partial files;
    inspect them rather than treating file existence as completion. This assumes
    a trusted quiescent filesystem and trusted executable, not adversarial races.
    """
    if not isinstance(reviewed_plan, dict) or reviewed_plan.get("schema") != "pharmaml-vina-launch-plan/1":
        raise DataValidationError("a reviewed Vina launch plan is required")
    p = reviewed_plan["preview"]
    engine = reviewed_plan["engine_prefix"]
    # Resolve the precise engine path; Wine, if needed, is discovered again and
    # is compared through the complete prefix and its hashes.
    engine_binary = engine[-1]["path"]
    fresh = build_vina_launch_plan(p["config"], p["ligands_dir"], p["output_dir"], vina_bin=engine_binary)
    if fresh != reviewed_plan:
        raise DataValidationError("inputs, engine or plan changed since review; review again")
    output = Path(p["output_dir"])
    try:
        output.mkdir()  # exclusive: no existing directory or clobber
    except FileExistsError as exc:
        raise DataValidationError("output directory already exists; review a fresh location") from exc
    manifest: dict[str, Any] = {"schema": "pharmaml-vina-campaign/1", "status": "preparing",
                                "plan": reviewed_plan, "results": [],
                                "notice": "Partial files may remain after failure or interruption; no resume."}
    _write_manifest(output, manifest)
    snapshots = output / "inputs"
    receptor = snapshots / "receptor.pdbqt"
    _copy_verified(Path(p["receptor"]), receptor, reviewed_plan["receptor"])
    original_config = snapshots / "reviewed_config.txt"
    _copy_verified(Path(p["config"]), original_config, reviewed_plan["config"])
    ligand_root = Path(p["ligands_dir"])
    staged_ligands = []
    for index, item in enumerate(reviewed_plan["ligands"], 1):
        path = snapshots / "ligands" / f"{index:06d}" / f"ligand_{index:06d}.pdbqt"
        _copy_verified(ligand_root / item["relative_path"], path, item)
        staged_ligands.append(path)
    keys = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    values = [*p["center_angstrom"], *p["size_angstrom"]]
    text = "receptor = receptor.pdbqt\n" + "".join(f"{k} = {v:.17g}\n" for k, v in zip(keys, values, strict=True))
    text += "".join(f"{name} = {value}\n" for name, value in reviewed_plan["options"].items())
    conf = snapshots / "vina.conf"
    conf.write_text(text, encoding="utf-8")
    manifest["status"] = "running"
    _write_manifest(output, manifest)
    total = len(staged_ligands)
    for index, (ligand, item) in enumerate(zip(staged_ligands, reviewed_plan["ligands"], strict=True)):
        if on_progress is not None and on_progress(index, total) is False:
            manifest["status"] = "cancelled"
            break
        current_prefix = [str(Path(part).resolve()) for part in engine_argv(engine_binary)]
        if current_prefix != [entry["path"] for entry in engine]:
            raise DataValidationError("engine launch prefix changed after review; stop and inspect partial output")
        for executable in engine:
            if fingerprint_file(executable["path"]) != {k: executable[k] for k in ("name", "sha256", "size_bytes")}:
                raise DataValidationError("engine changed after review; stop and inspect partial output")
        results = run_vina_batch(conf, ligand.parent, output / "results", max_parallel=1,
                                 vina_bin=engine_binary, timeout_s=timeout_s, skip_existing=False)
        if len(results) != 1:
            raise DataValidationError("unexpected ligand census; stop and inspect partial output")
        result = results[0]
        record: dict[str, Any] = {"ligand": item["relative_path"], "ok": result.ok,
                                  "seconds": result.seconds, "error": result.error}
        if result.ok:
            pose = output / "results" / f"{ligand.stem}_out.pdbqt"
            assert result.log_file is not None
            record.update({"pose": str(pose.relative_to(output)), "pose_fingerprint": fingerprint_file(pose),
                           "log": str(result.log_file.relative_to(output)),
                           "log_fingerprint": fingerprint_file(result.log_file)})
        manifest["results"].append(record)
        _write_manifest(output, manifest)
        if not result.ok:
            manifest["status"] = "failed"
            break
    else:
        manifest["status"] = "completed"
    _write_manifest(output, manifest)
    if on_progress is not None and manifest["status"] == "completed":
        on_progress(total, total)
    return manifest
