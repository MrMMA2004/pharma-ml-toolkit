"""Read-only Vina campaign review; no preparation, execution or file writes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pharmaml.docking.dockstring import parse_search_box
from pharmaml.docking.workflow import engine_argv, vina_batch_progress
from pharmaml.exceptions import DataValidationError, DockingError


def preview_vina_campaign(
    config_path: str | Path, ligands_dir: str | Path, output_dir: str | Path, *,
    vina_bin: str = "vina", engine_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Check candidate inputs and review geometry/engine without running Vina.

    Matches the existing batch's recursive, case-sensitive ``*.pdbqt`` scan
    and output census. Refuses receptor or output inside the ligand tree so
    those files are not mistaken for ligands. This is only a snapshot: file
    contents, chemistry, engine execution, output provenance and subsequent
    changes are not verified. No launch authorization is implied.
    """
    if not str(config_path).strip() or not str(ligands_dir).strip() or not str(output_dir).strip():
        raise DataValidationError("config, ligand directory and output directory are required")
    config = Path(config_path).resolve()
    ligands = Path(ligands_dir).resolve()
    output = Path(output_dir).resolve()
    if not config.is_file():
        raise DataValidationError("Vina config file not found")
    if not ligands.is_dir():
        raise DataValidationError("ligand directory not found")
    if output.exists() and not output.is_dir():
        raise DataValidationError("output path exists but is not a directory")
    if output == ligands or output.is_relative_to(ligands) or ligands.is_relative_to(output):
        raise DataValidationError("output and ligand directories must not overlap")
    # Fail closed on duplicate config directives: Vina's interpretation of an
    # ambiguous config must not differ from a superficially reassuring preview.
    params: dict[str, str] = {}
    try:
        for line_number, raw in enumerate(config.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not sep or not key or not value:
                raise DataValidationError(f"invalid config directive on line {line_number}")
            if key in params:
                raise DataValidationError(f"duplicate config directive: {key}")
            params[key] = value
    except (OSError, UnicodeError) as exc:
        raise DataValidationError(f"cannot read config: {exc}") from exc
    receptor_text = params.get("receptor")
    if not receptor_text:
        raise DataValidationError("config needs a receptor directive")
    receptor = (config.parent / receptor_text).resolve()
    if not receptor.is_file() or receptor.stat().st_size == 0:
        raise DataValidationError("receptor must be an existing nonempty file")
    if receptor.is_relative_to(ligands):
        raise DataValidationError("receptor is inside the ligand directory and would be docked as a ligand")
    keys = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    box = parse_search_box("\n".join(f"{key} = {params[key]}" for key in keys if key in params))
    ligand_files = sorted(p for p in ligands.rglob("*.pdbqt") if p.is_file())
    if not ligand_files:
        raise DataValidationError("no .pdbqt ligands found in the ligand directory")
    if any(p.resolve() == receptor for p in ligand_files):
        raise DataValidationError("a ligand path resolves to the receptor")
    if any(p.stat().st_size == 0 for p in ligand_files):
        raise DataValidationError("ligand directory contains an empty .pdbqt file")
    progress = vina_batch_progress(ligands, output)
    try:
        prefix = engine_argv(vina_bin, engine_dir)
        if len(prefix) == 1 and os.name != "nt" and not os.access(prefix[0], os.X_OK):
            raise DockingError("resolved native engine is not executable")
        engine: dict[str, Any] = {"status": "launch prefix resolved; not executed", "prefix": prefix}
    except DockingError as exc:
        engine = {"status": "unavailable", "reason": str(exc), "prefix": None}
    warnings = ["Preview does not execute Vina or validate molecular files or docking accuracy.",
                "Input files may change after review; re-check immediately before any run.",
                "Existing batch resume checks file presence, not input/config/engine provenance.",
                "Batch pose and log promotion is not crash-atomic."]
    if output.is_dir() and any(output.iterdir()):
        warnings.append("Output directory is nonempty; existing files can be overwritten by a batch run.")
    if any(name in params for name in ("ligand", "out", "log")):
        warnings.append("Config contains ligand/out/log directives; inspect for conflicts with batch arguments.")
    return {"config": str(config), "receptor": str(receptor), "ligands_dir": str(ligands),
            "output_dir": str(output), "center_angstrom": list(box.center),
            "size_angstrom": list(box.size), "ligands": progress.total,
            "matching_existing_pairs": progress.docked, "remaining_if_skip_existing": progress.remaining,
            "output_name_collisions": progress.collisions, "engine": engine,
            "other_config_directives": {k: v for k, v in params.items() if k not in {*keys, "receptor"}},
            "warnings": warnings, "executed": False}
