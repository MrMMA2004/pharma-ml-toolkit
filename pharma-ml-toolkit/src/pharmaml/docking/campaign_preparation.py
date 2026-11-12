"""Review and prepare a fresh Vina input tree; no docking is executed."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

from pharmaml.docking.ligands import load_ligand_csv, prepare_ligand_library
from pharmaml.docking.receptor import prepare_protein
from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger
from pharmaml.ml.provenance import environment_snapshot, fingerprint_file

_LOG = get_logger(__name__)


def _dependencies(protein_suffix: str) -> dict[str, str]:
    packages = ["rdkit", "meeko"] + (["gemmi"] if protein_suffix in {".cif", ".mmcif"} else [])
    result = {}
    for package in packages:
        if importlib.util.find_spec(package) is None:
            raise DependencyError(f"preparation requires {package}; install pharmaml[docking]")
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyError(f"cannot determine installed {package} version") from exc
    return result


def build_preparation_plan(
    receptor: str | Path, ligand_table: str | Path, output_dir: str | Path, *,
    smiles_col: str | None = None, name_col: str | None = None, chain: str | None = None,
    remove_waters: bool = True, keep_hetatm: bool = False, seed: int = 0,
) -> dict[str, Any]:
    """Review source hashes, syntax counts, dependencies and fixed preparation choices.

    This does not prepare molecules, validate biological chemistry or write files.
    Source files and selected dependency versions are checked again before use.
    """
    if any(not str(path).strip() for path in (receptor, ligand_table, output_dir)):
        raise DataValidationError("receptor, ligand table and output directory are required")
    receptor_path = Path(receptor).resolve()
    table_path = Path(ligand_table).resolve()
    # Preserve the final path component to reject dangling symlinks as existing.
    requested_output = Path(os.path.abspath(output_dir))
    if os.path.lexists(requested_output):
        raise DataValidationError("preparation requires a new, non-existing output directory")
    if not requested_output.parent.is_dir():
        raise DataValidationError("output parent directory must already exist")
    output = requested_output.parent.resolve() / requested_output.name
    suffix = receptor_path.suffix.lower()
    if suffix not in {".pdb", ".pdbqt", ".cif", ".mmcif"}:
        raise DataValidationError("receptor must be PDB, PDBQT or mmCIF")
    if table_path.suffix.lower() not in {".csv", ".tsv", ".txt", ".xlsx", ".xlsm"}:
        raise DataValidationError("ligand table must be CSV, TSV, TXT, XLSX or XLSM")
    if chain is not None and (not isinstance(chain, str) or len(chain) != 1 or not chain.isalnum()):
        raise DataValidationError("chain must be one alphanumeric PDB chain character or omitted")
    if not isinstance(remove_waters, bool) or not isinstance(keep_hetatm, bool):
        raise DataValidationError("receptor cleaning options must be boolean")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
        raise DataValidationError("ligand seed must be an integer from 0 to 2147483647")
    if receptor_path == table_path:
        raise DataValidationError("receptor and ligand table must be different files")
    receptor_hash = fingerprint_file(receptor_path)
    before = fingerprint_file(table_path)
    dependencies = _dependencies(suffix)
    library = load_ligand_csv(table_path, smiles_col=smiles_col, name_col=name_col)
    if before != fingerprint_file(table_path) or receptor_hash != fingerprint_file(receptor_path):
        raise DataValidationError("input changed during review; retry with stable files")
    valid_count = len(library.valid_frame)
    if not library.validated:
        raise DependencyError("RDKit SMILES syntax validation is unavailable")
    if not valid_count:
        raise DataValidationError("no syntactically valid SMILES in the ligand table")
    return {"schema": "pharmaml-preparation-plan/1", "receptor_path": str(receptor_path),
            "ligand_table_path": str(table_path), "output_dir": str(output),
            "receptor": receptor_hash, "ligand_table": before, "dependencies": dependencies,
            "options": {"smiles_col": smiles_col, "name_col": name_col, "chain": chain,
                        "remove_waters": remove_waters, "keep_hetatm": keep_hetatm, "seed": seed},
            "review": {"rows": len(library), "syntactically_valid": valid_count,
                       "invalid_smiles": len(library.invalid_frame), "smiles_column": library.smiles_column,
                       "name_column": library.name_column, "validated_by_rdkit": library.validated},
            "notices": ["No docking, box selection, or scientific validation is performed.",
                        "PDB/PDBQT cleaning removes waters and non-protein HETATM by default; review these choices.",
                        "First model and first alternate location are used; bad residues are not silently deleted.",
                        "Ligand 3D embedding is a single seeded conformer, not a protonation or tautomer workflow.",
                        "Source snapshots and partial results remain in the new output folder; no resume or stop.",
                        "Assumes a trusted quiescent filesystem; not crash-atomic or power-loss durable."]}


def _snapshot(source: Path, destination: Path, expected: dict[str, Any]) -> None:
    with source.open("rb") as src, destination.open("xb") as dest:
        shutil.copyfileobj(src, dest, length=1024 * 1024)
    actual = fingerprint_file(destination)
    if (actual["sha256"], actual["size_bytes"]) != (expected["sha256"], expected["size_bytes"]):
        raise DataValidationError(f"input changed since review: {source}")


def _manifest(output: Path, record: dict[str, Any]) -> None:
    # Output is new and trusted; a crash can leave a partial/stale manifest.
    with (output / "preparation.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prepare_reviewed_inputs(plan: dict[str, Any]) -> dict[str, Any]:
    """Recheck a review, snapshot sources and prepare into a fresh directory.

    No existing output is reused. Failed or interrupted preparation can leave a
    partial directory and stale manifest. Never treats a partial run as complete.
    """
    if not isinstance(plan, dict) or plan.get("schema") != "pharmaml-preparation-plan/1":
        raise DataValidationError("a reviewed preparation plan is required")
    fresh = build_preparation_plan(plan["receptor_path"], plan["ligand_table_path"], plan["output_dir"],
                                   **plan["options"])
    if fresh != plan:
        raise DataValidationError("input, dependencies or options changed since review; review again")
    output = Path(plan["output_dir"])
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise DataValidationError("output directory already exists; choose a new location") from exc
    record: dict[str, Any] = {"schema": "pharmaml-preparation-run/1", "status": "preparing", "plan": plan,
                              "environment": environment_snapshot(), "receptor": None, "ligands": None,
                              "notice": "Partial files may remain after failure or interruption; no resume."}
    _manifest(output, record)
    try:
        snapshots = output / "inputs"
        snapshots.mkdir()
        receptor = snapshots / ("receptor_source" + Path(plan["receptor_path"]).suffix.lower())
        ligands = snapshots / ("ligands_source" + Path(plan["ligand_table_path"]).suffix.lower())
        _snapshot(Path(plan["receptor_path"]), receptor, plan["receptor"])
        _snapshot(Path(plan["ligand_table_path"]), ligands, plan["ligand_table"])
        opts = plan["options"]
        protein = prepare_protein(receptor, output / "receptor.pdbqt", remove_waters=opts["remove_waters"],
                                  keep_hetatm=opts["keep_hetatm"], chain=opts["chain"],
                                  default_altloc="first", strip_hydrogens=True, allow_bad_res=False)
        receptor_out = output / "receptor.pdbqt"
        record["receptor"] = {"audit": protein.as_dict(), "output": "receptor.pdbqt",
                              "fingerprint": fingerprint_file(receptor_out)}
        _manifest(output, record)
        prepared = prepare_ligand_library(ligands, output / "ligands", smiles_col=opts["smiles_col"],
                                          name_col=opts["name_col"], folder_naming="index", seed=opts["seed"])
        rows = json.loads(prepared.frame.to_json(orient="records"))
        for row in rows:
            if row["ok"]:
                path = Path(row["pdbqt"])
                if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                    raise DataValidationError("prepared ligand is missing, empty or a symlink")
                row["pdbqt"] = path.relative_to(output).as_posix()
                row["fingerprint"] = fingerprint_file(path)
        record["ligands"] = {"prepared": prepared.n_prepared, "failed": prepared.n_failed, "rows": rows}
        record["status"] = ("failed" if prepared.n_prepared == 0 else
                            "completed_with_failures" if prepared.n_failed else "completed")
        _manifest(output, record)
        return record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        try:
            _manifest(output, record)
        except Exception as manifest_error:  # noqa: BLE001 - preserve the original failure
            _LOG.warning("Cannot record preparation failure: %s", manifest_error)
        raise
