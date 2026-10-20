#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/docking/prep`` route.

Runs inside the toolkit virtualenv (real RDKit + meeko + pharmaml) and wires
ONLY the public ``pharmaml.docking`` API:

    smiles_to_pdbqt -> extract_pdbqt_atom_types -> compute_grid_box
    -> write_gpf / write_dpf (text returned, files stay in a temp dir)

Request (stdin, JSON)::

    {
      "smiles": ["CCO", ...],
      "padding": 4.0, "spacing": 0.375,
      "gaRun": 100, "gaPopSize": 150
    }

Response (stdout, JSON, camelCase keys for the TypeScript route)::

    {
      "ligands": [{"smiles": "...", "ok": true, "name": "aspirin",
                    "atoms": 13, "torsdof": 4, "atomTypes": ["A", "C", ...],
                    "pdbqtText": "...", "error": null}, ...],
      "gridBox": {"centerX": 0.461, ..., "sizeX": 22.125, ...,
                   "npts": [60, 60, 60], "spacing": 0.375},
      "gpfText": "...", "dpfText": "...",
      "nOk": 1, "nFailed": 0, "elapsedSeconds": 0.4, "rdkitVersion": "2026.09.1"
    }

Errors print a one-line ``{"error": "..."}`` JSON on stdout with exit code 2
so the route can surface them as HTTP 400/500 cleanly.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 2


def _clean(value: Any) -> Any:
    """Recursively convert numpy scalars + NaN/inf to strict-JSON values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _num(request: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    raw = request.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        print(json.dumps({"error": f"{key} must be a number between {lo} and {hi}"}))
        raise SystemExit(EXIT_ERROR)
    value = float(raw)
    if not (lo <= value <= hi):
        print(json.dumps({"error": f"{key} must be between {lo} and {hi}, got {value}"}))
        raise SystemExit(EXIT_ERROR)
    return value


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(json.dumps({"error": f"bridge request is not valid JSON: {exc}"}))
        return EXIT_ERROR

    if not isinstance(request, dict):
        print(json.dumps({"error": "bridge request must be a JSON object"}))
        return EXIT_ERROR

    smiles = request.get("smiles")
    if not isinstance(smiles, list) or not smiles or not all(isinstance(s, str) for s in smiles):
        print(json.dumps({"error": "smiles must be a non-empty list of strings"}))
        return EXIT_ERROR

    padding = _num(request, "padding", 4.0, 0.0, 20.0)
    spacing = _num(request, "spacing", 0.375, 0.1, 2.0)
    ga_run = int(_num(request, "gaRun", 100, 1, 10000))
    ga_pop = int(_num(request, "gaPopSize", 150, 1, 10000))

    try:  # pragma: no cover - import guard for a broken environment
        from pharmaml.docking import (
            compute_grid_box,
            extract_pdbqt_atom_types,
            smiles_to_pdbqt,
            write_dpf,
            write_gpf,
        )
    except ImportError as exc:
        print(json.dumps({"error": f"pharmaml.docking unavailable: {exc}"}))
        return EXIT_ERROR

    import rdkit

    started = time.perf_counter()
    ligands: list[dict[str, Any]] = []
    gpf_text: str | None = None
    dpf_text: str | None = None
    grid_box: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="web_ligand_prep_") as tmp:
        root = Path(tmp)
        first_ok: Path | None = None
        for index, smi in enumerate(smiles):
            name = f"ligand_{index + 1:02d}"
            try:
                pdbqt_path = smiles_to_pdbqt(smi, root, name=name)
            except Exception as exc:  # noqa: BLE001 - per-molecule failure is data
                ligands.append(
                    {
                        "smiles": smi,
                        "ok": False,
                        "name": name,
                        "atoms": None,
                        "torsdof": None,
                        "atomTypes": [],
                        "pdbqtText": None,
                        "error": str(exc),
                    }
                )
                continue

            text = pdbqt_path.read_text(encoding="utf-8", errors="replace")
            torsdof = None
            for line in text.splitlines():
                if line.startswith("TORSDOF"):
                    try:
                        torsdof = int(line.split()[1])
                    except (IndexError, ValueError):
                        torsdof = None
                    break
            atom_types = extract_pdbqt_atom_types(pdbqt_path)
            ligands.append(
                {
                    "smiles": smi,
                    "ok": True,
                    "name": name,
                    "atoms": sum(1 for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))),
                    "torsdof": torsdof,
                    "atomTypes": atom_types,
                    "pdbqtText": text,
                    "error": None,
                }
            )
            if first_ok is None:
                first_ok = pdbqt_path

        if first_ok is not None:
            box = compute_grid_box(first_ok, padding=padding, spacing=spacing)
            grid_box = {
                "centerX": box.center_x,
                "centerY": box.center_y,
                "centerZ": box.center_z,
                "sizeX": box.size[0],
                "sizeY": box.size[1],
                "sizeZ": box.size[2],
                "npts": list(box.npts),
                "spacing": box.spacing,
            }
            gpf_path = write_gpf(first_ok, output_path=root / "r.gpf")
            dpf_path = write_dpf(
                first_ok,
                output_path=root / "r.dpf",
                ga_run=ga_run,
                ga_pop_size=ga_pop,
            )
            gpf_text = gpf_path.read_text(encoding="utf-8", errors="replace")
            dpf_text = dpf_path.read_text(encoding="utf-8", errors="replace")

    n_ok = sum(1 for lig in ligands if lig["ok"])
    if n_ok == 0:
        print(
            json.dumps(
                {
                    "error": "No SMILES could be prepared as a PDBQT ligand.",
                    "ligands": _clean(ligands),
                }
            )
        )
        return EXIT_ERROR

    response = {
        "ligands": ligands,
        "gridBox": grid_box,
        "gpfText": gpf_text,
        "dpfText": dpf_text,
        "nOk": n_ok,
        "nFailed": len(ligands) - n_ok,
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "rdkitVersion": rdkit.__version__,
    }
    json.dump(_clean(response), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
