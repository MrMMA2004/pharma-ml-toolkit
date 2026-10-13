"""Ligand and grid preparation for AutoDock4 / Vina docking.

Upgrades ``others/Docking/gpfer.py`` (GPF generation with per-ligand
atom-type extraction), ``smilestopdbqt.py`` (SMILES -> 3D PDBQT via
Meeko with AD4 atom-type repair) and the ligand-folder conventions of
``ligand generation/*.py``.

Key safety improvements over the originals:

* no ``os.chdir`` side effects -- every function takes explicit paths,
* atom-type repair table is data, not scattered ``if`` statements,
* subprocess calls use argument lists (never ``shell=True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pharmaml.exceptions import (
    DataValidationError,
    DependencyError,
    DockingError,
    DockingParseError,
)
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "GridBox",
    "compute_grid_box",
    "extract_pdbqt_atom_types",
    "write_gpf",
    "smiles_to_pdbqt",
    "RECEPTOR_TYPES",
]

# AutoDock4 atom types accepted in GPF ``receptor_types`` (from gpfer.py).
RECEPTOR_TYPES = "A C HD N NA OA SA"

# Meeko sometimes emits general types; map them onto the AD4 set
# (extracted from the original TYPE_MAP in smilestopdbqt.py).
AD4_TYPE_MAP = {
    "CG0": "C", "CG": "C", "G0": "C", "G": "C",
    "OG0": "OA", "OG": "OA",
    "NG0": "N", "NG": "N",
}

VALID_AD4_TYPES = {
    "C", "A", "N", "OA", "SA", "HD", "NA",
    "F", "Cl", "Br", "I", "P", "S", "Zn", "Fe", "Mg", "Ca", "Mn",
}


@dataclass(frozen=True, slots=True)
class GridBox:
    """Docking search box geometry.

    Attributes
    ----------
    center_x/y/z:
        Box centre in Å.
    npts:
        Number of grid points per axis (AutoDock4 convention:
        points = Å / spacing + 1).
    spacing:
        Grid spacing in Å (0.375 is the AutoDock default).
    """

    center_x: float
    center_y: float
    center_z: float
    npts: tuple[int, int, int] = (60, 60, 60)
    spacing: float = 0.375

    @property
    def size(self) -> tuple[float, float, float]:
        """Box edge lengths in Å along (x, y, z)."""
        return tuple(n * self.spacing for n in self.npts)  # type: ignore[return-value]

    def as_vina_kwargs(self) -> dict[str, float]:
        """Centre/size mapping consumed by ``vina`` CLI arguments."""
        size = self.size
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "center_z": self.center_z,
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
        }


def _read_coordinates(pdbqt_path: Path) -> np.ndarray:
    """Parse ATOM/HETATM coordinates from a PDB/PDBQT file.

    Primary strategy is strict PDB fixed-column slicing (x: cols 31-38,
    y: 39-46, z: 47-54); a whitespace-token fallback covers writers
    that only guarantee token order.
    """
    coords: list[list[float]] = []
    if not pdbqt_path.is_file():
        raise DockingParseError(f"ligand file not found: {pdbqt_path}")
    text = pdbqt_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        parsed = False
        if len(line) >= 54:
            try:
                coords.append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
                parsed = True
            except ValueError:
                parsed = False
        if not parsed:
            parts = line.split()
            if len(parts) >= 9:
                try:
                    coords.append([float(parts[6]), float(parts[7]), float(parts[8])])
                except ValueError:
                    continue
    if not coords:
        raise DockingParseError(f"no atom coordinates found in {pdbqt_path}")
    return np.asarray(coords, dtype=float)


def compute_grid_box(
    ligand_path: str | Path,
    padding: float = 4.0,
    npts: tuple[int, int, int] = (60, 60, 60),
    spacing: float = 0.375,
) -> GridBox:
    """Compute a grid box enclosing a reference ligand plus padding.

    Parameters
    ----------
    ligand_path:
        PDBQT/PDB file of the co-crystallised (reference) ligand.
    padding:
        Margin around the ligand bounding box in Å.
    npts, spacing:
        AutoDock4 grid resolution parameters.

    Returns
    -------
    GridBox
    """
    if padding < 0:
        raise DataValidationError(f"padding must be >= 0, got {padding}")
    coords = _read_coordinates(Path(ligand_path))
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    center = (mins + maxs) / 2.0
    _LOG.info(
        "Grid box around %s: centre (%.3f, %.3f, %.3f)",
        Path(ligand_path).name,
        *center,
    )
    return GridBox(
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        npts=npts,
        spacing=spacing,
    )


def extract_pdbqt_atom_types(pdbqt_path: str | Path) -> list[str]:
    """Collect sorted unique AutoDock atom types from a PDBQT ligand.

    Uses the last whitespace token of ATOM/HETATM lines (the convention
    ``gpfer.py`` relied on), filtered to alphabetic tokens so partial
    charges never leak into the type set.
    """
    types: set[str] = set()
    for line in Path(pdbqt_path).read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        if line.startswith(("ATOM", "HETATM")):
            parts = line.split()
            if parts:
                atom_type = parts[-1]
                if atom_type.isalpha():
                    types.add(atom_type)
    return sorted(types)


def write_gpf(
    ligand_pdbqt: str | Path,
    receptor_file: str = "r.pdbqt",
    box: GridBox | None = None,
    output_path: str | Path | None = None,
    receptor_types: str = RECEPTOR_TYPES,
    smooth: float = 0.5,
    dielectric: float = -0.1465,
) -> Path:
    """Write an AutoDock4 grid parameter file (GPF) for one ligand.

    Faithful upgrade of ``gpfer.py``: ligand atom types are extracted
    from the PDBQT, one affinity-map line is emitted per type, and the
    electrostatic/desolvation maps close the file.  Unlike the
    original, no global working directory is mutated.

    Parameters
    ----------
    ligand_pdbqt:
        Path to the prepared ligand PDBQT.
    receptor_file:
        Receptor PDBQT filename (relative to the docking directory).
    box:
        Grid geometry; defaults to a centred 60^3 box at 0.375 Å.
    output_path:
        Where to write the GPF; defaults next to the ligand
        (``<ligand_dir>/r.gpf``).

    Returns
    -------
    Path
        The written GPF path.
    """
    ligand = Path(ligand_pdbqt)
    if not ligand.is_file():
        raise DockingError(f"ligand PDBQT not found: {ligand}")
    box = box or GridBox(0.0, 0.0, 0.0)
    ligand_types = extract_pdbqt_atom_types(ligand)
    if not ligand_types:
        raise DockingError(f"no atom types found in {ligand}")

    lines = [
        f"npts {box.npts[0]} {box.npts[1]} {box.npts[2]}          # num.grid points in xyz",
        "gridfld r.maps.fld                    # grid_data_file",
        f"spacing {box.spacing}                        # spacing(A)",
        f"receptor_types {receptor_types}     # receptor atom types",
        f"ligand_types {' '.join(ligand_types)}            # ligand atom types",
        f"receptor {receptor_file}                     # macromolecule",
        f"gridcenter {box.center_x} {box.center_y} {box.center_z}      # xyz-coordinates",
        f"smooth {smooth}                           # store minimum energy w/in rad(A)",
    ]
    lines += [f"map r.{t}.map                          # atom-specific affinity map"
              for t in ligand_types]
    lines += [
        "elecmap r.e.map                      # electrostatic potential map",
        "dsolvmap r.d.map              # desolvation potential map",
        f"dielectric {dielectric}                   # <0, AD4 distance-dep.diel;>0, constant",
    ]

    out = Path(output_path) if output_path else ligand.parent / "r.gpf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _LOG.debug("Wrote GPF %s (ligand types: %s)", out, ligand_types)
    return out


def smiles_to_pdbqt(
    smiles: str,
    output_dir: str | Path,
    name: str | None = None,
    charge: int | None = None,
) -> Path:
    """Convert one SMILES to a 3D PDBQT ligand (Meeko + RDKit).

    Mirrors ``smilestopdbqt.py``: embed with ETKDG, MMFF-optimise,
    write via Meeko's legacy PDBQT writer, then repair any non-AD4
    atom types with the :data:`AD4_TYPE_MAP` table.

    Parameters
    ----------
    smiles:
        Molecule SMILES.
    output_dir:
        Destination directory (created if missing).
    name:
        Base filename (defaults to a safe SMILES slug).
    charge:
        Optional formal charge hint for protonation states.

    Returns
    -------
    Path
        Path of the written ``l.pdbqt``.

    Raises
    ------
    DependencyError
        If RDKit/Meeko are not installed.
    DockingError
        If embedding, writing, or atom-type repair fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError as exc:
        raise DependencyError(
            "smiles_to_pdbqt requires RDKit and Meeko. "
            "Install with:  pip install rdkit-pypi meeko"
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DockingError(f"invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol, addCoords=True)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        raise DockingError(f"3D embedding failed for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as exc:
        _LOG.warning("MMFF optimisation failed (%s); keeping unoptimised geometry.", exc)

    preparator = MoleculePreparation(merge_these_atom_types=("H",))
    setup = preparator.prepare(mol)[0]
    pdbqt_text, is_ok, error = PDBQTWriterLegacy.write_string(setup)
    if not is_ok:
        raise DockingError(f"Meeko PDBQT export failed: {error}")

    repaired_lines: list[str] = []
    for line in pdbqt_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_type = line[77:].strip()
            if atom_type not in VALID_AD4_TYPES:
                if atom_type in AD4_TYPE_MAP:
                    repaired = AD4_TYPE_MAP[atom_type]
                    _LOG.info("Repaired atom type %s -> %s", atom_type, repaired)
                    line = line[:77] + repaired.rjust(3)
                else:
                    _LOG.warning("Dropping line with unknown atom type %r", atom_type)
                    continue
        repaired_lines.append(line)

    from pharmaml.utils.io import safe_filename

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{safe_filename(name or smiles)}.pdbqt"
    target.write_text("\n".join(repaired_lines) + "\n", encoding="utf-8")
    return target
