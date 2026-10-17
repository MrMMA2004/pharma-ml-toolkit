"""Ligand and grid preparation for AutoDock4 / Vina docking.

Upgrades ``others/Docking/gpfer.py`` (GPF generation with per-ligand
atom-type extraction), ``others/dpfer.py`` (DPF generation with the
about-centroid, TORSDOF and the full AutoDock4 GA parameter block) and
``smilestopdbqt.py`` (SMILES -> 3D PDBQT via Meeko with AD4 atom-type
repair), plus the ligand-folder conventions of
``ligand generation/*.py``.

Key safety improvements over the originals:

* no ``os.chdir`` side effects -- every function takes explicit paths,
* atom-type repair table is data, not scattered ``if`` statements,
* subprocess calls use argument lists (never ``shell=True``).

Examples
--------
>>> from pharmaml.docking.preparation import GridBox
>>> box = GridBox(1.0, 2.0, 3.0, npts=(40, 40, 40), spacing=0.375)
>>> box.size            # fencepost: (npts - 1) * spacing per axis
(14.625, 14.625, 14.625)
>>> box.as_vina_kwargs()["size_x"]
14.625

>>> from pathlib import Path
>>> from tempfile import gettempdir
>>> from pharmaml.docking.preparation import compute_grid_box, extract_pdbqt_atom_types
>>> ligand = Path(gettempdir()) / "pharmaml_doctest_ligand.pdbqt"
>>> _ = ligand.write_text(
...     "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     C\\n"
...     "ATOM      2  O   LIG A   1       2.000   0.000   0.000  1.00  0.00     OA\\n"
...     "ATOM      3  N   LIG A   1       0.000   2.000   0.000  1.00  0.00     NA\\n"
...     "TORSDOF 1\\n"
... )
>>> extract_pdbqt_atom_types(ligand)
['C', 'NA', 'OA']
>>> box = compute_grid_box(ligand)   # centre = midpoint of the ligand bounding box
>>> (round(box.center_x, 3), round(box.center_y, 3), round(box.center_z, 3))
(1.0, 1.0, 0.0)

>>> from pharmaml.docking.preparation import read_gpf, write_gpf
>>> gpf = write_gpf(
...     ligand, box=GridBox(1.0, 1.0, 0.0, npts=(40, 50, 60)),
...     output_path=Path(gettempdir()) / "pharmaml_doctest_r.gpf")
>>> parsed = read_gpf(gpf)           # write/read round-trip
>>> parsed["ligand_types"], parsed["npts"], parsed["gridcenter"]
(['C', 'NA', 'OA'], (40, 50, 60), (1.0, 1.0, 0.0))
>>> parsed["receptor"], parsed["maps"]
('r.pdbqt', ['r.C.map', 'r.NA.map', 'r.OA.map'])
>>> ligand.unlink(); gpf.unlink()

>>> from pharmaml.docking.preparation import smiles_to_pdbqt
>>> pdbqt = smiles_to_pdbqt("CCO", output_dir=gettempdir(), name="pharmaml_doctest_etoh")
>>> pdbqt.name, extract_pdbqt_atom_types(pdbqt)   # seeded ETKDG + Meeko export
('pharmaml_doctest_etoh.pdbqt', ['C', 'HD', 'OA'])
>>> pdbqt.unlink()
>>> compute_grid_box("missing.pdbqt")
Traceback (most recent call last):
...
pharmaml.exceptions.DockingParseError: ligand file not found: missing.pdbqt
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

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
    "RECEPTOR_TYPES",
    "GridBox",
    "GridParameters",
    "compute_grid_box",
    "extract_pdbqt_atom_types",
    "read_gpf",
    "smiles_to_pdbqt",
    "write_dpf",
    "write_gpf",
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
        Number of grid points per axis.  AutoDock's fencepost
        convention: ``points = Å / spacing + 1`` -- the end points sit
        ON grid nodes, so the box edge length is ``(npts - 1) *
        spacing`` (see the :attr:`size` property).
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
        """Box edge lengths in Å along (x, y, z).

        Fencepost arithmetic: ``npts`` grid points at ``spacing`` span
        ``(npts - 1) * spacing`` Å per axis -- 60 points at 0.375 Å
        cover 22.125 Å, not 22.5 -- matching the AutoDock convention
        documented on :attr:`npts`.  The value feeds Vina's
        ``--size_x/y/z`` arguments via :meth:`as_vina_kwargs`.
        """
        return tuple(
            (n - 1) * self.spacing for n in self.npts
        )  # type: ignore[return-value]

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
                # Token-order fallback: locate the [resseq, x, y, z]
                # window -- the last three tokens of the first run of
                # four consecutive float-parsable tokens.  Chain-less
                # records (no chain ID) shift every field one token
                # left, so fixed offsets would read (y, z, occupancy)
                # instead of (x, y, z).
                for start in range(len(parts) - 3):
                    window = parts[start : start + 4]
                    try:
                        values = [float(token) for token in window]
                    except ValueError:
                        continue
                    coords.append(values[1:])
                    break
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

    The centre is the AutoDock convention: the midpoint of the ligand's
    bounding box (padding is symmetric, so the midpoint does not move).
    ``npts`` is a per-axis *minimum*: each axis grows as needed so the
    box size ``(npts - 1) * spacing`` encloses the ligand extent plus
    ``2 * padding`` on that axis -- the default ``60 x 60 x 60`` at
    0.375 A (22.5 A per axis) already covers drug-like ligands with the
    default padding, so growth only shows for large ligands or margins
    (round 100x-31: ``padding`` used to be validated and then silently
    ignored -- the box never grew).

    Parameters
    ----------
    ligand_path:
        PDBQT/PDB file of the co-crystallised (reference) ligand.
    padding:
        Margin around the ligand bounding box in Å (>= 0).
    npts:
        Per-axis minimum grid points; axes grow to honour ``padding``.
    spacing:
        AutoDock4 grid spacing in Å.

    Returns
    -------
    GridBox
    """
    if padding < 0:
        raise DataValidationError(f"padding must be >= 0, got {padding}")
    coords = _read_coordinates(Path(ligand_path))
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    center = (mins + maxs) / 2.0
    # per-axis growth: the box size (npts - 1) * spacing must cover the
    # ligand extent plus the symmetric padding margin
    required = (maxs - mins) + 2.0 * padding
    npts_out = (
        int(max(npts[0], math.ceil(required[0] / spacing) + 1)),
        int(max(npts[1], math.ceil(required[1] / spacing) + 1)),
        int(max(npts[2], math.ceil(required[2] / spacing) + 1)),
    )
    if npts_out != tuple(npts):
        _LOG.info(
            "Grid box around %s grew npts %s -> %s to enclose the ligand "
            "plus %.1f A padding",
            Path(ligand_path).name,
            tuple(npts),
            npts_out,
            padding,
        )
    _LOG.info(
        "Grid box around %s: centre (%.3f, %.3f, %.3f)",
        Path(ligand_path).name,
        *center,
    )
    return GridBox(
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        npts=npts_out,
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


class GridParameters(TypedDict):
    """Parsed AutoDock4 grid parameter file (GPF) content.

    The structural type of :func:`read_gpf`'s return value: a plain
    ``dict`` at runtime, with per-key types pinned so callers (and
    mypy) know what every GPF key holds.
    """

    ligand_types: list[str]
    receptor_types: list[str]
    npts: tuple[int, int, int]
    spacing: float
    gridcenter: tuple[float, float, float] | None
    receptor: str
    maps: list[str]
    smooth: float
    dielectric: float


def _gpf_float(field: str, tokens: list[str], path: Path) -> float:
    """Parse one GPF float field, raising a typed error when malformed."""
    if not tokens:
        raise DockingParseError(
            f"malformed {field} in {path}: expected a float value, got none"
        )
    try:
        return float(tokens[0])
    except ValueError as exc:
        raise DockingParseError(
            f"malformed {field} in {path}: cannot parse {tokens[0]!r} as a float"
        ) from exc


def read_gpf(gpf_path: str | Path) -> GridParameters:
    """Parse an AutoDock4 grid parameter file (GPF) into a dict.

    Companion reader for :func:`write_gpf` (and a typed replacement for
    the line-scraping in ``others/Docking/gpfreader.py``).  Inline ``#``
    comments are stripped before tokenising, and the affinity-map lines
    accumulate into the ``maps`` list.

    Returns
    -------
    GridParameters
        Keys: ``ligand_types`` (list[str]), ``receptor_types``
        (list[str]), ``npts`` (3 ints), ``spacing`` (float),
        ``gridcenter`` (3 floats or ``None`` when absent), ``receptor``
        (str), ``maps`` (list[str]), ``smooth`` (float) and
        ``dielectric`` (float).

    Raises
    ------
    DockingParseError
        If the file is unreadable, contains no ``ligand_types`` line,
        or any of the five numeric fields (``npts``, ``spacing``,
        ``gridcenter``, ``smooth``, ``dielectric``) is present but
        malformed -- instead of a raw ``ValueError`` or silently
        keeping the default value.
    """
    path = Path(gpf_path)
    if not path.is_file():
        raise DockingParseError(f"GPF file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        raise DockingParseError(f"cannot read GPF {path}: {exc}") from exc

    parsed: GridParameters = {
        "ligand_types": [],
        "receptor_types": [],
        "npts": (0, 0, 0),
        "spacing": 0.375,
        "gridcenter": None,
        "receptor": "",
        "maps": [],
        "smooth": 0.5,
        "dielectric": -0.1465,
    }

    def _fail(field: str, expected: str, got: object) -> DockingParseError:
        return DockingParseError(
            f"malformed {field} in {path}: expected {expected}, got {got!r}"
        )

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key, rest = parts[0], parts[1:]
        if key == "ligand_types":
            parsed["ligand_types"] = rest
        elif key == "receptor_types":
            parsed["receptor_types"] = rest
        elif key == "npts":
            if len(rest) < 3:
                raise _fail("npts", "3 integers", rest)
            try:
                parsed["npts"] = (int(rest[0]), int(rest[1]), int(rest[2]))
            except ValueError as exc:
                raise _fail("npts", "3 integers", rest) from exc
        elif key == "spacing":
            parsed["spacing"] = _gpf_float("spacing", rest, path)
        elif key == "gridcenter":
            if len(rest) < 3:
                raise _fail("gridcenter", "3 coordinates", rest)
            try:
                parsed["gridcenter"] = (
                    float(rest[0]),
                    float(rest[1]),
                    float(rest[2]),
                )
            except ValueError as exc:
                raise _fail("gridcenter", "3 coordinates", rest) from exc
        elif key == "receptor" and rest:
            parsed["receptor"] = rest[0]
        elif key == "map" and rest:
            parsed["maps"] = [*parsed["maps"], rest[0]]
        elif key == "smooth":
            parsed["smooth"] = _gpf_float("smooth", rest, path)
        elif key == "dielectric":
            parsed["dielectric"] = _gpf_float("dielectric", rest, path)

    if not parsed["ligand_types"]:
        raise DockingParseError(f"no ligand_types line found in {path}")
    return parsed


def write_dpf(
    ligand_pdbqt: str | Path,
    receptor_file: str = "r.pdbqt",
    output_path: str | Path | None = None,
    ga_run: int = 100,
    ga_pop_size: int = 150,
    ga_num_evals: int = 2500000,
    ga_num_generations: int = 27000,
    ga_elitism: int = 1,
    ga_mutation_rate: float = 0.02,
    ga_crossover_rate: float = 0.8,
    ga_window_size: int = 10,
    ga_cauchy_alpha: float = 0.0,
    ga_cauchy_beta: float = 1.0,
    sw_max_its: int = 300,
    sw_max_succ: int = 4,
    sw_max_fail: int = 4,
    sw_rho: float = 1.0,
    sw_lb_rho: float = 0.01,
    ls_search_freq: float = 0.06,
) -> Path:
    """Write an AutoDock4 docking parameter file (DPF) for one ligand.

    Faithful upgrade of ``others/dpfer.py``: the ligand atom types, the
    ``about`` centre (centroid of the ligand atoms) and the TORSDOF are
    read from the PDBQT, then the classic AutoDock4 parameter block is
    emitted verbatim -- Lamarckian GA settings, Solis-Wets local search,
    ``unbound_model bound`` and a final ``analysis`` section.  Unlike
    the original, the ligand is an explicit path (no folder guessing)
    and every GA/local-search knob is a keyword argument.

    Parameters
    ----------
    ligand_pdbqt:
        Path to the prepared ligand PDBQT.
    receptor_file:
        Receptor PDBQT filename; kept for parity with
        :func:`write_gpf` (the DPF itself only references the
        pre-computed maps).
    output_path:
        Where to write the DPF; defaults next to the ligand
        (``<ligand_dir>/r.dpf``).
    ga_run:
        Number of GA runs (``ga_run``).
    ga_pop_size, ga_num_evals, ga_num_generations, ...:
        Genetic-algorithm and Solis-Wets local-search parameters,
        defaulting to the values the original script used.

    Returns
    -------
    Path
        The written DPF path.

    Raises
    ------
    DockingError
        If the ligand file is missing or has no atom types.
    """
    ligand = Path(ligand_pdbqt)
    if not ligand.is_file():
        raise DockingError(f"ligand PDBQT not found: {ligand}")
    if ligand.name == receptor_file:
        _LOG.warning(
            "ligand %s has the receptor filename %r -- docking the receptor "
            "against itself?",
            ligand,
            receptor_file,
        )
    ligand_types = extract_pdbqt_atom_types(ligand)
    if not ligand_types:
        raise DockingError(f"no atom types found in {ligand}")

    coords = _read_coordinates(ligand)
    centre = coords.mean(axis=0)
    cx, cy, cz = (round(float(value), 3) for value in centre)

    torsdof = 0
    for line in ligand.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("TORSDOF"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                torsdof = int(parts[1])

    lines = [
        "autodock_parameter_version 4.2",
        "outlev 1",
        "intelec",
        "seed pid time",
        f"ligand_types {' '.join(ligand_types)}",
        "fld r.maps.fld",
    ]
    lines += [f"map r.{atom_type}.map" for atom_type in ligand_types]
    lines += [
        "elecmap r.e.map",
        "desolvmap r.d.map",
        f"move {ligand.name}",
        f"about {cx} {cy} {cz}",
        "tran0 random",
        "quaternion0 random",
        "dihe0 random",
        f"torsdof {torsdof}",
        "rmstol 2.0",
        "extnrg 1000.0",
        "e0max 0.0 10000",
        f"ga_pop_size {ga_pop_size}",
        f"ga_num_evals {ga_num_evals}",
        f"ga_num_generations {ga_num_generations}",
        f"ga_elitism {ga_elitism}",
        f"ga_mutation_rate {ga_mutation_rate}",
        f"ga_crossover_rate {ga_crossover_rate}",
        f"ga_window_size {ga_window_size}",
        f"ga_cauchy_alpha {ga_cauchy_alpha}",
        f"ga_cauchy_beta {ga_cauchy_beta}",
        "set_ga",
        f"sw_max_its {sw_max_its}",
        f"sw_max_succ {sw_max_succ}",
        f"sw_max_fail {sw_max_fail}",
        f"sw_rho {sw_rho}",
        f"sw_lb_rho {sw_lb_rho}",
        f"ls_search_freq {ls_search_freq}",
        "set_psw1",
        "unbound_model bound",
        f"ga_run {ga_run}",
        "analysis",
    ]

    out = Path(output_path) if output_path else ligand.parent / "r.dpf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _LOG.debug(
        "Wrote DPF %s (ligand types: %s, torsdof %d, about %.3f %.3f %.3f)",
        out,
        ligand_types,
        torsdof,
        cx,
        cy,
        cz,
    )
    return out


def smiles_to_pdbqt(
    smiles: str,
    output_dir: str | Path,
    name: str | None = None,
    charge: int | None = None,
    seed: int = 0,
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
        Formal-charge hint.  Currently **ignored** -- the neutral state
        is always generated (no protonation engine is invoked); passing
        a non-zero hint only triggers a :class:`UserWarning` so the
        caller knows the output will not match the intended species.
    seed:
        Random seed for the ETKDG embedding (default 0 = deterministic,
        unlike RDKit's unseeded default).  Pass different seeds to
        sample different conformers of a flexible molecule.

    Returns
    -------
    Path
        Path of the written ``l.pdbqt``.

    Raises
    ------
    DependencyError
        If RDKit/Meeko are not installed.
    DockingError
        If embedding or writing fails, or the molecule contains atom
        types with no AutoDock4 equivalent (e.g. boron) -- the PDBQT is
        refused rather than silently losing those atoms.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem

        # AllChem only re-exports these; importing from the defining
        # modules gives the fully-typed names in rdkit-stubs (identical
        # objects).
        from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
    except ImportError as exc:
        raise DependencyError(
            "smiles_to_pdbqt requires RDKit and Meeko. "
            "Install with:  pip install rdkit meeko"
        ) from exc

    if charge not in (None, 0):
        warnings.warn(
            "charge hint is currently ignored; generating the neutral state",
            UserWarning,
            stacklevel=2,
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DockingError(f"invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol, addCoords=True)
    embed_params = rdDistGeom.ETKDG()
    # unseeded ETKDG is nondeterministic; rdkit-stubs leaves randomSeed
    # untyped (mypy infers EmbedParameters), but at runtime it is a plain
    # int property.
    embed_params.randomSeed = seed  # type: ignore[assignment]
    if rdDistGeom.EmbedMolecule(mol, embed_params) != 0:
        raise DockingError(f"3D embedding failed for {smiles!r}")
    try:
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    except Exception as exc:  # noqa: BLE001 — Boost error zoo; geometry fallback is acceptable
        _LOG.warning("MMFF optimisation failed (%s); keeping unoptimised geometry.", exc)

    preparator = MoleculePreparation(merge_these_atom_types=("H",))
    setup = preparator.prepare(mol)[0]
    pdbqt_text, is_ok, error = PDBQTWriterLegacy.write_string(setup)
    if not is_ok:
        raise DockingError(f"Meeko PDBQT export failed: {error}")

    repaired_lines: list[str] = []
    dropped: dict[str, int] = {}
    for line in pdbqt_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_type = line[77:].strip()
            if atom_type not in VALID_AD4_TYPES:
                if atom_type in AD4_TYPE_MAP:
                    repaired = AD4_TYPE_MAP[atom_type]
                    _LOG.info("Repaired atom type %s -> %s", atom_type, repaired)
                    line = line[:77] + repaired.rjust(3)
                else:
                    dropped[atom_type] = dropped.get(atom_type, 0) + 1
                    continue
        repaired_lines.append(line)
    if dropped:
        # Silently dropping atoms would corrupt the ligand (missing mass,
        # wrong net charge) -- refuse instead, naming the elements.
        raise DockingError(
            f"SMILES {smiles!r} contains atom type(s) {sorted(dropped)} with no "
            "AutoDock4 parameterisation -- the PDBQT would silently lose "
            f"{sum(dropped.values())} atom(s) (element without an AD4 type); "
            "refusing to write a corrupted ligand"
        )

    from pharmaml.utils.io import safe_filename

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{safe_filename(name or smiles)}.pdbqt"
    target.write_text("\n".join(repaired_lines) + "\n", encoding="utf-8")
    return target
