"""Receptor (protein) import and preparation for AutoDock docking.

The original ``others/Docking`` scripts never prepared the protein:
``gpfer.py`` assumed a ready-made ``r.pdbqt`` sitting in every ligand
folder and hard-coded ``receptor_types = "A C HD N NA OA SA"``.  The
actual preparation happened outside the codebase, typically with the
retired AutoDockTools ``prepare_receptor4.py``, and distributing the
receptor into the campaign folders was ``others/adder.py``'s job
(alongside the engine binaries it needlessly copied everywhere).

This module closes that gap:

* :func:`prepare_protein` imports a protein from a PDB file (or an
  already-prepared PDBQT receptor), cleans it -- waters, non-protein
  HETATM records, alternate locations, chain selection -- and writes a
  rigid receptor PDBQT with AutoDock4 atom types and partial charges
  via Meeko's polymer pipeline,
* :func:`distribute_receptor` copies the prepared receptor into every
  ligand folder of a campaign,
* :class:`ProteinReport` audits every atom that went in, came out, or
  was dropped along the way.

The Meeko step is a lazy optional dependency (``pharmaml[docking]``):
importing this module needs nothing beyond the standard library, and
the PDBQT passthrough mode works without Meeko entirely.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pharmaml.exceptions import DataValidationError, DependencyError, DockingError
from pharmaml.logger import get_logger
from pharmaml.utils.io import ensure_directory

_LOG = get_logger(__name__)

__all__ = [
    "ProteinReport",
    "prepare_protein",
    "distribute_receptor",
    "WATER_RESIDUES",
    "MODIFIED_PROTEIN_RESIDUES",
]

# Residue names recognised as water across the common PDB dialects.
WATER_RESIDUES = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SOL", "SPC"}

# HETATM-recorded residues that are actually part of the polymer
# (selenomethionine, hydroxyproline, phosphorylated residues, ...).
# They are kept so expression-tag-containing constructs still dock.
MODIFIED_PROTEIN_RESIDUES = {
    "MSE", "SEC", "PYL", "CSO", "CSD", "CME", "OCS", "CSX", "CSW",
    "HYP", "PTR", "SEP", "TPO", "MLY", "M3L", "KCX", "LLP", "ACA", "CAS",
    "CAF", "YCM", "NEP", "FME", "CGU", "PCA", "TRQ", "ASX", "GLX",
}


@dataclass(slots=True)
class ProteinReport:
    """Audit of one :func:`prepare_protein` run.

    Attributes
    ----------
    source:
        Input protein file (PDB or PDBQT).
    output_pdbqt:
        Receptor PDBQT that was written.
    mode:
        ``"meeko"`` (PDB cleaned, typed and charged by Meeko) or
        ``"pdbqt"`` (existing PDBQT receptor re-cleaned in place).
    input_atoms / input_hetatms:
        ATOM / HETATM records found in the input.
    waters_removed / hetatm_removed / altloc_removed / hydrogens_removed:
        Records dropped by each cleaning rule.
    chains:
        Chain identifiers kept in the receptor.
    residues:
        Distinct residues (chain, number, insertion code).
    output_atoms:
        ATOM records in the written PDBQT.
    atom_types:
        AutoDock atom type -> count (e.g. ``{"A": 130, "HD": 190, ...}``).
    charge_min / charge_max / charge_mean:
        Range/mean of the partial charges in the PDBQT.
    notes:
        Human-readable remarks accumulated during preparation.
    """

    source: Path
    output_pdbqt: Path
    mode: str
    input_atoms: int = 0
    input_hetatms: int = 0
    waters_removed: int = 0
    hetatm_removed: int = 0
    altloc_removed: int = 0
    hydrogens_removed: int = 0
    chains: list[str] = field(default_factory=list)
    residues: int = 0
    output_atoms: int = 0
    atom_types: dict[str, int] = field(default_factory=dict)
    charge_min: float = 0.0
    charge_max: float = 0.0
    charge_mean: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Flat dictionary of every audit field (JSON-friendly)."""
        return {
            "source": str(self.source),
            "output_pdbqt": str(self.output_pdbqt),
            "mode": self.mode,
            "input_atoms": self.input_atoms,
            "input_hetatms": self.input_hetatms,
            "waters_removed": self.waters_removed,
            "hetatm_removed": self.hetatm_removed,
            "altloc_removed": self.altloc_removed,
            "hydrogens_removed": self.hydrogens_removed,
            "chains": list(self.chains),
            "residues": self.residues,
            "output_atoms": self.output_atoms,
            "atom_types": dict(self.atom_types),
            "charge_min": self.charge_min,
            "charge_max": self.charge_max,
            "charge_mean": self.charge_mean,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        types = "  ".join(f"{t}×{n}" for t, n in sorted(self.atom_types.items()))
        dropped = (
            f"{self.waters_removed} water, "
            f"{self.hetatm_removed} non-protein HETATM, "
            f"{self.altloc_removed} alt-location, "
            f"{self.hydrogens_removed} input H"
        )
        return "\n".join(
            [
                "Protein preparation audit",
                f"  source          : {self.source}",
                f"  mode            : {self.mode}"
                + (" (Meeko polymer pipeline)" if self.mode == "meeko" else " (PDBQT passthrough)"),
                f"  input           : {self.input_atoms} ATOM + {self.input_hetatms} HETATM records",
                f"  removed         : {dropped}",
                f"  chains kept     : {', '.join(self.chains) or '(none)'}",
                f"  residues        : {self.residues}",
                f"  receptor        : {self.output_atoms} atoms -> {self.output_pdbqt.name}",
                f"  atom types      : {types}",
                f"  charges         : [{self.charge_min:+.3f}, {self.charge_max:+.3f}]"
                f" (mean {self.charge_mean:+.3f})",
            ]
            + [f"  note            : {note}" for note in self.notes]
        )


def _parse_resname(line: str) -> str:
    return line[17:20].strip()


def _parse_chain(line: str) -> str:
    return line[21] if len(line) > 21 else " "


def _parse_resseq_key(line: str) -> tuple[str, str, str]:
    """(chain, resseq, icode) key distinguishing residues."""
    chain = _parse_chain(line)
    resseq = line[22:26].strip() if len(line) > 25 else ""
    icode = line[26].strip() if len(line) > 26 else ""
    return (chain, resseq, icode)


def _parse_element(line: str) -> str:
    if len(line) >= 78 and line[76:78].strip():
        return line[76:78].strip()
    name = line[12:16].strip() if len(line) > 16 else ""
    return name[:1] if name else ""


def _audit_pdbqt_lines(lines: list[str]) -> tuple[int, dict[str, int], list[float], int, list[str]]:
    """Count atoms, atom types, charges and residues in PDBQT ATOM lines."""
    types: dict[str, int] = {}
    charges: list[float] = []
    residues: set[tuple[str, str, str]] = set()
    chains: set[str] = set()
    atoms = 0
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atoms += 1
        residues.add(_parse_resseq_key(line))
        chains.add(_parse_chain(line).strip())
        parts = line.split()
        if parts and parts[-1].isalpha():
            types[parts[-1]] = types.get(parts[-1], 0) + 1
        try:
            charges.append(float(line[70:76]))
        except ValueError:
            pass
    return atoms, types, charges, len(residues), sorted(c for c in chains if c)


def _clean_records(
    text: str,
    remove_waters: bool,
    keep_hetatm: bool,
    chain: str | None,
    default_altloc: str | None,
    strip_hydrogens: bool,
) -> tuple[list[str], dict[str, int]]:
    """Filter PDB/PDBQT ATOM/HETATM records by the cleaning rules.

    Returns the kept coordinate lines plus counters of what was dropped.
    Chain-break ``TER`` records are re-emitted after each chain change,
    numbered after the last kept atom.  Multi-model files (NMR ensembles)
    are cut at the first ``ENDMDL`` so only the first model is kept.

    Alternate locations are resolved **per residue**: atoms without an
    altloc letter are conformer-independent and always kept; lettered
    atoms of a residue follow either the residue's first-seen letter
    (``"first"``) or the requested letter -- never a mix of both, which
    would build a chemically impossible hybrid conformer.
    """
    kept: list[str] = []
    counters = {
        "waters_removed": 0,
        "hetatm_removed": 0,
        "altloc_removed": 0,
        "hydrogens_removed": 0,
    }
    residue_letters: dict[tuple[str, str, str], str] = {}
    seen_names: set[tuple[tuple[str, str, str], str]] = set()
    last_chain: str | None = None
    last_serial = 0
    models = sum(1 for l in text.splitlines() if l.startswith("MODEL "))
    for line in text.splitlines():
        if line.startswith("ENDMDL") and models:
            # multi-model (NMR ensemble) file: keep only the first model
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = _parse_resname(line)
        if remove_waters and resname in WATER_RESIDUES:
            counters["waters_removed"] += 1
            continue
        if line.startswith("HETATM") and resname not in MODIFIED_PROTEIN_RESIDUES:
            if not keep_hetatm:
                counters["hetatm_removed"] += 1
                continue
        if chain is not None and _parse_chain(line).strip() != chain:
            continue
        altloc = line[16] if len(line) > 16 else " "
        residue_key = _parse_resseq_key(line)
        atom_key = (residue_key, line[12:16].strip())
        if altloc != " ":
            if default_altloc in (None, "first"):
                chosen = residue_letters.setdefault(residue_key, altloc)
            else:
                chosen = default_altloc
            if altloc != chosen or atom_key in seen_names:
                counters["altloc_removed"] += 1
                continue
        elif atom_key in seen_names:
            # pathological blank duplicate of an already-kept position
            counters["altloc_removed"] += 1
            continue
        seen_names.add(atom_key)
        if strip_hydrogens and _parse_element(line) in {"H", "D"}:
            counters["hydrogens_removed"] += 1
            continue
        current = _parse_chain(line)
        if last_chain is not None and current != last_chain:
            kept.append(f"TER   {last_serial + 1:5d}")
        kept.append(line)
        last_chain = current
        try:
            last_serial = int(line[6:11])
        except ValueError:
            last_serial = len(kept)
    counters["models"] = models
    return kept, counters


def prepare_protein(
    source: str | Path,
    output_pdbqt: str | Path | None = None,
    *,
    remove_waters: bool = True,
    keep_hetatm: bool = False,
    chain: str | None = None,
    default_altloc: str | None = "first",
    strip_hydrogens: bool = True,
) -> ProteinReport:
    """Import a protein and prepare a docking-ready receptor PDBQT.

    Upgrades the protein handling that ``gpfer.py`` took for granted
    (a pre-made ``r.pdbqt``): PDB input is cleaned and converted by
    Meeko's polymer pipeline -- AutoDock4 atom types, Gasteiger partial
    charges and polar hydrogens included.  A ``.pdbqt`` input is treated
    as an already-prepared receptor and only re-filtered.

    Parameters
    ----------
    source:
        Protein file (``.pdb`` or ``.pdbqt``).
    output_pdbqt:
        Receptor destination.  Defaults to ``<source stem>.pdbqt``
        next to the input (``r.pdbqt`` in the docking folder is the
        campaign convention used by :func:`~pharmaml.docking.write_gpf`).
    remove_waters:
        Drop water molecules (HOH/WAT/DOD/...).  Almost always wanted:
        buried waters freeze the electrostatic maps.
    keep_hetatm:
        Keep non-water HETATM records (cofactors, ions, glycans).
        Modified protein residues such as MSE are always kept.
    chain:
        Keep only this chain identifier (e.g. ``"A"``); ``None`` keeps
        all chains.
    default_altloc:
        Alternate-location rule: ``"first"`` keeps the first conformer
        encountered per atom; an explicit letter (``"A"``/``"B"``) keeps
        that conformer.
    strip_hydrogens:
        Remove input hydrogens before conversion.  Meeko re-adds polar
        hydrogens from its residue templates, so input H coordinates
        are not needed.

    Returns
    -------
    ProteinReport
        Full accounting of atoms in, atoms out and cleaning decisions.

    Raises
    ------
    DataValidationError
        Missing file, empty file or unparseable input.
    DependencyError
        PDB input requires Meeko (``pip install 'pharmaml[docking]'``).
    DockingError
        Meeko could not build the polymer (broken residues, unknown
        chemistry, ...).  The message carries Meeko's diagnosis.
    """
    src = Path(source)
    if not src.is_file():
        raise DataValidationError(f"protein file not found: {src}")
    if default_altloc not in (None, "first") and not (
        isinstance(default_altloc, str)
        and len(default_altloc) == 1
        and default_altloc.isalpha()
        and default_altloc.isupper()
    ):
        raise DataValidationError(
            f"default_altloc must be 'first' or a single uppercase letter, "
            f"got {default_altloc!r}"
        )
    if chain is not None:
        chain = chain.strip() or None
    text = src.read_text(encoding="utf-8-sig", errors="ignore")
    if not any(line.startswith(("ATOM", "HETATM")) for line in text.splitlines()):
        raise DataValidationError(f"no ATOM/HETATM records found in {src}")

    src_name = src.stem
    out = Path(output_pdbqt) if output_pdbqt is not None else src.with_suffix(".pdbqt")
    if out.resolve() == src.resolve():
        out = src.with_name(f"{src_name}_clean.pdbqt")
    ensure_directory(out.parent)

    mode = "pdbqt" if src.suffix.lower() == ".pdbqt" else "meeko"
    report = ProteinReport(source=src, output_pdbqt=out, mode=mode)
    report.input_atoms = sum(1 for line in text.splitlines() if line.startswith("ATOM"))
    report.input_hetatms = sum(1 for line in text.splitlines() if line.startswith("HETATM"))

    cleaned, counters = _clean_records(
        text,
        remove_waters=remove_waters,
        keep_hetatm=keep_hetatm,
        chain=chain,
        default_altloc=default_altloc,
        strip_hydrogens=strip_hydrogens and mode == "meeko",
    )
    report.waters_removed = counters["waters_removed"]
    report.hetatm_removed = counters["hetatm_removed"]
    report.altloc_removed = counters["altloc_removed"]
    report.hydrogens_removed = counters["hydrogens_removed"]
    if counters.get("models", 0) > 1:
        report.notes.append(
            f"multi-model file (NMR ensemble): kept MODEL 1 of "
            f"{counters['models']}"
        )
    if not cleaned:
        raise DataValidationError(
            f"all atoms were filtered away from {src} -- check the chain "
            f"selection and HETATM/keep rules"
        )

    if mode == "pdbqt":
        pdbqt_lines = cleaned
        report.notes.append(
            "input was already a PDBQT receptor: re-typed records as-is "
            "(no Meeko step, charges kept)"
        )
    else:
        pdbqt_text = _meeko_receptor_pdbqt("\n".join(cleaned) + "\nTER\nEND\n")
        pdbqt_lines = pdbqt_text.splitlines()

    out.write_text("\n".join(pdbqt_lines) + "\n", encoding="utf-8")

    atoms, types, charges, residues, chains = _audit_pdbqt_lines(pdbqt_lines)
    report.output_atoms = atoms
    report.atom_types = types
    report.residues = residues
    report.chains = chains
    if charges:
        report.charge_min = min(charges)
        report.charge_max = max(charges)
        report.charge_mean = sum(charges) / len(charges)
    if mode == "meeko" and not any(t.startswith("H") for t in types):
        report.notes.append("no polar hydrogens in the receptor -- dockings will lack H-bond donors")

    _LOG.info(
        "Prepared receptor %s (%d atoms, %d residues, %d waters removed)",
        out.name,
        atoms,
        residues,
        report.waters_removed,
    )
    return report


def _meeko_receptor_pdbqt(pdb_text: str) -> str:
    """Run the Meeko polymer pipeline and return rigid-receptor PDBQT text.

    Raises :class:`DependencyError` when Meeko is absent and
    :class:`DockingError` when the polymer cannot be built.
    """
    try:
        from meeko import PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    except ImportError as exc:
        raise DependencyError(
            "prepare_protein requires Meeko and RDKit for PDB input. "
            "Install with:  pip install 'pharmaml[docking]'  (or: meeko rdkit)"
        ) from exc

    try:
        templates = ResidueChemTemplates.create_from_defaults()
        polymer = Polymer.from_pdb_string(pdb_text, templates)
        rigid, flex = PDBQTWriterLegacy.write_from_polymer(polymer)
    except Exception as exc:  # PolymerCreationError and friends
        raise DockingError(
            "Meeko could not build the receptor from this PDB "
            f"({type(exc).__name__}: {exc}). Common causes: broken residues "
            "from unresolved electron density, non-standard chemistry, or "
            "HETATM records the templates do not know. Trim the offending "
            "residues or convert them to standard ones first."
        ) from exc
    if flex:
        _LOG.warning(
            "%d flexible sidechain(s) were requested by the polymer but a "
            "rigid receptor was written (flexible residues are out of scope)",
            len(flex),
        )
    return rigid


def distribute_receptor(
    receptor_path: str | Path,
    campaign_root: str | Path,
    *,
    filename: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Copy a prepared receptor into every ligand folder of a campaign.

    Upgrades ``others/adder.py``, which walked a campaign tree copying
    ``r.pdbqt`` (and the engine executables) into every sub-folder --
    the per-folder layout ``run_autodock4`` and ``write_gpf`` expect.

    Parameters
    ----------
    receptor_path:
        Prepared receptor PDBQT to distribute.
    campaign_root:
        Campaign directory whose immediate sub-directories are ligand
        folders.
    filename:
        Name inside each folder (defaults to the receptor's own name;
        ``"r.pdbqt"`` is the campaign convention).
    overwrite:
        Re-copy when the target already exists (existing copies are
        skipped by default, making the call idempotent).

    Returns
    -------
    list[Path]
        Folders that now contain the receptor (newly copied or
        pre-existing when ``overwrite`` is False).

    Raises
    ------
    DataValidationError
        Missing receptor file or campaign directory.
    """
    src = Path(receptor_path)
    if not src.is_file():
        raise DataValidationError(f"receptor file not found: {src}")
    root = Path(campaign_root)
    if not root.is_dir():
        raise DataValidationError(f"campaign directory not found: {root}")

    name = filename or src.name
    targets: list[Path] = []
    copied = 0
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    for folder in folders:
        target = folder / name
        if target.exists() and not overwrite:
            targets.append(target)
            continue
        shutil.copyfile(src, target)
        targets.append(target)
        copied += 1
    if not folders:
        _LOG.warning("no ligand folders found under %s", root)
    _LOG.info(
        "Receptor %s present in %d/%d ligand folders (%d copied)",
        name,
        len(targets),
        len(folders),
        copied,
    )
    return targets
