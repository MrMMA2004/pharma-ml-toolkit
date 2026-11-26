"""3D-RISM-KH-inspired solvation descriptors (RDKit-backed, lazy import).

Solvation-theory descriptors for QSAR, shaped after the quantities that
a 3D-RISM-KH analysis (three-dimensional Reference Interaction Site
Model with the Kovalenko-Hirata closure) reports for a solute in
water: site distribution peaks, first-shell coordination numbers,
an excluded-volume estimate and a solvation free energy decomposed
into Lennard-Jones and Coulomb (charging) parts.

Full 3D-RISM requires solving the molecular Ornstein-Zernike equation
numerically (grid hours per molecule).  This module computes the SAME
descriptor quantities from a documented *pairwise surrogate* that runs
in milliseconds:

1. the molecule is embedded in 3D (ETKDGv3, fixed seed) and relaxed
   (MMFF94, UFF fallback) — deterministic — with explicit hydrogens
   and Gasteiger partial charges per atom;
2. for every solute atom ``i`` and solvent site ``gamma`` (water:
   O_w / H_w with the correct site multiplicities: 1 O and 2 H per
   molecule, so the surrogate solvent is charge-neutral), the pair
   potential is

   ``u(d) = 4*eps_ij*[(sigma_ij/d)**12 - (sigma_ij/d)**6]
            + 332.06371 * q_i * q_gamma / (eps(d) * d)``

   with Lorentz-Berthelot combining rules (kcal/mol, Angstrom) and a
   smooth dielectric ramp ``eps(d)``: 1 at contact (``d <= 2.8`` A,
   unscreened hydrogen bonds) rising to the bulk-water value 78.4
   over ~4 A — the mean-field stand-in for the many-body screening a
   full RISM solve provides;
3. the pair distribution follows the KH closure branches, *tempered*
   where an isolated pair would diverge:

   * ``u <= 0`` (attractive): ``g = exp(min(-beta*u, K))`` with the
     tempering constant ``K = 2.0`` — real RISM water site peaks stay
     in the ~5-10 range, and ``e**K`` caps the surrogate there;
   * ``u >  0`` (repulsive): ``g = max(1 - beta*u, 0)`` — the KH
     linearised branch, clipped at full exclusion;
4. the free-energy descriptors are potential-energy integrals of the
   separated potentials against that distribution over the structured
   region ``d <= 6`` A (first + second solvation shells; beyond it the
   screened distributions approach bulk and carry no local structure):

   * ``RISM_Glj``   = ``sum_gamma rho_gamma * sum_i int u_LJ * g``
   * ``RISM_Gcoul`` = ``0.5 * sum_gamma rho_gamma * sum_i int u_C * g``
     (electrostatic charging at the textbook linear-response factor)
   * ``RISM_Gsolv`` = ``RISM_Glj + RISM_Gcoul``.

These descriptors are intended as *features for regression/classification*
(e.g. solubility, permeability, hydration-affinity endpoints), not as a
replacement for a production RISM calculation: the values order
molecules chemically sensibly (polar < apolar solvation energy,
H-bond donors raise the H_w coordination; the Gasteiger dipole of
ethanol lands at 1.6 D vs the experimental 1.69 D) but their absolute
energy scale is surrogate, not RISM-calibrated.  All parameters
(solvent model, temperature, radii, combining rules, tempering
constant, screening ramp) are fixed constants recorded in this
module, so the descriptor vector is fully deterministic.

Solvent models (site charges in elementary units, LJ in kcal/mol /
Angstrom, site density in sites/A**3):

* ``'spc'``  — SPC water: O_w q=-0.820, eps=0.1553, sigma=3.166,
  rho=0.0334; H_w q=+0.410, rho=0.0668 (two sites per molecule).
  The H sites carry the soft repulsive LJ of a solute hydrogen
  (eps=0.0157, sigma=2.42): real SPC/TIP3P water places no LJ on H,
  relying on the intramolecular O-H constraint the pairwise surrogate
  cannot express.
* ``'tip3p'`` — TIP3P water: O_w q=-0.834, eps=0.1521, sigma=3.1507;
  H_w q=+0.417 (same surrogate H LJ and site densities).

Examples
--------
>>> from pharmaml.chem.rism import compute_rism_descriptors, RISM_DESCRIPTORS
>>> len(RISM_DESCRIPTORS)                       # the descriptor contract
10
>>> frame = compute_rism_descriptors(["CCO"], progress=False)   # ethanol
>>> list(frame.columns[:3])
['SMILES', 'RISM_Gsolv_kcalmol', 'RISM_Glj_kcalmol']
>>> float(frame["RISM_Ow_cn1"].iloc[0]) > 0.0    # ethanol orders water
True
>>> round(float(frame["RISM_dipole_D"].iloc[0]), 1)   # Gasteiger dipole
1.6
>>> frame.index.name
'original_index'

>>> twin = compute_rism_descriptors(["CCO"], progress=False)
>>> abs(float(frame["RISM_Gsolv_kcalmol"].iloc[0])
...     - float(twin["RISM_Gsolv_kcalmol"].iloc[0])) < 1e-9   # deterministic
True

>>> seen = []                                    # (100x-169) live progress
>>> frame = compute_rism_descriptors(
...     ["CCO", "CCC"], progress=lambda d, t, n: seen.append((d, t, n)))
>>> seen[0], seen[-1]
((0, 2, 'RISM-KH descriptors'), (2, 2, 'RISM-KH descriptors'))

>>> frame = compute_rism_descriptors(["CCO", "not-a-smiles", "CCC"], progress=False)
>>> list(frame.index)                             # invalid row dropped
[0, 2]
>>> compute_rism_descriptors(["!!!"], progress=False)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: No valid SMILES found in the input
>>> compute_rism_descriptors([], progress=False)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: smiles is empty -- pass at least one SMILES string
>>> compute_rism_descriptors(["CCO"], solvent="sea", progress=False)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: solvent must be one of 'spc' or 'tip3p', got 'sea'
>>> compute_rism_descriptors(["CCO"], temperature=-5.0, progress=False)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: temperature must be > 0 in kelvin, got -5.0
>>> compute_rism_descriptors("CCO", progress=False)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: smiles must be a list or pandas.Series of SMILES strings, not a single string (got 'CCO'); wrap it in a list to compute one molecule
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = ["RISM_DESCRIPTORS", "compute_rism_descriptors"]


#: The ten descriptor columns, in output order (after ``SMILES``).
RISM_DESCRIPTORS: tuple[str, ...] = (
    "RISM_Gsolv_kcalmol",
    "RISM_Glj_kcalmol",
    "RISM_Gcoul_kcalmol",
    "RISM_Ow_gmax",
    "RISM_Hw_gmax",
    "RISM_Ow_cn1",
    "RISM_Hw_cn1",
    "RISM_Ow_exclvol_A3",
    "RISM_dipole_D",
    "RISM_q_neg_frac",
)

#: Solvent site definitions: charge (e), LJ epsilon (kcal/mol), LJ
#: sigma (Angstrom) and site number density (sites/A**3).  Water sites
#: are named ``"Ow"`` / ``Hw`` regardless of model; the H sites carry
#: the surrogate repulsive LJ and DOUBLE density (two H per molecule,
#: which keeps the surrogate solvent charge-neutral).
_SOLVENT_SITES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "spc": {
        "Ow": (-0.820, 0.1553, 3.166, 0.0334),
        "Hw": (0.410, 0.0157, 2.42, 0.0668),
    },
    "tip3p": {
        "Ow": (-0.834, 0.1521, 3.1507, 0.0334),
        "Hw": (0.417, 0.0157, 2.42, 0.0668),
    },
}

#: Gas constant in kcal mol^-1 K^-1.
_GAS_CONSTANT_KCAL = 0.0019872

#: Coulomb constant in kcal * Angstrom / (mol * e^2).
_COULOMB_KCAL = 332.06371

#: Solute-element Lennard-Jones parameters (epsilon kcal/mol, sigma A) —
#: OPLS/AMBER-flavoured values for the druglike elements.
_ELEMENT_LJ: dict[str, tuple[float, float]] = {
    "C": (0.066, 3.50),
    "N": (0.170, 3.25),
    "O": (0.170, 3.12),
    "S": (0.250, 3.55),
    "P": (0.200, 3.74),
    "F": (0.061, 3.10),
    "Cl": (0.265, 3.47),
    "Br": (0.330, 3.62),
    "I": (0.400, 3.78),
    "H": (0.0157, 2.42),
    "B": (0.155, 3.52),
    "Si": (0.200, 3.83),
}

#: Tempering constant for the attractive KH branch: the pair ``g`` peak
#: is capped at ``exp(K)`` (see the module docstring).
_TEMPER_K = 2.0

#: Dielectric screening ramp for the Coulomb term: ``eps(d) = 1`` at
#: contact (``d <= _SCREEN_ONSET``) rising smoothly to ``_EPS_BULK``
#: with length scale ``_SCREEN_LENGTH`` (Angstrom).
_SCREEN_ONSET = 2.8
_SCREEN_LENGTH = 4.0
_EPS_BULK = 78.4

#: Pair integrals run over the structured region only (first + second
#: solvation shells, Angstrom).
_STRUCTURED_CUTOFF = 6.0

#: First-shell window rule: anchored at the pair's ``g`` peak, closed
#: where ``g`` falls back within 20% of bulk (``g < 1.2``), capped at
#: this distance (Angstrom) — a per-atom shell, so shells shared
#: between atoms are counted once per atom (pairwise-sum semantics).
_SHELL_CLOSE_LEVEL = 1.2
_SHELL_MAX_RADIUS = 5.0

#: Radial pair grid: start, stop (exclusive) and step, in Angstrom.
_GRID_START, _GRID_STOP, _GRID_STEP = 1.2, 12.0, 0.05

#: Debye conversion: one e*Angstrom dipole = 4.80320 Debye.
_DEBYE_PER_EA = 4.80320

#: Fixed ETKDGv3 embedding seeds (first try, retry).
_EMBED_SEEDS: tuple[int, ...] = (42, 2024)


def _as_smiles_list(smiles: Iterable[str] | pd.Series, arg_name: str) -> list[str]:
    """Materialise a SMILES sequence once; reject a bare string.

    Same contract as :func:`pharmaml.chem.descriptors._as_smiles_list`
    (a bare ``"CCO"`` would otherwise become three single-atom
    "molecules").
    """
    if isinstance(smiles, str):
        preview = smiles if len(smiles) <= 40 else smiles[:40] + "..."
        raise DataValidationError(
            f"{arg_name} must be a list or pandas.Series of SMILES strings, not a "
            f"single string (got {preview!r}); wrap it in a list to compute "
            "one molecule"
        )
    return list(smiles)


def _pair_grid() -> np.ndarray:
    """The shared radial grid (Angstrom), half-open like ``np.arange``."""
    return np.arange(_GRID_START, _GRID_STOP, _GRID_STEP)


def _dielectric_ramp(grid: np.ndarray) -> np.ndarray:
    """Smooth dielectric ``eps(d)``: 1 at contact, ``_EPS_BULK`` far away."""
    excess = np.maximum(grid - _SCREEN_ONSET, 0.0)
    return 1.0 + (_EPS_BULK - 1.0) * (1.0 - np.exp(-excess / _SCREEN_LENGTH))


def _tempered_kh_distribution(potential: np.ndarray, beta: float) -> np.ndarray:
    """Pair site distribution ``g`` from the tempered KH closure branches.

    Attractive region (``u <= 0``): ``g = exp(min(-beta*u, K))`` — the
    KH Boltzmann branch, tempered at ``K`` where an isolated pair would
    diverge.  Repulsive region (``u > 0``): ``g = max(1 - beta*u, 0)``
    — the KH linearised branch, clipped at full exclusion.
    """
    attractive = potential <= 0.0
    g_att = np.exp(np.minimum(-beta * potential, _TEMPER_K))
    g_rep = np.maximum(1.0 - beta * potential, 0.0)
    return np.where(attractive, g_att, g_rep)


def _embed_molecule(mol: Any) -> np.ndarray:
    """Return the 3D conformer coordinates (Angstrom), deterministic.

    ETKDGv3 embedding with fixed seeds (retry once), then best-effort
    force-field relaxation: MMFF94 when the molecule types
    (``MMFFHasAllMoleculeParams``), else UFF, else the raw ETKDG
    geometry.  Raises :class:`ValueError` when both seeds fail (the
    caller's per-molecule guard turns that into a dropped row).
    """
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    params = rdDistGeom.ETKDGv3()
    embedded = False
    for seed in _EMBED_SEEDS:
        # rdkit-stubs types randomSeed as EmbedParameters (a stub bug);
        # the runtime attribute is the integer RNG seed.
        params.randomSeed = seed  # type: ignore[assignment]
        if rdDistGeom.EmbedMolecule(mol, params) == 0:
            embedded = True
            break
    if not embedded:
        raise ValueError("ETKDGv3 embedding failed for both fixed seeds")
    try:
        if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol, maxIters=500)
        elif rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            rdForceFieldHelpers.UFFOptimizeMolecule(mol, maxIters=500)
        # else: keep the raw ETKDG geometry (exotic elements) — the
        # descriptors remain deterministic either way.
    except Exception as exc:  # noqa: BLE001 — RDKit raises a zoo here
        _LOG.debug("force-field relaxation skipped (%s: %s)", type(exc).__name__, exc)
    conformer = mol.GetConformer(0)
    return np.asarray(
        [
            [
                conformer.GetAtomPosition(i).x,
                conformer.GetAtomPosition(i).y,
                conformer.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=np.float64,
    )


def _gasteiger_charges(mol: Any) -> np.ndarray:
    """Gasteiger partial charges per atom (elementary units)."""
    from rdkit.Chem import rdPartialCharges

    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = np.asarray(
        [float(atom.GetDoubleProp("_GasteigerCharge")) for atom in mol.GetAtoms()],
        dtype=np.float64,
    )
    # Parse artefacts (inf/NaN on exotic valences) must not poison the
    # descriptors — clip to a sane charge range instead.
    return np.clip(np.nan_to_num(charges, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0)


def _rism_vector(
    mol: Any,
    sites: Mapping[str, tuple[float, float, float, float]],
    kbt: float,
) -> dict[str, float]:
    """Compute the ten RISM descriptors for one embedded molecule.

    ``sites`` maps site name -> ``(charge, lj_epsilon, lj_sigma,
    site_density)``.
    """
    beta = 1.0 / kbt
    coords = _embed_molecule(mol)
    charges = _gasteiger_charges(mol)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    n_atoms = len(symbols)

    grid = _pair_grid()
    d2dd = 4.0 * np.pi * grid * grid * _GRID_STEP  # radial volume element
    structured = grid <= _STRUCTURED_CUTOFF  # structured-region mask
    dielectric = _dielectric_ramp(grid)

    g_lj = 0.0
    g_coul = 0.0
    gmax: dict[str, float] = {"Ow": 0.0, "Hw": 0.0}
    cn1: dict[str, float] = {"Ow": 0.0, "Hw": 0.0}
    exclvol = 0.0

    for site_name, (q_site, eps_site, sigma_site, rho_site) in sites.items():
        for i in range(n_atoms):
            eps_atom, sigma_atom = _ELEMENT_LJ.get(
                symbols[i], (0.100, 3.30)  # exotic element: modest defaults
            )
            sigma_ij = 0.5 * (sigma_atom + sigma_site)
            eps_ij = float(np.sqrt(eps_atom * eps_site))
            ratio = sigma_ij / grid
            u_lj = 4.0 * eps_ij * (ratio**12 - ratio**6)
            u_coul = _COULOMB_KCAL * charges[i] * q_site / (dielectric * grid)
            u_total = u_lj + u_coul

            g = _tempered_kh_distribution(u_total, beta)

            # Free-energy descriptors: potential-energy integrals of the
            # separated potentials against the tempered distribution of
            # the total one, over the structured region only (the
            # Coulomb part at the linear-response charging factor 1/2).
            g_lj += rho_site * float(np.sum((u_lj * g * d2dd)[structured]))
            g_coul += (
                0.5 * rho_site * float(np.sum((u_coul * g * d2dd)[structured]))
            )

            gmax[site_name] = max(gmax[site_name], float(g.max()))

            # First-shell coordination: window anchored at the pair's g
            # peak, closed where g falls back within 20% of bulk, capped
            # at the shell radius.  (Anchoring at the peak — not at the
            # first g >= 1 crossing — keeps the window open through the
            # gradual wall-side rise, where g sits between 1.0 and 1.2.)
            if float(g.max()) > 1.0:
                peak = int(np.argmax(g))
                closing = np.flatnonzero(g[peak:] < _SHELL_CLOSE_LEVEL)
                cap = int(np.searchsorted(grid, _SHELL_MAX_RADIUS))
                end = peak + int(closing[0]) if closing.size else cap
                end = min(end, cap)
                if end > peak:
                    cn1[site_name] += rho_site * float(
                        np.sum(g[peak:end] * d2dd[peak:end])
                    )

            # Excluded volume (O_w only, heavy atoms only — bound H
            # spheres sit inside their atom's sphere): sphere at the
            # onset of the STERIC (LJ-only) distribution — the total
            # potential's onset would drift outward for like-charge
            # pairs (O...Ow), where mild electrostatic repulsion keeps
            # g < 0.5 far into the screened tail.
            if site_name == "Ow" and symbols[i] != "H":
                g_steric = _tempered_kh_distribution(u_lj, beta)
                reached = g_steric >= 0.5
                if reached.any():
                    onset = grid[int(np.argmax(reached))]
                    exclvol += 4.0 * np.pi / 3.0 * onset**3

    # Molecular dipole from Gasteiger charges and 3D coordinates (Debye).
    dipole_ea = float(np.linalg.norm(charges @ coords)) if n_atoms else 0.0

    # Charge hydration asymmetry: share of negative site charge.
    abs_q = float(np.abs(charges).sum())
    neg_frac = float(-charges[charges < 0].sum() / abs_q) if abs_q > 0 else 0.0

    return {
        "RISM_Gsolv_kcalmol": g_lj + g_coul,
        "RISM_Glj_kcalmol": g_lj,
        "RISM_Gcoul_kcalmol": g_coul,
        "RISM_Ow_gmax": gmax["Ow"],
        "RISM_Hw_gmax": gmax["Hw"],
        "RISM_Ow_cn1": cn1["Ow"],
        "RISM_Hw_cn1": cn1["Hw"],
        "RISM_Ow_exclvol_A3": exclvol,
        "RISM_dipole_D": dipole_ea * _DEBYE_PER_EA,
        "RISM_q_neg_frac": neg_frac,
    }


def compute_rism_descriptors(
    smiles: pd.Series | list[str],
    solvent: str = "spc",
    temperature: float = 298.15,
    progress: bool | Callable[..., Any] | None = True,
) -> pd.DataFrame:
    """Compute 3D-RISM-KH-inspired solvation descriptors for SMILES.

    Parameters
    ----------
    smiles:
        Iterable of SMILES strings (a DataFrame column works directly).
        A bare string is rejected — wrap it in a list.
    solvent:
        Solvent site model: ``'spc'`` or ``'tip3p'`` (see the module
        docstring for the site parameters).
    temperature:
        Absolute temperature in kelvin (drives ``kT`` and the closure
        inverse temperature).  Must be positive.
    progress:
        (100x-169) the parameter finally does what this docstring
        always promised: ``True`` wraps the molecule loop in a tqdm
        bar (batches of 20+ molecules only), a CALLABLE receives
        ``progress(done, total, note)`` per molecule (the GUI's
        determinate-bar convention), and ``False``/``None`` stays
        silent.  Before the fix the flag was accepted and ignored --
        the "tqdm doesn't work" report.

    Returns
    -------
    pandas.DataFrame
        One row per *valid* molecule, indexed by the original position
        (index name ``original_index``), with a ``SMILES`` column
        followed by the ten :data:`RISM_DESCRIPTORS` columns.  Invalid
        SMILES are dropped with a warning naming their input index;
        molecules whose 3D embedding fails are dropped the same way
        (both ETKDG seeds exhausted).

    Raises
    ------
    DependencyError
        If RDKit is not installed.
    DataValidationError
        If ``smiles`` is a bare string or empty, ``solvent`` is unknown,
        ``temperature`` is not positive, no SMILES parses to a valid
        molecule, or descriptor computation fails for every valid
        molecule.

    Notes
    -----
    Determinism: the ETKDGv3 embedding uses fixed seeds and the whole
    pipeline has no other stochastic step, so repeated calls on the
    same input return identical descriptor values.

    Examples
    --------
    >>> frame = compute_rism_descriptors(["CCO"], progress=False)
    >>> list(frame.columns) == ["SMILES", *RISM_DESCRIPTORS]
    True
    """
    smiles_list = _as_smiles_list(smiles, "smiles")
    if not smiles_list:
        raise DataValidationError("smiles is empty -- pass at least one SMILES string")
    if solvent not in _SOLVENT_SITES:
        raise DataValidationError(
            f"solvent must be one of 'spc' or 'tip3p', got {solvent!r}"
        )
    if not np.isfinite(temperature) or float(temperature) <= 0:
        raise DataValidationError(f"temperature must be > 0 in kelvin, got {temperature!r}")

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for 3D-RISM-KH descriptor computation. "
            "From the toolkit package root, run:  python -m pip install -e '.[chem]'"
        ) from exc

    kbt = _GAS_CONSTANT_KCAL * float(temperature)
    sites = _SOLVENT_SITES[solvent]

    # (100x-169) the progress plumbing, finally wired: True -> a tqdm
    # bar over the molecule loop (large batches only, the docstring's
    # own wording); a callable -> (done, total, note) per molecule on
    # THIS thread (the GUI feeds it straight into its queue); the
    # pre-report + final-report pattern keeps the count monotone and
    # terminal at (total, total) whatever the drop paths do.
    total = len(smiles_list)
    iterator: Iterable[tuple[int, str]] = enumerate(smiles_list)
    if progress is True and total >= 20:
        try:
            from tqdm import tqdm

            iterator = enumerate(
                tqdm(smiles_list, desc="RISM-KH", unit="mol")  # type: ignore[assignment]
            )
        except ImportError:
            iterator = enumerate(smiles_list)
    report_progress = progress if callable(progress) else None

    rows: list[dict[str, float]] = []
    valid_idx: list[int] = []
    failures: list[str] = []
    n_parsed = 0
    done = 0
    for idx, smi in iterator:
        if report_progress is not None:
            report_progress(done, total, "RISM-KH descriptors")
        done += 1
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None
        if mol is None or mol.GetNumAtoms() == 0:
            failures.append(f"index {idx}: {smi!r}")
            continue
        n_parsed += 1
        try:
            # Explicit hydrogens: ETKDG embedding, MMFF/UFF optimisation
            # and the solute-site charge model all need real H atoms —
            # a SMILES-parsed mol keeps them implicit.
            rows.append(_rism_vector(Chem.AddHs(mol), sites, kbt))
        except Exception as exc:  # noqa: BLE001 — RDKit raises a zoo per molecule
            _LOG.warning(
                "Dropping molecule at index %d (%r): RISM computation failed "
                "(%s: %s)",
                idx,
                smi,
                type(exc).__name__,
                exc,
            )
            failures.append(f"index {idx}: {smi!r}")
            continue
        valid_idx.append(idx)
    if report_progress is not None:
        report_progress(total, total, "RISM-KH descriptors")
    if failures:
        _LOG.warning("Dropping %d molecules: %s", len(failures), failures[:5])
    if n_parsed == 0:
        raise DataValidationError("No valid SMILES found in the input")
    if not rows:
        raise DataValidationError(
            "3D-RISM-KH descriptor computation failed for every valid molecule"
        )

    frame = pd.DataFrame(rows, columns=list(RISM_DESCRIPTORS))
    frame.insert(0, "SMILES", [str(smiles_list[i]) for i in valid_idx])
    frame.index = pd.Index(valid_idx, name="original_index")
    _LOG.info(
        "Computed %d 3D-RISM-KH solvation descriptors for %d molecules "
        "(solvent=%s, T=%.2f K)",
        len(RISM_DESCRIPTORS),
        len(frame),
        solvent,
        float(temperature),
    )
    return frame


def _rism_family_columns() -> list[str]:
    """Feature-factory column names for the ``rism`` family.

    The ten descriptors, lower-cased without their ``RISM_`` prefix and
    re-prefixed with the family prefix ``rism_`` (so the frame
    vocabulary stays one flat prefix namespace): ``rism_gsolv_kcalmol``,
    ``rism_ow_cn1``, ...
    """
    return [f"rism_{name.removeprefix('RISM_').lower()}" for name in RISM_DESCRIPTORS]


def _rism_computer() -> Any:
    """Feature-factory computer: mol -> ten-value RISM vector (SPC, 298 K).

    The molecule must already carry explicit hydrogens (the feature
    factory parses with ``Chem.MolFromSmiles`` and adds them here).
    """
    sites = _SOLVENT_SITES["spc"]
    kbt = _GAS_CONSTANT_KCAL * 298.15

    def rism_vector(mol: Any) -> np.ndarray:
        from rdkit import Chem

        values = _rism_vector(Chem.AddHs(mol), sites, kbt)
        return np.asarray([values[name] for name in RISM_DESCRIPTORS], dtype=float)

    return rism_vector
