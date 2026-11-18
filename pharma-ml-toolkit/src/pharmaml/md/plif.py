"""Protein–ligand interaction fingerprints — ProLIF's geometric core.

``chemosim-lab/ProLIF`` scores docking poses by the *interactions*
they make: hydrophobic contacts, hydrogen bonds (both directions),
ionic pairs and π-stacking, each a geometric criterion over typed
atom coordinates.  This module is that core for explicit
coordinates — the caller assigns each ligand/protein point one of the
ProLIF interaction types and the fingerprint does the rest:

* ``HYD`` — hydrophobic;
* ``HBD`` / ``HBA`` — hydrogen-bond donor / acceptor;
* ``POS`` / ``NEG`` — cation / anion;
* ``PI`` — aromatic ring centroid (with optional ring normals for the
  stacking-orientation test).

Distances are the ProLIF defaults (hydrophobic 4.5 Å, H-bond 3.5 Å,
ionic 4.0 Å, π/π centroid 5.5 Å).  π-stacking, given ligand *and*
protein ring normals, additionally requires ``|cos θ| ≥ 0.94``
(parallel or anti-parallel); without normals the criterion honestly
degrades to the centroid distance, as documented.

* :func:`interaction_fingerprint` — the whole pose →
  :class:`InteractionFingerprint`.

Examples
--------
>>> lig = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
>>> ltypes = ["HYD", "HBD", "POS", "PI"]
>>> prot = np.array([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [16.0, 0.0, 0.0]])
>>> ptypes = ["HYD", "HBA", "NEG", "PI"]
>>> fp = interaction_fingerprint(lig, ltypes, prot, ptypes)
>>> (fp.counts["hydrophobic"], fp.counts["hbond_donor"], fp.counts["ionic"])
(1, 1, 1)
>>> fp.counts["pi_stack"]  # distance-only (no normals passed)
1
>>> list(fp.bits)
[True, True, False, True, True]
>>> perpendicular = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
>>> upright = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
>>> strict = interaction_fingerprint(
...     lig, ltypes, prot, ptypes, ligand_normals=upright, protein_normals=perpendicular
... )
>>> strict.counts["pi_stack"]
0
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pharmaml.exceptions import MDAnalysisError

__all__ = [
    "INTERACTION_CUTOFFS",
    "INTERACTION_ORDER",
    "INTERACTION_TYPES",
    "PI_COSINE_MIN",
    "InteractionFingerprint",
    "interaction_fingerprint",
]

INTERACTION_TYPES = ("HBD", "HBA", "HYD", "POS", "NEG", "PI")
"""ProLIF's interaction-capable atom types."""

INTERACTION_CUTOFFS: dict[str, float] = {
    "hydrophobic": 4.5,
    "hbond": 3.5,
    "ionic": 4.0,
    "pi": 5.5,
}
"""Default distance criteria (Å) per interaction family."""

INTERACTION_ORDER = ("hydrophobic", "hbond_donor", "hbond_acceptor", "ionic", "pi_stack")
"""Fingerprint bit order (ProLIF's canonical ordering)."""

PI_COSINE_MIN = 0.94
"""Minimum |cos| between ring normals for a π-stack (parallel/anti)."""


class InteractionFingerprint:
    """One pose's interaction record: counts, bits and residue pairs.

    ``bits`` follows :data:`INTERACTION_ORDER`; ``pairs`` carries the
    ``(ligand_index, protein_index)`` list behind every count so a UI
    can draw the contacts, not just count them.
    """

    def __init__(
        self,
        pairs: dict[str, list[tuple[int, int]]],
        *,
        ligand_labels: Sequence[str] | None = None,
        protein_labels: Sequence[str] | None = None,
    ) -> None:
        self.pairs = {key: list(pairs.get(key, [])) for key in INTERACTION_ORDER}
        self.counts = {key: len(self.pairs[key]) for key in INTERACTION_ORDER}
        self.bits = np.array(
            [self.counts[key] > 0 for key in INTERACTION_ORDER], dtype=bool
        )
        self._ligand_labels = list(ligand_labels) if ligand_labels is not None else None
        self._protein_labels = (
            list(protein_labels) if protein_labels is not None else None
        )

    def __len__(self) -> int:
        """Number of distinct interaction families present."""
        return int(self.bits.sum())

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return f"InteractionFingerprint({self.counts})"

    def to_dict(self) -> dict[str, object]:
        """JSON-ready copy for bridge payloads."""
        return {
            "counts": dict(self.counts),
            "bits": [bool(b) for b in self.bits],
            "pairs": {k: [list(p) for p in v] for k, v in self.pairs.items()},
        }


def _normals_stack(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two ring normals are (anti)parallel enough to stack."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return True  # a missing orientation never blocks the centroid test
    cosine = float(np.dot(a, b)) / (na * nb)
    return abs(np.clip(cosine, -1.0, 1.0)) >= PI_COSINE_MIN


def _typed_coords(
    coords: np.ndarray, types: Sequence[str], name: str
) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(coords, dtype=float)
    type_list = list(types)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise MDAnalysisError(
            f"{name} coords must have shape (n_points, 3), got {arr.shape}"
        )
    if len(type_list) != arr.shape[0]:
        raise MDAnalysisError(
            f"{name} types has {len(type_list)} entries but coords carry "
            f"{arr.shape[0]}"
        )
    bad = sorted({t for t in type_list if t not in INTERACTION_TYPES})
    if bad:
        raise MDAnalysisError(
            f"unknown {name} interaction type(s) {bad} — supported: "
            f"{list(INTERACTION_TYPES)}"
        )
    return arr, type_list


def interaction_fingerprint(
    ligand_coords: np.ndarray,
    ligand_types: Sequence[str],
    protein_coords: np.ndarray,
    protein_types: Sequence[str],
    *,
    cutoffs: dict[str, float] | None = None,
    ligand_normals: np.ndarray | None = None,
    protein_normals: np.ndarray | None = None,
) -> InteractionFingerprint:
    """Compute a pose's protein–ligand interaction fingerprint.

    Hydrophobic pairs (``HYD``–``HYD``), H-bonds in both directions
    (``HBD`` ligand → ``HBA`` protein counts as ``hbond_donor``; the
    reverse as ``hbond_acceptor``), ionic pairs (``POS``↔``NEG``) and
    π-stacks (``PI`` centroids within range, orientation-checked only
    when *both* normal arrays are supplied).

    >>> lig = np.array([[0.0, 0.0, 0.0]])
    >>> interaction_fingerprint(
    ...     lig, ["HYD"], np.array([[20.0, 0.0, 0.0]]), ["HYD"]
    ... ).counts["hydrophobic"]
    0
    """
    lig, lig_types = _typed_coords(ligand_coords, ligand_types, "ligand")
    prot, prot_types = _typed_coords(protein_coords, protein_types, "protein")
    rules = dict(INTERACTION_CUTOFFS)
    if cutoffs is not None:
        for key, value in cutoffs.items():
            if key not in rules:
                raise MDAnalysisError(
                    f"unknown cutoff key {key!r} — supported: {sorted(rules)}"
                )
            if value <= 0:
                raise MDAnalysisError(f"cutoff for {key!r} must be positive")
            rules[key] = float(value)

    have_lig_normals = ligand_normals is not None
    have_prot_normals = protein_normals is not None
    if have_lig_normals != have_prot_normals:
        raise MDAnalysisError(
            "pass both ligand_normals and protein_normals, or neither"
        )
    lig_n = (
        None
        if ligand_normals is None
        else np.asarray(ligand_normals, dtype=float)
    )
    prot_n = (
        None
        if protein_normals is None
        else np.asarray(protein_normals, dtype=float)
    )
    if (
        lig_n is not None
        and prot_n is not None
        and (lig_n.shape != lig.shape or prot_n.shape != prot.shape)
    ):
        raise MDAnalysisError(
            "normals must match their coordinate arrays' shapes"
        )

    pairs: dict[str, list[tuple[int, int]]] = {key: [] for key in INTERACTION_ORDER}
    for i, ltype in enumerate(lig_types):
        for j, ptype in enumerate(prot_types):
            delta = lig[i] - prot[j]
            distance = float(np.sqrt(delta @ delta))
            if ltype == "HYD" and ptype == "HYD" and distance <= rules["hydrophobic"]:
                pairs["hydrophobic"].append((i, j))
            elif ltype == "HBD" and ptype == "HBA" and distance <= rules["hbond"]:
                pairs["hbond_donor"].append((i, j))
            elif ltype == "HBA" and ptype == "HBD" and distance <= rules["hbond"]:
                pairs["hbond_acceptor"].append((i, j))
            elif {ltype, ptype} == {"POS", "NEG"} and distance <= rules["ionic"]:
                pairs["ionic"].append((i, j))
            elif (
                ltype == "PI"
                and ptype == "PI"
                and distance <= rules["pi"]
                and (
                    lig_n is None
                    or prot_n is None
                    or _normals_stack(lig_n[i], prot_n[j])
                )
            ):
                pairs["pi_stack"].append((i, j))
    return InteractionFingerprint(pairs)
