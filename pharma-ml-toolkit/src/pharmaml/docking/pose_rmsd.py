"""DiffDock's pose analysis, ported to the toolkit's stack.

The instruction (round 100x-164): *"Check the contents and files in
this GitHub link completely and one by one and add the missing items to
playground.py: github.com/gcorso/DiffDock."*  The repository was cloned
and walked file by file (the DiffDock-L ``main`` layout: ``utils/``,
``datasets/``, ``models/``, ``confidence/``, the vendored ``spyrmsd/``
and ``evaluate.py``).  The diffusion model itself (torch + e3nn +
ESMFold + trained checkpoints) cannot and should not land in this
RDKit/numpy toolkit -- what *does* land is every piece of pose
mathematics around it that a docking campaign needs:

* **the Kabsch superposition** (``utils/geometry.py::
  rigid_transform_Kabsch_3D_torch``) -- the numpy SVD port with the
  reflection fix, exposed as :func:`kabsch_transform` /
  :func:`kabsch_rmsd`;
* **the QCP RMSD** (Theobald's quaternion characteristic polynomial,
  ``spyrmsd/qcp.py``) -- the closed-form optimal-alignment RMSD without
  an SVD, :func:`qcp_rmsd`, Newton root with the eigenvalue fallback;
* **the Hungarian RMSD** (``spyrmsd/hungarian.py``) --
  :func:`hungarian_rmsd`, per-element linear assignment of atoms when
  the atom ordering is unknown;
* **the symmetry-corrected RMSD** (``spyrmsd/rmsd.py::symmrmsd`` --
  the metric DiffDock's own evaluation uses) --
  :func:`symmetry_corrected_rmsd`: enumerate the molecular graph's
  automorphisms (RDKit ``GetSubstructMatches(mol, mol,
  uniquify=False)`` -- element *and bond-order* aware, stricter than
  spyrmsd's plain VF2 over element-labelled graphs, so phenyl ring
  flips and carboxylate O-swaps are counted as the same pose), take the
  minimum RMSD over the automorphism permutations, cache the
  permutations across a whole pose list (spyrmsd's ``cache=True``),
  optionally return the winning permutation;
* **the torsion machinery** (``utils/torsion.py``) --
  :func:`torsion_bonds` (rotatable bonds by DiffDock's own rule: a
  bond whose removal disconnects the graph *and* whose smaller side
  holds more than one atom), :func:`rotatable_masks`,
  :func:`dihedrals`, :func:`perturb_torsions` (the exact
  ``modify_conformer_torsion_angles`` rotation of the smaller side
  about the bond axis) and :func:`randomize_pose` -- DiffDock's
  ``randomize_position`` initial state: uniform random torsions in
  ``[-pi, pi]``, a uniform random rotation, the centroid translated to
  a given centre (the receptor centroid in DiffDock; any point here);
* **the pose-evaluation metrics** (``evaluate.py``) --
  :func:`pose_evaluation_report`: mean/median/percentile RMSD, the
  classic ``< 2 A`` / ``< 5 A`` success fractions, centroid-distance
  stats, top-k best-of-N, the self-clash check (DiffDock's
  ``topk_self_intersect_fraction``: poses whose minimum intra-ligand
  atom distance drops under 0.4 A), plus greedy RMSD pose clustering
  (:func:`pose_clusters`) -- the confidence model's ranking job in
  DiffDock, approximated by cluster sizes;
* **the pocket-extraction logic** (``datasets/moad.py::get_by_name``
  + ``utils/utils.py::crop_beyond``) -- :func:`extract_pocket`: the
  residues within ``residue_cutoff`` (A) of any ligand atom, either the
  residues alone or **whole chains** (MOAD's ``chain_cutoff``
  convention -- a chain stays if *any* of its residues is in the
  pocket); :func:`crop_beyond`: the residue-level 20 A crop DiffDock
  applies before featurising.

Everything is pure numpy/scipy/RDKit -- no torch, no e3nn, no model
weights -- deterministic under the package's pinned seeds.

Examples
--------
>>> import numpy as np
>>> from pharmaml.docking.pose_rmsd import kabsch_rmsd, plain_rmsd
>>> square = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
>>> shifted = square + np.array([5.0, -3.0, 2.0])
>>> round(plain_rmsd(square, shifted), 6)     # the rigid shift, in place
6.164414
>>> round(kabsch_rmsd(square, shifted), 9)    # aligned away
0.0

A rotation is aligned away too (the Kabsch point):

>>> theta = np.pi / 2.0
>>> rot = np.array([[np.cos(theta), -np.sin(theta), 0],
...                 [np.sin(theta),  np.cos(theta), 0],
...                 [0, 0, 1]])
>>> round(kabsch_rmsd(square, square @ rot.T), 9)
0.0

DiffDock's torsion rule -- butane has exactly ONE rotatable bond (the
terminal C-C bonds hold single-atom sides, too small to rotate):

>>> from rdkit import Chem
>>> from pharmaml.docking.pose_rmsd import torsion_bonds
>>> butane = Chem.MolFromSmiles("CCCC")
>>> torsion_bonds(butane)
[(1, 2)]

The pocket extraction -- a two-residue "protein" around one ligand
atom:

>>> pdb_text = (
...     "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
...     "           N  \\n"
...     "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00  0.00"
...     "           C  \\n"
...     "ATOM      3  N   VAL B   1      50.000   0.000   0.000  1.00  0.00"
...     "           N  \\n")
>>> from pharmaml.docking.pose_rmsd import extract_pocket
>>> report = extract_pocket(pdb_text, ligand_coords=[[1.0, 0.0, 0.0]],
...     residue_cutoff=5.0, keep_whole_chains=False)
>>> report.residues
[('A', '1', ''), ('A', '2', '')]
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment, newton
from scipy.spatial.transform import Rotation

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from .receptor import _parse_resseq_key

_LOG = get_logger(__name__)

__all__ = [
    "AutomorphismCache",
    "PocketReport",
    "best_pose_rmsd",
    "centroid_distance",
    "crop_beyond",
    "dihedrals",
    "extract_pocket",
    "hungarian_rmsd",
    "kabsch_rmsd",
    "kabsch_transform",
    "min_intra_distance",
    "perturb_torsions",
    "plain_rmsd",
    "pose_clusters",
    "pose_evaluation_report",
    "pose_rmsd_matrix",
    "qcp_rmsd",
    "randomize_pose",
    "rmsd_wrapper",
    "rotatable_masks",
    "symmetry_corrected_rmsd",
    "torsion_bonds",
]


def _as_coords(coords: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    """Validate and normalise an (N, 3) coordinate array."""
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise DataValidationError(
            f"{name} must be a non-empty (N, 3) array of xyz coordinates, got "
            f"shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} holds non-finite coordinates")
    return array


# ---------------------------------------------------------------------------
# The rigid-alignment mathematics (utils/geometry.py + spyrmsd)
# ---------------------------------------------------------------------------


def plain_rmsd(
    coords_a: Sequence[Sequence[float]] | np.ndarray,
    coords_b: Sequence[Sequence[float]] | np.ndarray,
    center: bool = False,
) -> float:
    """The plain, no-superposition RMSD (``spyrmsd.rmsd``).

    ``sqrt(mean(|a_i - b_i|^2))`` atom-for-atom; ``center=True``
    subtracts each set's centroid first (spyrmsd's ``center`` flag --
    translation only, no rotation).

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> b = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> round(plain_rmsd(a, b), 6)
    1.0
    """

    a = _as_coords(coords_a, "coords_a")
    b = _as_coords(coords_b, "coords_b")
    if len(a) != len(b):
        raise DataValidationError(
            f"plain_rmsd needs equal atom counts, got {len(a)} and {len(b)}"
        )
    if center:
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def centroid_distance(
    coords_a: Sequence[Sequence[float]] | np.ndarray,
    coords_b: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Euclidean distance between the two geometries' centroids.

    One of DiffDock's ``evaluate.py`` metrics (the pocket-level success
    criterion, ``% centroid < 2 A``).
    """

    a = _as_coords(coords_a, "coords_a")
    b = _as_coords(coords_b, "coords_b")
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def kabsch_transform(
    coords_a: Sequence[Sequence[float]] | np.ndarray,
    coords_b: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The Kabsch rigid transformation superposing A onto B.

    The exact port of DiffDock's ``rigid_transform_Kabsch_3D_torch``
    (``utils/geometry.py``): centroids, the covariance ``H = A B^T``,
    SVD, the reflection fix ``det(R) < 0 -> flip the last V row``, and
    the translation ``t = -R @ centroid_A + centroid_B``.

    Returns
    -------
    ``(R, t)`` with ``B ~ (A @ R.T) + t`` (row-vector convention).

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> rot, t = kabsch_transform(a, a + 2.0)
    >>> bool(np.allclose(a @ rot.T + t, a + 2.0))
    True
    """

    a = _as_coords(coords_a, "coords_a")
    b = _as_coords(coords_b, "coords_b")
    if len(a) != len(b):
        raise DataValidationError(
            f"kabsch_transform needs equal atom counts, got {len(a)} and {len(b)}"
        )
    if len(a) < 3:
        raise DataValidationError(
            f"the Kabsch algorithm needs at least 3 points, got {len(a)}"
        )
    centroid_a = a.mean(axis=0)
    centroid_b = b.mean(axis=0)
    am = a - centroid_a
    bm = b - centroid_b
    h = am.T @ bm
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        # the reflection fix: flip the last row of vt (DiffDock's
        # diag(1, 1, -1) correction)
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = -rotation @ centroid_a + centroid_b
    return rotation, translation


def kabsch_rmsd(
    coords_a: Sequence[Sequence[float]] | np.ndarray,
    coords_b: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """The RMSD after the optimal Kabsch superposition.

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    >>> round(kabsch_rmsd(a, a * 2.0), 6)      # scaling is NOT rigid
    0.707107
    """

    a = _as_coords(coords_a, "coords_a")
    b = _as_coords(coords_b, "coords_b")
    if len(a) != len(b):
        raise DataValidationError(
            f"kabsch_rmsd needs equal atom counts, got {len(a)} and {len(b)}"
        )
    if len(a) < 3:
        # 1-2 points: translation-only alignment
        return plain_rmsd(a, b, center=True)
    rotation, translation = kabsch_transform(a, b)
    fitted = a @ rotation.T + translation
    return float(np.sqrt(np.mean(np.sum((fitted - b) ** 2, axis=1))))


def _k_matrix(m: np.ndarray) -> np.ndarray:
    """spyrmsd's 4x4 symmetric key matrix from the 3x3 inner product."""
    s_xx, s_xy, s_xz = m[0]
    s_yx, s_yy, s_yz = m[1]
    s_zx, s_zy, s_zz = m[2]
    return np.array(
        [
            [s_xx + s_yy + s_zz, s_yz - s_zy, s_zx - s_xz, s_xy - s_yx],
            [s_yz - s_zy, s_xx - s_yy - s_zz, s_xy + s_yx, s_zx + s_xz],
            [s_zx - s_xz, s_xy + s_yx, -s_xx + s_yy - s_zz, s_yz + s_zy],
            [s_xy - s_yx, s_zx + s_xz, s_yz + s_zy, -s_xx - s_yy + s_zz],
        ]
    )


def qcp_rmsd(
    coords_a: Sequence[Sequence[float]] | np.ndarray,
    coords_b: Sequence[Sequence[float]] | np.ndarray,
    atol: float = 1e-9,
) -> float:
    """The QCP RMSD (Theobald 2005, ``spyrmsd/qcp.py`` verbatim).

    The optimal-alignment RMSD from the largest eigenvalue of the
    quaternion key matrix -- no SVD, one quartic root solved by Newton
    iteration with the explicit-eigenvalue fallback.  Equal to
    :func:`kabsch_rmsd` to machine precision; both are shipped because
    spyrmsd uses QCP internally (and it is the faster route on large
    pose batches).

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    >>> b = a + np.array([3.0, 4.0, 0.0])
    >>> round(qcp_rmsd(a, b), 9)
    0.0
    """

    a = _as_coords(coords_a, "coords_a")
    b = _as_coords(coords_b, "coords_b")
    if len(a) != len(b):
        raise DataValidationError(
            f"qcp_rmsd needs equal atom counts, got {len(a)} and {len(b)}"
        )
    am = a - a.mean(axis=0)
    bm = b - b.mean(axis=0)
    g_a = float(np.trace(am.T @ am))
    g_b = float(np.trace(bm.T @ bm))
    m = bm.T @ am  # spyrmsd's M = B^T A
    k = _k_matrix(m)
    c2 = -2.0 * float(np.trace(m.T @ m))
    c1 = -8.0 * float(np.linalg.det(m))
    c0 = float(np.linalg.det(k))

    def quartic(x: float) -> float:
        return x**4 + c2 * x**2 + c1 * x + c0

    def quartic_prime(x: float) -> float:
        return 4.0 * x**3 + 2.0 * c2 * x + c1

    try:
        lam = float(newton(quartic, (g_a + g_b) / 2.0, fprime=quartic_prime))
        if not math.isfinite(lam):
            raise RuntimeError("non-finite root")
    except (RuntimeError, OverflowError):  # pragma: no cover - rare fallback
        lam = float(np.max(np.linalg.eigvalsh(k)))

    n = len(a)
    s = (g_a + g_b - 2.0 * lam) / n
    if abs(s) < atol:
        return 0.0
    return float(math.sqrt(max(s, 0.0)))


def hungarian_rmsd(
    coords_a: Sequence[Sequence[float]] | np.ndarray,
    coords_b: Sequence[Sequence[float]] | np.ndarray,
    atomicnums_a: Sequence[int],
    atomicnums_b: Sequence[int],
) -> float:
    """The Hungarian-assignment RMSD (``spyrmsd/hungarian.py``).

    When the atom *ordering* is unknown but the element multiset
    matches, solve the per-element linear assignment (scipy
    ``linear_sum_assignment`` on squared-distance cost matrices) and
    return the unaligned RMSD of the best assignment.  The symmetry
    route (:func:`symmetry_corrected_rmsd`) is the same idea with the
    assignment set restricted to graph automorphisms -- tighter and
    chemically aware; this function is the ordering-free baseline.

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([[0.0, 0, 0], [1, 0, 0]])
    >>> b = np.array([[1.0, 0, 0], [0, 0, 0]])     # same two atoms, swapped
    >>> hungarian_rmsd(a, b, [6, 8], [8, 6])
    0.0
    """

    a = _as_coords(coords_a, "coords_a")
    b = _as_coords(coords_b, "coords_b")
    nums_a = list(atomicnums_a)
    nums_b = list(atomicnums_b)
    if len(a) != len(nums_a) or len(b) != len(nums_b):
        raise DataValidationError("atomic number lists must match the coordinates")
    if sorted(nums_a) != sorted(nums_b):
        raise DataValidationError(
            "hungarian_rmsd needs matching element multisets "
            f"(got {sorted(nums_a)} vs {sorted(nums_b)})"
        )
    total = 0.0
    assigned = 0
    for element in sorted(set(nums_a)):
        idx_a = [i for i, num in enumerate(nums_a) if num == element]
        idx_b = [i for i, num in enumerate(nums_b) if num == element]
        cost = np.sum((a[idx_a][:, None, :] - b[idx_b][None, :, :]) ** 2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        total += float(cost[rows, cols].sum())
        assigned += len(rows)
    return float(math.sqrt(total / assigned))


# ---------------------------------------------------------------------------
# Symmetry-corrected RMSD (spyrmsd/rmsd.py::symmrmsd)
# ---------------------------------------------------------------------------

#: Cap on the automorphism enumeration (large symmetric species such as
#: cubane explode combinatorially; spyrmsd runs unbounded and relies on
#: DiffDock's external 10-second SIGALRM -- the toolkit caps instead).
_MAX_AUTOMORPHISMS = 200


def _automorphism_permutations(mol: Any) -> list[tuple[int, ...]]:
    """Every automorphism of the molecular graph (element + bond order).

    RDKit's ``GetSubstructMatches(mol, mol, uniquify=False)`` walks the
    full match set; the identity is always first (RDKit's ordering),
    which keeps degenerate comparisons on the fast path.  Capped at
    :data:`_MAX_AUTOMORPHISMS`.
    """
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False)
    permutations: list[tuple[int, ...]] = []
    for match in matches:
        if len(permutations) >= _MAX_AUTOMORPHISMS:
            _LOG.warning(
                "automorphism enumeration capped at %d (a highly symmetric "
                "molecule) -- the symmetry-corrected RMSD may overestimate",
                _MAX_AUTOMORPHISMS,
            )
            break
        permutations.append(tuple(match))
    return permutations


class AutomorphismCache:
    """spyrmsd's pose-list economy: enumerate the automorphisms once.

    DiffDock compares one reference against *many* poses of the same
    molecule -- the permutation list is identical every time, so it is
    computed on first sight and reused (spyrmsd's ``cache=True``
    parameter, here a per-instance dictionary keyed by ``id(mol)``).

    Example
    -------
    >>> from rdkit import Chem
    >>> cache = AutomorphismCache()
    >>> benzene = Chem.MolFromSmiles("c1ccccc1")
    >>> perms = cache.permutations(benzene)
    >>> len(perms)                    # the D6h automorphisms RDKit reports
    12
    >>> cache.permutations(benzene) is perms   # the cached list
    True
    """

    def __init__(self, max_permutations: int = _MAX_AUTOMORPHISMS) -> None:
        if not isinstance(max_permutations, int) or max_permutations < 1:
            raise DataValidationError(
                f"max_permutations must be a positive int, got {max_permutations!r}"
            )
        self.max_permutations = int(max_permutations)
        self._cache: dict[int, list[tuple[int, ...]]] = {}

    def permutations(self, mol: Any) -> list[tuple[int, ...]]:
        """The molecule's automorphisms (computed once, cached)."""
        key = id(mol)
        if key not in self._cache:
            self._cache[key] = _automorphism_permutations(mol)[: self.max_permutations]
        return self._cache[key]

    def clear(self) -> None:
        """Drop every cached molecule."""
        self._cache.clear()


def symmetry_corrected_rmsd(
    mol: Any,
    coords_ref: Sequence[Sequence[float]] | np.ndarray,
    coords: Sequence[Sequence[float]] | np.ndarray,
    minimize: bool = False,
    return_permutation: bool = False,
    cache: AutomorphismCache | None = None,
) -> Any:
    """The symmetry-corrected RMSD (spyrmsd's ``symmrmsd``).

    For every automorphism of the molecular graph, permute both
    coordinate sets and keep the minimum -- plain (in-place, the
    default) or Kabsch-aligned (``minimize=True``, spyrmsd's
    ``minimize``).  A phenyl flip or a carboxylate O-swap is *the same
    pose*, and this function says so.

    Parameters
    ----------
    mol:
        The RDKit molecule (defines the automorphism group).
    coords_ref, coords:
        The two ``(N, 3)`` geometries, same atom indexing as ``mol``.
    minimize:
        Kabsch-align each permutation before the RMSD.
    return_permutation:
        Also return the winning permutation (spyrmsd's
        ``return_permutation`` -- the atom mapping that realised the
        minimum).
    cache:
        An :class:`AutomorphismCache` to reuse permutations across a
        pose list.

    Example
    -------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> ethane = Chem.MolFromSmiles("CC")
    >>> a = np.array([[0.0, 0, 0], [1.54, 0, 0]])
    >>> b = np.array([[1.54, 0, 0], [0.0, 0, 0]])   # the two carbons swapped
    >>> round(symmetry_corrected_rmsd(ethane, a, b), 9)
    0.0
    """

    ref = _as_coords(coords_ref, "coords_ref")
    pos = _as_coords(coords, "coords")
    if len(ref) != len(pos) or len(ref) != mol.GetNumAtoms():
        raise DataValidationError(
            f"symmetry_corrected_rmsd needs {mol.GetNumAtoms()} coordinates "
            f"(the molecule's atom count); got {len(ref)} and {len(pos)}"
        )
    if cache is None:
        cache = AutomorphismCache()
    permutations = cache.permutations(mol)
    if not permutations:  # pragma: no cover - RDKit always returns identity
        permutations = [tuple(range(len(ref)))]

    best = math.inf
    best_perm = permutations[0]
    for perm in permutations:
        # spyrmsd's isomorphism loop: the reference stays in query
        # order, the pose is reordered by the permutation -- permuting
        # BOTH sides would compare the geometry to itself
        p_pos = pos[list(perm)]
        if minimize:
            value = kabsch_rmsd(ref, p_pos)
        else:
            value = float(np.sqrt(np.mean(np.sum((ref - p_pos) ** 2, axis=1))))
        if value < best:
            best = value
            best_perm = perm
    if return_permutation:
        return float(best), list(best_perm)
    return float(best)


def rmsd_wrapper(
    mol_ref: Any,
    mols: Sequence[Any],
    symmetry: bool = True,
    minimize: bool = False,
) -> list[float]:
    """spyrmsd's ``rmsdwrapper``: one reference vs a list of molecules.

    Every molecule's *first conformer* is compared against the
    reference's first conformer, atom-order-per-molecule (the wrapper
    assumes aligned indexing -- the caller's SDF pose lists always
    are).  ``symmetry=False`` compares plain in-place.

    Example
    -------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> ref = Chem.MolFromSmiles("CCO")
    >>> _ = AllChem.Compute2DCoords(ref)
    >>> same = Chem.Mol(ref)
    >>> [round(v, 6) for v in rmsd_wrapper(ref, [same])]
    [0.0]
    """

    if mol_ref is None or not mol_ref.GetNumConformers():
        raise DataValidationError("mol_ref needs at least one conformer")
    ref_pos = mol_ref.GetConformer().GetPositions()
    results: list[float] = []
    cache = AutomorphismCache()
    for mol in mols:
        if mol is None or not mol.GetNumConformers():
            raise DataValidationError(
                "every compared molecule needs at least one conformer"
            )
        pos = mol.GetConformer().GetPositions()
        if symmetry and mol.GetNumAtoms() == mol_ref.GetNumAtoms():
            results.append(
                symmetry_corrected_rmsd(
                    mol_ref, ref_pos, pos, minimize=minimize, cache=cache
                )
            )
        else:
            results.append(plain_rmsd(ref_pos, pos))
    return results


def best_pose_rmsd(
    mol: Any,
    coords_ref: Sequence[Sequence[float]] | np.ndarray,
    poses: Sequence[Sequence[Sequence[float]] | np.ndarray],
    minimize: bool = False,
) -> list[float]:
    """DiffDock's evaluation loop: one reference vs *many* poses.

    The automorphism permutations are enumerated once (the shared
    :class:`AutomorphismCache`) and reused for every pose --
    ``utils/molecules_utils.py::get_symmetry_rmsd``'s list-mode, the
    call behind ``rmsds_below_2`` / ``min_rmsds``.

    Example
    -------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> ethane = Chem.MolFromSmiles("CC")
    >>> ref = np.array([[0.0, 0, 0], [1.54, 0, 0]])
    >>> poses = [ref + 0.0, ref + 1.0]
    >>> [round(v, 6) for v in best_pose_rmsd(ethane, ref, poses)]
    [0.0, 1.732051]
    """

    cache = AutomorphismCache()
    return [
        float(
            symmetry_corrected_rmsd(
                mol, coords_ref, pose, minimize=minimize, cache=cache
            )
        )
        for pose in poses
    ]


def pose_rmsd_matrix(
    poses: Sequence[Sequence[Sequence[float]] | np.ndarray],
    minimize: bool = False,
) -> np.ndarray:
    """The pairwise in-place (or Kabsch) RMSD matrix of a pose set.

    The input to :func:`pose_clusters`.  ``minimize=True`` aligns every
    pair first (the rigid-body part of the deviation removed -- what
    DiffDock's clustering would see if it clustered on internal
    geometry rather than placement).

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    >>> matrix = pose_rmsd_matrix([a, a + 1.0, a + 5.0])
    >>> matrix.shape
    (3, 3)
    >>> bool(np.allclose(np.diag(matrix), 0.0))
    True
    """

    arrays = [_as_coords(pose, "pose") for pose in poses]
    n = len(arrays)
    sizes = {len(a) for a in arrays}
    if len(sizes) > 1:
        raise DataValidationError(
            f"pose_rmsd_matrix needs equal-size poses, got {sorted(sizes)}"
        )
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if minimize:
                value = kabsch_rmsd(arrays[i], arrays[j])
            else:
                value = plain_rmsd(arrays[i], arrays[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


# ---------------------------------------------------------------------------
# The torsion machinery (utils/torsion.py)
# ---------------------------------------------------------------------------


def torsion_bonds(mol: Any) -> list[tuple[int, int]]:
    """DiffDock's rotatable bonds (``get_transformation_mask`` rule).

    A bond is rotatable when **removing it disconnects the molecular
    graph** *and* the smaller connected component holds **more than one
    atom** -- terminal bonds never rotate (their small side is a single
    atom), ring bonds never disconnect.  This is deliberately *not*
    RDKit's ``NumRotatableBonds`` SMARTS definition; it is the torsion
    set DiffDock actually denoises.

    Example
    -------
    >>> from rdkit import Chem
    >>> torsion_bonds(Chem.MolFromSmiles("CCCC"))
    [(1, 2)]
    >>> torsion_bonds(Chem.MolFromSmiles("c1ccccc1"))   # rings: nothing rotates
    []
    """

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return []
    # adjacency (atom lists, not indices -- heavy graphs stay cheap)
    neighbours: list[list[int]] = [[] for _ in range(n_atoms)]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbours[a].append(b)
        neighbours[b].append(a)

    def component_without(bond: tuple[int, int], start: int) -> set[int]:
        """The connected component of ``start`` with ``bond`` removed."""
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in neighbours[node]:
                if {node, nxt} == set(bond):
                    continue
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    rotatable: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        side_a = component_without((a, b), a)
        if b in side_a:
            continue  # removing the bond keeps the graph connected (rings)
        smaller = side_a if len(side_a) <= n_atoms - len(side_a) else set(
            range(n_atoms)
        ) - side_a
        if len(smaller) > 1:
            rotatable.append((a, b))
    return rotatable


def rotatable_masks(mol: Any) -> tuple[list[tuple[int, int]], np.ndarray]:
    """The bonds and their rotation masks (``get_transformation_mask``).

    Returns ``(bonds, masks)`` where ``masks[k]`` is the boolean
    ``(n_atoms,)`` array of the atoms rotated when torsion *k* turns --
    the smaller side of bond *k*, exactly DiffDock's ``mask_rotate``.

    Example
    -------
    >>> from rdkit import Chem
    >>> butane = Chem.MolFromSmiles("CCCC")
    >>> bonds, masks = rotatable_masks(butane)
    >>> bonds
    [(1, 2)]
    >>> masks[0].astype(int).tolist()      # atoms {0, 1} rotate about 1-2
    [1, 1, 0, 0]
    """

    bonds = torsion_bonds(mol)
    n_atoms = mol.GetNumAtoms()
    masks = np.zeros((len(bonds), n_atoms), dtype=bool)
    neighbours: list[list[int]] = [[] for _ in range(n_atoms)]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbours[a].append(b)
        neighbours[b].append(a)

    def component_without(bond: tuple[int, int], start: int) -> set[int]:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in neighbours[node]:
                if {node, nxt} == set(bond):
                    continue
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    for k, (a, b) in enumerate(bonds):
        side_a = component_without((a, b), a)
        side_b = set(range(n_atoms)) - side_a
        smaller = side_a if len(side_a) <= len(side_b) else side_b
        masks[k, list(smaller)] = True
    return bonds, masks


def dihedrals(
    mol: Any,
    conf_id: int = -1,
) -> np.ndarray:
    """The torsion angles (radians) around :func:`torsion_bonds`.

    DiffDock's ``get_dihedrals``: for every rotatable bond ``(a, b)``
    the dihedral ``(c, a, b, d)`` with ``c``/``d`` the first
    non-bond neighbours of ``a``/``b``.

    Example
    -------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> butane = Chem.MolFromSmiles("CCCC")   # one rotatable bond 1-2
    >>> _ = AllChem.Compute2DCoords(butane)
    >>> angles = dihedrals(butane)
    >>> angles.shape
    (1,)
    >>> bool(np.all(np.abs(angles) <= np.pi))
    True
    """

    conformer = mol.GetConformer(conf_id)
    positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
    neighbours: list[list[int]] = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbours[a].append(b)
        neighbours[b].append(a)
    out: list[float] = []
    for a, b in torsion_bonds(mol):
        c = next((n for n in neighbours[a] if n != b), None)
        d = next((n for n in neighbours[b] if n != a), None)
        if c is None or d is None:  # pragma: no cover - torsion_bonds guarantees
            continue
        out.append(_dihedral_angle(positions[c], positions[a], positions[b], positions[d]))
    return np.asarray(out, dtype=np.float64)


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """The signed (c, a, b, d) dihedral in radians, IUPAC convention."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / max(np.linalg.norm(b1), 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.arctan2(y, x))


def perturb_torsions(
    coords: Sequence[Sequence[float]] | np.ndarray,
    bonds: Sequence[tuple[int, int]],
    masks: np.ndarray,
    torsion_updates: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply torsion updates to a geometry (``modify_conformer_torsion_angles``).

    For every rotatable bond *k* with update ``t_k``: rotate the
    bond's masked atoms about the ``(u, v)`` axis through the pivot
    ``v`` by ``t_k`` radians -- DiffDock's exact convention (positive
    rotation with the rotation vector pointing ``u -> v``):

    ``p_i <- (p_i - p_v) @ R(t_k * (p_u - p_v)/|..|)^T + p_v``.

    Example
    -------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> butane = Chem.MolFromSmiles("CCCC")
    >>> bonds, masks = rotatable_masks(butane)
    >>> flat = np.array([[0., 1, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0]])
    >>> turned = perturb_torsions(flat, bonds, masks, [np.pi / 2])
    >>> bool(np.allclose(turned[1], flat[1]))          # on the bond axis
    True
    >>> round(float(np.linalg.norm(turned[0] - flat[0])), 6)   # atom 0 swung
    1.414214
    """

    pos = _as_coords(coords, "coords").copy()
    updates = np.asarray(torsion_updates, dtype=np.float64)
    if len(bonds) != len(updates) or len(bonds) != len(masks):
        raise DataValidationError(
            f"perturb_torsions needs matching lengths: {len(bonds)} bonds, "
            f"{len(masks)} masks, {len(updates)} updates"
        )
    for k, (u, v) in enumerate(bonds):
        if updates[k] == 0.0:
            continue
        rot_vec = pos[u] - pos[v]
        norm = float(np.linalg.norm(rot_vec))
        if norm < 1e-12:  # pragma: no cover - degenerate geometry guard
            continue
        rot_vec = rot_vec * (updates[k] / norm)
        rot_mat = Rotation.from_rotvec(rot_vec).as_matrix()
        idx = list(np.nonzero(masks[k])[0])
        if not idx:
            continue
        # ensure the mask rotates the side NOT containing the pivot's
        # fixed partner (v's side stays; u's side turns) -- DiffDock
        # asserts it; the toolkit enforces it by construction
        if masks[k][v] and not masks[k][u]:
            idx = list(np.nonzero(~masks[k])[0])
        moved = pos[idx] - pos[v]
        pos[idx] = moved @ rot_mat.T + pos[v]
    return pos


def randomize_pose(
    mol: Any,
    conf_id: int = -1,
    seed: int | None = 42,
    center: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """DiffDock's ``randomize_position``: the diffusion initial state.

    Uniform random torsions over ``[-pi, pi]`` on every rotatable
    bond, a uniformly random rotation, and the centroid translated to
    ``center`` (the receptor centroid in DiffDock; the current centroid
    when ``None`` -- pure randomisation without relocation).  Seeded:
    the toolkit's deterministic-runs rule.

    Returns the randomised ``(N, 3)`` geometry (the conformer is left
    untouched; feed the result to :func:`perturb_torsions`-style
    pipelines or write it out yourself).

    Example
    -------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> butane = Chem.MolFromSmiles("CCCC")
    >>> _ = AllChem.Compute2DCoords(butane)
    >>> flat = butane.GetConformer().GetPositions()
    >>> rnd = randomize_pose(butane, seed=42, center=[0.0, 0.0, 0.0])
    >>> bool(np.allclose(rnd.mean(axis=0), 0.0, atol=1e-9))   # re-centred
    True
    >>> bool(not np.allclose(rnd, flat))                      # randomised
    True
    """

    pos = np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    bonds, masks = rotatable_masks(mol)
    if bonds:
        updates = rng.uniform(low=-np.pi, high=np.pi, size=len(bonds))
        pos = perturb_torsions(pos, bonds, masks, updates)
    # uniform random rotation (scipy's Rotation.random, seeded with the
    # same int -- the same distribution as DiffDock's R.random())
    rotation = Rotation.random(random_state=seed)
    pos = (pos - pos.mean(axis=0)) @ rotation.as_matrix().T
    if center is not None:
        centre = np.asarray(center, dtype=np.float64).reshape(3)
        pos = pos + centre
    return pos


# ---------------------------------------------------------------------------
# The pose-evaluation metrics (evaluate.py)
# ---------------------------------------------------------------------------


def min_intra_distance(coords: Sequence[Sequence[float]] | np.ndarray) -> float:
    """The minimum pairwise atom distance (the self-clash metric).

    DiffDock's ``topk_self_intersect_fraction`` flags poses whose
    minimum intra-ligand distance drops under 0.4 A (spyrmsd's
    connectivity tolerance); this function exposes the raw number.

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([[0.0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
    >>> min_intra_distance(a)
    1.5
    """

    pos = _as_coords(coords, "coords")
    if len(pos) < 2:
        raise DataValidationError("min_intra_distance needs at least 2 atoms")
    diffs = pos[:, None, :] - pos[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(distances, math.inf)
    return float(distances.min())


def pose_clusters(
    rmsd_matrix: np.ndarray,
    threshold: float = 2.0,
) -> list[list[int]]:
    """Greedy RMSD clustering of poses (the confidence approximation).

    Sort poses by their best distance to pose 0..k-1 (greedy leader
    selection), open a new cluster whenever a pose is further than
    ``threshold`` from every existing leader.  DiffDock's confidence
    model ranks pose clusters by a learned score; with no model
    available, cluster *size* is the honest ranking signal -- the
    basin the sampler keeps rediscovering.

    Example
    -------
    >>> import numpy as np
    >>> matrix = np.array([[0.0, 0.5, 5.0], [0.5, 0.0, 4.5], [5.0, 4.5, 0.0]])
    >>> pose_clusters(matrix, threshold=2.0)
    [[0, 1], [2]]
    """

    matrix = np.asarray(rmsd_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise DataValidationError(
            f"rmsd_matrix must be a non-empty square matrix, got {matrix.shape}"
        )
    if threshold <= 0:
        raise DataValidationError(f"threshold must be positive, got {threshold!r}")
    n = matrix.shape[0]
    unassigned = list(range(n))
    clusters: list[list[int]] = []
    while unassigned:
        leader = unassigned.pop(0)
        members = [leader]
        rest = []
        for pose in unassigned:
            if matrix[leader, pose] <= threshold:
                members.append(pose)
            else:
                rest.append(pose)
        unassigned = rest
        clusters.append(sorted(members))
    return clusters


@dataclass
class PoseEvaluationReport:
    """DiffDock's ``evaluate.py`` per-complex metrics, one pose set.

    Every field mirrors an ``evaluate.py`` output column: ``rmsds``
    (symmetry-corrected, in place), ``rmsds_below_2`` / ``below_5``
    (the classic success fractions), the 25/50/75 percentiles,
    ``centroid_distances`` and their <2 A / <5 A fractions, the top-k
    best RMSD (``min_rmsds`` -- "best of N sampled"), and the
    self-clash fraction (min intra-ligand distance < 0.4 A).
    """

    n_poses: int
    rmsd_mean: float
    rmsd_median: float
    rmsd_below_2_fraction: float
    rmsd_below_5_fraction: float
    rmsd_p25: float
    rmsd_p75: float
    best_rmsd: float
    centroid_mean: float
    centroid_below_2_fraction: float
    min_intra_distance_min: float
    self_clash_fraction: float
    clusters: list[list[int]] = field(default_factory=list)
    cluster_threshold: float = 2.0

    def summary(self) -> str:
        """The one-screen text report (the playground's text pane)."""
        lines = [
            f"poses               : {self.n_poses}",
            f"RMSD mean / median  : {self.rmsd_mean:.3f} / {self.rmsd_median:.3f} A",
            (
                f"RMSD < 2 A / < 5 A  : "
                f"{self.rmsd_below_2_fraction:.1%} / {self.rmsd_below_5_fraction:.1%}"
            ),
            f"RMSD p25 / p75      : {self.rmsd_p25:.3f} / {self.rmsd_p75:.3f} A",
            f"best pose RMSD      : {self.best_rmsd:.3f} A",
            (
                f"centroid mean       : {self.centroid_mean:.3f} A "
                f"(< 2 A: {self.centroid_below_2_fraction:.1%})"
            ),
            (
                f"min intra dist      : {self.min_intra_distance_min:.3f} A "
                f"(clash fraction: {self.self_clash_fraction:.1%})"
            ),
            (
                f"pose clusters       : {len(self.clusters)} "
                f"at {self.cluster_threshold:g} A "
                f"(sizes: {[len(c) for c in self.clusters]})"
            ),
        ]
        return "\n".join(lines)


def pose_evaluation_report(
    mol: Any,
    coords_ref: Sequence[Sequence[float]] | np.ndarray,
    poses: Sequence[Sequence[Sequence[float]] | np.ndarray],
    cluster_threshold: float = 2.0,
) -> PoseEvaluationReport:
    """The full DiffDock evaluation on one reference vs N poses.

    Symmetry-corrected in-place RMSDs (the ``get_symmetry_rmsd`` list
    mode with the shared automorphism cache), the ``<2 A`` / ``<5 A``
    fractions, percentiles, centroid distances, best-of-N, the clash
    fraction, and greedy :func:`pose_clusters`.

    Example
    -------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> ethane = Chem.MolFromSmiles("CC")
    >>> ref = np.array([[0.0, 0, 0], [1.54, 0, 0]])
    >>> report = pose_evaluation_report(
    ...     ethane, ref, [ref + 0.0, ref + 0.1, ref + 9.0])
    >>> report.n_poses
    3
    >>> round(report.rmsd_below_2_fraction, 6)
    0.666667
    """

    ref = _as_coords(coords_ref, "coords_ref")
    pose_list = [_as_coords(pose, "pose") for pose in poses]
    if not pose_list:
        raise DataValidationError("pose_evaluation_report needs at least one pose")
    sizes = {len(p) for p in pose_list} | {len(ref)}
    if len(sizes) > 1:
        raise DataValidationError(
            f"reference and poses must share atom counts, got {sorted(sizes)}"
        )
    rmsds = np.asarray(
        best_pose_rmsd(mol, ref, pose_list, minimize=False), dtype=np.float64
    )
    centroids = np.asarray([centroid_distance(ref, p) for p in pose_list])
    intra = [min_intra_distance(p) for p in pose_list]
    clash = float(np.mean([d < 0.4 for d in intra]))
    matrix = pose_rmsd_matrix(pose_list, minimize=False)
    clusters = pose_clusters(matrix, threshold=cluster_threshold)
    return PoseEvaluationReport(
        n_poses=len(pose_list),
        rmsd_mean=float(rmsds.mean()),
        rmsd_median=float(np.median(rmsds)),
        rmsd_below_2_fraction=float(np.mean(rmsds < 2.0)),
        rmsd_below_5_fraction=float(np.mean(rmsds < 5.0)),
        rmsd_p25=float(np.percentile(rmsds, 25)),
        rmsd_p75=float(np.percentile(rmsds, 75)),
        best_rmsd=float(rmsds.min()),
        centroid_mean=float(centroids.mean()),
        centroid_below_2_fraction=float(np.mean(centroids < 2.0)),
        min_intra_distance_min=float(min(intra)),
        self_clash_fraction=clash,
        clusters=clusters,
        cluster_threshold=float(cluster_threshold),
    )


# ---------------------------------------------------------------------------
# The pocket extraction (datasets/moad.py + utils/utils.py::crop_beyond)
# ---------------------------------------------------------------------------


@dataclass
class PocketReport:
    """The outcome of :func:`extract_pocket` / :func:`crop_beyond`.

    ``residues`` are ``(chain, resseq, icode)`` keys in file order;
    ``pdb_text`` is the pocket's ATOM/HETATM records verbatim, ready to
    write or to feed the receptor preparation pipeline.
    """

    residues: list[tuple[str, str, str]]
    chains: list[str]
    pdb_text: str
    n_atoms: int
    min_ligand_distance: float

    def summary(self) -> str:
        """The one-screen text report (the playground's text pane)."""
        return "\n".join(
            [
                f"pocket residues      : {len(self.residues)}",
                f"chains kept          : {', '.join(c or '?' for c in self.chains) or '(none)'}",
                f"pocket atoms         : {self.n_atoms}",
                f"closest ligand atom  : {self.min_ligand_distance:.2f} A",
            ]
        )


def _pdb_atom_lines(pdb_text: str) -> list[str]:
    return [line for line in pdb_text.splitlines() if line.startswith(("ATOM", "HETATM"))]


def _line_coords(line: str) -> np.ndarray | None:
    try:
        return np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    except ValueError:
        return None


def extract_pocket(
    protein_pdb_text: str,
    ligand_coords: Sequence[Sequence[float]] | np.ndarray,
    residue_cutoff: float = 10.0,
    keep_whole_chains: bool = True,
    include_hetatm: bool = True,
) -> PocketReport:
    """The MOAD pocket selection: residues near the ligand.

    DiffDock's ``datasets/moad.py::get_by_name`` logic, ported line by
    line: for every residue, the minimum distance from any ligand atom
    to any residue atom; a residue is *in the pocket* when that minimum
    is under ``residue_cutoff`` (their ``chain_cutoff``, default 10 A);
    with ``keep_whole_chains=True`` every chain holding at least one
    pocket residue stays **in full** (the MOAD convention), otherwise
    only the pocket residues are kept.

    Parameters
    ----------
    protein_pdb_text:
        The protein's PDB file content (ATOM/HETATM records).
    ligand_coords:
        The ligand's ``(N, 3)`` geometry (any pose -- the crystal pose,
        a docked pose, or a pocket-centre probe).
    residue_cutoff:
        The residue-level cutoff in Angstrom (default 10, MOAD's).
    keep_whole_chains:
        Keep whole chains (True, MOAD's behaviour) or pocket residues
        only (False, closer to DiffDock's ``crop_beyond``).
    include_hetatm:
        Include HETATM records (cofactors, ions) in the protein side.

    Example
    -------
    >>> pdb = (
    ...     "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
    ...     "           N  \\n"
    ...     "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00  0.00"
    ...     "           C  \\n"
    ...     "ATOM      3  N   VAL B  10      50.000   0.000   0.000  1.00  0.00"
    ...     "           N  \\n")
    >>> report = extract_pocket(pdb, ligand_coords=[[1.0, 0.0, 0.0]],
    ...     residue_cutoff=5.0, keep_whole_chains=False)
    >>> report.residues
    [('A', '1', ''), ('A', '2', '')]
    >>> report.n_atoms
    2
    """

    ligand = _as_coords(ligand_coords, "ligand_coords")
    if residue_cutoff <= 0:
        raise DataValidationError(
            f"residue_cutoff must be positive, got {residue_cutoff!r}"
        )
    lines = _pdb_atom_lines(protein_pdb_text)
    if not lines:
        raise DataValidationError(
            "protein_pdb_text holds no ATOM/HETATM records -- nothing to "
            "carve a pocket from"
        )
    # residue -> (lines, min distance so far)
    residues: dict[tuple[str, str, str], list[str]] = {}
    distances: dict[tuple[str, str, str], float] = {}
    for line in lines:
        if not include_hetatm and line.startswith("HETATM"):
            continue
        key = _parse_resseq_key(line)
        coords = _line_coords(line)
        if coords is None:
            continue
        residues.setdefault(key, []).append(line)
        if ligand.size:
            d_min = float(np.min(np.linalg.norm(ligand - coords, axis=1)))
            distances[key] = min(distances.get(key, math.inf), d_min)

    pocket_keys = [
        key for key, dist in distances.items() if dist < residue_cutoff
    ]
    if keep_whole_chains:
        pocket_chains = {key[0] for key in pocket_keys}
        keep = {key for key in residues if key[0] in pocket_chains}
    else:
        keep = set(pocket_keys)

    kept_lines: list[str] = []
    kept_keys: list[tuple[str, str, str]] = []
    for key, key_lines in residues.items():  # file order
        if key in keep:
            kept_lines.extend(key_lines)
            kept_keys.append(key)
    chains = sorted({key[0] for key in kept_keys})
    closest = min(distances.get(key, math.inf) for key in kept_keys)
    return PocketReport(
        residues=kept_keys,
        chains=chains,
        pdb_text="\n".join(kept_lines) + ("\n" if kept_lines else ""),
        n_atoms=len(kept_lines),
        min_ligand_distance=float(closest),
    )


def crop_beyond(
    protein_pdb_text: str,
    ligand_coords: Sequence[Sequence[float]] | np.ndarray,
    cutoff: float = 20.0,
) -> str:
    """DiffDock's ``utils/utils.py::crop_beyond``: the residue-level crop.

    Keep every residue with **any atom** within ``cutoff`` (default
    20 A -- DiffDock's receptor-pruning distance before featurising)
    of any ligand atom; drop the rest.  Returns the cropped PDB text.

    Example
    -------
    >>> pdb = (
    ...     "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
    ...     "           N  \\n"
    ...     "ATOM      2  N   VAL A  99      99.000   0.000   0.000  1.00  0.00"
    ...     "           N  \\n")
    >>> cropped = crop_beyond(pdb, ligand_coords=[[0.0, 0.0, 0.0]], cutoff=20.0)
    >>> cropped.count("ATOM")
    1
    """

    report = extract_pocket(
        protein_pdb_text,
        ligand_coords,
        residue_cutoff=cutoff,
        keep_whole_chains=False,
    )
    return report.pdb_text
