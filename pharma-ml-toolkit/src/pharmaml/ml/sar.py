"""Structure-activity relationship analytics: MMP, activity cliffs, R-groups.

Three med-chem workhorses the toolkit was missing, on one shared core:

* **Matched molecular pairs (MMP)** -- :func:`mmp_fragmentation` cuts one
  single, acyclic, non-aromatic bond in a molecule and reports the two
  fragments; the larger fragment is the *context* and the smaller is the
  *substituent*.  Two molecules that share a context and differ in their
  substituent form a matched pair, and the transformation between them is
  the substituent swap (Hussain & Rea 2010; the RDKit ``rdMMPA`` recipe,
  re-derived on the toolkit's own :mod:`pharmaml.chem.molgraph` engine so
  the pairing math runs without RDKit).
* **Activity cliffs** -- :func:`activity_cliffs` keeps the matched pairs
  whose activity delta clears a threshold: near-identical structures that
  disagree on potency, the cases a QSAR model is most likely to get wrong
  (Maggiora 2006).
* **R-group decomposition** -- :func:`rgroup_decomposition` matches a core
  (SMARTS with marked attachment atoms) against a series and reads the
  substituent at each attachment, the table a Free-Wilson / SAR walk
  starts from (RDKit's ``rdRGroupDecomposition``, absorbed as a
  lazily-imported ``[chem]`` surface).

The fragmentation half is pure Python over the parsed molecular graph, so
it is unit-tested and doctested without RDKit; only R-group decomposition
needs a real SMARTS matcher and raises :class:`DependencyError` without it.

Examples
--------
>>> from pharmaml.ml.sar import mmp_fragmentation
>>> mmp_fragmentation("CCO")          # ethanol: the CC core, the O substituent
[('CC', 'O'), ('CO', 'C')]
>>> mmp_fragmentation("c1ccccc1")     # benzene: no acyclic bond to cut
[]

>>> import pandas as pd
>>> from pharmaml.ml.sar import matched_pairs
>>> frame = pd.DataFrame(
...     {"smiles": ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"], "pic50": [5.0, 6.5, 5.1]})
>>> pairs = matched_pairs(frame, "smiles", "pic50")
>>> [(r["left_sub"], r["right_sub"], round(r["delta"], 4)) for r in
...  pairs[["left_sub", "right_sub", "delta"]].to_dict("records")]
[('O', 'N', 1.5), ('C', 'N', 1.4), ('O', 'C', 0.1)]

>>> from pharmaml.ml.sar import activity_cliffs
>>> cliffs = activity_cliffs(frame, "smiles", "pic50", min_delta=1.0)
>>> [(r["left_sub"], r["right_sub"], round(r["delta"], 4)) for r in
...  cliffs.pairs[["left_sub", "right_sub", "delta"]].to_dict("records")]
[('O', 'N', 1.5), ('C', 'N', 1.4)]
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "MATCHED_PAIR_COLUMNS",
    "ActivityCliffReport",
    "MatchedPair",
    "RGroup",
    "activity_cliffs",
    "matched_pairs",
    "mmp_deltas",
    "mmp_fragmentation",
    "rgroup_decomposition",
    "rgroup_table",
]

_RDKIT_HINT = (
    "R-group decomposition requires RDKit — from the toolkit package root run python -m pip install -e '.[chem]'"
    ""
)

#: Column order of the :func:`matched_pairs` frame (also the empty-frame schema).
MATCHED_PAIR_COLUMNS: tuple[str, ...] = (
    "context",
    "left_sub",
    "right_sub",
    "left_smiles",
    "right_smiles",
    "left_activity",
    "right_activity",
    "delta",
)


def _require_rdkit() -> Any:
    """Import and return ``rdkit.Chem``, or raise the typed error."""
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - env-dependent branch
        raise DependencyError(_RDKIT_HINT) from exc
    return Chem


# --------------------------------------------------------------------------- #
# public result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatchedPair:
    """Two molecules that differ by exactly one substituent swap.

    ``context`` is the shared fragment (identical canonical SMILES for
    both molecules); ``left_sub`` / ``right_sub`` are the swapped
    substituents.  ``delta`` is ``right_activity - left_activity`` with
    the *lower-activity* molecule on the left, so ``delta >= 0``.
    """

    context: str
    left_sub: str
    right_sub: str
    left_smiles: str
    right_smiles: str
    left_activity: float
    right_activity: float
    delta: float

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view (the row shape :func:`matched_pairs` builds)."""
        return {
            "context": self.context,
            "left_sub": self.left_sub,
            "right_sub": self.right_sub,
            "left_smiles": self.left_smiles,
            "right_smiles": self.right_smiles,
            "left_activity": self.left_activity,
            "right_activity": self.right_activity,
            "delta": self.delta,
        }


@dataclass
class ActivityCliffReport:
    """A matched-pair set filtered to the potency cliffs.

    Attributes
    ----------
    pairs:
        The cliff rows (the :func:`matched_pairs` frame filtered by
        ``delta >= min_delta``), sorted by ``delta`` descending.
    min_delta:
        The threshold that produced them.
    n_pairs_scanned:
        How many matched pairs were considered before filtering.
    """

    pairs: pd.DataFrame
    min_delta: float
    n_pairs_scanned: int

    def __len__(self) -> int:
        return len(self.pairs)

    def as_frame(self) -> pd.DataFrame:
        """The cliff rows as a :class:`pandas.DataFrame`."""
        return self.pairs

    def summary(self) -> str:
        """One-line reading of the report.

        Examples
        --------
        >>> import pandas as pd
        >>> from pharmaml.ml.sar import activity_cliffs
        >>> frame = pd.DataFrame(
        ...     {"smiles": ["c1ccccc1O", "c1ccccc1N"], "pic50": [5.0, 6.5]})
        >>> activity_cliffs(frame, "smiles", "pic50", min_delta=1.0).summary()
        '1 activity cliff(s) at |delta| >= 1, out of 1 matched pair(s)'
        """
        return (
            f"{len(self.pairs)} activity cliff(s) at |delta| >= "
            f"{self.min_delta:g}, out of {self.n_pairs_scanned} matched pair(s)"
        )


@dataclass
class RGroup:
    """One molecule's read-out in an R-group decomposition.

    ``groups`` maps attachment index (1-based, in the core's marked-atom
    order) to the substituent SMILES attached there; ``valid`` is False
    when the core did not match and the row is an honest empty.
    """

    smiles: str
    groups: dict[int, str] = field(default_factory=dict)
    valid: bool = False

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"smiles": self.smiles, "valid": self.valid}
        row.update({f"R{n}": sub for n, sub in sorted(self.groups.items())})
        return row


# --------------------------------------------------------------------------- #
# single-cut fragmentation over the pure graph (no RDKit)
# --------------------------------------------------------------------------- #
def _ring_bond(mol: Any, bond: Any) -> bool:
    """True when ``bond`` lies on a cycle (its endpoints stay connected without it)."""
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(mol.atoms))}
    for other in mol.bonds:
        if other is bond:
            continue
        adjacency[other.begin].add(other.end)
        adjacency[other.end].add(other.begin)
    seen = {bond.begin}
    stack = [bond.begin]
    while stack:
        node = stack.pop()
        if node == bond.end:
            return True
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _split_single_cut(mol: Any, cut: Any) -> tuple[set[int], set[int]]:
    """Atom indices on each side of one bond (BFS over the rest)."""
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(mol.atoms))}
    for other in mol.bonds:
        if other is cut:
            continue
        adjacency[other.begin].add(other.end)
        adjacency[other.end].add(other.begin)
    side = {cut.begin}
    stack = [cut.begin]
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in side:
                side.add(nxt)
                stack.append(nxt)
    return side, set(range(len(mol.atoms))) - side


def _fragment_smiles(mol: Any, keep: set[int]) -> str:
    """Canonical SMILES of the sub-graph induced by ``keep`` (``""`` on failure)."""
    from pharmaml.chem.molgraph import MolAtom, MolBond, Molecule, canonical_smiles

    if not keep:
        return ""
    remap = {old: new for new, old in enumerate(sorted(keep))}
    atoms = [
        MolAtom(
            index=remap[i],
            symbol=mol.atoms[i].symbol,
            atomic_num=mol.atoms[i].atomic_num,
            is_aromatic=mol.atoms[i].is_aromatic,
            charge=mol.atoms[i].charge,
            isotope=mol.atoms[i].isotope,
            bracket_h=mol.atoms[i].bracket_h,
        )
        for i in sorted(keep)
    ]
    bonds = [
        MolBond(
            begin=remap[b.begin],
            end=remap[b.end],
            order=b.order,
            is_aromatic=b.is_aromatic,
        )
        for b in mol.bonds
        if b.begin in keep and b.end in keep
    ]
    try:
        return canonical_smiles(Molecule(source="", atoms=atoms, bonds=bonds))
    except Exception:  # noqa: BLE001 - a hostile fragment is an honest ""
        return ""


def _rank(smiles: str) -> tuple[int, int]:
    """Fragment ranking key: (heavy-atom count, length); larger = context."""
    try:
        from pharmaml.chem.molgraph import parse_smiles

        return (len(parse_smiles(smiles, strict=False).atoms), len(smiles))
    except Exception:  # noqa: BLE001
        return (len(smiles), len(smiles))


def mmp_fragmentation(smiles: str, *, max_cuts: int = 1) -> list[tuple[str, str]]:
    """Single-cut fragmentations of one molecule: ``(context, substituent)``.

    Every acyclic, non-aromatic, single bond between two heavy atoms is
    cut once; the two fragments are canonicalised and the larger (more
    heavy atoms, ties broken by the longer SMILES) becomes the ``context``
    while the smaller is the ``substituent``.  Cutting a molecule that is
    itself one fragment (a bare ring, a single atom) yields nothing.

    Parameters
    ----------
    smiles:
        One SMILES string.  Unparsable input returns ``[]`` (an honest
        "no fragments" -- it never crashes a batch).
    max_cuts:
        Number of cuts; only ``1`` is supported (multi-cut fragmentation
        is a documented non-goal).

    Returns
    -------
    list of ``(context, substituent)`` string pairs, sorted and deduplicated.

    Raises
    ------
    DataValidationError
        If ``max_cuts`` is not ``1``.

    Examples
    --------
    >>> mmp_fragmentation("CCO")                 # ethanol
    [('CC', 'O'), ('CO', 'C')]
    >>> mmp_fragmentation("c1ccccc1")            # benzene: no acyclic bond
    []
    >>> mmp_fragmentation("not a smiles")        # hostile input: []
    []
    >>> mmp_fragmentation("CCO", max_cuts=2)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: only max_cuts=1 is supported, got 2
    """
    if max_cuts != 1:
        raise DataValidationError(f"only max_cuts=1 is supported, got {max_cuts}")
    from pharmaml.chem.molgraph import parse_smiles

    try:
        mol = parse_smiles(smiles, strict=False)
    except Exception:  # noqa: BLE001 - hostile input is an honest empty
        return []

    pairs: list[tuple[str, str]] = []
    for cut in mol.bonds:
        if cut.order != 1 or cut.is_aromatic:
            continue
        if _ring_bond(mol, cut):
            continue
        left, right = _split_single_cut(mol, cut)
        smi_l = _fragment_smiles(mol, left)
        smi_r = _fragment_smiles(mol, right)
        if not smi_l or not smi_r:
            continue
        if _rank(smi_l) >= _rank(smi_r):
            context, sub = smi_l, smi_r
        else:
            context, sub = smi_r, smi_l
        pairs.append((context, sub))
    return sorted(set(pairs))


# --------------------------------------------------------------------------- #
# matched pairs / MMP deltas / activity cliffs
# --------------------------------------------------------------------------- #
def _check_columns(frame: pd.DataFrame, smiles_col: str, activity_col: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("frame must be a pandas DataFrame")
    for col in (smiles_col, activity_col):
        if col not in frame.columns:
            raise DataValidationError(
                f"column {col!r} not in frame; available: {list(frame.columns)}"
            )


def matched_pairs(
    frame: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    *,
    min_shared: int = 0,
) -> pd.DataFrame:
    """All matched molecular pairs in a table.

    Each molecule is single-cut fragmented; two molecules sharing a
    context but differing in their substituent form a pair, emitted with
    the lower-activity molecule on the left so ``delta >= 0`` for every
    row.  Rows with a non-finite activity or an unparsable SMILES are
    dropped silently (a screening table routinely carries both).

    Parameters
    ----------
    frame:
        Table with one SMILES column and one numeric activity column.
    smiles_col, activity_col:
        Column names.
    min_shared:
        Keep only transformations whose context is shared by at least this
        many *distinct* molecules (0 = no filter).  ``min_shared=2`` keeps
        a context seen in two or more molecules, i.e. every genuine pair.

    Returns
    -------
    :class:`pandas.DataFrame`
        Columns ``context, left_sub, right_sub, left_smiles, right_smiles,
        left_activity, right_activity, delta`` -- one row per pair, sorted
        by ``delta`` descending.

    Raises
    ------
    DataValidationError
        On a missing column or a non-numeric activity column.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"smiles": ["CCO", "CCN"], "y": [5.0, 6.0]})
    >>> pairs = matched_pairs(frame, "smiles", "y")
    >>> pairs[["context", "left_sub", "right_sub", "delta"]].to_dict("records")
    [{'context': 'CC', 'left_sub': 'O', 'right_sub': 'N', 'delta': 1.0}]
    >>> matched_pairs(frame, "missing", "y")
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: column 'missing' not in frame; available: ['smiles', 'y']
    """
    _check_columns(frame, smiles_col, activity_col)
    if not pd.api.types.is_numeric_dtype(frame[activity_col]):
        raise DataValidationError(f"activity column {activity_col!r} must be numeric")
    if min_shared < 0:
        raise DataValidationError(f"min_shared must be >= 0, got {min_shared}")

    by_context: dict[str, list[tuple[str, str, float]]] = {}
    for smiles, activity in zip(frame[smiles_col], frame[activity_col], strict=False):
        if not isinstance(smiles, str) or not smiles:
            continue
        try:
            value = float(activity)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        for context, sub in mmp_fragmentation(smiles):
            by_context.setdefault(context, []).append((sub, smiles, value))

    rows: list[dict[str, Any]] = []
    for context, entries in by_context.items():
        if len({entry[1] for entry in entries}) < max(2, min_shared):
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                sub_i, smi_i, act_i = entries[i]
                sub_j, smi_j, act_j = entries[j]
                if smi_i == smi_j or sub_i == sub_j:
                    continue
                if act_i <= act_j:
                    left, right = (sub_i, smi_i, act_i), (sub_j, smi_j, act_j)
                else:
                    left, right = (sub_j, smi_j, act_j), (sub_i, smi_i, act_i)
                rows.append(
                    MatchedPair(
                        context=context,
                        left_sub=left[0],
                        right_sub=right[0],
                        left_smiles=left[1],
                        right_smiles=right[1],
                        left_activity=left[2],
                        right_activity=right[2],
                        delta=right[2] - left[2],
                    ).as_dict()
                )
    columns = [
        "context",
        "left_sub",
        "right_sub",
        "left_smiles",
        "right_smiles",
        "left_activity",
        "right_activity",
        "delta",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    out = out.drop_duplicates(subset=["context", "left_smiles", "right_smiles"])
    return out.sort_values("delta", ascending=False).reset_index(drop=True)


def mmp_deltas(
    frame: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    *,
    min_pairs: int = 1,
) -> pd.DataFrame:
    """Aggregate the per-transformation activity effect across a table.

    Groups :func:`matched_pairs` by ``(context, left_sub, right_sub)`` and
    summarises each substituent swap: how many pairs support it, the mean
    and sample standard deviation of ``delta``, and the mean of the two
    activity levels.  This is the classic "replace X with Y and potency
    moves by Z" readout.

    Parameters
    ----------
    frame:
        Table with one SMILES column and one numeric activity column.
    smiles_col, activity_col:
        Column names.
    min_pairs:
        Drop transformations supported by fewer than this many pairs.

    Returns
    -------
    :class:`pandas.DataFrame`
        Columns ``context, left_sub, right_sub, n_pairs, mean_delta,
        std_delta, mean_activity``, sorted by ``mean_delta`` descending.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame(
    ...     {"smiles": ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"],
    ...      "y": [5.0, 6.5, 5.1]})
    >>> deltas = mmp_deltas(frame, "smiles", "y")
    >>> [(r["left_sub"], r["right_sub"], round(r["mean_delta"], 4)) for r in
    ...  deltas[["left_sub", "right_sub", "mean_delta"]].to_dict("records")]
    [('O', 'N', 1.5), ('C', 'N', 1.4), ('O', 'C', 0.1)]
    """
    pairs = matched_pairs(frame, smiles_col, activity_col)
    columns = [
        "context",
        "left_sub",
        "right_sub",
        "n_pairs",
        "mean_delta",
        "std_delta",
        "mean_activity",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        pairs.assign(mean_activity=(pairs["left_activity"] + pairs["right_activity"]) / 2.0)
        .groupby(["context", "left_sub", "right_sub"], as_index=False)
        .agg(
            n_pairs=("delta", "size"),
            mean_delta=("delta", "mean"),
            std_delta=("delta", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
            mean_activity=("mean_activity", "mean"),
        )
    )
    grouped = grouped[grouped["n_pairs"] >= max(1, min_pairs)]
    grouped = grouped.sort_values("mean_delta", ascending=False)
    return grouped[columns].reset_index(drop=True)


def activity_cliffs(
    frame: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    *,
    min_delta: float = 1.0,
) -> ActivityCliffReport:
    """Matched pairs whose activity delta clears ``min_delta``.

    An activity cliff is a small structural change (one substituent swap)
    with a large potency change.  The default ``min_delta=1.0`` is one log
    unit -- the conventional pIC50 cliff threshold.

    Parameters
    ----------
    frame:
        Table with one SMILES column and one numeric activity column.
    smiles_col, activity_col:
        Column names.
    min_delta:
        Minimum ``delta`` to count as a cliff (must be >= 0).

    Returns
    -------
    ActivityCliffReport

    Raises
    ------
    DataValidationError
        If ``min_delta`` is negative.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"smiles": ["CCO", "CCN"], "y": [5.0, 6.5]})
    >>> report = activity_cliffs(frame, "smiles", "y", min_delta=1.0)
    >>> len(report), round(float(report.pairs["delta"].iloc[0]), 3)
    (1, 1.5)
    >>> activity_cliffs(frame, "smiles", "y", min_delta=-1.0)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: min_delta must be >= 0, got -1.0
    """
    if min_delta < 0:
        raise DataValidationError(f"min_delta must be >= 0, got {min_delta}")
    pairs = matched_pairs(frame, smiles_col, activity_col)
    cliffs = pairs[pairs["delta"] >= min_delta].reset_index(drop=True)
    return ActivityCliffReport(pairs=cliffs, min_delta=min_delta, n_pairs_scanned=len(pairs))


# --------------------------------------------------------------------------- #
# R-group decomposition (RDKit's rdRGroupDecomposition, lazily imported)
# --------------------------------------------------------------------------- #
def _strip_dummies(chem: Any, mol: Any) -> str:
    """SMILES of ``mol`` with the ``[*:n]`` attachment dummies removed.

    An R-group that is a bare hydrogen comes back from RDKit as
    ``[H][*:1]``; stripping the dummy yields ``[H]`` (RDKit writes the
    detached hydrogen as ``[HH]``, so the hydrogen case is special-cased
    back to the honest ``"[H]"``).
    """
    if mol is None:
        return "[H]"
    rw = chem.RWMol(mol)
    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummies:
        return chem.MolToSmiles(mol)
    for idx in sorted(dummies, reverse=True):
        rw.RemoveAtom(idx)
    stripped = rw.GetMol()
    if stripped.GetNumAtoms() == 0:
        return "[H]"
    # a lone hydrogen atom carries no bonds and is written [HH] by RDKit
    if stripped.GetNumAtoms() == 1 and stripped.GetAtomWithIdx(0).GetAtomicNum() == 1:
        return "[H]"
    try:
        return chem.MolToSmiles(stripped)
    except Exception:  # noqa: BLE001 - honest fallback over a crash
        return "[H]"


def rgroup_decomposition(smiles: Iterable[str], core: str) -> list[RGroup]:
    """Read the R-groups at each marked attachment of a common core.

    ``core`` is a SMARTS or SMILES string with one or more attachment
    atoms marked by an isotope (``[*:1]``, ``[*:2]`` ...); every molecule
    is matched to the core and the substituent growing from each marked
    atom is read out.  This is the table a Free-Wilson / R-group SAR
    analysis starts from.

    Parameters
    ----------
    smiles:
        The molecule series.
    core:
        SMARTS/SMILES core with marked attachment atoms.

    Returns
    -------
    list of :class:`RGroup`, one per input SMILES (in order); molecules
    that fail to parse or match come back with ``valid=False``.

    Raises
    ------
    DependencyError
        When RDKit is not installed.
    DataValidationError
        When ``core`` is not valid SMARTS or SMILES.

    Examples
    --------
    >>> from pharmaml.ml.sar import rgroup_decomposition
    >>> rows = rgroup_decomposition(["c1ccccc1O", "c1ccccc1N"], "c1ccccc1[*:1]")
    >>> [r.valid for r in rows]
    [True, True]
    >>> sorted(rows[0].groups.values()), sorted(rows[1].groups.values())
    (['O'], ['N'])
    """
    chem = _require_rdkit()
    from rdkit.Chem import rdRGroupDecomposition

    core_mol = chem.MolFromSmarts(core)
    if core_mol is None:
        core_mol = chem.MolFromSmiles(core)
    if core_mol is None:
        raise DataValidationError(f"core {core!r} is neither valid SMARTS nor SMILES")
    marked = [
        atom.GetIdx()
        for atom in core_mol.GetAtoms()
        if atom.GetIsotope() != 0 or atom.GetAtomicNum() == 0
    ]

    smiles_list = [str(s) for s in smiles]
    decomp = rdRGroupDecomposition.RGroupDecomposition(core_mol)
    accepted: list[int] = []
    for idx, smi in enumerate(smiles_list):
        mol = chem.MolFromSmiles(smi)
        if mol is not None and decomp.Add(mol) >= 0:
            accepted.append(idx)
    decomp.Process()
    columns = decomp.GetRGroupsAsColumns()

    out: list[RGroup] = [RGroup(smiles=smi, valid=False) for smi in smiles_list]
    for pos, idx in enumerate(accepted):
        groups: dict[int, str] = {}
        for order in range(1, len(marked) + 1):
            key = f"R{order}"
            if key not in columns:
                continue
            mol = columns[key][pos]
            groups[order] = _strip_dummies(chem, mol)
        out[idx] = RGroup(smiles=smiles_list[idx], groups=groups, valid=True)
    return out


def rgroup_table(
    frame: pd.DataFrame,
    smiles_col: str,
    core: str,
    *,
    activity_col: str | None = None,
) -> pd.DataFrame:
    """The R-group decomposition of a table as a frame.

    One row per input SMILES with ``R1``/``R2``/... columns holding the
    substituent at each marked core atom, ``valid`` flagging whether the
    core matched, and the activity column carried through when given.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"smiles": ["c1ccccc1O", "c1ccccc1N"], "y": [5.0, 6.0]})
    >>> table = rgroup_table(frame, "smiles", "c1ccccc1[*:1]", activity_col="y")
    >>> table[["smiles", "R1", "y"]].to_dict("records")
    [{'smiles': 'c1ccccc1O', 'R1': 'O', 'y': 5.0}, {'smiles': 'c1ccccc1N', 'R1': 'N', 'y': 6.0}]
    """
    if smiles_col not in frame.columns:
        raise DataValidationError(
            f"column {smiles_col!r} not in frame; available: {list(frame.columns)}"
        )
    groups = rgroup_decomposition(list(frame[smiles_col]), core)
    table = pd.DataFrame([g.as_dict() for g in groups])
    if activity_col is not None:
        if activity_col not in frame.columns:
            raise DataValidationError(
                f"column {activity_col!r} not in frame; available: {list(frame.columns)}"
            )
        table[activity_col] = list(frame[activity_col])
    r_cols = sorted(
        (c for c in table.columns if c.startswith("R") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    ordered = ["smiles", *r_cols, "valid"]
    if activity_col is not None:
        ordered.append(activity_col)
    return table[[c for c in ordered if c in table.columns]]
