"""Train/test split strategies for QSAR data (round 179).

:func:`~pharmaml.ml.preprocessing.make_train_test_split` ships exactly
one strategy: a shuffled (optionally stratified) random split.  Real
QSAR work needs more honest divisions, because a random split lets
near-duplicate molecules land on both sides of the fence and the
reported metric then measures memorisation, not generalisation.  This
module adds the strategies the field actually uses, all returning the
familiar ``(X_train, X_test, y_train, y_test)`` quartet:

* ``random`` — the shuffled split, byte-identical to the legacy
  ``preprocessing.make_train_test_split(..., stratify=False)`` path
  (both delegate to the same sklearn call on an index array).
* ``stratified`` — class-balanced split for classification targets
  (the legacy default, made explicit); continuous targets quietly fall
  back to ``random``, the same quiet path preprocessing takes.
* ``scaffold`` — Bemis–Murcko scaffold grouping: molecules sharing a
  scaffold ALWAYS land on the same side of the split.  The greedy
  assignment orders scaffold groups largest-first and fills the training
  target before the test target, reserving a group for testing. This
  tends to put rarer scaffolds in the test set, though sizes are only
  approximate and scaffold disjointness does not guarantee low similarity.
  The scaffold key itself (:func:`murcko_scaffold_key`) is computed by
  the toolkit's own numpy SMILES engine
  (:func:`pharmaml.chem.molgraph.parse_smiles`) — no RDKit needed.
* ``group`` — caller-supplied group labels (assay batch, supplier
  plate, compound series): whole groups are assigned together, group
  order shuffled by the seed.
* ``kennard_stone`` — the classic chemometrics maximal-minimal
  distance selection: the training set is built one molecule at a time
  by always picking the point FURTHEST (minimum distance) from the
  already-selected set, over autoscaled features.  The selected spread
  set is the training set; the remainder is the test set.  Deterministic
  and seed-free (the algorithm has no randomness at all).
* ``temporal`` — order-as-given: the first rows form the training set,
  the LAST rows the test set (the "train on the past, predict the
  future" protocol for time-ordered datasets).

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.ml.splits import (
...     SPLIT_STRATEGIES, make_train_test_split, murcko_scaffold_key)

>>> SPLIT_STRATEGIES
('random', 'stratified', 'scaffold', 'group', 'kennard_stone', 'temporal')

The scaffold key is the Bemis–Murcko framework — ring systems plus
linkers, side chains and exocyclic substituents pruned — written as a
canonical SMILES by the toolkit's own parser:

>>> murcko_scaffold_key("CC(=O)Oc1ccccc1C(=O)O")    # aspirin: the bare ring
'c1ccccc1'
>>> murcko_scaffold_key("c1ccccc1CCc1ccccc1")       # two rings + ethyl linker: kept
'c1ccc(cc1)CCc2ccccc2'
>>> murcko_scaffold_key("c1ccccc1CCCC(=O)O")        # phenylacetic acid: tail goes
'c1ccccc1'
>>> murcko_scaffold_key("CCO")                      # acyclic: empty framework
''

The scaffold split never splits a scaffold group across the fence:

>>> smiles = ["CC(=O)Oc1ccccc1C(=O)O"] * 4 + ["c1ccccc1CCc1ccccc1"] * 2
>>> X = pd.DataFrame({"a": np.arange(6.0)})
>>> y = pd.Series([0, 0, 0, 0, 1, 1])
>>> Xtr, Xte, ytr, yte = make_train_test_split(
...     X, y, strategy="scaffold", test_size=0.5, smiles=smiles)
>>> sorted(ytr.tolist())                      # the 4-molecule aspirin group trains
[0, 0, 0, 0]
>>> sorted(yte.tolist())                      # the diphenethyl group tests
[1, 1]

Kennard–Stone picks the spread set deterministically (no seed):

>>> X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]})
>>> y = pd.Series(range(6))
>>> _, Xte, _, yte = make_train_test_split(X, y, strategy="kennard_stone", test_size=0.5)
>>> sorted(int(v) for v in yte)
[1, 2, 4]

Temporal split: first rows train, last rows test:

>>> y = pd.Series([10, 20, 30, 40, 50])
>>> _, _, _, yte = make_train_test_split(X.iloc[:5], y, strategy="temporal", test_size=0.4)
>>> yte.tolist()
[40, 50]

Bad input is named, not crashed on:

>>> make_train_test_split(X, y, strategy="nope")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: unknown split strategy 'nope' -- valid strategies: random, stratified, scaffold, group, kennard_stone, temporal
>>> make_train_test_split(X, y, strategy="scaffold")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: strategy 'scaffold' needs one SMILES per row -- pass smiles=[...] aligned with X
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Sized

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.validation import looks_continuous

__all__ = [
    "SPLIT_STRATEGIES",
    "make_train_test_split",
    "murcko_scaffold_key",
    "scaffold_groups",
    "split_indices",
]

_LOG = get_logger(__name__)

#: The registry of split strategies (the playground's combobox values).
SPLIT_STRATEGIES: tuple[str, ...] = (
    "random",
    "stratified",
    "scaffold",
    "group",
    "kennard_stone",
    "temporal",
)


# ---------------------------------------------------------------------------
# the Bemis-Murcko scaffold key (numpy only -- the chem.molgraph engine)
# ---------------------------------------------------------------------------


def murcko_scaffold_key(smiles: str) -> str:
    """Canonical Bemis–Murcko scaffold SMILES for one molecule.

    The framework is ring systems plus the linkers between them; side
    chains and exocyclic substituents are pruned iteratively (a
    non-ring atom of residual degree 1 is a terminal branch and goes,
    repeatedly, until nothing terminal is left — aspirin's framework is
    the bare benzene ring, its carbonyls being substituents, not
    linkers).  The surviving subgraph is re-canonicalised by the
    toolkit's own :func:`pharmaml.chem.molgraph.canonical_smiles`, so
    the key is a plain string, comparable across tautomers of writing
    (``OCC`` vs ``C(C)O`` yield the same framework key when their
    frameworks agree).

    Parameters
    ----------
    smiles:
        One SMILES string. Hostile/unparsable input returns ``""``.
        Acyclic structures also return ``""``; scaffold grouping puts all
        such rows in the same group. This function does not distinguish
        invalid from acyclic input; validate structures before splitting.

    Examples
    --------
    >>> murcko_scaffold_key("c1ccccc1CCc1ccccc1")   # two rings + linker: the framework
    'c1ccc(cc1)CCc2ccccc2'
    >>> murcko_scaffold_key("CCCCCC")                # a plain chain: no framework
    ''
    >>> murcko_scaffold_key("not a smiles at all")   # hostile input: '', never a crash
    ''
    """
    try:
        from pharmaml.chem.molgraph import (
            MolAtom,
            MolBond,
            Molecule,
            canonical_smiles,
            find_rings,
            parse_smiles,
        )
    except Exception:  # noqa: BLE001 - any import-time failure in the chem
        # layer must degrade the scaffold key to "", never crash a split
        _LOG.warning("chem.molgraph unavailable -- scaffold key degrades to ''")
        return ""
    try:
        mol = parse_smiles(smiles)
    except Exception:  # noqa: BLE001 - hostile SMILES must not crash a split
        return ""
    ring_atoms: set[int] = set()
    for ring in find_rings(mol):
        ring_atoms.update(ring)
    # Iterative terminal pruning: a non-ring atom whose *kept* degree
    # drops to 1 is a side chain and goes.  Rings are never pruned.
    adjacency = mol.adjacency()
    kept = set(range(len(mol.atoms)))
    changed = True
    while changed:
        changed = False
        for index in sorted(kept):
            if index in ring_atoms:
                continue
            degree = sum(1 for neighbour in adjacency[index] if neighbour in kept)
            if degree <= 1:
                kept.discard(index)
                changed = True
    if not kept:
        return ""
    remap = {old: new for new, old in enumerate(sorted(kept))}
    atoms = [
        MolAtom(
            index=remap[atom.index],
            symbol=atom.symbol,
            atomic_num=atom.atomic_num,
            is_aromatic=atom.is_aromatic,
            charge=atom.charge,
            isotope=atom.isotope,
            bracket_h=atom.bracket_h,
        )
        for atom in mol.atoms
        if atom.index in kept
    ]
    bonds = [
        MolBond(
            begin=remap[bond.begin],
            end=remap[bond.end],
            order=bond.order,
            is_aromatic=bond.is_aromatic,
        )
        for bond in mol.bonds
        if bond.begin in kept and bond.end in kept
    ]
    scaffold = Molecule(source=smiles, atoms=atoms, bonds=bonds)
    try:
        return canonical_smiles(scaffold)
    except Exception:  # noqa: BLE001 - exotic fragments must not crash a split
        return ""


def scaffold_groups(smiles: Sequence[str]) -> list[str]:
    """Scaffold key per SMILES, in input order (``[""]`` for failures).

    Examples
    --------
    >>> scaffold_groups(["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CCO"])
    ['', 'c1ccccc1', '']
    """
    return [murcko_scaffold_key(str(item)) for item in smiles]


# ---------------------------------------------------------------------------
# the strategy core: indices only
# ---------------------------------------------------------------------------


def _validate_test_size(test_size: float) -> float:
    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool):
        raise DataValidationError(f"test_size must be a number, got {test_size!r}")
    if not math.isfinite(float(test_size)) or not 0.0 < float(test_size) < 1.0:
        raise DataValidationError(f"test_size must be in (0, 1), got {test_size}")
    return float(test_size)


def _greedy_group_assignment(
    groups: Sequence[object],
    n_train: int,
    n_test: int,
    order: Sequence[object],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign whole groups to train/test with a greedy train-first policy.

    Groups are consumed in ``order``; the training quota fills first,
    then the test quota; any leftover groups go back to the training
    side. If the quota would consume every group, reserve the final
    group for testing. Guarantees: no group spans both sides; both sides
    are non-empty when at least two groups exist. Requested sizes are
    targets, not exact quotas.
    """
    members: dict[object, list[int]] = {}
    for index, key in enumerate(groups):
        members.setdefault(key, []).append(index)
    train: list[int] = []
    test: list[int] = []
    for position, key in enumerate(order):
        if position == len(order) - 1 and not test and train:
            test.extend(members[key])
        elif len(train) < n_train:
            train.extend(members[key])
        elif len(test) < n_test:
            test.extend(members[key])
        else:
            train.extend(members[key])
    if not test:
        raise DataValidationError(
            "the whole dataset shares one group/scaffold -- no group-honest "
            "test set exists; use another strategy (random, kennard_stone, ...)"
        )
    if not train:
        raise DataValidationError("the split left the training side empty")
    return np.asarray(sorted(train), dtype=int), np.asarray(sorted(test), dtype=int)


def _kennard_stone_indices(X: np.ndarray, n_train: int) -> np.ndarray:
    """Max-min (Kennard–Stone) selection of the training set on autoscaled X."""
    n = X.shape[0]
    if n < 2:
        raise DataValidationError("kennard_stone needs at least two rows")
    if np.any(~np.isfinite(X)):
        raise DataValidationError(
            "kennard_stone needs finite features -- impute or drop NaN rows first"
        )
    if n > 3000:
        _LOG.warning(
            "kennard_stone is O(n^2): %d rows will take a moment", n
        )
    # autoscale so every feature contributes equally to the distance
    spread = X.std(axis=0)
    spread[spread == 0.0] = 1.0
    Z = (X - X.mean(axis=0)) / spread
    min_dist = np.full(n, np.inf)
    selected: list[int] = []
    # the classic first pick: the point furthest from the (scaled) centroid
    norms = np.linalg.norm(Z, axis=1)
    current = int(np.argmax(norms))
    for _ in range(n_train):
        selected.append(current)
        dist = np.linalg.norm(Z - Z[current], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected] = -np.inf  # never re-select
        current = int(np.argmax(min_dist))
    return np.asarray(sorted(selected), dtype=int)


def split_indices(
    X: object = None,
    y: object = None,
    *,
    strategy: str = "random",
    test_size: float = 0.2,
    random_state: int = 42,
    smiles: Sequence[str] | None = None,
    groups: Sequence[object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Core dispatcher: ``(train_indices, test_indices)`` for a strategy.

    At least one of ``X`` / ``y`` must be given (its length defines the
    dataset). If both are supplied they must have the same length; rows
    are aligned positionally, not by pandas index. ``random`` and
    ``stratified`` delegate to sklearn's own
    :func:`~sklearn.model_selection.train_test_split` on an index array,
    so the index semantics match the legacy
    :func:`~pharmaml.ml.preprocessing.make_train_test_split` exactly.

    Parameters
    ----------
    X:
        Feature matrix (any 2-D array/DataFrame); required by
        ``kennard_stone`` (distances are computed on it), ignored by
        the label strategies.
    y:
        Target vector; required by ``stratified``.
    strategy:
        One of :data:`SPLIT_STRATEGIES`.
    test_size:
        Fraction on (0, 1); target ``n_test = ceil(test_size * n)``.
        Group/scaffold splits keep groups whole, so actual sizes can differ.
    random_state:
        Seed for ``random`` / ``stratified`` / ``group`` shuffles.
    smiles:
        One SMILES per row (aligned with ``X``/``y``); required by
        ``scaffold``. Acyclic molecules and failed parses share the empty
        scaffold group; validate SMILES before use.
    groups:
        One group label per row; required by ``group``. Labels are
        converted to strings before grouping.

    Raises
    ------
    DataValidationError
        On an unknown strategy, a missing prerequisite (``smiles``/
        ``groups``/``y``), non-finite features for ``kennard_stone``,
        or a degenerate configuration that cannot produce both sides.
    """
    if strategy not in SPLIT_STRATEGIES:
        raise DataValidationError(
            f"unknown split strategy {strategy!r} -- valid strategies: "
            + ", ".join(SPLIT_STRATEGIES)
        )
    test_size = _validate_test_size(test_size)
    source: object = X if X is not None else y
    if source is None:
        raise DataValidationError("split_indices needs X or y to size the dataset")
    if not isinstance(source, Sized):
        raise DataValidationError(
            "X/y must be sized containers (arrays, DataFrames, lists)"
        )
    try:
        n = len(source)
    except TypeError as exc:
        raise DataValidationError("X/y must be sized containers") from exc
    if X is not None and y is not None:
        if not isinstance(y, Sized):
            raise DataValidationError("X/y must be sized containers")
        try:
            n_y = len(y)
        except TypeError as exc:
            raise DataValidationError("X/y must be sized containers") from exc
        if n != n_y:
            raise DataValidationError(
                f"X has {n} rows but y has {n_y} values; they must have the same length"
            )
    if n < 2:
        raise DataValidationError(
            f"a split needs at least two rows, got {n}"
        )
    n_test = math.ceil(test_size * n)
    n_train = n - n_test
    if n_train < 1 or n_test < 1:
        raise DataValidationError(
            f"test_size {test_size} leaves an empty side at n={n}"
        )

    if strategy in ("random", "stratified"):
        if strategy == "stratified":
            if y is None:
                raise DataValidationError(
                    "strategy 'stratified' needs y (the class labels)"
                )
            if looks_continuous(y):
                _LOG.warning(
                    "continuous target -- stratification is not a thing; "
                    "falling back to a plain random split"
                )
            else:
                indices = np.arange(n)
                train_idx, test_idx = _sklearn_split(
                    indices,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=np.asarray(y),
                )
                return np.sort(train_idx), np.sort(test_idx)
        indices = np.arange(n)
        train_idx, test_idx = _sklearn_split(
            indices, test_size=test_size, random_state=random_state
        )
        return np.sort(train_idx), np.sort(test_idx)

    if strategy == "scaffold":
        if smiles is None or len(smiles) != n:
            raise DataValidationError(
                "strategy 'scaffold' needs one SMILES per row -- pass "
                "smiles=[...] aligned with X"
            )
        if groups is not None:
            raise DataValidationError(
                "groups= is only used by strategy 'group' (got both)"
            )
        keys = scaffold_groups(smiles)
        order = sorted(set(keys), key=lambda k: (-keys.count(k), str(k)))
        return _greedy_group_assignment(keys, n_train, n_test, order)

    if strategy == "group":
        if groups is None or len(groups) != n:
            raise DataValidationError(
                "strategy 'group' needs one group label per row -- pass "
                "groups=[...] aligned with X"
            )
        if smiles is not None:
            raise DataValidationError(
                "smiles= is only used by strategy 'scaffold' (got both)"
            )
        rng = np.random.default_rng(random_state)
        unique = sorted({str(g) for g in groups}, key=str)
        shuffled = unique[:]
        rng.shuffle(shuffled)
        keys = [str(g) for g in groups]
        return _greedy_group_assignment(keys, n_train, n_test, list(shuffled))

    if strategy == "kennard_stone":
        if X is None:
            raise DataValidationError(
                "strategy 'kennard_stone' needs X (the feature matrix)"
            )
        matrix = np.asarray(
            X.to_numpy(dtype=float)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=float),
            dtype=float,
        )
        if matrix.ndim != 2 or matrix.shape[0] != n:
            raise DataValidationError(
                "kennard_stone needs a 2-D feature matrix aligned with y"
            )
        train_idx = _kennard_stone_indices(matrix, n_train)
        test_idx = np.setdiff1d(np.arange(n), train_idx)
        return train_idx, test_idx

    # strategy == "temporal"
    test_idx = np.arange(n - n_test, n)
    train_idx = np.arange(n - n_test)
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# the sklearn-shaped facade
# ---------------------------------------------------------------------------


def make_train_test_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    strategy: str = "random",
    test_size: float = 0.2,
    random_state: int = 42,
    smiles: Sequence[str] | None = None,
    groups: Sequence[object] | None = None,
    stratify: bool = True,
):
    """Strategy-aware train/test split, drop-in for the preprocessing twin.

    Same quartet return as
    :func:`pharmaml.ml.preprocessing.make_train_test_split` —
    ``(X_train, X_test, y_train, y_test)`` — but the division is chosen
    by ``strategy`` (see :data:`SPLIT_STRATEGIES`).  ``stratify`` is
    kept for signature symmetry and only consulted by the legacy
    ``random``-family behaviour: with ``strategy="random"`` and
    ``stratify=True`` (the default) a classification-like target is
    stratified exactly like the preprocessing twin, so
    ``strategy="random"`` alone reproduces the historical split.

    Raises
    ------
    DataValidationError
        When ``y`` is empty or holds NaN, ``test_size`` is out of
        range, or the strategy's prerequisites are missing (see
        :func:`split_indices`).
    """
    if len(np.asarray(y)) == 0:
        raise DataValidationError(
            "y is empty -- make_train_test_split needs at least one labeled sample"
        )
    n_missing = int(pd.isna(np.asarray(y)).sum())
    if n_missing:
        raise DataValidationError(
            f"y contains {n_missing} NaN value(s) -- drop or impute the target "
            "before splitting (load_dataset drops missing-target rows for you)"
        )
    effective = strategy
    if strategy == "random" and stratify and not looks_continuous(y):
        effective = "stratified"
    train_idx, test_idx = split_indices(
        X,
        y,
        strategy=effective,
        test_size=test_size,
        random_state=random_state,
        smiles=smiles,
        groups=groups,
    )
    if isinstance(X, pd.DataFrame):
        x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    else:
        x_train, x_test = np.asarray(X)[train_idx], np.asarray(X)[test_idx]
    if isinstance(y, pd.Series):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        y_train, y_test = np.asarray(y)[train_idx], np.asarray(y)[test_idx]
    return x_train, x_test, y_train, y_test
