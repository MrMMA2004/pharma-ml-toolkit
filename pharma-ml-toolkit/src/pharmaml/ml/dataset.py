"""The MoleculeNet data layer: Dataset, splitters, transformers (100x-175).

``deepchem/deepchem`` was reviewed for this module (round 175) file by
file -- ``data/datasets.py`` (the ``Dataset`` ABC, ``NumpyDataset``,
Welford statistics), ``splits/splitters.py`` (the splitter family and
above all ``ScaffoldSplitter.generate_scaffolds``'s exact greedy fill),
``splits/task_splitter.py`` (leave-task-out), ``trans/transformers.py``
(``Normalization``/``MinMax``/``Log``/``Clipping``/``Balancing``) and
``molnet/load_function/`` (the dataset registry that pins the canonical
benchmark configuration per dataset).  ``chemprop/chemprop``'s
``data/splitting.py`` delegates the same scaffold split to astartes,
and ``mims-harvard/TDC``'s ``utils/split.py`` implements the same
Murcko-group-then-greedy-fill idea with a big-set shuffle first; the
deepchem variant is the one ported verbatim here because it is the
reference the MoleculeNet leaderboard used.

The contract, kept faithful:

* :class:`Dataset` holds ``X, y, w, ids`` with deepchem's defaulting
  rules -- ``y=None`` means *unlabelled* (zeros plus ``w=0``, so loss
  terms vanish), ``w=None`` means all ones, ``ids`` default to
  ``arange`` and may be SMILES strings (the scaffold splitter relies
  on exactly that).
* ``get_statistics`` is Welford's streaming algorithm evaluated over
  batches, giving deepchem's *population* standard deviation (ddof=0)
  with the ``n < 2 -> 0`` guard.
* :func:`scaffold_split` reproduces deepchem's ``ScaffoldSplitter``:
  Murcko-scaffold groups (Bemis-Murcko frameworks -- side chains
  pruned, rings and ring-linkers kept), groups sorted descending by
  size with descending-first-member tie-break, and the float-cutoff
  greedy fill that puts large common scaffolds in train and rare ones
  in test -- the canonical "hard" extrapolation split.
* :func:`task_split` is deepchem's ``TaskSplitter``: leave-task-out
  holds out *task columns* of a sparse multitask matrix (X shared).
* :class:`BalancingTransformer` is deepchem's exact inverse-frequency
  rule ``w_c = N_task / n_c`` per task over labelled samples, with
  ``w == 0`` (missing label) preserved and the rest overwritten.
* :class:`MOLNET_REGISTRY` pins the canonical loader configuration per
  benchmark (tasks, type, splitter, transformers) as documentation
  constants: ``delaney``-ESOL through ``pcba``.

What is deliberately *not* ported, stated once: ``DiskDataset``
(sharded on-disk storage -- the toolkit is in-memory at this scale),
``ImageDataset``, the DataLoader CSV/SDF machinery (the toolkit's own
readers cover that), the Butina/MaxMin/Fingerprint splitters (need
RDKit picker loops; the random/scaffold/task trio covers the benchmark
defaults), and the CDF/DAG/IRV transformers.  The scaffold key itself
is computed by a pure-NumPy pruned-graph canonicalisation of the
toolkit's own SMILES parser (iterative leaf stripping -- exactly the
Bemis-Murcko framework definition); pass a custom ``scaffold_fn`` to
swap in RDKit's ``MurckoScaffoldSmiles`` when exactness against a
reference run matters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from pharmaml.chem.molgraph import Molecule, parse_smiles
from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "MOLNET_REGISTRY",
    "BalancingTransformer",
    "ClippingTransformer",
    "Dataset",
    "LogTransformer",
    "MinMaxTransformer",
    "NormalizationTransformer",
    "evaluate",
    "index_split",
    "molnet_registry_frame",
    "murcko_scaffold_key",
    "random_split",
    "scaffold_split",
    "task_split",
]

_LOG = get_logger(__name__)


# ------------------------------------------------------------------ #
# Dataset                                                              #
# ------------------------------------------------------------------ #


class Dataset:
    """An ``X / y / w / ids`` table -- deepchem's NumpyDataset contract.

    ``y`` may be ``None`` (unlabelled data: zeros with ``w=0`` so no
    loss term), ``w`` may be ``None`` (all samples count), and ``ids``
    defaults to ``0..n-1`` but accepts SMILES strings, which the
    scaffold splitter groups on.  ``y`` is always normalised to 2-D
    ``(n_samples, n_tasks)``; ``w`` follows its shape.

    Examples
    --------
    >>> ds = Dataset(X=np.arange(6.0).reshape(3, 2), y=np.array([0., 1., 1.]))
    >>> ds.n_tasks
    1
    >>> ds.w.T.tolist()
    [[1.0, 1.0, 1.0]]
    >>> unlabelled = Dataset(X=np.zeros((2, 2)))
    >>> float(unlabelled.w.sum())
    0.0
    >>> ds.get_statistics()[2].round(6)   # y mean, population stats
    array([0.666667])
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        w: np.ndarray | None = None,
        ids: Sequence[Any] | np.ndarray | None = None,
    ) -> None:
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X[:, None]
        if self.X.ndim != 2:
            raise DataValidationError(f"X must be 2-D, got shape {self.X.shape}")
        n = self.X.shape[0]
        if y is None:
            self.y = np.zeros((n, 1))
            self.w = np.zeros((n, 1))
        else:
            self.y = np.asarray(y, dtype=float)
            if self.y.ndim == 1:
                self.y = self.y[:, None]
            if self.y.shape[0] != n:
                raise DataValidationError(
                    f"X has {n} rows but y has {self.y.shape[0]}"
                )
            if w is None:
                self.w = np.ones(self.y.shape)
            else:
                self.w = np.asarray(w, dtype=float)
                if self.w.ndim == 1:
                    self.w = self.w[:, None]
                if self.w.shape != self.y.shape:
                    raise DataValidationError("w must match y's shape")
        if ids is None:
            self.ids = np.arange(n, dtype=object)
        else:
            ids_arr = np.asarray(ids, dtype=object)
            if ids_arr.shape[0] != n:
                raise DataValidationError(
                    f"ids has {ids_arr.shape[0]} entries but X has {n} rows"
                )
            self.ids = ids_arr

    # -- basic views ------------------------------------------------- #

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_tasks(self) -> int:
        """Number of task columns in ``y``."""
        return int(self.y.shape[1])

    @property
    def task_names(self) -> list[str]:
        """Task labels (``task_0..`` by default, deepchem's convention)."""
        return [f"task_{i}" for i in range(self.n_tasks)]

    def get_shape(self) -> tuple[tuple[int, ...], ...]:
        """``(X_shape, y_shape, w_shape, ids_shape)`` -- deepchem's contract."""
        return self.X.shape, self.y.shape, self.w.shape, (len(self.ids),)

    def label_mask(self, task: int) -> np.ndarray:
        """Samples with a measured label for ``task`` (``w != 0`` and finite y)."""
        return (self.w[:, task] != 0.0) & np.isfinite(self.y[:, task])

    # -- iteration --------------------------------------------------- #

    def itersamples(self) -> Iterable[tuple[np.ndarray, float, float, Any]]:
        """Per-sample ``(x_row, y_row, w_row, id)`` tuples."""
        for i in range(len(self)):
            yield self.X[i], self.y[i], self.w[i], self.ids[i]

    def iterbatches(
        self,
        batch_size: int | None = None,
        epochs: int = 1,
        deterministic: bool = True,
        seed: int | None = 0,
    ) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(X, y, w, ids)`` batches, optionally reshuffled per epoch.

        Non-deterministic iteration reshuffles the index order with a
        fresh permutation every epoch (deepchem's contract; the seed
        advances per epoch so epochs differ).

        Examples
        --------
        >>> ds = Dataset(X=np.arange(4.0).reshape(4, 1), y=[0, 1, 0, 1])
        >>> batches = list(ds.iterbatches(batch_size=3))
        >>> [b[0].shape[0] for b in batches]
        [3, 1]
        >>> orders = [list(b[3]) for b in ds.iterbatches(
        ...     batch_size=4, epochs=2, deterministic=False, seed=7)]
        >>> orders[0] == orders[1]
        False
        """
        n = len(self)
        if batch_size is None or batch_size <= 0:
            batch_size = n
        rng = np.random.default_rng(seed)
        for _ in range(max(1, int(epochs))):
            order = (
                np.arange(n)
                if deterministic
                else rng.permutation(n)
            )
            for start in range(0, n, batch_size):
                take = order[start : start + batch_size]
                yield self.X[take], self.y[take], self.w[take], self.ids[take]

    # -- slicing / merging ------------------------------------------- #

    def select(self, indices: Sequence[int] | np.ndarray) -> Dataset:
        """A new Dataset restricted to ``indices`` (the splitters' tool)."""
        idx = np.asarray(indices, dtype=int)
        return Dataset(X=self.X[idx], y=self.y[idx], w=self.w[idx], ids=self.ids[idx])

    @classmethod
    def merge(cls, datasets: Sequence[Dataset]) -> Dataset:
        """Concatenate datasets row-wise (deepchem's ``NumpyDataset.merge``)."""
        if not datasets:
            raise DataValidationError("merge needs at least one dataset")
        return cls(
            X=np.vstack([d.X for d in datasets]),
            y=np.vstack([d.y for d in datasets]),
            w=np.vstack([d.w for d in datasets]),
            ids=np.concatenate([d.ids for d in datasets]),
        )

    # -- statistics ---------------------------------------------------- #

    def get_statistics(
        self, X_stats: bool = True, y_stats: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-feature/per-task ``(X_mean, X_std, y_mean, y_std)``.

        Welford's streaming algorithm evaluated over batches of 1024 --
        mathematically the two-pass moments but numerically the same
        accumulation deepchem's per-sample loop performs.  The std is
        the *population* std (ddof=0) and ``n < 2`` gives zeros, both
        deepchem's contract.

        Examples
        --------
        >>> ds = Dataset(X=np.array([[1.0], [3.0]]), y=np.array([[2.0], [4.0]]))
        >>> xm, xs, ym, ys = ds.get_statistics()
        >>> [round(float(v), 4) for v in (xm[0], xs[0], ym[0], ys[0])]
        [2.0, 1.0, 3.0, 1.0]
        """
        out_X = (
            self._welford(self.X, columns=True)
            if X_stats
            else (np.zeros(self.X.shape[1]), np.zeros(self.X.shape[1]))
        )
        out_y = (
            self._welford(self.y, columns=True)
            if y_stats
            else (np.zeros(self.n_tasks), np.zeros(self.n_tasks))
        )
        return out_X[0], out_X[1], out_y[0], out_y[1]

    @staticmethod
    def _welford(
        array: np.ndarray, columns: bool = True, batch: int = 1024
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched Welford mean/variance along axis 0 (population std)."""
        n_features = array.shape[1] if columns else 1
        mean = np.zeros(n_features)
        m2 = np.zeros(n_features)
        count = 0
        for start in range(0, array.shape[0], batch):
            chunk = array[start : start + batch]
            if not columns:
                chunk = chunk.reshape(-1, 1)
            for row in chunk:
                count += 1
                delta = row - mean
                mean += delta / count
                m2 += delta * (row - mean)
        std = np.sqrt(m2 / count) if count >= 2 else np.zeros(n_features)
        if not columns:
            mean = np.array([mean[0]])
            std = np.array([std[0]])
        return mean, std

    # -- dataframe bridge ---------------------------------------------- #

    def to_dataframe(self) -> Any:
        """A tidy frame: ``X1..Xp``/``X``, ``y1..yn``, ``w1..wn``, ``ids``."""
        import pandas as pd

        frame: dict[str, Any] = {"ids": list(self.ids)}
        if self.X.shape[1] == 1:
            frame["X"] = self.X[:, 0]
        else:
            for j in range(self.X.shape[1]):
                frame[f"X{j + 1}"] = self.X[:, j]
        for t in range(self.n_tasks):
            frame[f"y{t + 1}"] = self.y[:, t]
            frame[f"w{t + 1}"] = self.w[:, t]
        return pd.DataFrame(frame)

    @classmethod
    def from_dataframe(cls, frame: Any) -> Dataset:
        """Rebuild from :meth:`to_dataframe` output."""
        x_cols = [c for c in frame.columns if c == "X" or (c.startswith("X") and c[1:].isdigit())]
        y_cols = [c for c in frame.columns if c.startswith("y") and c[1:].isdigit()]
        w_cols = [c for c in frame.columns if c.startswith("w") and c[1:].isdigit()]
        X = frame[x_cols].to_numpy(dtype=float)
        y = frame[y_cols].to_numpy(dtype=float)
        w = frame[w_cols].to_numpy(dtype=float) if w_cols else None
        return cls(X=X, y=y, w=w, ids=list(frame["ids"]))

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"Dataset(n={len(self)}, n_features={self.X.shape[1]}, "
            f"n_tasks={self.n_tasks}, labelled={int(self.w.sum())})"
        )


# ------------------------------------------------------------------ #
# Murcko scaffold key (pure path)                                     #
# ------------------------------------------------------------------ #


def murcko_scaffold_key(smiles: str) -> str:
    """A canonical Bemis-Murcko framework key, pure NumPy/graph only.

    The framework is the side-chain-pruned graph: atoms of degree <= 1
    are stripped iteratively (leaves first, so chains vanish and ring
    linkers survive), and what remains is serialised deterministically
    (elements sorted, bonds as sorted endpoint pairs with orders,
    ring-closure counts appended).  Two molecules share a key iff they
    share the pruned framework topology -- deepchem groups on exactly
    this equivalence class (it uses RDKit's ``MurckoScaffoldSmiles``;
    this pure twin exists so the split stays dependency-free and
    deterministic).

    Examples
    --------
    >>> murcko_scaffold_key("CCO")            # fully acyclic -> empty
    ''
    >>> murcko_scaffold_key("c1ccccc1O") == murcko_scaffold_key("Oc1ccccc1C")
    True
    >>> murcko_scaffold_key("C1CCCCC1CC") == murcko_scaffold_key("CCC1CCCCC1")
    True
    """
    try:
        mol: Molecule | None = parse_smiles(smiles, strict=False)
    except Exception:  # noqa: BLE001 - any parse failure -> no scaffold
        return ""
    if mol is None:
        return ""
    alive = {atom.index for atom in mol.atoms}
    adjacency: dict[int, list[int]] = {i: [] for i in alive}
    for bond in mol.bonds:
        adjacency[bond.begin].append(bond.end)
        adjacency[bond.end].append(bond.begin)
    changed = True
    while changed and len(alive) > 2:
        changed = False
        for i in list(alive):
            degree = sum(1 for j in adjacency[i] if j in alive)
            if degree <= 1:
                alive.discard(i)
                changed = True
    if not alive or (len(alive) == 2 and all(
        sum(1 for j in adjacency[i] if j in alive) <= 1 for i in alive
    )):
        # Nothing survived (acyclic molecule): empty scaffold, deepchem's
        # "no framework" class.
        return ""
    atoms = sorted(
        (mol.atoms[i].symbol, i) for i in alive
    )
    remap = {old: new for new, (_, old) in enumerate(atoms)}
    bonds = sorted(
        (
            remap[bond.begin],
            remap[bond.end],
            1.5 if bond.is_aromatic else bond.order,
        )
        for bond in mol.bonds
        if bond.begin in alive and bond.end in alive
    )
    symbols = "-".join(sym for sym, _ in atoms)
    bond_text = ",".join(f"{a}:{b}:{o:g}" for a, b, o in bonds)
    return f"{symbols}|{bond_text}"


# ------------------------------------------------------------------ #
# Splitters                                                           #
# ------------------------------------------------------------------ #


def _check_fracs(fracs: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(f) for f in fracs)
    if not values or any(f < 0 for f in values):
        raise DataValidationError("split fractions must be non-negative")
    total = sum(values)
    if total <= 0:
        raise DataValidationError("split fractions must sum > 0")
    if abs(total - 1.0) > 1e-9:
        raise DataValidationError(
            f"split fractions must sum to 1, got {total:.6f}"
        )
    return values


def random_split(
    dataset: Dataset,
    fracs: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[Dataset, ...]:
    """Random split -- deepchem's ``RandomSplitter`` (seeded permutation).

    Examples
    --------
    >>> ds = Dataset(X=np.arange(10.0).reshape(10, 1), y=np.arange(10.0))
    >>> train, val, test = random_split(ds, seed=1)
    >>> (len(train), len(val), len(test))
    (8, 1, 1)
    >>> sorted(int(v) for v in train.X[:, 0]) == list(range(8))
    False
    """
    fracs = _check_fracs(fracs)
    n = len(dataset)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    parts: list[Dataset] = []
    start = 0
    for frac in fracs[:-1]:
        stop = start + round(frac * n)
        parts.append(dataset.select(order[start:stop]))
        start = stop
    parts.append(dataset.select(order[start:]))
    return tuple(parts)


def index_split(
    dataset: Dataset, fracs: Sequence[float] = (0.8, 0.1, 0.1)
) -> tuple[Dataset, ...]:
    """Positional split, no shuffle -- deepchem's ``IndexSplitter``.

    For pre-ordered (time, scaffold-sorted) data.

    Examples
    --------
    >>> ds = Dataset(X=np.arange(10.0).reshape(10, 1), y=np.arange(10.0))
    >>> train, val, test = index_split(ds)
    >>> [int(v) for v in test.X[:, 0]]
    [9]
    """
    fracs = _check_fracs(fracs)
    n = len(dataset)
    parts: list[Dataset] = []
    start = 0
    for frac in fracs[:-1]:
        stop = start + round(frac * n)
        parts.append(dataset.select(np.arange(start, stop)))
        start = stop
    parts.append(dataset.select(np.arange(start, n)))
    return tuple(parts)


def scaffold_split(
    dataset: Dataset,
    fracs: Sequence[float] = (0.8, 0.1, 0.1),
    scaffold_fn: Callable[[str], str] | None = None,
) -> tuple[Dataset, ...]:
    """Deepchem's ``ScaffoldSplitter``: Murcko groups, greedy fill.

    Duplicate-symbol map (round 304, part12 A-6): ``scaffold_split``
    also exists at :mod:`pharmaml.ml.benchmark` — the
    registry-referenced index-array variant (SMILES in, index dict +
    ``dropped`` out, chemprop's atom-map zeroing). THIS one is the
    ``Dataset``-API library-surface variant (Datasets in, Datasets
    out, pluggable ``scaffold_fn``); the family's definition sites
    are pinned by ``tests/test_architecture_hygiene.py``.

    Groups are formed by scaffold key (default: the pure
    :func:`murcko_scaffold_key`; pass ``scaffold_fn`` for an RDKit
    ``MurckoScaffoldSmiles`` front end), sorted descending by group
    size with a descending-first-member tie-break, then greedily added
    whole to train/valid/test while they fit under deepchem's *float*
    cutoffs -- so large common frameworks land in train and rare ones
    in test, the extrapolation-hard benchmark split.

    Examples
    --------
    >>> smiles = ["CCO", "CCCO",                      # acyclic pair
    ...           "c1ccccc1O", "Cc1ccccc1N",          # benzene pair
    ...           "C1CCCCC1O", "CC1CCCCC1"]          # cyclohexane pair
    >>> ds = Dataset(
    ...     X=np.arange(6.0).reshape(6, 1), y=np.ones(6), ids=smiles)
    >>> train, val, test = scaffold_split(ds, fracs=(0.5, 0.25, 0.25))
    >>> (len(train), len(val), len(test))
    (2, 2, 2)
    >>> list(train.ids)          # size ties go to the later-index group first
    ['C1CCCCC1O', 'CC1CCCCC1']
    >>> list(val.ids)            # the benzene pair stays together
    ['c1ccccc1O', 'Cc1ccccc1N']
    >>> list(test.ids)
    ['CCO', 'CCCO']
    """
    fracs = _check_fracs(fracs)
    if len(fracs) != 3:
        raise DataValidationError("scaffold_split needs (train, valid, test) fractions")
    key_fn = scaffold_fn or murcko_scaffold_key
    groups: dict[str, list[int]] = {}
    for i, id_ in enumerate(dataset.ids):
        try:
            key = key_fn(str(id_))
        except Exception:  # noqa: BLE001 - unparseable ids join the empty class
            key = ""
        groups.setdefault(key, []).append(i)
    # deepchem's exact ordering: descending size, tie-break descending
    # smallest member index (their sort key reversed).
    ordered = sorted(
        groups.items(),
        key=lambda item: (len(item[1]), item[1][0]),
        reverse=True,
    )
    n = len(dataset)
    train_cutoff = fracs[0] * n
    valid_cutoff = (fracs[0] + fracs[1]) * n
    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for _, members in ordered:
        if len(train) + len(members) > train_cutoff:
            if len(train) + len(valid) + len(members) > valid_cutoff:
                test.extend(members)
            else:
                valid.extend(members)
        else:
            train.extend(members)
    return (
        dataset.select(sorted(train)),
        dataset.select(sorted(valid)),
        dataset.select(sorted(test)),
    )


def task_split(
    dataset: Dataset, frac_train: float = 0.8
) -> tuple[Dataset, Dataset]:
    """Leave-task-out split of the task columns (deepchem ``TaskSplitter``).

    The X block and ids are shared; the *tasks* of a sparse multitask
    matrix are divided into train/holdout sets so cross-task
    generalisation can be measured.

    Examples
    --------
    >>> ds = Dataset(
    ...     X=np.zeros((3, 1)),
    ...     y=np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=float))
    >>> train, held = task_split(ds, frac_train=2 / 3)
    >>> train.n_tasks, held.n_tasks
    (2, 1)
    >>> len(train) == len(ds)
    True
    """
    if not 0.0 < frac_train < 1.0:
        raise DataValidationError("frac_train must be in (0, 1)")
    n_tasks = dataset.n_tasks
    n_train = round(frac_train * n_tasks)
    if n_train == 0 or n_train >= n_tasks:
        raise DataValidationError(
            f"frac_train={frac_train} leaves no held-out task for {n_tasks} tasks"
        )
    train = Dataset(
        X=dataset.X,
        y=dataset.y[:, :n_train],
        w=dataset.w[:, :n_train],
        ids=dataset.ids,
    )
    held = Dataset(
        X=dataset.X,
        y=dataset.y[:, n_train:],
        w=dataset.w[:, n_train:],
        ids=dataset.ids,
    )
    return train, held


# ------------------------------------------------------------------ #
# Transformers                                                        #
# ------------------------------------------------------------------ #


class _ColumnTransformer:
    """Shared fit/apply plumbing for per-column y/X transformers."""

    def __init__(
        self,
        dataset: Dataset,
        transform_X: bool = False,
        transform_y: bool = True,
    ) -> None:
        if transform_X == transform_y:
            raise DataValidationError(
                "choose exactly one of transform_X / transform_y"
            )
        self.transform_X = transform_X
        self.transform_y = transform_y
        self._fit(dataset)

    def _target(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return X if self.transform_X else y

    def _set_target(
        self, X: np.ndarray, y: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.transform_X:
            return values, y
        return X, values

    def _fit(self, dataset: Dataset) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply the element transform to the chosen block (deepchem API)."""
        target = self._target(X, y)
        values = self._forward(target)
        X, y = self._set_target(X, y, values)
        return X, y, w, ids

    def transform(self, dataset: Dataset) -> Dataset:
        """A new Dataset with the chosen block transformed."""
        X, y, w, ids = self.transform_array(
            dataset.X.copy(), dataset.y.copy(), dataset.w.copy(), dataset.ids
        )
        return Dataset(X=X, y=y, w=w, ids=ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        """Invert predictions back to the original units."""
        return self._inverse(np.asarray(z, dtype=float))

    def _forward(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _inverse(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class NormalizationTransformer(_ColumnTransformer):
    """Standardise the chosen block: ``(a - mean) / std`` (std 0 -> 1).

    Statistics come from the *fit* dataset only (train), deepchem's
    leakage-free contract; ``untransform`` maps predictions back.

    Examples
    --------
    >>> ds = Dataset(X=np.array([[1.0], [3.0]]), y=np.array([[10.0], [20.0]]))
    >>> tr = NormalizationTransformer(ds, transform_y=True)
    >>> zd = tr.transform(ds)
    >>> zd.y.round(3).tolist()
    [[-1.0], [1.0]]
    >>> tr.untransform(zd.y).round(6).tolist()
    [[10.0], [20.0]]
    """

    def _fit(self, dataset: Dataset) -> None:
        if self.transform_X:
            mean, std = dataset.get_statistics(X_stats=True, y_stats=False)[:2]
        else:
            mean, std = dataset.get_statistics(X_stats=False, y_stats=True)[2:]
        std = np.where(std == 0.0, 1.0, std)
        self.mean = mean
        self.std = std

    def _forward(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class MinMaxTransformer(_ColumnTransformer):
    """Scale the chosen block into ``[0, 1]`` per column (range 0 -> 1).

    Examples
    --------
    >>> ds = Dataset(X=np.array([[0.0], [5.0]]), y=np.array([[10.0], [30.0]]))
    >>> tr = MinMaxTransformer(ds, transform_y=True)
    >>> tr.transform(ds).y.tolist()
    [[0.0], [1.0]]
    >>> tr.untransform(np.array([[0.5]])).tolist()
    [[20.0]]
    """

    def _fit(self, dataset: Dataset) -> None:
        target = dataset.X if self.transform_X else dataset.y
        self.lo = target.min(axis=0)
        span = target.max(axis=0) - self.lo
        self.span = np.where(span == 0.0, 1.0, span)

    def _forward(self, values: np.ndarray) -> np.ndarray:
        return (values - self.lo) / self.span

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.span + self.lo


class LogTransformer(_ColumnTransformer):
    """``log(a + 1)`` with inverse ``exp(z) - 1`` (deepchem's default base).

    Examples
    --------
    >>> ds = Dataset(X=np.zeros((2, 1)), y=np.array([[0.0], [6.389056]]))
    >>> tr = LogTransformer(ds, transform_y=True)
    >>> tr.transform(ds).y.round(6).tolist()
    [[0.0], [2.0]]
    >>> tr.untransform(np.array([[2.0]])).round(6).tolist()
    [[6.389056]]
    """

    def _fit(self, dataset: Dataset) -> None:
        return None

    def _forward(self, values: np.ndarray) -> np.ndarray:
        return np.log(values + 1.0)

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        return np.exp(values) - 1.0


class ClippingTransformer(_ColumnTransformer):
    """Clip the chosen block at ``±x_max`` (no inverse, deepchem's copy).

    Examples
    --------
    >>> ds = Dataset(X=np.zeros((3, 1)), y=np.array([[-9.0], [0.5], [9.0]]))
    >>> ClippingTransformer(ds, transform_y=True, x_max=5.0).transform(
    ...     ds).y.tolist()
    [[-5.0], [0.5], [5.0]]
    """

    def __init__(
        self,
        dataset: Dataset,
        transform_X: bool = False,
        transform_y: bool = True,
        x_max: float = 5.0,
    ) -> None:
        self.x_max = float(x_max)
        super().__init__(dataset, transform_X=transform_X, transform_y=transform_y)

    def _fit(self, dataset: Dataset) -> None:
        return None

    def _forward(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, -self.x_max, self.x_max)

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        return values  # clipping is not invertible; pass-through


class BalancingTransformer:
    """Class-balancing weights, deepchem's exact inverse-frequency rule.

    Per task (over *labelled* samples, ``w != 0``): ``w_c = N_task /
    n_c`` for class ``c``, so each class's total weight is ``N_task``
    -- for a binary task the positive:negative weight ratio equals the
    negative:positive count ratio.  Existing ``w == 0`` entries (the
    missing-label convention) are preserved; all other weights are
    overwritten, exactly as deepchem does.

    Examples
    --------
    >>> ds = Dataset(X=np.zeros((4, 1)), y=np.array([0, 0, 0, 1], dtype=float))
    >>> tr = BalancingTransformer(ds)
    >>> w = tr.transform(ds).w[:, 0]
    >>> w.round(4).tolist()
    [1.3333, 1.3333, 1.3333, 4.0]
    >>> round(float(w[3] / w[0]), 6)   # pos/neg weight == n_neg/n_pos
    3.0
    >>> sparse = Dataset(X=np.zeros((3, 1)),
    ...                  y=np.array([[0, 1], [1, 0], [1, 1]], dtype=float),
    ...                  w=np.array([[1, 0], [1, 1], [1, 1]]))
    >>> BalancingTransformer(sparse).transform(sparse).w.tolist()
    [[3.0, 0.0], [1.5, 2.0], [1.5, 2.0]]
    """

    def __init__(self, dataset: Dataset) -> None:
        self.classes = self._classes(dataset)
        self.weights = self._fit(dataset)

    @staticmethod
    def _classes(dataset: Dataset) -> list[float]:
        labelled = dataset.w != 0
        values = dataset.y[labelled]
        return sorted({float(v) for v in np.unique(values)})

    def _fit(self, dataset: Dataset) -> dict[int, list[float]]:
        weights: dict[int, list[float]] = {}
        for task in range(dataset.n_tasks):
            mask = dataset.w[:, task] != 0
            task_y = dataset.y[mask, task]
            n_task = task_y.size
            task_weights = []
            for c in self.classes:
                n_c = float(np.sum(task_y == c))
                task_weights.append(n_task / n_c if n_c > 0 else 0.0)
            weights[task] = task_weights
        return weights

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Overwrite non-zero weights with the inverse-frequency values."""
        w_out = w.copy()
        for task in range(y.shape[1]):
            for i, c in enumerate(self.classes):
                class_indices = (y[:, task] == c) & (w[:, task] != 0)
                w_out[class_indices, task] = self.weights[task][i]
        return X, y, w_out, ids

    def transform(self, dataset: Dataset) -> Dataset:
        X, y, w, ids = self.transform_array(
            dataset.X, dataset.y, dataset.w.copy(), dataset.ids
        )
        return Dataset(X=X, y=y, w=w, ids=ids)


# ------------------------------------------------------------------ #
# Evaluation loop (deepchem's Evaluator shape)                        #
# ------------------------------------------------------------------ #


def evaluate(
    dataset: Dataset,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    transformers: Sequence[_ColumnTransformer] = (),
) -> dict[str, float]:
    """Score ``predict_fn`` on a dataset, undoing y-transformers first.

    deepchem's ``Evaluator`` flow: predict on X, undo the target
    transformers *in reverse fit order* so scores are reported in the
    original units, then average each metric over tasks with the
    labelled-sample mask (via :func:`multitask_metric`).

    Examples
    --------
    >>> ds = Dataset(X=np.array([[0.0], [1.0], [2.0]]), y=np.array([1.0, 2.0, 3.0]))
    >>> scores = evaluate(ds, lambda X: X[:, :1] + 1.0,
    ...                   {"rmse": lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))})
    >>> scores["rmse"]
    0.0
    """
    y_pred = np.asarray(predict_fn(dataset.X), dtype=float)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    for transformer in reversed(list(transformers)):
        if transformer.transform_y:
            y_pred = transformer.untransform(y_pred)
    results: dict[str, float] = {}
    from pharmaml.ml.benchmetrics import multitask_metric

    for name, metric in metrics.items():
        value = multitask_metric(
            dataset.y, y_pred, metric, w=dataset.w, per_task=True
        )
        mean = value[0] if isinstance(value, tuple) else value
        results[name] = float(mean)
    return results


# ------------------------------------------------------------------ #
# MoleculeNet registry (documentation constants)                      #
# ------------------------------------------------------------------ #

#: The canonical MoleculeNet benchmark configuration per dataset, as
#: pinned by deepchem's loaders: ``(n_tasks, task_type, default
#: splitter, default transformers)``.  A documentation constant -- the
#: data itself is remote-hosted (deepchem S3), so the honest boundary
#: is the *configuration*, not the download.
MOLNET_REGISTRY: dict[str, tuple[int, str, str, tuple[str, ...]]] = {
    "delaney": (1, "regression", "scaffold", ("normalization",)),
    "freesolv": (1, "regression", "random", ("normalization",)),
    "sampl": (1, "regression", "scaffold", ("normalization",)),
    "lipo": (1, "regression", "scaffold", ("normalization",)),
    "clearance": (1, "regression", "scaffold", ("log",)),
    "bace_regression": (1, "regression", "scaffold", ("normalization",)),
    "bace_classification": (1, "classification", "scaffold", ("balancing",)),
    "qm7": (1, "regression", "random", ("normalization",)),
    "qm8": (16, "regression", "random", ("normalization",)),
    "qm9": (12, "regression", "random", ("normalization",)),
    "tox21": (12, "classification", "scaffold", ("balancing",)),
    "hiv": (1, "classification", "scaffold", ("balancing",)),
    "bbbp": (1, "classification", "scaffold", ("balancing",)),
    "clintox": (2, "classification", "scaffold", ("balancing",)),
    "sider": (27, "classification", "scaffold", ("balancing",)),
    "muv": (17, "classification", "scaffold", ("balancing",)),
    "pcba": (128, "classification", "scaffold", ("balancing",)),
    "toxcast": (600, "classification", "random", ("balancing",)),
}


def molnet_registry_frame() -> Any:
    """The MoleculeNet configuration table as a DataFrame.

    Examples
    --------
    >>> frame = molnet_registry_frame()
    >>> int(frame.loc["tox21", "n_tasks"])
    12
    >>> frame.loc["delaney", "splitter"]
    'scaffold'
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "n_tasks": {k: v[0] for k, v in MOLNET_REGISTRY.items()},
            "task_type": {k: v[1] for k, v in MOLNET_REGISTRY.items()},
            "splitter": {k: v[2] for k, v in MOLNET_REGISTRY.items()},
            "transformers": {
                k: "+".join(v[3]) for k, v in MOLNET_REGISTRY.items()
            },
        }
    )
