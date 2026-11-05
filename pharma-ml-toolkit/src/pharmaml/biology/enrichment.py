"""Gene-set enrichment scoring: the Precily feature engine.

``SmritiChawla/Precily`` was reviewed file by file for this module
(``README.md`` and the figure tree: ``Fig1/Fig1c/.../Precily_genes.py``
and ``Precily_pathways.py`` -- the Keras/TensorFlow deep-ensemble
trainers, ``Fig*/DrugsPred.R`` -- the five-model mean prediction, and
the ``c2.cp.v6.1.symbols.gmt`` MSigDB files every figure directory
ships).  Precily's whole design is *pathway-level* drug response
prediction: per-cell-line gene expression is compressed into
1329 GSVA pathway enrichment scores, those scores (plus drug
descriptors) feed the network, and the R ``GSVA`` package (v1.40.1)
does the compression.  The TensorFlow half stays external (Keras is
not in the venv, and the toolkit's numpy ensembles cover the concept
in :mod:`pharmaml.ml.deepensemble`); the *feature engine* -- the part
that turns an expression matrix into pathway scores -- had no native
home here, and lands as three functions:

* :func:`parse_gmt` -- the MSigDB ``.gmt`` format reader (the file
  format every Precily figure directory carries).
* :func:`gsva_scores` -- the rank-based KS core of GSVA
  (Hänzelmann et al. 2013): per-sample descending-expression ranks,
  weights ``|x - median|^tau`` (``tau = 1``, the package default),
  the gene-set running sum with in-set steps weighted by the
  weight-share and out-set steps ``-1/(n - n_H)``, the enrichment
  score as the running sum's maximum absolute excursion, and the
  final per-set z-normalisation across samples that makes GSVA
  scores comparable between cell lines.  The Gaussian-kernel density
  refinement (``kcdf="Gaussian"``) and the Kuiper ``mx.diff``
  variant of the R package are the two deliberate simplifications,
  stated here once.
* :func:`mean_signature_scores` -- the simpler mean-z signature score
  everyone computes before reaching for GSVA -- kept as the honest
  comparator (same input, same output shape, one line of math).

Riding along: :func:`knn_impute` -- Precily's ``impute.knn`` step
(the R ``impute`` package), as k-nearest-row weighted imputation over
the observed columns; and :func:`gene_set_overview` -- the set-level
coverage audit (size / present / missing) that tells you which sets
the expression table can actually score.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.biology.enrichment import (
...     gene_set_overview, gsva_scores, knn_impute, mean_signature_scores,
...     parse_gmt,
... )
>>> sets = parse_gmt(
...     "CELL_CYCLE\\tdescription\\tG1\\tG2\\tG3\\tG1\\n"
...     "DNA_REPAIR\\tdescription\\tG4\\tG5\\n")
>>> sorted(sets)
['CELL_CYCLE', 'DNA_REPAIR']
>>> sets["CELL_CYCLE"]
['G1', 'G2', 'G3']
>>> expression = pd.DataFrame(
...     [[3.0, 3.1, 2.9, 0.1, 0.2],
...      [0.2, 0.1, 0.3, 3.2, 2.8],
...      [1.5, 1.6, 1.4, 1.7, 1.5]],
...     index=["cellA", "cellB", "cellC"],
...     columns=["G1", "G2", "G3", "G4", "G5"])
>>> overview = gene_set_overview(expression, sets)
>>> int(overview.loc["CELL_CYCLE", "present"])
3
>>> scores = gsva_scores(expression, sets)
>>> scores.shape
(3, 2)
>>> bool(scores.loc["cellB", "DNA_REPAIR"] > scores.loc["cellA", "DNA_REPAIR"])
True
>>> signature = mean_signature_scores(expression, sets)
>>> bool(signature.loc["cellB", "DNA_REPAIR"] > signature.loc["cellA", "DNA_REPAIR"])
True
>>> holed = pd.DataFrame(
...     {"a": [1.0, 2.0, 3.0, 9.0], "b": [1.0, np.nan, 3.0, 9.0],
...      "c": [1.0, 2.5, 3.0, 9.0]})
>>> filled = knn_impute(holed, k=1)
>>> float(filled.loc[1, "b"])
3.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError

__all__ = [
    "gene_set_overview",
    "gsva_scores",
    "knn_impute",
    "mean_signature_scores",
    "parse_gmt",
]


def parse_gmt(source: str | Path) -> dict[str, list[str]]:
    """Parse MSigDB ``.gmt`` gene-set lines into ``{name: [genes]}``.

    The format every Precily figure directory ships
    (``c2.cp.v6.1.symbols.gmt``): one set per line, tab-separated,
    first field the set name, second field the description (dropped),
    remaining fields the member genes.  Duplicate genes are collapsed
    to first occurrence; empty fields are dropped.  A ``Path`` reads
    the file; a string is treated as the file's *content*.

    Examples
    --------
    >>> sets = parse_gmt("SET_A\\tdesc\\tX\\tY\\nSET_B\\tdesc\\tZ\\n")
    >>> sorted(sets)
    ['SET_A', 'SET_B']
    >>> parse_gmt("")   # an empty file is an error, not a silent {}
    Traceback (most recent call last):
        ...
    pharmaml.exceptions.DataValidationError: no gene sets parsed from the gmt input
    """
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    else:
        text = str(source)
    sets: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip("\r\n")
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("\t")]
        if len(fields) < 3 or not fields[0]:
            continue  # a set needs a name and at least one gene
        name = fields[0]
        genes: list[str] = []
        for gene in fields[2:]:
            if gene and gene not in genes:
                genes.append(gene)
        if genes:
            sets[name] = genes
    if not sets:
        raise DataValidationError("no gene sets parsed from the gmt input")
    return sets


def _expression_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce to float and refuse all-NaN / non-numeric columns."""
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise DataValidationError(
            f"expression needs >= 2 rows and >= 2 genes, got {frame.shape}"
        )
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    bad = [
        str(column)
        for column in frame.columns
        if numeric[column].isna().all() and frame[column].notna().any()
    ]
    if bad:
        raise DataValidationError(
            f"expression has non-numeric columns: {', '.join(bad[:5])}"
        )
    return numeric.astype(float)


def _match_sets(
    expression: pd.DataFrame, gene_sets: dict[str, list[str]]
) -> tuple[dict[str, list[str]], list[str]]:
    """Map set gene names onto the expression columns; report empties."""
    genes = [str(g) for g in expression.columns]
    lookup = {gene: k for k, gene in enumerate(genes)}
    matched: dict[str, list[str]] = {}
    empty: list[str] = []
    for name, members in gene_sets.items():
        present = [gene for gene in members if gene in lookup]
        if present:
            matched[str(name)] = present
        else:
            empty.append(str(name))
    if not matched:
        raise DataValidationError(
            "no gene set has a single gene in common with the expression "
            f"columns (checked {len(gene_sets)} sets)"
        )
    return matched, empty


def gsva_scores(
    expression: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    *,
    tau: float = 1.0,
) -> pd.DataFrame:
    """GSVA pathway scores: one z-scored KS statistic per cell x set.

    The rank-based core of the GSVA algorithm (Hänzelmann, Castelo &
    Guinney 2013 -- the paper behind the R package Precily calls),
    computed per sample row:

    1. genes ranked by descending expression;
    2. per-gene weight ``|x - row median| ** tau`` (``tau = 1`` -- the
       package default);
    3. the running sum over the ranked genes -- in-set genes step up
       by their share of the set's total weight, out-set genes step
       down by ``1 / (n_genes - n_set)``;
    4. the enrichment score is the running sum's value at its
       maximum absolute excursion -- SIGNED, so a set enriched among
       the sample's most-expressed genes scores positive and the
       same set depleted among them scores negative; and the final
       score z-normalises each set's enrichment scores ACROSS
       samples (the step that makes one cell line's pathway score
       comparable to another's).

    Two deliberate simplifications, stated once: the rank weights
    are the paper's original ``|x - median|^tau`` form rather than
    the package's Gaussian kernel refinement, and the score is the
    signed maximum excursion rather than the Kuiper ``mx.diff``
    variant.

    Gene sets with no gene in the expression columns are skipped
    (not silently zeroed); everything else is scored.

    Examples
    --------
    >>> import pandas as pd
    >>> expression = pd.DataFrame(
    ...     [[3.0, 3.1, 2.9, 0.1, 0.2],
    ...      [0.2, 0.1, 0.3, 3.2, 2.8],
    ...      [1.5, 1.6, 1.4, 1.7, 1.5]],
    ...     index=["cellA", "cellB", "cellC"],
    ...     columns=["G1", "G2", "G3", "G4", "G5"])
    >>> scores = gsva_scores(expression, {"P": ["G1", "G2", "G3"]})
    >>> scores.shape
    (3, 1)
    >>> bool(scores.loc["cellA", "P"] > 0.0)
    True
    """
    if tau <= 0.0:
        raise DataValidationError(f"tau must be > 0, got {tau}")
    matrix = _expression_matrix(expression)
    matched, _empty = _match_sets(matrix, gene_sets)
    values = matrix.to_numpy(dtype=float)
    genes = [str(g) for g in matrix.columns]
    samples = [str(i) for i in matrix.index]
    position = {gene: k for k, gene in enumerate(genes)}
    n_genes = values.shape[1]

    raw = np.zeros((values.shape[0], len(matched)), dtype=float)
    set_names = list(matched.keys())
    for col_index, (name, members) in enumerate(matched.items()):
        member_idx = np.array([position[gene] for gene in members])
        n_set = member_idx.size
        if n_set >= n_genes:
            raise DataValidationError(
                f"gene set {name!r} covers every gene -- the KS running "
                "sum is undefined (trivial set)"
            )
        for row in range(values.shape[0]):
            x = values[row]
            median = float(np.median(x))
            weights = np.abs(x - median) ** tau
            order = np.argsort(-x, kind="stable")
            in_set_weight = float(weights[member_idx].sum())
            if in_set_weight <= 0.0:
                continue  # flat row: no signal, raw ES stays 0
            running = 0.0
            best_value = 0.0
            best_abs = 0.0
            member_set = set(member_idx.tolist())
            step_out = 1.0 / (n_genes - n_set)
            for gene_k in order:
                if gene_k in member_set:
                    running += weights[gene_k] / in_set_weight
                else:
                    running -= step_out
                if abs(running) > best_abs:
                    best_abs = abs(running)
                    best_value = running
            raw[row, col_index] = best_value

    # the across-sample z-normalisation (GSVA's final step)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(std > 0.0, (raw - mean) / np.where(std > 0.0, std, 1.0), 0.0)
    return pd.DataFrame(normalized, index=samples, columns=set_names)


def mean_signature_scores(
    expression: pd.DataFrame, gene_sets: dict[str, list[str]]
) -> pd.DataFrame:
    """The simple comparator: per-set mean of per-gene z-scores.

    One line of math per set (standardise each gene across cell
    lines, average the members) -- the signature score people compute
    before reaching for GSVA, kept beside it so the two can be
    compared honestly on the same input.

    Examples
    --------
    >>> import pandas as pd
    >>> expression = pd.DataFrame(
    ...     [[3.0, 3.1, 2.9, 0.1, 0.2],
    ...      [0.2, 0.1, 0.3, 3.2, 2.8],
    ...      [1.5, 1.6, 1.4, 1.7, 1.5]],
    ...     index=["cellA", "cellB", "cellC"],
    ...     columns=["G1", "G2", "G3", "G4", "G5"])
    >>> signature = mean_signature_scores(expression, {"P": ["G1", "G2"]})
    >>> bool(signature.loc["cellA", "P"] > signature.loc["cellB", "P"])
    True
    """
    matrix = _expression_matrix(expression)
    matched, _empty = _match_sets(matrix, gene_sets)
    values = matrix.to_numpy(dtype=float)
    std = values.std(axis=0, keepdims=True)
    z = np.divide(
        values - values.mean(axis=0, keepdims=True),
        np.where(std > 0.0, std, 1.0),
    )
    frame = pd.DataFrame(z, index=[str(i) for i in matrix.index],
                         columns=[str(g) for g in matrix.columns])
    return pd.DataFrame(
        {name: frame[members].mean(axis=1) for name, members in matched.items()}
    )


def gene_set_overview(
    expression: pd.DataFrame, gene_sets: dict[str, list[str]]
) -> pd.DataFrame:
    """Per-set coverage audit: size, present, missing, coverage.

    The table that answers "which sets can this expression matrix
    actually score?" before any scoring runs.

    Examples
    --------
    >>> import pandas as pd
    >>> expression = pd.DataFrame(
    ...     {"G1": [1.0, 2.0], "G3": [2.0, 1.0], "G9": [0.5, 0.5]})
    >>> overview = gene_set_overview(
    ...     expression, {"SET_A": ["G1", "G3"], "SET_B": ["G7", "G8"]})
    >>> int(overview.loc["SET_A", "present"]), int(overview.loc["SET_B", "present"])
    (2, 0)
    >>> int(overview.loc["SET_B", "coverage_pct"])
    0
    """
    matrix = _expression_matrix(expression)
    genes = {str(g) for g in matrix.columns}
    rows = []
    for members in gene_sets.values():
        present = [gene for gene in members if gene in genes]
        size = len(members)
        rows.append(
            {
                "size": size,
                "present": len(present),
                "missing": size - len(present),
                "coverage_pct": round(100.0 * len(present) / size, 1) if size else 0.0,
            }
        )
    return pd.DataFrame(rows, index=[str(n) for n in gene_sets])


def knn_impute(frame: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """k-nearest-row weighted imputation (Precily's ``impute.knn``).

    Precily imputes missing pathway features with the R ``impute``
    package's ``impute.knn``; the numpy analogue fills each missing
    cell from the ``k`` nearest rows over the jointly observed
    columns, weighted by inverse distance (ties broken by row order,
    so the result is deterministic).  Rows missing a cell entirely
    fall back to the column median, stated in the output notes of
    callers, not silently.  A fully observed frame returns unchanged.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> holed = pd.DataFrame(
    ...     {"a": [1.0, 2.0, 3.0, 9.0], "b": [1.0, np.nan, 3.0, 9.0],
    ...      "c": [1.0, 2.5, 3.0, 9.0]})
    >>> float(knn_impute(holed, k=1).loc[1, "b"])
    3.0
    >>> round(float(knn_impute(holed, k=2).loc[1, "b"]), 3)
    2.234
    """
    if k < 1:
        raise DataValidationError(f"k must be >= 1, got {k}")
    filled = frame.copy()
    if not filled.isna().any().any():
        return filled
    values = filled.apply(pd.to_numeric, errors="coerce")
    missing_cells = np.argwhere(values.isna().to_numpy())
    if missing_cells.size == 0:
        return filled
    for row, col in missing_cells:
        column = values.iloc[:, col]
        candidates = [r for r in range(values.shape[0]) if r != row and np.isfinite(column.iloc[r])]
        if not candidates:
            filled.iloc[row, col] = float(np.nanmedian(values.to_numpy(dtype=float)[:, col]))
            continue
        own = values.iloc[row].to_numpy(dtype=float)
        distances: list[tuple[float, int]] = []
        for candidate in candidates:
            other = values.iloc[candidate].to_numpy(dtype=float)
            joint = np.isfinite(own) & np.isfinite(other)
            if not joint.any():
                distances.append((np.inf, candidate))
                continue
            distances.append(
                (float(np.linalg.norm(own[joint] - other[joint])), candidate)
            )
        distances.sort(key=lambda pair: (pair[0], pair[1]))
        nearest = distances[: min(k, len(distances))]
        usable = [(dist, candidate) for dist, candidate in nearest if np.isfinite(dist)]
        if not usable:
            filled.iloc[row, col] = float(np.nanmedian(values.to_numpy(dtype=float)[:, col]))
            continue
        if len(usable) == 1:
            filled.iloc[row, col] = float(column.iloc[usable[0][1]])
            continue
        weights = np.array([1.0 / (dist + 1e-12) for dist, _c in usable])
        estimates = np.array(
            [float(column.iloc[candidate]) for _d, candidate in usable]
        )
        filled.iloc[row, col] = float((weights * estimates).sum() / weights.sum())
    return filled
