"""Market-basket association mining: the Apriori algorithm in pure Pandas.

The ``Unsupervised Learning/ML_Apriori-Algorithm.ipynb`` notebook of the
absorbed repository madhurimarawat/Machine-Learning-Using-Python
(https://github.com/madhurimarawat/Machine-Learning-Using-Python) mines
shopping-basket associations from ``Market_Basket_Optimisation.csv``
through mlxtend's market-basket surface: ``TransactionEncoder`` to turn
the ragged basket rows into a one-hot boolean frame, then
``mlxtend.frequent_patterns.apriori`` to enumerate the frequent
itemsets (the notebook stops there and reads off the itemsets of
length >= 2 as its "associations").  This module reproduces exactly
that surface -- plus the ``association_rules`` call the notebook stops
one step short of -- without adding mlxtend to the dependency tree:

* :func:`transaction_encoder` -- baskets in, one-hot boolean frame out
  (one row per transaction, one column per sorted unique item, with the
  notebook's hand-rolled ``drop([''])`` padding fix folded in).
* :func:`apriori_frequent` -- levelwise Apriori: L1 from single-item
  supports, L(k) from L(k-1) self-joins that share their first k-2
  items, every candidate pruned unless all of its k-1 subsets are
  already frequent, supports counted over the boolean frame.
* :func:`association_rules` -- ``X -> Y`` rules with support,
  confidence, lift, leverage and conviction (single-item consequent --
  the convention the notebook's length >= 2 reads consume).
* :func:`apriori_pipeline` -- encoder -> itemsets -> rules in one call,
  returning the typed :class:`AprioriResult` record.

What did NOT land, stated honestly: mlxtend itself is not -- and will
not become -- a dependency of this toolkit; what you get here is a
pure NumPy/Pandas reimplementation of its three-function surface, so
the install footprint does not change.  The FP-Growth alternative
(frequent-pattern tree mining, mlxtend's ``fpgrowth``) is NOT
implemented -- levelwise Apriori only, which is what the notebook
teaches and what basket-shaped pharmacy data (co-dispensed OTC items,
co-ordered clinic supplies) needs.  And the notebook's
``Market_Basket_Optimisation.csv`` dataset (7500 grocery baskets) is
NOT bundled -- bring your own transaction list; the doctest below uses
a small pharmacy-flavoured one.

Determinism costs nothing here: Apriori is a counting argument, so the
same transactions always yield the same itemsets, supports and rules.
No randomness is involved anywhere, and every sort is keyed explicitly
(itemsets by support descending, ties broken by the itemset's sorted
tuple; rules by lift descending, ties broken by confidence, then
antecedent, then consequent).

Examples
--------
>>> from pharmaml.ml.apriori import (
...     apriori_frequent,
...     apriori_pipeline,
...     association_rules,
...     transaction_encoder,
... )

A small pharmacy basket log -- eight transactions over five items, with
``aspirin`` and ``vitamin C`` co-occurring in three of the eight baskets:

>>> baskets = [
...     ["aspirin", "vitamin C"],
...     ["aspirin", "vitamin C", "paracetamol"],
...     ["aspirin", "vitamin C", "ibuprofen"],
...     ["paracetamol", "ibuprofen"],
...     ["aspirin"],
...     ["vitamin C", "omeprazole"],
...     ["paracetamol", "omeprazole"],
...     [],
... ]

``transaction_encoder`` turns the baskets into a one-hot boolean frame
(one row per basket, sorted item columns; the empty eighth basket is an
all-``False`` row, kept -- empty transactions are allowed):

>>> onehot = transaction_encoder(baskets)
>>> onehot.shape
(8, 5)
>>> list(onehot.columns)
['aspirin', 'ibuprofen', 'omeprazole', 'paracetamol', 'vitamin C']
>>> int(onehot["aspirin"].sum())
4
>>> bool(onehot.loc[7].any())
False

The notebook's ``''``-padding wart (``read_csv(..., na_filter=False)``
fills short grocery rows with empty strings) is folded in -- blank
items never become a column:

>>> transaction_encoder([["aspirin", ""], [""]]).shape
(2, 1)

Frequent itemsets at ``min_support=0.25`` (2 of 8 baskets): all five
singletons survive (aspirin 4/8, vitamin C 4/8, paracetamol 3/8,
ibuprofen 2/8, omeprazole 2/8) and exactly one pair does -- the
aspirin + vitamin C pair at 3/8:

>>> frequent = apriori_frequent(onehot, min_support=0.25)
>>> len(frequent)
6
>>> [round(value, 3) for value in frequent["support"].tolist()]
[0.5, 0.5, 0.375, 0.375, 0.25, 0.25]
>>> frequent["itemsets"].apply(len).tolist()
[1, 1, 2, 1, 1, 1]
>>> aspirin_row = frequent[frequent["itemsets"] == frozenset({"aspirin"})]
>>> round(float(aspirin_row["support"].iloc[0]), 3)
0.5
>>> pair = frequent[frequent["itemsets"] == frozenset({"aspirin", "vitamin C"})]
>>> round(float(pair["support"].iloc[0]), 3)
0.375
>>> frequent[frequent["itemsets"] == frozenset({"aspirin", "ibuprofen"})].empty
True

Rules from those itemsets (single-item consequents): aspirin ->
vitamin C carries support 3/8 = 0.375, confidence (3/8) / (4/8) = 0.75,
lift 0.75 / 0.5 = 1.5, leverage 0.375 - 0.25 = 0.125 and conviction
(1 - 0.5) / (1 - 0.75) = 2.0; the mirror rule vitamin C -> aspirin has
the same numbers and loses the top spot only on the lexicographic
tie-break:

>>> rules = association_rules(frequent, onehot, min_confidence=0.5)
>>> rules.shape
(2, 9)
>>> list(rules.columns)[:4]
['antecedent', 'consequent', 'antecedent_support', 'consequent_support']
>>> list(rules.columns)[4:]
['support', 'confidence', 'lift', 'leverage', 'conviction']
>>> top = rules.iloc[0]
>>> top["antecedent"] == frozenset({"aspirin"})
True
>>> top["consequent"] == frozenset({"vitamin C"})
True
>>> round(float(top["support"]), 3)
0.375
>>> round(float(top["confidence"]), 3)
0.75
>>> round(float(top["lift"]), 3)
1.5
>>> round(float(top["leverage"]), 3)
0.125
>>> round(float(top["conviction"]), 3)
2.0

A perfect rule (confidence exactly 1) reports infinite conviction:

>>> strict = transaction_encoder([["gel", "syringe"], ["gel", "syringe"]])
>>> strict_rules = association_rules(
...     apriori_frequent(strict, min_support=0.5), strict, min_confidence=0.5
... )
>>> float(strict_rules.iloc[0]["conviction"])
inf

The notebook's whole flow (encode, mine itemsets, derive rules) in one
call, with the typed record reporting both counts and the top rule:

>>> result = apriori_pipeline(baskets, min_support=0.25, min_confidence=0.5)
>>> print(result)
Apriori market-basket analysis -- 8 transactions, 5 distinct items
  frequent itemsets (min_support=0.25): 6
  association rules (min_confidence=0.5): 2
  top rule (by lift): {aspirin} -> {vitamin C}  support=0.375  confidence=0.75  lift=1.5
>>> result.as_frame().shape
(2, 9)

``min_support`` must be a positive fraction -- mlxtend raises
``ValueError`` for the same misuse, and so does this reimplementation:

>>> apriori_frequent(onehot, min_support=0.0)
Traceback (most recent call last):
    ...
ValueError: min_support must be in (0, 1], got 0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Column contract of :func:`association_rules` output -- mlxtend's own
#: order: the two itemset sides first, then the five metric columns.
RULE_COLUMNS: tuple[str, ...] = (
    "antecedent",
    "consequent",
    "antecedent_support",
    "consequent_support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
)

#: Pandas dtype for each :data:`RULE_COLUMNS` entry, position by position
#: (``object`` for the two frozenset sides, ``float64`` for every metric).
#: ``zip(..., strict=True)`` in :func:`_empty_rules_frame` keeps this
#: aligned with :data:`RULE_COLUMNS` or fails loudly.
_RULE_DTYPES: tuple[object, ...] = (object, object) + ("float64",) * 7


def _bool_matrix(onehot: pd.DataFrame, *, caller: str) -> tuple[np.ndarray, list[str]]:
    """Validate a one-hot frame and convert it to a dense bool matrix.

    Bool columns pass through untouched; numeric columns coerce the
    0/1 way (0 -> ``False``, anything else -> ``True``, NaN included --
    :func:`transaction_encoder` never produces NaN, so this only bites
    hand-built frames); any other dtype is refused.  Returns the matrix
    plus the stringified column names (the item names).

    Raises
    ------
    TypeError
        When ``onehot`` is not a pandas DataFrame at all.
    ValueError
        When the frame has no rows / no item columns, or a column is
        neither boolean nor numeric.
    """
    if not isinstance(onehot, pd.DataFrame):
        raise TypeError(
            f"{caller}: expected a pandas DataFrame, got {type(onehot).__name__}"
        )
    if onehot.shape[1] == 0:
        raise ValueError(f"{caller}: the one-hot frame has no item columns")
    if onehot.shape[0] == 0:
        raise ValueError(f"{caller}: the one-hot frame has no transaction rows")
    parts: list[np.ndarray] = []
    for column in onehot.columns:
        series = onehot[column]
        if not (
            pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series)
        ):
            raise ValueError(
                f"{caller}: item column {column!r} is neither boolean nor numeric "
                f"({series.dtype}) -- pass the frame transaction_encoder built"
            )
        parts.append(series.to_numpy(dtype=bool))
    matrix: np.ndarray = np.column_stack(parts)
    names = [str(column) for column in onehot.columns]
    return matrix, names


def transaction_encoder(transactions: list[list[str]]) -> pd.DataFrame:
    """One-hot encode transactions (mlxtend's ``TransactionEncoder``).

    Parameters
    ----------
    transactions:
        One list of item names per transaction/basket.  Empty baskets
        are allowed (they become all-``False`` rows) and duplicate
        items inside one basket are harmless.

    Returns
    -------
    pd.DataFrame
        Boolean frame with one row per transaction, one column per
        distinct item, columns sorted by item name (mlxtend's own
        column order).

    Notes
    -----
    Empty-string items are dropped rather than encoded: the notebook's
    ``Market_Basket_Optimisation.csv`` read (``na_filter=False`` to
    survive the ragged rows) pads short baskets with ``''``, a padding
    column the notebook then had to drop by hand with
    ``df.drop([''], axis=1)`` -- this encoder never emits it in the
    first place.
    """
    baskets = [[item for item in basket if item != ""] for basket in transactions]
    items = sorted({item for basket in baskets for item in basket})
    position = {item: index for index, item in enumerate(items)}
    matrix = np.zeros((len(baskets), len(items)), dtype=bool)
    for row, basket in enumerate(baskets):
        for item in basket:
            matrix[row, position[item]] = True
    return pd.DataFrame(matrix, columns=items)


def _itemset_support(
    matrix: np.ndarray,
    itemset: tuple[str, ...],
    position: dict[str, int],
    n_transactions: int,
) -> float:
    """Share of transactions that contain every item in ``itemset``."""
    indices = [position[item] for item in itemset]
    covered = matrix[:, indices].all(axis=1)
    return int(np.count_nonzero(covered)) / n_transactions


def _candidate_joins(level: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Self-join the (k-1)-itemset level into k-itemset candidates.

    Two frequent (k-1)-itemsets join when they share their first k-2
    items (both tuples sorted, the level lexicographically ordered, so
    equal prefixes are contiguous -- the ``break`` is safe) and differ
    in the last; the candidate is the shared prefix plus both tails.
    """
    ordered = sorted(level)
    candidates: list[tuple[str, ...]] = []
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if left[:-1] != right[:-1]:
                break
            candidates.append(left + (right[-1],))
    return candidates


def _all_subsets_frequent(
    candidate: tuple[str, ...],
    level_set: set[tuple[str, ...]],
) -> bool:
    """The a-priori prune: every (k-1)-subset of the candidate is frequent."""
    return all(
        candidate[:index] + candidate[index + 1:] in level_set
        for index in range(len(candidate))
    )


def apriori_frequent(
    onehot: pd.DataFrame,
    min_support: float = 0.05,
    max_length: int = 4,
) -> pd.DataFrame:
    """Levelwise Apriori over a one-hot frame (mlxtend's ``apriori``).

    L1 comes from the single-item column supports; L(k) comes from
    L(k-1) self-joins (itemsets sharing their first k-2 items) with the
    a-priori prune (every k-1 subset already frequent), supports
    counted over the boolean frame.

    Parameters
    ----------
    onehot:
        The boolean frame from :func:`transaction_encoder` (numeric
        0/1 columns coerce).  One row per transaction, one column per
        item.
    min_support:
        Keep itemsets contained by at least this fraction of
        transactions.  Must be in (0, 1].  mlxtend's own default of 0.5
        empties anything but the densest baskets; this toolkit pins
        0.05, and the notebook hand-picks 0.01 for its 7500-basket run.
    max_length:
        Largest itemset length to consider (a combinatorics cap;
        mlxtend's ``max_len`` defaults to uncapped).

    Returns
    -------
    pd.DataFrame
        Columns ``'itemsets'`` (``frozenset`` of item names) and
        ``'support'`` (``float64``), one row per frequent itemset of
        length 1..``max_length``, sorted by support descending with
        ties broken by the itemset's sorted tuple.

    Raises
    ------
    ValueError
        If ``min_support`` is outside (0, 1], ``max_length`` < 1, the
        frame is empty / has no item columns, or a column is neither
        boolean nor numeric.
    TypeError
        If ``onehot`` is not a pandas DataFrame.
    """
    if not 0.0 < min_support <= 1.0:
        raise ValueError(f"min_support must be in (0, 1], got {min_support}")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    matrix, items = _bool_matrix(onehot, caller="apriori_frequent")
    n_transactions = matrix.shape[0]
    position = {item: index for index, item in enumerate(items)}

    found: list[tuple[str, ...]] = []
    support_of: dict[tuple[str, ...], float] = {}

    # L(k=1): single-item supports read straight off the columns.
    for index, item in enumerate(items):
        support = int(np.count_nonzero(matrix[:, index])) / n_transactions
        if support >= min_support:
            key = (item,)
            found.append(key)
            support_of[key] = support

    # L(k>=2): join, prune, count, keep -- until a level comes back
    # empty or the length cap is reached.
    level = list(found)
    length = 2
    while level and length <= max_length:
        level_set = set(level)
        survivors: list[tuple[str, ...]] = []
        for candidate in _candidate_joins(level):
            if not _all_subsets_frequent(candidate, level_set):
                continue
            support = _itemset_support(matrix, candidate, position, n_transactions)
            if support >= min_support:
                survivors.append(candidate)
                support_of[candidate] = support
        found.extend(survivors)
        level = survivors
        length += 1

    ordered = sorted(found, key=lambda key: (-support_of[key], key))
    frame = pd.DataFrame(
        {
            "itemsets": pd.Series([frozenset(key) for key in ordered], dtype=object),
            "support": pd.Series([support_of[key] for key in ordered], dtype="float64"),
        }
    )
    _LOG.info(
        "apriori_frequent: %d frequent itemset(s) of length <= %d at min_support=%.3g",
        len(frame),
        max_length,
        min_support,
    )
    return frame


def _empty_rules_frame() -> pd.DataFrame:
    """The rules table with the right columns and dtypes but no rows."""
    return pd.DataFrame(
        {
            name: pd.Series([], dtype=dtype)
            for name, dtype in zip(RULE_COLUMNS, _RULE_DTYPES, strict=True)
        }
    )


def _rule_sort_key(
    row: dict[str, Any],
) -> tuple[float, float, tuple[str, ...], tuple[str, ...]]:
    """Lift descending, then confidence, then antecedent, then consequent."""
    return (
        -float(row["lift"]),
        -float(row["confidence"]),
        tuple(sorted(row["antecedent"])),
        tuple(sorted(row["consequent"])),
    )


def association_rules(
    frequent: pd.DataFrame,
    onehot: pd.DataFrame,
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """Derive ``X -> Y`` association rules (mlxtend's ``association_rules``).

    Every frequent itemset of length >= 2 contributes one rule per
    possible single-item consequent ``Y`` (mlxtend emits every
    antecedent/consequent split; the single-item-consequent form is
    what the notebook's length >= 2 "association" reads consume).  With
    ``support(X -> Y)`` the support of the union, ``support(X)`` the
    antecedent's and ``support(Y)`` the consequent's:

    * ``confidence`` = support(X -> Y) / support(X)
    * ``lift`` = confidence / support(Y)
    * ``leverage`` = support(X -> Y) - support(X) * support(Y)
    * ``conviction`` = (1 - support(Y)) / (1 - confidence),
      ``inf`` when confidence is exactly 1

    All five support figures are (re)counted over ``onehot``, so the
    rule table is self-consistent even when ``frequent`` was mined
    with different settings; ``frequent`` contributes the candidate
    itemsets.  Itemsets that never occur together in ``onehot`` (a
    mismatched-frame situation) contribute no rule rather than a
    division-by-zero confidence.

    Parameters
    ----------
    frequent:
        The itemset table from :func:`apriori_frequent` (must carry the
        ``'itemsets'`` and ``'support'`` columns).
    onehot:
        The same one-hot frame the itemsets were mined from.
    min_confidence:
        Keep rules with confidence at least this value, in [0, 1]
        (mlxtend's own default threshold is 0.8; this toolkit pins 0.5).

    Returns
    -------
    pd.DataFrame
        One row per surviving rule with the :data:`RULE_COLUMNS`
        columns -- ``antecedent``/``consequent`` as ``frozenset`` (the
        consequent always a single item), the five metrics as
        ``float64`` -- sorted by lift descending, ties broken by
        confidence, then antecedent, then consequent.

    Raises
    ------
    ValueError
        If ``min_confidence`` is outside [0, 1], ``frequent`` lacks the
        itemset-table columns, the one-hot frame is empty or
        non-boolean/numeric, or an itemset names items missing from
        ``onehot``.
    TypeError
        If ``onehot`` is not a pandas DataFrame.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
    missing = [name for name in ("itemsets", "support") if name not in frequent.columns]
    if missing:
        raise ValueError(
            "association_rules expects the itemset table apriori_frequent produced "
            f"(columns 'itemsets' and 'support'); missing: {missing}"
        )
    matrix, items = _bool_matrix(onehot, caller="association_rules")
    n_transactions = matrix.shape[0]
    position = {item: index for index, item in enumerate(items)}

    rows: list[dict[str, Any]] = []
    for raw_itemset in frequent["itemsets"]:
        itemset = tuple(sorted(frozenset(raw_itemset)))
        if len(itemset) < 2:
            continue
        unknown = [item for item in itemset if item not in position]
        if unknown:
            raise ValueError(
                "association_rules: itemset names items missing from the one-hot "
                f"frame: {unknown} -- pass the SAME frame apriori_frequent used"
            )
        union_support = _itemset_support(matrix, itemset, position, n_transactions)
        if union_support <= 0.0:
            continue
        for index, consequent_item in enumerate(itemset):
            antecedent = itemset[:index] + itemset[index + 1:]
            antecedent_support = _itemset_support(
                matrix, antecedent, position, n_transactions
            )
            consequent_support = _itemset_support(
                matrix, (consequent_item,), position, n_transactions
            )
            confidence = union_support / antecedent_support
            if confidence < min_confidence:
                continue
            lift = confidence / consequent_support
            rows.append(
                {
                    "antecedent": frozenset(antecedent),
                    "consequent": frozenset((consequent_item,)),
                    "antecedent_support": antecedent_support,
                    "consequent_support": consequent_support,
                    "support": union_support,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": union_support - antecedent_support * consequent_support,
                    "conviction": (
                        math.inf
                        if confidence >= 1.0
                        else (1.0 - consequent_support) / (1.0 - confidence)
                    ),
                }
            )
    if not rows:
        return _empty_rules_frame()
    rows.sort(key=_rule_sort_key)
    _LOG.info(
        "association_rules: %d rule(s) at min_confidence=%.3g", len(rows), min_confidence
    )
    return pd.DataFrame(rows, columns=list(RULE_COLUMNS))


@dataclass
class AprioriResult:
    """Typed report of :func:`apriori_pipeline` (the notebook's whole flow).

    Attributes
    ----------
    n_transactions:
        Basket count the supports were computed over (empty baskets
        included -- they are transactions too).
    n_items:
        Distinct encoded items (the one-hot frame's column count).
    min_support:
        The itemset threshold actually used.
    min_confidence:
        The rule threshold actually used.
    frequent:
        The :func:`apriori_frequent` itemset table.
    rules:
        The :func:`association_rules` rule table.
    """

    n_transactions: int
    n_items: int
    min_support: float
    min_confidence: float
    frequent: pd.DataFrame
    rules: pd.DataFrame

    def as_frame(self) -> pd.DataFrame:
        """The association-rules table -- the analysis's deliverable."""
        return self.rules

    def __str__(self) -> str:
        lines = [
            (
                f"Apriori market-basket analysis -- {self.n_transactions} transactions, "
                f"{self.n_items} distinct items"
            ),
            f"  frequent itemsets (min_support={self.min_support:g}): {len(self.frequent)}",
            (
                f"  association rules (min_confidence={self.min_confidence:g}): "
                f"{len(self.rules)}"
            ),
        ]
        if self.rules.empty:
            lines.append("  no rule cleared the confidence threshold")
        else:
            top = self.rules.iloc[0]
            antecedent = ", ".join(sorted(top["antecedent"]))
            consequent = ", ".join(sorted(top["consequent"]))
            lines.append(
                f"  top rule (by lift): {{{antecedent}}} -> {{{consequent}}}  "
                f"support={float(top['support']):.3g}  "
                f"confidence={float(top['confidence']):.3g}  "
                f"lift={float(top['lift']):.3g}"
            )
        return "\n".join(lines)


def apriori_pipeline(
    transactions: list[list[str]],
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    max_length: int = 4,
) -> AprioriResult:
    """Run the notebook's whole flow: encode, mine itemsets, derive rules.

    One call chaining :func:`transaction_encoder` ->
    :func:`apriori_frequent` -> :func:`association_rules`, returning the
    typed :class:`AprioriResult` record (``as_frame()`` hands back the
    rules table; ``str()`` summarises the counts and the top rule).

    Parameters
    ----------
    transactions:
        One list of item names per basket (see
        :func:`transaction_encoder`).
    min_support:
        Frequent-itemset threshold, in (0, 1] (default 0.05).
    min_confidence:
        Rule threshold, in [0, 1] (default 0.5).
    max_length:
        Largest itemset length to consider (default 4).

    Returns
    -------
    AprioriResult

    Raises
    ------
    ValueError
        Whatever the three chained functions raise for invalid
        thresholds or malformed input (for example an empty transaction
        list, or every basket empty).
    """
    onehot = transaction_encoder(transactions)
    frequent = apriori_frequent(onehot, min_support=min_support, max_length=max_length)
    rules = association_rules(frequent, onehot, min_confidence=min_confidence)
    _LOG.info(
        "apriori_pipeline: %d transaction(s) -> %d frequent itemset(s) -> %d rule(s)",
        int(onehot.shape[0]),
        len(frequent),
        len(rules),
    )
    return AprioriResult(
        n_transactions=int(onehot.shape[0]),
        n_items=int(onehot.shape[1]),
        min_support=float(min_support),
        min_confidence=float(min_confidence),
        frequent=frequent,
        rules=rules,
    )


__all__ = [
    "RULE_COLUMNS",
    "AprioriResult",
    "apriori_frequent",
    "apriori_pipeline",
    "association_rules",
    "transaction_encoder",
]
