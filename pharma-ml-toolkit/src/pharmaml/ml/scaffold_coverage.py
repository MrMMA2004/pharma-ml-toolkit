"""Scaffold-aware coverage diagnostics — part13 §9.3's scaffold-aware
evaluation leg.

The conformal and coverage diagnostics in :mod:`pharmaml.ml.conformal`,
:mod:`pharmaml.ml.conformal_classification` and
:mod:`pharmaml.ml.coverage_diagnostics` summarize empirical coverage
overall and within caller-supplied groups.  Part13 §9.3 adds the
chemically-motivated grouping this module bridges: per-**Murcko
scaffold** coverage — the failure mode where marginal coverage looks
fine while every ring-system family is either fully covered or fully
missed.

The scaffold extraction itself is NOT duplicated here (§9.3's own
rule): the label comes from :func:`pharmaml.ml.splits.murcko_scaffold_key`,
the pure-Python framework extractor the split strategies already use.
This module contributes only the bridge and two honest sentinel labels
for the extractor's documented empty-key rows:

* rows whose SMILES fails to parse land in ``"invalid"`` — they were
  still evaluated rows; hiding them would inflate every scaffold's
  coverage;
* rows that parse but carry no ring systems (acyclic — ethanol,
  propane) land in ``"acyclic"`` — one extra parse of the same chem
  core disambiguates the two cases the extractor's contract cannot.

Every statistical caveat of
:func:`pharmaml.ml.coverage_diagnostics.coverage_by_group` applies
verbatim: the Wilson bounds describe sampling uncertainty for
prespecified groups with the fitted predictor and calibration held
fixed — they are not a conformal guarantee, and choosing groups after
seeing the results invalidates them.

Examples
--------
>>> from pharmaml.ml.scaffold_coverage import scaffold_aware_coverage
>>> smiles = ["c1ccccc1O", "c1ccccc1Cl", "CCO", "CCN"]
>>> report = scaffold_aware_coverage([True, False, True, True], smiles)
>>> [row.group for row in report.groups]
['c1ccccc1', 'acyclic']
>>> report.groups[0].n_test, report.groups[0].n_covered
(2, 1)
"""

from __future__ import annotations

from typing import Any

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.coverage_diagnostics import CoverageReport, coverage_by_group
from pharmaml.ml.splits import murcko_scaffold_key

__all__ = ["scaffold_aware_coverage"]

#: Group label for rows whose SMILES fails to parse.
INVALID_GROUP = "invalid"
#: Group label for rows that parse but have no ring systems.
ACYCLIC_GROUP = "acyclic"


def _smiles_vector(smiles: Any) -> list[str]:
    """Validate a nonempty one-dimensional vector of SMILES strings."""
    if isinstance(smiles, (str, bytes)):
        raise DataValidationError(
            "smiles must be a vector of strings, not one string")
    try:
        items = list(smiles)
    except TypeError as exc:
        raise DataValidationError(
            "smiles must be a one-dimensional vector of strings") from exc
    if not items:
        raise DataValidationError(
            "smiles must be a nonempty one-dimensional vector")
    for item in items:
        if not isinstance(item, str):
            raise DataValidationError("smiles must contain only string values")
    return items


def _label_for(smiles: str) -> str:
    """Scaffold label for one SMILES, with honest empty-key sentinels."""
    key = murcko_scaffold_key(smiles)
    if key != "":
        return key
    # The extractor's contract returns '' for both hostile input and
    # acyclic frameworks; one parse of the same chem core separates the
    # two (an unparseable SMILES never survives parse_smiles).
    try:
        from pharmaml.chem.molgraph import parse_smiles

        parse_smiles(smiles)
    except Exception:  # noqa: BLE001 - hostile SMILES is a label, not a crash
        return INVALID_GROUP
    return ACYCLIC_GROUP


def scaffold_aware_coverage(
    covered: Any,
    smiles: Any,
    *,
    sizes: Any = None,
    confidence: float = 0.95,
) -> CoverageReport:
    """Empirical coverage summarized per Murcko scaffold.

    ``covered`` (and optional ``sizes``) align positionally with
    ``smiles``; every argument and every statistical caveat is
    :func:`~pharmaml.ml.coverage_diagnostics.coverage_by_group`'s own —
    this function only derives the scaffold group labels via
    :func:`~pharmaml.ml.splits.murcko_scaffold_key`.  Groups are
    returned in first-appearance order, with the ``"invalid"`` and
    ``"acyclic"`` sentinels keeping unparseable and ring-less rows in
    the report rather than shrinking it.

    Examples
    --------
    >>> from pharmaml.ml.scaffold_coverage import scaffold_aware_coverage
    >>> report = scaffold_aware_coverage(
    ...     [True, True, False, True, False],
    ...     ["CCO", "c1ccccc1O", "CCO", "c1ccccc1Cl", "not a smiles"])
    >>> [row.group for row in report.groups]
    ['acyclic', 'c1ccccc1', 'invalid']
    >>> [row.coverage for row in report.groups]
    [0.5, 1.0, 0.0]
    """
    labels = [_label_for(one) for one in _smiles_vector(smiles)]
    return coverage_by_group(covered, labels, sizes=sizes,
                             confidence=confidence)
