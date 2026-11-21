"""Scaffold-aware coverage — part13 §9.3's scaffold-aware evaluation leg.

The bridge rides :func:`pharmaml.ml.splits.murcko_scaffold_key` (the
same pure-Python framework extractor the split strategies use) into
:func:`pharmaml.ml.coverage_diagnostics.coverage_by_group`, with the
two honest sentinels for the extractor's empty-key rows.  These tests
pin the grouping, the sentinels, the delegation contract, and the
end-to-end conformal integration.
"""


import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml import (
    calibrate_split_conformal,
    scaffold_aware_coverage,
)
from pharmaml.ml.scaffold_coverage import ACYCLIC_GROUP, INVALID_GROUP

BENZENE_DERIVATIVES = ["c1ccccc1O", "c1ccccc1Cl", "CC(=O)Oc1ccccc1C(=O)O"]
PYRIDINE_DERIVATIVES = ["c1ccncc1O", "c1ccncc1Cl"]
ACYCLIC = ["CCO", "CCN"]


def test_per_scaffold_groups_and_coverage():
    covered = [True, False, True, True, True, False, False]
    smiles = BENZENE_DERIVATIVES + PYRIDINE_DERIVATIVES + ACYCLIC
    report = scaffold_aware_coverage(covered, smiles)
    assert [row.group for row in report.groups] == ["c1ccccc1", "c1ccncc1", "acyclic"]
    benzene, pyridine, acyclic = report.groups
    assert (benzene.n_test, benzene.n_covered, benzene.coverage) == (3, 2, 2 / 3)
    assert (pyridine.n_test, pyridine.n_covered, pyridine.coverage) == (2, 2, 1.0)
    assert (acyclic.n_test, acyclic.n_covered, acyclic.coverage) == (2, 0, 0.0)
    assert report.overall.n_test == 7
    assert report.overall.coverage == pytest.approx(4 / 7)
    assert report.confidence == 0.95


def test_sentinels_keep_invalid_and_acyclic_rows_visible():
    report = scaffold_aware_coverage(
        [True, True, False, True, False, False],
        ["c1ccccc1O", "CCO", "not a smiles", "c1ccccc1Cl", "", "CCN"],
    )
    assert [row.group for row in report.groups] == [
        "c1ccccc1", ACYCLIC_GROUP, INVALID_GROUP,
    ]
    invalid = report.groups[2]
    # two hostile rows, both uncovered: 0/2, never silently dropped
    assert (invalid.n_test, invalid.n_covered, invalid.coverage) == (2, 0, 0.0)
    assert report.overall.n_test == 6


def test_first_appearance_order_and_sizes_passthrough():
    covered = [True, False, True, False, True]
    smiles = ["CCO", "c1ccncc1O", "CCN", "c1ccncc1Cl", "c1ccccc1O"]
    sizes = [1.0, 4.0, 2.0, 6.0, 8.0]
    report = scaffold_aware_coverage(covered, smiles, sizes=sizes)
    assert [row.group for row in report.groups] == [
        ACYCLIC_GROUP, "c1ccncc1", "c1ccccc1",
    ]
    acyclic, pyridine, benzene = report.groups
    assert (acyclic.n_test, acyclic.n_covered) == (2, 2)
    assert acyclic.coverage == 1.0
    assert acyclic.mean_size == pytest.approx(1.5)
    assert (pyridine.n_test, pyridine.n_covered, pyridine.mean_size) == (2, 0, 5.0)
    assert (benzene.n_test, benzene.n_covered, benzene.mean_size) == (1, 1, 8.0)
    assert report.overall.mean_size == pytest.approx(4.2)


def test_wilson_bounds_bracket_coverage_in_every_group():
    rng = np.random.default_rng(311)
    smiles = (BENZENE_DERIVATIVES * 7) + (PYRIDINE_DERIVATIVES * 5) + (ACYCLIC * 3)
    covered = list(rng.random(len(smiles)) < 0.6)
    report = scaffold_aware_coverage(covered, smiles)
    for row in report.groups:
        assert 0.0 <= row.lower <= row.coverage <= row.upper <= 1.0
        assert row.lower < row.upper


def test_dataframe_export_carries_group_rows():
    report = scaffold_aware_coverage(
        [True, False, True],
        ["c1ccccc1O", "c1ccccc1Cl", "CCO"],
    )
    frame = report.to_dataframe()
    assert list(frame.columns) == [
        "group", "n_test", "n_covered", "coverage", "lower", "upper", "mean_size",
    ]
    assert frame.group.tolist() == ["c1ccccc1", "acyclic"]
    assert frame.n_test.tolist() == [2, 1]


def test_conformal_intervals_integrate_with_scaffold_report():
    class Constant:
        def predict(self, X):
            return np.zeros(len(X))

    cal = calibrate_split_conformal(Constant(), [[0]] * 9, [1] * 9, alpha=0.1)
    smiles = ["c1ccccc1O", "c1ccccc1Cl", "CCO", "CCN", "not a smiles"]
    truth = [1.0, -1.0, 0.0, 2.0, 0.0]
    result = cal.predict_interval([[0]] * 5, truth)
    report = scaffold_aware_coverage(
        result.covered, smiles, sizes=result.upper - result.lower,
    )
    # the ±1 intervals around 0 (boundaries inclusive) cover the rows
    # with truth in [-1, 1] — and the hostile row lands in its own
    # visible group
    assert [row.group for row in report.groups] == [
        "c1ccccc1", ACYCLIC_GROUP, INVALID_GROUP,
    ]
    assert [row.coverage for row in report.groups] == [1.0, 0.5, 1.0]
    assert report.overall.coverage == result.coverage
    assert report.overall.mean_size == pytest.approx(result.mean_width)


@pytest.mark.parametrize("smiles", ["c1ccccc1O", 0])
def test_rejects_non_vector_smiles(smiles):
    with pytest.raises(DataValidationError):
        scaffold_aware_coverage([True], smiles)


@pytest.mark.parametrize("smiles", [[], [0, 1], ["c1ccccc1O", None], [["CCO"]]])
def test_rejects_invalid_smiles_vectors(smiles):
    with pytest.raises(DataValidationError):
        scaffold_aware_coverage([True, False], smiles)


def test_rejects_length_mismatch_with_clear_message():
    with pytest.raises(DataValidationError, match="same length"):
        scaffold_aware_coverage([True, False], ["CCO", "CCN", "c1ccccc1O"])


def test_confidence_passthrough_validates_in_the_delegate():
    with pytest.raises(DataValidationError, match="confidence"):
        scaffold_aware_coverage([True], ["CCO"], confidence=1.5)
    narrow = scaffold_aware_coverage([True, False], ["CCO", "CCO"], confidence=0.8)
    wide = scaffold_aware_coverage([True, False], ["CCO", "CCO"], confidence=0.99)
    assert wide.groups[0].lower < narrow.groups[0].lower
    assert narrow.groups[0].upper < wide.groups[0].upper


def test_smiles_vector_helpers_reject_scalar_and_iterables_of_nonstrings():
    from pharmaml.ml.scaffold_coverage import _smiles_vector

    for bad in ["CCO", b"CCO"]:
        with pytest.raises(DataValidationError, match="vector of strings"):
            _smiles_vector(bad)
    with pytest.raises(DataValidationError, match="nonempty"):
        _smiles_vector([])
    with pytest.raises(DataValidationError, match="only string"):
        _smiles_vector(["CCO", 5])
    assert _smiles_vector(("CCO", "CCN")) == ["CCO", "CCN"]


def test_label_for_helper_is_the_contract_documented():
    from pharmaml.ml.scaffold_coverage import _label_for

    assert _label_for("CCO") == ACYCLIC_GROUP
    assert _label_for("c1ccccc1O") == "c1ccccc1"
    assert _label_for("not a smiles") == INVALID_GROUP
