"""Split integrity: aligned rows, group isolation and feasible holdouts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.splits import make_train_test_split, scaffold_groups, split_indices


@pytest.mark.parametrize("strategy", ["random", "stratified", "temporal", "kennard_stone", "group", "scaffold"])
@pytest.mark.parametrize("n_y", [3, 5])
def test_mismatched_target_lengths_rejected_before_splitting(strategy, n_y):
    X = np.arange(4.0).reshape(-1, 1)
    kwargs = {"groups": ["a", "a", "b", "b"]} if strategy == "group" else {}
    if strategy == "scaffold":
        kwargs = {"smiles": ["c1ccccc1", "c1ccccc1", "C1CCCCC1", "C1CCCCC1"]}
    with pytest.raises(DataValidationError, match=f"X has 4 rows but y has {n_y}"):
        split_indices(X, np.zeros(n_y), strategy=strategy, **kwargs)
    with pytest.raises(DataValidationError, match=f"X has 4 rows but y has {n_y}"):
        make_train_test_split(X, np.arange(n_y), strategy=strategy, **kwargs)


def assert_partition(train, test, groups):
    assert len(train) and len(test)
    assert set(train).isdisjoint(test)
    assert sorted([*train, *test]) == list(range(len(groups)))
    assert {groups[i] for i in train}.isdisjoint({groups[i] for i in test})


@pytest.mark.parametrize("seed", range(8))
def test_group_split_reserves_last_group_if_needed(seed):
    groups = ["a", "a", "b", "b"]
    train, test = split_indices(y=np.arange(4), strategy="group", groups=groups,
                                test_size=0.25, random_state=seed)
    assert_partition(train, test, groups)
    assert len(train) == len(test) == 2  # whole groups, not the requested one row
    again = split_indices(y=np.arange(4), strategy="group", groups=groups,
                          test_size=0.25, random_state=seed)
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(test, again[1])


def test_scaffold_split_reserves_a_group_and_keeps_keys_disjoint():
    smiles = ["c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CC1CCCCC1"]
    keys = scaffold_groups(smiles)
    assert keys[0] == keys[1] and keys[2] == keys[3] and keys[0] != keys[2]
    train, test = split_indices(y=range(4), strategy="scaffold", smiles=smiles, test_size=0.25)
    assert_partition(train, test, keys)


@pytest.mark.parametrize("strategy,kwargs", [
    ("group", {"groups": ["same"] * 4}),
    ("scaffold", {"smiles": ["CCO", "CCC", "not a smiles at all", "CCO"]}),
])
def test_single_group_still_fails(strategy, kwargs):
    with pytest.raises(DataValidationError, match="no group-honest"):
        split_indices(y=range(4), strategy=strategy, test_size=0.25, **kwargs)


def test_empty_scaffolds_are_grouped_together():
    smiles = ["CCO", "CCC", "not a smiles at all", "c1ccccc1"]
    keys = scaffold_groups(smiles)
    assert keys[:3] == ["", "", ""] and keys[3] != ""
    train, test = split_indices(y=range(4), strategy="scaffold", smiles=smiles, test_size=0.5)
    assert_partition(train, test, keys)


def test_random_split_matches_sklearn_and_preserves_positional_pandas_rows():
    X = pd.DataFrame({"value": range(10)}, index=range(10, 20))
    y = pd.Series(range(10), index=range(100, 110))
    train, test = split_indices(X, y, strategy="random", test_size=0.3, random_state=12)
    expected_train, expected_test = train_test_split(np.arange(10), test_size=0.3, random_state=12)
    np.testing.assert_array_equal(train, np.sort(expected_train))
    np.testing.assert_array_equal(test, np.sort(expected_test))
    Xtr, Xte, ytr, yte = make_train_test_split(X, y, strategy="random", test_size=0.3,
                                               random_state=12, stratify=False)
    assert Xtr["value"].tolist() == ytr.tolist()
    assert Xte["value"].tolist() == yte.tolist()


def test_unsized_target_rejected():
    with pytest.raises(DataValidationError, match="sized containers"):
        split_indices(np.arange(5), iter(range(5)))
    with pytest.raises(DataValidationError, match="sized containers"):
        split_indices(np.arange(5), np.array(1))


def test_scaffold_split_preserves_train_first_when_test_already_nonempty():
    smiles = ["c1ccccc1"] * 4 + ["C1CCCCC1"] * 2 + ["C1CC1"]
    keys = scaffold_groups(smiles)
    assert len(set(keys)) == 3
    train, test = split_indices(y=range(7), strategy="scaffold", smiles=smiles, test_size=0.4)
    assert_partition(train, test, keys)
    assert list(train) == [0, 1, 2, 3]
    assert list(test) == [4, 5, 6]
