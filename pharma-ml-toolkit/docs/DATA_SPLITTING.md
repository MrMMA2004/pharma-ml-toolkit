# Choosing and checking data splits

`pharmaml.ml.splits.split_indices` returns sorted positional train and test
indices for random, stratified, scaffold, group, Kennard–Stone and temporal
splits. `make_train_test_split` applies the indices to features and labels.
When both are supplied, they must have the same number of rows; **pandas indexes
are not matched**. Align the data explicitly before splitting.

```python
from pharmaml.ml.splits import split_indices

# Rows in X, y and assay_series are already aligned.
train, test = split_indices(X, y, strategy="group", groups=assay_series,
                            test_size=0.2, random_state=42)
# Fit preprocessing and the model on X.iloc[train], y.iloc[train] only.
```

- Group and scaffold splits keep each group entirely on one side. Their
  `test_size` is a target, not a guarantee: large groups can make the actual
  fraction quite different. A group is reserved for the test side when the
  greedy training assignment would otherwise consume them all. One group alone
  cannot produce a group-disjoint split. Group labels are converted to strings,
  so, for example, `1` and `"1"` are treated as the same group.
- Scaffold splitting uses the toolkit's SMILES parser and its scaffold keys,
  not RDKit. Acyclic structures and parse failures both produce the empty key
  and are placed in the **same group**. Validate structures before splitting;
  an empty key alone cannot distinguish invalid molecules from acyclic ones.
  Scaffold disjointness is not a guarantee against chemical similarity, other
  duplicate leakage or distribution shift.
- Temporal splitting uses the rows as supplied, not a date column: sort by the
  relevant time first. Kennard–Stone selects training rows using all supplied
  feature values; consider whether that transductive selection is appropriate
  for the intended deployment. It does not use target labels to select rows.
- Splitting alone cannot detect repeated compounds, correlated measurements,
  assay or patient leakage, or whether groups capture all relevant relationships.
  Fit imputation, scaling, feature selection, and model tuning using training
  data only. Keep the final test set separate from model selection.
- A scaffold, group or temporal split can deliberately induce distribution
  shift; do not assume standard exchangeability-based guarantees such as
  [split-conformal coverage](CONFORMAL_REGRESSION.md) apply to that holdout.
