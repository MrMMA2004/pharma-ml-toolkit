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
  cannot produce a group-disjoint split. For `strategy="group"`, every row
  needs a known, nonblank label: missing values (including `None`, `NaN`,
  `pd.NA`, and `NaT`), blank strings and non-scalar labels are rejected.
  Tuples of nonblank scalar labels can represent composite groups. Labels are
  converted to strings without trimming, so `1` and `"1"` are treated as the
  same group and different representations can collide. Literal strings like
  `"NA"` are valid; file readers can interpret them as missing unless configured
  otherwise. Validate labels before splitting: a nonmissing but incorrect label
  can still hide a relationship. This validation is for holdout splits; validate
  group labels separately when using cross-validation splitters.
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

## Scaffold cross-validation

`cross_validate_model(..., strategy="scaffold", smiles=smiles)` groups folds
by scaffold keys computed from the row-aligned SMILES. An optional `groups`
argument does **not** change a scaffold run, and the result notes that it was
ignored; choose `strategy="group"` or `"stratified_group"` to use those labels
instead. Acyclic and invalid SMILES
share an empty key. Validate structures and check the other leakage units that
matter for your data. Scaffold-disjoint folds are not necessarily chemically
independent, and scores need not be lower than random-split scores.

`build_cv_splitter` returns an ordinary sklearn splitter, not one bound to its
input labels. If calling its `.split()` directly, supply the same aligned
scaffold keys (e.g. from `scaffold_groups(smiles)`) as `groups`. Passing other
labels changes the folds. Group labels for grouped cross-validation should be
validated separately; the holdout group checks above do not apply to CV.

## Audit exact identifier overlap

Split assignment alone does not establish that compounds, patients or assay
series are disjoint. After splitting, compare identifiers across **all**
training, calibration and test partitions:

```python
from pharmaml.ml import audit_split_overlap

report = audit_split_overlap({
    "train": ["compound-1", "compound-1", "compound-2"],
    "calibration": ["compound-3", "compound-2"],
    "test": ["compound-4", "compound-1"],
})
print(report.has_overlap, report.n_shared_keys)
print(report.to_dataframe())       # one row per shared key and counts per split
print(report.pairwise_dataframe())  # intersections count unique keys
print(report.overlap_rows)         # affected row counts in partition order
print(report.duplicate_rows)       # repeated rows beyond the first, within each split
```

Inputs are named, nonempty iterables of identifiers, such as compound IDs,
patient IDs or `(patient, assay)` tuples. Pandas indexes are not used. Missing,
empty and unsupported identifiers are rejected. Partition and key order follow
input order. Python equality is used: `1` and `1.0` match; strings are **not**
trimmed or canonicalized. A pairwise table's diagonal counts distinct keys in
that partition; its off-diagonal cells count distinct shared keys, not all row
pairs. The detailed table includes raw identifiers, which may be sensitive.

The audit detects **exact matches only**. It does not parse SMILES, standardize
salts, identify equivalent tautomers or detect similar compounds, patients in
the same family, duplicate records under different identifiers, or leakage from
preprocessing. Choose identifiers that represent each relevant leakage unit;
review standardization choices and audit compound, patient, batch and series
levels separately where appropriate. A clean report is not proof of an
independent or representative test set. The function reports overlap rather
than automatically removing records or resolving conflicting labels.

## Audit split files from the command line

```bash
pharmaml data audit-splits \
  --partition train=train.csv \
  --partition calibration=calibration.csv \
  --partition test=test.csv \
  --key compound_id
```

Provide two or more uniquely named partitions (names use letters, numbers, `_`
and `-`, starting with a letter). Repeat `--key` for composite identifiers,
for example `--key patient_id --key assay_id`. Every file must contain each
key column; headers are exact, case-sensitive and must be unique and nonempty.
Empty partitions, blank or whitespace-only identifiers and malformed rows fail
the audit. Input is UTF-8 CSV, TSV or TXT, or the first sheet of XLSX/XLSM.
CSV/TXT default to comma, TSV to tab; use `--delimiter semicolon`, `comma` or
`tab` to override the delimiter for text files. Quoting follows standard CSV
rules. Values are compared as text, preserving leading zeroes, literal `NA`,
case and whitespace; no identifier normalization is performed. Excel numeric
cells are converted from their stored values, **not their displayed number
formats**: export identifiers as text to preserve leading zeroes. The Excel
reader ignores trailing empty rows.

The summary shows duplicate rows, affected overlap rows and unique-key
intersections, without printing identifiers. Exit status is **0** for no
overlap, **3** for overlap, **1** for an input or execution error, and **2** for
argument parsing errors.
Within-partition duplicates alone do not change the exit status.

To review exact shared keys, add `--details-json shared.json`. This creates a
new UTF-8 JSON file with raw identifiers and per-partition counts; it refuses
to overwrite any existing file and uses owner-only permissions on POSIX.
**The detailed file may contain sensitive identifiers:** protect it and do not
commit or share it without authorization. An audit reads all keys into memory.
All exact-match limitations described above still apply.
