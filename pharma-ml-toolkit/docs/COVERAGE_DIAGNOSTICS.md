# Evaluating coverage by group

An overall coverage number can hide differences between compound series, assay
batches, classes or other groups. `coverage_by_group` summarizes **held-out**
boolean coverage outcomes overall and for each observed group, with sample
counts and two-sided Wilson score intervals. It optionally summarizes the mean
prediction-set size or interval width as well.

```python
from pharmaml.ml import coverage_by_group

# These rows come from an independent evaluation set. Align all inputs first.
covered = [True, False, True, True, False, True]
batch = ["A", "A", "B", "B", "B", "B"]
set_sizes = [1, 1, 2, 1, 0, 2]
report = coverage_by_group(covered, batch, sizes=set_sizes, confidence=0.95)
print(report.overall.coverage, report.overall.lower, report.overall.upper)
print(report.to_dataframe())  # group, n_test, n_covered, coverage, lower, upper, mean_size
```

For [conformal classification](CONFORMAL_CLASSIFICATION.md), pass the result of
`predict_set(X_test, y_true=y_test)`:

```python
report = coverage_by_group(result.covered, test_groups, sizes=result.set_sizes)
```

For [conformal regression](CONFORMAL_REGRESSION.md), `predict_interval` now
provides `result.covered` when `y_true` is supplied; use
`result.upper - result.lower` for `sizes`. Both types return `None` for
`covered` when no labels were provided. The existing regression DataFrame
columns remain unchanged.

Groups appear in first-seen order. Labels must be finite numeric or string
scalars; grouping follows Python equality (for example, `1` and `1.0` are the
same group). Arrays and pandas inputs are aligned **by position**, not index.
Empty or missing coverage/groups, non-boolean coverage, non-finite sizes and
length mismatches are rejected. No groups with zero observed test rows can be
reported. A small group can have a very wide interval.

## Interpretation

Wilson bounds approximate binomial sampling uncertainty for independent
observations with a common coverage probability within each group, conditional
on the fixed fitted model and calibration. They are not conformal prediction
intervals, do not adjust for multiple comparisons, and do not guarantee
coverage within a subgroup. Correlated test rows, distribution shift, group
selection after observing outcomes, and tuning on the evaluation set can make
these numbers misleading. Prespecify meaningful groups and evaluate on data
held out from both training and calibration; where rows are correlated, use a
study design and uncertainty analysis appropriate for the sampling unit. This
is research software, not clinical decision support.
