# Split-conformal classification sets

`calibrate_split_conformal_classifier` wraps a **fitted** probabilistic
classifier with prediction sets. A set contains every class whose probability
is high enough to pass a threshold calculated on separate calibration rows.
The model is not trained, copied or refitted by this API.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pharmaml.ml import (
    calibrate_label_conditional_classifier,
    calibrate_split_conformal_classifier,
    coverage_by_group,
)

X, y = make_classification(n_samples=600, n_features=8, n_informative=6,
                           n_classes=3, random_state=4)
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.4, random_state=5)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_hold, y_hold, test_size=0.5, random_state=6)

# Fit preprocessing and select hyperparameters on training data only.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
model.fit(X_train, y_train)
calibrated = calibrate_split_conformal_classifier(model, X_cal, y_cal, alpha=0.1)
result = calibrated.predict_set(X_test, y_true=y_test)
print(result.to_dataframe().head())  # prediction_set, set_size, covered
print("empirical coverage:", result.coverage)
print("mean set size:", result.mean_set_size)

# Optional: a separate threshold for each class, using the same held-out split.
by_class = calibrate_label_conditional_classifier(model, X_cal, y_cal, alpha=0.1)
print(by_class.calibration_summary())  # per-class counts, ranks, thresholds, fallbacks
class_sets = by_class.predict_set(X_test, y_true=y_test)
print(coverage_by_group(class_sets.covered, y_test, sizes=class_sets.set_sizes).to_dataframe())
```

## Method and outputs

For each calibration row the nonconformity score is `1 - P(true label | X)`.
For `n` calibration rows, select the `k = ceil((n + 1) * (1 - alpha))`-th
smallest score, using a one-based rank without interpolation. A query label is
included when its score is **at most** this threshold; ties are included. If
`k > n`, the threshold is 1, returning all fitted classes for every row rather
than clipping the rank. Sets can otherwise be empty, contain one label, or
contain several labels; `mean_set_size` measures their average size.

The classifier must have `classes_` and `predict_proba` whose columns follow
that exact class order. Probabilities must be finite, in `[0, 1]`, and sum to
one per row. Labels are finite numeric or string scalars, and `y_cal` must
contain only fitted classes. Alignment is positional: pandas indexes are **not**
checked. `membership` is a boolean matrix in `classes` order. When `y_true` is
provided, `covered`, `n_covered` and `coverage` describe that independent query
set; unknown query labels count as uncovered. Without `y_true`, they are
`None`. These are empirical measurements, not guarantees.

## Label-conditional sets

`calibrate_label_conditional_classifier` uses a separate threshold for each
true class. For a class with `n_c` calibration rows, the threshold is the
`ceil((n_c + 1) * (1 - alpha))`-th smallest true-class nonconformity score.
If that rank exceeds `n_c`, including when the class has **no calibration
rows**, the threshold is 1 and that class is included in every set. This is a
conservative fallback, not evidence that the classifier is good for that class.
The `calibration_summary()` table flags these fallbacks and shows each class's
count and rank; small classes can make sets substantially larger.

Label-conditional coverage of at least `1 - alpha` for a particular known
class requires calibration and future observations to be exchangeable
**within that class**, conditional on an independently trained and fixed model.
The method does not promise coverage for subgroups within a class, for an
individual, or under distribution shift. The fitted class list must contain the
possible deployment labels; an unseen label cannot appear in a set. Use
independent test labels to measure empirical coverage per class and review the
sample counts; [`coverage_by_group`](COVERAGE_DIAGNOSTICS.md) can help.

The returned `ConformalPredictionSets` has `class_thresholds` in `classes`
order and its scalar `threshold` is `None`. Marginal sets instead report a
scalar `threshold` and `class_thresholds` is `None`. Both expose the same
membership matrix, set sizes, and optional empirical coverage.

## Interpretation and limitations

- The marginal method's coverage of at least `1 - alpha` requires exchangeability
  of calibration and future observations, an independently trained and fixed
  model, and **all possible labels in the model's fixed class space**. An unseen
  class at deployment is never in a prediction set, even when the threshold is
  1; the guarantee does not apply to that event.
- Marginal coverage is not per class, subgroup or molecule. The label-conditional
  method targets known classes separately under the within-class assumptions
  above, but still not subgroups or individuals. Class imbalance and small or
  unrepresentative calibration sets can lead to uninformative or unequal sets. Neither a particular set nor its size
  is an individual confidence guarantee. Probabilities need not be calibrated
  for the marginal guarantee, but their quality affects set efficiency.
- Dataset shift, scaffold or temporal splits, correlated observations, repeated
  measurements, or tuning on calibration/test data can invalidate the
  assumptions. Use a separate test set to assess performance.
- The calibrator holds the model by reference. Do not refit or mutate the model
  or its preprocessing after calibration; only a changed `classes_` order is
  explicitly detected. The toolkit cannot verify disjointness or exchangeability.
- Use `covered` and `set_sizes` with [coverage diagnostics](COVERAGE_DIAGNOSTICS.md)
  to inspect empirical subgroup results; marginal coverage is not a subgroup guarantee.
- This is research software, not a clinical decision-support system.
