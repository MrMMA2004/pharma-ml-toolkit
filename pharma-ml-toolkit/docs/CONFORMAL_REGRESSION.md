# Split-conformal regression intervals

`calibrate_split_conformal` wraps an already-fitted, single-output regressor
with symmetric prediction intervals based on **held-out calibration residuals**.
It does not fit the estimator or its preprocessing. Separate training,
calibration and test data before fitting or selecting any model.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pharmaml.ml import calibrate_split_conformal

X, y = make_regression(n_samples=300, n_features=4, noise=20, random_state=2)
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.4, random_state=11)
X_cal, X_test, y_cal, y_test = train_test_split(X_hold, y_hold, test_size=0.5, random_state=12)

# Fit all preprocessing and model selection using training data only.
model = make_pipeline(StandardScaler(), Ridge()).fit(X_train, y_train)
calibrated = calibrate_split_conformal(model, X_cal, y_cal, alpha=0.1)
intervals = calibrated.predict_interval(X_test, y_true=y_test)
print(intervals.to_dataframe().head())  # prediction, lower, upper
print("empirical test coverage:", intervals.coverage)
print("mean interval width:", intervals.mean_width)
```

For `n` calibration rows, the radius is the `ceil((n + 1) * (1 - alpha))`-th
smallest absolute calibration residual, using 1-based ranks. Endpoints are
included when evaluating coverage. If that rank exceeds `n`, there is no finite
radius with the requested finite-sample guarantee; the function raises an error
rather than clipping the rank. For example, `alpha=0.1` needs at least nine
calibration rows. Data and predictions must be finite one-dimensional vectors.
Align rows before calling: arrays and pandas objects are matched by position,
not by index. Recognized classifiers are rejected, but the caller must ensure
any custom predictor is a regressor.

**Interpretation and limitations**

- With calibration and future observations exchangeable, and a model trained
  independently of both, split conformal provides at least `1 - alpha` marginal
  coverage. This is not a conditional, per-subgroup, or per-molecule guarantee.
- Dataset shift, scaffold or temporal splits, correlated observations, repeated
  measurements, calibration leakage, or tuning the model using calibration or
  test data can invalidate that guarantee. Validate the split against your use
  case and measure coverage on independent data.
- `coverage` is the empirical fraction on the labels passed to `predict_interval`,
  not a guarantee. It is `None` when labels are omitted. The same calibration
  radius is used for every query; it does not estimate molecule-specific error.
- The calibrator holds a reference to the model, not a copy. **Do not refit or
  mutate the model or preprocessing after calibration.** The toolkit cannot
  check that the supplied datasets are disjoint or exchangeable.
- This implementation covers single-output regression only. It is research
  software, not a clinical decision-support system.
