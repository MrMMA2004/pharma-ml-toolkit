"""100x-34 fail-first probe: the roc_auc column assumption inverts for
estimators whose predict_proba column order differs from sorted(y_true).

A conformant sklearn estimator always lays predict_proba columns out in
its (sorted) classes_ order, so proba[:, 1] happens to be the
sorted-last label's probability and roc_auc_score agrees.  A custom
estimator is free to expose classes_ in ITS OWN order -- then column 1
is not the class roc_auc_score treats as positive and a PERFECT
classifier reports AUC 0.0.  The fix: locate the positive-class column
via the estimator's own classes_ when it exposes one.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from pharmaml.ml import evaluate_model

X = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
y = np.array(["low", "high", "low", "high", "low", "high"])


class NonConformantPerfect:
    """Perfect classifier whose proba columns follow classes_ =
    ['low', 'high'] (NOT alphabetical) -- col 0 = P(low), col 1 =
    P(high).  Legal: predict_proba columns must follow classes_, and
    nothing forces a custom estimator to sort them."""

    classes_ = np.array(["low", "high"])

    def predict(self, X):
        return np.array(["low", "high", "low", "high", "low", "high"])

    def predict_proba(self, X):
        # col 0 = P(low), col 1 = P(high); perfect confidence
        return np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        )


report = evaluate_model(NonConformantPerfect(), X, y, "classification")
print(f"non-conformant perfect classifier: roc_auc = {report.roc_auc}")
assert report.roc_auc == 1.0, (
    f"FAIL-FIRST: perfect classifier reports roc_auc={report.roc_auc} "
    "(silently inverted -- proba[:, 1] is P(high) but roc_auc_score's "
    "positive class is y_true's sorted-last label 'low')"
)

# conformant sklearn estimator over the same string labels: correct today
# and must stay correct after the fix
clf = LogisticRegression().fit(X[:4], y[:4])
conformant = evaluate_model(clf, X, y, "classification")
print(f"conformant sklearn: roc_auc = {conformant.roc_auc}")
assert conformant.roc_auc is not None
print("PASS (unexpected -- bug already fixed?)")
