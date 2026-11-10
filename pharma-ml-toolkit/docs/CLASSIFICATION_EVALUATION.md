# Classification evaluation

`pharmaml.ml.evaluate_model(model, X_test, y_test, "classification")` returns
held-out accuracy, balanced accuracy, macro and weighted F1, macro precision and
recall, MCC, a confusion matrix and optional ROC-AUC. Fit and select the model
without using the held-out test rows. The confusion matrix uses the sorted
union of actual and predicted labels, with **true labels in rows** and
**predictions in columns**.

## ROC-AUC

AUC measures ranking, not classification accuracy or probability calibration.
Binary AUC uses the probability of the **sorted-last** label, consistent with
scikit-learn. Multiclass AUC is macro-averaged one-vs-rest. When `classes_` is
present, its order determines the probability columns; it must uniquely match
all observed true and predicted labels. Without `classes_`, column order is
assumed to follow the sorted label order. Custom predictors without that
convention must expose accurate `classes_` metadata.

`report.roc_auc` is `None` when no probability method is available, the test set
has only one true class, a class is absent from a multiclass evaluation, or
probability outputs are invalid or cannot be aligned. Probability matrices
must have one row per test sample and one column per observed label, finite
values in `[0, 1]`, and rows summing approximately to 1. A present but
inconsistent `classes_` is never ignored. Invalid or incompatible probability
outputs log a warning rather than invalidating the other classification metrics.
`None` means **unavailable**,
not zero or a performance score. This is stricter than earlier versions that
could silently use an unrelated column for AUC.

A model trained on additional classes may therefore have no reported AUC when
those classes are absent in the test split; do not compare that missing value
with a fully supported multiclass AUC. Report class supports and choose a test
split suitable for the intended use. Neither AUC nor a held-out split proves
clinical validity or performance under distribution shift.
