"""Probe: why does dnn score r2=-188 on the cell line demo data?"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from make_synthetic_cellline_data import (
    make_response_screen,
    make_expression_matrix,
)
from pharmaml.biology import load_response_table, collapse_replicates, ln_ic50_to_pic50
from pharmaml.biology.omics import normalize_expression
from pharmaml.ml import make_train_test_split
from pharmaml.ml.neural import DeepMLPRegressor

screen = load_response_table(make_response_screen(), tissue_column="tissue")
clean = collapse_replicates(screen)
expr_norm = normalize_expression(make_expression_matrix())
response = (
    clean.frame[clean.frame["compound"] == "erlotinib"]
    .set_index("cell_line")["LN_IC50"]
    .dropna()
)
X = expr_norm.loc[response.index]
y = ln_ic50_to_pic50(response)
print("X:", X.shape, "y range:", y.min(), y.max())

Xtr, Xte, ytr, yte = make_train_test_split(X, y, test_size=0.25, random_state=42)
print("train:", Xtr.shape, type(ytr), ytr.dtype)

m = DeepMLPRegressor(hidden_layers=(16, 8), epochs=60)
m.fit(Xtr, ytr)
pred = m.predict(Xte)
print("direct fit   rmse:", float(np.sqrt(np.mean((pred - yte.to_numpy()) ** 2))))
print("pred:", np.round(pred.ravel(), 3))
print("true:", yte.to_numpy().round(3))
print("n_iter_:", m.n_iter_)
