"""Probe: print the real numbers the Response AI page (audit row 12) can pin.

Run from the toolkit root: python scratch/probe_response_218.py

Every number the future page's tests assert is engine-printed here FIRST
(the round-205+ house rule): the Hill fit on a seeded four-parameter
logistic demo (seed 42), the closed-form dose-response curve + the
log-dose AUC, the response metrics on a seeded regression pair, the
rank correlations, the DrugResponseModel trained on a seeded
12-cell-line x 4-drug screen with a NaN mask (seed 42), the
leave-nothing-out cross-validation report over the same screen, and the
InformativeRegression / ResponseVAE / MultiOmicsFusion smokes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# == 1. the Hill fit: a seeded 4PL demo ====================================
print("== hill fit ==")
from pharmaml.ml.drugresponse import (
    auc_log_dose,
    cross_validate_response,
    dose_response_curve,
    fit_hill_curve,
    rank_correlations,
    response_metrics,
)

hill_true = dict(hill_coef=1.2, ic50=0.8, top=1.0, bottom=0.05)
doses = np.logspace(-3, 3, 21)
clean = dose_response_curve(doses, hill_true["hill_coef"], hill_true["ic50"],
                            top=hill_true["top"], bottom=hill_true["bottom"])
response = clean + rng.normal(0.0, 0.01, doses.size)
fit = fit_hill_curve(doses, response)
print("true hill/ic50/top/bottom:", hill_true)
params = fit.parameters
if isinstance(params, dict):
    print("fitted parameters:", {k: round(float(v), 4) for k, v in params.items()})
else:
    print("fitted parameters:", np.round(np.asarray(params, dtype=float), 4).tolist())
print("r_squared:", round(float(fit.r_squared), 6), "converged:", fit.converged,
      "n_points:", fit.n_points, "fitted_auc:", round(float(fit.fitted_auc), 6))

# == 2. the closed-form curve + the log-dose AUC ===========================
print("== dose response curve ==")
for d in (0.008, 0.8, 800.0):
    print(f"curve({d}):", round(float(dose_response_curve(d, **hill_true)), 6))
print("auc_log_dose(clean):", round(float(auc_log_dose(doses, clean)), 6))

# == 3. the response metrics ==============================================
print("== response metrics ==")
y_true = rng.uniform(0.0, 1.0, 60)
y_pred = np.clip(y_true + rng.normal(0.0, 0.08, 60), 0.0, 1.0)
metrics = response_metrics(y_true, y_pred)
print({k: (int(v) if k == "n" else round(float(v), 6)) for k, v in sorted(metrics.items())})

# == 4. the rank correlations =============================================
print("== rank correlations ==")
x = rng.uniform(0.0, 10.0, 48)
y = 0.7 * x + rng.normal(0.0, 1.5, 48)
rc = rank_correlations(x, y)
print("n:", rc.n, "pearson:", round(float(rc.pearson), 6),
      "spearman:", round(float(rc.spearman), 6), "kendall:", round(float(rc.kendall), 6))

# == 5. the DrugResponseModel on a seeded screen ===========================
print("== drug response model ==")
from pharmaml.ml.drugresponse import DrugResponseModel

n_cells, n_drugs, n_features = 12, 4, 8
X = pd.DataFrame(
    rng.normal(0.0, 1.0, (n_cells, n_features)),
    index=[f"CELL_{i:02d}" for i in range(n_cells)],
    columns=[f"feat_{j}" for j in range(n_features)],
)
true_factor = rng.normal(0.0, 1.0, n_cells)
drugs = ["DRUG_A", "DRUG_B", "DRUG_C", "DRUG_D"]
potency = np.array([0.9, 0.7, 0.5, 0.3])
table = pd.DataFrame(
    {
        d: 1.0 / (1.0 + np.exp(-(potency[k] * true_factor + rng.normal(0.0, 0.05, n_cells))))
        for k, d in enumerate(drugs)
    },
    index=X.index,
)
mask = rng.random(table.shape) < 0.15
table = table.mask(mask)
print("screen:", table.shape, "observed pairs:", int(table.notna().sum().sum()))
model = DrugResponseModel(n_factors=2, lr=0.1, epochs=120, l2=0.001, seed=42)
history = model.fit(X, table)
print("final loss:", round(float(history.final_loss), 6), "epochs:", history.n_epochs)
print("observed pairs:", history.n_observed)
print("in-sample pearson_r:", round(float(history.pearson_r), 6))
pred = model.predict(["CELL_00", "CELL_07"])
print("predict head:\n", pred.round(4).to_string())

# == 6. the cross-validation report ========================================
print("== cross validation ==")
report = cross_validate_response(
    lambda: DrugResponseModel(n_factors=2, lr=0.1, epochs=120, l2=0.001, seed=42),
    X, table, n_folds=4, seed=42,
)
pearsons = []
for fold in report.folds:
    m = fold.metrics
    pearsons.append(float(m["pearson"]))
    print(
        "fold", fold.fold, "test cells:", fold.n_test_cells,
        "observed:", fold.n_observed,
        "pearson:", round(float(m["pearson"]), 6),
        "rmse:", round(float(m["rmse"]), 6),
    )
print("mean pearson over", report.n_folds, "folds:", round(float(np.mean(pearsons)), 6))

# == 7. the InformativeRegression smoke ====================================
print("== informative regression ==")
from pharmaml.ml.drugresponse import InformativeRegression

Xr = np.column_stack([np.ones(40), rng.normal(0.0, 1.0, 40), rng.normal(0.0, 1.0, 40)])
beta = np.array([0.5, 1.5, -0.8])
yr = Xr @ beta + rng.normal(0.0, 0.1, 40)
prior = np.array([100.0, 4.0, 4.0])  # informative: betas near +-1.5/0.8
info = InformativeRegression(lr=0.05, epochs=300, l2=1.0)
losses = info.fit(Xr, yr, prior_variance=prior)
print("losses: first", round(float(losses[0]), 6), "last", round(float(losses[-1]), 6))
print("intercept:", round(float(info.intercept), 4), "coef:", np.round(np.asarray(info.coef), 4).tolist())

# == 8. the ResponseVAE smoke ==============================================
print("== response vae ==")
from pharmaml.ml.drugresponse import ResponseVAE

Xv = rng.normal(0.0, 1.0, (60, 10))
yv = (Xv @ rng.normal(0.0, 0.5, 10)) + rng.normal(0.0, 0.1, 60)
vae = ResponseVAE(latent_dim=4, hidden=16, lr=0.01, epochs=80, seed=42)
vae_history = vae.fit(Xv, yv, seed=42)
print("elbo: first", round(float(vae_history.elbo[0]), 6), "last", round(float(vae_history.elbo[-1]), 6),
      "(the bound must improve)")
print("reconstruction last:", round(float(vae_history.reconstruction[-1]), 6),
      "kl last:", round(float(vae_history.kl_divergence[-1]), 6),
      "response_loss last:", round(float(vae_history.response_loss[-1]), 6),
      "epochs:", vae_history.n_epochs)

# == 9. the MultiOmicsFusion smoke =========================================
print("== multi omics fusion ==")
from pharmaml.ml.drugresponse import MultiOmicsFusion

blocks = {
    "expr": pd.DataFrame(rng.normal(0.0, 1.0, (30, 12)), index=[f"C{i:02d}" for i in range(30)]),
    "mut": pd.DataFrame(rng.integers(0, 2, (30, 8)).astype(float), index=[f"C{i:02d}" for i in range(30)]),
}
fusion = MultiOmicsFusion(equal_block_variance=True)
latent = fusion.fit_transform(blocks)
print("latent shape:", latent.shape, "first columns:", [str(c) for c in latent.columns[:4]], "...")
print("block names:", fusion.block_names_)
print("block stds (per-block mean of the column stds):",
      {k: round(float(v.mean()), 6) for k, v in fusion.block_stds_.items()})
print("block means (per-block mean of the column means):",
      {k: round(float(v.mean()), 6) for k, v in fusion.block_means_.items()})
