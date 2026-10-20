"""102 verification (python side): sklearn GP vs the web mirror.

Reads /tmp/gp_web_side.json (the bun-side fit with FIXED kernel
hyperparameters), replicates the web fitter's spaces exactly (train-set
population standardization of X and y, kernel in scaled units), fits
sklearn's GaussianProcessRegressor with the same FIXED kernel, and
compares predictions, predictive sigmas and the log marginal likelihood
to machine precision.  The honest gate: max |Δ| across the grid.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    RBF,
    WhiteKernel,
)

web = json.load(open("/tmp/gp_web_side.json"))
X = np.asarray(web["train"]["X"], dtype=float)
y = np.asarray(web["train"]["y"], dtype=float)
grid = np.asarray(web["grid"], dtype=float)

# --- replicate the web fitter's spaces (train-set population stats) ---
mu_x = X.mean(axis=0)
sd_x = np.sqrt(((X - mu_x) ** 2).mean(axis=0))
sd_x = np.where(sd_x > 1e-12, sd_x, 1.0)
Z = (X - mu_x) / sd_x
Zg = (grid - mu_x) / sd_x
mu_y = y.mean()
sd_y = float(np.sqrt(((y - mu_y) ** 2).mean()))
sd_y = sd_y if sd_y > 1e-12 else 1.0
t = (y - mu_y) / sd_y

# kernel in scaled-y units (web: sf = signalStd/sigma_y, sn = noiseStd/sigma_y)
sf = web["signalStd"] / sd_y
sn = web["noiseStd"] / sd_y
ell = web["lengthScale"]

kernel = C(sf**2, constant_value_bounds="fixed") * RBF(
    length_scale=ell, length_scale_bounds="fixed"
) + WhiteKernel(noise_level=sn**2, noise_level_bounds="fixed")
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, optimizer=None)
gp.fit(Z, t)

mean_t, std_t = gp.predict(Zg, return_std=True)
# NOTE on semantics: sklearn's return_std evaluates the FULL kernel's
# diagonal (C·RBF + White) at the test point — the white-noise floor is
# already included once, exactly like the web fitter's posteriorVar
# (v = sf² − k*ᵀK⁻¹k* + sn²).  No extra sn² here or it double-counts.
sigmas_sk = std_t * sd_y
preds_sk = mean_t * sd_y + mu_y
lml_sk = float(gp.log_marginal_likelihood())

preds_w = np.asarray(web["preds"], dtype=float)
sigmas_w = np.asarray(web["sigmas"], dtype=float)

d_pred = float(np.max(np.abs(preds_sk - preds_w)))
d_sigma = float(np.max(np.abs(sigmas_sk - sigmas_w)))
d_lml = abs(lml_sk - web["logMarginalLikelihood"])

print(f"nSupport   web={web['nSupport']} sklearn={len(Z)}")
print(f"max |Δ pred|  = {d_pred:.3e}")
print(f"max |Δ sigma| = {d_sigma:.3e}")
print(f"|Δ LML|        = {d_lml:.3e}  (web {web['logMarginalLikelihood']:.6f} vs sklearn {lml_sk:.6f})")

# The honest gate: sklearn adds alpha=1e-10 jitter to K's diagonal by
# default (the web adds none), so the two Cholesky solves agree to
# ~1e-9 on this 40×40 system rather than to 1e-15 — that gap is the
# documented jitter, not an implementation difference.
ok = d_pred < 1e-8 and d_sigma < 1e-8 and d_lml < 1e-6
print("GATE:", "PASS — web GP matches sklearn (Δ within the alpha=1e-10 jitter scale)" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
