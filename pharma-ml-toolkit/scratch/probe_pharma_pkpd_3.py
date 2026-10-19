"""Round 100x-20-a probe batch 3: fix-design boundary checks + notebook drift quantification."""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "examples")
from make_synthetic_qsar_data import make_pk_profile  # noqa: E402

from pharmaml.pharma import pk  # noqa: E402

# --- Notebook cell §3 reproduction (committed output: AUC log 17.2347) ------
profile = make_pk_profile(ka=1.2, ke=0.35, dose=250.0, vd=35.0, f=0.9)
times = profile["time_h"].to_numpy()
conc = profile["conc_ng_ml"].to_numpy()
auc_linear = pk.auc_trapezoid(times, conc)
positive = conc > 0
auc_log_current = pk.auc_trapezoid(times[positive], conc[positive], method="log")

# True logarithmic trapezoid (Purves): segment = dt*(c2-c1)/ln(c2/c1)
tp, cp = times[positive], conc[positive]
c1, c2 = cp[:-1], cp[1:]
log_ratio = np.log(c2) - np.log(c1)
flat = log_ratio == 0.0
quotient = np.where(flat, 0.5 * (c1 + c2), (c2 - c1) / np.where(flat, 1.0, log_ratio))
auc_log_true = float((quotient * np.diff(tp)).sum())

print(f"notebook committed AUC log (geometric mean): 17.2347")
print(f"reproduced current value                   : {auc_log_current:.4f}")
print(f"true log trapezoid value                   : {auc_log_true:.4f}")
print(f"drift if fixed                             : {auc_log_true - auc_log_current:+.4f} "
      f"({(auc_log_true / auc_log_current - 1) * 100:+.3f}%)")
print(f"flat segments encountered                  : {int(flat.sum())}")

# --- tdm guard boundary design -----------------------------------------------
ratio = 12.0 / 8.0
print(f"\ntdm boundary: ratio exactly 5 -> {5.0 > 5}")

# --- log rule on mixed flat + decay profile -----------------------------------
mixed_t = [0.0, 1.0, 2.0, 3.0]
mixed_c = [5.0, 5.0, 5.0, 0.5]
c1m = np.array(mixed_c[:-1]); c2m = np.array(mixed_c[1:])
lr = np.log(c2m) - np.log(c1m)
fl = lr == 0.0
q = np.where(fl, 0.5 * (c1m + c2m), (c2m - c1m) / np.where(fl, 1.0, lr))
print(f"\nmixed flat/decay true-log AUC = {float((q * np.diff(mixed_t)).sum()):.6f} "
      f"(geometric would be {float((np.sqrt(c1m * c2m) * np.diff(mixed_t)).sum()):.6f})")

# --- Co molar mass hand-check ---------------------------------------------------
Co, N, H, O = 58.933, 14.007, 1.008, 15.999
expected = Co + 6 * (N + 3 * H) + 3 * (2 * H + O)
print(f"\nCo(NH3)6.3H2O expected molar mass = {expected:.3f}")
