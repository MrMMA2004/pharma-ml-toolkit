"""Cross-validation probe for the PK/PD Explorer frontend (Task 100x-20-b).

Runs the three preset concentration-time datasets + PD model checks through
the real pharmaml.pharma implementation so the TypeScript mirror can be pinned
against exact expected numbers.
"""

import numpy as np

from pharmaml.pharma import (
    auc_trapezoid,
    nca_summary,
    clearance,
    volume_of_distribution,
    half_life,
)
from pharmaml.pharma.pd import (
    emax_effect,
    sigmoid_emax_effect,
    inhibitory_emax_effect,
    linear_effect,
)

PRESETS = {
    "ORAL_CLEAN": {
        "t": [0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48],
        "c": [0, 1.28, 2.21, 3.33, 4.08, 3.85, 3.31, 2.82, 2.05, 0.79, 0.30, 0.12],
        "dose": 250.0,
        "f": 0.9,
    },
    "IV_BOLUS": {
        "t": [0, 0.5, 1, 2, 4, 6, 8, 10, 12],
        "c": [12.00, 10.59, 9.35, 7.28, 4.41, 2.68, 1.62, 0.99, 0.60],
        "dose": 500.0,
        "f": 1.0,
    },
    "NOISY_ORAL": {
        "t": [0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48],
        "c": [0.00, 1.35, 2.05, 3.45, 3.98, 4.12, 3.18, 2.90, 1.98, 0.85, 0.34, 0.08],
        "dose": 250.0,
        "f": 0.9,
    },
}


def terminal_fit(t, c, n):
    """Log-linear LS regression on the last n positive points after Tmax."""
    t = np.asarray(t, float)
    c = np.asarray(c, float)
    tmax = t[int(np.argmax(c))]
    mask = (c > 0) & (t > tmax)
    tt, cc = t[mask], c[mask]
    tt, cc = tt[-n:], cc[-n:]
    y = np.log(cc)
    slope, intercept = np.polyfit(tt, y, 1)
    yhat = slope * tt + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ke = -slope
    return ke, intercept, r2, len(tt), float(tt[0]), float(cc[-1])


for name, p in PRESETS.items():
    t, c = p["t"], p["c"]
    print(f"=== {name} ===")
    s = nca_summary(t, c)
    print(f"  auc_trapezoid (AUC0-last) = {auc_trapezoid(t, c):.6f}")
    print(f"  nca_summary: Cmax={s.c_max:.4f} Tmax={s.t_max:.4f} "
          f"thalf(last2)={s.half_life_terminal:.6f} MRT={s.mrt:.4f}")
    for n in (2, 3, 4):
        ke, b, r2, k, t0, clast = terminal_fit(t, c, n)
        auc = auc_trapezoid(t, c)
        clast_obs = c[-1] if c[-1] > 0 else None
        auc_extrap = clast / ke
        auc_inf = auc + auc_extrap
        pct = 100 * auc_extrap / auc_inf
        cl = p["f"] * p["dose"] / auc_inf
        vz = volume_of_distribution(cl, ke)
        print(f"  [N={n}] pts={k} ke={ke:.6f} t1/2={half_life(ke):.6f} "
              f"r2={r2:.6f} Clast={clast:.4f}")
        print(f"        AUCextrap={auc_extrap:.6f} AUCinf={auc_inf:.6f} "
              f"%extrap={pct:.4f} CL={cl:.6f} Vz={vz:.6f}")
    print()

print("=== PD models (pharmaml.pharma.pd) ===")
print(f"linear_effect: slope=8  C=[0, 1.28, 4.08, 12.80] -> {linear_effect([0, 1.28, 4.08, 12.80], 8.0)}")
print(f"emax_effect: E0=10 Emax=80 EC50=2 C=[0, 1.28, 4.08, 12.80] -> {emax_effect([0, 1.28, 4.08, 12.80], 10.0, 80.0, 2.0)}")
print(f"sigmoid: E0=10 Emax=80 EC50=2 h=1.5 C=[1.28, 4.08, 12.80] -> {sigmoid_emax_effect([1.28, 4.08, 12.80], 10.0, 80.0, 2.0, 1.5)}")
print(f"inhibitory: E0=10 Imax=10 IC50=2 C=[0, 1.28, 4.08, 12.80] -> {inhibitory_emax_effect([0, 1.28, 4.08, 12.80], 10.0, 10.0, 2.0)}")
print(f"clearance(250, AUCinf_iv?)... skipped; clearance() = F*D/AUC per current impl")
