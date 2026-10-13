"""Pharmaceutical-calculation example: full NCA + dosing design.

Demonstrates the new ``pharmaml.pharma`` domain end-to-end on a
synthetic oral profile:

1. non-compartmental analysis (AUC/AUMC/MRT/Cmax/Tmax/t½),
2. PD target-effect design with the Emax model,
3. dosing regimen design (loading + maintenance dose),
4. renal adjustment and TDM trough correction.

Usage::

    python examples/run_pk_analysis.py
"""

from __future__ import annotations

import numpy as np

from pharmaml.logger import configure_logging
from pharmaml.pharma.dosage import (
    bsa_mosteller,
    loading_dose,
    maintenance_dose,
    renal_dose_adjustment,
    tdm_trough_adjustment,
)
from pharmaml.pharma.pd import ec50_for_target_effect, sigmoid_emax_effect
from pharmaml.pharma.pk import (
    clearance,
    creatinine_clearance,
    elimination_rate_constant,
    nca_summary,
    oral_cmax,
    oral_tmax,
)
from pharmaml.pharma.units import UnitConverter, molar_mass


def main() -> None:
    configure_logging()

    # ---- 1. Non-compartmental analysis of a synthetic profile -------
    from make_synthetic_qsar_data import make_pk_profile

    profile = make_pk_profile(ka=1.2, ke=0.35, dose=250.0, vd=35.0, f=0.9)
    times = profile["time_h"].to_numpy()
    concs = profile["conc_ng_ml"].to_numpy()
    summary = nca_summary(times, concs)

    print("=== Non-compartmental analysis (synthetic oral profile) ===")
    for key, value in summary.as_dict().items():
        shown = "n/a" if not np.isfinite(value) else f"{value:.3f}"
        print(f"{key:<18} {shown}")

    # ---- 2. Derived PK parameters ------------------------------------
    # AUC after a 250 mg oral dose with F = 0.9 -> apparent clearance.
    auc = summary.auc
    cl_f = clearance(dose=250.0, auc=auc, f=0.9)  # CL/F actually
    ke = elimination_rate_constant(summary.half_life_terminal)
    print(f"\nCL/F              {cl_f:.2f} L/h (apparent)")
    print(f"ke                {ke:.4f} 1/h")

    # ---- 3. Analytic 1-compartment expectations ----------------------
    t_max = oral_tmax(ka=1.2, ke=0.35)
    c_max = oral_cmax(dose=250.0, ka=1.2, ke=0.35, vd=35.0, f=0.9)
    print(f"Tmax (analytic)   {t_max:.2f} h   vs. NCA {summary.t_max:.2f} h")
    print(f"Cmax (analytic)   {c_max:.2f} ng/mL vs. NCA {summary.c_max:.2f}")

    # ---- 4. PD target-effect design -----------------------------------
    # Sigmoid Emax: EC50 20 ng/mL, Hill 1.5; find concentration for 80% Emax.
    ec80 = ec50_for_target_effect(
        target_effect=80.0, ec50=20.0, e0=0.0, emax=100.0, hill=1.5
    )
    effect_check = sigmoid_emax_effect(ec80, e0=0.0, emax=100.0, ec50=20.0, hill=1.5)
    print(f"\nC for 80% effect {ec80:.2f} ng/mL (check: E = {effect_check:.1f}%)")

    # ---- 5. Dosage design ----------------------------------------------
    maintenance = maintenance_dose(cl_f, target_css=15.0, tau=12.0, f=0.9)
    loading = loading_dose(vd=35.0, target_concentration=15.0, f=0.9)
    print(f"\nMaintenance dose {maintenance:.1f} mg per 12 h")
    print(f"Loading dose     {loading:.1f} mg")

    # ---- 6. Patient-specific adjustments -------------------------------
    crcl = creatinine_clearance(age_years=67, weight_kg=58, serum_cr_mg_dl=1.4, sex="female")
    adjusted = renal_dose_adjustment(maintenance, current_creatinine_clearance=crcl)
    print(f"\nCockcroft-Gault CrCl {crcl:.1f} mL/min -> adjusted dose {adjusted:.1f} mg")
    print(f"BSA (170 cm, 58 kg)  {bsa_mosteller(170, 58):.2f} m^2")

    # ---- 7. TDM correction ----------------------------------------------
    new_dose = tdm_trough_adjustment(
        current_dose=adjusted, measured_trough=8.0, target_trough=12.0
    )
    print(f"TDM 8 -> 12 ng/mL trough: dose {adjusted:.1f} -> {new_dose:.1f} mg")

    # ---- 8. Unit sanity checks -------------------------------------------
    print(f"\nUnit checks: 500 mg = {UnitConverter.convert(500, 'mg', 'g')} g, "
          f"1 uM = {UnitConverter.convert(1, 'uM', 'nM')} nM")
    print(f"Molar mass caffeine (C8H10N4O2) = {molar_mass('C8H10N4O2'):.2f} g/mol")


if __name__ == "__main__":
    main()
