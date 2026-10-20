"""Round 100x-20-a adversarial probes for pharmaml.pharma (pk/pd/units/dosage/approvals).

Every suspicion is exercised EMPIRICALLY before any source edit.
Run: python scratch/probe_pharma_pkpd_1.py
"""

from __future__ import annotations

import math
import traceback

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, UnitConversionError
from pharmaml.pharma import approvals, pd as pdmod, pk, units
from pharmaml.pharma.dosage import (
    bsa_mosteller,
    maintenance_dose,
    renal_dose_adjustment,
    tdm_trough_adjustment,
)

FAILURES = []


def probe(name, fn):
    try:
        result = fn()
        print(f"[P] {name}: {result!r}")
    except Exception as exc:  # noqa: BLE001 - probe harness
        print(f"[P] {name}: RAISED {type(exc).__name__}: {exc}")
        return exc
    return result


print("=" * 72)
print("PK probes")
print("=" * 72)

# --- P1: log trapezoid vs the true logarithmic (Purves) rule --------------
# Single interval, exponential decay 10 -> 1 over dt=1.
# Exact integral = 9/ln(10) = 3.9089; the log trapezoid is EXACT here.
p1 = probe(
    "P1 log-trapezoid exactness (10x decay interval)",
    lambda: pk.auc_trapezoid([0.0, 1.0], [10.0, 1.0], method="log"),
)
print(f"    exact log-rule value = {9.0 / math.log(10):.4f}, "
      f"geometric-mean value = {math.sqrt(10.0):.4f}")
if p1 is not None and abs(p1 - 9.0 / math.log(10)) > 1e-9:
    FAILURES.append("P1: method='log' is not the logarithmic trapezoid")

# --- P2: same on a 2-decade decay ------------------------------------------
p2 = probe(
    "P2 log-trapezoid 100x decay interval",
    lambda: pk.auc_trapezoid([0.0, 1.0], [100.0, 1.0], method="log"),
)
print(f"    exact log-rule value = {99.0 / math.log(100):.4f}")
if p2 is not None and abs(p2 - 99.0 / math.log(100)) > 1e-9:
    FAILURES.append("P2: method='log' geometric mean off by "
                    f"{(p2 / (99.0 / math.log(100)) - 1) * 100:.1f}%")

# --- P3: log trapezoid on flat segment (equal concentrations) -------------
probe("P3 log-trapezoid flat segment", lambda: pk.auc_trapezoid(
    [0.0, 1.0, 2.0], [5.0, 5.0, 5.0], method="log"))

# --- P4: nca_summary with trailing BLQ zeros -------------------------------
probe("P4 nca trailing zeros (terminal slope from positives)",
      lambda: pk.nca_summary([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 6.0, 3.0, 0.0, 0.0]).as_dict())

# --- P5: nca_summary rising tail --------------------------------------------
probe("P5 nca rising tail (slope>0)", lambda: pk.nca_summary(
    [0.0, 1.0, 2.0], [1.0, 2.0, 4.0]).as_dict())

# --- P6: nca two-point profile ----------------------------------------------
probe("P6 nca two points", lambda: pk.nca_summary([0.0, 1.0], [5.0, 2.0]).as_dict())

# --- P7: nca single positive point ------------------------------------------
probe("P7 nca one positive point", lambda: pk.nca_summary(
    [0.0, 1.0, 2.0], [5.0, 0.0, 0.0]).as_dict())

# --- P8: iv_bolus_concentration with a plain list ---------------------------
probe("P8 iv_bolus list time", lambda: pk.iv_bolus_concentration(10.0, 0.5, [0.0, 1.0, 2.0]))

# --- P9: iv_bolus tuple time -------------------------------------------------
probe("P9 iv_bolus tuple time", lambda: pk.iv_bolus_concentration(10.0, 0.5, (0.0, 1.0)))

# --- P10: creatinine_clearance with non-string sex --------------------------
probe("P10 CrCl sex=None", lambda: pk.creatinine_clearance(40, 80, 1.0, None))

# --- P11: Series input (shuffled index) into auc_trapezoid ------------------
probe("P11 Series input to auc", lambda: pk.auc_trapezoid(
    pd.Series([0.0, 1.0, 2.0], index=[2, 0, 1]), [5.0, 4.0, 3.0]))

# --- P12: nca_summary duplicate-time guard through Series -------------------
probe("P12 nca unsorted Series", lambda: pk.nca_summary(
    pd.Series([2.0, 0.0, 1.0], index=[7, 8, 9]), [3.0, 5.0, 4.0]))

# --- P13: oral_tmax ka == ke -------------------------------------------------
probe("P13 oral_tmax ka==ke", lambda: pk.oral_tmax(0.5, 0.5))

# --- P14: accumulation_ratio large tau --------------------------------------
probe("P14 accumulation tau>>t12", lambda: pk.accumulation_ratio(1e6, 1.0))

# --- P15: mrt on single-interval profile ------------------------------------
probe("P15 mrt minimal", lambda: pk.mrt([0.0, 1.0], [2.0, 0.0]))

print()
print("=" * 72)
print("PD probes")
print("=" * 72)

# --- P16: NaN concentration through emax family ------------------------------
p16 = probe("P16 emax NaN conc", lambda: pdmod.emax_effect(float("nan"), 0.0, 100.0, 10.0))
if p16 is not None and math.isnan(p16):
    FAILURES.append("P16: emax_effect silently returns NaN for NaN concentration")

# --- P17: NaN inside array ----------------------------------------------------
p17 = probe("P17 sigmoid NaN in array",
            lambda: pdmod.sigmoid_emax_effect([0.0, float("nan"), 10.0], 0.0, 100.0, 10.0, 1.0))
if p17 is not None and isinstance(p17, np.ndarray) and np.isnan(p17).any():
    FAILURES.append("P17: sigmoid_emax_effect propagates NaN element silently")

# --- P18: inf concentration ---------------------------------------------------
probe("P18 emax inf conc", lambda: pdmod.emax_effect(float("inf"), 0.0, 100.0, 10.0))

# --- P19: inhibitory NaN ------------------------------------------------------
p19 = probe("P19 inhibitory NaN conc",
            lambda: pdmod.inhibitory_emax_effect(float("nan"), 90.0, 80.0, 5.0))
if p19 is not None and math.isnan(p19):
    FAILURES.append("P19: inhibitory_emax_effect silently returns NaN")

# --- P20: linear_effect NaN ---------------------------------------------------
probe("P20 linear NaN conc", lambda: pdmod.linear_effect(float("nan"), slope=2.0))

# --- P21: effect_table with NaN in grid ---------------------------------------
probe("P21 effect_table NaN grid", lambda: pdmod.effect_table(
    [0.0, float("nan"), 10.0], 0.0, 100.0, 10.0))

# --- P22: ec50_for_target_effect target == e0 (fraction exactly 0) -----------
probe("P22 ec50 target==e0", lambda: pdmod.ec50_for_target_effect(0.0, 10.0, 0.0, 100.0))

# --- P23: ec50 negative emax (inhibitory direction inversion) -----------------
probe("P23 ec50 negative emax",
      lambda: pdmod.ec50_for_target_effect(50.0, 10.0, 90.0, -80.0))

# --- P24: emax effect at 0 / saturation pins ---------------------------------
probe("P24 inhibitory conc=0", lambda: pdmod.inhibitory_emax_effect(0.0, 90.0, 80.0, 5.0, hill=3.0))

# --- P25: hill tiny ------------------------------------------------------------
probe("P25 sigmoid hill=1e-9", lambda: pdmod.sigmoid_emax_effect(3.0, 0.0, 100.0, 10.0, 1e-9))

print()
print("=" * 72)
print("Units probes")
print("=" * 72)

# --- P26: molar_mass docstring example Co(NH3)6.3H2O --------------------------
p26 = probe("P26 molar_mass Co(NH3)6.3H2O (docstring example)", lambda: units.molar_mass("Co(NH3)6.3H2O"))
if isinstance(p26, Exception):
    FAILURES.append("P26: docstring example Co(NH3)6.3H2O raises " + str(p26))

# --- P27: empty paren group -----------------------------------------------------
p27 = probe("P27 molar_mass Ca()2", lambda: units.molar_mass("Ca()2"))
if not isinstance(p27, Exception):
    FAILURES.append(f"P27: malformed 'Ca()2' silently accepted -> {p27}")

# --- P28: hydrate with empty tail ------------------------------------------------
p28 = probe("P28 molar_mass MgSO4.7 (dangling hydrate)", lambda: units.molar_mass("MgSO4.7"))
if not isinstance(p28, Exception):
    FAILURES.append(f"P28: malformed 'MgSO4.7' silently accepted -> {p28}")

# --- P29: molar_mass None ---------------------------------------------------------
probe("P29 molar_mass(None)", lambda: units.molar_mass(None))

# --- P30: nested parentheses -------------------------------------------------------
probe("P30 molar_mass K4(Fe(CN)6)",
      lambda: units.molar_mass("K4(Fe(CN)6)"))

# --- P31: lower-case formula --------------------------------------------------------
probe("P31 molar_mass h2o", lambda: units.molar_mass("h2o"))

# --- P32: convert negative ----------------------------------------------------------
probe("P32 convert -5 mg -> g", lambda: units.UnitConverter.convert(-5.0, "mg", "g"))

# --- P33: convert round trip precision ------------------------------------------------
probe("P33 roundtrip mg->ng->mg", lambda: units.UnitConverter.convert(
    units.UnitConverter.convert(0.123, "mg", "ng"), "ng", "mg"))

# --- P34: unknown unit error quality ---------------------------------------------------
probe("P34 unknown unit", lambda: units.UnitConverter.convert(1.0, "Mg", "g"))

# --- P35: molar mass with spaces --------------------------------------------------------
probe("P35 molar_mass 'C8 H10 N4 O2'", lambda: units.molar_mass("C8 H10 N4 O2"))

print()
print("=" * 72)
print("Dosage probes")
print("=" * 72)

# --- P36: tdm huge reduction (measured >> target) ---------------------------------------
p36 = probe("P36 tdm 10x reduction", lambda: tdm_trough_adjustment(1000.0, 50.0, 5.0))
if not isinstance(p36, Exception):
    FAILURES.append("P36: tdm_trough_adjustment cuts dose 10x with no "
                    "non-linear-kinetics guard (one-sided ratio check)")

# --- P37: renal adjustment when CrCl above normal --------------------------------------
p37 = probe("P37 renal CrCl 150 vs normal 100", lambda: renal_dose_adjustment(100.0, 150.0, 100.0))
if not isinstance(p37, Exception) and p37 > 100.0:
    print("    -> docstring says 'Linearly reduce a dose for impaired renal "
          "function' but the dose ESCALATES above normal")

# --- P38: maintenance dose f tolerance ---------------------------------------------------
probe("P38 maintenance f=1.0+1e-10", lambda: maintenance_dose(5.0, 10.0, 12.0, f=1.0 + 1e-10))

# --- P39: bsa bounds ------------------------------------------------------------------------
probe("P39 bsa 250cm/400kg", lambda: bsa_mosteller(250.0, 400.0))

# --- P40: dose_by_weight exact ceiling ------------------------------------------------------
probe("P40 dose_by_weight dose == max_dose", lambda: __import__(
    "pharmaml.pharma.dosage", fromlist=["dose_by_weight"]).dose_by_weight(10.0, 8.0, max_dose=80.0))

print()
print("=" * 72)
print("Approvals probes")
print("=" * 72)

# --- P41: blank drug cell collected as "" ---------------------------------------------------
blank_frame = pd.DataFrame({
    "drug": ["Aspirin", "   ", ""],
    "regulatory approval": ["Approved", "Approved", "Approved"],
})
p41 = probe("P41 blank drug cells", lambda: approvals.filter_approved_drugs(blank_frame))
if not isinstance(p41, Exception) and "" in p41:
    FAILURES.append("P41: blank drug cells collected as '' in output list")

# --- P42: approval_status empty string matches NaN rows -------------------------------------
nan_frame = pd.DataFrame({
    "drug": ["Aspirin", "Warfarin"],
    "regulatory approval": ["Approved", None],
})
probe("P42 status='' matches missing rows", lambda: approvals.filter_approved_drugs(
    nan_frame, approval_status=""))

# --- P43: whitespace-only approval cell ------------------------------------------------------
ws_frame = pd.DataFrame({
    "drug": ["Aspirin", "Warfarin"],
    "regulatory approval": [" Approved ", "withdrawn"],
})
probe("P43 whitespace status", lambda: approvals.filter_approved_drugs(ws_frame))

# --- P44: numeric drug names -------------------------------------------------------------------
num_frame = pd.DataFrame({"drug": [123, 4.5], "regulatory approval": ["Approved", "Approved"]})
probe("P44 numeric drug cells", lambda: approvals.filter_approved_drugs(num_frame))

print()
print("=" * 72)
print(f"CONFIRMED FAILURES: {len(FAILURES)}")
for f in FAILURES:
    print(f"  - {f}")
print("=" * 72)
