"""Round 100x-20-a probe batch 2: input-type robustness + CLI paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.pharma import pd as pdmod, pk
from pharmaml.utils.validation import validate_concentration_series

def probe(name, fn):
    try:
        result = fn()
        print(f"[P] {name}: {result!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"[P] {name}: RAISED {type(exc).__name__}: {exc}")
        return exc
    return result

# --- P45: string values in a concentration series ---------------------------
probe("P45 validate_conc_series strings", lambda: validate_concentration_series(
    [0.0, 1.0], ["1.0", "2.0"]))
probe("P46 validate_conc_series junk strings", lambda: validate_concentration_series(
    [0.0, 1.0], ["low", "high"]))
probe("P47 validate_conc_series None", lambda: validate_concentration_series(
    [0.0, None], [1.0, 2.0]))
probe("P48 validate_conc_series object dtype", lambda: validate_concentration_series(
    pd.Series([0.0, 1.0], dtype=object), [1.0, 2.0]))

# --- P49: effect_table with a generator --------------------------------------
probe("P49 effect_table generator", lambda: pdmod.effect_table(
    (x for x in [0.0, 10.0, 100.0]), 0.0, 100.0, 10.0))

# --- P50: emax with plain list -------------------------------------------------
probe("P50 emax list input", lambda: pdmod.emax_effect([0.0, 10.0], 0.0, 100.0, 10.0))

# --- P51: iv_bolus with Series -------------------------------------------------
probe("P51 iv_bolus Series time", lambda: pk.iv_bolus_concentration(
    10.0, 0.5, pd.Series([0.0, 1.0, 2.0])))

# --- P52: nca on Series pairs --------------------------------------------------
probe("P52 nca Series pair", lambda: pk.nca_summary(
    pd.Series([0.0, 1.0, 2.0]), pd.Series([5.0, 3.0, 1.0])).as_dict())

# --- P53: CLI pk on a CSV with a text cell -------------------------------------
from pharmaml.cli import main as cli_main
import tempfile, pathlib
with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    messy = tmp / "messy.csv"
    messy.write_text("time,conc\n0,5.0\n1,blq\n2,1.0\n", encoding="utf-8")
    probe("P53 CLI pk messy CSV", lambda: cli_main(
        ["pharma", "pk", "--data", str(messy), "--time", "time", "--conc", "conc"]))

    good = tmp / "good.csv"
    good.write_text("time,conc\n0,10.0\n1,6.0\n2,3.6\n3,2.2\n", encoding="utf-8")
    probe("P54 CLI pk good CSV", lambda: cli_main(
        ["pharma", "pk", "--data", str(good)]))

    probe("P55 CLI dose negative CL", lambda: cli_main(
        ["pharma", "dose", "--cl", "-5.0", "--vd", "35.0", "--target-css", "15.0"]))

    probe("P56 CLI dose normal", lambda: cli_main(
        ["pharma", "dose", "--cl", "5.0", "--vd", "35.0", "--target-css", "15.0",
         "--tau", "12.0", "--f", "0.9"]))

    probe("P57 CLI pk missing column", lambda: cli_main(
        ["pharma", "pk", "--data", str(good), "--conc", "nope"]))

# --- P58: mrt with interior zero (AUC>0, one zero point) ------------------------
probe("P58 mrt interior zero", lambda: pk.mrt([0.0, 1.0, 2.0], [4.0, 0.0, 4.0]))

# --- P59: nca interior zero terminal pick ----------------------------------------
probe("P59 nca interior zero (non-adjacent positives)",
      lambda: pk.nca_summary([0.0, 1.0, 2.0], [5.0, 0.0, 3.0]).as_dict())

# --- P60: oral_cmax flip-flop ------------------------------------------------------
probe("P60 oral_cmax ka<ke", lambda: pk.oral_cmax(100.0, 0.2, 0.5, 20.0, 0.9))

# --- P61: half_life accepts numpy scalar --------------------------------------------
probe("P61 half_life np scalar", lambda: pk.half_life(np.float64(0.35)))

# --- P62: accumulation ratio tiny tau -------------------------------------------------
probe("P62 accumulation tiny tau", lambda: pk.accumulation_ratio(1e-9, 1.0))

# --- P63: UnitConverter.convert junk value ---------------------------------------------
probe("P63 convert('abc')", lambda: __import__(
    "pharmaml.pharma.units", fromlist=["UnitConverter"]).UnitConverter.convert("abc", "mg", "g"))

# --- P64: filter_approved_drugs tuple input --------------------------------------------
from pharmaml.pharma.approvals import filter_approved_drugs
probe("P64 approvals tuple input", lambda: filter_approved_drugs(
    ("a.tsv", "b.tsv")))

# --- P65: nca on single point pair through Series with index ---------------------------
probe("P65 validate single point", lambda: validate_concentration_series([0.0], [1.0]))

# --- P66: effect_table empty grid --------------------------------------------------------
probe("P66 effect_table empty list", lambda: pdmod.effect_table([], 0.0, 100.0, 10.0))
