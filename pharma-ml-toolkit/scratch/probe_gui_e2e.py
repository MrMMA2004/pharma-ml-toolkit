"""End-to-end headless probe: exercise every playground calculation path."""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")
import tkinter.messagebox as mb

import playground

DIALOGS = []
mb.showerror = lambda *a, **k: DIALOGS.append(("error", a))
mb.showwarning = lambda *a, **k: DIALOGS.append(("warn", a))
mb.showinfo = lambda *a, **k: DIALOGS.append(("info", a))

app = playground.PlaygroundApp()


def pump_until_idle(timeout=300.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


# -- instant chemistry/PK callbacks ---------------------------------------
app._convert_affinity()
pic50 = app._aff_result.cget("text")
assert pic50 == "7.000", f"pIC50 wrong: {pic50!r}"

app._molar_mass()
mass = app._mass_result.cget("text")
assert "194.19" in mass, f"molar mass wrong: {mass!r}"

app._compare_similarity()
print("tanimoto:", app._tani_result.cget("text"),
      "|", app._tani_verdict.cget("text")[:40])

app._halflife_math()
print("halflife:", app._half_result.cget("text").replace("\n", " | "))

app._oral_pk()
print("oral:", app._oral_result.cget("text").replace("\n", " | "))

app._creatinine_math()
print("crcl:", app._cg_result.cget("text").replace("\n", " | "))

app._dose_design()
print("doses:", app._dose_result.cget("text").replace("\n", " | "))

app._bsa_math()
print("bsa:", app._bsa_result.cget("text").replace("\n", " | "))

app._pediatric_math()
print("peds:", app._ped_result.cget("text").replace("\n", " | "))

# -- threaded jobs ---------------------------------------------------------
app._analyse_smiles()
assert pump_until_idle(), "descriptor job hung"
desc_rows = app._desc_tree.get_children()
assert len(desc_rows) > 100, f"descriptor rows: {len(desc_rows)}"
print("descriptors:", len(desc_rows), "rows |", app._smiles_summary.cget("text")[:70])

app._plot_iv_bolus()
assert pump_until_idle(), "IV plot job hung"
assert len(app._iv_host.winfo_children()) == 1, "IV figure not embedded"
print("iv figure: embedded")

app._emax_math()
assert pump_until_idle(), "emax job hung"
assert len(app._pd_host.winfo_children()) == 1, "Emax figure not embedded"
print("emax:", app._pd_result.cget("text")[:64])

app._ml_load_path(Path("/home/z/my-project/pharma-ml-toolkit/examples/data/qsar_regression.csv"))
print("ml info:", app._ml_info.cget("text"))
assert "rows" in app._ml_info.cget("text")

app._ml_train()
assert pump_until_idle(timeout=300), "ML train job hung"
assert len(app._ml_metrics.winfo_children()) > 4, "metrics not rendered"
assert len(app._ml_host.winfo_children()) == 1, "parity figure not embedded"
print("ml train: metrics + parity figure OK |", app._target_combo.get())

app._ml_compare()
assert pump_until_idle(timeout=600), "compare job hung"
board_rows = app._board_tree.get_children()
assert len(board_rows) >= 3, f"leaderboard rows: {len(board_rows)}"
print("leaderboard:", len(board_rows), "models")

# -- cell lines --------------------------------------------------------------
app._bio_load_example()
audit = app._bio_audit_text.get("1.0", "end")
assert "duplicate pairs" in audit and "MCF-7 / crizotinib" in audit, audit[:120]
bio_info = app._bio_info.cget("text")
print("bio screen:", bio_info)
assert "185 rows" in bio_info and "23 cell lines" in bio_info
assert "erlotinib" in app._bio_compound_combo["values"]

app._bio_build_matrix()
assert pump_until_idle(), "matrix job hung"
matrix_rows = app._bio_matrix_tree.get_children()
assert len(matrix_rows) == 23, f"matrix rows: {len(matrix_rows)}"
assert app._bio_matrix.shape == (23, 8), app._bio_matrix.shape
print("bio matrix:", app._bio_matrix_info.cget("text"))

app._bio_ic50_entry.delete(0, "end")
app._bio_ic50_entry.insert(0, "3.2085197999117585")  # MCF-7 / erlotinib
app._bio_ln_to_pic50()
assert app._bio_ic50_result.cget("text") == "4.6066", app._bio_ic50_result.cget("text")
print("bio converter: LN_IC50 3.2085197999117585 -> pIC50",
      app._bio_ic50_result.cget("text"))

app._bio_compound_var.set("erlotinib")
app._bio_call_sensitive()
assert pump_until_idle(), "sensitive/resistant job hung"
print("bio calls:", app._bio_calls_summary.cget("text"))
assert len(app._bio_calls_tree.get_children()) == 23

app._bio_load_expr_example()
expr_info = app._bio_expr_info.cget("text")
print("bio expression:", expr_info)
assert "23 cell lines" in expr_info and "300 genes" in expr_info

app._bio_discover_biomarkers()
assert pump_until_idle(timeout=300), "biomarker job hung"
summary = app._bio_marker_summary.cget("text")
print("bio biomarkers:", summary)
assert "300 genes correlated" in summary and "1 significant" in summary
first = app._bio_marker_tree.get_children()[0]
values = app._bio_marker_tree.item(first, "values")
assert values[0] == "EGFR" and values[1] == "-0.9783", values
assert values[4] == "Yes" and "more sensitive" in values[5], values
print("bio top hit:", values[0], values[1], values[3], values[4], "|", values[5])

app._operator_combo.set("adaptive")
app._run_evolution()
assert pump_until_idle(timeout=300), "GA job hung"
print("ga summary:", app._ga_summary.cget("text"))
assert len(app._ga_tree.get_children()) >= 5, "leaderboard not filled"
stats = app._ga_stats_text.get("1.0", "end").strip()
assert "operator statistics" in stats, stats[:80]
assert len(app._ga_host.winfo_children()) == 1, "trajectory figure not embedded"
print("ga stats head:", stats.splitlines()[1] if len(stats.splitlines()) > 1 else stats)
print("ga trajectory: embedded")

app.destroy()
assert not DIALOGS, f"unexpected dialogs: {DIALOGS}"
print("PROBE_OK")
