"""End-to-end headless probe: exercise every playground calculation path."""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import tkinter.filedialog as fd
import tkinter.messagebox as mb

import pandas as pd

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

app._ml_load_path(Path("examples/data/qsar_regression.csv"))
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

# -- docking ------------------------------------------------------------------
dock_out = Path(tempfile.mkdtemp(prefix="probe_dock_"))

app._dock_load_example()
dock_info = app._dock_lib_info.cget("text")
print("dock library:", dock_info)
assert "10 rows" in dock_info and "8 valid" in dock_info

app._dock_out_dir = dock_out
app._dock_prepare_ligands()
assert pump_until_idle(timeout=120), "ligand prep job hung"
prep_rows = app._dock_lib_tree.get_children()
assert len(prep_rows) == 10, f"ligand prep rows: {len(prep_rows)}"
prep_summary = app._dock_prep_summary.cget("text")
print("dock prep:", prep_summary)
assert "8/10" in prep_summary

app._dock_load_receptor_example()
orig_saveas = fd.asksaveasfilename
fd.asksaveasfilename = lambda **_k: str(dock_out / "receptor.pdbqt")
app._dock_prepare_receptor()
assert pump_until_idle(timeout=180), "receptor prep job hung"
fd.asksaveasfilename = orig_saveas
assert app._dock_protein_report.output_atoms > 1000
protein_text = app._dock_protein_text.get("1.0", "end")
print("dock receptor:", " | ".join(protein_text.splitlines()[:6]))
assert "1,262" in protein_text

aspirin = Path(app._dock_report.frame[app._dock_report.frame["ok"]].iloc[0]["pdbqt"])
app._dock_ligand_path = aspirin
app._dock_compute_grid()
assert pump_until_idle(), "grid box job hung"
assert app._dock_box.size == (22.125, 22.125, 22.125)
assert aspirin.with_name("r.gpf").is_file() and aspirin.with_name("r.dpf").is_file()
print("dock grid:", " | ".join(app._dock_grid_text.get("1.0", "end").splitlines()[:2]))

app._dock_load_example_campaign()
assert pump_until_idle(), "campaign scan job hung"
scan_rows = app._dock_results_tree.get_children()
assert len(scan_rows) == 14, f"campaign scan rows: {len(scan_rows)}"
scan_info = app._dock_results_info.cget("text")
print("dock campaign:", scan_info)
assert "14" in scan_info and "12 OK" in scan_info

app._dock_heavyatoms_entry.insert(0, "13")
app._dock_open_example_dlg()
assert pump_until_idle(), "DLG parse job hung"
dlg_rows = app._dock_results_tree.get_children()
assert len(dlg_rows) == 5, f"DLG clusters: {len(dlg_rows)}"
dlg_values = app._dock_results_tree.item(dlg_rows[0], "values")
assert dlg_values[0] == "1" and dlg_values[1] == "-4.9100", dlg_values
assert "LE (kcal/mol/atom)" in app._dock_results_tree["columns"]
assert app._dock_results_tree.item(dlg_rows[0], "values")[-1] == "0.3777"
print("dock dlg:", app._dock_results_info.cget("text"))

engines = {name: shutil.which(name) is not None
           for name in ("autogrid4", "autodock4", "vina")}
print("dock engines on PATH:", engines)

# -- feature factory (Chemistry tab) ---------------------------------------
app._feat_k_entry.delete(0, "end")
app._feat_k_entry.insert(0, "50")
app._feat_ranker_combo.set("Variance (no target needed)")
app._feat_compute()
assert pump_until_idle(timeout=120), "feature factory job hung"
feat_rows = app._feat_tree.get_children()
assert len(feat_rows) == 50, f"feature factory rows: {len(feat_rows)}"
feat_summary = app._feat_summary.cget("text")
print("feature factory:", feat_summary)
assert " features" in feat_summary and "3 molecules" in feat_summary
assert "variance" in feat_summary and "median-survivors" in feat_summary

feat_out = Path(tempfile.mkdtemp(prefix="probe_feat_"))
feat_csv = feat_out / "selected_features.csv"
fd.asksaveasfilename = lambda **_k: str(feat_csv)
app._feat_save_csv()
fd.asksaveasfilename = orig_saveas
assert feat_csv.is_file()
exported = pd.read_csv(feat_csv)
assert exported.shape == (3, 51), f"exported shape: {exported.shape}"
assert exported.columns[0] == "SMILES"
assert list(exported.columns[1:]) == list(app._feat_selection.selected)
print("feature factory csv:", exported.shape, "| first column", exported.columns[0])

feat_json = feat_out / "selection_metadata.json"
fd.asksaveasfilename = lambda **_k: str(feat_json)
app._feat_save_json()
fd.asksaveasfilename = orig_saveas
meta = json.loads(feat_json.read_text(encoding="utf-8"))
assert len(meta["selected_features"]) == 50
assert meta["total_features"] > 10_000
assert meta["k_requested"] == 50
print("feature factory json:", len(meta["selected_features"]), "selected |",
      meta["total_features"], "total features")
# the two export confirmations are the only dialogs so far; clear them so
# the final no-unexpected-dialogs assert stays meaningful
assert [dialog[0] for dialog in DIALOGS] == ["info", "info"]
assert all("written to" in " ".join(map(str, dialog[1])) for dialog in DIALOGS)
DIALOGS.clear()

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

# -- 100x-119 additions: NCA / infusion / TDM / PD family ------------- #
DIALOGS.clear()
app._nca_run()
assert pump_until_idle(timeout=120), "NCA job hung"
nca_text = app._nca_result.cget("text")
print("nca:", nca_text[:110])
assert "AUC" in nca_text and "MRT" in nca_text and "t½(term)" in nca_text, nca_text
assert len(app._nca_host.winfo_children()) == 1, "NCA figure not embedded"

app._nca_plot()
assert pump_until_idle(timeout=120), "NCA plot job hung"
assert len(app._nca_host.winfo_children()) == 1

app._infusion_math()
inf_text = app._inf_result.cget("text")
print("infusion:", inf_text[:110])
assert "rate for Css" in inf_text and "Vd =" in inf_text, inf_text

app._tdm_math()
tdm_text = app._tdm_result.cget("text")
print("tdm:", tdm_text)
assert "new dose" in tdm_text, tdm_text

app._pd_family_math()
assert pump_until_idle(timeout=120), "PD family job hung"
fam_rows = app._fam_tree.get_children()
print("pd family:", app._fam_result.cget("text")[:80], "| rows:", len(fam_rows))
assert len(fam_rows) == 6, f"PD table rows: {len(fam_rows)}"
assert len(app._fam_host.winfo_children()) == 1, "PD family figure not embedded"

app._pd_target_ec50()
fam_result = app._fam_result.cget("text")
print("pd reverse:", fam_result[:90])
assert "EC50 =" in fam_result, fam_result

app._plot_accumulation()
assert pump_until_idle(timeout=120), "accumulation plot job hung"
assert len(app._acc_host.winfo_children()) == 1, "accumulation figure not embedded"
print("accumulation figure: embedded")

# -- chemistry: tautomer / 3D / MACCS / heatmap ------------------------ #
app._tautomer_run()
taut_text = app._taut_result.cget("text")
print("tautomer:", taut_text[:60])
assert taut_text and taut_text != "—", "tautomer result empty"

app._maccs_run()
assert pump_until_idle(timeout=120), "MACCS job hung"
maccs_text = app._maccs_result.cget("text")
print("maccs:", maccs_text.splitlines()[0][:80])
assert "/167" in maccs_text, maccs_text

# matrix scan first (heatmap fuel), then the heatmap figure
app._scan_mode.set("matrix: references × library")
app._scan_run()
assert pump_until_idle(timeout=180), "matrix scan job hung"
app._scan_plot_heatmap()
assert pump_until_idle(timeout=120), "heatmap job hung"
assert len(app._scan_host.winfo_children()) == 1, "heat-map figure not embedded"
print("similarity heat-map: embedded")

# -- biology: replicate variability + compound summary ------------------ #
app._bio_replicate_variability()
assert pump_until_idle(timeout=120), "replicate variability job hung"
rep_text = app._bio_audit_text.get("1.0", "end").strip()
print("replicate variability head:", rep_text.splitlines()[0][:70])

app._bio_summarize_compounds()
assert pump_until_idle(timeout=120), "compound summary job hung"
sum_text = app._bio_audit_text.get("1.0", "end").strip()
print("compound summary head:", sum_text.splitlines()[0][:70])
assert "compound summary" in sum_text, sum_text[:80]

# -- docking: atom types + top ligands ---------------------------------- #
app._dock_plot_atom_types()
assert pump_until_idle(timeout=120), "atom types job hung"
assert len(app._dock_receptor_host.winfo_children()) == 1, "atom-type figure not embedded"
print("receptor atom types: embedded")

# top-ligand bars need a campaign-scan frame (folder-ranked) -- reload it
app._dock_load_example_campaign()
assert pump_until_idle(timeout=180), "campaign reload job hung"
app._dock_plot_top_ligands()
assert pump_until_idle(timeout=120), "top ligands job hung"
assert len(app._dock_results_host.winfo_children()) == 1, "top-ligand figure not embedded"
print("top ligands figure: embedded")

# -- ML: calibration curve + leaderboard bars ---------------------------- #
app._example_combo.set("qsar_classification.csv")
app._ml_load_example()
app._task_combo.set("classification")
app._ml_task_changed()
if "activity" in app._target_combo["values"]:
    app._target_combo.set("activity")
app._model_combo.set("logistic_regression")
app._ml_train()
assert pump_until_idle(timeout=300), "classification train job hung"
app._ml_plot_calibration()
assert pump_until_idle(timeout=180), "calibration job hung"
assert len(app._ml_host.winfo_children()) == 1, "calibration figure not embedded"
print("calibration curve: embedded")

app._ml_compare()
assert pump_until_idle(timeout=600), "compare job hung"
app._ml_plot_leaderboard()
assert pump_until_idle(timeout=120), "leaderboard plot job hung"
assert len(app._board_host.winfo_children()) == 1, "leaderboard figure not embedded"
print("leaderboard bars: embedded")

# -- quick-jump chips + tab cycling -------------------------------------- #
app._jump_to_tab(1)
assert app._book.index(app._book.select()) == 1, "quick-jump failed"
class _FakeEvent:
    keysym = "Next"
app._cycle_tab(_FakeEvent())
assert app._book.index(app._book.select()) == 2, "tab cycling failed"
app._jump_to_tab(0)
print("quick-jump + tab cycling: OK")

# -- 100x-120 additions: the last library stragglers + card finder ------- #
# learning curve on the trained classification model
app._ml_plot_learning_curve()
assert pump_until_idle(timeout=300), "learning-curve job hung"
assert len(app._ml_host.winfo_children()) == 1, "learning-curve figure not embedded"
assert "learning curve" in app._ml_result_note.cget("text")
print("learning curve: embedded")

# descriptor distributions on a feature-factory compute
app._feat_k_entry.delete(0, "end")
app._feat_k_entry.insert(0, "50")
app._feat_ranker_combo.set("Variance (no target needed)")
app._feat_compute()
assert pump_until_idle(timeout=600), "feature factory job hung"
app._feat_plot_distributions()
assert pump_until_idle(timeout=180), "distributions job hung"
assert len(app._feat_host.winfo_children()) == 1, "distribution figure not embedded"
print("descriptor distributions: embedded")

# MAD gene ranking on the example expression table
app._bio_load_expr_example()
app._bio_rank_variable_genes()
assert pump_until_idle(timeout=180), "MAD ranking job hung"
mad_summary = app._bio_mad_summary.cget("text")
print("mad:", mad_summary[:80])
assert "top 20 of 300 genes by MAD" in mad_summary, mad_summary

# the card finder: dialog lists every card; a jump flashes the shell
cards = app._live_cards()
assert len(cards) >= 55, f"live cards: {len(cards)}"
app._find_card_dialog()
import tkinter as _tk

_dialogs = [c for c in app.winfo_children()
            if isinstance(c, _tk.Toplevel) and c.title() == "Find a card"]
assert len(_dialogs) == 1, "finder dialog did not open"
_win = _dialogs[0]
_tree = next(c for c in _win.winfo_children()
             if c.winfo_class() == "Treeview")
assert len(_tree.get_children()) >= 55, "finder tree not populated"
_win.destroy()
app._jump_to_card(1, "Affinity conversion")
assert app._book.index(app._book.select()) == 1
_shell = None
for _t, _s, _c in playground._CARD_INDEX:
    if _t == "Affinity conversion":
        try:
            if bool(_c.winfo_exists()):
                _shell = _c
                break
        except _tk.TclError:  # a destroyed earlier-app shell
            continue
assert _shell is not None
assert int(_shell.cget("highlightthickness")) == 3, "flash border missing"
app._unflash_card(_shell)
print("card finder:", len(cards), "cards listed · jump + flash OK")

# single-folder engine: label + registered binaries
app._dock_engine_combo.set("AutoDock4 single folder")
app._dock_refresh_campaign_info()
assert "r.gpf" in app._dock_campaign_info.cget("text")
assert playground.docking_engine_binaries(
    "AutoDock4 single folder") == ("autogrid4", "autodock4")
app._dock_engine_combo.set("AutoDock4 campaign")
print("single-folder engine: registered, labelled")

# Vina config parse: keys dumped + box copied into the single-run form
_tmp = Path(tempfile.mkdtemp(prefix="probe120_"))
_conf = _tmp / "batch.conf"
_conf.write_text(
    "receptor = r.pdbqt\ncenter_x = 2.5\ncenter_y = 3.5\ncenter_z = 4.5\n"
    "size_x = 20\nsize_y = 22\nsize_z = 24\nexhaustiveness = 12\n",
    encoding="utf-8")
app._dock_vbatch_conf = _conf
app._dock_vbatch_parse()
_info = app._dock_vbatch_info.cget("text")
assert "parsed 8 keys" in _info, _info
assert app._dock_center_entry.get() == "2.5 3.5 4.5"
assert app._dock_exhaust_entry.get() == "12"
shutil.rmtree(_tmp, ignore_errors=True)
print("vina config parse: 8 keys · form filled")

# -- 100x-121 additions: UnitConverter, export_table, DL curve, memory -- #
# the unit converter card: default mass conversion + dimension swap
app._uc_convert()
_uc_text = app._uc_result.cget("text")
assert "500 mg" in _uc_text and "=  0.5 g" in _uc_text, _uc_text
assert "factor" in _uc_text and "mass conversion" in _uc_text
app._uc_dim_combo.set("concentration")
app._uc_dimension_changed()
assert app._uc_from.get() == "uM" and app._uc_to.get() == "nM"
app._uc_convert()
assert "=  500000 nM" in app._uc_result.cget("text"), app._uc_result.cget("text")
print("unit converter:", _uc_text.replace("\n", " | "))

# the unknown-unit error contract (one dialog fires, then we clear it)
app._uc_from.delete(0, "end")
app._uc_from.insert(0, "stones")
app._uc_convert()
assert len(DIALOGS) == 1 and DIALOGS[0][0] == "error", DIALOGS
DIALOGS.clear()
app._uc_from.delete(0, "end")
app._uc_from.insert(0, "mg")
app._uc_dim_combo.set("mass")
app._uc_dimension_changed()
print("unit converter: unknown unit named in a dialog")

# export_table: the five-format save dialog writes md/html/json/tsv
_frame = pd.DataFrame({"name": ["aspirin", "caffeine"],
                        "score": [1.5, 2.5]})
_orig_ask = fd.asksaveasfilename
_tmp121 = Path(tempfile.mkdtemp(prefix="probe121_"))
for _suffix in ("md", "html", "json", "tsv"):
    _target = _tmp121 / f"table.{_suffix}"
    fd.asksaveasfilename = lambda t="", _p=_target, **k: str(_p)
    app._save_frame_dialog(_frame, "Save it", "table.csv", "The table")
    assert _target.exists(), f"{_suffix} export missing"
    assert "aspirin" in _target.read_text(encoding="utf-8")
fd.asksaveasfilename = _orig_ask
shutil.rmtree(_tmp121, ignore_errors=True)
assert len(DIALOGS) == 4 and all(d[0] == "info" for d in DIALOGS), DIALOGS
DIALOGS.clear()
print("export_table: md/html/json/tsv all written")

# the DL learning curve (small network, 2-fold, 3 epochs)
app._dl_load_example()
app._dl_hidden_entry.delete(0, "end")
app._dl_hidden_entry.insert(0, "8")
app._dl_epochs.delete(0, "end")
app._dl_epochs.insert(0, "3")
app._dl_train()
assert pump_until_idle(timeout=600), "DL train job hung"
app._dl_cv_entry.delete(0, "end")
app._dl_cv_entry.insert(0, "2")
app._dl_plot_learning_curve()
assert pump_until_idle(timeout=600), "DL learning-curve job hung"
assert len(app._dl_curve_host.winfo_children()) == 1, "DL curve not embedded"
print("DL learning curve: embedded")

# the finder remembers its last query (session payload + prefill)
app._finder_last_query = "unit converter"
_payload121 = app._collect_session()
assert _payload121["texts"]["finder"] == "unit converter"
app._find_card_dialog()
_d121 = [c for c in app.winfo_children()
         if isinstance(c, _tk.Toplevel) and c.title() == "Find a card"]
assert len(_d121) == 1
_e121 = next(c for c in _d121[0].winfo_children() if c.winfo_class() == "TEntry")
assert _e121.get() == "unit converter", _e121.get()
_d121[0].destroy()
print("finder memory: session payload + prefilled dialog")

# the DLG cluster summary rides the single-folder verdict line
from types import SimpleNamespace as _NS

_tmp_dlg = Path(tempfile.mkdtemp(prefix="probe121dlg_"))
_dlg = _tmp_dlg / "r.dlg"
_dlg.write_text(
    "________\nCLUSTERING HISTOGRAM\n__________________\n"
    "______________________________________________________________________\n"
    "Cluster Rank | Lowest Energy | Run | Mean Energy | # in Cluster\n"
    "_____________|_______________|_____|_____________|________________\n"
    "     1       |     -11.5     |  3  |    -11.2    |      7 |\n"
    "     2       |     -10.8     |  9  |    -10.5    |     34 |\n"
    "     3       |      -9.9     | 17  |    -9.7    |      52 |\n"
    "____________|_______________|_____|_____________|________________\n",
    encoding="utf-8")
_captured = {}


def _fake_start_job(label, work, render):
    _captured["render"] = render


_orig_start_job = app._start_job
_orig_binaries = playground.docking_engine_binaries
playground.docking_engine_binaries = lambda engine: ()
app._start_job = _fake_start_job
app._dock_engine_combo.set("AutoDock4 single folder")
app._dock_campaign_root = _tmp_dlg
app._dock_run()
app._start_job = _orig_start_job
playground.docking_engine_binaries = _orig_binaries
_captured["render"](_NS(ok=True, error="", seconds=12.5, dlg=_dlg))
_summary121 = app._dock_run_summary.cget("text")
assert "3 clusters" in _summary121 and "top -11.50" in _summary121, _summary121
assert "best-populated -9.90 (n=52)" in _summary121, _summary121
shutil.rmtree(_tmp_dlg, ignore_errors=True)
app._dock_engine_combo.set("AutoDock4 campaign")
print("dlg summary:", _summary121[:90])

app.destroy()
assert not DIALOGS, f"unexpected dialogs: {DIALOGS}"
print("PROBE_OK")
