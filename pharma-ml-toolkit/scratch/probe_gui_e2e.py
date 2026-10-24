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

# -- 100x-122 additions: reference objectives, surgery cards, PDF report -- #
# the three new objectives + the swapped reference row
app._objective_combo.set("Similarity to reference SMILES")
app._objective_hint()
assert app._obj_ref_label.cget("text") == "reference"
assert app._obj_ref_frame.grid_info() != {}
assert "Tanimoto similarity" in app._objective_hint_label.cget("text")
app._objective_combo.set("Contains substructure (SMARTS)")
app._objective_hint()
assert app._obj_ref_label.cget("text") == "SMARTS"
app._objective_combo.set("Novelty vs corpus")
app._objective_hint()
assert app._obj_ref_frame.grid_info() == {}
app._objective_combo.set("QED drug-likeness")
app._objective_hint()
print("objectives: reference row swaps for similarity/SMARTS, hides for novelty")

# the ring-topology moves card: piperidine's three mutants
app._gen_topology_moves()
assert pump_until_idle(timeout=240), "topology job hung"
_topo = app._topo_text.get("1.0", "end").strip()
assert "C1CCCC1" in _topo and "C1CCCNCC1" in _topo and "C1CCNC1" in _topo, _topo
assert app._topo_summary.cget("text").startswith("3 distinct")
print("ring topology moves:", _topo.replace("\n", " · "))

# the breed card: aspirin x ibuprofen, 4 seeded children
app._gen_breed()
assert pump_until_idle(timeout=240), "breed job hung"
_litter = app._breed_text.get("1.0", "end").strip().splitlines()
assert len(_litter) == 4, _litter
assert "viable children" in app._breed_summary.cget("text")
print("breed:", app._breed_summary.cget("text")[:80])

# the session PDF report: guard first (the registry is emptied AND every
# filled table is emptied -- the guard covers the report's two axes
# since 100x-123), then a real write with BOTH axes
app._session_figures.clear()


def _empty_filled_tables(widget):
    from tkinter import ttk as _ttk

    for child in widget.winfo_children():
        if isinstance(child, _ttk.Treeview) and len(child.get_children()):
            child.delete(*child.get_children())
        _empty_filled_tables(child)


_empty_filled_tables(app)
assert app._live_session_tables() == [], "probe failed to empty every table"
app._menu_save_session_report()
assert len(DIALOGS) == 1 and DIALOGS[0][0] == "info", DIALOGS
assert "Nothing to report yet" in str(DIALOGS[0][1]), DIALOGS
DIALOGS.clear()

# (100x-123) the surgery structure grids: enumerate -> draw, breed -> draw
app._gen_topology_moves()
assert pump_until_idle(timeout=240), "topology job hung"
assert app._topo_last[1], "no mutants enumerated"
app._gen_draw_topology()
assert pump_until_idle(timeout=240), "topology draw job hung"
assert getattr(app._topo_host, "_pg_figure", None) is not None, "no topo grid"
app._gen_breed()
assert pump_until_idle(timeout=240), "breed job hung"
app._gen_draw_breed()
assert pump_until_idle(timeout=240), "breed draw job hung"
assert getattr(app._breed_host, "_pg_figure", None) is not None, "no litter grid"
# the draw guards: without a last result the honest info fires
_saved_topo, app._topo_last = app._topo_last, None
app._gen_draw_topology()
assert any("No enumeration yet" in str(a) for a in DIALOGS), DIALOGS
app._topo_last = _saved_topo
DIALOGS.clear()

# the table axis: a REAL job fills the descriptor table (200+ rows)
app._analyse_smiles()
assert pump_until_idle(timeout=240), "analyse job hung"
_walked = app._live_session_tables()
assert any(t == "SMILES inspector" and len(r) > 150 for t, _h, r in _walked), _walked[:2]

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as _plt

_figA = _plt.figure()
_axA = _figA.add_subplot(111)
_axA.plot([0, 1], [0, 1])
_axA.set_title("alpha")
_figB = _plt.figure()
_axB = _figB.add_subplot(111)
_axB.plot([1, 0], [0, 1])
_axB.set_title("beta")
app._job_label_base = "figure alpha"
app._render_figure(app._ml_host, _figA)
app._job_label_base = "figure beta"
app._render_figure(app._feat_host, _figB)
# (100x-123) the surgery grids drawn earlier in this probe are ALSO
# live session figures now: topo + breed + ml + feat = 4 hosts
assert len(app._live_session_figures()) == 4
_figC = _plt.figure()
_figC.add_subplot(111).plot([0, 1], [1, 0])
app._render_figure(app._ml_host, _figC)  # replace in place
assert len(app._live_session_figures()) == 4, "registry must stay one-per-host"
_ml_entry = next(e for e in app._session_figures if e[0] is app._ml_host)
assert _ml_entry[2] is _figC

_tmp122 = Path(tempfile.mkdtemp(prefix="probe122_"))
_target122 = _tmp122 / "session-report.pdf"
fd.asksaveasfilename = lambda t="", _p=_target122, **k: str(_p)
app._menu_save_session_report()
assert pump_until_idle(timeout=240), "report job hung"
assert _target122.exists() and _target122.stat().st_size > 1_000
assert app._ml_host._pg_figure is _figC, "close=False must keep live figures"
assert len(DIALOGS) == 1 and "figure page(s)" in str(DIALOGS[0][1]), DIALOGS
assert "table page(s)" in str(DIALOGS[0][1]), DIALOGS
DIALOGS.clear()
import pypdf as _pypdf

_reader = _pypdf.PdfReader(str(_target122))
_pages = [(_page.extract_text() or "").lower() for _page in _reader.pages]
_joined = "\n".join(_pages)
assert "table page(s)" in _joined, "subtitle did not count tables"
assert "smiles inspector table" in _joined, "the table page did not ship"
assert "molwt" in _joined, "walked rows did not ship"
assert "more rows" in _joined, "the truncation note did not ship"
assert len(_reader.pages) >= 5, "expected title + 2 figures + 1 table at least"
shutil.rmtree(_tmp122, ignore_errors=True)
_plt.close("all")
print("session report: guard + registry replace + real PDF written")

# -- 100x-124 additions: the GA top-molecule grid + the HTML report -- #

# the leaderboard's scored structure grid: a real evolution run first
app._objective_combo.set("QED drug-likeness")
app._pop_size.set("8")
app._generations.set("2")
app._run_evolution()
assert pump_until_idle(timeout=300), "evolution job hung"
assert app._ga_result is not None
app._gen_draw_top()
assert pump_until_idle(timeout=300), "draw-top job hung"
_grid = getattr(app._ga_grid_host, "_pg_figure", None)
assert _grid is not None, "the top-molecule grid never landed"
_labels = [a.get_xlabel() for a in _grid.get_axes()]
assert any(l.startswith("score ") for l in _labels), _labels[:4]
_titles = [a.get_title() for a in _grid.get_axes()]
assert any(t.startswith("#1") for t in _titles), _titles[:4]
# the run's trajectory host + the grid host both join the session
# registry (4 -> 6 live figures: adaptive default plots the operator
# mix into _ga_host, the grid lands on _ga_grid_host)
assert len(app._live_session_figures()) == 6, len(app._live_session_figures())
# the guard: no run yet -> the honest info
_saved_ga, app._ga_result = app._ga_result, None
app._gen_draw_top()
assert any("No run yet" in str(a) for a in DIALOGS), DIALOGS
app._ga_result = _saved_ga
DIALOGS.clear()
print("GA top grid: run + draw + registry + guard OK")

# the HTML session report: the SAME two axes, one self-contained file
_tmp124 = Path(tempfile.mkdtemp(prefix="probe124_"))
_target124 = _tmp124 / "session-report.html"
fd.asksaveasfilename = lambda t="", _p=_target124, **k: str(_p)
app._menu_save_session_report_html()
assert pump_until_idle(timeout=240), "HTML report job hung"
assert _target124.exists() and _target124.stat().st_size > 50_000
_html = _target124.read_text(encoding="utf-8")
assert "<!DOCTYPE html>" in _html
assert "data:image" in _html, "figures were not embedded"
assert "<table" in _html, "tables missing"
assert "cards on screen" in _html, "metric cards missing"
assert "What this file contains" in _html
assert any("figure(s) + " in str(a) for a in DIALOGS), DIALOGS
DIALOGS.clear()
# the guard path: both axes empty -> the honest info
app._session_figures.clear()
_empty_filled_tables(app)
app._menu_save_session_report_html()
assert any("Nothing to report yet" in str(a) for a in DIALOGS), DIALOGS
DIALOGS.clear()
shutil.rmtree(_tmp124, ignore_errors=True)
_plt.close("all")
print("HTML session report: real write + guard OK")

# -- 100x-125 additions: the Pareto front's colour-coded grid ------- #
# a real multi-objective run, then the front card's draw-front button
app._jump_to_tab(7)
app._objective_combo.set("Multi-objective (QED × MW × TPSA)")
app._pop_size.set("12")
app._generations.set("4")
app._run_evolution()
assert pump_until_idle(timeout=240), "multi-objective evolution hung"
# NB: earlier sections leave mutation/crossover/operator state behind,
# so the front size is NOT the fresh-app pin here -- the colour split is
# derived from THIS front's own crowding verdict (the honest contract)
import math as _math

_front = app._pareto_front
assert _front is not None and len(_front) >= 2, "the front never filled"
_crowding = _front.crowding_distance()
_shown = min(len(_front.smiles), 12)
_expected_teal = sum(
    1 for _i in range(_shown) if _math.isinf(_crowding[_i])
)
_expected_neutral = _shown - _expected_teal
_registry_before = len(app._session_figures)
app._gen_draw_front()
assert pump_until_idle(timeout=240), "draw-front job hung"
_grid = getattr(app._pareto_host, "_pg_figure", None)
assert _grid is not None, "the front structure grid never landed"
_title_colors = [axis.title.get_color() for axis in _grid.get_axes()]
assert _title_colors.count("#0F766E") == _expected_teal, (
    _title_colors,
    _expected_teal,
)  # boundary corners
assert _title_colors.count("#525252") == _expected_neutral, (
    _title_colors,
    _expected_neutral,
)  # interior members
_labels = [axis.get_xlabel() for axis in _grid.get_axes()]
assert sum(label.startswith("qed ") for label in _labels) == _shown, _labels[:3]
# the grid replaced the scatter's registry entry IN PLACE (the registry
# is host-keyed) -- the front card's figure stays reportable content
_entry = next(
    (e for e in app._session_figures if e[0] is app._pareto_host), None
)
assert _entry is not None and _entry[2] is _grid, "the grid never registered"
assert _entry[1] == "Pareto front structure grid", _entry[1]
assert any(
    e[0] is app._pareto_host for e in app._live_session_figures()
), "the front grid is not live"
# Plot front restores the scatter (the replace/restore pair)
app._plot_pareto_front()
assert app._pareto_host._pg_figure is not None
_scatter_labels = [
    axis.get_xlabel() for axis in app._pareto_host._pg_figure.get_axes()
]
assert not any(label.startswith("qed ") for label in _scatter_labels)
# the guard: no front -> the honest info
_saved_front, app._pareto_front = app._pareto_front, None
_saved_result, app._pareto_result = app._pareto_result, None
app._gen_draw_front()
assert any("Nothing to draw" in str(a) for a in DIALOGS), DIALOGS
app._pareto_front, app._pareto_result = _saved_front, _saved_result
DIALOGS.clear()
_plt.close("all")
print("Pareto front grid: run + colours + registry + restore + guard OK")

# the live-card census grew with the two surgery cards
assert len(app._live_cards()) >= 68, f"live cards: {len(app._live_cards())}"
print("card census:", len(app._live_cards()))

# -- 100x-126 additions: the per-card save + the themed HTML report --- #
# the front card's per-card export saves the figure THE CARD shows even
# when another figure drew last somewhere else
app._jump_to_tab(7)
app._gen_draw_front()  # put the grid back on the host
assert pump_until_idle(timeout=240), "draw-front job hung"
_grid126 = getattr(app._pareto_host, "_pg_figure", None)
assert _grid126 is not None, "the front grid never landed"
import pharmaml.viz as _viz126

_real_save126 = _viz126.save_figure
_captured126: dict = {}


def _spy126(figure, path, *args, **kwargs):
    _captured126["figure"] = figure
    return _real_save126(figure, path, *args, **kwargs)


_viz126.save_figure = _spy126
try:
    import pathlib as _pl126

    _export126 = _pl126.Path("/tmp/probe126-front-export.png")
    playground.filedialog.asksaveasfilename = lambda **_k: str(_export126)
    _decoy126, _decoy_axis126 = _plt.subplots()
    _decoy_axis126.plot([0, 1], [1, 0])
    app._last_figure = _decoy126  # another figure "drew last" elsewhere
    app._save_pareto_figure()
    assert _captured126["figure"] is _grid126, "saved a figure that is not the card's"
    assert _captured126["figure"] is not _decoy126
    assert _export126.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert any("Figure saved" in str(a) for a in DIALOGS), DIALOGS
    _plt.close(_decoy126)
    DIALOGS.clear()
finally:
    _viz126.save_figure = _real_save126
# the guard: nothing on the host -> the honest info
app._pareto_host._pg_figure = None
app._save_pareto_figure()
assert any("No figure on the card yet" in str(a) for a in DIALOGS), DIALOGS
app._pareto_host._pg_figure = _grid126  # restore for the report below
DIALOGS.clear()
print("Pareto per-card save: card's figure over the decoy + guard OK")

# the HTML report OPENS in the app's theme and carries the toggle
app._theme_var.set("dark")
_report126 = "/tmp/probe126-report.html"
playground.filedialog.asksaveasfilename = lambda **_k: _report126
app._menu_save_session_report_html()
assert pump_until_idle(timeout=240), "the HTML report job hung"
with open(_report126, encoding="utf-8") as _fh126:
    _text126 = _fh126.read()
assert "<html lang='en' data-theme='dark'>" in _text126
assert "id='theme-toggle'" in _text126 and "localStorage" in _text126
assert "pharmaml-report-theme" in _text126
assert _text126.count("<script>") == 1
assert "data:image/png;base64," in _text126
assert "http://" not in _text126 and "https://" not in _text126
assert any("cycles it per reader" in str(a) for a in DIALOGS), DIALOGS
DIALOGS.clear()
app._theme_var.set("auto")  # leave the app as we found it
print("Themed HTML report: app theme baked + toggle + contract OK")

# -- 100x-127 additions: the per-card export on EVERY Generative figure
# card (two-host Results semantics + both surgery cards) and the
# REPORT_THEMES bake guard ------------------------------------------ #
# the ring-topology card: enumerate -> draw -> the CARD's own export
app._topo_smiles.delete(0, "end")
app._topo_smiles.insert(0, "C1CCNCC1")  # piperidine: 3 mutants
app._gen_topology_moves()
assert pump_until_idle(timeout=240), "topology job hung"
app._gen_draw_topology()
assert pump_until_idle(timeout=240), "draw-topology job hung"
_topo_grid127 = getattr(app._topo_host, "_pg_figure", None)
assert _topo_grid127 is not None, "the mutant grid never landed"
_real_save127 = _viz126.save_figure
_captured127: dict = {}


def _spy127(figure, path, *args, **kwargs):
    _captured127["figure"] = figure
    return _real_save127(figure, path, *args, **kwargs)


_viz126.save_figure = _spy127
try:
    _export_topo = "/tmp/probe127-topo-export.png"
    playground.filedialog.asksaveasfilename = lambda **_k: _export_topo
    _decoy127, _decoy_axis127 = _plt.subplots()
    _decoy_axis127.plot([0, 1], [1, 0])
    app._last_figure = _decoy127
    app._save_topo_figure()
    assert pump_until_idle(timeout=240), "topo save job hung"
    assert _captured127["figure"] is _topo_grid127, "topo: not the card's figure"
    assert _captured127["figure"] is not _decoy127
    with open(_export_topo, "rb") as _fh127:
        assert _fh127.read(8) == b"\x89PNG\r\n\x1a\n"
    assert any("Figure saved" in str(a) for a in DIALOGS), DIALOGS
    _plt.close(_decoy127)
    DIALOGS.clear()
finally:
    _viz126.save_figure = _real_save127
# the guard: breed card has no figure yet -> the honest info
app._breed_host._pg_figure = None
app._save_breed_figure()
assert any("No figure on the card yet" in str(a) for a in DIALOGS), DIALOGS
DIALOGS.clear()
print("Surgery per-card saves: topo semantics over decoy + breed guard OK")

# the Results card's TWO hosts: the save follows the newest of the
# card's own figures (grid vs history), never the global last-drawn
_hist127, _hist_axis127 = _plt.subplots()
_hist_axis127.set_title("history")
_grid127, _grid_axis127 = _plt.subplots()
_grid_axis127.set_title("grid")
_again127, _again_axis127 = _plt.subplots()
_again_axis127.set_title("history again")
try:
    app._render_figure(app._ga_host, _hist127)
    app._render_figure(app._ga_grid_host, _grid127)
    # stamps: the render counter is monotonic and rides every host
    assert (
        getattr(app._ga_host, "_pg_figure_seq", 0)
        < getattr(app._ga_grid_host, "_pg_figure_seq", 0)
    ), "render stamps are not ordered"
    _viz126.save_figure = _spy127
    try:
        _export_ga = "/tmp/probe127-ga-export.png"
        playground.filedialog.asksaveasfilename = lambda **_k: _export_ga
        app._last_figure = _decoy127  # a closed decoy: identity is what matters
        app._save_ga_figure()
        assert pump_until_idle(timeout=240), "ga save job hung"
        assert _captured127["figure"] is _grid127, "newest card figure lost"
        app._render_figure(app._ga_host, _again127)  # history re-drew last
        app._save_ga_figure()
        assert pump_until_idle(timeout=240), "ga save job hung"
        assert _captured127["figure"] is _again127, "newest card figure lost"
        with open(_export_ga, "rb") as _fh127:
            assert _fh127.read(8) == b"\x89PNG\r\n\x1a\n"
        assert any("Figure saved" in str(a) for a in DIALOGS), DIALOGS
        DIALOGS.clear()
    finally:
        _viz126.save_figure = _real_save127
    # the guard: both hosts empty -> the honest info
    _saved_ga_grid = app._ga_grid_host._pg_figure
    _saved_ga_host = app._ga_host._pg_figure
    app._ga_grid_host._pg_figure = None
    app._ga_host._pg_figure = None
    app._save_ga_figure()
    assert any("No figure on the card yet" in str(a) for a in DIALOGS), DIALOGS
    app._ga_grid_host._pg_figure = _saved_ga_grid
    app._ga_host._pg_figure = _saved_ga_host
    DIALOGS.clear()
finally:
    _plt.close("all")
print("Results two-host save: newest-of-card + stamps + guard OK")

# the four buttons ride the Generative tab (one per figure card)
_buttons127 = [
    str(w.cget("text"))
    for w in app._walk_widgets(app._book.nametowidget(app._book.tabs()[7]))
    if w.winfo_class() == "TButton"
]
assert _buttons127.count("Save figure…") == 4, _buttons127.count("Save figure…")

# the HTML report bake guards the library's REPORT_THEMES contract: a
# drifted theme value falls back to auto instead of raising mid-job
app._theme_var.set("neon")
_report127 = "/tmp/probe127-report.html"
playground.filedialog.asksaveasfilename = lambda **_k: _report127
app._menu_save_session_report_html()
assert pump_until_idle(timeout=240), "the guarded HTML report job hung"
with open(_report127, encoding="utf-8") as _fh127:
    _text127 = _fh127.read()
assert "<html lang='en' data-theme='auto'>" in _text127, "guard did not fall back to auto"
assert "data:image/png;base64," in _text127
app._theme_var.set("auto")
DIALOGS.clear()
print("REPORT_THEMES bake guard: drifted value -> auto OK")

# -- 100x-128 additions: the per-card export closes APP-WIDE (the 15
# remaining figure cards) -------------------------------------------- #
# a REAL end-to-end path: the IV bolus card's own Plot profile button
# draws the decay curve, then the card's own export saves THAT figure
# while a decoy holds the global last-drawn slot
app._iv_c0.delete(0, "end"); app._iv_c0.insert(0, "10")
app._iv_thalf.delete(0, "end"); app._iv_thalf.insert(0, "4")
app._iv_hours.delete(0, "end"); app._iv_hours.insert(0, "24")
app._plot_iv_bolus()
assert pump_until_idle(timeout=240), "IV profile job hung"
_iv_fig128 = getattr(app._iv_host, "_pg_figure", None)
assert _iv_fig128 is not None, "the IV decay curve never landed"
_real_save128 = _viz126.save_figure
_captured128: dict = {}


def _spy128(figure, path, *args, **kwargs):
    _captured128["figure"] = figure
    return _real_save128(figure, path, *args, **kwargs)


_viz126.save_figure = _spy128
try:
    _export_iv = "/tmp/probe128-iv-export.png"
    playground.filedialog.asksaveasfilename = lambda **_k: _export_iv
    _decoy128, _decoy_axis128 = _plt.subplots()
    _decoy_axis128.plot([0, 1], [1, 0])
    app._last_figure = _decoy128
    app._save_iv_figure()
    assert pump_until_idle(timeout=240), "IV save job hung"
    assert _captured128["figure"] is _iv_fig128, "IV: not the card's figure"
    assert _captured128["figure"] is not _decoy128
    with open(_export_iv, "rb") as _fh128:
        assert _fh128.read(8) == b"\x89PNG\r\n\x1a\n"
    assert any("Figure saved" in str(a) for a in DIALOGS), DIALOGS
    _plt.close(_decoy128)
    DIALOGS.clear()
finally:
    _viz126.save_figure = _real_save128
print("IV per-card save: real plot -> card's own figure over decoy OK")

# the two-host DL history card: newest of the card's OWN figures wins
_losses128, _l_axis128 = _plt.subplots()
_l_axis128.set_title("loss curves")
_curve128, _c_axis128 = _plt.subplots()
_c_axis128.set_title("learning curve")
_again128, _a_axis128 = _plt.subplots()
_a_axis128.set_title("loss curves again")
try:
    app._render_figure(app._dl_hist_host, _losses128)
    app._render_figure(app._dl_curve_host, _curve128)
    assert (
        getattr(app._dl_hist_host, "_pg_figure_seq", 0)
        < getattr(app._dl_curve_host, "_pg_figure_seq", 0)
    ), "DL render stamps are not ordered"
    _viz126.save_figure = _spy128
    try:
        _export_dl = "/tmp/probe128-dl-export.png"
        playground.filedialog.asksaveasfilename = lambda **_k: _export_dl
        app._save_dl_history_figure()
        assert pump_until_idle(timeout=240), "DL history save job hung"
        assert _captured128["figure"] is _curve128, "newest DL card figure lost"
        app._render_figure(app._dl_hist_host, _again128)  # losses re-drew last
        app._save_dl_history_figure()
        assert pump_until_idle(timeout=240), "second DL history save job hung"
        assert _captured128["figure"] is _again128, "newest DL card figure lost"
        with open(_export_dl, "rb") as _fh128:
            assert _fh128.read(8) == b"\x89PNG\r\n\x1a\n"
        assert any("Figure saved" in str(a) for a in DIALOGS), DIALOGS
        DIALOGS.clear()
    finally:
        _viz126.save_figure = _real_save128
finally:
    _plt.close("all")
print("DL history two-host save: newest-of-card + stamps OK")

# the honest guards across a representative spread of the new cards
for _method128 in (
    "_save_feat_figure",
    "_save_scan_figure",
    "_save_pd_figure",
    "_save_nca_figure",
    "_save_acc_figure",
    "_save_fam_figure",
    "_save_dock_receptor_figure",
    "_save_dock_results_figure",
    "_save_ml_figure",
    "_save_board_figure",
    "_save_ad_figure",
    "_save_dl_figure",
    "_save_util_cmp_figure",
):
    for _host128 in (
        "_feat_host", "_scan_host", "_pd_host", "_nca_host", "_acc_host",
        "_fam_host", "_dock_receptor_host", "_dock_results_host",
        "_ml_host", "_board_host", "_ad_host", "_dl_host", "_util_cmp_host",
    ):
        _host_obj128 = getattr(app, _host128, None)
        if _host_obj128 is not None and getattr(_host_obj128, "_pg_figure", None) is not None:
            _host_obj128._pg_figure = None
    getattr(app, _method128)()
    assert any("No figure on the card yet" in str(a) for a in DIALOGS), (_method128, DIALOGS)
    DIALOGS.clear()
print("Per-card guards: 13 representative new cards OK")

# the button census: one Save figure… per figure card on EVERY tab
for _idx128, _want128 in ((1, 2), (2, 5), (4, 2), (5, 3), (6, 2), (8, 1), (7, 4)):
    _tab_buttons128 = [
        str(w.cget("text"))
        for w in app._walk_widgets(app._book.nametowidget(app._book.tabs()[_idx128]))
        if w.winfo_class() == "TButton"
    ]
    assert _tab_buttons128.count("Save figure…") == _want128, (
        _idx128, _want128, _tab_buttons128.count("Save figure…")
    )
print("Button census: 19 Save figure… buttons app-wide (15 new + 4 Generative)")

# the live-card census: the new buttons joined EXISTING cards (the DL
# results card's new row is a frame INSIDE the card, not a new card)
assert len(app._live_cards()) >= 68, f"live cards: {len(app._live_cards())}"
print("card census:", len(app._live_cards()))

# -- 100x-129 additions: the PDF session report's dark-ink variant,
# riding the View > theme radios (light stays byte-classic) ---------- #
import re as _re129
import zlib as _zlib129


def _pdf_streams129(pdf_bytes: bytes) -> bytes:
    """Decompress every page-content stream (fpdf2 zlib-compresses)."""
    out: list[bytes] = []
    for match in _re129.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, _re129.DOTALL):
        try:
            out.append(_zlib129.decompress(match.group(1)))
        except _zlib129.error:
            pass
    return b"\n".join(out)


# a fresh live figure for the report walk (the 128 section closed all)
app._iv_c0.delete(0, "end"); app._iv_c0.insert(0, "12")
app._iv_thalf.delete(0, "end"); app._iv_thalf.insert(0, "6")
app._iv_hours.delete(0, "end"); app._iv_hours.insert(0, "24")
app._plot_iv_bolus()
assert pump_until_idle(timeout=240), "IV profile job (129) hung"
assert getattr(app._iv_host, "_pg_figure", None) is not None, "no live figure (129)"
app._desc_tree.insert("", "end", values=("MolWt", "194.19"))

# DARK: the radio picks the variant; the pages carry the palette
app._theme_var.set("dark")
app._on_theme_change()
assert app._theme == "dark"
_dark_pdf129 = "/tmp/probe129-session-dark.pdf"
playground.filedialog.asksaveasfilename = lambda **_k: _dark_pdf129
app._menu_save_session_report()
assert pump_until_idle(timeout=240), "dark session report job hung"
with open(_dark_pdf129, "rb") as _fh129:
    _dark_bytes129 = _fh129.read()
assert _dark_bytes129[:5] == b"%PDF-", "not a PDF"
_dark_ops129 = _pdf_streams129(_dark_bytes129)
assert b"0.0824 0.102 0.0941 rg" in _dark_ops129, "dark pages never filled"
assert b"1 1 1 rg" in _dark_ops129, "the figure plate never landed"
assert b"0.9059 0.9255 0.9137 rg" in _dark_ops129, "light body ink never landed"
assert any("dark ink variant" in str(a) for a in DIALOGS), DIALOGS
assert any("View" in str(a) for a in DIALOGS), "the flip hint never fired"
DIALOGS.clear()
print("Dark-ink session report: palette + plate + dialog hint OK")

# LIGHT: the classic paper carries NO colour operators at all
app._theme_var.set("light")
app._on_theme_change()
_light_pdf129 = "/tmp/probe129-session-light.pdf"
playground.filedialog.asksaveasfilename = lambda **_k: _light_pdf129
app._menu_save_session_report()
assert pump_until_idle(timeout=240), "light session report job hung"
with open(_light_pdf129, "rb") as _fh129:
    _light_bytes129 = _fh129.read()
assert _light_bytes129[:5] == b"%PDF-", "not a PDF"
assert _light_bytes129 != _dark_bytes129, "the variant changed nothing"
assert b" rg" not in _pdf_streams129(_light_bytes129), "light paper gained colour ops"
assert any("classic light paper" in str(a) for a in DIALOGS), DIALOGS
DIALOGS.clear()
print("Classic light paper: byte-classic, zero colour ops OK")

# AUTO resolves BEFORE the ink freezes (no reading-time toggle)
app._theme_var.set("auto")
app._resolve_auto_theme = lambda: "dark"
_auto_pdf129 = "/tmp/probe129-session-auto.pdf"
playground.filedialog.asksaveasfilename = lambda **_k: _auto_pdf129
app._menu_save_session_report()
assert pump_until_idle(timeout=240), "auto session report job hung"
with open(_auto_pdf129, "rb") as _fh129:
    _auto_ops129 = _pdf_streams129(_fh129.read())
assert b"0.0824 0.102 0.0941 rg" in _auto_ops129, "auto did not resolve to dark ink"
app._theme_var.set("light")
app._on_theme_change()
DIALOGS.clear()
print("Auto ride: resolved through _resolve_auto_theme at save time OK")


# -- 100x-130 additions: the Ctrl+F finder's "Recently jumped" row ---- #
import tkinter as _tk130

# three real jumps -> the memory is most-recent-first and de-duplicated
app._jump_to_card(1, "Affinity conversion")
app._jump_to_card(0, "PharmaML Playground")
app._jump_to_card(1, "Affinity conversion")
assert app._finder_recent == ["Affinity conversion", "PharmaML Playground"], (
    app._finder_recent
)
print("Recency memory: most-recent-first + de-duplication OK")

# the dialog: chips for the remembered cards, none for a bogus title
app._finder_recent = ["Affinity conversion", "Molar mass", "no such card"]
app._find_card_dialog()
_dialogs130 = [
    child
    for child in app.winfo_children()
    if isinstance(child, _tk130.Toplevel) and child.title() == "Find a card"
]
assert len(_dialogs130) == 1, "finder dialog did not open"
_win130 = _dialogs130[0]
_buttons130: list = []


def _walk_buttons130(widget):
    for child in widget.winfo_children():
        if child.winfo_class() == "TButton":
            _buttons130.append(child)
        _walk_buttons130(child)


_walk_buttons130(_win130)
_chip_labels130 = [str(b.cget("text")) for b in _buttons130]
assert "Affinity conversion" in _chip_labels130, _chip_labels130
assert "Molar mass" in _chip_labels130, _chip_labels130
assert not any("no such card" in label for label in _chip_labels130), (
    "a non-live title leaked into the chip row"
)
print("Chip row: live titles only, honest omission OK")

# a chip click jumps AND closes the dialog AND re-remembers
_chips130 = {str(b.cget("text")): b for b in _buttons130}
_chips130["Molar mass"].invoke()
app.update()
assert not bool(_win130.winfo_exists()), "chip click left the dialog open"
assert app._finder_recent[0] == "Molar mass", app._finder_recent
print("Chip click: jump + close + re-remember OK")

# the session round trip carries the memory as JSON
import json as _json130

_payload130 = app._collect_session()
_recent_raw130 = _payload130["texts"]["finder_recent"]
assert _json130.loads(_recent_raw130)[0] == "Molar mass", _recent_raw130
app._finder_recent = ["SMILES inspector"]
app._session = {
    "theme": "light",
    "geometry": "",
    "tab": 0,
    "variables": {},
    "families": {},
    "texts": {"finder_recent": _recent_raw130},
}
app._restore_session()
assert app._finder_recent[0] == "Molar mass", app._finder_recent
print("Session persistence: JSON round trip OK")

# an empty memory renders no chip row at all
app._finder_recent = []
app._find_card_dialog()
_dialogs130b = [
    child
    for child in app.winfo_children()
    if isinstance(child, _tk130.Toplevel) and child.title() == "Find a card"
]
_win130b = _dialogs130b[0]
_labels130: list = []


def _walk_labels130(widget):
    for child in widget.winfo_children():
        if child.winfo_class() == "TLabel":
            _labels130.append(str(child.cget("text")))
        _walk_labels130(child)


_walk_labels130(_win130b)
assert not any(label.startswith("Recently jumped") for label in _labels130)
_win130b.destroy()
print("Empty memory: no chip row OK")

# -- 100x-131 additions: the finder row's "Clear" affordance -------- #
import tkinter as _tk131

# seed the memory, open the dialog, and Clear must ride the row
app._jump_to_card(1, "Affinity conversion")
app._jump_to_card(0, "PharmaML Playground")
app._find_card_dialog()
_dialogs131 = [
    child
    for child in app.winfo_children()
    if isinstance(child, _tk131.Toplevel) and child.title() == "Find a card"
]
assert len(_dialogs131) == 1, "finder dialog did not open"
_win131 = _dialogs131[0]
_buttons131: list = []


def _walk_buttons131(widget):
    for child in widget.winfo_children():
        if child.winfo_class() == "TButton":
            _buttons131.append(child)
        _walk_buttons131(child)


_walk_buttons131(_win131)
_clear131 = {str(b.cget("text")): b for b in _buttons131}.get("Clear")
assert _clear131 is not None, "the Clear button is missing from the row"
print("Clear affordance: rides the recency row OK")

# one click: memory emptied, row collapsed in place, session follows
_clear131.invoke()
app.update()
assert app._finder_recent == [], app._finder_recent
_labels131: list = []


def _walk_labels131(widget):
    for child in widget.winfo_children():
        if child.winfo_class() == "TLabel":
            _labels131.append(str(child.cget("text")))
        _walk_labels131(child)


_walk_labels131(_win131)
assert not any(label.startswith("Recently jumped") for label in _labels131)
_payload131 = app._collect_session()
assert json.loads(_payload131["texts"]["finder_recent"]) == []
_win131.destroy()
print("Clear click: memory emptied + row collapsed + session [] OK")

# the next open renders no row and no Clear button at all
app._find_card_dialog()
_dialogs131b = [
    child
    for child in app.winfo_children()
    if isinstance(child, _tk131.Toplevel) and child.title() == "Find a card"
]
_win131b = _dialogs131b[0]
_buttons131b: list = []


def _walk_buttons131b(widget):
    for child in widget.winfo_children():
        if child.winfo_class() == "TButton":
            _buttons131b.append(child)
        _walk_buttons131b(child)


_walk_buttons131b(_win131b)
assert "Clear" not in [str(b.cget("text")) for b in _buttons131b]
_win131b.destroy()
print("Next open: no row, no Clear button OK")


app.destroy()
assert not DIALOGS, f"unexpected dialogs: {DIALOGS}"
print("PROBE_OK")

