"""The §17A workflow-templates tab — offscreen tests.

Everything runs against the REAL template library (no mocks): the
browser lists all seven part6 §17A templates with their owning cards,
the stage chain renders (each stage naming the existing backend it
calls), the parameter-light templates actually RUN through the engine
(the PK/PD chain completes in seconds and writes its run record), and
the data-dependent templates explain honestly that their quick form
collects no inputs instead of pretending to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def tab(qtbot, studio_app):
    from ui.widgets.workflow_templates import WorkflowTemplatesTab

    widget = WorkflowTemplatesTab()
    qtbot.addWidget(widget)
    return widget


def test_all_seven_templates_are_listed_with_cards(tab) -> None:
    names = [tab._choice.itemText(i) for i in range(tab._choice.count())]
    assert len(names) == 7
    for wanted in ("qsar_model_development", "molecular_descriptor_generation",
                   "drug_response_prediction", "molecular_generation_screening",
                   "docking_preparation_analysis", "pk_pd_modeling",
                   "md_analysis"):
        assert any(wanted in n for n in names), wanted
    # every entry carries its owning card
    assert any("[Machine Learning]" in n for n in names)
    assert any("[PK/PD]" in n for n in names)


def test_stage_chain_describes_real_backends(tab) -> None:
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("qsar_model_development"))
    text = tab._stages.toPlainText()
    assert "dataset — load" in text
    assert "pharmaml.data.load_dataset" in text
    assert "preprocessing" in text and "split_table" in text
    assert "training" in text and "compare_models" in text
    assert "evaluation" in text
    assert "export" in text


def test_pkpd_template_runs_and_writes_the_record(tab, qtbot, tmp_path) -> None:
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("pk_pd_modeling"))
    tab._inputs["ka"].setValue(1.2)     # ka
    tab._inputs["ke"].setValue(0.25)    # ke
    tab._inputs["vd"].setValue(40.0)    # Vd
    tab._inputs["dose"].setValue(250.0)  # dose
    tab._out_dir.setText(str(tmp_path))
    tab._run()
    log = tab._log.toPlainText()
    assert "running pk_pd_modeling" in log
    assert "done in" in log
    record_path = tmp_path / "pk_pd_modeling.run.json"
    assert record_path.exists()
    record = json.loads(record_path.read_text())
    assert record["ok"] is True
    assert [s["stage"] for s in record["stages"]] == [
        "modeling", "parameters", "export"]
    assert tab._last_record is record or tab._last_record == record


def test_md_template_runs_and_writes_the_record(tab, qtbot, tmp_path) -> None:
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("md_analysis"))
    tab._inputs["n_frames"].setValue(24)
    tab._out_dir.setText(str(tmp_path))
    tab._run()
    record = json.loads(
        (tmp_path / "md_analysis.run.json").read_text())
    assert record["ok"] is True
    assert len(record["stages"]) == 3
    assert (tmp_path / "rmsd.csv").exists()


def test_run_without_an_export_folder_is_an_inline_error(tab) -> None:
    tab._out_dir.setText("")
    tab._run()
    assert "pick an export folder first" in tab._log.toPlainText()


def _qsar_csv(tmp_path):
    import csv
    path = tmp_path / "qsar.csv"
    # the QSAR template trains on the numeric columns the dataset
    # already carries (a raw SMILES column is excluded, visibly) — so
    # the fixture carries two toy numeric features alongside smiles;
    # 16 rows (8 per class) because the 25% split needs one test
    # sample per class or sklearn refuses the split
    rows = [{"smiles": s, "mw": mw, "logp": logp, "activity": a}
            for s, mw, logp, a in [
                ("CCO", 46.1, -0.3, "1"), ("CCC", 44.1, 1.4, "0"),
                ("CCN", 45.1, -0.1, "1"), ("CCCl", 64.5, 1.7, "0"),
                ("CCCO", 60.1, -0.4, "1"), ("CCCC", 58.1, 2.2, "0"),
                ("CCCN", 59.1, 0.1, "1"), ("CCCCl", 78.5, 2.6, "0"),
                ("CCOCC", 74.1, 0.6, "1"), ("CCCCCC", 86.2, 3.1, "0"),
                ("CCNCC", 73.1, 0.8, "1"), ("CCCCCCl", 106.6, 3.5, "0"),
                ("CCOC", 60.1, 0.2, "1"), ("CCCCO", 74.1, 1.6, "0"),
                ("CCNC", 59.1, 0.4, "1"), ("CCCCN", 73.1, 1.8, "0"),
            ]]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["smiles", "mw", "logp", "activity"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_qsar_template_runs_end_to_end(tab, qtbot, tmp_path) -> None:
    """R26-c: the card-integrated form drives the full chain."""
    data = _qsar_csv(tmp_path)
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("qsar_model_development"))
    # typing the path fills the target combo from the CSV's header
    tab._inputs["data_path"].setText(str(data))
    combo_items = [tab._inputs["target"].itemText(i)
                   for i in range(tab._inputs["target"].count())]
    assert "activity" in combo_items
    tab._inputs["target"].setCurrentText("activity")
    tab._out_dir.setText(str(tmp_path / "out"))
    tab._run()
    log = tab._log.toPlainText()
    assert "running qsar_model_development" in log
    record = tmp_path / "out" / "qsar_model_development.run.json"
    assert record.exists()


def test_qsar_template_requires_the_data_path(tab, qtbot, tmp_path) -> None:
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("qsar_model_development"))
    tab._out_dir.setText(str(tmp_path))
    tab._run()
    assert "the dataset CSV is required" in tab._log.toPlainText()


def test_generative_template_form_runs_the_objective(
        tab, qtbot, tmp_path) -> None:
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("molecular_generation_screening"))
    tab._inputs["objective"].setCurrentText("qed")
    tab._inputs["generations"].setValue(2)
    tab._out_dir.setText(str(tmp_path / "out"))
    tab._run()
    log = tab._log.toPlainText()
    assert "running molecular_generation_screening" in log
    record = tmp_path / "out" / "molecular_generation_screening.run.json"
    assert record.exists()


def test_utilities_page_hosts_the_templates_tab(qtbot, studio_app) -> None:
    from ui.pages.utilities import UtilitiesWorkbenchPage

    page = UtilitiesWorkbenchPage()
    qtbot.addWidget(page)
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert "Workflow templates" in labels


def test_export_folder_is_remembered_per_template(
        tab, qtbot, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE",
                       str(tmp_path / "ws.json"))
    from pharmaml.studio.workspace import (
        last_export_folder,
        remember_export_folder,
    )
    remember_export_folder(str(tmp_path / "qsar-out"),
                           "qsar_model_development")
    # the tab fixture starts on template 0 (qsar) BEFORE the folder
    # was remembered — switch away and back so _refresh runs the
    # recall for the now-remembered folder
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("pk_pd_modeling"))
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("qsar_model_development"))
    assert tab._out_dir.text() == str(tmp_path / "qsar-out")
    # the pk/pd template has none remembered — the field clears
    remember_export_folder(str(tmp_path / "pk-out"), "pk_pd_modeling")
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("pk_pd_modeling"))
    assert tab._out_dir.text() == str(tmp_path / "pk-out")
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("qsar_model_development"))
    assert tab._out_dir.text() == str(tmp_path / "qsar-out")
    # the Remember button persists the CURRENT folder for the template
    tab._out_dir.setText(str(tmp_path / "new-place"))
    tab._remember_folder()
    assert last_export_folder(
        "qsar_model_development") == str(tmp_path / "new-place")


def _screen_csv(tmp_path):
    import csv
    path = tmp_path / "screen.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["cell_line", "compound", "auc"])
        w.writeheader()
        w.writerows([
            {"cell_line": "A549", "compound": "imatinib", "auc": "0.4"},
            {"cell_line": "A549", "compound": "erlotinib", "auc": "0.6"},
            {"cell_line": "MCF7", "compound": "imatinib", "auc": "0.5"},
            {"cell_line": "MCF7", "compound": "erlotinib", "auc": "0.3"},
        ])
    return path


def test_response_compound_combo_lists_real_values(
        tab, qtbot, tmp_path) -> None:
    """R27-c: the compound combo lists the screen's actual compounds."""
    screen = _screen_csv(tmp_path)
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("drug_response_prediction"))
    tab._inputs["screen_path"].setText(str(screen))
    items = [tab._inputs["compound"].itemText(i)
             for i in range(tab._inputs["compound"].count())]
    assert items == ["(auto)", "imatinib", "erlotinib"]


def test_docking_smiles_col_auto_detects_the_header(
        tab, qtbot, tmp_path) -> None:
    """R27-c: the docking SMILES combo adopts the CSV's own column."""
    ligands = tmp_path / "ligands.csv"
    ligands.write_text("name,Canonical_SMILES\n"
                       "L1,CCO\nL2,CCC\n")
    receptor = tmp_path / "rec.pdbqt"
    receptor.write_text("REMARK placeholder\n")
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("docking_preparation_analysis"))
    tab._inputs["ligands_path"].setText(str(ligands))
    # the header's columns are offered; no 'smiles' column exists so
    # the detected current stays (auto) and _run passes None (the
    # backend's own candidate resolution)
    items = [tab._inputs["smiles_col"].itemText(i)
             for i in range(tab._inputs["smiles_col"].count())]
    assert "(auto)" in items
    assert "Canonical_SMILES" in items


def test_descriptor_smiles_auto_resolves_case_insensitively(
        tab, qtbot, tmp_path) -> None:
    """R27-c: (auto) resolves to the header's actual smiles column."""
    data = tmp_path / "mols.csv"
    data.write_text("SMILES,mw\nCCO,46\nCCC,44\n")
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("molecular_descriptor_generation"))
    tab._inputs["data_path"].setText(str(data))
    # header auto-detect: the combo's current text becomes 'SMILES'
    assert tab._inputs["smiles_col"].currentText() == "SMILES"
