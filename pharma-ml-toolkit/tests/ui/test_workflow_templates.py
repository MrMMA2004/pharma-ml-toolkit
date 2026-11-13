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
    tab._param_a.setValue(1.2)   # ka
    tab._param_b.setValue(0.25)  # ke
    tab._param_c.setValue(40.0)  # Vd
    tab._param_d.setValue(250.0)  # dose
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
    tab._param_e.setValue(24)
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


def test_data_dependent_templates_explain_honestly(tab) -> None:
    tab._choice.setCurrentIndex(
        [tab._choice.itemData(i) for i in range(tab._choice.count())]
        .index("qsar_model_development"))
    tab._out_dir.setText("/tmp/whatever-r24")
    tab._run()
    log = tab._log.toPlainText()
    assert "needs data inputs this quick form does not collect" in log
    assert "build_template" in log


def test_utilities_page_hosts_the_templates_tab(qtbot, studio_app) -> None:
    from ui.pages.utilities import UtilitiesWorkbenchPage

    page = UtilitiesWorkbenchPage()
    qtbot.addWidget(page)
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert "Workflow templates" in labels
