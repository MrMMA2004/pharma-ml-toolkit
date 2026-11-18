"""§17D scientific report generation — the pins.

The contracts (part6 §17D):

* every report carries the eight sections, in order — Inputs,
  Methods, Parameters, Results, Plots, Statistical metrics,
  Provenance, Limitations — and empty sections SAY they are empty
  instead of vanishing;
* scientific units travel with their values (the known-key unit
  table), significant figures are preserved (five, matching §17C),
  and UNKNOWN units are never invented;
* the Limitations section is factual-only, and the report carries NO
  conclusion — the fixed closing statement says the toolkit does not
  interpret results;
* the composition rides the EXISTING self-contained HTML writer (the
  saved file renders standalone, zero external resources).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pharmaml.studio import runs as runs_mod
from pharmaml.studio.reports import (
    _SECTION_ORDER,
    UNITS,
    build_run_report,
    report_sections,
    save_run_report,
)

SECTIONS = list(_SECTION_ORDER)


@pytest.fixture()
def runs_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "studio-runs"
    monkeypatch.setattr(runs_mod, "runs_root", lambda: root)
    return root


def _ml_record(runs_root: Path) -> dict:
    assert runs_mod.record_run(
        "ml-evaluation", "family benchmark · activity (5-fold CV)",
        {"eval_kind": "cv", "metric": "roc_auc_ovo", "folds": 5,
         "task": "classification", "target": "activity",
         "families": 18, "best_family": "descriptors",
         "best_mean": 0.812345678, "n_failed": 0, "seconds": 12.5},
    )
    return runs_mod.latest_runs("ml-evaluation")[0]


def test_every_section_present_in_order(runs_root) -> None:
    record = _ml_record(runs_root)
    sections = report_sections(record)
    assert list(sections) == SECTIONS
    # populated sections carry their facts
    assert "target: activity" in sections["Inputs"]
    assert "metric: roc_auc_ovo" in sections["Methods"]
    assert "best_mean" in sections["Results"]
    assert "roc_auc_ovo" in sections["Methods"]
    assert "0.81235" in sections["Statistical metrics"]  # 5 sig figs
    assert "toolkit_version" in sections["Provenance"]


def test_units_travel_and_sig_figs_preserved(runs_root) -> None:
    assert runs_mod.record_run(
        "pkpd", "regimen · oral",
        {"route": "oral", "n_doses": 1, "c_max": 8.234567,
         "t_max": 1.5, "auc": 55.123456, "seconds": 3.25},
    )
    record = runs_mod.latest_runs("pkpd")[0]
    sections = report_sections(record)
    metrics = sections["Statistical metrics"]
    assert "8.2346 µg/mL" in metrics   # c_max, 5 sig figs + unit
    assert "1.5 h" in metrics          # t_max
    assert "55.123 µg·h/mL" in metrics  # auc
    assert "3.25 s" in sections["Parameters"]  # leftover → parameters
    # an unknown-unit key renders bare, never with a guessed unit
    assert "1" in sections["Inputs"]  # n_doses has no unit row


def test_unknown_units_are_never_invented(runs_root) -> None:
    assert runs_mod.record_run("viz-export", "tiff export",
                               {"kind": "tiff", "n": 3, "bitdepth": 16})
    record = runs_mod.latest_runs("viz-export")[0]
    sections = report_sections(record)
    text = (sections["Inputs"] + sections["Parameters"]
            + sections["Results"])
    assert "16" in text
    for unit in UNITS.values():
        assert f"16 {unit}" not in text


def test_limitations_are_factual_and_conclusions_never_auto_written(
        runs_root) -> None:
    record = _ml_record(runs_root)
    sections = report_sections(record)
    limits = sections["Limitations"]
    # facts about THIS record, not interpretations
    assert "single execution" in limits
    assert "no conclusions" in limits.lower()
    assert "does not interpret" in limits
    # the fixed closing statement is present verbatim
    assert "PharmaML does not interpret results" in limits
    # and there is no Conclusion section anywhere
    assert "Conclusion" not in SECTIONS
    html = build_run_report(record).save(runs_root.parent / "r.html")
    content = html.read_text(encoding="utf-8")
    assert "Conclusion" not in content
    assert "Limitations" in content


def test_empty_sections_say_so(runs_root) -> None:
    assert runs_mod.record_run("docking", "campaign scan",
                               {"campaign": "demo"})
    record = runs_mod.latest_runs("docking")[0]
    sections = report_sections(record)
    assert "no statistical metrics recorded" in sections[
        "Statistical metrics"]
    assert "no figures recorded" in sections["Plots"]
    assert "no held-out or cross-validated performance" in sections[
        "Limitations"]


def test_plots_section_lists_recorded_figures(runs_root) -> None:
    assert runs_mod.record_run(
        "generative", "pareto front",
        {"kind": "pareto-front", "objectives": ["qed", "mw_band"],
         "n_front": 9, "n_total": 276,
         "figures": ["/tmp/exports/front.png"]},
    )
    record = runs_mod.latest_runs("generative")[0]
    sections = report_sections(record)
    assert "/tmp/exports/front.png" in sections["Plots"]


def test_training_history_is_summarized_not_redrawn(runs_root) -> None:
    assert runs_mod.record_run(
        "dl-training", "DeepMLPRegressor on pIC50",
        {"engine": "torch", "epochs_asked": 40, "epochs_run": 40,
         "history": [{"epoch": i, "train_loss": 1.0 - i / 50,
                      "val_loss": 1.05 - i / 50} for i in range(1, 6)]},
    )
    record = runs_mod.latest_runs("dl-training")[0]
    sections = report_sections(record)
    assert "5 epochs recorded" in sections["Plots"]
    # the long history never floods a section: it is counted, not listed
    assert "epoch" not in sections["Inputs"]


def test_report_saves_self_contained(runs_root, tmp_path: Path) -> None:
    record = _ml_record(runs_root)
    out = tmp_path / "report.html"
    path = save_run_report(record, out)
    assert path.exists() and path.stat().st_size > 0
    content = path.read_text(encoding="utf-8")
    # all eight section headings render
    for name in SECTIONS:
        assert name in content
    # self-contained: no external stylesheet/script/link references
    assert "<link" not in content
    assert 'src="http' not in content
    assert 'href="http' not in content
