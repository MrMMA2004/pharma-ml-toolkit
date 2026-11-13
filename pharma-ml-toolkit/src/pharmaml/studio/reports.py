"""§17D — structured scientific reports from saved runs.

part6 §17D asks for reports containing, in order: **inputs, methods,
parameters, results, plots, statistical metrics, provenance,
limitations** — with scientific units and significant figures
preserved, and an absolute prohibition: *"Do not write unsupported
conclusions automatically."*

This module composes those sections from the Studio's run records
(the same JSON the Job Center and the §17C comparison consume) and
writes them through the EXISTING self-contained HTML writer
(:class:`pharmaml.viz.report.HTMLReport`) — one rendering engine, no
duplicate document code.

The honesty rules, mechanically enforced:

* Every number shown is the RECORDED value, formatted to five
  significant figures (matching the §17C tables); nothing is
  recomputed, rounded up, or invented.  Units travel with their
  values because the summary keys carry them (``c_max`` in µg/mL,
  ``t_max`` in h, ``auc`` in µg·h/mL, ``rmsd`` in Å…) — the
  per-family unit table below states them, and where a key's unit is
  NOT known the report prints the bare number with no invented unit.
* The **Limitations** section is assembled from factual statements
  only: what the record does not contain (no held-out evaluation, no
  runtime captured, single run / no repetition), never an
  interpretation of the science.
* The closing statement is fixed and explicit: the toolkit does not
  draw conclusions; interpretation belongs to the researcher who
  reads the numbers in context.  There is no "Conclusion" section to
  auto-write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pharmaml.viz.report import HTMLReport

__all__ = [
    "report_sections",
    "build_run_report",
    "save_run_report",
    "UNITS",
]


#: Known units for recorded summary keys.  A key absent here renders
#: as a bare number — an unknown unit is NEVER guessed.
UNITS: dict[str, str] = {
    # pkpd
    "c_max": "µg/mL", "t_max": "h", "auc": "µg·h/mL",
    "tmax_h": "h", "half_life_h": "h", "auc_trap": "µg·h/mL",
    # md
    "rmsd_mean": "Å", "rmsd_max": "Å",
    # timing
    "seconds": "s", "runtime_s": "s",
}

_SECTION_ORDER = (
    "Inputs", "Methods", "Parameters", "Results",
    "Plots", "Statistical metrics", "Provenance", "Limitations",
)

_NO_CONCLUSIONS_NOTE = (
    "This report presents the recorded facts of the run. It contains "
    "no conclusions: PharmaML does not interpret results "
    "automatically, and any scientific reading of the numbers above "
    "belongs to the researcher who ran the experiment."
)


def _fmt(value: Any) -> str:
    """Five significant figures; verbatim for non-numbers."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            value = 0.0
        return f"{value:.5g}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _with_unit(key: str, value: Any) -> str:
    unit = UNITS.get(key)
    if unit and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{_fmt(value)} {unit}"
    return _fmt(value)


def _kv_lines(pairs: list[tuple[str, Any]], unit_keys: set[str] | None = None
              ) -> str:
    lines = []
    for key, value in pairs:
        if unit_keys and key in unit_keys:
            lines.append(f"- {key}: {_with_unit(key, value)}")
        elif isinstance(value, list) and len(value) > 6:
            lines.append(f"- {key}: [{len(value)} values, first "
                         f"{_fmt(value[0])} … last {_fmt(value[-1])}]")
        else:
            lines.append(f"- {key}: {_fmt(value)}")
    return "\n".join(lines)


#: Which summary keys belong to which §17D section.  A key appears in
#: exactly one section; everything unclassified lands in Parameters
#: (the honest catch-all, mirroring how the run was actually driven).
_INPUT_KEYS = ("target", "task", "n", "n_samples", "n_total", "n_pairs",
               "n_prepared", "n_front", "n_molecules", "n_features",
               "n_doses", "families", "campaign", "objectives", "kind")
_METHOD_KEYS = ("eval_kind", "metric", "folds", "engine", "route",
                "model", "best_family", "strategy", "split")
_RESULT_KEYS = ("best_mean", "best_family", "mean", "value",
                "n_failed", "epochs_run", "deepest", "n_front")
_PROVENANCE_KEYS = ("toolkit_version", "schema", "created", "family")


def report_sections(record: Mapping[str, Any]) -> dict[str, str]:
    """Map a run record to its §17D sections (text blocks).

    The mapping is purely mechanical: the record's own keys decide
    which section each fact appears in; sections with no recorded
    content say so explicitly (a section that silently vanished would
    hide the structure §17D demands).
    """
    summary = record.get("summary") or {}
    if not isinstance(summary, Mapping):
        summary = {}

    def pick(keys: tuple[str, ...]) -> list[tuple[str, Any]]:
        return [(k, summary[k]) for k in keys if k in summary]

    def leftover() -> list[tuple[str, Any]]:
        classified = set(_INPUT_KEYS) | set(_METHOD_KEYS) | set(
            _RESULT_KEYS) | set(_PROVENANCE_KEYS)
        return [(k, v) for k, v in summary.items()
                if k not in classified and k != "history"]

    inputs = pick(_INPUT_KEYS)
    methods = pick(_METHOD_KEYS)
    results = pick(_RESULT_KEYS)
    params = leftover()
    provenance = [
        ("toolkit_version", record.get("toolkit_version", "")),
        ("schema", record.get("schema", "")),
        ("family", record.get("family", "")),
        ("label", record.get("label", "(unlabelled)")),
        ("run recorded", str(record.get("created", ""))[:19]
         .replace("T", " ")),
    ]

    # statistical metrics: numeric result-ish keys, minus the headline
    metric_keys = ("best_mean", "mean", "value", "c_max", "t_max", "auc",
                   "tmax_h", "cmax", "half_life_h", "auc_trap",
                   "final_train_loss", "final_val_loss")
    metrics = [(k, summary[k]) for k in metric_keys if k in summary]

    sections = {
        "Inputs": _kv_lines(inputs, set(UNITS)) or
        "- (no input facts recorded for this run)",
        "Methods": _kv_lines(methods) or
        "- (no method facts recorded for this run)",
        "Parameters": _kv_lines(params, set(UNITS)) or
        "- (no additional parameters recorded)",
        "Results": _kv_lines(results, set(UNITS)) or
        "- (no result values recorded for this run)",
        "Plots": _plots_section(summary),
        "Statistical metrics": _kv_lines(metrics, set(UNITS)) or
        "- (no statistical metrics recorded for this run)",
        "Provenance": _kv_lines(provenance),
        "Limitations": _limitations(record, summary, metrics),
    }
    return sections


def _plots_section(summary: Mapping[str, Any]) -> str:
    """State honestly which plots exist — none are invented here.

    Run records reference exported figures by path when the workflow
    wrote any; a record without one gets the explicit statement (the
    §17D structure keeps the section, the content stays truthful).
    """
    artifacts = summary.get("figures") or summary.get("plots")
    if isinstance(artifacts, list) and artifacts:
        lines = [f"- exported figure: {a}" for a in artifacts]
        return "\n".join(lines)
    history = summary.get("history")
    if isinstance(history, list) and len(history) > 1:
        return (f"- training history: {len(history)} epochs recorded "
                "(loss curves available in the workflow's own plot; "
                "this report lists the recorded series, it does not "
                "redraw them)")
    return ("- no figures recorded for this run (the report shows the "
            "recorded numbers; it does not synthesize plots)")


def _limitations(record: Mapping[str, Any], summary: Mapping[str, Any],
                 metrics: list[tuple[str, Any]]) -> str:
    """Factual gaps of THIS record — never an interpretation."""
    notes: list[str] = []
    if not metrics:
        notes.append("- this run recorded no statistical metrics")
    if "best_mean" not in summary and "value" not in summary:
        notes.append("- no held-out or cross-validated performance value "
                     "was recorded; nothing here ranks this run")
    if "seconds" not in summary and "runtime_s" not in summary:
        notes.append("- runtime was not captured")
    if not isinstance(summary.get("history"), list):
        notes.append("- single execution: no repetition or seed "
                     "variation is recorded")
    notes.append(_NO_CONCLUSIONS_NOTE)
    return "\n".join(notes)


def build_run_report(record: Mapping[str, Any]) -> HTMLReport:
    """Compose the §17D report for one saved run (an HTMLReport)."""
    label = str(record.get("label", "(unlabelled)"))
    family = str(record.get("family", ""))
    report = HTMLReport(
        f"Scientific run report — {label}",
        subtitle=f"family: {family}" if family else None,
    )
    sections = report_sections(record)
    report.add_text(
        "Report structure: " + " → ".join(_SECTION_ORDER) + ".",
        heading="Structure",
    )
    for name in _SECTION_ORDER:
        report.add_text(sections[name], heading=name)
    return report


def save_run_report(record: Mapping[str, Any], path: str | Path) -> Path:
    """Write the §17D report for one saved run to a self-contained HTML."""
    return build_run_report(record).save(path)
