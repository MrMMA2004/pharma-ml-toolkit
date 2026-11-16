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
from typing import Any, Mapping, Sequence

from pharmaml.viz.report import HTMLReport

__all__ = [
    "report_sections",
    "build_run_report",
    "save_run_report",
    "write_manifest",
    "MANIFEST_SCHEMA",
    "PINNED_STACK",
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
    # §17D figure embedding (R27): the Plots section's text stays the
    # honest inventory; every figure that EXISTS on disk as a PNG is
    # then embedded as its own page (base64, self-contained).  A
    # recorded path whose file is gone stays a text line with the
    # reason — never a broken image or a silent omission.
    summary = record.get("summary") or {}
    if isinstance(summary, Mapping):
        artifacts = summary.get("figures") or summary.get("plots")
        if isinstance(artifacts, list):
            for raw in artifacts:
                p = Path(str(raw))
                if p.suffix.lower() == ".png" and p.is_file():
                    report.add_image(p, caption=f"exported figure: {p.name}")
                elif p.exists():
                    report.add_text(
                        f"- recorded figure not embedded (not a PNG or "
                        f"unreadable): {p}", heading="Plots")
                else:
                    report.add_text(
                        f"- recorded figure file missing: {p}",
                        heading="Plots")
    return report


def save_run_report(record: Mapping[str, Any], path: str | Path) -> Path:
    """Write the §17D report for one saved run to a self-contained HTML."""
    return build_run_report(record).save(path)


# -- §14.2 Proposal B (round 7): reproducibility manifests ----------------------

#: The manifest schema stamp — readers refuse anything else.
MANIFEST_SCHEMA = "pharmaml-manifest/1"

#: The pinned-stack packages whose versions the manifest records.
#: A package that is not installed records as "" — an honest absence,
#: never a guess (the core toolkit is importable without any of them).
PINNED_STACK = (
    "numpy", "pandas", "scikit-learn", "scipy", "rdkit", "PySide6",
    "matplotlib", "pyqtgraph", "xgboost", "lightgbm", "catboost",
    "shap", "torch",
)

_version_cache: dict[str, str] = {}


def _package_version(name: str) -> str:
    """importlib.metadata.version(), cached after first use ("" if absent)."""
    if name not in _version_cache:
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                _version_cache[name] = version(name)
            except PackageNotFoundError:
                _version_cache[name] = ""
        except ImportError:  # pragma: no cover - py>=3.8 always has it
            _version_cache[name] = ""
    return _version_cache[name]


def _platform_facts() -> dict[str, str]:
    """The descriptive platform block (never executable, never a guess)."""
    import platform
    import sys

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def write_manifest(
    path: str | Path,
    *,
    runs: "Sequence[Mapping[str, Any]] | None" = None,
    datasets: "Sequence[Mapping[str, Any]] | None" = None,
    config: "Mapping[str, Any] | PipelineConfig | None" = None,
) -> Path:
    """Write one reproducibility manifest (§14.2 Proposal B) as JSON.

    The manifest is *descriptive, not executable*: it captures the
    environment (pharmaml + Python + pinned-stack versions, platform)
    and the workflow slice (per-run recorded hyperparameters, seeds,
    ``n_jobs``; dataset fingerprints + Proposal A content versions) so
    "I trained this in the Studio yesterday" becomes a checkable
    claim.  It never reinstalls, executes, or recomputes anything.

    Parameters
    ----------
    runs:
        Run records (``latest_runs()`` dicts) — each contributes its
        ``family``, ``label``, ``created`` and the summary keys the
        workflow chose to record (``seeds``, ``n_jobs``, hyperpar-
        ameters…), exactly as stored, never recomputed.
    datasets:
        Workspace dataset entries (``WorkspaceEntry``-shaped mappings
        or dataclasses) — each contributes name, path, sha-256,
        Proposal A ``content_version`` and ``column_signature``.
    config:
        A :class:`~pharmaml.config.PipelineConfig` (or mapping of its
        fields) — its ``random_state`` / ``test_size`` / ``cv_folds`` /
        ``n_jobs`` ride along when given; the defaults are never
        invented when it is not.

    Returns the written path (parents created).  Raises nothing the
    caller cannot handle: OSError/TypeError propagate like any honest
    file write — this is a user-invoked export, not a best-effort
    side-record.
    """
    import dataclasses
    import datetime as _dt
    import json

    from pharmaml import __version__ as _pharmaml_version

    def _as_mapping(item: Any) -> Mapping[str, Any]:
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return dataclasses.asdict(item)
        return dict(item)

    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "pharmaml_version": _pharmaml_version,
        "platform": _platform_facts(),
        "pinned_stack": {
            name: _package_version(name) for name in PINNED_STACK
        },
        "runs": [
            {
                "family": str(r.get("family", "")),
                "label": str(r.get("label", "")),
                "created": str(r.get("created", "")),
                "summary": dict(r.get("summary") or {}),
            }
            for r in (_as_mapping(item) for item in (runs or []))
        ],
        "datasets": [
            {
                "name": str(d.get("name", "")),
                "path": str(d.get("path", "")),
                "sha256": str(d.get("sha256", "")),
                "content_version": int(d.get("content_version", 1) or 1),
                "column_signature": str(d.get("column_signature", "")),
            }
            for d in (_as_mapping(item) for item in (datasets or []))
        ],
    }
    if config is not None:
        cfg = (
            dataclasses.asdict(config)
            if dataclasses.is_dataclass(config)
            else dict(config)
        )
        payload["config"] = {
            key: cfg[key] for key in (
                "random_state", "test_size", "cv_folds", "n_jobs",
            ) if key in cfg
        }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target
