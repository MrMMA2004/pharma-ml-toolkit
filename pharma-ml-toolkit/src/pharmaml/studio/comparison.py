"""§17C — experiment comparison over the Studio's saved runs.

The Studio records every finished workflow as a JSON run record under
``~/.pharmaml/studio-runs/<family>/`` (:mod:`pharmaml.studio.runs`).
This module turns those records into *scientific* comparisons:

1. **Extract** the comparable dimensions of each run — dataset
   identity (target + task + family context), model configuration,
   hyperparameters, validation protocol (eval kind + metric + folds),
   performance metrics, runtime and resource consumption (where the
   recording workflow captured them).
2. **Group** the runs by a *compatibility key*: two runs may be ranked
   against each other only when their task, metric and validation
   protocol agree — comparing a ROC-AUC classification leaderboard
   against an R² regression run is meaningless and must never be
   silently merged (part6 §17C: "support clear comparisons without
   mixing incompatible experimental conditions").
3. **Compare** within each compatible group: a side-by-side table of
   every dimension, per-metric deltas against the group's best run,
   and — for metric-bearing groups — a dominance verdict (which run is
   best on the shared metric, and by how much).  No verdict is emitted
   for groups lacking a common metric: the table still shows, the
   ranking stays silent.

The verdicts are arithmetic facts about the recorded numbers, nothing
more.  There is no auto-written conclusion here and none in the §17D
report surface that consumes this module — "X scored 0.09 higher
ROC-AUC than Y under the same 5-fold protocol" is a comparison, not an
interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from pharmaml.studio.runs import (
    RUN_SCHEMA,
    latest_runs,
    run_families,
)

__all__ = [
    "ComparisonDimension",
    "ComparisonGroup",
    "RunView",
    "collect_runs",
    "compatibility_key",
    "compare_runs",
    "comparison_summary",
]


# ---------------------------------------------------------------------------
# The comparable view of one run record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunView:
    """One saved run flattened into its comparable dimensions."""

    family: str
    label: str
    created: str
    toolkit_version: str
    # dataset identity
    target: str | None = None
    task: str | None = None
    n_samples: int | None = None
    # model configuration
    model: str | None = None
    engine: str | None = None
    # hyperparameters (the workflow-recorded subset, verbatim)
    params: Mapping[str, Any] = field(default_factory=dict)
    # validation protocol
    eval_kind: str | None = None
    metric: str | None = None
    folds: int | None = None
    # performance
    metric_value: float | None = None
    extra_metrics: Mapping[str, float] = field(default_factory=dict)
    # runtime / resources (present when the workflow captured them)
    seconds: float | None = None
    epochs_run: int | None = None
    # the untouched source record (§17D reports consume it)
    record: Mapping[str, Any] | None = None

    @property
    def when(self) -> str:
        return str(self.created)[:19].replace("T", " ")


_PARAM_KEYS = (
    # ml-evaluation records
    "models", "families", "best_family",
    # dl-training records
    "engine", "hidden", "epochs_asked", "learning_rate", "batch_size",
    # generative records
    "objectives", "kind",
    # pkpd records
    "route", "n_doses", "dose", "ka", "ke", "vd",
    # docking records
    "campaign",
)

_METRIC_KEYS = (
    "roc_auc_ovo", "balanced_accuracy", "r2", "rmse", "mae",
    "roc_auc", "auc", "accuracy",
)

#: keys that hold a *value* of the run's declared metric, not a metric
#: themselves (ml-evaluation's "best_mean" is the value OF "metric").
_VALUE_KEYS = ("best_mean", "mean", "value")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _run_view(record: Mapping[str, Any]) -> RunView:
    summary = record.get("summary") or {}
    if not isinstance(summary, Mapping):
        summary = {}

    params = {k: summary[k] for k in _PARAM_KEYS if k in summary}
    metric = summary.get("metric")
    value = None
    for key in _VALUE_KEYS:
        if key in summary:
            value = _number(summary[key])
            if value is not None:
                break

    extra: dict[str, float] = {}
    for key in _METRIC_KEYS:
        if key in summary and key != metric:
            got = _number(summary[key])
            if got is not None:
                extra[key] = got
    # dl-training: the final epoch's losses are the run's metrics
    history = summary.get("history")
    if isinstance(history, list) and history and isinstance(history[-1], Mapping):
        for key in ("train_loss", "val_loss"):
            got = _number(history[-1].get(key))
            if got is not None:
                extra[f"final_{key}"] = got

    n = summary.get("n_samples") or summary.get("n") or summary.get("n_total")
    seconds = _number(summary.get("seconds") or summary.get("runtime_s")
                      or summary.get("elapsed_s"))

    return RunView(
        family=str(record.get("family", "")),
        label=str(record.get("label", "(unlabelled)")),
        created=str(record.get("created", "")),
        toolkit_version=str(record.get("toolkit_version", "")),
        target=(str(summary["target"]) if summary.get("target") else None),
        task=(str(summary["task"]) if summary.get("task") else None),
        n_samples=(int(n) if isinstance(n, (int, float)) else None),
        model=(str(summary["model"]) if summary.get("model") else None),
        engine=(str(summary["engine"]) if summary.get("engine") else None),
        params=params,
        eval_kind=(str(summary["eval_kind"]) if summary.get("eval_kind") else None),
        metric=(str(metric) if metric else None),
        folds=(int(summary["folds"]) if isinstance(summary.get("folds"), int) else None),
        metric_value=value,
        extra_metrics=extra,
        seconds=seconds,
        epochs_run=(int(summary["epochs_run"])
                    if isinstance(summary.get("epochs_run"), int) else None),
        record=dict(record),
    )


def collect_runs(family: str | None = None, limit: int = 25) -> list[RunView]:
    """Load saved runs as comparable views (newest first).

    ``family=None`` loads every family's newest records.  Foreign or
    corrupt records are skipped by :func:`latest_runs` itself.
    """
    families = [family] if family else sorted(run_families())
    views: list[RunView] = []
    for fam in families:
        for record in latest_runs(fam, limit=limit):
            if record.get("schema") == RUN_SCHEMA:
                views.append(_run_view(record))
    views.sort(key=lambda v: v.created, reverse=True)
    return views


# ---------------------------------------------------------------------------
# Compatibility grouping
# ---------------------------------------------------------------------------

def compatibility_key(view: RunView) -> tuple[str, ...]:
    """The experimental conditions two runs must share to be ranked.

    Task, metric and validation protocol are the non-negotiable axes;
    the dataset identity (target + family context) joins a run to its
    peers — runs on *different* targets can still be shown side by
    side but their metric delta is only computed when the identity
    matches.
    """
    return (
        f"task={view.task or '?'}",
        f"metric={view.metric or '?'}",
        f"protocol={view.eval_kind or '?'}",
        f"folds={view.folds if view.folds is not None else '?'}",
    )


@dataclass
class ComparisonDimension:
    """One column of the side-by-side table (or one divergence note)."""

    name: str
    values: dict[str, str] = field(default_factory=dict)  # run label -> shown value
    numeric: bool = False
    unit: str = ""


@dataclass
class ComparisonGroup:
    """A set of runs sharing one compatibility key."""

    key: tuple[str, ...]
    runs: list[RunView] = field(default_factory=list)

    @property
    def comparable(self) -> bool:
        """True when the group carries a shared metric to rank on."""
        return (
            len(self.runs) >= 2
            and self.runs[0].metric is not None
            and all(v.metric_value is not None for v in self.runs)
            and all(v.metric == self.runs[0].metric for v in self.runs)
        )


def _format(value: Any, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        # significant-figure discipline: five sig figs; a zero delta
        # must render "0", never "-0"
        if value == 0:
            value = 0.0
        text = f"{value:.5g}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def _dimensions(runs: list[RunView]) -> list[ComparisonDimension]:
    """The side-by-side columns: identity rows first, then the numbers."""

    def dim(name: str, pick, unit: str = "", numeric: bool = False):
        return ComparisonDimension(
            name=name,
            values={r.label: _format(pick(r), unit) for r in runs},
            numeric=numeric,
            unit=unit,
        )

    out = [
        dim("recorded", lambda r: r.when),
        dim("family", lambda r: r.family),
        dim("toolkit", lambda r: r.toolkit_version or "—"),
        dim("target", lambda r: r.target),
        dim("task", lambda r: r.task),
        dim("model", lambda r: r.model or r.engine),
        dim("protocol", lambda r: r.eval_kind),
        dim("folds", lambda r: r.folds, numeric=True),
    ]
    if any(r.params for r in runs):
        keys: list[str] = []
        for r in runs:
            for k in r.params:
                if k not in keys:
                    keys.append(k)
        for k in keys:
            out.append(dim(f"param:{k}", lambda r, k=k: r.params.get(k)))
    metric = runs[0].metric
    if metric is not None:
        out.append(dim(metric, lambda r: r.metric_value, numeric=True))
    extra_keys: list[str] = []
    for r in runs:
        for k in r.extra_metrics:
            if k not in extra_keys:
                extra_keys.append(k)
    for k in extra_keys:
        out.append(dim(k, lambda r, k=k: r.extra_metrics.get(k), numeric=True))
    out.append(dim("runtime (s)", lambda r: r.seconds, numeric=True))
    if any(r.epochs_run is not None for r in runs):
        out.append(dim("epochs run", lambda r: r.epochs_run, numeric=True))
    return out


@dataclass
class ComparisonResult:
    """The §17C deliverable for a set of selected runs."""

    groups: list[ComparisonGroup] = field(default_factory=list)
    # incompatible pairs the comparison REFUSED to rank together
    refused: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return any(g.comparable for g in self.groups)

    def to_text(self) -> str:
        return comparison_summary(self)


def compare_runs(views: Iterable[RunView]) -> ComparisonResult:
    """Group the selected runs and compute per-group comparisons.

    Every pair whose compatibility keys differ lands in ``refused``
    with the reason (the differing axis) — the caller shows these as
    explicitly excluded, never as silently merged rows.
    """
    runs = list(views)
    by_key: dict[tuple[str, ...], list[RunView]] = {}
    for view in runs:
        by_key.setdefault(compatibility_key(view), []).append(view)

    groups = [ComparisonGroup(key=key, runs=sorted(
        members, key=lambda v: v.created, reverse=True))
        for key, members in by_key.items()]
    groups.sort(key=lambda g: (not g.comparable, g.key))

    refused: list[tuple[str, str, str]] = []
    keys = list(by_key)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            differing = []
            for part_a, part_b in zip(a, b):
                if part_a != part_b:
                    differing.append(part_a.split("=")[0])
            if differing:
                label_a = by_key[a][0].label
                label_b = by_key[b][0].label
                refused.append((label_a, label_b, ", ".join(differing)))

    result = ComparisonResult(groups=groups, refused=refused)

    # per-group verdicts live on the group (deltas + best run)
    for group in result.groups:
        if group.comparable:
            _annotate_verdict(group)
    return result


def _annotate_verdict(group: ComparisonGroup) -> None:
    """Attach deltas + the best-run verdict to a comparable group.

    Stored as the last dimensions of the group's table (the module's
    dataclass has no dedicated verdict field — the table IS the
    verdict, keeping one rendering path for the UI and the text
    summary).
    """
    runs = group.runs
    metric = runs[0].metric or "metric"
    higher = not str(metric).startswith(("rmse", "mae", "loss"))
    if higher:
        best = max(runs, key=lambda r: r.metric_value or float("-inf"))
    else:
        best = min(runs, key=lambda r: r.metric_value or float("inf"))
    setattr(group, "_verdict", (best.label, metric, higher))
    delta_values = {}
    for r in runs:
        delta = _delta(r, best, higher)
        delta_values[r.label] = _format(delta)
    deltas = ComparisonDimension(
        name=f"Δ {metric} vs best",
        values=delta_values,
        numeric=True,
    )
    group_dimensions = _dimensions(runs)
    group_dimensions.append(deltas)
    setattr(group, "_table", group_dimensions)


def _delta(run: RunView, best: RunView, higher_is_better: bool) -> float | None:
    """How far ``run`` trails ``best`` on the shared metric (≥ 0).

    The convention keeps one reading for both metric polarities: the
    best run's delta is 0 and every other run's delta counts the gap
    it must close — ``best - run`` when higher is better,
    ``run - best`` when lower is better.
    """
    if run.metric_value is None or best.metric_value is None:
        return None
    if higher_is_better:
        return best.metric_value - run.metric_value
    return run.metric_value - best.metric_value


def comparison_summary(result: ComparisonResult) -> str:
    """The plain-text rendering (the CLI/report twin of the UI table)."""
    lines: list[str] = []
    if not result.groups:
        return "No runs selected."
    for group in result.groups:
        lines.append(f"== group {' · '.join(group.key)}")
        table = getattr(group, "_table", None) or _dimensions(group.runs)
        headers = ["dimension"] + [r.label for r in group.runs]
        lines.append(" | ".join(headers))
        for dim in table:
            row = [dim.name] + [dim.values.get(r.label, "—")
                                for r in group.runs]
            lines.append(" | ".join(row))
        verdict = getattr(group, "_verdict", None)
        if verdict:
            best, metric, _ = verdict
            lines.append(f"best on {metric}: {best}")
        else:
            reasons = ("no shared metric" if group.runs[0].metric is None
                       else "missing metric values")
            lines.append(f"no ranking: {reasons} (side-by-side only)")
        lines.append("")
    for a, b, why in result.refused:
        lines.append(f"not compared — {a} vs {b}: differing {why}")
    return "\n".join(lines).rstrip() + "\n"
