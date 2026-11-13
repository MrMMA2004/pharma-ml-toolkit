"""§17C experiment comparison — the pins.

The contracts:

* every saved run record becomes a comparable :class:`RunView` with
  dataset identity, model config, hyperparameters, validation
  protocol, metrics, runtime and resources (present when recorded);
* runs are GROUPED by compatibility key (task + metric + protocol +
  folds) — two runs with different conditions are never ranked
  together; every cross-condition pair is reported as explicitly
  refused with the differing axes;
* within a compatible group the comparison is a side-by-side table of
  every dimension plus per-run metric deltas against the best, with
  lower-is-better metrics (rmse/mae/loss) handled correctly;
* no verdict is emitted for groups without a shared metric — the table
  shows, the ranking stays silent.

The fixtures write REAL run records through the existing
``record_run`` API into a monkeypatched runs root, exactly like the
Studio workflows do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharmaml.studio import runs as runs_mod
from pharmaml.studio.comparison import (
    RunView,
    collect_runs,
    compare_runs,
    compatibility_key,
    comparison_summary,
)


@pytest.fixture()
def runs_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "studio-runs"
    monkeypatch.setattr(runs_mod, "runs_root", lambda: root)
    return root


def _record(runs_root: Path, family: str, label: str, **summary) -> None:
    path = runs_mod.record_run(family, label, summary)
    assert path is not None, f"record_run failed for {label}"


# -- the comparable view ------------------------------------------------------

def test_run_view_extracts_every_dimension(runs_root) -> None:
    _record(
        runs_root, "ml-evaluation", "family benchmark · activity (5-fold CV)",
        eval_kind="cv", metric="roc_auc_ovo", folds=5, task="classification",
        target="activity", families=18, best_family="descriptors",
        best_mean=0.812345, n_failed=0, seconds=12.5,
    )
    views = collect_runs("ml-evaluation")
    assert len(views) == 1
    v = views[0]
    assert v.task == "classification"
    assert v.target == "activity"
    assert v.metric == "roc_auc_ovo"
    assert v.folds == 5
    assert v.eval_kind == "cv"
    assert v.metric_value == pytest.approx(0.812345)
    assert v.params["families"] == 18
    assert v.params["best_family"] == "descriptors"
    assert v.seconds == pytest.approx(12.5)
    assert v.toolkit_version  # stamped by record_run


def test_dl_run_view_reads_final_epoch_losses(runs_root) -> None:
    _record(
        runs_root, "dl-training", "DeepMLPRegressor on pIC50",
        engine="torch", epochs_asked=40, epochs_run=40,
        history=[{"epoch": 1, "train_loss": 0.99, "val_loss": 0.98},
                 {"epoch": 2, "train_loss": 0.90, "val_loss": 0.91}],
        seconds=45.0,
    )
    v = collect_runs("dl-training")[0]
    assert v.engine == "torch"
    assert v.epochs_run == 40
    assert v.extra_metrics["final_train_loss"] == pytest.approx(0.90)
    assert v.extra_metrics["final_val_loss"] == pytest.approx(0.91)
    assert v.metric is None  # a training run has no declared metric


# -- compatibility ------------------------------------------------------------

def test_incompatible_conditions_are_refused_not_merged(runs_root) -> None:
    _record(runs_root, "ml-evaluation", "clf run (roc_auc, 5-fold)",
            eval_kind="cv", metric="roc_auc_ovo", folds=5,
            task="classification", target="activity", best_mean=0.81)
    _record(runs_root, "ml-evaluation", "reg run (r2, 5-fold)",
            eval_kind="cv", metric="r2", folds=5,
            task="regression", target="pIC50", best_mean=0.54)
    result = compare_runs(collect_runs("ml-evaluation"))
    assert len(result.groups) == 2
    assert all(not g.comparable for g in result.groups)
    assert len(result.refused) == 1
    a, b, why = result.refused[0]
    assert {a, b} == {"clf run (roc_auc, 5-fold)", "reg run (r2, 5-fold)"}
    assert "task" in why and "metric" in why
    text = comparison_summary(result)
    assert "not compared" in text
    assert "no ranking" in text


def test_different_folds_are_different_conditions(runs_root) -> None:
    for folds, label in ((2, "2-fold"), (5, "5-fold")):
        _record(runs_root, "ml-evaluation", label,
                eval_kind="cv", metric="roc_auc_ovo", folds=folds,
                task="classification", target="activity", best_mean=0.7)
    result = compare_runs(collect_runs("ml-evaluation"))
    assert len(result.groups) == 2
    assert result.refused and "folds" in result.refused[0][2]


# -- within-group comparison ---------------------------------------------------

def test_comparable_group_ranks_and_deltas(runs_root) -> None:
    _record(runs_root, "ml-evaluation", "descriptors board",
            eval_kind="cv", metric="roc_auc_ovo", folds=5,
            task="classification", target="activity",
            best_family="descriptors", best_mean=0.82)
    _record(runs_root, "ml-evaluation", "fingerprints board",
            eval_kind="cv", metric="roc_auc_ovo", folds=5,
            task="classification", target="activity",
            best_family="fingerprints", best_mean=0.75)
    result = compare_runs(collect_runs("ml-evaluation"))
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.comparable
    best, metric, higher = getattr(group, "_verdict")
    assert best == "descriptors board"
    assert metric == "roc_auc_ovo" and higher is True
    table = getattr(group, "_table")
    names = [d.name for d in table]
    assert "roc_auc_ovo" in names
    assert "Δ roc_auc_ovo vs best" in names
    deltas = next(d for d in table if d.name.startswith("Δ"))
    assert deltas.values["descriptors board"] == "0"
    assert deltas.values["fingerprints board"].startswith("0.07")  # trails by 0.07
    text = comparison_summary(result)
    assert "best on roc_auc_ovo: descriptors board" in text
    assert "fingerprints board" in text


def test_lower_is_better_metric_picks_the_minimum(runs_root) -> None:
    _record(runs_root, "ml-evaluation", "model A",
            eval_kind="holdout", metric="rmse", folds=None,
            task="regression", target="pIC50", best_mean=1.20)
    _record(runs_root, "ml-evaluation", "model B",
            eval_kind="holdout", metric="rmse", folds=None,
            task="regression", target="pIC50", best_mean=0.90)
    result = compare_runs(collect_runs("ml-evaluation"))
    group = result.groups[0]
    best, _, higher = getattr(group, "_verdict")
    assert best == "model B" and higher is False
    deltas = next(d for d in getattr(group, "_table") if d.name.startswith("Δ"))
    assert deltas.values["model B"] == "0"
    assert deltas.values["model A"].startswith("0.3")


def test_runs_without_metric_never_get_a_verdict(runs_root) -> None:
    _record(runs_root, "pkpd", "regimen · oral",
            route="oral", n_doses=1, c_max=8.2, t_max=1.5, auc=55.0)
    _record(runs_root, "pkpd", "regimen · iv",
            route="iv", n_doses=1, c_max=12.4, t_max=0.5, auc=48.0)
    result = compare_runs(collect_runs("pkpd"))
    # same key (no task/metric) → one group, side-by-side, no ranking
    assert len(result.groups) == 1
    group = result.groups[0]
    assert not group.comparable
    assert getattr(group, "_verdict", None) is None
    text = comparison_summary(result)
    assert "no ranking" in text
    assert "c_max" not in text  # unknown metrics are not invented
    # but the recorded numbers DO show as param rows
    assert "param:route" in text


# -- collection ---------------------------------------------------------------

def test_collect_runs_reads_every_family_newest_first(runs_root) -> None:
    _record(runs_root, "viz-export", "tiff export", kind="tiff", n=3)
    _record(runs_root, "docking", "campaign scan", campaign="demo", n=14)
    views = collect_runs()
    assert {v.family for v in views} == {"viz-export", "docking"}
    # newest-first across families (created timestamps are monotonic
    # in the filename order record_run writes)
    stamps = [v.created for v in views]
    assert stamps == sorted(stamps, reverse=True)


def test_collect_runs_skips_foreign_schema(runs_root) -> None:
    (runs_root / "ml-evaluation").mkdir(parents=True, exist_ok=True)
    (runs_root / "ml-evaluation" / "foreign.json").write_text(
        json.dumps({"schema": "something-else", "family": "ml-evaluation"}))
    _record(runs_root, "ml-evaluation", "real run",
            eval_kind="cv", metric="r2", folds=5,
            task="regression", target="pIC50", best_mean=0.5)
    views = collect_runs("ml-evaluation")
    assert len(views) == 1 and views[0].label == "real run"


def test_empty_state_is_honest(runs_root) -> None:
    assert collect_runs() == []
    result = compare_runs([])
    assert comparison_summary(result) == "No runs selected."
