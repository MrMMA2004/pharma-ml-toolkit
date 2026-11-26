"""part5 §11 — the nine-card evidence contract: real or precisely empty.

Pinned here, per the master prompt's own demands:

* every one of the eight evidence cards has a precise empty state that
  names the action which fills it (never a blank, never a module count);
* every real preview number comes from a seeded run record, a loaded
  frame, or a parsed study — the collectors are fed REAL data in these
  tests and must echo it back exactly;
* the run records themselves are atomic, best-effort (never raise), and
  refuse foreign/corrupt payloads on read;
* no unseeded randomness anywhere in the preview surface (source scan).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pharmaml.studio import runs as runs_mod
from pharmaml.studio.evidence import (
    EMPTY_STATES,
    CardEvidence,
    collect_evidence,
    dataset_evidence,
    dl_evidence,
    docking_evidence,
    empty_evidence,
    generative_evidence,
    md_evidence,
    ml_evidence,
    pkpd_evidence,
    viz_evidence,
)

#: The eight evidence-bearing cards (Introduction keeps its version string).
EVIDENCE_CARDS = {
    "datasets",
    "machine-learning",
    "deep-learning",
    "generative-ai",
    "molecular-docking",
    "pkpd",
    "molecular-dynamics-analysis",
    "visualization",
}


@pytest.fixture()
def runs_home(tmp_path, monkeypatch):
    """An isolated runs directory so tests never touch the real home."""
    home = tmp_path / "studio-runs"
    monkeypatch.setenv(runs_mod.RUNS_DIR_ENV, str(home))
    return home


def _seed(family: str, label: str, summary: dict) -> None:
    assert runs_mod.record_run(family, label, summary) is not None


class TestRunRecords:
    def test_record_and_latest_round_trip(self, runs_home) -> None:
        _seed(
            "dl-training",
            "NumpyMLPRegressor on pIC50",
            {"epochs_run": 4, "history": [{"epoch": 1, "train_loss": 0.5}]},
        )
        records = runs_mod.latest_runs("dl-training")
        assert len(records) == 1
        record = records[0]
        assert record["schema"] == runs_mod.RUN_SCHEMA
        assert record["family"] == "dl-training"
        assert record["label"] == "NumpyMLPRegressor on pIC50"
        assert record["summary"]["epochs_run"] == 4
        assert record["created"]
        assert record["toolkit_version"]

    def test_newest_record_first(self, runs_home) -> None:
        _seed("viz-export", "a.png", {"format": "PNG"})
        _seed("viz-export", "b.png", {"format": "PNG"})
        records = runs_mod.latest_runs("viz-export")
        assert len(records) == 2
        assert records[0]["created"] >= records[1]["created"]

    def test_writing_never_raises_on_unwritable_home(self, tmp_path, monkeypatch) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where the runs dir would be", encoding="utf-8")
        monkeypatch.setenv(runs_mod.RUNS_DIR_ENV, str(blocker))
        assert runs_mod.record_run("viz-export", "x", {"a": 1}) is None

    def test_numpy_scalars_and_paths_serialize(self, runs_home) -> None:
        _seed(
            "viz-export",
            "panel",
            {"score": np.float64(0.5), "array": np.arange(3), "where": Path("/tmp/x.png")},
        )
        record = runs_mod.latest_runs("viz-export")[0]
        assert record["summary"]["score"] == 0.5
        assert record["summary"]["array"] == [0, 1, 2]
        assert record["summary"]["where"] == "/tmp/x.png"

    def test_foreign_schema_and_corrupt_files_are_skipped(self, runs_home) -> None:
        family_dir = runs_home / "viz-export"
        family_dir.mkdir(parents=True)
        (family_dir / "foreign.json").write_text(
            json.dumps({"schema": "something-else/9", "family": "viz-export"}), encoding="utf-8"
        )
        (family_dir / "corrupt.json").write_text("{not json", encoding="utf-8")
        assert runs_mod.latest_runs("viz-export") == []

    def test_run_families_counts_only_real_directories(self, runs_home) -> None:
        _seed("docking", "scan", {"n_ok": 1})
        _seed("docking", "scan2", {"n_ok": 2})
        _seed("pkpd", "regimen", {"auc": 1.0})
        assert runs_mod.run_families() == {"docking": 2, "pkpd": 1}

    def test_slugify(self) -> None:
        assert runs_mod.slugify("family benchmark · pIC50 (5-fold CV)") == (
            "family-benchmark-pic50-5-fold-cv"
        )
        assert runs_mod.slugify("!!!") == "run"
        assert runs_mod.slugify("") == "run"


class TestEmptyStates:
    def test_every_evidence_card_has_a_precise_empty_state(self) -> None:
        assert set(EMPTY_STATES) == EVIDENCE_CARDS

    def test_empty_states_name_the_filling_action(self) -> None:
        for card_id, text in EMPTY_STATES.items():
            assert text.startswith("No "), f"{card_id}: empty state must state what is absent"
            assert " — " in text, f"{card_id}: empty state must say what fills it"
            assert len(text) > 60, f"{card_id}: empty state too thin to be precise"

    def test_unknown_card_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown evidence card"):
            empty_evidence("definitely-not-a-card")

    def test_collectors_without_data_return_their_empty_states(self, runs_home) -> None:
        board = collect_evidence()
        assert set(board) == EVIDENCE_CARDS
        for card_id, evidence in board.items():
            assert isinstance(evidence, CardEvidence)
            assert evidence.empty, f"{card_id} must be honestly empty on a fresh home"
            if card_id == "machine-learning":
                # the registry count is real data that always exists: the
                # headline stays numeric (the part-4 pin) and the precise
                # empty state rides the detail line.
                assert "model families in the live registry" in evidence.headline
                assert evidence.detail == EMPTY_STATES[card_id]
            else:
                assert evidence.headline == EMPTY_STATES[card_id]


class TestDatasetEvidence:
    def test_no_frame_is_the_empty_state(self) -> None:
        evidence = dataset_evidence(None)
        assert evidence.empty and evidence.headline == EMPTY_STATES["datasets"]

    def test_real_frame_shape_and_missingness(self) -> None:
        frame = pd.DataFrame(
            {
                "smiles": ["CCO", "CCC", None, "CCN"],
                "pic50": [5.1, 6.2, np.nan, 4.4],
            }
        )
        evidence = dataset_evidence(frame, source="/data/qsar.csv", active_column="pic50")
        assert not evidence.empty
        text = evidence.text()
        assert "qsar.csv" in text
        assert "4 rows x 2 columns" in text
        assert "2 missing cells" in text and "25.0%" in text
        assert "smiles" in text  # SMILES detection is real, not assumed
        assert "min 4.4" in text and "median 5.1" in text and "max 6.2" in text
        assert evidence.data["rows"] == 4 and evidence.data["missing_cells"] == 2

    def test_categorical_column_shows_its_real_balance(self) -> None:
        frame = pd.DataFrame({"class": ["active", "active", "inactive", "inactive"]})
        evidence = dataset_evidence(frame, "toy.csv", active_column="class")
        assert "active 2" in evidence.text() and "inactive 2" in evidence.text()

    def test_empty_frame_is_the_empty_state(self) -> None:
        evidence = dataset_evidence(pd.DataFrame(), "empty.csv")
        assert evidence.empty and evidence.headline == EMPTY_STATES["datasets"]


class TestSavedRunEvidence:
    def test_ml_evaluation_cv(self, runs_home) -> None:
        _seed(
            "ml-evaluation",
            "family benchmark · pIC50 (5-fold CV)",
            {
                "eval_kind": "cv",
                "folds": 5,
                "families": 9,
                "metric": "r2",
                "best_mean": 0.612,
                "best_family": "morgan",
            },
        )
        evidence = ml_evidence()
        assert not evidence.empty
        text = evidence.text()
        assert "model families in the live registry" in text  # the registry stays real
        assert "5-fold CV over 9 families" in text
        assert "best r2 0.612 (morgan)" in text

    def test_ml_held_out_is_distinguished_from_cv(self, runs_home) -> None:
        _seed(
            "ml-evaluation",
            "held-out evaluation",
            {"eval_kind": "held-out", "metric": "rmse", "best_mean": 0.71, "best_family": "ridge"},
        )
        assert "held-out rmse 0.71 (ridge)" in ml_evidence().text()

    def test_dl_history_curves(self, runs_home) -> None:
        history = [
            {"epoch": i + 1, "train_loss": 1.0 - 0.9 * i / 39, "val_loss": 1.1 - 0.8 * i / 39}
            for i in range(40)
        ]
        _seed(
            "dl-training",
            "NumpyMLPRegressor on pIC50",
            {
                "engine": "numpy",
                "epochs_run": 40,
                "epochs_asked": 200,
                "history": history,
            },
        )
        evidence = dl_evidence()
        assert not evidence.empty
        text = evidence.text()
        assert "NumpyMLPRegressor on pIC50" in text
        assert "early stop at epoch 40" in text
        assert "train loss 1 -> 0.1" in text
        assert "val loss 0.3" in text
        assert len(evidence.data["history"]) == 40  # the curve rides the evidence

    def test_generative_sa_batch(self, runs_home) -> None:
        _seed(
            "generative",
            "SA batch",
            {
                "kind": "sa-batch",
                "n_scored": 24,
                "n_invalid": 1,
                "unique": 23,
                "mean": 2.41,
                "std": 0.58,
                "min": 1.2,
                "max": 4.1,
            },
        )
        text = generative_evidence().text()
        assert "24 scored / 1 invalid" in text
        assert "validity 96%" in text
        assert "23 unique canonical SMILES" in text
        assert "SA mean 2.41 ± 0.58" in text

    def test_generative_pareto_front(self, runs_home) -> None:
        _seed(
            "generative",
            "Pareto front",
            {
                "kind": "pareto-front",
                "objectives": ["sa_score", "qed"],
                "n_front": 7,
                "n_total": 40,
                "deepest": 3,
            },
        )
        assert "7 of 40 molecules" in generative_evidence().text()
        assert "sa_score, qed" in generative_evidence().text()

    def test_docking_campaign(self, runs_home) -> None:
        _seed(
            "docking",
            "campaign scan",
            {
                "n_folders": 12,
                "n_ok": 9,
                "n_incomplete": 2,
                "n_missing": 1,
                "top_ligand": "L07",
                "top_score": 0.834,
            },
        )
        text = docking_evidence().text()
        assert "12 folders" in text
        assert "9 complete, 2 incomplete, 1 missing log" in text
        assert "top ligand L07 (consensus 0.834)" in text

    def test_pkpd_regimen_with_units_and_parameters(self, runs_home) -> None:
        _seed(
            "pkpd",
            "regimen · oral",
            {
                "route": "oral",
                "n_doses": 5,
                "c_max": 8.423,
                "t_max": 2.1,
                "auc": 96.31,
                "duration_h": 48,
                "parameters": {"ka": 1.2, "cl": 3.1, "v2": 28.0},
            },
        )
        text = pkpd_evidence().text()
        assert "oral · 5 dose(s)" in text
        assert "Cmax 8.423 mg/L at 2.1 h" in text
        assert "AUC(0-48 h) 96.31 mg·h/L" in text
        assert "ka=1.2" in text and "v2=28" in text

    def test_viz_export(self, runs_home) -> None:
        _seed(
            "viz-export",
            "imagelab.png",
            {
                "name": "imagelab.png",
                "format": "PNG",
                "width": 1024,
                "height": 768,
                "size_bytes": 45210,
                "kind": "image panel",
            },
        )
        text = viz_evidence().text()
        assert "imagelab.png (PNG)" in text
        assert "1024x768 px" in text
        assert "45,210 bytes" in text


class TestMDEvidence:
    def test_no_study_is_the_empty_state(self) -> None:
        evidence = md_evidence(None)
        assert evidence.empty and evidence.headline == EMPTY_STATES["molecular-dynamics-analysis"]

    def test_loaded_study_reports_real_timelines(self) -> None:
        series = SimpleNamespace(
            time=np.linspace(0.0, 2.0, 5), values=np.array([0.098, 0.15, 0.22, 0.27, 0.312])
        )
        ligand = SimpleNamespace(metric_keys=["ligand_rmsd"], series={"ligand_rmsd": series})
        study = SimpleNamespace(ligand_names=["lig1"], ligand=lambda name: ligand)
        evidence = md_evidence(study)
        assert not evidence.empty
        text = evidence.text()
        assert "loaded study: 1 ligand(s) — lig1" in text
        assert "ligand_rmsd 0.098 -> 0.312" in text
        assert "n=5" in text

    def test_a_broken_study_degrades_to_the_empty_state(self) -> None:
        evidence = md_evidence(SimpleNamespace(ligand_names=None))  # type: ignore[arg-type]
        assert evidence.empty and evidence.headline == EMPTY_STATES["molecular-dynamics-analysis"]


class TestCollectBoard:
    def test_context_feeds_the_dataset_card(self, runs_home) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        board = collect_evidence(
            {"data_frame": frame, "data_source": "toy.csv", "data_column": "b"}
        )
        assert not board["datasets"].empty
        assert "3 rows x 2 columns" in board["datasets"].text()
        # the other cards stay honestly empty on this fresh home
        assert board["deep-learning"].empty

    def test_every_card_survives_an_exotic_context(self, runs_home) -> None:
        board = collect_evidence({"data_frame": object(), "md_study": 12345})
        assert set(board) == EVIDENCE_CARDS  # nothing raised, nothing lost


class TestNoUnseededRandomness:
    """§11: 'Never use arbitrary random values' — enforced by source scan.

    The scan covers every module on the preview path: the collectors, the
    runs writer, and the Qt grid that renders them.  ``random_state``
    (an honest seeded-estimator keyword) contains no dot and cannot trip
    the patterns.
    """

    FORBIDDEN = (
        re.compile(r"^\s*import random\b", re.MULTILINE),
        re.compile(r"^\s*from random\b", re.MULTILINE),
        re.compile(r"\bnp\.random\b"),
        re.compile(r"\brandom\.(?:uniform|randint|choice|random|sample|shuffle|rand|randn)\b"),
        re.compile(r"\btorch\.(?:rand|randn|randint)\b"),
        re.compile(r"\bdefault_rng\b"),
    )

    @pytest.mark.parametrize(
        "path",
        [
            Path("src/pharmaml/studio/__init__.py"),
            Path("src/pharmaml/studio/runs.py"),
            Path("src/pharmaml/studio/evidence.py"),
            Path("legacy/ui/nine_cards.py"),
        ],
    )
    def test_preview_surface_has_no_unseeded_randomness(self, path) -> None:
        source = path.read_text(encoding="utf-8")
        for pattern in self.FORBIDDEN:
            match = pattern.search(source)
            assert match is None, f"{path} uses unseeded randomness: {match!r}"
