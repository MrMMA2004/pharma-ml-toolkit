"""§17A workflow templates + §17B sequential pipeline engine — the pins.

Two contracts are pinned:

Engine (§17B)
    stages thread artifacts, run in order, time themselves, fail fast
    while preserving partial outcomes, refuse to clobber artifact keys,
    and gate their optional dependencies behind the actionable
    DependencyError (never a silent empty).

Templates (§17A)
    all seven named templates exist, declare their owning card, and
    ACTUALLY RUN end-to-end against the bundled example data — every
    stage calls the existing backend implementation it names in its
    description (a template that named an API it does not call is the
    §18 "honesty" violation these tests exist to catch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DependencyError
from pharmaml.studio.pipelines import (
    TEMPLATES,
    Pipeline,
    Stage,
    build_template,
    run_pipeline,
    template_documentation,
    template_names,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data"


# -- engine -------------------------------------------------------------------

def test_stage_requires_identifier_name() -> None:
    with pytest.raises(ValueError):
        Stage(name="not a name!", fn=lambda a: {})
    Stage(name="valid_name", fn=lambda a: {})  # fine


def test_stages_thread_artifacts_in_order() -> None:
    calls: list[str] = []

    def make(n: str, produces: dict):
        def fn(artifacts: dict) -> dict:
            calls.append(n)
            return produces
        return fn

    def see_c(artifacts: dict) -> dict:
        calls.append("c")
        return {"z": artifacts["x"] + artifacts["y"]}

    result = run_pipeline("threading", [
        Stage(name="a", fn=make("a", {"x": 1})),
        Stage(name="b", fn=make("b", {"y": 1})),
        Stage(name="c", fn=see_c),
    ])
    assert calls == ["a", "b", "c"]
    assert result.ok
    assert result.artifacts == {"x": 1, "y": 1, "z": 2}
    assert [s.produced for s in result.stages] == [("x",), ("y",), ("z",)]
    assert result.seconds >= 0.0


def test_fail_fast_preserves_partial_outcomes() -> None:
    def boom(artifacts: dict) -> dict:
        raise RuntimeError("stage two exploded")

    with pytest.raises(RuntimeError, match="stage two exploded"):
        run_pipeline("failing", [
            Stage(name="one", fn=lambda a: {"x": 1}),
            Stage(name="two", fn=boom),
            Stage(name="three", fn=lambda a: {"never": True}),
        ])
    # reached via Pipeline.run to check the result plumbing instead


def test_pipeline_run_records_partial_failure() -> None:
    def boom(artifacts: dict) -> dict:
        raise RuntimeError("nope")

    pipe = Pipeline(name="partial", stages=(
        Stage(name="one", fn=lambda a: {"x": 1}),
        Stage(name="two", fn=boom),
    ))
    captured: list = []

    def keep(outcome) -> None:
        captured.append(outcome)

    with pytest.raises(RuntimeError):
        pipe.run(on_stage=keep)
    assert len(captured) == 2
    assert captured[0].ok and captured[0].produced == ("x",)
    assert not captured[1].ok and "nope" in captured[1].error


def test_duplicate_artifact_key_is_a_loud_error() -> None:
    with pytest.raises(KeyError, match="would overwrite"):
        run_pipeline("clobber", [
            Stage(name="a", fn=lambda a: {"x": 1}),
            Stage(name="b", fn=lambda a: {"x": 2}),
        ])


def test_non_dict_stage_return_is_a_loud_error() -> None:
    with pytest.raises(TypeError, match="returned list"):
        run_pipeline("badreturn", [Stage(name="a", fn=lambda a: [1, 2])])


def test_missing_known_dependency_uses_a_source_local_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    stage = Stage(name="needs_chemistry", fn=lambda a: {}, requires=("rdkit",))
    with pytest.raises(
        DependencyError,
        match=r"python -m pip install -e '\.\[chem\]'",
    ):
        run_pipeline("deps", [stage])


def test_missing_unknown_dependency_does_not_guess_a_package_command() -> None:
    stage = Stage(
        name="needs_nothing_real",
        fn=lambda a: {},
        requires=("definitely_not_a_module_xyz",),
    )
    with pytest.raises(DependencyError, match="supported package manager"):
        run_pipeline("deps", [stage])


def test_on_stage_callback_sees_every_outcome() -> None:
    seen: list[str] = []
    run_pipeline("cb", [
        Stage(name="a", fn=lambda a: {"x": 1}),
        Stage(name="b", fn=lambda a: {"y": 2}),
    ], on_stage=lambda o: seen.append(o.stage))
    assert seen == ["a", "b"]


def test_result_json_is_complete_and_stable() -> None:
    import json

    result = run_pipeline("json", [Stage(name="a", fn=lambda a: {"x": 1})])
    payload = json.loads(result.to_json())
    assert payload["pipeline"] == "json"
    assert payload["ok"] is True
    assert payload["stages"][0]["produced"] == ["x"]
    assert payload["stages"][0]["error"] is None
    assert payload["artifact_keys"] == ["x"]


# -- template registry --------------------------------------------------------

def test_the_seventeen_a_templates_exist_with_cards() -> None:
    assert template_names() == [
        "qsar_model_development",
        "molecular_descriptor_generation",
        "drug_response_prediction",
        "molecular_generation_screening",
        "docking_preparation_analysis",
        "pk_pd_modeling",
        "md_analysis",
    ]
    for name, entry in TEMPLATES.items():
        assert entry["card"], name
        assert entry["summary"], name
    assert "[Machine Learning]" in template_documentation()


def test_unknown_template_is_a_loud_error() -> None:
    with pytest.raises(ValueError, match="known templates"):
        build_template("definitely_not_a_template")


# -- templates that actually run ----------------------------------------------

def test_qsar_template_runs_end_to_end(tmp_path: Path) -> None:
    pipe = build_template(
        "qsar_model_development",
        out_dir=tmp_path, data_path=EXAMPLES / "qsar_classification.csv",
        target=None, models=["random_forest", "logistic_regression"], cv=3)
    assert [s.name for s in pipe.stages] == [
        "dataset", "preprocessing", "training", "evaluation", "export"]
    result = pipe.run()
    assert result.ok
    assert result.artifacts["metric_name"] == "balanced_accuracy"
    assert 0.0 <= result.artifacts["metric_value"] <= 1.0
    assert result.artifacts["heldout_n"] > 0
    export = Path(result.artifacts["exports"]["leaderboard"])
    assert export.exists() and export.stat().st_size > 0


def test_descriptors_template_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pipe = build_template(
        "molecular_descriptor_generation",
        out_dir=tmp_path, data_path=EXAMPLES / "seed_molecules.csv",
        smiles_col="smiles", kind="morgan")
    result = pipe.run()
    assert result.ok
    assert result.artifacts["n_molecules"] > 0
    assert result.artifacts["n_features"] > 0


def test_response_template_runs_end_to_end(tmp_path: Path) -> None:
    # drug response = the bundled screen + expression matrix (the
    # notebook's own chain), predicting LN_IC50 from expression
    pipe = build_template(
        "drug_response_prediction",
        out_dir=tmp_path,
        screen_path=EXAMPLES / "cellline_screen.csv",
        expression_path=EXAMPLES / "cellline_expression.csv",
        compound="erlotinib",
        models=["random_forest", "ridge"], cv=3)
    result = pipe.run()
    assert result.ok
    assert result.artifacts["metric_name"] == "r2"
    assert result.artifacts["heldout_n"] > 0


def test_generative_template_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pipe = build_template(
        "molecular_generation_screening",
        out_dir=tmp_path, objective="qed", generations=3, seed=42)
    result = pipe.run()
    assert result.ok
    board = result.artifacts["leaderboard"]
    assert {"smiles", "score"} <= set(board.columns)
    assert len(board) > 0
    # ranked best-first
    assert board["score"].is_monotonic_decreasing


def test_generative_template_rejects_unknown_objective(tmp_path: Path) -> None:
    pipe = build_template(
        "molecular_generation_screening",
        out_dir=tmp_path, objective="not_an_objective", generations=2, seed=1)
    with pytest.raises(ValueError, match="unknown objective"):
        pipe.run()


def test_pkpd_template_runs_end_to_end(tmp_path: Path) -> None:
    pipe = build_template(
        "pk_pd_modeling",
        out_dir=tmp_path, ka=1.2, ke=0.25, vd=40.0, dose=250.0)
    result = pipe.run()
    assert result.ok
    art = result.artifacts
    assert art["tmax_h"] == pytest.approx(1.6512, abs=1e-3)
    assert art["cmax"] > 0.0
    assert art["half_life_h"] == pytest.approx(
        np.log(2) / 0.25, abs=1e-6)
    assert art["auc"] > 0.0
    curve = pd.read_csv(art["exports"]["pk_curve"])
    assert {"time_h", "concentration"} == set(curve.columns)


def test_md_template_runs_end_to_end(tmp_path: Path) -> None:
    pipe = build_template(
        "md_analysis", out_dir=tmp_path, n_frames=20, seed=7)
    result = pipe.run()
    assert result.ok
    rmsd = result.artifacts["rmsd"]
    assert len(rmsd) == 20
    assert rmsd[0] == pytest.approx(0.0)  # first frame is the reference
    out = pd.read_csv(result.artifacts["exports"]["rmsd"])
    assert len(out) == 20


def test_docking_template_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("meeko")
    # the bundled ligand library (SMILES csv) + the demo ligand PDBQT
    # as the grid-box reference
    ligands = EXAMPLES / "ligand_library.csv"
    receptor = EXAMPLES / "contacts_demo_ligand.pdbqt"
    pipe = build_template(
        "docking_preparation_analysis",
        out_dir=tmp_path, ligands_path=ligands, receptor_pdbqt=receptor,
        smiles_col="SMILES")
    result = pipe.run()
    assert result.ok
    assert result.artifacts["n_prepared"] >= 1
    assert result.artifacts["grid_box"] is not None
