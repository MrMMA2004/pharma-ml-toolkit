"""SA reward integration in the generative CLI (real RDKit, no torch)."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

from pharmaml.chem.sa_scoring import sa_score
from pharmaml.cli import _generative_objective, main
from pharmaml.exceptions import DependencyError
from pharmaml.generative import GenerationResult, MultiObjective


def test_ga_sa_reward_and_pareto_output(tmp_path, capsys):
    destination = tmp_path / "candidates.csv"
    code = main([
        "generative", "run", "--method", "ga", "--recipe", "qed",
        "--sa-weight", "2", "--iterations", "2", "--population", "8",
        "--seed", "42", "--pareto", "--out", str(destination),
    ])
    output = capsys.readouterr()
    assert code == 0, output.err
    assert "SA reward (weight 2)" in output.out
    assert "0-1 reward, not raw SA" in output.out
    assert "pareto" in output.out
    frame = pd.read_csv(destination)
    assert len(frame) > 0 and {"qed", "sa_score"}.issubset(frame.columns)
    # The output is an optimization reward, not the raw Ertl score.
    for row in frame.itertuples():
        raw = sa_score(row.smiles).score
        assert raw is not None
        reward = min(max((7.0 - raw) / 4.0, 0.0), 1.0)
        assert row.sa_score == pytest.approx(reward)
        assert row.score == pytest.approx((row.qed * reward**2) ** (1 / 3))


def test_rl_uses_the_same_weighted_objective(monkeypatch, capsys):
    import pharmaml.generative as gen

    seen = {}

    class FakeRL:
        def __init__(self, objective, **_kwargs):
            seen["objective"] = objective

        def optimize(self):
            total, breakdown = seen["objective"].score_with_breakdown("CCO")
            return GenerationResult(smiles=["CCO"], scores=[total], breakdown=[breakdown])

    monkeypatch.setattr(gen, "ReinventOptimizer", FakeRL)
    code = main(["generative", "run", "--method", "rl", "--recipe", "druglike",
                 "--sa-weight", "0.4", "--iterations", "1"])
    assert code == 0, capsys.readouterr().err
    objective = seen["objective"]
    assert isinstance(objective, MultiObjective)
    assert [obj.name for obj in objective.objectives] == ["qed", "mw_band", "tpsa_band", "sa_score"]
    assert objective.weights == [1.0, 1.0, 1.0, 0.4]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_weight_rejected_before_generation(value, capsys):
    code = main(["generative", "run", "--sa-weight", value])
    output = capsys.readouterr()
    assert code == 1
    assert "--sa-weight must be a finite positive number" in output.err


def test_similar_recipe_retains_its_weights_when_sa_is_enabled(monkeypatch, capsys):
    import pharmaml.generative as gen

    seen = {}

    class FakeRL:
        def __init__(self, objective, **_kwargs):
            seen["objective"] = objective

        def optimize(self):
            score, breakdown = seen["objective"].score_with_breakdown("CCO")
            return GenerationResult(smiles=["CCO"], scores=[score], breakdown=[breakdown])

    monkeypatch.setattr(gen, "ReinventOptimizer", FakeRL)
    code = main(["generative", "run", "--method", "rl", "--recipe", "similar",
                 "--seed-smiles", "CCO", "--sa-weight", "0.5", "--iterations", "1"])
    assert code == 0, capsys.readouterr().err
    assert seen["objective"].weights == [0.7, 0.3, 0.5]
    assert [item.name for item in seen["objective"].objectives] == ["similarity", "qed", "sa_score"]


def test_existing_recipe_weights_and_defaults_are_unchanged(capsys):
    objective, _ = _generative_objective("similar", ["CCO"])
    assert objective.weights == [0.7, 0.3]
    assert [item.name for item in objective.objectives] == ["similarity", "qed"]
    code = main(["generative", "run", "--recipe", "qed", "--pareto"])
    assert code == 1
    assert "single objective" in capsys.readouterr().err


def test_missing_rdkit_gives_dependency_error(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("RDKit blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(DependencyError, match="pip install"):
        sa_score("CCO")
