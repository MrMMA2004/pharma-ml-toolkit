"""The GA backend exposes a partial result on cooperative Stop without checkpointing."""
import pytest

pytest.importorskip("rdkit.Chem")

from pharmaml.exceptions import JobCancelled
from pharmaml.generative import GeneticOptimizer, QedObjective


def test_genetic_stop_without_disk_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def progress(done, total, label):
        calls.append((done, total, label))
        return done < 2

    optimizer = GeneticOptimizer(QedObjective(), population_size=8, generations=4, seed=42,
                                 mutation="topology", crossover="splice", operator_selection="uniform")
    with pytest.raises(JobCancelled) as caught:
        optimizer.optimize(progress=progress)
    result = caught.value.partial_results
    assert calls == [(1, 4, "generative search"), (2, 4, "generative search")]
    assert result.metadata["partial"] is True and len(result.history) == 2
    assert result.smiles and result.to_dataframe().shape[0] == len(result.smiles)
    assert list(tmp_path.iterdir()) == []
