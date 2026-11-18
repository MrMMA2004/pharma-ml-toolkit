"""Fail-closed genetic checkpoint identity and preservation tests."""
from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("rdkit.Chem")

from pharmaml.exceptions import PersistenceError
from pharmaml.generative import GeneticOptimizer, LogPBandObjective, MultiObjective, QedObjective


def optimizer(objective=None, **kwargs):
    return GeneticOptimizer(objective or QedObjective(), population_size=8, generations=4, seed=7, **kwargs)


def tree_bytes(path):
    return {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}


def rewrite_record(path, mutate):
    record = json.loads(path.read_text())
    mutate(record)
    record.pop("integrity")
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    record["integrity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(record), encoding="utf-8")


def test_changed_objective_parameters_and_corpus_refused_without_overwrite(tmp_path):
    home = tmp_path / "checkpoints"
    optimizer(LogPBandObjective(0, 1)).optimize(checkpoint_dir=home, corpus=["CCO", "CCN"])
    original = tree_bytes(home)
    for objective, corpus in ((LogPBandObjective(3, 4), ["CCO", "CCN"]),
                              (LogPBandObjective(0, 1), ["CCO", "CCC"])):
        with pytest.raises(PersistenceError, match="identity or version mismatch"):
            optimizer(objective).optimize(checkpoint_dir=home, corpus=corpus)
        assert tree_bytes(home) == original


def test_changed_multi_objective_weights_refused(tmp_path):
    home = tmp_path / "multi"
    objective = MultiObjective([QedObjective(), LogPBandObjective(1, 3)], weights=[1, 2])
    optimizer(objective).optimize(checkpoint_dir=home)
    before = tree_bytes(home)
    changed = MultiObjective([QedObjective(), LogPBandObjective(1, 3)], weights=[2, 1])
    with pytest.raises(PersistenceError, match="identity or version mismatch"):
        optimizer(changed).optimize(checkpoint_dir=home)
    assert tree_bytes(home) == before


def test_custom_or_modified_objective_cannot_use_checkpoints(tmp_path):
    class Custom(QedObjective):
        pass

    home = tmp_path / "custom"
    with pytest.raises(PersistenceError, match="custom and stateful"):
        optimizer(Custom()).optimize(checkpoint_dir=home)
    assert not home.exists()
    modified = QedObjective()
    modified.score = lambda _smiles: 0.5
    with pytest.raises(PersistenceError, match="modified objective"):
        optimizer(modified).optimize(checkpoint_dir=home)
    assert not home.exists()
    assert len(optimizer(Custom()).optimize().history) == 4


def test_invalid_and_legacy_files_never_fallback_or_quarantine(tmp_path):
    home = tmp_path / "broken"
    optimizer().optimize(checkpoint_dir=home)
    last = max(home.glob("*.json"))
    last.write_text("{broken", encoding="utf-8")
    before = tree_bytes(home)
    with pytest.raises(PersistenceError, match="invalid genetic checkpoint"):
        optimizer().optimize(checkpoint_dir=home)
    assert tree_bytes(home) == before
    last.rename(last.with_suffix(".json.corrupt"))
    before = tree_bytes(home)
    with pytest.raises(PersistenceError, match="remnants"):
        optimizer().optimize(checkpoint_dir=home)
    assert tree_bytes(home) == before


def test_legacy_fingerprint_and_malformed_state_fail_closed(tmp_path):
    home = tmp_path / "legacy"
    optimizer().optimize(checkpoint_dir=home)
    files = sorted(home.glob("*.json"))
    rewrite_record(files[-1], lambda record: record["state"].update(config_fp="legacy"))
    before = tree_bytes(home)
    with pytest.raises(PersistenceError, match="identity or version mismatch"):
        optimizer().optimize(checkpoint_dir=home)
    assert tree_bytes(home) == before

    home2 = tmp_path / "malformed"
    optimizer().optimize(checkpoint_dir=home2)
    last = max(home2.glob("*.json"))
    rewrite_record(last, lambda record: record["state"].update(history=[]))
    before = tree_bytes(home2)
    with pytest.raises(PersistenceError, match="invalid genetic checkpoint state"):
        optimizer().optimize(checkpoint_dir=home2)
    assert tree_bytes(home2) == before


def test_new_run_refuses_existing_checkpoint_dir(tmp_path):
    home = tmp_path / "fresh"
    optimizer().optimize(checkpoint_dir=home)
    before = tree_bytes(home)
    with pytest.raises(PersistenceError, match="choose a new directory"):
        optimizer().optimize(checkpoint_dir=home, checkpoint_resume=False)
    assert tree_bytes(home) == before


def test_changed_setup_during_run_preserves_prior_checkpoint(tmp_path):
    home = tmp_path / "changed-during-run"
    objective = LogPBandObjective(1, 3)

    def change_after_first(done, _total, _label):
        if done == 1:
            objective.lo = 0.0
        return True

    with pytest.raises(PersistenceError, match="setup changed during the run"):
        optimizer(objective).optimize(checkpoint_dir=home, progress=change_after_first)
    files = sorted(home.glob("*.json"))
    assert len(files) == 1 and files[0].name.endswith("000001.json")
    # The first checkpoint still belongs to the original objective, not the
    # mutated parameters; a matching caller can resume it.
    result = optimizer(LogPBandObjective(1, 3)).optimize(checkpoint_dir=home)
    assert len(result.history) == 4


def test_wrong_envelope_version_or_step_never_falls_back(tmp_path):
    for field in ("toolkit_version", "resume_unit"):
        home = tmp_path / field
        optimizer().optimize(checkpoint_dir=home)
        last = max(home.glob("*.json"))
        def mutate(record, field=field):
            envelope = record["envelope"]
            envelope[field] = "wrong-version" if field == "toolkit_version" else 0
        rewrite_record(last, mutate)
        before = tree_bytes(home)
        with pytest.raises(PersistenceError, match="genetic checkpoint"):
            optimizer().optimize(checkpoint_dir=home)
        assert tree_bytes(home) == before


def test_malformed_adaptive_book_is_not_partially_restored(tmp_path):
    home = tmp_path / "adaptive"
    optimizer(operator_selection="adaptive").optimize(checkpoint_dir=home)
    last = max(home.glob("*.json"))
    rewrite_record(last, lambda record: record["state"]["mutation_book"]["draws"].clear())
    before = tree_bytes(home)
    with pytest.raises(PersistenceError, match="invalid genetic checkpoint state"):
        optimizer(operator_selection="adaptive").optimize(checkpoint_dir=home)
    assert tree_bytes(home) == before


def test_save_failure_is_reported(tmp_path, monkeypatch):
    from pharmaml.utils.checkpoints import CheckpointManager

    def broken_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(CheckpointManager, "save", broken_save)
    with pytest.raises(PersistenceError, match="checkpoint write failed"):
        optimizer().optimize(checkpoint_dir=tmp_path / "disk-full")
