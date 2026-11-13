"""The Part-3 SS20.4/SS20.5 NumPy-trainer checkpoint-depth suite.

The contract (preserved in the worklog after rollback #27 ate the
first implementation): ``fit(checkpoint_dir=...)`` writes the COMPLETE
resumable state through the round-229 ``CheckpointManager`` (atomic,
sha-256 self-verified, JSON + base64 -- never pickle), and the
acceptance bar is BIT-EXACTNESS -- ``np.array_equal`` predictions for
interrupted-then-resumed vs uninterrupted, never ``allclose`` (which
would hide a rewound Adam bias correction or a shifted batch
permutation).  SS20.5's incompatibility contract: config + dataset
fingerprints, free ``epochs``/``patience`` budgets, discard-with-
warning (never fatal), broken homes degrade to warnings.
"""

from __future__ import annotations

import json
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from pharmaml.exceptions import ModelError
from pharmaml.ml.autograd import SGD, Adam, Tensor
from pharmaml.ml.numpy_neural import (
    NumpyMLPClassifier,
    NumpyMLPRegressor,
    _decode_optimizer_state,
    _encode_optimizer_state,
)


def _regression_data(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n, 3))
    # the noise keeps the validation loss climbing eventually, so early
    # stopping genuinely fires within the test budgets (the continuity
    # and checkpoint-at-the-stopped-epoch tests exercise the real path)
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.3 + 0.35 * rng.normal(size=n)
    return X, y


def _classification_data(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    X = np.vstack(
        [rng.normal(loc=-1.0, size=(n // 2, 3)), rng.normal(loc=1.0, size=(n - n // 2, 3))]
    )
    y = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    return X, y


def _fit(X, y, epochs: int, **kw):
    return NumpyMLPRegressor(hidden_layers=(12,), epochs=epochs, random_state=5, **kw).fit(X, y)


# ---------------------------------------------------------------------------
# the optimizer-state layer (SS20.4 "optimizer moment buffers")
# ---------------------------------------------------------------------------


class TestOptimizerState:
    def test_adam_round_trip_continues_bit_identically(self):
        rng = np.random.default_rng(0)
        params = [Tensor(rng.normal(size=(4, 2)), requires_grad=True)]
        first, second = Adam(params, lr=0.05), Adam(
            [Tensor(p.data.copy(), requires_grad=True) for p in params], lr=0.05
        )
        for _ in range(3):
            loss = ((first.params[0] * 2.0) ** 2).sum()
            first.zero_grad()
            loss.backward()
            first.step()
        saved = first.state_dict()
        # the weights ride the network snapshot in the real wiring; the
        # optimizer state alone needs the post-step weights to be meaningful
        second.params[0].data = first.params[0].data.copy()
        second.load_state_dict(_decode_optimizer_state(_encode_optimizer_state(saved)))
        for _ in range(4):
            for opt in (first, second):
                loss = ((opt.params[0] * 2.0) ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        assert np.array_equal(first.params[0].data, second.params[0].data)
        assert second.state_dict()["step_count"] == 7

    def test_sgd_velocity_restores(self):
        rng = np.random.default_rng(1)
        params = [Tensor(rng.normal(size=(3,)), requires_grad=True)]
        opt = SGD(params, lr=0.1, momentum=0.9)
        for _ in range(2):
            loss = (opt.params[0] ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        state = opt.state_dict()
        assert state["step_count"] == 2
        assert "_velocity" in state
        assert np.array_equal(state["_velocity"][0], opt._velocity[0])

    def test_missing_step_count_raises(self):
        opt = SGD([Tensor(np.ones(2), requires_grad=True)], lr=0.1)
        with pytest.raises(ModelError, match="missing 'step_count'"):
            opt.load_state_dict({"_velocity": []})

    def test_non_int_step_count_raises(self):
        opt = SGD([Tensor(np.ones(2), requires_grad=True)], lr=0.1)
        with pytest.raises(ModelError, match="step_count must be an int"):
            opt.load_state_dict({"step_count": True})
        with pytest.raises(ModelError, match="step_count must be an int"):
            opt.load_state_dict({"step_count": "3"})

    def test_foreign_buffer_name_raises(self):
        opt = SGD([Tensor(np.ones(2), requires_grad=True)], lr=0.1)
        with pytest.raises(ModelError, match="has none"):
            opt.load_state_dict({"step_count": 1, "_m": []})

    def test_wrong_buffer_shape_raises(self):
        opt = Adam([Tensor(np.ones((4, 2)), requires_grad=True)], lr=0.1)
        bad = {"step_count": 1, "_m": [np.ones(5)], "_v": [np.ones((4, 2))]}
        with pytest.raises(ModelError, match="does not match the live parameter"):
            opt.load_state_dict(bad)


# ---------------------------------------------------------------------------
# the bit-exactness acceptance bar (SS20.4)
# ---------------------------------------------------------------------------


class TestBitExactResume:
    def test_adam_regressor_resume_is_bit_exact(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            part = _fit(X, y, 5, checkpoint_dir=home, checkpoint_every=3)
            full = _fit(X, y, 9, checkpoint_dir=home, checkpoint_every=3)
            assert full.n_iter_ == 9
            del part
        uninterrupted = _fit(X, y, 9)
        assert np.array_equal(full.predict(X), uninterrupted.predict(X))

    def test_sgd_momentum_resume_is_bit_exact(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 5, optimizer="sgd", momentum=0.9, checkpoint_dir=home)
            full = _fit(X, y, 9, optimizer="sgd", momentum=0.9, checkpoint_dir=home)
        uninterrupted = _fit(X, y, 9, optimizer="sgd", momentum=0.9)
        assert np.array_equal(full.predict(X), uninterrupted.predict(X))

    def test_classifier_path_resume_is_bit_exact(self):
        X, y = _classification_data()
        with tempfile.TemporaryDirectory() as home:
            NumpyMLPClassifier(
                hidden_layers=(8,), epochs=5, random_state=2,
                checkpoint_dir=home, checkpoint_every=2,
            ).fit(X, y)
            full = NumpyMLPClassifier(
                hidden_layers=(8,), epochs=10, random_state=2, checkpoint_dir=home
            ).fit(X, y)
        uninterrupted = NumpyMLPClassifier(
            hidden_layers=(8,), epochs=10, random_state=2
        ).fit(X, y)
        assert np.array_equal(full.predict(X), uninterrupted.predict(X))

    def test_chained_resume_5_14_24_is_bit_exact(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 5, checkpoint_dir=home)
            _fit(X, y, 14, checkpoint_dir=home)
            full = _fit(X, y, 24, checkpoint_dir=home)
        uninterrupted = _fit(X, y, 24)
        assert full.n_iter_ == 24
        assert np.array_equal(full.predict(X), uninterrupted.predict(X))

    def test_dropout_resume_is_bit_exact(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 6, dropout=0.2, checkpoint_dir=home, checkpoint_every=3)
            full = _fit(X, y, 12, dropout=0.2, checkpoint_dir=home)
        uninterrupted = _fit(X, y, 12, dropout=0.2)
        assert np.array_equal(full.predict(X), uninterrupted.predict(X))


# ---------------------------------------------------------------------------
# early-stopping continuity
# ---------------------------------------------------------------------------


class TestEarlyStoppingContinuity:
    def test_resumed_run_stops_at_the_same_epoch(self):
        X, y = _regression_data(60)
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 4, learning_rate=0.05, patience=2, checkpoint_dir=home)
            full = _fit(X, y, 80, learning_rate=0.05, patience=2, checkpoint_dir=home)
        uninterrupted = _fit(X, y, 80, learning_rate=0.05, patience=2)
        assert full.n_iter_ < 80  # early stopping genuinely fired
        assert full.n_iter_ == uninterrupted.n_iter_
        assert full.history_[-1]["epoch"] == uninterrupted.history_[-1]["epoch"]
        assert np.array_equal(full.predict(X), uninterrupted.predict(X))

    def test_history_matches_epoch_for_epoch(self):
        X, y = _regression_data(60)
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 4, learning_rate=0.05, patience=2, checkpoint_dir=home)
            full = _fit(X, y, 80, learning_rate=0.05, patience=2, checkpoint_dir=home)
        uninterrupted = _fit(X, y, 80, learning_rate=0.05, patience=2)
        for restored, expected in zip(full.history_, uninterrupted.history_, strict=True):
            assert restored["epoch"] == expected["epoch"]
            assert restored["train_loss"] == expected["train_loss"]
            assert restored["val_loss"] == expected["val_loss"]


# ---------------------------------------------------------------------------
# the SS20.5 incompatibility / hygiene contract
# ---------------------------------------------------------------------------


class TestResumeContract:
    def test_budget_reached_restores_and_trains_nothing(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            part = _fit(X, y, 5, checkpoint_dir=home)
            again = _fit(X, y, 5, checkpoint_dir=home)
            assert again.n_iter_ == 5
            assert len(again.history_) == 5
            assert np.array_equal(again.predict(X), part.predict(X))

    def test_resume_disabled_fresh_while_still_writing(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 4, checkpoint_dir=home)
            fresh = _fit(X, y, 9, checkpoint_dir=home, checkpoint_resume=False)
            assert fresh.n_iter_ == 9  # trained every epoch, nothing resumed
            files = list(Path(home).glob("numpy-mlp.step-*.json"))
            assert any("step-000009" in f.name for f in files)

    @contextmanager
    def _logs_visible(self):
        """Re-enable propagation on the ``pharmaml`` logger for caplog.

        When CLI tests run earlier in the same process they call
        ``configure_logging``, which sets ``propagate=False`` on the
        ``pharmaml`` root logger; pytest's caplog handler lives on the
        true root logger, so records only reach it while propagation
        is on (restored on exit).  Same dance as
        ``tests/test_docking.py`` and ``tests/test_ml_pipeline.py``.
        """
        pharma_root = logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        pharma_root.propagate = True
        try:
            yield
        finally:
            pharma_root.propagate = prev_propagate

    def test_incompatible_config_discards_with_warning(self, caplog):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 5, checkpoint_dir=home)
            with self._logs_visible():
                with caplog.at_level("WARNING", logger="pharmaml.ml.numpy_neural"):
                    full = _fit(X, y, 9, learning_rate=1e-2, checkpoint_dir=home)
        assert "different config/dataset" in caplog.text
        assert full.n_iter_ == 9  # started fresh, ran every epoch

    def test_different_dataset_discards_with_warning(self, caplog):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 5, checkpoint_dir=home)
            X2, y2 = _regression_data(50)
            with self._logs_visible():
                with caplog.at_level("WARNING", logger="pharmaml.ml.numpy_neural"):
                    full = _fit(X2, y2, 9, checkpoint_dir=home)
        assert "different config/dataset" in caplog.text
        assert full.n_iter_ == 9

    def test_corrupt_checkpoint_quarantined_fresh_start(self, caplog):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 5, checkpoint_dir=home)
            newest = max(Path(home).glob("numpy-mlp.step-*.json"))
            newest.write_bytes(newest.read_bytes()[:40])  # truncate mid-record
            with caplog.at_level("WARNING", logger="pharmaml.ml.numpy_neural"):
                full = _fit(X, y, 9, checkpoint_dir=home)
            assert full.n_iter_ == 9  # degraded to fresh, never fatal
            assert list(Path(home).glob("*.corrupt"))

    def test_broken_home_degrades_to_warning(self, caplog, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("a file where the checkpoint home should be")
        X, y = _regression_data()
        with caplog.at_level("WARNING", logger="pharmaml.ml.numpy_neural"):
            full = _fit(X, y, 6, checkpoint_dir=str(blocker))
        assert full.n_iter_ == 6
        assert np.isfinite(full.predict(X)).all()

    def test_scan_failure_degrades_to_warning(self, caplog, tmp_path):
        X, y = _regression_data()
        home = tmp_path / "home"
        home.mkdir()
        (home / "numpy-mlp.step-000003.json").write_bytes(b"\x00binary noise")
        with caplog.at_level("WARNING", logger="pharmaml.ml.numpy_neural"):
            full = _fit(X, y, 6, checkpoint_dir=str(home))
        assert full.n_iter_ == 6  # the binary file is quarantined, run is fresh


class TestWriteHygiene:
    def test_cadence_and_final_epoch_writes(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 6, checkpoint_dir=home, checkpoint_every=2)
            steps = sorted(
                int(f.name.split("step-")[1].split(".")[0])
                for f in Path(home).glob("numpy-mlp.step-*.json")
            )
            assert steps == [2, 4, 6]

    def test_rotation_keeps_newest_only(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 8, checkpoint_dir=home, checkpoint_every=1)
            steps = sorted(
                int(f.name.split("step-")[1].split(".")[0])
                for f in Path(home).glob("numpy-mlp.step-*.json")
            )
            assert steps == [6, 7, 8]  # the manager's default keep=3

    def test_early_stop_final_epoch_checkpoint(self):
        X, y = _regression_data(60)
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 80, learning_rate=0.05, patience=2, checkpoint_dir=home, checkpoint_every=100)
            steps = sorted(
                int(f.name.split("step-")[1].split(".")[0])
                for f in Path(home).glob("numpy-mlp.step-*.json")
            )
            assert len(steps) >= 1
            assert steps[0] < 80  # the early-stopped epoch, not the budget

    def test_checkpoint_files_are_json_never_pickle(self):
        X, y = _regression_data()
        with tempfile.TemporaryDirectory() as home:
            _fit(X, y, 5, checkpoint_dir=home)
            newest = max(Path(home).glob("numpy-mlp.step-*.json"))
            record = json.loads(newest.read_text(encoding="utf-8"))
            assert record["schema"] == "pharmaml-checkpoint/1"
            state = record["state"]
            assert state["config_fp"] and state["dataset_fp"]
            assert "rng_state" in state and "optimizer" in state
            for layer in state["weights"]:
                assert layer["w"]["dtype"] == "float64"
                assert isinstance(layer["w"]["data"], str)
