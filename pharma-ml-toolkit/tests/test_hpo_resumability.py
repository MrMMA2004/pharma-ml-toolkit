"""The §13 exact-restore proofs for the resumable HPO study.

Part5-continuation round 8: ``tune_model(checkpoint_dir=...)`` is the
third of the six pending §13 operation classes to earn registry entry.
These tests are the evidence: an interrupted-then-resumed study must
equal an uninterrupted one BIT FOR BIT -- trial history, best params,
best score.  The round's specific hazard: replaying the trial history
alone is NOT bit-exact (the TPE sampler draws candidates from its
``random.Random``, whose state advances per trial), so the checkpoint
must carry the RNG state too; one test pins exactly that distinction.
A cooperative stop preserves the partial report on the exception, and
a stale checkpoint (different model/space/sampler/seed or different
data) is refused -- exact restore or nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pharmaml.exceptions import JobCancelled
from pharmaml.ml.hpo import SearchSpace, tune_model
from pharmaml.utils.checkpoint_schema import (
    ENVELOPE_KEY,
    OPERATION_REGISTRY,
    OperationSpec,
    validate_checkpoint,
)
from pharmaml.utils.checkpoints import CheckpointManager

N_TRIALS = 14  # enough to cross the TPE startup (10) into guided sampling


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _space() -> SearchSpace:
    return SearchSpace.uniform("alpha", 0.01, 1.0, log=True)


def _tune(X: np.ndarray, y: np.ndarray, **kwargs: object):
    return tune_model(
        "ridge_classifier", _space(), X, y, task="classification",
        cv=2, n_trials=N_TRIALS, sampler="tpe", seed=1, **kwargs
    )


class TestHPOResume:
    def test_resumed_study_is_bit_exact(self, tmp_path: Path) -> None:
        """THE registry proof: interrupted + resumed == uninterrupted,
        bit for bit -- history, best params, best score."""
        X, y = _data()
        home = tmp_path / "ckpt"
        # interrupt: compute everything but pretend the process died --
        # checkpoint_resume=False forces a from-scratch run that still
        # writes checkpoints (the fresh-run twin)
        fresh = _tune(X, y, checkpoint_dir=home, checkpoint_resume=False)
        assert fresh.n_trials == N_TRIALS
        # the resume: same inputs, resume allowed, and the progress
        # callback counts how many trials were actually re-evaluated
        reevaluated: list[int] = []
        resumed = _tune(
            X, y, checkpoint_dir=home,
            progress=lambda done, total, note: reevaluated.append(done),
        )
        # every trial was already in the checkpoint: NOTHING re-evaluated
        assert reevaluated == []
        # bit-identical outputs
        assert resumed.history == fresh.history
        assert resumed.best_params == fresh.best_params
        assert resumed.best_score == fresh.best_score
        assert resumed.n_trials == fresh.n_trials
        # and identical to the checkpoint-less historical call
        plain = _tune(X, y)
        assert resumed.history == plain.history
        assert resumed.best_params == plain.best_params
        assert resumed.best_score == plain.best_score

    def test_partial_checkpoint_resumes_only_the_tail(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint from a genuinely partial run: the resume
        evaluates only the remaining trials and still matches bit for
        bit."""
        X, y = _data()
        home = tmp_path / "partial"
        with pytest.raises(JobCancelled):
            _tune(
                X, y, checkpoint_dir=home,
                progress=lambda done, total, note: False if done >= 5 else None,
            )
        steps = sorted(home.glob("hpo-study.step-*.json"))
        assert steps, "the partial run must have written checkpoints"
        stopped_at = max(int(s.stem.split(".step-")[1]) for s in steps)
        assert stopped_at == 5
        # resume and count what was re-evaluated: the tail past 5
        reevaluated: list[int] = []
        report = _tune(
            X, y, checkpoint_dir=home,
            progress=lambda done, total, note: reevaluated.append(done),
        )
        assert reevaluated, "the tail after the checkpoint must run"
        assert reevaluated[0] == 6  # trial 6 is the first after the resume
        plain = _tune(X, y)
        assert report.history == plain.history
        assert report.best_params == plain.best_params
        assert report.best_score == plain.best_score

    def test_resume_requires_the_rng_state_not_just_history(
        self, tmp_path: Path
    ) -> None:
        """The round's specific hazard, pinned: replaying the trial
        history WITHOUT the RNG state produces a DIFFERENT continuation
        (the TPE sampler's candidate draws depend on the rng state).
        The checkpoint therefore carries both, and the resume is exact
        -- this test proves the distinction is real by showing the
        naive replay diverges."""
        from pharmaml.ml.hpo import HPOStudy

        # an uninterrupted 14-trial study vs 5-then-9 with history-only
        # replay: the continuation MUST differ (that is why capture_state
        # persists the rng state)
        space = _space()

        def obj(p: object) -> float:
            return abs(float(p["alpha"]) - 0.1)  # deterministic objective

        full = HPOStudy(objective=obj, space=space, sampler="tpe",
                        seed=1, n_startup=4)
        full.run(N_TRIALS)
        part = HPOStudy(objective=obj, space=space, sampler="tpe",
                        seed=1, n_startup=4)
        part.run(5)
        naive = HPOStudy(objective=obj, space=space, sampler="tpe",
                         seed=1, n_startup=4)
        for t in part.trials:
            naive._engine._trials.append(t)  # history only, NO rng state
        naive.run(N_TRIALS - 5)
        with_rng = HPOStudy(objective=obj, space=space, sampler="tpe",
                            seed=1, n_startup=4)
        with_rng.restore_state(part.capture_state())  # history + rng state
        with_rng.run(N_TRIALS - 5)
        # the naive replay diverges somewhere in the tail...
        naive_tail = [t.params for t in naive.trials[5:]]
        full_tail = [t.params for t in full.trials[5:]]
        assert naive_tail != full_tail
        # ...while the rng-state restore matches the uninterrupted run
        assert with_rng.trials == full.trials

    def test_cancel_preserves_partial_report_on_the_exception(
        self, tmp_path: Path
    ) -> None:
        """§13: preserve completed partial results after cancellation --
        the partial TuneReport rides JobCancelled."""
        X, y = _data()
        home = tmp_path / "stop"
        with pytest.raises(JobCancelled) as caught:
            _tune(
                X, y, checkpoint_dir=home,
                progress=lambda done, total, note: False if done >= 7 else None,
            )
        partial = caught.value.partial_results
        assert partial is not None
        assert partial.n_trials == 7
        # the partial best is the best of the first 7 trials of the
        # full run (bit-exact prefix semantics)
        plain = _tune(X, y)
        first7 = plain.history[:7]
        assert partial.history == first7
        best7 = min(first7, key=lambda t: float(t.value or np.inf))
        assert partial.best_params == dict(best7.params)
        assert partial.best_score == -float(best7.value or np.inf)

    def test_stale_config_checkpoint_refuses_to_resume(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint from a different search config (seed, sampler,
        space bounds, n_trials, model): exact restore or nothing."""
        X, y = _data()
        home = tmp_path / "stale"
        _tune(X, y, checkpoint_dir=home, checkpoint_resume=False)
        reevaluated: list[int] = []
        # different seed
        tune_model(
            "ridge_classifier", _space(), X, y, task="classification",
            cv=2, n_trials=N_TRIALS, sampler="tpe", seed=2,
            checkpoint_dir=home,
            progress=lambda done, total, note: reevaluated.append(done),
        )
        assert reevaluated and reevaluated[0] == 1
        reevaluated.clear()
        # different space bounds
        tune_model(
            "ridge_classifier", SearchSpace.uniform("alpha", 0.5, 2.0),
            X, y, task="classification", cv=2, n_trials=N_TRIALS,
            sampler="tpe", seed=1, checkpoint_dir=home,
            progress=lambda done, total, note: reevaluated.append(done),
        )
        assert reevaluated and reevaluated[0] == 1
        reevaluated.clear()
        # different sampler
        tune_model(
            "ridge_classifier", _space(), X, y, task="classification",
            cv=2, n_trials=N_TRIALS, sampler="random", seed=1,
            checkpoint_dir=home,
            progress=lambda done, total, note: reevaluated.append(done),
        )
        assert reevaluated and reevaluated[0] == 1

    def test_stale_dataset_checkpoint_refuses_to_resume(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint from different data: detected, not guessed."""
        X, y = _data()
        home = tmp_path / "stale-data"
        _tune(X, y, checkpoint_dir=home, checkpoint_resume=False)
        X2 = X + 0.5  # different content
        reevaluated: list[int] = []
        tune_model(
            "ridge_classifier", _space(), X2, y, task="classification",
            cv=2, n_trials=N_TRIALS, sampler="tpe", seed=1,
            checkpoint_dir=home,
            progress=lambda done, total, note: reevaluated.append(done),
        )
        assert reevaluated and reevaluated[0] == 1

    def test_checkpoints_carry_the_v2_envelope(self, tmp_path: Path) -> None:
        """The §13 envelope rides every checkpoint: registered
        operation, resume_unit in trials, validate_checkpoint green."""
        X, y = _data()
        home = tmp_path / "envelope"
        _tune(X, y, checkpoint_dir=home, checkpoint_resume=False)
        files = sorted(home.glob("hpo-study.step-*.json"))
        assert files
        for path in files:
            report = validate_checkpoint(path, manager_name="hpo-study")
            assert report.ok, report.errors
            assert not report.legacy
            envelope = report.envelope
            assert envelope is not None
            assert envelope["operation"]["type"] == "hpo"
            assert envelope["operation"]["restore_exact"] is True
            assert envelope["operation"]["partial_results_policy"] == "preserve"
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record[ENVELOPE_KEY]["resume_unit"] == record["step"]

    def test_registry_entry_pins_the_proof_test(self) -> None:
        """The registered entry points at THIS test (earning registration
        is the point -- the proof must be findable from the registry)."""
        entry = OPERATION_REGISTRY["hpo"]
        assert entry.implementation == "pharmaml.ml.hpo"
        assert "test_resumed_study_is_bit_exact" in entry.exact_restore_proof
        assert entry.unit_description == "trials evaluated"
        # (round 9) generative_search and (round 10) docking_batch
        # have since earned their own rounds: preprocessing is the
        # pending frontier now
        assert "generative_search" in OPERATION_REGISTRY
        assert "docking_batch" in OPERATION_REGISTRY
        with pytest.raises(Exception, match="preprocessing"):
            OperationSpec(operation_type="preprocessing", operation_id="x")

    def test_checkpoint_every_validation(self, tmp_path: Path) -> None:
        X, y = _data()
        with pytest.raises(ValueError, match="checkpoint_every"):
            _tune(X, y, checkpoint_dir=tmp_path, checkpoint_every=0)

    def test_checkpoint_every_thins_the_saves(self, tmp_path: Path) -> None:
        """checkpoint_every=3 saves only every third trial; the resume
        is still bit-exact."""
        X, y = _data()
        home = tmp_path / "thin"
        _tune(X, y, checkpoint_dir=home, checkpoint_every=3,
              checkpoint_resume=False)
        files = sorted(home.glob("hpo-study.step-*.json"))
        # rotation keeps only the newest 3 (the storage layer's keep=3):
        # steps 9, 12 and the final 14
        assert [int(f.name.split(".step-")[1].split(".")[0]) for f in files] == [
            9, 12, 14,
        ]
        report = _tune(X, y, checkpoint_dir=home)
        plain = _tune(X, y)
        assert report.history == plain.history

    def test_no_checkpoint_dir_keeps_historical_behaviour(self) -> None:
        """Without checkpoint_dir a cooperative stop raises with
        partial_results=None (the round-205 convention for
        checkpoint-less loops)."""
        X, y = _data()
        with pytest.raises(JobCancelled) as caught:
            _tune(
                X, y,
                progress=lambda done, total, note: False if done >= 3 else None,
            )
        assert caught.value.partial_results is None

    def test_manager_names_the_study(self, tmp_path: Path) -> None:
        """The checkpoint files carry the job stem the manager defines."""
        X, y = _data()
        home = tmp_path / "names"
        _tune(X, y, checkpoint_dir=home, checkpoint_resume=False)
        files = sorted(home.glob("*.json"))
        assert files[-1].name == "hpo-study.step-000014.json"
        info = CheckpointManager(home, "hpo-study").latest()
        assert info is not None and info.resumable
