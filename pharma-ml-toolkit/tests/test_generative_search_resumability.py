"""The §13 exact-restore proofs for the resumable generative search.

Part5-continuation round 9: ``GeneticOptimizer.optimize(checkpoint_dir=...)``
is the fourth of the six pending §13 operation classes to earn registry
entry.  These tests are the evidence: an interrupted-then-resumed
evolution must equal an uninterrupted one BIT FOR BIT (``elapsed_s``
aside, which is wall-clock by nature) -- molecules, scores, history,
metadata.  The round-8 lesson carried over and pinned here: the
checkpoints carry the sampler's RNG state (every tournament / mutation
/ crossover draw advances it), so a resume proposes exactly the
offspring the uninterrupted run would have.  On adaptive runs the
three operator credit books and the trajectory restore too.  A
cooperative stop preserves the partial :class:`GenerationResult` on
the exception, and a stale-config checkpoint (different GA knobs or a
different objective) is refused -- exact restore or nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharmaml.exceptions import JobCancelled, PersistenceError
from pharmaml.generative import (
    GeneticOptimizer,
    LogPBandObjective,
    QedObjective,
)
from pharmaml.utils.checkpoint_schema import (
    ENVELOPE_KEY,
    OPERATION_REGISTRY,
    PENDING_OPERATION_TYPES,
    validate_checkpoint,
)

GENS = 8  # generations per run


def _ga(**kwargs: object) -> GeneticOptimizer:
    params: dict[str, object] = {
        "population_size": 10,
        "generations": GENS,
        "seed": 7,
    }
    params.update(kwargs)
    return GeneticOptimizer(QedObjective(), **params)


def _strip(result: object) -> tuple:
    """Everything bit-exact-comparable about a GenerationResult."""
    m = dict(result.metadata)  # type: ignore[attr-defined]
    m.pop("elapsed_s", None)  # wall-clock by nature
    return (
        result.smiles,  # type: ignore[attr-defined]
        result.scores,  # type: ignore[attr-defined]
        result.history,  # type: ignore[attr-defined]
        m,
        result.best,  # type: ignore[attr-defined]
        result.best_score,  # type: ignore[attr-defined]
        result.breakdown,  # type: ignore[attr-defined]
    )


class TestGenerativeSearchResume:
    def test_resumed_search_is_bit_exact(self, tmp_path: Path) -> None:
        """THE registry proof: interrupted + resumed == uninterrupted,
        bit for bit (elapsed_s aside) -- molecules, scores, history,
        metadata, breakdown."""
        home = tmp_path / "ckpt"
        # the fresh-run twin: computes everything, checkpoints land, but
        # resume is suppressed so the NEXT call below must replay them
        fresh = _ga().optimize(
            checkpoint_dir=home, checkpoint_resume=False
        )
        assert len(fresh.history) == GENS
        # the resume: same config, resume allowed, and the progress
        # callback counts how many generations actually re-ran
        ran: list[int] = []
        resumed = _ga().optimize(
            checkpoint_dir=home,
            progress=lambda done, total, note: ran.append(done),
        )
        assert ran == []  # every generation was in the checkpoint
        assert _strip(resumed) == _strip(fresh)
        # and identical to the checkpoint-less historical call
        plain = _ga().optimize()
        assert _strip(resumed) == _strip(plain)

    def test_partial_checkpoint_resumes_only_the_tail(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint from a genuinely partial run: the resume
        evolves only the remaining generations and still matches."""
        home = tmp_path / "partial"
        with pytest.raises(JobCancelled):
            _ga().optimize(
                checkpoint_dir=home,
                progress=lambda done, total, note: (
                    False if done >= 3 else None
                ),
            )
        steps = sorted(home.glob("generative-search.step-*.json"))
        assert steps, "the partial run must have written checkpoints"
        stopped_at = max(int(s.stem.split(".step-")[1]) for s in steps)
        assert stopped_at == 3
        ran: list[int] = []
        report = _ga().optimize(
            checkpoint_dir=home,
            progress=lambda done, total, note: ran.append(done),
        )
        assert ran and ran[0] == 4  # generation 4 is the first re-run
        plain = _ga().optimize()
        assert _strip(report) == _strip(plain)

    def test_adaptive_search_resumes_bit_exact(self, tmp_path: Path) -> None:
        """The adaptive operator selection: the three credit books
        (reward windows, tallies, best offspring) and the trajectory
        restore too -- the resumed mix evolves identically."""
        home = tmp_path / "adaptive"
        with pytest.raises(JobCancelled):
            _ga(
                operator_selection="adaptive", generations=GENS + 2
            ).optimize(
                checkpoint_dir=home,
                progress=lambda done, total, note: (
                    False if done >= 4 else None
                ),
            )
        ran: list[int] = []
        resumed = _ga(
            operator_selection="adaptive", generations=GENS + 2
        ).optimize(
            checkpoint_dir=home,
            progress=lambda done, total, note: ran.append(done),
        )
        assert ran and ran[0] == 5
        plain = _ga(
            operator_selection="adaptive", generations=GENS + 2
        ).optimize()
        assert _strip(resumed) == _strip(plain)
        # the trajectory is complete and generation-aligned
        trajectory = resumed.metadata["operator_trajectory"]
        assert [rec["generation"] for rec in trajectory] == list(
            range(GENS + 2)
        )
        # the credit books' tallies survived the resume intact
        stats = resumed.metadata["operator_stats"]
        assert stats["level1"] == plain.metadata["operator_stats"]["level1"]

    def test_cancel_preserves_partial_result_on_the_exception(
        self, tmp_path: Path
    ) -> None:
        """§13: preserve completed partial results after cancellation --
        the partial GenerationResult rides JobCancelled."""
        home = tmp_path / "stop"
        with pytest.raises(JobCancelled) as caught:
            _ga(generations=10).optimize(
                checkpoint_dir=home,
                progress=lambda done, total, note: (
                    False if done >= 5 else None
                ),
            )
        partial = caught.value.partial_results
        assert partial is not None
        # 5 generations completed: history carries 5 records, ranked
        # molecules are the union of what those generations produced
        assert len(partial.history) == 5
        assert partial.metadata["iterations"] == 5
        assert partial.metadata["partial"] is True
        assert partial.smiles  # something was produced
        # the partial best matches the best of the 5-generation prefix
        # of the uninterrupted 10-generation run's produced book
        plain = _ga(generations=10).optimize()
        prefix_best = max(
            (
                (s, sc)
                for s, sc in zip(plain.smiles, plain.scores)
                if s in set(partial.smiles)
            ),
            key=lambda kv: (kv[1], kv[0]),
            default=("", 0.0),
        )
        assert partial.best_score == pytest.approx(prefix_best[1])

    def test_stale_config_checkpoint_refuses_to_resume(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint from a different GA config (or a different
        objective): exact restore or nothing."""
        home = tmp_path / "stale"
        _ga().optimize(checkpoint_dir=home, checkpoint_resume=False)
        before = {p.name: p.read_bytes() for p in home.iterdir()}
        alternatives = [
            GeneticOptimizer(QedObjective(), population_size=10, generations=GENS, seed=8),
            GeneticOptimizer(LogPBandObjective(1.0, 3.0), population_size=10, generations=GENS, seed=7),
            GeneticOptimizer(QedObjective(), population_size=10, generations=GENS, seed=7,
                             mutation="bioisostere"),
        ]
        for optimizer in alternatives:
            with pytest.raises(PersistenceError, match="identity or version mismatch"):
                optimizer.optimize(checkpoint_dir=home)
            assert {p.name: p.read_bytes() for p in home.iterdir()} == before

    def test_checkpoints_carry_the_v2_envelope(self, tmp_path: Path) -> None:
        """The §13 envelope rides every checkpoint: registered
        operation, resume_unit in generations, validate green."""
        home = tmp_path / "envelope"
        _ga().optimize(checkpoint_dir=home, checkpoint_resume=False)
        files = sorted(home.glob("generative-search.step-*.json"))
        assert files
        for path in files:
            report = validate_checkpoint(
                path, manager_name="generative-search"
            )
            assert report.ok, report.errors
            assert not report.legacy
            envelope = report.envelope
            assert envelope is not None
            assert envelope["operation"]["type"] == "generative_search"
            assert envelope["operation"]["restore_exact"] is True
            assert (
                envelope["operation"]["partial_results_policy"] == "preserve"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record[ENVELOPE_KEY]["resume_unit"] == record["step"]

    def test_registry_entry_pins_the_proof_test(self) -> None:
        """The registered entry points at THIS test (earning
        registration is the point -- the proof must be findable from
        the registry)."""
        entry = OPERATION_REGISTRY["generative_search"]
        assert (
            entry.implementation == "pharmaml.generative.genetic"
        )
        assert (
            "test_resumed_search_is_bit_exact" in entry.exact_restore_proof
        )
        assert entry.unit_description == "generations evolved"
        # (round 10) docking_batch has since earned its own round:
        # the pending frontier is preprocessing alone now
        assert "docking_batch" in OPERATION_REGISTRY
        # (round 11) preprocessing has since earned its own round:
        # ALL SIX §13 classes are registered -- the registry is
        # complete and the pending list is empty
        assert "preprocessing" in OPERATION_REGISTRY
        assert PENDING_OPERATION_TYPES == ()

    def test_checkpoint_every_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checkpoint_every"):
            _ga().optimize(
                checkpoint_dir=tmp_path, checkpoint_every=0
            )

    def test_checkpoint_every_thins_the_saves(
        self, tmp_path: Path
    ) -> None:
        """checkpoint_every=2 saves every second generation boundary;
        the storage layer's rotation keeps the newest three -- and the
        resume is still bit-exact."""
        home = tmp_path / "thin"
        _ga(generations=10).optimize(
            checkpoint_dir=home, checkpoint_every=2, checkpoint_resume=False
        )
        files = sorted(home.glob("generative-search.step-*.json"))
        steps = [int(f.name.split(".step-")[1].split(".")[0]) for f in files]
        # saves at the 2,4,6,8 boundaries (the 10th and final generation
        # produces no offspring, so its "boundary" carries no new state)
        # -- rotation keeps the newest three: 6, 8, 10? no: the final
        # save fires at the 9th generation's boundary (generation >=
        # generations - 2 forces the last two boundaries), so 6, 8, 9.
        assert steps == [6, 8, 9]
        report = _ga(generations=10).optimize(checkpoint_dir=home)
        plain = _ga(generations=10).optimize()
        assert _strip(report) == _strip(plain)

    def test_no_checkpoint_dir_keeps_historical_behaviour(self) -> None:
        """Without checkpoint_dir a cooperative stop still raises
        JobCancelled, and the partial GenerationResult is computable
        from the loop state alone -- R9 attaches it in BOTH modes (a
        strictly richer stop than the round-205 None convention)."""
        with pytest.raises(JobCancelled) as caught:
            _ga().optimize(
                progress=lambda done, total, note: (
                    False if done >= 3 else None
                ),
            )
        partial = caught.value.partial_results
        assert partial is not None
        assert len(partial.history) == 3
        assert partial.metadata["iterations"] == 3
        assert partial.metadata["partial"] is True
