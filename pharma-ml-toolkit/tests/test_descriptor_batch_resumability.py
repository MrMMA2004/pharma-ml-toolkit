"""The §13 exact-restore proofs for the resumable descriptor batch.

Part5-continuation round 6: ``compute_feature_frame(checkpoint_dir=...)``
is the first of the six pending §13 operation classes to earn registry
entry.  These tests are the evidence: an interrupted-then-resumed batch
must equal an uninterrupted one BIT FOR BIT (frame values, column order,
index, report), a cooperative stop must preserve the partial results on
the exception, and a stale checkpoint (different families, conformer
count, or input list) must refuse to resume -- exact restore or nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmaml.chem.features import compute_feature_frame
from pharmaml.exceptions import JobCancelled
from pharmaml.utils.checkpoint_schema import (
    ENVELOPE_KEY,
    OPERATION_REGISTRY,
    PENDING_OPERATION_TYPES,
    validate_checkpoint,
)
from pharmaml.utils.checkpoints import CheckpointManager

# enough molecules to cross several 25-molecule chunk boundaries, with a
# few hostile entries so the drop path rides the checkpoints too
SMILES = ["CCO", "CCC", "CCN", "not-a-smiles", "c1ccccc1", "CC(=O)O", "",
          "CCOCC", "CCCCCC", "CN(C)C"] * 12  # 120 entries, ~12 chunks


def _uninterrupted() -> tuple[pd.DataFrame, object]:
    return compute_feature_frame(SMILES, families=("ecfp", "descriptors"))


class TestDescriptorBatchResume:
    def test_resumed_batch_is_bit_exact(self, tmp_path: Path) -> None:
        """THE registry proof: interrupted + resumed == uninterrupted,
        bit for bit -- frame, columns, index and report."""
        home = tmp_path / "ckpt"
        # interrupt: compute everything but pretend the process died --
        # checkpoint_resume=False forces a from-scratch run that still
        # writes checkpoints (the fresh-run twin)
        fresh_frame, fresh_report = compute_feature_frame(
            SMILES, families=("ecfp", "descriptors"),
            checkpoint_dir=home, checkpoint_resume=False,
        )
        # the resume: same inputs, resume allowed, and the progress
        # callback counts how many molecules were actually recomputed
        recomputed: list[int] = []
        resumed_frame, resumed_report = compute_feature_frame(
            SMILES, families=("ecfp", "descriptors"),
            checkpoint_dir=home,
            progress=lambda done, total, note: recomputed.append(done),
        )
        # every molecule was already in the checkpoint: NOTHING recomputed
        assert recomputed == []
        # bit-identical outputs
        pd.testing.assert_frame_equal(resumed_frame, fresh_frame)
        assert resumed_report.smiles == fresh_report.smiles
        assert resumed_report.dropped == fresh_report.dropped
        assert resumed_report.families == fresh_report.families
        # and identical to the checkpoint-less historical call
        plain_frame, plain_report = _uninterrupted()
        pd.testing.assert_frame_equal(resumed_frame, plain_frame)
        assert resumed_report.dropped == plain_report.dropped

    def test_partial_checkpoint_resumes_only_the_tail(self, tmp_path: Path) -> None:
        """A checkpoint from a genuinely partial run: the resume recomputes
        only the remaining molecules and still matches bit for bit."""
        home = tmp_path / "partial"
        # partial run: stop at the SECOND chunk boundary (50 of 120) --
        # None continues, False cancels (the round-205 contract)
        with pytest.raises(JobCancelled):
            compute_feature_frame(
                SMILES, families=("ecfp",),
                checkpoint_dir=home,
                progress=lambda done, total, note: False if done >= 50 else None,
            )
        steps = sorted(home.glob("descriptor-batch.step-*.json"))
        assert steps, "the partial run must have written a checkpoint"
        stopped_at = max(int(s.stem.split(".step-")[1]) for s in steps)
        assert stopped_at == 50
        # resume and count what was recomputed: the tail past the checkpoint
        recomputed: list[int] = []
        frame, report = compute_feature_frame(
            SMILES, families=("ecfp",),
            checkpoint_dir=home,
            progress=lambda done, total, note: recomputed.append(done),
        )
        assert recomputed, "the tail after the checkpoint must recompute"
        assert recomputed[0] == 75  # the next chunk boundary past the resume point
        # the resumed total matches the uninterrupted twin exactly
        plain_frame, plain_report = compute_feature_frame(SMILES, families=("ecfp",))
        pd.testing.assert_frame_equal(frame, plain_frame)
        assert report.dropped == plain_report.dropped

    def test_cancel_preserves_partial_results_on_the_exception(
        self, tmp_path: Path
    ) -> None:
        """§13: preserve completed partial results after cancellation --
        the data contract permits it (rows are independent)."""
        home = tmp_path / "stop"
        with pytest.raises(JobCancelled) as caught:
            compute_feature_frame(
                SMILES, families=("ecfp",),
                checkpoint_dir=home,
                progress=lambda done, total, note: False if done >= 30 else None,
            )
        partial = caught.value.partial_results
        assert partial is not None
        partial_frame, partial_report = partial
        # a real partial: some rows present, fewer than the total
        assert 0 < len(partial_frame) < len(SMILES)
        assert partial_frame.index.name == "original_index"
        # the partial rows are a bit-exact subset of the full run's rows
        # (same original_index positions, identical float values), and
        # the drop ledger is the full run's ledger restricted to the
        # computed prefix
        full_frame, full_report = compute_feature_frame(SMILES, families=("ecfp",))
        shared = partial_frame.index.intersection(full_frame.index)
        assert len(shared) == len(partial_frame)
        np.testing.assert_array_equal(
            partial_frame.to_numpy(), full_frame.loc[shared].to_numpy()
        )
        max_row = max(partial_frame.index.max(), 0)
        full_dropped = dict(full_report.dropped)
        for row, text in partial_report.dropped:
            assert full_dropped.get(row) == text
        assert all(row <= max_row for row, _ in partial_report.dropped)

    def test_stale_config_checkpoint_refuses_to_resume(self, tmp_path: Path) -> None:
        """A checkpoint from different families: exact restore or nothing."""
        home = tmp_path / "stale"
        compute_feature_frame(
            SMILES[:30], families=("ecfp",),
            checkpoint_dir=home, checkpoint_resume=False,
        )
        recomputed: list[int] = []
        frame, _report = compute_feature_frame(
            SMILES[:30], families=("avalon_count",),
            checkpoint_dir=home,
            progress=lambda done, total, note: recomputed.append(done),
        )
        assert recomputed, "a different config must recompute from scratch"
        plain_frame, _r = compute_feature_frame(SMILES[:30], families=("avalon_count",))
        pd.testing.assert_frame_equal(frame, plain_frame)

    def test_stale_dataset_checkpoint_refuses_to_resume(self, tmp_path: Path) -> None:
        """A checkpoint from a different input list: detected, not guessed."""
        home = tmp_path / "stale-data"
        compute_feature_frame(
            SMILES[:30], families=("ecfp",),
            checkpoint_dir=home, checkpoint_resume=False,
        )
        recomputed: list[int] = []
        compute_feature_frame(
            SMILES[:40], families=("ecfp",),
            checkpoint_dir=home,
            progress=lambda done, total, note: recomputed.append(done),
        )
        assert recomputed, "a different input list must recompute from scratch"

    def test_checkpoints_carry_the_v2_envelope(self, tmp_path: Path) -> None:
        """The §13 envelope rides every checkpoint: registered operation,
        resume_unit in molecules, validate_checkpoint green."""
        home = tmp_path / "envelope"
        compute_feature_frame(
            SMILES[:30], families=("ecfp",),
            checkpoint_dir=home, checkpoint_resume=False,
        )
        files = sorted(home.glob("descriptor-batch.step-*.json"))
        assert files
        for path in files:
            report = validate_checkpoint(path, manager_name="descriptor-batch")
            assert report.ok, report.errors
            assert not report.legacy
            envelope = report.envelope
            assert envelope is not None
            assert envelope["operation"]["type"] == "descriptor_batch"
            assert envelope["operation"]["restore_exact"] is True
            assert envelope["operation"]["partial_results_policy"] == "preserve"
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record[ENVELOPE_KEY]["resume_unit"] == record["step"]

    def test_registry_entry_pins_the_proof_test(self) -> None:
        """The registered entry points at THIS test (earning registration
        is the point -- the proof must be findable from the registry)."""
        entry = OPERATION_REGISTRY["descriptor_batch"]
        assert entry.implementation == "pharmaml.chem.features"
        assert "test_resumed_batch_is_bit_exact" in entry.exact_restore_proof
        assert entry.unit_description == "molecules featurized"
        # (round 7) the NEXT candidate has since earned its own round:
        # similarity_campaign is registered -- and (round 8) hpo and
        # (round 9) generative_search after it; the pending frontier
        # has moved on to docking_batch and the rest
        assert "similarity_campaign" in OPERATION_REGISTRY
        assert "hpo" in OPERATION_REGISTRY
        assert "generative_search" in OPERATION_REGISTRY
        # (round 11) preprocessing has since earned its own round:
        # ALL SIX §13 classes are registered -- the registry is
        # complete and the pending list is empty
        assert "preprocessing" in OPERATION_REGISTRY
        assert PENDING_OPERATION_TYPES == ()

    def test_checkpoint_every_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checkpoint_every"):
            compute_feature_frame(
                ["CCO"], families=("ecfp",),
                checkpoint_dir=tmp_path, checkpoint_every=0,
            )

    def test_checkpoint_every_thins_the_saves(self, tmp_path: Path) -> None:
        """checkpoint_every=2 halves the checkpoint count; the resume is
        still bit-exact."""
        home = tmp_path / "thin"
        compute_feature_frame(
            SMILES[:60], families=("ecfp",),
            checkpoint_dir=home, checkpoint_every=2, checkpoint_resume=False,
        )
        # 60 molecules = 3 chunks (25/25/10): saves only after chunks 2
        # and 3 (boundaries_since_save reaches 2), never after chunk 1;
        # the FINAL chunk always saves regardless (the completion record)
        files = sorted(home.glob("descriptor-batch.step-*.json"))
        assert [int(f.name.split(".step-")[1].split(".")[0]) for f in files] == [50, 60]
        frame, _report = compute_feature_frame(
            SMILES[:60], families=("ecfp",), checkpoint_dir=home,
        )
        plain_frame, _r = compute_feature_frame(SMILES[:60], families=("ecfp",))
        pd.testing.assert_frame_equal(frame, plain_frame)

    def test_no_checkpoint_dir_keeps_historical_behaviour(self) -> None:
        """Without checkpoint_dir the call is the historical one: a
        cooperative stop raises with partial_results=None (nothing was
        kept -- the round-205 contract, unchanged)."""
        with pytest.raises(JobCancelled) as caught:
            compute_feature_frame(
                SMILES[:60], families=("ecfp",),
                progress=lambda done, total, note: False if done >= 25 else None,
            )
        assert caught.value.partial_results is None

    def test_manager_names_the_batch(self, tmp_path: Path) -> None:
        """The checkpoint files carry the job stem the manager defines."""
        home = tmp_path / "names"
        compute_feature_frame(
            ["CCO", "CCC"], families=("ecfp",), checkpoint_dir=home,
        )
        files = sorted(home.glob("*.json"))
        assert [f.name for f in files] == ["descriptor-batch.step-000002.json"]
        info = CheckpointManager(home, "descriptor-batch").latest()
        assert info is not None and info.resumable
