"""The §13 exact-restore proofs for the resumable similarity campaign.

Part5-continuation round 7: ``run_similarity_campaign(checkpoint_dir=...)``
is the second of the six pending §13 operation classes to earn registry
entry.  These tests are the evidence: an interrupted-then-resumed campaign
must equal an uninterrupted one BIT FOR BIT (and equal the one-shot
:func:`~pharmaml.chem.similarity.similarity_scan` on the same inputs), a
cooperative stop must preserve the partial rows on the exception, and a
stale checkpoint (different scan config or different input lists) must
refuse to resume -- exact restore or nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pharmaml.chem.similarity import run_similarity_campaign, similarity_scan
from pharmaml.exceptions import JobCancelled
from pharmaml.utils.checkpoint_schema import (
    ENVELOPE_KEY,
    OPERATION_REGISTRY,
    OperationSpec,
    validate_checkpoint,
)
from pharmaml.utils.checkpoints import CheckpointManager

# enough references to cross several 25-reference chunk boundaries, with a
# few hostile entries so the drop ledger rides the checkpoints too
REFERENCES = ["CCO", "CCC", "CCN", "not-a-smiles", "c1ccccc1", "CC(=O)O", "",
              "CCOCC", "CCCCCC", "CN(C)C"] * 12  # 120 entries, 5 chunks
TARGETS = ["CCO", "CCC", "c1ccccc1O", "CC(=O)OC(=O)c1ccccc1C(=O)O",
           "CN(C)CCN(C)C", "CCOCC"] * 8  # 48-entry library


def _scan() -> pd.DataFrame:
    """The one-shot ground truth on the same inputs."""
    return similarity_scan(REFERENCES, TARGETS, threshold=0.3)


class TestSimilarityCampaignResume:
    def test_resumed_campaign_is_bit_exact(self, tmp_path: Path) -> None:
        """THE registry proof: interrupted + resumed == uninterrupted,
        bit for bit -- and == the one-shot similarity_scan."""
        home = tmp_path / "ckpt"
        # interrupt: compute everything but pretend the process died --
        # checkpoint_resume=False forces a from-scratch run that still
        # writes checkpoints (the fresh-run twin)
        fresh = run_similarity_campaign(
            REFERENCES, TARGETS, threshold=0.3,
            checkpoint_dir=home, checkpoint_resume=False,
        )
        # the resume: same inputs, resume allowed, and the progress
        # callback counts how many references were actually rescanned
        rescanned: list[int] = []
        resumed = run_similarity_campaign(
            REFERENCES, TARGETS, threshold=0.3, checkpoint_dir=home,
            progress=lambda done, total, note: rescanned.append(done),
        )
        # every reference was already in the checkpoint: NOTHING rescanned
        assert rescanned == []
        # bit-identical outputs
        pd.testing.assert_frame_equal(resumed, fresh)
        # and identical to the one-shot historical scan
        pd.testing.assert_frame_equal(resumed, _scan())

    def test_partial_checkpoint_resumes_only_the_tail(self, tmp_path: Path) -> None:
        """A checkpoint from a genuinely partial run: the resume rescans
        only the remaining references and still matches bit for bit."""
        home = tmp_path / "partial"
        # partial run: stop at the SECOND chunk boundary (50 of 120) --
        # None continues, False cancels (the round-205 contract)
        with pytest.raises(JobCancelled):
            run_similarity_campaign(
                REFERENCES, TARGETS, threshold=0.3, checkpoint_dir=home,
                progress=lambda done, total, note: False if done >= 50 else None,
            )
        steps = sorted(home.glob("similarity-campaign.step-*.json"))
        assert steps, "the partial run must have written a checkpoint"
        stopped_at = max(int(s.stem.split(".step-")[1]) for s in steps)
        assert stopped_at == 50
        # resume and count what was rescanned: the tail past the checkpoint
        rescanned: list[int] = []
        frame = run_similarity_campaign(
            REFERENCES, TARGETS, threshold=0.3, checkpoint_dir=home,
            progress=lambda done, total, note: rescanned.append(done),
        )
        assert rescanned, "the tail after the checkpoint must rescan"
        assert rescanned[0] == 75  # the next chunk boundary past the resume point
        # the resumed total matches the one-shot scan exactly
        pd.testing.assert_frame_equal(frame, _scan())

    def test_cancel_preserves_partial_rows_on_the_exception(
        self, tmp_path: Path
    ) -> None:
        """§13: preserve completed partial results after cancellation --
        the data contract permits it (reference rows are independent)."""
        home = tmp_path / "stop"
        with pytest.raises(JobCancelled) as caught:
            run_similarity_campaign(
                REFERENCES, TARGETS, threshold=0.3, checkpoint_dir=home,
                progress=lambda done, total, note: False if done >= 30 else None,
            )
        partial = caught.value.partial_results
        assert partial is not None
        # a real partial: some rows present, a strict prefix of the campaign
        assert 0 < len(partial) < len(_scan())
        # the partial rows are a bit-exact subset of the full run's rows
        # (same reference positions, identical float similarities)
        full = _scan()
        partial_refs = set(partial["reference_index"])
        full_at_refs = full[full["reference_index"].isin(partial_refs)]
        pd.testing.assert_frame_equal(
            partial.reset_index(drop=True), full_at_refs.reset_index(drop=True)
        )
        # only references from the computed prefix appear: the cancel fires
        # at the first chunk boundary with done >= 30, i.e. after the
        # SECOND chunk (references 0..49 complete, done == 50)
        assert partial["reference_index"].max() < 50

    def test_stale_config_checkpoint_refuses_to_resume(self, tmp_path: Path) -> None:
        """A checkpoint from a different scan config: exact restore or
        nothing."""
        home = tmp_path / "stale"
        run_similarity_campaign(
            REFERENCES[:30], TARGETS, threshold=0.3,
            checkpoint_dir=home, checkpoint_resume=False,
        )
        rescanned: list[int] = []
        frame = run_similarity_campaign(
            REFERENCES[:30], TARGETS, threshold=0.6,  # different threshold
            checkpoint_dir=home,
            progress=lambda done, total, note: rescanned.append(done),
        )
        assert rescanned, "a different config must rescan from scratch"
        plain = similarity_scan(REFERENCES[:30], TARGETS, threshold=0.6)
        pd.testing.assert_frame_equal(frame, plain)

    def test_stale_dataset_checkpoint_refuses_to_resume(self, tmp_path: Path) -> None:
        """A checkpoint from different input lists: detected, not guessed."""
        home = tmp_path / "stale-data"
        run_similarity_campaign(
            REFERENCES[:30], TARGETS, threshold=0.3,
            checkpoint_dir=home, checkpoint_resume=False,
        )
        rescanned: list[int] = []
        run_similarity_campaign(
            REFERENCES[:40], TARGETS, threshold=0.3,  # different references
            checkpoint_dir=home,
            progress=lambda done, total, note: rescanned.append(done),
        )
        assert rescanned, "a different reference list must rescan from scratch"
        rescanned.clear()
        run_similarity_campaign(
            REFERENCES[:30], TARGETS[:20], threshold=0.3,  # different library
            checkpoint_dir=home,
            progress=lambda done, total, note: rescanned.append(done),
        )
        assert rescanned, "a different target list must rescan from scratch"

    def test_checkpoints_carry_the_v2_envelope(self, tmp_path: Path) -> None:
        """The §13 envelope rides every checkpoint: registered operation,
        resume_unit in references, validate_checkpoint green."""
        home = tmp_path / "envelope"
        run_similarity_campaign(
            REFERENCES[:30], TARGETS, threshold=0.3,
            checkpoint_dir=home, checkpoint_resume=False,
        )
        files = sorted(home.glob("similarity-campaign.step-*.json"))
        assert files
        for path in files:
            report = validate_checkpoint(path, manager_name="similarity-campaign")
            assert report.ok, report.errors
            assert not report.legacy
            envelope = report.envelope
            assert envelope is not None
            assert envelope["operation"]["type"] == "similarity_campaign"
            assert envelope["operation"]["restore_exact"] is True
            assert envelope["operation"]["partial_results_policy"] == "preserve"
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record[ENVELOPE_KEY]["resume_unit"] == record["step"]

    def test_registry_entry_pins_the_proof_test(self) -> None:
        """The registered entry points at THIS test (earning registration
        is the point -- the proof must be findable from the registry)."""
        entry = OPERATION_REGISTRY["similarity_campaign"]
        assert entry.implementation == "pharmaml.chem.similarity"
        assert "test_resumed_campaign_is_bit_exact" in entry.exact_restore_proof
        assert entry.unit_description == "references scanned"
        # (round 8) the NEXT candidate has since earned its own round:
        # hpo is registered -- and (round 9) generative_search after
        # it; the pending frontier has moved on to docking_batch and
        # the rest
        assert "hpo" in OPERATION_REGISTRY
        assert "generative_search" in OPERATION_REGISTRY
        with pytest.raises(Exception, match="docking_batch"):
            OperationSpec(operation_type="docking_batch", operation_id="x")

    def test_checkpoint_every_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checkpoint_every"):
            run_similarity_campaign(
                ["CCO"], ["CCC"], checkpoint_dir=tmp_path, checkpoint_every=0,
            )

    def test_checkpoint_every_thins_the_saves(self, tmp_path: Path) -> None:
        """checkpoint_every=2 halves the checkpoint count; the resume is
        still bit-exact."""
        home = tmp_path / "thin"
        run_similarity_campaign(
            REFERENCES[:60], TARGETS, threshold=0.3,
            checkpoint_dir=home, checkpoint_every=2, checkpoint_resume=False,
        )
        # 60 references = 3 chunks (25/25/10): saves only after chunks 2
        # and 3 (boundaries_since_save reaches 2), never after chunk 1;
        # the FINAL chunk always saves regardless (the completion record)
        files = sorted(home.glob("similarity-campaign.step-*.json"))
        assert [int(f.name.split(".step-")[1].split(".")[0]) for f in files] == [50, 60]
        frame = run_similarity_campaign(
            REFERENCES[:60], TARGETS, threshold=0.3, checkpoint_dir=home,
        )
        plain = similarity_scan(REFERENCES[:60], TARGETS, threshold=0.3)
        pd.testing.assert_frame_equal(frame, plain)

    def test_no_checkpoint_dir_keeps_historical_behaviour(self) -> None:
        """Without checkpoint_dir a cooperative stop raises with
        partial_results=None (nothing was kept -- the round-205
        convention for checkpoint-less loops)."""
        with pytest.raises(JobCancelled) as caught:
            run_similarity_campaign(
                REFERENCES[:60], TARGETS, threshold=0.3,
                progress=lambda done, total, note: False if done >= 25 else None,
            )
        assert caught.value.partial_results is None

    def test_manager_names_the_campaign(self, tmp_path: Path) -> None:
        """The checkpoint files carry the job stem the manager defines."""
        home = tmp_path / "names"
        run_similarity_campaign(["CCO", "CCC"], ["CCO"], checkpoint_dir=home)
        files = sorted(home.glob("*.json"))
        assert [f.name for f in files] == ["similarity-campaign.step-000002.json"]
        info = CheckpointManager(home, "similarity-campaign").latest()
        assert info is not None and info.resumable
