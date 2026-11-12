"""The §13 exact-restore proofs for the resumable chunked preprocessing.

Part5-continuation round 11 — the LAST of the six §13 operation
classes: with this round the registry is complete (descriptor_batch
R6, similarity_campaign R7, hpo R8, generative_search R9,
docking_batch R10, preprocessing R11).  These tests are the evidence
for ``apply_preprocessor_chunked``: the chunked (impute -> scale)
transform of a large table is resumable because two properties hold —
the pipeline's fit is DETERMINISTIC given the table (a resume refits
instead of deserializing sklearn state) and ``transform`` is ROW-PURE
(the transform of a slice equals the slice of the transform).  The
checkpoint is the completed chunks' transformed rows; a resume skips
them BIT-EXACTLY.  The fingerprint covers the strategy knobs AND the
table itself (columns, dtypes, shape, content checksum) — any edit
refuses the resume.  A cooperative stop preserves the partial
``(frame, report)`` on ``JobCancelled.partial_results``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import JobCancelled
from pharmaml.ml.preprocessing import (
    apply_preprocessor_chunked,
    build_preprocessor,
)
from pharmaml.utils.checkpoint_schema import (
    ENVELOPE_KEY,
    OPERATION_REGISTRY,
    PENDING_OPERATION_TYPES,
    OperationSpec,
    validate_checkpoint,
)

N_ROWS = 37
CHUNK = 10  # -> 4 chunks (10 + 10 + 10 + 7)


def _table() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    table = pd.DataFrame(
        rng.standard_normal((N_ROWS, 4)), columns=["a", "b", "c", "d"]
    )
    table.iloc[5, 2] = np.nan  # imputation has real work
    table.iloc[20, 0] = np.nan
    return table


class TestPreprocessingResume:
    def test_resumed_table_is_bit_exact(self, tmp_path: Path) -> None:
        """THE registry proof: a full fresh run (checkpoints written,
        resume suppressed) vs a resume of it -- the resume transforms
        NOTHING and the output frame is bit-identical."""
        home = tmp_path / "ckpt"
        fresh, report = apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK, checkpoint_dir=home,
            checkpoint_resume=False,
        )
        assert report["rows"] == N_ROWS
        assert report["chunks"] == 4
        ran: list[int] = []
        resumed, report2 = apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK, checkpoint_dir=home,
            progress=lambda done, total, note: ran.append(done),
        )
        assert ran == []  # every chunk was in the checkpoint
        assert report2["resumed_from_chunks"] == 4
        assert np.array_equal(resumed.to_numpy(), fresh.to_numpy())
        assert resumed.columns.tolist() == fresh.columns.tolist()
        # and identical to the one-shot historical call
        pipeline = build_preprocessor()
        oneshot = pd.DataFrame(
            pipeline.fit_transform(_table()),
            columns=[str(c) for c in pipeline.get_feature_names_out()],
        )
        assert np.allclose(fresh.to_numpy(), oneshot.to_numpy())
        assert fresh.columns.tolist() == oneshot.columns.tolist()

    def test_partial_checkpoint_resumes_only_the_tail(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint from a genuinely partial run: the resume
        transforms only the remaining chunks and still matches."""
        home = tmp_path / "partial"
        with pytest.raises(JobCancelled):
            apply_preprocessor_chunked(
                _table(), chunk_rows=CHUNK, checkpoint_dir=home,
                progress=lambda done, total, note: (
                    False if done >= 2 else None
                ),
            )
        steps = sorted(home.glob("preprocessing.step-*.json"))
        assert steps, "the partial run must have written checkpoints"
        # the cancel fires after chunk 2's rows but BEFORE its save --
        # the newest saved boundary is chunk 1 (boundary semantics)
        stopped_at = max(
            int(s.name.split(".step-")[1].split(".")[0]) for s in steps
        )
        assert stopped_at == 1
        ran: list[int] = []
        report_frame, report = apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK, checkpoint_dir=home,
            progress=lambda done, total, note: ran.append(done),
        )
        assert ran and ran[0] == 1  # the resume counts from its own 1
        assert report["resumed_from_chunks"] == 1
        fresh, _ = apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK
        )
        assert np.array_equal(report_frame.to_numpy(), fresh.to_numpy())

    def test_cancel_preserves_partial_frame_on_the_exception(
        self, tmp_path: Path
    ) -> None:
        """§13: preserve completed partial results after cancellation --
        the partial (frame, report) rides JobCancelled: the completed
        rows transformed, the pending rows absent."""
        home = tmp_path / "stop"
        with pytest.raises(JobCancelled) as caught:
            apply_preprocessor_chunked(
                _table(), chunk_rows=CHUNK, checkpoint_dir=home,
                progress=lambda done, total, note: (
                    False if done >= 2 else None
                ),
            )
        partial = caught.value.partial_results
        assert partial is not None
        partial_frame, partial_report = partial
        assert len(partial_frame) == 2 * CHUNK  # 20 completed rows
        assert partial_report["rows"] == 2 * CHUNK
        assert partial_report["partial"] is True
        assert partial_report["chunks"] == 2
        # the partial rows are the exact prefix of the full transform
        fresh, _ = apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK
        )
        assert np.array_equal(
            partial_frame.to_numpy(), fresh.to_numpy()[: 2 * CHUNK]
        )

    def test_stale_table_checkpoint_refuses_to_resume(
        self, tmp_path: Path
    ) -> None:
        """Any edit to the table (values, columns, shape) refuses the
        resume: exact restore or nothing."""
        home = tmp_path / "stale"
        table = _table()
        apply_preprocessor_chunked(
            table, chunk_rows=CHUNK, checkpoint_dir=home,
            checkpoint_resume=False,
        )
        variants = []
        edited = table.copy()
        edited.iloc[0, 0] = 99.0  # a value edit
        variants.append(("value", edited))
        renamed = table.copy()
        renamed.columns = ["a", "b", "c", "e"]  # a column rename
        variants.append(("rename", renamed))
        extended = pd.concat(
            [table, table.iloc[:3]], ignore_index=True
        )  # a shape edit
        variants.append(("shape", extended))
        for label, variant in variants:
            ran: list[int] = []

            def _count(
                done: int, total: int, note: str, _ran: list[int] = ran
            ) -> None:
                _ran.append(done)

            _, report = apply_preprocessor_chunked(
                variant, chunk_rows=CHUNK, checkpoint_dir=home,
                progress=_count,
            )
            n_chunks = (len(variant) + CHUNK - 1) // CHUNK
            assert len(ran) == n_chunks, (
                f"stale variant {label} must re-transform everything"
            )
            assert report["resumed_from_chunks"] == 0

    def test_stale_config_checkpoint_refuses_to_resume(
        self, tmp_path: Path
    ) -> None:
        """A different strategy knob refuses the resume too."""
        home = tmp_path / "stale-config"
        apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK, checkpoint_dir=home,
            checkpoint_resume=False,
        )
        ran: list[int] = []
        _, report = apply_preprocessor_chunked(
            _table(), impute_strategy="mean", chunk_rows=CHUNK,
            checkpoint_dir=home,
            progress=lambda done, total, note: ran.append(done),
        )
        assert len(ran) == 4
        assert report["resumed_from_chunks"] == 0

    def test_checkpoints_carry_the_v2_envelope(self, tmp_path: Path) -> None:
        """The §13 envelope rides every checkpoint: registered
        operation, resume_unit in chunks, validate green."""
        home = tmp_path / "envelope"
        apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK, checkpoint_dir=home,
            checkpoint_resume=False,
        )
        files = sorted(home.glob("preprocessing.step-*.json"))
        assert files
        for path in files:
            report = validate_checkpoint(
                path, manager_name="preprocessing"
            )
            assert report.ok, report.errors
            assert not report.legacy
            envelope = report.envelope
            assert envelope is not None
            assert envelope["operation"]["type"] == "preprocessing"
            assert envelope["operation"]["restore_exact"] is True
            assert (
                envelope["operation"]["partial_results_policy"]
                == "preserve"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record[ENVELOPE_KEY]["resume_unit"] == record["step"]
            # the payload is the transformed rows themselves
            rows = record["state"]["rows"]
            assert rows and all(isinstance(v, float) for r in rows for v in r)

    def test_registry_complete_and_pins_the_proof_test(self) -> None:
        """Round 11 makes the registry COMPLETE: all six §13 classes
        the schema header names are registered, the pending list is
        EMPTY, and this suite's proof node is pinned."""
        expected = {
            "numpy_mlp_training",
            "benchmark_family_scoring",
            "descriptor_batch",
            "similarity_campaign",
            "hpo",
            "generative_search",
            "docking_batch",
            "preprocessing",
        }
        assert expected <= set(OPERATION_REGISTRY)
        assert PENDING_OPERATION_TYPES == ()
        entry = OPERATION_REGISTRY["preprocessing"]
        assert entry.implementation == "pharmaml.ml.preprocessing"
        assert (
            "test_resumed_table_is_bit_exact" in entry.exact_restore_proof
        )
        assert entry.unit_description == "chunks transformed"
        # and a made-up operation type is still refused (the gate stays
        # armed for future classes)
        with pytest.raises(Exception, match="unknown operation type"):
            OperationSpec(
                operation_type="time_travel", operation_id="x"
            )

    def test_chunk_rows_and_every_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="chunk_rows"):
            apply_preprocessor_chunked(_table(), chunk_rows=0)
        with pytest.raises(ValueError, match="checkpoint_every"):
            apply_preprocessor_chunked(
                _table(), checkpoint_dir=tmp_path, checkpoint_every=0
            )
        with pytest.raises(Exception, match="empty"):
            apply_preprocessor_chunked(pd.DataFrame())

    def test_checkpoint_every_thins_the_saves(self, tmp_path: Path) -> None:
        """checkpoint_every=2 saves every second chunk boundary; the
        rotation keeps the newest three -- and the resume is still
        bit-exact."""
        home = tmp_path / "thin"
        apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK, checkpoint_dir=home,
            checkpoint_every=2, checkpoint_resume=False,
        )
        files = sorted(home.glob("preprocessing.step-*.json"))
        steps = [
            int(f.name.split(".step-")[1].split(".")[0]) for f in files
        ]
        # saves at chunk boundaries 2 and 4 (every=2) -- rotation keeps
        # both (only two exist)
        assert steps == [2, 4]
        resumed, report = apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK, checkpoint_dir=home
        )
        fresh, _ = apply_preprocessor_chunked(
            _table(), chunk_rows=CHUNK
        )
        assert report["resumed_from_chunks"] == 4
        assert np.array_equal(resumed.to_numpy(), fresh.to_numpy())

    def test_categorical_table_round_trips_resumably(
        self, tmp_path: Path
    ) -> None:
        """encode_categorical=True: the ColumnTransformer branch (one-hot
        columns) restores bit-exactly too."""
        rng = np.random.default_rng(11)
        table = pd.DataFrame(
            {
                "a": rng.standard_normal(25),
                "cls": ["A", "B", "C"] * 8 + ["A"],
            }
        )
        home = tmp_path / "cat"
        fresh, _fresh_report = apply_preprocessor_chunked(
            table, encode_categorical=True, chunk_rows=CHUNK,
            checkpoint_dir=home, checkpoint_resume=False,
        )
        assert len(fresh) == 25
        resumed, report2 = apply_preprocessor_chunked(
            table, encode_categorical=True, chunk_rows=CHUNK,
            checkpoint_dir=home,
        )
        assert report2["resumed_from_chunks"] == 3  # complete already
        assert np.array_equal(resumed.to_numpy(), fresh.to_numpy())
