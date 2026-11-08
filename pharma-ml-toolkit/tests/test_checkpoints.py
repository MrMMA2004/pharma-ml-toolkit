"""Tests for :mod:`pharmaml.utils.checkpoints` -- Part-3 §20 checkpoint depth.

The checkpoint contract the suite pins:

* every save is atomic (temp + fsync + ``os.replace``) and self-verifying
  (SHA-256 over the canonical record) -- a crash mid-write leaves the
  previous checkpoint intact, and no reader ever sees a half file;
* rotation keeps exactly ``keep`` checkpoints -- a 40-hour job can
  checkpoint thousands of times without filling the disk;
* a corrupt newest checkpoint (truncated, tampered, foreign schema) is
  QUARANTINED to ``*.json.corrupt`` on read and the resume falls back
  to the previous valid one instead of failing the resuming run;
* validation is strict about the inputs (plain filename stems,
  non-negative int steps, dict states) because a checkpoint written
  with sloppy metadata is a resume that lies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharmaml.exceptions import PersistenceError
from pharmaml.utils.checkpoints import (
    SCHEMA,
    CheckpointInfo,
    CheckpointManager,
    default_directory,
)


# ---------------------------------------------------------------------------
# save / read round-trip
# ---------------------------------------------------------------------------
class TestSaveAndRead:
    def test_round_trip_preserves_every_field(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=3)
        info = mgr.save(
            step=7,
            total=12,
            state={"rows": [1, 2, 3], "metric": "roc_auc"},
            note="family 'ecfp' scored",
        )
        assert isinstance(info, CheckpointInfo)
        assert info.step == 7
        assert info.total == 12
        assert info.state == {"rows": [1, 2, 3], "metric": "roc_auc"}
        assert info.note == "family 'ecfp' scored"
        assert info.saved_at  # an ISO-ish timestamp landed
        assert info.path.name == "job.step-000007.json"
        assert len(info.integrity) == 64  # a sha256 hex digest

        read = mgr.latest()
        assert read is not None
        assert read.step == info.step
        assert read.state == info.state
        assert read.note == info.note

    def test_unicode_state_round_trips(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(step=1, state={"family": "拓扑", "note": "π-STACK"})
        assert mgr.latest().state == info.state

    def test_default_state_is_empty_dict(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(step=0)
        assert info.state == {}
        assert info.total is None
        assert info.note == ""

    def test_files_are_valid_json_documents(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        mgr.save(step=1, state={"a": 1})
        record = json.loads((tmp_path / "job.step-000001.json").read_text("utf-8"))
        assert record["schema"] == SCHEMA
        assert record["name"] == "job"
        assert "integrity" in record
        # no temp files survive a successful save
        assert list(tmp_path.glob("*.tmp*")) == []


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
class TestValidation:
    @pytest.mark.parametrize("bad_step", [-1, 1.5, True, "3", None])
    def test_step_must_be_non_negative_int(self, tmp_path: Path, bad_step) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        with pytest.raises(PersistenceError):
            mgr.save(step=bad_step)

    def test_state_must_be_a_dict(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        with pytest.raises(PersistenceError, match="state must be a dict"):
            mgr.save(step=1, state=[1, 2, 3])  # type: ignore[arg-type]

    def test_unserialisable_state_is_rejected(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        with pytest.raises(PersistenceError, match="JSON-serialisable"):
            mgr.save(step=1, state={"frame": object()})

    @pytest.mark.parametrize("bad_name", ["", "a/b", "a\\b", ".", "..", " pad "])
    def test_name_must_be_a_plain_filename(self, tmp_path: Path, bad_name: str) -> None:
        with pytest.raises(PersistenceError):
            CheckpointManager(tmp_path, bad_name)

    @pytest.mark.parametrize("bad_keep", [0, -1, 1.5, True])
    def test_keep_must_be_positive_int(self, tmp_path: Path, bad_keep) -> None:
        with pytest.raises(PersistenceError):
            CheckpointManager(tmp_path, "job", keep=bad_keep)

    def test_directory_is_created_on_demand(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested"
        mgr = CheckpointManager(target, "job")
        mgr.save(step=1)
        assert (target / "job.step-000001.json").exists()


# ---------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------
class TestRotation:
    def test_only_the_newest_keep_survive(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=2)
        for step in range(1, 5):
            mgr.save(step=step)
        names = sorted(path.name for path in tmp_path.glob("job.step-*.json"))
        assert names == ["job.step-000003.json", "job.step-000004.json"]
        assert mgr.latest().step == 4

    def test_keep_equals_one_leaves_exactly_one(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=1)
        for step in (1, 2, 3):
            mgr.save(step=step)
        assert [path.name for path in tmp_path.glob("job.step-*.json")] == [
            "job.step-000003.json"
        ]

    def test_listing_is_newest_first(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=5)
        for step in (1, 2, 3):
            mgr.save(step=step)
        assert [info.step for info in mgr.list()] == [3, 2, 1]


# ---------------------------------------------------------------------------
# corruption quarantine and fallback
# ---------------------------------------------------------------------------
class TestCorruption:
    def test_truncated_newest_falls_back_to_previous(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=5)
        mgr.save(step=1, state={"a": 1})
        mgr.save(step=2, state={"a": 2})
        (tmp_path / "job.step-000002.json").write_text('{"schema": "pharm', encoding="utf-8")

        info = mgr.latest()
        assert info.step == 1  # the previous checkpoint survived
        quarantined = list(tmp_path.glob("job.step-000002.json.corrupt*"))
        assert len(quarantined) == 1

    def test_tampered_state_fails_the_digest(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=5)
        mgr.save(step=1, state={"a": 1})
        mgr.save(step=2, state={"a": 2})
        path = tmp_path / "job.step-000002.json"
        record = json.loads(path.read_text("utf-8"))
        record["state"]["a"] = 999  # the digest no longer covers this
        path.write_text(json.dumps(record), encoding="utf-8")

        assert mgr.latest().step == 1
        assert list(tmp_path.glob("job.step-000002.json.corrupt*"))

    def test_foreign_schema_is_quarantined(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=5)
        mgr.save(step=1, state={"a": 1})
        (tmp_path / "job.step-000002.json").write_text(
            json.dumps({"schema": "someone-elses/9", "step": 2}), encoding="utf-8"
        )
        assert mgr.latest().step == 1

    def test_quarantine_never_overwrites_a_previous_one(
        self, tmp_path: Path
    ) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=5)
        mgr.save(step=1, state={"a": 1})
        for step in (2, 3):
            path = tmp_path / f"job.step-{step:06d}.json"
            path.write_text("garbage", encoding="utf-8")
        assert mgr.latest().step == 1
        quarantined = sorted(
            path.name for path in tmp_path.glob("job.step-*.json.corrupt*")
        )
        assert len(quarantined) == 2
        # each corrupt file keeps its own step in the name; the -N
        # suffix only disambiguates a RE-quarantine of the same step
        assert quarantined[0] == "job.step-000002.json.corrupt"
        assert quarantined[1] == "job.step-000003.json.corrupt"

    def test_all_corrupt_returns_none(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=5)
        mgr.save(step=1, state={"a": 1})
        (tmp_path / "job.step-000001.json").write_text("garbage", encoding="utf-8")
        assert mgr.latest() is None

    def test_empty_directory_returns_none(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "not-yet-created", "job")
        assert mgr.latest() is None
        assert mgr.list() == []

    def test_foreign_name_files_are_invisible(self, tmp_path: Path) -> None:
        CheckpointManager(tmp_path, "other-job").save(step=1)
        mgr = CheckpointManager(tmp_path, "job")
        assert mgr.latest() is None
        assert (tmp_path / "other-job.step-000001.json").exists()


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------
class TestClear:
    def test_clear_removes_checkpoints_and_quarantine(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=5)
        mgr.save(step=1)
        mgr.save(step=2)
        (tmp_path / "job.step-000003.json").write_text("garbage", encoding="utf-8")
        mgr.list()  # quarantines the garbage
        removed = mgr.clear()
        assert removed == 3  # two valid + one quarantined
        assert mgr.latest() is None
        assert list(tmp_path.glob("job.step-*")) == []

    def test_clear_without_a_directory_is_zero(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "absent", "job")
        assert mgr.clear() == 0


# ---------------------------------------------------------------------------
# default_directory
# ---------------------------------------------------------------------------
class TestDefaultDirectory:
    def test_lives_under_the_pharmaml_home(self) -> None:
        path = default_directory("family-benchmark")
        assert path.parts[-3:] == (".pharmaml", "checkpoints", "family-benchmark")

    def test_job_name_is_sanitised(self) -> None:
        path = default_directory("weird/name:field")
        assert "/" not in path.name
