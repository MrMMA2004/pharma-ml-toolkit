"""Tests for :mod:`pharmaml.utils.checkpoint_schema` — Part-5 §13 envelope.

The §13 contract this suite pins:

* the envelope is written only for **registered** operation classes — the
  §13 candidates (descriptor batches, similarity campaigns, HPO, generative
  searches, docking batches, preprocessing) are rejected until the round
  that lands their bit-exact resume proof registers them;
* ``restore_exact`` must be True: an operation that cannot restore its
  complete state exactly may not emit envelope-carrying checkpoints;
* inputs are fingerprinted (SHA-256 over sorted name/hash/size triples) and
  the fingerprint is recomputed on validation — a resume against different
  data is detected, never guessed at;
* ``resume_unit`` must agree with the storage step — a resume that would
  start from the wrong unit is a validation failure;
* legacy checkpoints (Part-3 §20 files without an envelope) are reported
  as legacy/not-resumable and are **never silently upgraded**;
* a future envelope version is a hard refusal asking for migration, not a
  best-effort interpretation;
* the storage layer's guarantees (atomicity, integrity digest, rotation,
  quarantine) are untouched — a save with an envelope is byte-compatible
  with the old format plus one top-level key.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pharmaml.exceptions import PersistenceError
from pharmaml.utils.checkpoint_schema import (
    ENVELOPE_KEY,
    ENVELOPE_SCHEMA,
    ENVELOPE_SCHEMA_VERSION,
    OPERATION_REGISTRY,
    PENDING_OPERATION_TYPES,
    OperationSpec,
    build_envelope,
    describe_report,
    implementation_fingerprint,
    input_fingerprint,
    validate_checkpoint,
)
from pharmaml.utils.checkpoints import SCHEMA, CheckpointManager


def _spec(operation_type: str = "numpy_mlp_training", operation_id: str = "job-1") -> OperationSpec:
    return OperationSpec(operation_type=operation_type, operation_id=operation_id)


def _reseal(path: Path, mutate) -> None:
    """Rewrite a checkpoint after mutating it, re-sealing the storage digest.

    §13 threat model: the storage integrity hash covers the whole record,
    so tampering that wants to *reach* envelope validation must re-forge
    the outer seal — these tests verify the envelope catches it anyway.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    record.pop("integrity", None)
    canonical = json.dumps(
        record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    record["integrity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# OperationSpec: the registry gate
# ---------------------------------------------------------------------------
class TestOperationSpec:
    def test_registered_class_defaults_from_registry(self) -> None:
        spec = _spec()
        entry = OPERATION_REGISTRY["numpy_mlp_training"]
        assert spec.unit_description == entry.unit_description
        assert spec.partial_results_policy == entry.partial_results_policy
        assert spec.restore_exact is True

    def test_pending_section13_class_is_rejected_until_proven(self) -> None:
        # every §13 candidate must be refused: earning registration is the point
        assert PENDING_OPERATION_TYPES, "the §13 candidate list must not be empty"
        for pending in PENDING_OPERATION_TYPES:
            assert pending not in OPERATION_REGISTRY
            with pytest.raises(PersistenceError, match="candidate"):
                OperationSpec(operation_type=pending, operation_id="x")

    def test_unknown_class_is_rejected(self) -> None:
        with pytest.raises(PersistenceError, match="unknown operation type"):
            _spec(operation_type="teleportation")

    def test_restore_exact_false_is_rejected(self) -> None:
        with pytest.raises(PersistenceError, match="restore_exact"):
            OperationSpec(
                operation_type="numpy_mlp_training",
                operation_id="x",
                restore_exact=False,
            )

    def test_policy_contradiction_is_rejected(self) -> None:
        with pytest.raises(PersistenceError, match="contradicts"):
            OperationSpec(
                operation_type="numpy_mlp_training",
                operation_id="x",
                partial_results_policy="discard",
            )

    def test_empty_operation_id_is_rejected(self) -> None:
        with pytest.raises(PersistenceError, match="operation_id"):
            _spec(operation_id="   ")

    def test_every_registry_entry_cites_a_proof_test(self) -> None:
        # registration is earned: each entry names a real pytest node id
        for name, entry in OPERATION_REGISTRY.items():
            assert entry.exact_restore_proof.startswith("tests/"), name
            assert "::" in entry.exact_restore_proof, name
            assert entry.implementation.startswith("pharmaml."), name
            assert entry.partial_results_policy in ("preserve", "discard"), name


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------
class TestFingerprints:
    def test_input_fingerprint_is_order_insensitive(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
        a.write_text("smiles\nCCO\n")
        b.write_text("smiles\nCCC\n")
        first = input_fingerprint([a, b])
        second = input_fingerprint([b, a])
        assert first["fingerprint"] == second["fingerprint"]
        assert {art["name"] for art in first["artifacts"]} == {"a.csv", "b.csv"}

    def test_input_fingerprint_tracks_content(self, tmp_path: Path) -> None:
        data = tmp_path / "data.csv"
        data.write_text("one")
        first = input_fingerprint([data])
        data.write_text("two")
        second = input_fingerprint([data])
        assert first["fingerprint"] != second["fingerprint"]
        assert first["artifacts"][0]["size_bytes"] == 3

    def test_input_fingerprint_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceError, match="not readable"):
            input_fingerprint([tmp_path / "ghost.csv"])

    def test_implementation_fingerprint_of_a_module(self) -> None:
        import pharmaml.utils.checkpoint_schema as module

        digest = implementation_fingerprint(module)
        assert len(digest) == 64
        assert digest == implementation_fingerprint(module)  # stable

    def test_implementation_fingerprint_rejects_moduleless_object(self) -> None:
        with pytest.raises(PersistenceError, match="no __file__"):
            implementation_fingerprint(object())


# ---------------------------------------------------------------------------
# envelope construction and validation
# ---------------------------------------------------------------------------
class TestBuildEnvelope:
    def test_resume_unit_follows_step(self) -> None:
        envelope = build_envelope(_spec(), step=4)
        assert envelope["resume_unit"] == 4
        assert envelope["schema"] == ENVELOPE_SCHEMA
        assert envelope["operation"]["restore_exact"] is True
        assert envelope["toolkit_version"]

    def test_step_must_be_a_non_negative_int(self) -> None:
        for bad in (-1, 1.5, True, "3"):
            with pytest.raises(PersistenceError):
                build_envelope(_spec(), step=bad)  # type: ignore[arg-type]

    def test_inputs_recorded_when_given(self, tmp_path: Path) -> None:
        data = tmp_path / "d.parquet"
        data.write_text("x")
        envelope = build_envelope(_spec(), step=0, inputs=[data])
        assert envelope["inputs"]["artifacts"][0]["name"] == "d.parquet"
        assert envelope["inputs"]["fingerprint"] == input_fingerprint([data])["fingerprint"]

    def test_inputs_absent_when_empty(self) -> None:
        assert build_envelope(_spec(), step=0)["inputs"] is None


class TestValidateCheckpoint:
    def test_valid_envelope_round_trip(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(
            step=6, total=20, state={"epoch": 6}, operation=_spec(),
        )
        assert info.resumable
        assert info.envelope is not None
        assert info.envelope["resume_unit"] == 6

        report = validate_checkpoint(info.path, manager_name="job")
        assert report.ok, report.errors
        assert not report.legacy
        assert report.storage_valid
        assert describe_report(report).startswith("checkpoint:")

    def test_legacy_checkpoint_reported_not_upgraded(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "old")
        info = mgr.save(step=3, total=9, state={"rows": [1, 2, 3]})
        before = info.path.read_bytes()

        report = validate_checkpoint(info.path)
        assert not report.ok
        assert report.legacy
        assert report.storage_valid
        assert "not resumable" in " ".join(report.errors).lower()
        # never upgraded: the file is byte-identical after validation
        assert info.path.read_bytes() == before
        # and the manager still reads it exactly as before
        assert mgr.latest() is not None and not mgr.latest().resumable  # type: ignore[union-attr]

    def test_tampered_resume_unit_is_caught(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(step=2, state={}, operation=_spec())
        _reseal(info.path, lambda r: r[ENVELOPE_KEY].__setitem__("resume_unit", 9))
        report = validate_checkpoint(info.path)
        assert not report.ok
        assert any("resume_unit" in error for error in report.errors)

    def test_tampered_input_fingerprint_is_caught(self, tmp_path: Path) -> None:
        data = tmp_path / "d.csv"
        data.write_text("x")
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(step=1, state={}, operation=_spec(), inputs=[data])
        _reseal(
            info.path,
            lambda r: r[ENVELOPE_KEY]["inputs"]["artifacts"][0].__setitem__(
                "sha256", "0" * 64
            ),
        )
        report = validate_checkpoint(info.path)
        assert not report.ok
        assert any("fingerprint mismatch" in error for error in report.errors)

    def test_unregistered_operation_type_on_disk_is_caught(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(step=1, state={}, operation=_spec())
        _reseal(
            info.path,
            lambda r: r[ENVELOPE_KEY]["operation"].__setitem__("type", "docking_batch"),
        )
        report = validate_checkpoint(info.path)
        assert not report.ok
        assert any("not registered" in error for error in report.errors)

    def test_future_envelope_version_refuses_with_migration_error(
        self, tmp_path: Path
    ) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(step=1, state={}, operation=_spec())
        _reseal(
            info.path,
            lambda r: r[ENVELOPE_KEY].__setitem__(
                "schema", f"pharmaml-checkpoint-envelope/{ENVELOPE_SCHEMA_VERSION + 1}"
            ),
        )
        report = validate_checkpoint(info.path)
        assert not report.ok
        assert any("migration" in error for error in report.errors)

    def test_storage_layer_failure_short_circuits(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(step=1, state={}, operation=_spec())
        record = json.loads(info.path.read_text(encoding="utf-8"))
        record["step"] = 99  # storage tamper WITHOUT resealing
        info.path.write_text(json.dumps(record), encoding="utf-8")
        report = validate_checkpoint(info.path)
        assert not report.ok and not report.legacy
        assert not report.storage_valid
        assert any("integrity" in error for error in report.errors)

    def test_manager_name_mismatch_is_caught(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        info = mgr.save(step=1, state={}, operation=_spec())
        report = validate_checkpoint(info.path, manager_name="other-job")
        assert not report.ok
        assert any("other-job" in error for error in report.errors)

    def test_missing_and_garbage_files(self, tmp_path: Path) -> None:
        missing = validate_checkpoint(tmp_path / "ghost.json")
        assert not missing.ok and not missing.legacy
        junk = tmp_path / "junk.json"
        junk.write_text("{not json", encoding="utf-8")
        report = validate_checkpoint(junk)
        assert not report.ok and not report.legacy
        array = tmp_path / "array.json"
        array.write_text("[1, 2]", encoding="utf-8")
        assert not validate_checkpoint(array).ok


# ---------------------------------------------------------------------------
# storage-layer compatibility (nothing about Part-3 §20 may regress)
# ---------------------------------------------------------------------------
class TestStorageCompatibility:
    def test_envelope_save_keeps_every_storage_field(
        self, tmp_path: Path
    ) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=2)
        info = mgr.save(
            step=7, total=12, state={"rows": [1]}, note="unit 7",
            operation=_spec(),
        )
        record = json.loads(info.path.read_text(encoding="utf-8"))
        assert record["schema"] == SCHEMA
        assert record["name"] == "job"
        assert record["step"] == 7 and record["total"] == 12
        assert record["note"] == "unit 7"
        assert set(record) >= {
            "schema", "name", "step", "total", "note", "saved_at",
            "elapsed_s", "state", "integrity", ENVELOPE_KEY,
        }

    def test_reserved_state_key_is_rejected(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        with pytest.raises(PersistenceError, match="reserved"):
            mgr.save(step=1, state={ENVELOPE_KEY: {"forged": True}})

    def test_rotation_still_applies_to_envelope_saves(
        self, tmp_path: Path
    ) -> None:
        mgr = CheckpointManager(tmp_path, "job", keep=2)
        for step in range(1, 6):
            mgr.save(step=step, state={"n": step}, operation=_spec())
        survivors = sorted(p.name for p in tmp_path.glob("job.step-*.json"))
        assert survivors == ["job.step-000004.json", "job.step-000005.json"]

    def test_corrupt_envelope_file_is_quarantined_like_any_other(
        self, tmp_path: Path
    ) -> None:
        mgr = CheckpointManager(tmp_path, "job")
        good = mgr.save(step=1, state={}, operation=_spec())
        newest = mgr.save(step=2, state={}, operation=_spec())
        newest.path.write_text("{truncated", encoding="utf-8")
        resumed = mgr.latest()
        assert resumed is not None
        assert resumed.step == 1  # fell back to the previous valid checkpoint
        assert resumed.resumable
        quarantined = list(tmp_path.glob("*.corrupt"))
        assert len(quarantined) == 1
        assert good.path.exists()
