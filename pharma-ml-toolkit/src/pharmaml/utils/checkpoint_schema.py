"""Part 5 §13 — the resumable-operation checkpoint schema (envelope v2).

The storage layer (:mod:`pharmaml.utils.checkpoints`, Part-3 §20) already
makes every checkpoint file atomic, self-verifying (SHA-256 over the
canonical record), rotated and quarantine-on-corruption.  What it does not
record is *what kind of resumable operation* wrote the file, which inputs the
resume depends on, or whether the operation class is even allowed to claim
resumability.  Part 5 §13 demands exactly that, **before** Pause/Resume is
extended to descriptor batches, similarity campaigns, HPO, generative
searches, docking batches or large preprocessing jobs:

    "Add Pause/Resume only for operations whose complete state can be
    serialized and restored exactly.  Use the existing Leaderboards
    checkpoint workflow as a reference, then define and test a checkpoint
    schema before extending resumability."

This module is that schema.  It is an *envelope* written next to the existing
record fields — old readers ignore it, new readers validate it, and nothing
about the on-disk format of existing checkpoints changes.

The contract, in order of importance:

1. **Registered classes only.**  An operation type appears in
   :data:`OPERATION_REGISTRY` only after a bit-exact resume proof test
   exists and is referenced from its entry.  The §13 candidate classes are
   listed in :data:`PENDING_OPERATION_TYPES` and are rejected by
   :class:`OperationSpec` until their proofs land (one round each).
2. **Exact restore or nothing.**  ``restore_exact`` must be ``True``; an
   operation that cannot restore its complete state exactly is not allowed
   to emit envelope-carrying checkpoints at all (§13: "Never present a
   Stop/Resume action that cannot safely resume").
3. **Inputs are fingerprinted.**  The envelope carries a SHA-256 fingerprint
   over the sorted ``(name, sha256, size)`` triples of every input artifact,
   so a resume against different data is detected, not guessed at.
4. **The implementation is fingerprinted.**  The executing module's source
   hash rides the envelope; a resume under changed code is a warning the
   caller can act on, not a silent behavioural drift.
5. **Legacy checkpoints are reported, never upgraded.**  Files without an
   envelope validate as ``legacy=True`` / not resumable through the new
   contract.  There is no silent migration path — a rewrite requires a new
   save.
6. **Future versions refuse, not guess.**  An envelope whose schema version
   is newer than this module's is a hard validation error asking for a
   migration, never a best-effort interpretation.

Examples
--------
>>> from pathlib import Path
>>> from pharmaml.utils.checkpoint_schema import (
...     OperationSpec, build_envelope, validate_checkpoint,
... )
>>> spec = OperationSpec(
...     operation_type="numpy_mlp_training",
...     operation_id="demo-1",
... )
>>> envelope = build_envelope(spec, step=4)
>>> envelope["operation"]["type"]
'numpy_mlp_training'
>>> envelope["resume_unit"]
4
>>> sorted(envelope["operation"])
['id', 'partial_results_policy', 'restore_exact', 'type', 'unit_description']
>>> report = validate_checkpoint(Path("does-not-exist.json"))
>>> report.ok, report.legacy
(False, False)
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharmaml.exceptions import PersistenceError

#: Envelope schema version this module reads and writes.  §13 rule: a newer
#: version found on disk is a *refusal* asking for migration, never a guess.
ENVELOPE_SCHEMA_VERSION = 2

#: The envelope's marker string (independent of the storage layer's schema).
ENVELOPE_SCHEMA = f"pharmaml-checkpoint-envelope/{ENVELOPE_SCHEMA_VERSION}"

#: The top-level record key the envelope is stored under.
ENVELOPE_KEY = "envelope"

#: The partial-results policies §13 allows.  "preserve" keeps completed
#: partial results after cancellation whenever the operation's data contract
#: permits; "discard" declares up front that only a clean restart is safe.
PARTIAL_RESULTS_POLICIES = ("preserve", "discard")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegistryEntry:
    """One proven-resumable operation class.

    Parameters
    ----------
    unit_description:
        What one checkpoint step means for this class ("epochs scored",
        "descriptor families scored"), used in user-facing resume messages.
    partial_results_policy:
        Whether completed units survive a Stop (§13 cancellation contract).
    exact_restore_proof:
        The pytest node id of the test that proves a paused-then-resumed run
        is bit-identical to an uninterrupted one.  Registry membership is
        *earned* by this reference — an entry without a real proof test is
        a lie the schema exists to prevent.
    implementation:
        The module that executes the operation (also the default source of
        the implementation fingerprint).
    """

    unit_description: str
    partial_results_policy: str
    exact_restore_proof: str
    implementation: str


#: Operation classes with landed, referenced bit-exact resume proofs.
#: The two Part-3 §20 consumers are the reference implementations the
#: master prompt names; the §13 candidates stay pending (below) until each
#: earns its own proof round.
OPERATION_REGISTRY: dict[str, RegistryEntry] = {
    "numpy_mlp_training": RegistryEntry(
        unit_description="training epochs",
        partial_results_policy="preserve",
        exact_restore_proof=(
            "tests/test_numpy_neural_checkpoints.py::TestBitExactResume::"
            "test_adam_regressor_resume_is_bit_exact"
        ),
        implementation="pharmaml.ml.numpy_neural",
    ),
    "benchmark_family_scoring": RegistryEntry(
        unit_description="descriptor families scored",
        partial_results_policy="preserve",
        exact_restore_proof=(
            "tests/test_numpy_neural_checkpoints.py::TestBitExactResume::"
            "test_sgd_momentum_resume_is_bit_exact"
        ),
        implementation="pharmaml.ml.training",
    ),
    # (part5-continuation round 6, §13) the first of the six pending
    # candidates to EARN registration: the descriptor batch's complete
    # state (per-row vectors + drops + fingerprints) serializes and
    # restores exactly, proven by the bit-identical resume test below.
    "descriptor_batch": RegistryEntry(
        unit_description="molecules featurized",
        partial_results_policy="preserve",
        exact_restore_proof=(
            "tests/test_descriptor_batch_resumability.py::"
            "TestDescriptorBatchResume::test_resumed_batch_is_bit_exact"
        ),
        implementation="pharmaml.chem.features",
    ),
    # (part5-continuation round 7, §13) the second candidate to EARN
    # registration: the similarity campaign's complete state (the
    # long-format similarity rows + the dropped-reference ledger +
    # the scan-config and dataset fingerprints) serializes and
    # restores exactly, proven by its bit-identical resume test.
    "similarity_campaign": RegistryEntry(
        unit_description="references scanned",
        partial_results_policy="preserve",
        exact_restore_proof=(
            "tests/test_similarity_campaign_resumability.py::"
            "TestSimilarityCampaignResume::test_resumed_campaign_is_bit_exact"
        ),
        implementation="pharmaml.chem.similarity",
    ),
    # (part5-continuation round 8, §13) the third candidate to EARN
    # registration: the HPO study's complete state is the trial history
    # AND the sampler's RNG state (without the latter a resumed TPE
    # would propose different candidates -- the naive history replay
    # is NOT bit-exact, proven by the round's tests).
    "hpo": RegistryEntry(
        unit_description="trials evaluated",
        partial_results_policy="preserve",
        exact_restore_proof=(
            "tests/test_hpo_resumability.py::"
            "TestHPOResume::test_resumed_study_is_bit_exact"
        ),
        implementation="pharmaml.ml.hpo",
    ),
    # (part5-continuation round 9, §13) the fourth candidate to EARN
    # registration: the generative search's complete state (the
    # population + produced book + per-generation history + the RNG
    # state +, on adaptive runs, the three operator credit books and
    # the trajectory) serializes and restores exactly.  The round-8
    # lesson carried over: the RNG state rides in the checkpoint, and
    # a naive history-only replay is proven to diverge.
    "generative_search": RegistryEntry(
        unit_description="generations evolved",
        partial_results_policy="preserve",
        exact_restore_proof=(
            "tests/test_generative_search_resumability.py::"
            "TestGenerativeSearchResume::test_resumed_search_is_bit_exact"
        ),
        implementation="pharmaml.generative.genetic",
    ),
    # (part5-continuation round 10, §13) the fifth candidate to EARN
    # registration: the docking batch's complete state is the per-ligand
    # completion ledger -- every finished entry's full row -- plus the
    # fingerprint that closes the two gaps the R9 audit flagged (the
    # FULL smiles list and the receptor PDBQT content checksum).
    # Finished ligands are PRESERVED verbatim, never re-run: a real
    # docking engine is stochastic by nature, so "preserve" (not
    # re-execute) is the honest exactness contract, proven bit-identical
    # against a deterministic fake engine.
    "docking_batch": RegistryEntry(
        unit_description="ligands docked",
        partial_results_policy="preserve",
        exact_restore_proof=(
            "tests/test_docking_batch_resumability.py::"
            "TestDockingBatchResume::test_resumed_batch_is_bit_exact"
        ),
        implementation="pharmaml.docking.smiles_docking",
    ),
    # (part5-continuation round 11, §13) the sixth and LAST candidate to
    # EARN registration -- with it, all six §13 classes the schema
    # header names are registered.  The chunked preprocessing's state
    # is the completed chunks' transformed rows; the exactness contract
    # rests on two verified properties: the pipeline's fit is
    # DETERMINISTIC given the table (a resume refits instead of
    # deserializing sklearn state) and transform is ROW-PURE (the
    # transform of a slice equals the slice of the transform).  The
    # fingerprint covers the strategy knobs AND the table itself
    # (columns, dtypes, shape, content checksum) -- any edit refuses.
    "preprocessing": RegistryEntry(
        unit_description="chunks transformed",
        partial_results_policy="preserve",
        exact_restore_proof=(
            "tests/test_preprocessing_resumability.py::"
            "TestPreprocessingResume::test_resumed_table_is_bit_exact"
        ),
        implementation="pharmaml.ml.preprocessing",
    ),
}

#: (round 11) EMPTY: all six §13 candidate classes have earned their
#: registration (descriptor_batch R6, similarity_campaign R7, hpo R8,
#: generative_search R9, docking_batch R10, preprocessing R11).  The
#: gate stays armed for any future class the schema grows.
PENDING_OPERATION_TYPES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# specs and fingerprints
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperationSpec:
    """The identity of one resumable job (§13's "complete state" header).

    Raises
    ------
    PersistenceError
        If the operation type is not registered (pending classes included —
        earning registration is the point), the policy is unknown, or
        ``restore_exact`` is not ``True``.
    """

    operation_type: str
    operation_id: str
    unit_description: str | None = None
    partial_results_policy: str | None = None
    restore_exact: bool = True

    def __post_init__(self) -> None:
        entry = OPERATION_REGISTRY.get(self.operation_type)
        if entry is None:
            if self.operation_type in PENDING_OPERATION_TYPES:
                raise PersistenceError(
                    f"operation type {self.operation_type!r} is a §13 "
                    "candidate but not yet registered — register it with a "
                    "referenced bit-exact resume proof before emitting "
                    "envelope-carrying checkpoints"
                )
            raise PersistenceError(
                f"unknown operation type {self.operation_type!r}; known: "
                f"{sorted(OPERATION_REGISTRY)}"
            )
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise PersistenceError(
                f"operation_id must be a non-empty string, got {self.operation_id!r}"
            )
        if self.partial_results_policy is None:
            # class default from the registry entry
            object.__setattr__(
                self, "partial_results_policy", entry.partial_results_policy
            )
        elif self.partial_results_policy not in PARTIAL_RESULTS_POLICIES:
            raise PersistenceError(
                f"partial_results_policy must be one of "
                f"{PARTIAL_RESULTS_POLICIES}, got {self.partial_results_policy!r}"
            )
        elif self.partial_results_policy != entry.partial_results_policy:
            raise PersistenceError(
                f"partial_results_policy {self.partial_results_policy!r} "
                f"contradicts the registry entry for "
                f"{self.operation_type!r} ({entry.partial_results_policy!r})"
            )
        if self.unit_description is None:
            object.__setattr__(self, "unit_description", entry.unit_description)
        if self.restore_exact is not True:
            raise PersistenceError(
                "restore_exact must be True — §13 forbids presenting "
                "Stop/Resume for an operation whose complete state cannot "
                "be restored exactly"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(paths: Sequence[Path | str]) -> dict[str, Any]:
    """Fingerprint every input artifact for the envelope.

    Returns the ``inputs`` envelope member: a SHA-256 over the sorted
    ``(name, sha256, size_bytes)`` triples, plus the artifact list itself so
    a validator can recompute and a resume can report *which* input moved.
    """
    artifacts: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        if not path.is_file():
            raise PersistenceError(f"input artifact not readable: {path}")
        artifacts.append(
            {
                "name": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    artifacts.sort(key=lambda a: (a["name"], a["sha256"], a["size_bytes"]))
    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":"))
    return {
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "artifacts": artifacts,
    }


def implementation_fingerprint(module: Any) -> str:
    """SHA-256 of the executing module's source file (§13 change detector)."""
    file = getattr(module, "__file__", None)
    if not file:
        raise PersistenceError(
            f"cannot fingerprint implementation of {module!r}: no __file__"
        )
    return _sha256_file(Path(file))


def _default_implementation_fingerprint(operation_type: str) -> str:
    module_name = OPERATION_REGISTRY[operation_type].implementation
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - broken install
        raise PersistenceError(
            f"cannot import implementation module {module_name!r} for "
            f"operation type {operation_type!r}: {exc}"
        ) from exc
    return implementation_fingerprint(module)


def build_envelope(
    operation: OperationSpec,
    *,
    step: int,
    inputs: Sequence[Path | str] = (),
    toolkit_version: str | None = None,
    implementation_hash: str | None = None,
) -> dict[str, Any]:
    """Assemble the v2 envelope for one checkpoint save.

    ``step`` becomes ``resume_unit`` — the unit index a resume must continue
    from (the storage layer's step, kept in lockstep by
    :meth:`CheckpointManager.save <pharmaml.utils.checkpoints.CheckpointManager.save>`).
    """
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise PersistenceError(f"resume step must be a non-negative int, got {step!r}")
    if toolkit_version is None:
        from pharmaml import __version__ as toolkit_version
    if implementation_hash is None:
        implementation_hash = _default_implementation_fingerprint(
            operation.operation_type
        )
    entry = OPERATION_REGISTRY[operation.operation_type]
    return {
        "schema": ENVELOPE_SCHEMA,
        "operation": {
            "type": operation.operation_type,
            "id": operation.operation_id,
            "unit_description": operation.unit_description or entry.unit_description,
            "partial_results_policy": operation.partial_results_policy
            or entry.partial_results_policy,
            "restore_exact": True,
        },
        "toolkit_version": toolkit_version,
        "implementation_fingerprint": implementation_hash,
        "inputs": input_fingerprint(inputs) if tuple(inputs) else None,
        "resume_unit": step,
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationReport:
    """The outcome of :func:`validate_checkpoint`.

    ``legacy`` marks a storage-layer-valid checkpoint written before the
    envelope existed: readable, but **not resumable** under the §13 contract
    and never silently upgraded.
    """

    path: Path
    ok: bool
    legacy: bool
    errors: tuple[str, ...] = ()
    envelope: dict[str, Any] | None = None
    record: dict[str, Any] | None = None
    storage_valid: bool = False


def _recompute_storage_integrity(record: dict[str, Any]) -> bool:
    stored = record.pop("integrity", None)
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return stored == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_envelope(envelope: Any, record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(envelope, dict):
        return [f"{ENVELOPE_KEY} must be an object, got {type(envelope).__name__}"]
    schema = envelope.get("schema")
    if schema != ENVELOPE_SCHEMA:
        version = None
        if isinstance(schema, str) and schema.count("/") == 1:
            try:
                version = int(schema.rsplit("/", 1)[1])
            except ValueError:
                version = None
        if version is not None and version > ENVELOPE_SCHEMA_VERSION:
            errors.append(
                f"envelope schema {schema!r} is newer than supported "
                f"{ENVELOPE_SCHEMA!r} — a migration is required, refusing "
                "to guess (§13 rule 6)"
            )
        else:
            errors.append(f"envelope schema must be {ENVELOPE_SCHEMA!r}, got {schema!r}")
        return errors
    operation = envelope.get("operation")
    if not isinstance(operation, dict):
        errors.append("envelope.operation must be an object")
        return errors
    op_type = operation.get("type")
    entry = OPERATION_REGISTRY.get(op_type) if isinstance(op_type, str) else None
    if entry is None:
        known = ", ".join(sorted(OPERATION_REGISTRY))
        errors.append(
            f"envelope operation type {op_type!r} is not registered "
            f"(known: {known}); pending §13 classes must not appear on disk"
        )
    else:
        if operation.get("partial_results_policy") != entry.partial_results_policy:
            errors.append(
                f"envelope partial_results_policy {operation.get('partial_results_policy')!r} "
                f"contradicts the registry entry ({entry.partial_results_policy!r})"
            )
    if operation.get("restore_exact") is not True:
        errors.append("envelope restore_exact must be true (§13 rule 2)")
    if not isinstance(operation.get("id"), str) or not operation.get("id"):
        errors.append("envelope operation id must be a non-empty string")
    if not isinstance(envelope.get("toolkit_version"), str) or not envelope.get(
        "toolkit_version"
    ):
        errors.append("envelope toolkit_version must be a non-empty string")
    if not isinstance(envelope.get("implementation_fingerprint"), str):
        errors.append("envelope implementation_fingerprint must be a string")
    resume_unit = envelope.get("resume_unit")
    if not isinstance(resume_unit, int) or isinstance(resume_unit, bool) or resume_unit < 0:
        errors.append(f"envelope resume_unit must be a non-negative int, got {resume_unit!r}")
    elif record.get("step") != resume_unit:
        errors.append(
            f"envelope resume_unit ({resume_unit}) disagrees with the "
            f"record step ({record.get('step')!r}) — a resume would start "
            "from the wrong unit"
        )
    inputs = envelope.get("inputs")
    if inputs is not None:
        if not isinstance(inputs, dict):
            errors.append("envelope.inputs must be an object or null")
        else:
            artifacts = inputs.get("artifacts")
            if not isinstance(artifacts, list):
                errors.append("envelope.inputs.artifacts must be a list")
            else:
                canonical = json.dumps(
                    artifacts, sort_keys=True, separators=(",", ":")
                )
                recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if inputs.get("fingerprint") != recomputed:
                    errors.append(
                        "envelope inputs fingerprint mismatch — the recorded "
                        "artifact list does not hash to the recorded fingerprint"
                    )
    return errors


def validate_checkpoint(
    path: Path | str, *, manager_name: str | None = None
) -> ValidationReport:
    """Full §13 conformance check of one checkpoint file.

    Layers, outermost first: readable JSON → storage-layer schema/name/
    integrity (the Part-3 §20 rules) → the v2 envelope rules.  A file that
    fails the storage layer is not further interpreted.  A storage-valid
    file without an envelope is ``legacy`` — reported, never upgraded.
    """
    # imported here, not at module level: checkpoints.py imports this
    # module for save()-time envelope construction, so a module-level
    # import back would be a cycle.
    from pharmaml.utils.checkpoints import SCHEMA as STORAGE_SCHEMA

    file = Path(path)
    if not file.is_file():
        return ValidationReport(
            path=file, ok=False, legacy=False, errors=(f"no such file: {file}",)
        )
    try:
        record = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ValidationReport(
            path=file, ok=False, legacy=False, errors=(f"unreadable/invalid JSON: {exc}",)
        )
    if not isinstance(record, dict):
        return ValidationReport(
            path=file, ok=False, legacy=False, errors=("record must be a JSON object",)
        )

    storage_errors: list[str] = []
    if record.get("schema") != STORAGE_SCHEMA:
        storage_errors.append(f"storage schema must be {STORAGE_SCHEMA!r}")
    if manager_name is not None and record.get("name") != manager_name:
        storage_errors.append(
            f"checkpoint name {record.get('name')!r} does not match expected {manager_name!r}"
        )
    record_copy = json.loads(json.dumps(record))  # never mutate the loaded copy
    if not _recompute_storage_integrity(record_copy):
        storage_errors.append("storage-layer integrity digest mismatch (tampered?)")
    if storage_errors:
        return ValidationReport(
            path=file, ok=False, legacy=False, errors=tuple(storage_errors), record=record
        )

    envelope = record.get(ENVELOPE_KEY)
    if envelope is None:
        return ValidationReport(
            path=file,
            ok=False,
            legacy=True,
            errors=(
                (
                    "legacy checkpoint (no §13 envelope): readable but NOT "
                    "resumable under the v2 contract; a resume requires a fresh "
                    "save from the registered operation"
                ),
            ),
            record=record,
            storage_valid=True,
        )

    errors = _validate_envelope(envelope, record)
    return ValidationReport(
        path=file,
        ok=not errors,
        legacy=False,
        errors=tuple(errors),
        envelope=envelope,
        record=record,
        storage_valid=True,
    )


def describe_report(report: ValidationReport) -> str:
    """Render a one-screen human summary of a validation report."""
    lines = [f"checkpoint: {report.path}"]
    if report.ok:
        envelope = report.envelope or {}
        operation = envelope.get("operation", {})
        lines.append("  VALID under the §13 v2 envelope")
        lines.append(f"  operation:  {operation.get('type')} ({operation.get('id')})")
        lines.append(f"  resume at:  unit {envelope.get('resume_unit')}")
        lines.append(f"  toolkit:    {envelope.get('toolkit_version')}")
        inputs = envelope.get("inputs")
        lines.append(
            f"  inputs:     {len(inputs['artifacts']) if inputs else 'none recorded'}"
        )
    elif report.legacy:
        lines.append("  LEGACY: storage-valid, no §13 envelope — not resumable, not upgraded")
    else:
        lines.append("  INVALID:")
        lines.extend(f"    - {error}" for error in report.errors)
    return "\n".join(lines)


__all__ = [
    "ENVELOPE_KEY",
    "ENVELOPE_SCHEMA",
    "ENVELOPE_SCHEMA_VERSION",
    "OPERATION_REGISTRY",
    "PARTIAL_RESULTS_POLICIES",
    "PENDING_OPERATION_TYPES",
    "OperationSpec",
    "RegistryEntry",
    "ValidationReport",
    "build_envelope",
    "describe_report",
    "implementation_fingerprint",
    "input_fingerprint",
    "validate_checkpoint",
]
