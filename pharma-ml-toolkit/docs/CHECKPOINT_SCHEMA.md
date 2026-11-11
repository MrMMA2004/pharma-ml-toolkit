# The §13 checkpoint schema (envelope v2)

**Status:** landed in `part5-continuation` round 2 (2026-11-12)
**Spec:** MASTER_PROMPT part5 §13 — "define and test a checkpoint schema
**before** extending resumability to descriptor batches, similarity
campaigns, HPO, generative searches, docking batches, or large
preprocessing jobs"
**Implementation:** `src/pharmaml/utils/checkpoint_schema.py` (envelope +
registry + validator) on top of `src/pharmaml/utils/checkpoints.py`
(Part-3 §20 storage layer — untouched guarantees)
**Tests:** `tests/test_checkpoint_schema.py` (29 focused tests) + the
existing §20 suites (regression) + module doctests

## What exists vs. what this adds

The Part-3 §20 storage layer already guarantees: atomic saves
(temp + fsync + `os.replace`), a SHA-256 integrity digest over the canonical
record, rotation (`keep` newest), and quarantine-on-corruption with fallback
to the previous valid checkpoint. What it does not record is *what kind of
resumable operation* wrote a file, which inputs a safe resume depends on, or
whether the operation class has earned the right to claim resumability at
all. The envelope adds exactly that, as one extra top-level JSON key — old
readers ignore it, old files remain readable, nothing about the on-disk
format otherwise changes.

## The envelope (written by `CheckpointManager.save(..., operation=spec)`)

```json
{
  "schema": "pharmaml-checkpoint-envelope/2",
  "operation": {
    "type": "numpy_mlp_training",
    "id": "<job-unique string>",
    "unit_description": "training epochs",
    "partial_results_policy": "preserve",
    "restore_exact": true
  },
  "toolkit_version": "1.2.174",
  "implementation_fingerprint": "<sha256 of the executing module's source>",
  "inputs": {
    "fingerprint": "<sha256 over sorted (name, sha256, size) triples>",
    "artifacts": [{"name": "...", "sha256": "...", "size_bytes": 0}]
  },
  "resume_unit": 5
}
```

(`inputs` is `null` when no artifacts were declared. `resume_unit` always
equals the storage `step` — a mismatch is a validation failure, because a
resume would start from the wrong unit.)

## The rules (each one is pinned by tests)

1. **Registered classes only.** `OperationSpec(operation_type=...)` accepts
   only types present in `OPERATION_REGISTRY`. Registry membership requires
   a referenced pytest node id proving bit-exact resume for that class
   (`RegistryEntry.exact_restore_proof`).
2. **Pending §13 classes are refused.** `descriptor_batch`,
   `similarity_campaign`, `hpo`, `generative_search`, `docking_batch`,
   `preprocessing` are listed in `PENDING_OPERATION_TYPES` and rejected with
   an explicit "not yet registered" error until their proof round lands
   (planned rounds R6+ in
   `docs/plans/PART5_S11_S14_CONTINUATION.md`).
3. **Exact restore or nothing.** `restore_exact` must be `True`
   (construction-time and on-disk validation). §13: "Never present a
   Stop/Resume action that cannot safely resume."
4. **Inputs are fingerprinted** and the fingerprint is recomputed on
   validation; `input_fingerprint()` is order-insensitive and
   content-sensitive.
5. **The implementation is fingerprinted** (module source hash) so a resume
   under changed code can be detected by the caller.
6. **Legacy checkpoints are reported, never upgraded.** A storage-valid
   file without an envelope validates as `legacy=True`, "readable but NOT
   resumable under the v2 contract", and the file is left byte-identical.
   `CheckpointInfo.resumable` is `False` for those.
7. **Future versions refuse.** An envelope schema newer than v2 is a hard
   error asking for a migration — never a best-effort interpretation.
8. **The reserved key.** `save()` rejects a state dict that claims the
   `envelope` key; the envelope is passed via `operation=`, not forged
   through user state.
9. **Storage guarantees are untouched.** Envelope-carrying saves are still
   atomic, integrity-sealed, rotated and quarantine-on-corruption (the
   corrupt-newest → fall-back-to-previous behaviour is regression-tested
   with envelope files).

## Current registry

| Type | Unit | Partial results | Proof | Implementation |
| --- | --- | --- | --- | --- |
| `numpy_mlp_training` | training epochs | preserve | `tests/test_numpy_neural_checkpoints.py::TestBitExactResume::test_adam_regressor_resume_is_bit_exact` | `pharmaml.ml.numpy_neural` |
| `benchmark_family_scoring` | descriptor families scored | preserve | `tests/test_numpy_neural_checkpoints.py::TestBitExactResume::test_sgd_momentum_resume_is_bit_exact` | `pharmaml.ml.training` |

The six §13 candidates are pending; each earns its row in its own round
(descriptor batches first), exactly as the master prompt sequences it.

## Using it

```python
from pharmaml.utils import CheckpointManager, OperationSpec

spec = OperationSpec(operation_type="numpy_mlp_training", operation_id="run-42")
mgr = CheckpointManager(checkpoint_dir, "numpy-mlp")
info = mgr.save(step=epoch, total=epochs, state=..., operation=spec,
                inputs=[dataset_path])          # envelope rides along
resumed = mgr.latest()                          # .resumable, .envelope
```

```bash
python tools/checkpoint.py --validate <checkpoint.json>
# exit 0 = valid v2 envelope; 1 = invalid OR legacy (the report says which)
```

## Migration policy

Writers stamp `pharmaml-checkpoint-envelope/2`; readers validate before
use. A backward-incompatible change bumps the version and lands a migration
together with its tests **before** any writer emits the new version. There
is no silent migration of legacy files: a resume under the v2 contract
requires a fresh save from a registered operation.
