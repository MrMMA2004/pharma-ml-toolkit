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
2. **Unknown operation classes are refused.** (round 11) the §13
   registry is COMPLETE — all six candidate classes earned their
   registration (descriptor_batch R6, similarity_campaign R7, hpo
   R8, generative_search R9, docking_batch R10, preprocessing R11) —
   so `PENDING_OPERATION_TYPES` is empty.  The gate stays armed: a
   type not in `OPERATION_REGISTRY` is rejected with an explicit
   error, so any future candidate class must earn its proof round
   before it can claim resumability.
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
| `descriptor_batch` | molecules featurized | preserve | `tests/test_descriptor_batch_resumability.py::TestDescriptorBatchResume::test_resumed_batch_is_bit_exact` | `pharmaml.chem.features` |
| `similarity_campaign` | references scanned | preserve | `tests/test_similarity_campaign_resumability.py::TestSimilarityCampaignResume::test_resumed_campaign_is_bit_exact` | `pharmaml.chem.similarity` |
| `hpo` | trials evaluated | preserve | `tests/test_hpo_resumability.py::TestHPOResume::test_resumed_study_is_bit_exact` | `pharmaml.ml.hpo` |
| `generative_search` | generations evolved | preserve | `tests/test_generative_search_resumability.py::TestGenerativeSearchResume::test_resumed_search_is_bit_exact` | `pharmaml.generative.genetic` |
| `docking_batch` | ligands docked | preserve | `tests/test_docking_batch_resumability.py::TestDockingBatchResume::test_resumed_batch_is_bit_exact` | `pharmaml.docking.smiles_docking` |
| `preprocessing` | chunks transformed | preserve | `tests/test_preprocessing_resumability.py::TestPreprocessingResume::test_resumed_table_is_bit_exact` | `pharmaml.ml.preprocessing` |

The first of the six §13 candidates earned its row in round 6: the
descriptor batch (`compute_feature_frame(checkpoint_dir=...)`)
checkpoints every chunk boundary with its complete state (per-row float
vectors, the drop ledger, config + dataset fingerprints), resumes by
skipping the completed prefix, refuses a stale config/dataset
checkpoint (exact restore or nothing), and preserves the partial
`(frame, report)` on a cooperative stop via
`JobCancelled.partial_results`. Round 7 added the second:
`run_similarity_campaign(checkpoint_dir=...)` wraps the
`similarity_scan` math in a 25-reference chunk loop whose checkpoints
carry the long-format similarity rows, the dropped-reference ledger,
and the scan-config + dataset fingerprints; a resume replays them
(bit-identical to both the uninterrupted campaign AND the one-shot
`similarity_scan`), a cooperative stop preserves the partial frame on
`JobCancelled.partial_results`, and a stale config/dataset checkpoint
is refused. Round 8 added the third: `tune_model(checkpoint_dir=...)`
— and the round's specific lesson is that the trial history ALONE is
not the complete state: the TPE sampler draws candidates from its
`random.Random`, so the checkpoint carries the RNG state too
(`HPOStudy.capture_state()` / `restore_state()`), making the resumed
study bit-identical to the uninterrupted one where a naive
history-only replay demonstrably diverges. Round 9 added the fourth:
`GeneticOptimizer.optimize(checkpoint_dir=...)` — the generation
boundary's complete state (population, produced book, history, RNG
state, and on adaptive runs the three operator credit books +
trajectory; the round-8 lesson carried over) restores so the resumed
evolution is bit-identical to the uninterrupted run (`elapsed_s`
aside, wall-clock by nature), and a cooperative stop attaches the
partial `GenerationResult` in BOTH the checkpointed and the
checkpoint-less mode. Round 10 added the fifth:
`dock_smiles_array(checkpoint_dir=...)` — the resume unit is the
LIGAND (the per-ligand completion ledger; finished ligands are
PRESERVED verbatim, never re-run — a real engine is stochastic by
nature, so "preserve", not re-execute, is the honest exactness
contract, proven bit-identical against a deterministic fake engine).
Its fingerprint closes the two gaps the R9 audit flagged: the FULL
smiles list + names (a reordered or edited dataset refuses) and the
receptor PDBQT content checksum (an edited receptor refuses). Round
11 completed the registry with `apply_preprocessor_chunked()`: the
chunked (impute -> scale -> encode) transform rests on two verified
properties — the fit is DETERMINISTIC given the table (a resume
refits instead of deserializing sklearn state) and transform is
ROW-PURE (the transform of a slice equals the slice of the
transform) — so the checkpoint is the completed chunks' transformed
rows and a resume skips them bit-exactly; the fingerprint covers the
strategy knobs AND the table itself (columns, dtypes, shape, content
checksum). With that, all six §13 classes the schema header names
are registered and the pending list is empty.

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
