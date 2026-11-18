# Part 5 Continuation Plan — §11–§14 onto main @ v1.2.174

**Prepared:** 2026-11-12, `part5-continuation` round 1
**Base:** `main` @ `e2f88d1` (round 267 close-out; v1.2.174; remote main-only)
**Authoritative spec:** MASTER_PROMPT part5 §11–§16
**Status inherited:** the Part 5 phase-0 audit
(`docs/PART5_PHASE0_AUDIT.md`) plus round 267's landing of the arena work
(transactional restore, estimator validation, engine discovery tests,
portability markers, release-audit tooling, docs-index/example-data/drift
tooling). Per that matrix, the remaining open areas are:

| Part 5 area | Phase-0 matrix status | Round-267 delta |
| --- | --- | --- |
| §11 dashboard real evidence | **Not implemented** | none |
| §12 GUI/viz/export gaps | **Not implemented** (partially covered by earlier Studio work) | none |
| §13 pause/resume schema | **Not implemented** (Part-3 §20 `CheckpointManager` exists as the reference) | checkpoint build-safety hardening only |
| §14 release QA artifact | **Partially implemented** (`tools/audit_release.py`; v1.2.174 shipped with honest totals) | shipped for v1.2.174 |

Working rules carried over from part5 §15: reproduce → classify → focused
regression test → smallest safe patch → focused test → adjacent tests →
profile → diff review → docs → checkpoint. Never count unexecuted tests as
passed.

---

## Round map (branch: `part5-continuation`, pushed after every round)

| Round | Scope | Part 5 hook | Exit gate |
| --- | --- | --- | --- |
| R1 (done) | Security kit (this doc set) + registry reconciliation + `.gitignore` gap | precondition | pushed `a31dbac` |
| R2 (done) | Checkpoint schema v1 — formal envelope + validator + conformance tests | §13 first half | pushed; 29 new tests + §20 regressions + doctests green; ruff + mypy clean |
| R3 (done) | §11 nine-card live previews — real evidence per card, honest empty states | §11 (P3) | pushed; 44 new tests (32 evidence + 12 wiring) + 26 part-4 pins green; randomness-ban source scan green; docs/NINE_CARD_EVIDENCE.md |
| R4 (done) | §12 audit-first GUI gap table + first confirmed fixes | §12 (P3) | pushed; `docs/GUI_GAP_AUDIT.md` (17 rows: 3 fixed, 14 not-a-gap with citations); inspector dense-plot coalescing + MD export reopen-validation; 6 new focused GUI tests; ruff zero on round files, mypy clean on src/pharmaml; full UI battery 532/532 on the pinned stack (the rdkit 2026.9.1 chase — R3's drift attribution corrected) |
| R5 (done) | §14 report schema v2 — fold profile results + gates into the release audit artifact | §14 (P0/1) | pushed; `--profile-results` (schema pharmaml-profile-results/2, fail-closed honest-labelling validation) folds per-profile counts/gates/backup/limitations into report.json + SUMMARY.md from one source; 22 new tests (28 in the suite); the validator's collected-before-compare ordering bug caught by the new tests and fixed; a real report shipped under docs/release-audits/v1.2.174-part5-r5-* |
| R6 (done) | First §13 extension on the schema: descriptor batches (exact-restore proof) | §13 second half | pushed; `compute_feature_frame(checkpoint_dir=…)` — chunk-boundary envelope checkpoints carrying the complete state (per-row b64 float vectors, drop ledger, config+dataset fingerprints); resume skips the completed prefix bit-identically; stale config/dataset refuses (exact restore or nothing); cooperative stop preserves the partial `(frame, report)` on `JobCancelled.partial_results` (§13 "whenever the data contract permits"); `descriptor_batch` registered in OPERATION_REGISTRY with the proof node; 11-test proof suite; focused battery 593 passed (features/descriptors/schema/tooling/numpy-neural/pipeline/cli + doctests); ruff + mypy clean |
| R7 (done) | Second §13 extension: similarity campaigns (exact-restore proof) | §13 second half | pushed; `run_similarity_campaign(checkpoint_dir=…)` — the `similarity_scan` math in a 25-reference chunk loop with envelope checkpoints carrying the long-format similarity rows + dropped-reference ledger + scan-config/dataset fingerprints; a resume is bit-identical to BOTH the uninterrupted campaign and the one-shot `similarity_scan`; stale config/dataset refuses; cooperative stop preserves the partial frame on `JobCancelled.partial_results`; `similarity_campaign` registered in OPERATION_REGISTRY with the proof node; 11-test proof suite; focused battery 107 passed (similarity/descriptor-batch/schema/tooling + doctests); ruff + mypy clean |
| R8 (done) | Third §13 extension: HPO studies (exact-restore proof, RNG-state lesson) | §13 second half | pushed; `tune_model(checkpoint_dir=…)` — the trial loop with envelope checkpoints carrying the COMPLETE state: trial history AND the sampler's RNG state (`HPOStudy.capture_state()/restore_state()`); without the RNG state a resumed TPE demonstrably diverges (naive history replay pinned by its own test); a resume is bit-identical to the uninterrupted study AND the checkpoint-less call; stale config (model/space/sampler/seed/cv/n_trials) or dataset refuses (exact restore or nothing); cooperative stop preserves the partial TuneReport on `JobCancelled.partial_results`; `hpo` registered in OPERATION_REGISTRY with the proof node; 12-test proof suite; focused battery 282 passed (hpo/presets/multiobjective/resumability×3/schema/tooling/numpy-neural/carriers + doctests); ruff + mypy clean |
| R9 (done) | Fourth §13 extension: generative search (exact-restore proof, adaptive books) | §13 second half | pushed; `GeneticOptimizer.optimize(checkpoint_dir=…)` — generation-boundary envelope checkpoints carrying the COMPLETE state: population + produced book + history + the RNG state (round-8 lesson carried over) + the three operator credit books and trajectory on adaptive runs; a resume is bit-identical to the uninterrupted run AND the checkpoint-less call (elapsed_s aside, wall-clock by nature); stale config (any GA knob or the objective identity) refuses (exact restore or nothing); a cooperative stop attaches the partial GenerationResult on `JobCancelled.partial_results` in BOTH modes (strictly richer than the round-205 None convention); `generative_search` registered in OPERATION_REGISTRY with the proof node; 10-test proof suite; focused battery 295 passed + generative regression 322 passed; ruff + mypy clean |
| R10 (done) | Fifth §13 extension: the docking batch (per-ligand ledger, preserve semantics, the R9-gap closures) | §13 second half | pushed; `dock_smiles_array(checkpoint_dir=…)` — the resume unit is the LIGAND: the per-ligand completion ledger (every finished entry's full row) rides envelope checkpoints every checkpoint_every finished ligands; finished ligands are PRESERVED verbatim, never re-run (a real engine is stochastic by nature — "preserve", not re-execute; proven bit-identical against a deterministic fake engine via the monkeypatchable prep/engine seams); the fingerprint closes both R9-audit gaps: the FULL smiles list + names (shortened/extended/reordered/renamed all refuse) AND the receptor PDBQT content checksum (an edited receptor refuses); a cooperative stop raises JobCancelled with the partial result frame (workers>1 runs in waves so the stop lands at a clean barrier — never mid-wave); preparation failures are final and ledgered (never re-prepared); `docking_batch` registered in OPERATION_REGISTRY with the proof node; 11-test proof suite; focused battery 322 passed; ruff + mypy (217 files) clean |
| R11 (done) | Sixth and final §13 extension: the chunked preprocessing (refit-on-resume determinism, row-pure transform) — THE REGISTRY IS COMPLETE | §13 second half | pushed; `apply_preprocessor_chunked(checkpoint_dir=…)` in `pharmaml/ml/preprocessing.py` — the standard (impute -> scale -> encode) pipeline fitted ONCE (deterministically: a resume refits rather than deserializing sklearn state) and applied chunk_rows at a time; transform verified ROW-PURE (transform of a slice == slice of the transform) so the checkpoint is the completed chunks' transformed rows and a resume skips them bit-exactly (equal to BOTH the uninterrupted chunked run AND the one-shot fit_transform); the fingerprint covers the strategy knobs AND the table itself (columns, dtypes, shape, content checksum — a value edit, a rename, or a shape edit all refuse); a cooperative stop preserves the partial (frame, report) on JobCancelled.partial_results; the complete-run checkpoint restores with ZERO chunks re-transformed (the smoke-caught double-append bug — a complete ledger re-appending the last chunk — fixed before the proofs); `preprocessing` registered, PENDING_OPERATION_TYPES now EMPTY; 10-test proof suite; six-proof battery 94/94; ruff + mypy (217 files) clean |
| R12 (done) | §12 round 2 — the preserved-partial channel: the GUI surfaces what a §13 Stop kept | §12 + §13 | pushed; `ui/workers/job_worker.py` — the round-4 audit's row 18: the six registered §13 operations attach `partial_results` to `JobCancelled` but the worker dropped it; new `JobSignals.preserved(object)` emits right after `cancelled` whenever a payload rode the exception (the historical `cancelled(float)` contract untouched — all 15 pages byte-identical, pinned by a dedicated back-compat test); `JobRunner.submit(on_preserved=…)` follows the round-208 connect-before-pool pattern; first consumer: the chemistry feature-factory page — the R6 partial `(frame, report)` binds to the table with the honest "kept the N completed molecule(s) of M requested" line; 5 new offscreen tests (3 worker + 2 page); ruff clean, mypy clean on the touched files (7 pre-existing transitive errors proven identical via stash), full UI battery green |

Ordering rationale: the schema precedes the previews and the extension (the
master prompt demands the schema be defined and tested **before** extending
resumability, and the run summaries two §11 cards consume are state-serialization
adjacent). §14's report is generated last so it can honestly include R2–R4.

## R2 — Checkpoint schema v1 (§13, first half)

**Existing reference (do not rebuild):** `src/pharmaml/utils/checkpoints.py`
(Part-3 §20, round 229): atomic saves, canonical-JSON SHA-256 integrity,
rotation, quarantine-on-corruption; consumers in `ml/training.py` and
`ml/numpy_neural.py`; `SCHEMA = "pharmaml-checkpoint/1"`.

**Gap:** the record carries `step/total/state/note` but no formal envelope for
*resumable-operation identity*: operation type, toolkit version,
implementation fingerprint, input fingerprints, partial-results policy,
resume semantics, or a versioned migration path. §13 requires exactly that
before resumability extends to new operation classes.

**Design (incremental, format-stays-readable):**

1. New module `src/pharmaml/utils/checkpoint_schema.py`:
   - `CHECKPOINT_SCHEMA_VERSION = 2` (envelope level; the storage layer's
     `"pharmaml-checkpoint/1"` string stays untouched).
   - `@dataclass(frozen=True) OperationSpec`: `operation_type` (registry
     key), `operation_id`, `unit_description`, `partial_results_policy`
     (`preserve`/`discard`), `restore_exact: Literal[True]`.
   - `OPERATION_REGISTRY`: the §13 candidate classes (descriptor_batch,
     similarity_campaign, hpo, generative_search, docking_batch,
     preprocessing) — each entry registered with its unit semantics and a
     pointer to the test that proves exact restore (initially: only classes
     with proofs may register — the two existing consumers' classes).
   - `input_fingerprint(paths) -> str`: SHA-256 over sorted
     `(name, sha256, size)` triples.
   - `implementation_fingerprint(module) -> str`: SHA-256 of the executing
     module's source file at save time.
   - `validate_checkpoint(path) -> CheckpointInfo`: full conformance check
     (envelope fields, registry membership, fingerprint recompute,
     integrity digest, `restore_exact` invariant, `resume_unit ==
     step` for unit-indexed operations); legacy records (no envelope)
     validate as `legacy=True` and are reported non-resumable, never
     silently upgraded.
2. `CheckpointManager.save()` gains optional `operation: OperationSpec` and
   `inputs: Sequence[Path]` parameters; when given, the envelope is written
   inside the existing `state` under a reserved key (`__envelope__`) so old
   readers ignore it and the integrity digest still covers everything.
3. `tools/checkpoint.py` gains `--validate <file>` reusing the validator.
4. Tests `tests/test_checkpoint_schema.py`: envelope round-trip; tamper
   detection; unknown operation type rejected; unregistered class rejected;
   `restore_exact=False` rejected at construction; input fingerprint
   recompute; legacy record flagged not upgraded; migration refusal on
   future-version markers.

## R3 — §11 nine-card live previews

Replace the module-count previews in `ui/nine_cards.py::live_preview` with
real evidence, preserving card order, the 18-page navigation, card identity,
reflow and theme. Data sources only: loaded files, saved runs, parsed
campaign outputs. Every card ships an honest empty state. No randomness
(`random`/`np.random` outside explicitly-seeded, labelled synthetic demos).

| Card | Real evidence | Empty state |
| --- | --- | --- |
| Datasets | loaded-dataset shape, missingness, target distribution (cached per artifact hash) + the existing molecule thumbnail | "No dataset loaded…" |
| Machine Learning | saved evaluation summaries (metric, CV vs held-out) from the runs directory | "No saved evaluations…" |
| Deep Learning | loss/validation curves read from actual saved run event files | "No saved runs…" |
| Generative AI | validity/uniqueness/diversity + real SA histogram from generated outputs | "No generative runs…" |
| Docking | campaign completion state + engine score summaries from parsed outputs | "No docking results…" (explicitly demanded) |
| PK/PD | saved simulation curves with units/parameters; examples allowed but labelled | "No PK/PD runs…" |
| MD | RMSD/RMSF timelines only from actually loaded trajectories | "No trajectory loaded…" |
| Visualization | last exported artifacts (thumbnail, format, size) + inspection state | "No plots exported…" |
| Introduction | unchanged (version string) | n/a |

Parsing stays in `QThreadPool` workers with queued delivery (existing
architecture). Tests: per-card empty-state strings; per-card seeded real-data
diagnostics; a source scan banning unseeded randomness in preview modules.

## R4 — §12 GUI/viz/export

Audit-first: produce `docs/GUI_GAP_AUDIT.md` (gap → source-confirmed? → fix
round → test) against the current Studio tree, then fix only confirmed gaps:
image pixel/channel inspection; nearest-point selection with per-sample
metadata; debounced dense-plot metadata; adaptive ticks/legend/label
robustness; export validation (reopen, format/dimensions/bit-depth/metadata;
JPEG lossy-conversion made explicit or rejected; high-bit TIFF preserved).
No font-size increases or label clipping as layout substitutes. Focused GUI
regression tests per closed gap (offscreen QApplication fixture).

## R5 — §14 release QA artifact

Extend `tools/audit_release.py` (do not fork it) with a `--profile-results`
input so the shipped report gains: exact passed/failed/skipped/blocked/
not-executed counts per profile (never merged), the gates table, runtime/
platform limitations, and backup/recovery status — producing both the JSON
artifact and the rendered SUMMARY.md from one source of truth. The v1.2.174
audit already covers manifest/integrity/engine-checksum/exclusions; this
round closes the §14 remaining bullet: one persistent, machine-readable
release report per ship.

## R6 — §13 extension: descriptor batches

With the schema proven (R2), wire `OperationSpec` into the descriptor-batch
path and demonstrate exact restore: a paused-then-resumed batch produces
byte-identical outputs to an uninterrupted run (seeded fixture), Stop
preserves completed partial results per the data contract, and the proof
test is referenced from the `OPERATION_REGISTRY` entry. Only then may the
remaining §13 classes (similarity, HPO, generative, docking, preprocessing)
follow, one round each.

**Done (round 6, v1.2.176) — and round 7 followed the same pattern for the
similarity campaign: `run_similarity_campaign(checkpoint_dir=...)` in
`pharmaml/chem/similarity.py`, registered as `similarity_campaign` with its
proof suite `tests/test_similarity_campaign_resumability.py` (11 tests,
led by the bit-exact resume proof against both the uninterrupted twin and
the one-shot `similarity_scan`). Round 8 followed for HPO:
`tune_model(checkpoint_dir=...)` in `pharmaml/ml/hpo.py`, registered as
`hpo` with its proof suite `tests/test_hpo_resumability.py` (12 tests) —
the round's lesson: the trial history alone is NOT the complete state
(the TPE sampler's RNG advances per trial), so the checkpoints carry
the RNG state too and a naive history-only replay is proven to
diverge. Round 9 followed for the generative search:
`GeneticOptimizer.optimize(checkpoint_dir=...)` in
`pharmaml/generative/genetic.py`, registered as `generative_search`
with its proof suite `tests/test_generative_search_resumability.py`
(10 tests) — the generation boundary's complete state (population,
produced book, history, RNG state, the three adaptive operator credit
books + trajectory) restores bit-exactly (`elapsed_s` aside), and the
cooperative stop attaches the partial `GenerationResult` in both the
checkpointed and the checkpoint-less mode. Two candidates remain:
docking_batch, preprocessing.**

## Definition of done (this continuation)

1. §11: all nine cards real-or-honest; randomness-ban test green.
2. §12: audit table complete; every fixed row has a regression test; every
   "not a gap" row cites source evidence.
3. §13: schema v1 + tests landed; ≥1 operation class demonstrates
   bit-identical pause/resume; Stop preserves partials per contract.
4. §14: next ship's report carries the full profile/gate/limitation fields.
5. Every round: focused tests run and reported honestly (this sandbox runs
   the pure-Python profiles; GUI-with-display and engine profiles are
   recorded as not-executed with cause, never as passed).
