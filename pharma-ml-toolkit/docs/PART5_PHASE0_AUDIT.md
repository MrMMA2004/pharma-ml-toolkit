# Part 5 — phase 0 baseline and restore audit

This is an **incremental, partial assessment**, not a full release certification.
The requested v1.2.173 ZIP and v1.2.173-specific master prompt were not supplied
in the attachment for this task. The checkout is a Git monorepo rooted at
`<TOOLKIT-ROOT>`; the Python toolkit and Studio are in the inner
`pharma-ml-toolkit/` directory and the companion Next.js web source is at the
repository root. The branch began at main commit `732f5538b5c69aa3997a5686b4aaaf47a6fed767`;
its prior landing-page change was committed as `8c1ca49` before this work.
The actual source `VERSION`, Python package version and web version are
**1.2.172**. The root citation intentionally has no version.

The available checked-in **v1.2.172** ZIP is 8,883,247 bytes with 531 entries
(486 files), SHA-256
`c2676af8db050b8f31c40bd36216e3de7ba4b8216a7067e321c421e6c5eb9857`.
Its CRC check passed. It has 45 example-data files recursively (16 top-level,
29 nested), and its three Windows engine binaries match the recorded MD5s in
`engines/README.md`. They were **not executed**: this Linux x86-64 environment
has no Wine. Presence, PE headers and checksums do not prove runtime behavior.
The [machine-readable baseline report](PART5_BASELINE_V1.2.172.json) includes
all archive file sizes and SHA-256 hashes, engine hashes, exclusions, smoke-test
scope and outstanding profiles. It describes the old checked-in archive, **not**
a new archive containing this patch.

Before editing, the archive was copied to the workspace backup directory
`<BACKUP-DIR>/` and its SHA-256 matched the source. This is
not an off-site disaster-recovery backup. The original checked-in ZIP was not
modified or rebuilt. The prior branch work was already committed and pushed.

## Findings and completion matrix

Statuses refer only to the evidence in this phase. No unrun tests are counted as
passed. The PDF's requirements span several phases; this table is not a claim
that the entire prompt has been completed.

| Area | Status | Evidence / remaining work |
| --- | --- | --- |
| Bounded ZIP validation and exception rollback | **Partially implemented** | Root-level and historical prefixed archives are checked for unsafe or colliding paths, special files, size/ratio limits, CRC and required version carriers. Existing destinations are backed up and verified; caught install, mkdir, journal and verification failures roll back. Fault tests compare file bytes, modes and mtimes. Actual disk exhaustion, power loss, process kill and restart recovery remain untested. |
| Recovery information | **Partially implemented** | A sibling private recovery directory contains staged files, backup hashes and a journal; incomplete rollback retains it. Crash recovery is manual, not an automatic or crash-atomic transaction. Concurrent writers are unsupported. |
| Real archive smoke | **Completed and verified** | The available v1.2.172 ZIP was dry-run and restored into an isolated temporary directory; every restored member's SHA-256 matched the ZIP and an unrelated user file survived. This does not test the unavailable v1.2.173 archive. |
| Release ZIP QA report | **Partially implemented** | A read-only archive auditor now produces machine-readable manifests and a human-readable summary, with explicit not-executed statuses. The report covers the old v1.2.172 archive, not a new release; comprehensive documentation, profile and full release certification remain outstanding. |
| Standalone release portability and canonical builders | **Partially implemented** | Original data, index and notebook-stripping generators now ship in `tools/`; checkout-only CLI forwarders preserve their historical paths. An explicitly selected targeted standalone profile and isolated candidate ZIP regression passed; the full release suite remains untested. See the continuation evidence below and [standalone testing](STANDALONE_TESTING.md). |
| Dependency profiles and CI | **Partially implemented** | Named targeted profiles and a GUI test extra are documented; local core, chemistry and Meeko/gemmi preparation selections ran. A chemistry/preparation CI job and GUI extra installation have been added; broader profiles remain incomplete. The CI portability failure is traced to example-data byte drift on the GitHub runner; root cause and a safe cross-platform remedy remain under investigation. See continuation evidence below. |
| Estimator validation / scikit-learn compatibility | **Partially implemented** | The 1.8 classifier failure was reproduced and fixed with sklearn type/fitted-state APIs and an explicit legacy-custom fallback. Focused and adjacent local tests passed on 1.3.0, 1.3.2, 1.5.2 and 1.8.0; the new compatibility CI matrix passed on commit 89c3cde. Full-suite and all combinations in the declared open-ended range are not certified. |
| Engine discovery and actual docking | **Not implemented** | `docking/workflow.py` has engine lookup APIs; the current engine README and feature inventory make conflicting claims. Search-order and runtime behavior were not changed or fully tested. No binary was executed. |
| Documentation and scientific feature inventory | **Partially implemented** | The example count mismatch and current-Tkinter and bundled-engine claims were located; the broader reconciliation and feature inventory have not been completed. |
| Dashboard, visualization and resumable jobs | **Not implemented** | Existing Studio architecture is preserved. No GUI behavior or scientific output was changed or tested. |
| v1.2.173-specific archive and validation | **Blocked by an external dependency/source** | The requested v1.2.173 ZIP and its version-specific prompt were not provided in this task. Do not treat v1.2.172 observations as v1.2.173 results. |
| Standalone-only absence of web source and Git | **Partially implemented** | Two checkout-only checks are skipped only when `--standalone-release` is explicitly selected. The isolated targeted selection passed; broader tests and web-parity behavior are not certified. |

Source inspection also found that the prior restore implementation staged files
but then used sequential `copy2` without backups or rollback. A later failure
could leave a mixture of old and new files. `--restore --dry-run` could also
materialize a missing Git-tracked ZIP, contrary to its no-write description.
Both paths are covered by this phase's regression tests. Invalid or missing
archive metadata now causes a refusal, including in dry-run; older synthetic
unit-test fixtures were updated to represent valid project metadata.

## Environment and commands

Linux x86-64, Python 3.11.2, Git 2.39.5; isolated environment: NumPy 1.26.4,
pandas 2.2.3, scikit-learn 1.5.2, SciPy 1.14.1, openpyxl 3.1.5, pytest 9.2.0,
Ruff 0.17.1. RDKit, Meeko, gemmi, PySide6, pyqtgraph and pytest-qt were not
installed in that environment. Bun and Wine were not available on PATH.

Run from the inner toolkit directory with the pinned core requirements and
pytest installed:

```bash
python -m pytest tests/test_checkpoint_restore_transaction.py -o addopts="-q --tb=short"
python -m pytest tests/test_checkpoint_tooling.py -o addopts="-q --tb=short"
python -m pytest tests/test_checkpoints.py tests/test_docs_index.py tests/test_conformal.py -o addopts="-q --tb=short"
ruff check tools/checkpoint.py tests/test_checkpoint_restore_transaction.py tests/test_checkpoint_tooling.py
```

Observed locally: **45 passed, 0 failed, 0 skipped** in the restore regression
module; **22 passed, 0 failed, 0 skipped** in adjacent checkpoint tooling;
**83 passed, 0 failed, 0 skipped** across checkpoint utilities, docs index and
conformal regression on scikit-learn 1.5.2; the root GitHub-page validator
passed and its unittest module had **10 passed, 0 failed, 0 skipped**; Ruff
passed. Full backend, optional chemistry/docking, GUI, notebooks, web parity,
standalone ZIP tests, release build and full suite: **not executed**. The
isolated actual-archive restore/byte comparison is a smoke check, not a full
standalone ZIP test profile. See [RESUME](RESUME.md#release-restore-safety-part-5-phase-0)
for constraints and manual recovery guidance.

## Continuation — standalone portability (targeted, not release certification)

The baseline v1.2.172 ZIP was restored into a separate temporary directory. With
RDKit 2026.9.1, the original provenance and checkpoint selection produced **27
passed, 5 failed, 0 skipped**: three failures depended on the excluded data
builder, one on excluded scratch forwarders, and one on Git origin metadata.
The data builder itself passed `--check` in the checkout with RDKit 2026.9.1;
RDKit 2024.9.6 instead reported drift in two outputs. No example data was
changed to force a match.

The existing toolkit-local generators were moved to `tools/`, with checkout
command-line shims at their former paths. The separately located top-level
historical index script is not the original toolkit generator and was not used.
Notebook source references were updated in both full and no-output variants,
without regenerating or fabricating execution outputs. The published notebook
execution script does not regenerate twins itself; documentation now gives the
separate stripping command. The standalone option skips exactly the two marked
checkout-only tests; the default still executes them.

A **temporary candidate**, not a published release, was made using the release
builder's file-selection rules without running its publication/deletion routine.
Its ZIP had 513 files and passed CRC. In its isolated extraction, the package
import resolved to the extracted `src/` tree. The provenance, checkpoint, index,
and notebook-variant selection reported **179 passed, 0 failed, 2 skipped**;
the skipped tests are the checkout-only forwarders and origin checks. The data
builder `--check` reported 44 files byte-identical, 46 owned, no orphans and
passing verify pins (the synthetic inventory intentionally includes invalid
SMILES and failed docking examples). The same selection in the Git checkout
reported **181 passed, 0 failed, 0 skipped**. The new portability regression
reported **2 passed, 0 failed, 0 skipped**, including an isolated candidate and
proof that those two checks fail rather than silently skip without the profile.
These results use Python 3.11.2, scikit-learn 1.5.2, RDKit 2026.9.1 and nbformat.
The pre-existing archive and its backup were not overwritten. This is not the
requested v1.2.173 archive, the full standalone suite, a real engine test, or a
release build. The CI addition for this regression is pending its next run.
A combined checkout run including the 45 restore fault tests and the new
portability regression reported **228 passed, 0 failed, 0 skipped**. The old
scratch data-builder command also passed `--check`; both the original archive
and its backup still have SHA-256
`c2676af8db050b8f31c40bd36216e3de7ba4b8216a7067e321c421e6c5eb9857`.

## Continuation — test profiles and dependency evidence

Named targeted commands and requirements are in [TEST_PROFILES.md](TEST_PROFILES.md).
The `gui-test` extra declares the PySide6, PyQtGraph, theme and pytest-qt
requirements; CI installs this extra with explicit pins. An additional CI job
runs descriptor and receptor/ligand preparation tests with RDKit, Mordred,
Meeko and gemmi; that job passed on commit 7fd4a09.

Local Python 3.11.2 results:

- Core-only environment (pinned core and pytest 9.0.2; no RDKit, Mordred,
  Meeko, gemmi or GUI): **104 passed, 0 failed, 0 skipped** in data-loading,
  PK and conformal selections.
- RDKit 2026.9.1 + Mordred 1.2.0 descriptor selection: **73 passed, 0 failed,
  0 skipped** (329 deprecation warnings).
- RDKit 2026.9.1 + Meeko 0.8.0 + gemmi 0.7.5 preparation selection: **84
  passed, 0 failed, 0 skipped** (5 warnings). This prepares inputs; it is not
  execution of Vina, AutoGrid or AutoDock.
- Local GUI selection: **blocked before collection**, since native
  `libGL.so.1` is absent even after the Python GUI extra installed; no GUI
  tests were claimed as executed.
- An attempted web-parity selection without optional dependencies produced
  **49 passed, 5 failed, 27 skipped** (the five bridge failures lacked RDKit).
  With RDKit and matplotlib it produced **50 passed, 4 failed, 27 skipped**:
  four rendering bridge tests could not import the system `libXrender.so.1`;
  Bun-dependent tests were skipped. Web parity is not certified.
- Checkout portability/provenance/index/notebook selection after the profile
  changes: **184 passed, 0 failed, 0 skipped**. Metadata validation now runs
  without Git as well as the separate checkout-only origin comparison.

On the previously pushed commit 45d3480, six GitHub CI jobs passed and the
Generative SA job failed in the newly added portability step. GitHub's job
log download host was inaccessible in this sandbox; the exact failing case was
not available from the API. Reproduction with the same pinned Python packages
and an archived clean source tree passed locally. The workflow now emits test
failure details as GitHub annotations to support diagnosis on its next run;
this failure is not considered resolved merely because local reproduction passed.
The v1.2.173 inputs, full release profile and real docking engine execution
remain unavailable/unverified.

### Follow-up — broader standalone selection and CI failure diagnosis

A broader extracted-candidate run including web-parity and version-carrier
tests initially reported **184 passed, 15 failed, 64 skipped**: 15 tests
assumed the excluded Web Console source was present. These tests now carry
an explicit `web_checkout_only` marker and skip only under the selected
standalone profile, with a specific reason. Default full-checkout behavior is
unchanged; the temporary-archive regression verifies a representative missing
web file still fails without the flag. Re-running that broader selection in a
fresh extracted candidate yielded **184 passed, 0 failed, 79 skipped**. The
79 include absent web/Bun tests and the two checkout-only guards; these are not
passes. This is still not the full release suite.

The CI failure on commits 45d3480 and 7fd4a09 was diagnosed using the added
JUnit annotations: the isolated data builder reports byte drift in
`qsar_multiclass.csv` and `pk_profile.csv` on GitHub's runner, causing three
provenance assertions to fail within the portability regression. The same
check passes in the local pinned environment. As an additional local
experiment, disabling NumPy CPU features (`AVX512F,AVX2,FMA3`) caused
`pk_profile.csv` byte drift; this suggests platform-dependent numerical output
may be involved but does not establish the precise CI root cause. The
byte-identity guard was **not** relaxed and generated data was not replaced.
On commit 7fd4a09, the other seven jobs, including Chemistry and docking
preparation and Studio MD, passed; the Generative SA job remains red. Real
docking engines were not executed.

## Continuation — reusable read-only archive audit

`tools/audit_release.py` produces a machine-readable report and human-readable
summary without publishing or overwriting an existing output directory. It
checks bounded path/collision validation, CRC, toolkit version carriers,
exclusions, sorted file sizes/SHA-256, engine MD5 references, recursive data
counts, archive stability and an isolated restore with per-file hash comparison
and an unrelated sentinel. Its own report explicitly marks test profiles,
real engines and backup verification as not executed. The engine references
come from the archive itself, not an independent authenticity source. Details
and limitations are in [RELEASE_AUDIT.md](RELEASE_AUDIT.md).

The original v1.2.172 archive was audited to the separate workspace directory
`<AUDIT-DIR>/`: 531 entries, 486 files,
8,883,247 bytes, SHA-256
`c2676af8db050b8f31c40bd36216e3de7ba4b8216a7067e321c421e6c5eb9857`;
all eight archive checks passed and 45 example-data files were counted
recursively (16 top-level). This command did not rebuild or overwrite the
archive or independently reverify the earlier backup. It did not run engines
or test profiles. A separate temporary candidate also contained the audit tool,
which successfully audited the old archive from the extracted standalone tree.

Focused audit, restore-fault and checkpoint-tooling tests reported **74 passed,
0 failed, 0 skipped**; Ruff passed. On commit b092837, all eight GitHub jobs
passed, including the previously failing Generative SA job. The preceding
runner reported data-builder byte drift, so this green rerun does not establish
cross-platform reproducibility or explain the earlier failure. No tolerance or
data was changed to hide it. Full release certification, v1.2.173 inputs and
real engine execution remain outstanding.

A further read-only numerical experiment reproduced the **same two filenames**
from the failed runner without changing any data: with
`OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 NPY_DISABLE_CPU_FEATURES=AVX512F`,
the builder reported drift in `qsar_multiclass.csv` and `pk_profile.csv`. Comparing
the in-memory frames against round-trip CSV parsing showed eight changed numeric
values in the former (largest absolute difference `4.440892098500626e-16`) and
12 in the latter (largest `8.881784197001252e-16`); categorical labels and
shapes were unchanged in this local experiment. This demonstrates a reproducible
CPU-dispatch/BLAS sensitivity locally; the failed CI runner's exact runtime was
not captured, so it does not establish that the same cause applied there. A
future portability failure now annotates the NumPy runtime to aid diagnosis.
No tolerance was substituted for the byte-identity check, and no generated
scientific data or model semantics were changed. A portable byte-identical
regeneration strategy across supported hardware remains unresolved.
The combined audit, isolated-portability, restore-fault, checkpoint, provenance,
index and notebook-variant selection subsequently reported **235 passed, 0
failed, 0 skipped**. A separate SHA-256 check again matched the original ZIP and
its existing local backup; this remains a local copy, not an off-site backup.

## Continuation — estimator validation and compatibility (targeted)

On Python 3.11 with scikit-learn 1.8.0, the existing conformal regression
selection reproduced the reported failure: **16 passed, 1 failed, 0 skipped**,
because an unfitted classifier reached `predict` and leaked `NotFittedError`.
A new focused regression module initially exposed **6 passed, 16 failed, 0
skipped** under 1.8.0, including fitted classifiers accepted as regressors and
DALEX using hard class labels instead of binary probabilities.

The implementation now uses sklearn's classifier/regressor and fitted-state
APIs for supported estimator instances and Pipelines. It rejects recognized
classifiers and unknown/invalid sklearn types, translates `NotFittedError` to
`DataValidationError` with the cause, checks input/prediction row alignment,
and preserves unrelated computational exceptions. A narrowly documented
legacy `_estimator_type` fallback protects older lightweight custom predictors;
untagged predict-only objects remain supported but cannot be proven to be
regressors. Untagged `predict_proba` objects are rejected as ambiguous. DALEX
uses the second binary probability column for recognized classifiers and
requires an explicit prediction function for other classifier conventions.
No model is fit or refit by calibration, and the conformal rank and coverage
formula are unchanged. See [CONFORMAL_REGRESSION.md](CONFORMAL_REGRESSION.md).

The same five-module estimator/conformal/coverage selection reported **136
passed, 0 failed, 0 skipped** in each of four separate Python 3.11 environments
using scikit-learn 1.3.0, 1.3.2, 1.5.2 and 1.8.0. The pinned 1.5.2 core stack
was not changed. A broader pinned-core selection including splitting, evaluation
and documentation checks reported **290 passed, 0 failed, 0 skipped** (three
single-class metric warnings); the DALEX/conformal doctest selection on 1.8.0
reported **1 passed, 0 failed, 0 skipped**. Ruff passed for changed code and
mypy reported no issues in 213 source files. CI now has a separate Python 3.11
matrix for scikit-learn 1.3.0, 1.5.2 and 1.8.0; all three jobs passed on
commit 89c3cde, as did the eight other jobs in that run. These are selected
tests, not proof of the full open-ended dependency range or full release suite.
Cross-platform example-data byte reproducibility remains unresolved.

## Continuation — strict data-drift diagnosis

On the documentation-only commit ceb5b94, **10 CI jobs passed and the
Generative SA job failed** again at the strict standalone provenance check.
Its NumPy-runtime annotation identified OpenBLAS architecture `Zen`, no AVX512F,
NumPy 1.26.4 and Python 3.11.18. The failing filenames were again
`qsar_multiclass.csv` and `pk_profile.csv`. This closely matches the controlled
local CPU-dispatch experiment, but does not prove every mechanism affecting
those files. The strict byte-identity test remains active and still fails on
this runtime; no CSV, hash or numerical tolerance has been changed to pass it.

Two adjacent test fixtures were made independent of existing byte drift: the
documentation ownership test now checks the ownership map even if the separate
strict drift gate fails, and the planted-drift test compares before/after drift
against a deterministic fixed-text file. On the controlled Zen/no-AVX512
configuration the full provenance module reported **9 passed, 1 failed, 0
skipped**, with the sole failure still the actual byte-identical check. It does
not hide or accept the existing drift.

`tools/diagnose_data_drift.py` runs the original builder in `--check` mode and
reports the differing CSV cells, schema and NumPy runtime without writing data
or replacing the strict result with a tolerance. It returns the original
nonzero status on drift; its numeric summaries are descriptive, not evidence of
byte identity or scientific equivalence. In the controlled local Zen/no-AVX512
experiment it reported eight changed text cells (largest numeric difference
`4.440892098500626e-16`) in the multiclass CSV and 12 changed text cells
(largest `8.881784197001252e-16`) in the PK CSV, with no changed nonnumeric
cells and unchanged shapes/headers. All 46 example/PDB file hashes were checked
before and after and matched. A diagnostic JSON was saved separately at
`<DRIFT-REPORT>.json`.

The new diagnostic/fixture/portability/provenance selection reported **187
passed, 0 failed, 0 skipped** on the local AVX512 environment. In the core-only
environment the diagnostic, release-audit and restore-fault selection reported
**54 passed, 0 failed, 0 skipped**. Ruff passed. The CI failure handler now
emits the diagnostic if strict portability fails; it does not change the test
status. A cross-hardware byte-identical generation strategy remains unresolved,
and no release has been certified.


## Continuation — engine discovery and launch prerequisites

The backend now searches explicit binary paths, explicit engine folders, PATH,
then checksum- and PE-machine-validated source/ZIP bundled engines. PATH
precedes bundled engines to avoid replacing existing native installations.
Installed wheels can only use package-local engines when explicitly packaged;
the current wheel does not promise them. PE files require a supported x86 host
and Wine on non-Windows; finding Wine is not a runtime capability test. SMILES
preflight now checks those prerequisites, and relative engine and batch file
paths are anchored before changing the subprocess working directory. Engine
launch `OSError`s are reported as failed runs. Documentation and card text
distinguish bundled Windows files from actual Studio functionality and data.

The local chemistry environment ran engine discovery, campaign, docking,
preparation and feature-inventory tests: **278 passed, 0 failed, 0 skipped**
(5 warnings). A core-only selection ran **237 passed, 0 failed, 0 skipped**. A broader
chemistry/docking CI selection subsequently ran **442 passed, 0 failed, 0
skipped** (335 warnings).
An earlier core-only attempt included an unguarded RDKit pose test and stopped
during collection with one missing-RDKit error; no tests ran in that attempt.
Subprocess calls in the new discovery/launch tests are mocked; **actual engine
execution remains not executed / platform blocked** because this Linux host has
no Wine. Windows execution, Wine's 32-bit support, actual scores and the
full release suite remain unverified. The updated Studio card assertion was
not run locally because `libGL.so.1` is missing; it is included in the Studio
CI job. These are targeted results, not release certification. The existing baseline matrix above describes the original audit,
not this continuation.

CI run 43011012196 on commit 49912d6: **10 jobs passed, 1 failed**.
Chemistry/docking and Studio (including the revised card assertion) passed.
Generative SA failed its strict portability test on the same two CSV drifts
and also exposed a stale generated `docs/INDEX.md` after the docking docstring
changes. The index has been regenerated and its focused identity test passes
locally; strict cross-platform CSV drift remains unresolved. No data was changed.


## Continuation — controlled floating-point attribution and ZIP build safety

CI on 50bb25a reported **11 jobs passed, 0 failed**. This does not resolve
the intermittent strict data drift. In a local pinned-stack experiment, with
the checkout's 1.2.172 CSVs unchanged, forcing OpenBLAS `Zen` (or `Haswell`)
while leaving NumPy SIMD unchanged reproduced the 8 QSAR cell differences and
no PK differences. Disabling NumPy `AVX512F` while leaving BLAS unchanged
reproduced the 12 PK differences and no QSAR differences. Combining both
settings reproduced both sets. This isolates the observed configuration
dependence, not a proof of all underlying operations or cross-platform
equivalence. Scientific recipes, release data and the strict byte check remain
unchanged; forcing unsupported hardware, rounding output or copying old bytes
would not be a justified general fix.

A separate synthetic-fixture regression reproduced the checkpoint ZIP builder
overwriting an existing same-version ZIP, deleting an unrelated file in its
fixed staging directory, replacing a dangling symlink, and mishandling an
invalid version: **4 failed** before the fix. The individual build step now
uses a unique private staging directory, verifies ZIP CRC and atomically
creates the final name without replacement using a same-filesystem hard link.
Existing targets, concurrent target creation and unavailable hard links fail
closed; an integrity failure leaves no published ZIP. Focused plus adjacent
checkpoint, restore, release-audit and drift-diagnostic tests: **86 passed, 0
failed, 0 skipped**. A broader portability/provenance/index/notebook and
checkpoint selection: **247 passed, 0 failed, 0 skipped** on the local
pinned chemistry stack. Source-walk errors also fail closed rather than
silently producing an incomplete archive. The existing
v1.2.172 archive and its separate workspace backup have not been replaced.
The later `--commit` safe-copy and retention operations and the independent
`tools/build_zip.py` publication ritual remain unmodified and **not certified
safe**; see [RELEASE_AUDIT](RELEASE_AUDIT.md). This change is not a release
build or full release certification.


## Continuation — new v1.2.174 baseline and checkpoint safe copies

The subsequent branch began at main commit `e2f88d1`, with source version
**1.2.174**, not the earlier v1.2.172. The prior baseline observations above
are historical, not proof of this newer release. The available root download
and toolkit dist-archive copies of v1.2.174 both have SHA-256
`bf8dbfb58e2d0d07ecc79c3f36f28595d33d59890f439cb242e992c7525b54e7`.
A separate, newly created workspace backup was verified against the root copy;
it is not off-site and does not survive every workspace loss. A read-only
archive audit of the root copy reported **8 passed, 0 failed** for the
auditor's scoped checks, 569 entries / 524 files / 8,884,591 bytes, 45
recursive example-data files, matching recorded engine MD5s, and an isolated
restore with every member hash checked. Engine execution and test profiles
were not executed by that audit; it is not release certification. Its JSON
manifest and summary were written outside the repository.

Synthetic ship fixtures reproduced four failures in the old checkpoint ship
path: a same-version safe copy could be overwritten, a failed copy did not
stop retention/commit, and retention could delete the only or a different
version of a backup. The checkpoint ritual now refuses existing output names,
uses staged, SHA-256-verified no-clobber safe copies and stops before retention
and Git commit on copy failure. Retention keeps its existing count policy only
for archives with an independent matching configured backup; it skips unsafe
retirements. Mock Git and synthetic archives were used: no real ship, commit,
retention sweep or release build was run for this test. All created copies
remain on a partial failure; version carriers may already have changed and
recovery is manual. Filesystem/hard-link support, concurrent writers, power
loss and the separate `build_zip.py` publication ritual remain limitations.

Focused ship/build/restore/tooling/archive-audit tests in the current core
environment: **95 passed, 0 failed, 0 skipped**. A broader targeted selection
covering portability, provenance, checkpoint, documentation and notebook tests
with RDKit 2026.9.1: **261 passed, 0 failed, 0 skipped**. No production
release or full suite is certified by these results.

## v1.2.176 continuation baseline

The current branch started at consolidated main `f0f40a19f829354164761f84ed606743a5b43595`,
not the prior PR branch. Closed, unmerged PR #11 contained follow-up changes not
present in that baseline; the CV parameter and group-label fixes, four-home ZIP
safeguards, recovery inventory, drift diagnostics and Windows smoke were restored
selectively. Other new baseline changes (including checkpoint schema validation)
were retained. The source fixes are **not** included in the existing v1.2.176 ZIP.

Both baseline ZIP copies and a separate local backup have SHA-256
`da2ba527083cec7672f7799977df45deeaea82477cd63a2c5ee36db9afad63ed`.
The backup is on the same machine, not off-site. The read-only archive audit
reported eight scoped checks passed, 597 entries / 547 files; it did not execute
any test profile or engine and is not a release certification. No release was
republished. The historical Windows job from the closed PR executed Vina on toy
geometry; that result was not an execution of this new v1.2.176 archive.

On this branch the restored focused CV/release selection passed 190 tests;
provenance, notebook variants, docs-index and checkout release-portability
selection passed 161 tests, with pinned RDKit 2026.9.1. These are targeted
selections, not full profile certification. See [test profiles](TEST_PROFILES.md)
for requirements and [recovery inventory](INTERRUPTED_RELEASE_RECOVERY.md) for
manual interrupted-publication guidance.
