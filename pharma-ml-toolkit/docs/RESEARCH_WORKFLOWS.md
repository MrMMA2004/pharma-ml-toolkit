# Incremental research workflow additions

These are narrow tools, not a complete validation platform. CLI commands below
print JSON; the CLI does not write reports unless you choose to redirect output. Hash and checkpoint inspections assume files are stable while read.

**Interface status:** The supported desktop is the React + TypeScript/Tauri 2
application in `frontend/`; the separate Next.js console is another product.
Any UI walkthroughs in this guide describe only the retired PySide6 implementation
(`legacy/studio.py` and `ui/`). They are archival and unsupported, are not part
of either current app, and do not imply feature parity. The Python API and CLI
workflows are documented independently; see [`frontend/README.md`](../frontend/README.md)
for current desktop scope.

## Data quality audit

The CLI command `pharmaml data audit-splits` is the current documented route for
auditing exact-key overlap across separate files. The UI walkthrough below is
archival and describes only the retired PySide6 implementation, not the current
React/Tauri Studio.

In the retired PySide6 interface's **Data Explorer**, load a table, optionally
select a class, an exact ID key, and a split column, then choose **Run data audit**. The result reports missing
cells, duplicate rows, extra duplicate keys, class counts and, if both key and
split are selected, cross-split exact-key overlap. Audits run in the shared
background job runner; they are not cancellable or resumable. A new load
invalidates the previous report. Split auditing refuses missing/blank keys and
split labels, and requires at least two partitions. The regular table reader
infers types, which may alter identifiers such as leading-zero IDs; inspect the
loaded values. For separate files, choose **Data Explorer → Audit separate
files…** or use `pharmaml data audit-splits`. Both read identifier columns as
literal text, preserving values such as `001` and `NA` instead of inferring
numeric or missing values. In the dialog, add at least two unique `NAME=PATH`
entries, enter one exact column name per line (multiple columns form a
composite key), and select a delimiter if automatic detection is inappropriate.
The worker reports per-partition rows, extra duplicates, overlap rows and
pairwise unique-key intersections. It cannot be stopped or resumed; closing
the dialog does not stop it. Changing inputs clears the prior result. The
optional JSON summary is created as a new file and omits raw shared keys and
input paths; column and partition names can still be sensitive. Failed saves
may leave a partial file. The reader does not snapshot files: keep them stable
while the audit runs. Excel cells stored as numbers cannot recover leading zeros
that were never stored as text.

No SMILES canonicalization or similarity detection is performed. A clean
report does not prove independent data or absence of other leakage. Select the
correct leakage unit (compound, patient, batch, etc.). Duplicate-key counts
include missing and blank values, which are separately counted. Group/class
names can be sensitive: review output before sharing it.

## Held-out regression uncertainty

```sh
pharmaml ml conformal-report --train train.csv --calibration cal.csv \
  --test test.csv --features descriptor1 descriptor2 --target activity \
  --key compound_id --group assay_group --alpha 0.1
```

This is a fixed ridge regression workflow. Median imputation, scaling, and the
model are fit **only** on train; calibration determines a split-conformal
radius; test measures coverage. Files must share the explicit numeric features,
target and exact ID columns; the optional group is taken from test. Exact
cross-partition ID overlap is refused and duplicate IDs within each partition
are reported. Numeric feature blanks are imputed, target blanks are refused.
Calibration must be large enough for a finite conformal quantile. Select
partitions before model selection, and do not use test data for tuning.

Exact disjointness is not independence. Marginal conformal coverage relies on
exchangeability; subgroup coverage is empirical, **not** a conditional coverage
guarantee. Wilson intervals assume independent evaluation observations and
common coverage within groups; small groups, correlations, subgroup selection
and shift can invalidate comparisons. This regression workflow uses one fixed
model. The retired PySide6 interface had a **Model Analysis → Uncertainty…**
dialog for these three files; that screen and its background-job controls are
archival only and are not part of current React/Tauri Studio. Its historical
workflow did not reuse the current trained model and did not support cancelling
or resuming the operation. Input edits cleared the report; saving created a new
JSON file, and comparison inspected input-hash, parameter, environment and
summary differences. A failed save could leave a partial file. Reports may
contain sensitive subgroup labels.

## Retired PySide6 Statistics-tab conformal and subgroup tools

The retired PySide6 **ML Workbench → Statistical tools** conformal buttons
reused the selected fitted model and its scored rows. They split those rows by
position, with a fixed random seed: approximately half calibrated and the rest
were evaluated. Regression coverage and classification set summaries therefore
did not evaluate the same rows used for calibration. Too few calibration rows
could prevent a finite regression radius; classification could fall back to
full sets. The split was not stratified, and rare classes could have little or
no calibration data. These historical tools could not verify that the source
model was fitted and selected independently of those rows, particularly for a
loaded bundle or a previously explored run. Exchangeability, independence and
prospective validity were not established. This retired PySide6 workflow is not
available in current React/Tauri Studio; use the separate-file CLI workflow above.

In that retired implementation, coverage-by-group used the evaluation rows
and alpha saved with the last regression conformal run, including mean interval
width in its CSV output. Subgroup metrics used indexed scored rows rather than a
reconstructed 25% split. The tools refused mismatched indexed features/targets
and unavailable or ambiguous row identity. Group labels came from the selected
dataset and were **not authenticated**: identical features and targets could
not prove original metadata or independence, especially for duplicate rows.
These historical tools could not establish conditional coverage or eliminate
leakage.

## Held-out classification uncertainty

The retired PySide6 uncertainty dialog offered **Classification — logistic
prediction sets**; that control is not part of current React/Tauri Studio. For
the independently documented CLI workflow, run:

```sh
pharmaml ml conformal-classification --train train.csv --calibration cal.csv \
  --test test.csv --features descriptor1 descriptor2 --target class_label \
  --key compound_id --group assay_group --alpha 0.1
```

This fixed logistic-regression classifier and its median imputer and scaler
are fit on training only. A separate calibration set produces marginal
split-conformal prediction sets, and the separate test set reports coverage,
set-size counts (including empty and multiple-label sets), per-class and
optional subgroup summaries. File-based labels are read as nonempty literal
strings; the direct DataFrame API also requires nonempty string labels. Train
must contain at least two classes; calibration and test labels must occur in
training. Exact ID overlap is refused. When the calibration sample is too small
for the requested finite quantile, the method uses the full class set and
reports `full_set_fallback`; this is conservative but uninformative.

Coverage is **marginal**, not class-conditional or subgroup-conditional, and
requires exchangeability and a complete fixed class space. Exact IDs do not
rule out related compounds or other leakage. Small or correlated groups and
adaptive group selection undermine Wilson interval interpretation. Do not use
test data for model or threshold selection. This workflow does not provide
probability recalibration, label-conditional conformal calibration,
classification nested CV, or scientific validation of the model.

## Optional held-out model diagnostics

The retired PySide6 uncertainty dialog had an **Include kNN distance and
classification probability diagnostics** checkbox; it is not a current
React/Tauri Studio control. In the current CLI workflow, add `--diagnostics` to
either uncertainty command. These checks use the same previously held-out test
set and do not change the fitted model, conformal thresholds or prediction
sets. They can be expensive on large descriptor tables and are off by default.

For classification, the report includes a multiclass Brier score defined as
mean **sum** of squared errors over classes per observation (range 0–2, including
binary), and one-vs-rest reliability tables for ten fixed probability bins.
Bins are left-inclusive and right-exclusive, except the final bin includes 1;
empty bins are omitted. These are empirical evaluation summaries, not fitted
probability calibration, and small or dependent test samples can mislead.

For either task, kNN distances use the fitted train-only median imputer and a
separate train-fitted scaler/index, with `k=5` and the 95th percentile of
leave-self-out training distances as the reference threshold. The report
counts test rows inside/outside and describes their empirical conformal coverage.
At least six training rows are required; otherwise the diagnostic is marked
unavailable rather than silently changing `k`. This is a descriptor-distance
heuristic, **not** evidence of molecular similarity or prediction reliability,
nor a coverage/safety guarantee. Treat slice counts and coverage as descriptive;
do not choose models or thresholds using the evaluation set. Dependence,
missing descriptors, feature scaling and high-dimensional distance effects can
change interpretation. Full individual predictions are not included.

## Prepare docking inputs: archived PySide6 interface

The retired PySide6 UI exposed **Docking Workbench → Campaign summary →
Prepare docking inputs…**. Its historical procedure was to select a receptor
(PDB, PDBQT or mmCIF), a CSV/TSV/Excel ligand table with a SMILES column, and a
**new** output directory whose parent already exists. This screen is not part of
the current React/Tauri Studio.
SMILES and name columns can be specified or auto-detected. Choose the optional
single-character PDB chain, water removal, non-water HETATM retention and
ligand embedding seed. Review lists source hashes, chosen columns, counts of
syntactically valid/invalid SMILES and required RDKit/Meeko (and for mmCIF,
gemmi) versions. Review is read-only; it does not establish receptor quality,
chemical suitability, correct charges or a docking box.

After separately confirming, the workflow rechecks source/dependency
fingerprints, creates the directory exclusively, snapshots both input files,
prepares `receptor.pdbqt` and one ligand PDBQT per numbered directory under
`ligands/`, and records a `preparation.json` audit with source choices,
receptor cleaning, per-ligand outcomes and output hashes. Invalid SMILES and
individual preparation failures are included; `completed_with_failures` means
some ligands were not prepared, not that the whole input succeeded. The
original table, failure CSV and audit can include sensitive structures, names,
paths and errors. Do not share them without review. A failed run or crash can
leave a partial folder or stale/partial manifest; there is no resume, stop or
crash-atomic guarantee. Closing the dialog does not stop a running job. The
checks assume trusted, quiescent files, not adversarial concurrent writers.

Receptor cleaning can alter scientific meaning: by default waters and
non-protein HETATM (including many cofactors/ions) are removed, first MODEL and
first alternate locations are retained, and PDB hydrogens are replaced through
Meeko. Existing PDBQT input is re-filtered rather than retyped; inspect charges
and chemistry. Bad residues are not automatically deleted. The ligand path
performs one seeded 3D embedding/optimization, not protonation, tautomer
selection, or exhaustive conformer sampling. Check structures and biological
context independently; reproducibility across versions or hardware is not
certified. From the toolkit package root, install the local extra with
`python -m pip install -e '.[docking]'` for preparation.

Preparation does not write a Vina config, choose a box, validate engine output,
or launch docking. After inspecting the prepared files, create a separate
Vina config referring to `receptor.pdbqt`, choose the box explicitly, and use
the review/launch workflow below with the prepared `ligands/` folder. Never
point the launch output at the preparation folder; launch requires a different
new directory.

## Vina campaign review and launch: archived PySide6 interface

The retired PySide6 UI exposed **Docking Workbench → Campaign summary →
Review / launch Vina campaign…**. Its historical procedure was to select an
existing Vina config, prepared PDBQT ligand directory, and a **new,
non-existing** output directory (type the new path; its parent must exist).
This screen is not part of the current React/Tauri Studio.
Review displays the receptor, explicit Vina box in Å, engine launch prefix,
input fingerprints and output census. Review itself is read-only and does not
prove the engine works or the files are chemically valid. Unsupported config
directives, unavailable engines, missing inputs, overlapping trees or an
existing output directory prevent launch, while a read-only review may still
be shown. Launch supports `exhaustiveness`, `num_modes`, `cpu`, `seed`, and
`energy_range`; other directives must be removed or handled elsewhere.

After a successful review, **Launch reviewed campaign…** requires a separate
confirmation. It rechecks the plan, exclusively creates the new output directory,
then copies and verifies the config, receptor, and ligands against their reviewed
hashes. These snapshots remain in `inputs/`; unique numbered poses and logs are
in `results/`, and `campaign.json` records per-ligand status and output hashes.
The engine runs sequentially with the reviewed box and supported options. A
fixed seed is recommended; if omitted, the plan explicitly warns that engine
randomness is not controlled. Only execute a trusted engine with scientifically
prepared input. Inspect poses, logs and scores independently.

Stop is cooperative **between ligands**; it cannot interrupt a running engine
process (the per-ligand timeout is one hour). Closing the wizard does not stop
the job. No skip-existing or resume is offered. Failure, cancellation or a crash
can leave snapshots, partial outputs, stale `campaign.json` status, or temporary
files; pose/log promotion and the manifest are **not** one crash-atomic transaction
and are not power-loss durable. A new output directory is required for each run.
The checks assume a trusted, quiescent filesystem and cannot prevent adversarial
concurrent changes or engine side effects. Input snapshots, manifest, paths and
logs can be sensitive and may consume substantial disk space. AutoGrid-to-Vina
box conversions are not implicit. This workflow does not prepare PDBQT inputs,
validate chemistry or establish biological accuracy.

## Inspecting an interrupted Vina campaign

```sh
pharmaml docking campaign-status --dir /path/to/campaign
# Optional: also read and hash original input and engine paths in the manifest
pharmaml docking campaign-status --dir /path/to/campaign --check-originals
```

The retired PySide6 UI offered a read-only inspection at **Docking Workbench →
Campaign summary → Inspect Vina campaign progress…**. That screen applied only
to its review-first launcher and is not part of current React/Tauri Studio. The
CLI command above remains the current documented inspection path. The historical
screen reported the manifest's recorded status, counts of recorded
successful/failed and unrecorded ligands, and whether recorded successful
pose/log pairs still matched their fingerprints. It
also checks the copied source config, receptor and ligand snapshots, verifies
the generated config against the recorded plan (line endings normalized), and
lists unrecorded files/directories under `results/`. A recorded `running` status
may be stale after a crash; inspection cannot determine whether a process is
still active. A pose or log not recorded in the manifest is **not** promoted to
completed, even if both files exist. Missing/changed files are reported rather
than silently repaired, deleted or quarantined.

Optionally checking originals reads and hashes the **external paths named by
the manifest**, compares planned ligands and engine-prefix files, and reports
new/missing ligand names. Only use this option with a trusted manifest and
paths; engine resolution itself is not re-run. With the option off, originals
are not read. Symlinks inside campaign outputs are refused. The CLI returns
zero for a readable inspection even when it reports missing, changed, failed,
unfinished or unrecorded items; inspect the JSON rather than treating exit zero
as successful docking. The CLI and the retired UI could disclose names and paths.

Inspection and fingerprints assume a trusted, quiescent filesystem. Manifests
are unsigned and can be edited; a matching record does not prove the engine
ran, the original config and plan agreed, output validity, reproducibility or
safe recovery. The current launcher still offers **no resume**; keep or back up
partial results and, after investigation, make a new review/run in a different
new output directory. The inspector does not signal, stop or restart a process.
Generalized recovery remains unavailable.

## Binary probability recalibration (separate from conformal)

```sh
pharmaml ml calibrate-binary --train training.csv --calibration probability_cal.csv \
  --test test.csv --features descriptor1 descriptor2 --target activity \
  --key compound_id --positive-label active
```

The retired PySide6 UI offered this workflow under **Model Analysis →
Probability recalibration…**; it is not a current React/Tauri Studio control.
The CLI/API procedure is independent of that interface. Specify the exact text
of the positive class; the input reader preserves literal label and identifier
text. The new base model is a binary logistic regression fitted with median
imputation and scaling **only on train**. A second L2-regularized logistic
sigmoid (C=1) is fitted to the base model's decision score **only on the
separate probability-calibration file**. Both training and
probability-calibration data must contain both classes; test must contain only
those classes. A convergence warning aborts rather than silently reporting an
unconverged fit. This is a fixed procedure, not automatic model or
calibration-method selection.

The report compares uncalibrated and recalibrated test-set binary Brier score
(range 0–1) and log loss, plus fixed-bin one-vs-rest reliability summaries. The
existing multiclass Brier sum shown in the reliability section is twice the
binary Brier score. Class counts, ID-duplicate counts, settings, input hashes
and environment are recorded. Test labels are never used for fitting or
selection. No model or per-row predictions are saved. The retired PySide6
dialog could save a report to a **new** JSON file; a failed save could leave a
partial file. Its historical job could not be stopped or resumed.

This is **probability** calibration, not conformal calibration. For a
subsequent conformal procedure, use an additional independent conformal
calibration partition and a separate evaluation set; this report does not
connect its model to the conformal workflow. Exact ID-disjoint checks cannot
prove independence or detect related compounds. Small, dependent or shifted
samples make evaluation unstable; do not repeatedly tune on the test set.
Recalibration may worsen performance and cannot guarantee future calibration.
Input hashes assume stable files during reading; reports are unsigned and can
contain sensitive metadata. The job cannot be stopped or resumed.

## Nested validation

```sh
pharmaml ml nested-ridge --data training.csv --features descriptor1 descriptor2 \
  --target activity --group compound_id --outer-folds 3 --inner-folds 3
pharmaml ml nested-logistic --data classes.csv --features descriptor1 descriptor2 \
  --target class --group compound_id --outer-folds 3 --inner-folds 3
```

Outer folds evaluate the procedure; each outer training fold runs an inner grid
search. Ridge regression tunes alpha (0.1, 1, 10), and reports per-fold and mean
MAE. Logistic classification tunes C (0.1, 1, 10) with `balanced_accuracy`
(mean class recall), reporting per-fold and unweighted mean balanced accuracy,
per-fold accuracy and confusion matrices (rows true, columns predicted, listed
class order). Median imputation and scaling are fitted inside each training fold,
never on an evaluation fold. Logistic uses a fixed 2000-iteration limit and
fails rather than silently accepting a convergence warning.

Classification requires at least two nonempty string class labels. File reads
preserve literal text (for example `01` and `1` are different classes). It
prechecks that **every** class appears in every inner and outer training and
evaluation fold and that no training fold has an entirely missing feature; if
this is impossible with the selected data and fold counts it refuses the run.
Row splits use shuffled stratification. Group splits use deterministic GroupKFold
at both levels, which is **not stratified** and can fail even if another group
allocation might work. With `--group`, both workflows keep groups disjoint;
without it, row folds can leak related compounds or patients. Users are
responsible for selecting leakage-unit groups and appropriate features. Split
checks cannot prove independence. Fold scores can be correlated, small classes
make them unstable, and the unweighted fold mean does not weight by fold size.
Neither workflow produces a final fitted model, calibrates probabilities, or
substitutes for an untouched external evaluation.

The retired PySide6 UI offered both tasks under **Model Analysis → Nested
validation…** in its shared background runner. Progress counted completed outer
folds. **Stop at next fold boundary** was cooperative: it could not interrupt
an active fold, and cancelled partial results were not saved or resumable. This
screen is not part of the current React/Tauri Studio. The file-based report
records a hash checked around parsing and the environment; the hash is not a
race-free snapshot and does not prove independence or reproducibility.

## Observed `ml train` run records (opt-in)

```sh
pharmaml ml train --data training.csv --target activity --model rf --no-tune \
  --run-record run1.json
pharmaml research compare-training run1.json run2.json
```

`--run-record` is optional and requires a **new `.json` path in an existing
parent directory**. Before training it fingerprints the input file around the
actual dataset load and refuses an ordinary change during parsing. After a
successful train, evaluation, model save and any requested visual report, it
records the input hash, resolved target/task/features and dropped columns,
actual holdout stratification/fallback, seed, row counts and ordered split-index
hashes, fitted pipeline settings and selected parameters, tuning/CV details,
held-out metrics, saved-model byte hash and runtime environment. Unsupported or
deeply nested parameter values are explicitly tagged as omitted; such records
are incomplete. A second file hash after the run is marked `match`, `changed`
or `unavailable`; the latter two emit a warning. Split-index hashes identify
ordered positions in the **cleaned table after missing-target rows are dropped**,
not original source line numbers or molecular IDs.

The existing `ml train` model save remains unchanged: it writes under
`./artifacts/` and **can replace an existing model**. Only the optional JSON
record is new-file-only. The exclusive record write happens after the model
and any report have already been written: failures or a competing writer can
leave an overwritten model and no record, or a partial record. This is not a
transaction, snapshot, resume system or signed audit log; it assumes a trusted,
quiescent filesystem. Input hashes around reading catch ordinary changes, not
adversarial races or changes reverted between reads. Random row splits can
leak related compounds or patients; recorded hashes do not establish split
independence. Do not repeatedly tune against held-out metrics. The serialized
model's hash includes bundle metadata (such as training time) and can differ
even for equivalent fitted models. Records may expose sensitive settings,
column names and environment details. Untrusted joblib models must never be
loaded; the comparison command reads JSON only.

The retired PySide6 UI exposed **Utilities Workbench → Reproducibility
diagnostics → Compare training records…** to compare two such JSON files in
the background; it did not train a model or require loading its artifact. This
screen is not part of current React/Tauri Studio. Matching records do not
establish scientific validity or reproducibility. The integration covers the
CLI `ml train` path only, not other model workflows.

## Provenance and reproducibility diagnostics

```sh
pharmaml research manifest --data training.csv --split-strategy grouped \
  --parameter model=ridge --parameter seed=42 > run1.json
pharmaml research compare run1.json run2.json
pharmaml research environment
```

Manifests record a file SHA-256, size, user-declared parameters and split
strategy, and selected software/environment details. They are **not** generated
by model training and do not prove the recorded setup was used; they omit many
sources of numerical drift, dependencies, data transformations and randomness.
The comparison is descriptive and treats a matching record as no proof of
reproducibility.

The file-based uncertainty workflow also includes **automatically captured**
input hashes, used parameters and environment in its report, distinct from the
user-declared manifest. Save two CLI reports and compare them with
`pharmaml research compare-uncertainty first.json second.json`. The file reader
hashes before and after each read and refuses an observed change, but this is
not a race-free snapshot and assumes files are stable (a change reverted during
reading can go undetected). The reports are unsigned, omit other influences,
and matching metadata or metrics do not prove validity or reproducibility.
Other training workflows do not yet produce this provenance automatically. Do not include sensitive parameters in shared reports.
Existing `tools/diagnose_data_drift.py` performs a separate, read-only generated
CSV comparison; environment inspection does not modify any results or strict
byte comparison policy.

The retired PySide6 UI exposed **Utilities Workbench → Reproducibility &
checkpoint diagnostics… → Create declared manifest…**. That historical screen
fingerprinted one selected file and recorded user-entered split strategy and
optional `NAME=VALUE` parameters; it did not run or observe training. The
fingerprint assumed a stable file while reading. Its background job could not
be stopped or resumed, and closing the dialog did not stop it. Saving created a
new JSON file exclusively, though a failed save could leave a partial file.
**Compare declared manifests…** compared manifests created by the retired UI or
the CLI and reported dataset-hash, parameter, split-strategy and environment
differences. Neither tool verified what was trained or established
reproducibility. The same archived dialog provided environment details,
checkpoint inspection and the checkout's `tools/diagnose_data_drift.py` strict
generated-data check. The latter ran in an isolated child process and required
the diagnostic script, example data and its dependencies; installations
without them reported failure rather than a fictitious clean result. Exit 1 was
reported as **strict byte drift**, even when numeric differences were tiny; no
rounding, tolerance or data rewrite was applied. This UI is not part of the
current React/Tauri Studio. The checks assume files are not concurrently
modified and do not establish numerical or scientific equivalence. Reports may
contain sensitive paths and environment information.

## Checkpoint inspection

```sh
pharmaml research checkpoint path/to/checkpoint.json --input data.csv
```

This checks the existing v2 checkpoint envelope and recorded implementation
fingerprint. If inputs were recorded, provide the **complete** set of input
files using repeated `--input` options to compare fingerprints. A changed
input or implementation produces a warning and nonzero exit; omitted inputs
are explicitly marked unchecked. Legacy checkpoints and invalid envelopes are
not upgraded or resumed. A match alone does not guarantee safe recovery; this
command never resumes a job, validates no external side effects, and does not
extend resumability to additional operations. Existing proof-backed checkpoint
support and the separate release recovery tooling retain their documented
limitations.

## Genetic search: archived PySide6 interface

The retired PySide6 UI's **Generative Workbench → Evolve** workflow reported
progress at generation boundaries. **Stop** requested cooperative cancellation
at the next boundary and could not interrupt a generation in progress. A Stop
requested too late, including during the final generation, could finish normally
rather than cancel. When the backend returned a partial result, that interface
labelled it partial and displayed its leaderboard and completed history. The
partial was in memory only: the retired UI did not save a GA checkpoint, offer
resume, or extract a Pareto front from a partial run. Closing the page did not
stop a running job. RL, MCTS and the other generative operations remained
single-shot and did not gain this Stop behavior. This history does not describe
controls in the current React/Tauri Studio or separate Next.js console.

The lower-level genetic checkpoint mechanism is not documented as a React/Tauri
Studio resume feature. Its revised identity covers GA settings, the ordered canonical corpus
and seeds, supported objective parameters, package Python source, Python, RDKit
and platform metadata. Unsupported or modified objectives, older identities,
mismatches and malformed files fail rather than silently restarting. Use a new
checkpoint directory for a fresh run and preserve old or damaged files for
inspection. The checks are not authentication, do not cover every dependency
build or runtime alteration, and are not a cross-machine reproducibility guarantee;
never share a checkpoint directory between concurrent writers. See the
[checkpoint schema](CHECKPOINT_SCHEMA.md) for supported objectives and limits.

## Still to build

Provenance integration across other model-training workflows, generalized safe
resume, broader docking preparation/chemistry review,
broader classification validation, multiclass recalibration and applicability methods, and a
full reproducibility certification system are not delivered here. Existing docking engines and
campaign APIs, label-conditional conformal APIs, and proof-backed checkpoints
remain available independently; they are not equivalent to those additions.
