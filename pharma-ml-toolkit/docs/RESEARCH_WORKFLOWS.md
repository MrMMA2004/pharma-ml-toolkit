# Incremental research workflow additions

These are narrow tools, not a complete validation platform. CLI commands below
print JSON; the CLI does not write reports unless you choose to redirect output. Hash and checkpoint inspections assume files are stable while read.

## Data quality in Studio

In **Data Explorer**, load a table, optionally select a class, an exact ID key,
and a split column, then choose **Run data audit**. The result reports missing
cells, duplicate rows, extra duplicate keys, class counts and, if both key and
split are selected, cross-split exact-key overlap. Audits run in the shared
background job runner; they are not cancellable or resumable. A new load
invalidates the previous report. Split auditing refuses missing/blank keys and
split labels, and requires at least two partitions. The regular table reader
infers types, which may alter identifiers such as leading-zero IDs; inspect the
loaded values and use the existing `pharmaml data audit-splits` CLI for
text-preserving, separate-file exact-key audits.

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
and shift can invalidate comparisons. This regression workflow uses one fixed model. In Studio, open
**Model Analysis → Uncertainty…** and select the same three separate files;
list one numeric feature name per line. The Studio dialog runs in the shared
background runner and does not reuse the current trained model. This operation
is not cancellable or resumable. Editing any input clears the previous report.
Use **Save report…** to create a new JSON file (existing files are never
replaced), and **Compare saved reports…** to inspect input-hash, parameter,
environment and summary differences. A failed save can leave a partial new
file; inspect it before reuse. Reports can contain sensitive subgroup labels.

## Held-out classification uncertainty

Choose **Classification — logistic prediction sets** in the same Studio
dialog, or run:

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

Select **Include kNN distance and classification probability diagnostics**
in the uncertainty dialog, or add `--diagnostics` to either uncertainty CLI
command. These checks use the same previously held-out test set and do not
change the fitted model, conformal thresholds or prediction sets. They can be
expensive on large descriptor tables; they are off by default and the job is
not cancellable or resumable.

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

## Prepare docking inputs in Studio

Under **Docking Workbench → Campaign summary → Prepare docking inputs…**,
select a receptor (PDB, PDBQT or mmCIF), a CSV/TSV/Excel ligand table with a
SMILES column, and a **new** output directory whose parent already exists.
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
certified. Install `pharmaml[docking]` for preparation.

Preparation does not write a Vina config, choose a box, validate engine output,
or launch docking. After inspecting the prepared files, create a separate
Vina config referring to `receptor.pdbqt`, choose the box explicitly, and use
the review/launch workflow below with the prepared `ligands/` folder. Never
point the launch output at the preparation folder; launch requires a different
new directory.

## Vina campaign review and launch in Studio

Under **Docking Workbench → Campaign summary → Review / launch Vina campaign…**,
select an existing Vina config, prepared PDBQT ligand directory, and a **new,
non-existing** output directory (type the new path; its parent must exist).
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

Studio offers the same read-only inspection at **Docking Workbench → Campaign
summary → Inspect Vina campaign progress…**. It applies only to the
review-first Vina launcher described above. It reports the manifest's recorded
status, counts of recorded successful/failed and unrecorded ligands, and
whether recorded successful pose/log pairs still match their fingerprints. It
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
as successful docking. Both UI and CLI may disclose names and paths.

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

Studio offers the same workflow under **Model Analysis → Probability
recalibration…**. Specify the exact text of the positive class; the input reader
preserves literal label and identifier text. The new base model is a binary
logistic regression fitted with median imputation and scaling **only on train**.
A second L2-regularized logistic sigmoid (C=1) is fitted to the base model's
decision score **only on the separate probability-calibration file**. Both
training and probability-calibration data must contain both classes; test must
contain only those classes. A convergence warning aborts rather than silently
reporting an unconverged fit. This is a fixed procedure, not automatic model or
calibration-method selection.

The report compares uncalibrated and recalibrated test-set binary Brier score
(range 0–1) and log loss, plus fixed-bin one-vs-rest reliability summaries. The
existing multiclass Brier sum shown in the reliability section is twice the
binary Brier score. Class counts, ID-duplicate counts, settings, input hashes
and environment are recorded. Test labels are never used for fitting or
selection. No model or per-row predictions are saved. The Studio report can be
saved to a **new** JSON file; a failed save may leave a partial file.

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

In Studio, **Model Analysis → Nested validation…** offers both tasks in the
shared background runner. Progress counts completed outer folds. **Stop at next
fold boundary** is cooperative: it cannot interrupt an active fold, and
cancelled partial results are not saved or resumable. The file-based report
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

Studio **Utilities Workbench → Reproducibility diagnostics → Compare training
records…** compares two such JSON files in the background; it does not train a
model or require loading its artifact. Matching records do not establish
scientific validity or reproducibility. This integration covers the CLI `ml
train` path only, not the Studio training path or other model workflows.

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

In Studio, **Utilities Workbench → Reproducibility & checkpoint diagnostics…**
provides environment details, checkpoint inspection and the checkout's
`tools/diagnose_data_drift.py` strict generated-data check. The latter runs in
an isolated child process and requires the diagnostic script, example data and
its dependencies; installations without them report failure rather than a
fictitious clean result. Exit 1 is reported as **strict byte drift**, even when
numeric differences are tiny; no rounding, tolerance or data rewrite is applied.
The job cannot be stopped or resumed. The checks assume the files are not
concurrently modified and do not establish numerical or scientific equivalence.
Reports can contain sensitive paths and environment information.

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

## Still to build

Provenance integration across other model and Studio training runs, generalized
safe resume, broader docking preparation/chemistry review,
broader classification validation, multiclass recalibration and applicability methods, and a
full reproducibility certification system are not delivered here. Existing docking engines and
campaign APIs, label-conditional conformal APIs, and proof-backed checkpoints
remain available independently; they are not equivalent to those additions.
