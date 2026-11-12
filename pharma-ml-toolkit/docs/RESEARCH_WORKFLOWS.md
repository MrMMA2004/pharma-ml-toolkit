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

A guided docking **preparation** workflow, provenance integration across other
model runs, generalized safe resume,
broader classification validation/probability recalibration and applicability methods, and a
full reproducibility certification system are not delivered here. Existing docking engines and
campaign APIs, label-conditional conformal APIs, and proof-backed checkpoints
remain available independently; they are not equivalent to those additions.
