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

## Vina campaign review in Studio

Under **Docking Workbench → Campaign summary → Review a Vina campaign…**,
the wizard selects a Vina config, a prepared ligand directory and an output
directory, then reviews the engine launch prefix, receptor and six search-box
values in Å. It rejects ambiguous duplicate config directives, missing or
empty inputs, overlapping input/output trees and a receptor in the ligand
tree. It reports the existing batch runner's recursive `.pdbqt` census and
file-presence-based skip count. This is a **read-only review**, not a launcher,
preparation wizard, engine runtime test or scientific validation; no files or
results are written. It does not confirm that the molecules are chemically
valid or that a discovered engine works. Recheck after changing inputs.

A nonempty output directory may contain unrelated or outdated results. The
batch skip check uses only pose/log presence, not input, config, engine or seed
provenance; pose/log promotion is not crash-atomic. Use a fresh output directory
and independently review results. Extra config directives and paths may be
sensitive when sharing the review. AutoGrid and Vina box conventions are not
implicitly converted; this wizard displays the Vina config's explicit sizes.

## Nested validation

```sh
pharmaml ml nested-ridge --data training.csv --features descriptor1 descriptor2 \
  --target activity --group compound_id --outer-folds 3 --inner-folds 3
```

Outer folds evaluate the procedure; each outer training fold runs an inner grid
search over ridge alpha values 0.1, 1, and 10, with preprocessing fitted inside
each fold. Output is per-fold and mean MAE. With `--group`, both inner and outer
folds keep groups disjoint; without it, shuffled row folds can leak related
compounds or patients. Supply enough groups for both fold levels. This does not
produce a final fitted model and does not address calibration, applicability
domain, classification, or every split strategy. Fold estimates may be
correlated; nested CV is not a substitute for an untouched external evaluation.

In Studio, **Model Analysis → Nested validation…** runs this regression-only
workflow in the shared background runner. Progress counts completed outer folds.
**Stop at next fold boundary** is cooperative: it cannot interrupt an active
fold, and cancelled partial results are not saved or resumable. The file-based
report records a hash checked around parsing and the environment; the hash is
not a race-free snapshot and does not prove independence or reproducibility.

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

A guided docking campaign **launcher/preparation** workflow, provenance
integration across other model runs, generalized safe resume,
broader classification validation/probability recalibration and applicability methods, and a
full reproducibility certification system are not delivered here. Existing docking engines and
campaign APIs, label-conditional conformal APIs, and proof-backed checkpoints
remain available independently; they are not equivalent to those additions.
