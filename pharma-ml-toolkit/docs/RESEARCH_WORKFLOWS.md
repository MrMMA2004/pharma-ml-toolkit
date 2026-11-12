# Incremental research workflow additions

These are narrow tools, not a complete validation platform. Commands below
print JSON; the toolkit does not write reports unless you choose to redirect
output. Hash and checkpoint inspections assume files are stable while read.

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
and shift can invalidate comparisons. This command is regression-only, uses
one fixed model, and has no Studio uncertainty page yet.

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
reproducibility. Do not include sensitive parameters in shared reports.
Existing `tools/diagnose_data_drift.py` performs a separate, read-only generated
CSV comparison; environment inspection does not modify any results or strict
byte comparison policy.

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

The proposed guided docking campaign wizard, uncertainty GUI, integrated
experiment comparison/provenance for actual runs, generalized safe resume,
classification validation/calibration/applicability reports, and consolidated
reproducibility dashboard are not delivered here. Existing docking engines and
campaign APIs, conformal classification APIs, and proof-backed checkpoints
remain available independently; they are not equivalent to those additions.
