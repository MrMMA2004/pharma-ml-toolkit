# §14.2 Feature Proposals — Round 5 (v1.2.209)

Status: **PROPOSAL — not yet implemented.** Per the directive: *"Do not
implement these blindly. Present a short justified feature proposal
before expanding scope."* This document is that presentation. Of the
seven §14.2 candidates, two are proposed for the next round; the other
five are evaluated and declined with reasons.

---

## Proposal A — Dataset-level provenance and versioning

- **Existing gap.** `pharmaml.studio.workspace` records *that* a dataset
  was loaded (path + sha-256 via `input_fingerprint`, capped at 24
  entries) but nothing about *which version of its content* a model was
  trained on. When a CSV is re-exported or corrected in place, every
  stored run and bundle silently refers to a file whose bytes have
  changed. There is no way to answer "what exactly did the model see?"
  after the fact — the round-3 honesty report identified this as the
  largest remaining reproducibility hole.
- **Scientific usefulness.** QSAR results are only interpretable against
  the exact labeled set: a corrected SMILES, a dropped duplicate, or a
  re-split changes every downstream number. Provenance versioning makes
  model claims auditable (which dataset version, which split, which
  feature columns) — the minimum standard for regulatory-adjacent ML
  and for the toolkit's own no-silent-numbers policy.
- **Backend requirements.** Extend `studio.workspace.WorkspaceEntry`
  with a `content_version` (monotone integer, bumped when the same
  logical path arrives with a changed fingerprint) and a
  `column_signature` (names + dtypes hash). `training_provenance.py`
  already writes run records — add the entry's version + signature to
  each new record. No new tables, no database: the same best-effort
  JSON registry.
- **GUI location.** Model Analysis dataset row (a version badge next to
  the loaded dataset) + the Job Center's run rows. No new workspace.
- **Dependencies.** None (stdlib hashing; `input_fingerprint` already
  exists).
- **Resource implications.** One sha-256 per load (already paid today)
  plus ~200 bytes per registry entry. The 24-entry cap stays.
- **GitHub source.** Native; no external repository fills this gap
  better than the existing workspace registry (DVC and lakeFS were
  considered and rejected: both drag a runtime dependency and a
  foreign data model into a desktop app that deliberately keeps files
  first-class — https://github.com/iterative/dvc,
  https://github.com/lakefs/lakefs).
- **Tests.** (1) same content re-loaded → same version; (2) changed
  content under the same path → version bumps and the old run records
  keep their original version; (3) the Model Analysis badge shows the
  version; (4) a run record written at version N still reads N after a
  bump.
- **Implementation status.** Proposed (this document). Estimated one
  round: registry extension + record wiring + badge + 4 tests.

## Proposal B — Reproducibility manifests

- **Existing gap.** A trained bundle (`ml.persistence`) plus the
  notebook export paths leave the user to reconstruct the environment:
  pharmaml version, dependency versions, random seeds, `n_jobs`, split
  strategy. The §13 guides now *show* the workflow, but nothing
  *captures* the workflow's exact configuration for later replay.
- **Scientific usefulness.** A manifest turns "I trained this in the
  Studio yesterday" into a checkable claim — the same standard as
  `pip freeze` + seeds + data fingerprint in one JSON sidecar. It is
  the enabling artifact for the toolkit's re-run story: without it,
  every reproduction is an act of memory.
- **Backend requirements.** One function `write_manifest(path, *,
  runs, datasets)` in `pharmaml.studio.reports` (the §17D report
  machinery already collects runs/datasets/artifacts — this exports
  the same registry slice as JSON). Fields: pharmaml + Python version,
  `importlib.metadata` versions of the pinned stack, per-run
  hyperparameters + seeds + `n_jobs` (now user-configurable per §5),
  the dataset entries' fingerprints and (with Proposal A) content
  versions, platform, and a timestamp. The manifest is *descriptive,
  not executable*: no auto-reinstall, no pip execution.
- **GUI location.** Persistence tab ("Write reproducibility manifest…")
  next to the model-card writer; the file lands beside the bundle.
- **Dependencies.** None (stdlib `importlib.metadata`, `platform`,
  `json`).
- **Resource implications.** One small JSON write per bundle save;
  `importlib.metadata.version()` calls are cached after first use.
- **GitHub source.** Native. (Precedent studied, not depended on:
  `reprozip` https://github.com/ViDA-NYU/reprozip captures OS-level
  traces — far heavier than a desktop app needs.)
- **Tests.** (1) manifest JSON parses and carries all field groups;
  (2) versions match the running interpreter's actual packages;
  (3) a run with manual `n_jobs=6` records `n_jobs: 6` (the §5 field
  now has an end-to-end audit trail); (4) round-trip: the manifest for
  a bundle re-opened later names the same dataset fingerprint.
- **Implementation status.** Proposed. Estimated one round; halves if
  Proposal A lands first (manifests want content versions).

## Evaluated and declined (with reasons)

- **More comprehensive prediction uncertainty** — already covered:
  `bootstrap_uncertainty`, conformal regression/classification, the
  calibration report, and the uncertainty dialog shipped in rounds
  1–3; a new layer would duplicate a validated one.
- **More effective model comparison** — the §19 CV report, leaderboards
  and §17C experiment comparison cover this; no demonstrated gap.
- **Improved class imbalance diagnostics** — `ml.subgroups` + the
  coverage-by-group audit address it; class-weight guidance exists in
  the model tooltips. Weak marginal value.
- **Better chemical-space visualization** — UMAP/t-SNE/manifold tabs and
  the Image Lab matrices already ship; the plot-coverage registry
  (round 3) lists what remains backend-only honestly.
- **Additional validated QSAR diagnostics** — no specific diagnostic
  named by any user surface; adding unspecified diagnostics is exactly
  the feature-count padding §14 forbids.

## Decision requested

Proceed with **A then B** in the next round (A first: manifests depend
on content versions). Both stay inside the studio modules' existing
contracts (best-effort, file-based, no new dependencies), and both
close gaps this audit round could demonstrate rather than suppose.
