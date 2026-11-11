# The §11 nine-card evidence contract

**Status:** landed in `part5-continuation` round 3 (2026-11-12)
**Spec:** MASTER_PROMPT part5 §11 — "Improve previews only where
meaningful, using actual loaded files, persisted runs, or parsed campaign
outputs … Never use arbitrary random values, stale session state,
synthetic scores presented as real scores, or fabricated saved runs. If
no suitable data exist, show a precise empty state."
**Implementation:** `src/pharmaml/studio/` (runs + evidence) with writers
in seven Studio pages and the render path in `ui/nine_cards.py`
**Tests:** `tests/test_studio_evidence.py` (32) +
`tests/ui/test_nine_cards_evidence.py` (12) + the part-4 pins in
`tests/ui/test_nine_cards.py` (26, unchanged and green)

## What changed on the cards

The part-4 previews reported module counts ("14 modules under
pharmaml.md"). §11 replaces them with **real evidence or a precise empty
state** — there is no third branch:

| Card | Real source | Empty state begins |
| --- | --- | --- |
| Introduction | installed `pharmaml.__version__` (unchanged) | n/a — always real |
| Datasets | the **loaded** Data Explorer frame: shape, missing cells, SMILES-column detection, the plotted column's distribution | "No dataset loaded — …" |
| Machine Learning | the live `MODEL_REGISTRY` count + saved `ml-evaluation` records (CV vs held-out labelled) + the round-229 family-benchmark §20 checkpoint (reported as CV, never resumed) | "No saved evaluations yet — …" |
| Deep Learning | saved `dl-training` records: the actual `history_` loss/validation curves, early-stop epoch | "No saved runs yet — …" |
| Generative AI | saved `generative` records: SA batches (validity, unique canonical SMILES, mean ± std) and Pareto fronts | "No generative runs yet — …" |
| Docking | saved `docking` records: folder completion state, top ligand + consensus | "No docking results yet — …" |
| PK/PD | saved `pkpd` records: route, doses, Cmax/tmax, AUC **with units and the ka/cl/v2/q/v3/dose/tau parameters**, plus the curve points | "No PK/PD runs yet — …" |
| MD | **only** the study the MD page actually loaded: ligand count, first metric's first→last/mean | "No trajectory loaded — …" |
| Visualization | saved `viz-export` records: name, format, dimensions, bytes | "No plots exported yet — …" |

Every empty state names the action that fills it. The Machine Learning
card keeps its numeric-first headline (the part-4 pin): the registry
count is real data that always exists, and the precise empty state rides
the detail line when no evaluations are saved yet.

## The run records (`pharmaml.studio.runs`)

One small JSON file per finished Studio job, under
`~/.pharmaml/studio-runs/<family>/`:

* schema `"pharmaml-studio-run/1"`, UTC-microsecond filenames, atomic
  `os.replace` writes;
* **best-effort and never raises** — `record_run()` returns `None` on an
  unwritable home; a failed record can never fail the job that produced
  the results (mirroring the §20 checkpoint contract);
* readers skip unreadable / foreign-schema files — a corrupt record is
  reported by its absence, never guessed;
* families written today: `ml-evaluation`, `dl-training`, `generative`,
  `docking`, `pkpd`, `viz-export`.

Writers live in the pages' GUI-thread done-slots (one per family): the
ML benchmark board, the DL `_render_train` history, the generative SA
batch and Pareto renders, the docking campaign scan, the PK/PD regimen
(now carrying its parameters on `RegimenOutcome`), and both export paths
(Image Lab panels and MD figures).

## The render path (`ui/nine_cards.py`)

* `live_preview(card_id, context=None)` — the §11 entry point; `context`
  is the GUI-thread snapshot of live page state (loaded frame + source +
  plotted column, loaded MD study, benchmark checkpoint dir).
* `NineCardGrid` labels every card's preview by objectName
  `nineCardLivePreview`; `refresh_evidence()` re-reads the sources into
  the **same** labels (the §8 widget-reuse contract extended to evidence
  text). Collectors are deliberately cheap — in-memory frames and tiny
  JSON reads — so no worker round-trip is needed for the refresh; the
  heavy jobs that *create* the evidence stay in the pages'
  `QThreadPool` workers.
* `MainWindow._card_context()` feeds the grid; `DashboardPage.showEvent`
  re-reads on every visit.

## The honesty guarantees (each pinned by a test)

1. Empty states are precise, action-naming, and verbatim-pinned.
2. Every real number in a preview comes from a seeded record, a loaded
   frame, or a parsed study — the tests feed real data and assert the
   exact echoed numbers (e.g. "5-fold CV over 9 families — best r2 0.612
   (morgan)", "Cmax 8.423 mg/L at 2.1 h", "validity 96%").
3. CV and held-out evaluations are labelled distinctly.
4. MD evidence exists **only** while a study is actually loaded — no
   stale session state.
5. A source-scan test bans unseeded randomness (`import random`,
   `np.random`, `torch.rand*`, `default_rng`) across the whole preview
   surface: `pharmaml.studio.*` and `ui/nine_cards.py`.
6. A raising context provider degrades to the empty states, never to a
   crash or a fabricated line.

## Environment record for this round

Focused suites (this sandbox): 32 + 12 new tests green; the 26 part-4
nine-card pins green; the seven touched-page suites plus main-window and
pages-smoke green (215 passed total). Four seeded-number probes in the
Deep Learning / Generative suites (`test_umap_after_a_run_real_numbers`,
`test_ga_multi_objective_seeded_real_numbers`,
`test_pareto_front_real_numbers`,
`test_pareto_front_hover_carries_smiles`) fail **identically on the
pristine tree** under this sandbox's torch 2.15.0+cpu / numpy 2.1.3 —
pre-existing environment-version drift, recorded here as failed-with-
cause, not as regressions and not as passed.
