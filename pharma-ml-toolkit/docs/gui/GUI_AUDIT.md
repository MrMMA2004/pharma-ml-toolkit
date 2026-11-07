# GUI Audit — the Tkinter playground mapped for the Studio redesign

Directive #16 (GUI redesign, `upload/MASTER_PROMPT part2.pdf`, 16.2/16.4):
before redesigning the interface, inventory every existing screen and map
it to its backend. This is the increment-1 audit (round 206); it is kept
current as sections migrate.

## The existing application

`playground.py` (Tkinter + ttk, ~45k lines) is the current GUI: one
`ttk.Notebook` with thirteen top-level tabs, 129 `make_card` cards, the
`WORKFLOWS` registry (11 guided workflows / 58 steps), per-card
descriptions, per-tab sample data, CSV **and** Excel I/O (v1.2.116+), and
the v1.2.119 heavy-work safety kit (Stop button, memory watchdog,
pre-flight batch guards, ML n_jobs cap, cooperative `JobCancelled`
checkpoints).

## Tab inventory → backend map → Studio status

| # | Playground tab | Cards (approx. clusters) | Backend modules | Studio page (increment 1) |
|---|----------------|--------------------------|-----------------|---------------------------|
| 1 | Home | quick-jump chips, workflow guide | `WORKFLOWS` registry | **Dashboard** (live census, dataset index) |
| 2 | Chemistry | standardize, descriptors, fingerprints, feature factory, feature recalculation | `pharmaml.chem.*` | **Chemistry Workbench** (increment 2: standardize + multi-family feature factory with the batch guard) |
| 3 | PK / PD | dose response, exposure, pregnancy PK | `pharmaml.pkpd.*` | **PK/PD Workbench** (increment 3: exact two-compartment regimen simulator + trapezoid/linear-up-log-down NCA + the NONMEM event grammar) |
| 4 | Cell Lines | expression, screening | `pharmaml.biology.*` | **Biology Workbench** (increment 4: screen audit + sensitivity matrix with quartile calls + MAD gene ranking + biomarker discovery) |
| 5 | Docking | DOCKSTRING lab, affinity, campaigns | `pharmaml.docking.*` | planned |
| 6 | Data & ML | QSAR training, leaderboards, AD | `pharmaml.ml.*` | **Model Analysis** (registry fit + curves) |
| 7 | Unsupervised | clustering, reduction | `pharmaml.ml.unsupervised` | planned |
| 8 | Deep Learning | runboard, UMAP, curation, figures | `pharmaml.ml.*` (torch) | planned |
| 9 | Generative | RNN/RL optimization | `pharmaml.generative.*` | planned |
| 10 | Foundations | statistics, table prep | `pharmaml.utils.*`, `profiling` | planned |
| 11 | Best-of ML | curated model cards | `pharmaml.ml.*` | planned |
| 12 | Response AI | drug response | `pharmaml.ml.drugresponse` | planned |
| 13 | Utilities | CSV standardize, converters, IO | `pharmaml.utils.io`, `chem.csv_standardize` | **Data Explorer** (tables + distributions) |

Plus the cross-cutting concerns, all live in Studio increment 1:
theme/appearance (Settings page), navigation (sidebar + Ctrl+1..9),
background execution (JobWorker/QThreadPool with cooperative stop),
state persistence (QSettings: geometry/nav/theme).

## Preserved / redesigned / consolidated / replaced

- **Preserved (backend)**: every `pharmaml` module is reused as-is;
  Studio calls the same public API surface the playground cards call
  (`read_table`, `train_model`, `MODEL_REGISTRY`, …). No backend
  behaviour changes in this round.
- **Preserved (GUI)**: the Tkinter playground itself stays fully
  functional and shipped until every section above has a verified
  Studio page (directive 16.18 step 15).
- **Redesigned**: the presentation layer only — notebook tabs become
  sidebar navigation; card grids become dedicated pages; matplotlib
  report figures become live PyQtGraph panels.
- **Consolidated**: the Stop button + memory watchdog survive as the
  Studio's `JobWorker` cooperative-stop channel (same
  `JobCancelled` contract as v1.2.119).
- **Replaced**: nothing yet (increment 1; retirement happens only after
  each replacement is verified).

## Known issues found during the audit (16.4)

1. The Tkinter GUI blocks on every card action — long jobs freeze the
   whole window (the v1.2.119 Stop/watchdog kit mitigates the OOM
   failure mode, but the thread model remains single-threaded). The
   Studio's QThreadPool workers fix the root cause.
2. Tab order encodes history, not workflow frequency — the sidebar's
   seven live sections are ordered by usage (dashboard → data →
   chemistry → pkpd → biology → models → settings).
3. No theme or appearance control exists in Tkinter; Studio ships the
   full provider stack (Qt-Material primary, QDarkStyleSheet
   alternative, native fallback) with runtime switching.
