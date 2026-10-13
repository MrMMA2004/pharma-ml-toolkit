# Worklog — PharmaML Toolkit Upgrade & Codebase Explorer

---
Task ID: 1
Agent: Main orchestrator (Z.ai Code)
Task: Extract and analyze the uploaded RAR archive `کدهای ماشین لرنینگ.rar` (machine learning codes) and assess the current code quality.

Work Log:
- Extracted `/home/z/my-project/upload/کدهای ماشین لرنینگ.rar` (648 KB, 138 entries) with `unrar x` into `/home/z/my-project/extracted/` — 118 files recovered (1 entry failed with a seek error, appears to be a stray binary entry).
- Analyzed representative scripts: `Classification/RF.py`, `Convert IC50(nM) to pIC50.py`, `others/Docking/main.py`, `others/Docking/readerwithdetails.py`, `others/Docking/gpfer.py`, `others/Docking/smilestopdbqt.py`, `others/Docking/checker.py`, `Split CSV file.py`, `Tanimoto calculator.py`, `others/Ml/Descriptor calculator.py`.
- Environment recon: Python 3.12.15 (numpy 2.1.3, pandas 2.2.3, sklearn 1.5.2, scipy 1.14.1, pytest) — RDKit NOT installed. `unrar`, `zip`, `unzip` CLI available. Next.js dev server already running on port 3000 (HTTP 200).

Stage Summary:
- Original codebase is a computational-pharmacy / QSAR research toolkit (~30 near-duplicate model scripts for classification & regression: RF, SVM, XGBoost, LightGBM, CatBoost, AdaBoost, ANN, KNN, Stacking, Voting, Logistic, Ridge/Lasso/ElasticNet, Decision tree), plus:
  - Data utilities: IC50→pIC50 conversion, CSV↔SDF↔SMI converters, CSV splitter, non-numeric remover, Tanimoto similarity scan, RDKit descriptor calculator, ECFP/Morgan/Mordred calculators.
  - Docking: AutoDock4 batch orchestration (autogrid4/autodock4 via subprocess + ProcessPoolExecutor), GPF generator with atom-type extraction, DLG cluster-table parser, docking completeness checker, SMILES→PDBQT via Meeko with AD4 atom-type repair.
- Quality issues found: no package structure (flat scripts), massive duplication, hardcoded Windows file paths (`C:\Users\...`), no type hints, no docstrings, bare `except:`, print-based logging, `os.chdir` side effects, `shell=True` subprocess calls, no tests, no CLI, mixed Persian/English messages, no pharmaceutical PK/PD/dosage calculation modules (user explicitly requests these as a new domain).
- Upgrade plan: build `pharma-ml-toolkit/` — a typed, documented, tested Python 3.10+ package `pharmaml` with subpackages `chem`, `ml`, `pharma` (NEW: PK/PD/dosage), `docking`, `utils`, plus examples, pytest suite, pyproject, Dockerfile, CI; then a Next.js "Codebase Explorer" web app on `/` to browse/download the upgraded codebase.

---
Task ID: 2
Agent: Main orchestrator (Z.ai Code)
Task: Finalize the Python package deliverable (pytest path fix, README, CI) and verify the suite.

Work Log:
- Assessed `pharma-ml-toolkit/` state: ~6,600 lines across `src/pharmaml/` (chem/ml/pharma/docking/utils + cli/config/logger/exceptions), 10 test modules, 5 examples, Dockerfile, pyproject.toml, requirements.txt. All written in a prior session but never logged.
- Fixed pytest import path: added `pythonpath = ["src"]` to `[tool.pytest.ini_options]` in pyproject.toml (tests previously failed with ModuleNotFoundError unless PYTHONPATH was set manually).
- Wrote the missing `pharma-ml-toolkit/README.md` (referenced by pyproject but absent): overview, design assumptions, install, package layout, quickstart per domain, CLI reference, error-handling/logging, testing, Docker, design-choices & trade-offs.
- Added `.github/workflows/ci.yml`: matrix CI (Python 3.10/3.11/3.12) — install → ruff → mypy (non-blocking) → pytest with coverage; plus a full-deps job (chem+boosting extras).
- Re-ran the suite: **159 passed, 1 skipped** with zero env setup. Smoke-tested imports & math: CrCl(68y/70kg/1.1)=63.64 mL/min ✓, Mosteller BSA=1.845 m² ✓, pIC50(100 nM)=7.0 ✓, ligand efficiency=0.3 ✓.

Stage Summary:
- Python deliverable COMPLETE and verified: package imports cleanly, CLI exists (`pharmaml ml train|compare`, `chem pic50|descriptors`, `pharma pk|dose`, `docking scan|check`), model registry lists sklearn models with XGBoost/LightGBM/CatBoost behind lazy optional imports.
- Remaining work: the Next.js "Codebase Explorer" web app on `/` (currently default placeholder page.tsx), browser verification, cron setup.

---
Task ID: 5
Agent: full-stack-developer
Task: Build the PharmaML Toolkit Explorer web app on /

Work Log:
- Read worklog + audited the working tree: a prior (interrupted, unlogged) session had already written the full app — page.tsx (4-tab single page), layout.tsx (ThemeProvider + sonner Toaster + metadata), explorer components (overview, code-explorer, file-tree, code-block, playground/{playground, pharma-calcs, calc-ui, ml-playground, docking-parser}, download-panel, stat-chip), libs (store/toolkit-types/toolkit-fs/ml-math/docking-parse/docking-samples/format) and all 5 API routes. @types/react-syntax-highlighter already installed.
- Verified all 5 API routes end-to-end with curl: GET /api/toolkit/tree (48 files, 12 dirs, 7,097 LOC, 237,586 B; hidden .github included; 60s TTL cache visibly working: 114ms → 8ms); GET /api/toolkit/file (README 200; traversal `../` → 400; illegal ext → 400; missing → 404; .github/workflows/ci.yml viewable); GET /api/toolkit/download (streams zip -r via child_process, correct Content-Type/Disposition, 60 entries / 237,586 B matching tree stats); POST /api/ml/train (linear R²=0.9077±CV, ridge α, polynomial deg-2 with 2 coefs, logistic moons acc=0.853 with 3,600-point 60×60 boundary grid, k-fold CV mean±std, zod 400 on bad body); POST /api/docking/parse (Vina 9-pose table, DLG cluster table, CSV with 2 score columns → consensus z-score rescore + re-rank, garbage input → 400 with actionable message).
- Fixed a real bug in src/components/explorer/code-block.tsx: the grammar lookup `SYNTAX_MAP[language] === undefined ? SYNTAX_MAP[language] : language` always resolved to undefined (missing keys are undefined too), so NO file ever got Prism highlighting. Replaced with a REGISTERED_GRAMMARS whitelist; also registered the bash grammar so the download-panel install block highlights as Shell.
- Polish: fixed odd JSX indentation of the "best" badge in docking-parser.tsx; updated fallback hero/footer LOC stats from stale "6,993" to the live value "7,097" (48 files); confirmed registry stat uses the true count (15 models) from list_models() rather than the spec's "20+".
- bun run lint → zero errors. Curled / and every route to force recompiles; dev.log tail shows only 200/400/404s, no compile or runtime errors.

Stage Summary:
- Files: src/app/page.tsx, src/app/layout.tsx, src/app/api/{toolkit/tree,toolkit/file,toolkit/download,ml/train,docking/parse}/route.ts, src/components/explorer/{overview,code-explorer,file-tree,code-block,stat-chip,download-panel}.tsx, src/components/explorer/playground/{playground,pharma-calcs,calc-ui,ml-playground,docking-parser}.tsx, src/lib/{store,toolkit-types,toolkit-fs,ml-math,docking-parse,docking-samples,format}.ts, globals.css (emerald/amber palette, thin-scrollbar + pill-scroll rules, sticky footer via mt-auto + safe-area inset).
- Features: Overview (hero w/ live stats, 4 domain cards, before→after table, 5 design-choice cards from README, quickstart + copy); Code Explorer (collapsible tree w/ line counts, breadcrumb + badges + copy, Prism light/dark, line numbers, max-h-[70vh], README.md default, zustand-persisted selection, skeletons + retry); Playground (7 pharma calculators with validation/formulas/copy + recharts AUC & PD Hill charts; ML trainer with sliders/selects, metric cards, CV table, fitted-curve and moons+boundary charts, learned params — all math server-side & seeded; docking parser with 3 embedded samples, ranked table, LE, stat chips, affinity bar chart, consensus z-score column); Download (live file counts, install code block, blob download w/ loading + toasts).
- Status: lint clean (exit 0), dev.log clean, all API contracts verified. Known notes: docs/ dir in the package is empty (shown as placeholder); README says "152 tests" while the verified suite is 159 — the app consistently displays 159.

---
Task ID: 6
Agent: Main orchestrator (Z.ai Code)
Task: End-to-end browser verification of the PharmaML Explorer web app + fixes.

Work Log:
- Loaded agent-browser skill and drove the full golden path on http://localhost:3000:
  - Overview tab: hero, live stats, before→after upgrade table, design-choice cards, quickstart — all render.
  - Code Explorer: real file tree from /api/toolkit/tree; opened src/pharmaml/ml/models.py — 14,546 chars of Prism-highlighted code rendered; copy button present.
  - Playground → Calculators: verified math against hand calculations — CrCl(68y/70kg/1.1/male)=63.64 mL/min ✓, Mosteller BSA=1.845 m² ✓, pediatric 330 mg & 990 mg/day ✓, LD=540 mg ✓, MD=153.6 mg/τ ✓, ke=0.1155 h⁻¹ ✓, AUC trapezoid=32.65 µg·h/mL ✓ (hand-computed).
  - Playground → ML Training: regression (5-fold CV R²=0.912–0.941, mean 0.92 ± 0.0132 — std verified by hand) and classification (5-fold acc 0.79 ± 0.034, moons scatter + boundary dots + recharts SVGs confirmed).
  - Playground → Docking Parser: Vina sample → 9 poses with LE(−10.3/32)=−0.322 ✓, RMSD columns, bar chart; CSV sample → consensus z-score rescore (−2.546, −2.735, …) ✓.
  - Download: button shows ≈232.0 KB; curl of /api/toolkit/download → HTTP 200, 93,589 bytes, unzip lists 60 files / 237,586 bytes matching tree stats exactly.
- Checks: zero page errors, zero console errors/warnings; responsive at 360×800 (horizontal pill-scroll nav, overflow-x:auto); sticky-footer CSS confirmed (wrapper flex min-h-screen flex-col + footer mt-auto + safe-area inset); theme toggle renders correct dark palette.
- Visual QA with VLM (z-ai vision) on mobile + desktop-dark screenshots: desktop clean & professional; mobile revealed real defect — StatChip `truncate` clipped "232.0 KB"/"uncompressed"/"passing tests" at 360px.
- Fix 1: stat-chip.tsx — replaced truncate with break-words (+leading-snug on label); re-verified at 360px: 0 clipped elements.
- Fix 2: next.config.ts — added `allowedDevOrigins: ["*.space-z.ai"]` to silence the preview-origin dev warning; server auto-restarted cleanly.
- Fix 3: README.md test counts 152 → 159 (matches verified suite).
- bun run lint → exit 0; dev.log tail → only 200/400/404s, no compile/hydration/runtime errors.

Stage Summary:
- Web app fully browser-verified: every tab, every API, mobile + desktop, light + dark. All interactive flows work end-to-end with correct math.
- dev server healthy on port 3000.

---
Task ID: 7
Agent: Main orchestrator (Z.ai Code)
Task: Schedule the recurring web-dev review job and close out.

Work Log:
- Created cron job "PharmaML Explorer — 15-min dev review" (job_id 457247, kind=webDevReview, Quartz cron `0 */15 * * * ?`, tz Asia/Tehran, priority 5) with the standard QA/continue-development prompt (worklog review → agent-browser testing → fix-or-extend → worklog update).
- Closed the agent-browser session.

Stage Summary:
- Project COMPLETE and verified: (1) `pharma-ml-toolkit/` Python package — 48 files, ~7.1k LOC, 159 passing tests, README, CI, Dockerfile, CLI; (2) Next.js single-page Explorer app on `/` — overview, code browser, interactive playground (pharma calculators / ML training / docking parser), ZIP download; lint-clean, browser-verified.
- Known minor items: empty `docs/` directory in the package (rendered as a harmless placeholder in the explorer); PyTorch not used (sklearn backbone by design).

---
Task ID: R1 (cron review round 1)
Agent: Main orchestrator (Z.ai Code)
Task: Scheduled 15-min review: QA the explorer app, then extend both deliverables (fix + features + styling).

Work Log:
- QA pass (agent-browser): cycled all 4 tabs — zero page/console errors, dev server healthy (HTTP 200).
- BUG FOUND & FIXED — README/overview API drift: the README quickstart and the web app's QUICKSTART block used keyword names that don't exist (`age=`, `scr_mg_dl=`, `target_conc=`, `train_model(data=…)`, `report.metrics`, `ligand_efficiency(affinity=, heavy_atoms=)`, `consensus_rescore(dict)`) and stale registry names (`logistic`, `ada_boost`). Cross-checked every signature against the actual source and rewrote both snippets with the verified API; executed a mirror of every code cell as a script to prove correctness (all values verified: CV ROC-AUC 0.924 ± 0.031, acc 0.833, NCA AUC 63.7, CrCl 63.6, LD 540, LE 0.322…).
- NEW Python deliverable — `examples/pharmaml_walkthrough.ipynb`: built with nbformat and EXECUTED with nbclient (10 code cells, 10 outputs, 0 errors, kernel path = examples/); covers ML train/eval/leaderboard/persistence, NCA PK, dosing design, Vina parsing + LE, pIC50 conversion. Package installed editable (`python -m pip install -e . --no-deps` into the venv) to enable kernel import.
- NEW Python deliverable — `docs/INDEX.md`: auto-generated module index (126 lines, 7 modules) from each module's `__all__` + docstrings, via `scratch/build_docs_index.py`.
- NEW web feature — Code Explorer search: filter input with clear button, live match count, empty state, pruned tree with auto-expanded dirs (file-tree.tsx `searchActive` prop).
- NEW web feature — PK Simulator (4th Playground tab): exact 1-compartment superposition for repeated IV-bolus/oral dosing, 5–7 sliders, recharts concentration–time profile with dashed Css,avg reference, derived readouts (ke, t½, R, Css,avg, Cmax), flip-flop kinetics warning at ka ≤ 1.25·ke. Browser-verified: readouts ke=0.0711, t½=9.75 h, R=2.30×, Css,avg=9.77 mg/L, Cmax=5.56 mg/L (all hand-checked); warning toggles correctly via slider keyboard interaction.
- NEW support — `.ipynb`/`.json` viewable in the explorer (allowlist + JSON grammar registered in Prism).
- BUG FOUND & FIXED — egg-info pollution: the editable install created `src/pharmaml.egg-info/` which leaked into the tree and ZIP; excluded via suffix check in the tree walk and `*.egg-info*` patterns in the zip command. Verified: 0 PKG matches in tree, ZIP clean, notebook + INDEX.md included.
- Styling polish: hero redesigned as a framed panel with layered radial glow + faint blueprint grid (multi background-image — no z-index paint-order issues), mono uppercase eyebrow labels on 4 overview sections, data-table zebra + emerald hover on all three tables (best-pose highlight moved to inline style to win specificity), download panel docs note updated, playground tab grid now 2×2 on mobile (4 tabs).
- Verification: `bun run lint` exit 0; console/page errors zero after reload; 360px check: 0 clipped elements, playground tabs fit; VLM QA of new hero: "professional, technical blueprint aesthetic, no contrast issues"; dev.log tail: all 200s, only the transient import-typo error (fixed) from hot-reload.
- Cleaned scratch build artifacts (kept scratch/*.py generator scripts for regeneration).

Stage Summary:
- Fixed: 1 real doc bug (API drift in README + web quickstart), 1 pollution bug (egg-info in tree/ZIP), 1 transient import typo.
- Added: executed Jupyter walkthrough notebook, auto-generated docs/INDEX.md, file-tree search, PK Simulator tab, ipynb viewing, hero/eyebrow/table styling refresh.
- Python package now at 50 files (notebook + INDEX); all 159 tests still pass; web app fully re-verified.
- Known minor items / next-round candidates: (a) README "Design assumptions" could mention the notebook; (b) the notebooks' matplotlib-free charting means outputs are text/HTML tables only; (c) could add an "API reference" tab in the web app rendering docs/INDEX.md as a page instead of raw markdown; (d) example ML playground could add a regression-degree selector effect on the chart legend.

---
Task ID: R2 (cron review round 2)
Agent: Main orchestrator (Z.ai Code)
Task: Scheduled 15-min review: QA pass, then implement the R1 next-round candidate — an API Reference tab rendering docs/INDEX.md.

Work Log:
- QA pass: cycled all tabs incl. PK Simulator (charts+sliders render), zero console/page errors, Python suite re-run → 159 passed, 1 skipped. Phase stable → feature work.
- NEW API route `GET /api/toolkit/reference`: parses `docs/INDEX.md` into structured JSON (modules → symbols with kind/summary), resolves each module's source file via fs stat (`src/<dotted>.py` vs `src/<dotted>/__init__.py`), 60s TTL cache mirroring the tree route, helpful 500 if INDEX.md missing. Verified via curl: 7 modules / 62 symbols, source paths correct.
- NEW component `api-reference.tsx` + 5th top-level tab "API Reference" (store TabId + TABS + BookMarked icon + page.tsx render):
  - Header: eyebrow label, stat badges (7 modules · 62 public symbols · "N shown" while filtering), symbol search input with clear button, kind filter pills (All/Functions/Classes/Constants).
  - Module cards: mono name, blurb, symbol count, "View source" button → switches to Code Explorer with the module file selected + toast (browser-verified: clicking on pharmaml opened src/pharmaml/__init__.py).
  - Symbol rows: mono name (break-all), color-coded kind badges (function=emerald, class=amber, module=teal, constant=stone), docstring summary; row hover; per-module framer-motion entrance.
  - Loading skeletons, error state with retry, empty state with clear-filters; module-level response cache so tab switches never refetch.
- Integration: Overview quickstart section now has a second link "Browse the API reference" (verified via accessible-name click: lands on the reference tab).
- README (R1 leftover a): layout tree now lists examples notebook + docs/INDEX.md + .github/workflows; Quickstart ML section mentions the executed walkthrough notebook.
- BUG FIXED (mobile): module blurbs used `truncate` → clipped at 360px (same defect class as R1's stat chips); changed to break-words/leading-snug and symbol names to break-all → re-verified 0 clipped elements at 360px.
- Fixed lint error during development: react-hooks/set-state-in-effect in the fetch effect (restructured to async closure, status initialized from cache presence).
- Verification: `bun run lint` exit 0; fresh-load console completely clean (earlier Fast-Refresh warnings were transient mid-edit hot-reload artifacts — dev.log shows one mid-edit 500 that resolved; final state all 200s, reference route 5-20ms cached); search ("tanimoto" → 1 module/2 rows), kind filter (Classes-only → all badges 'class'), view-source navigation, overview link — all browser-verified; VLM QA of the tab: "clean and professional, no overlapping or misaligned elements, header well laid out".

Stage Summary:
- Added: API Reference tab (server-parsed INDEX.md → searchable/filterable symbol browser with view-source deep links into the Code Explorer) — completes the "index documentation" deliverable as an interactive surface.
- Fixed: mobile blurb clipping, one lint rule violation, transient mid-edit 500 (hot reload).
- Python tests: 159 passed, 1 skipped (unchanged). Web app: 5 tabs, lint-clean, browser-verified desktop + 360px.
- Next-round candidates: (a) deep-link from symbol rows to the exact definition line in Code Explorer (would need a line-anchor feature in the code viewer); (b) kind counts in filter pills (e.g. "Functions 48"); (c) export the API reference as JSON download; (d) ML playground: show dataset preview table button.

---
Task ID: R3 (cron review round 3)
Agent: Main orchestrator (Z.ai Code)
Task: Scheduled 15-min review: QA pass, then implement the R2 next-round candidates (symbol→definition deep links, kind counts, JSON export, dataset preview) + styling polish.

Work Log:
- QA pass: all 5 tabs cycled, ML regression re-trained (CV R²=0.92 ± 0.0132 matches known-good), zero page/console errors, dev.log clean → phase stable → feature work.
- BUG FOUND & FIXED — docs generator misclassification: `scratch/build_docs_index.py` used `inspect.isfunction()`, which returns False for `@lru_cache`-wrapped functions, so `pharmaml.pharma.molar_mass` was listed in docs/INDEX.md as a "constant" without a summary. Fixed with `inspect.isroutine(obj) or inspect.isfunction(inspect.unwrap(obj))`; regenerated INDEX.md (molar_mass now "function" + docstring; counts now 52 functions / 2 constants).
- NEW FEATURE — definition-line deep links (R2 candidate a): reference route now (1) parses each module's `from … import …` statements (single-line + multi-line parenthesized, absolute + relative) into an import map, (2) resolves every symbol to the file that actually defines it by following re-export chains through `__init__.py` files, (3) returns per-symbol 1-based `line` + `sourcePath` override. Verified 62/62 symbols resolve; spot-checked L94 `def ic50_to_pic50`, L146 `MODEL_REGISTRY`, L229 `def creatinine_clearance` (pk.py) — all exact.
- NEW UI — clickable symbol rows in API Reference: whole row is a button (aria-label "Jump to <name> definition at line N…"); jumps to Code Explorer, opens the correct submodule file, highlights the line (emerald inset bar + 13% tint + 1.8s flash, `prefers-reduced-motion` respected), smooth-scrolls it to the pane center, and shows a dismissible "line N" chip in the code header; `cl-N` per-line classes avoid TS data-attr typing issues in `lineProps`; hover arrow affordance + L-number chip per row.
- NEW FEATURE — kind counts in filter pills (R2 candidate b): "All 62 / Functions 52 / Classes 8 / Constants 2" (mono tabular-nums, opacity-60).
- NEW FEATURE — API reference JSON export (R2 candidate c): blob download button → pharmaml-api-reference.json + success toast (browser-verified toast).
- NEW FEATURE — ML dataset preview (R2 candidate d): new `POST /api/ml/dataset` route (zod-validated; same mulberry32/makeRegressionData/makeMoons generators as /api/ml/train, so preview ≡ training data) returning first 10 rows, per-column mean/σ/min/max, Pearson r (regression) or class balance (classification). Playground: "Preview data" button beside Train (icon-only "Data" <sm), collapsible card with data-table, colored label dots, r with qualitative strength ("very strong" ≥0.9 …), class-balance callout; resets on task switch. Cross-check: preview r=0.9614 → r²=0.9243 ≈ train R² 0.92 ✓; rows/stats match API byte-for-byte.
- NEW store API: `targetLine` + `openFileAtLine(path, line)`; manual file selection clears the highlight.
- BUG FOUND & FIXED — cross-tab scroll inheritance: deep-linking from the (tall) reference list kept the old page scroll, pushing the code header off-screen. Added scroll-to-top on tab change in page.tsx (standard SPA page semantics); re-verified chip visible at y=215 after deep-link from a scrolled list.
- DEV ISSUE RESOLVED — stale CSS: after editing globals.css, the dev pipeline served the old stylesheet (`.code-line` rules missing despite being in the file); a trivial re-touch of globals.css forced recompile and the rules appeared. No source change needed — watch-mode hiccup only.
- Styling polish: animated active-tab pill (framer-motion layoutId spring, tracks active button exactly — verified geometry), custom emerald `::selection` color, line-flash keyframes with reduced-motion fallback, symbol-row focus-visible styles.
- Verification: `bun run lint` exit 0; Python suite 159 passed, 1 skipped (regenerated docs don't affect it); browser: 62 jump buttons, deep-link end-to-end (light + dark), chip dismiss, collapse/expand preview, JSON export toast — all verified; mobile 360px: 0 clipped elements, filter pills (328px) and Train/Preview buttons fit, no horizontal scroll; VLM QA: deep-link highlight "clearly visible" both themes, API reference "correctly rendered according to specifications"; ZIP download HTTP 200 (102,174 bytes) and contains the corrected INDEX.md.

Stage Summary:
- Current status: STABLE & COMPLETE. Python package (159 tests green) + web app (5 tabs, 7 API routes) all verified end-to-end; zero console/page errors; lint clean.
- Completed this round: fixed docs-generator lru_cache misclassification bug; added symbol→definition-line deep linking with re-export resolution (62/62), kind-count filter pills, reference JSON export, ML dataset preview (server route + collapsible UI); styling: animated nav pill, line-flash highlight, selection color; fixed cross-tab scroll bug.
- Unresolved / next-round candidates: (a) Code Explorer: click a line number to copy a permalink-style "#L94" reference or deep-link INTO it (inverse direction); (b) preview could show a mini scatter of the full dataset (not just first 10 rows); (c) reference search could fuzzy-match (subsequence) instead of substring; (d) README "Design assumptions" could document the new /api/ml/dataset route parity guarantee; (e) minor: `docs/INDEX.md` regeneration is manual — could add a pytest that fails on drift between INDEX.md and live __all__.
