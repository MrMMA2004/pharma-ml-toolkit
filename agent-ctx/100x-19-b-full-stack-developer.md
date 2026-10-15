# Task 100x-19-b — Biology playground tab + /api/biology/biomarkers route (frontend subagent)

Sibling task: 100x-19-a (Python package edits under `/home/z/my-project/pharma-ml-toolkit/` — untouched by me).

## What landed

- **NEW `src/app/api/biology/biomarkers/route.ts`** (468 LOC) — pure-TypeScript mirror of `examples/make_synthetic_cellline_data.py`: 24 canonical cell lines × 10 tissues, 8 compounds with baselines + tissue effects, planted EGFR biomarker driven from its own fixed seed (mulberry32(2024), lung N(9.0,0.8) / other N(5.5,0.9)), screen noise drawn in the Python loop order (cell-outer, compound-inner across all 8 compounds), AUC via the same sigmoid, expression matrix from seed+1 (GENE_001… noise + EGFR + KRT8 breast-hot). Statistics: Spearman = Pearson on midranks; two-sided p via normal approximation with tie-corrected variance `Var = (1/(n−1))(1−Tx)(1−Ty)`, `T = Σ(t³−t)/(n³−n)` (documented in-code, conservative at n≈24); Benjamini–Hochberg step-up FDR; zod schema (compound enum, genes 60–600 def 300, seed 0–2³¹−1 def 42, nCellLines 6–24 def 24). Returns top-15 rows, EGFR row, topHit + EGFR scatter payloads (24 points + rank-space trend mapped to value space), tissues, meta (plantedHit = "EGFR" iff rank #1 AND FDR<0.05).
- **NEW `src/components/explorer/playground/biology-playground.tsx`** (769 LOC) — controls (compound select w/ tissue-effect hints + EGFR-linked chip, genes slider, seed + dice, panel-size select, Discover + Reset), stats strip, recharts scatter (10 tissue-colored series, dashed rank trend, custom tooltip w/ cell line + tissue), ranked table (max-h-96 thin-scrollbar, sign-colored ρ + sensitizing/resisting badges, FDR check, EGFR tint), planted-biomarker emerald badge, top-hit↔EGFR chart toggle, empty/loading/error states per ml-playground conventions.
- **Wiring** — playground.tsx: "Cell Line Bio" tab after ML Training (sm:grid-cols-6); command-palette.tsx: SUB_TABS now all 6 sub-tabs (added biology + the previously-missing reconciler — the list had drifted to 4-of-5); overview.tsx: "Cell Line Biology" domain card (emerald/Microscope), heading "Six domains, one import", grid xl:cols-3; toolkit-types.ts: +88 LOC types.
- **Shared-file fixes found during verification** — `fmtNum` passed directly as a recharts `tickFormatter` receives the tick INDEX as `sig` → `toPrecision(0)` RangeError crashed the client (fixed at call site with `fmtTick` + hardened `fmtNum` to clamp sig into the 1–100 domain); radix Select trigger showed the concatenated rich item text (fixed with a short controlled label); 3 pre-existing `tsc` errors in store.ts fixed (2× repeated-cast narrowing via hoisted `csvMeta`; `setPkParam` was dropping `mlCsv` from the sessionStorage snapshot — a real persistence bug).

## Verified numbers (live dev server)

- erlotinib, 300 genes, seed 42, 24 lines: **EGFR rank #1, ρ = −0.9673, p = 3.51e-6, FDR = 1.05e-3, significant**; exactly 1 significant gene of 300; UI shows planted badge.
- cisplatin control: **EGFR rank #254, ρ = +0.059 (|ρ| < 0.5), p = 0.777, FDR = 0.914, NOT significant**; top hit GENE_166 (ρ −0.707, FDR 0.131, ns).
- gefitinib: rank #1, ρ −0.882, FDR 7.0e-3, significant. genes=600/seed=2024: rank #1, ρ −0.9836. genes=60/seed=7/n=12: EGFR rank #2, FDR 0.085 ns (honest small-panel power loss).
- Runtime 5–22 ms per request.

## Gates

- `bun run lint` exit 0 · `bunx tsc --noEmit`: **src/ = 0 errors** (3 pre-existing store.ts errors fixed this round; remaining errors are pre-existing in examples/websocket + skills/*).
- agent-browser E2E: tab in tablist + palette; Discover → EGFR #1 + significant check; scatter = 24 symbols, 10-tissue legend, 1 trend line; cisplatin via UI select → GENE_166 top, EGFR #254 ns; dice works; invalid seed → inline Alert, no request; Reset → defaults + empty state; console/page errors: none.
- Mobile 360×844: `document.scrollWidth == 360`; table scrolls inside its pane (horizontal in the inner overflow-x-auto container, vertical in max-h-96 thin-scrollbar pane); controls stack.
- Screenshots: `download/biology-r19-{light,dark,mobile,cisplatin}.png`. dev.log: POST /api/biology/biomarkers 200.

## Deliberately NOT touched (for the orchestrator)

- overview hero "Three scientific domains." tagline; registry chip (hardcoded 18); UPGRADE_ROWS "6 subpackages" + StatChip 900/908 counts; examples/ + skills/ tsc errors; biology panel state is transient (like docking/reconciler) — store persistence is a possible follow-up.
