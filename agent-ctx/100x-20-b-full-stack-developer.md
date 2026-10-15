# Task 100x-20-b — PK/PD Explorer (PD effect layer + NCA observed-data panel)

Sibling task: 100x-20-a (Python PK/PD deep review — already landed; its Purves log-trapezoid
correction is documented in the pk-pd.ts header; the TS NCA path uses the LINEAR trapezoid
only, so it is unaffected).

## Situation on pickup

An interrupted first draft of this exact task was already in the working tree, uncommitted,
with NO worklog entry: pk-simulator.tsx had become a 3-inner-tab shell, pd-effect.tsx /
nca-panel.tsx / src/lib/pk-pd.ts existed, store.ts was wired, and scratch/probe_pkpd_20b.py
existed but had never been verified end-to-end. This round verified the math against the real
Python package, fixed 3 real defects found in QA, and ran the full gate suite.

## What landed

- **NEW `src/lib/pk-pd.ts`** (616 LOC) — shared math: one-compartment superposition
  concentrationAt/derivePk (moved out of pk-simulator.tsx so the PD panel reuses the exact
  curve), the four pharmaml.pharma.pd models (linear / Emax / sigmoid Emax / inhibitory Emax)
  with degenerate guards (negative C → 0, EC50/IC50 ≤ 0 → ε, h ≤ 0 → ε — never NaN), timeAbove
  (linear-interpolation threshold crossing integral), and the full NCA engine
  (computeNca: auto-sort + dedup + duplicate/trailing-zero/monotonic warnings, linear
  trapezoid AUC, log-linear terminal fit on the last N positive points after Tmax with r²,
  AUCextrap = Clast/ke, AUC∞, %extrap, CL = F·D/AUC∞, Vz = CL/ke, chart tail endpoint at
  0.01·Cmax). Header comment block pins every Python cross-validation number.
- **NEW `pd-effect.tsx`** (539 LOC) — model Select with live equation caption, per-model
  ParamSliders (slope / E0+Emax+EC50 / +Hill / E0+Imax+IC50), dual-axis ComposedChart
  (emerald conc + rose effect, right-hand axis), Conc/Both/Effect view toggle, EC50 + half-max
  + Css,avg reference lines, readout cards (E@Cmax, E@Css,avg, t > EC50 with interpolated
  crossings, EC50/Cmax ratio), amber notes for "EC50 ≫ Cmax" and near-saturation.
- **NEW `nca-panel.tsx`** (624 LOC) — 3 presets (clean oral / IV bolus / noisy oral), editable
  table (add/remove/edit, aria-labelled row inputs, invalid-row highlighting + destructive
  alert, 40-row cap, max-h-72 thin-scrollbar pane), dose + F NumberFields, terminal-N select
  (2–6), metric cards (AUC₀₋last, Cmax, Tmax, λz, t½, r², AUC₀₋∞, %extrap, CL, Vz),
  observations scatter + amber terminal points + dashed fitted tail extended to 1 % of Cmax,
  inline amber/destructive warnings.
- **CHANGED `pk-simulator.tsx`** (341 → 309) — regimen panel intact; shell with three inner
  tabs; pkInnerTab persisted.
- **CHANGED `store.ts`** (+~230) — pdParams / ncaState / pkInnerTab + field-by-field
  sessionStorage validators + a single persistPlayground() snapshot helper (the 19-b
  dropped-field bug class cannot recur).
- **CHANGED `calc-ui.tsx`** (+ParamSlider, +DedupeLegend shared atoms), **`ui/slider.tsx`**
  (forwards root aria-label to single-value thumbs — app-wide a11y win), **`playground.tsx`**
  (sub-tab "PK / PD"), **`command-palette.tsx`** ("PK / PD Explorer").

## Cross-validation (Python pharmaml vs TS vs live UI)

Probe: `scratch/probe_pkpd_20b.py` via `/home/z/.venv/bin/python`; TS via `bun -e` import of
pk-pd.ts; UI read from the rendered metric cards. Every number identical to all printed digits:

| preset | AUC | ke (N=3) | t½ | r² | AUC∞ | %ext | CL | Vz |
|---|---|---|---|---|---|---|---|---|
| oral | 62.746250 | 0.078523 | 8.827365 | 0.999747 | 64.274473 | 2.3777 | 3.500612 | 44.580978 |
| iv | 46.227500 | 0.248313 | 2.791426 | 0.999977 | 48.643806 | 4.9673 | 10.278801 | 41.394544 |
| noisy | 63.563750 | 0.098467 | 7.039380 | 0.983472 | 64.376204 | 1.2620 | 3.495080 | 35.494914 |

N=2 reproduces `nca_summary`'s last-2 secant exactly (0.076358/9.077650 · 0.250388/2.768296 ·
0.120577/5.748605). PD spot-checks identical (linear 8×{1.28,4.08,12.8} → 10.24/32.64/102.4;
emax → 41.2195/63.6842/79.1892; sigmoid h=1.5 → 37.0899/69.5590/85.3464; inhibitory →
6.0976/3.2895/1.3514).

## Defects fixed during QA

1. Mobile 360 px: "NCA (observed)" inner-tab label clipped in its grid cell → responsive label.
2. PD effect axis rendered raw float ticks ("78.8343"), top tick clipped 6 px past the chart
   edge → fmtTick on both PD Y-axes, width 44. Post-fix: zero clipped elements (DOM-verified).
3. Slider thumbs had no accessible names (aria-label on the radix root div is inert) →
   ui/slider.tsx forwards the label to single-value thumbs; range sliders keep Minimum/Maximum.

## Gates

lint exit 0 · src/ tsc 0 errors (4 pre-existing outside src/) · agent-browser E2E both features
+ persistence across reload + all warning cases · console/page errors 0 · mobile 360×844
scrollWidth == 360 · dark mode via app toggle · screenshots download/pkpd-r20-*.png.

## Not touched (for the orchestrator)

page.tsx / overview.tsx / download-panel.tsx fallback numbers (the pharmaml.pharma domain card
has no playground deep-link, so nothing to re-point); examples/ + skills/ tsc errors; the two
old "Fast Refresh full reload" dev.log lines from the interrupted draft's development (not
reproducible now — console clean).
