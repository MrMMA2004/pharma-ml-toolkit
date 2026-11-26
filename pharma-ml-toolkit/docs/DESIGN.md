# DESIGN — the PharmaML Studio design system (part18 §12.4)

This document is the single reference for the product's visual
language.  It describes what IS (the shipped token set, the Mineral
Design System direction the part11 §10.3 migration established) and
the rules that keep scientific output readable.  It is authored from
the live token sources — `frontend/src/styles/globals.css` (the
React/Tauri app), `ui/design_tokens.py` (the legacy GUI's token twin,
kept in lockstep) — not from aspirations.

## 1. Color tokens

### 1.1 The React/Tauri app (Tailwind v4, oklch)

| Token | Light | Dark | Role |
|---|---|---|---|
| `--background` | `oklch(1 0 0)` | `oklch(0.145 0 0)` | canvas |
| `--foreground` | `oklch(0.145 0 0)` | `oklch(0.985 0 0)` | primary text |
| `--card` / `--card-foreground` | `oklch(1 0 0)` / `oklch(0.145 0 0)` | `oklch(0.205 0 0)` / `oklch(0.985 0 0)` | surface + text |
| `--muted` / `--muted-foreground` | `oklch(0.97 0 0)` / `oklch(0.556 0 0)` | `oklch(0.269 0 0)` / `oklch(0.709 0 0)` | quiet fills / secondary text |
| `--primary` / `--primary-foreground` | `oklch(0.596 0.145 163.225)` / `oklch(0.985 0 0)` | `oklch(0.696 0.17 162.48)` / `oklch(0.393 0.095 163.225)` | the teal action colour |
| `--secondary` / `--accent` | `oklch(0.97 0 0)` | `oklch(0.269 0 0)` | neutral fills |
| `--destructive` | `oklch(0.577 0.245 27.325)` | `oklch(0.704 0.191 22.216)` | errors / destructive actions |
| `--border` / `--input` | `oklch(0.922 0 0)` | `oklch(1 0 0 / 10%)` / `oklch(1 0 0 / 15%)` | outlines |
| `--ring` | `--primary` | `--primary` | focus ring |

The Mineral direction is canonical: **teal action, neutral greys,
no indigo/blue primaries** (the part11 rule the web console
established; the design-research sources of §12 — Liquid Glass,
Aceternity, etc. — are references for restraint, never wholesale
adoptions).

### 1.2 Status colours (the legacy GUI's semantic twin, in lockstep)

| Status | Foreground | Soft background |
|---|---|---|
| error | `#B33840` | `#FCECEE` |
| warning | `#805400` | `#FFF4DC` |
| success | `#246C48` | `#E8F4EC` |
| info | `#2468A0` | `#E9F2FA` |

Contrast was tuned for the accessibility gates (v1.2.204): text
muted ≥ 4.5:1 on canvas AND surface; disabled ≥ 3:1.

## 2. Typography

* Family: `ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, sans-serif` — the native stack (Windows
  workstations get Segoe UI; no webfont dependency for the product's
  core UI).
* Monospace (`font-mono`) renders code, identifiers, model IDs, and
  scientific result summaries verbatim.
* Scale: Tailwind's default type scale; body text `text-sm` (14px),
  card descriptions `text-xs`, numeric tables `text-xs`–`text-sm`,
  stage titles `text-base` semibold.  Density switches (comfortable
  16px / compact 14px root) ride `html[data-density]`.

## 3. Spacing

The 4px base grid (Tailwind's spacing scale): intra-card gaps
`gap-2`/`gap-3`, control rows `gap-3`, section spacing `mt-8`,
card padding `p-4`–`p-6`.  Cards never mix paddings within one
stage; lane stacks use `space-y-6`.

## 4. Border radii

`--radius: 0.625rem` (10px); derived `--radius-sm = radius − 4px`,
`--radius-md = radius − 2px`, `--radius-lg = radius`.  Controls take
`--radius-md`, cards `--radius-lg`, chips `rounded` (4px).

## 5. Shadows and surface hierarchy

Three levels, all in oklch/neutral — the app is flat-by-default:

1. **canvas** (`--background`);
2. **card** (`--card`, the workspace surface, 1px `--border`);
3. **elevated** (menus/dialogs; the legacy twin's `#FFFFFF`-on-surface
   layer), a shadow of `0 1px 3px oklch(0 0 0 / 8%)` and up.

## 6. Light / dark theming

Both themes share the same token names (§1.1); the `.dark` class on
the root flips them.  First paint respects `prefers-color-scheme`;
the host toggle is instant (no flash).  Plotly charts re-theme live
through the `PlotlyFigure` MutationObserver on the same class
attribute — charts never carry stale theme colours.

## 7. Chart palette

| Slot | Light | Dark |
|---|---|---|
| `--chart-1` | teal `oklch(0.596 0.145 163.225)` | bright teal `oklch(0.696 0.17 162.48)` |
| `--chart-2` | amber `oklch(0.769 0.188 70.08)` | amber |
| `--chart-3` | red `oklch(0.645 0.246 16.439)` | red |
| `--chart-4` | cyan `oklch(0.704 0.14 182.503)` | cyan |
| `--chart-5` | purple `oklch(0.553 0.016 285.938)` | purple |

The **colorblind-safe override** (`.charts-cvd`, Okabe-Ito: blue
`#0072b2`, orange `#e69f00`, green `#009e73`, purple `#cc79a7`, sky
`#56b4e9`) wins in both themes by source order — a settings toggle,
never a silent replacement.

## 8. Motion principles

* Motion is decoration, never information: transitions exist for
  focus/expansion feedback (150–200ms, ease-out).
* `prefers-reduced-motion: reduce` collapses every animation and
  transition to ~0ms — a hard, global contract.
* No WebGL/WebGPU decorative layers, no heavy background video, no
  second animation engine (Motion is the sanctioned engine class if
  richer transitions are ever introduced; no Anime.js alongside it).

## 9. Responsive breakpoints

Mobile-first Tailwind prefixes (`sm` 640, `md` 768, `lg` 1024,
`xl` 1280).  The desktop app's minimum ergonomic target is ~1024px;
the lanes wrap (`flex-wrap`) rather than clip.  Touch targets stay
≥44px on touch-capable hosts.  Long tables and lists scroll inside
bounded containers (`max-h` + overflow) — the page never grows
unbounded.

## 10. Accessibility requirements

* Semantic landmarks (`main`/`nav`/`section`) and labelled controls
  (`<label>` wrapping or `aria-labelledby`); every interactive icon
  is `aria-hidden` with a text label sibling.
* Errors surface through `role="alert"` paragraphs; status text is
  text (colour never the only signal).
* Focus rings (`--ring`) never removed; keyboard order follows DOM
  order.
* Contrast gates: body text ≥ 4.5:1, large text/UI ≥ 3:1 (the
  v1.2.204 tuning, §1.2).

## 11. Scientific data-visualization rules

1. **Data dominates**: a result view's space goes to the table/plot,
   not to illustration; empty-state art is allowed only where no
   data exists yet.
2. **Honest axes**: scientific units on every axis title; legends name
   series exactly; sampling/truncation always labelled (the existing
   bounded-preview conventions: "preview rows", "top-N features").
3. **One engine**: Plotly is the interactive chart engine — no second
   chart library without a demonstrated need.
4. **Never fabricated**: synthetic/demo results carry their labels
   wherever they render; degenerate statistical answers (`n/a`, null
   metrics) print as their own states, never as 0.
5. **Colourblind lane**: the Okabe-Ito override is one toggle away
   and theme-stable.
6. **Export parity**: PNG/SVG/PDF exports come from the same figure
   the screen shows.

## 12. The legacy GUI twin

`ui/design_tokens.py` keeps the retired PySide6 GUI's QSS on the same
semantic values (see `DESIGN_SYSTEM_MIGRATION.md`).  It is
maintenance-only: the quarantined legacy surface (part18 §10) adopts
no new design features — new UI work happens in the React/Tauri app
only.
