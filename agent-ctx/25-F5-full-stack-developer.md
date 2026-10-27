# Task 25-F5 — full-stack-developer (Code Explorer data preview)

**Task:** CSV/PDB preview in the Code Explorer (the "notebooks must work entirely
based on the files I give them" story — data files were size-only nodes before).

## Files touched (writable scope respected)

- `src/lib/toolkit-types.ts` — added `DataFileKind`, `CsvPreviewResponse`,
  `PdbPreviewResponse`, `DataPreviewResponse`, `FilePayload` union
  (`FileContentResponse` unchanged → source path stays byte-compatible).
- `src/lib/data-file.ts` (new) — client-safe `dataFileKind(path)` extension
  sniff (`.csv`/`.pdb`); imported by server (toolkit-fs, file route) AND client
  (code-explorer skeleton pick). No Node imports.
- `src/lib/toolkit-fs.ts` — tree walk: data nodes get `language:"csv"|"pdb"`
  but are never line-counted (LOC stat stays source-only, 62,679 was synced
  with README/hero). Comment documents why `.csv/.pdb` are NOT in
  `ALLOWED_EXTENSIONS`.
- `src/app/api/toolkit/file/route.ts` — data branch before the source gate.
  CSV: full parse ≤ 512 KB via shared `parseCsv` (lib/csv.ts) → exact
  rowCount, ≤ 50 previewRows. PDB: streamed readline scan ≤ 8 MB → exact
  atomCount (ATOM+HETATM), residueCount (unique chain:resSeq keys),
  totalLineCount; only first 200 lines ever enter JSON (`previewText`).
  Larger files → actionable 400 pointing at the Download tab. Traversal
  guards (`..` check + `resolveToolkitPath`) untouched — still 400 for
  `../../etc/passwd` and `/etc/passwd`.
- `src/components/explorer/data-file-preview.tsx` (new) — `DataFilePreview`:
  CSV table (shadcn TableHeader/TableRow/TableHead/TableBody/TableCell over a
  raw `<table>` — the shadcn `Table` wrapper's nested overflow container
  breaks sticky thead; border-separate + cell borders make sticky survive
  scroll), zebra via `.data-table`, sticky thead, # gutter, numeric
  right-align + tabular-nums (≥90 % parse heuristic), title-attr tooltips,
  28ch truncated cells, "showing N of M rows · K columns · X KB" caption +
  "+N more rows in file" badge, empty state. PDB: atoms/residues/lines/size
  Badge chips + CodeBlock monospace preview (200 lines) + caption.
- `src/components/explorer/code-explorer.tsx` — `FilePayload` union content
  state/cache, kind-aware header badges (rows·cols / atoms·residues / size),
  `DataSkeleton` (table-shaped for csv, chip+lines for pdb) while loading,
  `DataFilePreview` render branch (FileOutline/CodeBlock only for source).
- `src/components/explorer/file-tree.tsx` — `Table2` icon (csv) / `Atom` icon
  (pdb) in primary accent = clickable-previewable affordance; size-on-right
  label (fmtBytes) instead of line count; aria labels
  "…, csv data file, 77.5 KB, click to preview".

## NOT touched

pharma-ml-toolkit/** (read-only), README.md, docs/INDEX.md, others' worklog
entries, examples/data (byte-unchanged).

## Verification evidence

- `bun run lint` exit 0; `bunx tsc --noEmit` → 0 errors in src/ (pre-existing
  errors only in examples/ + skills/, untouched).
- curl matrix: qsar_classification.csv {420 rows, 11 cols, truncated};
  lysozyme_1aki.pdb {1,079 atoms = 1001 ATOM + 78 HETATM (matches round-24
  pins), 207 residues = 129 protein + 78 waters, 1,437 lines, 200-line
  preview}; tests/data/1crn.pdb {327 atoms, 46 residues (1CRN = 46-residue
  crambin ✓), 611 lines}; seed_molecules.csv {12 rows, 443 B, not truncated};
  traversal `../../etc/passwd` → 400, `/etc/passwd` → 400, missing → 404;
  non-viewable .dlg → same 400 message as before; README.md / .py source
  payloads byte-identical shape {content,language,lines,path,size}.
- Tree API: all 16 csv/pdb nodes have language markers, lines=null; stats
  drift 62,679→63,578 lines (+43 KB) observed mid-session is a PARALLEL
  writer editing scratch/ (file count unchanged at 219; data nodes never
  counted) — not caused by this task.
- agent-browser E2E (session task25f5, 1280×900): tree data nodes render new
  affordance; CSV click → table with 50 real rows (c1ccccc1O / CMPD-0000 /
  0.15175264883764342), sticky thead (computed position:sticky, top:0), th
  scope=col, numeric cells right-aligned, title tooltip on SMILES cell,
  caption "showing 50 of 420 rows · 11 columns · 77.5 KB" + "+370 more rows
  in file"; PDB click → "1,079 atoms · 207 residues" header badge, 4 summary
  chips, HEADER/TITLE/COMPND records in monospace CodeBlock with line
  numbers, caption "showing 200 of 1,437 lines · 113.7 KB · full structure
  ships in the downloadable archive"; zero console errors. 360×800: table
  1,702 px scrolls inside its 300 px `overflow:auto` container, document
  scrollWidth = 360 (no page-level overflow — inner scroll is the project
  convention); dark-mode screenshot captured. Screenshots:
  `download/explorer-25f5-{csv,pdb,csv-mobile,csv-dark}.png`.
- Empty states verified by code review only — no shipped empty CSV/PDB
  fixture exists and the toolkit is read-only for this task.
