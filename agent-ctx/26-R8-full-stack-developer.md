# Task 26-R8 — full-stack-developer (data preview rebuild)

## Summary
Rebuilt the Code Explorer's data-file preview lost to the sandbox reset: `/api/toolkit/file`
now returns kind-tagged previews (CSV / PDB / plain text) for the shipped data files under
`pharma-ml-toolkit/examples/data/`, with a kind-aware viewer wired into the Explorer UI.

## Files touched
- `src/app/api/toolkit/file/route.ts` — kind dispatch, caps (CSV ≤512 KB, PDB/text ≤8 MB), guards unchanged
- `src/lib/data-file.ts` (new) — pure isomorphic parsers (RFC4180 CSV, PDB census, text slicing)
- `src/lib/toolkit-fs.ts` — `readToolkitFileBounded` (top-level `createReadStream`), data-node language markers, no LOC counting
- `src/lib/toolkit-types.ts` — CsvPreviewResponse / PdbPreviewResponse / TextPreviewResponse / AnyFileResponse
- `src/components/explorer/data-file-view.tsx` (new) — CSV table, PDB chips + records, text viewer, skeletons, badges
- `src/components/explorer/code-explorer.tsx` — kind-aware badges / skeletons / rendering
- `src/components/explorer/file-tree.tsx` — Table2/Atom icons, size labels, aria labels

## Verification highlights
- curl matrix: qsar 420 rows/11 cols/truncated; lysozyme 1,079 atoms/207 residues/1,437 lines; seed 12 rows untruncated; r.dlg text; traversal + absolute → 400; missing → 404; README.md byte-identical shape
- `bun run lint` exit 0; `bunx tsc --noEmit` clean in src/ (4 pre-existing out-of-scope errors in examples/websocket + skills/)
- agent-browser E2E 1280×900 (real table, chips, zero console errors) + 360px (no page overflow, inner-only scroll)
- Tree stats: data nodes never line-counted (lines=null), LOC stat source-only

Full log: see worklog.md entry "Task ID: 26-R8".
