## [1.2.228] — 2026-11-18 — round 287 (release ritual: the §11.5 Molecular Docking + §11.6 PK/PD workspaces ship)

- **The release the last two interactive rounds deferred**: the
  `v1.2.228` zip carries both un-shipped payloads — the r285 §11.5
  Molecular Docking workspace (features/docking/DockingPage.tsx, the
  ten-stage sequential workflow on the real backend: Select Ligands →
  Prepare Ligands (ligand/pdbqt artifact) → Prepare Receptor
  (receptor/prepare artifact) → Define Binding Site/Grid (manual box
  or the 58-target dockstring catalog's own center/size) → Configure
  Engine (the honest engine catalog: bundled/launchable/boundary_note,
  the Windows-PE boundary stated) → Run Docking (job mode + sync lane,
  the unlaunchable-engine 422 surfaced as the stage's honest error) →
  Inspect Poses (parse/vina + parse/dlg) → Analyze Interactions
  (contacts + PLIF + symmetry-aware pose-rmsd + consensus +
  ligand-efficiency) → Visualize → Export; services/docking/client.ts,
  all 13 routes, 16 drift guards) and the r286 §11.6 PK/PD workspace
  (features/pkpd/PKPDPage.tsx, the eight-stage sequential workflow:
  Import Data → Inspect (the real NCA + both AUC variants + the
  terminal-phase regression) → Select Model (the backend's own
  /pkpd/catalog inventory) → Configure Parameters → Fit/Simulate (sync
  lanes + the population job mode with per-batch job.progress) →
  Diagnostics → Exposure/Response Visualization → Export;
  services/pkpd/client.ts, all nine routes, 14 drift guards). The
  registry flip lands with both rows: pkpd shell → workflow, 7 of 9
  workspaces live.
- **The v1.2.228 archive is the FIRST built with the path-anchored
  exclusion** (`tools/build_zip.py` EXCLUDE_DIR_PATHS={"data/artifacts"},
  the r285 fix): the bare "artifacts" entry that made v1.2.225/226/227
  ship without frontend/src/services/artifacts/client.ts can no longer
  bite — the frontend services lane always ships.
- The dev-only backend proxy (vite.config.ts
  PHARMAML_DEV_BACKEND_PORT /api proxy, the r286 find: browser dev
  sessions could never exercise the real backend across ports without
  it) rides this archive; production untouched — the Tauri host loads
  the static dist/.
- Gates this firing: the eleven-leg suite (scratch/
  run_suite_chunks_287.py — test_pkpd_workspace.py now EXPLICIT in
  chunk 2, the chunk's list was per-file so the new family would have
  ridden no leg) green RC=0 at the payload revision; chunk 3 + chunk 7
  re-run green after the version bump (the carrier-lockstep and
  contract surfaces); tsc 0 errors, vite build RC=0, ruff clean on the
  touched file; census re-trued live through /api/toolkit/tree?fresh=1
  and the web pins synced; v1.2.228 shipped across the four homes
  (download/, toolkit download/, toolkit dist-archive/, upload/) via
  the digest-verified publish path, the offsite home
  ~/pharma-ml-toolkit-zips/ restored post-rollback with the 227+228
  pair, and the root dist-archive re-pinned; the eight-check
  tools/audit_release.py audit passed (scratch/audit287/); counter
  82 → 83.
