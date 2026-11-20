# Bug Report (part12 Phase 1–2)

Evidence-driven findings from the part12 audit session (2026-11-20).
Severity reflects user/packaging impact. Every fix this round carries its
verification; regression coverage is listed per bug.

## Fixed this round

### B-1 — `artifacts` any-depth exclusion dropped real frontend source (HIGH, packaging)

- **Evidence**: `git ls-files pharma-ml-toolkit/` (1,582 tracked) vs
  `unzip -l pharma-ml-toolkit-v1.2.231.zip` (919 files): exactly one
  non-excluded tracked source file absent —
  `frontend/src/services/artifacts/client.ts`. The only `artifacts`-named
  dir in the tree IS that real source dir. Tree walk undercounted by the
  same file (1,581 vs 1,582); the file was unbrowsable in the explorer.
- **Root cause**: `EXCLUDED_DIRS`/`ZIP_EXCLUDE_ARGS` matched the bare
  basename `artifacts` at any depth — intended only for Python
  generated-output dirs (`examples/artifacts/` leaderboards).
- **Impact**: a rebuild-from-zip of the frontend fails tsc (workspaces
  import the artifacts client); the shipped source archive is incomplete.
- **Fix**: scoped, relative-path-anchored exclusions
  (`artifacts`, `examples/artifacts`) in `src/lib/toolkit-fs.ts` (tree walk
  + ZIP pattern list). Verified: live walk 1,581→1,582 with the file
  browsable; test-zip semantics show real source included + generated
  output excluded. **Regression**: the round's zip builder + the archive
  audit (`tools/audit_release.py`) re-verify composition; the tracked-vs-
  zip diff must stay empty of viewable source.
- The published v1.2.231 release (EXE/MSI) is unaffected — it builds from
  the repo tree, not the source zip.

### B-1b — `frontend/dist` leak into census + archive (MEDIUM, latent)

- **Evidence**: the session's local `npm run build` created
  `frontend/dist` (index.html + 6 bundles, ~5.7 MB); the tree census
  jumped +7 files until the fix. Shipped v1.2.231 zip has 0 dist entries
  (historically no local builds existed — node_modules wiped by
  rollbacks), which is why this never fired before.
- **Fix**: scoped `frontend/dist` exclusion (tree + ZIP). Verified by the
  live walk returning to the tracked-set count.
- **Regression**: any round that builds the frontend locally re-runs the
  census parity check (the close-out ritual).

### B-2 — `/api/v1/database` persistence surface had no GUI consumer (MEDIUM, integration)

- **Evidence**: `routes/database.py` (status/jobs/preferences/backup/
  restore, 67 persistence tests green) consumed by zero of the three
  frontends, while `IntroductionPage.tsx:64-66` claimed "backup and
  restore built in".
- **Fix**: new runtime-narrowed `services/database/client.ts` + the
  Introduction "studio database" card (live status + row counts,
  persisted job history, backup → digest-verified artifact download,
  two-step-confirm restore with the replace-registry warning). tsc clean,
  frontend drift guards green, production build green.
- **Regression**: covered by the existing persistence test suite
  (backend) + the round's E2E browser pass (frontend). A dedicated
  frontend unit for the card lands with the next frontend-test campaign
  (the repo's frontend tests are contract guards, not component renders —
  noted honestly).

### B-3 — malformed nested CI workflow (LOW, hygiene)

- **Evidence**: `pharma-ml-toolkit/.github/workflows/ci.yml` trigger read
  `branches: ain]` (corrupted edit); the file is invisible to Actions
  (root-level registration rule) and a stale subset of the root CI.
- **Fix**: removed (same sweep precedent as round-301's nested
  windows-release.yml). Census effect documented (-1 file / -52 lines;
  pins re-trued this round).

## Documented, not fixed this round (owner decisions / later phases) — r307 reconciliation: B-4/B-5/B-6 closed and B-7 dispositioned by the r303/r304 follow-ups (statuses below trued; evidence in CHANGELOG [1.2.231] + RELEASE_READINESS.md)

### B-4 — Applicability Domain: no standalone leverage/Williams method (MEDIUM, scientific)

kNN-distance + bootstrap-uncertainty + DomainBox are implemented and GUI-
plotted; the leverage/Williams hat-matrix exists only inside Cook's
distance diagnostics. Part12 PART 11 names it "where implemented" —
currently it is not, and the GUI does not pretend otherwise (no Williams
plot is exposed). See SCIENTIFIC_VALIDATION_REPORT.md §AD for the
add-or-document decision frame.

**CLOSED r303**: `ml/applicability.py::leverage_williams` (hat-matrix
leverage, h\* = 3(p+1)/n, standardized-residual band, regression-only
guards) + `plot_williams` + `/api/v1/applicability-domain/leverage-williams`
(job/sync + the figure artifact + the four-entry catalog) + the
React/Tauri ML workspace's Williams lane; 23 unit + 3 figure + 7 route
tests. Verified in-tree at the r307 reconciliation.

### B-5 — chirality fixed OFF in fingerprint feature factory (LOW, scientific)

`chem/features.py` generators pin `includeChirality=False`; the
fingerprint configuration exposes nBits/radius/count modes but no chirality
toggle. Documented as a gap (FEATURE_GAP §fingerprint parameters) —
enabling it changes feature semantics (a data-compat decision, not a
mechanical edit).

**CLOSED r303**: `include_chirality` exposed through
`compute_fingerprints`/`FeatureConfig`/`/api/v1/chemistry/fingerprints` +
the Datasets fingerprint-card toggle (morgan-only, default OFF — the
data-compat caution honored, byte-identical history preserved); 8
descriptor + 8 factory + 3 route tests.

### B-6 — Tk playground retired-but-shipped (MEDIUM, dead weight)

`playground.py` (30,047 lines) is declared RETIRED (studio docstring r228;
README says retired at v1.2.141) yet still ships, with
`tests/test_playground.py` (16,370 lines, the largest test file) still
collected. Removing it changes test counts (~980 of its tests were already
retired once per README) and zip composition — an owner-gated Phase-8
decision under the same rule as PySide6 retirement (preserve working
functionality; retire legacy only with evidence).

**CLOSED r304**: both files left the shipped tree (the v1.2.141
declaration executed; the final version preserved in git history); the
collection re-trued 8,643 → 8,362; the zip builder's NEGATIVE gate pins
the pair's absence. Verified in-tree at the r307 reconciliation.

### B-7 — 9 doctest-only modules + 3 duplicate-symbol families (LOW, maintainability)

See ARCHITECTURE_REVIEW A-6. No user-facing defect; matrix rows carry the
"verified-by-doctest-only" tier and the duplicate-symbol map for the next
hardening campaign.

**Dispositioned r304**: pinned-not-rewritten — the ten docstring-map
cross-reference sites + `tests/test_architecture_hygiene.py` (472 tests)
guard the definition-site sets, the zero-importer census, and the numeric
parities; drift fires a test, not an audit.

## Cleared (audited, no defect found)

- WebSocket reliability (hello/replay/heartbeat/backoff/dedupe + REST
  reconciliation) — guards green; the uvicorn shutdown bug class was fixed
  with `TestGracefulShutdownPush` in r280.
- Version-carrier lockstep — 8 carriers green this session, guarded by two
  named tests.
- Release artifact integrity — assets verified against the live API
  (sizes/hashes/state); manifest + SHA256SUMS present.
- No token/PAT exposure in URLs/config/logs this session (askpass shim;
  recovery scripts print length + sha12 only).
