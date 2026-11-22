# PART14 PHASE-1 AUDIT — the factual repository and testing audit

Part14 §18 deliverable #9 ("Produce a prioritized audit report and a
proposed implementation plan") — the FIRST deliverable of the part14
era, executed read-only per §18 ("The first deliverable must be a
factual repository and testing audit, not a code rewrite").

- **Audit date**: 2026-11-23 (round 326, the 118th recorded run; the
  part14 arrival — the interactive part14 continuation, trace
  1a203001d56e82d9).
- **Audited state**: the post-r325 disk tree over the sixteenth
  rollback (see §A) with the remote terminal `ad5f00a7` verified
  read-only via `git fetch` (the PAT credential is gone — no
  authenticated API; the public git protocol sufficed for ls-remote +
  fetch; the REST API is rate-limited for this egress IP).
- **Scope**: §18 tasks 1–9 to their read-only extent. No project
  source changed, nothing committed, nothing pushed (part14 §16.24 —
  "Do not commit, push, publish, tag, or delete branches without
  explicit authorization").

## A. Repository State (§18.1, §17A)

| Item | Verified value | Evidence |
| --- | --- | --- |
| Local HEAD | `2eb0ede4` (the round-186 machinery checkpoint) | `git rev-parse HEAD` this session |
| Remote main | `ad5f00a7` = the round-325 close-out push (2026-11-22 11:05Z) | `git ls-remote` + read-only `git fetch`; commit message read verbatim |
| Synchronized? | **NO** — local is 528 commits behind; the branch pointer was rewound to the r186 checkpoint by the machinery rollback (the sixteenth recorded arrival of the deepest class) | `git rev-list --count HEAD..FETCH_HEAD` = 528 |
| Working tree | 953 modified/untracked paths, **fully understood at hash level**: every terminal-tracked source file is byte-identical on disk except `.zscripts/dev.pid` (runtime PID noise); ghost additions = 5 × `pharmaml.egg-info/*` (the venv editable install) + `tests/test_playground.py` (a rollback-resurrected file — tracked at r186, deleted in the terminal era; it alone carries 753 collected tests, exactly explaining the 9,308-disk vs 8,555-terminal census delta); 245 terminal files absent from disk, ALL under `extracted/…` (the user's legacy uploaded code family, not application source) | full-tree `git hash-object` vs `ls-tree` sweep this session |
| Version carriers | `1.2.240` across `VERSION`, `pyproject.toml`, `pharmaml.__version__`, frontend `package.json`, `tauri.conf.json` (§2.4 reconciliation — no drift; the root web console's own carrier is `0.2.1`, a distinct surface) | read this session; venv repair printed `pharmaml 1.2.240` |
| Published release | `v1.2.231` per the status doc (r325 verified its assets live at the ledger bytes); the unauthenticated API is rate-limited this session — NOT re-verified live here (honest: cited, not re-proven) | `docs/PROJECT_STATUS.md` + this session's API block |
| Release-zip homes | v1.2.240 at the EXACT ledger bytes 22,449,138 across `upload/`, `download/`, `pharma-ml-toolkit/download/` (disk homes intact); the web console's zip-size API answers live `{"zipBytes":22449138,"version":"1.2.240","exact":true,"source":"release"}` | `ls -la` byte sizes + live HTTP 200 this session |
| CI at the terminal | 19/19 per the r325 evidence — NOT re-verified this session (API rate-limited; PAT gone) | `docs/PROJECT_STATUS.md` |

**The sixteenth rollback, reconciled** (the documented r306–r313/r322
deepest class, now verified rather than assumed): history reset to
`2eb0ede4`, origin detached, the PAT credential gone, the venv gutted
(numpy floated to 2.1.3; rdkit/torch/pharmaml/PySide6/mypy/ruff/
pytest-xdist/pyinstaller missing — fastapi/uvicorn/pydantic/httpx/
pytest survived), the Studio frontend node_modules wiped, the root web
console node_modules under an ACTIVE per-call wipe (the r323–r325
documented class — re-verified live: three consecutive wipes observed
mid-session). Recovery executed this round to the read-only extent:
the venv re-pinned via the established idempotent playbook (final
import surface verified incl. the offscreen PySide6 GUI stack), the
root web console reinstalled + the double-fork dev daemon restarted
(the r258 lesson — single spawns die at tool-call boundaries). **The
branch-pointer recovery (reset onto `ad5f00a7`, sweeping the
resurrected ghost) is deliberately NOT executed** — part14 §16.24
requires explicit authorization; it is proposed in §G.

## B. Findings (§17B)

### Confirmed defects — the critical incident reproduced from current source (§18.4–5)

Live probe over the REAL `create_app()` (as the Tauri host launches
it), browser-shaped headers, evidence at
`scratch/p14_r326_origin_probe.py` + `.txt`:

| Leg | Request shape | Result |
| --- | --- | --- |
| P1 | `Origin: http://tauri.localhost` + valid token | **403 "Non-loopback Origin rejected."** |
| P2 | `Origin: https://tauri.localhost` + valid token | **403** (same) |
| P3 | OPTIONS preflight from `tauri.localhost` | **405 Method Not Allowed** |
| P4 | OPTIONS preflight from the dev origin `localhost:5174` | **405** (any origin) |
| P5 | loopback origin, no token | **401 "Invalid or missing session token."** |
| P6 | loopback origin + valid token | **200** (the only lane that works) |
| P7/P8 | non-loopback Host (tauri.localhost / LAN) | **403 Host rejected** |

- **D-1 (Critical, confirmed)**: `src/pharmaml/api/security.py`
  `LOOPBACK_HOSTS = {127.0.0.1, localhost, ::1, [::1]}` lacks
  `tauri.localhost` — the production WebView2 origin of the packaged
  Tauri 2 app (`tauri.conf.json` has no origin override; Tauri 2 on
  Windows serves from `tauri.localhost`). Every packaged-app REST
  request is 403-rejected even with a valid session token. The dev
  origin (`http://localhost:5174` per `devUrl`) IS loopback —
  production and development origins differ, which is why dev works
  and the installed app fails.
- **D-2 (Critical, confirmed)**: `create_app()` wires the guard as a
  route-level dependency and adds NO `CORSMiddleware` — an OPTIONS
  preflight (mandatory cross-origin for the JSON + `X-Session-Token`
  request shape `client.ts` sends) is answered 405, so the browser
  blocks the actual request and `fetch` rejects. Dev survives only
  because the Vite dev proxy makes requests same-origin (no
  preflight). CSP is NOT the blocker (`connect-src` already permits
  `http://127.0.0.1:*`/`ws://127.0.0.1:*`).
- **D-3 (High, confirmed at source)**: `frontend/src/services/api/
  startup.ts` wraps every health-probe failure as
  `StartupError("health", "the backend health probe failed: …")` —
  the reported "The backend did not start / stage: health / Failed to
  fetch" is the mislabeled CORS/403 failure chain (part14 §3.1 item
  12: errors are NOT reported accurately).
- **D-4 (Medium, potential — code-review finding, needs Phase-2
  confirmation)**: `src-tauri/src/backend_manager.rs` handshake
  parse-failure path formats the RAW `PHARMAML-READY` line (which
  contains the session token) into the error surface — a §3.3
  redaction violation if that string reaches a log/UI message.
- **D-5 (Medium, gap)**: the same Rust host performs no handshake
  field validation (port bounds, pid sanity, token presence/format,
  version shape) — §3.3's list, currently unenforced.

### Architectural risks

- The fix surface is security-sensitive: §3.2 forbids weakening
  (no `*` origins, no auth bypass, no non-loopback bind). The
  trusted-origin policy must be narrowly scoped to the actual Tauri
  app origins with regression tests proving arbitrary origins/hosts
  stay rejected.
- `ws_router` is mounted WITHOUT the guard dependency — WebSocket
  auth is handled elsewhere; the §3.1 item 8 (WS/REST auth
  consistency) needs a Phase-2 audit leg.

### Documentation inconsistencies

- `docs/PROJECT_STATUS.md` is current through r325 (the first clean
  arrival); its "HEAD == origin/main, tree 0/0" claims were true at
  r325 close but are stale NOW (the sixteenth rollback landed after).
  This audit supersedes on current state; the r325 record stands as
  history (per §2.4: historical records stay historical).
- The part13-era "push each round" protocol is superseded by part14
  §16.24 (no commit/push/publish/tag without explicit authorization).

## C. Changes (§17C)

**Project source: NONE** (Phase 1 read-only — honored; §16.1
"Inspect before editing" satisfied). Environment/restoration changes
(all reversible, all documented):

- `/home/z/.venv` re-pinned via the established idempotent
  `scratch/venv_repair.sh` (the r306–r313 playbook — the round-196
  shap/no-deps rule, the EGL stub copy, the rdkit-stub patch, the
  editable `pharmaml` install). Pins verified: numpy 1.26.4, torch
  2.14.1+cpu, matplotlib 3.10.9, rdkit 2026.9.1, pharmaml 1.2.240,
  fastapi 0.128.0, uvicorn 0.44.0; the PySide6 6.8.3 GUI stack
  imports and instantiates offscreen.
- Root web console node_modules reinstalled (`bun install`, 828
  packages) three times against the active per-call wipe; the dev
  server restarted via the double-fork daemon (`scratch/dev_daemon.py`
  — the r258 pattern). No security implication (dev-only deps).
- New artifacts: this report; `scratch/p14_r326_origin_probe.py` +
  `.txt` (the reproduction evidence); the pins pair
  `docs/gui/screenshots/p14r326-pins-{desktop,mobile}.png`; the
  worklog/counter ledger entries. All uncommitted per §16.24.

## D. Test Results (§17D) — executed this round

| Test | Result | Evidence |
| --- | --- | --- |
| Origin/OPTIONS reproduction (8 legs) | 8/8 as predicted (D-1/D-2) | `scratch/p14_r326_origin_probe.txt` |
| pytest collection census | 9,308 tests / 471 files on disk = terminal's 8,555/470 + the resurrected ghost's 753/1; zero collection errors (exit 0) | this session |
| venv pin restoration | all pins verified; GUI stack offscreen OK | repair output |
| Web console render | HTTP 200 / 202 KB; title verified; ML-card interaction renders the registry/training surface; zero fresh page errors | live probes + `p14r326` pins pair |
| zip-size API | 22,449,138 `exact:true source:release` (ledger parity) | live HTTP 200 |
| ECC inspection | `ecc-universal@2.2.3` confirmed current (shasum `11f410d5…`, MIT, node≥18); Windows-native limits verified from the official README (continuous-learning v2 observer daemon + memory-vault defects #2489/#2626; shell-backed features need Git Bash/WSL); NOT installed per §18 | `npm view` + README fetch |
| Vitest / tsc (Studio frontend) | **NOT re-run** — node_modules wiped by the rollback; re-install + re-run queued (§G) | — |
| ruff / mypy / ten-chunk gate | **NOT re-run this round** — the r325 terminal evidence (clean) stands as last-verified; re-rides next round with the recovery | `docs/PROJECT_STATUS.md` |
| Rust / tauri build / EXE-MSI | **NOT re-run** — no Windows toolchain in this sandbox (BLOCKED, §E) | — |

## E. Blocked Tests (§17E)

| Blocker | Why | Required environment | Lowest-cost solution |
| --- | --- | --- | --- |
| Clean-machine Windows 10/11 smoke; NSIS/MSI installers; offline WebView2; signed-EXE validation; Stage-5 PySide6 retirement gate; physical GPU/docking benchmarks; 45–60-min stability | This development environment is a **Linux sandbox**, not the documented Windows 10 box (§9's hardware claims cannot be re-verified here either) | a physical clean Windows 10/11 machine (a VM is acceptable for most legs per §9.5) | the §9.5 clean-VM procedure; stays BLOCKED until then — never PASS |
| CI re-verification at HEAD | PAT credential gone; unauthenticated API rate-limited for this egress IP | any GitHub token or a different egress | read-only `git fetch` (already used); CI verdict re-check queued with the remote re-attach |
| Studio frontend vitest/tsc | frontend node_modules wiped by the rollback | a one-window `npm ci` + run (the r320 precedent) | queued for the recovery round |
| ECC installation (dry-run + guided) | §18: forbidden until reviewed and approved; also no Claude Code/Codex harness exists in THIS sandbox (the coding agent here is Z.ai Code) — §4.1's "identify the actual available harness" answers "none of the listed ones" | the Windows dev box (or an explicit harness decision) | proposed plan in §G; metadata/license/shasum already captured |

## F. Remaining Risks (§17F)

1. D-1/D-2 fix must stay inside §3.2's constraints (narrow trusted
   origins, no auth weakening, loopback-only bind).
2. The branch pointer still sits 528 commits behind the pushed
   terminal; until the authorized reset, every disk-vs-git comparison
   carries the ghost-file noise (understood, but error-prone).
3. D-4: a malformed handshake could leak the session token into an
   error surface (unconfirmed severity — Phase-2 test needed).
4. WS-vs-REST auth consistency unaudited (§3.1 item 8).
5. The three release lanes (clean-machine smoke, PySide6 retirement,
   installer signing) remain BLOCKED — unchanged from r325.

## G. Next Actions — the proposed implementation plan (§15 Phase order)

1. **Phase 2 (needs the user's go — the §16.24 authorization for the
   tree reset rides with it)**: at next round-open, execute the
   documented recovery: re-attach `origin`
   (https://github.com/MrMMA2004/pharma-ml-toolkit.git), reset the
   branch onto the verified terminal `ad5f00a7`, sweep the ghost
   residues (`tests/test_playground.py`, `pharmaml.egg-info/`), re-verify
   the tree 0/0, then re-run ruff/mypy/vitest/tsc to re-pin the
   quality gates at the terminal.
2. **Phase 2 proper (the incident fix)**: the narrowly-scoped
   trusted-origin policy — a `TRUSTED_APP_ORIGINS` addition
   (`http://tauri.localhost` + `https://tauri.localhost`, scheme-strict,
   host-exact) in `security.py`; a scoped CORS preflight lane that
   answers OPTIONS for the trusted origins WITHOUT bypassing the
   token guard on actual requests; the D-3 mislabel fix (typed
   error stages: network/CORS/auth vs "backend did not start");
   regression tests for every leg of the P1–P8 matrix + arbitrary
   origins/hosts staying 403.
3. **Phase 2 (§3.3 hardening)**: handshake field validation (port
   bounds/pid/token format/version) in `backend_manager.rs` + the
   token redaction in the parse-failure path + malformed-handshake
   regression tests.
4. **Phase 3 (ECC)**: on the Windows box (or with an explicit harness
   decision), the reviewed dry-run
   `npx ecc-universal@2.2.3 install --guided --harness codex --dry-run`
   — the exact pinned version verified current this round; native
   Windows limits documented from the official README.
5. **Phases 5–8**: the `docs/QA/` deliverable set
   (MASTER_TEST_PLAN, MANUAL_TEST_CHECKLIST, AUTOMATED_TEST_MATRIX,
   WINDOWS_TEST_MATRIX, PERFORMANCE_BASELINES, SECURITY_TEST_REPORT,
   BLOCKED_TESTS_AND_SOLUTIONS, DEFECT_REGISTER,
   RELEASE_ACCEPTANCE_REPORT) built on the §13 structure, with D-1…
   D-5 as the defect register's opening rows.

## H. Release readiness gates (§14, current honest status)

| Gate | Status | Basis |
| --- | --- | --- |
| A — Repository integrity | **PARTIAL** | tree understood at hash level; versions consistent; HEAD NOT synchronized (rollback); no push (unauthorized) |
| B — Backend and security | **FAIL** | "Tauri origins are handled correctly" is FALSE (D-1/D-2 reproduced live); loopback bind/token guard themselves verified working (P5–P8) |
| C — Frontend/scientific | **PARTIAL** | collection parity + web console verified live this round; full suites not re-run |
| D — Automated tests | **PARTIAL** | collection 8,555-equivalent / 0 errors re-verified; run-time suites cited from r325, not re-executed |
| E — Performance/stability | **BLOCKED** | needs the physical Windows legs |
| F — Windows release | **BLOCKED** | unchanged three lanes |
| G — Release integrity | **PARTIAL** | disk homes at exact ledger bytes + live exact:true API; CI/SBOM cited from r325 |

— recorded by round 326; every PASS above names its evidence; nothing
converts absence of evidence into a pass.
