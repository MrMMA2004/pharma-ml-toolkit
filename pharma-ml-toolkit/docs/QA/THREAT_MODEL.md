# THREAT MODEL — part17 §4.5 deliverable (Round 356, lane P1)

The application penetration-testing program (part17 §4) begins with
this model: what the PharmaML Studio product is, what it protects,
who touches it, where the trust boundaries sit, and what a motivated
attacker tries at each boundary. Every row names its evidence — the
source file that implements the control, the test suite that pins
it — so the model is falsifiable, not aspirational.

The four scope surfaces (part17 §4.2 A–D) map to the product's real
architecture:

| Surface | The real component | The boundary it lives behind |
| --- | --- | --- |
| A. FastAPI backend | `src/pharmaml/api/` (uvicorn, loopback-bound) | the session token + loopback Host + trusted-Origin guard (`api/security.py`) |
| B. React/Tauri desktop | `frontend/` + `src-tauri/` (the packaged WebView2 app) | the Tauri capability allowlist + the WebView2 CSP (`src-tauri/tauri.conf.json`) |
| C. Next.js web console | the repository's web console (the build-gated `bun run build` surface; local-only in the shipped topology) | the console's own route validators |
| D. Supply chain | `requirements.txt`, `frontend/package-lock.json`, `src-tauri/Cargo.lock`, the SBOM, the CI workflows | the CI-side audits (`security-scan.yml`, CodeQL, the tree hygiene guard) |

## 1. Assets (what an attacker wants)

| Asset | Where | Loss scenario |
| --- | --- | --- |
| The user's scientific datasets | in-session `DatasetStore`; `PHARMAML_STUDIO_DATA` vault (`persistence.py` `FileVault`) | disclosure, silent corruption, destruction |
| Trained models + experiment lineage | `ModelStore`, `RunStore`, SQLite registry | poisoning (a swapped joblib), provenance loss |
| Artifacts + exports | `ArtifactStore` + the download lane | exfiltration, header/response splitting via export names |
| The session token | in-memory `SessionSecurity` (`api/security.py` | any authenticated operation the guard exists to gate |
| The SQLite database + file vault | `studio.db` + `files/` under the data dir | the backup-bundle restore lane (ZIP-slip class) |
| The host filesystem | reached only through the vault's path policy | traversal/symlink escape writes outside the data dir |
| The release pipeline | `.github/workflows/*`, the PAT | workflow injection, secret theft, artifact poisoning |

## 2. Actors (who can touch what)

* **The owner-user** — the single local user. Every route is theirs;
  there is deliberately no multi-tenant split (an honest design
  decision this model records, not hides: the product is a local
  single-user scientific workbench).
* **The packaged WebView2 app** (Tauri 2, `tauri.localhost`) — the
  trusted app origin, holding the token via the startup handshake.
* **A local non-browser tool** — absent Origin/Referer headers is a
  designed lane (curl, the Rust host) — still token-gated.
* **A remote network attacker** — can only reach the host if the
  loopback bind is violated; the design goal is that this actor has
  **zero** reach: the guard rejects non-loopback Host before routing
  (pinned in `tests/api/test_trusted_origins.py`).
* **A malicious web page in the user's browser** (CSRF/DNS-rebinding
  class) — the Origin/Referer predicate + `allow_credentials=False`
  CORS lane is the control; pinned by the 34-leg origin matrix and
  the WS-origin suite.
* **A malicious upload** — bytes the user is tricked into loading:
  the extension allowlist, size cap, safe-stem lanes, and the
  restore-bundle validation are the controls.
* **The CI supply chain** — a compromised dependency or workflow
  injects itself at build time; the pinned lockfiles, SBOM,
  pip-audit/npm-audit/CodeQL lanes are the controls.

## 3. Trust boundaries and their controls (with evidence)

| Boundary | Control (source) | Pinned by (test) |
| --- | --- | --- |
| Browser/page → backend REST | loopback Host, Origin/Referer predicate, session token (`api/security.py::guard_request`) | `test_trusted_origins.py` (34 legs), `test_penetration_security.py` (§4.3 battery) |
| Preflight/CORS | the scoped anchored-regex CORS lane, `allow_credentials=False` (`api/__init__.py`) | the preflight legs in the origin matrix + the CORS-complement class |
| WebSocket connect | the same origin predicate pre-accept + close 4401/4403 (`routes/events.py`) | `test_ws_origin.py` (12/12) |
| WebView2 → backend | the capability allowlist (`core:default` ONLY) + the loopback-only CSP (`src-tauri/`) | the static pins in `test_penetration_security.py` (the desktop-surface class) |
| Host process → token handoff | the single `PHARMAML-READY` handshake line, token never in diagnostics (`api/server.py`) | handshake-shape pins; the Rust-side `validate_handshake()` cases (compile-blocked on this host, ride the Windows release pipeline) |
| Upload bytes → data registry | extension allowlist, 64 MiB cap, safe-stem (`routes/datasets.py`, `utils/io.py::safe_filename`) | upload-attack class + the existing 415/413/422 lanes |
| Registry → filesystem | the vault's resolve-and-contain policy (`persistence.py::_resolve`) | vault traversal/symlink class |
| Backup bundle → restore | the member allowlist (`files/` prefix only), lexical + resolved containment (`persistence.py::restore_bundle`) | the ZIP-slip class |
| Export names → response headers | registration-time `safe_filename` + header-time CR/LF/quote strip (`routes/artifacts.py`) | the F-356-01 class (fixed this round) |
| Any input → SQLite | fully parameterized `_execute` (`persistence.py`) | the injection class |

## 4. STRIDE over the four surfaces (summary verdicts)

| Surface | Spoofing | Tampering | Repudiation | Info disclosure | DoS | Elevation |
| --- | --- | --- | --- | --- | --- | --- |
| A. backend | the token lanes (constant-time verify, per-session rotation) | SQL is parameterized; the vault is contained | single-user local product: job/run history is the audit lane | log hygiene pins + the digest-not-token handshake | bounded stores + upload caps + job slots | the guard runs before routing on every route |
| B. desktop | the handshake validation + the trusted-origin lane | no shell/fs capability exists to abuse | n/a (local single user) | CSP `object-src 'none'`, no wildcard sources | the Rust host supervises the sidecar | the ONE capability is `core:default` |
| C. web console | build-gated; no runtime auth surface of its own in the shipped topology | route validators; JSON-only API responses | n/a | no secrets in the console tree (the supply-chain scan pins this) | static build output | no server-side execution lane |
| D. supply chain | lockfiles pinned; SBOM recorded | pip-audit + npm audit + CodeQL + the tree scan in CI | CI run history | the secret-pattern scan (the PAT lives disk-only, `.gitignore`-quarantined) | billing-bounded runner use | workflows `contents: read` only |

## 5. Accepted risks (documented, deliberate)

1. **No TLS by design** — the backend is loopback-only; the token
   never crosses a network segment it could be sniffed on.
2. **Single-user, no authorization model** — a local scientific
   workbench; the "roles" lane (API5 BFLA) degenerates honestly to
   "every route equally token-gated" (pinned by the BFLA lane).
3. **`style-src 'unsafe-inline'` in the desktop CSP** — the Tauri
   inline-style requirement; `script-src` stays strict (pinned).
4. **The CI push-trigger filter is cosmetically malformed**
   (`branches: ain]` — see the register, F-356-02): observed
   behavior is that pushes to main still fire CI (verified this round
   via the Actions API: runs 371–376, `push`-event, all `success`),
   so the defect is hygiene-grade, recorded for cleanup.

## 6. Out of scope (honest boundaries)

* No active scanning of anything except the project-owned ephemeral
  test instance (part17 §4.4's own rule).
* No network-facing scanner is ever part of the installed runtime.
* The Windows-only lanes (installer signing, clean-machine
  installation) stay with their long-standing BLOCKED_ENVIRONMENT
  register entries.

The living evidence for every row above: the automated suite
`tests/api/test_penetration_security.py` (this round, 67 tests) plus
the standing suites it complements — see
`PENETRATION_TEST_PLAN.md` for the full test-ID map and
`PENETRATION_TEST_REPORT.md` for the executed verdicts.
