# SECURITY AUDIT — part15 §13 deliverable #4

The part15 Phase-Twelve security and supply-chain review over the
commit `0132d409` (round r340). The executed-lane evidence lives in
`tests/test_security_supply_chain.py` (8/8 green this round),
`tests/api/test_trusted_origins.py` (34 legs), `tests/api/test_ws_origin.py`
(12/12, r339) and the CI security lanes; the prior part14 report
(`SECURITY_TEST_REPORT.md`) stands as the historical record this layer
supersedes-and-extends.

## 1. Transport and origin posture (the D-1..D-6 closure)

| Control | Policy | Evidence |
| --- | --- | --- |
| Binding | loopback-only; never 0.0.0.0 (§4.2's never-list) | server boot lanes |
| Host validation | loopback host policy | security lanes |
| REST Origin/Referer | scheme-strict, host-exact `TRUSTED_APP_ORIGINS` (`http(s)://tauri.localhost`, loopback-any-port); lookalike subdomains/suffixes/ports rejected | `test_trusted_origins.py` 34 legs |
| CORS | scoped middleware: trusted origins only; exact-origin echo; `allow_credentials=False`; arbitrary origins receive NO permissive response | P-lanes (r327/r329/r339) |
| OPTIONS preflight | 405 → 200 for trusted origins; the auth lanes never authenticate | preflight legs |
| REST auth | 401 unauthenticated; 403 untrusted origin (valid token included) | the 34-leg matrix |
| WS auth (token) | close 4401 unauthenticated | events lanes |
| WS Origin (D-6, r339) | present Origin/Referer held to the same predicate pre-accept; untrusted → close 4403 BEFORE accept; absent Origin = the non-browser lane | `test_ws_origin.py` 12/12 FAIL-FIRST |
| Filesystem | traversal/null-byte/absolute-path rejection; upload size/type validation | FS lanes |
| Secrets | D-4 redaction (Rust source, compile-blocked); strict pattern scan over the shipped tree zero-hit | supply-chain suite |

The incident history (tauri.localhost rejected / 405 preflight /
misclassified startup errors) is closed with regression evidence at
D-1/D-2/D-3; the defense-in-depth core (a leaked token must not open the
WS stream) is pinned at the D-6 refusal lanes.

## r357 dated addendum (part17 §4 — the penetration-testing program)

This audit (r340) stands as the historical record it is; the part17
round-357 lane supersedes-and-extends it with a full application
penetration-testing program: the focused automated security suite
`tests/api/test_penetration_security.py` (67 OWASP-mapped tests over
the real `create_app()` stack — the token battery, the
Origin/Referer bypass complements, the CORS complement, the
IDOR/BOLA/BFLA 404 lanes, the vault traversal/symlink class, the
restore-bundle ZIP-slip class, the upload attack shapes, the unsafe
export names, the resource caps, the injection/XSS lanes, the log
hygiene, the desktop CSP + capability static pins, the handshake
shape, and the ZAP tooling pins), the four program documents
(`THREAT_MODEL.md`, `PENETRATION_TEST_PLAN.md`,
`PENETRATION_TEST_REPORT.md`, `SECURITY_REMEDIATION_REGISTER.md`),
and the isolated ZAP workflow lane (`.github/workflows/
penetration-test.yml` + `tools/zap_target.py` + `tools/zap_gate.py`
+ `packaging/security/`). The round's real find F-356-01 (the
artifact-filename Content-Disposition injection) is FIXED+PINNED;
F-356-02 (the root ci.yml `branches: ain]` cosmetic trigger defect
— pushes to main still fire CI, verified via the Actions API) is
recorded OPEN in the register. The ZAP workflow stays
NOT_EXECUTED until its first runner firing.

## 2. Supply-chain review (§12)

- **Python dependencies**: `requirements.txt` exact-pinned (the
  active-lines-are-exact-pins guard); numpy 1.26.4 constraint anchor
  with the documented shap metadata-lie workaround (`--no-deps` + real
  runtime deps, `scratch/venv_repair.sh`).
- **npm dependencies**: lockfile-committed for the nested frontend
  (`package-lock.json`, `npm ci` discipline — 210 packages this round);
  the root console's lockfile per its own `package.json`.
- **Rust dependencies**: `src-tauri` manifests lockfile-committed;
  cargo-audit rides the Windows box (no Rust toolchain here).
- **SBOM**: `packaging/sbom/sbom.cdx.json` — valid CycloneDX, covering
  the pinned core stack (the two supply-chain guards).
- **Known vulnerabilities**: `pip-audit --strict` + `npm audit
  --audit-level=high` run in CI on the weekly + dispatch
  `security-scan.yml` lane (network-facing scanning stays CI-side,
  never the runtime — the suite pins the workflow's CI-only posture).
- **Secret scanning**: the strict full-pattern tier over
  src/tests/docs/tools/workflows — zero hits, proven by the tree scan
  guard; the `.credentials/` quarantine guard holds (the token lanes
  stay disk-only, 600-mode, gitignored — the f2607a2 lesson).
- **GitHub Actions permissions**: the four workflows (`ci.yml`,
  `security-scan.yml`, `windows-install-smoke.yml`,
  `windows-release.yml`) carry least-privilege declarations; the
  smoke/release lanes auto-run on published releases only.
- **Build artifact provenance**: v1.2.240's five assets verified
  digest-level (SHA256SUMS + RELEASE_MANIFEST cross-check, the ninth
  smoke dispatch green end-to-end).
- **Tauri capabilities**: the capability/permission audit lane is the
  `test-tauri.ps1` script (SS11) — honest BLOCKED here (no Tauri
  toolchain); no broad filesystem/shell/network permission grants
  without documented requirement were found in the conf inspection.

## 3. Standing rules honored

No token ever in URLs/argv/logs (the askpass + curl-config lanes); owner
uploads (`GitHub_token.*`, `skill.txt`, the MASTER_PROMPT set) stay
disk-only per the gitignore conventions; no security check was disabled
to make a test pass (§18); the blocked legs are named, never faked.

— audited round 340 (the 132nd recorded run) at `0132d409`.
