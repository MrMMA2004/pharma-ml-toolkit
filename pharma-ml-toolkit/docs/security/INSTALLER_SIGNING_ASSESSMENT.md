# Installer Digital-Signing Assessment (part13 Phase 2, Blocker C)

Part13 §4 "Blocker C — Digital signing" evaluation deliverable:
"Digital signing requires an appropriate certificate. Evaluate:
Authenticode signing, Timestamping, Certificate storage in GitHub
Actions secrets, Signature verification, Release provenance. Do not
store certificates or private signing keys in the repository. If no
certificate is available, keep the gate BLOCKED and document the exact
requirement."

- **Assessed**: 2026-11-21, round 309 (the 99th→100th recorded-run
  session, the interactive part13 Phase-2 continuation).
- **Audited base**: `origin/main` = `19effec8` (the round-308 close-out).
- **Verdict**: **BLOCKED — certificate-gated** (no certificate exists in
  this environment, none may be fabricated). Everything below documents
  the exact requirement and the staged design so the gate opens the day
  a certificate arrives; nothing here claims signing is done.

## 1. The exact requirement

| Element | Requirement |
| --- | --- |
| Certificate type | Authenticode **code-signing certificate** (OV or EV) for `MrMMA2004` / the PharmaML Studio publishing identity, issued by a public CA on the Microsoft root-of-trust chain (e.g. DigiCert, Sectigo, GlobalSign, SSL.com). EV additionally requires a hardware token (USB/HSM) per CA/Browser Forum rules — it removes the SmartScreen "unknown publisher" reputation lag but does not change the engineering below. |
| What gets signed | Both Windows installers: `PharmaML_Studio_Setup_v{V}.exe` (NSIS) and `PharmaML_Studio_v{V}.mssi` (WiX) — plus, optionally, the frozen sidecar `pharmaml-backend.exe` for a deeper provenance chain. |
| Signing scheme | Authenticode (PKCS #7) via `signtool sign /fd SHA-256 /tr <TSA> /td SHA-256`; both the file digest AND the timestamp digest are SHA-256. Legacy /t RFC-3161-less timestamping is not acceptable (single-TSA SHA-1 dependency). |
| Timestamping | RFC 3161 TSA (e.g. DigiCert `http://timestamp.digicert.com`, Sectigo `http://timestamp.sectigo.com`) — mandatory so signatures outlive certificate expiry. |
| Private-key custody | The PFX / token NEVER enters the repository (this is the standing PAT-exposure rule, one class harder: a signing key can mint trusted binaries). Storage is GitHub Actions **encrypted secrets** (repo → Settings → Secrets → Actions): the base64 PFX as one secret, its passphrase as a separate secret; or Azure Trusted Signing / a cloud HSM identity for the EV path. Secrets are masked in logs, scoped to the `windows-release` workflow, and rotatable without a tree change. |
| Signature verification | `signtool verify /pa /all` (default Authenticode policy, all nested signatures) in the release workflow, plus `Get-AuthenticodeSignature` checks in the clean-machine smoke protocol; the signature's `SignerCertificate` subject must match the publishing identity. |
| Release provenance | SHA256SUMS.txt + RELEASE_MANIFEST.json (already shipped) gain the signature's thumbprint and the TSA timestamp so the release page carries an independently recomputable chain: installer bytes → SHA-256 → signature → certificate → timestamp. |
| Repo invariant | `.github/workflows/windows-release.yml` gains the signing step **conditioned on the secrets being present** (`if: github.event_name == 'push' && vars.SIGNING_ENABLED == 'true'`); with the secrets absent the step self-reports SKIP and the release proceeds UNSIGNED-BLOCKED exactly as today — the lane must never fail the build for lacking a certificate (the PDF's own rule), and must never silently pretend. |

## 2. Current state (verified in-tree this round)

- **No signing step exists** in `windows-release.yml` (grep: zero
  `signtool`/`sign` steps) — honest: the v1.2.231 release EXE/MSI are
  UNSIGNED and say so implicitly via the SmartScreen "unknown
  publisher" path a clean machine will hit (the clean-install ladder's
  row context).
- **No certificate material exists anywhere in the repository** —
  verified this round by the ship ritual's leak scan (token shapes +
  absolute paths) and by direct grep for `signtool`/`.pfx`/`.p12`: zero
  hits. The PAT-exposure remediation (docs/security/
  PAT_EXPOSURE_REMEDIATION.md) is the standing proof that this project
  has already fought and won one credential-hygiene battle; the signing
  key inherits the same rules.
- **The provenance half is already real**: the release run generates
  SHA256SUMS.txt over every artifact (step 22) + RELEASE_MANIFEST.json
  (step 21) + the SBOM with its hash self-registered into SHA256SUMS
  (step 22b). Signing plugs into an existing, verified provenance
  spine — it does not have to build one.

## 3. The staged design (what the lane looks like the day the cert arrives)

```yaml
# windows-release.yml — the conditional sign leg (design only, NOT
# landed: no certificate exists to gate it on)
- name: "22c. Authenticode-sign the installers (certificate-gated)"
  if: vars.SIGNING_ENABLED == 'true'
  env:
    PFX_B64: ${{ secrets.CODESIGN_PFX_B64 }}   # base64 PFX
    PFX_PASS: ${{ secrets.CODESIGN_PFX_PASS }} # its passphrase
  shell: pwsh
  run: |
    [IO.File]::WriteAllBytes("$env:RUNNER_TEMP\cert.pfx",
      [Convert]::FromBase64String($env:PFX_B64))
    signtool sign /f "$env:RUNNER_TEMP\cert.pfx" /p "$env:PFX_PASS" `
      /fd SHA-256 /tr http://timestamp.digicert.com /td SHA-256 `
      "$nsis" "$msi"
    Remove-Item "$env:RUNNER_TEMP\cert.pfx" -Force
    signtool verify /pa /all "$nsis"; signtool verify /pa /all "$msi"
```

The EV/hardware-token variant swaps the PFX block for Azure Trusted
Signing (or a network HSM session) — the signtool invocation and the
verification line stay identical.

## 4. What unblocks it

An Authenticode code-signing certificate issued to the publishing
identity, delivered either as an exportable PFX (OV) or a hardware
token / Trusted-Signing identity (EV). Cost is the owner's call (OV
roughly tens of USD/yr, EV hundreds). The moment the secret exists in
the repo's Actions secrets, the lane above lands, the clean-machine
smoke ladder gains "signature verifies" as a row, and this gate moves
from BLOCKED to PASS with the thumbprint recorded in
RELEASE_MANIFEST.json.

Until then, every honest close repeats: **installer signing —
BLOCKED, certificate-gated; the requirement above is the exact
specification.**
