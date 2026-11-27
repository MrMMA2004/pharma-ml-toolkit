
# RETIRED PACKAGING LANE (r368 / part19 Phase 2): this Inno-Setup path is the
# part9-era LEGACY installer builder for the retired PySide6 GUI app (it froze
# pharmaml.spec -- the legacy GUI spec -- with the Qt stack embedded). The
# supported production path is .github/workflows/windows-release.yml: the
# Qt-free pharmaml_backend.spec sidecar + the Tauri 2 host (NSIS Setup EXE +
# WiX MSI). This script is kept for archival/reproducibility only and is NOT
# part of any supported release workflow.
# PharmaML Studio — the one-command Windows installer build (part9 §16).
#
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
#
# Produces, under release\windows\:
#   PharmaML_Studio_Setup_vX.Y.Z.exe   — the offline installer
#   SHA256SUMS.txt                     — checksums of every release artifact
#   RELEASE_MANIFEST.json              — versions, sizes, tool versions
#   BUILD_REPORT.md                    — the honest build report
#
# Every step fails loudly (nonzero exit) — a half-built installer is
# never presented as success (part9 §16).  Optional switches:
#   -SkipTests   dev-only convenience; the default runs the full pytest
#                gate BEFORE freezing (the same suites the release
#                evidence records).
#   -SkipClean   reuse the build venv (NOT for release builds).

param(
  [switch]$SkipTests,
  [switch]$SkipClean
)

$ErrorActionPreference = 'Stop'
$ErrorView = 'NormalView'   # full errors, even from native commands

# ---- 0. locate ourselves ----------------------------------------------------
$PackDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # ...packaging\windows
$Root    = Resolve-Path (Join-Path $PackDir '..\..')         # toolkit root
$Release = Join-Path $Root 'release\windows'
Set-Location $Root

# ---- 1. verify the source version (part9 §16 step 1) --------------------------
$Version = (Get-Content (Join-Path $Root 'VERSION') -Raw).Trim()
if ($Version -notmatch '^\d+\.\d+\.\d+$') {
  throw "VERSION file does not carry a clean semver: '$Version'"
}
Write-Host "== PharmaML Studio $Version — Windows installer build =="
Write-Host "== toolkit root: $Root"

# ---- 2. validate the Python interpreter (x64, 3.12) --------------------------
$Py = Get-Command python -ErrorAction SilentlyContinue
if (-not $Py) { $Py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $Py) { throw 'python/py not on PATH — install Python 3.12 x64 first' }
$PyVer  = & $Py.Source --version
$PyArch = & $Py.Source -c "import struct; print(struct.calcsize('P')*8)"
if ($PyVer -notmatch '^Python 3\.12') {
  throw "build requires Python 3.12 (the pinned stack's QA family); found: $PyVer"
}
if ($PyArch -ne '64') { throw "build requires 64-bit Python; found: ${PyArch}-bit" }
Write-Host "== interpreter: $PyVer ($PyArch-bit)"

# ---- 3. clean-build environment (part9 §10) ----------------------------------
$Venv = Join-Path $PackDir 'build-venv'
if ((-not $SkipClean) -and (Test-Path $Venv)) {
  Remove-Item -Recurse -Force $Venv | Out-Null
}
if (-not (Test-Path (Join-Path $Venv 'Scripts\python.exe'))) {
  & $Py.Source -m venv $Venv
  if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$VenvPython = Join-Path $Venv 'Scripts\python.exe'

# ---- 4. dependencies: torch CPU wheel, then the pinned stack -----------------
# (probe with Continue so a failed torch import writes no terminating
#  ErrorRecord under $ErrorActionPreference='Stop')
$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$TorchOk = & $VenvPython -c "import torch; print(torch.__version__)" 2>$null
$ErrorActionPreference = $prev
if ("$TorchOk" -notmatch '^\d+\.\d+\.\d+\+cpu$') {
  & $VenvPython -m pip install --disable-pip-version-check `
      torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
  if ($LASTEXITCODE -ne 0) { throw 'torch CPU wheel install failed' }
}
& $VenvPython -m pip install --disable-pip-version-check `
    -r (Join-Path $PackDir 'installer-requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'pinned dependency install failed' }

# the toolkit itself (frozen from source; --no-deps so the pins above rule):
& $VenvPython -m pip install --disable-pip-version-check --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'pharmaml editable install failed' }

# ---- 5. pre-build tests (part9 §16 step 4) -----------------------------------
if (-not $SkipTests) {
  Write-Host "== running the full pytest gate (the release evidence suites)..."
  & $VenvPython -m pytest tests -q -W ignore -p no:cacheprovider
  if ($LASTEXITCODE -ne 0) { throw "pre-build test gate FAILED (rc=$LASTEXITCODE)" }
}

# ---- 6. clean old artifacts (part9 §16 step 5) -------------------------------
foreach ($d in @((Join-Path $Release 'dist'), (Join-Path $Release 'build'))) {
  if (Test-Path $d) { Remove-Item -Recurse -Force $d | Out-Null }
}
New-Item -ItemType Directory -Force -Path $Release | Out-Null

# ---- 7. freeze (part9 §16 step 6) --------------------------------------------
& $VenvPython -m PyInstaller (Join-Path $PackDir 'pharmaml.spec') `
    --noconfirm `
    --distpath (Join-Path $Release 'dist') `
    --workpath (Join-Path $Release 'build')
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (rc=$LASTEXITCODE)" }

# ---- 8. packaging validation (part9 §16 steps 8-9) ----------------------------
& $VenvPython (Join-Path $PackDir 'packaging_validation.py') `
    --dist (Join-Path $Release 'dist\PharmaMLStudio') --frozen-smoke
if ($LASTEXITCODE -ne 0) { throw "packaging validation FAILED (rc=$LASTEXITCODE)" }

# collect the third-party license inventory into the release dir:
& $VenvPython (Join-Path $Root 'tools\packaging_audit.py') `
    --licenses-dir (Join-Path $Release 'THIRD_PARTY_LICENSES')
if ($LASTEXITCODE -ne 0) { throw 'license inventory generation failed' }

# ---- 9. the installer (part9 §16 step 7) --------------------------------------
$Iscc = $env:INNO_SETUP
if (-not $Iscc -or -not (Test-Path $Iscc)) {
  $Iscc = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
}
if (-not (Test-Path $Iscc)) {
  $hit = Get-ChildItem 'C:\Program Files*' -Recurse -Filter ISCC.exe `
         -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($hit) { $Iscc = $hit.FullName }
}
if (-not (Test-Path $Iscc)) {
  throw 'Inno Setup 6 not found — install it (or set INNO_SETUP to ISCC.exe)'
}
Write-Host "== compiling the installer with: $Iscc"
& $Iscc "/DMyAppVersion=$Version" (Join-Path $PackDir 'installer.iss')
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed (rc=$LASTEXITCODE)" }

# ---- 10. checksums + manifest + report (part9 §16 steps 10-11) ---------------
$SetupExe = Join-Path $Release "PharmaML_Studio_Setup_v$Version.exe"
if (-not (Test-Path $SetupExe)) { throw "installer EXE missing: $SetupExe" }

& $VenvPython (Join-Path $Root 'tools\packaging_audit.py') `
    --manifest $SetupExe --release-dir $Release --python $VenvPython
if ($LASTEXITCODE -ne 0) { throw 'RELEASE_MANIFEST/BUILD_REPORT generation failed' }

Write-Host ''
Write-Host "== BUILD COMPLETE: $SetupExe"
Write-Host "== next: clean-machine test per packaging\windows\README.md (part9 §14)"
exit 0
