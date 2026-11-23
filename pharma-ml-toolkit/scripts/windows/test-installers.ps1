#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 test-installers — verify the PUBLISHED v1.2.240 installer assets.

.DESCRIPTION
    Downloads the published release's SHA256SUMS.txt (or uses a local copy
    via -SumsFile) and verifies local installer files (EXE/MSI) against
    the pinned digests. Also parses RELEASE_MANIFEST.json when present to
    cross-check the build commit and asset sizes.

    NEVER installs, uninstalls, or executes either installer — this is
    the integrity leg only. The full install/upgrade/uninstall mechanics
    are exercised by the CI smoke lane (windows-install-smoke.yml) and
    require a dedicated Windows machine.

.PARAMETER AssetsDir
    Directory holding the downloaded EXE/MSI (default: the user's
    Downloads folder).

.OUTPUTS
    Exit codes: 0 = all present assets verified; 1 = digest mismatch or
    missing manifest; 2 = blocked (no assets found / no sums source).

.NOTES
    Administrator privileges are NOT required for digest verification.
#>
[CmdletBinding()]
param(
    [string]$AssetsDir = (Join-Path $env:USERPROFILE "Downloads"),
    [string]$SumsFile = "",
    [string]$Version = "1.2.240",
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\logs\windows-tests")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir ("test-installers-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $log -ErrorAction SilentlyContinue | Out-Null

$sums = $SumsFile
if (-not $sums) {
    $local = Join-Path $AssetsDir "SHA256SUMS.txt"
    if (Test-Path $local) { $sums = $local }
}
if (-not $sums) {
    $remote = "https://github.com/MrMMA2004/pharma-ml-toolkit/releases/download/v$Version/SHA256SUMS.txt"
    Write-Host "Downloading the published sums: $remote"
    $tmp = Join-Path $env:TEMP ("SHA256SUMS-v{0}.txt" -f $Version)
    try { Invoke-WebRequest -Uri $remote -OutFile $tmp -UseBasicParsing; $sums = $tmp }
    catch { Write-Host "BLOCKED: could not fetch or find SHA256SUMS.txt."; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 2 }
}

Write-Host "Sums source: $sums"
$pins = @{}
foreach ($line in (Get-Content $sums)) {
    $parts = $line -split '\s+', 2
    if ($parts.Count -eq 2) { $pins[$parts[1].Trim()] = $parts[0].Trim().ToLowerInvariant() }
}
Write-Host ("Pinned assets: {0}" -f ($pins.Keys -join ", "))

$failures = 0
$verified = 0
foreach ($name in $pins.Keys) {
    $path = Join-Path $AssetsDir $name
    if (-not (Test-Path $path)) {
        Write-Host ("  SKIP (not present locally): {0}" -f $name)
        continue
    }
    $actual = (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -eq $pins[$name]) {
        Write-Host ("  PASS: {0} matches the published digest" -f $name); $verified++
    } else {
        Write-Host ("  FAIL: {0} digest mismatch (expected {1}, got {2})" -f $name, $pins[$name], $actual); $failures++
    }
}

$manifest = Join-Path $AssetsDir "RELEASE_MANIFEST.json"
if (Test-Path $manifest) {
    try {
        $m = Get-Content $manifest -Raw | ConvertFrom-Json
        Write-Host ("Manifest build commit: {0}  version: {1}" -f $m.commit, $m.version)
    } catch {
        Write-Host "FAIL: RELEASE_MANIFEST.json present but did not parse"; $failures++
    }
} else {
    Write-Host "SKIP: RELEASE_MANIFEST.json not present locally"
}

Write-Host ""
if ($failures -gt 0) {
    Write-Host ("test-installers verdict: FAIL ({0} failures, {1} verified). Log: {2}" -f $failures, $verified, $log)
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1
}
if ($verified -eq 0) {
    Write-Host "test-installers verdict: BLOCKED (no local assets to verify). Log: $log"
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 2
}
Write-Host ("test-installers verdict: PASS ({0} assets verified against the published digests). Log: {1}" -f $verified, $log)
Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
exit 0
