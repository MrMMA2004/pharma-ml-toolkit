#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 test-tauri — Tauri 2 build checks and capability audit on Windows.

.DESCRIPTION
    From pharma-ml-toolkit/src-tauri/: runs `cargo check` (the fastest
    compile gate), validates the tauri.conf.json JSON parses, and audits
    the capability files under capabilities/ (names + the permissions they
    grant are listed for human review — no broad fs/shell/http scope
    should appear there without a documented requirement).

    This script does NOT build installers and does NOT publish
    anything — the release lanes stay gated on the owner.

.OUTPUTS
    Exit codes: 0 = pass; 1 = failure; 2 = blocked (cargo missing or the
    src-tauri tree absent).

.NOTES
    No administrator privileges are required for cargo check. Full
    `tauri build` runs (NSIS/MSI generation) need the WebView2 SDK
    presence via the build tools but NOT admin rights by default.
#>
[CmdletBinding()]
param(
    [string]$Cargo = "cargo",
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\logs\windows-tests")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq (Get-Command $Cargo -ErrorAction SilentlyContinue)) {
    Write-Host "BLOCKED: cargo not found on PATH."; exit 2
}
$srcTauri = Resolve-Path (Join-Path $PSScriptRoot "..\..\src-tauri")
Set-Location $srcTauri
Write-Host "Working directory: $srcTauri"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir ("test-tauri-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $log -ErrorAction SilentlyContinue | Out-Null

try {
    Write-Host ""; Write-Host "== tauri.conf.json parses =="
    $conf = Get-Content "tauri.conf.json" -Raw | ConvertFrom-Json
    Write-Host ("identifier: {0}" -f $conf.identifier)
    if (-not $conf.identifier) { Write-Host "FAILED: identifier missing in tauri.conf.json"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
    Write-Host "PASS: tauri.conf.json parses"

    Write-Host ""; Write-Host "== capability files =="
    $caps = Get-ChildItem "capabilities" -Filter "*.json" -ErrorAction SilentlyContinue
    if ($null -eq $caps -or $caps.Count -eq 0) {
        Write-Host "FAILED: no capability files found under capabilities/"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1
    }
    foreach ($c in $caps) {
        $json = Get-Content $c.FullName -Raw | ConvertFrom-Json
        $perms = $json.permissions -join ", "
        Write-Host ("  {0}: permissions = {1}" -f $c.Name, $perms)
        if ($perms -match "(?i)fs:default|shell:default|http:default") {
            Write-Host "  NOTE: review broad default permissions against the documented requirement (part15 SS12)."
        }
    }
    Write-Host "PASS: capabilities audited (permissions listed above for review)"

    Write-Host ""; Write-Host "== cargo check (compile gate) =="
    & $Cargo check 2>&1 | Select-Object -Last 6
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: cargo check"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
    Write-Host "PASS: cargo check"

    Write-Host ""
    Write-Host "test-tauri verdict: PASS. Log: $log"
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 0
} catch {
    Write-Host ("FAILED with an unexpected error: {0}" -f $_.Exception.Message)
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 1
}
