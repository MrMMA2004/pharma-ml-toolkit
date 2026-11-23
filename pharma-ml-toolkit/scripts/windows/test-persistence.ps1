#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 test-persistence — SQLite/FileVault persistence suite on Windows.

.DESCRIPTION
    Runs the persistence regression suites (tests/api/test_persistence.py,
    tests/api/test_persistence_database.py) over the real backend's
    isolated temp-directory databases — never the user's app data.
    Then exercises the restart-recovery contract: create -> restart
    (a fresh create_app() over the same directory) -> recover.

    The script never deletes or touches any existing user database; all
    state lives under a per-run temp directory that the tests own.

.OUTPUTS
    Exit codes: 0 = pass; 1 = failure; 2 = blocked (python missing).

.NOTES
    No administrator privileges are required.
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\logs\windows-tests")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Host "BLOCKED: python not found on PATH."; exit 2
}
$toolkitRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $toolkitRoot
Write-Host "Working directory: $toolkitRoot"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir ("test-persistence-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $log -ErrorAction SilentlyContinue | Out-Null

try {
    Write-Host ""; Write-Host "== persistence suites (isolated temp DBs) =="
    & $Python -m pytest tests/api/test_persistence.py tests/api/test_persistence_database.py -q -rs 2>&1 | Select-Object -Last 8
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: persistence suites"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
    Write-Host "PASS: persistence suites (schema, transactions, WAL, recovery, restart)"

    Write-Host ""
    Write-Host "test-persistence verdict: PASS. Log: $log"
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 0
} catch {
    Write-Host ("FAILED with an unexpected error: {0}" -f $_.Exception.Message)
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 1
}
