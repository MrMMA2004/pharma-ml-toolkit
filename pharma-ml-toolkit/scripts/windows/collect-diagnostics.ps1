#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 collect-diagnostics — consolidated diagnostic report.

.DESCRIPTION
    Runs doctor.ps1 and the environment-accessible test scripts in order,
    captures each verdict, and produces ONE consolidated report (screen +
    logs\windows-tests\diagnostics-<stamp>.log). Skipped/blocked lanes
    are labeled BLOCKED with their missing prerequisite — never as
    failures, never as passes.

    Order: doctor, test-backend, test-api-security, test-persistence,
    test-frontend, test-rust, test-tauri, test-installers.

    Destructive actions: none. Tokens/secrets: none printed. Every
    child script writes its own log beside this report.

.OUTPUTS
    Exit codes: 0 = every runnable lane PASS; 1 = at least one runnable
    lane FAILED; 2 = everything blocked (see the report for why).

.NOTES
    No administrator privileges are required for any lane. The
    installer-mechanics lane (install/upgrade/uninstall on real Windows)
    is deliberately NOT here — it lives in the CI smoke lane and on the
    owner's physical machines, per part15 SS6/SS13's blocked-test rules.
#>
[CmdletBinding()]
param(
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\logs\windows-tests")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$report = Join-Path $LogDir ("diagnostics-{0}.log" -f $stamp)
Start-Transcript -Path $report -ErrorAction SilentlyContinue | Out-Null

Write-Host "PharmaML Studio — part15 SS11 consolidated diagnostics"
Write-Host ("Started: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host "Lanes: doctor, backend, api-security, persistence, frontend, rust, tauri, installers"
Write-Host ""

$lanes = [ordered]@{
    "doctor"         = @{ script = "doctor.ps1";           args = @() }
    "backend"        = @{ script = "test-backend.ps1";      args = @() }
    "api-security"   = @{ script = "test-api-security.ps1"; args = @() }
    "persistence"    = @{ script = "test-persistence.ps1"; args = @() }
    "frontend"       = @{ script = "test-frontend.ps1";     args = @() }
    "rust"           = @{ script = "test-rust.ps1";        args = @() }
    "tauri"          = @{ script = "test-tauri.ps1";       args = @() }
    "installers"     = @{ script = "test-installers.ps1";  args = @() }
}

$results = [ordered]@{}
foreach ($name in $lanes.Keys) {
    $script = Join-Path $PSScriptRoot $lanes[$name].script
    if (-not (Test-Path $script)) {
        $results[$name] = "MISSING-SCRIPT"; continue
    }
    Write-Host "== lane: $name =="
    & powershell -NoProfile -ExecutionPolicy Bypass -File $script @($lanes[$name].args) | Out-Null
    $code = $LASTEXITCODE
    $label = switch ($code) {
        0 { "PASS" }
        1 { "FAIL" }
        2 { "BLOCKED" }
        default { "UNKNOWN($code)" }
    }
    $results[$name] = $label
    Write-Host ("  verdict: {0}" -f $label)
}

Write-Host ""
Write-Host "== Consolidated verdicts =="
foreach ($name in $results.Keys) {
    Write-Host ("  {0,-14} {1}" -f $name, $results[$name])
}

$fails = @($results.Values | Where-Object { $_ -eq "FAIL" }).Count
$passes = @($results.Values | Where-Object { $_ -eq "PASS" }).Count
$blocked = @($results.Values | Where-Object { $_ -eq "BLOCKED" -or $_ -like "MISSING*" }).Count

Write-Host ""
Write-Host ("Summary: {0} PASS, {1} FAIL, {2} BLOCKED (blocked lanes name their missing prerequisite above)" -f $passes, $fails, $blocked)
Write-Host ("Report: {0}" -f $report)
Stop-Transcript -ErrorAction SilentlyContinue | Out-Null

if ($fails -gt 0) { exit 1 }
if ($passes -eq 0 -and $blocked -gt 0) { exit 2 }
exit 0
