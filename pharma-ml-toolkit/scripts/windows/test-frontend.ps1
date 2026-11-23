#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 test-frontend — the React/Vite desktop frontend lane on Windows.

.DESCRIPTION
    From pharma-ml-toolkit/frontend/ (the NESTED Vite desktop frontend —
    deliberately NOT the root Next.js web console): npm ci (fresh
    install), TypeScript typecheck, Vitest component/API-service tests,
    and the Vite production build.

    The npm cache is preserved; no destructive reinstall is forced. If
    node_modules already satisfies the lockfile, `npm ci` is the honest
    reproducible lane and is what CI uses too.

.OUTPUTS
    Exit codes: 0 = pass; 1 = failure; 2 = blocked (node/npm missing, or
    the frontend tree not found).

.NOTES
    No administrator privileges are required.
#>
[CmdletBinding()]
param(
    [string]$Npm = "npm",
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\logs\windows-tests")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "BLOCKED: node not found on PATH."; exit 2
}
if ($null -eq (Get-Command $Npm -ErrorAction SilentlyContinue)) {
    Write-Host "BLOCKED: npm not found on PATH."; exit 2
}
$frontendRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\frontend")
Set-Location $frontendRoot
Write-Host "Working directory: $frontendRoot"
Write-Host ("node: {0}" -f (node --version))
Write-Host ("npm:  {0}" -f (npm --version))

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir ("test-frontend-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $log -ErrorAction SilentlyContinue | Out-Null

try {
    Write-Host ""; Write-Host "== npm ci (lockfile-exact install) =="
    & $Npm ci --no-audit --no-fund 2>&1 | Select-Object -Last 4
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: npm ci"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }

    Write-Host ""; Write-Host "== TypeScript typecheck =="
    & $Npm run typecheck 2>&1 | Select-Object -Last 4
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: typecheck (run 'npm run typecheck' for the full output)"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }

    Write-Host ""; Write-Host "== Vitest (component + API-service tests) =="
    & $Npm test -- --run 2>&1 | Select-Object -Last 12
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: vitest"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }

    Write-Host ""; Write-Host "== Vite production build =="
    & $Npm run build 2>&1 | Select-Object -Last 6
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: vite build"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }

    Write-Host ""
    Write-Host "test-frontend verdict: PASS. Log: $log"
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 0
} catch {
    Write-Host ("FAILED with an unexpected error: {0}" -f $_.Exception.Message)
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 1
}
