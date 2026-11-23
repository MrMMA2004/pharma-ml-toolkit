#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 doctor — environment prerequisite check for PharmaML Studio
    development and testing on Windows.

.DESCRIPTION
    Detects missing prerequisites, prints exact tool versions, and reports
    which part15 test lanes are runnable on this machine. Read-only: it
    changes nothing, deletes nothing, and prints no secrets or tokens.

    Lanes and their prerequisites:
      python  (backend tests, api security, persistence)  -> python + venv
      node    (frontend typecheck / vitest / vite build)    -> node + npm
      rust    (cargo fmt / clippy / test, D-4/D-5)         -> cargo + rustc
      tauri   (tauri build checks)                         -> cargo + node
      git     (repository state checks)                     -> git

.OUTPUTS
    A consolidated table of tool -> version/missing, plus the runnable-lane
    verdict. Exit codes: 0 = all core prerequisites present; 1 = one or
    more core prerequisites missing (runnable lanes listed); 2 = blocked
    (could not determine the repository layout).

.NOTES
    No administrator privileges are required.
#>
[CmdletBinding()]
param(
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\logs\windows-tests")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ToolVersion {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { return $null }
    try {
        & $Name --version 2>$null | Select-Object -First 1
    } catch {
        try { (& $Name -V) | Select-Object -First 1 } catch { "<present, version unavailable>" }
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir ("doctor-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $log -ErrorAction SilentlyContinue | Out-Null

$toolkitRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Write-Host "Repository toolkit root: $toolkitRoot"
if (-not (Test-Path (Join-Path $toolkitRoot "src\pharmaml"))) {
    Write-Host "BLOCKED: cannot locate the toolkit tree from this script's location."
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 2
}

$tools = [ordered]@{
    "python" = "python"
    "node"   = "node"
    "npm"    = "npm"
    "git"    = "git"
    "cargo"  = "cargo"
    "rustc"  = "rustc"
}

$versions = [ordered]@{}
$missing  = [System.Collections.Generic.List[string]]::new()
foreach ($name in $tools.Keys) {
    $v = Get-ToolVersion -Name $tools[$name]
    $versions[$name] = if ($null -ne $v) { $v } else { "MISSING" }
    if ($null -eq $v) { $missing.Add($name) }
}

Write-Host ""
Write-Host "== Tool versions =="
foreach ($name in $versions.Keys) {
    Write-Host ("  {0,-8} {1}" -f $name, $versions[$name])
}

Write-Host ""
Write-Host "== Runnable lanes =="
$lane = {
    param($label, $needs)
    $ok = $true
    foreach ($n in $needs) { if ($versions[$n] -eq "MISSING") { $ok = $false } }
    if ($ok) { Write-Host ("  {0,-24} RUNNABLE" -f $label); $true }
    else      { Write-Host ("  {0,-24} BLOCKED (missing: {1})" -f $label, (($needs | Where-Object { $versions[$_] -eq "MISSING" }) -join ", ")); $false }
}
$null = & $lane "backend (python)"     @("python")
$null = & $lane "frontend (node)"      @("node", "npm")
$null = & $lane "rust (D-4/D-5)"       @("cargo", "rustc")
$null = & $lane "tauri (build checks)" @("cargo", "rustc", "node")

Write-Host ""
Write-Host ("Log: {0}" -f $log)
if ($missing.Count -eq 0) {
    Write-Host "Doctor verdict: PASS — all core prerequisites present."
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 0
} else {
    Write-Host ("Doctor verdict: FAIL — missing prerequisites: {0}" -f ($missing -join ", "))
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 1
}
