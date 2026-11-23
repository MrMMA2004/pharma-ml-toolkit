#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 test-backend — run the Python backend test suite on Windows.

.DESCRIPTION
    Runs the PharmaML backend test batteries from the correct working
    directory (pharma-ml-toolkit/): pytest collection, the focused API
    suites (backend service, trusted origins, WS origin, contracts),
    ruff, and mypy over src/. Optionally the full suite with -Full.

    Skipped tests are recorded with reasons (pytest -rs); the script
    never deletes data, never touches the user's app data, and prints
    no tokens.

.PARAMETER Full
    Run the full pytest suite (longer; includes the scientific suite).
    Default runs the API + security-focused baseline.

.OUTPUTS
    Exit codes: 0 = pass; 1 = test failure; 2 = blocked (python or the
    toolkit tree missing; install Python 3.11+ and the pinned
    requirements.txt into your venv first).

.NOTES
    No administrator privileges are required.
#>
[CmdletBinding()]
param(
    [switch]$Full,
    [string]$Python = "python",
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\logs\windows-tests")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Host "BLOCKED: python not found on PATH. Install Python 3.11+ and retry."
    exit 2
}
$toolkitRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $toolkitRoot
Write-Host "Working directory: $toolkitRoot"
Write-Host ("Python: {0}" -f (& $Python --version 2>&1 | Select-Object -First 1))

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $LogDir ("test-backend-{0}.log" -f $stamp)
Start-Transcript -Path $log -ErrorAction SilentlyContinue | Out-Null

function Invoke-Step {
    param([string]$Label, [scriptblock]$Body)
    Write-Host ""
    Write-Host "== $Label =="
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("FAILED: {0} (exit {1})" -f $Label, $LASTEXITCODE)
        Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
        exit 1
    }
    Write-Host "PASS: $Label"
}

try {
    Invoke-Step "pytest collection census" {
        & $Python -m pytest tests --collect-only -q 2>&1 | Select-Object -Last 5
    }

    $targets = if ($Full) { "tests" } else {
        "tests/api/test_backend_service.py", "tests/api/test_trusted_origins.py",
        "tests/api/test_ws_origin.py", "tests/api/test_contracts.py",
        "tests/api/test_persistence.py"
    }
    Invoke-Step ("pytest focused suite" + $(if ($Full) { " (FULL)" } else { "" })) {
        & $Python -m pytest @targets -q -rs 2>&1 | Select-Object -Last 12
    }

    Invoke-Step "ruff (src + tests)" {
        & $Python -m ruff check src tests 2>&1 | Select-Object -Last 3
    }

    Invoke-Step "mypy (src)" {
        & $Python -m mypy src 2>&1 | Select-Object -Last 3
    }

    Write-Host ""
    Write-Host "test-backend verdict: PASS (all steps green). Log: $log"
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 0
} catch {
    Write-Host ("FAILED with an unexpected error: {0}" -f $_.Exception.Message)
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 1
}
