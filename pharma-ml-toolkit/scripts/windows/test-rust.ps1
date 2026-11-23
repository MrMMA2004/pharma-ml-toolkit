#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 test-rust — the BLOCKED D-4/D-5 lanes' carrier on Windows.

.DESCRIPTION
    Runs the Rust/Tauri host's quality gates from pharma-ml-toolkit/
    src-tauri/: `cargo fmt --check`, `cargo clippy` (the pinned profile,
    warnings-as-failures off — clippy's own exit code decides), and
    `cargo test` (which executes the D-4/D-5 #[cfg(test)] suites:
    the redacted-handshake parsing tests and the validate_handshake
    field-validation cases).

    These legs are exactly the ones the sandbox could not run (no Rust
    toolchain in that environment — the defect register's D-4/D-5
    "compile-blocked" class). Run this script on any Windows machine
    with the Rust toolchain to convert those rows to plain FIXED with
    executed evidence. Never mark them passed without this run.

.OUTPUTS
    Exit codes: 0 = pass (fmt clean, clippy clean, tests green);
    1 = failure (the failing step names itself);
    2 = blocked (cargo/rustc missing — install the Rust toolchain from
    https://rustup.rs and retry, or the src-tauri tree is absent).

.NOTES
    No administrator privileges are required. The Rust toolchain link
    target (~\.cargo) does not need elevation on a per-user install.
#>
[CmdletBinding()]
param(
    [string]$Cargo = "cargo",
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\logs\windows-tests")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq (Get-Command $Cargo -ErrorAction SilentlyContinue)) {
    Write-Host "BLOCKED: cargo not found on PATH (the D-4/D-5 prerequisite)."
    Write-Host "Install the Rust toolchain (https://rustup.rs) and re-run this script."
    exit 2
}
try { Write-Host ("cargo: {0}" -f (& cargo --version)) } catch {}
try { Write-Host ("rustc: {0}" -f (& rustc --version)) } catch {}

$srcTauri = Resolve-Path (Join-Path $PSScriptRoot "..\..\src-tauri")
Set-Location $srcTauri
Write-Host "Working directory: $srcTauri"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir ("test-rust-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $log -ErrorAction SilentlyContinue | Out-Null

try {
    Write-Host ""; Write-Host "== cargo fmt --check =="
    & $Cargo fmt --check 2>&1 | Select-Object -Last 4
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: cargo fmt --check (run 'cargo fmt' to fix)"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
    Write-Host "PASS: cargo fmt --check"

    Write-Host ""; Write-Host "== cargo clippy =="
    & $Cargo clippy 2>&1 | Select-Object -Last 6
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: cargo clippy"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
    Write-Host "PASS: cargo clippy"

    Write-Host ""; Write-Host "== cargo test (the D-4/D-5 #[cfg(test)] suites) =="
    & $Cargo test 2>&1 | Select-Object -Last 12
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: cargo test — the D-4/D-5 rows stay compile-blocked"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
    Write-Host "PASS: cargo test — record this run as the D-4/D-5 executed evidence"

    Write-Host ""
    Write-Host "test-rust verdict: PASS (fmt + clippy + test). Log: $log"
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 0
} catch {
    Write-Host ("FAILED with an unexpected error: {0}" -f $_.Exception.Message)
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 1
}
