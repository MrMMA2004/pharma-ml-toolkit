#Requires -Version 5.1
<#
.SYNOPSIS
    part15 SS11 test-api-security — the API security suite on Windows.

.DESCRIPTION
    Runs the focused security regressions over the real create_app():
    the session-security/loopback-guard suite (test_backend_service.py),
    the trusted-origin REST matrix (test_trusted_origins.py — the D-1/D-2
    pins), and the WebSocket handshake Origin policy (test_ws_origin.py —
    the part15 D-6 pins).

    Optional live-lane probes against a RUNNING local backend
    (-BaseUrl http://127.0.0.1:<port> -Token <session token>): without
    them the suite is the offline test-client lane. The token, when
    supplied, is used only in-memory for the probe request and is never
    printed or logged.

.OUTPUTS
    Exit codes: 0 = pass; 1 = failure; 2 = blocked (python missing).

.NOTES
    No administrator privileges are required.
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$BaseUrl = "",
    [string]$Token = "",
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
$log = Join-Path $LogDir ("test-api-security-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $log -ErrorAction SilentlyContinue | Out-Null

try {
    Write-Host ""; Write-Host "== security suites (test-client lane) =="
    & $Python -m pytest `
        tests/api/test_backend_service.py `
        tests/api/test_trusted_origins.py `
        tests/api/test_ws_origin.py `
        -q -rs 2>&1 | Select-Object -Last 8
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: the security suites"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
    Write-Host "PASS: security suites (session/loopback guard, REST trusted origins, WS origin policy)"

    # Windows-PowerShell-5.1-compatible status probe: 4xx/5xx raise,
    # so read the code from the caught HttpWebResponse.
    function Get-Status {
        param([string]$Url, [hashtable]$Headers)
        try {
            $r = Invoke-WebRequest -Uri $Url -Method Get -Headers $Headers -UseBasicParsing
            return [int]$r.StatusCode
        } catch {
            $resp = $_.Exception.Response
            if ($resp) { return [int]$resp.StatusCode }
            return -1
        }
    }

    if ($BaseUrl -and $Token) {
        Write-Host ""; Write-Host "== live probe against $BaseUrl =="
        # 403 lane: a hostile Origin must be refused even before token concerns
        $code = Get-Status -Url "$BaseUrl/api/v1/health" -Headers @{ "Origin" = "https://evil.example" }
        if ($code -ne 403) { Write-Host "FAILED: hostile-origin probe returned $code (expected 403)"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
        Write-Host "  hostile-origin probe: 403 as expected"

        # 401 lane: trusted origin, NO token
        $code = Get-Status -Url "$BaseUrl/api/v1/health" -Headers @{ "Origin" = "http://tauri.localhost" }
        if ($code -ne 401) { Write-Host "FAILED: no-token probe returned $code (expected 401)"; Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
        Write-Host "  no-token probe: 401 as expected"

        # 200 lane: trusted origin + token
        $code = Get-Status -Url "$BaseUrl/api/v1/health" -Headers @{ "Origin" = "http://tauri.localhost"; "X-Session-Token" = $Token }
        if ($code -ne 200) { Write-Host ("FAILED: trusted+token probe returned {0}" -f $code); Stop-Transcript -ErrorAction SilentlyContinue | Out-Null; exit 1 }
        Write-Host "  trusted+token probe: 200 as expected"
    } else {
        Write-Host "(no -BaseUrl/-Token supplied: live probes skipped — offline suite only)"
    }

    Write-Host ""
    Write-Host "test-api-security verdict: PASS. Log: $log"
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 0
} catch {
    Write-Host ("FAILED with an unexpected error: {0}" -f $_.Exception.Message)
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 1
}
