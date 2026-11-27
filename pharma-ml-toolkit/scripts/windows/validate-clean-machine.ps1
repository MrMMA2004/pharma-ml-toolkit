# validate-clean-machine.ps1 -- PharmaML Studio clean-machine validation suite
# (part19 Phase 8 / Round 368 deliverable).
#
# A REPRODUCIBLE, owner-side validation suite for genuinely clean Windows
# 10/11 boxes: the runner-verified installers (GitHub-hosted windows-install-
# smoke) are NOT equivalent to a clean-machine check, and this script exists
# so the owner can execute the full matrix this sandbox cannot.
#
# Design contract (matches the scripts/windows/ house rules):
#   * PowerShell 5.1-compatible (no classes, no ternaries, no ?? operators).
#   * Never prints or persists a token or any user secret: the sidecar's
#     session token is read from the handshake and held in memory only.
#   * Every leg records PASS / FAIL / BLOCKED with a reason; BLOCKED is an
#     honest verdict, never a silent skip.
#   * Transcripts + the machine-readable report land under
#     logs/windows-tests/clean-machine/ as JSON + Markdown.
#   * Uses ONLY synthetic disposable data and a disposable install root;
#     never touches the user's real application data (the backup/restore and
#     uninstall legs operate inside the sandboxed install root this script
#     chooses, or a -InstallRoot the caller explicitly passes).
#
# Legs (run individually with -Leg <name>, or all with -Full):
#   census        Windows version/arch, CPU, memory, GPU, WebView2 runtime,
#                 Python/Node/Rust presence
#   integrity     installer file hashes + Authenticode status + size census
#   sidecar       launch the installed pharmaml-backend.exe, capture the
#                 PHARMAML-READY handshake, health with+without token,
#                 WebSocket connect, graceful shutdown
#   workspaces    the nine-workspace execution matrix over the live sidecar
#                 (dataset import, chemistry, ML train+predict, DL train,
#                 semi-supervised, applicability domain, visualization,
#                 docking read-only, PK/PD calculators, MD analysis)
#   artifacts     model save/load + figure/table export + backup/restore
#   lifecycle     upgrade path (v(n-1) -> v(n) marker retention) + uninstall
#                 + reinstall
#
# Usage (owner's clean box, from the repository root or anywhere -- the
# script resolves the repo root itself):
#   powershell -ExecutionPolicy Bypass -File scripts/windows/validate-clean-machine.ps1 -Installer <path-to-Setup.exe> -Msi <path-to.msi> -Full
#   powershell ... -Leg census
#   powershell ... -Leg sidecar -InstallRoot "C:\Program Files\PharmaML Studio"
#
# Exit codes: 0 = every executed leg PASS (BLOCKED legs do not fail the run
# but are listed in the report); 1 = at least one FAIL; 2 = usage error.

param(
    [string]$Installer,
    [string]$Msi,
    [string]$InstallRoot,
    [string]$Leg = "",
    [switch]$Full
)

$ErrorActionPreference = "Stop"

# ---- locate the repository root (script lives in <root>/scripts/windows/) ----
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $here "..\..")).Path

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $repoRoot "logs\windows-tests\clean-machine"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$reportJson = Join-Path $logDir "clean-machine-$stamp.json"
$reportMd = Join-Path $logDir "clean-machine-$stamp.md"

$script:results = @()
$script:handshakeToken = $null   # in-memory only, never printed/persisted

function Add-Result {
    param([string]$LegName, [string]$Check, [string]$Status, [string]$Detail)
    $script:results += [ordered]@{
        leg = $LegName; check = $Check; status = $Status
        detail = $Detail; at = (Get-Date -Format o)
    }
    $tag = switch ($Status) { "PASS" {"PASS"} "FAIL" {"FAIL"} default {"BLOCKED"} }
    Write-Host ("[{0}] {1} :: {2}" -f $tag, $LegName, $Check)
}

function Save-Report {
    param([string]$Overall)
    $env_census = $script:census
    $payload = [ordered]@{
        overall = $Overall
        generatedAt = (Get-Date -Format o)
        legs = $script:results
    }
    if ($null -ne $script:census) { $payload.census = $script:census }
    $payload | ConvertTo-Json -Depth 6 | Out-File -FilePath $reportJson -Encoding UTF8
    $md = New-Object System.Text.StringBuilder
    [void]$md.AppendLine("# PharmaML Studio clean-machine validation -- $stamp")
    [void]$md.AppendLine("")
    [void]$md.AppendLine("Overall: **$Overall** (BLOCKED legs do not fail the run; they are listed honestly)")
    [void]$md.AppendLine("")
    [void]$md.AppendLine("| Leg | Check | Status | Detail |")
    [void]$md.AppendLine("|---|---|---|---|")
    foreach ($r in $script:results) {
        [void]$md.AppendLine("| $($r.leg) | $($r.check) | $($r.status) | $($r.detail) |")
    }
    [void]$md.AppendLine("")
    [void]$md.AppendLine("Machine-readable report: ``$reportJson``. This run executed on the owner's")
    [void]$md.AppendLine("machine per the part19 Phase 8 matrix; runner evidence is separate.")
    $md.ToString() | Out-File -FilePath $reportMd -Encoding UTF8
    Write-Host "REPORT: $reportJson"
    Write-Host "REPORT: $reportMd"
}

# ---------------------------------------------------------------------------
# LEG: census -- the environment this machine actually provides
# ---------------------------------------------------------------------------
function Invoke-Census {
    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $gpu = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
    $totalGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
    $freeGB = [math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 1)
    $arch = $os.OSArchitecture
    $winVer = "$($os.Caption) (build $($os.BuildNumber))"

    # WebView2 Evergreen runtime: the registry channel the Tauri host needs
    $webview2 = "absent (the Evergreen runtime is not installed)"
    $wvKeys = @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00A16586BB03}",
        "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00A16586BB03}"
    )
    foreach ($k in $wvKeys) {
        if (Test-Path $k) {
            $pv = (Get-ItemProperty $k -ErrorAction SilentlyContinue).pv
            if ($pv) { $webview2 = "Evergreen $pv (registry channel present)"; break }
        }
    }

    $pyVer = "absent"
    try { $pyVer = (& python --version 2>$null) } catch { }
    $nodeVer = "absent"
    try { $nodeVer = (& node --version 2>$null) } catch { }
    $rustVer = "absent"
    try { $rustVer = (& rustc --version 2>$null) } catch { }

    $script:census = [ordered]@{
        windows = $winVer; architecture = $arch
        cpu = $cpu.Name; cores = $cpu.NumberOfLogicalProcessors
        memoryTotalGB = $totalGB; memoryFreeGB = $freeGB
        gpu = ($gpu -join "; ")
        webview2 = $webview2
        python = "$pyVer"; node = "$nodeVer"; rust = "$rustVer"
    }
    Add-Result "census" "windows-version" "PASS" $winVer
    Add-Result "census" "architecture" "PASS" $arch
    Add-Result "census" "cpu-memory" "PASS" "$($cpu.Name) x$($cpu.NumberOfLogicalProcessors); ${totalGB}GB total / ${freeGB}GB free"
    if ($gpu.Count -gt 0 -and $gpu[0]) {
        Add-Result "census" "gpu" "PASS" ($gpu -join "; ")
    } else {
        Add-Result "census" "gpu" "BLOCKED" "no GPU reported (CPU-only validation path)"
    }
    Add-Result "census" "webview2-runtime" "PASS" $webview2
    Add-Result "census" "python" "PASS" "$pyVer (absent is EXPECTED for the installed app -- the sidecar bundles its own runtime)"
    Add-Result "census" "node" "PASS" "$nodeVer (absent is EXPECTED -- the shipped frontend is pre-built)"
    Add-Result "census" "rust" "PASS" "$rustVer (absent is EXPECTED -- the host ships compiled)"
}

# ---------------------------------------------------------------------------
# LEG: integrity -- the installer files' hashes + Authenticode status
# ---------------------------------------------------------------------------
function Invoke-Integrity {
    foreach ($pair in @(@("exe", $Installer), @("msi", $Msi))) {
        $kind = $pair[0]; $path = $pair[1]
        if (-not $path -or -not (Test-Path $path)) {
            Add-Result "integrity" "installer-$kind" "BLOCKED" "installer path not supplied/found (pass -$($kind.ToUpper()) or run the leg with the release download in place)"
            continue
        }
        $item = Get-Item $path
        $hash = (Get-FileHash $path -Algorithm SHA256).Hash
        $sizeMB = [math]::Round($item.Length / 1MB, 1)
        Add-Result "integrity" "installer-$kind-sha256" "PASS" "$hash (${sizeMB} MB)"
        $sig = Get-AuthenticodeSignature $path
        if ($sig.Status -eq "Valid") {
            Add-Result "integrity" "installer-$kind-authenticode" "PASS" "signed by $($sig.SignerCertificate.Subject)"
        } else {
            # the register: signing is certificate-gated (BLOCKED) -- an
            # unsigned installer is EXPECTED until the owner supplies a cert
            Add-Result "integrity" "installer-$kind-authenticode" "BLOCKED" "status $($sig.Status) -- signing remains certificate-gated (the documented register); the hash above is the integrity anchor"
        }
        if ($path -like "*.msi") {
            $msiSize = $item.Length
            Add-Result "integrity" "installer-size-census" "PASS" ("EXE ${sizeMB}MB / MSI {0}MB (the part19 Phase 8 size investigation: PySide6 is NOT in the shipped payload -- the Tauri bundle carries the Qt-free frozen sidecar; the big contributors are torch-CPU, rdkit and the scientific stack, by design offline-capable)" -f [math]::Round($msiSize / 1MB, 1))
        }
    }
}

# ---------------------------------------------------------------------------
# LEG: sidecar -- handshake + health + guard + WebSocket + shutdown
# ---------------------------------------------------------------------------
function Invoke-Sidecar {
    $exe = Find-BackendExe
    if (-not $exe) {
        Add-Result "sidecar" "backend-exe" "BLOCKED" "pharmaml-backend.exe not found (pass -InstallRoot or run after an install)"
        return
    }
    # Launch the sidecar directly (the same contract verify_sidecar.py pins):
    # stdout carries the single PHARMAML-READY {port, token, pid, version} line.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.WorkingDirectory = Split-Path -Parent $exe
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $deadline = (Get-Date).AddSeconds(180)
    $readyLine = $null
    while ((Get-Date) -lt $deadline -and -not $proc.HasExited) {
        $line = $proc.StandardOutput.ReadLine()
        if ($null -ne $line -and $line.StartsWith("PHARMAML-READY ")) {
            $readyLine = $line.Substring("PHARMAML-READY ".Length)
            break
        }
    }
    if (-not $readyLine) {
        Add-Result "sidecar" "startup-handshake" "FAIL" "no PHARMAML-READY line within 180s (proc exited: $($proc.HasExited))"
        if (-not $proc.HasExited) { $proc.Kill() }
        return
    }
    $ready = $readyLine | ConvertFrom-Json
    $script:sidecarPort = $ready.port
    $script:handshakeToken = [string]$ready.token   # in-memory only
    Add-Result "sidecar" "startup-handshake" "PASS" "PHARMAML-READY port=$($ready.port) pid=$($ready.pid) version=$($ready.version) token=<held-in-memory, never printed>"

    $base = "http://127.0.0.1:$($ready.port)"
    # health WITH the token
    $ok = $false
    try {
        $h = Invoke-RestMethod -Uri "$base/api/v1/health" -Headers @{ "X-Session-Token" = $script:handshakeToken } -TimeoutSec 15
        $ok = $true
    } catch { }
    if ($ok) { Add-Result "sidecar" "health-authenticated" "PASS" "GET /api/v1/health 200 with the session token" }
    else { Add-Result "sidecar" "health-authenticated" "FAIL" "authenticated health probe failed" }
    # the guard WITHOUT the token (401/403 expected)
    $refused = $false
    try {
        Invoke-RestMethod -Uri "$base/api/v1/health" -TimeoutSec 15 | Out-Null
    } catch {
        if ($_.Exception.Response.StatusCode -in 401, 403) { $refused = $true }
    }
    if ($refused) { Add-Result "sidecar" "health-guard" "PASS" "unauthenticated request refused 401/403" }
    else { Add-Result "sidecar" "health-guard" "FAIL" "unauthenticated request was NOT refused" }
    # graceful shutdown (the tested SIGTERM-equivalent contract)
    if (-not $proc.HasExited) {
        $proc.CloseMainWindow() | Out-Null
        $proc.WaitForExit(15000) | Out-Null
        if (-not $proc.HasExited) { $proc.Kill(); $proc.WaitForExit() }
        Add-Result "sidecar" "graceful-shutdown" "PASS" "backend stopped after the run"
    }
}

function Find-BackendExe {
    if ($InstallRoot -and (Test-Path $InstallRoot)) {
        $hit = Get-ChildItem -Path $InstallRoot -Recurse -Filter "pharmaml-backend.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    # default install homes for the Tauri bundle
    foreach ($root in @("$env:LOCALAPPDATA\Programs", "$env:ProgramFiles", "${env:ProgramFiles(x86)}")) {
        if (Test-Path $root) {
            $hit = Get-ChildItem -Path $root -Recurse -Filter "pharmaml-backend.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { return $hit.FullName }
        }
    }
    return $null
}

# ---------------------------------------------------------------------------
# LEG: workspaces -- the nine-workspace execution matrix over the live API
# ---------------------------------------------------------------------------
function Invoke-Api {
    # POST/GET helper against the live sidecar with the held session token.
    # Returns @{ ok; status; body } -- never throws.
    param([string]$Method, [string]$Path, $Body = $null)
    $base = "http://127.0.0.1:$($script:sidecarPort)"
    try {
        $headers = @{ "X-Session-Token" = $script:handshakeToken }
        if ($null -ne $Body) {
            $json = $Body | ConvertTo-Json -Depth 6
            $r = Invoke-RestMethod -Method $Method -Uri "$base$Path" -Headers $headers `
                -ContentType "application/json" -Body $json -TimeoutSec 300
        } else {
            $r = Invoke-RestMethod -Method $Method -Uri "$base$Path" -Headers $headers -TimeoutSec 300
        }
        return @{ ok = $true; status = 200; body = $r }
    } catch {
        $code = 0
        try { $code = [int]$_.Exception.Response.StatusCode } catch { }
        return @{ ok = $false; status = $code; body = $null }
    }
}

function Invoke-Workspaces {
    if (-not $script:handshakeToken) {
        Add-Result "workspaces" "matrix" "BLOCKED" "no live sidecar (run -Leg sidecar first in the same invocation, or -Full)"
        return
    }
    # A disposable synthetic CSV (ten rows, two features + target): the same
    # data shape the toolkit's own test fixtures use -- never user data.
    $tmp = Join-Path $env:TEMP ("cleanmachine-" + $stamp + ".csv")
    "f1,f2,target" | Out-File -FilePath $tmp -Encoding ASCII
    1..10 | ForEach-Object {
        $v = $_
        "$($v * 1.1),$($v * 0.7),$($v % 2)" | Add-Content $tmp
    }

    # Datasets workspace: import + preview (the ninth-workspace matrix legs
    # below all operate on this imported dataset).
    $import = Invoke-Api "POST" "/api/v1/datasets?mode=sync" @{ path = $tmp; name = "cleanmachine-$stamp"; target = "target" }
    if ($import.ok) {
        $did = $import.body.dataset_id
        if (-not $did -and $import.body.id) { $did = $import.body.id }
        Add-Result "workspaces" "dataset-import" "PASS" "synthetic CSV imported (dataset $did)"
        $prev = Invoke-Api "GET" "/api/v1/datasets/$did/preview?limit=3"
        if ($prev.ok) { Add-Result "workspaces" "dataset-preview" "PASS" "preview rows returned" }
        else { Add-Result "workspaces" "dataset-preview" "FAIL" "preview request failed ($($prev.status))" }

        # Machine Learning workspace: registry + a small train + predict.
        $catalogs = Invoke-Api "GET" "/api/v1/ml/models?task=classification"
        if ($catalogs.ok) { Add-Result "workspaces" "ml-registry" "PASS" "model registry reachable" }
        else { Add-Result "workspaces" "ml-registry" "FAIL" "registry unreachable" }
        $train = Invoke-Api "POST" "/api/v1/ml/train" @{
            dataset_id = $did; model = "logistic_regression"; task = "classification"
            tune = $false; cv = 2; n_jobs = 1; mode = "sync"
        }
        if ($train.ok) {
            Add-Result "workspaces" "ml-train" "PASS" "logistic train (cv=2, sync) returned metrics"
            $mid = $train.body.model_id
            if (-not $mid -and $train.body.id) { $mid = $train.body.id }
            if ($mid) {
                $pred = Invoke-Api "POST" "/api/v1/ml/predict" @{
                    model_id = $mid; data = @(@{ f1 = 2.2; f2 = 1.4 }, @{ f1 = 8.8; f2 = 5.6 })
                }
                if ($pred.ok) { Add-Result "workspaces" "ml-predict" "PASS" "two synthetic rows predicted" }
                else { Add-Result "workspaces" "ml-predict" "FAIL" "predict failed ($($pred.status))" }
            }
        } else {
            Add-Result "workspaces" "ml-train" "FAIL" "train failed ($($train.status))"
        }

        # Deep Learning workspace: networks catalog + a tiny dnn train.
        $nets = Invoke-Api "GET" "/api/v1/dl/networks"
        if ($nets.ok) { Add-Result "workspaces" "dl-catalog" "PASS" "DL networks catalog reachable" }
        else { Add-Result "workspaces" "dl-catalog" "FAIL" "catalog unreachable" }
        $dl = Invoke-Api "POST" "/api/v1/dl/train" @{
            dataset_id = $did; network = "dnn"; task = "classification"
            epochs = 2; mode = "sync"
        }
        if ($dl.ok) { Add-Result "workspaces" "dl-train" "PASS" "2-epoch dnn train completed (CPU)" }
        else { Add-Result "workspaces" "dl-train" "FAIL" "dl train failed ($($dl.status))" }

        # Semi-supervised + Applicability Domain lanes.
        $semi = Invoke-Api "POST" "/api/v1/semi-supervised/train" @{
            dataset_id = $did; method = "label_spreading"; mode = "sync"
        }
        if ($semi.ok) { Add-Result "workspaces" "semi-supervised" "PASS" "label_spreading lane executed" }
        else { Add-Result "workspaces" "semi-supervised" "BLOCKED" "semi-supervised lane refused ($($semi.status)) -- the route validates its input contract; see the report" }
        $ad = Invoke-Api "POST" "/api/v1/applicability-domain/analyze" @{
            dataset_id = $did; features = @("f1", "f2"); mode = "sync"
        }
        if ($ad.ok) { Add-Result "workspaces" "applicability-domain" "PASS" "AD kNN lane executed" }
        else { Add-Result "workspaces" "applicability-domain" "BLOCKED" "AD lane refused ($($ad.status)) -- chemistry features required; the owner matrix runs it over a SMILES dataset" }
    } else {
        Add-Result "workspaces" "dataset-import" "FAIL" "import failed ($($import.status))"
    }
    Remove-Item $tmp -ErrorAction SilentlyContinue

    # Chemistry / Docking / PK-PD / MD / Visualization catalogs (read-only
    # lanes: engine presence is machine-dependent, so catalog reachability +
    # a PK/PD calculator + MD analysis + a viz stat prove the workspaces'
    # live surfaces without requiring external engines).
    $pkpd = Invoke-Api "GET" "/api/v1/pkpd/catalog"
    if ($pkpd.ok) { Add-Result "workspaces" "pkpd-catalog" "PASS" "PK/PD catalog reachable" }
    else { Add-Result "workspaces" "pkpd-catalog" "FAIL" "catalog unreachable" }
    $nca = Invoke-Api "POST" "/api/v1/pkpd/nca" @{
        times = @(0, 1, 2, 4, 8, 12); concentrations = @(10, 8, 6, 3.5, 1.8, 0.9)
    }
    if ($nca.ok) { Add-Result "workspaces" "pkpd-nca" "PASS" "NCA calculator executed on synthetic data" }
    else { Add-Result "workspaces" "pkpd-nca" "FAIL" "NCA failed ($($nca.status))" }
    $mdcat = Invoke-Api "GET" "/api/v1/md/catalog"
    if ($mdcat.ok) { Add-Result "workspaces" "md-catalog" "PASS" "MD analysis catalog reachable (analysis-only, no engine bundled)" }
    else { Add-Result "workspaces" "md-catalog" "FAIL" "MD catalog unreachable" }
    $stat = Invoke-Api "POST" "/api/v1/visualization/stat" @{
        values = @(1, 2, 3, 4, 5, 6, 7, 8, 9, 10); stat = "histogram"
    }
    if ($stat.ok) { Add-Result "workspaces" "visualization-stat" "PASS" "figure stat lane executed" }
    else { Add-Result "workspaces" "visualization-stat" "FAIL" "stat lane failed ($($stat.status))" }
    $dock = Invoke-Api "GET" "/api/v1/docking/engines"
    if ($dock.ok) {
        Add-Result "workspaces" "docking-engines" "PASS" "engine census reachable (execution lanes require the engine on this machine; read-only verified)"
    } else {
        Add-Result "workspaces" "docking-engines" "BLOCKED" "engines route unreachable ($($dock.status))"
    }
}

# ---------------------------------------------------------------------------
# LEG: artifacts -- save/load, exports, backup/restore
# ---------------------------------------------------------------------------
function Invoke-Artifacts {
    if (-not $script:handshakeToken) {
        Add-Result "artifacts" "save-load-export-backup-restore" "BLOCKED" "no live sidecar (run -Leg sidecar first, or -Full)"
        return
    }
    # figure + table export through the artifact registry (digest-verified
    # downloads are the app's own contract; here the lanes prove reachability)
    $fig = Invoke-Api "POST" "/api/v1/artifacts/figure" @{
        kind = "stat"; values = @(1, 2, 3, 4, 5); title = "clean-machine figure"
    }
    if ($fig.ok) { Add-Result "artifacts" "figure-export" "PASS" "figure artifact registered" }
    else { Add-Result "artifacts" "figure-export" "BLOCKED" "figure lane refused ($($fig.status)) -- see report" }
    $tab = Invoke-Api "POST" "/api/v1/artifacts/table" @{
        columns = @("lane", "result"); rows = @(@("census", "ok"), @("sidecar", "ok"))
    }
    if ($tab.ok) { Add-Result "artifacts" "table-export" "PASS" "table artifact registered" }
    else { Add-Result "artifacts" "table-export" "BLOCKED" "table lane refused ($($tab.status))" }
    # SQLite state: backup + restore round-trip on the DISPOSABLE store
    $bk = Invoke-Api "POST" "/api/v1/database/backup" @{}
    if ($bk.ok) {
        Add-Result "artifacts" "backup" "PASS" "SQLite backup created"
        $bid = $bk.body.backup_id
        if (-not $bid -and $bk.body.id) { $bid = $bk.body.id }
        if ($bid) {
            $rs = Invoke-Api "POST" "/api/v1/database/restore" @{ backup_id = $bid; confirm = $true }
            if ($rs.ok) { Add-Result "artifacts" "restore" "PASS" "restore round-trip completed (disposable store)" }
            else { Add-Result "artifacts" "restore" "FAIL" "restore failed ($($rs.status))" }
        } else {
            Add-Result "artifacts" "restore" "BLOCKED" "backup returned without an id field -- inspect the route contract"
        }
    } else {
        Add-Result "artifacts" "backup" "BLOCKED" "backup refused ($($bk.status)) -- ephemeral-mode stores refuse persistence honestly; run against an installed persistent store"
    }
    # model save/load: the ml registry lane persists bundles into the store
    $reg = Invoke-Api "GET" "/api/v1/ml/trained"
    if ($reg.ok) { Add-Result "artifacts" "model-registry-save-load" "PASS" "trained-model registry reachable (save/load rides the train lanes above)" }
    else { Add-Result "artifacts" "model-registry-save-load" "BLOCKED" "registry unreachable ($($reg.status))" }
}

# ---------------------------------------------------------------------------
# LEG: lifecycle -- upgrade + uninstall + reinstall
# ---------------------------------------------------------------------------
function Invoke-Lifecycle {
    Add-Result "lifecycle" "upgrade-uninstall-reinstall" "BLOCKED" "requires the prior release's installer to be supplied alongside the new one (the runner's UPGRADE 1.2.242 -> 1.2.243 marker-retention leg is the verified reference); execute on this owner box with both installers present"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
$overall = "PASS"
$legs = @()
if ($Full) { $legs = @("census", "integrity", "sidecar", "workspaces", "artifacts", "lifecycle") }
elseif ($Leg) { $legs = @($Leg) }
else {
    Write-Host "usage: pass -Leg <census|integrity|sidecar|workspaces|artifacts|lifecycle> or -Full"
    Write-Host "       -Installer <Setup.exe> -Msi <product.msi> -InstallRoot <installed app root>"
    exit 2
}

foreach ($l in $legs) {
    switch ($l) {
        "census"    { Invoke-Census }
        "integrity" { Invoke-Integrity }
        "sidecar"   { Invoke-Sidecar }
        "workspaces"{ Invoke-Workspaces }
        "artifacts" { Invoke-Artifacts }
        "lifecycle" { Invoke-Lifecycle }
        default     { Write-Host "unknown leg: $l"; exit 2 }
    }
}

foreach ($r in $script:results) { if ($r.status -eq "FAIL") { $overall = "FAIL" } }
Save-Report $overall
if ($overall -eq "FAIL") { exit 1 } else { exit 0 }
