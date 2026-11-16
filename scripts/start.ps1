param([switch]$NoBrowser, [switch]$Demo)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Split-Path $PSScriptRoot -Parent)).Path
$ProjectPython = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$DashboardPort = 8765
if (Test-Path -LiteralPath (Join-Path $ProjectRoot '.env')) {
    foreach ($SettingLine in (Get-Content -LiteralPath (Join-Path $ProjectRoot '.env'))) {
        if ($SettingLine -match '^SAFETY_PORT=(\d+)$') { $DashboardPort = [int]$Matches[1] }
    }
}
if ($env:SAFETY_PORT) { $DashboardPort = [int]$env:SAFETY_PORT }
$DashboardUrl = "http://127.0.0.1:$DashboardPort"
if (!(Test-Path -LiteralPath $ProjectPython)) { throw 'Run scripts/setup.ps1 first.' }
if ($Demo) {
    if (!(Test-Path -LiteralPath (Join-Path $ProjectRoot 'configs/cameras.demo.local.yaml'))) { throw 'Run python -m scripts.make_demo_videos first.' }
    $env:SAFETY_CAMERAS = Join-Path $ProjectRoot 'configs/cameras.demo.local.yaml'
}
$RuntimeDirectory = Join-Path $ProjectRoot '.runtime'
New-Item -ItemType Directory -Path $RuntimeDirectory -Force | Out-Null
$PidFile = Join-Path $RuntimeDirectory 'server.json'
if (Test-Path -LiteralPath $PidFile) {
    $Owned = Get-Content -LiteralPath $PidFile -Raw | ConvertFrom-Json
    $Running = Get-Process -Id $Owned.pid -ErrorAction SilentlyContinue
    if ($Running -and $Running.StartTime.ToUniversalTime().Ticks.ToString() -eq $Owned.start_ticks) {
        Write-Host 'This project is already running.'
        if (!$NoBrowser) { Start-Process $DashboardUrl }
        exit 0
    }
}
$ServerProcess = Start-Process -FilePath $ProjectPython -ArgumentList @('-m','app.main') -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeDirectory 'server.out.log') -RedirectStandardError (Join-Path $RuntimeDirectory 'server.err.log')
@{pid=$ServerProcess.Id; start_ticks=$ServerProcess.StartTime.ToUniversalTime().Ticks.ToString(); project=$ProjectRoot; executable=$ServerProcess.Path} | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding UTF8
for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    Start-Sleep -Milliseconds 500
    if ($ServerProcess.HasExited) { throw 'Server failed. Read .runtime/server.err.log.' }
    try {
        $Response = Invoke-WebRequest -Uri "$DashboardUrl/health/live" -UseBasicParsing -TimeoutSec 1
        if ($Response.StatusCode -eq 200) { break }
    } catch { }
}
if (!$Response -or $Response.StatusCode -ne 200) { throw 'Server readiness timed out. Read .runtime/server.err.log.' }
Write-Host "Dashboard: $DashboardUrl"
if (!$NoBrowser) { Start-Process $DashboardUrl }
