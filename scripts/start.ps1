param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Split-Path $PSScriptRoot -Parent)).Path
$ProjectPython = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $ProjectPython)) { throw 'Run scripts/setup.ps1 first.' }
$RuntimeDirectory = Join-Path $ProjectRoot '.runtime'
New-Item -ItemType Directory -Path $RuntimeDirectory -Force | Out-Null
$PidFile = Join-Path $RuntimeDirectory 'server.json'
if (Test-Path -LiteralPath $PidFile) {
    $Owned = Get-Content -LiteralPath $PidFile -Raw | ConvertFrom-Json
    $Running = Get-Process -Id $Owned.pid -ErrorAction SilentlyContinue
    if ($Running -and $Running.StartTime.ToUniversalTime().Ticks.ToString() -eq $Owned.start_ticks) {
        Write-Host 'This project is already running.'
        if (!$NoBrowser) { Start-Process 'http://127.0.0.1:8765' }
        exit 0
    }
}
$ServerProcess = Start-Process -FilePath $ProjectPython -ArgumentList @('-m','app.main') -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeDirectory 'server.out.log') -RedirectStandardError (Join-Path $RuntimeDirectory 'server.err.log')
@{pid=$ServerProcess.Id; start_ticks=$ServerProcess.StartTime.ToUniversalTime().Ticks.ToString(); project=$ProjectRoot; executable=$ServerProcess.Path} | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding UTF8
for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    Start-Sleep -Milliseconds 500
    if ($ServerProcess.HasExited) { throw 'Server failed. Read .runtime/server.err.log.' }
    try {
        $Response = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/health/live' -UseBasicParsing -TimeoutSec 1
        if ($Response.StatusCode -eq 200) { break }
    } catch { }
}
Write-Host 'Dashboard: http://127.0.0.1:8765'
if (!$NoBrowser) { Start-Process 'http://127.0.0.1:8765' }
