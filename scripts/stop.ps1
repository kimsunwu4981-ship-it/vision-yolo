$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Split-Path $PSScriptRoot -Parent)).Path
Set-Location -LiteralPath $ProjectRoot
& .venv/Scripts/python.exe -m scripts.stop_server
exit $LASTEXITCODE
