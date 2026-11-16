param([int]$Seconds=1800,[string]$Cameras='configs/cameras.local.yaml')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
& .venv/Scripts/python.exe -m scripts.benchmark --seconds $Seconds --cameras $Cameras
exit $LASTEXITCODE
