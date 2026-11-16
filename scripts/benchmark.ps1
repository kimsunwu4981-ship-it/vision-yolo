param([int]$Seconds=1800,[string]$Cameras='configs/cameras.local.yaml',[ValidateSet(1,2,4)][int]$Channels=4,[ValidateSet(1,2,4)][int]$Microbatch=1)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
& .venv/Scripts/python.exe -m scripts.benchmark --seconds $Seconds --cameras $Cameras --channels $Channels --microbatch $Microbatch
exit $LASTEXITCODE
