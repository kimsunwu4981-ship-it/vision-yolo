$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $ProjectRoot
& .venv/Scripts/python.exe -m scripts.diagnose
exit $LASTEXITCODE
