param([string]$Python = 'python', [switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $ProjectRoot
if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or 3.12 is required.' }
}
$ProjectPython = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$TorchIndex = if ($CpuOnly) {'https://download.pytorch.org/whl/cpu'} else {'https://download.pytorch.org/whl/cu128'}
& $ProjectPython -m pip install torch==2.10.0 torchvision==0.25.0 --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $ProjectPython -m pip install -r requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $ProjectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts found.' }
if (!(Test-Path -LiteralPath 'configs/cameras.local.yaml')) { Copy-Item -LiteralPath 'configs/cameras.example.yaml' -Destination 'configs/cameras.local.yaml' }
& $ProjectPython -m scripts.diagnose
Write-Host 'Ready. Run scripts/start.ps1. Configure .env and cameras.local.yaml for real input.'
exit 0
