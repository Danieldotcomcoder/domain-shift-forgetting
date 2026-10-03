$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program $Arguments" }
}
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    Invoke-Checked 'python' @('-m', 'venv', '.venv')
}
$LabPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
# CUDA 12.6 supports the Turing GPU; use this explicit build for the local lab.
Invoke-Checked $LabPython @('-m', 'pip', 'install', 'torch==2.10.0+cu126', '--extra-index-url', 'https://download.pytorch.org/whl/cu126')
Invoke-Checked $LabPython @('-m', 'pip', 'install', '-e', '.[model,test]')
Invoke-Checked $LabPython @('-c', "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU only')")
