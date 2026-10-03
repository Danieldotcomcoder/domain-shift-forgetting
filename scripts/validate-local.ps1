param(
    [int]$WaitForProcessId = 0,
    [string]$Output = ('artifacts/local-lab/validation-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$LabValidation = New-Item -ItemType Directory -Path $Output
$LabStatus = Join-Path $LabValidation.FullName 'status.json'
function Set-LabStatus([string]$State, [string]$Detail) {
    @{ status = $State; detail = $Detail; time = (Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $LabStatus -Encoding utf8
}
Start-Transcript -Path (Join-Path $LabValidation.FullName 'validation.log') | Out-Null
try {
    if ($WaitForProcessId -gt 0 -and (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue)) {
        Set-LabStatus 'waiting_for_download' "Waiting for installer process $WaitForProcessId"
        Wait-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    }
    Set-LabStatus 'installing' 'Resolving the local CUDA environment'
    . (Join-Path $PSScriptRoot 'setup-local.ps1')
    $LabPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
    Set-LabStatus 'testing' 'Running the prepared tests and two short GPU smoke runs'
    $env:OMP_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    Invoke-Checked $LabPython @('-m', 'pytest', '-q', '--basetemp', (Join-Path $LabValidation.FullName 'pytest-temp'), '--junitxml', (Join-Path $LabValidation.FullName 'tests.xml'))
    Invoke-Checked $LabPython @('-m', 'domain_shift_forgetting.local_lab', '--device', 'cuda', '--smoke', '--output', (Join-Path $LabValidation.FullName 'tiny-smoke'))
    Invoke-Checked $LabPython @('-m', 'domain_shift_forgetting.local_lab', '--device', 'cuda', '--model', 'pilot-rms', '--smoke', '--batch-size', '1', '--accumulation', '1', '--output', (Join-Path $LabValidation.FullName 'pilot-rms-smoke'))
    & $LabPython -m pip freeze | Set-Content -LiteralPath (Join-Path $LabValidation.FullName 'packages.txt') -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw 'Could not record installed packages' }
    Set-LabStatus 'completed' 'Tests and both GPU smoke runs passed; learning-only evidence'
} catch {
    Set-LabStatus 'failed' $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
