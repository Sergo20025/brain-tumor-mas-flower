param(
    [string]$EnvName = "brain-tumor-fl"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repoRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is not available in this PowerShell session."
}

$condaHook = conda shell.powershell hook | Out-String
Invoke-Expression $condaHook
conda activate $EnvName

$env:PATH = "$env:CONDA_PREFIX\Scripts;$env:CONDA_PREFIX;$env:PATH"
$env:FLWR_HOME = Join-Path $repoRoot ".flwr-home"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = "536870912"
$env:RAY_OVERRIDE_RESOURCES = '{"object_store_memory":536870912}'

New-Item -ItemType Directory -Force -Path $env:FLWR_HOME | Out-Null

Write-Host "Activated conda env: $EnvName" -ForegroundColor Green
Write-Host "Repo root: $repoRoot" -ForegroundColor DarkGray
Write-Host "FLWR_HOME: $env:FLWR_HOME" -ForegroundColor DarkGray
