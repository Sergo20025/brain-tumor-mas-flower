param(
    [int]$NumServerRounds = 3,
    [int]$NumClients = 10,
    [string]$DatasetRoot = "brain_tumor_mri",
    [string]$PartitionMode = "dirichlet",
    [double]$DirichletAlpha = 0.5,
    [bool]$DecentralizedMode = $true,
    [bool]$UsePretrained = $false,
    [int]$LocalEpochs = 1,
    [int]$BatchSize = 16,
    [double]$LearningRate = 0.0003,
    [double]$WeightDecay = 0.00001
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$env:FLWR_HOME = Join-Path $PSScriptRoot ".flwr-home"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

New-Item -ItemType Directory -Force -Path $env:FLWR_HOME | Out-Null

$runConfig = @(
    "dataset-root='$DatasetRoot'"
    "num-server-rounds=$NumServerRounds"
    "num-clients=$NumClients"
    "min-available-clients=$NumClients"
    "partition-mode='$PartitionMode'"
    "dirichlet-alpha=$DirichletAlpha"
    "decentralized-mode=$($DecentralizedMode.ToString().ToLower())"
    "use-pretrained=$($UsePretrained.ToString().ToLower())"
    "local-epochs=$LocalEpochs"
    "batch-size=$BatchSize"
    "learning-rate=$LearningRate"
    "weight-decay=$WeightDecay"
) -join " "

Write-Host "Starting Flower training with streamed logs..." -ForegroundColor Cyan
Write-Host "Run config: $runConfig" -ForegroundColor DarkGray

flwr run . --stream --run-config $runConfig
