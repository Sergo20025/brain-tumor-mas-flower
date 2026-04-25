param(
    [string]$ExperimentName = "exp_default",
    [string]$EnvName = "brain-tumor-fl",
    [int]$NumServerRounds = 30,
    [int]$NumClients = 10,
    [string]$DatasetRoot = "brain_tumor_mri",
    [string]$PartitionMode = "dirichlet",
    [double]$DirichletAlpha = 0.5,
    [bool]$DecentralizedMode = $true,
    [string]$TopologyMode = "augmented_ring",
    [bool]$UsePretrained = $false,
    [int]$LocalEpochs = 2,
    [int]$BatchSize = 16,
    [double]$LearningRate = 0.0002,
    [double]$WeightDecay = 0.00001,
    [double]$FractionFit = 1.0,
    [double]$FractionEvaluate = 1.0,
    [int]$TopologyExtraOffset = 2,
    [switch]$ResetRuntime
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "activate_flwr_env.ps1") -EnvName $EnvName

if ($ResetRuntime) {
    foreach ($proc in @("flower-superlink", "flower-superexec", "raylet", "gcs_server")) {
        Get-Process -Name $proc -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    }
}

$experimentDir = Join-Path $repoRoot "outputs\experiments\$ExperimentName"
New-Item -ItemType Directory -Force -Path $experimentDir | Out-Null

$saveMetricsPath = "outputs/experiments/$ExperimentName/round_metrics.jsonl"

Write-Host "Running experiment: $ExperimentName" -ForegroundColor Cyan
Write-Host "Results directory: $experimentDir" -ForegroundColor DarkGray
$args = @(
    "-m", "brain_tumor_fl.decentralized_simulation",
    "--dataset-root", $DatasetRoot,
    "--num-server-rounds", "$NumServerRounds",
    "--num-clients", "$NumClients",
    "--partition-mode", $PartitionMode,
    "--dirichlet-alpha", "$DirichletAlpha",
    "--decentralized-mode", $($DecentralizedMode.ToString().ToLower()),
    "--topology-mode", $TopologyMode,
    "--topology-extra-offset", "$TopologyExtraOffset",
    "--use-pretrained", $($UsePretrained.ToString().ToLower()),
    "--local-epochs", "$LocalEpochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--save-metrics-path", $saveMetricsPath
)

Write-Host "Python args: $($args -join ' ')" -ForegroundColor DarkGray
python @args
