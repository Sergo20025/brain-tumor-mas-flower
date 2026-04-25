param(
    [string]$ExperimentName = "server/exp_server_fedavg",
    [string]$EnvName = "brain-tumor-fl",
    [string]$StrategyName = "fedavg",
    [double]$ProximalMu = 0.01,
    [int]$NumServerRounds = 30,
    [int]$NumClients = 10,
    [string]$DatasetRoot = "brain_tumor_mri",
    [string]$PartitionMode = "shards_quantity_skew",
    [double]$DirichletAlpha = 0.5,
    [bool]$UsePretrained = $false,
    [int]$LocalEpochs = 2,
    [int]$BatchSize = 16,
    [double]$LearningRate = 0.0002,
    [double]$WeightDecay = 0.00001
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "activate_flwr_env.ps1") -EnvName $EnvName

$experimentDir = Join-Path $repoRoot "outputs\experiments\$ExperimentName"
New-Item -ItemType Directory -Force -Path $experimentDir | Out-Null
$saveMetricsPath = "outputs/experiments/$ExperimentName/round_metrics.jsonl"

$args = @(
    "-m", "brain_tumor_fl.server_simulation",
    "--dataset-root", $DatasetRoot,
    "--num-server-rounds", "$NumServerRounds",
    "--num-clients", "$NumClients",
    "--partition-mode", $PartitionMode,
    "--dirichlet-alpha", "$DirichletAlpha",
    "--strategy-name", $StrategyName,
    "--proximal-mu", "$ProximalMu",
    "--use-pretrained", $($UsePretrained.ToString().ToLower()),
    "--local-epochs", "$LocalEpochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--save-metrics-path", $saveMetricsPath
)

Write-Host "Running server simulation: $ExperimentName [$StrategyName]" -ForegroundColor Cyan
python @args
python scripts\plot_experiment.py --experiment-dir "outputs/experiments/$ExperimentName" --dataset-root $DatasetRoot --num-clients "$NumClients" --partition-mode $PartitionMode --alpha "$DirichletAlpha"
python scripts\analyze_experiment.py --experiment-dir "outputs/experiments/$ExperimentName" --partition-mode $PartitionMode
