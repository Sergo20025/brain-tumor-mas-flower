param(
    [string]$ExperimentName = "local/exp_local_baseline",
    [string]$EnvName = "brain-tumor-fl",
    [int]$NumServerRounds = 30,
    [string]$DatasetRoot = "brain_tumor_mri",
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
    "-m", "brain_tumor_fl.local_training",
    "--dataset-root", $DatasetRoot,
    "--num-server-rounds", "$NumServerRounds",
    "--use-pretrained", $($UsePretrained.ToString().ToLower()),
    "--local-epochs", "$LocalEpochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--save-metrics-path", $saveMetricsPath
)

Write-Host "Running local baseline: $ExperimentName" -ForegroundColor Cyan
python @args
python scripts\plot_experiment.py --experiment-dir "outputs/experiments/$ExperimentName" --dataset-root $DatasetRoot --num-clients 1 --partition-mode local
python scripts\analyze_experiment.py --experiment-dir "outputs/experiments/$ExperimentName" --partition-mode local
