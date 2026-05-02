param(
    [string]$ExperimentName = "decentralized/exp_decentralized_augmented_ring",
    [string]$EnvName = "brain-tumor-fl",
    [int]$NumServerRounds = 30,
    [int]$NumClients = 10,
    [string]$DatasetRoot = "brain_tumor_mri",
    [string]$PartitionMode = "shards_quantity_skew",
    [double]$DirichletAlpha = 0.5,
    [double]$SoftMixRatio = 0.15,
    [int]$SoftMinExtraClasses = 5,
    [string]$ModelName = "efficientnet_b0",
    [string]$TopologyMode = "augmented_ring",
    [int]$TopologyExtraOffset = 2,
    [bool]$UsePretrained = $false,
    [int]$LocalEpochs = 2,
    [int]$BatchSize = 16,
    [double]$LearningRate = 0.0002,
    [double]$WeightDecay = 0.00001,
    [bool]$AsyncMode = $false,
    [double]$AsyncDropoutRate = 0.0,
    [int]$MaxAsyncDropouts = 0,
    [bool]$HeterogeneousNodes = $false,
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "activate_flwr_env.ps1") -EnvName $EnvName

$experimentDir = Join-Path $repoRoot "outputs\experiments\$ExperimentName"
New-Item -ItemType Directory -Force -Path $experimentDir | Out-Null
$saveMetricsPath = "outputs/experiments/$ExperimentName/round_metrics.jsonl"

$args = @(
    "-m", "brain_tumor_fl.decentralized_simulation",
    "--dataset-root", $DatasetRoot,
    "--num-server-rounds", "$NumServerRounds",
    "--num-clients", "$NumClients",
    "--partition-mode", $PartitionMode,
    "--dirichlet-alpha", "$DirichletAlpha",
    "--soft-mix-ratio", "$SoftMixRatio",
    "--soft-min-extra-classes", "$SoftMinExtraClasses",
    "--model-name", $ModelName,
    "--decentralized-mode", "true",
    "--topology-mode", $TopologyMode,
    "--topology-extra-offset", "$TopologyExtraOffset",
    "--use-pretrained", $($UsePretrained.ToString().ToLower()),
    "--local-epochs", "$LocalEpochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--async-mode", $($AsyncMode.ToString().ToLower()),
    "--async-dropout-rate", "$AsyncDropoutRate",
    "--max-async-dropouts", "$MaxAsyncDropouts",
    "--heterogeneous-nodes", $($HeterogeneousNodes.ToString().ToLower()),
    "--save-metrics-path", $saveMetricsPath
)

if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    $args += @("--resume-checkpoint", $ResumeCheckpoint)
}

Write-Host "Running decentralized simulation: $ExperimentName [$TopologyMode]" -ForegroundColor Cyan
python @args
python scripts\plot_experiment.py --experiment-dir "outputs/experiments/$ExperimentName" --dataset-root $DatasetRoot --num-clients "$NumClients" --partition-mode $PartitionMode --alpha "$DirichletAlpha" --soft-mix-ratio "$SoftMixRatio" --soft-min-extra-classes "$SoftMinExtraClasses"
python scripts\analyze_experiment.py --experiment-dir "outputs/experiments/$ExperimentName" --partition-mode $PartitionMode
