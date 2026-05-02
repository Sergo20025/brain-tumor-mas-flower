param(
    [string]$EnvName = "brain-tumor-fl"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent

& (Join-Path $scriptDir "run_decentralized_experiment.ps1") `
    -ExperimentName "cifar100/decentralized/exp_cifar100_augmented_ring_async_heterogeneous_soft20_r70" `
    -EnvName $EnvName `
    -DatasetRoot "cifar100" `
    -PartitionMode "shards_quantity_skew_soft" `
    -SoftMixRatio 0.20 `
    -SoftMinExtraClasses 12 `
    -ModelName "efficientnet_b0" `
    -TopologyMode "augmented_ring" `
    -TopologyExtraOffset 2 `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -BatchSize 32 `
    -LearningRate 0.0005 `
    -WeightDecay 0.0001 `
    -NumServerRounds 70 `
    -AsyncMode $true `
    -AsyncDropoutRate 0.1 `
    -MaxAsyncDropouts 2 `
    -HeterogeneousNodes $true
