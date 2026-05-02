param(
    [string]$EnvName = "brain-tumor-fl"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$repoRoot = Split-Path -Path $scriptDir -Parent

. (Join-Path $scriptDir "activate_flwr_env.ps1") -EnvName $EnvName

$experiments = @(
    @{
        Dir = "outputs/experiments/decentralized/exp_decentralized_ring"
        Dataset = "brain_tumor_mri"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/decentralized/exp_decentralized_augmented_ring"
        Dataset = "brain_tumor_mri"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/decentralized/exp_decentralized_full_graph"
        Dataset = "brain_tumor_mri"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/decentralized/exp_decentralized_augmented_ring_async_heterogeneous_mri_r60_d20"
        Dataset = "brain_tumor_mri"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/cifar100/decentralized/exp_cifar100_augmented_ring"
        Dataset = "cifar100"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/cifar100/decentralized/exp_cifar100_full_graph"
        Dataset = "cifar100"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/cifar100/decentralized/exp_cifar100_augmented_ring_async_heterogeneous_r60"
        Dataset = "cifar100"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/cifar100/decentralized/exp_cifar100_augmented_ring_async_heterogeneous_r60_fix1"
        Dataset = "cifar100"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/cifar100/decentralized/exp_cifar100_augmented_ring_async_heterogeneous_r70"
        Dataset = "cifar100"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/cifar100/decentralized/exp_cifar100_augmented_ring_async_heterogeneous_resnet50_r70"
        Dataset = "cifar100"
        Clients = 10
        Partition = "shards_quantity_skew"
        Alpha = 0.5
        SoftMix = 0.15
        SoftExtra = 5
    },
    @{
        Dir = "outputs/experiments/cifar100/decentralized/exp_cifar100_augmented_ring_async_heterogeneous_soft20_r70"
        Dataset = "cifar100"
        Clients = 10
        Partition = "shards_quantity_skew_soft"
        Alpha = 0.5
        SoftMix = 0.20
        SoftExtra = 12
    }
)

foreach ($exp in $experiments) {
    $roundMetrics = Join-Path $repoRoot $exp.Dir | Join-Path -ChildPath "round_metrics.jsonl"
    if (-not (Test-Path -LiteralPath $roundMetrics)) {
        Write-Host "Skip missing experiment: $($exp.Dir)" -ForegroundColor DarkYellow
        continue
    }

    Write-Host "Rebuilding plots: $($exp.Dir)" -ForegroundColor Cyan
    python scripts\plot_experiment.py `
        --experiment-dir $exp.Dir `
        --dataset-root $exp.Dataset `
        --num-clients $exp.Clients `
        --partition-mode $exp.Partition `
        --alpha $exp.Alpha `
        --soft-mix-ratio $exp.SoftMix `
        --soft-min-extra-classes $exp.SoftExtra
}
