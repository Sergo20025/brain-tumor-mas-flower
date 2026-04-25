$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "activate_flwr_env.ps1") -EnvName "brain-tumor-fl"

python scripts\compare_experiments.py `
    --title-prefix "CIFAR-100 Method Progress" `
    --output-dir "outputs/experiments/cifar100/comparison" `
    --experiment "Local=outputs/experiments/cifar100/local/exp_cifar100_local_baseline" `
    --experiment "FedAvg Server=outputs/experiments/cifar100/server/exp_cifar100_fedavg" `
    --experiment "FedProx Server=outputs/experiments/cifar100/server/exp_cifar100_fedprox" `
    --experiment "Decentralized Ring=outputs/experiments/cifar100/decentralized/exp_cifar100_ring" `
    --experiment "Decentralized Augmented Ring=outputs/experiments/cifar100/decentralized/exp_cifar100_augmented_ring" `
    --experiment "Decentralized Full Graph=outputs/experiments/cifar100/decentralized/exp_cifar100_full_graph"
