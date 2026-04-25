& "$PSScriptRoot\run_server_experiment.ps1" `
    -ExperimentName "cifar100/server/exp_cifar100_fedavg" `
    -DatasetRoot "cifar100" `
    -StrategyName "fedavg" `
    -PartitionMode "shards_quantity_skew" `
    -DirichletAlpha 0.5 `
    -UsePretrained $false `
    -LocalEpochs 1 `
    -BatchSize 32 `
    -NumServerRounds 10
