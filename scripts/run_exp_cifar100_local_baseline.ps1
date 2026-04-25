& "$PSScriptRoot\run_local_experiment.ps1" `
    -ExperimentName "cifar100/local/exp_cifar100_local_baseline" `
    -DatasetRoot "cifar100" `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -BatchSize 32 `
    -LearningRate 0.0005 `
    -WeightDecay 0.0001 `
    -NumServerRounds 30
