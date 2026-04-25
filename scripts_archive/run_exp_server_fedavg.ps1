& "$PSScriptRoot\run_server_experiment.ps1" `
    -ExperimentName "server/exp_server_fedavg" `
    -StrategyName "fedavg" `
    -PartitionMode "shards_quantity_skew" `
    -DirichletAlpha 0.5 `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -NumServerRounds 30
