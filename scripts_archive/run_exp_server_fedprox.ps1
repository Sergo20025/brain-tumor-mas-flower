& "$PSScriptRoot\run_server_experiment.ps1" `
    -ExperimentName "server/exp_server_fedprox" `
    -StrategyName "fedprox" `
    -ProximalMu 0.01 `
    -PartitionMode "shards_quantity_skew" `
    -DirichletAlpha 0.5 `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -NumServerRounds 30
