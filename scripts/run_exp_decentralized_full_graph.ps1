& "$PSScriptRoot\run_decentralized_experiment.ps1" `
    -ExperimentName "decentralized/exp_decentralized_full_graph" `
    -PartitionMode "shards_quantity_skew" `
    -DirichletAlpha 0.5 `
    -TopologyMode "full_graph" `
    -TopologyExtraOffset 2 `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -NumServerRounds 30
