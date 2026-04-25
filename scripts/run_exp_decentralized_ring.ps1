& "$PSScriptRoot\run_decentralized_experiment.ps1" `
    -ExperimentName "decentralized/exp_decentralized_ring" `
    -PartitionMode "shards_quantity_skew" `
    -DirichletAlpha 0.5 `
    -TopologyMode "ring" `
    -TopologyExtraOffset 2 `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -NumServerRounds 30
