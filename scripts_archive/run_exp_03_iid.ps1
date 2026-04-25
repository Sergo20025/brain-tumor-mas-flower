& "$PSScriptRoot\run_flwr_experiment.ps1" `
    -ExperimentName "exp_03_iid" `
    -PartitionMode "iid" `
    -DirichletAlpha 0.5 `
    -DecentralizedMode $true `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -NumServerRounds 30 `
    -ResetRuntime
