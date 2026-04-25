& "$PSScriptRoot\run_flwr_experiment.ps1" `
    -ExperimentName "exp_07_local_epochs_3" `
    -PartitionMode "dirichlet" `
    -DirichletAlpha 0.5 `
    -DecentralizedMode $true `
    -UsePretrained $false `
    -LocalEpochs 3 `
    -NumServerRounds 30 `
    -ResetRuntime
