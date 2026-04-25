& "$PSScriptRoot\run_flwr_experiment.ps1" `
    -ExperimentName "exp_04_alpha_03" `
    -PartitionMode "dirichlet" `
    -DirichletAlpha 0.3 `
    -DecentralizedMode $true `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -NumServerRounds 30 `
    -ResetRuntime
