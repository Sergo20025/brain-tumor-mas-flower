& "$PSScriptRoot\run_flwr_experiment.ps1" `
    -ExperimentName "exp_05_alpha_10" `
    -PartitionMode "dirichlet" `
    -DirichletAlpha 1.0 `
    -DecentralizedMode $true `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -NumServerRounds 30 `
    -ResetRuntime
