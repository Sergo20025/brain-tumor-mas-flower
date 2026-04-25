& "$PSScriptRoot\run_local_experiment.ps1" `
    -ExperimentName "local/exp_local_baseline" `
    -UsePretrained $false `
    -LocalEpochs 2 `
    -NumServerRounds 30
