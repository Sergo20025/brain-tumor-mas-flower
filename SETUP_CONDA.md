# Conda + GPU setup

This project runs well in a dedicated Conda environment on Windows.
To avoid OpenMP conflicts with PyTorch on Windows, prefer an OpenBLAS-based Conda stack instead of MKL.

## 1. Create and activate the environment

```powershell
conda env create -f environment.yml
conda activate brain-tumor-fl
```

If you already created the environment before this fix, recreate it once:

```powershell
conda deactivate
conda env remove -n brain-tumor-fl
conda env create -f environment.yml
conda activate brain-tumor-fl
```

## 2. Install GPU-enabled PyTorch

PyTorch's official Windows install page currently offers CUDA 12.6 and 12.8 wheels.
For a modern NVIDIA driver/GPU setup, CUDA 12.8 is the best first choice.

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If that exact CUDA build gives dependency issues in your environment, use the official selector and switch to CUDA 12.6:

https://pytorch.org/get-started/locally/

## 3. Install the project itself

```powershell
python -m pip install -e .
```

## 4. Verify GPU access

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 5. Avoid Windows permission issues with Flower

Flower stores local state in `%FLWR_HOME%` (or `%USERPROFILE%\\.flwr` if `FLWR_HOME` is not set).
To keep everything inside the project and avoid permission issues, set a project-local path before running Flower:

```powershell
$env:FLWR_HOME = (Join-Path (Get-Location) ".flwr-home")
New-Item -ItemType Directory -Force -Path $env:FLWR_HOME | Out-Null
```

## 6. Run the project

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
flwr run . --run-config "dataset-root='brain_tumor_mri' num-server-rounds=3 num-clients=10 partition-mode='dirichlet' dirichlet-alpha=0.5 decentralized-mode=true use-pretrained=false"
```

## Required libraries

- `torch`, `torchvision`, `torchaudio`: model training and GPU execution
- `flwr`: federated learning framework
- `ray`: backend used by Flower simulation runtime
- `numpy`, `scikit-learn`: metrics and data partitioning
- `pillow`: image loading
- `pandas`, `matplotlib`, `seaborn`, `tqdm`: experiment analysis and visualization
- `kaggle`: dataset downloads

## Recommended note

`Python 3.11` is the safest choice for this project.
This is a practical recommendation based on current ecosystem stability for `Flower + Ray + PyTorch` on Windows, not a strict project limitation.

If you see `OMP: Error #15`, it usually means the environment mixed PyTorch's bundled OpenMP runtime with MKL/Intel OpenMP from Conda packages.
The clean fix is to recreate the environment from this file and avoid mixing MKL-based packages into the same environment.
