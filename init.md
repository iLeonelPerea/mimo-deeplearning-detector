# Environment Setup Instructions (Windows)

## Create and activate virtual environment
```powershell
py -3.11 -m virtualenv venv
.\venv\Scripts\activate
```

## Force Windows to use Python from venv
These commands ensure Windows uses the Python interpreter inside the venv directory:
```powershell
function python { & (Resolve-Path .\venv\Scripts\python.exe) @args }
function pip    { & (Resolve-Path .\venv\Scripts\pip.exe)    @args }
python --version
pip --version
Get-Command python | Select-Object Source
```

## Install PyTorch with CUDA 12.1 support
```powershell
pip install torch==2.5.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Install project dependencies
```powershell
pip install -r requirements.txt
```

## Verify CUDA availability
```powershell
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```
