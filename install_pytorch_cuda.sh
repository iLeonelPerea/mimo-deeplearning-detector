#!/bin/bash
# Script to install PyTorch with CUDA support for NVIDIA RTX 4090
# CUDA 12.x compatible

echo "=============================================="
echo "PyTorch CUDA Installation Script"
echo "=============================================="

# Check if nvidia-smi works
echo ""
echo "1. Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

nvidia-smi
echo ""

# Check Python version
echo "2. Checking Python version..."
python3 --version
echo ""

# Uninstall existing PyTorch
echo "3. Uninstalling existing PyTorch packages..."
pip3 uninstall -y torch torchvision torchaudio
echo ""

# Install PyTorch with CUDA 12.1
echo "4. Installing PyTorch with CUDA 12.1 support..."
echo "   This may take several minutes..."
echo ""

# Option 1: CUDA 12.1 (recommended for RTX 4090)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If the above fails, try CUDA 11.8 (more compatible)
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  CUDA 12.1 installation failed. Trying CUDA 11.8..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo ""
echo "5. Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not available\"}')"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Next step: Run the test script"
echo "  python3 Pytorch/test_cuda.py"
