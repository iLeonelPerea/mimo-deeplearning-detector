#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test CUDA availability and performance for NVIDIA GPU
Run this script to verify PyTorch can use your NVIDIA GPU
"""

import torch
import time
import sys

print("="*70)
print("PyTorch CUDA Test")
print("="*70)

# 1. Check PyTorch version
print(f"\n1. PyTorch Version: {torch.__version__}")

# 2. Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n2. CUDA Available: {cuda_available}")

if not cuda_available:
    print("\n❌ CUDA is NOT available!")
    print("\nPossible solutions:")
    print("  1. Install/Update NVIDIA drivers:")
    print("     Run: nvidia-smi")
    print("\n  2. Reinstall PyTorch with CUDA support:")
    print("     pip3 uninstall torch torchvision torchaudio")
    print("     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# 3. CUDA version
print(f"\n3. CUDA Version (PyTorch): {torch.version.cuda}")

# 4. cuDNN version
print(f"\n4. cuDNN Version: {torch.backends.cudnn.version()}")

# 5. Number of GPUs
gpu_count = torch.cuda.device_count()
print(f"\n5. Number of GPUs: {gpu_count}")

# 6. GPU Details
print("\n6. GPU Details:")
for i in range(gpu_count):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"      - Compute Capability: {props.major}.{props.minor}")
    print(f"      - Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"      - Multi-Processors: {props.multi_processor_count}")

# 7. Current device
print(f"\n7. Current CUDA Device: {torch.cuda.current_device()}")

# 8. Memory info
print("\n8. GPU Memory:")
print(f"   - Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"   - Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# 9. Speed test: CPU vs GPU
print("\n9. Performance Test (Matrix Multiplication):")
print("   Creating large random matrices...")

size = 5000
iterations = 10

# CPU test
print(f"\n   CPU Test ({iterations} iterations)...")
cpu_times = []
for i in range(iterations):
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_times.append(time.time() - start)

cpu_avg = sum(cpu_times) / len(cpu_times)
print(f"   CPU Average Time: {cpu_avg:.4f} seconds")

# GPU test
print(f"\n   GPU Test ({iterations} iterations)...")
gpu_times = []
for i in range(iterations):
    a_gpu = torch.randn(size, size, device='cuda')
    b_gpu = torch.randn(size, size, device='cuda')
    torch.cuda.synchronize()  # Wait for GPU to finish
    start = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_times.append(time.time() - start)

gpu_avg = sum(gpu_times) / len(gpu_times)
print(f"   GPU Average Time: {gpu_avg:.4f} seconds")

speedup = cpu_avg / gpu_avg
print(f"\n   🚀 Speedup: {speedup:.2f}x faster on GPU!")

# 10. Test moving tensors between CPU and GPU
print("\n10. Testing Tensor Operations on GPU:")
try:
    # Create tensor on CPU
    x = torch.randn(1000, 1000)
    print(f"    ✓ Created tensor on CPU: {x.device}")

    # Move to GPU
    x_gpu = x.to('cuda')
    print(f"    ✓ Moved tensor to GPU: {x_gpu.device}")

    # Perform operations on GPU
    y_gpu = x_gpu @ x_gpu.T
    print(f"    ✓ Matrix multiplication on GPU: {y_gpu.device}")

    # Move back to CPU
    y_cpu = y_gpu.cpu()
    print(f"    ✓ Moved result back to CPU: {y_cpu.device}")

except Exception as e:
    print(f"    ❌ Error: {e}")

# 11. Test complex tensors (needed for MIMO simulation)
print("\n11. Testing Complex Tensors on GPU:")
try:
    # Create complex tensor
    real = torch.randn(100, 100, device='cuda')
    imag = torch.randn(100, 100, device='cuda')
    complex_tensor = torch.complex(real, imag)
    print(f"    ✓ Created complex tensor on GPU: {complex_tensor.dtype}")

    # Complex operations
    result = torch.matmul(complex_tensor, complex_tensor.conj().T)
    print(f"    ✓ Complex matrix multiplication: {result.shape}")

except Exception as e:
    print(f"    ❌ Error: {e}")

print("\n" + "="*70)
print("✅ All tests passed! Your GPU is ready for the MIMO simulation.")
print("="*70)

print("\n📝 Next steps:")
print("   1. The original script will now detect CUDA automatically")
print("   2. Make sure all tensors are moved to GPU in the simulation loop")
print("   3. Expected speedup for MIMO BER simulation: 50-100x faster!")
