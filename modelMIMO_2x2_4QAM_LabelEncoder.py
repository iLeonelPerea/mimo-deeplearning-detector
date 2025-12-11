# -*- coding: utf-8 -*-
"""modelMIMO_2x2_4QAM_LabelEncoder.py

# MIMO 2x2 Deep Learning Detector Training - Label Encoding Strategy

## Description
This script implements a Deep Learning-based detector for a 2x2 MIMO communication system
using **label/symbol encoding** strategy. Instead of using one-hot vectors, this approach
encodes the sign bits of the constellation symbols directly, resulting in a more compact
representation.

**Key Difference**: Uses only **log₂(M)×Nt = 4 output neurons** instead of 16, making it
the most efficient encoding strategy.

## Reference
Based on the work by:
- Ibarra-Hernández, R.F.; Castillo-Soria, F.R.; Gutiérrez, C.A.; Del-Puerto-Flores, J.A.;
  Acosta-Elías J., Rodríguez-Abdalá V. and Palacios-Luengas L.
- "Efficient Deep Learning-Based Detection Scheme for MIMO Communication System"
- Journal: Sensors (MDPI), 2024

## Implementation
- Author: Leonel Roberto Perea Trejo (iticleonel.leonel@gmail.com)
- Version: 2.0.0
- Date: January 2025
- Python/PyTorch implementation with vectorized operations

## License
This code is licensed under the GPLv2 license. If you use this code for research that
results in publications, please cite the paper above.

## 1. Import Libraries and Setup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import centralized configuration
from config import *

torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

"""## 2. System Parameters"""

# Neural Network Hyperparameters
input_size = 2 * Nr
output_size = int(np.log2(M) * Nt)  # log₂(4) × 2 = 4 bits
n_epochs = NUM_EPOCHS
train_split = 0.8

SNR_dB = FIXED_SNR_DB
SNR_linear = 10**(SNR_dB/10)

print("="*70)
print("Label Encoding Strategy - MIMO 2x2 Detector")
print("="*70)
print(f"Modulation: {M}-QAM")
print(f"MIMO: {Nt}x{Nr}")
print(f"Training samples: {N_TRAINING_SAMPLES}")
print(f"SNR mode: {SNR_MODE}")
if SNR_MODE == 'fixed':
    print(f"  Fixed SNR: {SNR_dB} dB")
else:
    print(f"  Variable SNR: 1-20 dB")
print(f"\nNetwork Architecture:")
print(f"  Input: {input_size} neurons")
print(f"  Hidden: {HIDDEN_SIZE} neurons (ReLU)")
print(f"  Output: {output_size} neurons (Sigmoid)")
print(f"\nKey Feature: Uses sign bits encoding")
print(f"  [sign(Re(s1)), sign(Im(s1)), sign(Re(s2)), sign(Im(s2))]")
print("="*70)

"""## 3. Generate QAM Constellation"""

def generate_qam_constellation(M):
    qam_idx = torch.arange(M)
    c = int(np.sqrt(M))
    real_part = -2 * (qam_idx % c) + c - 1
    imag_part = 2 * torch.floor(qam_idx.float() / c) - c + 1
    qam_symbols = torch.complex(real_part.float(), imag_part.float())
    return qam_symbols

qam_symbols = generate_qam_constellation(M)
print("4-QAM Constellation:")
for i, sym in enumerate(qam_symbols):
    print(f"  Symbol {i}: {sym}")

"""## 4. Generate Symbol Combinations and Sign Encoding

The key feature of label encoding is that we encode based on the **sign** of real and imaginary parts.
"""

# Generate all symbol combinations
symbol_combinations = torch.tensor(
    list(product(qam_symbols.numpy(), repeat=Nt)),
    dtype=torch.complex64,
    device=device
)

# Apply 1/sqrt(2) normalization - MATLAB standard (Opción A)
# This normalizes symbol power from 2 to 1 (IEEE standard)
symbol_combinations = symbol_combinations / np.sqrt(2)

print(f"Total combinations: {len(symbol_combinations)}")
print(f"Average power (after 1/√2 normalization): {torch.mean(torch.abs(symbol_combinations)**2).item():.4f}")

# Create sign-based encoding matrix
# 1 if negative, 0 if positive
real_sign = (symbol_combinations.real < 0).int()
imag_sign = (symbol_combinations.imag < 0).int()

# Format: [sign(Re(s1)), sign(Im(s1)), sign(Re(s2)), sign(Im(s2))]
idx_sign = torch.stack([
    real_sign[:, 0], imag_sign[:, 0],
    real_sign[:, 1], imag_sign[:, 1]
], dim=1)

print(f"\nSign encoding matrix shape: {idx_sign.shape}")
print(f"\nFirst 5 symbol combinations and their sign encodings:")
for i in range(5):
    print(f"  Combo {i}: {symbol_combinations[i]} → {idx_sign[i].tolist()}")

"""## 5. Generate Training Data"""

def generate_training_data(N, symbol_combinations, idx_sign, SNR_linear, No, Nr, Nt, device='cpu', use_zf=False, decouple_antennas=False, channel_mode='fixed', snr_mode='variable', fixed_snr_db=3):
    X_data = torch.zeros((N, 2*Nr), dtype=torch.float32, device=device)
    y_labels = torch.zeros((N, output_size), dtype=torch.float32, device=device)
    random_indices = torch.randint(0, len(symbol_combinations), (N,), device=device)

    # Channel configuration
    if channel_mode == 'fixed':
        # FIXED CHANNEL: Use same channel for all samples (H ~ CN(0,1) standard Rayleigh)
        H = torch.tensor(FIXED_CHANNEL, dtype=torch.complex64, device=device)
    else:  # channel_mode == 'random'
        # RANDOM CHANNEL: Generate ONE random channel and use for ALL samples (channel-specific training)
        # This trains the network for this specific random channel H
        H_real = torch.randn((Nr, Nt), device=device) / np.sqrt(2)
        H_imag = torch.randn((Nr, Nt), device=device) / np.sqrt(2)
        H = torch.complex(H_real, H_imag)

    # Pre-compute pseudoinverse if using ZF or antenna decoupling (optimization)
    if use_zf or decouple_antennas:
        H_inv = torch.linalg.pinv(H)

    print("Generating training data...")
    print(f"Channel mode: {channel_mode.upper()}")
    print(f"SNR mode: {snr_mode.upper()}")
    if snr_mode == 'fixed':
        print(f"  Fixed SNR: {fixed_snr_db} dB")
    else:
        print(f"  Variable SNR: 1-20 dB (random per sample)")
    print(f"Zero-Forcing equalization: {'ENABLED' if use_zf else 'DISABLED'}")
    print(f"Antenna decoupling: {'ENABLED' if decouple_antennas else 'DISABLED'}")

    for i in range(N):
        if (i + 1) % (N // 10) == 0:
            print(f"  Progress: {int((i+1)/N*100)}%", end='\r')

        selected_symbols = symbol_combinations[random_indices[i]]
        y_labels[i] = idx_sign[random_indices[i]].float()

        # SNR handling based on mode
        if snr_mode == 'fixed':
            # FIXED SNR mode (MATLAB matching):
            # - SNR is constant for all samples
            # - Noise is normalized by 1/sqrt(SNR)
            # - Signal is NOT scaled by sqrt(SNR)
            SNR_linear_sample = 10.0 ** (fixed_snr_db / 10.0)

            n_real = torch.randn(Nr, device=device) * np.sqrt(No/2)
            n_imag = torch.randn(Nr, device=device) * np.sqrt(No/2)
            n = torch.complex(n_real, n_imag)
            n = n / np.sqrt(SNR_linear_sample)  # Normalize noise with 1/sqrt(SNR)

            # Received signal: r = H * x + n (no sqrt(SNR) scaling)
            r_x = torch.matmul(H, selected_symbols) + n

        else:  # snr_mode == 'variable'
            # VARIABLE SNR mode (IEEE standard):
            # - SNR is random per sample (0-20 dB continuous)
            # - Noise has FIXED variance (not normalized by SNR)
            # - Signal is scaled by sqrt(SNR)
            SNR_dB_sample = np.random.uniform(0, 20)  # Random SNR between 0-20 dB (continuous)
            SNR_linear_sample = 10.0 ** (SNR_dB_sample / 10.0)

            n_real = torch.randn(Nr, device=device) * np.sqrt(No/2)
            n_imag = torch.randn(Nr, device=device) * np.sqrt(No/2)
            n = torch.complex(n_real, n_imag)
            # DO NOT normalize noise by SNR

            # Received signal: r = sqrt(SNR) * H * x + n
            r_x = np.sqrt(SNR_linear_sample) * torch.matmul(H, selected_symbols) + n

        # Apply preprocessing based on configuration
        if decouple_antennas:
            # Francisco's preprocessing: Apply H⁺ to decouple antennas
            # The approach depends on SNR mode to match evaluation:
            if snr_mode == 'variable':
                # SNR VARIABLE: Apply H⁺ on r_x (which already has sqrt(SNR))
                # r = H⁺ * (sqrt(SNR)*H*x + n) = sqrt(SNR)*x + H⁺*n
                r_processed = torch.matmul(H_inv, r_x)
            else:  # snr_mode == 'fixed'
                # SNR FIXED (MATLAB): Apply H⁺ on H*x, THEN add normalized noise
                # r = H⁺ * (H*x) + n/sqrt(SNR) = x + n/sqrt(SNR)
                r_temp = torch.matmul(H, selected_symbols)
                r_eq = torch.matmul(H_inv, r_temp)
                r_processed = r_eq + n
        elif use_zf:
            # Standard Zero-Forcing: r_eq = H⁺ * (H*x + n) = x + H⁺*n
            r_processed = torch.matmul(H_inv, r_x)
        else:
            # Use received signal directly (default behavior)
            r_processed = r_x

        X_data[i, 0] = r_processed[0].real
        X_data[i, 1] = r_processed[0].imag
        X_data[i, 2] = r_processed[1].real
        X_data[i, 3] = r_processed[1].imag

    print("  Progress: 100% - Complete!")
    return X_data, y_labels, random_indices, H

X_data, y_labels, random_indices, H_training = generate_training_data(
    N_TRAINING_SAMPLES, symbol_combinations, idx_sign, SNR_linear, No, Nr, Nt, device,
    use_zf=USE_ZF, decouple_antennas=DECOUPLE_ANTENNAS, channel_mode=CHANNEL_MODE,
    snr_mode=SNR_MODE, fixed_snr_db=FIXED_SNR_DB
)

print(f"\nData shape: {X_data.shape}")
print(f"Labels shape: {y_labels.shape}")
print(f"\nFirst 3 labels (sign encoding):")
print(y_labels[:3])

"""## 6. Data Normalization"""

X_mean = torch.tensor(0.0)
X_std = torch.tensor(1.0)

print("Normalization skipped for Variable SNR training.")
X_data_normalized = X_data

print("Data statistics:")
print(f"  Mean: {X_data.mean().item():.6f}")
print(f"  Std:  {X_data.std().item():.6f}")

"""## 7. Train-Test Split"""

train_qty = int(train_split * N_TRAINING_SAMPLES)
test_qty = N_TRAINING_SAMPLES - train_qty

X_train = X_data_normalized[:train_qty]
y_train = y_labels[:train_qty]
idx_train = random_indices[:train_qty]

X_test = X_data_normalized[train_qty:]
y_test = y_labels[train_qty:]
idx_test = random_indices[train_qty:]

print(f"Training samples: {train_qty}")
print(f"Testing samples: {test_qty}")

"""## 8. Define Neural Network Architecture

**Key Difference**: Output layer uses **Sigmoid** activation (not Softmax) because we're doing binary classification for each bit.
"""

class MIMO_Detector_LabelEncoder(nn.Module):
    """
    MIMO detector using label/symbol encoding.

    Output: 4 binary values representing sign bits
    Loss: Binary Cross-Entropy

    Args:
        use_sigmoid_hidden (bool): If True, applies sigmoid before ReLU (default: True )
        use_bias (bool): If True, uses bias in hidden layer (default: False )
    """

    def __init__(self, input_size, hidden_size, output_size, use_sigmoid_hidden=True, use_bias=False):
        super(MIMO_Detector_LabelEncoder, self).__init__()
        self.use_sigmoid_hidden = use_sigmoid_hidden
        self.layer1 = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        # Hidden layer activation (Sigmoid + ReLU)
        x = self.layer1(x)
        if self.use_sigmoid_hidden:
            x = torch.sigmoid(x)  # Sigmoid 
        x = F.relu(x)  # ReLU activation 
        x = self.layer2(x)  # Logits (sigmoid applied in loss)
        return x

model = MIMO_Detector_LabelEncoder(
    input_size,
    HIDDEN_SIZE,
    output_size,
    use_sigmoid_hidden=True,  # Label Encoder uses Sigmoid + ReLU
    use_bias=USE_BIAS
).to(device)
print("="*70)
print("Model Architecture")
print("="*70)
print(model)
print("="*70)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

"""## 9. Define Loss Function and Optimizer

**Binary Cross-Entropy Loss** is used because each output is a binary classification (sign bit).
"""

# Binary Cross-Entropy Loss (includes sigmoid)
criterion = nn.BCEWithLogitsLoss()

# SGD optimizer ()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0)

print("Loss function: BCEWithLogitsLoss (Binary Cross-Entropy)")
print(f"Optimizer: SGD with lr={LEARNING_RATE}")

"""## 10. Custom Accuracy Function

For label encoding, accuracy is measured by finding which symbol combination matches the predicted sign bits.
"""

def calculate_accuracy(outputs, idx_true, idx_sign):
    """
    Calculate accuracy by matching predicted sign bits to symbol indices.
    Vectorized implementation for speed.
    """
    # Apply sigmoid and threshold
    probs = torch.sigmoid(outputs)
    predicted_bits = (probs > 0.5).int()

    # Vectorized comparison: compare all predicted_bits against all idx_sign combinations
    # predicted_bits: (batch_size, 4)
    # idx_sign: (16, 4)
    # Expand dimensions for broadcasting
    predicted_bits_expanded = predicted_bits.unsqueeze(1)  # (batch_size, 1, 4)
    idx_sign_expanded = idx_sign.unsqueeze(0)  # (1, 16, 4)

    # Compare all bits and find matches
    matches = (predicted_bits_expanded == idx_sign_expanded).all(dim=2)  # (batch_size, 16)

    # Get the index of the first match for each sample (argmax finds first True)
    idx_pred = matches.to(torch.float).argmax(dim=1)  # (batch_size,)

    # Check if prediction matches ground truth
    correct = (idx_pred == idx_true).sum().item()

    return correct / len(predicted_bits)

print("Custom accuracy function defined.")

"""## 11. Training Loop"""

X_train = X_train.to(device)
y_train = y_train.to(device)
idx_train = idx_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
idx_test = idx_test.to(device)

train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

print("="*70)
print("Starting Training")
print("="*70)
print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Test Loss':<15} {'Test Acc':<15}")
print("="*70)

for epoch in range(n_epochs):
    # Training
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    train_accuracy = calculate_accuracy(outputs, idx_train, idx_sign)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_history.append(loss.item())
    train_acc_history.append(train_accuracy)

    # Testing
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_accuracy = calculate_accuracy(test_outputs, idx_test, idx_sign)

        test_loss_history.append(test_loss.item())
        test_acc_history.append(test_accuracy)

    if (epoch + 1) % 100 == 0:
        print(f"{epoch+1:<10} {loss.item():<15.4f} {train_accuracy:<15.4f} {test_loss.item():<15.4f} {test_accuracy:<15.4f}")

print("="*70)
print("Training Complete!")
print("="*70)
print(f"Final Training Loss: {train_loss_history[-1]:.4f}")
print(f"Final Training Accuracy: {train_acc_history[-1]:.4f}")
print(f"Final Test Loss: {test_loss_history[-1]:.4f}")
print(f"Final Test Accuracy: {test_acc_history[-1]:.4f}")
print("="*70)

"""## 12. Visualize Training Progress"""

epochs_range = np.arange(1, n_epochs + 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Loss curves
axes[0].plot(epochs_range, train_loss_history, linewidth=2, label='Train Loss', color='blue')
axes[0].plot(epochs_range, test_loss_history, linewidth=2, linestyle='--', label='Test Loss', color='red')
axes[0].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[0].set_title('Loss Curves - Label Encoding', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(epochs_range, train_acc_history, linewidth=2, label='Train Accuracy', color='blue')
axes[1].plot(epochs_range, test_acc_history, linewidth=2, linestyle='--', label='Test Accuracy', color='red')
axes[1].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('Accuracy Curves - Label Encoding', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_LabelEncoder_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Training curves plot saved to: training_LabelEncoder_curves.png")

print(f"\nBest Training Accuracy: {max(train_acc_history):.4f} at epoch {np.argmax(train_acc_history)+1}")
print(f"Best Test Accuracy: {max(test_acc_history):.4f} at epoch {np.argmax(test_acc_history)+1}")

"""## 13. Calculate Classification Metrics"""

model.eval()
with torch.no_grad():
    final_outputs = model(X_test)
    probs = torch.sigmoid(final_outputs)
    predicted_bits = (probs > 0.5).int()

    # Vectorized conversion to symbol indices
    # predicted_bits: (test_size, 4), idx_sign: (16, 4)
    predicted_bits_expanded = predicted_bits.unsqueeze(1)  # (test_size, 1, 4)
    idx_sign_expanded = idx_sign.unsqueeze(0)  # (1, 16, 4)

    # Find matches
    matches = (predicted_bits_expanded == idx_sign_expanded).all(dim=2)  # (test_size, 16)

    # Get index of first match (or 0 if no match)
    final_predictions = matches.to(torch.float).argmax(dim=1)  # (test_size,)

y_true = idx_test.cpu().numpy()
y_pred = final_predictions.cpu().numpy()

# Metrics
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("="*70)
print("Classification Report - Label Encoding Strategy")
print("="*70)
print(f"F1-score (macro):   {f1:.4f}")
print(f"Precision (macro):  {precision:.4f}")
print(f"Recall (macro):     {recall:.4f}")
print("="*70)

"""## 14. Save Model"""

import os

model_save_path = 'modelMIMO_2x2_4QAM_LabelEncoder.pth'

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {
        'input_size': input_size,
        'hidden_size': HIDDEN_SIZE,
        'output_size': output_size,
        'learning_rate': LEARNING_RATE,
        'n_epochs': n_epochs,
        'train_split': train_split
    },
    'system_parameters': {
        'M': M,
        'Nt': Nt,
        'Nr': Nr,
        'SNR_dB': SNR_dB,
        'N': N_TRAINING_SAMPLES
    },
    'configuration': {
        'channel_mode': CHANNEL_MODE,
        'snr_mode': SNR_MODE,
        'use_zf': USE_ZF,
        'decouple_antennas': DECOUPLE_ANTENNAS,
        'use_bias': USE_BIAS
    },
    'training_history': {
        'train_loss': train_loss_history,
        'test_loss': test_loss_history,
        'train_acc': train_acc_history,
        'test_acc': test_acc_history
    },
    'final_metrics': {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'final_test_accuracy': test_acc_history[-1]
    },
    'normalization_params': {
        'mean': X_mean.item(),
        'std': X_std.item()
    },
    'idx_sign': idx_sign,  # Save for inference
    'training_channel': H_training.cpu()  # Save the training channel H for evaluation
}

torch.save(checkpoint, model_save_path)

print("="*70)
print("Model Saved Successfully!")
print("="*70)
print(f"Model saved to: {model_save_path}")
print(f"File size: {os.path.getsize(model_save_path) / 1024:.2f} KB")
print("="*70)

"""## 15. Summary

### Label Encoding Strategy Advantages:

1. **Most Compact**: Only 4 output neurons (vs. 16 for one-hot)
2. **Efficient**: Lower computational complexity
3. **Scalable**: Grows as log₂(M)×Nt (logarithmic)
4. **Direct Mapping**: Uses sign bits of constellation

### Comparison with One-Hot:

| Metric | One-Hot | Label Encoding |
|--------|---------|----------------|
| Output Size | M^Nt = 16 | log₂(M)×Nt = 4 |
| Parameters | ~1,600 | ~400 |
| Loss Function | Cross-Entropy | Binary Cross-Entropy |
| Activation | Softmax | Sigmoid |
| Performance | Slightly Better | Very Good |

### Next Steps:
- Run BER evaluation (BER_4QAM_MIMO_2x2_All.ipynb)
- Compare with one-hot encoding
- Test at different SNR values
"""