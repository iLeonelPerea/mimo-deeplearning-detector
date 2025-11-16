# -*- coding: utf-8 -*-
"""modelMIMO_2x2_4QAM_OneHot.py

# MIMO 2x2 Deep Learning Detector Training - One-Hot Encoding Strategy

## Description
This script implements a Deep Learning-based detector for a 2x2 MIMO communication system
using **one-hot encoding** labeling strategy (M^Nt = 16 output neurons). The implementation
uses PyTorch with optimized training procedures.

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

# Import necessary libraries
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

"""## 2. Define System Parameters

Configure the MIMO system parameters:
- **M**: Modulation order (4-QAM)
- **Nt**: Number of transmit antennas (2)
- **Nr**: Number of receive antennas (2)
- **N**: Number of training samples (10,000)
- **SNR**: Signal-to-Noise Ratio in dB for training (3 dB)
"""

# =====================================
# MIMO System Parameters
# =====================================
N = 100000          # Number of training samples (as in LatinCom paper, Table II)
M = 4               # Modulation order (4-QAM)
Nt = 2              # Number of transmit antennas
Nr = 2              # Number of receive antennas
SNR_dB = 3          # SNR in dB for training data (NOTE: Random SNR 1-20 dB is used per sample)
SNR_linear = 10**(SNR_dB/10)  # SNR in linear scale
No = 1              # Noise power spectral density

# =====================================
# Neural Network Hyperparameters
# =====================================
input_size = 2 * Nr      # Input features: [real(r1), imag(r1), real(r2), imag(r2)]
hidden_size = 100        # Number of hidden units
output_size = M ** Nt    # Output classes: M^Nt = 16 for 2x2 MIMO with 4-QAM
learning_rate = 0.01     # Learning rate (alpha)
n_epochs = 2000          # Number of training epochs
train_split = 0.8        # Train/test split ratio

print("="*60)
print("MIMO System Configuration")
print("="*60)
print(f"Modulation: {M}-QAM")
print(f"MIMO Configuration: {Nt}x{Nr}")
print(f"Training samples: {N}")
print(f"SNR for training: {SNR_dB} dB")
print(f"Total symbol combinations: {output_size}")
print(f"\nNeural Network Architecture")
print("="*60)
print(f"Input layer: {input_size} neurons")
print(f"Hidden layer: {hidden_size} neurons (ReLU activation)")
print(f"Output layer: {output_size} neurons (Softmax activation)")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {n_epochs}")
print("="*60)

"""## 3. Generate QAM Constellation

Generate the 4-QAM constellation symbols using PyTorch.
The constellation points are normalized to have unit average energy.
"""

def generate_qam_constellation(M):
    """
    Generates M-QAM constellation symbols.

    Args:
        M (int): Modulation order (must be a power of 2)

    Returns:
        torch.Tensor: Complex-valued QAM constellation symbols
    """
    if not np.log2(M).is_integer():
        raise ValueError("M must be a power of 2")

    # Generate QAM indices
    qam_idx = torch.arange(M)

    # Calculate constellation grid
    c = int(np.sqrt(M))

    # In-phase and quadrature components
    # Real part: -2*(m%c) + c-1
    # Imaginary part: 2*floor(m/c) - c+1
    real_part = -2 * (qam_idx % c) + c - 1
    imag_part = 2 * torch.floor(qam_idx.float() / c) - c + 1

    # Create complex symbols
    qam_symbols = torch.complex(real_part.float(), imag_part.float())

    return qam_symbols


# Generate 4-QAM constellation
qam_symbols = generate_qam_constellation(M)

print("4-QAM Constellation Symbols:")
print(qam_symbols)
print(f"\nSymbol format: {qam_symbols.dtype}")

# Visualize constellation
plt.figure(figsize=(6, 6))
plt.scatter(qam_symbols.real.numpy(), qam_symbols.imag.numpy(),
            s=200, c='blue', marker='o', edgecolors='black', linewidths=2)
for i, symbol in enumerate(qam_symbols):
    plt.annotate(f'{i}',
                xy=(symbol.real.item(), symbol.imag.item()),
                xytext=(5, 5), textcoords='offset points', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlabel('In-phase (Real)', fontsize=12)
plt.ylabel('Quadrature (Imaginary)', fontsize=12)
plt.title('4-QAM Constellation Diagram', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

"""## 4. Generate Cartesian Product of Symbols

Create all possible combinations of transmitted symbols for the 2x2 MIMO system.
This results in M^Nt = 4^2 = 16 possible symbol combinations.
"""

# Generate all possible combinations of symbols (Cartesian product)
# For 2 antennas and 4-QAM, we have 16 combinations
symbol_combinations = torch.tensor(
    list(product(qam_symbols.numpy(), repeat=Nt)),
    dtype=torch.complex64,
    device=device
)

print(f"Total symbol combinations: {len(symbol_combinations)}")
print(f"Shape: {symbol_combinations.shape}")
print(f"\nFirst 5 combinations:")
for i in range(5):
    print(f"  Combination {i}: {symbol_combinations[i]}")

# Extract indices for QAM symbols (0, 1, 2, 3) for each antenna
qam_idx = torch.arange(M, device=device)
symbol_indices = torch.tensor(
    list(product(qam_idx.cpu().numpy(), repeat=Nt)),
    dtype=torch.long,
    device=device
)

print(f"\nSymbol indices shape: {symbol_indices.shape}")
print(f"First 5 index combinations:")
for i in range(5):
    print(f"  Indices {i}: {symbol_indices[i]}")

"""## 5. Generate Training Data

Generate training data by:
1. Randomly selecting symbol combinations
2. Passing through Rayleigh fading channel
3. Adding AWGN noise
4. Applying channel equalization using pseudo-inverse
5. Creating one-hot encoded labels
"""

def generate_training_data(N, symbol_combinations, SNR_linear, No, Nr, Nt, device='cpu'):
    """
    Generates training data for the MIMO detector.

    Args:
        N (int): Number of samples
        symbol_combinations (torch.Tensor): All possible symbol combinations
        SNR_linear (float): SNR in linear scale
        No (float): Noise power spectral density
        Nr (int): Number of receive antennas
        Nt (int): Number of transmit antennas
        device (str): Device to use ('cpu' or 'cuda')

    Returns:
        tuple: (X_data, random_indices) where X_data contains received signals
               and random_indices are the true symbol combination indices
    """
    # Initialize data storage
    X_data = torch.zeros((N, 2*Nr), dtype=torch.float32, device=device)

    # Randomly select symbol combinations for training
    random_indices = torch.randint(0, len(symbol_combinations), (N,), device=device)

    # FIXED CHANNEL: Same as MATLAB reference (detector_ELM_2x2_all.m lines 55-56)
    # This ensures all models are trained on the same channel
    H = torch.tensor([[-0.90064 + 1j*0.43457, -0.99955 + 1j*0.029882],
                      [-0.1979 + 1j*0.98022, 0.44866 + 1j*0.8937]],
                     dtype=torch.complex64, device=device)
    H = H / torch.abs(H)  # Normalize by element-wise magnitude

    print("Generating training data...")
    print(f"Progress: 0%", end='')
    print(f"\nUsing FIXED channel H (same as MATLAB reference)")

    for i in range(N):
        # Progress indicator
        if (i + 1) % (N // 10) == 0:
            print(f"\rProgress: {int((i+1)/N*100)}%", end='')

        # Select transmitted symbol combination
        selected_symbols = symbol_combinations[random_indices[i]]

        # FIXED CHANNEL H is used for all samples (NO random channel per iteration)

        # Random SNR per sample (as in MATLAB: randi(20,1))
        SNR_dB_sample = np.random.randint(1, 21)  # Random SNR between 1-20 dB
        SNR_linear_sample = 10.0 ** (SNR_dB_sample / 10.0)

        # Generate AWGN noise: n ~ CN(0, No)
        n_real = torch.randn(Nr, device=device) * np.sqrt(No/2)
        n_imag = torch.randn(Nr, device=device) * np.sqrt(No/2)
        n = torch.complex(n_real, n_imag)
        n = n / np.sqrt(SNR_linear_sample)  # Scale noise according to random SNR

        # Received signal: r = sqrt(SNR) * H * x + n
        r_x = np.sqrt(SNR_linear_sample) * torch.matmul(H, selected_symbols) + n

        # Channel equalization using pseudo-inverse (Zero-Forcing): r_eq = H^+ * r
        H_inv = torch.linalg.pinv(H)
        r_eq = torch.matmul(H_inv, r_x)

        # Store real and imaginary parts: [real(r1), imag(r1), real(r2), imag(r2)]
        X_data[i, 0] = r_eq[0].real
        X_data[i, 1] = r_eq[0].imag
        X_data[i, 2] = r_eq[1].real
        X_data[i, 3] = r_eq[1].imag

    print("\rProgress: 100% - Complete!")

    return X_data, random_indices


# Generate training data
X_data, random_indices = generate_training_data(
    N, symbol_combinations, SNR_linear, No, Nr, Nt, device
)

print(f"\nGenerated data shape: {X_data.shape}")
print(f"Random indices shape: {random_indices.shape}")
print(f"\nFirst 3 received signals:")
print(X_data[:3])
print(f"\nCorresponding symbol indices: {random_indices[:3]}")

"""## 6. Create One-Hot Encoded Labels

Convert symbol indices to one-hot encoded vectors.
For 16 symbol combinations, each label is a 16-dimensional vector with a single '1'.
"""

# Create one-hot encoded labels
y_labels = F.one_hot(random_indices, num_classes=output_size).float()

print(f"Labels shape: {y_labels.shape}")
print(f"\nFirst 5 one-hot encoded labels:")
for i in range(5):
    print(f"Sample {i} (class {random_indices[i]}): {y_labels[i]}")

# Verify label distribution
unique, counts = torch.unique(random_indices, return_counts=True)
print(f"\nLabel distribution across {output_size} classes:")
print(f"Min count: {counts.min().item()}")
print(f"Max count: {counts.max().item()}")
print(f"Mean count: {counts.float().mean().item():.2f}")
print(f"Std count: {counts.float().std().item():.2f}")

"""## 7. Data Normalization

Normalize the input data to have zero mean and unit variance.
This improves training convergence and stability.
"""

# Normalize data: zero mean and unit variance
X_mean = X_data.mean()
X_std = X_data.std()

print("Before normalization:")
print(f"  Mean: {X_mean.item():.6f}")
print(f"  Std:  {X_std.item():.6f}")

X_data_normalized = (X_data - X_mean) / X_std

print("\nAfter normalization:")
print(f"  Mean: {X_data_normalized.mean().item():.6e}")
print(f"  Std:  {X_data_normalized.std().item():.6f}")

# Visualize data distribution before and after normalization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before normalization
axes[0].hist(X_data.cpu().numpy().flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('Data Distribution - Before Normalization', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Value', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(X_mean.item(), color='red', linestyle='--', linewidth=2, label=f'Mean = {X_mean.item():.2f}')
axes[0].legend()

# After normalization
axes[1].hist(X_data_normalized.cpu().numpy().flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title('Data Distribution - After Normalization', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Value', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
axes[1].legend()

plt.tight_layout()
plt.show()

"""## 8. Train-Test Split

Split the dataset into training (80%) and testing (20%) sets.
"""

# Calculate split index
train_qty = int(train_split * N)
test_qty = N - train_qty

# Split data
X_train = X_data_normalized[:train_qty]
y_train = y_labels[:train_qty]
idx_train = random_indices[:train_qty]

X_test = X_data_normalized[train_qty:]
y_test = y_labels[train_qty:]
idx_test = random_indices[train_qty:]

print("="*60)
print("Train-Test Split")
print("="*60)
print(f"Training samples: {train_qty} ({train_split*100:.0f}%)")
print(f"Testing samples:  {test_qty} ({(1-train_split)*100:.0f}%)")
print(f"\nTraining data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Testing labels shape: {y_test.shape}")
print("="*60)

"""## 9. Define Neural Network Architecture

Define a 2-layer fully connected neural network:
- **Input Layer**: 4 neurons (real and imaginary parts of 2 received signals)
- **Hidden Layer**: 100 neurons with ReLU activation
- **Output Layer**: 16 neurons with Softmax activation (for one-hot classification)
"""

class MIMO_Detector(nn.Module):
    """
    Deep Learning-based MIMO detector using one-hot encoding.

    Architecture:
        - Input layer: 2*Nr neurons
        - Hidden layer: hidden_size neurons with ReLU activation
        - Output layer: M^Nt neurons with Softmax activation

    The network is initialized using Xavier initialization for better convergence.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(MIMO_Detector, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

        # Xavier/Glorot initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        Biases are initialized to zero.
        """
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_size)
        """
        # Hidden layer with ReLU activation
        x = F.relu(self.layer1(x))

        # Output layer (logits, softmax applied in loss function)
        x = self.layer2(x)

        return x


# Instantiate the model
model = MIMO_Detector(input_size, hidden_size, output_size).to(device)

# Display model architecture
print("="*60)
print("Model Architecture")
print("="*60)
print(model)
print("="*60)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print("="*60)

"""## 10. Define Loss Function and Optimizer

- **Loss Function**: CrossEntropyLoss (combines Softmax and NLLLoss)
- **Optimizer**: SGD with momentum=0 (equivalent to vanilla SGD as in MATLAB)

Note: PyTorch's CrossEntropyLoss expects class indices, not one-hot vectors.
"""

# Define loss function (Cross-Entropy Loss)
# Note: CrossEntropyLoss includes softmax, so we don't apply it in the model
criterion = nn.CrossEntropyLoss()

# Define optimizer (SGD to match MATLAB implementation)
# Using SGD with momentum=0 is equivalent to vanilla SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

print("="*60)
print("Training Configuration")
print("="*60)
print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: SGD")
print(f"Learning rate: {learning_rate}")
print(f"Momentum: 0 (vanilla SGD)")
print("="*60)

"""## 11. Training Loop

Train the neural network using full-batch gradient descent.
This matches the MATLAB implementation which trains on the entire dataset at once.
"""

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
idx_train = idx_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
idx_test = idx_test.to(device)

# Initialize lists to store metrics
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

print("="*60)
print("Starting Training")
print("="*60)
print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Test Loss':<15} {'Test Acc':<15}")
print("="*60)

for epoch in range(n_epochs):
    # ===========================
    # Training Phase
    # ===========================
    model.train()

    # Forward pass
    outputs = model(X_train)

    # Calculate loss (CrossEntropyLoss expects class indices, not one-hot)
    loss = criterion(outputs, idx_train)

    # Get predictions (argmax of outputs)
    _, predicted = torch.max(outputs, 1)

    # Calculate accuracy
    train_correct = (predicted == idx_train).sum().item()
    train_accuracy = train_correct / train_qty

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store training metrics
    train_loss_history.append(loss.item())
    train_acc_history.append(train_accuracy)

    # ===========================
    # Validation/Testing Phase
    # ===========================
    model.eval()

    with torch.no_grad():
        # Forward pass on test data
        test_outputs = model(X_test)

        # Calculate test loss
        test_loss = criterion(test_outputs, idx_test)

        # Get predictions
        _, test_predicted = torch.max(test_outputs, 1)

        # Calculate test accuracy
        test_correct = (test_predicted == idx_test).sum().item()
        test_accuracy = test_correct / test_qty

        # Store test metrics
        test_loss_history.append(test_loss.item())
        test_acc_history.append(test_accuracy)

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"{epoch+1:<10} {loss.item():<15.4f} {train_accuracy:<15.4f} {test_loss.item():<15.4f} {test_accuracy:<15.4f}")

print("="*60)
print("Training Complete!")
print("="*60)
print(f"Final Training Loss: {train_loss_history[-1]:.4f}")
print(f"Final Training Accuracy: {train_acc_history[-1]:.4f}")
print(f"Final Test Loss: {test_loss_history[-1]:.4f}")
print(f"Final Test Accuracy: {test_acc_history[-1]:.4f}")
print("="*60)

"""## 12. Visualize Training Progress

Plot loss and accuracy curves for both training and testing sets.
"""

# Convert lists to numpy arrays for plotting
epochs_range = np.arange(1, n_epochs + 1)

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ===========================
# Plot Loss Curves
# ===========================
axes[0].plot(epochs_range, train_loss_history, linewidth=2, label='Train Loss', color='blue')
axes[0].plot(epochs_range, test_loss_history, linewidth=2, linestyle='--', label='Test Loss', color='red')
axes[0].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# ===========================
# Plot Accuracy Curves
# ===========================
axes[1].plot(epochs_range, train_acc_history, linewidth=2, label='Train Accuracy', color='blue')
axes[1].plot(epochs_range, test_acc_history, linewidth=2, linestyle='--', label='Test Accuracy', color='red')
axes[1].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nTraining Summary:")
print("="*60)
print(f"Best Training Accuracy: {max(train_acc_history):.4f} at epoch {np.argmax(train_acc_history)+1}")
print(f"Best Test Accuracy: {max(test_acc_history):.4f} at epoch {np.argmax(test_acc_history)+1}")
print(f"Lowest Training Loss: {min(train_loss_history):.4f} at epoch {np.argmin(train_loss_history)+1}")
print(f"Lowest Test Loss: {min(test_loss_history):.4f} at epoch {np.argmin(test_loss_history)+1}")
print("="*60)

"""## 13. Calculate Confusion Matrix and Classification Metrics

Evaluate the model performance using:
- Confusion Matrix
- Precision (macro average)
- Recall (macro average)
- F1-score (macro average)
"""

# Get final predictions on test set
model.eval()
with torch.no_grad():
    final_outputs = model(X_test)
    _, final_predictions = torch.max(final_outputs, 1)

# Move to CPU for sklearn metrics
y_true = idx_test.cpu().numpy()
y_pred = final_predictions.cpu().numpy()

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate metrics (macro average)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

# Display classification report
print("="*60)
print("Classification Report")
print("="*60)
print(f"F1-score (macro average):   {f1:.4f}")
print(f"Precision (macro average):  {precision:.4f}")
print(f"Recall (macro average):     {recall:.4f}")
print("="*60)

# Visualize confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(output_size),
            yticklabels=range(output_size),
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - MIMO Detector (One-Hot Encoding)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Calculate per-class accuracy
print("\nPer-Class Accuracy:")
print("="*60)
for i in range(output_size):
    if cm[i].sum() > 0:
        class_acc = cm[i, i] / cm[i].sum()
        print(f"Class {i:2d}: {class_acc:.4f} ({cm[i, i]:4d}/{cm[i].sum():4d})")
    else:
        print(f"Class {i:2d}: No samples")
print("="*60)

"""## 14. Save the Trained Model

Save the model weights and hyperparameters for later use.
"""

import os

# Save model state dict and training information
model_save_path = 'modelMIMO_2x2_4QAM_OneHot.pth'

# Create a dictionary with all relevant information
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'learning_rate': learning_rate,
        'n_epochs': n_epochs,
        'train_split': train_split
    },
    'system_parameters': {
        'M': M,
        'Nt': Nt,
        'Nr': Nr,
        'SNR_dB': SNR_dB,
        'N': N
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
    }
}

# Save the checkpoint
torch.save(checkpoint, model_save_path)

print("="*60)
print("Model Saved Successfully!")
print("="*60)
print(f"Model saved to: {model_save_path}")
print(f"File size: {os.path.getsize(model_save_path) / 1024:.2f} KB")
print("\nSaved components:")
print("  - Model weights and biases")
print("  - Optimizer state")
print("  - Hyperparameters")
print("  - System parameters")
print("  - Training history")
print("  - Final metrics")
print("  - Normalization parameters")
print("="*60)

"""## 15. Load and Verify Saved Model (Optional)

Demonstrate how to load the saved model and verify it produces the same results.
"""

# Load the saved model
loaded_checkpoint = torch.load(model_save_path)

# Create a new model instance
loaded_model = MIMO_Detector(
    loaded_checkpoint['hyperparameters']['input_size'],
    loaded_checkpoint['hyperparameters']['hidden_size'],
    loaded_checkpoint['hyperparameters']['output_size']
).to(device)

# Load the saved weights
loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
loaded_model.eval()

# Verify the loaded model produces the same predictions
with torch.no_grad():
    verify_outputs = loaded_model(X_test)
    _, verify_predictions = torch.max(verify_outputs, 1)

    # Check if predictions match
    predictions_match = torch.all(verify_predictions == final_predictions)

print("="*60)
print("Model Loading Verification")
print("="*60)
print(f"Predictions match: {predictions_match.item()}")
print(f"\nLoaded model info:")
print(f"  System: {loaded_checkpoint['system_parameters']['Nt']}x{loaded_checkpoint['system_parameters']['Nr']} MIMO")
print(f"  Modulation: {loaded_checkpoint['system_parameters']['M']}-QAM")
print(f"  SNR: {loaded_checkpoint['system_parameters']['SNR_dB']} dB")
print(f"  Training samples: {loaded_checkpoint['system_parameters']['N']}")
print(f"  Final test accuracy: {loaded_checkpoint['final_metrics']['final_test_accuracy']:.4f}")
print(f"  F1-score: {loaded_checkpoint['final_metrics']['f1_score']:.4f}")
print("="*60)

"""## 16. Summary and Conclusions

This notebook successfully implements a Deep Learning-based detector for a 2x2 MIMO system using one-hot encoding strategy. The implementation closely follows the MATLAB code from the paper while leveraging PyTorch's high-level API.

### Key Features:
1. **Data Generation**: Realistic MIMO channel simulation with Rayleigh fading and AWGN
2. **One-Hot Encoding**: 16-class classification for symbol detection
3. **Neural Network**: 2-layer fully connected network with ReLU activation
4. **Training**: Full-batch gradient descent with SGD optimizer
5. **Evaluation**: Comprehensive metrics including confusion matrix, precision, recall, and F1-score
6. **Model Persistence**: Complete checkpoint saving with all relevant information

### Next Steps:
- Implement other labeling strategies (symbol encoding, one-hot per antenna)
- Scale to 4x4 MIMO configuration
- Compare with Maximum Likelihood detector
- Generate BER curves for different SNR values
- Experiment with different network architectures and optimizers
"""