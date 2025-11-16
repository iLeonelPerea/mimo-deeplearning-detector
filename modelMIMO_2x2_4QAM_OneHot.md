# MIMO 2x2 Deep Learning Detector - One-Hot Encoding Strategy

## Overview

This notebook implements a **Deep Learning-based detector** for a **2×2 MIMO (Multiple-Input Multiple-Output)** communication system using a **one-hot encoding** labeling strategy. The detector leverages a neural network to perform symbol detection in the presence of Rayleigh fading channels and additive white Gaussian noise (AWGN).

## Table of Contents

- [Introduction](#introduction)
- [System Model](#system-model)
- [Neural Network Architecture](#neural-network-architecture)
- [Dataset Generation](#dataset-generation)
- [Training Procedure](#training-procedure)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [References](#references)
- [License](#license)

---

## Introduction

### Motivation

In MIMO communication systems, the receiver needs to detect transmitted symbols from multiple antennas simultaneously. Traditional detection methods like **Maximum Likelihood (ML)** detection have exponential complexity O(M^Nt), making them computationally expensive for high-order modulations and large antenna arrays.

**Deep Learning-based detection** offers a promising alternative with:
- Reduced computational complexity
- Near-optimal performance
- Adaptability to channel conditions
- Scalability to larger systems

### Problem Statement

Given:
- **Nt = 2** transmit antennas
- **Nr = 2** receive antennas
- **4-QAM** modulation (M = 4)
- **Rayleigh fading** channel
- **AWGN** noise

**Goal**: Train a neural network to correctly classify the transmitted symbol combination from the received signal.

---

## System Model

### MIMO Channel Model

The received signal vector **r** is given by:

```
r = H·x + n
```

Where:
- **r** ∈ ℂ^(Nr×1): Received signal vector
- **H** ∈ ℂ^(Nr×Nt): Channel matrix (Rayleigh fading)
- **x** ∈ ℂ^(Nt×1): Transmitted symbol vector
- **n** ∈ ℂ^(Nr×1): AWGN noise vector

### Channel Model and Equalization

The received signal is generated using the standard MIMO model:

```
r = √SNR · H · x + n
```

Before feeding to the neural network, **Zero-Forcing (ZF)** equalization is applied:

```
r_eq = H⁺ · r
```

Where **H⁺** is the Moore-Penrose pseudo-inverse of **H**.

This gives us:
```
r_eq = H⁺ · (√SNR · H · x + n)
```

### Signal Processing

The complex-valued received signal is converted to real-valued features:

```
Input = [Re(r̃₁), Im(r̃₁), Re(r̃₂), Im(r̃₂)]
```

This creates a 4-dimensional real-valued input vector.

---

## Neural Network Architecture

### Network Structure

```
Input Layer (4 neurons)
      ↓
Hidden Layer (100 neurons) + ReLU
      ↓
Output Layer (16 neurons) + Softmax
```

### Layer Details

| Layer | Input Size | Output Size | Activation | Parameters |
|-------|-----------|-------------|------------|------------|
| Linear 1 | 4 | 100 | ReLU | 500 |
| Linear 2 | 100 | 16 | Softmax | 1,616 |

**Total Parameters**: 2,116

### Weight Initialization

- **Method**: Xavier/Glorot Uniform Initialization
- **Bias**: Initialized to zero

Xavier initialization ensures proper gradient flow during training:

```
W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]
```

---

## Dataset Generation

### Training Data Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **N** | 10,000 | Total number of samples |
| **M** | 4 | Modulation order (4-QAM) |
| **Nt** | 2 | Number of TX antennas |
| **Nr** | 2 | Number of RX antennas |
| **SNR** | 3 dB | Signal-to-Noise Ratio |
| **Train/Test Split** | 80/20 | 8,000 train / 2,000 test |

### 4-QAM Constellation

The 4-QAM constellation consists of 4 complex symbols:

```
Symbol 0: -1 + 1j    Symbol 1: -1 - 1j
Symbol 2:  1 + 1j    Symbol 3:  1 - 1j
```

### Symbol Combinations

For 2 transmit antennas with 4-QAM, there are **M^Nt = 4² = 16** possible symbol combinations:

```
[Symbol_Antenna1, Symbol_Antenna2]
```

Examples:
- Combination 0: [-1+1j, -1+1j]
- Combination 5: [-1-1j, -1-1j]
- Combination 10: [1+1j, 1+1j]
- Combination 15: [1-1j, 1-1j]

### Data Generation Pipeline

1. **Random Selection**: Randomly select a symbol combination
2. **Channel Generation**: Generate Rayleigh fading channel H ~ CN(0, 1)
3. **Noise Generation**: Generate AWGN noise n ~ CN(0, 1/SNR)
4. **Signal Transmission**: Compute r = √SNR · H · x + n
5. **Equalization**: Apply r_eq = H⁺ · r (Zero-Forcing)
6. **Feature Extraction**: Extract [Re(r_eq₁), Im(r_eq₁), Re(r_eq₂), Im(r_eq₂)]
7. **Normalization**: Apply z-score normalization

### One-Hot Encoding

Each of the 16 symbol combinations is encoded as a one-hot vector:

```
Combination 0  → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Combination 5  → [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Combination 15 → [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

---

## Training Procedure

### Hyperparameters

```python
learning_rate = 0.01        # Learning rate (α)
n_epochs = 2000             # Number of training epochs
batch_size = full_batch     # Full-batch gradient descent
optimizer = SGD             # Stochastic Gradient Descent
momentum = 0                # Vanilla SGD
```

### Loss Function

**Cross-Entropy Loss** is used for multi-class classification:

```
L = -∑ y_true · log(y_pred)
```

This loss function is appropriate for one-hot encoded labels and combines:
- Softmax activation
- Negative log-likelihood

### Training Algorithm

```
For each epoch:
    1. Forward Pass:
        - Compute Z₁ = W₁·X + b₁
        - Apply A₁ = ReLU(Z₁)
        - Compute Z₂ = W₂·A₁ + b₂
        - Apply Softmax to get predictions

    2. Loss Calculation:
        - Compute CrossEntropyLoss(predictions, labels)

    3. Backward Pass:
        - Compute gradients via autograd

    4. Weight Update:
        - W ← W - α·∇W
        - b ← b - α·∇b

    5. Validation:
        - Evaluate on test set
        - Record metrics
```

### Learning Rate Schedule

- **Fixed learning rate**: α = 0.01 throughout training
- No learning rate decay or scheduling

---

## Evaluation Metrics

### Classification Metrics

The model is evaluated using standard classification metrics:

#### 1. **Accuracy**
```
Accuracy = (Number of Correct Predictions) / (Total Predictions)
```

#### 2. **Precision (Macro Average)**
```
Precision = (1/N_classes) · ∑ (TP_i / (TP_i + FP_i))
```

#### 3. **Recall (Macro Average)**
```
Recall = (1/N_classes) · ∑ (TP_i / (TP_i + FN_i))
```

#### 4. **F1-Score (Macro Average)**
```
F1 = (1/N_classes) · ∑ (2 · Precision_i · Recall_i / (Precision_i + Recall_i))
```

### Confusion Matrix

A 16×16 confusion matrix visualizes the classification performance across all symbol combinations:

- **Diagonal elements**: Correctly classified samples
- **Off-diagonal elements**: Misclassifications

---

## Usage

### Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install torch numpy matplotlib scikit-learn seaborn
```

### Running the Notebook

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook training_2x2_detector_OneHot.ipynb
   ```

2. **Execute All Cells**: Run all cells sequentially from top to bottom

3. **Monitor Training**: Observe training progress printed every 100 epochs

### Loading a Trained Model

```python
import torch

# Load checkpoint
checkpoint = torch.load('modelMIMO_2x2_4QAM_OneHot_3dB_pytorch.pth')

# Create model instance
model = MIMO_Detector(
    input_size=checkpoint['hyperparameters']['input_size'],
    hidden_size=checkpoint['hyperparameters']['hidden_size'],
    output_size=checkpoint['hyperparameters']['output_size']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Using the Model for Inference

```python
# Normalize input data
X_normalized = (X_new - checkpoint['normalization_params']['mean']) / \
               checkpoint['normalization_params']['std']

# Move to device
X_normalized = X_normalized.to(device)

# Get predictions
with torch.no_grad():
    outputs = model(X_normalized)
    _, predictions = torch.max(outputs, 1)

print(f"Predicted symbol combination: {predictions}")
```

---

## Results

### Expected Performance

After 2000 epochs of training at SNR = 3 dB, the model typically achieves:

| Metric | Expected Range |
|--------|----------------|
| **Training Accuracy** | 95-99% |
| **Test Accuracy** | 95-98% |
| **F1-Score (Macro)** | 0.95-0.98 |
| **Precision (Macro)** | 0.95-0.98 |
| **Recall (Macro)** | 0.95-0.98 |

### Training Curves

**Loss Curves**: Both training and test loss should decrease monotonically and converge to low values (< 0.2).

**Accuracy Curves**: Both training and test accuracy should increase and plateau at high values (> 95%).

### Confusion Matrix Characteristics

- **Strong diagonal**: High values along the diagonal indicate correct classifications
- **Weak off-diagonal**: Low values elsewhere indicate few misclassifications
- **Symmetric patterns**: May reveal similar symbol combinations that are harder to distinguish

---

## Dependencies

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | ≥ 3.7 | Programming language |
| **PyTorch** | ≥ 1.8.0 | Deep learning framework |
| **NumPy** | ≥ 1.19.0 | Numerical computations |
| **Matplotlib** | ≥ 3.3.0 | Visualization |
| **Scikit-learn** | ≥ 0.24.0 | Metrics and evaluation |
| **Seaborn** | ≥ 0.11.0 | Statistical visualization |

### Installation

Install all dependencies using pip:

```bash
pip install torch numpy matplotlib scikit-learn seaborn jupyter
```

For GPU support (CUDA):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## File Structure

```
Pytorch/
│
├── training_2x2_detector_OneHot.ipynb    # Main notebook
├── training_2x2_detector_OneHot.md       # This documentation
└── modelMIMO_2x2_4QAM_OneHot_3dB_pytorch.pth  # Saved model (after training)
```

---

## Model Checkpoint Contents

The saved model file contains:

```python
{
    'model_state_dict': {},           # Trained weights and biases
    'optimizer_state_dict': {},       # Optimizer state
    'hyperparameters': {              # Network architecture
        'input_size': 4,
        'hidden_size': 100,
        'output_size': 16,
        'learning_rate': 0.01,
        'n_epochs': 2000,
        'train_split': 0.8
    },
    'system_parameters': {            # MIMO system config
        'M': 4,
        'Nt': 2,
        'Nr': 2,
        'SNR_dB': 3,
        'N': 10000
    },
    'training_history': {             # Training metrics
        'train_loss': [...],
        'test_loss': [...],
        'train_acc': [...],
        'test_acc': [...]
    },
    'final_metrics': {                # Performance metrics
        'f1_score': 0.xx,
        'precision': 0.xx,
        'recall': 0.xx,
        'final_test_accuracy': 0.xx
    },
    'normalization_params': {         # For inference
        'mean': x.xx,
        'std': x.xx
    }
}
```

---

## Key Concepts

### Why One-Hot Encoding?

**Advantages**:
- ✅ Simple and intuitive representation
- ✅ Standard for multi-class classification
- ✅ Compatible with softmax output
- ✅ Well-established training procedures

**Disadvantages**:
- ❌ High dimensionality (M^Nt outputs)
- ❌ Doesn't exploit bit-level structure
- ❌ Scales poorly for large M or Nt

### Why 100 Hidden Neurons?

The hidden layer size is a trade-off:
- **Too small**: Insufficient capacity to learn complex patterns
- **Too large**: Overfitting and computational cost

100 neurons provides:
- Adequate capacity for 16-class classification
- Good generalization to test data
- Reasonable training time

### Why Full-Batch Training?

Full-batch gradient descent (using entire dataset) offers:
- **Stable gradients**: Less noise in gradient estimates
- **Reproducibility**: Deterministic training process
- **Simplicity**: No mini-batch sampling needed

Trade-offs:
- Higher memory usage
- No regularization from mini-batch noise
- Slower convergence for very large datasets

---

## Computational Complexity

### Training Complexity

**Forward Pass**:
```
O(input_size × hidden_size + hidden_size × output_size)
= O(4 × 100 + 100 × 16)
= O(2,000) operations per sample
```

**Backward Pass**: Similar complexity

**Per Epoch**: O(2,000 × N) = O(20,000,000) for N=10,000

### Inference Complexity

**Per Prediction**: O(2,000) operations

**Comparison with ML Detection**:
- ML Detection: O(Nr × Nt × M^Nt) = O(2 × 2 × 16) = O(64)
- **DL Detection**: O(2,000)

**Note**: While DL has higher per-sample complexity, it's:
1. Highly parallelizable (GPU acceleration)
2. Does not grow exponentially with M or Nt
3. Can be optimized with quantization/pruning

---

## Extensions and Future Work

### Possible Improvements

1. **Different Labeling Strategies**:
   - Symbol encoding (log₂(M) × Nt outputs)
   - One-hot per antenna (M × Nt outputs)
   - Hybrid approaches

2. **Larger Systems**:
   - 4×4 MIMO
   - 8×8 MIMO
   - Massive MIMO

3. **Higher Modulation Orders**:
   - 16-QAM (M=16)
   - 64-QAM (M=64)

4. **Advanced Architectures**:
   - Deeper networks (3-4 layers)
   - Residual connections
   - Attention mechanisms

5. **Training Enhancements**:
   - Mini-batch training
   - Learning rate scheduling
   - Data augmentation
   - Dropout regularization

6. **BER Analysis**:
   - Evaluate at multiple SNR values
   - Compare with theoretical bounds
   - Generate BER curves

7. **Real-World Channels**:
   - Time-varying channels
   - Correlated fading
   - Non-Gaussian noise

---

## Troubleshooting

### Common Issues

#### 1. **GPU Out of Memory**

**Solution**: Reduce batch size or move to CPU
```python
device = 'cpu'  # Force CPU usage
```

#### 2. **Poor Convergence**

**Solutions**:
- Check learning rate (try 0.001 or 0.1)
- Verify data normalization
- Increase number of epochs
- Check for NaN/Inf values

#### 3. **Overfitting**

**Symptoms**: High train accuracy, low test accuracy

**Solutions**:
- Increase training data (N)
- Add dropout layers
- Use weight decay
- Early stopping

#### 4. **Underfitting**

**Symptoms**: Low train and test accuracy

**Solutions**:
- Increase hidden layer size
- Add more layers
- Train for more epochs
- Reduce regularization

---

## References

### Academic Papers

1. **Original Work**:
   - Ibarra-Hernández, R.F.; Castillo-Soria, F.R.; Gutiérrez, C.A.; Del-Puerto-Flores, J.A.; Acosta-Elías J., Rodríguez-Abdalá V. and Palacios-Luengas L.
   - "Efficient Deep Learning-Based Detection Scheme for MIMO Communication System"
   - Submitted to the Journal Sensors of MDPI

2. **Deep Learning for MIMO Detection**:
   - Samuel, N., Diskin, T., & Wiesel, A. (2017). "Deep MIMO detection." IEEE 18th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC).

3. **Neural Network Basics**:
   - Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press.

### Online Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **MIMO Systems**: https://en.wikipedia.org/wiki/MIMO
- **QAM Modulation**: https://en.wikipedia.org/wiki/Quadrature_amplitude_modulation

---

## License

This code is licensed under the **GPLv2 license**.

### Citation

If you use this code for research that results in publications, please cite:

```bibtex
@article{ibarra2024efficient,
  title={Efficient Deep Learning-Based Detection Scheme for MIMO Communication System},
  author={Ibarra-Hern{\'a}ndez, Roilhi Frajo and Castillo-Soria, Francisco Rub{\'e}n and Guti{\'e}rrez, Carlos A and Del-Puerto-Flores, Jos{\'e} A and Acosta-El{\'i}as, J and Rodr{\'i}guez-Abdal{\'a}, V and Palacios-Luengas, L},
  journal={Sensors},
  publisher={MDPI},
  year={2024}
}
```

---

## Authors

- **Roilhi Frajo Ibarra Hernández** - roilhi.ibarra@uaslp.mx
- **Francisco Rubén Castillo-Soria** - ruben.soria@uaslp.mx

**Affiliation**: Universidad Autónoma de San Luis Potosí (UASLP)

---

## Contact

For questions, issues, or contributions:

- Open an issue in the repository
- Contact the authors via email
- Refer to the main project documentation

---

## Acknowledgments

This work was supported by the research team at UASLP and contributes to the advancement of efficient MIMO detection schemes using deep learning techniques.

---

**Last Updated**: 2025

**Version**: 1.0.0
