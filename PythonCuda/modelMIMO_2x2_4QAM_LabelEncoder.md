# MIMO 2x2 Deep Learning Detector - Label Encoding Strategy

## Overview

This notebook trains a Deep Learning-based MIMO detector using **label/symbol encoding** strategy. This is the most efficient encoding approach, using only **4 output neurons** instead of 16 (one-hot) or 8 (one-hot per antenna).

## Key Concept: Sign Bit Encoding

Instead of encoding full symbol combinations, this strategy encodes the **sign** of real and imaginary parts:

```
Symbol [-1+1j, +1-1j] → [1, 0, 0, 1]
                         ↑  ↑  ↑  ↑
                         |  |  |  └─ sign(Im(s2)) = negative → 1
                         |  |  └──── sign(Re(s2)) = positive → 0
                         |  └─────── sign(Im(s1)) = positive → 0
                         └────────── sign(Re(s1)) = negative → 1
```

## Architecture

### Network Structure

```
Input Layer (4 neurons)
      ↓
Hidden Layer (100 neurons) + ReLU
      ↓
Output Layer (4 neurons) + Sigmoid
```

### Key Differences from One-Hot

| Feature | One-Hot | Label Encoding |
|---------|---------|----------------|
| **Output Size** | M^Nt = 16 | log₂(M)×Nt = 4 |
| **Total Parameters** | ~1,700 | ~500 |
| **Loss Function** | Cross-Entropy | Binary Cross-Entropy |
| **Output Activation** | Softmax | Sigmoid |
| **Encoding** | One class per combination | 4 independent bits |

## Training Parameters

```python
N = 10,000                    # Training samples
M = 4                         # 4-QAM
Nt = 2, Nr = 2               # 2x2 MIMO
SNR = 3 dB                   # Training SNR
hidden_size = 100            # Hidden neurons
output_size = 4              # Sign bits
learning_rate = 0.01         # SGD learning rate
n_epochs = 2000              # Training epochs
```

## Label Format

For 2×2 MIMO with 4-QAM, the label is a 4-bit vector:

```python
Label = [sign(Re(s1)), sign(Im(s1)), sign(Re(s2)), sign(Im(s2))]

where sign(x) = {
    1  if x < 0 (negative)
    0  if x ≥ 0 (positive)
}
```

### Example Encodings

```
Symbol Combination         Sign Encoding
[+1+1j, +1+1j]      →     [0, 0, 0, 0]
[-1-1j, -1-1j]      →     [1, 1, 1, 1]
[-1+1j, +1-1j]      →     [1, 0, 0, 1]
[+1-1j, -1+1j]      →     [0, 1, 1, 0]
```

## Advantages

### 1. Most Compact Representation
- **Smallest output layer**: 4 neurons vs. 16 (one-hot)
- **Fewer parameters**: ~500 vs. ~1,700
- **Lower memory**: Reduced model size

### 2. Best Scalability
- **Logarithmic growth**: O(log₂(M) × Nt)
- For 16-QAM: 4×2 = 8 outputs (vs. 256 one-hot)
- For 64-QAM: 6×2 = 12 outputs (vs. 4,096 one-hot)

### 3. Computational Efficiency
- **Faster inference**: Fewer computations
- **Lower complexity**: O(4×100 + 100×4) = O(800)
- **Better for embedded systems**: Small footprint

### 4. Direct Bit-Level Mapping
- **Natural representation**: Uses actual constellation structure
- **Interpretable**: Each output is a physical property (sign)

## Disadvantages

### 1. Slight Performance Loss
- Typically 0.5-2 dB gap from ML detector
- Slightly worse than one-hot encoding
- May have more errors in high SNR

### 2. Decoding Step Required
- Must match bit pattern to symbol index
- If no exact match, needs fallback logic
- More complex inference code

### 3. Bit Errors vs. Symbol Errors
- One wrong bit → wrong symbol
- No partial correctness
- All-or-nothing detection

## Usage

### Training

```bash
jupyter notebook training_2x2_detector_LabelEncoder.ipynb
# Run all cells
```

### Loading Trained Model

```python
import torch

checkpoint = torch.load('modelMIMO_2x2_4QAM_LabelEncoder_3dB_pytorch.pth')

model = MIMO_Detector_LabelEncoder(
    input_size=4,
    hidden_size=100,
    output_size=4
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get sign encoding matrix
idx_sign = checkpoint['idx_sign']
```

### Inference

```python
# Prepare input
x_input = torch.tensor([real_r1, imag_r1, real_r2, imag_r2])

# Forward pass
outputs = model(x_input.unsqueeze(0))

# Apply sigmoid and threshold
probs = torch.sigmoid(outputs)
predicted_bits = (probs > 0.5).int()

# Find matching symbol combination
matches = (idx_sign == predicted_bits[0]).all(dim=1)
symbol_idx = torch.where(matches)[0]

if len(symbol_idx) > 0:
    detected_symbol = symbol_idx[0].item()
else:
    # Fallback: find closest match
    distances = (idx_sign.float() - predicted_bits[0].float()).abs().sum(dim=1)
    detected_symbol = torch.argmin(distances).item()
```

## Training Process

### 1. Data Generation
- Generate random symbol combinations
- Pass through Rayleigh fading channel H ~ CN(0,1)
- Transmit: r = √SNR · H · x + n
- Apply Zero-Forcing equalization: r_eq = H⁺ · r
- Extract features: [Re(r_eq), Im(r_eq)]
- Extract sign bits as labels

### 2. Network Training
- **Loss**: Binary Cross-Entropy (BCEWithLogitsLoss)
- **Optimizer**: SGD (vanilla, no momentum)
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Epochs**: 2000

### 3. Evaluation
- Calculate accuracy via bit pattern matching
- Compute precision, recall, F1-score
- Generate loss and accuracy curves

## Expected Performance

### At SNR = 3 dB (Training)

```
Training Accuracy:  95-98%
Test Accuracy:      94-97%
F1-Score:           0.94-0.97
```

### BER Performance (vs. One-Hot)

```
SNR (dB)    One-Hot BER    Label Enc BER    Gap
   10          1e-2            1.5e-2       +0.5 dB
   15          1e-3            2e-3         +0.7 dB
   20          1e-4            3e-4         +1.0 dB
```

**Note**: Exact values depend on training and channel conditions.

## When to Use Label Encoding

### Best For:
✅ **Large-scale systems**: 4×4, 8×8 MIMO
✅ **High modulation**: 16-QAM, 64-QAM
✅ **Resource-constrained**: Embedded devices
✅ **Real-time processing**: Low-latency requirements

### Not Ideal For:
❌ **Small systems**: 2×2 QPSK (overhead not worth it)
❌ **Maximum accuracy**: Use one-hot instead
❌ **Simple implementation**: One-hot is easier

## Comparison with Other Strategies

### Label Encoding vs. One-Hot

| Metric | Label Encoding | One-Hot |
|--------|----------------|---------|
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Scalability** | ⭐⭐⭐⭐⭐ | ⭐ |
| **Simplicity** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Output Size Comparison

```
System          One-Hot    Label Enc    Reduction
2×2 4-QAM         16           4          75%
2×2 16-QAM        256         8          97%
4×4 4-QAM         256         16         94%
4×4 16-QAM        65,536      32         99.95%
```

## Files Generated

After training:

1. **modelMIMO_2x2_4QAM_LabelEncoder_3dB_pytorch.pth**
   - Trained model weights
   - Hyperparameters
   - Training history
   - Sign encoding matrix (idx_sign)

2. **Loss and accuracy plots**
   - Training curves
   - Convergence visualization

## Troubleshooting

### Issue: Low Accuracy (<90%)

**Possible causes**:
- Incorrect sign bit encoding
- Wrong activation function (should be Sigmoid)
- Label mismatch

**Solutions**:
```python
# Verify sign encoding
print(idx_sign[:5])

# Check output activation
print(torch.sigmoid(model(X_train[:1])))

# Verify labels match data
print(y_train[:5])
```

### Issue: No Matching Symbol Found

**Cause**: Bit pattern doesn't match any combination

**Solution**: Use distance-based fallback:
```python
if len(matches) == 0:
    distances = (idx_sign.float() - predicted_bits).abs().sum(dim=1)
    symbol_idx = torch.argmin(distances)
```

### Issue: Training Unstable

**Possible causes**:
- Learning rate too high
- Wrong loss function

**Solutions**:
```python
# Reduce learning rate
learning_rate = 0.001

# Ensure correct loss
criterion = nn.BCEWithLogitsLoss()  # Not CrossEntropyLoss!
```

## Dependencies

Same as one-hot encoding:
```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

## References

1. Original paper implementing this strategy
2. Binary classification for MIMO detection
3. Efficient neural network architectures

## License

GPLv2 License

## Authors

- Roilhi Frajo Ibarra Hernández (roilhi.ibarra@uaslp.mx)
- Francisco Rubén Castillo-Soria (ruben.soria@uaslp.mx)

---

**Last Updated**: 2025
**Version**: 1.0.0
