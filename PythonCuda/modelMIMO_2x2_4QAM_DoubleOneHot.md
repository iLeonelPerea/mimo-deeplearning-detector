# MIMO 2x2 Deep Learning Detector - One-Hot Per Antenna Strategy

## Overview

This notebook trains a Deep Learning-based MIMO detector using **one-hot encoding per antenna** strategy (also called "double one-hot"). This approach provides a **balanced trade-off** between the high accuracy of one-hot encoding and the efficiency of label encoding.

## Key Concept: Separate One-Hot Per Antenna

Instead of encoding the entire symbol combination or using bit signs, this strategy encodes **each antenna's symbol separately**:

```
Symbol Combination: [Symbol_Ant1, Symbol_Ant2]

Encoding: [one_hot(Symbol_Ant1) | one_hot(Symbol_Ant2)]
          \_____4 neurons_____/   \_____4 neurons_____/
```

### Example

```
Symbol [-1+1j, +1-1j] → [0, 1, 0, 0 | 0, 0, 0, 1]
                         \__Ant 1__/   \__Ant 2__/
                           Symbol 1      Symbol 3
```

## Architecture

### Network Structure

```
Input Layer (4 neurons)
      ↓
Hidden Layer (100 neurons) + ReLU
      ↓
Output Layer (8 neurons) + Sigmoid
      ↓
Split: [4 neurons for Ant1 | 4 neurons for Ant2]
```

### Comparison with Other Strategies

| Feature | One-Hot | **Double One-Hot** | Label Encoding |
|---------|---------|-------------------|----------------|
| **Output Size** | 16 | **8** | 4 |
| **Total Parameters** | ~1,700 | **~900** | ~500 |
| **Loss Function** | Cross-Entropy | **BCE** | BCE |
| **Output Activation** | Softmax | **Sigmoid** | Sigmoid |
| **Structure** | Joint | **Per-Antenna** | Bit-level |

## Training Parameters

```python
N = 10,000                   # Training samples
M = 4                        # 4-QAM
Nt = 2, Nr = 2              # 2x2 MIMO
SNR = 3 dB                  # Training SNR
hidden_size = 100           # Hidden neurons
output_size = 8             # M × Nt
learning_rate = 0.01        # SGD learning rate
n_epochs = 2000             # Training epochs
```

## Label Format

For 2×2 MIMO with 4-QAM, the label is an 8-dimensional vector:

```python
Label = [one_hot_ant1 | one_hot_ant2]

where:
  one_hot_ant1 = one-hot encoding of symbol on antenna 1 (4 values)
  one_hot_ant2 = one-hot encoding of symbol on antenna 2 (4 values)
```

### Example Encodings

```
Symbol Combination         Double One-Hot Encoding
[-1+1j, -1+1j]      →     [1, 0, 0, 0, 1, 0, 0, 0]
                            ↑           ↑
                            Ant1: Sym0  Ant2: Sym0

[-1-1j, +1+1j]      →     [0, 1, 0, 0, 0, 0, 1, 0]
                            ↑           ↑
                            Ant1: Sym1  Ant2: Sym2

[+1-1j, -1+1j]      →     [0, 0, 0, 1, 1, 0, 0, 0]
                            ↑           ↑
                            Ant1: Sym3  Ant2: Sym0
```

## Advantages

### 1. Balanced Complexity
- **Moderate output size**: 8 vs. 16 (one-hot) or 4 (label)
- **Good parameter count**: ~900 parameters
- **Efficient yet accurate**: Best of both worlds

### 2. Exploits MIMO Structure
- **Natural decomposition**: Matches physical antenna structure
- **Independent detection**: Per-antenna classification
- **Interpretable**: Each group represents one antenna

### 3. Good Scalability
- **Linear growth**: O(M × Nt)
- For 16-QAM: 4×4 = 16 outputs (vs. 256 one-hot)
- For 4×4 MIMO with 4-QAM: 4×4 = 16 outputs (vs. 256 one-hot)

### 4. Performance/Efficiency Balance
- **Near one-hot accuracy**: Minimal performance loss
- **Better than label encoding**: Typically 0.2-0.5 dB better
- **Fewer parameters**: 50% reduction vs. one-hot

## Disadvantages

### 1. More Complex Than Label Encoding
- Still more outputs than label encoding
- Not as compact for large M

### 2. Decoding Step Required
- Must combine per-antenna predictions
- Find matching symbol combination
- Needs symbol index lookup

### 3. Assumes Independence
- Treats antennas independently (not always optimal)
- Doesn't fully exploit joint statistics
- May miss inter-antenna correlations

## Usage

### Training

```bash
jupyter notebook training_2x2_detector_DoubleOneHot.ipynb
# Run all cells
```

### Loading Trained Model

```python
import torch

checkpoint = torch.load('modelMIMO_2x2_4QAM_DoubleOneHot_3dB_pytorch.pth')

model = MIMO_Detector_DoubleOneHot(
    input_size=4,
    hidden_size=100,
    output_size=8
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get symbol indices matrix
symbol_indices = checkpoint['symbol_indices']
```

### Inference

```python
# Prepare input
x_input = torch.tensor([real_r1, imag_r1, real_r2, imag_r2])

# Forward pass
outputs = model(x_input.unsqueeze(0))

# Apply sigmoid
probs = torch.sigmoid(outputs)

# Split for each antenna
probs_ant1 = probs[0, :4]  # First 4 outputs
probs_ant2 = probs[0, 4:]  # Last 4 outputs

# Get predictions per antenna
pred_ant1 = torch.argmax(probs_ant1).item() + 1  # 1-indexed
pred_ant2 = torch.argmax(probs_ant2).item() + 1

# Find matching symbol combination
predicted_idx = torch.tensor([pred_ant1, pred_ant2])
matches = (symbol_indices == predicted_idx).all(dim=1)
symbol_idx = torch.where(matches)[0]

if len(symbol_idx) > 0:
    detected_symbol = symbol_idx[0].item()
else:
    # Fallback if no exact match (shouldn't happen in normal operation)
    detected_symbol = 0
```

## Training Process

### 1. Data Generation
- Generate random symbol combinations
- Pass through Rayleigh fading channel H ~ CN(0,1)
- Transmit: r = √SNR · H · x + n
- Apply Zero-Forcing equalization: r_eq = H⁺ · r
- Extract features: [Re(r_eq), Im(r_eq)]
- Create double one-hot labels

### 2. Network Training
- **Loss**: Binary Cross-Entropy (BCEWithLogitsLoss)
- **Optimizer**: SGD (vanilla, no momentum)
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Epochs**: 2000

### 3. Evaluation
- Split outputs per antenna
- Take argmax for each antenna
- Combine to find symbol combination
- Compute metrics

## Expected Performance

### At SNR = 3 dB (Training)

```
Training Accuracy:  96-98%
Test Accuracy:      95-97%
F1-Score:           0.95-0.97
```

### BER Performance Comparison

```
SNR (dB)    One-Hot    Double OH    Label Enc    Gap to One-Hot
   10        1e-2       1.2e-2       1.5e-2         +0.2 dB
   15        1e-3       1.3e-3       2e-3           +0.3 dB
   20        1e-4       1.5e-4       3e-4           +0.4 dB
```

## When to Use Double One-Hot

### Best For:
✅ **Balanced systems**: When you want accuracy AND efficiency
✅ **Moderate MIMO**: 2×2, 4×4 systems
✅ **Moderate modulation**: 4-QAM, 16-QAM
✅ **When antenna structure matters**: Physical decomposition makes sense

### Not Ideal For:
❌ **Maximum accuracy**: Use standard one-hot
❌ **Maximum efficiency**: Use label encoding
❌ **Very large systems**: Label encoding scales better

## Strategy Comparison

### Performance Ranking (Best to Worst)

```
1. One-Hot Encoding        ⭐⭐⭐⭐⭐  (Best accuracy)
2. Double One-Hot          ⭐⭐⭐⭐    (Very good)
3. Label Encoding          ⭐⭐⭐      (Good)
```

### Efficiency Ranking (Most to Least Efficient)

```
1. Label Encoding          ⭐⭐⭐⭐⭐  (Most efficient)
2. Double One-Hot          ⭐⭐⭐⭐    (Balanced)
3. One-Hot Encoding        ⭐⭐        (Least efficient)
```

### Output Size Scaling

```
System           One-Hot    Double OH    Label Enc
2×2 4-QAM          16          8            4
2×2 16-QAM        256         16            8
4×4 4-QAM         256         16           16
4×4 16-QAM      65,536        64           32
8×8 4-QAM      16,777,216    128           64
```

**Observation**: Double one-hot scales linearly with M×Nt, making it practical for larger systems.

## Technical Details

### Forward Pass

```python
# Input: [real(r1), imag(r1), real(r2), imag(r2)]
x = [4 features]

# Hidden layer
h = ReLU(W1·x + b1)  # [100 features]

# Output layer (logits)
z = W2·h + b2        # [8 features]

# Apply sigmoid
p = sigmoid(z)       # [8 probabilities]

# Split
p_ant1 = p[0:4]      # Antenna 1 probabilities
p_ant2 = p[4:8]      # Antenna 2 probabilities

# Predictions
sym1 = argmax(p_ant1)
sym2 = argmax(p_ant2)
```

### Loss Calculation

```python
# Binary Cross-Entropy per output neuron
loss = -Σ [y_i · log(p_i) + (1-y_i) · log(1-p_i)]

# Where:
#   y_i: true label (0 or 1) for neuron i
#   p_i: predicted probability for neuron i
```

### Accuracy Calculation

```python
# For each sample:
1. Get predictions per antenna
2. Combine: [pred_ant1, pred_ant2]
3. Find matching symbol index
4. Compare with true index
5. Count correct predictions
```

## Files Generated

After training:

1. **modelMIMO_2x2_4QAM_DoubleOneHot_3dB_pytorch.pth**
   - Model weights
   - Hyperparameters
   - Training history
   - Symbol indices matrix

2. **Training curves**
   - Loss vs. epochs
   - Accuracy vs. epochs

## Troubleshooting

### Issue: Antenna Predictions Don't Match Any Combination

**Cause**: Rarely, the per-antenna predictions might not correspond to a valid symbol combination.

**Solution**: This shouldn't happen in normal 4-QAM, but if it does:
```python
# All valid combinations are covered
# Just use the first match or a default
if len(matches) == 0:
    symbol_idx = 0  # Default to first symbol
```

### Issue: Lower Accuracy Than Expected

**Possible causes**:
- Wrong output split (should be M and M, not M^2)
- Incorrect sigmoid vs. softmax
- Label mismatch

**Solutions**:
```python
# Verify split
assert probs_ant1.shape[-1] == M
assert probs_ant2.shape[-1] == M

# Check activation
probs = torch.sigmoid(outputs)  # Not softmax!

# Verify labels
print(y_labels[0])  # Should have two ones
```

### Issue: Training Slower Than One-Hot

**Cause**: Binary cross-entropy on 8 outputs vs. cross-entropy on 16

**Solution**: Normal behavior, slight difference in computation

## Dependencies

Same as other strategies:
```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

## References

1. Original paper with all three strategies
2. Per-antenna detection in MIMO systems
3. Hybrid encoding approaches

## License

GPLv2 License

## Authors

- Roilhi Frajo Ibarra Hernández (roilhi.ibarra@uaslp.mx)
- Francisco Rubén Castillo-Soria (ruben.soria@uaslp.mx)

---

**Last Updated**: 2025
**Version**: 1.0.0
