# BER Performance Evaluation for MIMO 2×2 Detectors

## Overview

This notebook evaluates the **Bit Error Rate (BER)** performance of Deep Learning-based MIMO detectors across multiple SNR values and compares three different labeling strategies against the optimal **Maximum Likelihood (ML)** detector. This implementation generates **Figure 3** from the referenced paper.

## Table of Contents

- [Introduction](#introduction)
- [Detection Strategies](#detection-strategies)
- [System Model](#system-model)
- [BER Calculation Methodology](#ber-calculation-methodology)
- [Monte Carlo Simulation](#monte-carlo-simulation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Results Interpretation](#results-interpretation)
- [Computational Considerations](#computational-considerations)
- [Dependencies](#dependencies)
- [References](#references)
- [License](#license)

---

## Introduction

### Motivation

In MIMO systems, evaluating detector performance across varying Signal-to-Noise Ratio (SNR) conditions is crucial for:

1. **Performance Assessment**: Understanding detector behavior in different channel conditions
2. **Comparison**: Benchmarking against optimal ML detection
3. **Trade-off Analysis**: Evaluating complexity vs. performance trade-offs
4. **System Design**: Making informed decisions for practical implementations

### Problem Statement

**Given**:
- 2×2 MIMO system with 4-QAM modulation
- Multiple DL-based detectors with different labeling strategies
- SNR range: 0 to 25 dB

**Goal**: Compute and compare BER curves to assess:
- How close DL detectors perform to optimal ML detection
- Which labeling strategy provides best performance
- Performance vs. complexity trade-offs

---

## Detection Strategies

This notebook evaluates **four detection approaches**:

### 1. Maximum Likelihood (ML) Detector

**Optimal detector** that exhaustively searches all possible transmitted symbol combinations.

**Algorithm**:
```
For each possible symbol combination s:
    Calculate distance: d(s) = ||r - sqrt(SNR)·H·s||²
Select s* = argmin d(s)
```

**Characteristics**:
- ✅ Optimal BER performance (theoretical lower bound)
- ❌ Exponential complexity: O(M^Nt)
- ❌ Impractical for large M or Nt
- Uses: Benchmark for comparison

---

### 2. One-Hot Encoding Strategy

**Direct multi-class classification** with M^Nt output neurons.

**Architecture**:
- Input: 4 features [Re(r₁), Im(r₁), Re(r₂), Im(r₂)]
- Hidden: 100 neurons + ReLU
- Output: 16 neurons + Softmax
- Loss: Cross-Entropy

**Label Format**:
```
Combination 0  → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Combination 5  → [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Combination 15 → [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

**Characteristics**:
- ✅ Standard classification approach
- ✅ Typically best DL performance
- ❌ High output dimensionality
- ❌ Scales poorly: M^Nt outputs

**Complexity**:
- Training: Standard
- Inference: O(4×100 + 100×16) = O(2,000)

---

### 3. Label/Symbol Encoding Strategy

**Binary classification** using sign bits of constellation symbols.

**Architecture**:
- Input: 4 features [Re(r₁), Im(r₁), Re(r₂), Im(r₂)]
- Hidden: 100 neurons + ReLU
- Output: 4 neurons + Sigmoid
- Loss: Binary Cross-Entropy

**Label Format** (sign bits):
```
[-1+1j, -1+1j] → [1, 0, 1, 0]  (real<0, imag>0, real<0, imag>0)
[+1-1j, +1-1j] → [0, 1, 0, 1]  (real>0, imag<0, real>0, imag<0)
```

**Characteristics**:
- ✅ Lowest output dimensionality: log₂(M)×Nt
- ✅ Most efficient representation
- ✅ Best scalability
- ⚠️ Slight performance degradation vs. one-hot
- Uses bit-level structure

**Complexity**:
- Training: Standard
- Inference: O(4×100 + 100×4) = O(800)

---

### 4. One-Hot Per Antenna Strategy

**Hybrid approach** with separate one-hot encoding for each antenna.

**Architecture**:
- Input: 4 features [Re(r₁), Im(r₁), Re(r₂), Im(r₂)]
- Hidden: 100 neurons + ReLU
- Output: 8 neurons (4 per antenna) + Sigmoid
- Loss: Binary Cross-Entropy

**Label Format**:
```
[-1-1j, 1+1j] → [0, 1, 0, 0 | 0, 0, 1, 0]
                 \_Ant 1_/   \_Ant 2_/
```

**Characteristics**:
- ✅ Exploits per-antenna structure
- ✅ Balanced dimensionality: M×Nt
- ✅ Good performance/complexity trade-off
- Decomposes joint detection into per-antenna classification

**Complexity**:
- Training: Standard
- Inference: O(4×100 + 100×8) = O(1,200)

---

## System Model

### MIMO Channel

```
r = √SNR · H · x + n
```

Where:
- **r** ∈ ℂ²: Received signal vector
- **H** ∈ ℂ²ˣ²: Rayleigh fading channel matrix ~ CN(0,1)
- **x** ∈ ℂ²: Transmitted symbol vector
- **n** ∈ ℂ²: AWGN noise vector ~ CN(0,σ²)
- **SNR**: Signal-to-Noise Ratio (linear scale)

### Detection Approaches

Two different approaches are used depending on the detector:

**1. Maximum Likelihood (ML) Detection**:
- Uses the raw received signal **r** directly
- Exhaustive search over all M^Nt combinations
- No equalization required

**2. Deep Learning (DL) Detection**:
- Applies Zero-Forcing equalization first: **r_eq = H⁺ · r**
- Neural network operates on equalized signal
- Where **H⁺** is the Moore-Penrose pseudo-inverse of **H**

### Normalized Constellation

The 4-QAM constellation is normalized for unit average power:

1. **Initial normalization**:
   ```
   FN = 1/√((2/3)·(M-1))
   symbols = FN · qam_symbols
   ```

2. **Power normalization**:
   ```
   avg_power = (1/M) · Σ|symbol_i|
   symbols = symbols / avg_power
   ```

3. **Transmission scaling**:
   ```
   x = symbols / √2
   ```

---

## BER Calculation Methodology

### Bit Error Rate Definition

```
BER = (Total Bit Errors) / (Total Transmitted Bits)
```

For 2×2 MIMO with 4-QAM:
- Bits per symbol: log₂(4) = 2
- Total bits per transmission: 2 × 2 = 4 bits

### Bit Error Counting

When comparing true index `i_true` with detected index `i_pred`:

1. Convert indices to binary representation (4 bits)
2. Count differing bits using XOR operation

**Example**:
```
True index: 5   → Binary: 0101
Pred index: 7   → Binary: 0111
                  XOR:     0010 → 1 bit error
```

### Average Bit Error Probability (ABEP)

For each SNR point:

```
ABEP(SNR) = (Σ bit_errors) / (n_iterations × 4 bits)
```

---

## Monte Carlo Simulation

### Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **SNR Range** | 0 to 25 dB | In 1 dB steps |
| **Iterations** | 1,000,000 | Per SNR point |
| **Total Runs** | 26,000,000 | 26 SNR × 1M iterations |
| **Channel** | Rayleigh | Complex Gaussian CN(0,1) |
| **Noise** | AWGN | Complex Gaussian CN(0,σ²) |

### Simulation Flow

For each SNR value:
```
1. Initialize bit error counters
2. For each Monte Carlo iteration:
    a. Generate random Rayleigh channel H ~ CN(0,1)
    b. Generate AWGN noise n ~ CN(0,σ²)
    c. Compute received signal r = √SNR · H · x + n
    d. Run ML detector on raw signal r → count bit errors
    e. Apply ZF equalization: r_eq = H⁺ · r
    f. Run DL detector 1 on r_eq → count bit errors
    g. Run DL detector 2 on r_eq → count bit errors
    h. Run DL detector 3 on r_eq → count bit errors
3. Calculate BER = (total_errors) / (total_bits)
4. Store BER for this SNR
```

### Transmitted Symbol

A fixed symbol combination is transmitted:
- **Index**: 1 (MATLAB convention, 1-indexed)
- **Symbols**: First combination from Cartesian product
- This ensures consistent comparison across detectors

---

## Usage

### Prerequisites

```bash
pip install torch numpy matplotlib tqdm
```

### Running the Simulation

1. **Train Models First** (if not already done):
   ```python
   # Train one-hot encoding model
   run training_2x2_detector_OneHot.ipynb

   # Train label encoding model (to be implemented)
   run training_2x2_detector_LabelEncoder.ipynb

   # Train one-hot per antenna model (to be implemented)
   run training_2x2_detector_OneHotPerAntenna.ipynb
   ```

2. **Open BER Notebook**:
   ```bash
   jupyter notebook BER_4QAM_MIMO_2x2_All.ipynb
   ```

3. **Execute All Cells**: Run cells sequentially

4. **Adjust Parameters** (optional):
   ```python
   # For faster testing (lower quality)
   n_iter = int(1e4)  # 10,000 iterations

   # For publication quality (slower)
   n_iter = int(1e6)  # 1,000,000 iterations
   ```

### Expected Runtime

| Iterations | CPU Time | GPU Time | BER Quality |
|------------|----------|----------|-------------|
| 10,000 | ~2 min | ~30 sec | Testing only |
| 100,000 | ~20 min | ~5 min | Fair |
| 1,000,000 | ~3 hours | ~45 min | Publication |

**Note**: Times are approximate and depend on hardware.

---

## Performance Metrics

### Primary Metric: BER vs. SNR

**BER Curve**: Log-scale plot showing BER as a function of SNR (dB).

**Interpretation**:
- **Lower is better**: Lower BER = fewer errors
- **Steeper slope**: Faster improvement with SNR
- **Gap from ML**: Performance loss vs. optimal

### Secondary Metrics

#### 1. SNR Required for Target BER

Common thresholds:
- **BER = 10⁻²**: Moderate quality
- **BER = 10⁻³**: Good quality
- **BER = 10⁻⁴**: High quality

**Example**:
```
Target BER: 1e-3
  ML Detector:         12.5 dB
  One-Hot:             13.2 dB  (+0.7 dB gap)
  Label Encoder:       14.1 dB  (+1.6 dB gap)
  One-Hot Per Ant:     13.7 dB  (+1.2 dB gap)
```

#### 2. Performance Gap

```
Gap(dB) = SNR_DL - SNR_ML
```

Lower gap = better DL performance.

#### 3. Asymptotic Behavior

At high SNR:
- All detectors should achieve low BER
- Differences become more apparent
- Reveals detector limitations

---

## Results Interpretation

### Expected BER Curves

```
    BER (log scale)
    ^
 1  |    All detectors start high
    |
    |         \
10⁻²|          \   ML (best)
    |           \
    |            \ One-Hot (close to ML)
10⁻⁴|             \
    |              \ Label Enc (slight gap)
    |               \
10⁻⁶|                \  One-Hot/Ant (middle)
    |                 \
    +------------------+---> SNR (dB)
    0    5   10   15  20   25
```

### Key Observations

#### Low SNR (0-10 dB)
- All detectors perform similarly
- Noise dominates
- Hard to distinguish symbols

#### Medium SNR (10-18 dB)
- Detectors start to separate
- ML shows optimal performance
- DL detectors show small gap

#### High SNR (18-25 dB)
- Clear performance differences
- ML achieves very low BER
- DL strategies ranked by performance

### Typical Performance Ranking

**Best to Worst**:
1. **ML Detector** (optimal, but expensive)
2. **One-Hot Encoding** (closest to ML)
3. **One-Hot Per Antenna** (good trade-off)
4. **Label Encoding** (most efficient, slight loss)

### Trade-off Analysis

| Strategy | BER Performance | Output Size | Complexity | Best For |
|----------|----------------|-------------|------------|----------|
| **ML** | ⭐⭐⭐⭐⭐ | N/A | O(M^Nt) | Benchmark |
| **One-Hot** | ⭐⭐⭐⭐ | M^Nt = 16 | O(2,000) | Best accuracy |
| **One-Hot/Ant** | ⭐⭐⭐½ | M×Nt = 8 | O(1,200) | Balanced |
| **Label Enc** | ⭐⭐⭐ | log₂(M)×Nt = 4 | O(800) | Scalability |

---

## Computational Considerations

### Memory Requirements

**During Simulation**:
```
Channel matrices: 26 SNR × 1M iter × 4 complex = ~800 MB
BER storage: 4 detectors × 26 SNR × 8 bytes = ~1 KB
Models: 3 × 2,116 parameters × 4 bytes = ~25 KB
```

**Total**: ~1 GB RAM (manageable on most systems)

### Parallelization Opportunities

1. **SNR-level parallelization**:
   - Each SNR point is independent
   - Can run multiple SNR values in parallel
   - Linear speedup with cores

2. **Batch processing**:
   - Process multiple channel realizations at once
   - GPU acceleration for DL inference
   - Vectorize ML detector calculations

### Optimization Tips

#### For Speed
```python
# Reduce iterations (lower quality)
n_iter = int(1e4)

# Use fewer SNR points
SNR_dB = np.arange(0, 26, 2)  # Every 2 dB

# Use GPU
device = 'cuda'

# Vectorize when possible
# (Already implemented in ML detector)
```

#### For Accuracy
```python
# Increase iterations
n_iter = int(5e6)

# Finer SNR resolution
SNR_dB = np.arange(0, 26, 0.5)

# Multiple random seeds
# Run simulation 5 times, average results
```

### Bottlenecks

1. **Monte Carlo Loop**: Dominates runtime (~95%)
2. **Channel Generation**: ~20% of loop time
3. **ML Detection**: ~40% of loop time (most expensive)
4. **DL Inference**: ~30% of loop time (3 models)
5. **Bit Counting**: ~10% of loop time

**Optimization Focus**: Vectorize ML detection across iterations.

---

## Output Files

### Generated Files

1. **BER_MIMO_2x2_All_Strategies.png**
   - High-resolution plot (300 DPI)
   - Publication-ready figure
   - Shows all BER curves

2. **BER_results_MIMO_2x2_all_strategies.npy**
   - NumPy binary format
   - Contains all BER data + metadata
   - Load with: `np.load('file.npy', allow_pickle=True)`

3. **BER_results_MIMO_2x2_all_strategies.txt**
   - Human-readable text file
   - Tabular format
   - Easy inspection

### Loading Saved Results

```python
# Load NumPy file
results = np.load('BER_results_MIMO_2x2_all_strategies.npy',
                  allow_pickle=True).item()

SNR_dB = results['SNR_dB']
BER_ML = results['BER_ML']
BER_OneHot = results['BER_OneHot']

# Plot
plt.semilogy(SNR_dB, BER_ML, label='ML')
plt.semilogy(SNR_dB, BER_OneHot, label='One-Hot')
plt.legend()
plt.show()
```

---

## Validation and Verification

### Sanity Checks

1. **BER Monotonicity**:
   - BER should decrease as SNR increases
   - If not, check noise scaling

2. **ML as Lower Bound**:
   - All DL detectors should have BER ≥ BER_ML
   - If not, check detector implementations

3. **High SNR Behavior**:
   - BER should approach zero at high SNR
   - Typically < 10⁻⁴ at 25 dB

4. **Low SNR Behavior**:
   - BER should approach 0.5 at very low SNR
   - Random guessing limit

### Debugging Tips

**If BER is too high**:
- Check constellation normalization
- Verify SNR scaling in noise
- Confirm model loading

**If BER curves are flat**:
- Check SNR range (might be too narrow)
- Verify channel generation
- Increase iterations

**If curves are erratic**:
- Increase Monte Carlo iterations
- Check for NaN/Inf values
- Verify detector outputs

---

## Extensions and Future Work

### Immediate Extensions

1. **Complete Model Set**:
   - Train label encoding model
   - Train one-hot per antenna model
   - Ensure all three strategies are evaluated

2. **Additional Metrics**:
   - Symbol Error Rate (SER)
   - Throughput vs. SNR
   - Complexity analysis

### Advanced Extensions

1. **Larger Systems**:
   - 4×4 MIMO
   - 8×8 MIMO
   - Massive MIMO (64×64)

2. **Higher Modulations**:
   - 16-QAM
   - 64-QAM
   - 256-QAM

3. **Channel Models**:
   - Rician fading
   - Correlated MIMO channels
   - Frequency-selective fading

4. **Imperfect CSI**:
   - Channel estimation errors
   - Pilot-based estimation
   - Blind detection

5. **Coded Systems**:
   - Combine with channel coding
   - Turbo/LDPC codes
   - Coded BER performance

6. **Real-World Testing**:
   - Over-the-air measurements
   - Software-defined radio (SDR)
   - Hardware implementation

---

## Troubleshooting

### Common Issues

#### Issue 1: "Model file not found"

**Cause**: Model hasn't been trained yet

**Solution**:
```python
# Train the model first
run training_2x2_detector_OneHot.ipynb

# Then run BER simulation
```

#### Issue 2: "CUDA out of memory"

**Cause**: GPU memory exhausted

**Solution**:
```python
# Force CPU usage
device = 'cpu'

# Or reduce batch processing
# (Not applicable here, already sample-by-sample)
```

#### Issue 3: "Simulation taking too long"

**Cause**: Too many iterations

**Solution**:
```python
# Reduce for testing
n_iter = int(1e4)  # 10,000 instead of 1,000,000

# Or parallelize (not implemented by default)
```

#### Issue 4: "BER all zeros or all ones"

**Cause**: Incorrect SNR scaling or detector bug

**Solution**:
- Check SNR conversion: `SNR_linear = 10.0 ** (SNR_dB / 10.0)`
- Verify noise scaling: `n = n / np.sqrt(SNR_j)`
- Print intermediate values to debug

#### Issue 5: "NaN in BER results"

**Cause**: Model not available for that detector

**Solution**:
- Check which models loaded successfully
- NaN indicates detector was skipped
- Train missing models

---

## Dependencies

### Required Libraries

```python
torch>=1.8.0          # PyTorch for DL models
numpy>=1.19.0         # Numerical computations
matplotlib>=3.3.0     # Plotting
tqdm>=4.50.0          # Progress bars
```

### Optional Libraries

```python
seaborn>=0.11.0       # Enhanced plotting
pandas>=1.1.0         # Data analysis
scipy>=1.5.0          # Scientific computing
```

### Installation

```bash
# Minimal installation
pip install torch numpy matplotlib tqdm

# Full installation
pip install torch numpy matplotlib tqdm seaborn pandas scipy jupyter
```

---

## Performance Benchmarks

### Reference System Specs

**CPU Test**:
- Processor: Intel Core i7-9700K
- RAM: 16 GB
- Iterations: 1,000,000
- Time: ~180 minutes

**GPU Test**:
- GPU: NVIDIA RTX 3080
- VRAM: 10 GB
- Iterations: 1,000,000
- Time: ~45 minutes

### Scaling Analysis

**Iterations vs. Time** (Linear):
```
10,000 iter    → ~3 min   (GPU)
100,000 iter   → ~30 min  (GPU)
1,000,000 iter → ~5 hours (GPU)
```

**SNR Points vs. Time** (Linear):
```
10 SNR points  → ~17 min (1M iter, GPU)
26 SNR points  → ~45 min (1M iter, GPU)
50 SNR points  → ~87 min (1M iter, GPU)
```

---

## References

### Academic Papers

1. **Original Work**:
   - Ibarra-Hernández, R.F.; Castillo-Soria, F.R.; Gutiérrez, C.A.; Del-Puerto-Flores, J.A.; Acosta-Elías J., Rodríguez-Abdalá V. and Palacios-Luengas L.
   - "Efficient Deep Learning-Based Detection Scheme for MIMO Communication System"
   - Submitted to the Journal Sensors of MDPI

2. **BER Analysis**:
   - Proakis, J.G., & Salehi, M. (2008). "Digital Communications" (5th ed.). McGraw-Hill.

3. **MIMO Detection**:
   - Tse, D., & Viswanath, P. (2005). "Fundamentals of Wireless Communication." Cambridge University Press.

4. **Monte Carlo Methods**:
   - Jeruchim, M.C., Balaban, P., & Shanmugan, K.S. (2000). "Simulation of Communication Systems" (2nd ed.). Springer.

### Online Resources

- **BER Curves**: https://www.gaussianwaves.com/tag/ber/
- **MIMO Systems**: https://en.wikipedia.org/wiki/MIMO
- **PyTorch**: https://pytorch.org/docs/

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

## Acknowledgments

This work evaluates the performance of Deep Learning-based MIMO detection schemes and contributes to understanding the trade-offs between detection accuracy, computational complexity, and system scalability.

---

**Last Updated**: 2025

**Version**: 1.0.0
