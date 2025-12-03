"""
Benchmark riguroso de optimizaciones para artículo de conferencia
Mide tiempos reales de cada optimización incremental

Autor: Leonel Roberto Perea Trejo
Fecha: Diciembre 2024
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Parámetros del sistema MIMO
Nr = 2  # Antenas receptoras
Nt = 2  # Antenas transmisoras
M = 4   # 4-QAM

# Parámetros de benchmark
N_WARMUP = 100      # Iteraciones de calentamiento
N_ITERATIONS = 10000  # Iteraciones para benchmark
SNR_DB = 10.0       # SNR fijo para benchmark

# =============================================================================
# UTILIDADES DE MEDICIÓN
# =============================================================================

class GPUTimer:
    """Timer preciso para operaciones GPU usando CUDA events"""
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

    def elapsed(self):
        return self.elapsed_time_ms

def benchmark_function(func, n_iterations=N_ITERATIONS, n_warmup=N_WARMUP, use_gpu_timer=True):
    """
    Benchmark de una función con warmup y múltiples iteraciones

    Returns:
        (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(n_warmup):
        func()

    # Benchmark
    times = []
    for _ in range(n_iterations):
        if use_gpu_timer and torch.cuda.is_available():
            timer = GPUTimer()
            with timer:
                func()
            times.append(timer.elapsed())
        else:
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convertir a ms

    return np.mean(times), np.std(times)

# =============================================================================
# GENERACIÓN DE DATOS DE PRUEBA
# =============================================================================

# Canal fijo para todas las pruebas
H_fixed = torch.randn(Nr, Nt, dtype=torch.complex64, device=device) / np.sqrt(2)

# Símbolos 4-QAM
constellation_4qam = torch.tensor([-1-1j, -1+1j, 1-1j, 1+1j],
                                   dtype=torch.complex64, device=device) / np.sqrt(2)

# Todas las combinaciones de símbolos (16 para 2 antenas)
symbol_combinations = torch.stack([
    torch.stack([constellation_4qam[i], constellation_4qam[j]])
    for i in range(M) for j in range(M)
], dim=0).to(device)

# SNR
SNR_linear = 10.0 ** (SNR_DB / 10.0)
sqrt_SNR = np.sqrt(SNR_linear)
inv_sqrt_SNR = 1.0 / sqrt_SNR

# Símbolo transmitido aleatorio
x_transmitted = constellation_4qam[torch.randint(0, M, (Nt,))].to(device)

# =============================================================================
# IMPLEMENTACIONES: BASELINE VS OPTIMIZADAS
# =============================================================================

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN 1: Pre-cómputo de Pseudoinversa
# -----------------------------------------------------------------------------

def baseline_pinv():
    """Baseline: Calcular pinv en cada iteración"""
    H_inv = torch.linalg.pinv(H_fixed)
    return H_inv

def optimized_pinv():
    """Optimizado: Usar pinv pre-computada"""
    # En la versión optimizada, esto se hace UNA vez antes del loop
    return H_inv_precomputed

# Pre-computar una vez
H_inv_precomputed = torch.linalg.pinv(H_fixed)

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN 2: Eliminación de Transferencias CPU↔GPU
# -----------------------------------------------------------------------------

def baseline_cpu_gpu_transfer():
    """Baseline: Transferencias CPU↔GPU"""
    # Generar señal recibida
    n = torch.randn(Nr, dtype=torch.complex64, device=device) * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n
    r_eq = H_inv_precomputed @ r

    # MALO: Transferir a CPU y de vuelta a GPU
    x_input = torch.tensor([
        r_eq[0].real.item(),
        r_eq[0].imag.item(),
        r_eq[1].real.item(),
        r_eq[1].imag.item()
    ], device=device)

    return x_input

def optimized_cpu_gpu_transfer():
    """Optimizado: Todo en GPU"""
    # Generar señal recibida
    n = torch.randn(Nr, dtype=torch.complex64, device=device) * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n
    r_eq = H_inv_precomputed @ r

    # BUENO: Operaciones nativas en GPU
    x_input = torch.stack([
        r_eq[0].real,
        r_eq[0].imag,
        r_eq[1].real,
        r_eq[1].imag
    ])

    return x_input

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN 3: Pre-cómputo de Productos ML
# -----------------------------------------------------------------------------

def baseline_ml_products():
    """Baseline: Calcular H·s en cada iteración"""
    n = torch.randn(Nr, dtype=torch.complex64, device=device) * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n

    # MALO: Calcular productos cada vez
    Hs = symbol_combinations @ H_fixed.T
    distances = torch.abs(r.unsqueeze(0) - sqrt_SNR * Hs)**2
    idx = torch.argmin(distances.sum(dim=1))

    return idx

def optimized_ml_products():
    """Optimizado: Usar productos pre-computados"""
    n = torch.randn(Nr, dtype=torch.complex64, device=device) * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n

    # BUENO: Usar pre-computado
    distances = torch.abs(r.unsqueeze(0) - sqrt_SNR * Hs_precomputed)**2
    idx = torch.argmin(distances.sum(dim=1))

    return idx

# Pre-computar productos
Hs_precomputed = symbol_combinations @ H_fixed.T

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN 4: Pre-cómputo de √SNR
# -----------------------------------------------------------------------------

def baseline_sqrt_snr():
    """Baseline: Calcular sqrt(SNR) cada vez"""
    n = torch.randn(Nr, dtype=torch.complex64, device=device) / np.sqrt(2)

    # MALO: Calcular sqrt cada vez
    n = n / np.sqrt(SNR_linear)
    r = np.sqrt(SNR_linear) * (H_fixed @ x_transmitted) + n

    return r

def optimized_sqrt_snr():
    """Optimizado: Usar sqrt pre-computado"""
    n = torch.randn(Nr, dtype=torch.complex64, device=device) / np.sqrt(2)

    # BUENO: Usar pre-computado
    n = n * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n

    return r

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN 5: XOR Bitwise para Conteo de Errores
# -----------------------------------------------------------------------------

def baseline_bit_counting():
    """Baseline: Conversión a strings"""
    idx_true = np.random.randint(0, 16)
    idx_pred = np.random.randint(0, 16)

    # MALO: Conversión a strings
    true_bits = format(idx_true, '04b')
    pred_bits = format(idx_pred, '04b')
    errors = sum(t != p for t, p in zip(true_bits, pred_bits))

    return errors

def optimized_bit_counting():
    """Optimizado: XOR bitwise"""
    idx_true = np.random.randint(0, 16)
    idx_pred = np.random.randint(0, 16)

    # BUENO: XOR bitwise
    xor_result = idx_true ^ idx_pred
    errors = bin(xor_result).count('1')

    return errors

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN 6: Generación Directa de Ruido Complejo
# -----------------------------------------------------------------------------

def baseline_complex_noise():
    """Baseline: Generación separada real/imag"""
    # MALO: 2 llamadas + complex()
    n_real = torch.randn(Nr, device=device) / np.sqrt(2)
    n_imag = torch.randn(Nr, device=device) / np.sqrt(2)
    n = torch.complex(n_real, n_imag)

    return n

def optimized_complex_noise():
    """Optimizado: Generación directa"""
    # BUENO: 1 llamada directa
    n = torch.randn(Nr, dtype=torch.complex64, device=device) / np.sqrt(2)

    return n

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN 7: Omisión de Softmax
# -----------------------------------------------------------------------------

# Modelo simple para demostración
simple_model = torch.nn.Sequential(
    torch.nn.Linear(4, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 16)
).to(device)

def baseline_softmax():
    """Baseline: Calcular softmax antes de argmax"""
    x_input = torch.randn(1, 4, device=device)

    # MALO: Softmax innecesario
    logits = simple_model(x_input)
    probs = torch.softmax(logits, dim=1)
    idx = torch.argmax(probs, dim=1)

    return idx

def optimized_softmax():
    """Optimizado: Argmax directo sobre logits"""
    x_input = torch.randn(1, 4, device=device)

    # BUENO: Argmax directo
    logits = simple_model(x_input)
    idx = torch.argmax(logits, dim=1)

    return idx

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN 8: Lookup Table para Errores de Bit
# -----------------------------------------------------------------------------

# Pre-computar lookup table
bit_error_lut = torch.zeros(16, 16, dtype=torch.int32, device=device)
for i in range(16):
    for j in range(16):
        bit_error_lut[i, j] = bin(i ^ j).count('1')

def baseline_bit_lut():
    """Baseline: XOR + bin().count()"""
    idx_true = torch.randint(0, 16, (1,), device=device).item()
    idx_pred = torch.randint(0, 16, (1,), device=device).item()

    # MALO: Python bin().count()
    xor_result = idx_true ^ idx_pred
    errors = bin(xor_result).count('1')

    return errors

def optimized_bit_lut():
    """Optimizado: Lookup en GPU"""
    idx_true = torch.randint(0, 16, (1,), device=device).item()
    idx_pred = torch.randint(0, 16, (1,), device=device).item()

    # BUENO: Lookup en tensor GPU
    errors = bit_error_lut[idx_true, idx_pred].item()

    return errors

# =============================================================================
# EJECUCIÓN DE BENCHMARKS
# =============================================================================

def run_benchmarks():
    """Ejecutar todos los benchmarks y reportar resultados"""

    print("\n" + "="*80)
    print("BENCHMARK DE OPTIMIZACIONES - Sistema MIMO 2×2 4-QAM")
    print("="*80)
    print(f"\nConfiguración:")
    print(f"  - Iteraciones: {N_ITERATIONS:,}")
    print(f"  - Warmup: {N_WARMUP}")
    print(f"  - Dispositivo: {device}")
    print(f"  - SNR: {SNR_DB} dB")

    results = {}

    # -------------------------------------------------------------------------
    # Optimización 1: Pseudoinversa
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("OPTIMIZACIÓN 1: Pre-cómputo de Pseudoinversa")
    print("-"*80)

    print("Midiendo baseline (pinv en cada iteración)...", end=" ")
    time_baseline_pinv, std_baseline = benchmark_function(baseline_pinv)
    print(f"{time_baseline_pinv:.6f} ± {std_baseline:.6f} ms")

    print("Midiendo optimizado (pinv pre-computada)...", end=" ")
    time_optimized_pinv, std_optimized = benchmark_function(optimized_pinv)
    print(f"{time_optimized_pinv:.6f} ± {std_optimized:.6f} ms")

    speedup_1 = time_baseline_pinv / time_optimized_pinv
    print(f"➜ Speedup: {speedup_1:.2f}×")

    results['pinv'] = {
        'baseline': time_baseline_pinv,
        'optimized': time_optimized_pinv,
        'speedup': speedup_1
    }

    # -------------------------------------------------------------------------
    # Optimización 2: CPU↔GPU Transfers
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("OPTIMIZACIÓN 2: Eliminación de Transferencias CPU↔GPU")
    print("-"*80)

    print("Midiendo baseline (con transferencias)...", end=" ")
    time_baseline_transfer, std_baseline = benchmark_function(baseline_cpu_gpu_transfer)
    print(f"{time_baseline_transfer:.6f} ± {std_baseline:.6f} ms")

    print("Midiendo optimizado (sin transferencias)...", end=" ")
    time_optimized_transfer, std_optimized = benchmark_function(optimized_cpu_gpu_transfer)
    print(f"{time_optimized_transfer:.6f} ± {std_optimized:.6f} ms")

    speedup_2 = time_baseline_transfer / time_optimized_transfer
    print(f"➜ Speedup: {speedup_2:.2f}×")

    results['transfer'] = {
        'baseline': time_baseline_transfer,
        'optimized': time_optimized_transfer,
        'speedup': speedup_2
    }

    # -------------------------------------------------------------------------
    # Optimización 3: Productos ML
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("OPTIMIZACIÓN 3: Pre-cómputo de Productos ML (H·s)")
    print("-"*80)

    print("Midiendo baseline (calcular cada vez)...", end=" ")
    time_baseline_ml, std_baseline = benchmark_function(baseline_ml_products)
    print(f"{time_baseline_ml:.6f} ± {std_baseline:.6f} ms")

    print("Midiendo optimizado (pre-computado)...", end=" ")
    time_optimized_ml, std_optimized = benchmark_function(optimized_ml_products)
    print(f"{time_optimized_ml:.6f} ± {std_optimized:.6f} ms")

    speedup_3 = time_baseline_ml / time_optimized_ml
    print(f"➜ Speedup: {speedup_3:.2f}×")

    results['ml_products'] = {
        'baseline': time_baseline_ml,
        'optimized': time_optimized_ml,
        'speedup': speedup_3
    }

    # -------------------------------------------------------------------------
    # Optimización 4: √SNR
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("OPTIMIZACIÓN 4: Pre-cómputo de √SNR")
    print("-"*80)

    print("Midiendo baseline (calcular cada vez)...", end=" ")
    time_baseline_sqrt, std_baseline = benchmark_function(baseline_sqrt_snr)
    print(f"{time_baseline_sqrt:.6f} ± {std_baseline:.6f} ms")

    print("Midiendo optimizado (pre-computado)...", end=" ")
    time_optimized_sqrt, std_optimized = benchmark_function(optimized_sqrt_snr)
    print(f"{time_optimized_sqrt:.6f} ± {std_optimized:.6f} ms")

    speedup_4 = time_baseline_sqrt / time_optimized_sqrt
    print(f"➜ Speedup: {speedup_4:.2f}×")

    results['sqrt_snr'] = {
        'baseline': time_baseline_sqrt,
        'optimized': time_optimized_sqrt,
        'speedup': speedup_4
    }

    # -------------------------------------------------------------------------
    # Optimización 5: XOR Bitwise
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("OPTIMIZACIÓN 5: XOR Bitwise para Conteo de Errores")
    print("-"*80)

    print("Midiendo baseline (strings)...", end=" ")
    time_baseline_xor, std_baseline = benchmark_function(baseline_bit_counting, use_gpu_timer=False)
    print(f"{time_baseline_xor:.6f} ± {std_baseline:.6f} ms")

    print("Midiendo optimizado (XOR)...", end=" ")
    time_optimized_xor, std_optimized = benchmark_function(optimized_bit_counting, use_gpu_timer=False)
    print(f"{time_optimized_xor:.6f} ± {std_optimized:.6f} ms")

    speedup_5 = time_baseline_xor / time_optimized_xor
    print(f"➜ Speedup: {speedup_5:.2f}×")

    results['xor'] = {
        'baseline': time_baseline_xor,
        'optimized': time_optimized_xor,
        'speedup': speedup_5
    }

    # -------------------------------------------------------------------------
    # Optimización 6: Ruido Complejo
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("OPTIMIZACIÓN 6: Generación Directa de Ruido Complejo")
    print("-"*80)

    print("Midiendo baseline (separado real/imag)...", end=" ")
    time_baseline_noise, std_baseline = benchmark_function(baseline_complex_noise)
    print(f"{time_baseline_noise:.6f} ± {std_baseline:.6f} ms")

    print("Midiendo optimizado (directo)...", end=" ")
    time_optimized_noise, std_optimized = benchmark_function(optimized_complex_noise)
    print(f"{time_optimized_noise:.6f} ± {std_optimized:.6f} ms")

    speedup_6 = time_baseline_noise / time_optimized_noise
    print(f"➜ Speedup: {speedup_6:.2f}×")

    results['noise'] = {
        'baseline': time_baseline_noise,
        'optimized': time_optimized_noise,
        'speedup': speedup_6
    }

    # -------------------------------------------------------------------------
    # Optimización 7: Softmax
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("OPTIMIZACIÓN 7: Omisión de Softmax Innecesario")
    print("-"*80)

    print("Midiendo baseline (con softmax)...", end=" ")
    time_baseline_softmax, std_baseline = benchmark_function(baseline_softmax)
    print(f"{time_baseline_softmax:.6f} ± {std_baseline:.6f} ms")

    print("Midiendo optimizado (sin softmax)...", end=" ")
    time_optimized_softmax, std_optimized = benchmark_function(optimized_softmax)
    print(f"{time_optimized_softmax:.6f} ± {std_optimized:.6f} ms")

    speedup_7 = time_baseline_softmax / time_optimized_softmax
    print(f"➜ Speedup: {speedup_7:.2f}×")

    results['softmax'] = {
        'baseline': time_baseline_softmax,
        'optimized': time_optimized_softmax,
        'speedup': speedup_7
    }

    # -------------------------------------------------------------------------
    # Optimización 8: Lookup Table
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("OPTIMIZACIÓN 8: Lookup Table para Errores de Bit")
    print("-"*80)

    print("Midiendo baseline (XOR + bin().count())...", end=" ")
    time_baseline_lut, std_baseline = benchmark_function(baseline_bit_lut, use_gpu_timer=False)
    print(f"{time_baseline_lut:.6f} ± {std_baseline:.6f} ms")

    print("Midiendo optimizado (lookup en GPU)...", end=" ")
    time_optimized_lut, std_optimized = benchmark_function(optimized_bit_lut, use_gpu_timer=False)
    print(f"{time_optimized_lut:.6f} ± {std_optimized:.6f} ms")

    speedup_8 = time_baseline_lut / time_optimized_lut
    print(f"➜ Speedup: {speedup_8:.2f}×")

    results['lut'] = {
        'baseline': time_baseline_lut,
        'optimized': time_optimized_lut,
        'speedup': speedup_8
    }

    # -------------------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)

    # Calcular speedup acumulado (multiplicativo)
    total_speedup = 1.0
    for key, value in results.items():
        total_speedup *= value['speedup']

    print("\nTabla de Speedups:")
    print(f"{'Optimización':<40} {'Speedup Individual':<20} {'Speedup Acumulado':<20}")
    print("-" * 80)

    cumulative = 1.0
    opt_names = [
        ('pinv', 'Pre-cómputo Pseudoinversa'),
        ('transfer', 'Eliminar CPU↔GPU'),
        ('ml_products', 'Pre-cómputo Productos ML'),
        ('sqrt_snr', 'Pre-cómputo √SNR'),
        ('xor', 'XOR Bitwise'),
        ('noise', 'Ruido Complejo Directo'),
        ('softmax', 'Skip Softmax'),
        ('lut', 'Lookup Table Bits')
    ]

    for key, name in opt_names:
        speedup = results[key]['speedup']
        cumulative *= speedup
        print(f"{name:<40} {speedup:>6.2f}× {cumulative:>18.2f}×")

    print("-" * 80)
    print(f"{'SPEEDUP TOTAL':<40} {total_speedup:>26.2f}×")

    # Guardar resultados
    np.save('benchmark_results.npy', results)
    print("\n✓ Resultados guardados en: benchmark_results.npy")

    return results

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def plot_results(results):
    """Generar gráficos de los resultados"""

    opt_names = [
        'Pre-cómputo\nPseudoinversa',
        'Eliminar\nCPU↔GPU',
        'Pre-cómputo\nProductos ML',
        'Pre-cómputo\n√SNR',
        'XOR\nBitwise',
        'Ruido\nComplejo',
        'Skip\nSoftmax',
        'Lookup\nTable'
    ]

    speedups = [
        results['pinv']['speedup'],
        results['transfer']['speedup'],
        results['ml_products']['speedup'],
        results['sqrt_snr']['speedup'],
        results['xor']['speedup'],
        results['noise']['speedup'],
        results['softmax']['speedup'],
        results['lut']['speedup']
    ]

    # Gráfico de barras
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Speedups individuales
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(speedups)))
    bars = ax1.bar(range(len(speedups)), speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(speedups)))
    ax1.set_xticklabels(opt_names, rotation=45, ha='right')
    ax1.set_ylabel('Speedup Individual (×)', fontsize=12, fontweight='bold')
    ax1.set_title('Speedup por Optimización', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0×)')
    ax1.legend()

    # Añadir valores sobre las barras
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}×',
                ha='center', va='bottom', fontweight='bold')

    # Speedup acumulado
    cumulative = np.cumprod(speedups)
    ax2.plot(range(len(cumulative)), cumulative, marker='o', linewidth=3,
             markersize=10, color='#2E86AB')
    ax2.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='#2E86AB')
    ax2.set_xticks(range(len(cumulative)))
    ax2.set_xticklabels([f'Opt {i+1}' for i in range(len(cumulative))], rotation=45)
    ax2.set_ylabel('Speedup Acumulado (×)', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup Acumulado (Multiplicativo)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')

    # Añadir valor final
    ax2.text(len(cumulative)-1, cumulative[-1],
             f'Total: {cumulative[-1]:.2f}×',
             ha='right', va='bottom', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax2.legend()

    plt.tight_layout()
    plt.savefig('benchmark_speedups.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado en: benchmark_speedups.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n🚀 Iniciando benchmarks de optimizaciones...\n")

    if not torch.cuda.is_available():
        print("⚠️  ADVERTENCIA: CUDA no disponible, usando CPU")
        print("   Los resultados no serán representativos de optimizaciones GPU\n")

    results = run_benchmarks()
    plot_results(results)

    print("\n✅ Benchmark completado exitosamente!")
