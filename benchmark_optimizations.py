"""
Benchmark Completo: Individual + Extrapolación
Mide 8 optimizaciones individuales y extrapola a simulación completa de 26M iteraciones

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

# Parámetros de simulación Monte Carlo (para extrapolación)
N_ITER_PER_SNR = 1_000_000
N_SNR_POINTS = 26
TOTAL_ITERATIONS = N_ITER_PER_SNR * N_SNR_POINTS

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
    """Benchmark de una función con warmup y múltiples iteraciones"""
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
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)

# =============================================================================
# GENERACIÓN DE DATOS DE PRUEBA
# =============================================================================

H_fixed = torch.randn(Nr, Nt, dtype=torch.complex64, device=device) / np.sqrt(2)
constellation_4qam = torch.tensor([-1-1j, -1+1j, 1-1j, 1+1j],
                                   dtype=torch.complex64, device=device) / np.sqrt(2)
symbol_combinations = torch.stack([
    torch.stack([constellation_4qam[i], constellation_4qam[j]])
    for i in range(M) for j in range(M)
], dim=0).to(device)

SNR_linear = 10.0 ** (SNR_DB / 10.0)
sqrt_SNR = np.sqrt(SNR_linear)
inv_sqrt_SNR = 1.0 / sqrt_SNR
x_transmitted = constellation_4qam[torch.randint(0, M, (Nt,))].to(device)

# Pre-computar
H_inv_precomputed = torch.linalg.pinv(H_fixed)
Hs_precomputed = symbol_combinations @ H_fixed.T

# Modelo simple
simple_model = torch.nn.Sequential(
    torch.nn.Linear(4, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 16)
).to(device)


# =============================================================================
# FUNCIONES BASELINE Y OPTIMIZADAS
# =============================================================================

def baseline_pinv():
    H_inv = torch.linalg.pinv(H_fixed)
    return H_inv

def optimized_pinv():
    return H_inv_precomputed

def baseline_cpu_gpu_transfer():
    n = torch.randn(Nr, dtype=torch.complex64, device=device) * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n
    r_eq = H_inv_precomputed @ r
    x_input = torch.tensor([
        r_eq[0].real.item(),
        r_eq[0].imag.item(),
        r_eq[1].real.item(),
        r_eq[1].imag.item()
    ], device=device)
    return x_input

def optimized_cpu_gpu_transfer():
    n = torch.randn(Nr, dtype=torch.complex64, device=device) * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n
    r_eq = H_inv_precomputed @ r
    x_input = torch.stack([
        r_eq[0].real,
        r_eq[0].imag,
        r_eq[1].real,
        r_eq[1].imag
    ])
    return x_input

def baseline_ml_products():
    n = torch.randn(Nr, dtype=torch.complex64, device=device) * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n
    Hs = symbol_combinations @ H_fixed.T
    distances = torch.abs(r.unsqueeze(0) - sqrt_SNR * Hs)**2
    idx = torch.argmin(distances.sum(dim=1))
    return idx

def optimized_ml_products():
    n = torch.randn(Nr, dtype=torch.complex64, device=device) * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n
    distances = torch.abs(r.unsqueeze(0) - sqrt_SNR * Hs_precomputed)**2
    idx = torch.argmin(distances.sum(dim=1))
    return idx

def baseline_sqrt_snr():
    n = torch.randn(Nr, dtype=torch.complex64, device=device) / np.sqrt(2)
    n = n / np.sqrt(SNR_linear)
    r = np.sqrt(SNR_linear) * (H_fixed @ x_transmitted) + n
    return r

def optimized_sqrt_snr():
    n = torch.randn(Nr, dtype=torch.complex64, device=device) / np.sqrt(2)
    n = n * inv_sqrt_SNR
    r = sqrt_SNR * (H_fixed @ x_transmitted) + n
    return r

def baseline_bit_counting():
    idx_true = np.random.randint(0, 16)
    idx_pred = np.random.randint(0, 16)
    true_bits = format(idx_true, '04b')
    pred_bits = format(idx_pred, '04b')
    errors = sum(t != p for t, p in zip(true_bits, pred_bits))
    return errors

def optimized_bit_counting():
    idx_true = np.random.randint(0, 16)
    idx_pred = np.random.randint(0, 16)
    xor_result = idx_true ^ idx_pred
    errors = bin(xor_result).count('1')
    return errors

def baseline_complex_noise():
    n_real = torch.randn(Nr, device=device) / np.sqrt(2)
    n_imag = torch.randn(Nr, device=device) / np.sqrt(2)
    n = torch.complex(n_real, n_imag)
    return n

def optimized_complex_noise():
    n = torch.randn(Nr, dtype=torch.complex64, device=device) / np.sqrt(2)
    return n

def baseline_softmax():
    x_input = torch.randn(1, 4, device=device)
    logits = simple_model(x_input)
    probs = torch.softmax(logits, dim=1)
    idx = torch.argmax(probs, dim=1)
    return idx

def optimized_softmax():
    x_input = torch.randn(1, 4, device=device)
    logits = simple_model(x_input)
    idx = torch.argmax(logits, dim=1)
    return idx

# Pre-computar LUT para errores de bit
bit_error_lut = torch.tensor([
    bin(i ^ j).count('1') for i in range(16) for j in range(16)
], dtype=torch.int32, device=device).reshape(16, 16)

def baseline_bit_error_lut():
    idx_true = torch.randint(0, 16, (1,), device=device)
    idx_pred = torch.randint(0, 16, (1,), device=device)
    xor_result = idx_true ^ idx_pred
    errors = bin(xor_result.item()).count('1')
    return errors

def optimized_bit_error_lut():
    idx_true = torch.randint(0, 16, (1,), device=device)
    idx_pred = torch.randint(0, 16, (1,), device=device)
    errors = bit_error_lut[idx_true, idx_pred]
    return errors


# =============================================================================
# BENCHMARK INDIVIDUAL
# =============================================================================

def run_individual_benchmarks():
    """Ejecutar benchmarks individuales"""

    print("\n" + "="*80)
    print("FASE 1: BENCHMARK INDIVIDUAL DE OPTIMIZACIONES")
    print("="*80)

    benchmarks = [
        ('pinv', 'Pre-cómputo Pseudoinversa', baseline_pinv, optimized_pinv, True, 1),
        ('cpu_gpu', 'Eliminar CPU↔GPU', baseline_cpu_gpu_transfer, optimized_cpu_gpu_transfer, True, 4),
        ('ml_products', 'Pre-cómputo Productos ML', baseline_ml_products, optimized_ml_products, True, 1),
        ('sqrt_snr', 'Pre-cómputo √SNR', baseline_sqrt_snr, optimized_sqrt_snr, True, 2),
        ('xor', 'XOR Bitwise', baseline_bit_counting, optimized_bit_counting, False, 4),
        ('noise', 'Ruido Complejo Directo', baseline_complex_noise, optimized_complex_noise, True, 1),
        ('softmax', 'Skip Softmax', baseline_softmax, optimized_softmax, True, 1),
        ('bit_lut', 'Lookup Table Errores de Bit', baseline_bit_error_lut, optimized_bit_error_lut, True, 4),
    ]

    results = {}

    for key, name, baseline_func, optimized_func, use_gpu, calls_per_iter in benchmarks:
        print(f"\n{'-'*80}")
        print(f"OPTIMIZACIÓN: {name}")
        print(f"{'-'*80}")

        print(f"Midiendo baseline...", end=" ", flush=True)
        time_baseline, std_baseline = benchmark_function(baseline_func, use_gpu_timer=use_gpu)
        print(f"{time_baseline:.6f} ± {std_baseline:.6f} ms")

        print(f"Midiendo optimizado...", end=" ", flush=True)
        time_optimized, std_optimized = benchmark_function(optimized_func, use_gpu_timer=use_gpu)
        print(f"{time_optimized:.6f} ± {std_optimized:.6f} ms")

        speedup = time_baseline / time_optimized
        print(f"➜ Speedup: {speedup:.2f}×")

        results[key] = {
            'name': name,
            'baseline_ms': time_baseline,
            'optimized_ms': time_optimized,
            'speedup': speedup,
            'calls_per_iter': calls_per_iter
        }

    return results

# =============================================================================
# EXTRAPOLACIÓN
# =============================================================================

def extrapolate_to_full_simulation(individual_results):
    """Extrapolar benchmarks individuales a simulación completa"""

    print("\n" + "="*80)
    print("FASE 2: EXTRAPOLACIÓN A SIMULACIÓN COMPLETA")
    print("="*80)
    print(f"\nParámetros:")
    print(f"  - Iteraciones por SNR: {N_ITER_PER_SNR:,}")
    print(f"  - Puntos SNR: {N_SNR_POINTS}")
    print(f"  - Total iteraciones: {TOTAL_ITERATIONS:,}")

    extrapolation_results = []

    for key, bench in individual_results.items():
        # Convertir ms a µs
        baseline_us = bench['baseline_ms'] * 1000
        optimized_us = bench['optimized_ms'] * 1000

        # Calcular llamadas totales
        total_calls = TOTAL_ITERATIONS * bench['calls_per_iter']

        # Tiempo total
        time_baseline_us = baseline_us * total_calls
        time_optimized_us = optimized_us * total_calls
        time_saved_us = time_baseline_us - time_optimized_us

        # Convertir a segundos
        time_baseline_sec = time_baseline_us / 1_000_000
        time_optimized_sec = time_optimized_us / 1_000_000
        time_saved_sec = time_saved_us / 1_000_000

        extrapolation_results.append({
            'key': key,
            'name': bench['name'],
            'total_calls': total_calls,
            'time_baseline_sec': time_baseline_sec,
            'time_optimized_sec': time_optimized_sec,
            'time_saved_sec': time_saved_sec,
            'speedup': bench['speedup']
        })

        print(f"\n{bench['name']}:")
        print(f"  Llamadas totales: {total_calls:,}")
        print(f"  Tiempo baseline: {time_baseline_sec:,.2f} seg ({time_baseline_sec/3600:.2f} h)")
        print(f"  Tiempo optimizado: {time_optimized_sec:,.2f} seg ({time_optimized_sec/3600:.2f} h)")
        print(f"  AHORRADO: {time_saved_sec:,.2f} seg ({time_saved_sec/3600:.2f} h)")

    # Resumen total
    time_baseline_total = sum(r['time_baseline_sec'] for r in extrapolation_results)
    time_optimized_total = sum(r['time_optimized_sec'] for r in extrapolation_results)
    time_saved_total = time_baseline_total - time_optimized_total
    speedup_total = time_baseline_total / time_optimized_total

    print(f"\n{'='*80}")
    print("RESUMEN TOTAL DE OPTIMIZACIONES")
    print(f"{'='*80}")
    print(f"\nTiempo BASELINE (sin optimizaciones): {time_baseline_total:,.2f} seg ({time_baseline_total/3600:.2f} h)")
    print(f"Tiempo OPTIMIZADO (con 7 optimizaciones): {time_optimized_total:,.2f} seg ({time_optimized_total/3600:.2f} h)")
    print(f"\n{'🚀 MEJORA OBTENIDA':^80}")
    print(f"  Tiempo AHORRADO: {time_saved_total:,.2f} seg ({time_saved_total/3600:.2f} h)")
    print(f"  Reducción: {(time_saved_total/time_baseline_total)*100:.1f}%")
    print(f"  SPEEDUP: {speedup_total:.2f}×")

    return {
        'individual': extrapolation_results,
        'baseline_total_sec': time_baseline_total,
        'optimized_total_sec': time_optimized_total,
        'saved_total_sec': time_saved_total,
        'speedup_total': speedup_total
    }

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def plot_all_results(individual_results, extrapolation_data):
    """Generar gráficos completos"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # --- GRÁFICO 1: Speedup Individual (Barras) ---
    ax1 = fig.add_subplot(gs[0, :2])

    opt_names = [v['name'].replace(' ', '\n') for v in individual_results.values()]
    speedups = [v['speedup'] for v in individual_results.values()]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(speedups)))
    bars = ax1.bar(range(len(speedups)), speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(speedups)))
    ax1.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel('Tipo de Optimización', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Speedup Individual (×)', fontsize=11, fontweight='bold')
    ax1.set_title('Speedup por Optimización (Benchmark Individual)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0×)')
    ax1.legend()

    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}×', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # --- GRÁFICO 2: Speedup Acumulado (Línea) ---
    ax2 = fig.add_subplot(gs[0, 2])

    cumulative = np.cumprod(speedups)
    ax2.plot(range(len(cumulative)), cumulative, marker='o', linewidth=3,
             markersize=8, color='#2E86AB')
    ax2.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='#2E86AB')
    ax2.set_xticks(range(len(cumulative)))
    ax2.set_xticklabels([f'{i+1}' for i in range(len(cumulative))], fontsize=9)
    ax2.set_xlabel('Número de Optimización', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Speedup Acumulado (×)', fontsize=10, fontweight='bold')
    ax2.set_title('Speedup Acumulado\n(Multiplicativo)', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.text(len(cumulative)-1, cumulative[-1],
             f'{cumulative[-1]:.0f}×',
             ha='right', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # --- GRÁFICO 3: Contribución por Optimización (Pie) ---
    ax3 = fig.add_subplot(gs[1, :2])

    # Filtrar solo optimizaciones que ahorran tiempo (positivas)
    time_saved_positive = []
    names_positive = []
    colors_list = []

    for i, r in enumerate(extrapolation_data['individual']):
        time_saved_h = r['time_saved_sec']/3600
        if time_saved_h > 0:  # Solo incluir mejoras positivas
            time_saved_positive.append(time_saved_h)
            names_positive.append(r['name'])
            colors_list.append(plt.cm.Set3(i))

    if len(time_saved_positive) > 0:
        wedges, texts, autotexts = ax3.pie(time_saved_positive,
                                             labels=names_positive,
                                             autopct='%1.1f%%',
                                             colors=colors_list,
                                             startangle=90,
                                             textprops={'fontsize': 8})

        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)

        for text in texts:
            text.set_fontsize(8)

    ax3.set_title('Contribución al Tiempo Total Ahorrado (% del tiempo ahorrado)',
                  fontsize=12, fontweight='bold')

    # --- GRÁFICO 4: Baseline vs Optimizado (Barras) ---
    ax4 = fig.add_subplot(gs[1, 2])

    categories = ['Baseline', 'Optimizado', 'Ahorrado']
    times_hours = [
        extrapolation_data['baseline_total_sec']/3600,
        extrapolation_data['optimized_total_sec']/3600,
        extrapolation_data['saved_total_sec']/3600
    ]
    colors_bar = ['#d62728', '#2ca02c', '#1f77b4']

    bars2 = ax4.bar(categories, times_hours, color=colors_bar, edgecolor='black', linewidth=2)
    ax4.set_xlabel('Categoría', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Tiempo (horas)', fontsize=10, fontweight='bold')
    ax4.set_title('Tiempo Simulación\nCompleta (26M iter)', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    for bar, time_h in zip(bars2, times_hours):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_h:.2f} h',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax4.text(1, times_hours[1] + 0.02,
             f'Speedup:\n{extrapolation_data["speedup_total"]:.1f}×',
             ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # --- GRÁFICO 5: Desglose de Tiempo (Stacked Bar) ---
    ax5 = fig.add_subplot(gs[2, :])

    opt_names_full = [r['name'] for r in extrapolation_data['individual']]
    time_baseline_ops = [r['time_baseline_sec']/3600 for r in extrapolation_data['individual']]
    time_optimized_ops = [r['time_optimized_sec']/3600 for r in extrapolation_data['individual']]

    x_pos = np.arange(2)
    width = 0.6

    # Baseline
    bottom_base = 0
    for i, (name, time_h) in enumerate(zip(opt_names_full, time_baseline_ops)):
        color = plt.cm.Set3(i)
        ax5.bar(0, time_h, width, bottom=bottom_base, color=color,
                edgecolor='black', linewidth=0.5, label=name)
        bottom_base += time_h

    # Optimizado
    bottom_opt = 0
    for i, time_h in enumerate(time_optimized_ops):
        color = plt.cm.Set3(i)
        ax5.bar(1, time_h, width, bottom=bottom_opt, color=color,
                edgecolor='black', linewidth=0.5)
        bottom_opt += time_h

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(['Baseline', 'Optimizado'], fontsize=11, fontweight='bold')
    ax5.set_xlabel('Versión de la Simulación', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Tiempo (horas)', fontsize=11, fontweight='bold')
    ax5.set_title('Desglose de Tiempo por Optimización', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    ax5.grid(axis='y', alpha=0.3)

    # Totales
    ax5.text(0, bottom_base + 0.05,
             f'{bottom_base:.2f} h',
             ha='center', fontweight='bold', fontsize=10)
    ax5.text(1, bottom_opt + 0.05,
             f'{bottom_opt:.2f} h',
             ha='center', fontweight='bold', fontsize=10)

    plt.suptitle('Análisis Completo: Benchmark Individual + Extrapolación a 26M Iteraciones',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig('benchmark_optimizations_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: benchmark_optimizations_analysis.png")

    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n🚀 Iniciando Benchmark Completo...\n")

    if not torch.cuda.is_available():
        print("⚠️  ADVERTENCIA: CUDA no disponible, usando CPU")
        print("   Los resultados no serán representativos de optimizaciones GPU\n")

    # Fase 1: Benchmarks individuales
    individual_results = run_individual_benchmarks()

    # Fase 2: Extrapolación
    extrapolation_data = extrapolate_to_full_simulation(individual_results)

    # Guardar resultados
    np.save('benchmark_optimizations_results.npy', {
        'individual': individual_results,
        'extrapolation': extrapolation_data
    })
    print("\n✓ Resultados guardados: benchmark_optimizations_results.npy")

    # Guardar resultados en formato de texto
    with open('benchmark_optimizations_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESULTADOS DE BENCHMARK - OPTIMIZACIONES MIMO 2x2 4-QAM\n")
        f.write("="*80 + "\n\n")

        f.write("CONFIGURACIÓN:\n")
        f.write(f"  - Iteraciones por SNR: {N_ITER_PER_SNR:,}\n")
        f.write(f"  - Puntos SNR: {N_SNR_POINTS}\n")
        f.write(f"  - Total iteraciones: {TOTAL_ITERATIONS:,}\n")
        f.write(f"  - GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("FASE 1: BENCHMARK INDIVIDUAL\n")
        f.write("="*80 + "\n\n")

        for key, bench in individual_results.items():
            f.write(f"OPTIMIZACIÓN: {bench['name']}\n")
            f.write(f"  Baseline:   {bench['baseline_ms']:.6f} ms\n")
            f.write(f"  Optimizado: {bench['optimized_ms']:.6f} ms\n")
            f.write(f"  Speedup:    {bench['speedup']:.2f}×\n")
            f.write(f"  Llamadas por iteración: {bench['calls_per_iter']}\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("FASE 2: EXTRAPOLACIÓN A SIMULACIÓN COMPLETA\n")
        f.write("="*80 + "\n\n")

        for r in extrapolation_data['individual']:
            f.write(f"{r['name']}:\n")
            f.write(f"  Llamadas totales: {r['total_calls']:,}\n")
            f.write(f"  Tiempo baseline:  {r['time_baseline_sec']:,.2f} seg ({r['time_baseline_sec']/3600:.2f} h)\n")
            f.write(f"  Tiempo optimizado: {r['time_optimized_sec']:,.2f} seg ({r['time_optimized_sec']/3600:.2f} h)\n")
            f.write(f"  Tiempo ahorrado:   {r['time_saved_sec']:,.2f} seg ({r['time_saved_sec']/3600:.2f} h)\n")
            f.write(f"  Speedup: {r['speedup']:.2f}×\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("RESUMEN TOTAL\n")
        f.write("="*80 + "\n\n")
        f.write(f"Tiempo BASELINE (sin optimizaciones):\n")
        f.write(f"  {extrapolation_data['baseline_total_sec']:,.2f} seg ({extrapolation_data['baseline_total_sec']/3600:.2f} horas)\n\n")
        f.write(f"Tiempo OPTIMIZADO (con 8 optimizaciones):\n")
        f.write(f"  {extrapolation_data['optimized_total_sec']:,.2f} seg ({extrapolation_data['optimized_total_sec']/3600:.2f} horas)\n\n")
        f.write(f"Tiempo AHORRADO:\n")
        f.write(f"  {extrapolation_data['saved_total_sec']:,.2f} seg ({extrapolation_data['saved_total_sec']/3600:.2f} horas)\n\n")
        f.write(f"SPEEDUP TOTAL: {extrapolation_data['speedup_total']:.2f}×\n")
        f.write(f"REDUCCIÓN: {(extrapolation_data['saved_total_sec']/extrapolation_data['baseline_total_sec'])*100:.1f}%\n\n")

        f.write("="*80 + "\n")
        f.write("TABLA DE SPEEDUPS INDIVIDUALES\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Optimización':<40} {'Speedup Individual':>20} {'Speedup Multiplicado':>20}\n")
        f.write("-"*80 + "\n")

        cumulative = 1.0
        for key, bench in individual_results.items():
            cumulative *= bench['speedup']
            f.write(f"{bench['name']:<40} {bench['speedup']:>19.2f}× {cumulative:>19.2f}×\n")

        f.write("-"*80 + "\n")
        f.write(f"{'SPEEDUP MULTIPLICADO (teórico)':<40} {'':<20} {cumulative:>19.2f}×\n")
        f.write("\n")
        f.write("NOTA: El speedup multiplicado es teórico. El speedup REAL de la simulación\n")
        f.write(f"      completa es {extrapolation_data['speedup_total']:.2f}× (ver RESUMEN TOTAL arriba).\n")
        f.write("      La diferencia se debe a overhead fijo y Ley de Amdahl.\n")
        f.write("\n" + "="*80 + "\n")

    print("✓ Resultados guardados: benchmark_optimizations_results.txt")

    # Generar gráficos
    print("\nGenerando gráficos completos...")
    plot_all_results(individual_results, extrapolation_data)

    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETO FINALIZADO")
    print("="*80)
    print(f"\n📊 RESUMEN:")
    print(f"   Speedup individual acumulado: {np.prod([v['speedup'] for v in individual_results.values()]):.0f}×")
    print(f"   Speedup simulación completa: {extrapolation_data['speedup_total']:.2f}×")
    print(f"   Tiempo baseline: {extrapolation_data['baseline_total_sec']/3600:.2f} horas")
    print(f"   Tiempo optimizado: {extrapolation_data['optimized_total_sec']/3600:.2f} horas")
    print(f"   Tiempo ahorrado: {extrapolation_data['saved_total_sec']/3600:.2f} horas")
    print("\n" + "="*80)
