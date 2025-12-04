#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de Resultados BER guardados
====================================

Este script muestra cómo cargar y analizar los resultados BER
guardados en formato .npy después de ejecutar la simulación.

Author: Leonel Roberto Perea Trejo
Date: Noviembre 2024
"""

import numpy as np
import matplotlib
# Configurar backend apropiado para macOS
import sys
if sys.platform == 'darwin':  # macOS
    try:
        matplotlib.use('MacOSX')  # Backend nativo de macOS
    except:
        matplotlib.use('Agg')  # Fallback: solo guardar archivos (sin ventanas)
else:
    try:
        matplotlib.use('TkAgg')  # Backend para Windows/Linux
    except:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CARGAR RESULTADOS GUARDADOS
# ==============================================================================

print("="*70)
print("Cargando resultados BER guardados...")
print("="*70)

# Cargar el archivo .npy
# El parámetro allow_pickle=True es necesario para objetos Python complejos
results = np.load('resultados/BER_results_MIMO_2x2_all_strategies.npy',
                  allow_pickle=True).item()

# ==============================================================================
# 2. EXPLORAR ESTRUCTURA DE DATOS
# ==============================================================================

print("\n" + "="*70)
print("ESTRUCTURA DE DATOS GUARDADOS")
print("="*70)

# Ver qué contiene el diccionario
print("\nClaves disponibles en el archivo:")
for key in results.keys():
    print(f"  - {key}")

# Mostrar información detallada
print("\n" + "-"*70)
print("INFORMACIÓN DETALLADA:")
print("-"*70)

# SNR values
print(f"\nRango SNR:")
print(f"  Valores: {results['SNR_dB']}")
print(f"  Número de puntos: {len(results['SNR_dB'])}")
print(f"  Mínimo: {results['SNR_dB'][0]} dB")
print(f"  Máximo: {results['SNR_dB'][-1]} dB")

# Metadata
print(f"\nMetadatos de la simulación:")
print(f"  Iteraciones: {results['n_iterations']:,}")
print(f"  Configuración MIMO: {results['MIMO_config']}")
print(f"  Modulación: {results['modulation']}")

# Detectores disponibles
print(f"\nDetectores evaluados:")
detectors = []
for key in results.keys():
    if key.startswith('BER_'):
        detector_name = key.replace('BER_', '')
        detectors.append(detector_name)
        ber_values = results[key]
        print(f"  - {detector_name}: {len(ber_values)} puntos BER")
        print(f"    BER mínimo: {np.min(ber_values):.2e}")
        print(f"    BER máximo: {np.max(ber_values):.2e}")

# ==============================================================================
# 3. ACCEDER A DATOS ESPECÍFICOS
# ==============================================================================

print("\n" + "="*70)
print("ACCESO A DATOS ESPECÍFICOS")
print("="*70)

# Extraer vectores de datos
SNR_dB = results['SNR_dB']
BER_ML = results['BER_ML']
BER_OneHot = results['BER_OneHot']
BER_LabelEncoder = results['BER_LabelEncoder']

# Buscar un SNR específico
target_snr = 10  # dB
idx = np.where(SNR_dB == target_snr)[0]
if len(idx) > 0:
    idx = idx[0]
    print(f"\nResultados @ SNR = {target_snr} dB:")
    print(f"  ML:            BER = {BER_ML[idx]:.4e}")
    print(f"  One-Hot:       BER = {BER_OneHot[idx]:.4e}")
    print(f"  Label Encoder: BER = {BER_LabelEncoder[idx]:.4e}")

# ==============================================================================
# 4. ANÁLISIS: ENCONTRAR SNR PARA BER OBJETIVO
# ==============================================================================

print("\n" + "="*70)
print("ANÁLISIS: SNR REQUERIDO PARA BER = 10⁻³")
print("="*70)

target_ber = 1e-3

def find_snr_for_target_ber(snr_array, ber_array, target_ber):
    """
    Encuentra el SNR requerido para alcanzar un BER objetivo
    mediante interpolación logarítmica.
    """
    # Verificar que el BER objetivo está en el rango
    if np.min(ber_array) > target_ber:
        return None, "BER objetivo no alcanzado (aumentar SNR máximo)"
    if np.max(ber_array) < target_ber:
        return None, "BER objetivo superado en todo el rango"

    # Encontrar puntos que rodean el BER objetivo
    idx = np.where(ber_array < target_ber)[0]
    if len(idx) == 0:
        return None, "No se encontró el BER objetivo"

    idx = idx[0]  # Primer punto que cumple BER < target

    if idx == 0:
        return snr_array[0], "Exacto (primer punto)"

    # Interpolación logarítmica
    snr1, snr2 = snr_array[idx-1], snr_array[idx]
    ber1, ber2 = ber_array[idx-1], ber_array[idx]

    # Evitar log de cero
    ber1 = max(ber1, 1e-10)
    ber2 = max(ber2, 1e-10)

    log_ber1 = np.log10(ber1)
    log_ber2 = np.log10(ber2)
    log_target = np.log10(target_ber)

    # Interpolación lineal en escala logarítmica
    snr_interp = snr1 + (snr2 - snr1) * (log_target - log_ber1) / (log_ber2 - log_ber1)

    return snr_interp, "Interpolado"

# Calcular SNR requerido para cada detector
print(f"\nSNR requerido para alcanzar BER = {target_ber:.0e}:\n")

for detector_name in detectors:
    ber_key = f'BER_{detector_name}'
    if ber_key in results:
        ber_values = results[ber_key]
        snr_req, status = find_snr_for_target_ber(SNR_dB, ber_values, target_ber)

        if snr_req is not None:
            # Calcular gap respecto a ML
            snr_ml, _ = find_snr_for_target_ber(SNR_dB, BER_ML, target_ber)
            gap = snr_req - snr_ml if snr_ml else 0

            print(f"  {detector_name:20s}: {snr_req:6.2f} dB  (Gap: {gap:+.2f} dB)")
        else:
            print(f"  {detector_name:20s}: {status}")

# ==============================================================================
# 5. GRAFICAR RESULTADOS
# ==============================================================================

print("\n" + "="*70)
print("GENERANDO GRÁFICOS...")
print("="*70)

# Crear figura con múltiples subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis Completo de Resultados BER', fontsize=16, fontweight='bold')

# --- Subplot 1: Curvas BER completas ---
ax1 = axes[0, 0]
ax1.semilogy(SNR_dB, BER_ML, 's-', linewidth=2, label='ML (Optimal)')
ax1.semilogy(SNR_dB, BER_OneHot, 'o--', linewidth=2, label='One-Hot')
ax1.semilogy(SNR_dB, BER_LabelEncoder, '^-.', linewidth=2, label='Label Encoder')

# Línea de referencia BER = 10^-3
ax1.axhline(y=1e-3, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='BER = 10⁻³')
ax1.grid(True, which='both', alpha=0.3)
ax1.set_xlabel('SNR (dB)', fontsize=11)
ax1.set_ylabel('Bit Error Rate (BER)', fontsize=11)
ax1.set_title('Curvas BER Completas', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylim([1e-6, 1])

# --- Subplot 2: Zoom en región de interés (BER = 10^-3) ---
ax2 = axes[0, 1]
ax2.semilogy(SNR_dB, BER_ML, 's-', linewidth=2, label='ML (Optimal)')
ax2.semilogy(SNR_dB, BER_OneHot, 'o--', linewidth=2, label='One-Hot')
ax2.semilogy(SNR_dB, BER_LabelEncoder, '^-.', linewidth=2, label='Label Encoder')
ax2.axhline(y=1e-3, color='red', linestyle=':', linewidth=2, alpha=0.7, label='BER = 10⁻³')
ax2.grid(True, which='both', alpha=0.3)
ax2.set_xlabel('SNR (dB)', fontsize=11)
ax2.set_ylabel('Bit Error Rate (BER)', fontsize=11)
ax2.set_title('Zoom: Región @ BER = 10⁻³', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim([8, 15])
ax2.set_ylim([1e-4, 1e-2])

# --- Subplot 3: Gap vs ML detector ---
ax3 = axes[1, 0]

# Calcular gap para cada punto SNR
gaps_onehot = []
gaps_label = []
valid_snrs = []

for i, snr in enumerate(SNR_dB):
    if BER_ML[i] > 0 and BER_ML[i] < 0.5:  # Solo donde tiene sentido calcular gap
        # Gap en dB (aproximado, basado en SNR donde se alcanza mismo BER)
        # Simplificación: diferencia de BER en log scale
        gap_oh = 10 * np.log10(BER_OneHot[i] / BER_ML[i]) if BER_OneHot[i] > 0 else 0
        gap_le = 10 * np.log10(BER_LabelEncoder[i] / BER_ML[i]) if BER_LabelEncoder[i] > 0 else 0

        gaps_onehot.append(gap_oh)
        gaps_label.append(gap_le)
        valid_snrs.append(snr)

ax3.plot(valid_snrs, gaps_onehot, 'o-', linewidth=2, label='One-Hot vs ML')
ax3.plot(valid_snrs, gaps_label, '^-', linewidth=2, label='Label Encoder vs ML')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('SNR (dB)', fontsize=11)
ax3.set_ylabel('Gap vs ML (dB)', fontsize=11)
ax3.set_title('Diferencia de Rendimiento vs ML', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)

# --- Subplot 4: Estadísticas por detector ---
ax4 = axes[1, 1]
ax4.axis('off')

# Crear tabla de estadísticas
stats_text = "ESTADÍSTICAS DE RENDIMIENTO\n"
stats_text += "="*50 + "\n\n"

for detector_name in detectors:
    ber_key = f'BER_{detector_name}'
    if ber_key in results:
        ber_values = results[ber_key]
        snr_req, _ = find_snr_for_target_ber(SNR_dB, ber_values, target_ber)

        stats_text += f"{detector_name}:\n"
        if snr_req is not None:
            snr_ml, _ = find_snr_for_target_ber(SNR_dB, BER_ML, target_ber)
            gap = snr_req - snr_ml if snr_ml else 0

            stats_text += f"  SNR @ 10⁻³:  {snr_req:.2f} dB\n"
            stats_text += f"  Gap vs ML:   {gap:+.2f} dB\n"
        else:
            stats_text += f"  No alcanzó BER = 10⁻³\n"

        stats_text += f"  BER mínimo:  {np.min(ber_values):.2e}\n"
        stats_text += f"  BER máximo:  {np.max(ber_values):.2e}\n"
        stats_text += "\n"

ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('resultados/Analisis_Resultados_BER.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: resultados/Analisis_Resultados_BER.png")

plt.show()

# ==============================================================================
# 6. EXPORTAR RESULTADOS A CSV (para Excel, etc.)
# ==============================================================================

print("\n" + "="*70)
print("EXPORTANDO A CSV...")
print("="*70)

# Crear un archivo CSV con los datos
csv_data = np.column_stack([SNR_dB, BER_ML, BER_OneHot, BER_LabelEncoder])

header = "SNR_dB,BER_ML,BER_OneHot,BER_LabelEncoder"
csv_filename = 'resultados/BER_results_table.csv'

np.savetxt(csv_filename, csv_data, delimiter=',', header=header, comments='',
           fmt=['%.1f', '%.6e', '%.6e', '%.6e'])

print(f"✓ Datos exportados a: {csv_filename}")
print("  (Puedes abrir este archivo en Excel, LibreOffice, etc.)")

# ==============================================================================
# 7. COMPARACIÓN CON RESULTADOS ESPERADOS
# ==============================================================================

print("\n" + "="*70)
print("COMPARACIÓN CON RESULTADOS ESPERADOS (PAPER)")
print("="*70)

# Valores esperados del paper LatinCom 2025
expected_results = {
    'ML': {'snr_at_1e-3': 10.50, 'gap': 0.00},
    'OneHot': {'snr_at_1e-3': 11.50, 'gap': 1.00},
    'LabelEncoder': {'snr_at_1e-3': 11.00, 'gap': 0.50}  # ELM en paper
}

print("\nComparación @ BER = 10⁻³:")
print("-" * 70)
print(f"{'Detector':<20} {'Esperado (dB)':<15} {'Obtenido (dB)':<15} {'Diferencia':<10}")
print("-" * 70)

for detector_name in ['ML', 'OneHot', 'LabelEncoder']:
    ber_key = f'BER_{detector_name}'
    if ber_key in results:
        ber_values = results[ber_key]
        snr_obtained, _ = find_snr_for_target_ber(SNR_dB, ber_values, target_ber)

        if snr_obtained is not None and detector_name in expected_results:
            snr_expected = expected_results[detector_name]['snr_at_1e-3']
            diff = snr_obtained - snr_expected

            symbol = "✓" if abs(diff) < 1.0 else "⚠"
            print(f"{detector_name:<20} {snr_expected:<15.2f} {snr_obtained:<15.2f} {diff:+.2f} dB {symbol}")

print("\n" + "="*70)
print("ANÁLISIS COMPLETADO")
print("="*70)
print("\nArchivos generados:")
print("  1. resultados/Analisis_Resultados_BER.png")
print("  2. resultados/BER_results_table.csv")
print("\nPara re-ejecutar este análisis en el futuro:")
print("  python analizar_resultados_BER.py")
print("="*70)
