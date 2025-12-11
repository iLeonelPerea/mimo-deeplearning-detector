# MIMO Deep Learning Detector - Python Implementation

> Implementación completa en Python/PyTorch de detectores MIMO basados en Deep Learning con backpropagation, como alternativa al enfoque Extreme Learning Machine (ELM).

**Basado en:** [roilhi/mimo-dl-detector](https://github.com/roilhi/mimo-dl-detector) - Implementación original MATLAB/ELM

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)

---

## 🎯 Descripción

Detectores basados en Deep Learning para sistemas MIMO 2×2 con modulación 4-QAM, implementando tres estrategias de etiquetado diferentes. El código está optimizado para coincidir exactamente con la implementación MATLAB de referencia.

### Sistema Evaluado

- **MIMO:** 2×2 (2 transmisores, 2 receptores)
- **Modulación:** 4-QAM
- **Canal:** Rayleigh fading + AWGN
- **SNR:** 0-25 dB (26 puntos)
- **Iteraciones Monte Carlo:** 1,000,000/punto

---

## ✨ Características Principales

### 1. Tres Estrategias de Etiquetado

| Estrategia | Salidas | Activación Oculta | Activación Salida |
|-----------|---------|-------------------|-------------------|
| **One-Hot** | 16 (M^Nt) | ReLU | Softmax |
| **Label Encoder** | 4 (log₂(M)×Nt) | Sigmoid + ReLU | Sigmoid |
| **One-Hot Per Antenna** | 8 (M×Nt) | Sigmoid + ReLU | Sigmoid |

### 2. Configuración Flexible

Dos parámetros configurables en todos los scripts:

```python
USE_ZF = False    # Zero-Forcing equalization
USE_BIAS = False  # Bias en capa oculta
```

**Configuraciones disponibles:**

| Config | USE_ZF | USE_BIAS | Matching MATLAB | Parámetros |
|--------|--------|----------|-----------------|------------|
| **Default** ✅ | False | False | ✅ Sí | ~1,600 |
| Opción 2 | True | True | ❌ No | ~1,700 |
| Opción 3 | True | False | ❌ No | ~1,600 |
| Opción 4 | False | True | ❌ No | ~1,700 |

### 3. Optimizaciones de Rendimiento

**7 optimizaciones implementadas** logrando ~15× speedup:

1. ⚡ Eliminación transferencias CPU↔GPU (3-5× speedup)
2. 🔥 Pre-cómputo productos H*s para ML (1.3× speedup)
3. 📊 Pre-cómputo √SNR (1.2× speedup)
4. 📌 XOR para conteo bits (5× en conteo)
5. 🚀 Generación directa ruido complejo (1.2× speedup)
6. ⚡ Saltar softmax innecesario (1.3× speedup)
7. 🔧 Lookup table errores bit (2-3× speedup)

**Resultado:** ~15 horas → ~90 minutos (GPU RTX 4090)

---

## 🏗️ Arquitectura

### Red Neuronal

```
Input (4) → Hidden (100) + [Sigmoid] + ReLU → Output (16/4/8)
                             ^
                             |
                    Opcional según estrategia
```

**Configuración matching MATLAB:**
- ✅ Sin bias en capa oculta (`USE_BIAS=False`)
- ✅ Sin Zero-Forcing (`USE_ZF=False`)
- ✅ One-Hot: solo ReLU
- ✅ Label Encoder/Per-Antenna: Sigmoid + ReLU

### Modelo de Canal

```
r = √SNR · H · x + n
```

- **H**: Canal fijo normalizado
- **x**: Símbolos 4-QAM transmitidos
- **n**: Ruido AWGN ~ CN(0, 1) con **varianza fija** (no escalado por SNR)
- **SNR**: Controlado únicamente escalando la señal, no el ruido

---

## 🚀 Instalación

### Requisitos

- Python 3.11+
- PyTorch 2.5+
- CUDA 12.1+ (opcional, recomendado)

### Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/mimo-dl-detector.git
cd mimo-dl-detector

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o venv\Scripts\activate  # Windows

# Instalar PyTorch con CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias
pip install numpy matplotlib tqdm scikit-learn seaborn
```

---

## 💻 Uso

### 1. Entrenamiento (Matching MATLAB)

```bash
# Configuración por defecto: USE_ZF=False, USE_BIAS=False
python modelMIMO_2x2_4QAM_OneHot.py
python modelMIMO_2x2_4QAM_LabelEncoder.py
python modelMIMO_2x2_4QAM_DoubleOneHot.py
```

**Características del entrenamiento:**
- ✅ **SNR Variable**: 1-20 dB aleatorio por muestra
- ✅ **Sin normalización de datos**: Red aprende magnitudes reales
- ✅ **Ruido sin escalar**: `n ~ CN(0, 1)` con varianza fija

**Salida:** Modelos guardados en raíz del proyecto
- `modelMIMO_2x2_4QAM_OneHot.pth`
- `modelMIMO_2x2_4QAM_LabelEncoder.pth`
- `modelMIMO_2x2_4QAM_DoubleOneHot.pth`

### 2. Evaluación BER

```bash
python ber_4qam_mimo_2x2_all.py
```

**Configuración en el script:**
```python
# Líneas 95-96
USE_ZF = False    # Matching MATLAB
USE_BIAS = False  # Matching MATLAB
```

**⚠️ IMPORTANTE:** `USE_ZF` y `USE_BIAS` deben coincidir con la configuración usada en entrenamiento.

**Monitoreo en tiempo real:**
- Durante la simulación se guarda `BER_MIMO_2x2_4QAM_progress.png` (actualizado cada SNR)
- Puedes abrir/refrescar este archivo para ver el progreso sin interrumpir la ejecución
- Se elimina automáticamente al finalizar

**Salidas finales:**
- `BER_MIMO_2x2_4QAM.png` - Gráfica BER final
- `BER_results_MIMO_2x2.npy` - Datos NumPy
- `BER_results_MIMO_2x2.txt` - Tabla texto con tiempos por SNR

### 3. Cambiar Configuración

Para experimentar con otras configuraciones:

```python
# En archivos modelMIMO_*.py (líneas 292-293)
USE_ZF = True     # Habilitar Zero-Forcing
USE_BIAS = True   # Habilitar bias

# Ejecutar entrenamiento
python modelMIMO_2x2_4QAM_OneHot.py

# Actualizar en ber_4qam_mimo_2x2_all.py (líneas 95-96)
USE_ZF = True     # Debe coincidir con entrenamiento
USE_BIAS = True   # Debe coincidir con entrenamiento

# Ejecutar evaluación
python ber_4qam_mimo_2x2_all.py
```

---

## 📁 Estructura del Proyecto

```
mimo-dl-detector/
│
├── README.md                              # Este archivo
├── CONFIGURATION.md                       # Guía de configuración detallada
│
├── modelMIMO_2x2_4QAM_OneHot.py          # Entrenamiento One-Hot
├── modelMIMO_2x2_4QAM_LabelEncoder.py    # Entrenamiento Label Encoder
├── modelMIMO_2x2_4QAM_DoubleOneHot.py    # Entrenamiento Per-Antenna
│
├── ber_4qam_mimo_2x2_all.py              # Evaluación BER
│
├── modelMIMO_*.pth                        # Modelos entrenados
│
└── detector_ELM_2x2_all.m                # Referencia MATLAB
```

---

## 📊 Diferencias Python vs MATLAB

| Aspecto | MATLAB (ELM) | Python (Este Código) |
|---------|--------------|----------------------|
| **Método** | Extreme Learning Machine | Deep Learning (backprop) |
| **Pesos entrada** | Aleatorios fijos | Aprendidos |
| **Pesos salida** | Pseudoinversa analítica | SGD iterativo |
| **Pseudoinversa** | ❌ No usa | ❌ No usa (default) |
| **Bias oculta** | ❌ No usa (b_oh=0) | ❌ No usa (default) |
| **Activación OH** | ReLU | ReLU ✅ |
| **Activación LE** | Sigmoid + ReLU | Sigmoid + ReLU ✅ |
| **Activación OHA** | Sigmoid + ReLU | Sigmoid + ReLU ✅ |
| **Tiempo entrena** | ~segundos | ~2-3 minutos |
| **Framework** | MATLAB | PyTorch |

**Con configuración default (`USE_ZF=False`, `USE_BIAS=False`):** Coincidencia exacta con MATLAB en arquitectura y procesamiento de señales.

---

## ⚙️ Parámetros de Configuración

### USE_ZF (Zero-Forcing Equalization)

**False (default):** Sin pseudoinversa, matching MATLAB
```python
r_processed = r  # Señal directa
```

**True:** Con ecualización Zero-Forcing
```python
r_processed = H_inv @ r  # Señal ecualizada
```

### USE_BIAS (Bias en capa oculta)

**False (default):** Sin bias, matching MATLAB `b_oh=0`
```python
nn.Linear(input_size, hidden_size, bias=False)
# Parámetros: ~1,600
```

**True:** Con bias aprendido
```python
nn.Linear(input_size, hidden_size, bias=True)
# Parámetros: ~1,700 (+100 bias)
```

---

## 🤝 Contribuciones

### Implementación Python

**Autor:** Leonel Roberto Perea Trejo
**Email:** iticleonel.leonel@gmail.com
**Fecha:** Enero 2025

**Contribuciones:**
- ✅ Implementación completa Python/PyTorch
- ✅ Configuración flexible (USE_ZF, USE_BIAS)
- ✅ Matching exacto con MATLAB
- ✅ 8 optimizaciones de rendimiento
- ✅ Documentación técnica
- ✅ Compatibilidad cross-platform

### Trabajo de Referencia

**Autores:** Roilhi Frajo Ibarra Hernández, Francisco Rubén Castillo-Soria
**Email:** roilhi.ibarra@uaslp.mx
**Repositorio:** [roilhi/mimo-dl-detector](https://github.com/roilhi/mimo-dl-detector)

---

## 📄 Licencia

GPLv2 License - Ver LICENSE para detalles.

```
Copyright (C) 2025 Leonel Roberto Perea Trejo

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2.
```

---

## 📞 Contacto

**Implementación Python:**
Leonel Roberto Perea Trejo - iticleonel.leonel@gmail.com

**Referencia MATLAB/ELM:**
Prof. Roilhi Ibarra - roilhi.ibarra@uaslp.mx

---

**Última Actualización:** Enero 2025
**Versión:** 2.0.0
**Estado:** Mantenido activamente
