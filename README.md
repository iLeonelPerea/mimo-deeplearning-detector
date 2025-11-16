# MIMO Deep Learning Detector - Comparative Study

> Implementación de Deep Learning con backpropagation completo para detección de señales MIMO, como estudio comparativo con el enfoque Extreme Learning Machine (ELM).

**Based on:** [roilhi/mimo-dl-detector](https://github.com/roilhi/mimo-dl-detector) - Original MATLAB/ELM implementation

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Características Principales](#-características-principales)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Resultados Clave](#-resultados-clave)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Documentación Técnica](#-documentación-técnica)
- [Contribuciones y Referencias](#-contribuciones-y-referencias)
- [Licencia](#-licencia)

---

## 🎯 Descripción General

Este proyecto implementa **detectores basados en Deep Learning** para sistemas de comunicación MIMO (Multiple-Input Multiple-Output) utilizando **optimización basada en gradientes con backpropagation completo**, como parte de un estudio comparativo con el enfoque **Extreme Learning Machine (ELM)** propuesto en:

> **Ibarra-Hernández, R.F. et al.** (2025). "Extreme Learning Machine Signal Detection for MIMO Channels." *IEEE LatinCom 2025*.

### Diferencias Metodológicas Clave

| Aspecto | ELM (Trabajo de Referencia) | Deep Learning (Este Trabajo) |
|---------|----------------------------|------------------------------|
| **Método de Aprendizaje** | Extreme Learning Machine | Deep Learning con Backpropagation |
| **Pesos de Entrada** | Aleatorios (fijos) | **Aprendidos vía gradiente** |
| **Pesos de Salida** | Pseudoinversa (analítico) | **Aprendidos vía SGD** |
| **Tiempo de Entrenamiento** | ~segundos (más rápido) | ~2-3 minutos (más lento) |
| **Rendimiento BER (Label Encoder)** | ~0.5 dB gap del ML óptimo | **~0.3 dB gap** ⭐ **40% mejor** |
| **Muestras de Entrenamiento** | 10,000 @ SNR fijo (3 dB) | 100,000 @ SNR variable (0-20 dB) |
| **Aceleración GPU** | No disponible | **Soporte completo CUDA** |
| **Framework** | MATLAB (manual) | PyTorch (autograd) |

### Sistema Evaluado

- **Configuración MIMO:** 2×2 (2 transmisores, 2 receptores)
- **Modulación:** 4-QAM (4 símbolos por antena)
- **Canal:** Rayleigh fading con ruido AWGN
- **Rango SNR:** 0-25 dB (26 puntos)
- **Iteraciones Monte Carlo:** 1,000,000 por punto SNR

---

## ✨ Características Principales

### 1. Tres Estrategias de Etiquetado Implementadas

#### **One-Hot Encoding**
- 16 salidas (M^Nt)
- Clasificación directa de combinaciones de símbolos
- Activación: Softmax
- **Rendimiento:** Gap de 1.0 dB vs ML óptimo

#### **Label Encoder (Direct Symbol Encoding)**
- 4 salidas (log₂(M)×Nt)
- Predicción de bits de signo
- Activación: ReLU (mejor que Sigmoid en Deep Learning)
- **Rendimiento:** Gap de 0.3 dB vs ML óptimo ⭐ **Mejor estrategia**

#### **Per-Antenna (Double One-Hot)**
- 8 salidas (M×Nt)
- One-hot por antena
- Activación: Sigmoid (crítico para estructura dual)
- **Rendimiento:** Gap de ~0.8-1.0 dB vs ML óptimo

### 2. Optimizaciones de Rendimiento

Este proyecto incluye **8 optimizaciones mayores** que logran una aceleración de **~17×**:

1. ⚡ **Eliminación de transferencias CPU↔GPU** (3-5× speedup)
2. 🔥 **Pre-cómputo de pseudoinversa** (5× speedup)
3. 🔥 **Pre-cómputo de productos ML** (1.3× speedup)
4. 📊 **Pre-cómputo de √SNR** (1.2× speedup)
5. 📌 **XOR para conteo de bits** (5× en conteo)
6. 🚀 **Generación directa de ruido complejo** (1.2× speedup)
7. ⚡ **Saltar softmax innecesario** (1.3× speedup)
8. 🔧 **Lookup table para errores de bit** (2-3× speedup)

**Impacto combinado:** Reducción de ~15 horas → ~90 minutos (GPU RTX 4090)

### 3. Análisis Automático en BER = 10⁻³

Implementa la metodología del paper LatinCom 2025 con:
- Interpolación logarítmica precisa
- Cálculo automático de gaps vs ML
- Clasificación de rendimiento (Excellent/Good/Acceptable)
- Tablas comparativas y visualizaciones mejoradas

### 4. Visualización Interactiva en Tiempo Real

- **Backend no bloqueante** (TkAgg)
- Zoom/pan durante la simulación
- Actualización de curvas BER en tiempo real
- Compatibilidad Windows/Linux/macOS

---

## 🏗️ Arquitectura del Sistema

### Red Neuronal (Común a todas las estrategias)

```
Capa de Entrada (4 neuronas)
       ↓
       [Re(r₁), Im(r₁), Re(r₂), Im(r₂)]
       ↓
Capa Oculta (100 neuronas) + ReLU
       ↓
Capa de Salida (16/4/8 según estrategia)
       ↓
Softmax/Sigmoid (según estrategia)
```

**Parámetros totales:** ~2,116 (compacto y eficiente)

### Modelo de Canal MIMO

```
r = √SNR · H · x + n
```

Donde:
- **H** ∈ ℂ²ˣ² : Matriz de canal Rayleigh ~ CN(0,1)
- **x** ∈ ℂ² : Vector de símbolos transmitidos (4-QAM)
- **n** ∈ ℂ² : Ruido AWGN con **varianza fija** ~ CN(0,σ²)
- **SNR**: Relación señal-ruido (escala lineal)

**Nota crítica:** La varianza del ruido es **constante** (estándar universal en comunicaciones inalámbricas)

### Proceso de Detección

**Detector ML (referencia óptima):**
```python
# Búsqueda exhaustiva sobre todas las 16 combinaciones
distances = ||r - √SNR · H · s||² for all s
s_hat = argmin(distances)
```

**Detectores Deep Learning:**
```python
# 1. Ecualización Zero-Forcing
r_eq = H⁺ · r

# 2. Extracción de características
features = [Re(r_eq₁), Im(r_eq₁), Re(r_eq₂), Im(r_eq₂)]

# 3. Red neuronal
output = model(features)
s_hat = decode(output)  # Según estrategia
```

---

## 📊 Resultados Clave

### Comparación de Rendimiento @ BER = 10⁻³

| Detector | SNR Requerido | Gap vs ML | Mejora vs ELM | Clasificación |
|----------|---------------|-----------|---------------|---------------|
| **ML (Óptimo)** | 10.50 dB | 0.00 dB | - | Referencia |
| **One-Hot (ReLU)** | 11.50 dB | 1.00 dB | Similar | ✅ Excellent |
| **Label Encoder (ReLU)** | 10.80 dB | **0.30 dB** | **0.2 dB mejor** | ✅✅ Outstanding |
| **Label Encoder (Sigmoid)** | ~11.20 dB | ~0.70 dB | Similar | ✅ Excellent |
| **Per-Antenna (Sigmoid)** | ~11.30 dB | ~0.80 dB | Similar | ✅ Excellent |

### Hallazgos Científicos Principales

#### 1. Deep Learning Supera a ELM para Label Encoder

**Resultado experimental:**
- Deep Learning (este trabajo): **0.3 dB gap**
- ELM (paper LatinCom): **~0.5 dB gap**
- **Mejora: 40% en reducción del gap de SNR**

**Explicación:**
- La optimización basada en gradientes aprende mejores representaciones de características que las proyecciones aleatorias fijas de ELM
- Todos los pesos (entrada, ocultos, salida) se optimizan iterativamente
- SNR variable en entrenamiento (0-20 dB) vs fijo (3 dB) mejora la generalización

#### 2. Selección de Función de Activación Depende de la Estructura de Salida

| Estrategia | Tipo de Salida | Mejor Activación | Razonamiento |
|-----------|---------------|------------------|--------------|
| **One-Hot** | Clase única | Softmax | Clasificación multi-clase estándar |
| **Label Encoder** | Bits binarios | **ReLU** (Deep Learning) | Fronteras de decisión nítidas |
| **Per-Antenna** | One-hot dual | **Sigmoid** | Interpretación probabilística por grupo |

**Descubrimiento crítico:** ReLU falla para Per-Antenna (2.0 dB gap) porque las salidas no acotadas [0,∞) causan competencia global. Sigmoid [0,1] proporciona separación por antena.

#### 3. Trade-off Velocidad vs Calidad

**ELM (Referencia):**
- ⚡ Entrenamiento: ~segundos
- 📊 BER (Label Encoder): ~0.5 dB gap
- 🔧 Implementación: Simple (pseudoinversa directa)

**Deep Learning (Este Trabajo):**
- ⏱️ Entrenamiento: ~2-3 minutos
- 📊 BER (Label Encoder): **~0.3 dB gap** (mejor)
- 🔧 Implementación: Más compleja pero estándar (PyTorch)

**Conclusión:** Para aplicaciones críticas donde cada décima de dB importa, Deep Learning justifica el costo computacional adicional de entrenamiento.

---

## 🚀 Instalación

### Requisitos del Sistema

**Requeridos:**
- Python 3.11-3.13 (recomendado 3.11)
- GPU NVIDIA con soporte CUDA (opcional pero recomendado para evaluación BER)

**Recomendado:**
- GPU: NVIDIA RTX 3080 o superior
- RAM: 16 GB
- CUDA: 12.1+ o 13.0+

### Instalación Paso a Paso

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/mimo-dl-detector.git
cd mimo-dl-detector
```

2. **Crear entorno virtual (recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows
```

3. **Instalar PyTorch con CUDA:**
```bash
# Para NVIDIA GPU (compatible con CUDA 12.1/13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Para CPU only
pip install torch torchvision torchaudio
```

4. **Instalar dependencias restantes:**
```bash
pip install numpy matplotlib tqdm scikit-learn seaborn
```

5. **Verificar instalación:**
```python
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

### Compatibilidad de Plataforma

**Windows:**
- ✅ Totalmente funcional con GPU CUDA
- ⚠️ `torch.compile()` no disponible (pérdida de ~1.5× speedup)
- Solución: `torch._dynamo` deshabilitado automáticamente

**Linux:**
- ✅ Rendimiento óptimo con `torch.compile()`
- ✅ Backend Triton disponible
- Recomendado para máximo rendimiento

**macOS:**
- ✅ Funcional en CPU
- ⚠️ Sin aceleración GPU (MPS experimental)

---

## 💻 Uso

### 1. Entrenamiento de Modelos

#### Entrenar todas las estrategias:

```bash
# One-Hot Encoding
python modelMIMO_2x2_4QAM_OneHot.py

# Label Encoder (ReLU)
python modelMIMO_2x2_4QAM_LabelEncoder.py

# Label Encoder (Sigmoid)
python modelMIMO_2x2_4QAM_LabelEncoder_Sigmoid.py

# Per-Antenna (ReLU)
python modelMIMO_2x2_4QAM_DoubleOneHot.py

# Per-Antenna (Sigmoid)
python modelMIMO_2x2_4QAM_DoubleOneHot_Sigmoid.py
```

**Salida esperada:**
- Modelos entrenados guardados en `trained_models/`
- Tiempo de entrenamiento: ~2-3 minutos por modelo (GPU)
- Accuracies típicos: 95-98% en conjunto de prueba

### 2. Evaluación de BER

```bash
python ber_4qam_mimo_2x2_all.py
```

**Parámetros de simulación:**
- Iteraciones: 1,000,000 por punto SNR
- Rango SNR: 0-25 dB (paso 1 dB)
- Tiempo estimado: ~90 minutos (GPU RTX 4090)

**Salidas generadas:**
- `BER_MIMO_2x2_All_Strategies.png` - Curvas BER (alta resolución)
- `BER_results_MIMO_2x2_all_strategies.npy` - Datos en formato NumPy
- `BER_results_MIMO_2x2_all_strategies.txt` - Tabla legible

### 3. Análisis de Resultados

```python
import numpy as np
import matplotlib.pyplot as plt

# Cargar resultados
results = np.load('BER_results_MIMO_2x2_all_strategies.npy', allow_pickle=True).item()

# Extraer datos
SNR_dB = results['SNR_dB']
BER_ML = results['BER_ML']
BER_OneHot = results['BER_OneHot']
BER_LabelEncoder = results['BER_LabelEncoder']

# Visualizar
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_dB, BER_ML, 's-', label='ML (Óptimo)')
plt.semilogy(SNR_dB, BER_OneHot, 'o--', label='One-Hot Encoding')
plt.semilogy(SNR_dB, BER_LabelEncoder, 'x-.', label='Label Encoder')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('BER_comparison.png', dpi=300)
plt.show()
```

---

## 📁 Estructura del Proyecto

```
mimo-dl-detector/
│
├── README.md                                    # Este archivo
├── CHANGELOG.md                                 # Historial de versiones y cambios técnicos
├── RESULTS.md                                   # Resultados experimentales y análisis
│
├── Comparacion_MATLAB_vs_PYTHON.md              # Análisis comparativo ELM vs Deep Learning
├── ELM_vs_DeepLearning_Resultados.md            # Resultados detallados del estudio comparativo
│
├── modelMIMO_2x2_4QAM_OneHot.py                 # Script entrenamiento One-Hot
├── modelMIMO_2x2_4QAM_LabelEncoder.py           # Script entrenamiento Label Encoder (ReLU)
├── modelMIMO_2x2_4QAM_LabelEncoder_Sigmoid.py   # Script entrenamiento Label Encoder (Sigmoid)
├── modelMIMO_2x2_4QAM_DoubleOneHot.py           # Script entrenamiento Per-Antenna (ReLU)
├── modelMIMO_2x2_4QAM_DoubleOneHot_Sigmoid.py   # Script entrenamiento Per-Antenna (Sigmoid)
│
├── ber_4qam_mimo_2x2_all.py                     # Script evaluación BER (optimizado)
│
├── trained_models/                              # Modelos entrenados (.pth)
│   ├── modelMIMO_2x2_4QAM_OneHot.pth
│   ├── modelMIMO_2x2_4QAM_LabelEncoder.pth
│   ├── modelMIMO_2x2_4QAM_LabelEncoder_Sigmoid.pth
│   ├── modelMIMO_2x2_4QAM_DoubleOneHot.pth
│   └── modelMIMO_2x2_4QAM_DoubleOneHot_Sigmoid.pth
│
├── modelMIMO_2x2_4QAM_OneHot.md                 # Documentación técnica One-Hot
├── modelMIMO_2x2_4QAM_LabelEncoder.md           # Documentación técnica Label Encoder
├── modelMIMO_2x2_4QAM_DoubleOneHot.md           # Documentación técnica Per-Antenna
├── BER_4QAM_MIMO_2x2_All.md                     # Documentación evaluación BER
│
└── Matlab/                                      # Código MATLAB de referencia (ELM)
    └── (código original del repositorio base)
```

---

## 📚 Documentación Técnica

### Documentos Principales

1. **[CHANGELOG.md](CHANGELOG.md)**
   - Historial completo de versiones
   - Detalles técnicos de las 8 optimizaciones
   - Correcciones críticas (modelo de ruido)
   - Notas de compatibilidad Windows/Linux

2. **[Comparacion_MATLAB_vs_PYTHON.md](Comparacion_MATLAB_vs_PYTHON.md)**
   - Análisis exhaustivo ELM vs Deep Learning
   - Diferencias arquitectónicas fundamentales
   - Comparación de estrategias de etiquetado
   - Análisis de bugs y hallazgos críticos

3. **[ELM_vs_DeepLearning_Resultados.md](ELM_vs_DeepLearning_Resultados.md)**
   - Resultados experimentales del estudio comparativo
   - Análisis de trade-offs (velocidad vs calidad)
   - Reproducibilidad y accesibilidad
   - Tablas de rendimiento detalladas

4. **[RESULTS.md](RESULTS.md)**
   - Resultados experimentales actualizados
   - Estudios de ablación (ReLU vs Sigmoid)
   - Insights científicos
   - Protocolo experimental para reproducibilidad

### Documentación por Estrategia

- **[modelMIMO_2x2_4QAM_OneHot.md](modelMIMO_2x2_4QAM_OneHot.md)** - Codificación One-Hot
- **[modelMIMO_2x2_4QAM_LabelEncoder.md](modelMIMO_2x2_4QAM_LabelEncoder.md)** - Label Encoder
- **[modelMIMO_2x2_4QAM_DoubleOneHot.md](modelMIMO_2x2_4QAM_DoubleOneHot.md)** - Per-Antenna
- **[BER_4QAM_MIMO_2x2_All.md](BER_4QAM_MIMO_2x2_All.md)** - Evaluación BER

---

## 🤝 Contribuciones y Referencias

### Implementación Deep Learning (Este Trabajo)

**Autor:** Leonel Roberto Perea Trejo
**Email:** iticleonel.leonel@gmail.com
**Fecha:** Enero 2025

**Contribuciones:**
- ✅ Implementación Python/PyTorch con backpropagation completo
- ✅ 8 optimizaciones de rendimiento (17× speedup)
- ✅ Análisis comparativo exhaustivo ELM vs Deep Learning
- ✅ Estudios de ablación (activaciones ReLU vs Sigmoid)
- ✅ Corrección de modelo de ruido (estándar científico)
- ✅ Análisis automático @ BER = 10⁻³
- ✅ Documentación técnica comprensiva
- ✅ Compatibilidad cross-platform

### Trabajo de Referencia (Enfoque ELM)

**Autores:** Roilhi Frajo Ibarra Hernández, Francisco Rubén Castillo-Soria
**Afiliación:** Universidad Autónoma de San Luis Potosí (UASLP)
**Email:** roilhi.ibarra@uaslp.mx

**Papers:**
1. Ibarra-Hernández, R.F. et al. (2024). "Efficient Deep Learning-Based Detection Scheme for MIMO Communication System." *Sensors (MDPI)*.
2. Ibarra-Hernández, R.F. et al. (2025). "Extreme Learning Machine Signal Detection for MIMO Channels." *IEEE LatinCom 2025*.

**Repositorio Original:** [roilhi/mimo-dl-detector](https://github.com/roilhi/mimo-dl-detector)

### Cómo Citar

Si utilizas este código en investigación que resulte en publicaciones, por favor cita:

```bibtex
@article{ibarra2024efficient,
  title={Efficient Deep Learning-Based Detection Scheme for MIMO Communication System},
  author={Ibarra-Hern{\'a}ndez, Roilhi Frajo and Castillo-Soria, Francisco Rub{\'e}n and others},
  journal={Sensors},
  publisher={MDPI},
  year={2024}
}

@inproceedings{ibarra2025elm,
  title={Extreme Learning Machine Signal Detection for MIMO Channels},
  author={Ibarra-Hern{\'a}ndez, Roilhi Frajo and Castillo-Soria, Francisco Rub{\'e}n and others},
  booktitle={IEEE LatinCom},
  year={2025}
}
```

### Referencias Teóricas

1. **Shannon, C.E.** (1948). "A Mathematical Theory of Communication"
2. **Telatar, E.** (1999). "Capacity of Multi-antenna Gaussian Channels"
3. **Tse, D., & Viswanath, P.** (2005). "Fundamentals of Wireless Communication." Cambridge University Press.
4. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). "Deep Learning." MIT Press.
5. **Huang, G.-B., et al.** (2006). "Extreme learning machine: Theory and applications." *Neurocomputing*.

---

## 📄 Licencia

Este proyecto está licenciado bajo **GPLv2 License**.

```
Copyright (C) 2025 Leonel Roberto Perea Trejo

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
```

### Reconocimientos

Este trabajo se basa en la investigación original del equipo de la UASLP y contribuye al avance de esquemas eficientes de detección MIMO mediante técnicas de Deep Learning. Agradecemos especialmente al Prof. Roilhi Ibarra por proporcionar el código de referencia MATLAB/ELM que permitió realizar este estudio comparativo.

---

## 📞 Contacto y Soporte

**Para preguntas sobre la implementación Deep Learning:**
- Leonel Roberto Perea Trejo
- Email: iticleonel.leonel@gmail.com
- GitHub: [Issues en este repositorio]

**Para preguntas sobre el enfoque ELM (referencia):**
- Prof. Roilhi Frajo Ibarra Hernández
- Email: roilhi.ibarra@uaslp.mx
- GitHub: [roilhi/mimo-dl-detector](https://github.com/roilhi/mimo-dl-detector)

---

## 🔮 Trabajo Futuro

### Extensiones Inmediatas
- [ ] Sistemas MIMO más grandes (4×4, 8×8)
- [ ] Modulaciones de orden superior (16-QAM, 64-QAM)
- [ ] Canales Rician y selectivos en frecuencia

### Investigación Avanzada
- [ ] Arquitecturas profundas (3-4 capas, residual connections)
- [ ] Mecanismos de atención para detección MIMO
- [ ] Implementación en hardware (FPGA/ASIC)
- [ ] Pruebas sobre el aire (SDR)

---

**Última Actualización:** Noviembre 2025
**Versión:** 1.0.0
**Estado:** Activo y mantenido

---

<p align="center">
  <i>Desarrollado como parte de investigación en detección MIMO basada en Deep Learning</i><br>
  <i>Contribuyendo al avance de sistemas de comunicación inalámbrica eficientes</i>
</p>
