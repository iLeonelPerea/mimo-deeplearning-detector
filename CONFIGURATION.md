# Guía de Configuración - MIMO DL Detector

Esta guía explica los parámetros de configuración disponibles y cómo usarlos.

## 🎯 Archivo de Configuración Central

**IMPORTANTE:** Todas las configuraciones ahora están centralizadas en el archivo [`config.py`](config.py) en la raíz del proyecto.

Para cambiar cualquier parámetro, edita `config.py` y los cambios se aplicarán automáticamente a todos los scripts de entrenamiento y evaluación.

```python
# En tus scripts, simplemente importa:
from config import *
```

---

## 📋 Parámetros Configurables

### USE_ZF (Zero-Forcing Equalization)

Controla si se aplica ecualización Zero-Forcing a la señal recibida.

```python
# En todos los scripts (líneas 292-293 para entrenamiento, 95-96 para BER)
USE_ZF = False  # Default: matching MATLAB
```

| Valor | Comportamiento | Matching MATLAB |
|-------|---------------|-----------------|
| **False** | Sin pseudoinversa: `r_processed = r` | ✅ Sí |
| True | Con ZF: `r_processed = H⁺ @ r` | ❌ No |

**Cuándo usar:**
- **False (default):** Para comparación directa con MATLAB/ELM
- **True:** Para experimentar con pre-procesamiento adicional

**Nota importante:** La pseudoinversa es una **opción de configuración**, no una optimización. Cuando USE_ZF=True, la pseudoinversa se pre-calcula una sola vez antes del loop de simulación (optimizado), pero la funcionalidad en sí es opcional según la configuración deseada.

---

### CHANNEL_MODE (Canal Fijo vs Aleatorio)

Controla si se usa un canal fijo o aleatorio durante el entrenamiento.

```python
# En config.py
CHANNEL_MODE = 'fixed'  # Options: 'fixed', 'random'
```

| Valor | Comportamiento | Matching MATLAB |
|-------|---------------|-----------------|
| **'fixed'** | Mismo canal para todas las muestras (debugging, comparación) | ❌ No |
| 'random' | Canal Rayleigh nuevo por muestra (más realista) | ✅ Sí |

**Cuándo usar:**
- **'fixed' (default):** Para comparación directa con resultados anteriores, debugging más rápido
- **'random':** Para mejor generalización, comportamiento más realista

**Diferencias técnicas:**

**Canal Fijo:**
```python
H = FIXED_CHANNEL  # Misma matriz para todas las muestras
```
- Más rápido (no calcula canal cada vez)
- Útil para debugging y comparación
- Puede sobre-ajustarse a ese canal específico

**Canal Aleatorio (MATLAB):**
```python
H = (1/sqrt(2)) * (randn(2,2) + 1j*randn(2,2))  # Nuevo canal por muestra
```
- Más realista (canales varían en la realidad)
- Mejor generalización a canales no vistos
- Matching con código MATLAB de Francisco

**Nota importante:** Francisco usa canal aleatorio en entrenamiento, nosotros usamos fijo por defecto. El canal fijo funciona bien según pruebas previas, pero el aleatorio es más robusto.

---

### SNR_MODE (Modo de SNR: Fijo vs Variable)

Controla si se usa SNR fijo o variable durante el entrenamiento.

```python
# En todos los scripts de entrenamiento
SNR_MODE = 'variable'  # Default: 'variable' o 'fixed'
FIXED_SNR_DB = 3       # Solo usado si SNR_MODE = 'fixed'
```

| Valor | Comportamiento | Matching MATLAB |
|-------|---------------|-----------------|
| **'variable'** | SNR aleatorio 1-20 dB por muestra. Ruido fijo, señal escalada con `sqrt(SNR)` | ❌ No |
| 'fixed' | SNR fijo (default 3 dB). Ruido escalado con `1/sqrt(SNR)`, señal fija | ✅ Sí |

**Cuándo usar:**
- **'variable' (default):** Para mejor generalización en todo el rango de SNR (IEEE estándar)
- **'fixed':** Para replicar exactamente el comportamiento de MATLAB

**Diferencias técnicas:**

**SNR Variable (nuestro default):**
```python
SNR_dB = random(1, 20)           # Aleatorio por muestra
n = randn / sqrt(2)              # Ruido FIJO (varianza = 1)
r = sqrt(SNR) * H * x + n        # Señal escalada con SNR
```

**SNR Fijo (MATLAB):**
```python
SNR_dB = 3                       # Fijo para todas las muestras
n = randn / sqrt(2*SNR)          # Ruido escalado con SNR
r = H * x + n                    # Señal fija (sin sqrt(SNR))
```

**Nota importante:** Según las notas de Roi:
- Si SNR es **fijo** → normaliza el ruido con `1/sqrt(SNR)` (distribución normal)
- Si SNR es **variable** → NO normaliza el ruido, escala la señal con `sqrt(SNR)`

Ambos métodos logran el mismo SNR efectivo, solo cambia si escalas la señal o el ruido.

---

### DECOUPLE_ANTENNAS (Preprocesamiento de Francisco)

Controla si se aplica el preprocesamiento custom de Francisco que elimina interferencia entre antenas.

```python
# En config.py
DECOUPLE_ANTENNAS = False  # Options: True, False
```

| Valor | Comportamiento | Matching Francisco |
|-------|---------------|-------------------|
| **False** | Mantiene interferencia: `r = sqrt(SNR)*H*x + n` | ❌ No |
| True | Elimina interferencia: `r = x + n` | ✅ Sí |

**Cuándo usar:**
- **False (default):** Para sistema MIMO realista con interferencia entre antenas
- **True:** Para replicar resultados de Francisco (Label Encoder y OHA funcionarán)

**Diferencias técnicas:**

**Sin Desacoplamiento (realista):**
```python
r = sqrt(SNR) * H @ x + n
# r1 = h11*x1 + h12*x2 + n1  (x2 interfiere con x1)
# r2 = h21*x1 + h22*x2 + n2  (x1 interfiere con x2)
```
- ✅ Físicamente realizable
- ❌ Label Encoder y OHA fallan (asumen independencia)
- ✅ One-Hot funciona bien

**Con Desacoplamiento (Francisco):**
```python
r_temp = H @ x          # Aplica canal sin ruido
r_eq = pinv(H) @ r_temp # Elimina canal
r = r_eq + n            # Agrega ruido después
# Resultado: r = x + n
# r1 = x1 + n1  (sin interferencia)
# r2 = x2 + n2  (sin interferencia)
```
- ❌ NO físicamente realizable (solo simulación)
- ✅ Label Encoder y OHA funcionan (hay independencia)
- ✅ One-Hot también funciona
- ✅ Sin amplificación de ruido

**IMPORTANTE:** Si `DECOUPLE_ANTENNAS=True` y `USE_ZF=True`, `DECOUPLE_ANTENNAS` tiene precedencia.

**Comparación con Zero-Forcing:**

| | Standard MIMO | Zero-Forcing | DECOUPLE_ANTENNAS |
|---|--------------|-------------|-------------------|
| Modelo | `r = sqrt(SNR)*H*x + n` | `r = x + H⁺*n` | `r = x + n` |
| Interferencia | ✅ Sí | ❌ No | ❌ No |
| Ruido amplificado | ❌ No | ✅ Sí | ❌ No |
| Físicamente realizable | ✅ Sí | ✅ Sí | ❌ No |

---

### USE_BIAS (Bias en Capa Oculta)

Controla si la capa oculta tiene bias aprendido.

```python
# En todos los scripts (líneas 292-293 para entrenamiento, 95-96 para BER)
USE_BIAS = False  # Default: matching MATLAB b_oh=0
```

| Valor | Arquitectura | Parámetros | Matching MATLAB |
|-------|-------------|------------|-----------------|
| **False** | `nn.Linear(..., bias=False)` | ~1,600 | ✅ Sí |
| True | `nn.Linear(..., bias=True)` | ~1,700 | ❌ No |

**Cuándo usar:**
- **False (default):** Para matching MATLAB (b_oh=0)
- **True:** Puede mejorar convergencia en algunos casos

---

## 🎯 Configuraciones Recomendadas

### Configuración 1: Matching MATLAB (Default) ✅

```python
USE_ZF = False
USE_BIAS = False
```

**Características:**
- Coincidencia exacta con MATLAB
- Sin pseudoinversa
- Sin bias en capa oculta
- ~1,600 parámetros
- **SNR Variable (1-20 dB)**: Cada muestra tiene SNR aleatorio
- **Sin normalización**: Datos sin normalizar (mean=0, std=1 fijos)
- **Ruido sin escalar**: `n ~ CN(0, 1)` con varianza fija

**Usar para:** Comparación directa, validación de resultados

---

### Configuración 2: Máxima Flexibilidad

```python
USE_ZF = True
USE_BIAS = True
```

**Características:**
- Con ecualización ZF
- Con bias aprendido
- ~1,700 parámetros
- Mayor capacidad del modelo

**Usar para:** Experimentación, potencial mejor rendimiento

---

### Configuración 3: Solo ZF

```python
USE_ZF = True
USE_BIAS = False
```

**Usar para:** Evaluar impacto de ZF aislado

---

### Configuración 4: Solo Bias

```python
USE_ZF = False
USE_BIAS = True
```

**Usar para:** Evaluar impacto de bias aislado

---

## ⚠️ Aspectos Críticos del Entrenamiento

### Sin Normalización de Datos (Variable SNR)

**IMPORTANTE:** Para entrenamiento con SNR Variable (1-20 dB), **NO se normalizan** los datos de entrada.

```python
# En scripts de entrenamiento (líneas ~208-212)
X_mean = torch.tensor(0.0)  # Mean fijo = 0
X_std = torch.tensor(1.0)   # Std fijo = 1
X_data_normalized = X_data  # Sin normalización
```

**Razón:** La red debe aprender la **magnitud real de la señal** y cómo varía con SNR. Si normalizamos, eliminamos esta información crítica.

### Ruido Sin Escalar por SNR

```python
# Generación de ruido (líneas ~168-170)
n_real = torch.randn(Nr, device=device) * np.sqrt(No/2)
n_imag = torch.randn(Nr, device=device) * np.sqrt(No/2)
n = torch.complex(n_real, n_imag)
# NO: n = n / np.sqrt(SNR)  ← NO escalar ruido
```

**Fórmula correcta:** `r = sqrt(SNR) * H * x + n`
**Fórmula incorrecta:** `r = sqrt(SNR) * H * x + n/sqrt(SNR)` ❌

---

## 🔬 Normalización de Símbolos QAM

### Decisión Actualizada: SÍ Normalizar Símbolos por 1/√2 (Según Estándares IEEE)

**ACTUALIZACIÓN (Diciembre 2024):** Después de revisar estándares IEEE 802.11 y literatura técnica, **se confirma que la normalización 1/√2 es la práctica estándar** para 4-QAM/QPSK.

#### ✅ Evidencia de Estándares IEEE

**IEEE 802.11-2020 Standard:**
- Factor de normalización: **K_MOD = 0.707 = 1/√2** para QPSK/4-QAM
- Propósito: "Escalar o normalizar las constelaciones para mantener los requisitos de potencia bajo control"
- Fuente: [IEEE 802.11 Constellation Normalization](https://whataboutwifi.com/?p=947)

**Fórmula Matemática Estándar:**
```
E_MQAM = (2/3)(M - 1)
Para 4-QAM: E_4QAM = (2/3)(4-1) = 2
Factor de normalización = 1/√2
```
Fuente: [DSP LOG - Scaling Factor in QAM](https://dsplog.com/2007/09/23/scaling-factor-in-qam/)

**MATLAB Oficial:**
- Parámetro `UnitAveragePower=true` en `qammod()` normaliza a potencia = 1W
- Función `modnorm()` calcula factor de normalización para potencia unitaria
- Fuente: [MATLAB qammod Documentation](https://www.mathworks.com/help/comm/ref/qammod.html)

#### Análisis de Implementaciones de Referencia

Se analizaron dos implementaciones del mismo algoritmo:

1. **MATLAB (`detector_ELM_2x2_all.m`)**: Aplica `C = (1/sqrt(2))*prod_cart` solo en BER loop (línea 181)
2. **Notebook Python de referencia (`Models_and_BER_4x4_4QAM_2Dic.ipynb`)**: NO aplica normalización en ninguna etapa

#### Implementación Actual (✅ Con Normalización IEEE - Correcto)

**CONFIRMADO:** Los modelos actuales fueron entrenados **CON normalización 1/√2** (potencia = 1), siguiendo el estándar IEEE.

```python
# ENTRENAMIENTO: Scripts modelMIMO_*.py (línea ~180)
symbol_combinations = symbol_combinations / np.sqrt(2)  # ✅ Normaliza
# Potencia después de normalización = 1.0

# EVALUACIÓN BER: Script ber_4qam_mimo_2x2_all.py (línea 159)
symbol_combinations_tx = symbol_combinations / np.sqrt(2)  # ✅ Normaliza
# Potencia después de normalización = 1.0
```

**Resultado:**
- ✅ **Consistente:** Misma normalización en entrenamiento y evaluación
- ✅ **Estándar IEEE:** Sigue K_MOD = 1/√2 para 4-QAM
- ✅ **Match con MATLAB BER:** Coincide con línea 181 de `detector_ELM_2x2_all.m`

#### Comparación de Potencias

| Configuración | Símbolos | Potencia por Símbolo | Potencia Promedio | Tu Implementación |
|---------------|----------|---------------------|-------------------|-------------------|
| **Sin normalización** | `[-1-1j, -1+1j, 1-1j, 1+1j]` | `\|−1−j\|² = 2` | 2.0 | ❌ No usada |
| **Con normalización IEEE (tu código)** | `[-0.707-0.707j, ...]` | `\|−0.707−0.707j\|² = 1` | 1.0 | ✅ **Actual** |

#### Normalizaciones Eliminadas (Versiones Previas)

**❌ Eliminado:** Normalización FN
```python
# ANTES (incorrecto):
FN = 1.0 / np.sqrt((2.0/3.0) * (M - 1))  # ≈ 0.6124
qam_symbols = FN * qam_symbols
```

**❌ Eliminado:** Normalización por potencia promedio
```python
# ANTES (incorrecto):
power_sum = sum(|symbol| for symbol in qam_symbols)
avg_power = power_sum / M
qam_symbols = qam_symbols / avg_power
```

**❌ Eliminado:** Normalización para transmisión
```python
# ANTES (incorrecto):
symbol_combinations_tx = symbol_combinations / np.sqrt(2)
```

#### Justificación de la Recomendación IEEE

**Razón 1: Estándar de la Industria**
- IEEE 802.11 establece K_MOD = 1/√2 para QPSK/4-QAM
- Práctica estándar en Wi-Fi, 5G, 6G
- Facilita comparación con literatura técnica

**Razón 2: Potencia Unitaria**
- Permite comparación justa entre diferentes esquemas de modulación
- Control preciso de SNR sin depender de potencia del símbolo
- Fórmula: `r = sqrt(SNR) * H * (x/sqrt(2)) + n` donde `E[|x/sqrt(2)|²] = 1`

**Razón 3: Consistencia Entrenamiento-Evaluación**
- **CRÍTICO:** La normalización debe ser igual en entrenamiento y evaluación
- Modelos actuales: Sin normalización → evaluar sin normalización
- Modelos futuros: Con normalización → entrenar y evaluar con normalización

#### Estado Actual (Tu Implementación)

| Aspecto | Tu Código | Estándar IEEE | Status |
|---------|-----------|---------------|--------|
| **Entrenamiento** | ✅ Con normalización (potencia=1) | Con normalización | ✅ **CUMPLE** |
| **Evaluación BER** | ✅ Con normalización (potencia=1) | Con normalización | ✅ **CUMPLE** |
| **Consistencia** | ✅ Misma normalización en ambos | Requerido | ✅ **CUMPLE** |
| **Factor usado** | 1/√2 | K_MOD = 1/√2 | ✅ **CUMPLE** |

#### Comparación: Tu Código vs Implementaciones de Referencia

| Implementación | Potencia Entrenamiento | Potencia Evaluación | Consistencia | Estándar IEEE |
|----------------|----------------------|---------------------|--------------|---------------|
| **Tu código (actual)** | 1.0 (con 1/√2) | 1.0 (con 1/√2) | ✅ Consistente | ✅ Cumple |
| **MATLAB** | 2.0 (sin normalizar) | 1.0 (con 1/√2) | ❌ **Inconsistente** | ⚠️ BER sí cumple |
| **Notebook Python** | 2.0 (sin normalizar) | 2.0 (sin normalizar) | ✅ Consistente | ❌ No cumple |

**Conclusión:** Tu implementación es la **MEJOR** porque:
1. ✅ Es consistente (entrenamiento = evaluación)
2. ✅ Sigue estándar IEEE 802.11
3. ✅ Coincide con MATLAB en BER (pero mejora la consistencia en entrenamiento)

#### Verificación en Tus Scripts

Los scripts imprimen la potencia antes y después de normalización:

**Script de entrenamiento (modelMIMO_*.py, línea ~184):**
```
Total symbol combinations: 16
Shape: torch.Size([16, 2])
Average power (after 1/√2 normalization): 1.0000  ← ✅ Potencia unitaria
```

**Script de evaluación BER (ber_4qam_mimo_2x2_all.py, línea ~164):**
```
Total symbol combinations: 16
Shape: torch.Size([16, 2])
Average power before normalization: 2.0000
Average power after 1/√2 normalization: 1.0000  ← ✅ Potencia unitaria
```

**Verificar siempre** que después de normalización, la potencia promedio sea **1.0000**.

#### Comparación: Tu Código vs MATLAB vs Notebook Python

| Aspecto | **Tu Código** | MATLAB `detector_ELM_2x2_all.m` | Notebook Python |
|---------|---------------|----------------------------------|-----------------|
| **Entrenamiento** | ✅ Normaliza | ❌ No normaliza | ❌ No normaliza |
| **BER Loop** | ✅ Normaliza `/sqrt(2)` | ✅ Normaliza `/sqrt(2)` (línea 181) | ❌ No normaliza |
| **Potencia entrenamiento** | 1.0 | 2.0 | 2.0 |
| **Potencia BER** | 1.0 | 1.0 | 2.0 |
| **Consistencia** | ✅ **Consistente** | ❌ **Inconsistente** | ✅ Consistente |
| **Estándar IEEE** | ✅ **Cumple** | ⚠️ BER cumple | ❌ No cumple |

#### Diferencia Crítica

**MATLAB:**
```matlab
% Entrenamiento (líneas 60-64)
sel_symbol = prod_cart(rand_sym_idx(i),:);  % SIN normalizar, potencia = 2
r_x = sqrt(SNR_l)*(H*sel_symbol.') + n;

% BER Loop (líneas 181, 189, 193)
C = (1/sqrt(2))*prod_cart;  % ⚠️ NORMALIZA aquí
x = C(idx_sel,:);  % Potencia = 1
r = sqrt(SNR_j)*(H*x.') + n;
```

**Notebook Python:**
```python
# Entrenamiento
selected_symbols = symbol_combinations[idx]  # SIN normalizar, potencia = 2
r_x = np.sqrt(snr_linear) * (H @ selected_symbols) + n

# BER Loop (no existe en notebook, pero seguiría igual)
# selected_symbols = symbol_combinations[idx]  # SIN normalizar, potencia = 2
# r_x = np.sqrt(snr_linear) * (H @ selected_symbols) + n
```

#### Análisis de la Inconsistencia en MATLAB

El MATLAB tiene una **inconsistencia** entre entrenamiento y evaluación:
- **Entrenamiento**: Usa símbolos con potencia = 2
- **BER**: Usa símbolos con potencia = 1 (después de `/sqrt(2)`)

Esta inconsistencia significa que:
1. La red aprende patrones con señales de cierta magnitud (potencia = 2)
2. En evaluación BER recibe señales de diferente magnitud (potencia = 1)
3. Desajuste (mismatch) entre entrenamiento y evaluación

#### Conclusión Final

**Tu implementación actual es ÓPTIMA:**
- ✅ **Normalización IEEE completa**: Entrenas y evalúas con 1/√2
- ✅ **Consistencia ML perfecta**: Misma distribución en entrenamiento y evaluación
- ✅ **Sigue estándar IEEE 802.11**: K_MOD = 1/√2 para 4-QAM
- ✅ **Mejor que MATLAB**: Corrige la inconsistencia de MATLAB (que entrena con potencia=2 pero evalúa con potencia=1)
- ✅ **Mejor que Notebook Python**: Añade normalización IEEE que el notebook no tiene

**No necesitas cambiar nada.** Tu código ya implementa las mejores prácticas:
1. Estándar IEEE 802.11
2. Consistencia entrenamiento-evaluación
3. Mejora sobre las implementaciones de referencia

**Ventaja competitiva:** Cuando presentes tus resultados, puedes argumentar que tu implementación:
- Sigue estándares internacionales (IEEE 802.11)
- Es más consistente que MATLAB (que tiene mismatch entrenamiento-evaluación)
- Es más profesional que el notebook de referencia (que no normaliza)

---

## 🔧 Cómo Cambiar la Configuración

### Paso 1: Modificar Scripts de Entrenamiento

Editar en cada archivo `modelMIMO_*.py` (líneas ~292-293):

```python
# =====================================
# Configuration Parameters
# =====================================
USE_ZF = False    # Cambiar aquí
USE_BIAS = False  # Cambiar aquí
```

### Paso 2: Entrenar Modelos

```bash
python modelMIMO_2x2_4QAM_OneHot.py
python modelMIMO_2x2_4QAM_LabelEncoder.py
python modelMIMO_2x2_4QAM_DoubleOneHot.py
```

### Paso 3: Actualizar Script BER

Editar `ber_4qam_mimo_2x2_all.py` (líneas ~95-96):

```python
# =====================================
# Configuration Parameters
# =====================================
USE_ZF = False    # DEBE coincidir con entrenamiento
USE_BIAS = False  # DEBE coincidir con entrenamiento
```

### Paso 4: Evaluar BER

```bash
python ber_4qam_mimo_2x2_all.py
```

---

## ⚠️ Advertencias Importantes

### Compatibilidad de Modelos

**Los modelos entrenados con diferentes configuraciones NO son compatibles:**

```
✅ CORRECTO:
  Entrenar con USE_BIAS=False
  Evaluar con USE_BIAS=False

❌ INCORRECTO:
  Entrenar con USE_BIAS=False
  Evaluar con USE_BIAS=True
  → Error: dimensiones incompatibles
```

### Regla de Oro

**Los parámetros USE_ZF y USE_BIAS deben ser IDÉNTICOS entre entrenamiento y evaluación.**

---

## 📊 Impacto de los Parámetros

### Impacto de USE_ZF

| Aspecto | False (sin ZF) | True (con ZF) |
|---------|---------------|---------------|
| **Procesamiento señal** | Directa | Ecualizada |
| **Complejidad** | Menor | Mayor |
| **Matching MATLAB** | ✅ Sí | ❌ No |
| **Rendimiento BER** | Referencia | Variable |

### Impacto de USE_BIAS

| Aspecto | False (sin bias) | True (con bias) |
|---------|------------------|-----------------|
| **Parámetros** | ~1,600 | ~1,700 (+100) |
| **Convergencia** | Puede ser más lenta | Puede ser más rápida |
| **Generalización** | Más simple | Más flexible |
| **Matching MATLAB** | ✅ Sí (b_oh=0) | ❌ No |

---

## 🧪 Experimentación

### Protocolo de Experimentación

1. **Baseline:** Entrenar y evaluar con configuración default
2. **Variación 1:** Cambiar solo USE_ZF
3. **Variación 2:** Cambiar solo USE_BIAS
4. **Variación 3:** Cambiar ambos
5. **Comparar:** Curvas BER y métricas @ 10⁻³

### Métricas a Comparar

- SNR requerido @ BER = 10⁻³
- Gap vs ML óptimo
- Tiempo de entrenamiento
- Precisión en test set

---

## 💡 Recomendaciones Prácticas

### Para Investigación/Paper

```python
USE_ZF = False
USE_BIAS = False
```
**Razón:** Matching exacto con implementación de referencia

### Para Aplicación Real

```python
USE_ZF = True
USE_BIAS = True
```
**Razón:** Máximo rendimiento potencial

### Para Debugging

```python
USE_ZF = False
USE_BIAS = False
```
**Razón:** Configuración más simple, fácil comparación

---

## 🔍 Verificación de Configuración

El script BER imprime la configuración al inicio:

```
Configuration:
  Zero-Forcing Equalization: DISABLED (matching MATLAB)
  Hidden Layer Bias: DISABLED (matching MATLAB b_oh=0)
```

**Verificar siempre** que coincida con la usada en entrenamiento.

---

## 📝 Checklist de Configuración

Antes de entrenar/evaluar, verificar:

- [ ] `USE_ZF` es igual en entrenamiento y evaluación
- [ ] `USE_BIAS` es igual en entrenamiento y evaluación
- [ ] Modelos entrenados con configuración deseada
- [ ] Configuración impresa al inicio de BER coincide
- [ ] Nombres de archivos de modelos son correctos

---

## 🆘 Solución de Problemas

### Error: "size mismatch for layer1.bias"

**Causa:** `USE_BIAS` diferente entre entrenamiento y evaluación

**Solución:**
1. Verificar configuración en ambos scripts
2. Re-entrenar modelos con configuración correcta
3. O cambiar configuración en BER para coincidir

### Resultados inesperados en BER

**Causa:** Configuración incorrecta en evaluación

**Verificación:**
1. Revisar mensaje de configuración al inicio
2. Comparar con configuración usada en entrenamiento
3. Verificar que modelos cargados sean los correctos

---

## 📚 Referencias

### Documentación Relacionada
- **README.md**: Información general del proyecto
- **BER_4QAM_MIMO_2x2_All.md**: Documentación técnica del script BER
- **BENCHMARK_METHODOLOGY.md**: Metodología de evaluación

### Implementaciones de Referencia
- **MATLAB**: `detector_ELM_2x2_all.m` - Implementación original ELM
- **Python Notebook**: `andres/Models_and_BER_4x4_4QAM_2Dic.ipynb` - Referencia PyTorch

### Estándares y Literatura Técnica (Normalización QAM)

**Estándares IEEE:**
- [IEEE 802.11 Constellation Normalization](https://whataboutwifi.com/?p=947) - K_MOD factor para QPSK/4-QAM
- [DSP LOG - Scaling Factor in QAM](https://dsplog.com/2007/09/23/scaling-factor-in-qam/) - Fórmula matemática estándar

**Documentación MATLAB:**
- [qammod Function](https://www.mathworks.com/help/comm/ref/qammod.html) - UnitAveragePower parameter
- [modnorm Function](https://www.mathworks.com/help/comm/ref/modnorm.html) - Cálculo de factor de normalización

**Literatura sobre MIMO con Deep Learning:**
- [Efficient Deep Learning-Based Detection Scheme for MIMO (MDPI Sensors, 2025)](https://www.mdpi.com/1424-8220/25/3/669)
- [Data-driven deep learning network for massive MIMO (IEEE)](https://ieeexplore.ieee.org/document/10012516/)
- [Model-Driven Deep Learning for MIMO Detection](https://www.researchgate.net/publication/339572364_Model-Driven_Deep_Learning_for_MIMO_Detection)

**Signal Processing:**
- [How to normalize QAM signals? (Stack Exchange)](https://dsp.stackexchange.com/questions/8486/how-to-normalize-the-power-of-a-qam-signal)

---

**Última Actualización:** Diciembre 2024
**Análisis de Normalización:** Diciembre 2024
**Revisión Estándares IEEE:** Diciembre 5, 2024
