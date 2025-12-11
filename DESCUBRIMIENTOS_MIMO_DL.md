# Descubrimientos Clave - MIMO Deep Learning Detection

Este documento recopila los hallazgos importantes durante el desarrollo y análisis del detector MIMO basado en Deep Learning.

---

## 1. SNR Fijo vs Variable (Descubrimiento de Roi)

### Regla Fundamental

**Según las notas de Roi:**

> **Si SNR es FIJO** → normalizar ruido para distribución normal (`n/sqrt(SNR)`)
>
> **Si SNR es VARIABLE** → NO normalizar ruido, escalar señal (`sqrt(SNR)*H*x`)

### ¿Por qué `sqrt(SNR)` y no `SNR`?

Porque SNR se define en términos de **POTENCIA**, no amplitud:

```
SNR = Potencia_señal / Potencia_ruido
```

Y potencia es el cuadrado de la amplitud:

```
Potencia = |amplitud|²
```

Por lo tanto:
- Si multiplicas amplitud por `sqrt(SNR)` → potencia aumenta por `SNR`
- `|sqrt(SNR) * x|² = SNR * |x|²` ✓

**Ejemplo:** Si SNR = 100 (lineal)
- Multiplicas señal por `sqrt(100) = 10`
- Potencia de señal aumenta por `10² = 100` ✓

### ¿Por qué esta diferencia?

El SNR (Signal-to-Noise Ratio) se define como:

```
SNR = Potencia_señal / Potencia_ruido
```

Puedes lograr el mismo SNR de **dos formas equivalentes**:

---

### Forma 1: SNR Fijo - Normalizar Ruido (MATLAB)

**Cuándo usar:** Cuando entrenas con un SNR constante para todas las muestras.

**Implementación:**
```matlab
SNR_dB = 3;  % SNR fijo para todas las muestras
SNR = 10^(SNR_dB/10);

% Genera ruido estándar
n = (randn(Nr,1) + 1i*randn(Nr,1)) / sqrt(2);

% NORMALIZA el ruido con SNR
n = n / sqrt(SNR);  % ← Clave: divide por sqrt(SNR)

% Señal NO se escala con SNR
r = H*x + n;
```

**Resultado:**
- Señal: potencia fija
- Ruido: potencia = `1/(2*SNR)` (depende de SNR)
- SNR efectivo = `Señal / Ruido` ✓

**Distribución del ruido:**
- Varianza: `σ²_n = 1/(2*SNR)`
- Es una distribución normal con varianza que depende del SNR
- Por eso se dice que "se normaliza para distribución normal"

---

### Forma 2: SNR Variable - NO Normalizar Ruido (IEEE)

**Cuándo usar:** Cuando entrenas con SNR aleatorio por muestra (1-20 dB).

**Implementación:**
```python
SNR_dB = random.randint(1, 21)  # SNR aleatorio por muestra
SNR = 10**(SNR_dB/10)

# Genera ruido estándar
n = (randn(Nr) + 1j*randn(Nr)) / sqrt(2)

# NO normalices el ruido con SNR
# n = n / sqrt(SNR)  ← NO HAGAS ESTO

# Señal SÍ se escala con SNR
r = sqrt(SNR) * H*x + n  # ← Clave: multiplica señal por sqrt(SNR)
```

**Resultado:**
- Señal: potencia = `SNR * |H*x|²` (depende de SNR)
- Ruido: potencia fija = `1`
- SNR efectivo = `Señal / Ruido` ✓

**Distribución del ruido:**
- Varianza: `σ²_n = 1` (constante)
- NO se normaliza con SNR
- La variabilidad del SNR se controla escalando la SEÑAL, no el ruido

---

### Comparación Visual

```
┌─────────────────────────────────────────────────────────┐
│  SNR FIJO (MATLAB)                                      │
├─────────────────────────────────────────────────────────┤
│  Señal:  ████████ (constante)                          │
│  Ruido:  █ (pequeño con SNR alto)                      │
│                                                          │
│  Control de SNR: Ajustando el RUIDO                    │
│  Ruido normalizado: σ² = 1/(2*SNR)                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  SNR VARIABLE (IEEE)                                    │
├─────────────────────────────────────────────────────────┤
│  Señal:  ████████████ (grande con SNR alto)            │
│  Ruido:  ████ (constante)                              │
│                                                          │
│  Control de SNR: Ajustando la SEÑAL                    │
│  Ruido SIN normalizar: σ² = 1                          │
└─────────────────────────────────────────────────────────┘
```

---

### Analogía con Volumen de Audio 🔊

Imagina que quieres que la música suene 10× más fuerte que el ruido de fondo:

**SNR Fijo (bajar ruido):**
- Música: volumen 10 (constante)
- Ruido: volumen 1 (reduces el ruido)
- Relación: 10/1 = 10 ✓

**SNR Variable (subir música):**
- Música: volumen 10, 20, 30... (ajustas según desees)
- Ruido: volumen 1 (constante, no tocas)
- Relación: Música/1 = variable ✓

---

### ¿Cuál es mejor?

| Aspecto | SNR Fijo | SNR Variable |
|---------|----------|--------------|
| **Generalización** | Se especializa en un SNR | Generaliza en todo el rango |
| **Uso típico** | Académico, papers clásicos | Industria, IEEE estándar |
| **Complejidad** | Más simple | Requiere gestión de SNR variable |
| **Robustez** | Funciona bien en ese SNR | Funciona bien en cualquier SNR |

**Recomendación:**
- **SNR Variable** para aplicaciones reales (mejor generalización)
- **SNR Fijo** para replicar papers académicos como MATLAB de Francisco

---

### Implementación en Nuestro Código

Hemos implementado ambos modos con la variable `SNR_MODE`:

```python
# CONFIGURACIÓN
SNR_MODE = 'variable'  # O 'fixed'
FIXED_SNR_DB = 3       # Solo usado si SNR_MODE = 'fixed'

# El código maneja automáticamente ambos casos
if snr_mode == 'fixed':
    # SNR fijo → normaliza ruido
    SNR = 10**(FIXED_SNR_DB/10)
    n = randn / sqrt(2)
    n = n / sqrt(SNR)  # Normaliza ruido
    r = H @ x + n       # Señal sin escalar
else:
    # SNR variable → NO normaliza ruido
    SNR_dB = random(1, 21)
    SNR = 10**(SNR_dB/10)
    n = randn / sqrt(2)  # Ruido fijo
    r = sqrt(SNR) * H @ x + n  # Escala señal
```

---

## 2. Preprocesamiento de Francisco: Desacoplamiento de Antenas

### Descubrimiento

Francisco elimina la interferencia entre antenas ANTES de agregar ruido.

**Código MATLAB (líneas 67-69):**
```matlab
r_x = H*sel_symbol.';  % Aplica canal
H_inv = pinv(H);       % Calcula pseudoinversa
r_x = H_inv*r_x+n;     % Elimina canal, LUEGO agrega ruido
```

**Resultado matemático:**
```
r_x = H⁺ * (H*x) + n
    = I * x + n
    = x + n
```

### ¿Por qué importa?

**Sin preprocesamiento (MIMO estándar):**
```matlab
r1 = h11*x1 + h12*x2 + n1  % x2 interfiere con x1
r2 = h21*x1 + h22*x2 + n2  % x1 interfiere con x2
```
Las antenas NO son independientes.

**Con preprocesamiento de Francisco:**
```matlab
r1 = x1 + n1  % Sin interferencia de x2
r2 = x2 + n2  % Sin interferencia de x1
```
Las antenas SON independientes.

### Impacto

| Método | Con Preprocesamiento | Sin Preprocesamiento |
|--------|---------------------|----------------------|
| **One-Hot (16 salidas)** | ✅ Funciona | ✅ Funciona |
| **Label Encoder (4 salidas)** | ✅ Gap ~2 dB | ❌ Gap ~6 dB |
| **OHA (8 salidas)** | ✅ Gap ~0.5 dB | ❌ Satura en 1.5×10⁻² |

**Razón:** Label Encoder y OHA asumen independencia entre antenas. Sin preprocesamiento, esa independencia no existe.

---

## 3. Zero-Forcing vs Preprocesamiento de Francisco

### Zero-Forcing Estándar (IEEE)

```matlab
r = H*x + n           % Recibe señal con ruido
H_inv = pinv(H)
r_eq = H_inv * r      % Aplica ZF DESPUÉS del ruido
```

**Resultado:**
```
r_eq = H⁺*(H*x + n)
     = x + H⁺*n      ← Ruido AMPLIFICADO
```

### Preprocesamiento de Francisco

```matlab
r_temp = H*x          % Aplica canal (sin ruido)
H_inv = pinv(H)
r_eq = H_inv * r_temp % Elimina canal
r = r_eq + n          % Agrega ruido DESPUÉS
```

**Resultado:**
```
r = H⁺*(H*x) + n
  = x + n            ← Ruido NO amplificado
```

### Diferencia Clave

| | Zero-Forcing | Francisco |
|---|-------------|-----------|
| **Orden** | Canal → Ruido → Ecualización | Canal → Ecualización → Ruido |
| **Ruido final** | `H⁺*n` (amplificado) | `n` (original) |
| **Físicamente realizable** | ✅ Sí | ❌ No |
| **Interferencia** | Eliminada | Eliminada |

**Conclusión:** Ambos eliminan interferencia, pero Francisco evita amplificación de ruido a costa de no ser físicamente realizable.

---

## 4. Arquitecturas que Requieren Independencia entre Antenas

### One-Hot (16 salidas) - Robusto ✅

```
Input(4) → Hidden(100) → Output(16)
```

**Por qué funciona con interferencia:**
- 16 salidas = 4² combinaciones (todas las combinaciones posibles de 2 símbolos 4-QAM)
- Puede aprender la función compleja `f(r1, r2) → (x1, x2)` incluso con interferencia
- Tiene suficiente capacidad para capturar dependencias entre antenas

---

### Label Encoder (4 salidas) - Sensible ⚠️

```
Input(4) → Hidden(100) → Output(4)
```

**Por qué falla con interferencia:**
- 4 salidas codifican 4 signos: [sign(Re{x1}), sign(Im{x1}), sign(Re{x2}), sign(Im{x2}})]
- Asume que puede decodificar `x1` y `x2` independientemente
- Pero con `r1 = h11*x1 + h12*x2 + n1`, NO puede separar `x1` de `x2`
- Sin suficientes salidas para aprender la dependencia completa

**Con preprocesamiento (`r = x + n`):**
- Puede decodificar `sign(Re{x1})` directamente de `Re{r1}`
- Funciona porque las antenas son independientes

---

### One-Hot per Antenna - OHA (8 salidas) - Muy Sensible ❌

```
Input(4) → Hidden(100) → Output(8)
                          ├─ 4 para antena 1
                          └─ 4 para antena 2
```

**Por qué falla con interferencia:**
- Arquitectura split: cabezas separadas para cada antena
- **Asume estructuralmente** que `r1` solo contiene info de `x1` y `r2` solo de `x2`
- Pero con interferencia: `r1` contiene AMBOS `x1` y `x2`
- La arquitectura no puede aprender dependencias cruzadas entre cabezas

**Con preprocesamiento (`r = x + n`):**
- `r1 = x1 + n1` → cabeza 1 solo necesita decodificar `x1`
- `r2 = x2 + n2` → cabeza 2 solo necesita decodificar `x2`
- Perfectamente alineado con la arquitectura split

---

## 5. Canal Fijo vs Aleatorio

### MATLAB (Francisco)

```matlab
% Cada muestra tiene un canal ALEATORIO
H = (1/sqrt(2))*(randn(Nr,Nt) + 1i*randn(Nr,Nt));
```

- Entrena con múltiples realizaciones del canal
- Más realista (canales varían en la realidad)
- Mejor generalización a canales no vistos

### Nuestro Código

```python
# Todas las muestras usan el MISMO canal
H = torch.tensor([[-0.90064 + 1j*0.43457, ...]], ...)
```

- Entrena con una sola realización del canal
- Puede sobre-ajustarse a ese canal específico
- Más simple, útil para debugging y comparación

**Impacto:** Posible sobreajuste al canal específico, menor generalización.

---

## 6. Normalización de Símbolos (SIEMPRE Requerida)

### Descubrimiento del Profesor

**Tu profesor confirmó: Los símbolos SIEMPRE deben normalizarse.**

### ¿Por qué normalizar?

La normalización de símbolos es **obligatoria** por las siguientes razones:

1. **Control de Potencia Transmitida**
   - En sistemas reales, hay límites de potencia de transmisión
   - La normalización asegura potencia unitaria: `E[|x|²] = 1`
   - Esto garantiza uso eficiente de la potencia disponible

2. **Definición Correcta del SNR**
   - SNR se define como: `SNR = Potencia_señal / Potencia_ruido`
   - Si los símbolos no están normalizados, el SNR efectivo cambia
   - Con normalización: SNR se controla solo con el término `sqrt(SNR)`

3. **Estándar IEEE (802.11, LTE, 5G)**
   - Todos los estándares de comunicación usan símbolos normalizados
   - 4-QAM normalizado: `{±1/√2 ± 1j/√2}` → `E[|x|²] = 1`
   - 16-QAM, 64-QAM, etc. también se normalizan

4. **Comparación Justa entre Modulaciones**
   - Con normalización, diferentes modulaciones (4-QAM, 16-QAM, 64-QAM) usan la misma potencia
   - Puedes comparar BER de forma justa a un SNR dado

### Ejemplo Numérico

**Sin normalización (INCORRECTO):**
```python
symbols = {-1-1j, -1+1j, 1-1j, 1+1j}
E[|x|²] = (1² + 1²) = 2  # Potencia = 2

# Si quieres SNR = 10 dB:
r = sqrt(10) * H*x + n
SNR_efectivo = 10 * 2 / 1 = 20 (13 dB) ← INCORRECTO!
```

**Con normalización (CORRECTO):**
```python
symbols = {-1-1j, -1+1j, 1-1j, 1+1j} / sqrt(2)
E[|x|²] = (1/2 + 1/2) = 1  # Potencia = 1

# Si quieres SNR = 10 dB:
r = sqrt(10) * H*x + n
SNR_efectivo = 10 * 1 / 1 = 10 (10 dB) ← CORRECTO!
```

### ¿Qué pasa con el código de MATLAB de Francisco?

**MATLAB (Francisco) - SIN normalización:**
```matlab
Xx = [-1 1];
Yy = [-1 1];
```
Símbolos: `{±1 ± 1j}`, Potencia: `E[|x|²] = 2`

**¿Es incorrecto?** No necesariamente, PERO:
- Francisco usa un modelo de potencia custom
- Compensa la falta de normalización en otras partes del código
- Su SNR efectivo es `4*SNR` en lugar de `SNR`
- Esto hace que sus resultados no sean directamente comparables con estándares IEEE

**Nuestro Código (IEEE estándar):**
```python
symbols = {±1 ± 1j}
symbols = symbols / sqrt(2)  # SIEMPRE normalizar
```
Símbolos: `{±1/√2 ± 1j/√2}`, Potencia: `E[|x|²] = 1`

### Regla de Oro

**🔥 SIEMPRE normaliza los símbolos para que `E[|x|²] = 1`**

Esto asegura:
- ✅ Potencia unitaria transmitida
- ✅ Control correcto del SNR
- ✅ Comparabilidad con literatura científica
- ✅ Cumplimiento con estándares IEEE
- ✅ Resultados reproducibles

**Conclusión:** Nuestro código está correcto al normalizar. El código de MATLAB de Francisco usa una convención no estándar que requiere ajustes en el modelo de potencia.

---

## 7. Modelo de Potencia Completo

### MATLAB

```
H:      E[|h|²] = 1/2           (normalización 1/sqrt(2))
x:      E[|x|²] = 2             (sin normalizar)
n:      E[|n|²] = 1/(2*SNR)     (normalizado con SNR)

r = x + n  (después de eliminar H)

SNR_efectivo = 2 / (1/(2*SNR)) = 4*SNR
```

Si configuras SNR=3dB, el SNR efectivo es ~6dB.

### Nuestro Código

```
H:      E[|H|²] = ?             (normalización element-wise)
x:      E[|x|²] = 1             (IEEE estándar)
n:      E[|n|²] = 1             (fijo)

r = sqrt(SNR) * H*x + n

SNR_efectivo = SNR * E[|H|²]
```

Con normalización apropiada de H, SNR efectivo ≈ SNR.

**Conclusión:** Modelos de potencia diferentes pero ambos válidos. Importante ser consistente.

---

## 8. Resumen de Descubrimientos

1. **✅ SNR fijo → normaliza ruido, SNR variable → NO normaliza ruido** (Roi)
2. **✅ Francisco elimina interferencia antes de agregar ruido** (no físicamente realizable)
3. **✅ Label Encoder y OHA requieren independencia entre antenas**
4. **✅ One-Hot es robusto, funciona con o sin interferencia**
5. **✅ Canal aleatorio vs fijo afecta generalización**
6. **✅ Normalización de símbolos: convención, no corrección**
7. **✅ Zero-Forcing amplifica ruido, preprocesamiento de Francisco no**
8. **✅ Modelo de potencia: diferentes convenciones, mismo resultado**

---

## Referencias

- Paper: Ibarra-Hernández et al., "Efficient Deep Learning-Based Detection Scheme for MIMO Communication Systems", Sensors 2025
- Código MATLAB: `roilhi/Matlab/training_2x2_detector_*.m`
- Notas de Roi sobre SNR fijo vs variable
- IEEE 802.11 standard para normalización de símbolos
- Nuestro análisis: `CODE_COMPARISON_MATLAB_VS_PYTHON.md`, `FRANCISCO_PREPROCESSING_ANALYSIS.md`
