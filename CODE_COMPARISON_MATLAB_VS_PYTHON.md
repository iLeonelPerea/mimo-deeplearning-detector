# Comparación Línea por Línea: Código MATLAB vs Python (Nuestro)

## 1. Generación del Canal H

### MATLAB
```matlab
% training_2x2_detector_OneHot.m línea 50
H = (1/sqrt(2))*(randn(Nr,Nt) + 1i*randn(Nr,Nt));
```
- **Canal ALEATORIO** por cada muestra de entrenamiento
- Normalización: `1/sqrt(2)`
- Rayleigh fading: `CN(0, 1)`

### Python (Nuestro)
```python
# modelMIMO_2x2_4QAM_DoubleOneHot.py líneas 159-162
H = torch.tensor([[-0.90064 + 1j*0.43457, -0.99955 + 1j*0.029882],
                  [-0.1979 + 1j*0.98022, 0.44866 + 1j*0.8937]],
                 dtype=torch.complex64, device=device)
H = H / torch.abs(H)  # Normalize by element-wise magnitude
```
- **Canal FIJO** para todas las muestras
- Normalización: element-wise `|h_ij|`

### 🔴 DIFERENCIA #1: Canal aleatorio vs fijo
**Impacto:** En Matlab entrena con múltiples realizaciones del canal (más realista), nosotros con un solo canal (sobreajuste posible).

---

## 2. Generación de Ruido

### MATLAB
```matlab
% training_2x2_detector_OneHot.m líneas 51-52
n = (No/sqrt(2))*(randn(Nr,1) + 1i*randn(Nr,1));
n = (1/sqrt(SNR_l))*n;
```
Expandiendo:
```matlab
n = (No/sqrt(2)) * (randn + 1i*randn) * (1/sqrt(SNR))
```
Con `No = 1`:
```matlab
n = (1/sqrt(2)) * (randn + 1i*randn) * (1/sqrt(SNR))
n = (1/sqrt(2*SNR)) * (randn + 1i*randn)
```

Varianza del ruido:
- Parte real: `Var(real) = (1/(2*SNR)) * 1/2 = 1/(4*SNR)`
- Parte imag: `Var(imag) = (1/(2*SNR)) * 1/2 = 1/(4*SNR)`
- Varianza total: `σ²_n = 1/(4*SNR) + 1/(4*SNR) = 1/(2*SNR)`

### Python (Nuestro)
```python
# modelMIMO_2x2_4QAM_DoubleOneHot.py líneas 184-186
n_real = torch.randn(Nr, device=device) * np.sqrt(No/2)
n_imag = torch.randn(Nr, device=device) * np.sqrt(No/2)
n = torch.complex(n_real, n_imag)
```
Con `No = 1`:
```python
n = sqrt(1/2) * randn + 1j * sqrt(1/2) * randn
```

Varianza del ruido:
- Parte real: `Var(real) = 1/2 * 1 = 1/2`
- Parte imag: `Var(imag) = 1/2 * 1 = 1/2`
- Varianza total: `σ²_n = 1/2 + 1/2 = 1`

### 🔴 DIFERENCIA #2: Varianza del ruido
| | En Matlab | Nuestro |
|---|-----------|---------|
| Varianza ruido | `1/(2*SNR)` | `1` (fijo) |
| Depende de SNR | ✅ Sí | ❌ No |

**Impacto:** Nuestro ruido tiene varianza fija, En Matlab escala el ruido con SNR.

---

## 3. SNR de Entrenamiento

### MATLAB
```matlab
% training_2x2_detector_OneHot.m línea 43
SNR_dB = 3; % SNR for add noise to training data
SNR_l = 10.^(SNR_dB./10);
```
- **SNR FIJO: 3 dB** para todas las muestras

### Python (Nuestro)
```python
# modelMIMO_2x2_4QAM_DoubleOneHot.py líneas 181-182
SNR_dB_sample = np.random.randint(1, 21)  # Random SNR between 1-20 dB
SNR_linear_sample = 10.0 ** (SNR_dB_sample / 10.0)
```
- **SNR ALEATORIO: 1-20 dB** por cada muestra

### 🔴 DIFERENCIA #3: SNR de entrenamiento
**Impacto:** Nosotros entrenamos con múltiples SNR (más robusto), En Matlab con un solo SNR (especializado).

---

## 4. Preprocesamiento de la Señal Recibida

### MATLAB - ENTRENAMIENTO
```matlab
% training_2x2_detector_OneHot.m líneas 53-55
r_x = H*sel_symbol.';     % Paso 1: Aplica canal
H_inv = pinv(H);          % Paso 2: Calcula H⁺
r_x = H_inv*r_x+n;        % Paso 3: Elimina canal, agrega ruido
```
**Resultado:** `r = x + n` (sin interferencia entre antenas)

### Python (Nuestro) - ENTRENAMIENTO
```python
# modelMIMO_2x2_4QAM_DoubleOneHot.py línea 189
r_x = np.sqrt(SNR_linear_sample) * torch.matmul(H, selected_symbols) + n

# Con use_zf=False (por defecto)
r_processed = r_x  # línea 197
```
**Resultado:** `r = sqrt(SNR) * H*x + n` (CON interferencia entre antenas)

### 🔴 DIFERENCIA #4: Preprocesamiento (LA MÁS CRÍTICA)
| | En Matlab | Nuestro (actual) | Nuestro (antiguo con ZF) |
|---|-----------|------------------|--------------------------|
| Modelo | `r = x + n` | `r = sqrt(SNR)*H*x + n` | `r = x + H⁺*n` |
| Interferencia | ❌ No | ✅ Sí | ❌ No |
| Ruido amplificado | ❌ No | ❌ No | ✅ Sí |
| Físicamente realizable | ❌ No | ✅ Sí | ✅ Sí |

**Impacto CRÍTICO:**
- En Matlab elimina interferencia SIN amplificar ruido (solo simulación)
- Nuestro código actual mantiene interferencia (realista)
- Nuestro código antiguo eliminaba interferencia CON amplificación (ZF estándar)

---

## 5. Normalización de Símbolos

### MATLAB
```matlab
% training_2x2_detector_OneHot.m líneas 28-31
M = 4; % 4-QAM
Xx = [-1 1];
Yy = [-1 1];
prod_cart = [Xx(:) Yy(:)];  % Símbolos sin normalizar
```
Símbolos: `{-1-1j, -1+1j, 1-1j, 1+1j}`

Potencia: `E[|x|²] = (1² + 1²) = 2`

### Python (Nuestro)
```python
# modelMIMO_2x2_4QAM_DoubleOneHot.py líneas 125-130
symbol_map = {
    0: -1 - 1j, 1: -1 + 1j,
    2: 1 - 1j,  3: 1 + 1j
}
symbol_combinations = torch.tensor([...], dtype=torch.complex64)
symbol_combinations = symbol_combinations / np.sqrt(2)  # línea 180
```
Símbolos normalizados: `{-1/√2 - 1j/√2, ...}`

Potencia: `E[|x|²] = (1/2 + 1/2) = 1`

### 🔴 DIFERENCIA #5: Normalización de símbolos
| | En Matlab | Nuestro |
|---|-----------|---------|
| Símbolos | `±1 ± 1j` | `±1/√2 ± 1j/√2` |
| Potencia | 2 | 1 (IEEE) |

**Impacto:** Escalamiento diferente de la señal.

---

## 6. Modelo de Señal Recibida

### MATLAB - ENTRENAMIENTO
```matlab
% Combinando todo:
H = (1/sqrt(2))*(randn(Nr,Nt) + 1i*randn(Nr,Nt));  % E[|h|²] = 1/2
x = sel_symbol.';                                   % E[|x|²] = 2
n = (1/sqrt(2*SNR))*(randn + 1i*randn);            % E[|n|²] = 1/(2*SNR)

r_temp = H*x;         % E[|H*x|²] = E[|h|²]*E[|x|²] = (1/2)*2 = 1
r_eq = H⁺ * r_temp;   % E[|r_eq|²] ≈ E[|x|²] = 2
r = r_eq + n;         % E[|r|²] ≈ 2 + 1/(2*SNR)
```

**SNR efectivo:**
```
SNR_efectivo = E[|señal|²] / E[|ruido|²]
             = 2 / (1/(2*SNR))
             = 4*SNR
```

### Python (Nuestro)
```python
H = fixed_matrix / |H|              # Normalización element-wise (no clear E[|h|²])
x = symbols / sqrt(2)               # E[|x|²] = 1
n = randn/sqrt(2) + 1j*randn/sqrt(2)  # E[|n|²] = 1

r = sqrt(SNR) * H*x + n
```

**SNR efectivo:**
```
SNR_efectivo = E[|sqrt(SNR)*H*x|²] / E[|n|²]
             = SNR * E[|H|²] * E[|x|²] / 1
             = SNR * E[|H|²] * 1
```

Con normalización element-wise de H, `E[|H|²]` no está claro.

### 🔴 DIFERENCIA #6: Modelo de señal completo
En Matlab y nosotros usamos modelos de potencia diferentes.

---

## 7. Evaluación BER

### MATLAB
```matlab
% BER_4QAM_MIMO_2x2_All.m líneas 102-105
Hinv = pinv(H);
H_eqz = H*Hinv;           % H * H⁺ ≈ I
r = H_eqz*x.' + n;        % r ≈ x + n

% Línea 120: Alimenta a los modelos
Xinput = [real_r(1) imag_r(1) real_r(2) imag_r(2)];
```
**También en evaluación elimina interferencia.**

### Python (Nuestro)
```python
# ber_4qam_mimo_2x2_all.py línea 630
r = sqrt_SNR_j * (H_fixed @ x_transmitted) + n

# Con USE_ZF=False (líneas 396-399)
if use_zf and H_inv is not None:
    r_processed = H_inv @ r
else:
    r_processed = r  # Usa señal directa
```
**Por defecto mantiene interferencia.**

### 🔴 DIFERENCIA #7: Consistencia entrenamiento-evaluación

| | Entrenamiento | Evaluación | ¿Consistente? |
|---|---------------|------------|---------------|
| **En Matlab** | `r = x + n` | `r = x + n` | ✅ Sí |
| **Nuestro (actual)** | `r = sqrt(SNR)*H*x + n` | `r = sqrt(SNR)*H*x + n` | ✅ Sí |

Ambos somos consistentes, pero usamos **modelos diferentes**.

---

## 8. Orden de Características de Entrada

### MATLAB
```matlab
% training_2x2_detector_OneHot.m línea 56
X(i,:) = [real(r_x.') imag(r_x.')];
% Resultado: [real(r1) real(r2) imag(r1) imag(r2)]

% Línea 59-60: REORDENA
orden = [1,3,2,4]; % [real(x1) imag(x1) real(x2) imag(x2)]
X = X(:,orden);
% Resultado FINAL: [real(r1) imag(r1) real(r2) imag(r2)]
```

### Python (Nuestro)
```python
# modelMIMO_2x2_4QAM_DoubleOneHot.py líneas 199-202
X_data[i, 0] = r_processed[0].real
X_data[i, 1] = r_processed[0].imag
X_data[i, 2] = r_processed[1].real
X_data[i, 3] = r_processed[1].imag
# Resultado: [real(r1) imag(r1) real(r2) imag(r2)]
```

### ✅ SIN DIFERENCIA: Orden de características
Ambos usan `[real(r1) imag(r1) real(r2) imag(r2)]`.

---

## 9. Arquitectura de Red Neuronal

### MATLAB - One-Hot
```matlab
% training_2x2_detector_OneHot.m líneas 70-92
% Input(4) -> Hidden(100) -> Output(16)
W1 = randn(hidden_size, input_size);
b1 = randn(hidden_size, 1);
W2 = randn(output_size, hidden_size);
b2 = randn(output_size, 1);

% Forward pass (líneas 137-143)
Z1 = W1*Xinput'+b1;
A1 = max(0,Z1);        % ReLU
Z2 = W2*A1+b2;
A2 = exp(Z2)./sum(exp(Z2));  % Softmax
```

### Python (Nuestro)
```python
# modelMIMO_2x2_4QAM_DoubleOneHot.py líneas 86-101
# Input(4) -> Hidden(100) -> Output(16)
self.layer1 = nn.Linear(input_size, hidden_size, bias=False)  # Sin bias!
self.layer2 = nn.Linear(hidden_size, output_size)

def forward(self, x):
    x = self.layer1(x)
    x = F.relu(x)
    x = self.layer2(x)
    return x  # Softmax se aplica después en CrossEntropyLoss
```

### 🔴 DIFERENCIA #8: Bias en capa oculta
| | En Matlab | Nuestro |
|---|-----------|---------|
| Bias en capa 1 | ✅ Sí | ❌ No |
| Bias en capa 2 | ✅ Sí | ✅ Sí |

**Impacto:** En Matlab tiene más parámetros entrenables.

---

## 10. Activación de Salida

### MATLAB - One-Hot
```matlab
% training_2x2_detector_OneHot.m línea 143
A2 = exp(Z2)./sum(exp(Z2));  % Softmax explícito
[~,idx_DL_1] = max(A2);
```

### Python (Nuestro) - Entrenamiento
```python
# modelMIMO_2x2_4QAM_DoubleOneHot.py líneas 215-217
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss incluye softmax
loss = criterion(outputs, labels)
```

### Python (Nuestro) - Evaluación
```python
# ber_4qam_mimo_2x2_all.py líneas 359-362
with torch.no_grad():
    outputs = model(x_input)
    probs = F.softmax(outputs, dim=1)
    _, predicted = torch.max(probs, 1)
```

### ✅ SIN DIFERENCIA: Activación de salida
Ambos usan softmax para One-Hot.

---

## Resumen de Diferencias Críticas

| # | Diferencia | En Matlab | Nuestro | Impacto |
|---|-----------|-----------|---------|---------|
| **1** | Canal | Aleatorio por muestra | Fijo | 🔴 Alto |
| **2** | Varianza ruido | `1/(2*SNR)` | `1` fijo | 🔴 Alto |
| **3** | SNR entrenamiento | Fijo (3 dB) | Aleatorio (1-20 dB) | 🟡 Medio |
| **4** | **Preprocesamiento** | **`r = x + n`** | **`r = H*x + n`** | 🔴 **CRÍTICO** |
| **5** | Normalización símbolos | `E[|x|²] = 2` | `E[|x|²] = 1` | 🟡 Medio |
| **6** | Modelo de potencia | Custom | IEEE estándar | 🟡 Medio |
| **7** | Consistencia train/eval | `r = x + n` ambos | `r = H*x + n` ambos | ✅ OK |
| **8** | Bias en capa oculta | Sí | No | 🟢 Bajo |

---

## Conclusión: ¿Qué está causando la diferencia en resultados?

### Diferencia #4 es LA CLAVE 🎯

**En Matlab:**
- Entrena con `r = x + n` (sin interferencia)
- Evalúa con `r = x + n` (sin interferencia)
- **Label Encoder y OHA funcionan** porque las antenas son independientes

**Nosotros:**
- Entrenamos con `r = sqrt(SNR)*H*x + n` (CON interferencia)
- Evaluamos con `r = sqrt(SNR)*H*x + n` (CON interferencia)
- **Label Encoder y OHA fallan** porque asumen independencia que no existe

---

## Recomendación

Para replicar los resultados de En Matlab, necesitamos implementar su preprocesamiento:

```python
# En entrenamiento y evaluación
r_temp = sqrt(SNR) * H @ x  # Aplica canal
H_inv = torch.linalg.pinv(H)
r = H_inv @ r_temp + n      # Elimina canal, luego agrega ruido
# Resultado: r = x + n
```

Esto hace que Label Encoder y OHA funcionen, pero **NO es físicamente realizable** en sistemas reales (solo para simulación/investigación).

Alternativamente, podríamos usar Zero-Forcing estándar:

```python
# En entrenamiento y evaluación
r = sqrt(SNR) * H @ x + n   # Recibe con interferencia
H_inv = torch.linalg.pinv(H)
r_eq = H_inv @ r            # Aplica ZF
# Resultado: r_eq = x + H^+*n (con ruido amplificado)
```

Esto también hace que Label Encoder y OHA funcionen, y **SÍ es físicamente realizable** (receptores reales pueden hacer esto).
