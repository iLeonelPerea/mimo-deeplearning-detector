# Análisis del Preprocesamiento Custom de Francisco

## Resumen Ejecutivo

Francisco Rubén Castillo-Soria y colaboradores implementan un preprocesamiento custom en su paper "Efficient Deep Learning-Based Detection Scheme for MIMO Communication Systems" (Sensors 2025) que **elimina la interferencia entre antenas antes de agregar ruido**.

Este preprocesamiento es similar a Zero-Forcing pero con una diferencia clave: **elimina el canal antes de agregar ruido**, mientras que Zero-Forcing estándar elimina el canal después de recibir la señal con ruido.

---

## 1. Preprocesamiento Custom de Francisco

### Código MATLAB (presente en los 3 métodos)

```matlab
% Ubicación: líneas 67-69 en training_2x2_detector_onehot_perAntenna.m
% Ubicación: líneas 62-64 en training_2x2_detector_SymbolEncoding.m
% Ubicación: líneas 54-56 en training_2x2_detector_OneHot.m

r_x = H*sel_symbol.';    % Paso 1: Aplica el canal H
H_inv = pinv(H);         % Paso 2: Calcula pseudoinversa H⁺
r_x = H_inv*r_x+n;       % Paso 3: Elimina canal, luego agrega ruido
```

### Análisis Matemático

**Paso 1:** Transmite a través del canal
```
r_temp = H * x
```

**Paso 2:** Elimina el canal aplicando la pseudoinversa
```
r_eq = H⁺ * r_temp = H⁺ * H * x ≈ I * x = x
```

**Paso 3:** Agrega ruido DESPUÉS de eliminar el canal
```
r = r_eq + n = x + n
```

**Resultado final:**
```
r = x + n
```

Las señales recibidas en cada antena son:
```matlab
r1 = x1 + n1  % Sin interferencia de x2
r2 = x2 + n2  % Sin interferencia de x1
```

---

## 2. Comparación con Zero-Forcing Estándar

### Zero-Forcing Estándar (IEEE)

```matlab
r_received = H*x + n;     % Recibe señal con interferencia + ruido
H_inv = pinv(H);          % Calcula pseudoinversa
r_eq = H_inv * r_received; % Aplica ZF
```

**Resultado:**
```
r_eq = H⁺ * (H*x + n)
     = H⁺*H*x + H⁺*n
     = x + H⁺*n
```

Las señales ecualizadas son:
```matlab
r1_eq = x1 + (H⁺*n)₁  % Sin interferencia, pero RUIDO AMPLIFICADO
r2_eq = x2 + (H⁺*n)₂  % Sin interferencia, pero RUIDO AMPLIFICADO
```

### Preprocesamiento Custom de Francisco

```matlab
r_temp = H*x;             % Aplica canal (sin ruido aún)
H_inv = pinv(H);          % Calcula pseudoinversa
r_eq = H_inv * r_temp;    % Elimina canal
r = r_eq + n;             % Agrega ruido DESPUÉS
```

**Resultado:**
```
r = H⁺*H*x + n
  = x + n
```

Las señales recibidas son:
```matlab
r1 = x1 + n1  % Sin interferencia, ruido NO amplificado
r2 = x2 + n2  % Sin interferencia, ruido NO amplificado
```

---

## 3. Diferencia Clave

| Aspecto | Zero-Forcing Estándar | Preprocesamiento de Francisco |
|---------|----------------------|-------------------------------|
| **Orden de operaciones** | Canal → Ruido → Ecualización | Canal → Ecualización → Ruido |
| **Modelo recibido** | `r = H*x + n` | `r = H*x` (temporal) |
| **Señal ecualizada** | `r_eq = x + H⁺*n` | `r = x + n` |
| **Interferencia entre antenas** | Eliminada | Eliminada |
| **Amplificación de ruido** | SÍ (factor ||H⁺||) | NO |
| **Realismo físico** | Sí (modelo IEEE) | No (solo para entrenamiento) |

---

## 4. ¿Por Qué es Similar a Zero-Forcing?

Ambos métodos eliminan la interferencia entre antenas aplicando `H⁺`:

### Sin preprocesamiento (MIMO estándar):
```matlab
r1 = h11*x1 + h12*x2 + n1  % x2 interfiere con x1
r2 = h21*x1 + h22*x2 + n2  % x1 interfiere con x2
```
**Problema:** Las antenas NO son independientes.

### Con Zero-Forcing o preprocesamiento de Francisco:
```matlab
r1 = x1 + ruido_efectivo_1  % Sin interferencia de x2
r2 = x2 + ruido_efectivo_2  % Sin interferencia de x1
```
**Beneficio:** Las antenas SON independientes.

La diferencia está en el "ruido_efectivo":
- **Zero-Forcing:** `ruido_efectivo = H⁺*n` (amplificado)
- **Francisco:** `ruido_efectivo = n` (sin amplificar)

---

## 5. Impacto en los Tres Métodos de Codificación

### 5.1 One-Hot (16 salidas)

**Arquitectura:** 4 entradas → 100 neuronas ocultas → 16 salidas

**Con interferencia (`r = H*x + n`):**
- ✅ Funciona bien (gap ~0.2 dB vs ML)
- Tiene 16 salidas, puede aprender todas las combinaciones `(x1, x2)` interferentes

**Sin interferencia (`r = x + n`):**
- ✅ Funciona bien (gap ~0.5 dB en paper de Francisco)
- Sigue teniendo capacidad suficiente

**Conclusión:** One-Hot es robusto, funciona con o sin interferencia.

---

### 5.2 Label Encoder / Direct Symbol Encoding (4 salidas)

**Arquitectura:** 4 entradas → 100 neuronas ocultas → 4 salidas

**Con interferencia (`r = H*x + n`):**
- ❌ Falla (gap ~6 dB vs ML en nuestro código)
- Las 4 salidas codifican símbolos `{0,1,2,3}` pero NO pueden capturar la dependencia entre `x1` y `x2`
- Ejemplo: `r1 = h11*00 + h12*01` vs `r1 = h11*00 + h12*10` generan diferentes `r1`, pero el modelo debe predecir el mismo símbolo para antena 1

**Sin interferencia (`r = x + n`):**
- ✅ Funciona bien (gap ~2 dB en paper de Francisco)
- Con `r1 = x1 + n1` y `r2 = x2 + n2`, cada antena es independiente
- Las 4 salidas son suficientes para clasificar símbolos QAM independientes

**Conclusión:** Label Encoder **requiere independencia entre antenas**.

---

### 5.3 One-Hot per Antenna / OHA (8 salidas: 4+4)

**Arquitectura:** 4 entradas → 100 neuronas ocultas → 8 salidas (4 para antena 1, 4 para antena 2)

**Con interferencia (`r = H*x + n`):**
- ❌ Falla gravemente (satura en BER = 1.5×10⁻², no baja más)
- El modelo tiene cabezas separadas para cada antena, pero `r1` contiene información de `x2`
- Arquitecturalmente asume independencia, pero la entrada tiene dependencia

**Sin interferencia (`r = x + n`):**
- ✅ Funciona bien (gap ~0.5 dB en paper de Francisco)
- Con `r1 = x1 + n1` y `r2 = x2 + n2`, la arquitectura split es perfecta
- Cada cabeza aprende a decodificar su antena independientemente

**Conclusión:** OHA **estrictamente requiere independencia entre antenas**.

---

## 6. Verificación en el Código de Francisco

### 6.1 Script de Entrenamiento OHA

Archivo: `roilhi/Matlab/training_2x2_detector_onehot_perAntenna.m`

```matlab
% Líneas 67-69
r_x = H*sel_symbol.';
H_inv = pinv(H);
r_x = H_inv*r_x+n;      % r = x + n
```

```matlab
% Líneas 72-74: Orden de características
orden = [1,3,2,4]; % [real(x1) imag(x1) real(x2) imag(x2)]
X = X(:,orden);    % [real(r1) imag(r1) real(r2) imag(r2)]
```

```matlab
% Líneas 155-162: Arquitectura split en salida
A2 = 1./(1+exp(-Z2));                    % Sigmoid en 8 salidas
A2_first_rows = A2(1:output_size/2,:);   % Primeras 4 salidas (antena 1)
A2_last_rows = A2((output_size/2)+1:end,:); % Últimas 4 salidas (antena 2)
[~, y_hat1] = max(A2_first_rows);        % argmax para antena 1
[~, y_hat2] = max(A2_last_rows);         % argmax para antena 2
```

**Análisis:** El preprocesamiento `r = x + n` es esencial para que esta arquitectura split funcione.

---

### 6.2 Script de Entrenamiento Label Encoder

Archivo: `roilhi/Matlab/training_2x2_detector_SymbolEncoding.m`

```matlab
% Líneas 62-64: MISMO preprocesamiento
r_x = H*sel_symbol.';
H_inv = pinv(H);
r_x = H_inv*r_x+n;      % r = x + n
```

```matlab
% Líneas 47-50: Codificación de símbolos
real_sign = real(prod_cart)<0;  % Signo de parte real
imag_sign = imag(prod_cart)<0;  % Signo de parte imaginaria
idx_sign = [real_sign(:,1) imag_sign(:,1) real_sign(:,2) imag_sign(:,2)];
```

```matlab
% Línea 155: Activación sigmoide
A2 = 1./(1+exp(-Z2));
```

```matlab
% Línea 160: Decodificación
[~, y_hat] = ismember((A2>0.5)',idx_sign,'rows');
```

**Análisis:** Label Encoder usa 4 salidas binarias para codificar signos. Requiere independencia entre antenas.

---

### 6.3 Script de Entrenamiento One-Hot

Archivo: `roilhi/Matlab/training_2x2_detector_OneHot.m`

```matlab
% Líneas 54-56: MISMO preprocesamiento
H_inv = pinv(H);
r_x = H_inv*r_x+n;      % r = x + n
```

```matlab
% Líneas 31-43: One-hot encoding de 16 combinaciones
prod_cart = [Xx(:) Yy(:)];  % Producto cartesiano M^Nt
y = zeros(N,M^Nt);          % 16 salidas
```

```matlab
% Línea 143: Activación softmax
A2 = exp(Z2)./sum(exp(Z2));
```

```matlab
% Línea 145: Decodificación
[~, y_hat] = max(A2);  % argmax de 16 clases
```

**Análisis:** One-Hot funciona con o sin preprocesamiento, pero Francisco también lo aplica.

---

## 7. Script de Evaluación BER

Archivo: `roilhi/Matlab/BER_4QAM_MIMO_2x2_All.m`

```matlab
% Líneas 102-105
Hinv = pinv(H);
H_eqz = H*Hinv;         % H * H⁺ ≈ I (matriz identidad)
r = H_eqz*x.' + n;      % r ≈ x + n
```

**Análisis:** También en evaluación aplica el mismo preprocesamiento.

---

## 8. Documentación en el Paper

### Lo que dice el paper (Sensors 2025, página 9):

> "the received samples are equalized"

### Lo que NO dice el paper:

- ❌ No explica QUÉ tipo de ecualización
- ❌ No menciona que elimina canal antes de agregar ruido
- ❌ No justifica por qué este preprocesamiento
- ❌ No compara con Zero-Forcing estándar
- ❌ Algorithm 1 (página 6) muestra `r = Hs + n` sin preprocesamiento

**Conclusión:** El preprocesamiento custom está en el código pero NO documentado en el paper.

---

## 9. ¿Es Zero-Forcing?

### Respuesta: **Casi, pero no exactamente**

**Similitudes con Zero-Forcing:**
- ✅ Aplica pseudoinversa `H⁺` para eliminar interferencia
- ✅ Resultado final: señales independientes por antena
- ✅ Permite decodificación separada de símbolos

**Diferencias con Zero-Forcing:**
- ❌ No amplifica ruido (ruido se agrega DESPUÉS)
- ❌ No es físicamente realizable (¿cómo transmites por canal sin ruido?)
- ❌ Solo aplicable en simulación/entrenamiento

### Nombre más apropiado:

**"Desacoplamiento de Antenas"** o **"Channel-Free Training"**

Es una técnica de preprocesamiento de datos de entrenamiento, NO un algoritmo de ecualización de sistemas reales.

---

## 10. Implicaciones para Nuestro Código

### Situación actual:

**Con Zero-Forcing (commit a1bf68c, 16-nov-2025):**
```python
r_x = sqrt(SNR) * H * x + n  # Recibe con interferencia
H_inv = pinv(H)
r_eq = H_inv * r_x           # r_eq = x + H_inv*n
# Entrena con r_eq
```

**Resultados:**
- One-Hot: ~0.5 dB gap vs ML
- Label Encoder: ~2 dB gap vs ML
- OHA: ~2-3 dB gap vs ML

**Sin preprocesamiento (código actual):**
```python
r_x = sqrt(SNR) * H * x + n  # Entrena directamente con interferencia
```

**Resultados:**
- One-Hot: ~0.2 dB gap vs ML ✅
- Label Encoder: ~6 dB gap vs ML ❌
- OHA: Satura en 1.5×10⁻² ❌

---

## 11. Pregunta Pendiente para Francisco

¿Es correcto usar Zero-Forcing (o el preprocesamiento `r = x + n`) como preprocesamiento para Deep Learning?

**Opciones:**

### Opción A: Sistema con ZF + DL
```
[Transmisor] → Canal → [Receptor con ZF] → [Detector DL]
                r = H*x+n    r_eq = x+H⁺*n
```
El detector DL recibe señales ecualizadas con ruido amplificado.

### Opción B: Sistema solo DL (sin ZF)
```
[Transmisor] → Canal → [Detector DL]
                r = H*x+n
```
El detector DL recibe señales con interferencia.

### Opción C: Preprocesamiento solo para entrenamiento
```
Entrenamiento: r = x + n (datos sintéticos sin interferencia)
Despliegue:    r = H*x + n (datos reales con interferencia)
```
⚠️ **Problema:** Mismatch entre entrenamiento y despliegue.

---

## 12. Conclusiones

1. **Francisco usa preprocesamiento custom** en los 3 métodos: One-Hot, Label Encoder, OHA
2. El preprocesamiento **elimina interferencia entre antenas** creando `r = x + n`
3. Es **similar a Zero-Forcing** pero sin amplificación de ruido
4. **Label Encoder y OHA requieren este preprocesamiento** para funcionar
5. **One-Hot funciona con o sin preprocesamiento** (arquitectura más robusta)
6. El preprocesamiento **NO está documentado en el paper**
7. Necesitamos clarificación de Francisco sobre la justificación y aplicabilidad real

---

## Referencias

- Paper: Ibarra-Hernández et al., "Efficient Deep Learning-Based Detection Scheme for MIMO Communication Systems", Sensors 2025
- Código MATLAB: `roilhi/Matlab/training_2x2_detector_*.m`
- Nuestros resultados: Commits a1bf68c (con ZF) y actuales (sin ZF)
