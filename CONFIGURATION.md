# Guía de Configuración - MIMO DL Detector

Esta guía explica los parámetros de configuración disponibles y cómo usarlos.

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

**Última Actualización:** Enero 2025
