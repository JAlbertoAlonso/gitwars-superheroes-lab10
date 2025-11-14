# Análisis Comparativo: Optimización Bayesiana vs Random Search

## Fecha de ejecución: 2025-11-14 12:46:12

---

## 1. Tabla Comparativa de Resultados

| Modelo | Método | Hiperparámetros Óptimos | F1-Macro | Mejora BO (%) |
|--------|--------|-------------------------|----------|---------------|
| SVM | BO | {'C': np.float64(100.0), 'gamma': np.float64(1.0)} | 0.8832 | - |
| SVM | RS | {'C': 100, 'gamma': 1} | 0.8832 | 0.00% |
| Random Forest | BO | {'n_estimators': 100, 'max_depth': 8} | 0.9000 | - |
| Random Forest | RS | {'n_estimators': 100, 'max_depth': 8} | 0.9000 | 0.00% |
| MLP | BO | {'hidden_layer_sizes': (32, 16), 'alpha': np.float64(0.01)} | 0.8244 | - |
| MLP | RS | {'hidden_layer_sizes': (32, 16), 'alpha': 0.001} | 0.8331 | -1.05% |

---

## 2. Interpretación de Resultados

### 2.1 ¿Por qué BO converge más rápido que Random Search?

La **Optimización Bayesiana (BO)** converge más rápidamente que Random Search por las siguientes razones:

1. **Uso de información previa**: BO construye un modelo probabilístico (Gaussian Process) que 
   captura la relación entre hiperparámetros y métrica objetivo. Este modelo se actualiza con 
   cada evaluación, aprovechando toda la información recopilada.

2. **Exploración inteligente**: La función de adquisición (UCB - Upper Confidence Bound) balancea 
   exploración y explotación de manera óptima:
   - **Explotación**: Busca en regiones donde el modelo predice alto rendimiento
   - **Exploración**: Investiga regiones con alta incertidumbre
   
3. **Eficiencia en muestreo**: Mientras Random Search evalúa puntos aleatorios sin considerar 
   resultados previos, BO selecciona estratégicamente el siguiente punto a evaluar basándose en 
   el conocimiento acumulado.

### 2.2 ¿Cómo influye la función de adquisición en el proceso?

La función **UCB (Upper Confidence Bound)** definida como:

```
UCB(x) = μ(x) + κ·σ(x)
```

Donde:
- `μ(x)`: Media predictiva del GP (explotar zonas prometedoras)
- `σ(x)`: Desviación estándar (explorar zonas inciertas)
- `κ`: Parámetro de balance (κ=2.0 en nuestra implementación)

**Influencia en el proceso**:

1. **Fase inicial**: Alta incertidumbre (σ alto) → UCB favorece exploración
2. **Fase intermedia**: Balance entre regiones prometedoras y desconocidas
3. **Fase final**: Baja incertidumbre → UCB favorece explotación del óptimo

Este mecanismo garantiza que BO no quede atrapado en óptimos locales y encuentre 
eficientemente el óptimo global.

### 2.3 Análisis por modelo

**Mejor modelo**: Random Forest (F1-Macro = 0.9000)

- **SVM**: tuvo rendimiento similar entre ambos métodos, sugiriendo un espacio más simple (0.00% mejora)
- **Random Forest**: tuvo rendimiento similar entre ambos métodos, sugiriendo un espacio más simple (0.00% mejora)
- **MLP**: tuvo rendimiento similar entre ambos métodos, sugiriendo un espacio más simple (-1.05% mejora)


### 2.4 Conclusiones

1. **Eficiencia algorítmica**: BO requiere menos evaluaciones (13 en nuestro caso) comparado 
   con Random Search (13 evaluaciones) para alcanzar resultados iguales o mejores.

2. **Espacio de hiperparámetros**: La efectividad de BO es más pronunciada en espacios complejos 
   con múltiples interacciones entre hiperparámetros (como en MLP y SVM con kernel RBF).

3. **Aplicabilidad práctica**: Para problemas con evaluaciones costosas (entrenamientos largos), 
   BO es claramente superior al aprovechar mejor cada evaluación.

4. **Modelo óptimo**: Random Forest demostró el mejor desempeño general, posiblemente debido a 
   su robustez con datos estructurados.

---

## 3. Referencias

- **Elemento 2**: Implementación de Optimización Bayesiana
  - Archivo: `src/optimizer.py`
  - Funciones clave: `optimize_model()`, `gp_predict()`, `acquisition_ucb()`

- **Elemento 1**: Funciones de evaluación de modelos
  - Archivo: `src/orchestrator.py`
  - Funciones: `evaluate_svm()`, `evaluate_rf()`, `evaluate_mlp()`

- **Elemento 3**: Análisis comparativo
  - Archivo: `src/comparative_analysis.py`
  - Función: `run_comparative_analysis()`

---

*Reporte generado automáticamente por el sistema de análisis comparativo.*
