import numpy as np
from orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp

# -----------------------------------------------------------
# Kernel RBF
# -----------------------------------------------------------
def rbf_kernel(x1, x2, length_scale=1.0):
    """
    Implementa el kernel RBF (Radial Basis Function).
    
    k(x1, x2) = exp(-||x1 - x2||^2 / (2 * length_scale^2))
    
    Args:
        x1: Vector o matriz de puntos
        x2: Vector o matriz de puntos
        length_scale: Parámetro de escala del kernel
    
    Returns:
        Valor del kernel RBF
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    # Calcular distancia euclidiana al cuadrado
    sq_dist = np.sum((x1[:, np.newaxis] - x2[np.newaxis, :]) ** 2, axis=2)
    
    # Aplicar función RBF
    return np.exp(-sq_dist / (2 * length_scale ** 2))

# -----------------------------------------------------------
# Ajuste del GP
# -----------------------------------------------------------
def fit_gp(X, y, length_scale=1.0, noise=1e-6):
    """
    Ajusta el Gaussian Process a los datos observados.
    
    Resuelve el sistema: (K + noise*I)^(-1) * y = alpha
    
    Args:
        X: Matriz de puntos observados (n x d)
        y: Vector de valores observados (n,)
        length_scale: Parámetro del kernel RBF
        noise: Término de ruido para estabilidad numérica
    
    Returns:
        Diccionario con parámetros del GP ajustado
    """
    X = np.atleast_2d(X)
    y = np.atleast_1d(y)
    
    # Construir matriz de covarianza K
    K = rbf_kernel(X, X, length_scale)
    
    # Añadir ruido a la diagonal para estabilidad numérica
    K_noise = K + noise * np.eye(len(X))
    
    # Resolver sistema lineal: alpha = (K + noise*I)^(-1) * y
    alpha = np.linalg.solve(K_noise, y)
    
    return {
        'X_train': X,
        'y_train': y,
        'alpha': alpha,
        'K_noise': K_noise,
        'length_scale': length_scale,
        'noise': noise
    }

# -----------------------------------------------------------
# Predicción del GP
# -----------------------------------------------------------
def gp_predict(X_train, y_train, X_test, length_scale=1.0, noise=1e-6):
    """
    Realiza predicciones con el GP en nuevos puntos.
    
    Calcula:
    - Media: mu(x*) = k(x*)^T * alpha
    - Varianza: sigma^2(x*) = k(x*, x*) - k(x*)^T * (K + noise*I)^(-1) * k(x*)
    
    Args:
        X_train: Puntos de entrenamiento (n x d)
        y_train: Valores observados (n,)
        X_test: Puntos donde predecir (m x d)
        length_scale: Parámetro del kernel
        noise: Término de ruido
    
    Returns:
        mu: Vector de medias predictivas (m,)
        sigma: Vector de desviaciones estándar (m,)
    """
    X_train = np.atleast_2d(X_train)
    X_test = np.atleast_2d(X_test)
    y_train = np.atleast_1d(y_train)
    
    # Ajustar GP
    gp_params = fit_gp(X_train, y_train, length_scale, noise)
    alpha = gp_params['alpha']
    K_noise = gp_params['K_noise']
    
    # Calcular k(x*) para cada punto de test
    k_star = rbf_kernel(X_test, X_train, length_scale)
    
    # Media predictiva: mu(x*) = k(x*)^T * alpha
    mu = k_star @ alpha
    
    # Varianza predictiva
    k_star_star = rbf_kernel(X_test, X_test, length_scale)
    
    # sigma^2(x*) = k(x*, x*) - k(x*)^T * (K + noise*I)^(-1) * k(x*)
    v = np.linalg.solve(K_noise, k_star.T)
    var = np.diag(k_star_star) - np.sum(k_star * v.T, axis=1)
    
    # Asegurar que la varianza no sea negativa (por errores numéricos)
    var = np.maximum(var, 1e-10)
    
    # Retornar desviación estándar
    sigma = np.sqrt(var)
    
    return mu, sigma

# -----------------------------------------------------------
# Función de adquisición UCB
# -----------------------------------------------------------
def acquisition_ucb(mu, sigma, kappa=2.0):
    """
    Calcula Upper Confidence Bound (UCB).
    
    UCB = mu + kappa * sigma
    
    Args:
        mu: Media predictiva
        sigma: Desviación estándar
        kappa: Parámetro de balance entre exploración y explotación
    
    Returns:
        Valor de UCB
    """
    return mu + kappa * sigma

# -----------------------------------------------------------
# BO principal
# -----------------------------------------------------------
def optimize_model(model_name, n_init=3, n_iter=10, return_history=False):
    """
    Optimiza hiperparámetros usando Optimización Bayesiana.
    
    Args:
        model_name: Nombre del modelo ('svm', 'rf', 'mlp')
        n_init: Número de puntos iniciales aleatorios
        n_iter: Número de iteraciones de BO
        return_history: Si True, retorna también el historial completo
    
    Returns:
        best_params: Mejor configuración encontrada
        best_metric: Mejor métrica (float) alcanzada
        history (opcional): Lista de (params, metric) si return_history=True
    """
    # Definir dominios discretos según el modelo
    if model_name == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        eval_func = evaluate_svm
    elif model_name == 'rf':
        param_grid = {
            'n_estimators': [10, 20, 50, 100],
            'max_depth': [2, 4, 6, 8]
        }
        eval_func = evaluate_rf
    elif model_name == 'mlp':
        param_grid = {
            'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16)],
            'alpha': [1e-4, 1e-3, 1e-2]
        }
        eval_func = evaluate_mlp
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    
    # Generar rejilla completa de hiperparámetros
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Crear todas las combinaciones posibles
    from itertools import product
    all_combinations = list(product(*param_values))
    
    # Convertir combinaciones a formato numérico para el GP
    def params_to_vector(params_tuple):
        """Convierte tupla de parámetros a vector numérico"""
        vec = []
        for i, val in enumerate(params_tuple):
            if isinstance(val, tuple):
                # Para hidden_layer_sizes, usar longitud y valores
                vec.extend(val)
                # Rellenar para mantener dimensionalidad fija
                vec.extend([0] * (2 - len(val)))
            else:
                vec.append(val)
        return np.array(vec, dtype=float)
    
    def vector_to_params(vec):
        """Convierte vector numérico de vuelta a diccionario de parámetros"""
        params = {}
        if model_name == 'svm':
            params['C'] = vec[0]
            params['gamma'] = vec[1]
        elif model_name == 'rf':
            params['n_estimators'] = int(vec[0])
            params['max_depth'] = int(vec[1])
        elif model_name == 'mlp':
            # Reconstruir hidden_layer_sizes
            if vec[1] == 0:
                params['hidden_layer_sizes'] = (int(vec[0]),)
            else:
                params['hidden_layer_sizes'] = (int(vec[0]), int(vec[1]))
            params['alpha'] = vec[2] if len(vec) > 2 else vec[-1]
        return params
    
    # Inicialización: seleccionar n_init puntos aleatorios
    n_total = len(all_combinations)
    init_indices = np.random.choice(n_total, size=min(n_init, n_total), replace=False)
    
    # Historial de observaciones
    X_observed = []
    y_observed = []
    params_history = []
    iteration_history = []  # Para tracking de evolución
    
    print(f"\n=== Optimizando {model_name.upper()} con BO ===")
    print(f"Espacio de búsqueda: {n_total} combinaciones")
    print(f"Inicialización: {n_init} puntos aleatorios")
    print(f"Iteraciones BO: {n_iter}")
    
    # Fase de inicialización
    print("\n--- Fase de Inicialización ---")
    for idx in init_indices:
        params_tuple = all_combinations[idx]
        x_vec = params_to_vector(params_tuple)
        params_dict = vector_to_params(x_vec)
        
        # Evaluar modelo
        metric = eval_func(params_dict)
        
        X_observed.append(x_vec)
        y_observed.append(metric)
        params_history.append(params_dict)
        iteration_history.append({
            'iteration': len(X_observed),
            'params': params_dict.copy(),
            'metric': metric,
            'best_so_far': max(y_observed),
            'phase': 'initialization'
        })
        
        print(f"Punto {len(X_observed)}: {params_dict} -> Métrica: {metric:.4f}")
    
    X_observed = np.array(X_observed)
    y_observed = np.array(y_observed)
    
    # Fase iterativa de BO
    print("\n--- Fase de Optimización Bayesiana ---")
    for iteration in range(n_iter):
        # Ajustar GP con datos observados
        gp_params = fit_gp(X_observed, y_observed, length_scale=1.0, noise=1e-6)
        
        # Evaluar función de adquisición en toda la rejilla
        X_candidates = np.array([params_to_vector(c) for c in all_combinations])
        
        # Filtrar puntos ya evaluados
        evaluated_mask = np.zeros(len(X_candidates), dtype=bool)
        for x_obs in X_observed:
            distances = np.sum((X_candidates - x_obs) ** 2, axis=1)
            evaluated_mask |= (distances < 1e-6)
        
        unevaluated_indices = np.where(~evaluated_mask)[0]
        
        if len(unevaluated_indices) == 0:
            print("Todos los puntos han sido evaluados.")
            break
        
        X_unevaluated = X_candidates[unevaluated_indices]
        
        # Predecir con GP
        mu, sigma = gp_predict(X_observed, y_observed, X_unevaluated, 
                              length_scale=1.0, noise=1e-6)
        
        # Calcular UCB
        ucb_values = acquisition_ucb(mu, sigma, kappa=2.0)
        
        # Seleccionar punto con mayor UCB
        best_ucb_idx = np.argmax(ucb_values)
        next_idx = unevaluated_indices[best_ucb_idx]
        next_params_tuple = all_combinations[next_idx]
        next_x_vec = params_to_vector(next_params_tuple)
        next_params_dict = vector_to_params(next_x_vec)
        
        # Evaluar modelo en el nuevo punto
        next_metric = eval_func(next_params_dict)
        
        # Agregar a historial
        X_observed = np.vstack([X_observed, next_x_vec])
        y_observed = np.append(y_observed, next_metric)
        params_history.append(next_params_dict)
        iteration_history.append({
            'iteration': len(X_observed),
            'params': next_params_dict.copy(),
            'metric': next_metric,
            'best_so_far': max(y_observed),
            'phase': 'bayesian_optimization',
            'ucb': float(ucb_values[best_ucb_idx]),
            'mu': float(mu[best_ucb_idx]),
            'sigma': float(sigma[best_ucb_idx])
        })
        
        print(f"Iteración {iteration + 1}: {next_params_dict}")
        print(f"  -> Métrica: {next_metric:.4f} (UCB: {ucb_values[best_ucb_idx]:.4f})")
    
    # Encontrar mejor configuración
    best_idx = np.argmax(y_observed)
    best_metric = float(y_observed[best_idx])
    best_params = params_history[best_idx]
    
    print(f"\n=== Mejor configuración encontrada ===")
    print(f"Parámetros: {best_params}")
    print(f"Métrica: {best_metric:.4f}")
    
    if return_history:
        return best_params, best_metric, iteration_history
    else:
        return best_params, best_metric


# Función auxiliar para pruebas
if __name__ == "__main__":
    # Probar con cada modelo
    models = ['svm', 'rf', 'mlp']
    
    for model in models:
        print(f"\n{'='*60}")
        best_params, best_metric = optimize_model(model, n_init=3, n_iter=5)
        print(f"Modelo {model.upper()}: Mejor métrica = {best_metric:.4f}")