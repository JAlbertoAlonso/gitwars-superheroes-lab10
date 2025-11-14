import numpy as np
from orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp

def random_search_optimize(model_name, n_iterations=15):
    """
    Optimiza hiperparámetros usando Random Search.
    
    Args:
        model_name: Nombre del modelo ('svm', 'rf', 'mlp')
        n_iterations: Número de combinaciones aleatorias a probar
    
    Returns:
        best_params: Mejor configuración encontrada
        best_metric: Mejor métrica (float) alcanzada
        history: Lista de tuplas (params, metric) para análisis
    """
    # Definir dominios según el modelo
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
    
    # Generar todas las combinaciones posibles
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    print(f"\n=== Optimizando {model_name.upper()} con Random Search ===")
    print(f"Espacio de búsqueda: {len(all_combinations)} combinaciones")
    print(f"Iteraciones: {n_iterations}")
    
    # Seleccionar combinaciones aleatorias
    n_samples = min(n_iterations, len(all_combinations))
    sampled_indices = np.random.choice(len(all_combinations), size=n_samples, replace=False)
    
    best_params = None
    best_metric = -np.inf
    history = []
    
    print("\n--- Evaluaciones Random Search ---")
    for i, idx in enumerate(sampled_indices, 1):
        # Construir diccionario de parámetros
        params_tuple = all_combinations[idx]
        params_dict = dict(zip(param_names, params_tuple))
        
        # Evaluar modelo
        metric = eval_func(params_dict)
        history.append((params_dict.copy(), metric))
        
        # Actualizar mejor configuración
        if metric > best_metric:
            best_metric = metric
            best_params = params_dict.copy()
        
        print(f"Iteración {i}: {params_dict} -> Métrica: {metric:.4f}")
    
    print(f"\n=== Mejor configuración encontrada ===")
    print(f"Parámetros: {best_params}")
    print(f"Métrica: {best_metric:.4f}")
    
    return best_params, best_metric, history


# Función auxiliar para pruebas
if __name__ == "__main__":
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    # Probar con cada modelo
    models = ['svm', 'rf', 'mlp']
    
    for model in models:
        print(f"\n{'='*60}")
        best_params, best_metric, history = random_search_optimize(model, n_iterations=15)
        print(f"Modelo {model.upper()}: Mejor métrica = {best_metric:.4f}")
