# src/random_search.py
import numpy as np
import pandas as pd
from orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp
from optimizer import get_search_domain

def random_search(model_name, n_iter=10):
    """
    Implementa Random Search para comparar con BO
    """
    print(f"🎲 Ejecutando Random Search para {model_name.upper()}...")
    
    domain = get_search_domain(model_name)
    best_score = -1
    best_params = None
    history = []
    
    # Realizar búsqueda aleatoria
    for i in range(min(n_iter, len(domain))):
        # Seleccionar punto aleatorio no repetido
        available_indices = [idx for idx in range(len(domain)) if idx not in [h[0] for h in history]]
        
        if not available_indices:
            break
            
        idx = np.random.choice(available_indices)
        domain_point = domain[idx]
        params = domain_point['params']
        
        # Evaluar modelo
        if model_name == "svm":
            score = evaluate_svm(C=params[0], gamma=params[1])
        elif model_name == "rf":
            score = evaluate_rf(n_estimators=params[0], max_depth=params[1])
        elif model_name == "mlp":
            score = evaluate_mlp(hidden_layer_sizes=params[0], alpha=params[1])
        
        history.append((idx, params, score))
        
        # Actualizar mejor
        if score > best_score:
            best_score = score
            best_params = params
            
        print(f"  Iteración {i+1}: {params} -> R² = {score:.4f}")
    
    print(f"✅ Random Search completado - Mejor R²: {best_score:.4f}")
    return best_params, best_score, history