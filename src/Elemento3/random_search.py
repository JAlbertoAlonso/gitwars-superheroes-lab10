import numpy as np
from itertools import product

from src.orchestrator import (
    evaluate_svm,
    evaluate_rf,
    evaluate_mlp
)


# Espacios discretos para cada modelo


def get_domain(model_name):
    if model_name == "svm":
        C = [0.1, 1, 10, 100]
        gamma = [0.001, 0.01, 0.1, 1]
        return list(product(C, gamma))

    if model_name == "rf":
        n_estimators = [10, 20, 50, 100]
        max_depth = [2, 4, 6, 8]
        return list(product(n_estimators, max_depth))

    if model_name == "mlp":
        hidden = [(16,), (32,), (64,), (32, 16)]
        alpha = [1e-4, 1e-3, 1e-2]
        return list(product(hidden, alpha))

    raise ValueError("Modelo no reconocido")



# Llamar al evaluador correcto


def eval_model(model_name, params):
    if model_name == "svm":
        C, gamma = params
        return evaluate_svm(C, gamma)

    if model_name == "rf":
        n_est, max_d = params
        return evaluate_rf(n_est, max_d)

    if model_name == "mlp":
        hl, alpha = params
        return evaluate_mlp(hl, alpha)

    raise ValueError("Modelo no reconocido")



# Random Search


def random_search(model_name, n_iter=20):
    domain = get_domain(model_name)

    best_score = None
    best_params = None

    for _ in range(n_iter):
        # Elegir parámetros aleatorios
        x = domain[np.random.randint(len(domain))]

        # Evaluar
        score = eval_model(model_name, x)

        # Guardar el mejor (aquí menor RMSE)
        if (best_score is None) or (score < best_score):
            best_score = score
            best_params = x

    return best_params, best_score
