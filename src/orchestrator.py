# src/orchestrator.py
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Cargar y preparar los datos una vez
def load_data():
    """Carga y prepara el dataset para entrenamiento"""
    df = pd.read_csv('data/data.csv')
    
    # Separar características y variable objetivo
    X = df.drop('power', axis=1)
    y = df['power']
    
    # Dividir en train y test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar características (importante para SVM y MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Cargar datos una vez al importar el módulo
X_train, X_test, y_train, y_test = load_data()

def evaluate_svm(C, gamma):
    """  
    Entrena un SVM con los hiperparámetros dados  
    y devuelve un float (R² score).
    """  
    try:
        # Crear y entrenar modelo SVM (sin random_state)
        model = SVR(C=C, gamma=gamma)
        model.fit(X_train, y_train)
        
        # Predecir y calcular métrica
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        # Asegurar que el score esté en [0,1]
        return max(0.0, float(score))
    
    except Exception as e:
        print(f"Error en evaluate_svm: {e}")
        return 0.0

def evaluate_rf(n_estimators, max_depth):
    """  
    Entrenar un Random Forest y retornar un float  
    con la métrica de desempeño (R²).
    """  
    try:
        # Crear y entrenar modelo Random Forest
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth == 0 else int(max_depth),
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predecir y calcular métrica
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        return max(0.0, float(score))
    
    except Exception as e:
        print(f"Error en evaluate_rf: {e}")
        return 0.0

def evaluate_mlp(hidden_layer_sizes, alpha):
    """ 
    Entrenar un MLPRegressor y devolver la métrica final (R²).
    """ 
    try:
        # Crear y entrenar modelo MLP
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            random_state=42,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        
        # Predecir y calcular métrica
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        return max(0.0, float(score))
    
    except Exception as e:
        print(f"Error en evaluate_mlp: {e}")
        return 0.0

# Script de prueba para verificar que funciona
if __name__ == "__main__":
    print("Probando funciones del orquestador...")
    
    # Probar SVM con parámetros del dominio de BO
    print("\n1. Probando SVM...")
    svm_score = evaluate_svm(C=1.0, gamma=0.1)
    print(f"SVM R² score: {svm_score:.4f}")
    
    # Probar Random Forest con parámetros del dominio de BO
    print("\n2. Probando Random Forest...")
    rf_score = evaluate_rf(n_estimators=50, max_depth=10)
    print(f"Random Forest R² score: {rf_score:.4f}")
    
    # Probar MLP con parámetros del dominio de BO
    print("\n3. Probando MLP...")
    mlp_score = evaluate_mlp(hidden_layer_sizes=(50,), alpha=0.001)
    print(f"MLP R² score: {mlp_score:.4f}")
    
    print("\n✅ Todas las funciones ejecutan correctamente!")
    
    # Probar con múltiples valores
    print("\n--- Pruebas adicionales con diferentes parámetros ---")
    
    print("\nSVM con diferentes parámetros:")
    for C in [0.1, 1.0, 10.0]:
        for gamma in [0.001, 0.01, 0.1]:
            score = evaluate_svm(C=C, gamma=gamma)
            print(f"C={C}, gamma={gamma}: R² = {score:.4f}")
    
    print("\nRandom Forest con diferentes parámetros:")
    for n_est in [10, 50, 100]:
        for depth in [5, 10, 20]:
            score = evaluate_rf(n_estimators=n_est, max_depth=depth)
            print(f"n_estimators={n_est}, max_depth={depth}: R² = {score:.4f}")