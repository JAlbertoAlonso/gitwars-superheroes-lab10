import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score

# Cargar datos globalmente
def load_data():
    """
    Carga el dataset de superhéroes y prepara X, y para entrenamiento.
    """
    df = pd.read_csv('../data/data.csv')
    
    # Features: todas las columnas excepto 'power' (variable objetivo)
    X = df.drop('power', axis=1).values
    
    # Target: clasificación binaria basada en la mediana de 'power'
    # power >= mediana -> clase 1 (alto poder)
    # power < mediana -> clase 0 (bajo poder)
    median_power = df['power'].median()
    y = (df['power'] >= median_power).astype(int).values
    
    return X, y

# Cargar datos una sola vez
X_global, y_global = load_data()

def evaluate_svm(params):
    """
    Entrena y evalúa SVM con los parámetros dados.
    
    Args:
        params: dict con 'C' y 'gamma'
    
    Returns:
        f1_macro: métrica F1 macro (float)
    """
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_global, y_global, test_size=0.2, random_state=42, stratify=y_global
    )
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = SVC(C=params['C'], gamma=params['gamma'], random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = model.predict(X_test_scaled)
    
    # Calcular F1 macro
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    return f1_macro

def evaluate_rf(params):
    """
    Entrena y evalúa Random Forest con los parámetros dados.
    
    Args:
        params: dict con 'n_estimators' y 'max_depth'
    
    Returns:
        f1_macro: métrica F1 macro (float)
    """
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_global, y_global, test_size=0.2, random_state=42, stratify=y_global
    )
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = model.predict(X_test_scaled)
    
    # Calcular F1 macro
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    return f1_macro

def evaluate_mlp(params):
    """
    Entrena y evalúa MLP con los parámetros dados.
    
    Args:
        params: dict con 'hidden_layer_sizes' y 'alpha'
    
    Returns:
        f1_macro: métrica F1 macro (float)
    """
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_global, y_global, test_size=0.2, random_state=42, stratify=y_global
    )
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        alpha=params['alpha'],
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = model.predict(X_test_scaled)
    
    # Calcular F1 macro
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    return f1_macro

# Función de prueba
if __name__ == "__main__":
    print("Probando funciones de evaluación...")
    
    # Probar SVM
    svm_params = {'C': 1.0, 'gamma': 0.01}
    svm_score = evaluate_svm(svm_params)
    print(f"\nSVM con {svm_params}: F1-macro = {svm_score:.4f}")
    
    # Probar Random Forest
    rf_params = {'n_estimators': 50, 'max_depth': 4}
    rf_score = evaluate_rf(rf_params)
    print(f"Random Forest con {rf_params}: F1-macro = {rf_score:.4f}")
    
    # Probar MLP
    mlp_params = {'hidden_layer_sizes': (32,), 'alpha': 0.001}
    mlp_score = evaluate_mlp(mlp_params)
    print(f"MLP con {mlp_params}: F1-macro = {mlp_score:.4f}")