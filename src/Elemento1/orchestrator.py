"""
Orchestrator for evaluating ML models (SVM, Random Forest, MLP).
Each function trains a model and returns a performance metric (RMSE).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os


def load_data():
    """
    Load the dataset and split into features and target.
    Returns: X_train, X_test, y_train, y_test
    """
    # Try multiple possible paths for the data file
    possible_paths = [
        'data/data.csv',
        '../data/data.csv',
        'src/Elemento0/data/data.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv'),
        os.path.join(os.path.dirname(__file__), 'Elemento0', 'data', 'data.csv')
    ]
    
    df = None
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        except:
            continue
    
    if df is None:
        raise FileNotFoundError("Could not find data.csv in any expected location")
    
    # Separate features and target
    # Target is 'power', features are all other columns
    X = df.drop('power', axis=1)
    y = df['power']
    
    # Split the data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def evaluate_svm(C, gamma):
    """
    Debe entrenar un SVM con los hiperparámetros dados
    y devolver un float (RMSE).
    
    Args:
        C: Regularization parameter
        gamma: Kernel coefficient for 'rbf'
    
    Returns:
        float: RMSE score on test data
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM regressor
    model = SVR(C=C, gamma=gamma, kernel='rbf')
    model.fit(X_train_scaled, y_train)
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Normalize RMSE to [0, 1] range for consistency
    # Since power ranges from 0 to 100, we divide by 100
    normalized_rmse = rmse / 100.0
    
    return float(normalized_rmse)


def evaluate_rf(n_estimators, max_depth):
    """
    Entrenar un Random Forest y retornar un float
    con la métrica de desempeño (RMSE).
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
    
    Returns:
        float: RMSE score on test data
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train Random Forest regressor
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Normalize RMSE to [0, 1] range
    normalized_rmse = rmse / 100.0
    
    return float(normalized_rmse)


def evaluate_mlp(hidden_layer_sizes, alpha):
    """
    Entrenar un MLPClassifier y devolver la métrica final (RMSE).
    
    Args:
        hidden_layer_sizes: Tuple defining the number of neurons in each hidden layer
        alpha: L2 regularization parameter
    
    Returns:
        float: RMSE score on test data
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP regressor
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Normalize RMSE to [0, 1] range
    normalized_rmse = rmse / 100.0
    
    return float(normalized_rmse)


if __name__ == "__main__":
    # Test the functions with sample hyperparameters
    print("Testing orchestrator functions...")
    
    # Test SVM
    print("\n1. Testing SVM...")
    svm_score = evaluate_svm(C=1.0, gamma='scale')
    print(f"   SVM RMSE (normalized): {svm_score:.4f}")
    
    # Test Random Forest
    print("\n2. Testing Random Forest...")
    rf_score = evaluate_rf(n_estimators=100, max_depth=10)
    print(f"   RF RMSE (normalized): {rf_score:.4f}")
    
    # Test MLP
    print("\n3. Testing MLP...")
    mlp_score = evaluate_mlp(hidden_layer_sizes=(100, 50), alpha=0.0001)
    print(f"   MLP RMSE (normalized): {mlp_score:.4f}")
    
    print("\n✓ All functions executed successfully!")
