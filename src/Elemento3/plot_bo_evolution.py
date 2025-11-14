import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optimizer import optimize_model

def plot_bo_evolution(model_name, n_init=3, n_iter=10, output_path=None):
    """
    Genera visualización de la evolución de Bayesian Optimization.
    
    Args:
        model_name: 'svm', 'rf', o 'mlp'
        n_init: Número de puntos iniciales
        n_iter: Número de iteraciones de BO
        output_path: Ruta donde guardar la figura
    """
    # Ejecutar BO con historial
    print(f"\nGenerando datos de evolución para {model_name.upper()}...")
    best_params, best_metric, history = optimize_model(
        model_name, 
        n_init=n_init, 
        n_iter=n_iter,
        return_history=True
    )
    
    # Extraer datos del historial
    iterations = [h['iteration'] for h in history]
    metrics = [h['metric'] for h in history]
    best_so_far = [h['best_so_far'] for h in history]
    phases = [h['phase'] for h in history]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- Subplot 1: Evolución de métricas ---
    ax1 = axes[0]
    
    # Separar fases de inicialización y BO
    init_mask = np.array([p == 'initialization' for p in phases])
    bo_mask = ~init_mask
    
    # Plotear puntos de inicialización
    ax1.scatter(
        np.array(iterations)[init_mask], 
        np.array(metrics)[init_mask],
        c='#E63946', 
        s=100, 
        alpha=0.7, 
        label='Inicialización (Random)',
        marker='o',
        edgecolors='black',
        linewidths=1.5
    )
    
    # Plotear puntos de BO
    ax1.scatter(
        np.array(iterations)[bo_mask], 
        np.array(metrics)[bo_mask],
        c='#2E86AB', 
        s=100, 
        alpha=0.7, 
        label='Bayesian Optimization',
        marker='s',
        edgecolors='black',
        linewidths=1.5
    )
    
    # Línea de mejor valor acumulado
    ax1.plot(
        iterations, 
        best_so_far, 
        'g-', 
        linewidth=2.5, 
        label='Mejor valor acumulado',
        marker='D',
        markersize=6,
        alpha=0.8
    )
    
    ax1.set_xlabel('Iteración', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Macro Score', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'Evolución de Optimización Bayesiana - {model_name.upper()}\nConvergencia hacia el óptimo',
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir línea vertical separando fases
    if n_init > 0:
        ax1.axvline(x=n_init, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax1.text(n_init, ax1.get_ylim()[0], '  Fase BO →', 
                rotation=0, va='bottom', ha='left', color='red', fontweight='bold')
    
    # --- Subplot 2: Mejora acumulada ---
    ax2 = axes[1]
    
    # Calcular mejora porcentual respecto al primer valor
    initial_metric = metrics[0]
    improvement = [(m - initial_metric) / initial_metric * 100 for m in best_so_far]
    
    ax2.plot(
        iterations, 
        improvement, 
        'o-', 
        color='#06A77D', 
        linewidth=2.5,
        markersize=8,
        markeredgecolor='black',
        markeredgewidth=1.5,
        alpha=0.8
    )
    
    # Rellenar área bajo la curva
    ax2.fill_between(iterations, 0, improvement, alpha=0.3, color='#06A77D')
    
    ax2.set_xlabel('Iteración', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mejora respecto a inicio (%)', fontsize=12, fontweight='bold')
    ax2.set_title(
        'Mejora Acumulada durante el Proceso de Optimización',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Añadir línea vertical separando fases
    if n_init > 0:
        ax2.axvline(x=n_init, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Añadir anotación del mejor valor
    best_iter = iterations[np.argmax(metrics)]
    best_val = max(metrics)
    ax1.annotate(
        f'Mejor: {best_val:.4f}',
        xy=(best_iter, best_val),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'),
        fontsize=10,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Guardar figura
    if output_path is None:
        output_path = f'../results/bo_evolution_{model_name}.png'
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada en: {output_path}")
    
    plt.close()
    
    return history


def plot_all_models_evolution(n_init=3, n_iter=10):
    """
    Genera gráficas de evolución para todos los modelos.
    """
    models = ['svm', 'rf', 'mlp']
    model_names = {
        'svm': 'SVM',
        'rf': 'Random Forest',
        'mlp': 'MLP'
    }
    
    print("\n" + "="*70)
    print("GENERACIÓN DE GRÁFICAS DE EVOLUCIÓN - BAYESIAN OPTIMIZATION")
    print("="*70)
    
    histories = {}
    for model in models:
        print(f"\n{'#'*70}")
        print(f"# MODELO: {model_names[model]}")
        print(f"{'#'*70}")
        
        history = plot_bo_evolution(model, n_init=n_init, n_iter=n_iter)
        histories[model] = history
    
    # Crear gráfica comparativa de todas las evoluciones
    create_combined_evolution_plot(histories)
    
    print("\n" + "="*70)
    print("✅ Todas las gráficas generadas exitosamente")
    print("="*70)
    
    return histories


def create_combined_evolution_plot(histories, output_path='../results/bo_evolution_combined.png'):
    """
    Crea una gráfica combinada mostrando la evolución de todos los modelos.
    
    Args:
        histories: Diccionario con historiales de cada modelo
        output_path: Ruta donde guardar la figura
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = {
        'svm': 'SVM',
        'rf': 'Random Forest',
        'mlp': 'MLP'
    }
    
    colors = {
        'svm': '#E63946',
        'rf': '#2E86AB',
        'mlp': '#06A77D'
    }
    
    for model, history in histories.items():
        iterations = [h['iteration'] for h in history]
        best_so_far = [h['best_so_far'] for h in history]
        
        ax.plot(
            iterations,
            best_so_far,
            'o-',
            color=colors[model],
            linewidth=2.5,
            markersize=6,
            label=model_names[model],
            alpha=0.8
        )
    
    ax.set_xlabel('Iteración', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mejor F1-Macro Score', fontsize=12, fontweight='bold')
    ax.set_title(
        'Comparación de Convergencia: Optimización Bayesiana\nTodos los Modelos',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfica combinada guardada en: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    # Fijar semilla
    np.random.seed(42)
    
    # Generar todas las gráficas
    plot_all_models_evolution(n_init=3, n_iter=10)
