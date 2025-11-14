#!/usr/bin/env python3
"""
Script principal para ejecutar el análisis comparativo completo.

Ejecuta:
1. Optimización Bayesiana para todos los modelos
2. Random Search para todos los modelos
3. Generación de tablas comparativas
4. Generación de visualizaciones
5. Generación de reporte interpretativo
"""

import sys
import os
import numpy as np

# Agregar directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..','..','src'))

def main():
    """Función principal que ejecuta el análisis completo."""
    
    print("\n" + "="*80)
    print(" "*20 + "ANÁLISIS COMPARATIVO COMPLETO")
    print(" "*15 + "Bayesian Optimization vs Random Search")
    print("="*80)
    
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    # Paso 1: Ejecutar análisis comparativo
    print("\n📊 PASO 1/2: Ejecutando análisis comparativo...")
    print("-"*80)
    
    try:
        from Elemento3.comparative_analysis import (
            run_comparative_analysis,
            create_comparison_table,
            plot_comparison_metrics,
            generate_analysis_report
        )
        
        # Ejecutar análisis
        results_df, bo_histories = run_comparative_analysis()
        
        # Crear tabla
        create_comparison_table(results_df)
        
        # Crear gráfica de comparación
        plot_comparison_metrics(results_df)
        
        # Generar reporte
        generate_analysis_report(results_df)
        
    except Exception as e:
        print(f"\n❌ Error en análisis comparativo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Paso 2: Generar gráficas de evolución
    print("\n📈 PASO 2/2: Generando gráficas de evolución de BO...")
    print("-"*80)
    
    try:
        from Elemento3.plot_bo_evolution import plot_all_models_evolution
        
        plot_all_models_evolution(n_init=3, n_iter=10)
        
    except Exception as e:
        print(f"\n❌ Error en generación de gráficas: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Resumen final
    print("\n" + "="*80)
    print(" "*25 + "✅ ANÁLISIS COMPLETADO")
    print("="*80)
    
    print("\n📁 Archivos generados en el directorio 'results/':")
    print("   • comparison_table.csv ........... Tabla comparativa de resultados")
    print("   • comparison_plot.png ............ Gráfica de comparación de métricas")
    print("   • analysis_report.md ............. Reporte de análisis interpretativo")
    print("   • bo_evolution_svm.png ........... Evolución de BO para SVM")
    print("   • bo_evolution_rf.png ............ Evolución de BO para Random Forest")
    print("   • bo_evolution_mlp.png ........... Evolución de BO para MLP")
    print("   • bo_evolution_combined.png ...... Comparación de convergencia")
    print("   • README.md ...................... Documentación del análisis")
    
    print("\n💡 Próximos pasos:")
    print("   1. Revisar los resultados en 'results/analysis_report.md'")
    print("   2. Examinar las visualizaciones generadas")
    print("   3. Incluir estos archivos en tu PR")
    
    print("\n" + "="*80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
