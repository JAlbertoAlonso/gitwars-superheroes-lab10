"""
Script de validación para el dataset generado
Verifica que cumple todos los criterios del Elemento 0
"""

import pandas as pd
import os


def validate_dataset():
    """
    Valida que el dataset cumple con todos los requisitos:
    - Archivo existe en data/data.csv
    - Contiene exactamente 600 registros
    - Tiene las 8 columnas correctas
    - No hay valores faltantes
    - Todas las columnas son numéricas
    """
    print("=" * 60)
    print("VALIDACIÓN DEL DATASET - ELEMENTO 0")
    print("=" * 60)

    # 1. Verificar que el archivo existe
    file_path = 'data/data.csv'
    if not os.path.exists(file_path):
        print(f"❌ ERROR: El archivo {file_path} no existe")
        return False

    print(f"[OK] Archivo {file_path} encontrado")

    # 2. Leer el dataset
    df = pd.read_csv(file_path)

    # 3. Verificar número de registros
    num_records = len(df)
    if num_records == 600:
        print(f"✓ Número de registros: {num_records} (CORRECTO)")
    else:
        print(f"❌ Número de registros: {num_records} (SE REQUIEREN 600)")
        return False

    # 4. Verificar columnas
    expected_columns = ['intelligence', 'strength', 'speed', 'durability',
                       'combat', 'height_cm', 'weight_kg', 'power']

    if list(df.columns) == expected_columns:
        print(f"✓ Columnas correctas: {list(df.columns)}")
    else:
        print(f"❌ Columnas incorrectas")
        print(f"   Esperadas: {expected_columns}")
        print(f"   Obtenidas: {list(df.columns)}")
        return False

    # 5. Verificar valores faltantes
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print(f"✓ No hay valores faltantes")
    else:
        print(f"❌ Hay {missing_values} valores faltantes:")
        print(df.isnull().sum())
        return False

    # 6. Verificar que todas las columnas son numéricas
    non_numeric = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    if len(non_numeric) == 0:
        print(f"✓ Todas las columnas son numéricas")
    else:
        print(f"❌ Columnas no numéricas: {non_numeric}")
        return False

    # 7. Mostrar estadísticas básicas
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DEL DATASET")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("VALIDACIÓN EXITOSA ✓")
    print("=" * 60)
    print("El dataset cumple con todos los requisitos del Elemento 0:")
    print("  ✓ 600 registros")
    print("  ✓ 8 columnas correctas")
    print("  ✓ Sin valores faltantes")
    print("  ✓ Todas las columnas numéricas")
    print("=" * 60)

    return True


if __name__ == "__main__":
    validate_dataset()
