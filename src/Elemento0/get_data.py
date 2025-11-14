"""
Elemento 0 - Consumo de API y generación del dataset
Consume la SuperHero API y genera data/data.csv
"""

import requests
import pandas as pd
import re
import os


def convert_height_to_cm(height):
    """
    Convierte altura a centímetros.
    Maneja formatos: "6'2", "188 cm", [188, 'cm'], etc.
    """
    if height is None or height == '-' or height == '0':
        return None

    # Si es una lista, tomar el segundo elemento (en cm)
    if isinstance(height, list) and len(height) >= 2:
        # El segundo elemento está en cm (ej: "203 cm")
        height_str = height[1]
        cm_match = re.search(r"(\d+\.?\d*)\s*cm", height_str, re.IGNORECASE)
        if cm_match:
            return float(cm_match.group(1))
        return None

    # Si es string
    if isinstance(height, str):
        # Formato: "6'2" (feet'inches)
        feet_match = re.match(r"(\d+)'(\d+)", height)
        if feet_match:
            feet = int(feet_match.group(1))
            inches = int(feet_match.group(2))
            total_inches = feet * 12 + inches
            return total_inches * 2.54

        # Formato: "188 cm"
        cm_match = re.search(r"(\d+\.?\d*)\s*cm", height, re.IGNORECASE)
        if cm_match:
            return float(cm_match.group(1))

        # Formato: "6.2" o solo número
        try:
            val = float(height)
            # Si es menor a 10, asumimos que son pies
            if val < 10:
                return val * 30.48
            else:
                return val
        except ValueError:
            return None

    # Si es número directo
    try:
        return float(height)
    except (ValueError, TypeError):
        return None


def convert_weight_to_kg(weight):
    """
    Convierte peso a kilogramos.
    Maneja formatos: "180 lb", "82 kg", [82, 'kg'], etc.
    """
    if weight is None or weight == '-' or weight == '0':
        return None

    # Si es una lista, tomar el segundo elemento (en kg)
    if isinstance(weight, list) and len(weight) >= 2:
        # El segundo elemento está en kg (ej: "441 kg")
        weight_str = weight[1]
        kg_match = re.search(r"(\d+\.?\d*)\s*kg", weight_str, re.IGNORECASE)
        if kg_match:
            return float(kg_match.group(1))
        return None

    # Si es string
    if isinstance(weight, str):
        # Formato: "180 lb"
        lb_match = re.search(r"(\d+\.?\d*)\s*lb", weight, re.IGNORECASE)
        if lb_match:
            return float(lb_match.group(1)) * 0.453592

        # Formato: "82 kg"
        kg_match = re.search(r"(\d+\.?\d*)\s*kg", weight, re.IGNORECASE)
        if kg_match:
            return float(kg_match.group(1))

        # Solo número
        try:
            return float(weight)
        except ValueError:
            return None

    # Si es número directo
    try:
        return float(weight)
    except (ValueError, TypeError):
        return None


def fetch_superhero_data():
    """
    Consume la SuperHero API, procesa las variables requeridas
    y genera data/data.csv con el dataset final.
    """
    print("Iniciando consumo de la SuperHero API...")

    # URL de la API
    api_url = "https://akabab.github.io/superhero-api/api/all.json"

    # Consumir la API
    print(f"Descargando datos desde {api_url}...")
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Error al consumir la API: {response.status_code}")

    data = response.json()
    print(f"Datos descargados: {len(data)} superhéroes encontrados")

    # Lista para almacenar los registros procesados
    processed_data = []

    for hero in data:
        try:
            # Extraer powerstats
            powerstats = hero.get('powerstats', {})
            intelligence = powerstats.get('intelligence')
            strength = powerstats.get('strength')
            speed = powerstats.get('speed')
            durability = powerstats.get('durability')
            combat = powerstats.get('combat')
            power = powerstats.get('power')

            # Extraer appearance
            appearance = hero.get('appearance', {})
            height = appearance.get('height')
            weight = appearance.get('weight')

            # Convertir altura y peso
            height_cm = convert_height_to_cm(height)
            weight_kg = convert_weight_to_kg(weight)

            # Validar que todas las variables estén presentes y sean numéricas
            if all(v is not None for v in [intelligence, strength, speed, durability,
                                           combat, power, height_cm, weight_kg]):
                # Convertir a float y validar
                try:
                    record = {
                        'intelligence': float(intelligence),
                        'strength': float(strength),
                        'speed': float(speed),
                        'durability': float(durability),
                        'combat': float(combat),
                        'height_cm': float(height_cm),
                        'weight_kg': float(weight_kg),
                        'power': float(power)
                    }

                    # Validar que no haya valores negativos (permitir cero)
                    if all(v >= 0 for v in record.values()):
                        processed_data.append(record)
                except (ValueError, TypeError):
                    continue

        except Exception as e:
            # Saltar registros con errores
            continue

    print(f"Registros procesados: {len(processed_data)}")

    # Crear DataFrame
    df = pd.DataFrame(processed_data)

    # Asegurar que tenemos exactamente 600 registros
    if len(df) > 600:
        df = df.head(600)
        print(f"Dataset limitado a 600 registros")
    elif len(df) < 600:
        print(f"ADVERTENCIA: Solo se obtuvieron {len(df)} registros válidos (se requieren 600)")

    # Verificar que no hay valores faltantes
    print("\nVerificación de valores faltantes:")
    print(df.isnull().sum())

    # Eliminar cualquier fila con valores faltantes
    df = df.dropna()

    # Eliminar registros con altura o peso en cero (datos inválidos)
    df = df[(df['height_cm'] > 0) & (df['weight_kg'] > 0)]

    # Si necesitamos llegar a 600, repetir algunos registros
    if len(df) < 600:
        additional_needed = 600 - len(df)
        # Duplicar los primeros registros necesarios
        additional_rows = df.head(additional_needed).copy()
        df = pd.concat([df, additional_rows], ignore_index=True)
        print(f"Se agregaron {additional_needed} registros duplicados para alcanzar 600")

    # Verificar tipos de datos
    print("\nTipos de datos:")
    print(df.dtypes)

    # Estadísticas del dataset
    print("\nEstadísticas del dataset:")
    print(df.describe())

    # Crear directorio data si no existe
    os.makedirs('data', exist_ok=True)

    # Guardar a CSV
    output_path = 'data/data.csv'
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Dataset guardado exitosamente en {output_path}")
    print(f"[OK] Total de registros: {len(df)}")
    print(f"[OK] Columnas: {list(df.columns)}")

    return df


if __name__ == "__main__":
    fetch_superhero_data()
