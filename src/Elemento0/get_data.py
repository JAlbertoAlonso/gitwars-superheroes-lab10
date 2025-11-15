# src/Elemento0/get_data.py
import requests
import pandas as pd
import numpy as np
import json
import os

def convert_to_cm(height_list):
    """
    Convierte la altura a centímetros.
    La altura viene como lista: ["6'8", "203 cm"] o ["-"]
    """
    if not height_list or height_list == ["-"]:
        return np.nan
    
    # Tomar el primer elemento que no sea "-"
    for height_str in height_list:
        if height_str != "-":
            height_str = str(height_str).strip()
            
            # Si ya está en cm
            if "cm" in height_str.lower():
                try:
                    return float(height_str.lower().replace("cm", "").strip())
                except:
                    continue
            
            # Si está en pies y pulgadas (formato: "6'8"")
            if "'" in height_str:
                try:
                    # Formato: "6'8""
                    parts = height_str.split("'")
                    feet = float(parts[0])
                    inches_str = parts[1].replace('"', '').strip()
                    inches = float(inches_str) if inches_str else 0
                    return (feet * 30.48) + (inches * 2.54)
                except:
                    continue
            
            # Intentar convertir directamente
            try:
                return float(height_str)
            except:
                continue
    
    return np.nan

def convert_to_kg(weight_list):
    """
    Convierte el peso a kilogramos.
    El peso viene como lista: ["980 lb", "443 kg"] o ["-"]
    """
    if not weight_list or weight_list == ["-"]:
        return np.nan
    
    # Tomar el primer elemento que no sea "-"
    for weight_str in weight_list:
        if weight_str != "-":
            weight_str = str(weight_str).strip()
            
            # Si ya está en kg
            if "kg" in weight_str.lower():
                try:
                    return float(weight_str.lower().replace("kg", "").strip())
                except:
                    continue
            
            # Si está en libras
            if "lb" in weight_str.lower():
                try:
                    lbs = float(weight_str.lower().replace("lb", "").strip())
                    return lbs * 0.453592
                except:
                    continue
            
            # Intentar convertir directamente
            try:
                return float(weight_str)
            except:
                continue
    
    return np.nan

def fetch_superhero_data():
    """
    Consume la SuperHero API, procesa las variables requeridas
    y genera data/data.csv con el dataset final.
    """
    print("Consumiendo SuperHero API...")
    
    # URL de la API
    url = "https://akabab.github.io/superhero-api/api/all.json"
    
    try:
        # Hacer la petición a la API
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        print(f"Se obtuvieron {len(data)} superhéroes de la API")
        
    except requests.exceptions.RequestException as e:
        print(f"Error al consumir la API: {e}")
        return
    
    processed_data = []
    
    for hero in data:
        try:
            powerstats = hero.get('powerstats', {})
            appearance = hero.get('appearance', {})
            
            # Obtener height y weight como listas (formato original de la API)
            height_list = appearance.get('height', ["-"])
            weight_list = appearance.get('weight', ["-"])
            
            # Convertir unidades
            height_cm = convert_to_cm(height_list)
            weight_kg = convert_to_kg(weight_list)
            
            hero_data = {
                'intelligence': powerstats.get('intelligence', np.nan),
                'strength': powerstats.get('strength', np.nan),
                'speed': powerstats.get('speed', np.nan),
                'durability': powerstats.get('durability', np.nan),
                'combat': powerstats.get('combat', np.nan),
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'power': powerstats.get('power', np.nan)
            }
            
            processed_data.append(hero_data)
                    
        except Exception as e:
            print(f"Error procesando héroe {hero.get('name', 'Unknown')}: {e}")
            continue
    
    print(f"Se procesaron {len(processed_data)} héroes")
    
    df = pd.DataFrame(processed_data)
    print(f"Dataset final con {len(df)} registros")
    
    # Calcular medias para reemplazar NaN
    height_mean = df['height_cm'].mean()
    weight_mean = df['weight_kg'].mean()
    
    print(f"\nMedias calculadas:")
    print(f"Altura media: {height_mean:.2f} cm")
    print(f"Peso medio: {weight_mean:.2f} kg")
    
    # Reemplazar NaN con las medias
    df['height_cm'].fillna(height_mean, inplace=True)
    df['weight_kg'].fillna(weight_mean, inplace=True)
    
    # Crear directorio data si no existe
    os.makedirs('data', exist_ok=True)
    
    # Guardar el dataset
    output_path = 'data/data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset guardado en: {output_path}")
    
    # Mostrar información del dataset
    print("\nInformación del dataset:")
    print(f"Columnas: {list(df.columns)}")
    print(f"Forma: {df.shape}")
    
    # Mostrar valores nulos por columna (deberían ser 0 ahora)
    print("\nValores nulos por columna (después de reemplazar con media):")
    null_counts = df.isnull().sum()
    for col, null_count in null_counts.items():
        print(f"  {col}: {null_count} nulos")
    
    print(f"Total de valores nulos: {df.isnull().sum().sum()}")
    
    print("\nPrimeras 10 filas:")
    print(df.head(10))
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Mostrar algunos ejemplos de conversión
    print("\nEjemplos de conversión:")
    sample_heroes = data[:5]
    for i, hero in enumerate(sample_heroes):
        name = hero.get('name', 'Unknown')
        height_orig = hero.get('appearance', {}).get('height', ['-'])
        weight_orig = hero.get('appearance', {}).get('weight', ['-'])
        height_cm = df.iloc[i]['height_cm']
        weight_kg = df.iloc[i]['weight_kg']
        
        print(f"{name}:")
        print(f"  Altura original: {height_orig} -> {height_cm:.2f} cm")
        print(f"  Peso original: {weight_orig} -> {weight_kg:.2f} kg")

if __name__ == "__main__":
    fetch_superhero_data()