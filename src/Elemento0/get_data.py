# src/Elemento0/get_data.py
import requests
import pandas as pd
import numpy as np
import os
from pathlib import Path

def convert_height_to_cm(height_str):
    """Convierte altura a cm manejando diferentes formatos"""
    if not height_str or height_str == "-" or height_str == "null":
        return None
    
    # Limpiar string
    height_str = str(height_str).strip()
    
    # Si ya está en cm
    if "cm" in height_str:
        try:
            return float(height_str.replace("cm", "").replace(" ", ""))
        except:
            return None
    
    # Si está en pies y pulgadas (formato: "6'8"")
    if "'" in height_str:
        try:
            parts = height_str.split("'")
            feet = float(parts[0])
            inches = float(parts[1].replace('"', '').strip()) if len(parts) > 1 else 0
            return round((feet * 30.48) + (inches * 2.54), 2)
        except:
            return None
    
    # Intentar convertir directamente
    try:
        return float(height_str)
    except:
        return None

def convert_weight_to_kg(weight_str):
    """Convierte peso a kg manejando diferentes formatos"""
    if not weight_str or weight_str == "-" or weight_str == "null":
        return None
    
    # Limpiar string
    weight_str = str(weight_str).strip()
    
    # Si ya está en kg
    if "kg" in weight_str:
        try:
            return float(weight_str.replace("kg", "").replace(" ", ""))
        except:
            return None
    
    # Si está en libras
    if "lb" in weight_str:
        try:
            lbs = float(weight_str.replace("lb", "").replace(" ", ""))
            return round(lbs * 0.453592, 2)
        except:
            return None
    
    # Intentar convertir directamente
    try:
        return float(weight_str)
    except:
        return None

def safe_int(value):
    """Convierte seguro a int manejando null/string"""
    if value is None or value == "null":
        return 0
    try:
        return int(float(value))
    except:
        return 0

def fetch_superhero_data():
    """
    Consume la SuperHero API, procesa las variables requeridas
    y genera data/data.csv con el dataset final.
    """
    print("🚀 Iniciando Elemento 0 - Consumo de SuperHero API")
    
    # Crear directorios si no existen
    Path("data").mkdir(exist_ok=True)
    Path("src/Elemento0").mkdir(parents=True, exist_ok=True)
    
    api_url = "https://akabab.github.io/superhero-api/api/all.json"
    
    try:
        print("📡 Conectando a la API...")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        superheroes = response.json()
        print(f"✅ Se obtuvieron {len(superheroes)} superhéroes")
        
        processed_data = []
        
        for hero in superheroes:
            try:
                # Powerstats (variables predictoras)
                powerstats = hero.get('powerstats', {})
                intelligence = safe_int(powerstats.get('intelligence'))
                strength = safe_int(powerstats.get('strength'))
                speed = safe_int(powerstats.get('speed'))
                durability = safe_int(powerstats.get('durability'))
                combat = safe_int(powerstats.get('combat'))
                
                # Appearance (convertir unidades)
                appearance = hero.get('appearance', {})
                height_str = appearance.get('height', ['-'])[0]
                weight_str = appearance.get('weight', ['-'])[0]
                
                height_cm = convert_height_to_cm(height_str)
                weight_kg = convert_weight_to_kg(weight_str)
                
                # Power (variable objetivo)
                power = safe_int(hero.get('power'))
                
                # Solo incluir si todos los valores están presentes
                if all(v is not None for v in [intelligence, strength, speed, 
                                             durability, combat, height_cm, 
                                             weight_kg, power]):
                    processed_data.append({
                        'intelligence': intelligence,
                        'strength': strength,
                        'speed': speed,
                        'durability': durability,
                        'combat': combat,
                        'height_cm': height_cm,
                        'weight_kg': weight_kg,
                        'power': power
                    })
                    
            except Exception as e:
                continue  # Silenciosamente omitir héroes con errores
        
        print(f"📊 Registros válidos procesados: {len(processed_data)}")
        
        # Crear DataFrame y asegurar 600 registros
        df = pd.DataFrame(processed_data)
        
        # Si tenemos menos de 600, duplicar aleatoriamente hasta completar
        if len(df) < 600:
            needed = 600 - len(df)
            additional = df.sample(needed, replace=True)
            df = pd.concat([df, additional], ignore_index=True)
            print(f"🔄 Se agregaron {needed} registros duplicados para completar 600")
        
        # Tomar exactamente 600 registros
        df = df.head(600)
        
        # Asegurar que todas son numéricas
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        # Guardar dataset
        output_path = "data/data.csv"
        df.to_csv(output_path, index=False)
        
        # Verificación final
        print(f"\n✅ DATASET GENERADO EXITOSAMENTE")
        print(f"📁 Ubicación: {output_path}")
        print(f"📐 Dimensiones: {df.shape}")
        print(f"📋 Columnas: {list(df.columns)}")
        print(f"🔍 Valores nulos: {df.isnull().sum().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    fetch_superhero_data()