import requests
import pandas as pd
import re


def convert_height_to_cm(height_list):
    """
    Convert height to centimeters from various formats.
    Expected format: [feet'inches, cm]
    """
    if not height_list or len(height_list) < 2:
        return None
    
    # Try to get cm value directly
    cm_value = height_list[1]
    if cm_value and cm_value != '-':
        try:
            # Remove ' cm' or any text, extract number
            cm_str = re.sub(r'[^\d.]', '', str(cm_value))
            if cm_str:
                return float(cm_str)
        except (ValueError, TypeError):
            pass
    
    # Try to convert from feet and inches
    feet_inches = height_list[0]
    if feet_inches and feet_inches != '-':
        try:
            # Parse format like "5'10" or "6'2"
            match = re.match(r"(\d+)'(\d+)", str(feet_inches))
            if match:
                feet = int(match.group(1))
                inches = int(match.group(2))
                return round(feet * 30.48 + inches * 2.54, 2)
        except (ValueError, TypeError, AttributeError):
            pass
    
    return None


def convert_weight_to_kg(weight_list):
    """
    Convert weight to kilograms from various formats.
    Expected format: [lbs, kg]
    """
    if not weight_list or len(weight_list) < 2:
        return None
    
    # Try to get kg value directly
    kg_value = weight_list[1]
    if kg_value and kg_value != '-':
        try:
            # Remove ' kg' or any text, extract number
            kg_str = re.sub(r'[^\d.]', '', str(kg_value))
            if kg_str:
                return float(kg_str)
        except (ValueError, TypeError):
            pass
    
    # Try to convert from lbs
    lbs_value = weight_list[0]
    if lbs_value and lbs_value != '-':
        try:
            # Remove ' lb' or any text, extract number
            lbs_str = re.sub(r'[^\d.]', '', str(lbs_value))
            if lbs_str:
                return round(float(lbs_str) * 0.453592, 2)
        except (ValueError, TypeError):
            pass
    
    return None


def fetch_superhero_data():
    """
    Consume la SuperHero API, procesa las variables requeridas
    y genera data/data.csv con el dataset final.
    """
    print("Fetching data from SuperHero API...")
    
    # URL of the API
    api_url = "https://akabab.github.io/superhero-api/api/all.json"
    
    try:
        # Fetch data from API
        response = requests.get(api_url)
        response.raise_for_status()
        heroes = response.json()
        
        print(f"Fetched {len(heroes)} heroes from API")
        
        # Extract and process data
        processed_data = []
        
        for hero in heroes:
            try:
                # Extract powerstats
                powerstats = hero.get('powerstats', {})
                intelligence = powerstats.get('intelligence')
                strength = powerstats.get('strength')
                speed = powerstats.get('speed')
                durability = powerstats.get('durability')
                combat = powerstats.get('combat')
                
                # Extract appearance
                appearance = hero.get('appearance', {})
                height = appearance.get('height', [])
                weight = appearance.get('weight', [])
                
                # Convert height and weight (if conversion fails, use 0)
                height_cm = convert_height_to_cm(height)
                if height_cm is None:
                    height_cm = 0.0
                    
                weight_kg = convert_weight_to_kg(weight)
                if weight_kg is None:
                    weight_kg = 0.0
                
                # Extract power
                power = hero.get('powerstats', {}).get('power')
                
                # Check if all required powerstats fields are present (not None)
                if all(v is not None for v in [
                    intelligence, strength, speed, durability, combat, power
                ]):
                    processed_data.append({
                        'intelligence': int(intelligence),
                        'strength': int(strength),
                        'speed': int(speed),
                        'durability': int(durability),
                        'combat': int(combat),
                        'height_cm': float(height_cm),
                        'weight_kg': float(weight_kg),
                        'power': int(power)
                    })
            
            except (ValueError, TypeError, KeyError) as e:
                # Skip heroes with invalid data
                continue
        
        print(f"Processed {len(processed_data)} valid records")
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Remove any remaining rows with missing values
        df = df.dropna()
        
        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows that became NaN after conversion
        df = df.dropna()
        
        # Limit to exactly 600 records if we have more
        if len(df) > 600:
            df = df.head(600)
            print(f"Limited dataset to 600 records")
        elif len(df) < 600:
            print(f"Warning: Only {len(df)} valid records found (expected 600)")
            print(f"Note: The API does not provide 600 complete records.")
        
        # Save to CSV in the data folder at repository root
        output_path = "data/data.csv"
        df.to_csv(output_path, index=False)
        
        print(f"\nDataset successfully created!")
        print(f"Output file: {output_path}")
        print(f"Total records: {len(df)}")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        print(f"\nDataset statistics:")
        print(df.describe())
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        raise
    except Exception as e:
        print(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    fetch_superhero_data()
