import requests
import json

# Obtener datos de la API
api_url = "https://akabab.github.io/superhero-api/api/all.json"
response = requests.get(api_url)
data = response.json()

# Mostrar la estructura del primer héroe
print("Estructura del primer superhéroe:")
print(json.dumps(data[0], indent=2))
