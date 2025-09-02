import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

WEATHER_BASE_URL = "http://api.weatherapi.com/v1/current.json"
DISTANCE_BASE_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

def get_weather(city="New York"):
    params = {"key": WEATHER_API_KEY, "q": city}
    response = requests.get(WEATHER_BASE_URL, params=params)
    data = response.json()
    if "current" not in data:
        raise Exception(f"Weather API error: {data}")
    current = data["current"]
    return {
        "temp": current["temp_c"],
        "clouds": current["cloud"],
        "pressure": current["pressure_mb"],
        "rain": current.get("precip_mm", 0),
        "humidity": current["humidity"],
        "wind": current["wind_kph"] / 3.6
    }

def get_distance(source, destination):
    params = {"origins": source, "destinations": destination, "key": GOOGLE_API_KEY}
    response = requests.get(DISTANCE_BASE_URL, params=params)
    data = response.json()
    if "rows" not in data or len(data["rows"]) == 0:
        raise Exception(f"Distance API error: {data}")
    distance_meters = data["rows"][0]["elements"][0]["distance"]["value"]
    return distance_meters / 1000.0
