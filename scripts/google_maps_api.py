import requests
import os
from dotenv import load_dotenv

# Load API key from .env file (recommended for security)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_route_info(origin: str, destination: str, departure_time="now"):
    """
    Fetch distance, duration, and traffic info from Google Maps Distance Matrix API.
    
    Args:
        origin (str): Starting location (e.g., "Bangalore, India").
        destination (str): Destination location (e.g., "Airport, Bangalore").
        departure_time (str): "now" for current traffic or timestamp (in seconds).
    
    Returns:
        dict: distance (km), duration (mins), duration_in_traffic (mins)
    """
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    params = {
        "origins": origin,
        "destinations": destination,
        "departure_time": departure_time,
        "traffic_model": "best_guess",
        "key": GOOGLE_API_KEY
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] != "OK":
        raise Exception(f"API Error: {data['status']}")
    
    element = data["rows"][0]["elements"][0]
    
    if element["status"] != "OK":
        raise Exception(f"Route Error: {element['status']}")

    result = {
        "distance_km": element["distance"]["value"] / 1000,  # meters → km
        "duration_mins": element["duration"]["value"] / 60,  # seconds → mins
        "duration_in_traffic_mins": element.get("duration_in_traffic", element["duration"])["value"] / 60
    }
    
    return result


if __name__ == "__main__":
    # Example run
    origin = "MG Road, Bangalore"
    destination = "Kempegowda International Airport, Bangalore"
    result = get_route_info(origin, destination)
    print(result)
