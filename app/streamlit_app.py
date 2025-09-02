import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from xgboost import XGBRegressor

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load model, scaler, and feature columns
model = pickle.load(open("model_artifacts/xgboost_uber_model.pkl", "rb"))
scaler = pickle.load(open("model_artifacts/scaler.pkl", "rb"))
feature_columns = pickle.load(open("model_artifacts/feature_columns.pkl", "rb"))

# Dropdown options (from your dataset)
CAB_TYPES = ["Uber", "Lyft"]
SOURCES = ["Back Bay", "Beacon Hill", "Boston University", "Fenway", "Financial District", 
           "Haymarket Square", "North End", "Northeastern University", "South Station", "Theatre District", "West End"]
DESTINATIONS = ["Beacon Hill", "Boston University", "Fenway", "Financial District", "Haymarket Square", 
                "North End", "Northeastern University", "South Station", "Theatre District", "West End"]

# Google Maps Distance API
def get_distance(source, destination):
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={source}&destinations={destination}&key={GOOGLE_API_KEY}"
    response = requests.get(url).json()
    try:
        distance_km = response["rows"][0]["elements"][0]["distance"]["value"] / 1000
        return round(distance_km, 2)
    except:
        return None

# Weather API (Boston by default)
def get_weather():
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q=Boston"
    response = requests.get(url).json()
    try:
        return {
            "temp": response["current"]["temp_c"],
            "clouds": response["current"]["cloud"],
            "pressure": response["current"]["pressure_mb"],
            "humidity": response["current"]["humidity"],
            "wind": response["current"]["wind_kph"],
            "rain": response["current"].get("precip_mm", 0)
        }
    except:
        return None

# Streamlit UI
st.title("🚖 Cab Price Predictor")

cab_type = st.selectbox("Select Cab Type", CAB_TYPES)
source = st.selectbox("Select Source", SOURCES)
destination = st.selectbox("Select Destination", DESTINATIONS)
surge = st.number_input("Enter Surge Multiplier", min_value=1.0, max_value=3.0, step=0.1)

if st.button("Predict Price"):
    distance = get_distance(source, destination)
    weather = get_weather()

    if distance is None or weather is None:
        st.error("Error fetching distance or weather. Check API keys.")
    else:
        # Build feature row
        input_data = {
            "distance": distance,
            "surge_multiplier": surge,
            "cab_type_" + cab_type: 1,
            "source_" + source: 1,
            "destination_" + destination: 1,
            "temp": weather["temp"],
            "clouds": weather["clouds"],
            "pressure": weather["pressure"],
            "humidity": weather["humidity"],
            "wind": weather["wind"],
            "rain": weather["rain"]
        }

        # Fill missing columns with 0
        row = pd.DataFrame(columns=feature_columns)
        row.loc[0] = 0
        for col, val in input_data.items():
            if col in row.columns:
                row.loc[0, col] = val

        # Scale numerical features
        row_scaled = scaler.transform(row)

        # Predict
        prediction = model.predict(row_scaled)[0]
        st.success(f"💰 Estimated Cab Price: ${prediction:.2f}")
