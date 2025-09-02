import numpy as np
import pandas as pd
from scripts.weather_api import get_weather, get_distance
import pickle

with open("model_artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model_artifacts/feature_columns.pkl", "rb") as f:
    FEATURE_COLUMNS = pickle.load(f)

def build_features(source, destination, cab_type, cab_name, timestamp, surge_multiplier=1.0):
    weather = get_weather(destination)
    distance = get_distance(source, destination)

    features = {col: 0 for col in FEATURE_COLUMNS}
    features["distance"] = distance
    features["surge_multiplier"] = surge_multiplier
    features["temp"] = weather["temp"]
    features["clouds"] = weather["clouds"]
    features["pressure"] = weather["pressure"]
    features["rain"] = weather["rain"]
    features["humidity"] = weather["humidity"]
    features["wind"] = weather["wind"]

    features[f"cab_type_{cab_type}"] = 1
    if f"name_{cab_name}" in features:
        features[f"name_{cab_name}"] = 1
    features[f"source_{source}"] = 1
    features[f"destination_{destination}"] = 1

    hour = timestamp.hour
    minute = timestamp.minute
    day = timestamp.day
    month = timestamp.month
    dow = timestamp.weekday()

    features["hour_sin"] = np.sin(2*np.pi*hour/24)
    features["hour_cos"] = np.cos(2*np.pi*hour/24)
    features["minute_sin"] = np.sin(2*np.pi*minute/60)
    features["minute_cos"] = np.cos(2*np.pi*minute/60)
    features["day_of_week_sin"] = np.sin(2*np.pi*dow/7)
    features["day_of_week_cos"] = np.cos(2*np.pi*dow/7)
    features["month_sin"] = np.sin(2*np.pi*month/12)
    features["month_cos"] = np.cos(2*np.pi*month/12)
    features["day_sin"] = np.sin(2*np.pi*day/31)
    features["day_cos"] = np.cos(2*np.pi*day/31)

    df = pd.DataFrame([features])[FEATURE_COLUMNS]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=FEATURE_COLUMNS)
    return df_scaled
