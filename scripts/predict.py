from features import build_features
from datetime import datetime
import pickle

with open("model_artifacts/xgboost_uber_model.pkl", "rb") as f:
    model = pickle.load(f)

if __name__ == "__main__":
    source = "Northeastern University"
    destination = "Beacon Hill"
    cab_type = "Uber"
    cab_name = "UberX"
    timestamp = datetime.now()
    surge_multiplier = 1.0

    X = build_features(source, destination, cab_type, cab_name, timestamp, surge_multiplier)
    price_pred = model.predict(X)[0]
    print(f"Predicted Fare: ${price_pred:.2f}")
