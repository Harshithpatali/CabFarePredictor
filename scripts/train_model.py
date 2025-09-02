import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import os

# Load your preprocessed dataset
data_path = r"C:\Users\devar\Downloads\cab_price_eta_predictor\data\processed\processed_cab_rides.csv"
df = pd.read_csv(data_path)

# ------------------------------
# Features & Target
# ------------------------------
X = df.drop(columns=["price"])   # target is price
y = df["price"]

# Save feature columns list
feature_columns = list(X.columns)

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Scaling
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Train XGBoost Model
# ------------------------------
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ------------------------------
# Evaluation
# ------------------------------
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ XGBoost RMSE: {rmse:.2f}")
print(f"✅ XGBoost R²: {r2:.4f}")

# ------------------------------
# Save Artifacts
# ------------------------------
os.makedirs("../model_artifacts", exist_ok=True)

with open("../model_artifacts/xgboost_uber_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model_artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("../model_artifacts/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("🎉 Model, Scaler, and Feature Columns saved in model_artifacts/")
