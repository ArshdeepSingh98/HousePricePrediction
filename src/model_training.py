import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import TRAIN_CLEANED_UNSCALED, MODEL_FILE
from helpers import load_data

# Load processed data
df = load_data(TRAIN_CLEANED_UNSCALED)


# Define features and target
X = df.drop(columns=["log_price", "price"], axis=1)
y = df["price"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred = model.fit(X, y)

# Save model
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("Model training completed and saved.")
