import pickle
import pandas as pd
from config import TEST_CLEANED_UNSCALED, MODEL_FILE
from helpers import load_data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Load model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)


test_df = load_data(TEST_CLEANED_UNSCALED)
X_test = test_df.drop(columns=["log_price", "price"], axis=1)
y_test = test_df['price']

predictions = model.predict(X_test)

# Save predictions
test_df["predicted_price"] = predictions
test_df.to_csv("data/processed/predictions.csv", index=False)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print results
print(f"Random Forest Model Evaluation Metrics for Price Prediction:")
print(f"✅ R² Score: {r2:.4f}  (Higher is better, max = 1)")
print(f"✅ MAE: ${mae:.2f} (Lower is better, avg absolute error)")
print(f"✅ MSE: {mse:.2f} (Lower is better, penalizes large errors)")
print(f"✅ RMSE: ${rmse:.2f} (Lower is better, interpretable error in price units)")

print("Predictions saved.")
