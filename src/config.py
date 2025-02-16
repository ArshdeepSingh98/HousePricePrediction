import os

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Go to project root
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")  
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  
MODEL_DIR = os.path.join(BASE_DIR, "models")  
TRAIN_FILE = os.path.join(DATA_DIR, "train_set")
TEST_FILE = os.path.join(DATA_DIR, "test_set")

TRAIN_CLEANED = os.path.join(PROCESSED_DIR, "train_cleaned_data.csv")
TEST_CLEANED = os.path.join(PROCESSED_DIR, "test_cleaned_data.csv")
TRAIN_CLEANED_UNSCALED = os.path.join(PROCESSED_DIR, "train_cleaned_data_unscaled.csv")
TEST_CLEANED_UNSCALED = os.path.join(PROCESSED_DIR, "test_cleaned_data_unscaled.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "random_forest_price_model.pkl")

# Model Hyperparameters
RANDOM_STATE = 42
N_ESTIMATORS = 100
N_REGION_CLUSTERS = 10