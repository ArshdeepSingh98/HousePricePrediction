import pandas as pd
import os

def load_data(file_path):
    """Load CSV file as DataFrame"""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File {file_path} not found.")

def save_data(df, file_path):
    """Save DataFrame to CSV"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
