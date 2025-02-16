import subprocess
import sys

print("Running feature engineering...")
subprocess.run([sys.executable, "src/feature_engineering.py"])
print("Feature engineering completed.")
