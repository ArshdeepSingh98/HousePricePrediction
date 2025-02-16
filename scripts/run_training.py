import subprocess
import sys

print("Running model training...")
subprocess.run([sys.executable, "src/model_training.py"])
print("Model training completed.")
