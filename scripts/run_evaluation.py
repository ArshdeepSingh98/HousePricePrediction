import subprocess
import sys

print("Running model inference...")
subprocess.run([sys.executable, "src/model_inference.py"])
print("Model inference completed.")
