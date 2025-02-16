import logging

# Configure logging
logging.basicConfig(
    filename="logs/ml_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_logger():
    return logging.getLogger()
