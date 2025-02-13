import logging
import os
from datetime import datetime

# Define log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define log directory path
log_path = os.path.join(os.getcwd(), "logs")

# Create the log directory if it does not exist
os.makedirs(log_path, exist_ok=True)

# Define the full log file path
LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # File handler to write logs to a file
        logging.FileHandler(LOG_FILEPATH),
        # Stream handler to print logs to console
        logging.StreamHandler()
    ]
)

logging.info("Logger setup is complete.")