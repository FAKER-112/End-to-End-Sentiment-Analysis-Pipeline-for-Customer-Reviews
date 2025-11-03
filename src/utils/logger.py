import os, sys, io
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")

sys.stdout.reconfigure(encoding='utf-8')
# Force UTF-8 output for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
# Create logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

# File handler (rotating)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
