import os
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

import yaml

# Load environment variables from .env file
# This should be done at the module level to ensure environment
# variables are loaded before any other imports or code execution
load_dotenv()

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging to write to both console and file
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging for the agent.
    
    Args:
        log_level (str): The logging level to use
    
    Returns:
        logging.Logger: The configured logger
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create a logger
    logger = logging.getLogger("agent")
    logger.setLevel(numeric_level)
    logger.handlers = []  # Clear any existing handlers
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    # Create file handler
    log_filename = "agent.log"
    file_handler = logging.FileHandler(LOGS_DIR / log_filename)
    file_handler.setLevel(numeric_level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log a startup message to verify file logging is working
    logger.info(f"Logging initialized with level {log_level}")
    
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: The configuration dictionary
    """
    # Read config from yaml file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    if config is None:
        raise ValueError(f"Config file {config_path} is empty")
    
    return config
