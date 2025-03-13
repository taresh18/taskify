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
def setup_logging() -> logging.Logger:
    """
    Set up logging for the agent.
    
    Returns:
        logging.Logger: The configured logger
    """
    # Load config to get logging settings
    config = load_config()
    logging_config = config.get("logging")
    
    log_level = logging_config.get("level")
    
    # Create logs directory
    log_directory = logging_config.get("log_directory")
    logs_dir = Path(log_directory)
    logs_dir.mkdir(exist_ok=True)
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler 
    log_file = logs_dir / "conversify.log"
    file_handler = logging.FileHandler(log_file, mode='a')  # 'a' for append mode
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Create and return the conversify logger as a child of root
    logger = logging.getLogger("conversify")
    logger.debug("Logging initialized with level %s to %s", log_level, log_directory)
    
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
