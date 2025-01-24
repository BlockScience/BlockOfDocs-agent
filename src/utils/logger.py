import logging
from utils.config import load_config

config = load_config()

def setup_logger() -> logging.Logger:
    """Configure and return a logger instance."""
    
    # Create logger
    logger = logging.getLogger("rag_app")
    logger.setLevel(config.LOG_LEVEL)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add formatter to console handler
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    return logger
