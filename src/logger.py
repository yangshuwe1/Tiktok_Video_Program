import logging
import time
import os
from functools import wraps

def get_logger(name="tiktok"):
    """Get logger with default configuration to avoid circular dependency"""
    # Default configuration
    log_file = 'logs/processing.log'
    log_level = 'INFO'
    enable_console = True
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        if enable_console:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = True
    return logger

def log_time(section):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start = time.time()
            logger.info(f"[{section}] started")
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"[{section}] finished, elapsed: {elapsed:.2f} seconds")
            return result
        return wrapper
    return decorator 