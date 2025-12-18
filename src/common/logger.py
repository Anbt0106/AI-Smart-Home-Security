import logging
import os
import sys

def setup_logger(name="SmartHomeAI", log_file=None, level=logging.INFO):
    """
    Setup a logger with console and file handlers.
    
    Args:
        name (str): Name of the logger.
        log_file (str or None): Path to the log file. If None, defaults to 'logs/app.log'.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Define default log file path if not provided
    if log_file is None:
        # Determine absolute path to logs directory relative to project root
        # Assuming src/common/logger.py -> project_root/logs
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(base_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, "app.log")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicate logs if function called multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File Handler
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file handler for logger: {e}")

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Create a default logger instance for easy import
logger = setup_logger()
