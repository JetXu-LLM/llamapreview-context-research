import logging
import sys
from typing import Optional

def setup_logger(name: str = "LlamaPReview", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with a console handler and a standard formatter.
    
    This function implements a singleton-like pattern for handlers to avoid 
    duplicate log messages if called multiple times.

    Args:
        name (str): The name of the logger. Defaults to "LlamaPReview".
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO). 
                     Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers to prevent duplicate logs
    if logger.handlers:
        return logger

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Define a professional log format: Timestamp - LoggerName - Level - Message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid double logging if root is configured elsewhere
    logger.propagate = False

    return logger