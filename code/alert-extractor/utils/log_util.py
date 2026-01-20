"""
Log Utilities for Alert Extraction
"""
import logging

def get_logger(name=__name__, level=logging.INFO, fmt="%(asctime)s - %(levelname)s - %(message)s"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
   
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    
    logger.propagate = False
    
    return logger