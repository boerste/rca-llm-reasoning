import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    """
    A logging handler that writes messages using tqdm.write().
    
    This ensures that log messages do not interfere with tqdm progress bars
    generated in the console.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # <--- write via tqdm.write instead of print()
            self.flush()
        except Exception:
            self.handleError(record)
            
def setup_logger(log_file: str | None = None):
    """
    Sets up the root logger.
    
    Configures a TqdmLoggingHandler for console output (compatible with progress bars)
    and optionally a FileHandler if a log file path is provided.

    Args:
        log_file (str | None): Optional path to a log file. If provided, logs will
            also be written to this file. Defaults to None.

    Returns:
        logging.Logger: The configured root logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_file:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    th = TqdmLoggingHandler()
    th.setFormatter(formatter)
    logger.addHandler(th)

    return logger

