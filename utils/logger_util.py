import logging
import os
import sys


def setup_logging(log_dir=None, file_name='train.log'):
    """Initialize `logging` module."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.close()
    root_logger.handlers.clear()
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, file_name))
        file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
