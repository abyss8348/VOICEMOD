"""Logging configuration for FlowVoice."""

import logging
import sys
from .config import settings


def setup_logger(name: str = "flowvoice") -> logging.Logger:
    """Configure and return a structured logger for FlowVoice."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        level = getattr(logging, settings.log_level, logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
