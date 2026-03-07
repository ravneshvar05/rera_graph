"""
logger.py — Loguru-based logger used across all modules.
"""

import sys
from loguru import logger

from config import settings

# Remove default handler
logger.remove()

# Console handler
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:{line} — {message}",
    colorize=True,
)

# File handler — rolling daily
logger.add(
    "logs/rera_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="14 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
)

__all__ = ["logger"]
