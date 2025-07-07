"""trading.utils.logging

Centralised logging configuration so every CLI / notebook can call
`setup_logging()` instead of duplicating boilerplate.
"""
from __future__ import annotations

import logging
from datetime import datetime

__all__ = ["setup_logging"]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once.

    Safe to call multiple times (second calls are ignored).
    """
    if logging.getLogger().handlers:
        return  # already configured

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=level, format=log_format, datefmt=date_format)

    # quiet noisy libs
    for noisy in ("matplotlib", "pandas", "sklearn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
