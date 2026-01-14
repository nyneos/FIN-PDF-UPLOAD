"""
Structured logging configuration.
Provides request-level logging with timestamps and module labels.
"""

import logging
import sys
from typing import Optional


def configure_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure and return a module logger named `bank_parser`.

    The function is idempotent and safe to call multiple times.
    """

    log_level = (level or "INFO").upper()

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    # Ensure basicConfig is set only once
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:
        for h in logging.getLogger().handlers:
            h.setLevel(log_level)

    # Silence uvicorn default handlers when used under uvicorn
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.error").handlers.clear()

    return logging.getLogger("bank_parser")