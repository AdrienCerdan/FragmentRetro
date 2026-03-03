import logging
import logging.config
import os
from typing import Any

from rdkit import rdBase

# --- Hardcoded Configuration ---
LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "fragment": {
            "handlers": ["console"],
            "propagate": False,
            "level": "INFO",  # Default level
        }
    },
}
# --- End Hardcoded Configuration ---


def suppress_rdkit_logs() -> None:
    """Disable common RDKit warnings (valence, aromaticity, etc.) from console.

    Highly recommended during retrosynthesis searches where reverse SMARTS
    application produces many expected but noisy warnings.
    """
    rdBase.DisableLog("rdApp.*")


def setup_logging() -> None:
    """Setup logging configuration from hardcoded dict with environment variable override"""

    # Get log level from environment variable, default to INFO if not set
    log_level = os.getenv("FRAGMENT_LOG_LEVEL", "INFO").upper()

    # Validate the log level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        print(f"Invalid log level {log_level}, defaulting to INFO")
        log_level = "INFO"  # Make sure to reset if invalid

    # Override the log level in the copied config
    LOGGING_CONFIG["loggers"]["fragment"]["level"] = log_level

    logging.config.dictConfig(LOGGING_CONFIG)

    # Automatically suppress RDKit logs if not in DEBUG mode
    if log_level != "DEBUG":
        suppress_rdkit_logs()


logger = logging.getLogger("fragment")
