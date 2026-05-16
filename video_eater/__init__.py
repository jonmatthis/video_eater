import logging

from skellylogs import configure_logging, LogLevels

LOG_LEVEL = LogLevels.TRACE
configure_logging(level=LOG_LEVEL)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
