# Add to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from video_eater.logging_configuration.configure_logging import configure_logging
from video_eater.logging_configuration.log_levels import LogLevels

LOG_LEVEL = LogLevels.TRACE
configure_logging(LOG_LEVEL)
