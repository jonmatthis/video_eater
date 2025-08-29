# logging_config.py
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Centralized logging for the pipeline."""

    def __init__(self,
                 log_level: str = "INFO",
                 log_file: Optional[Path] = None,
                 console_output: bool = True):

        self.logger = logging.getLogger("video_pipeline")
        self.logger.setLevel(getattr(logging, log_level))

        # Console handler with clean formatting
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter('%(message)s')
            )
            self.logger.addHandler(console_handler)

        # File handler with detailed formatting
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)

    def step(self, step_num: int, total: int, description: str):
        """Log a pipeline step."""
        self.logger.info(f"\n[Step {step_num}/{total}] {description}")
        self.logger.info("-" * 40)

    def progress(self, current: int, total: int, task: str):
        """Log progress within a step."""
        self.logger.debug(f"  [{current}/{total}] {task}")

    def success(self, message: str):
        """Log success message."""
        self.logger.info(f"✅ {message}")

    def cache_hit(self, resource: str):
        """Log cache hit."""
        self.logger.debug(f"📂 Using cached {resource}")

    def error(self, message: str):
        """Log error message."""
        self.logger.error(f"❌ {message}")