import logging
from logging.config import dictConfig

from .handlers.colored_console import ColoredConsoleHandler
from .log_levels import LogLevels


class LoggerBuilder:

    def __init__(self,
                 level: LogLevels ):
        self.level = level
        dictConfig({"version": 1, "disable_existing_loggers": False})

    def _configure_root_logger(self):
        root = logging.getLogger()
        root.setLevel(self.level.value)

        # Clear existing handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)



        root.addHandler(self._build_console_handler())


    def _build_console_handler(self):
        handler = ColoredConsoleHandler()
        handler.setLevel(self.level.value)
        return handler



    def configure(self):
        if len(logging.getLogger().handlers) == 0:
            self._configure_root_logger()

