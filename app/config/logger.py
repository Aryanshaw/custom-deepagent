import logging


class Logger:
    # ANSI escape codes for colors
    COLORS = {
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
    }

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.handlers = []
        self.logger.propagate = False  # Prevent propagation to root logger
        self.setup_logger()

    def setup_logger(self):
        """Configure the logger with basic settings"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # Custom formatter that includes colors
            handler.setFormatter(self.ColoredFormatter())
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            # Format the log record first with default formatter
            formatted_msg = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ).format(record)

            # Apply color to the entire formatted message based on level
            if record.levelno == logging.INFO:
                return f"{Logger.COLORS['BLUE']}{formatted_msg}{Logger.COLORS['RESET']}"
            elif record.levelno == logging.DEBUG:
                return f"{Logger.COLORS['GREEN']}{formatted_msg}{Logger.COLORS['RESET']}"
            elif record.levelno == logging.WARNING:
                return f"{Logger.COLORS['YELLOW']}{formatted_msg}{Logger.COLORS['RESET']}"
            elif record.levelno == logging.ERROR:
                return f"{Logger.COLORS['RED']}{formatted_msg}{Logger.COLORS['RESET']}"
            elif record.levelno == logging.CRITICAL:
                return f"{Logger.COLORS['BOLD']}{Logger.COLORS['RED']}{formatted_msg}{Logger.COLORS['RESET']}"
            else:
                return formatted_msg

    def info(self, message):
        """Log info messages in blue"""
        self.logger.info(message)

    def debug(self, message):
        """Log debug messages in green"""
        self.logger.debug(message)

    def warning(self, message):
        """Log warning messages in yellow"""
        self.logger.warning(message)

    def error(self, message):
        """Log error messages in red"""
        self.logger.error(message)

    def critical(self, message):
        """Log critical messages in bold red"""
        self.logger.critical(message)


logger = Logger("app")
