"""Setup a logger to be used in all modules in the library.

To use the logger, import it in any module and use it as follows:

    ```
    from param_decomp.log import logger

    logger.info("Info message")
    logger.warning("Warning message")
    ```
"""

import logging
import shutil
from collections.abc import Mapping
from logging.config import dictConfig
from pathlib import Path
from typing import Literal

DEFAULT_LOGFILE: Path = Path(__file__).resolve().parent.parent / "logs" / "logs.log"

DIV_CHAR: str = "="
LogFormat = Literal["default", "terse"]
_PARAM_DECOMP_LOGGER_NAME: str = "param_decomp"

_FORMATTERS: dict[LogFormat, dict[Literal["fmt", "datefmt"], str]] = {
    "terse": {"fmt": "%(message)s"},
    "default": {
        "fmt": "%(asctime)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
}


class _ParamDecompLogger(logging.Logger):
    """`logging.Logger` with `values` and `section` convenience helpers."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def values(
        self,
        data: Mapping[str, None | bool | int | float | str] | list[None | bool | int | float | str],
        msg: str | None = None,
    ) -> None:
        """log a dict of metrics"""
        output: str
        if isinstance(data, list):
            output = "\n  ".join(str(v) for v in data)
        else:
            # otherwise, assume it's a dict
            longest_key: int = max(len(k) for k in data)
            lines: list[str] = [f"  {k:<{longest_key + 1}}: {v}" for k, v in data.items()]
            output = "\n".join(lines)

        if msg:
            self.info(f"{msg}:\n{output}")
        else:
            self.info("\n" + output)

    def section(
        self,
        msg: str,
    ) -> None:
        """Emit a visually separated section header"""
        # term width
        term_width: int = shutil.get_terminal_size(fallback=(50, 20)).columns
        self.info("\n" + DIV_CHAR * term_width + "\n" + msg + "\n" + DIV_CHAR * term_width)

    def set_format(self, handler: str, style: LogFormat) -> None:
        """Swap this logger's handler formatters in place.

        it would be nicer to do this when we initialize the logger, but that's done on module import
        """
        fmt: logging.Formatter = logging.Formatter(**_FORMATTERS[style])
        found_handler: bool = False
        for h in self.handlers:
            if getattr(h, "name", None) == handler:
                h.setFormatter(fmt)
                found_handler = True
                break
        if not found_handler:
            raise ValueError(
                f"Handler '{handler}' not found in logger '{self.name}' handlers: {self.handlers}. "
                f"could not set {style = }"
            )


def setup_logger(logfile: Path = DEFAULT_LOGFILE) -> _ParamDecompLogger:
    """Setup a logger to be used in all modules in the library.

    Sets up logging configuration with a console handler and a file handler.
    Console handler logs messages with INFO level, file handler logs WARNING level.
    The root logger is configured to use both handlers.

    Returns:
        _ParamDecompLogger: A configured logger object.

    Example:
        >>> logger = setup_logger()
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
    """
    logging.setLoggerClass(_ParamDecompLogger)

    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        "formatters": _FORMATTERS,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(logfile),
                "formatter": "default",
                "level": "WARNING",
            },
        },
        "loggers": {
            _PARAM_DECOMP_LOGGER_NAME: {
                "handlers": ["console", "file"],
                "level": "INFO",
            },
        },
    }

    dictConfig(logging_config)
    # we have to pass the name, or we always get the root logger
    _logger: _ParamDecompLogger = logging.getLogger(_PARAM_DECOMP_LOGGER_NAME)  # pyright:ignore[reportAssignmentType]
    return _logger


logger: _ParamDecompLogger = setup_logger()
