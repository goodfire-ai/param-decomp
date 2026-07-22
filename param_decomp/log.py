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


def setup_logger(logfile: Path) -> _ParamDecompLogger:
    """Attach a console (INFO) + file (WARNING) handler to the `param_decomp` logger.

    Called once by the run entry point with the run's logfile; until then `logger` only
    carries a `NullHandler` (library-safe — no output unless the application opts in).
    """
    logging.setLoggerClass(_ParamDecompLogger)

    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        # dictConfig defaults this to True, which DISABLES every already-created logger —
        # including jax's, silencing JAX_LOG_COMPILES / persistent-cache diagnostics.
        "disable_existing_loggers": False,
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


logging.setLoggerClass(_ParamDecompLogger)
logger: _ParamDecompLogger = logging.getLogger(_PARAM_DECOMP_LOGGER_NAME)  # pyright:ignore[reportAssignmentType]
logger.addHandler(logging.NullHandler())
