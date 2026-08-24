"""Setup a logger to be used in all modules in the library.

To use the logger, import it in any module and use it as follows:

    ```
    from param_decomp.core.log import logger

    logger.info("Info message")
    logger.warning("Warning message")
    ```
"""

import logging
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

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


def _configure_handler(
    handler: logging.Handler, *, log_format: LogFormat, level: int
) -> logging.Handler:
    formatter_config = _FORMATTERS[log_format]
    handler.setFormatter(
        logging.Formatter(
            fmt=formatter_config["fmt"],
            datefmt=formatter_config.get("datefmt"),
        )
    )
    handler.setLevel(level)
    return handler


def _configure_logger(handlers: list[logging.Handler], *, propagate: bool) -> _ParamDecompLogger:
    for existing_handler in tuple(logger.handlers):
        logger.removeHandler(existing_handler)
        existing_handler.close()
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.propagate = propagate
    return logger


def setup_logger(logfile: Path) -> _ParamDecompLogger:
    """Attach a console (INFO) + file (WARNING) handler to the application logger."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    return _configure_logger(
        [
            _configure_handler(logging.StreamHandler(), log_format="default", level=logging.INFO),
            _configure_handler(
                logging.FileHandler(logfile), log_format="default", level=logging.WARNING
            ),
        ],
        propagate=True,
    )


def setup_console_logger() -> _ParamDecompLogger:
    """Write the application logger's INFO output to stdout."""
    return _configure_logger(
        [
            _configure_handler(
                logging.StreamHandler(sys.stdout), log_format="terse", level=logging.INFO
            )
        ],
        propagate=False,
    )


logging.setLoggerClass(_ParamDecompLogger)
logger = cast(_ParamDecompLogger, logging.getLogger(_PARAM_DECOMP_LOGGER_NAME))
logger.addHandler(logging.NullHandler())
