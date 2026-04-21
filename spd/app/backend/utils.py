"""Shared utilities for the SPD backend."""

import functools
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from fastapi import HTTPException

from spd.log import logger


def log_errors[T: Callable[..., Any]](func: T) -> T:
    """Decorator to log errors with full traceback for easier debugging."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise  # Let FastAPI handle HTTP exceptions normally
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            traceback.print_exc()
            raise

    return wrapper  # pyright: ignore[reportReturnType]


# Characters that don't get a space prefix in wordpiece
_PUNCT_NO_SPACE = set(".,!?;:'\")-]}>/")


def delimit_tokens(tokens: list[tuple[str, bool]]) -> str:
    """Join token strings, wrapping active spans in <<delimiters>>.

    Consecutive active tokens are grouped: [(" over", T), (" the", T), (" moon", T)]
    produces " <<over the moon>>".
    """
    parts: list[str] = []
    in_span = False
    for tok, active in tokens:
        if active and not in_span:
            stripped = tok.lstrip()
            parts.append(tok[: len(tok) - len(stripped)])
            parts.append("<<")
            parts.append(stripped)
            in_span = True
        elif active:
            parts.append(tok)
        elif in_span:
            parts.append(">>")
            parts.append(tok)
            in_span = False
        else:
            parts.append(tok)
    if in_span:
        parts.append(">>")
    return "".join(parts)


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{name} took {end - start:.1f}ms")
