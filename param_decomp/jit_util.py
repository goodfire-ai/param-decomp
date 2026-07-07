"""`eqx.filter_jit` + `compiler_options` passthrough.

`equinox.filter_jit` forwards `**jitkwargs` to `jax.jit` at runtime, but its typed
`@overload`s expose only `fun` + `donate` — so passing `compiler_options` (the native,
in-process way to set XLA compiler flags, and the way they enter the compile-cache key)
fails basedpyright with "no overloads match". This wrapper centralizes that one cast so
call sites stay clean and typed.
"""

from collections.abc import Callable
from typing import Literal

import equinox as eqx

DonateMode = Literal["all", "all-except-first", "warn", "warn-except-first", "none"]


def filter_jit[**P, T](
    fn: Callable[P, T],
    *,
    donate: DonateMode = "none",
    compiler_options: dict[str, bool | int | str] | None = None,
) -> Callable[P, T]:
    """`eqx.filter_jit(fn, donate=…, compiler_options=…)` — `compiler_options` is forwarded
    to `jax.jit` (XLA compiler flags, native + in the compile-cache key). `None` = no
    options. CPU backends accept and ignore GPU flags."""
    return eqx.filter_jit(fn, donate=donate, compiler_options=compiler_options)  # pyright: ignore[reportCallIssue]
