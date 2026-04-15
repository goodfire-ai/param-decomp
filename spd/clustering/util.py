from collections.abc import Callable
from typing import Literal


def format_scientific_latex(value: float) -> str:
    """Format a number in LaTeX scientific notation style."""
    if value == 0:
        return r"$0$"

    import math

    exponent: int = int(math.floor(math.log10(abs(value))))
    mantissa: float = value / (10**exponent)

    return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"


ModuleFilterSource = str | Callable[[str], bool] | set[str] | None
ModuleFilterFunc = Callable[[str], bool]
DeadComponentFilterStat = Literal["max", "mean"]
