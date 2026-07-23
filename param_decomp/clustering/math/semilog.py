import math


def semilog(
    value: float,
    epsilon: float = 1e-3,
) -> float:
    if abs(value) < epsilon:
        return value
    else:
        sign: int = 1 if value >= 0 else -1
        # log10 here is safe, since we know the value is not close to zero
        return sign * epsilon * math.log1p(abs(value) / epsilon)
